"""
Python database backends for `scan_database`, called from Rust.

Rust picks the backend and runs the SQL it compiled; this module only opens connections
for Python drivers and streams results back as DataFrames. Query execution reuses the
`ConnectionExecutor` that `read_database` uses, so every driver it supports works here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import re
import threading
import weakref
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from polars import DataFrame

# Rows per batch when the user gives no `batch_size`.
DEFAULT_BATCH_SIZE = 50_000

# Python package (distribution) names, for version lookups.
_DISTRIBUTIONS = {
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "mysql.connector": "mysql-connector-python",
    "snowflake.connector": "snowflake-connector-python",
    "google.cloud.bigquery": "google-cloud-bigquery",
    "databricks.sql": "databricks-sql-connector",
    "clickhouse_connect": "clickhouse-connect",
    "clickhouse_driver": "clickhouse-driver",
    "arrow_odbc": "arrow-odbc",
    "mssql_python": "mssql-python",
    "redshift_connector": "redshift-connector",
}


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(module: str) -> str | None:
    root = module.split(".")[0]
    if module == "sqlite3":
        # the standard library's module: the SQLite library is what has a version
        import sqlite3

        return sqlite3.sqlite_version
    for dist in (_DISTRIBUTIONS.get(module), root, root.replace("_", "-")):
        if dist:
            with contextlib.suppress(PackageNotFoundError, ValueError):
                return _dist_version(dist)
    return None


def probe(kind: str) -> tuple[str | None, str | None] | None:
    """Whether a backend is installed: `(version, location)`, or None."""
    head, _, tail = kind.partition(":")
    if head == "adbc":
        # a pip-installed `adbc_driver_<name>` package carries the driver library
        module = f"adbc_driver_{tail}"
        if not re.fullmatch(r"\w+", tail) or not _has_module(module):
            return None
        try:
            driver = importlib.import_module(module)
            path = driver._driver_path()
        except Exception:
            return None
        return _version(module), str(path)
    if head == "connectorx":
        return (_version("connectorx"), None) if _has_module("connectorx") else None
    if head == "sqlalchemy":
        return (_version("sqlalchemy"), None) if _has_module("sqlalchemy") else None
    if head in ("python", "odbc") and tail:
        return (_version(tail), None) if _has_module(tail) else None
    return None


def _module_of(obj: Any) -> str:
    module = type(obj).__module__.split(".", 1)[0]
    # compiled driver modules can have a leading underscore (eg: "_duckdb")
    public = module.lstrip("_")
    return public if public != module and _has_module(public) else module


def connection_info(obj: Any, kind: str) -> tuple[str, str]:
    """`(dialect, module)` of a connection factory or live connection object."""
    if kind == "factory":
        conn = obj()
        try:
            return connection_info(conn, "live")
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    module = _module_of(obj)
    if module == "sqlalchemy":
        engine = getattr(obj, "engine", None) or getattr(obj, "bind", None)
        dialect = getattr(getattr(engine, "dialect", None), "name", "") or ""
        return dialect, "sqlalchemy"
    return "", module


def credentials(provider: Callable[[], Any]) -> tuple[str | None, str | None] | None:
    """Evaluates a credential provider: `(user, password)`."""
    creds = provider()
    if isinstance(creds, tuple) and len(creds) == 2:
        if all(isinstance(c, str) for c in creds):
            msg = (
                "a database credential provider must return a dict, e.g. "
                "{'user': ..., 'password': ...}, not a (user, password) tuple"
            )
            raise TypeError(msg)
        # `(credentials, expiry)`, as the cloud storage credential providers return
        creds = creds[0]
    if creds is None:
        return None
    if isinstance(creds, dict):
        user = creds.get("user", creds.get("username"))
        password = creds.get("password", creds.get("token"))
        if user is None and password is None:
            # e.g. cloud storage keys: using them would drop the URI's password
            msg = (
                "the database credential provider returned neither 'user' nor "
                f"'password' (keys: {sorted(map(str, creds))})"
            )
            raise ValueError(msg)
        return user, password
    msg = f"a database credential provider must return a dict, got {type(creds).__name__!r}"
    raise TypeError(msg)


# A host, `host:port` or `[v6]:port`
_HOST = re.compile(r"(?:\[[^\]]+\]|[A-Za-z0-9._%-]+)(?::\d+)?")


def _delimiter(s: str) -> int:
    """The index of the first `/`, `?` or `#` (the end of a URI's authority)."""
    return min((i for i in map(s.find, "/?#") if i >= 0), default=len(s))


def _later_userinfo_end(
    rest: str, end: int, *, routing: bool, query_guard: bool
) -> int | None:
    """
    The last `@` after the authority's end that can end a userinfo.

    For `routing`, `host:port/` ends the authority: `localhost:5432/my@db` has no
    userinfo (a password like `12/x` is then only hidden). With `query_guard`, what
    lies between mustn't look like a query (`=`, `&`); hiding a password that the
    authority doesn't already end drops it, as the password may be base64.
    """
    fragment = rest[end : end + 1] == "#"
    if routing and rest[end : end + 1] == "/":
        _, colon, port = rest[:end].rpartition(":")
        if colon and port.isdigit():
            return None
    at = len(rest)
    while (at := rest.rfind("@", 0, at)) > end:
        after = rest[at + 1 :]
        hosts = after[: _delimiter(after)].split(",")
        if all(_HOST.fullmatch(h) for h in hosts) and (
            not query_guard or fragment or not re.search("[=&]", rest[end:at])
        ):
            return at
    return None


def _userinfo_end(rest: str, *, secret: bool = False) -> int | None:
    """
    The index of the `@` that ends the userinfo of `rest` (a URI after `scheme://`).

    As `userinfo_end` in Rust (`connection.rs`): the last `@` before the first `/`, `?`
    or `#`. Otherwise a password may hold one of them unencoded (`u:pa/ss@host`): a later
    `@` followed by a host ends the userinfo, unless what lies between looks like a query
    (`h:5432/db?x=a@b`). A `#` before a later `@` is in the password: database URIs have
    no fragments. With `secret`, for hiding the password, a password holding both an `@`
    and a `/` (`u:p@ss/w@host`) is taken whole.
    """
    end = _delimiter(rest)
    at = rest.rfind("@", 0, end)
    if at >= 0:
        later = (
            _later_userinfo_end(rest, end, routing=False, query_guard=True)
            if secret
            else None
        )
        return at if later is None else max(at, later)
    if ":" not in rest[:end] or rest.startswith("["):
        return None
    return _later_userinfo_end(rest, end, routing=not secret, query_guard=not secret)


def _split_authority(uri: str) -> tuple[str, str | None, str, str, bool] | None:
    """
    `(scheme, userinfo, host[:port], rest, unencoded)` of a `scheme://` URI, or None.

    `unencoded` marks a password holding an unencoded `/`, `?` or `#` (see
    `_userinfo_end`).
    """
    scheme, sep, rest = uri.partition("://")
    if not sep:
        return None
    at = _userinfo_end(rest)
    unencoded = at is not None and at > _delimiter(rest)
    userinfo = rest[:at] if at is not None else None
    rest = rest[at + 1 :] if at is not None else rest
    end = _delimiter(rest)
    return scheme, userinfo, rest[:end], rest[end:], unencoded


def _with_credentials(uri: str, creds: tuple[str | None, str | None] | None) -> str:
    split = _split_authority(uri)
    if split is None or not (creds or split[4]):
        return uri
    scheme, userinfo, host, rest, _ = split
    old_user, colon, old_password = (userinfo or "").partition(":")
    user, password = creds or (None, None)
    auth = quote(user, safe="") if user is not None else old_user
    if password is not None:
        auth += ":" + quote(password, safe="")
    elif colon:
        # the provider gave no password: keep the URI's (encoded, see `_split_authority`)
        auth += ":" + quote(unquote(old_password), safe="")
    return f"{scheme}://{auth}@{host}{rest}" if auth else f"{scheme}://{host}{rest}"


# Seconds before a connect attempt gives up, where the URI sets none: libpq waits for
# the operating system's TCP timeout (minutes) by default, and attempts are retried.
DEFAULT_CONNECT_TIMEOUT = 10

# SQLAlchemy's Postgres DB-API drivers built on libpq, which takes `connect_timeout`
_LIBPQ_SQLALCHEMY_DRIVERS = ("psycopg", "psycopg2", "psycopg2cffi")


def _with_connect_timeout(uri: str, dialect: str, kind: str) -> str:
    if dialect not in ("postgresql", "redshift") or "connect_timeout=" in uri:
        return uri
    head, _, tail = kind.partition(":")
    if head == "sqlalchemy":
        # `postgresql://` and sqlalchemy-redshift's `redshift://` default to psycopg2
        driver = uri.split(":", 1)[0].partition("+")[2] or "psycopg2"
        libpq = driver in _LIBPQ_SQLALCHEMY_DRIVERS
    else:
        # ConnectorX reads it too; pg8000 and redshift_connector reject it
        libpq = head == "connectorx" or tail in ("psycopg", "psycopg2")
    if not libpq:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}connect_timeout={DEFAULT_CONNECT_TIMEOUT}"


def _strip_driver(uri: str) -> str:
    scheme, sep, rest = uri.partition(":")
    return scheme.split("+", 1)[0] + sep + rest


def _libpq_uri(uri: str) -> str:
    """A URI for libpq, which takes only the `postgresql://` and `postgres://` schemes."""
    uri = _strip_driver(uri)
    scheme, sep, rest = uri.partition(":")
    return f"postgresql{sep}{rest}" if scheme.lower() == "redshift" else uri


# SQLite URI filename parameters: they make the path a `file:` URI
_SQLITE_URI_PARAMS = ("cache", "immutable", "mode", "modeof", "nolock", "psow", "vfs")


def _file_target(uri: str) -> tuple[str, dict[str, str]]:
    """
    The path of a `sqlite:` / `duckdb:` URI and its parameters, as SQLAlchemy reads it.

    `sqlite:///rel.db` is relative, `sqlite:////abs.db` absolute; `sqlite://` is
    ':memory:'. A `file:` URI (`sqlite:///file:a.db?mode=ro`) is returned whole.
    """
    rest = re.sub(r"^\w+(\+\w+)?:", "", uri, count=1)
    if rest.startswith("//"):
        # an authority (`sqlite://rel.db`, ConnectorX's form) is read as part of the path
        rest = rest[2:]
        rest = rest.removeprefix("/")
    if rest.startswith("file:"):
        return rest, {}
    path, _, query = rest.partition("?")
    params = dict(parse_qsl(query, keep_blank_values=True))
    path = unquote(path)
    if path in ("", ":memory:", "/:memory:"):
        return ":memory:", params
    # `sqlite:////C:/path` -> `C:/path`: on Windows, `/C:/path` is a network path
    if re.match(r"^/[A-Za-z]:[\\/]", path):
        path = path[1:]
    return path, params


def _file_path(uri: str) -> str:
    return _file_target(uri)[0]


def _sqlite_connect_args(uri: str) -> tuple[str, dict[str, Any]]:
    """`sqlite3.connect` arguments for a `sqlite:` URI."""
    path, params = _file_target(uri)
    # sessions move between threads; each is used by one thread at a time
    kw: dict[str, Any] = {"check_same_thread": False}
    file_params, unknown = {}, []
    for name, value in params.items():
        if name in _SQLITE_URI_PARAMS:
            file_params[name] = value
        elif name in ("timeout", "detect_types", "cached_statements"):
            kw[name] = _convert(
                "sqlite3", name, value, float if name == "timeout" else int
            )
        elif name == "isolation_level":
            kw[name] = value or None
        elif name not in ("uri", "check_same_thread"):
            # `uri=true` is implied by the file parameters, SQLAlchemy's form; the
            # thread check is always off
            unknown.append(name)
    _raise_unmapped("sqlite3", unknown)
    if file_params:
        safe = "/:\\"
        path = f"file:{quote(path, safe=safe)}?{urlencode(file_params)}"
    if path.startswith("file:"):
        kw["uri"] = True
    return path, kw


def _sqlalchemy_sqlite_uri(uri: str) -> str:
    """
    A `sqlite:` URI in SQLAlchemy's form, for the file the other backends open.

    SQLAlchemy reads `sqlite:rel.db` as no URI at all, and drops `?mode=ro` without
    `uri=true` (with `uri=true`, it wants a `file:` path).
    """
    path, kw = _sqlite_connect_args(uri)
    scheme = uri.split(":", 1)[0]
    if path == ":memory:":
        return f"{scheme}://"
    query = {
        k: str(v)
        for k, v in kw.items()
        if k not in ("check_same_thread", "uri") and v is not None
    }
    if kw.get("uri"):
        path, _, file_query = path.partition("?")
        query = {**dict(parse_qsl(file_query, keep_blank_values=True)), **query}
        query["uri"] = "true"
    # SQLAlchemy decodes the path
    path = quote(path, safe="/:\\")
    return f"{scheme}:///{path}" + (f"?{urlencode(query)}" if query else "")


def _duckdb_connect_args(uri: str) -> tuple[str, dict[str, Any]]:
    """`duckdb.connect` arguments for a `duckdb:` URI: parameters are DuckDB settings."""
    path, params = _file_target(uri)
    kw: dict[str, Any] = {}
    if "read_only" in params:
        kw["read_only"] = _convert(
            "duckdb", "read_only", params.pop("read_only"), _bool
        )
    if params:
        # DuckDB rejects an unknown setting itself
        kw["config"] = params
    return path, kw


def _connectorx_uri(uri: str, dialect: str) -> str:
    uri = _strip_driver(uri)
    if dialect != "sqlite":
        return uri
    # ConnectorX joins the host and path of `sqlite://…` (lowercasing the host) and
    # can't open `file:` URIs: hand it the same file as the other backends, absolute
    path, params = _file_target(uri)
    if params or path.startswith("file:"):
        msg = (
            f"connectorx can't open SQLite with URI parameters ({uri.split('?', 1)[-1]!r}); "
            "pin another driver with driver='adbc' or driver='sqlite3'"
        )
        raise ValueError(msg)
    if path == ":memory:":
        return "sqlite://:memory:"
    return "sqlite://" + quote(os.path.abspath(path), safe="/:\\")


def _url_parts(uri: str) -> dict[str, Any]:
    split = _split_authority(_strip_driver(uri))
    if split is None:
        return dict.fromkeys(("host", "port", "user", "password", "database"))
    _, userinfo, hostport, rest, _ = split
    user, colon, password = (userinfo or "").partition(":")
    if "," in hostport:
        msg = (
            "this driver takes one host, and the URI names several; "
            "pin a libpq driver (driver='psycopg' or 'adbc') for multiple hosts"
        )
        raise ValueError(msg)
    location = urlsplit(f"//{hostport}")
    try:
        port = location.port
    except ValueError:
        # the text after `:` is shown nowhere: it may be part of a password
        msg = (
            "the connection URI has an invalid port; an unencoded `/`, `?`, `#` or "
            "`@` in the password must be percent-encoded"
        )
        raise ValueError(msg) from None
    path = rest.partition("?")[0].partition("#")[0]
    return {
        "host": location.hostname,
        "port": port,
        "user": unquote(user) if user else None,
        "password": unquote(password) if colon and password else None,
        "database": unquote(path.lstrip("/")) or None,
    }


def _url_query(uri: str) -> dict[str, str]:
    split = _split_authority(uri)
    rest = split[3] if split is not None else uri
    query = rest.partition("?")[2].partition("#")[0]
    return dict(parse_qsl(query, keep_blank_values=True))


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _bool(value: str) -> bool:
    low = value.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    msg = f"expected a boolean, got {value!r}"
    raise ValueError(msg)


def _convert(module: str, name: str, value: str, convert: Callable[[str], Any]) -> Any:
    try:
        return convert(value)
    except ValueError as e:
        msg = f"URI parameter {name}={value!r} for {module}: {e}"
        raise ValueError(msg) from None


def _raise_unmapped(module: str, names: list[str]) -> None:
    if names:
        # a name that isn't one may be part of a password the URI parse missed
        shown = [n for n in names if re.fullmatch(r"[A-Za-z_][\w.-]*", n)]
        listed = f"{sorted(shown)}" if len(shown) == len(names) else f"({len(names)})"
        msg = (
            f"{module} has no connect option for URI parameter(s) {listed}; "
            "pass the driver's own keyword arguments with connect_options={...}, "
            "or pin another driver with driver=..."
        )
        raise ValueError(msg)


def _tls_context(
    *,
    cafile: str | None = None,
    certfile: str | None = None,
    keyfile: str | None = None,
    key_password: str | None = None,
    verify: bool = True,
    check_hostname: bool = True,
) -> Any:
    """An `ssl.SSLContext`: `cafile="system"` (or None) uses the system's CAs."""
    import ssl

    try:
        ctx = ssl.create_default_context(cafile=None if cafile == "system" else cafile)
        if not verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        else:
            ctx.check_hostname = check_hostname
        if certfile is not None:
            ctx.load_cert_chain(certfile, keyfile, key_password)
    except OSError as e:
        files = [f for f in (cafile, certfile, keyfile) if f and f != "system"]
        msg = f"can't load the TLS certificate file(s) {files}: {e.strerror or e}"
        raise ValueError(msg) from None
    return ctx


def _pg8000_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    # libpq's TLS parameters as an SSL context
    files = ("sslrootcert", "sslcert", "sslkey", "sslpassword")
    mode = params.pop("sslmode", None)
    if mode is None and any(f in params for f in files):
        mode = "require"
    if mode in ("require", "verify-ca", "verify-full"):
        cafile = params.get("sslrootcert")
        kw["ssl_context"] = _tls_context(
            cafile=cafile,
            certfile=params.get("sslcert"),
            keyfile=params.get("sslkey"),
            key_password=params.get("sslpassword"),
            # as libpq: `require` verifies the server when given a root certificate
            verify=mode != "require" or cafile is not None,
            check_hostname=mode == "verify-full",
        )
    elif mode not in (None, "disable", "allow", "prefer"):
        msg = f"URI parameter sslmode={mode!r} for pg8000: unknown mode"
        raise ValueError(msg)
    for f in files:
        params.pop(f, None)
    # pg8000 can't fall back between plain and TLS connections: `allow` and `prefer`
    # connect without TLS, as when there is no `sslmode`
    if (host := params.pop("host", None)) is not None:
        # libpq's socket directory
        if host.startswith("/"):
            kw["unix_sock"] = f"{host.rstrip('/')}/.s.PGSQL.{port or 5432}"
        else:
            kw["host"] = host
    if (options := params.pop("options", None)) is not None:
        kw["startup_params"] = {"options": options}
    if (timeout := params.pop("connect_timeout", None)) is not None:
        # pg8000's `timeout` holds for every socket operation: `_connect_uri` lifts it
        # once connected (unless `timeout` is given too)
        seconds = _convert("pg8000", "connect_timeout", timeout, float)
        kw[_PG8000_CONNECT_TIMEOUT] = seconds if seconds > 0 else None
    if (name := params.pop("fallback_application_name", None)) is not None:
        kw.setdefault("application_name", name)
    # libpq parameters pg8000 has no option for: accepted where pg8000 behaves the same,
    # rejected where it would ignore what they ask for
    same = {
        "target_session_attrs": ("any",),
        "client_encoding": ("utf8", "utf-8", "unicode"),
        "gssencmode": ("disable", "prefer"),
    }
    for name, values in same.items():
        value = params.pop(name, None)
        if value is not None and value.lower() not in values:
            msg = (
                f"URI parameter {name}={value!r} for pg8000: pg8000 can't honour it "
                f"(supported: {', '.join(values)})"
            )
            raise ValueError(msg)
    for name in ("keepalives_idle", "keepalives_interval", "keepalives_count"):
        # tuning of the TCP keepalives that `keepalives` turns on
        params.pop(name, None)


def _mysql_ssl_mode(module: str, params: dict[str, str], kw: dict[str, Any]) -> None:
    # MySQL's `--ssl-mode`, as the ADBC driver and ConnectorX read it
    mode = params.pop("ssl_mode", params.pop("ssl-mode", None))
    if module != "mariadb" and (ssl := params.pop("ssl", None)) is not None:
        # `ssl=true` (SQLAlchemy URLs, mariadb's option): TLS is required
        tls = _convert(module, "ssl", ssl, _bool)
        mode = mode or ("REQUIRED" if tls else "DISABLED")
    if mode is None:
        return
    mode = mode.upper()
    if mode not in (
        "DISABLED",
        "PREFERRED",
        "REQUIRED",
        "VERIFY_CA",
        "VERIFY_IDENTITY",
    ):
        msg = f"URI parameter ssl_mode={mode!r} for {module}: unknown mode"
        raise ValueError(msg)
    if module == "MySQLdb":
        kw["ssl_mode"] = mode
        return
    if mode == "DISABLED":
        kw["ssl" if module == "mariadb" else "ssl_disabled"] = module != "mariadb"
    elif mode == "REQUIRED":
        if module == "mariadb":
            kw["ssl"] = True
        elif module == "mysql.connector":
            # it only prefers TLS without a certificate check: `_connect_uri` checks
            kw["ssl_disabled"] = False
            kw[_REQUIRE_TLS] = True
        elif module == "pymysql" and not {"ssl_ca", "ssl_cert", "ssl_key"} & set(
            params
        ):
            # without a certificate parameter, pymysql only prefers TLS
            kw["ssl"] = _tls_context(verify=False)
    elif mode != "PREFERRED":
        if module == "mariadb":
            kw["ssl"] = True
        kw["ssl_verify_cert"] = True
        if mode == "VERIFY_IDENTITY" and module != "mariadb":
            kw["ssl_verify_identity"] = True
        if module == "mysql.connector":
            kw[_REQUIRE_TLS] = True


# `_query_kwargs` markers for `_connect_uri`, not driver arguments
_PG8000_CONNECT_TIMEOUT = "__polars_connect_timeout"
_REQUIRE_TLS = "__polars_require_tls"


def _check_mysql_tls(conn: Any) -> None:
    """Raises, closing the connection, if a MySQL connection isn't encrypted."""
    try:
        cur = conn.cursor()
        cur.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
        row = cur.fetchone()
        cur.close()
    except BaseException:
        conn.close()
        raise
    if not row or not row[1]:
        conn.close()
        msg = "ssl_mode requires TLS, but the server doesn't support it"
        raise ValueError(msg)


def _trino_params(params: dict[str, str], kw: dict[str, Any], port: int | None) -> None:
    if (verify := params.pop("verify", None)) is not None:
        # a boolean, or the CA bundle to verify against
        try:
            kw["verify"] = _bool(verify)
        except ValueError:
            kw["verify"] = verify
    # the JDBC driver's form
    if (tls := params.pop("SSL", None)) is not None and _convert(
        "trino", "SSL", tls, _bool
    ):
        kw["http_scheme"] = "https"


def _redshift_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    mode = params.pop("sslmode", None)
    if mode == "disable":
        kw["ssl"] = False
    elif mode in ("verify-ca", "verify-full"):
        kw["ssl"], kw["sslmode"] = True, mode
    elif mode is not None:
        msg = (
            f"URI parameter sslmode={mode!r} for redshift_connector: it supports "
            "only 'disable', 'verify-ca' and 'verify-full'"
        )
        raise ValueError(msg)


def _pymssql_params(
    params: dict[str, str], kw: dict[str, Any], port: int | None
) -> None:
    # the ODBC / JDBC drivers' `encrypt`
    if (encrypt := params.pop("encrypt", None)) is not None:
        kw["encryption"] = (
            "require" if _convert("pymssql", "encrypt", encrypt, _bool) else "off"
        )


_S, _I, _F, _B = str, int, float, _bool
_MYSQL_COMMON: dict[str, Any] = {
    "unix_socket": _S,
    "init_command": _S,
    "connect_timeout": _I,
    "read_timeout": _I,
    "write_timeout": _I,
    "ssl_ca": _S,
    "ssl_cert": _S,
    "ssl_key": _S,
    "compress": _B,
    "local_infile": _B,
    "autocommit": _B,
}

# URI query parameters a DB-API driver takes, as `{parameter: converter}` (the keyword
# argument of the same name) or `{parameter: (keyword, converter)}`; for drivers whose
# connect arguments come from the URI's parts. A parameter not listed raises.
_QUERY_KWARGS: dict[str, dict[str, Any]] = {
    "pg8000": {
        "application_name": _S,
        "timeout": _F,
        "tcp_keepalive": _B,
        "keepalives": ("tcp_keepalive", _B),
        "source_address": _S,
        "unix_sock": _S,
    },
    "pymysql": {
        **_MYSQL_COMMON,
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "program_name": _S,
        "bind_address": _S,
        "read_default_file": _S,
        "read_default_group": _S,
        "server_public_key": _S,
        "connect_timeout": _F,
        "read_timeout": _F,
        "write_timeout": _F,
        "max_allowed_packet": _I,
        "client_flag": _I,
        "ssl_key_password": _S,
        "ssl_verify_cert": _B,
        "ssl_verify_identity": _B,
        "ssl_check_hostname": ("ssl_verify_identity", _B),
        "ssl_disabled": _B,
        "binary_prefix": _B,
        "use_unicode": _B,
    },
    "mysql.connector": {
        **_MYSQL_COMMON,
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "time_zone": _S,
        "auth_plugin": _S,
        "ssl_cipher": _S,
        "tls_versions": ("tls_versions", lambda v: v.split(",")),
        "tls_ciphersuites": ("tls_ciphersuites", lambda v: v.split(",")),
        "connection_timeout": _I,
        "ssl_verify_cert": _B,
        "ssl_verify_identity": _B,
        "ssl_check_hostname": ("ssl_verify_identity", _B),
        "ssl_disabled": _B,
        "local_infile": ("allow_local_infile", _B),
        "allow_local_infile": _B,
        "use_pure": _B,
        "use_unicode": _B,
        "force_ipv6": _B,
    },
    "MySQLdb": {
        **{k: v for k, v in _MYSQL_COMMON.items() if not k.startswith("ssl_")},
        "charset": _S,
        "collation": _S,
        "sql_mode": _S,
        "auth_plugin": _S,
        "read_default_file": _S,
        "read_default_group": _S,
        "client_flag": _I,
        # TLS files: collected into MySQLdb's `ssl` dict
        "ssl_ca": _S,
        "ssl_cert": _S,
        "ssl_key": _S,
        "ssl_capath": _S,
        "ssl_cipher": _S,
    },
    "mariadb": {
        **_MYSQL_COMMON,
        "ssl_capath": _S,
        "ssl_cipher": _S,
        "ssl_crlpath": _S,
        "tls_version": _S,
        "default_file": _S,
        "default_group": _S,
        "plugin_dir": _S,
        "ssl": _B,
        "ssl_verify_cert": _B,
    },
    "pymssql": {
        "charset": _S,
        "appname": _S,
        "tds_version": _S,
        "conn_properties": _S,
        "encryption": _S,
        "timeout": _I,
        "login_timeout": _I,
        "read_only": _B,
        "autocommit": _B,
        "use_datetime2": _B,
    },
    "oracledb": {
        "service_name": _S,
        "sid": _S,
        "protocol": _S,
        "ssl_server_cert_dn": _S,
        "ssl_server_dn_match": _B,
        "use_sni": _B,
        "wallet_location": _S,
        "wallet_password": _S,
        "config_dir": _S,
        "https_proxy": _S,
        "https_proxy_port": _I,
        "tcp_connect_timeout": _F,
        "connect_timeout": ("tcp_connect_timeout", _F),
        "retry_count": _I,
        "retry_delay": _I,
        "expire_time": _I,
        "sdu": _I,
        "stmtcachesize": _I,
        "disable_oob": _B,
        "edition": _S,
        "program": _S,
        "machine": _S,
        "terminal": _S,
        "osuser": _S,
    },
    "redshift_connector": {
        "ssl": _B,
        "timeout": _I,
        "tcp_keepalive": _B,
        "application_name": _S,
        "region": _S,
        "cluster_identifier": _S,
        "db_user": _S,
        "iam": _B,
        "profile": _S,
        "max_prepared_statements": _I,
    },
    "trino": {
        "source": _S,
        "http_scheme": _S,
        "timezone": _S,
        "request_timeout": _F,
        "max_attempts": _I,
        "client_tags": ("client_tags", lambda v: v.split(",")),
        "legacy_primitive_types": _B,
    },
    "clickhouse_connect": {
        "interface": _S,
        "secure": _B,
        "verify": _B,
        "ca_cert": _S,
        "client_cert": _S,
        "client_cert_key": _S,
        "server_host_name": _S,
        "tls_mode": _S,
        "http_proxy": _S,
        "https_proxy": _S,
        "compress": _S,
        "connect_timeout": _F,
        "send_receive_timeout": _F,
        "query_limit": _I,
        "session_id": _S,
    },
}

# Parameters a driver reads in its own way: they're popped before the table lookup
_SPECIAL_PARAMS: dict[
    str, Callable[[dict[str, str], dict[str, Any], int | None], None]
] = {
    "pg8000": _pg8000_params,
    "redshift_connector": _redshift_params,
    "trino": _trino_params,
    "pymssql": _pymssql_params,
    **{
        m: (lambda m: lambda q, kw, _port: _mysql_ssl_mode(m, q, kw))(m)
        for m in ("pymysql", "mysql.connector", "MySQLdb", "mariadb")
    },
}


def _query_kwargs(module: str, uri: str, port: int | None = None) -> dict[str, Any]:
    """A driver's connect keyword arguments from the URI's query parameters."""
    params = _url_query(uri)
    kw: dict[str, Any] = {}
    if (special := _SPECIAL_PARAMS.get(module)) is not None:
        special(params, kw, port)
    table = _QUERY_KWARGS.get(module, {})
    unknown = []
    for name, value in params.items():
        if (spec := table.get(name)) is None:
            unknown.append(name)
            continue
        keyword, convert = spec if isinstance(spec, tuple) else (name, spec)
        kw[keyword] = _convert(module, name, value, convert)
    _raise_unmapped(module, unknown)
    if module == "MySQLdb":
        # TLS files go in one `ssl` dict
        ssl = {
            k[4:]: kw.pop(k)
            for k in list(kw)
            if k.startswith("ssl_") and k != "ssl_mode"
        }
        if ssl:
            kw["ssl"] = ssl
    return kw


def _connect_uri(module: str, uri: str, options: dict[str, Any]) -> Any:
    """A DB-API connection to a URI through a driver module."""
    mod = importlib.import_module(module)
    p = _url_parts(uri)
    if module == "sqlite3":
        path, kw = _sqlite_connect_args(uri)
        return mod.connect(path, **{**kw, **options})
    if module == "duckdb":
        path, kw = _duckdb_connect_args(uri)
        return mod.connect(path, **{**kw, **options})
    if module == "psycopg":
        # libpq reads (and checks) the query parameters itself
        return mod.connect(_libpq_uri(uri), **options)
    if module == "psycopg2":
        conn = mod.connect(_libpq_uri(uri), **options)
        # bytea as bytes, not memoryview (which would become an Object column)
        ext = importlib.import_module("psycopg2.extensions")
        as_bytes = ext.new_type(
            mod.BINARY.values,
            "POLARS_BYTES",
            lambda v, cur: None if v is None else mod.BINARY(v, cur).tobytes(),
        )
        ext.register_type(as_bytes, conn)
        return conn

    # the driver's keyword arguments: the URI's parts, its query, then `connect_options`
    kw = _query_kwargs(module, uri, p["port"])
    if module in (
        "pg8000",
        "pymysql",
        "mysql.connector",
        "mariadb",
        "redshift_connector",
    ):
        kw = {**_drop_none(**p), **kw}
    elif module == "MySQLdb":
        base = _drop_none(host=p["host"], port=p["port"], user=p["user"])
        kw = {**base, **_drop_none(passwd=p["password"], db=p["database"]), **kw}
    elif module == "pymssql":
        base = _drop_none(server=p["host"], port=p["port"], user=p["user"])
        kw = {
            **base,
            **_drop_none(password=p["password"], database=p["database"]),
            **kw,
        }
    elif module == "oracledb":
        # `/name` is the service name, as is `?service_name=name` (SQLAlchemy's form)
        if "sid" not in kw:
            kw.setdefault("service_name", p["database"] or "")
        base = _drop_none(user=p["user"], password=p["password"])
        kw = {**base, "host": p["host"], "port": p["port"] or 1521, **kw}
    elif module == "trino":
        from trino.dbapi import connect  # type: ignore[import-not-found]

        catalog, _, schema = (p["database"] or "").partition("/")
        base = _drop_none(host=p["host"], port=p["port"], user=p["user"])
        base.update(_drop_none(catalog=catalog or None, schema=schema or None))
        if p["password"] is not None:
            from trino.auth import BasicAuthentication  # type: ignore[import-not-found]

            # Trino takes a password only over HTTPS
            base["auth"] = BasicAuthentication(p["user"] or "", p["password"])
            base["http_scheme"] = "https"
        return connect(**{**base, **kw, **options})
    elif module == "clickhouse_connect":
        from clickhouse_connect.dbapi import connect  # type: ignore[import-not-found]

        base = _drop_none(host=p["host"], port=p["port"], username=p["user"])
        base.update(_drop_none(password=p["password"], database=p["database"]))
        return connect(**{**base, **kw, **options})
    else:
        msg = (
            f"scan_database can't open a {module!r} connection from a URI; "
            "pass a connection factory instead, e.g. `lambda: <module>.connect(...)`"
        )
        raise NotImplementedError(msg)
    require_tls = kw.pop(_REQUIRE_TLS, False)
    # libpq's `connect_timeout` for pg8000, whose `timeout` holds for every read
    lift_timeout = _PG8000_CONNECT_TIMEOUT in kw and not ({"timeout"} & {*kw, *options})
    if _PG8000_CONNECT_TIMEOUT in kw:
        kw.setdefault("timeout", kw.pop(_PG8000_CONNECT_TIMEOUT))
    conn = mod.connect(**{**kw, **options})
    if lift_timeout:
        with contextlib.suppress(AttributeError):
            conn._usock.settimeout(None)
    if require_tls:
        _check_mysql_tls(conn)
    return conn


# Connect options holding a secret, by a part of their name (as `is_secret_key` in
# Rust), compared without `_` and `-`
_SECRET_OPTIONS = (
    "password",
    "passwd",
    "pwd",
    "passphrase",
    "secret",
    "token",
    "credential",
    "apikey",
    "accesskey",
    "privatekey",
    "accountkey",
    "encryptionkey",
    "masterkey",
    # DuckDB's `azure_storage_connection_string` holds an `AccountKey=`
    "connectionstring",
)


def _is_secret_option(key: str) -> bool:
    key = key.lower().replace("_", "").replace("-", "")
    # SQLCipher's `?key=`
    return key == "key" or any(s in key for s in _SECRET_OPTIONS)


def _secrets(
    target: Any,
    creds: tuple[str | None, str | None] | None,
    options: dict[str, Any],
) -> list[str]:
    """The secrets of a connection, to scrub from driver errors: longest first."""
    out: list[str] = []
    if isinstance(target, str):
        scheme, sep, rest = target.partition("://")
        at = _userinfo_end(rest, secret=True) if sep else None
        if at is not None and ":" in rest[:at]:
            password = rest[:at].partition(":")[2]
            out += [password, unquote(password)]
            # a driver that misreads the URI echoes a piece (`invalid port "kq7"`)
            out += [p for p in re.split("[/?#@]", password) if p != password]
        if sep:
            query = rest[at + 1 if at is not None else 0 :].partition("?")[2]
            out += [v for k, v in parse_qsl(query) if _is_secret_option(k)]
        else:
            # an ODBC connection string: `PWD=…` or `PWD={…;…}`
            for m in re.finditer(
                r"(?i)(?:^|;)\s*([^=;]*)=\s*(\{(?:[^}]|\}\})*\}|[^;]*)",
                target,
            ):
                if not _is_secret_option(m.group(1)):
                    continue
                value = m.group(2).strip()
                out.append(value)
                if value.startswith("{"):
                    out.append(value[1:-1].replace("}}", "}"))
    if creds and creds[1]:
        out += [creds[1], quote(creds[1], safe="")]
    for key, value in options.items():
        if isinstance(value, str) and _is_secret_option(key):
            out.append(value)
    return sorted({s for s in out if s}, key=len, reverse=True)


# Secrets shorter than this are only replaced where they stand as a value (`:pw@`,
# `=pw`): as bare substrings they would garble the message
_MIN_BARE_SECRET_LEN = 4


def _scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if len(secret) >= _MIN_BARE_SECRET_LEN:
            text = text.replace(secret, "***")
        else:
            text = re.sub(
                rf"(?<=[:='\"]){re.escape(secret)}(?=$|[@;&\s'\"}},)])", "***", text
            )
    return text


@contextlib.contextmanager
def _scrubbed(secrets: list[str]) -> Iterator[None]:
    """Driver errors without the connection's secrets: some echo the DSN."""
    try:
        yield
    except Exception as e:
        message = _scrub(str(e), secrets)
        if message == str(e):
            raise
        e.args = (message,)
        e.__cause__ = None
        e.__suppress_context__ = True
        if _scrub(str(e), secrets) != str(e):
            # a message not built from `args`
            raise RuntimeError(f"{type(e).__name__}: {message}") from None
        raise


def _no_panics(frames: Iterator[DataFrame]) -> Iterator[DataFrame]:
    """Batches, with a Polars panic on an Arrow type it can't import as an error."""
    from polars.exceptions import ComputeError

    try:
        yield from frames
    except BaseException as e:
        # e.g. Arrow month-day-nano intervals: a panic would abort the whole query
        if type(e).__name__ != "PanicException":
            raise
        msg = (
            f"the driver returned data Polars can't read ({e}); convert the column "
            "in a query, e.g. CAST(<column> AS VARCHAR)"
        )
        raise ComputeError(msg) from None


def _object_columns_as_text(frames: Iterator[DataFrame]) -> Iterator[DataFrame]:
    """
    Batches without `Object` columns: a row driver's `uuid.UUID`, `ipaddress` or other
    Python values have no Arrow type (their export would carry pointers). They are read
    as text; `memoryview` and `bytearray` values (psycopg2's bytea) as binary.
    """
    import uuid
    from decimal import Decimal
    from ipaddress import _BaseAddress, _BaseNetwork

    import polars as pl
    from polars.exceptions import ComputeError

    as_text = (uuid.UUID, _BaseAddress, _BaseNetwork, Decimal)
    as_binary = (memoryview, bytearray, bytes)
    for df in frames:
        objects = [c for c, dt in df.schema.items() if dt == pl.Object]
        for name in objects:
            values = df[name].to_list()
            present = [v for v in values if v is not None]
            if present and all(isinstance(v, as_binary) for v in present):
                binary = [None if v is None else bytes(v) for v in values]
                df = df.with_columns(pl.Series(name, binary, pl.Binary))
                continue
            for v in present:
                if not isinstance(v, as_text):
                    msg = (
                        f"the driver returned column {name!r} as Python "
                        f"{type(v).__name__!r} values, which Polars can't read; "
                        "convert the column in a query, e.g. CAST(<column> AS VARCHAR)"
                    )
                    raise ComputeError(msg)
            df = df.with_columns(
                pl.Series(
                    name, [None if v is None else str(v) for v in values], pl.String
                )
            )
        yield df


def _params_keyword(cx: Any) -> str:
    """The keyword of a cursor's `execute()` that takes the bound parameters."""
    from inspect import Parameter, signature

    if cx.driver_name in ("sqlalchemy", "duckdb") or "oracledb" in cx.driver_name:
        return "parameters"  # the executor maps it
    try:
        ps = [
            p
            for p in signature(cx.cursor.execute).parameters.values()
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        # no signature (C methods: sqlite3, psycopg2): passed positionally
        return "parameters"
    # (query, parameters, ...): psycopg `params`, PyMySQL and pg8000 `args`
    return ps[1].name if len(ps) >= 2 else "parameters"


def _transaction_idle(conn: Any) -> bool | None:
    """Whether a connection is outside a transaction; `None` if unknown."""
    with contextlib.suppress(Exception):
        if (info := getattr(conn, "info", None)) is not None and hasattr(
            info, "transaction_status"
        ):
            return bool(info.transaction_status == 0)  # psycopg: IDLE
        if callable(status := getattr(conn, "get_transaction_status", None)):
            return bool(status() == 0)  # psycopg2: TRANSACTION_STATUS_IDLE
        in_transaction = getattr(conn, "in_transaction", None)
        if callable(in_transaction):
            # a SQLAlchemy Connection, which begins one on its first statement
            return not in_transaction()
        if isinstance(in_transaction, bool) and _module_of(conn) == "sqlite3":
            # (mysql.connector has the property too, but its connections can't be
            # used from two threads at once: the user's, and the scan's at its end)
            return not in_transaction
    return None


def _unbuffered_mysql(conn: Any, module: str) -> bool:
    """Whether a MySQL-family connection's cursors stream their result (unbuffered)."""
    with contextlib.suppress(Exception):
        if module in ("pymysql", "MySQLdb"):
            cursors = importlib.import_module(f"{module}.cursors")
            return issubclass(conn.cursorclass, cursors.SSCursor)
        if module == "mysql":
            # mysql.connector: unbuffered unless connected with `buffered=True`
            return not getattr(conn, "_buffered", False)
    return False


def _buffered_cursor(conn: Any, module: str) -> Any:
    """A cursor that reads its whole result on `execute()`."""
    if module == "mysql":
        return conn.cursor(buffered=True)
    cursors = importlib.import_module(f"{module}.cursors")
    return conn.cursor(cursors.Cursor)


def _exec_driver_sql(target: Any) -> Callable[..., Any] | None:
    """`exec_driver_sql` of a SQLAlchemy Connection, or of an ORM Session's."""
    if callable(f := getattr(target, "exec_driver_sql", None)):
        return cast("Callable[..., Any]", f)
    if type(target).__module__.startswith("sqlalchemy.orm") and callable(
        connection := getattr(target, "connection", None)
    ):
        with contextlib.suppress(Exception):
            return cast("Callable[..., Any]", connection().exec_driver_sql)
    return None


def _sql_str(value: Any, *, dialect: str) -> str:
    """A string literal, read the same whatever the session's string settings."""
    s = str(value).replace("'", "''")
    if "\\" not in s:
        return f"'{s}'"
    s = s.replace("\\", "\\\\")
    # MySQL's backslash escapes are on (the setup removes NO_BACKSLASH_ESCAPES); a
    # Postgres escape string doesn't depend on `standard_conforming_strings`
    return f"E'{s}'" if dialect == "postgresql" else f"'{s}'"


def _session_settings(conn: Any) -> list[str] | None:
    """
    Statements that restore what the session setup changes on a SQLAlchemy connection;
    `None` if unknown.
    """

    def row(sql: str) -> Any:
        options = {"no_parameters": True, "stream_results": False}
        result = conn.exec_driver_sql(sql, execution_options=options)
        try:
            return result.fetchone()
        finally:
            result.close()

    dialect = getattr(getattr(conn, "dialect", None), "name", "")
    try:
        if dialect == "postgresql":
            names = ("application_name", "TimeZone", "default_transaction_read_only")
            values = row(
                "SELECT " + ", ".join(f"current_setting('{n}')" for n in names)
            )
            # `set_config` is allowed in a read-only transaction; values are quoted
            # with standard strings
            sets = (
                f"set_config('{n}', {_sql_str(v, dialect='postgresql')}, false)"
                for n, v in zip(names, values, strict=True)
            )
            return ["SELECT " + ", ".join(sets)]
        if dialect in ("mysql", "mariadb"):
            tz, mode = row("SELECT @@SESSION.time_zone, @@SESSION.sql_mode")
            for var in ("transaction_read_only", "tx_read_only"):
                with contextlib.suppress(Exception):
                    (read_only,) = row(f"SELECT @@SESSION.{var}")
                    break
            else:
                return None
            # backslash escapes are on: the setup removes NO_BACKSLASH_ESCAPES
            return [
                f"SET SESSION time_zone = {_sql_str(tz, dialect='mysql')}, "
                f"SESSION sql_mode = {_sql_str(mode, dialect='mysql')}, "
                f"SESSION {var} = {int(read_only)}"
            ]
        if dialect == "oracle":
            rows = conn.exec_driver_sql(
                "SELECT parameter, value FROM nls_session_parameters "
                "WHERE parameter IN ('NLS_COMP', 'NLS_SORT')",
                execution_options={"stream_results": False},
            ).fetchall()
            values = {str(p): str(v) for p, v in rows}
            if sorted(values) != ["NLS_COMP", "NLS_SORT"] or not all(
                re.fullmatch(r"\w+", v) for v in values.values()
            ):
                return None
            return [
                f"ALTER SESSION SET NLS_COMP = {values['NLS_COMP']} "
                f"NLS_SORT = {values['NLS_SORT']}"
            ]
    except Exception:
        with contextlib.suppress(Exception):
            conn.rollback()
    return None


def _shutdown_socket(conn: Any) -> None:
    """Stops a PyMySQL connection's result: the server stops sending once it can't."""
    import socket

    if (sock := getattr(conn, "_sock", None)) is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


def _empty_frame(result: Any) -> DataFrame:
    import polars as pl

    names: list[str] = []
    dtypes: list[Any] = []
    with contextlib.suppress(Exception):
        if (description := getattr(result, "description", None)) is not None:
            names = [d[0] for d in description]
            dtypes = _description_dtypes(result, description)
        elif callable(keys := getattr(result, "keys", None)):
            names = [str(k) for k in cast("Any", keys())]
            # a SQLAlchemy result: the DB-API cursor has the types
            if (cursor := getattr(result, "cursor", None)) is not None and (
                description := getattr(cursor, "description", None)
            ) is not None:
                dtypes = _description_dtypes(cursor, description)
    if len(dtypes) != len(names):
        dtypes = [pl.Null] * len(names)
    return pl.DataFrame(schema=dict(zip(names, dtypes, strict=True)))


# Postgres type OIDs whose dtype is fixed, as ADBC reads them. `cursor.description` has
# them for an empty result too, where a row-wise driver has no values to type columns by.
_POSTGRES_OID_DTYPES = {
    16: "Boolean",  # bool
    17: "Binary",  # bytea
    18: "String",  # "char"
    19: "String",  # name
    20: "Int64",  # int8
    21: "Int16",  # int2
    23: "Int32",  # int4
    25: "String",  # text
    700: "Float32",  # float4
    701: "Float64",  # float8
    1042: "String",  # bpchar
    1043: "String",  # varchar
    1082: "Date",  # date
    1083: "Time",  # time
    1114: "Datetime",  # timestamp
    1184: "Datetime[UTC]",  # timestamptz (the session time zone is UTC)
    2950: "String",  # uuid, as a table's is read (as text)
}

# Postgres array OIDs, by their element's OID: lists of the element dtype.
_POSTGRES_ARRAY_OIDS = {
    1000: 16,
    1001: 17,
    1002: 18,
    1003: 19,
    1016: 20,
    1005: 21,
    1007: 23,
    1009: 25,
    1021: 700,
    1022: 701,
    1014: 1042,
    1015: 1043,
    1182: 1082,
    1183: 1083,
    1115: 1114,
    1185: 1184,
    2951: 2950,
}

# MySQL field types (`cursor.description` of PyMySQL, MySQLdb and mysql.connector), as
# these drivers read a non-empty result: every integer as Int64, `FLOAT` as a Python
# float, `TIMESTAMP` naive (`BIGINT UNSIGNED` as UInt64, see `_mysql_unsigned`). Others (strings, which may be binary; decimals; `TIME`,
# read as a duration) are typed from values.
_MYSQL_FIELD_DTYPES = {
    1: "Int64",  # TINY
    2: "Int64",  # SHORT
    3: "Int64",  # LONG
    4: "Float64",  # FLOAT
    5: "Float64",  # DOUBLE
    7: "Datetime",  # TIMESTAMP
    8: "Int64",  # LONGLONG
    9: "Int64",  # INT24
    10: "Date",  # DATE
    12: "Datetime",  # DATETIME
    13: "Int64",  # YEAR
    14: "Date",  # NEWDATE
}


def _mysql_unsigned(cursor: Any, description: Any, i: int) -> bool | None:
    """Whether a MySQL result column is `UNSIGNED`; `None` if unknown."""
    unsigned_flag = 32
    with contextlib.suppress(Exception):
        if _module_of(cursor) == "mysql":
            # mysql.connector: the column flags are the description's 8th item
            return bool(description[i][7] & unsigned_flag)
        if _module_of(cursor) == "pymysql":
            return bool(cursor._result.fields[i].flags & unsigned_flag)
    return None


def _description_dtypes(cursor: Any, description: Any) -> list[Any]:
    """Dtypes of a Postgres- or MySQL-family result's columns, from their type codes,
    which an empty result has too; `Null` (typed later from a sample) for other types
    and other drivers."""
    import polars as pl

    def named(name: str) -> Any:
        if name.startswith("Datetime"):
            return pl.Datetime("us", "UTC" if name.endswith("[UTC]") else None)
        return getattr(pl, name)

    def postgres(d: Any) -> Any:
        type_code = d[1]
        if type(type_code) is not int:
            return pl.Null
        if type_code == 1700:  # numeric(p, s); without a precision, typed from values
            p, s = d[4], d[5]
            if type(p) is int and type(s) is int and 0 < p <= 38 and 0 <= s <= p:
                return pl.Decimal(p, s)
            return pl.Null
        if (element := _POSTGRES_ARRAY_OIDS.get(type_code)) is not None:
            return pl.List(named(_POSTGRES_OID_DTYPES[element]))
        name = _POSTGRES_OID_DTYPES.get(type_code)
        return pl.Null if name is None else named(name)

    def mysql(i: int, d: Any) -> Any:
        name = _MYSQL_FIELD_DTYPES.get(d[1]) if type(d[1]) is int else None
        if d[1] == 8:  # LONGLONG: `BIGINT UNSIGNED` goes past Int64
            unsigned = _mysql_unsigned(cursor, description, i)
            if unsigned is None:
                return pl.Null  # typed from values
            if unsigned:
                return pl.UInt64
        return pl.Null if name is None else named(name)

    module = _module_of(cursor)
    if module in ("psycopg", "psycopg2", "pg8000"):
        return [postgres(d) for d in description]
    if module in ("pymysql", "MySQLdb", "mysql"):
        return [mysql(i, d) for i, d in enumerate(description)]
    return [pl.Null] * len(description)


# Names of server-side cursors: unique in the process, as a cursor left open in a live
# connection's transaction (see `_Session.execute`) may outlive its session.
_CURSOR_IDS = itertools.count()


def _close_if_open(conn: Any, name: str) -> None:
    """
    Closes a psycopg server-side cursor if it still exists: a cursor lives in its
    transaction, so then that is still the scan's (one statement, so nothing else runs
    between the check and the CLOSE). Nothing is begun on an idle connection.
    """
    with contextlib.suppress(Exception):
        if conn.info.transaction_status != 2:  # INTRANS
            return
        conn.execute(
            "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_catalog.pg_cursors "
            f"WHERE name = '{name}') THEN EXECUTE 'CLOSE {name}'; END IF; END $$"
        )


class _ConnectionState:
    """Shared by every session over one connection."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # executions in progress, and whether the first found no transaction open
        # (`None`: unknown)
        self.active = 0
        self.idle_before: bool | None = False


# A live connection is the user's one object: the scans of a query (a self-join, a
# union) each have a session over it and read at the same time, from other threads.
_LIVE_STATES: weakref.WeakKeyDictionary[Any, _ConnectionState] = (
    weakref.WeakKeyDictionary()
)
_LIVE_STATES_BY_ID: dict[int, _ConnectionState] = {}
_LIVE_STATES_LOCK = threading.Lock()


def _live_state(conn: Any) -> _ConnectionState:
    with _LIVE_STATES_LOCK:
        try:
            return _LIVE_STATES.setdefault(conn, _ConnectionState())
        except TypeError:
            # not weak-referenceable or not hashable: kept for the process
            return _LIVE_STATES_BY_ID.setdefault(id(conn), _ConnectionState())


class _Session:
    """A connection used by one partition reader at a time."""

    can_cancel = False

    def __init__(
        self,
        conn: Any,
        *,
        owned: bool,
        execute_options: dict[str, Any],
        close: Callable[[], None] | None = None,
        params: Any = None,
    ) -> None:
        self.conn = conn
        self.owned = owned
        self.execute_options = execute_options
        self.params = params
        self._close = close
        self._state = _ConnectionState() if owned else _live_state(conn)
        # the session setup changed the connection's settings (see `close`)
        self._setup_ran = False
        # statements that restore what the setup changed on a pooled connection
        # (`None`: unknown, so the connection is discarded instead)
        self._restore: list[str] | None = None
        # stopped by `cancel` through its socket: the connection is unusable
        self._aborted = False
        module = _module_of(conn)
        # PyMySQL has no cancel, and closing its unbuffered cursor reads the rest of
        # the result; a cancelled session is never reused, so its socket can go
        self._abort_socket = owned and module == "pymysql"
        # A live connection is never cancelled: a cancel stops whatever runs on it (the
        # user's next statement, once ours ended), and on Postgres aborts the
        # transaction. Its scans' cursors are closed instead.
        self.can_cancel = owned and (
            self._abort_socket
            or any(callable(getattr(conn, m, None)) for m in ("cancel", "interrupt"))
        )
        # SQLAlchemy's `text()` reads `:name` and `\:` inside the string literals
        # written into the SQL; without parameters, the SQL goes to the driver as is
        self._raw_sqlalchemy = (
            module == "sqlalchemy"
            and params is None
            and "parameters" not in execute_options
            and "params" not in execute_options
        )

    def execute(self, sql: str, batch_size: int | None) -> Iterator[DataFrame]:
        from polars.io.database._executor import ConnectionExecutor

        batch_size = batch_size or DEFAULT_BATCH_SIZE
        state = self._state
        entered = False
        cursor = None
        cx_cursor: Any = None
        cx: Any = None
        stopped = False
        frames: Iterator[DataFrame] | None = None
        try:
            # driver calls hold the lock, yields don't: other scans of the same live
            # connection interleave their calls, and none waits for another's consumer
            with state.lock:
                if state.active == 0:
                    # a live connection that wasn't in a transaction is left without one
                    state.idle_before = (
                        None if self.owned else _transaction_idle(self.conn)
                    )
                state.active += 1
                entered = True
                cursor = self._server_cursor(batch_size)
                cx = ConnectionExecutor(cursor if cursor is not None else self.conn)
                cx_cursor = cx.cursor if cx.cursor is not cursor else None
                options = dict(self.execute_options)
                oracle_arrow = (
                    cx.driver_name == "oracledb_proxy"
                    or cx._oracle_arrow_proxy() is not None
                )
                if oracle_arrow:
                    # NUMBER(p, s) as decimals, not floats, like the other drivers
                    options.setdefault("fetch_decimals", True)
                if (
                    _module_of(self.conn) == "clickhouse_connect"
                    and "parameters" not in options
                    and self.params is None
                ):
                    # it %-formats the SQL even without parameters, which would turn
                    # LIKE's '%%' into '%'; with parameters, '%' is the user's to write
                    sql = sql.replace("%", "%%")
                if self.params is not None:
                    # each driver names the parameters argument of `execute()` itself
                    options[_params_keyword(cx)] = self.params
                exec_driver_sql = (
                    _exec_driver_sql(cx.cursor)
                    if self._raw_sqlalchemy
                    and not oracle_arrow
                    and cx.driver_name in ("sqlalchemy", "duckdb")
                    else None
                )
                if exec_driver_sql is not None:
                    # as `cx.execute`, but without `text()` (see `__init__`); with
                    # `no_parameters`, `%`-style drivers don't format the SQL either
                    # (`text()` escaped `%` for them)
                    execution_options = options.pop("execution_options", None) or {}
                    result = exec_driver_sql(
                        sql,
                        execution_options={**execution_options, "no_parameters": True},
                        **options,
                    )
                    if cx.driver_name == "duckdb" and cx._is_alchemy_result(result):
                        result = result.cursor
                    cx.result = result
                else:
                    cx.execute(sql, options=options or None)
                if cursor is not None and cursor.description is None:
                    # psycopg2's named cursors describe the result only after a fetch;
                    # `FETCH FORWARD 0` returns no row
                    cursor.fetchmany(0)
                frames = _object_columns_as_text(
                    _no_panics(
                        cx.to_polars(  # type: ignore[arg-type]
                            iter_batches=True,
                            batch_size=batch_size,
                            # every row types a batch: a column whose first rows are NULL
                            # would fail on its first value otherwise
                            infer_schema_length=None,
                        )
                    )
                )
                # some drivers yield no batch for an empty result, and close the
                # cursor when done: take the column names first
                empty_frame = _empty_frame(cx.result)
            empty = True
            while True:
                with state.lock:
                    df = next(frames, None)
                if df is None:
                    break
                empty = False
                try:
                    yield df
                except GeneratorExit:
                    stopped = True
                    raise
            if empty:
                yield empty_frame
        finally:
            # also runs when the reader stops early and drops this generator
            # Polars releases an early-stopped read after `collect()` returned, while the
            # user may run statements on a live connection again: the scan's
            # transaction and server-side cursor are then left for the user's commit
            # or rollback to end, as a CLOSE or ROLLBACK now would hit theirs.
            late = stopped and not self.owned
            with state.lock:
                if (
                    self._aborted
                    and (result := getattr(self.conn, "_result", None)) is not None
                ):
                    # the socket is gone: nothing more to read (closing the cursor,
                    # or the result's `__del__`, would try)
                    result.unbuffered_active = False
                if frames is not None:
                    with contextlib.suppress(Exception):
                        frames.close()  # type: ignore[attr-defined]
                if cursor is not None and late and _module_of(cursor) == "psycopg":
                    # psycopg would CLOSE it even in a transaction the user began
                    # since (aborting it): closed only if it's still there
                    # (psycopg2 skips the CLOSE itself once its transaction ended)
                    cursor._closed = True
                    _close_if_open(self.conn, cursor.name)
                elif cursor is not None:
                    with contextlib.suppress(Exception):
                        cursor.close()
                if (
                    cx is not None
                    and _module_of(cx.result) == "sqlalchemy"
                    and callable(getattr(cx.result, "close", None))
                ):
                    # a SQLAlchemy result (a server-side cursor with `stream_results`)
                    # left unfinished: closed before its connection goes on
                    with contextlib.suppress(Exception):
                        cx.result.close()
                if cx_cursor is not None and cx_cursor is not self.conn:
                    # the executor's own cursor (or an Engine's connection), which it
                    # closes only after the last batch
                    with contextlib.suppress(Exception):
                        cx_cursor.close()
                if entered:
                    state.active -= 1
                if (self.owned and cursor is not None and not self._aborted) or (
                    state.active == 0
                    and state.idle_before
                    and not late
                    and _transaction_idle(self.conn) is False
                ):
                    with contextlib.suppress(Exception):
                        # the query opened a transaction (a server-side cursor lives
                        # in one), holding a snapshot and locks: end it, once no other
                        # scan reads in it. A transaction the user had open is theirs.
                        self.conn.rollback()

    def _reset_pooled(self) -> None:
        """
        A pooled connection (an Engine's) goes back to the user's pool: the settings
        the session setup changed (READ ONLY, UTC, sql_mode, ...) are restored, or the
        connection is discarded.
        """
        conn = self.conn
        if not callable(invalidate := getattr(conn, "invalidate", None)):
            return
        if self._restore is not None:
            try:
                conn.rollback()
                for sql in self._restore:
                    # not on the server-side cursor that `stream_results` would open
                    options = {"no_parameters": True, "stream_results": False}
                    conn.exec_driver_sql(sql, execution_options=options).close()
                conn.commit()
            except Exception:
                pass
            else:
                return
        with contextlib.suppress(Exception):
            invalidate()

    def _server_cursor(self, batch_size: int) -> Any:
        """A server-side cursor, so rows stream instead of being fetched up front."""
        module = _module_of(self.conn)
        if not self.owned:
            # a named cursor needs a transaction; in autocommit it would have to be
            # declared WITH HOLD, which materializes the whole result
            if module in ("psycopg", "psycopg2") and not getattr(
                self.conn, "autocommit", True
            ):
                cursor = self.conn.cursor(name=f"polars_scan_{next(_CURSOR_IDS)}")
                cursor.itersize = batch_size
                return cursor
            if _unbuffered_mysql(self.conn, module):
                # MySQL can't keep two results open on one connection: another scan's
                # statement would end this one's result early, silently with PyMySQL
                return _buffered_cursor(self.conn, module)
            return None
        if module in ("psycopg", "psycopg2"):
            cursor = self.conn.cursor(name=f"polars_scan_{next(_CURSOR_IDS)}")
            cursor.itersize = batch_size
            return cursor
        if module == "pymysql":
            import pymysql.cursors  # type: ignore[import-untyped]

            return self.conn.cursor(pymysql.cursors.SSCursor)
        if module == "sqlalchemy" and callable(
            opts := getattr(self.conn, "execution_options", None)
        ):
            # the dialect's own streaming cursor (named, SSCursor, ...)
            opts(stream_results=True, max_row_buffer=batch_size)
        return None

    def execute_update(self, sql: str) -> None:
        """Run a session setup statement (read-only transactions, time zone, ...)."""
        conn = self.conn
        if not self.owned:
            # a live connection is the user's: its session settings are left alone
            return
        if _module_of(conn) == "sqlalchemy":
            if not self._setup_ran and self._close is None:
                # an Engine's pooled connection: its settings are restored on `close`
                self._restore = _session_settings(conn)
            self._setup_ran = True
            conn.exec_driver_sql(sql)
            conn.commit()
            return
        self._setup_ran = True
        cursor = conn.cursor() if hasattr(conn, "cursor") else conn
        try:
            cursor.execute(sql)
        finally:
            if cursor is not conn:
                with contextlib.suppress(Exception):
                    cursor.close()
        # drivers that open a transaction implicitly (psycopg, pymysql, ...) apply
        # session characteristics like READ ONLY to the transactions after this one
        if callable(commit := getattr(conn, "commit", None)):
            with contextlib.suppress(Exception):
                commit()

    def cancel(self) -> None:
        if not self.owned:
            # see `can_cancel`
            return
        if self._abort_socket:
            self._aborted = True
            _shutdown_socket(self.conn)
            return
        for m in ("cancel", "interrupt"):
            if callable(f := getattr(self.conn, m, None)):
                f()
                return

    def close(self) -> None:
        if self._setup_ran and self._close is None:
            self._reset_pooled()
        if self._close is not None:
            self._close()
        elif self.owned and callable(getattr(self.conn, "close", None)):
            if self._aborted:
                with contextlib.suppress(Exception):
                    self.conn.close()
            else:
                self.conn.close()


# (URI, protocol) pairs ConnectorX has connected to in this process
_CONNECTORX_CHECKED: set[tuple[str, str | None]] = set()
_CONNECTORX_CHECKED_LOCK = threading.Lock()


class _ConnectorXSession:
    """ConnectorX has no sessions: every query opens its own connections."""

    can_cancel = False

    def __init__(self, uri: str, dialect: str, protocol: str | None = None) -> None:
        self.uri = uri
        self.dialect = dialect
        self.protocol = protocol
        self._check_connection()

    def _check_connection(self) -> None:
        """
        Connects once per URI: a failed login surfaces here, after ConnectorX's ~30s
        connect timeout, instead of after the planner's catalog lookups (which carry on
        past a failure) each waited for it.
        """
        if self.dialect == "sqlite":
            # ConnectorX opens SQLite files read-only, never creating one: a missing
            # file fails each connect after the same wait
            path = unquote(self.uri.removeprefix("sqlite://"))
            if path != ":memory:" and not os.path.isfile(path):
                msg = f"no SQLite database file at {path!r}"
                raise FileNotFoundError(msg)
            return
        key = (self.uri, self.protocol)
        with _CONNECTORX_CHECKED_LOCK:
            if key in _CONNECTORX_CHECKED:
                return
        from polars.io.database._utils import _read_sql_connectorx

        sql = "SELECT 1 FROM DUAL" if self.dialect == "oracle" else "SELECT 1"
        _read_sql_connectorx(sql, connection_uri=self.uri, protocol=self.protocol)
        with _CONNECTORX_CHECKED_LOCK:
            _CONNECTORX_CHECKED.add(key)

    def execute(self, sql: str, batch_size: int | None) -> Iterator[DataFrame]:
        # ConnectorX returns whole results; there is no batch size to apply
        from polars.io.database._utils import _read_sql_connectorx

        try:
            # ConnectorX connects per query: its errors can echo the URI
            with _scrubbed(_secrets(self.uri, None, {})):
                df = _read_sql_connectorx(
                    sql, connection_uri=self.uri, protocol=self.protocol
                )
        except RuntimeError as e:
            # ConnectorX can't type an empty SQLite result (or a leading NULL); sqlite3
            # reads the same file
            if self.dialect != "sqlite" or "Cannot infer type" not in str(e):
                raise
            # `sqlite://<path>` (see `_connectorx_uri`) as SQLAlchemy's form
            sqlite_uri = "sqlite:///" + self.uri.removeprefix("sqlite://")
            conn = _connect_uri("sqlite3", sqlite_uri, {})
            try:
                yield from _Session(conn, owned=True, execute_options={}).execute(
                    sql, batch_size
                )
            finally:
                conn.close()
            return
        yield df

    def execute_update(self, sql: str) -> None:
        msg = "connectorx can't run session statements"
        raise NotImplementedError(msg)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class _ArrowOdbcSession:
    """`arrow-odbc` reads batches straight from an ODBC connection string."""

    can_cancel = False

    def __init__(self, connection_string: str, options: dict[str, Any]) -> None:
        self.connection_string = connection_string
        self.options = options

    def execute(self, sql: str, batch_size: int | None) -> Iterator[DataFrame]:
        import arrow_odbc

        import polars as pl

        reader = arrow_odbc.read_arrow_batches_from_odbc(
            query=sql,
            connection_string=self.connection_string,
            batch_size=batch_size or DEFAULT_BATCH_SIZE,
            **self.options,
        )
        for batch in reader:
            yield pl.DataFrame(batch)

    def execute_update(self, sql: str) -> None:
        msg = "arrow-odbc can't run session statements"
        raise NotImplementedError(msg)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class _AdbcSession:
    """
    An ADBC connection opened through `adbc_driver_manager`.

    Used when `POLARS_DATABASE_ADBC_LOADER=python`. Rust finds the driver and builds
    the options, as for its own loader; results go back as the driver's Arrow C
    stream, which Rust reads without the GIL.
    """

    def __init__(
        self,
        driver: str,
        entrypoint: str | None,
        options: list[tuple[str, str]],
        execute_options: list[tuple[str, str]],
    ) -> None:
        try:
            from adbc_driver_manager import AdbcConnection, AdbcDatabase
            from adbc_driver_manager._lib import ArrowArrayStreamHandle
        except ImportError:
            msg = (
                "POLARS_DATABASE_ADBC_LOADER=python needs adbc_driver_manager: "
                "pip install adbc-driver-manager"
            )
            raise ModuleNotFoundError(msg) from None
        if not hasattr(ArrowArrayStreamHandle, "__arrow_c_stream__"):
            msg = (
                "POLARS_DATABASE_ADBC_LOADER=python needs a newer adbc_driver_manager "
                f"(found {_version('adbc_driver_manager')}): "
                "pip install -U adbc-driver-manager"
            )
            raise ModuleNotFoundError(msg)

        # the driver manager's own arguments: as options they would load another driver
        reserved = {"driver", "entrypoint", "load_flags", "init_func"} & dict(
            options
        ).keys()
        if reserved:
            msg = (
                f"connect option(s) {sorted(reserved)} can't be used with "
                "POLARS_DATABASE_ADBC_LOADER=python"
            )
            raise ValueError(msg)
        kwargs: dict[str, Any] = {"driver": driver, **dict(options)}
        if entrypoint:
            kwargs["entrypoint"] = entrypoint
        self._db = AdbcDatabase(**kwargs)
        try:
            self._conn = AdbcConnection(self._db)
        except BaseException:
            self._db.close()
            raise
        self._execute_options = dict(execute_options)
        self._lock = threading.Lock()
        # held while a statement or the connection is cancelled or closed: the driver
        # manager calls the driver for both without a lock of its own, and a cancel must
        # never run on a handle being released
        self._handle_lock = threading.Lock()
        # statements whose result streams Rust still reads; the connection closes after
        # the last one
        self._open: list[Any] = []
        self._closed = False

    def _statement(self, sql: str, hint: tuple[str, str] | None) -> Any:
        from adbc_driver_manager import AdbcStatement

        stmt = AdbcStatement(self._conn)
        try:
            if hint is not None:
                # a hint: drivers that reject it still work
                with contextlib.suppress(Exception):
                    stmt.set_options(**{hint[0]: hint[1]})
            if self._execute_options:
                stmt.set_options(**self._execute_options)
            stmt.set_sql_query(sql)
        except BaseException:
            stmt.close()
            raise
        return stmt

    def execute(self, sql: str, hint: tuple[str, str] | None) -> tuple[Any, Any]:
        """Run a query: `(Arrow C stream capsule, statement)`; see `release`."""
        stmt = self._statement(sql, hint)
        with self._lock:
            self._open.append(stmt)
        try:
            handle, _ = stmt.execute_query()
            return handle.__arrow_c_stream__(), stmt
        except BaseException:
            self.release(stmt)
            raise

    def execute_update(self, sql: str) -> None:
        stmt = self._statement(sql, None)
        with self._lock:
            self._open.append(stmt)
        try:
            stmt.execute_update()
        finally:
            self.release(stmt)

    def release(self, stmt: Any) -> None:
        """Close a statement once its result stream is released."""
        with self._lock:
            with contextlib.suppress(ValueError):
                self._open.remove(stmt)
            close_conn = self._closed and not self._open
        try:
            with self._handle_lock:
                stmt.close()
        finally:
            if close_conn:
                self._close_connection()

    def cancel(self) -> None:
        from adbc_driver_manager import AdbcStatusCode, NotSupportedError

        with self._lock:
            stmt = self._open[-1] if self._open else None
        if stmt is None:
            return
        try:
            with self._handle_lock:
                # released meanwhile: nothing to cancel
                if stmt not in self._open:
                    return
                try:
                    stmt.cancel()
                except NotSupportedError:
                    try:
                        self._conn.cancel()
                    except NotSupportedError:
                        # an ADBC 1.0 driver: it can't cancel, as with the Rust loader
                        return
        except Exception as e:
            # nothing running: nothing to cancel
            if getattr(e, "status_code", None) != AdbcStatusCode.INVALID_STATE:
                raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._open:
                return
        self._close_connection()

    def _close_connection(self) -> None:
        with self._handle_lock:
            try:
                self._conn.close()
            finally:
                self._db.close()


def adbc_connect(
    driver: str,
    entrypoint: str | None,
    options: list[tuple[str, str]],
    execute_options: list[tuple[str, str]],
) -> _AdbcSession:
    """Open an ADBC connection through `adbc_driver_manager`."""
    return _AdbcSession(driver, entrypoint, options, execute_options)


def connect(
    kind: str,
    target_kind: str,
    target: Any,
    connect_options: list[tuple[str, str]],
    execute_options: list[tuple[str, str]],
    dialect: str,
    credential_provider: Callable[[], Any] | None,
    params: Any = None,
    python_options: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> Any:
    """Opens a session for a Python backend; the batch size is given per query."""
    # the values as given in Python: driver keyword arguments aren't all strings
    connect_opts: dict[str, Any]
    execute_opts: dict[str, Any]
    if python_options is not None:
        connect_opts, execute_opts = dict(python_options[0]), dict(python_options[1])
    else:
        connect_opts, execute_opts = dict(connect_options), dict(execute_options)
    # every statement of a query source has the user query and its placeholders
    session = {"execute_options": execute_opts, "params": params}
    head, _, tail = kind.partition(":")

    if target_kind == "live":
        return _Session(target, owned=False, **session)
    if target_kind == "factory":
        return _Session(target(), owned=True, **session)

    creds = credentials(credential_provider) if credential_provider else None
    with _scrubbed(_secrets(target, creds, connect_opts)):
        return _open(kind, target_kind, target, creds, connect_opts, dialect, session)


def _open(
    kind: str,
    target_kind: str,
    target: str,
    creds: tuple[str | None, str | None] | None,
    connect_opts: dict[str, Any],
    dialect: str,
    session: dict[str, Any],
) -> Any:
    """A session for a URI or ODBC connection string."""
    head, _, tail = kind.partition(":")
    if target_kind == "odbc":
        if tail == "arrow_odbc":
            return _ArrowOdbcSession(target, connect_opts)
        mod = importlib.import_module(tail)
        if tail == "turbodbc":
            conn = mod.connect(connection_string=target, **connect_opts)
        else:
            conn = mod.connect(target, **connect_opts)
        return _Session(conn, owned=True, **session)

    uri = _with_connect_timeout(_with_credentials(target, creds), dialect, kind)
    if head == "connectorx":
        unknown = sorted(set(connect_opts) - {"protocol"}) + sorted(
            session["execute_options"]
        )
        if unknown:
            msg = (
                f"connectorx takes no options {unknown}: only "
                "connect_options={'protocol': ...}"
            )
            raise ValueError(msg)
        return _ConnectorXSession(
            _connectorx_uri(uri, dialect),
            dialect,
            protocol=connect_opts.get("protocol"),
        )
    if head == "sqlalchemy":
        from sqlalchemy import create_engine

        if dialect == "sqlite":
            uri = _sqlalchemy_sqlite_uri(uri)
        try:
            engine = create_engine(uri, **connect_opts)
        except Exception as e:
            if type(e).__name__ != "NoSuchModuleError":
                raise
            scheme = uri.split(":", 1)[0]
            msg = (
                f"SQLAlchemy has no dialect for {scheme!r}; install the SQLAlchemy "
                "dialect package for this database, pin another driver with "
                "driver=..., or pass a connection factory"
            )
            raise ValueError(msg) from None
        conn = engine.connect()

        def close() -> None:
            conn.close()
            engine.dispose()

        return _Session(conn, owned=True, close=close, **session)
    if head == "python":
        conn = _connect_uri(tail, uri, connect_opts)
        return _Session(conn, owned=True, **session)

    msg = f"no Python backend for {kind!r} (dialect {dialect!r})"
    raise ValueError(msg)
