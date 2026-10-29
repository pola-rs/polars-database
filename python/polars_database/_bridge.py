"""
Python database backends for `scan_database`, called from Rust.

Rust picks the backend and runs the SQL it compiled; this module only opens connections
for Python drivers and streams results back as DataFrames. Query execution reuses the
`ConnectionExecutor` that `read_database` uses, so every driver it supports works here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import threading
import weakref
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

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


def _with_credentials(uri: str, creds: tuple[str | None, str | None] | None) -> str:
    if not creds:
        return uri
    user, password = creds
    parts = urlsplit(uri)
    host = parts.netloc.rsplit("@", 1)[-1]
    old_user = unquote(parts.username or "")
    user = user if user is not None else old_user
    auth = quote(user, safe="")
    if password is not None:
        auth += ":" + quote(password, safe="")
    return urlunsplit(parts._replace(netloc=f"{auth}@{host}" if auth else host))


# Seconds before a connect attempt gives up, where the URI sets none: libpq waits for
# the operating system's TCP timeout (minutes) by default, and attempts are retried.
DEFAULT_CONNECT_TIMEOUT = 10


def _with_connect_timeout(uri: str, dialect: str) -> str:
    if dialect not in ("postgresql", "redshift") or "connect_timeout=" in uri:
        return uri
    sep = "&" if "?" in uri else "?"
    return f"{uri}{sep}connect_timeout={DEFAULT_CONNECT_TIMEOUT}"


def _strip_driver(uri: str) -> str:
    scheme, sep, rest = uri.partition(":")
    return scheme.split("+", 1)[0] + sep + rest


def _file_path(uri: str) -> str:
    # `sqlite:///path` -> `/path`; `sqlite://` -> ':memory:'
    rest = re.sub(r"^\w+(\+\w+)?:(//)?", "", uri)
    if rest in ("", "/", ":memory:", "/:memory:"):
        return ":memory:"
    # `sqlite:///C:/path` -> `C:/path`: on Windows, `/C:/path` is a network path
    if re.match(r"^/[A-Za-z]:[\\/]", rest):
        return rest[1:]
    return rest


def _url_parts(uri: str) -> dict[str, Any]:
    parts = urlsplit(_strip_driver(uri))
    return {
        "host": parts.hostname,
        "port": parts.port,
        "user": unquote(parts.username) if parts.username else None,
        "password": unquote(parts.password) if parts.password else None,
        "database": unquote(parts.path.lstrip("/")) or None,
    }


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _connect_uri(module: str, uri: str, options: dict[str, Any]) -> Any:
    """A DB-API connection to a URI through a driver module."""
    mod = importlib.import_module(module)
    p = _url_parts(uri)
    if module == "sqlite3":
        # sessions move between threads; each is used by one thread at a time
        return mod.connect(_file_path(uri), check_same_thread=False, **options)
    if module == "duckdb":
        return mod.connect(_file_path(uri), **options)
    if module == "psycopg":
        return mod.connect(_strip_driver(uri), **options)
    if module == "psycopg2":
        conn = mod.connect(_strip_driver(uri), **options)
        # bytea as bytes, not memoryview (which would become an Object column)
        ext = importlib.import_module("psycopg2.extensions")
        as_bytes = ext.new_type(
            mod.BINARY.values,
            "POLARS_BYTES",
            lambda v, cur: None if v is None else mod.BINARY(v, cur).tobytes(),
        )
        ext.register_type(as_bytes, conn)
        return conn
    if module == "pg8000":
        return mod.connect(**_drop_none(**p), **options)
    if module == "pymysql":
        return mod.connect(**_drop_none(**p), **options)
    if module == "MySQLdb":
        kw = _drop_none(host=p["host"], port=p["port"], user=p["user"])
        kw.update(_drop_none(passwd=p["password"], db=p["database"]))
        return mod.connect(**kw, **options)
    if module == "mysql.connector":
        return mod.connect(**_drop_none(**p), **options)
    if module == "mariadb":
        return mod.connect(**_drop_none(**p), **options)
    if module == "pymssql":
        kw = _drop_none(server=p["host"], port=p["port"], user=p["user"])
        kw.update(_drop_none(password=p["password"], database=p["database"]))
        return mod.connect(**kw, **options)
    if module == "oracledb":
        # `/name` is the service name, as is `?service_name=name` (SQLAlchemy's form)
        query = dict(parse_qsl(urlsplit(uri).query))
        service = query.get("service_name") or p["database"] or ""
        dsn = f"{p['host']}:{p['port'] or 1521}/{service}"
        return mod.connect(
            **_drop_none(user=p["user"], password=p["password"]), dsn=dsn, **options
        )
    if module == "redshift_connector":
        return mod.connect(**_drop_none(**p), **options)
    if module == "trino":
        from trino.dbapi import connect  # type: ignore[import-not-found]

        catalog, _, schema = (p["database"] or "").partition("/")
        kw = _drop_none(host=p["host"], port=p["port"], user=p["user"])
        kw.update(_drop_none(catalog=catalog or None, schema=schema or None))
        return connect(**kw, **options)
    if module == "clickhouse_connect":
        from clickhouse_connect.dbapi import connect  # type: ignore[import-not-found]

        kw = _drop_none(host=p["host"], port=p["port"], username=p["user"])
        kw.update(_drop_none(password=p["password"], database=p["database"]))
        return connect(**kw, **options)
    msg = (
        f"scan_database can't open a {module!r} connection from a URI; "
        "pass a connection factory instead, e.g. `lambda: <module>.connect(...)`"
    )
    raise NotImplementedError(msg)


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
    """Whether a psycopg(2) connection is outside a transaction; `None` if unknown."""
    with contextlib.suppress(Exception):
        if (info := getattr(conn, "info", None)) is not None and hasattr(
            info, "transaction_status"
        ):
            return bool(info.transaction_status == 0)  # IDLE
        if callable(status := getattr(conn, "get_transaction_status", None)):
            return bool(status() == 0)  # TRANSACTION_STATUS_IDLE
    return None


def _empty_frame(result: Any) -> DataFrame:
    import polars as pl

    names: list[str] = []
    with contextlib.suppress(Exception):
        if (description := getattr(result, "description", None)) is not None:
            names = [d[0] for d in description]
        elif callable(keys := getattr(result, "keys", None)):
            names = [str(k) for k in cast("Any", keys())]
    return pl.DataFrame(schema=dict.fromkeys(names, pl.Null))


class _ConnectionState:
    """Shared by every session over one connection."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # executions in progress, and whether the first found no transaction open
        self.active = 0
        self.idle_before = False


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
        self._cursors = 0
        self._state = _ConnectionState() if owned else _live_state(conn)
        self.can_cancel = any(
            callable(getattr(conn, m, None)) for m in ("cancel", "interrupt")
        )

    def execute(self, sql: str, batch_size: int | None) -> Iterator[DataFrame]:
        from polars.io.database._executor import ConnectionExecutor

        batch_size = batch_size or DEFAULT_BATCH_SIZE
        state = self._state
        entered = False
        cursor = None
        frames: Iterator[DataFrame] | None = None
        try:
            # driver calls hold the lock, yields don't: other scans of the same live
            # connection interleave their calls, and none waits for another's consumer
            with state.lock:
                if state.active == 0:
                    # a live connection that wasn't in a transaction is left without one
                    state.idle_before = not self.owned and bool(
                        _transaction_idle(self.conn)
                    )
                state.active += 1
                entered = True
                cursor = self._server_cursor(batch_size)
                cx = ConnectionExecutor(cursor if cursor is not None else self.conn)
                options = dict(self.execute_options)
                if (
                    cx.driver_name == "oracledb_proxy"
                    or cx._oracle_arrow_proxy() is not None
                ):
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
                cx.execute(sql, options=options or None)
                if cursor is not None and cursor.description is None:
                    # psycopg2's named cursors describe the result only after a fetch;
                    # `FETCH FORWARD 0` returns no row
                    cursor.fetchmany(0)
                frames = _no_panics(
                    cx.to_polars(iter_batches=True, batch_size=batch_size)  # type: ignore[arg-type]
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
                yield df
            if empty:
                yield empty_frame
        finally:
            # also runs when the reader stops early and drops this generator
            with state.lock:
                if frames is not None:
                    with contextlib.suppress(Exception):
                        frames.close()  # type: ignore[attr-defined]
                if cursor is not None:
                    with contextlib.suppress(Exception):
                        cursor.close()
                if entered:
                    state.active -= 1
                if (self.owned and cursor is not None) or (
                    state.active == 0
                    and state.idle_before
                    and _transaction_idle(self.conn) is False
                ):
                    with contextlib.suppress(Exception):
                        # the query opened a transaction (a server-side cursor lives
                        # in one), holding a snapshot and locks: end it, once no other
                        # scan reads in it. A transaction the user had open is theirs.
                        self.conn.rollback()

    def _server_cursor(self, batch_size: int) -> Any:
        """A server-side cursor, so rows stream instead of being fetched up front."""
        module = _module_of(self.conn)
        if not self.owned:
            # a named cursor needs a transaction; in autocommit it would have to be
            # declared WITH HOLD, which materializes the whole result
            if module in ("psycopg", "psycopg2") and not getattr(
                self.conn, "autocommit", True
            ):
                self._cursors += 1
                cursor = self.conn.cursor(
                    name=f"polars_scan_{id(self)}_{self._cursors}"
                )
                cursor.itersize = batch_size
                return cursor
            return None
        if module in ("psycopg", "psycopg2"):
            self._cursors += 1
            cursor = self.conn.cursor(name=f"polars_scan_{self._cursors}")
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
            conn.exec_driver_sql(sql)
            conn.commit()
            return
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
        for m in ("cancel", "interrupt"):
            if callable(f := getattr(self.conn, m, None)):
                f()
                return

    def close(self) -> None:
        if self._close is not None:
            self._close()
        elif self.owned and callable(getattr(self.conn, "close", None)):
            self.conn.close()


class _ConnectorXSession:
    """ConnectorX has no sessions: every query opens its own connections."""

    can_cancel = False

    def __init__(self, uri: str, dialect: str, protocol: str | None = None) -> None:
        self.uri = uri
        self.dialect = dialect
        self.protocol = protocol

    def execute(self, sql: str, batch_size: int | None) -> Iterator[DataFrame]:
        # ConnectorX returns whole results; there is no batch size to apply
        from polars.io.database._utils import _read_sql_connectorx

        try:
            df = _read_sql_connectorx(
                sql, connection_uri=self.uri, protocol=self.protocol
            )
        except RuntimeError as e:
            # ConnectorX can't type an empty SQLite result (or a leading NULL); sqlite3
            # reads the same file
            if self.dialect != "sqlite" or "Cannot infer type" not in str(e):
                raise
            conn = _connect_uri("sqlite3", self.uri, {})
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
    if target_kind == "odbc":
        if tail == "arrow_odbc":
            return _ArrowOdbcSession(target, connect_opts)
        mod = importlib.import_module(tail)
        if tail == "turbodbc":
            conn = mod.connect(connection_string=target, **connect_opts)
        else:
            conn = mod.connect(target, **connect_opts)
        return _Session(conn, owned=True, **session)

    uri = _with_connect_timeout(_with_credentials(target, creds), dialect)
    if head == "connectorx":
        unknown = sorted(set(connect_opts) - {"protocol"}) + sorted(execute_opts)
        if unknown:
            msg = (
                f"connectorx takes no options {unknown}: only "
                "connect_options={'protocol': ...}"
            )
            raise ValueError(msg)
        return _ConnectorXSession(
            _strip_driver(uri), dialect, protocol=connect_opts.get("protocol")
        )
    if head == "sqlalchemy":
        from sqlalchemy import create_engine

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
