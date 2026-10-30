"""
`scan_database` against real database servers: the differential suite.

These run only when server URIs are given:

    POLARS_TEST_POSTGRES_URI, POLARS_TEST_POSTGRES14_URI, POLARS_TEST_MYSQL_URI,
    POLARS_TEST_MARIADB_URI, POLARS_TEST_ORACLE_URI, POLARS_TEST_CLICKHOUSE_URI

Every (server, driver) cell must pass: a failing cell is a bug, and an xfail would hide
a regression in it. Cells that can't run here are skipped with the reason
(`KNOWN_UNRUNNABLE`), at once rather than after driver timeouts.

By default the `live` cell of each server runs the whole predicate corpus, and the
other cells what they plan differently (see `_scan_corpus`).
`POLARS_TEST_DATABASE_FULL=1` runs it on every cell, as the nightly job does.

A server whose URI is set but which doesn't accept connections fails the run when the
tests are collected; `POLARS_TEST_DATABASE_SKIP_UNREACHABLE=1` skips its tests instead.

Run them in parallel (`pytest -n 4`); `test_scan_load.py` runs on its own, serially.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, date
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests._scan_corpus import (
    CORPUS_MODES,
    FULL,
    N_EDGE_ROWS,
    SAMPLE,
    CorpusPlans,
    cell_predicates,
    check_corpus,
    corpus_for_mode,
    expected_ids,
    ids,
    predicate_strategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytest.importorskip("sqlglot")

# Random predicate trees per server; the nightly job raises it.
N_EXAMPLES = int(
    os.environ.get("POLARS_TEST_DATABASE_EXAMPLES", "200" if FULL else "50")
)

# The cell that runs the whole corpus: a live connection doesn't connect per query, and
# plans the same SQL as the other cells (the SQL is the dialect's, not the driver's)
BASE_CELL = "live"


def _parts(uri: str) -> dict[str, Any]:
    p = urlsplit(uri)
    return {
        "host": p.hostname,
        "port": p.port,
        "user": p.username,
        "password": p.password,
        "database": p.path.lstrip("/"),
    }


def _pg_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import psycopg

        return psycopg.connect(uri)

    return connect


def _mysql_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import pymysql  # type: ignore[import-untyped]

        return pymysql.connect(**_parts(uri))

    return connect


def _oracle_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        import oracledb

        p = _parts(uri)
        return oracledb.connect(
            user=p["user"],
            password=p["password"],
            dsn=f"{p['host']}:{p['port']}/FREEPDB1",
        )

    return connect


def _clickhouse_factory(uri: str) -> Callable[[], Any]:
    def connect() -> Any:
        from clickhouse_connect.dbapi import connect

        p = _parts(uri)
        return connect(
            host=p["host"],
            port=p["port"],
            username=p["user"],
            password=p["password"],
            database=p["database"],
        )

    return connect


@functools.cache
def _reachable(uri: str) -> bool:
    """
    Whether the server accepts connections, checked once per run.

    A server that is down skips its tests at once, instead of each one waiting for
    connect retries and driver timeouts.
    """
    import socket

    p = urlsplit(uri)
    scheme = p.scheme.split("+", 1)[0]
    port = p.port or _DEFAULT_PORTS.get(scheme)
    if not p.hostname or port is None:
        # a socket path or an unknown scheme: let the tests find out
        return True
    try:
        with socket.create_connection((p.hostname, port), timeout=2):
            return True
    except OSError:
        return False


_DEFAULT_PORTS = {
    "postgresql": 5432,
    "postgres": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "oracle": 1521,
    "clickhouse": 8123,
}


@dataclass
class Server:
    """A server of the backend matrix and its driver cells."""

    env: str
    # the edge table as a query, in the dialect's quoting
    query: str
    # driver pin -> blocking?
    drivers: dict[str, bool]
    factory: Callable[[str], Callable[[], Any]] | None = None
    factory_blocking: bool = False
    live_blocking: bool = False
    # a driver pin that needs a different URI form (SQLAlchemy dialect+driver)
    uri_for: dict[str, Callable[[str], str]] = field(default_factory=dict)

    @property
    def uri(self) -> str | None:
        """The server's URI, or None when it isn't given or isn't reachable."""
        uri = os.environ.get(self.env)
        if uri is None or _reachable(uri):
            return uri
        if os.environ.get("POLARS_TEST_DATABASE_SKIP_UNREACHABLE"):
            return None
        # a server that didn't start must not turn its tests into skips
        msg = (
            f"{self.env} is set, but the server is not reachable "
            "(POLARS_TEST_DATABASE_SKIP_UNREACHABLE=1 skips its tests)"
        )
        raise RuntimeError(msg)


SERVERS = {
    "postgres17": Server(
        "POLARS_TEST_POSTGRES_URI",
        "SELECT * FROM edge",
        {
            "adbc": True,
            "connectorx": True,
            "psycopg": True,
            "psycopg2": False,
            "pg8000": False,
            "sqlalchemy": True,
        },
        factory=_pg_factory,
        factory_blocking=True,
        live_blocking=True,
    ),
    "postgres14": Server(
        "POLARS_TEST_POSTGRES14_URI",
        "SELECT * FROM edge",
        {"adbc": True, "connectorx": False, "psycopg": False, "sqlalchemy": False},
        factory=_pg_factory,
    ),
    "mysql": Server(
        "POLARS_TEST_MYSQL_URI",
        "SELECT * FROM edge",
        {
            "adbc": True,
            "connectorx": True,
            "pymysql": True,
            "MySQLdb": False,
            "mysql.connector": False,
            "sqlalchemy": True,
        },
        factory=_mysql_factory,
        factory_blocking=True,
        uri_for={"sqlalchemy": lambda u: u.replace("mysql://", "mysql+pymysql://", 1)},
    ),
    "mariadb": Server(
        "POLARS_TEST_MARIADB_URI",
        "SELECT * FROM edge",
        {"adbc": False, "connectorx": False, "pymysql": False, "mariadb": False},
        factory=_mysql_factory,
    ),
    "oracle": Server(
        "POLARS_TEST_ORACLE_URI",
        'SELECT * FROM "edge"',
        {"oracledb": False, "connectorx": False, "sqlalchemy": False},
        factory=_oracle_factory,
        uri_for={
            "sqlalchemy": lambda u: u.replace("oracle://", "oracle+oracledb://", 1)
        },
    ),
    "clickhouse": Server(
        "POLARS_TEST_CLICKHOUSE_URI",
        "SELECT * FROM edge",
        {"adbc": False, "clickhouse_connect": False, "connectorx": False},
        factory=_clickhouse_factory,
    ),
}

DRIVER_MODULES = {
    "adbc": None,
    "connectorx": "connectorx",
    "sqlalchemy": "sqlalchemy",
    "clickhouse_connect": "clickhouse_connect",
    "mysql.connector": "mysql.connector",
}


# Oracle Free's listener refuses bursts of connections (ORA-12516) beyond what connect
# retries absorb: its tests run on one xdist worker, one at a time.
ORACLE_GROUP = pytest.mark.xdist_group("oracle")


def _server_marks(name: str) -> list[Any]:
    return [ORACLE_GROUP] if name == "oracle" else []


def _no_oracle_client() -> str | None:
    import ctypes.util

    if ctypes.util.find_library("clntsh") is None and not os.environ.get("ORACLE_HOME"):
        return "ConnectorX reads Oracle through Oracle Instant Client, not installed"
    return None


# Cells that may not run here: (server, driver) -> the reason, or None if it can.
KNOWN_UNRUNNABLE: dict[tuple[str, str], Callable[[], str | None]] = {
    ("oracle", "connectorx"): _no_oracle_client,
}


def _cells(*, connections: bool = True, group: bool = False) -> list[Any]:
    """
    (server, driver) pairs; `factory` and `live` are connection forms.

    `group` keeps the tests of a server on one xdist worker, where they share the cached
    references, baselines and plans of its cells, the base cell's above all.
    """
    out = []
    for name, server in SERVERS.items():
        cells = list(server.drivers.items())
        if connections and server.factory is not None:
            cells += [
                ("factory", server.factory_blocking),
                ("live", server.live_blocking),
            ]
        for driver, _blocking in cells:
            marks = [
                pytest.mark.skipif(
                    server.uri is None, reason=f"{server.env} unset or unreachable"
                )
            ]
            marks += _server_marks(name)
            if group and name != "oracle":
                marks.append(pytest.mark.xdist_group(f"corpus-{name}"))
            unrunnable = KNOWN_UNRUNNABLE.get((name, driver))
            if unrunnable is not None and (reason := unrunnable()) is not None:
                marks.append(pytest.mark.skip(reason=reason))
            out.append(pytest.param(name, driver, id=f"{name}-{driver}", marks=marks))
    return out


# One live connection per server, shared by the module's tests (closed after them).
_LIVE: dict[str, Any] = {}


def _live(name: str) -> Any:
    if name not in _LIVE:
        server = SERVERS[name]
        assert server.factory is not None
        assert server.uri is not None
        _LIVE[name] = server.factory(server.uri)()
    return _LIVE[name]


@pytest.fixture(scope="module", autouse=True)
def _close_live_connections() -> Iterator[None]:
    yield
    for conn in _LIVE.values():
        with contextlib.suppress(Exception):
            conn.close()
    _LIVE.clear()


def _scan(name: str, driver: str, source: str = "edge", **kwargs: Any) -> pl.LazyFrame:
    server = SERVERS[name]
    uri = server.uri
    assert uri is not None
    if name == "oracle" and source.isidentifier() and source.islower():
        # the shared tables are created quoted lower-case; a bare name means upper case
        source = f'"{source}"'
    if driver == "factory":
        assert server.factory is not None
        return pl.scan_database(source, server.factory(uri), **kwargs)
    if driver == "live":
        return pl.scan_database(source, _live(name), **kwargs)
    module = DRIVER_MODULES.get(driver, driver)
    if module is not None:
        pytest.importorskip(module)
    convert = server.uri_for.get(driver)
    if convert is not None:
        uri = convert(uri)
    return pl.scan_database(source, uri, driver=driver, **kwargs)


@functools.cache
def _reference(name: str, driver: str) -> pl.DataFrame:
    # the table as read without pushdown; Polars then does all the filtering
    return _scan(name, driver, pushdown=False, partition_on=None).collect()


@functools.cache
def _plans(name: str, driver: str, source: str) -> CorpusPlans:
    lf = _scan(name, driver, source, partition_on=None)
    return CorpusPlans.of(lf, _reference(name, driver))


def _default_driver(name: str) -> str:
    return next(iter(SERVERS[name].drivers))


@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_pushdown_off_reads_every_row(name: str, driver: str) -> None:
    # harness self-check: the reference really is the whole table
    assert ids(_reference(name, driver)) == list(range(N_EDGE_ROWS))


def _corpus_cells() -> list[Any]:
    """
    (server, driver, mode) of the corpus test.

    Outside the full run, partitioned modes run on the base cell only: the other cells
    plan the same SQL plus a range, and `test_partitions` reads partitions with each
    driver. `factory` cells don't run: they read through the driver module of a pinned
    cell, and the schema and round-trip tests cover the connection form.
    """
    return [
        pytest.param(*cell.values, mode, id=f"{cell.id}-{mode}", marks=cell.marks)
        for cell in _cells(group=True)
        for mode in CORPUS_MODES
        if FULL
        or cell.values[1] == BASE_CELL
        or (cell.values[1] != "factory" and not mode.endswith("p4"))
    ]


@pytest.mark.parametrize(("name", "driver", "mode"), _corpus_cells())
def test_differential_corpus(name: str, driver: str, mode: str) -> None:
    """Pushed predicates may only ever widen what the database returns."""
    server = SERVERS[name]
    source = server.query if mode.startswith("query") else "edge"
    kwargs: dict[str, Any] = (
        {"partition_on": "id", "partition_num": 4}
        if mode.endswith("p4")
        else {"partition_on": None}
    )
    reference = _reference(name, driver)
    lf = _scan(name, driver, source, **kwargs)
    predicates = corpus_for_mode(mode)
    # the base cell's table runs the whole corpus; its query is compared with it
    if not FULL and (driver != BASE_CELL or source != "edge"):
        base = (BASE_CELL, source) if driver != BASE_CELL else (BASE_CELL, "edge")
        predicates = cell_predicates(
            mode,
            _plans(name, driver, source),
            _plans(name, *base),
            sample=SAMPLE[mode],
            offset=[*server.drivers, "factory", BASE_CELL].index(driver),
        )
    failures, stats = check_corpus(
        lambda: lf, reference, predicates, check_exact=mode == "table"
    )
    if mode == "table":
        print(f"\n{name}-{driver}: {stats}")
    assert not failures, f"of {len(predicates)}:\n" + "\n".join(failures[:40])


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=[
                pytest.mark.skipif(
                    s.uri is None, reason=f"{s.env} unset or unreachable"
                ),
                *_server_marks(n),
            ],
        )
        for n, s in SERVERS.items()
    ],
)
def test_property(name: str) -> None:
    """Random predicate trees; a failing example belongs in the fixed corpus."""
    from hypothesis import HealthCheck, given, settings

    driver = _default_driver(name) if FULL else BASE_CELL
    reference = _reference(name, driver)
    lf = _scan(name, driver, partition_on=None)

    @settings(
        max_examples=N_EXAMPLES,
        deadline=None,
        suppress_health_check=list(HealthCheck),
        database=None,
    )
    @given(predicate_strategy(boolean_column=reference["b"].dtype == pl.Boolean))
    def check(p: pl.Expr) -> None:
        expected = expected_ids(reference, p)
        if isinstance(expected, Exception):
            with pytest.raises(Exception):  # noqa: B017
                lf.filter(p).select("id").collect()
            return
        assert ids(lf.filter(p).select("id").collect()) == expected, str(p)

    check()


@pytest.mark.parametrize("table", ["edge", "orders"])
@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_same_schema_for_every_driver(name: str, driver: str, table: str) -> None:
    # the dialect fixes the dtypes, not the driver
    assert _scan(name, driver, table).collect_schema() == _baseline_schema(name, table)


@functools.cache
def _baseline_schema(name: str, table: str) -> pl.Schema:
    return _scan(name, _default_driver(name), table).collect_schema()


# The rows `test_partitions` reads: all of `orders` (100k) in the full run, else a
# pushed range of them, still split over every partition
ORDERS = pl.lit(value=True) if FULL else pl.col("id") < 5_000


@pytest.mark.parametrize(("name", "driver"), _cells(connections=False))
def test_partitions(name: str, driver: str) -> None:
    lf = _scan(
        name, driver, "orders", partition_on="id", partition_num=8, max_connections=4
    ).filter(ORDERS)
    assert "partitions: 8" in lf.explain()
    reference = _orders(name)
    assert lf.select(pl.len()).collect().item() == reference.height
    assert_frame_equal(lf.collect().sort("id"), reference, check_dtypes=False)


@functools.cache
def _orders(name: str) -> pl.DataFrame:
    lf = _scan(name, _default_driver(name), "orders", partition_on=None)
    return lf.filter(ORDERS).collect().sort("id")


POSTGRES = SERVERS["postgres17"].uri
needs_postgres = pytest.mark.skipif(POSTGRES is None, reason="no Postgres URI")


def active_polars_queries(uri: str, marker: str = "") -> int:
    """
    Active Polars statements, only those whose SQL holds `marker` if given.

    Tests run in parallel: other tests' statements may be running too.
    """
    like = f" AND query LIKE '%{marker}%' AND pid != pg_backend_pid()" if marker else ""
    return int(
        pl.read_database_uri(
            "SELECT COUNT(*) AS n FROM pg_stat_activity "
            f"WHERE application_name = 'polars' AND state = 'active'{like}",
            uri,
        ).item()
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_cancel_on_early_stop(driver: str) -> None:
    assert POSTGRES is not None
    marker = f"cancel_on_early_stop_{driver}"
    slow = (
        f"SELECT i AS id, pg_sleep(0.002) IS NULL AS {marker} "
        "FROM generate_series(1, 5000) AS i"
    )
    start = time.monotonic()
    out = (
        _scan(
            "postgres17",
            driver,
            slow,
            batch_size=100,
            schema={"id": pl.Int32, marker: pl.Boolean},
        )
        .filter(pl.col("id") > 0)
        .head(5)
        .collect()
    )
    assert out.height == 5
    # the whole query takes 10s; stopping early must not wait for it (the bound leaves
    # room for the other workers' load on the same server)
    assert time.monotonic() - start < 9
    # the cancel reaches the server asynchronously: wait a little, not 10s
    deadline = time.monotonic() + 3
    while active_polars_queries(POSTGRES, marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert active_polars_queries(POSTGRES, marker) == 0


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_credential_provider(driver: str) -> None:
    assert POSTGRES is not None
    password = urlsplit(POSTGRES).password
    without_password = POSTGRES.replace(f":{password}@", "@", 1)
    lf = pl.scan_database(
        "orders",
        without_password,
        driver=driver,
        credential_provider=lambda: {"password": password},
    )
    assert lf.select(pl.len()).collect().item() > 0
    assert str(password) not in lf.explain()


# Declared types and their dtypes, per dialect, whichever driver reads them
EDGE_TYPES = {
    "postgres17": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        # unconstrained `numeric` with 12 decimal places: Decimal(38, 9) would round it,
        # which is an error; read as text, it is the same for every driver
        "nu": pl.String,
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.Time,
    },
    "mysql": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        "nu": pl.Decimal(30, 12),
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.Time,
    },
    "oracle": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.Binary,
        "nu": pl.String,
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.String,
    },
    "clickhouse": {
        "j": pl.String,
        "u": pl.String,
        "bin": pl.String,
        "nu": pl.Decimal(38, 12),
        "tstz": pl.Datetime("us", "UTC"),
        "t": pl.String,
    },
}
EDGE_TYPES["postgres14"] = EDGE_TYPES["postgres17"]
EDGE_TYPES["mariadb"] = EDGE_TYPES["mysql"]


@pytest.mark.parametrize(("name", "driver"), _cells(group=True))
def test_edge_types_round_trip(name: str, driver: str) -> None:
    schema = _scan_edge_types(name, driver).collect_schema()
    expected = {"id": schema["id"], **EDGE_TYPES[name]}
    assert dict(schema) == expected
    got = _scan_edge_types(name, driver).collect().sort("id")
    assert_frame_equal(got, _edge_types(name))


def _scan_edge_types(name: str, driver: str) -> pl.LazyFrame:
    # unconstrained `numeric` / `NUMBER` with 12 decimal places: read as text
    kwargs: dict[str, Any] = (
        {"schema_overrides": {"nu": pl.String}}
        if name.startswith("postgres") or name == "oracle"
        else {}
    )
    return _scan(name, driver, "edge_types", **kwargs)


@functools.cache
def _edge_types(name: str) -> pl.DataFrame:
    return _scan_edge_types(name, _default_driver(name)).collect().sort("id")


# Pushdown regressions: each check compares a pushed-down
# filter with the same filter applied by Polars to the whole table.


def _execute(name: str, statements: list[str]) -> None:
    server = SERVERS[name]
    if server.uri is None:
        pytest.skip(f"no {name} URI")
    assert server.factory is not None
    conn = server.factory(server.uri)()
    try:
        cur = conn.cursor()
        for sql in statements:
            cur.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _check_filters(lf: pl.LazyFrame, predicates: list[pl.Expr]) -> None:
    table = lf.collect()
    for p in predicates:
        got = lf.filter(p).collect().sort("id")
        expected = table.filter(p).sort("id")
        assert got["id"].to_list() == expected["id"].to_list(), str(p)


@needs_postgres
def test_postgres_converted_and_padded_columns() -> None:
    from decimal import Decimal

    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_pg",
            "DROP COLLATION IF EXISTS review_ci",
            "CREATE COLLATION review_ci "
            "(provider = icu, locale = 'und-u-ks-level2', deterministic = false)",
            "CREATE TABLE review_pg (id int, u uuid, j jsonb, ch char(3), "
            "ci text COLLATE review_ci, nu numeric, d numeric(10,2))",
            "INSERT INTO review_pg VALUES "
            "(1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{\"a\":1}', 'a', 'abc', "
            "0.100000001, 1.50), "
            "(2, 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{\"b\":2}', 'b  ', 'ABC', 0.1, 2.00), "
            "(3, NULL, NULL, NULL, 'x', 0.2, 3.00)",
        ],
    )
    for driver in ["adbc", "psycopg"]:
        lf = _scan("postgres17", driver, "review_pg")
        _check_filters(
            lf,
            [
                pl.col("u") == "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",
                pl.col("u") != "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
                pl.col("u").str.starts_with("a0"),
                pl.col("j") == '{"a":1}',
                pl.col("ch") == "a",
                pl.col("ch") != "a",
                pl.col("ci") == "abc",
                pl.col("ci") != "abc",
                ~pl.col("ci").is_in(["abc"]),
                pl.col("nu") == Decimal("0.1"),
                pl.col("nu") > Decimal("0.1"),
                pl.col("d") < pl.lit(Decimal("1.501")),
                pl.col("d") > pl.lit(Decimal("1.499")),
            ],
        )


@needs_postgres
def test_postgres_scans_differing_in_options_are_not_merged() -> None:
    _execute(
        "postgres17",
        [
            "CREATE SCHEMA IF NOT EXISTS review_s1",
            "CREATE SCHEMA IF NOT EXISTS review_s2",
            "DROP TABLE IF EXISTS review_s1.t",
            "DROP TABLE IF EXISTS review_s2.t",
            "CREATE TABLE review_s1.t AS SELECT 's1' AS s",
            "CREATE TABLE review_s2.t AS SELECT 's2' AS s",
        ],
    )

    def scan(schema: str) -> pl.LazyFrame:
        return _scan(
            "postgres17",
            "psycopg",
            "SELECT s FROM t",
            connect_options={"options": f"-c search_path={schema}"},
        )

    both = pl.concat([scan("review_s1"), scan("review_s2")])
    for engine in ["in-memory", "streaming"]:
        assert both.collect(engine=engine)["s"].to_list() == ["s1", "s2"]  # type: ignore[call-overload]


@needs_postgres
def test_postgres_bare_table_names_fold_to_lower_case() -> None:
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS e_fold",
            'DROP TABLE IF EXISTS "E_Fold_Quoted"',
            "CREATE TABLE e_fold AS SELECT 1 AS id",
            'CREATE TABLE "E_Fold_Quoted" AS SELECT 2 AS id',
        ],
    )
    for driver in ["adbc", "psycopg"]:
        assert _scan("postgres17", driver, "E_Fold").collect()["id"].to_list() == [1]
        assert _scan("postgres17", driver, "Public.E_FOLD").collect().height == 1
        quoted = _scan("postgres17", driver, '"E_Fold_Quoted"')
        assert quoted.collect()["id"].to_list() == [2]


@needs_postgres
def test_postgres_bare_names_created_quoted_are_found_as_written() -> None:
    _execute(
        "postgres17",
        [
            'DROP TABLE IF EXISTS "E_Written"',
            'CREATE TABLE "E_Written" AS SELECT 1 AS id',
        ],
    )
    for driver in ["adbc", "psycopg"]:
        assert _scan("postgres17", driver, "E_Written").collect()["id"].to_list() == [1]
        # a user schema means no lookup: the name as written
        lf = _scan("postgres17", driver, "E_Written", schema={"id": pl.Int32})
        assert lf.collect()["id"].to_list() == [1]


@needs_postgres
def test_postgres_credential_provider_with_an_at_in_a_parameter() -> None:
    assert POSTGRES is not None
    uri = POSTGRES + ("&" if "?" in POSTGRES else "?") + "application_name=a@b"
    for driver in ["adbc", "psycopg"]:
        lf = pl.scan_database(
            "SELECT 1 AS one",
            uri,
            driver=driver,
            credential_provider=lambda: {"user": urlsplit(POSTGRES).username},
        )
        assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_bare_table_names_fold_to_upper_case() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE e_fold'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE e_fold ("id" NUMBER(9))',
            "INSERT INTO e_fold VALUES (1)",
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE \"e_written\"'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE "e_written" ("id" NUMBER(9))',
            'INSERT INTO "e_written" VALUES (1)',
        ],
    )
    # a bare name created quoted is found as written
    for source in ["e_fold", "E_Fold", '"E_FOLD"', "e_written"]:
        lf = pl.scan_database(source, SERVERS["oracle"].uri)
        assert lf.collect()["id"].to_list() == [1], source


@needs_postgres
def test_postgres_date_partitions_past_year_9999() -> None:
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS e_far_dates",
            "CREATE TABLE e_far_dates (id int, d date)",
            "INSERT INTO e_far_dates VALUES (1, '2000-01-01'), (2, '5000-06-01'), "
            "(3, '20000-01-01'), (4, '900000-01-01'), (5, NULL)",
        ],
    )
    lf = _scan("postgres17", "adbc", "e_far_dates", partition_on="d", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3, 4, 5]


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_far_date_literals() -> None:
    lf = _scan("mysql", "adbc", "edge")
    far = pl.lit(2_933_897).cast(pl.Date)
    old = pl.lit(date(1, 6, 1))
    _check_filters(lf, [pl.col("d") < far, pl.col("d") != far, pl.col("d") > old])


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_char_and_float_columns() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora ("id" NUMBER(9), ch CHAR(3), fl FLOAT)',
            "INSERT INTO review_ora VALUES (1, 'a', 0.1000000000000000001)",
            "INSERT INTO review_ora VALUES (2, 'b', 0.1)",
            "INSERT INTO review_ora VALUES (3, NULL, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_ORA")
    _check_filters(
        lf,
        [
            pl.col("CH") == "a",
            pl.col("CH") != "a",
            pl.col("FL") == 0.1,
            pl.col("FL") > 0.1,
            pl.col("FL") != 0.1,
        ],
    )


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
@pytest.mark.parametrize("driver", ["adbc", "clickhouse_connect"])
def test_clickhouse_datetime_and_enum(driver: str) -> None:
    from datetime import datetime

    table = f"review_ch_{driver}"
    _execute(
        "clickhouse",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id Int32, dt DateTime('UTC'), "
            "e Enum8('a' = 1, 'b' = 2)) ENGINE = MergeTree ORDER BY id",
            f"INSERT INTO {table} VALUES "
            "(1, '2020-01-02 03:04:05', 'a'), (2, '2021-01-01 00:00:00', 'b')",
        ],
    )
    lf = _scan("clickhouse", driver, table)
    assert lf.collect_schema()["dt"] == pl.Datetime("ms", "UTC")
    assert lf.collect_schema()["e"] == pl.String
    df = lf.collect().sort("id")
    assert df["dt"][0] == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert df["e"].to_list() == ["a", "b"]
    _check_filters(
        lf,
        [
            pl.col("dt") > datetime(2020, 6, 1, tzinfo=UTC),
            pl.col("e") == "b",
        ],
    )


@needs_postgres
def test_postgres_live_connection_left_outside_a_transaction() -> None:
    import psycopg

    assert POSTGRES is not None
    with psycopg.connect(POSTGRES) as conn:
        assert pl.scan_database("orders", conn).select(pl.len()).collect().item() > 0
        assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
        # a transaction the user opened is theirs: it stays open
        conn.execute("SELECT 1")
        pl.scan_database("orders", conn).head(1).collect()
        status = conn.info.transaction_status
        assert status == psycopg.pq.TransactionStatus.INTRANS  # type: ignore[comparison-overlap]


# `params=` with each Postgres and MySQL driver: placeholders, and a pushed filter
# around the query that would hold a `%`
PARAM_CELLS = [
    ("postgres17", d) for d in ["psycopg", "psycopg2", "pg8000", "sqlalchemy"]
] + [("mysql", d) for d in ["pymysql", "mysql.connector", "sqlalchemy"]]
# clickhouse_connect %-formats every query: the query's own placeholders must survive
PARAM_CELLS += [("clickhouse", "clickhouse_connect")]


@pytest.mark.parametrize(("name", "driver"), PARAM_CELLS)
def test_params(name: str, driver: str) -> None:
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    placeholder = ":n" if driver == "sqlalchemy" else "%s"
    params: Any = {"n": 30} if driver == "sqlalchemy" else [30]
    lf = _scan(
        name,
        driver,
        f"SELECT id, s FROM edge WHERE id < {placeholder}",
        params=params,
        partition_on=None,
    )
    out = lf.filter(pl.col("id") >= 10, pl.col("s").str.starts_with("a")).collect()
    expected = _reference(name, _default_driver(name)).filter(
        pl.col("id").is_between(10, 29), pl.col("s").str.starts_with("a")
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@pytest.mark.parametrize("name", ["mysql", "postgres17"])
def test_scans_sharing_a_live_connection(name: str) -> None:
    # each scan has its own session over the one connection, and they read at once
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    conn = _live(name)
    lo = pl.scan_database("edge", conn).filter(pl.col("id") < 40)
    hi = pl.scan_database("edge", conn).filter(pl.col("id") >= 10)
    out = lo.select("id", "s").join(hi.select("id"), on="id").collect()
    expected = _reference(name, _default_driver(name)).filter(
        pl.col("id").is_between(10, 39)
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_live_connection_without_backslash_escapes() -> None:
    # a live session is the user's: its sql_mode stays, so literals that need
    # backslash escapes aren't pushed (a connection of its own: it changes sql_mode)
    assert SERVERS["mysql"].uri is not None
    conn = _mysql_factory(SERVERS["mysql"].uri)()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET SESSION sql_mode = "
                "CONCAT(@@SESSION.sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
        expected = _reference("mysql", _default_driver("mysql"))
        for predicate in [
            pl.col("s") == "\\",
            pl.col("s").is_in(["\\", "abc"]),
            pl.col("s").str.contains("\\", literal=True),
            pl.col("s").str.starts_with("a"),
        ]:
            out = pl.scan_database("edge", conn).filter(predicate).collect()
            want = expected.filter(predicate)
            assert sorted(out["id"].to_list()) == sorted(want["id"].to_list())
    finally:
        conn.close()


# Types outside the ones every driver reads natively are read as text (or a fixed
# dtype), so all drivers agree; none of them may crash or return other values.
UNUSUAL_TYPES: dict[str, tuple[list[str], str, list[str]]] = {
    "postgres17": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "DROP TYPE IF EXISTS review_mood CASCADE",
            "CREATE EXTENSION IF NOT EXISTS citext",
            "CREATE TYPE review_mood AS ENUM ('sad', 'ok')",
            "CREATE TABLE review_unusual (id int, ci citext, ip inet, mac macaddr, "
            "iv interval, m money, arr int[], e review_mood, b bit(4), r int4range, "
            "oi oid, w numeric(50, 10), n numeric)",
            "INSERT INTO review_unusual VALUES (1, 'AbC', '10.0.0.1/24', "
            "'08:00:2b:01:02:03', '1 day 2 hours', 12.34, '{1,2}', 'ok', B'1010', "
            "'[1,5)', 42, 1234567890123456789012345678901234567890.0123456789, "
            "12345678901234567890.123)",
        ],
        "review_unusual",
        ["adbc", "psycopg", "psycopg2", "connectorx", "pg8000", "sqlalchemy"],
    ),
    "mysql": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id int, y year, bt bit(4), st set('x','y'), "
            "g point, fl float, dc decimal(65, 30))",
            "INSERT INTO review_unusual VALUES (1, 2024, b'1010', 'x,y', "
            "ST_GeomFromText('POINT(1 2)'), 1.5, 1.123456789012345678901234567890)",
        ],
        "review_unusual",
        ["adbc", "pymysql", "connectorx", "mysql.connector", "sqlalchemy"],
    ),
    "clickhouse": (
        [
            "DROP TABLE IF EXISTS review_unusual",
            "CREATE TABLE review_unusual (id Int32, arr Array(Int32), "
            "mp Map(String, Int32), tp Tuple(Int32, String), ip IPv4, fs FixedString(3), "
            "i128 Int128, u256 UInt256, d256 Decimal(60, 10)) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_unusual VALUES (1, [1, 2], {'a': 1}, (1, 'x'), "
            "'10.0.0.1', 'abc', 170141183460469231731687303715884105727, 1, 1.5)",
        ],
        "review_unusual",
        ["adbc", "clickhouse_connect", "connectorx"],
    ),
    "oracle": (
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_unusual'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_unusual ("id" NUMBER(9), ids INTERVAL DAY TO SECOND, '
            "iym INTERVAL YEAR TO MONTH, ri ROWID, n NUMBER)",
            "INSERT INTO review_unusual VALUES (1, INTERVAL '1 02:03:04' DAY TO SECOND, "
            "INTERVAL '1-2' YEAR TO MONTH, NULL, 12345678901234567890.123)",
        ],
        "REVIEW_UNUSUAL",
        ["oracledb", "sqlalchemy"],
    ),
}


def _installed(module: str) -> bool:
    # `find_spec("a.b")` imports `a`, and raises when `a` is missing
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_server_marks(n)) for n in sorted(UNUSUAL_TYPES)]
)
def test_unusual_types_read_the_same_by_every_driver(name: str) -> None:
    statements, table, drivers = UNUSUAL_TYPES[name]
    _execute(name, statements)
    frames = {}
    for driver in drivers:
        module = DRIVER_MODULES.get(driver, driver)
        if module is not None and not _installed(module):
            continue
        frames[driver] = _scan(name, driver, table).collect()
    assert len(frames) >= 2, frames.keys()
    first, *rest = frames
    for driver in rest:
        assert_frame_equal(frames[driver], frames[first]), driver
    row = frames[first].row(0, named=True)
    assert frames[first].height == 1
    if name == "postgres17":
        assert row["iv"] == "1 day 02:00:00"
        assert row["w"] == "1234567890123456789012345678901234567890.0123456789"
        assert row["n"] == Decimal("12345678901234567890.123")
    if name == "oracle":
        assert row["N"] == Decimal("12345678901234567890.123")
    if name == "mysql":
        assert row["g"] == "POINT(1 2)"
        assert row["bt"] == "1010"


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "connectorx"])
def test_decimal_that_would_round_is_an_error(driver: str) -> None:
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS review_round_{driver}",
            f"CREATE TABLE review_round_{driver} (id int, n numeric, d numeric(20, 15))",
            f"INSERT INTO review_round_{driver} VALUES (1, 0.1000000001, 1.123456789012345)",
        ],
    )
    with pytest.raises(pl.exceptions.PolarsError, match="would round"):
        _scan("postgres17", driver, f"review_round_{driver}").select("n").collect()
    # a query has no declared types; ConnectorX writes every decimal with 10 places
    query = _scan("postgres17", driver, f"SELECT id, d FROM review_round_{driver}")
    if driver == "psycopg":
        assert query.collect()["d"].to_list() == [Decimal("1.123456789012345")]
    else:
        with pytest.raises(pl.exceptions.PolarsError, match="would round"):
            query.collect()


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_local_time_zone_and_numbers_on_a_live_session() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_tz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_tz ("id" NUMBER(9), tsl TIMESTAMP WITH LOCAL TIME ZONE, '
            "n NUMBER)",
            "INSERT INTO review_tz VALUES (1, TIMESTAMP '2020-01-02 03:04:05 +00:00', 0.5)",
        ],
    )
    server = SERVERS["oracle"]
    assert server.factory is not None
    assert server.uri is not None
    conn = server.factory(server.uri)()
    try:
        cur = conn.cursor()
        cur.execute("ALTER SESSION SET TIME_ZONE = 'Asia/Tokyo'")
        cur.execute("ALTER SESSION SET NLS_NUMERIC_CHARACTERS = ',.'")
        row = pl.scan_database("REVIEW_TZ", conn).collect().row(0)
    finally:
        conn.close()
    assert row == (1, datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), Decimal("0.5"))


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
def test_clickhouse_nan_in_a_float_partition_column_and_naive_timestamps() -> None:
    from datetime import datetime

    _execute(
        "clickhouse",
        [
            "DROP TABLE IF EXISTS review_nan",
            "CREATE TABLE review_nan (id Int32, f Float64, ts DateTime64(6)) "
            "ENGINE = MergeTree ORDER BY id",
            "INSERT INTO review_nan VALUES (1, 0, '2020-01-01 00:00:00'), "
            "(2, nan, '2020-01-02 00:00:00'), (3, 10, '2020-01-03 00:00:00')",
        ],
    )
    lf = _scan("clickhouse", "adbc", "review_nan", partition_on="f", partition_num=3)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3]
    # a literal without a zone would be read in the server's time zone: not pushed
    naive = _scan("clickhouse", "adbc", "review_nan").filter(
        pl.col("ts") >= datetime(2020, 1, 2)
    )
    assert "filters 0/1" in naive.explain()
    assert sorted(naive.collect()["id"].to_list()) == [2, 3]


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_ordered_read_with_more_partitions_than_sessions(
    driver: str, tmp_path: Path
) -> None:
    # An ordered sink consumes partition 0 first: a later partition must never hold a
    # session it waits for. Run in a subprocess, so a deadlock fails instead of hanging.
    import subprocess
    import sys

    assert POSTGRES is not None
    out = tmp_path / "out.parquet"
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database('SELECT i AS id, md5(i::text) AS s "
        f"FROM generate_series(1, 1000000) AS i', {POSTGRES!r}, driver={driver!r}, "
        "partition_on='id', partition_num=16, max_connections=4, batch_size=10000)\n"
        f"for _ in range({3 if FULL else 1}):\n"
        f"    lf.sink_parquet({str(out)!r})\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("the ordered partitioned read hung")
    assert pl.scan_parquet(out).select(pl.len()).collect().item() == 1_000_000


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
def test_mysql_year_literals(name: str) -> None:
    # MySQL reads an integer compared with YEAR as a year: 70 is 1970, 1 is 2001
    _execute(
        name,
        [
            "DROP TABLE IF EXISTS review_year",
            "CREATE TABLE review_year (id int, y year)",
            "INSERT INTO review_year VALUES (1, 1970), (2, 2001), (3, 2050), (4, 0), "
            "(5, NULL)",
        ],
    )
    predicates = [
        pl.col("y") == 70,
        pl.col("y") == 1,
        pl.col("y") > 50,
        pl.col("y") > 99,
        pl.col("y") < 1,
        pl.col("y").is_in([1, 70]),
        pl.col("y").is_in([5, 1970]),
        pl.col("y").is_between(1, 2100),
        pl.col("y") == 2001,
        pl.col("y") == 0,
        # every other integer compares as itself
        pl.col("y") > 100,
        pl.col("y") < 1000,
        pl.col("y") == 2156,
        pl.col("y") == -1,
        pl.col("y").is_between(100, 1950),
        pl.col("y").is_in([0, 1970]),
    ]
    _check_filters(_scan(name, "pymysql", "review_year"), predicates)
    # a query's columns read by a row driver have no type: any integer may be a YEAR,
    # so only literals 1-99 are kept from the database
    query = _scan(name, "pymysql", "SELECT id, y FROM review_year")
    _check_filters(query, predicates)
    assert "filters 1/1 (1 exact)" in query.filter(pl.col("id") == 500).explain()
    assert "filters 0/1" in query.filter(pl.col("id") == 50).explain()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "connectorx", "live"])
def test_postgres_literals_without_standard_conforming_strings(driver: str) -> None:
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_scs",
            "CREATE TABLE review_scs (id int, s text)",
            "INSERT INTO review_scs VALUES (1, 'a\\b'), (2, 'ab'), (3, '50%'), "
            "(4, 'a''b')",
        ],
    )
    off = "-c standard_conforming_strings=off"
    conn = None
    if driver == "live":
        # a connection of its own: it changes a session setting
        server = SERVERS["postgres17"]
        assert server.factory is not None
        assert server.uri is not None
        conn = server.factory(server.uri)()
        conn.execute("SET standard_conforming_strings = off")
        conn.commit()
        lf = pl.scan_database("review_scs", conn)
    elif driver == "psycopg":
        lf = _scan("postgres17", driver, "review_scs", connect_options={"options": off})
    else:
        assert POSTGRES is not None
        if driver == "connectorx":
            pytest.importorskip(driver)
        uri = f"{POSTGRES}?options={quote(off)}"
        lf = pl.scan_database("review_scs", uri, driver=driver)
    try:
        _check_filters(
            lf,
            [
                pl.col("s") == "a\\b",
                pl.col("s") != "a\\b",
                pl.col("s").str.contains("%", literal=True),
                pl.col("s").str.starts_with("a\\"),
                pl.col("s") == "a'b",
                # Postgres text can't hold NUL
                pl.col("s") == "a\0b",
                pl.col("s") != "a\0b",
            ],
        )
    finally:
        if conn is not None:
            conn.close()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_statement_timeout_is_not_retried(driver: str) -> None:
    timeout = "-c statement_timeout=300"
    if driver == "adbc":
        assert POSTGRES is not None
        lf = pl.scan_database(
            "SELECT pg_sleep(2) AS x", f"{POSTGRES}?options={quote(timeout)}"
        )
    else:
        lf = _scan(
            "postgres17",
            driver,
            "SELECT pg_sleep(2) AS x",
            connect_options={"options": timeout},
        )
    start = time.monotonic()
    with pytest.raises(pl.exceptions.PolarsError, match="statement timeout"):
        lf.collect()
    # retried, it would take 4 x 0.3 s plus 2.6 s of backoff
    assert time.monotonic() - start < 2


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
@pytest.mark.parametrize("connection", ["live", "factory"])
def test_oracle_linguistic_session(connection: str) -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_nls'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_nls ("id" NUMBER(9), s VARCHAR2(10))',
            "INSERT INTO review_nls VALUES (1, 'abc')",
            "INSERT INTO review_nls VALUES (2, 'ABC')",
            "INSERT INTO review_nls VALUES (3, 'x')",
        ],
    )
    assert SERVERS["oracle"].uri is not None
    factory = _oracle_factory(SERVERS["oracle"].uri)

    def linguistic() -> Any:
        conn = factory()
        conn.cursor().execute(
            "ALTER SESSION SET NLS_COMP = LINGUISTIC NLS_SORT = BINARY_CI"
        )
        return conn

    # a live session keeps its settings; a factory's sessions get the setup
    source = linguistic() if connection == "live" else linguistic
    lf = pl.scan_database("REVIEW_NLS", source)
    _check_filters(
        lf,
        [
            pl.col("S") == "abc",
            pl.col("S") != "abc",
            pl.col("S").is_in(["abc"]),
            pl.col("S").str.starts_with("ab"),
            pl.col("S") > "a",
        ],
    )


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_time_zone_offsets_and_bc_dates() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_tstz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_tstz ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE, '
            "d DATE)",
            "INSERT INTO review_tstz VALUES "
            "(1, TIMESTAMP '2020-01-02 12:00:00 +05:00', DATE '2020-01-02')",
            "INSERT INTO review_tstz VALUES "
            "(2, TIMESTAMP '2020-01-02 12:00:00 -08:00', "
            "TO_DATE('-0044-03-15', 'SYYYY-MM-DD'))",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_TSTZ")
    # the offsets are applied, not dropped
    assert lf.select("id", "TS").sort("id").collect().rows() == [
        (1, datetime(2020, 1, 2, 7, tzinfo=UTC)),
        (2, datetime(2020, 1, 2, 20, tzinfo=UTC)),
    ]
    noon = datetime(2020, 1, 2, 12, tzinfo=UTC)
    assert lf.filter(pl.col("TS") < noon).select("id").collect().item() == 1
    # Python's datetime has no year -44
    with pytest.raises(pl.exceptions.PolarsError, match="outside years 1..9999"):
        lf.select("D").collect()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["adbc", "connectorx", "pymysql"])
def test_mysql_unsigned_decimals_and_zero_dates(driver: str) -> None:
    from datetime import datetime

    table = f"review_d_{driver}"
    _execute(
        "mysql",
        [
            "SET SESSION sql_mode = ''",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, a decimal(30,15) unsigned, "
            "b decimal(20,5) unsigned zerofill, d date, dt datetime, ts timestamp NULL)",
            f"INSERT INTO {table} VALUES "
            "(1, 0.123456789012345, 123456789012345.12345, '0000-00-00', "
            "'0000-00-00 00:00:00', '0000-00-00 00:00:00'), "
            "(2, 1, 1, '2020-00-15', '2020-01-02 03:04:05', '2020-01-02 03:04:05'), "
            "(3, 1, 1, '2019-01-01', NULL, NULL), (4, 1, 1, '2021-01-01', NULL, NULL)",
        ],
    )
    lf = _scan("mysql", driver, table)
    assert lf.collect_schema()["a"] == pl.Decimal(30, 15)
    assert lf.collect_schema()["b"] == pl.Decimal(20, 5)
    # every digit, not cut to 10 places or to 19 digits
    exact = Decimal("0.123456789012345")
    assert lf.filter(pl.col("a") == exact).select("id").collect().item() == 1
    rows = lf.sort("id").collect().rows()
    assert rows[0][1:3] == (exact, Decimal("123456789012345.12345"))
    # zero dates (and a zero day or month) are no date, also to pushed filters
    assert rows[0][3:] == (None, None, None)
    assert rows[1][3:5] == (None, datetime(2020, 1, 2, 3, 4, 5))
    # bounds and partitions take zero dates (and a zero day or month) as NULL, and so
    # does the order, also when the column isn't read
    for kwargs in [{"partition_on": None}, {"partition_on": "d", "partition_num": 3}]:
        ordered = _scan("mysql", driver, table, order_by="d", **kwargs)
        assert ordered.collect()["d"].to_list() == [
            None,
            None,
            date(2019, 1, 1),
            date(2021, 1, 1),
        ], kwargs
        ids = ordered.select("id").collect()["id"].to_list()
        assert ids[2:] == [3, 4], kwargs
    _check_filters(
        lf,
        [
            pl.col("d").is_null(),
            pl.col("d").is_not_null(),
            pl.col("dt") < datetime(2021, 1, 1),
            pl.col("dt").is_null() | (pl.col("ts") > datetime(2000, 1, 1, tzinfo=UTC)),
        ],
    )


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "pg8000", "sqlalchemy"])
def test_postgres_empty_table_is_typed(driver: str) -> None:
    from datetime import datetime

    table = f"review_empty_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, s text, v float8, ts timestamptz, "
            "n numeric(10, 2))",
        ],
    )
    expected = {
        "id": pl.Int32,
        "s": pl.String,
        "v": pl.Float64,
        "ts": pl.Datetime("us", "UTC"),
        "n": pl.Decimal(10, 2),
    }
    row = (1, "x", 1.5, datetime(2020, 1, 2, tzinfo=UTC), Decimal("1.25"))
    # a query's `numeric` has no declared precision for every driver
    for source, n in [(table, 5), (f"SELECT id, s, v, ts FROM {table}", 4)]:
        lf = _scan("postgres17", driver, source)
        assert dict(lf.collect_schema()) == dict(list(expected.items())[:n]), source
        # rows added after planning still fit the planned schema
        _execute(
            "postgres17",
            [f"INSERT INTO {table} VALUES (1, 'x', 1.5, '2020-01-02', 1.25)"],
        )
        assert lf.collect().rows() == [row[:n]]
        _execute("postgres17", [f"DELETE FROM {table}"])


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "connectorx"])
def test_postgres_query_columns_compare_by_their_type(driver: str) -> None:
    # a query has no catalog: its result types decide what is compared, and how
    table = f"review_pgq_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"DROP TYPE IF EXISTS {table}_mood",
            f"DROP COLLATION IF EXISTS {table}_ci",
            f"CREATE COLLATION {table}_ci "
            "(provider = icu, locale = 'und-u-ks-level2', deterministic = false)",
            f"CREATE TYPE {table}_mood AS ENUM ('a', 'b')",
            f"CREATE TABLE {table} (id int, code char(5), u uuid, ip inet, "
            f"m {table}_mood, ci text COLLATE {table}_ci, v text, n int)",
            f"INSERT INTO {table} VALUES "
            "(1, 'ab', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '10.0.0.1', 'a', "
            "'abc', 'x', 1), "
            "(2, 'cd', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '10.0.0.2/31', 'b', "
            "'ABC', 'y', 2)",
        ],
    )
    lf = _scan(
        "postgres17", driver, f"SELECT id, code, u, ip, m, ci, v, n FROM {table}"
    )
    predicates = [
        pl.col("code") == "ab",
        pl.col("m") == "nope",
        pl.col("ci") == "abc",
        pl.col("v") == "x",
        pl.col("v") < "y",
        pl.col("n") > 1,
    ]
    if lf.collect_schema()["u"] == pl.String:
        predicates += [
            pl.col("u") == "{a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11}",
            pl.col("u") == "zzz",
            pl.col("u") < "b",
            pl.col("ip") == "10.0.0.1/32",
        ]
    _check_filters(lf, predicates)
    # text and numbers are still compared in the database
    for p in [pl.col("v") == "x", pl.col("n") > 1]:
        assert "filters 1/1 (1 exact)" in lf.filter(p).explain(), str(p)


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_integer_is_a_38_digit_integer() -> None:
    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora_int'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora_int ("id" NUMBER(9), i INTEGER, s NUMBER(*, 2))',
            "INSERT INTO review_ora_int VALUES "
            "(1, 12345678901234567890123456789012345, 1.25)",
            "INSERT INTO review_ora_int VALUES (2, 7, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", "REVIEW_ORA_INT")
    assert lf.collect_schema()["I"] == pl.Decimal(38, 0)
    assert lf.collect_schema()["S"] == pl.Decimal(38, 2)
    df = lf.collect().sort("id")
    assert df["I"].to_list() == [Decimal(12345678901234567890123456789012345), 7]
    _check_filters(lf, [pl.col("I") > 7, pl.col("S") == Decimal("1.25")])


@pytest.mark.skipif(SERVERS["clickhouse"].uri is None, reason="no ClickHouse URI")
@pytest.mark.parametrize("driver", ["adbc", "clickhouse_connect"])
def test_clickhouse_query_columns_compare_by_their_type(driver: str) -> None:
    lf = _scan(
        "clickhouse",
        driver,
        "SELECT number AS id, toString(number) AS s, "
        "CAST(if(number = 1, 'b', 'a') AS Enum8('a' = 1, 'b' = 2)) AS e "
        "FROM numbers(3)",
    )
    if lf.collect_schema()["e"] == pl.String:
        # the database casts the literal to the enum
        _check_filters(lf, [pl.col("e") == "nope", pl.col("e") > "a"])
        assert "filters 0/1" in lf.filter(pl.col("e") == "b").explain()
    assert "filters 1/1 (1 exact)" in lf.filter(pl.col("s") == "1").explain()


def _sqlalchemy_url(uri: str, driver: str) -> str:
    scheme, rest = uri.split("://", 1)
    return f"{scheme}+{driver}://{rest}"


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "psycopg2"])
def test_postgres_engine_connections_dont_keep_the_session_setup(driver: str) -> None:
    # the scan's READ ONLY and UTC settings don't go back into the user's pool: they
    # are restored, and the pooled connection is kept
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    engine = sqlalchemy.create_engine(
        _sqlalchemy_url(POSTGRES, driver), pool_size=1, max_overflow=0
    )

    @sqlalchemy.event.listens_for(engine, "connect")
    def setup(dbapi_conn: Any, _: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("SET TIME ZONE 'Asia/Tokyo'")
        cur.execute("SET application_name = 'it''s 100%'")
        cur.close()
        dbapi_conn.commit()

    try:
        with engine.connect() as conn:
            pid = conn.exec_driver_sql("SELECT pg_backend_pid()").scalar()
        assert pl.scan_database("edge", engine).select(pl.len()).collect().item() > 0
        with engine.connect() as conn:
            show = conn.exec_driver_sql
            assert show("SELECT pg_backend_pid()").scalar() == pid
            assert show("SHOW default_transaction_read_only").scalar() == "off"
            assert show("SHOW TimeZone").scalar() == "Asia/Tokyo"
            assert show("SHOW application_name").scalar() == "it's 100%"
            conn.exec_driver_sql("CREATE TEMPORARY TABLE review_engine_rw (i int)")
            conn.exec_driver_sql("INSERT INTO review_engine_rw VALUES (1)")
            conn.rollback()
    finally:
        engine.dispose()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_engine_connections_dont_keep_the_session_setup() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    uri = SERVERS["mysql"].uri
    assert uri is not None
    engine = sqlalchemy.create_engine(
        _sqlalchemy_url(uri, "pymysql"), pool_size=1, max_overflow=0
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET time_zone = '+09:00'")
            conn.exec_driver_sql(
                "SET SESSION sql_mode = CONCAT(@@sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
            before = conn.exec_driver_sql(
                "SELECT CONNECTION_ID(), @@time_zone, @@sql_mode, "
                "@@transaction_read_only"
            ).one()
            conn.commit()
        assert pl.scan_database("edge", engine).select(pl.len()).collect().item() > 0
        with engine.connect() as conn:
            after = conn.exec_driver_sql(
                "SELECT CONNECTION_ID(), @@time_zone, @@sql_mode, "
                "@@transaction_read_only"
            ).one()
        assert tuple(after) == tuple(before)
    finally:
        engine.dispose()


@needs_postgres
@pytest.mark.parametrize("module", ["psycopg", "psycopg2"])
def test_postgres_live_connection_after_an_early_stop(module: str) -> None:
    # Polars releases an early-stopped read after `collect()` returned: the scan
    # neither cancels nor rolls back then, which would hit the user's next statements
    import importlib

    driver = importlib.import_module(module)
    table = f"review_early_stop_{module}"
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (k int)",
        ],
    )
    conn = driver.connect(POSTGRES)
    try:
        q = "SELECT g AS id FROM generate_series(1, 300000) g"
        for k in range(10):
            lf = pl.scan_database(q, conn, batch_size=1000)
            lf.filter((pl.col("id").hash() % 7) == 0).head(2).collect()
            cur = conn.cursor()
            cur.execute(f"INSERT INTO {table} VALUES (%s)", (k,))
            cur.close()
            conn.commit()
        cur = conn.cursor()
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] == 10
        cur.close()
        conn.commit()
    finally:
        conn.close()


@needs_postgres
def test_postgres_engine_params_and_literals_with_colons() -> None:
    # `text()` binds `:lo`, also inside a pushed literal: those stay in Polars
    sqlalchemy = pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            "DROP TABLE IF EXISTS review_colons",
            "CREATE TABLE review_colons (id int PRIMARY KEY, s text)",
            "INSERT INTO review_colons VALUES "
            "(1, ':lo'), (2, '0'), (3, 'ab'), (4, '\\:x'), (5, ':x')",
        ],
    )
    engine = sqlalchemy.create_engine(_sqlalchemy_url(POSTGRES, "psycopg2"))
    try:
        q = pl.scan_database(
            "SELECT id, s FROM review_colons WHERE id > :lo", engine, params={"lo": 0}
        )
        t = pl.scan_database("review_colons", engine)
        # parameters given as `execute()`'s own argument
        e = pl.scan_database(
            "SELECT id, s FROM review_colons WHERE id > :lo",
            engine,
            execute_options={"parameters": {"lo": 0}},
        )
        with sqlalchemy.orm.Session(engine) as session:
            # an ORM Session has no `exec_driver_sql`: its connection's is used
            o = pl.scan_database("review_colons", session)
            for lf in (q, t, e, o):
                for value, want in [
                    (":lo", [1]),
                    ("\\:x", [4]),
                    (":abc", []),
                    ("ab", [3]),
                ]:
                    out = lf.filter(pl.col("s") == value).collect()
                    assert out["id"].to_list() == want, value
    finally:
        engine.dispose()


@needs_postgres
def test_postgres_percent_literals_with_execute_options_parameters() -> None:
    # psycopg binds `parameters` given in `execute_options` and reads `%` in the SQL
    assert POSTGRES is not None
    lf = pl.scan_database(
        "SELECT id, s FROM edge WHERE id < %s",
        POSTGRES,
        driver="psycopg",
        execute_options={"params": [30]},
        partition_on=None,
    )
    out = lf.filter(pl.col("s") != "a%b").collect()
    expected = _reference("postgres17", _default_driver("postgres17")).filter(
        pl.col("id") < 30, pl.col("s") != "a%b"
    )
    assert sorted(out["id"].to_list()) == sorted(expected["id"].to_list())


@needs_postgres
def test_postgres_cancel_on_a_shared_live_connection() -> None:
    # one scan's cancel must not stop another scan's statement on the same connection:
    # a live connection is never cancelled
    import threading

    import psycopg

    from polars_database import _bridge

    assert POSTGRES is not None
    with psycopg.connect(POSTGRES) as conn:
        running = _bridge._Session(conn, owned=False, execute_options={})
        other = _bridge._Session(conn, owned=False, execute_options={})
        assert not other.can_cancel
        other_batches = other.execute("SELECT i FROM generate_series(1, 10) i", 1)
        next(other_batches)
        stop = threading.Event()

        def cancel() -> None:
            while not stop.is_set():
                other.cancel()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            sql = "SELECT i, md5(i::text) AS s FROM generate_series(1, 200000) i ORDER BY 2"
            n = sum(df.height for df in running.execute(sql, 5000))
        finally:
            stop.set()
            thread.join()
            other_batches.close()  # type: ignore[attr-defined]
        assert n == 200_000
        # `other` stopped early: its transaction is the user's to end
        status = conn.info.transaction_status
        assert status == psycopg.pq.TransactionStatus.INTRANS


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_pymysql_cancel_doesnt_read_the_rest() -> None:
    # PyMySQL's unbuffered cursor would read the rest of the result when closed
    from polars_database import _bridge

    uri = SERVERS["mysql"].uri
    assert uri is not None
    conn = _mysql_factory(uri)()
    session = _bridge._Session(conn, owned=True, execute_options={})
    assert session.can_cancel
    sql = (
        "WITH RECURSIVE r (i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM r "
        "WHERE i < 1000) SELECT a.i, MD5(a.i * b.i) AS s FROM r a CROSS JOIN r b"
    )
    batches = session.execute(sql, 1000)
    next(batches)
    packets = 0
    read_packet = conn._read_packet

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal packets
        packets += 1
        return read_packet(*args, **kwargs)

    conn._read_packet = counted
    session.cancel()
    batches.close()  # type: ignore[attr-defined]
    session.close()
    # not the ~1M rows left
    assert packets < 100


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_scans_sharing_an_unbuffered_live_connection(driver: str) -> None:
    # MySQL keeps one result open per connection: interleaved unbuffered reads would
    # end each other's results early
    uri = SERVERS["mysql"].uri
    assert uri is not None
    if driver == "pymysql":
        import pymysql

        conn = pymysql.connect(**_parts(uri), cursorclass=pymysql.cursors.SSCursor)
    else:
        mysql_connector = pytest.importorskip("mysql.connector")
        conn = mysql_connector.connect(**_parts(uri))
    try:
        a = pl.scan_database("edge", conn, batch_size=7)
        b = pl.scan_database("edge", conn, batch_size=7)
        n = pl.concat([a, b]).select(pl.len()).collect().item()
        assert n == 2 * a.select(pl.len()).collect().item()
        assert a.join(b, on="id").select(pl.len()).collect().item() == n // 2
    finally:
        conn.close()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_user_statements_during_an_unbuffered_live_scan(driver: str) -> None:
    # the scan's results are buffered: the user's statements between its batches,
    # or after it was dropped early, don't meet an unread result
    import gc

    uri = SERVERS["mysql"].uri
    assert uri is not None
    if driver == "pymysql":
        import pymysql

        conn = pymysql.connect(**_parts(uri), cursorclass=pymysql.cursors.SSCursor)
    else:
        mysql_connector = pytest.importorskip("mysql.connector")
        conn = mysql_connector.connect(**_parts(uri))

    def user_statement() -> None:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM edge")
        cur.fetchall()
        cur.close()

    try:
        lf = pl.scan_database("edge", conn, batch_size=7)
        total = 0
        for df in lf.collect_batches():
            total += df.height
            user_statement()
        assert total == lf.select(pl.len()).collect().item()
        batches = lf.collect_batches()
        next(iter(batches))
        del batches
        gc.collect()
        user_statement()
    finally:
        conn.close()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg", "pg8000", "sqlalchemy"])
def test_postgres_credential_provider_with_user_only(driver: str) -> None:
    # the URI's password stays
    pytest.importorskip(
        {"adbc": "polars", "sqlalchemy": "sqlalchemy"}.get(driver, driver)
    )
    assert POSTGRES is not None
    user = urlsplit(POSTGRES).username
    lf = pl.scan_database(
        "SELECT 1 AS x",
        POSTGRES,
        driver=driver,
        credential_provider=lambda: {"user": user},
    )
    assert lf.collect().item() == 1


@needs_postgres
def test_postgres_uri_query_parameters_reach_pg8000() -> None:
    pytest.importorskip("pg8000")
    assert POSTGRES is not None
    lf = pl.scan_database(
        "SELECT current_setting('search_path') AS p",
        f"{POSTGRES}?options=-c%20search_path%3Da_path",
        driver="pg8000",
    )
    assert lf.collect().item() == "a_path"
    # TLS is required: pg8000 must not connect without it
    tls = f"{POSTGRES}?sslmode=verify-full&sslrootcert=/nonexistent"
    with pytest.raises(pl.exceptions.ComputeError, match="/nonexistent"):
        pl.scan_database("SELECT 1 AS x", tls, driver="pg8000").collect()
    with pytest.raises(pl.exceptions.ComputeError, match=r"\['bogus'\]"):
        pl.scan_database(
            "SELECT 1 AS x", f"{POSTGRES}?bogus=1", driver="pg8000"
        ).collect()


@needs_postgres
def test_postgres_sqlalchemy_pg8000_gets_no_connect_timeout() -> None:
    pytest.importorskip("pg8000")
    pytest.importorskip("sqlalchemy")
    assert POSTGRES is not None
    uri = POSTGRES.replace("postgresql://", "postgresql+pg8000://", 1)
    lf = pl.scan_database("SELECT 1 AS x", uri, driver="sqlalchemy")
    assert lf.collect().item() == 1


@needs_postgres
def test_redshift_uri_through_psycopg2() -> None:
    # a Postgres server stands in for Redshift: libpq takes only `postgresql://`
    pytest.importorskip("psycopg2")
    assert POSTGRES is not None
    uri = POSTGRES.replace("postgresql://", "redshift://", 1)
    lf = pl.scan_database("SELECT 1 AS x", uri, driver="psycopg2")
    assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_uri_query_parameters_reach_the_driver(driver: str) -> None:
    pytest.importorskip(driver)
    uri = SERVERS["mysql"].uri
    assert uri is not None
    lf = pl.scan_database(
        "SELECT @@character_set_client AS c", f"{uri}?charset=latin1", driver=driver
    )
    assert lf.collect().item() == "latin1"
    with pytest.raises(pl.exceptions.ComputeError, match=r"\['bogus'\]"):
        pl.scan_database("SELECT 1 AS x", f"{uri}?bogus=1", driver=driver).collect()


@needs_postgres
def test_postgres_common_libpq_parameters_with_pg8000() -> None:
    pytest.importorskip("pg8000")
    assert POSTGRES is not None
    query = "connect_timeout=5&target_session_attrs=any&client_encoding=utf8"
    lf = pl.scan_database("SELECT 1 AS x", f"{POSTGRES}?{query}", driver="pg8000")
    assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_ssl_parameter(driver: str) -> None:
    pytest.importorskip(driver)
    uri = SERVERS["mysql"].uri
    assert uri is not None
    for q in ["ssl=true", "ssl=false", "ssl_mode=REQUIRED"]:
        lf = pl.scan_database("SELECT 1 AS x", f"{uri}?{q}", driver=driver)
        assert lf.collect().item() == 1


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["adbc", "pymysql"])
def test_mysql_empty_table_is_typed(driver: str) -> None:
    table = f"review_empty_{driver}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, d date, dt datetime(6), x double)",
        ],
    )
    for source in [table, f"SELECT d, dt, x FROM {table}"]:
        schema = _scan("mysql", driver, source).collect_schema()
        assert schema["d"] == pl.Date, source
        assert schema["dt"] == pl.Datetime("us"), source
        assert schema["x"] == pl.Float64, source


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "pg8000"])
def test_postgres_empty_query_arrays_are_typed(driver: str) -> None:
    table = f"review_empty_arr_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (a int4[], t text[], f float8[])",
        ],
    )
    lf = _scan("postgres17", driver, f"SELECT * FROM {table}")
    assert dict(lf.collect_schema()) == {
        "a": pl.List(pl.Int32),
        "t": pl.List(pl.String),
        "f": pl.List(pl.Float64),
    }
    _execute(
        "postgres17", [f"INSERT INTO {table} VALUES ('{{1}}', '{{x}}', '{{1.5}}')"]
    )
    assert lf.collect().rows() == [([1], ["x"], [1.5])]


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_query_time_zone_offsets() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_qtz'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_qtz ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE)',
            "INSERT INTO review_qtz VALUES (1, TIMESTAMP '2020-01-02 12:00:00 +05:00')",
            "INSERT INTO review_qtz VALUES (2, TIMESTAMP '2020-01-02 12:00:00 -08:00')",
        ],
    )
    noon = datetime(2020, 1, 2, 12, tzinfo=UTC)
    # read as UTC with pushdown or without
    for pushdown in [True, False]:
        lf = _scan(
            "oracle", "oracledb", 'SELECT "id", ts FROM review_qtz', pushdown=pushdown
        )
        assert lf.select("TS").sort("TS").collect()["TS"].to_list() == [
            datetime(2020, 1, 2, 7, tzinfo=UTC),
            datetime(2020, 1, 2, 20, tzinfo=UTC),
        ]
        assert lf.filter(pl.col("TS") < noon).select("id").collect().item() == 1


@pytest.mark.parametrize("name", ["mysql", "mariadb"])
@pytest.mark.parametrize("driver", ["adbc", "pymysql", "connectorx"])
def test_mysql_query_year_and_json_columns(name: str, driver: str) -> None:
    if SERVERS[name].uri is None:
        pytest.skip(f"no {name} URI")
    table = f"review_myq_{driver}"
    _execute(
        name,
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int, y year, j json, s varchar(5))",
            f"INSERT INTO {table} VALUES (1, 1970, '{{\"a\": 1}}', 'x'), "
            "(2, 2001, '\"x\"', 'y'), (3, 0, NULL, NULL), (4, NULL, NULL, NULL)",
        ],
    )
    sources = [
        (f"SELECT id, y, j, s FROM {table}", {}),
        # the user's schema doesn't say an integer isn't a YEAR
        (table, {"schema": {"id": pl.Int32, "y": pl.Int16}}),
    ]
    for source, kwargs in sources:
        lf = _scan(name, driver, source, **kwargs)
        dtype = lf.collect_schema()["y"]
        if dtype.is_integer():
            predicates = [pl.col("y") == 70, pl.col("y") == 1, pl.col("y") > 50]
            predicates += [pl.col("y").is_in([1, 70]), pl.col("y") == 1970]
        else:
            predicates = [pl.col("y") == "0", pl.col("y") == "70"]
        if "j" in lf.collect_schema():
            predicates += [pl.col("j") == '{"a": 1}', pl.col("j") == "x"]
            predicates += [pl.col("s") == "x"]
        _check_filters(lf, predicates)


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_query_types_and_collations() -> None:
    from datetime import datetime

    _execute(
        "oracle",
        [
            "BEGIN EXECUTE IMMEDIATE 'DROP TABLE review_ora_q'; "
            "EXCEPTION WHEN OTHERS THEN NULL; END;",
            'CREATE TABLE review_ora_q ("id" NUMBER(9), ts TIMESTAMP WITH TIME ZONE, '
            "s VARCHAR2(5), cl CLOB)",
            "INSERT INTO review_ora_q VALUES "
            "(1, TIMESTAMP '2024-01-01 12:00:00 +05:00', 'a', 'x')",
            "INSERT INTO review_ora_q VALUES "
            "(2, TIMESTAMP '2024-01-01 12:00:00 -08:00', 'b', 'y')",
            "INSERT INTO review_ora_q VALUES (3, NULL, NULL, NULL)",
        ],
    )
    lf = _scan("oracle", "oracledb", 'SELECT "id", ts, s, cl FROM review_ora_q')
    utc = datetime(2024, 1, 1, 7)
    _check_filters(
        lf,
        [
            pl.col("TS") == pl.lit(utc).dt.replace_time_zone("UTC"),
            pl.col("TS") > pl.lit(utc).dt.replace_time_zone("UTC"),
            pl.col("CL") == "x",
            pl.col("S") == "a",
        ],
    )
    # the table's column collations are looked up: a default one is compared
    table = _scan("oracle", "oracledb", "REVIEW_ORA_Q")
    assert "filters 1/1" in table.filter(pl.col("S") == "a").explain()


@needs_postgres
@pytest.mark.parametrize("driver", ["psycopg", "pg8000", "sqlalchemy"])
def test_postgres_uuid_from_row_drivers(driver: str) -> None:
    # row drivers return `uuid.UUID` objects: read as text, not as an Object column
    table = f"review_uuid_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id int PRIMARY KEY, u uuid)",
            f"INSERT INTO {table} VALUES "
            "(1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'), (2, NULL)",
        ],
    )
    for source in [table, f"SELECT * FROM {table}"]:
        lf = _scan("postgres17", driver, source)
        assert lf.collect().sort("id")["u"].to_list() == [
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            None,
        ]
        uuid = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        out = lf.filter(pl.col("u") == uuid).collect()
        assert out["id"].to_list() == [1]


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
def test_mysql_nul_literal_without_backslash_escapes() -> None:
    # NUL is written as `\0`, which NO_BACKSLASH_ESCAPES reads as backslash, zero
    _execute(
        "mysql",
        [
            "DROP TABLE IF EXISTS review_nul",
            "CREATE TABLE review_nul (id int, s varchar(10) COLLATE utf8mb4_bin)",
            "INSERT INTO review_nul VALUES (1, 'a\\0b'), (2, 'a'), (3, 'a\\\\0b')",
        ],
    )
    assert SERVERS["mysql"].uri is not None
    conn = _mysql_factory(SERVERS["mysql"].uri)()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET SESSION sql_mode = "
                "CONCAT(@@SESSION.sql_mode, ',NO_BACKSLASH_ESCAPES')"
            )
        _check_filters(
            pl.scan_database("review_nul", conn),
            [
                pl.col("s") == "a\0b",
                pl.col("s").is_in(["a\0b"]),
                pl.col("s") == "a\\0b",
            ],
        )
    finally:
        conn.close()


@needs_postgres
@pytest.mark.parametrize("driver", ["adbc", "psycopg"])
def test_postgres_domains_compare_as_their_base_type(driver: str) -> None:
    table = f"review_pgdom_{driver}"
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"DROP DOMAIN IF EXISTS {table}_d2",
            f"DROP DOMAIN IF EXISTS {table}_d1",
            f"CREATE DOMAIN {table}_d1 AS varchar(5)",
            f"CREATE DOMAIN {table}_d2 AS {table}_d1",
            f"CREATE TABLE {table} (id int, a {table}_d1, b {table}_d2)",
            f"INSERT INTO {table} VALUES (1, 'x', 'y'), (2, 'z', NULL)",
        ],
    )
    for source in [table, f"SELECT * FROM {table}"]:
        lf = _scan("postgres17", driver, source)
        _check_filters(lf, [pl.col("a") == "x", pl.col("b") == "y"])
        # a domain over a domain over varchar: compared as text
        assert "(1 exact)" in lf.filter(pl.col("b") == "y").explain(), source


@pytest.mark.skipif(SERVERS["oracle"].uri is None, reason="no Oracle URI")
@ORACLE_GROUP
def test_oracle_wide_query_is_described() -> None:
    # the description of 300 columns is longer than a VARCHAR2
    cols = ", ".join(f"{i} AS some_longer_column_name_{i}" for i in range(300))
    lf = _scan(
        "oracle",
        "oracledb",
        f"SELECT {cols}, CAST('x' AS VARCHAR2(5)) AS s FROM dual",
    )
    assert "filters 1/1" in lf.filter(pl.col("S") == "x").explain()
    assert lf.filter(pl.col("S") == "x").collect().height == 1


@needs_postgres
def test_postgres_live_early_stops_dont_leave_cursors_behind() -> None:
    # early-stopped scans' cursors are closed in the transaction they live in, and
    # names aren't reused: another scan never declares a cursor that still exists
    import psycopg

    assert POSTGRES is not None
    q = "SELECT g AS id FROM generate_series(1, 200000) g"
    with psycopg.connect(POSTGRES) as conn:
        for _ in range(20):
            lf = pl.scan_database(q, conn, batch_size=1000, pushdown=False)
            assert lf.head(3).collect().height == 3
            open_cursors = conn.execute(
                "SELECT count(*) FROM pg_cursors WHERE name LIKE 'polars_scan_%'"
            ).fetchone()
            assert open_cursors is not None
            assert open_cursors[0] <= 2
        conn.rollback()


@pytest.mark.skipif(SERVERS["mysql"].uri is None, reason="no MySQL URI")
@pytest.mark.parametrize("driver", ["pymysql", "mysql.connector"])
def test_mysql_bigint_unsigned_query_source(driver: str) -> None:
    uri = SERVERS["mysql"].uri
    assert uri is not None
    table = f"review_unsigned_{driver.replace('.', '_')}"
    _execute(
        "mysql",
        [
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (id INT, bu BIGINT UNSIGNED, iu INT UNSIGNED)",
            f"INSERT INTO {table} VALUES (1, 18446744073709551615, 4294967295), "
            "(2, 5, 0)",
        ],
    )
    for source in [table, f"SELECT id, bu, iu FROM {table}"]:
        df = _scan("mysql", driver, source).collect().sort("id")
        assert df["bu"].to_list() == [2**64 - 1, 5]
        assert df["iu"].to_list() == [2**32 - 1, 0]


@needs_postgres
@pytest.mark.parametrize("kind", ["live", "factory"])
def test_postgres_psycopg2_bytea(kind: str) -> None:
    # psycopg2 returns bytea as memoryview values
    import psycopg2

    assert POSTGRES is not None
    _execute(
        "postgres17",
        [
            f"DROP TABLE IF EXISTS review_bytea_{kind}",
            f"CREATE TABLE review_bytea_{kind} (id int, b bytea)",
            f"INSERT INTO review_bytea_{kind} VALUES (1, '\\x00ff'), (2, NULL)",
        ],
    )
    for source in [f"review_bytea_{kind}", f"SELECT * FROM review_bytea_{kind}"]:
        if kind == "live":
            with contextlib.closing(psycopg2.connect(POSTGRES)) as conn:
                df = pl.scan_database(source, conn).collect()
        else:
            df = pl.scan_database(source, lambda: psycopg2.connect(POSTGRES)).collect()
        assert df.sort("id")["b"].to_list() == [b"\x00\xff", None]
