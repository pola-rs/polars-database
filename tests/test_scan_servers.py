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

import functools
import importlib.util
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, date
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

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
    from collections.abc import Callable
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


@functools.cache
def _live(name: str) -> Any:
    server = SERVERS[name]
    assert server.factory is not None
    assert server.uri is not None
    return server.factory(server.uri)()


def _scan(name: str, driver: str, source: str = "edge", **kwargs: Any) -> pl.LazyFrame:
    server = SERVERS[name]
    uri = server.uri
    assert uri is not None
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


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_server_marks(n)) for n in sorted(UNUSUAL_TYPES)]
)
def test_unusual_types_read_the_same_by_every_driver(name: str) -> None:
    statements, table, drivers = UNUSUAL_TYPES[name]
    _execute(name, statements)
    frames = {}
    for driver in drivers:
        module = DRIVER_MODULES.get(driver, driver)
        if module is not None and importlib.util.find_spec(module) is None:
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
