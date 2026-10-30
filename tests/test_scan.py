from __future__ import annotations

import io
import os
import sqlite3
from contextlib import closing
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from polars._typing import PolarsDataType

pytest.importorskip("sqlglot")

N_ROWS = 1000


def _rows(n: int) -> list[tuple[Any, ...]]:
    return [
        (
            i,
            f"c{i % 7}",
            None if i % 11 == 0 else i * 1.5,
            "paid" if i % 3 else "open",
            f"2026-01-{1 + i % 28:02d}",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE orders ("
        "id INTEGER PRIMARY KEY, customer TEXT, amount REAL, status TEXT, day TEXT)"
    )
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _rows(N_ROWS))
    conn.execute("CREATE TABLE empty (a INTEGER, b TEXT)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def shop_uri(shop_db: Path) -> str:
    return f"sqlite:///{shop_db}"


@pytest.fixture
def expected(shop_db: Path) -> pl.DataFrame:
    conn = sqlite3.connect(shop_db)
    try:
        return pl.read_database("SELECT * FROM orders", conn)
    finally:
        conn.close()


def _sql(lf: pl.LazyFrame) -> str:
    plan = lf.explain()
    return next(line for line in plan.splitlines() if "sql[0]:" in line)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns)


@pytest.mark.parametrize(
    "driver", [None, "adbc", "sqlite3", "connectorx", "sqlalchemy"]
)
def test_scan_table_drivers(
    shop_uri: str, expected: pl.DataFrame, driver: str | None
) -> None:
    pytest.importorskip(
        {"connectorx": "connectorx", "sqlalchemy": "sqlalchemy"}.get(
            driver or "", "sqlite3"
        )
    )
    if driver in (None, "adbc"):
        pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri, driver=driver)
    assert lf.collect_schema() == expected.schema
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))


@pytest.mark.parametrize(
    ("predicate", "exact"),
    [
        (pl.col("status") == "paid", True),
        (pl.col("status") != "paid", True),
        (pl.col("id") > 500, True),
        (pl.col("id").is_between(10, 20), True),
        (pl.col("id").is_between(10, 20, closed="none"), True),
        (pl.col("customer").is_in(["c1", "c3"]), True),
        (pl.col("customer").is_in(["c1", None], nulls_equal=True), True),
        (pl.col("amount").is_null(), True),
        (pl.col("amount").is_not_null() & (pl.col("amount") < 30), True),
        ((pl.col("id") < 10) | (pl.col("id") > 990), True),
        (~(pl.col("id") < 990), True),
        (pl.col("customer").str.starts_with("c1"), True),
        (pl.col("customer").str.ends_with("3"), True),
        (pl.col("customer").str.contains("_", literal=True), True),
        (pl.col("amount").ne_missing(3.0), True),
        (pl.col("amount").eq_missing(None), True),
        # not pushed: arithmetic is not in tier 1
        ((pl.col("id") * 2) > 100, False),
        # partly pushed: the AND keeps its translatable side as a pre-filter
        ((pl.col("id") > 900) & ((pl.col("id") % 2) == 0), False),
    ],
)
def test_scan_filter_pushdown(
    shop_uri: str, expected: pl.DataFrame, predicate: pl.Expr, exact: bool
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri).filter(predicate)
    out = lf.collect()
    assert_frame_equal(_sorted(out), _sorted(expected.filter(predicate)))
    plan = lf.explain()
    if exact:
        assert "WHERE" in _sql(lf)
        assert "(1 exact)" in plan or "exact)" in plan


def test_scan_partitioning(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "orders", shop_uri, partition_on="id", partition_num=4, max_connections=2
    )
    plan = lf.explain()
    assert "partitions: 4 (range on" in plan
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))

    # bounds use the pushed filters, and every row lands in exactly one partition
    q = lf.filter(pl.col("id") > 700)
    assert_frame_equal(
        _sorted(q.collect()), _sorted(expected.filter(pl.col("id") > 700))
    )


def test_scan_partition_column_with_nulls(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri, partition_on="amount", partition_num=3)
    assert lf.explain().count("range on") == 1
    out = lf.collect()
    assert out.height == N_ROWS
    assert out["amount"].null_count() == expected["amount"].null_count()


def test_scan_partition_range_and_predicates(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    # the range only sets the stride: rows outside it still land in the end partitions
    lf = pl.scan_database(
        "orders",
        shop_uri,
        partition_on="id",
        partition_num=4,
        partition_range=(400, 600),
    )
    assert "bounds from partition_range" in lf.explain()
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))

    lf = pl.scan_database(
        "orders",
        shop_uri,
        partition_predicates=["id % 2 = 0", "id % 2 = 1"],
    )
    assert "2 (partition_predicates)" in lf.explain()
    assert_frame_equal(_sorted(lf.collect()), _sorted(expected))


def test_scan_order_by(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "orders", shop_uri, order_by="id", partition_on="id", partition_num=4
    )
    assert "ORDER BY" in _sql(lf)
    assert_frame_equal(lf.collect(), expected.sort("id"))


def test_scan_slice_and_count(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri)
    assert "LIMIT 5" in _sql(lf.head(5))
    assert lf.head(5).collect().height == 5
    assert lf.slice(10, 5).collect().height == 5
    assert lf.select(pl.len()).collect().item() == N_ROWS
    assert lf.filter(pl.col("status") == "paid").select(pl.len()).collect().item() == (
        N_ROWS - N_ROWS // 3
    )


def test_scan_row_index_keeps_filters_in_polars(
    shop_uri: str, expected: pl.DataFrame
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = (
        pl.scan_database("orders", shop_uri, order_by="id", partition_on=None)
        .with_row_index()
        .filter(pl.col("id") > 990)
    )
    out = lf.collect()
    assert out["index"].to_list() == list(range(990, 1000))


def test_scan_query(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "SELECT id, customer, amount FROM orders WHERE status = 'paid'", shop_uri
    ).filter(pl.col("id") < 50)
    assert (
        "FROM (SELECT id, customer, amount FROM orders WHERE status = 'paid')"
        in _sql(lf)
    )
    exp = expected.filter(pl.col("status") == "paid", pl.col("id") < 50).select(
        "id", "customer", "amount"
    )
    assert_frame_equal(_sorted(lf.collect()), _sorted(exp))


def test_scan_query_without_pushdown(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    # duplicate output names can't be wrapped, nor held in a DataFrame
    lf = pl.scan_database("SELECT id, id FROM orders", shop_uri)
    with pytest.raises(pl.exceptions.DuplicateError):
        lf.collect_schema()

    # a query that sqlglot can't parse runs as given
    lf = pl.scan_database(
        "SELECT id FROM orders WHERE id < 3 /* trailing */",
        shop_uri,
        source_type="query",
    )
    assert lf.collect()["id"].to_list() == [1, 2]

    lf = pl.scan_database("orders", shop_uri, pushdown=False).filter(pl.col("id") < 3)
    assert "pushdown: off (pushdown=False)" in lf.explain()
    assert lf.collect()["id"].to_list() == [1, 2]


def test_scan_user_schema_and_overrides(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    schema = {"id": pl.Int32, "customer": pl.String}
    lf = pl.scan_database("SELECT id, customer FROM orders", shop_uri, schema=schema)
    assert lf.collect_schema() == pl.Schema(schema)
    assert lf.collect().schema == pl.Schema(schema)

    lf = pl.scan_database("orders", shop_uri, schema_overrides={"id": pl.UInt16})
    assert lf.collect_schema()["id"] == pl.UInt16
    assert lf.collect()["id"].dtype == pl.UInt16


def test_scan_empty_table(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    out = pl.scan_database("empty", shop_uri).collect()
    assert out.height == 0
    assert out.columns == ["a", "b"]


def test_scan_connection_factory_and_live(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT range AS id, range % 5 AS g, "
            "DATE '2026-01-01' + CAST(range AS INTEGER) AS d FROM range(100)"
        )

    def factory() -> Any:
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory).filter(
        pl.col("g") == 2, pl.col("d") >= date(2026, 2, 1)
    )
    assert "python:duckdb" in lf.explain()
    out = lf.collect()
    assert sorted(out["id"].to_list()) == [i for i in range(31, 100) if i % 5 == 2]

    live = duckdb.connect(path, read_only=True)
    try:
        lf = pl.scan_database("t", live).select("id").filter(pl.col("id") < 3)
        assert sorted(lf.collect()["id"].to_list()) == [0, 1, 2]
    finally:
        live.close()

    # a URI picks the duckdb module too
    out = pl.scan_database("t", f"duckdb:///{path}").select(pl.len()).collect()
    assert out.item() == 100


def test_scan_live_connection_is_not_serializable(shop_db: Path) -> None:
    conn = sqlite3.connect(shop_db, check_same_thread=False)
    try:
        lf = pl.scan_database("orders", conn, schema={"id": pl.Int64})
        with pytest.raises(Exception, match="live connection"):
            lf.serialize()
    finally:
        conn.close()


def test_scan_serialize_roundtrip(shop_uri: str, expected: pl.DataFrame) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("orders", shop_uri).filter(pl.col("id") <= 10)
    buf = io.BytesIO(lf.serialize())
    out = pl.LazyFrame.deserialize(buf).collect()
    assert_frame_equal(_sorted(out), _sorted(expected.filter(pl.col("id") <= 10)))


def test_scan_secrets_not_in_explain(shop_db: Path) -> None:
    lf = pl.scan_database(
        "orders",
        # a host that can't resolve fails at once; a refused connect is retried
        "postgresql://user:hunter2@nohost.invalid:1/db",
        schema={"id": pl.Int64},
        driver="sqlalchemy",
    )
    with pytest.raises(Exception) as exc:
        lf.collect()
    assert "hunter2" not in str(exc.value)


def test_scan_errors(shop_uri: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    with pytest.raises(Exception, match="not installed"):
        pl.scan_database(
            "orders", shop_uri, driver="python:not_a_real_module"
        ).collect()
    with pytest.raises(Exception, match="range partitioning"):
        pl.scan_database(
            "orders", shop_uri, partition_on="customer", partition_num=2
        ).collect()
    with pytest.raises(ValueError, match="connection URI"):
        pl.scan_database("orders", "not a uri")
    with pytest.raises(Exception, match="snapshot"):
        pl.scan_database(
            "orders",
            shop_uri,
            partition_on="id",
            partition_num=2,
            consistency="snapshot",
        ).collect()


def test_show_drivers() -> None:
    df = pl.io.database.show_drivers("sqlite")
    assert df.columns == [
        "dialect",
        "backend",
        "available",
        "version",
        "location",
        "install",
    ]
    assert "python:sqlite3" in df["backend"].to_list()
    assert df.filter(pl.col("backend") == "python:sqlite3")["available"].item()
    assert pl.io.database.show_drivers("postgresql://u@h/db")[
        "dialect"
    ].unique().to_list() == ["postgresql"]


def test_read_database_uri_adbc_manifest_driver(
    shop_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an ADBC driver installed with `dbc` (a manifest, not a pip package) is found
    adbc_sqlite = pytest.importorskip("adbc_driver_sqlite")
    pytest.importorskip("adbc_driver_manager")
    manifest_dir = tmp_path / "drivers"
    manifest_dir.mkdir()
    library = adbc_sqlite._driver_path()
    (manifest_dir / "sqlite.toml").write_text(
        f'name = "SQLite"\nversion = "1.0.0"\n[Driver]\nshared = "{library}"\n'
    )
    monkeypatch.setenv("ADBC_DRIVER_PATH", str(manifest_dir))

    import polars.io.database._utils as utils

    real_import: Callable[..., Any] = utils._import_optional_adbc_driver

    def no_pip_driver(module_name: str, **kwargs: Any) -> Any:
        if module_name == "adbc_driver_sqlite":
            msg = "ADBC 'adbc_driver_sqlite' driver not detected."
            raise ModuleNotFoundError(msg)
        return real_import(module_name, **kwargs)

    monkeypatch.setattr(utils, "_import_optional_adbc_driver", no_pip_driver)
    out = pl.read_database_uri(
        "SELECT COUNT(*) AS n FROM orders", f"sqlite:///{shop_db}", engine="adbc"
    )
    assert out.item() == N_ROWS


# The differential corpus on SQLite and DuckDB, the in-process databases, on every PR.
# The server variant is in `test_scan_servers.py`.

EDGE_OVERRIDES: dict[str, PolarsDataType] = {
    "dec": pl.Decimal(18, 4),
    "d": pl.Date,
    "ts": pl.Datetime("us"),
    "b": pl.Boolean,
}


@pytest.fixture(scope="module")
def edge_sqlite(tmp_path_factory: pytest.TempPathFactory) -> str:
    from tests._scan_corpus import edge_rows

    path = tmp_path_factory.mktemp("edge") / "edge.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE edge (id INTEGER PRIMARY KEY, i16 INTEGER, i32 INTEGER, "
        "i64 INTEGER, f64 REAL, dec TEXT, s TEXT, sb TEXT, d TEXT, ts TEXT, b INTEGER)"
    )
    rows = [
        tuple(str(v) if isinstance(v, (Decimal, date)) else v for v in row)
        for row in edge_rows()
    ]
    conn.executemany("INSERT INTO edge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


@pytest.fixture(scope="module")
def edge_duckdb(tmp_path_factory: pytest.TempPathFactory) -> str:
    duckdb = pytest.importorskip("duckdb")
    from tests._scan_corpus import edge_frame

    path = str(tmp_path_factory.mktemp("edge") / "edge.duckdb")
    frame = edge_frame()  # noqa: F841 (read by DuckDB by name)
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE edge AS SELECT * FROM frame")
    return path


# as `_scan_corpus.CORPUS_MODES`, which is imported lazily
_CORPUS_MODES = ["table", "query", "table-p4", "query-p4"]


def _corpus_scan(
    source_uri: Any, driver: str | None, mode: str, **kwargs: Any
) -> pl.LazyFrame:
    source = "SELECT * FROM edge" if mode.startswith("query") else "edge"
    partitions: dict[str, Any] = (
        {"partition_on": "id", "partition_num": 4}
        if mode.endswith("p4")
        else {"partition_on": None}
    )
    return pl.scan_database(source, source_uri, driver=driver, **partitions, **kwargs)


def _cell_corpus(
    mode: str,
    cell: int,
    scan: Callable[[str | None], pl.LazyFrame],
    base_scan: Callable[[str | None], pl.LazyFrame] | None,
) -> list[pl.Expr]:
    """
    The corpus of a cell (see `_scan_corpus.cell_predicates`).

    `scan(source)` scans the "table" or "query" source, or the whole table without
    pushdown for None; `base_scan` is the base cell's, None on the base cell. The base
    cell's table runs the whole corpus; its query is compared with it.
    """
    from tests._scan_corpus import (
        FULL,
        SAMPLE,
        CorpusPlans,
        cell_predicates,
        corpus_for_mode,
    )

    source = mode.removesuffix("-p4")
    if FULL or (base_scan is None and source == "table"):
        return corpus_for_mode(mode)

    def plans(scan: Callable[[str | None], pl.LazyFrame], source: str) -> CorpusPlans:
        return CorpusPlans.of(scan(source), scan(None).collect())

    base = plans(scan, "table") if base_scan is None else plans(base_scan, source)
    return cell_predicates(
        mode, plans(scan, source), base, sample=SAMPLE[mode], offset=cell
    )


def _corpus_params(cells: list[str]) -> list[Any]:
    """(cell, mode); outside the full run, partitioned modes on the base cell only."""
    full = bool(os.environ.get("POLARS_TEST_DATABASE_FULL"))
    return [
        pytest.param(cell, mode, id=f"{cell}-{mode}")
        for cell in cells
        for mode in _CORPUS_MODES
        if full or cell == cells[0] or not mode.endswith("p4")
    ]


# the first is the base cell
_SQLITE_DRIVERS = ["sqlite3", "adbc", "connectorx", "sqlalchemy"]


@pytest.mark.parametrize(("driver", "mode"), _corpus_params(_SQLITE_DRIVERS))
def test_corpus_sqlite(edge_sqlite: str, driver: str, mode: str) -> None:
    from tests._scan_corpus import check_corpus, ids

    pytest.importorskip(
        {"adbc": "adbc_driver_sqlite", "sqlite3": "sqlite3"}.get(driver, driver)
    )

    def scan(driver: str, mode: str | None) -> pl.LazyFrame:
        # mode None: the whole table, without pushdown
        if mode is None:
            return pl.scan_database(
                "edge",
                edge_sqlite,
                driver=driver,
                pushdown=False,
                schema_overrides=EDGE_OVERRIDES,
            )
        return _corpus_scan(edge_sqlite, driver, mode, schema_overrides=EDGE_OVERRIDES)

    reference = scan(driver, None).collect()
    assert ids(reference) == list(range(60))
    predicates = _cell_corpus(
        mode,
        _SQLITE_DRIVERS.index(driver),
        lambda m: scan(driver, m),
        None if driver == "sqlite3" else lambda m: scan("sqlite3", m),
    )
    lf = scan(driver, mode)
    failures, _ = check_corpus(lambda: lf, reference, predicates)
    assert not failures, "\n".join(failures[:40])


# DuckDB refuses connections to one file with different configurations at once: the
# base cell connects per query, read-only like the live connection
_DUCKDB_CONNECTIONS = ["factory", "uri", "live"]


@pytest.mark.parametrize(("connection", "mode"), _corpus_params(_DUCKDB_CONNECTIONS))
def test_corpus_duckdb(edge_duckdb: str, connection: str, mode: str) -> None:
    import duckdb

    from tests._scan_corpus import check_corpus, ids

    def connect() -> Any:
        # one thread: the base cell connects per query, and a pool of threads per
        # connection is costly next to the other test workers
        return duckdb.connect(edge_duckdb, read_only=True, config={"threads": 1})

    live = connect() if connection == "live" else None
    connections: dict[str, Any] = {
        "uri": f"duckdb:///{edge_duckdb}",
        "factory": connect,
        "live": live,
    }

    def scan(connection: str, mode: str | None) -> pl.LazyFrame:
        conn = connections[connection]
        if mode is None:
            return pl.scan_database("edge", conn, pushdown=False, partition_on=None)
        return _corpus_scan(conn, None, mode)

    try:
        reference = scan(connection, None).collect()
        assert ids(reference) == list(range(60))
        predicates = _cell_corpus(
            mode,
            _DUCKDB_CONNECTIONS.index(connection),
            lambda m: scan(connection, m),
            None if connection == "factory" else lambda m: scan("factory", m),
        )
        lf = scan(connection, mode)
        failures, _ = check_corpus(lambda: lf, reference, predicates)
        assert not failures, "\n".join(failures[:40])
    finally:
        if live is not None:
            live.close()


def test_property_sqlite(edge_sqlite: str) -> None:
    """Random predicate trees on SQLite; failing seeds belong in the fixed corpus."""
    pytest.importorskip("hypothesis")
    pytest.importorskip("adbc_driver_sqlite")
    from hypothesis import HealthCheck, given, settings

    from tests._scan_corpus import (
        expected_ids,
        ids,
        predicate_strategy,
    )

    reference = pl.scan_database(
        "edge", edge_sqlite, pushdown=False, schema_overrides=EDGE_OVERRIDES
    ).collect()
    lf = pl.scan_database("edge", edge_sqlite, schema_overrides=EDGE_OVERRIDES)

    @settings(
        max_examples=int(os.environ.get("POLARS_TEST_DATABASE_EXAMPLES", "50")),
        deadline=None,
        suppress_health_check=list(HealthCheck),
        database=None,
    )
    @given(predicate_strategy())
    def check(p: pl.Expr) -> None:
        expected = expected_ids(reference, p)
        if isinstance(expected, Exception):
            with pytest.raises(Exception):  # noqa: B017
                lf.filter(p).select("id").collect()
            return
        assert ids(lf.filter(p).select("id").collect()) == expected, str(p)

    check()


# Pushdown and planning regressions.


def test_scan_overridden_column_is_not_pushed(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT, r REAL, a TEXT, b TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
            [
                (1, "5", 1.0, "2020-1-5", "2020-01-10"),
                (2, "10", 1.7, "2020-01-01", "2020-01-02"),
            ],
        )
        conn.commit()
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        schema_overrides={"s": pl.Int64, "r": pl.Int64, "a": pl.Date, "b": pl.Date},
    )
    table = lf.collect()
    for p in [pl.col("s") > 5, pl.col("r") == 1, pl.col("a") < pl.col("b")]:
        assert_frame_equal(lf.filter(p).collect(), table.filter(p))


def test_scan_decimal_literal_is_not_rounded(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 1.50::DECIMAL(10,2)), "
            "(2, 2.00::DECIMAL(10,2))) v(id, d)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    table = lf.collect()
    for p in [
        pl.col("d") < pl.lit(Decimal("1.501")),
        pl.col("d") > pl.lit(Decimal("1.499")),
        pl.col("d") == pl.lit(Decimal("1.5")),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


def test_scan_enum_column_and_infinite_partition_bounds(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TYPE mood AS ENUM ('a', 'b')")
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'::mood, 'inf'::DOUBLE), "
            "(2, 'b'::mood, '-inf'::DOUBLE), (3, 'a'::mood, 0.0)) v(id, e, f)"
        )
    uri = f"duckdb:///{path}"
    for pushdown in [True, False]:
        lf = pl.scan_database("t", uri, pushdown=pushdown)
        out = lf.filter(pl.col("e") == "a").collect()
        assert sorted(out["id"].to_list()) == [1, 3]
    lf = pl.scan_database("t", uri, partition_on="f", partition_num=3)
    assert sorted(lf.collect()["id"].to_list()) == [1, 2, 3]


def test_scan_query_with_trailing_comment(shop_uri: str) -> None:
    lf = pl.scan_database("SELECT * FROM orders WHERE id <= 10 -- note", shop_uri)
    assert lf.filter(pl.col("id") > 5).select(pl.len()).collect().item() == 5


def test_scan_order_by_needs_one_ordered_read(shop_uri: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="global order"):
        pl.scan_database(
            "orders",
            shop_uri,
            order_by="id",
            partition_predicates=["id > 500", "id <= 500"],
        ).collect()
    # without pushdown a table scan still orders; a query can't be wrapped
    out = pl.scan_database("orders", shop_uri, order_by="id", pushdown=False).collect()
    assert out["id"].is_sorted()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="order_by"):
        pl.scan_database(
            "SELECT * FROM orders", shop_uri, order_by="id", pushdown=False
        ).collect()


def test_scan_query_that_isnt_repeatable_counts_its_own_rows(shop_uri: str) -> None:
    # each run picks other rows: the count that a negative slice needs is taken from
    # the rows read, not from a second run (which "changed the table")
    lf = pl.scan_database("SELECT * FROM orders WHERE abs(random()) % 2 = 0", shop_uri)
    tail = lf.tail(5).collect()
    assert tail.height == 5
    assert tail["id"].is_sorted()
    n = lf.select(pl.len()).collect().item()
    assert 0 < n < N_ROWS


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_sampled_integer_column_with_a_later_float(
    tmp_path: Path, driver: str
) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, x NUMERIC)")
        # typed Int64 from the first 1000 rows
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, i if i < 1000 else i + 0.5) for i in range(1100)],
        )
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    assert lf.collect_schema()["x"] == pl.Int64
    # never truncated to 1000
    with pytest.raises(pl.exceptions.PolarsError, match=r"1000\.5|Float64"):
        lf.collect()
    out = pl.scan_database(
        "t", f"sqlite:///{path}", driver=driver, schema_overrides={"x": pl.Float64}
    ).collect()
    assert out.filter(pl.col("id") == 1000)["x"].item() == 1000.5


@pytest.mark.parametrize("partition_num", [1, 4])
def test_scan_order_by_with_range_partitions_keeps_nulls_first(
    tmp_path: Path, partition_num: int
) -> None:
    # SQLite sorts NULLs first: they must come from the first partition, not the last
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, v TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, f"v{i}") for i in range(1, 101)] + [(None, "n1"), (None, "n2")],
        )
        conn.commit()
    lf = pl.scan_database(
        "t",
        f"sqlite:///{path}",
        order_by="id",
        partition_on="id",
        partition_num=partition_num,
    )
    ids = lf.collect()["id"].to_list()
    assert ids == [None, None, *range(1, 101)]


def test_scan_errors_dont_leak_passwords() -> None:
    # unencoded `#`, `?` and `/` in a URI password, and `;` in a braced ODBC value
    for uri in [
        "teradata://u:pa#Sekr3t@h/db",
        "teradata://u:pa?Sekr3t@h/db",
        "teradata://u:pa/Sekr3t@h/db",
        "Driver={No Such Driver 1};Server=s;UID=u;PWD={a;Sekr3t}",
    ]:
        with pytest.raises(Exception) as exc:  # noqa: PT011
            pl.scan_database("t", uri).collect()
        assert "Sekr3t" not in str(exc.value), uri


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_string_literal_with_nul(tmp_path: Path, driver: str) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, s TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a\0b"), (2, "ab")])
        conn.commit()
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    table = lf.collect()
    for p in [
        pl.col("s") == "a\0b",
        pl.col("s") != "a\0b",
        pl.col("s").is_in(["a\0b"]),
        pl.col("s").str.starts_with("a\0"),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, unicode(randomblob(1)) AS b FROM orders",
        "SELECT id, random() % 256 AS b FROM orders",
    ],
)
def test_scan_query_with_a_volatile_function_isnt_partitioned(
    shop_uri: str, query: str
) -> None:
    lf = pl.scan_database(
        query, shop_uri, partition_on="b", partition_num=4, partition_range=(0, 255)
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        lf.collect()


def test_scan_hugeint_literals(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    big = 99_999_999_999_999_999_999_999_999_999_999_999_999
    with duckdb.connect(path) as conn:
        # HUGEINT is read as Decimal(38, 0), though its maximum has 39 digits
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 5::HUGEINT), "
            f"(2, {big}::HUGEINT), (3, NULL::HUGEINT), (4, {2**127 - 1}::HUGEINT)) "
            "v(id, h)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}")
    table = lf.collect()
    assert table.height == 4
    values = pl.Series([5, big], dtype=pl.Int128).cast(pl.Decimal(38, 0))
    for p in [
        pl.col("h") == 5,
        pl.col("h") >= pl.lit(big, dtype=pl.Decimal(38, 0)),
        pl.col("h").is_in(values),
    ]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


def _assert_filters_match(lf: pl.LazyFrame, predicates: list[pl.Expr]) -> None:
    table = lf.collect()
    for p in predicates:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))


@pytest.mark.parametrize("source", ["t", "SELECT * FROM t"])
def test_scan_duckdb_collations_and_non_text_columns(
    tmp_path: Path, source: str
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, s VARCHAR COLLATE NOCASE, v VARCHAR, "
            "u UUID, e ENUM('a', 'b'))"
        )
        conn.execute(
            "INSERT INTO t VALUES "
            "(1, 'abc', 'abc', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a'), "
            "(2, 'ABC', 'ABC', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'b')"
        )
    lf = pl.scan_database(source, f"duckdb:///{path}")
    assert lf.collect_schema()["u"] == pl.String
    _assert_filters_match(
        lf,
        [
            pl.col("s") == "abc",
            pl.col("s") != "abc",
            pl.col("s") < "b",
            pl.col("s").is_in(["abc"]),
            # the database casts the literal to UUID / the enum
            pl.col("u") == "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",
            pl.col("u") == "zzz",
            pl.col("u") < "b",
            pl.col("e") == "nope",
        ],
    )
    # a VARCHAR is still compared, as a superset: the table declares a collation
    assert "filters 1/1" in lf.filter(pl.col("v") == "abc").explain()


def test_scan_duckdb_session_default_collation() -> None:
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (id INTEGER, v VARCHAR)")
    conn.execute("INSERT INTO t VALUES (1, 'abc'), (2, 'ABC')")
    conn.execute("SET default_collation = 'nocase'")
    lf = pl.scan_database("t", conn)
    _assert_filters_match(lf, [pl.col("v") == "abc", pl.col("v") < "b"])
    assert "(0 exact)" in lf.filter(pl.col("v") == "abc").explain()


@pytest.mark.parametrize("source", ["t", "SELECT * FROM t"])
def test_scan_sqlite_strings_without_text_affinity(tmp_path: Path, source: str) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, n NUMERIC, x, s TEXT COLLATE NOCASE)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [
                (1, "abc", "10", "abc"),
                (2, 1, 10, "ABC"),
                (3, 10, None, "b"),
                (4, 9, None, None),
                (5, "x", None, None),
            ],
        )
        conn.commit()
    lf = pl.scan_database(source, f"sqlite:///{path}")
    assert lf.collect_schema()["n"] == pl.String
    _assert_filters_match(
        lf,
        [
            pl.col("n") == "01",
            pl.col("n") == "1.0",
            pl.col("n") > "9",
            pl.col("x") == "10",
            pl.col("s") == "abc",
            pl.col("s") != "abc",
            pl.col("s") < "b",
        ],
    )


@pytest.mark.parametrize("driver", ["sqlite3", "sqlalchemy", "live"])
def test_scan_row_driver_with_leading_nulls(tmp_path: Path, driver: str) -> None:
    # more leading NULLs than a row driver's default inference length (100)
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, None if i < 150 else "x") for i in range(200)],
        )
        conn.commit()
    if driver == "live":
        with closing(sqlite3.connect(path, check_same_thread=False)) as live:
            df = pl.scan_database("t", live).collect()
    else:
        df = pl.scan_database("t", f"sqlite:///{path}", driver=driver).collect()
    assert df.schema["s"] == pl.String
    assert df["s"].null_count() == 150


def test_scan_sqlalchemy_literals_with_colons(tmp_path: Path) -> None:
    # SQLAlchemy's `text()` would read `:name` in a pushed literal as a placeholder,
    # and `\:` as an escaped `:`
    pytest.importorskip("sqlalchemy")
    path = tmp_path / "t.db"
    rows = [(1, "\\:x"), (2, ":x"), (3, "y"), (4, ":abc"), (5, ":lo"), (6, "0")]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", rows)
        conn.commit()
    uri = f"sqlite:///{path}"
    expected = pl.DataFrame(rows, schema=["id", "s"], orient="row")
    predicates = [
        pl.col("s") == "\\:x",
        pl.col("s") != "\\:x",
        pl.col("s").is_in(["\\:x"]),
        pl.col("s") == ":abc",
        pl.col("s").str.starts_with(":"),
        pl.col("s") == ":lo",
    ]
    lf = pl.scan_database("t", uri, driver="sqlalchemy")
    # with SQLAlchemy's `:name` parameters, the query's own placeholders are bound
    q = pl.scan_database(
        "SELECT id, s FROM t WHERE id > :lo", uri, driver="sqlalchemy", params={"lo": 0}
    )
    for p in predicates:
        want = sorted(expected.filter(p)["id"].to_list())
        assert sorted(lf.filter(p).collect()["id"].to_list()) == want, p
        assert sorted(q.filter(p).collect()["id"].to_list()) == want, p


def test_scan_sqlalchemy_live_connection_left_outside_a_transaction(
    shop_uri: str,
) -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(shop_uri)
    try:
        with engine.connect() as conn:
            assert pl.scan_database("orders", conn).collect().height == N_ROWS
            assert not conn.in_transaction()
            # a transaction the user opened is theirs: it stays open
            conn.exec_driver_sql("SELECT 1")
            pl.scan_database("orders", conn).head(1).collect()
            assert conn.in_transaction()
    finally:
        engine.dispose()


def test_scan_engine_with_a_small_pool(shop_uri: str, expected: pl.DataFrame) -> None:
    # more partitions than the pool has connections: they share them, rather than
    # wait for the pool to time out
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        shop_uri,
        poolclass=sqlalchemy.pool.QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    try:
        lf = pl.scan_database(
            "orders", engine, partition_on="id", partition_num=4, max_connections=4
        )
        assert_frame_equal(_sorted(lf.collect()), expected)
    finally:
        engine.dispose()


def test_connectorx_checks_the_connection_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a failed login fails `connect`, rather than every catalog lookup in turn
    from polars.io.database import _utils

    from polars_database import _bridge

    calls: list[str] = []

    def read(sql: str, *, connection_uri: str, protocol: str | None) -> Any:
        calls.append(connection_uri)
        if "wrong" in connection_uri:
            msg = "timed out waiting for connection: password authentication failed"
            raise RuntimeError(msg)
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr(_utils, "_read_sql_connectorx", read)
    monkeypatch.setattr(_bridge, "_CONNECTORX_CHECKED", set())
    with pytest.raises(RuntimeError, match="password authentication failed"):
        _bridge._ConnectorXSession("postgresql://u:wrong@h/db", "postgresql")
    good = "postgresql://u:right@h/db"
    _bridge._ConnectorXSession(good, "postgresql")
    _bridge._ConnectorXSession(good, "postgresql")
    assert calls == ["postgresql://u:wrong@h/db", good]


@pytest.mark.parametrize("driver", _SQLITE_DRIVERS)
def test_sqlite_uri_paths_mean_one_file_for_every_driver(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        {"adbc": "adbc_driver_sqlite", "sqlite3": "sqlite3"}.get(driver, driver)
    )
    monkeypatch.chdir(tmp_path)
    for name, x in [("rel.db", 1), ("a b.db", 2)]:
        with closing(sqlite3.connect(tmp_path / name)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (?)", (x,))
            conn.commit()

    def read(uri: str) -> int:
        return pl.scan_database("t", uri, driver=driver).collect().item()

    # as SQLAlchemy reads them: three slashes are relative, four absolute
    assert read("sqlite:///rel.db") == 1
    assert read(f"sqlite:///{tmp_path}/rel.db") == 1
    assert read("sqlite:rel.db") == 1
    assert read("sqlite:///a%20b.db") == 2
    if driver == "connectorx":
        with pytest.raises(Exception, match="connectorx can't open SQLite with URI"):
            read("sqlite:///rel.db?mode=ro")
    else:
        assert read("sqlite:///rel.db?mode=ro") == 1
    # no file was created elsewhere
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a b.db", "rel.db"]


def _skip_without_duckdb_adbc() -> None:
    try:
        pl.scan_database("SELECT 1 AS x", "duckdb://", driver="adbc").collect()
    except pl.exceptions.ComputeError as e:
        if "is not installed" in str(e):
            pytest.skip("no DuckDB ADBC driver")
        raise


@pytest.mark.parametrize("driver", ["duckdb", "adbc"])
def test_scan_duckdb_string_literal_with_nul(tmp_path: Path, driver: str) -> None:
    duckdb = pytest.importorskip("duckdb")
    if driver == "adbc":
        _skip_without_duckdb_adbc()
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute(
            "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a' || chr(0) || 'b'), "
            "(2, 'ab')) v(id, s)"
        )
    lf = pl.scan_database("t", f"duckdb:///{path}", driver=driver)
    table = lf.collect()
    for p in [pl.col("s") == "a\0b", pl.col("s") != "a\0b"]:
        assert_frame_equal(lf.filter(p).collect().sort("id"), table.filter(p))
        # the Python module reads the whole query text; C drivers stop at the NUL
        pushed = "filters 1/1" in lf.filter(p).explain()
        assert pushed is (driver == "duckdb")


def test_scan_engine_with_a_shared_connection_pool() -> None:
    # a StaticPool's one connection is the user's: used as a live connection, so its
    # temporary tables are visible and it isn't discarded
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=sqlalchemy.pool.StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("CREATE TEMPORARY TABLE tmp (id INTEGER, s TEXT)")
            conn.exec_driver_sql("INSERT INTO tmp VALUES (1, 'a'), (2, 'b')")
            conn.commit()
        lf = pl.scan_database("tmp", engine)
        assert sorted(lf.collect()["id"].to_list()) == [1, 2]
        assert lf.filter(pl.col("s") == "b").collect()["id"].to_list() == [2]
        assert sorted(lf.collect()["id"].to_list()) == [1, 2]
    finally:
        engine.dispose()


def test_engine_pool_capacity() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from polars.io.database._scan import _engine_pool_capacity

    def capacity(**kwargs: Any) -> int | None:
        engine = sqlalchemy.create_engine(
            "sqlite:///unused.db", poolclass=sqlalchemy.pool.QueuePool, **kwargs
        )
        return _engine_pool_capacity(engine)

    assert capacity(pool_size=2, max_overflow=3) == 5
    # unbounded
    assert capacity(pool_size=0) is None
    assert capacity(pool_size=2, max_overflow=-1) is None


def test_connectorx_missing_sqlite_file(tmp_path: Path) -> None:
    # ConnectorX would retry opening it for minutes
    pytest.importorskip("connectorx")
    path = tmp_path / "missing.db"
    with pytest.raises(Exception, match="no SQLite database file"):
        pl.scan_database("t", f"sqlite:///{path}", driver="connectorx").collect()
    assert not path.exists()


def test_connectorx_sqlite_windows_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # ConnectorX parses the URI as a URL: `sqlite://C:\\Users\\a` would come out as
    # `C/\\Users\\a`. Fully encoded, nothing in the path is a URL delimiter
    import ntpath

    from polars_database import _bridge

    monkeypatch.setattr(_bridge.os.path, "abspath", ntpath.abspath)
    path = "C:\\Users\\A b#1\\t.db"
    uri = _bridge._connectorx_uri(f"sqlite:///{path}", "sqlite")
    rest = uri.removeprefix("sqlite://")
    assert not set(rest) & set(":/\\?# "), uri
    assert unquote(rest) == path


def test_connectorx_sqlite_path_with_url_characters(tmp_path: Path) -> None:
    pytest.importorskip("connectorx")
    path = tmp_path / "A b#1%2F" / "t.db"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1 AS id")
        conn.commit()
    uri = f"sqlite:///{quote(str(path))}"
    lf = pl.scan_database("t", uri, driver="connectorx")
    assert lf.collect()["id"].to_list() == [1]


@pytest.mark.parametrize("driver", ["adbc", "sqlite3"])
def test_scan_empty_sqlite_table_is_typed_by_affinity(
    tmp_path: Path, driver: str
) -> None:
    if driver == "adbc":
        pytest.importorskip("adbc_driver_sqlite")
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE t (id INTEGER, s VARCHAR(10), f DOUBLE, b BLOB, d DATE)"
        )
    lf = pl.scan_database("t", f"sqlite:///{path}", driver=driver)
    assert dict(lf.collect_schema()) == {
        "id": pl.Int64,
        "s": pl.String,
        "f": pl.Float64,
        "b": pl.Binary,
        "d": pl.String,
    }
    # rows added after planning fit the planned schema
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO t VALUES (1, 'x', 1.5, x'00', '2020-01-02')")
        conn.commit()
    assert lf.collect().rows() == [(1, "x", 1.5, b"\x00", "2020-01-02")]


def test_scan_sqlite_user_schema_string_isnt_taken_as_text(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, x, c CHAR(3))")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?)", [(1, "10", "a"), (2, 10, "a ")]
        )
        conn.commit()
    schema = {"id": pl.Int64, "x": pl.String, "c": pl.String}
    lf = pl.scan_database("t", f"sqlite:///{path}", schema=schema)
    _assert_filters_match(lf, [pl.col("x") == "10", pl.col("x") != "10"])
    # SQLite doesn't pad CHAR(n): its strings compare exactly
    exact = pl.scan_database("t", f"sqlite:///{path}").filter(pl.col("c") == "a")
    assert "(1 exact)" in exact.explain()
    assert exact.collect()["id"].to_list() == [1]


def test_scan_duckdb_names_resolve_case_insensitively(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, u UUID, v VARCHAR)")
        conn.execute(
            "INSERT INTO t VALUES (1, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a')"
        )
    for source in ["T", "MAIN.T", "t"]:
        lf = pl.scan_database(source, f"duckdb:///{path}")
        _assert_filters_match(lf, [pl.col("u") == "zzz", pl.col("v") != "a"])
        assert "(1 exact)" in lf.filter(pl.col("v") == "a").explain(), source


def test_scan_duckdb_uri_paths_via_adbc(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    _skip_without_duckdb_adbc()
    with duckdb.connect(str(tmp_path / "d.db")) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1 AS id")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # relative, as the Python backends (and SQLAlchemy) read it
        for uri in ["duckdb:///d.db", "duckdb:///d.db?read_only=true"]:
            lf = pl.scan_database("t", uri, driver="adbc")
            assert lf.collect()["id"].to_list() == [1], uri
    finally:
        os.chdir(cwd)


def test_scan_shared_connection_pool_keeps_the_users_transaction() -> None:
    # the pool's reset on return would roll back the one shared connection
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(
        "sqlite://",
        poolclass=sqlalchemy.pool.StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE t (id INTEGER)")
            conn.exec_driver_sql("INSERT INTO t VALUES (1), (2)")
        user = engine.connect()
        user.exec_driver_sql("INSERT INTO t VALUES (3)")  # not committed
        assert pl.scan_database("t", engine).collect().height == 3
        assert user.in_transaction()
        user.commit()
        user.close()
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SELECT count(*) FROM t").scalar() == 3
    finally:
        engine.dispose()


def test_restored_setting_literals() -> None:
    from polars_database._bridge import _sql_str

    assert _sql_str("UTC", dialect="postgresql") == "'UTC'"
    # independent of `standard_conforming_strings`
    assert _sql_str("a\\b'c", dialect="postgresql") == "E'a\\\\b''c'"
    assert _sql_str("a\\b", dialect="mysql") == "'a\\\\b'"
