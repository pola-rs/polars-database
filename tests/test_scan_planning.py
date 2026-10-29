"""
`scan_database` planning checks that need no database server.

SQL and classifier snapshots per dialect are planned offline: with `schema=` given and
one partition, planning does no I/O, so SQL Server, Oracle and the warehouses are
covered here without servers. Regenerate the snapshot after a reviewed change with
`POLARS_UPDATE_SNAPSHOTS=1`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from tests._scan_corpus import EDGE_SCHEMA

pytest.importorskip("sqlglot")
pytest.importorskip("sqlalchemy")

SNAPSHOT = Path(__file__).parent / "snapshots" / "scan_sql.json"

DIALECT_URIS = {
    "postgresql": "postgresql://u@h/db",
    "mysql": "mysql://u@h/db",
    "mssql": "mssql://u@h/db",
    "oracle": "oracle://u@h/db",
    "sqlite": "sqlite:///x.db",
    "duckdb": "duckdb:///x.duckdb",
    "snowflake": "snowflake://u@acct/db",
    "bigquery": "bigquery://project/dataset",
    "databricks": "databricks://h/db",
    "clickhouse": "clickhouse://u@h/db",
    "trino": "trino://u@h/catalog",
    "redshift": "redshift://u@h/db",
}

CASES: dict[str, Any] = {
    "all": None,
    "head": "head",
    "select": ["id", "s"],
    "int_cmp": pl.col("i32") > 42,
    "int_between": pl.col("i64").is_between(0, 10),
    "float_cmp": pl.col("f64") > 1.5,
    "decimal_cmp": pl.col("dec") >= Decimal("12.5"),
    "str_eq": pl.col("s") == "abc",
    "str_ne": pl.col("s") != "abc",
    "str_lt": pl.col("s") < "b",
    "str_quote": pl.col("s") == "it's",
    "str_backslash": pl.col("s") == "a\\b",
    "str_unicode": pl.col("s") == "é",
    "str_empty": pl.col("s") == "",
    "like_prefix": pl.col("s").str.starts_with("a%_"),
    "like_contains": pl.col("s").str.contains("[x]", literal=True),
    "is_in": pl.col("s").is_in(["a", "b", None], nulls_equal=True),
    "is_in_long": pl.col("i64").is_in(list(range(5000))),
    "is_null": pl.col("d").is_null(),
    "eq_missing_null": pl.col("i16").eq_missing(None),
    "ne_missing": pl.col("i16").ne_missing(3),
    "not_exact": ~(pl.col("i16") > 0),
    "not_superset": ~(pl.col("s") == "abc"),
    "or": (pl.col("i16") == 1) | (pl.col("b")),
    "date_cmp": pl.col("d") >= date(2026, 1, 1),
    "ts_cmp": pl.col("ts") < datetime(2026, 1, 1, 12, 30, 0, 123456),
    "bool_col": pl.col("b"),
    "arithmetic": (pl.col("i16") * 2) > 4,
}


def _plan(dialect: str, case: str) -> dict[str, str]:
    lf = pl.scan_database(
        "s.t",
        DIALECT_URIS[dialect],
        schema=EDGE_SCHEMA,
        driver="sqlalchemy",
        partition_on=None,
    )
    op = CASES[case]
    if isinstance(op, str):
        lf = lf.head(10)
    elif isinstance(op, list):
        lf = lf.select(op)
    elif op is not None:
        lf = lf.filter(op)
    lines = {
        k.strip(): v.strip()
        for k, _, v in (
            line.partition(": ")
            for line in lf.explain().splitlines()
            if line.strip().startswith(("pushed:", "sql[0]:"))
        )
    }
    return {"pushed": lines["pushed"], "sql": lines["sql[0]"]}


def _all_plans() -> dict[str, dict[str, dict[str, str]]]:
    return {d: {c: _plan(d, c) for c in CASES} for d in DIALECT_URIS}


def test_sql_snapshots() -> None:
    """Quoting, literals, LIMIT forms, IN chunking and exactness, per dialect."""
    plans = _all_plans()
    if os.environ.get("POLARS_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(plans, indent=2, ensure_ascii=False) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    diffs = [
        f"{d}/{c}: {plans[d][c]} != {expected.get(d, {}).get(c)}"
        for d in plans
        for c in plans[d]
        if plans[d][c] != expected.get(d, {}).get(c)
    ]
    assert not diffs, "\n".join(diffs)


def test_sql_snapshot_details() -> None:
    plans = _all_plans()
    # LIMIT forms
    assert plans["postgresql"]["head"]["sql"].endswith("LIMIT 10")
    assert "SELECT TOP 10" in plans["mssql"]["head"]["sql"]
    assert plans["oracle"]["head"]["sql"].endswith("FETCH FIRST 10 ROWS ONLY")
    # literals
    assert "N'é'" in plans["mssql"]["str_unicode"]["sql"]
    assert "'it''s'" in plans["postgresql"]["str_quote"]["sql"]
    assert "'a\\\\b'" in plans["mysql"]["str_backslash"]["sql"]
    assert "DATE '2026-01-01'" in plans["oracle"]["date_cmp"]["sql"]
    assert "[[]x]" in plans["mssql"]["like_contains"]["sql"] or (
        "\\[x]" in plans["mssql"]["like_contains"]["sql"]
    )
    # SQL Server lists are kept short: a long IN becomes a BETWEEN superset
    assert "BETWEEN 0 AND 4999" in plans["mssql"]["is_in_long"]["sql"]
    assert "(0 exact)" in plans["mssql"]["is_in_long"]["pushed"]
    assert " IN (" in plans["postgresql"]["is_in_long"]["sql"]
    # classification: NOT of a superset is never pushed, NOT of an exact one is
    assert "filters 0/1" in plans["mysql"]["not_superset"]["pushed"]
    assert "filters 1/1 (1 exact)" in plans["mysql"]["not_exact"]["pushed"]
    assert "filters 1/1 (0 exact)" in plans["mysql"]["str_eq"]["pushed"]
    assert "filters 1/1 (1 exact)" in plans["postgresql"]["str_eq"]["pushed"]
    # floats: pushed only where comparisons agree with Polars (total order NaN)
    assert "filters 0/1" in plans["bigquery"]["float_cmp"]["pushed"]
    assert "filters 1/1 (1 exact)" in plans["postgresql"]["float_cmp"]["pushed"]
    # Oracle: '' is NULL, so an empty-string literal isn't pushed
    assert "filters 0/1" in plans["oracle"]["str_empty"]["pushed"]
    # arithmetic is not tier 1
    assert "filters 0/1" in plans["postgresql"]["arithmetic"]["pushed"]
    # an empty projection selects a constant
    assert "projection 2/11" in plans["postgresql"]["select"]["pushed"]


@pytest.mark.parametrize(
    ("dialect", "query", "reason"),
    [
        ("mssql", "WITH a AS (SELECT 1 AS x) SELECT x FROM a", "CTE"),
        ("mssql", "SELECT x FROM t ORDER BY x", "ORDER BY"),
        ("postgresql", "SELECT x, x FROM t", "duplicate"),
    ],
)
def test_query_wrapping_falls_back(dialect: str, query: str, reason: str) -> None:
    lf = pl.scan_database(
        query, DIALECT_URIS[dialect], schema={"x": pl.Int64}, driver="sqlalchemy"
    )
    plan = lf.filter(pl.col("x") > 1).explain()
    assert "pushdown: off" in plan
    assert reason in plan
    assert f"sql[0]: {query}" in plan


def test_query_wrapping() -> None:
    lf = pl.scan_database(
        "SELECT x FROM t ORDER BY x OFFSET 0 ROWS",
        DIALECT_URIS["mssql"],
        schema={"x": pl.Int64},
        driver="sqlalchemy",
    ).filter(pl.col("x") > 1)
    assert "FROM (SELECT x FROM t ORDER BY x OFFSET 0 ROWS) AS [_pl]" in lf.explain()


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM t",
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "SELECT * INTO copy FROM t",
        "SELECT 1; DROP TABLE t",
        "/* sneaky */ UPDATE t SET x = 1",
    ],
)
@pytest.mark.parametrize("pushdown", [True, False])
def test_write_queries_are_rejected(query: str, pushdown: bool) -> None:
    lf = pl.scan_database(
        query,
        DIALECT_URIS["postgresql"],
        schema={"x": pl.Int64},
        driver="sqlalchemy",
        pushdown=pushdown,
        source_type="query",
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="read"):
        lf.collect_schema()


def test_cse_of_identical_scans(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT)")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(i, "g") for i in range(10)])
    lf = pl.scan_database("t", f"sqlite:///{db}")
    joined = lf.join(lf, on="id")
    assert "CACHE" in joined.explain()
    assert joined.collect().height == 10


# secrets never appear in plans, errors or verbose logs

SECRET = "s3cr3t-Pa55word"


@pytest.mark.parametrize(
    ("connection", "driver"),
    [
        # a host that can't resolve fails at once; a refused connect is retried with
        # backoff (2.6s), which the MySQL cell covers: its retry log mustn't leak either
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "adbc"),
        # ConnectorX's pool waits 30s on a refused connection; a bad option fails now
        (f"postgresql://user:{SECRET}@127.0.0.1:1/db?sslmode=bogus", "connectorx"),
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "psycopg"),
        (f"postgresql://user:{SECRET}@nohost.invalid:1/db", "sqlalchemy"),
        (f"mysql://user:{SECRET}@127.0.0.1:1/db", "pymysql"),
        (f"Driver={{Nope}};Server=127.0.0.1,1;UID=u;PWD={SECRET}", "pyodbc"),
    ],
)
def test_secrets_absent(
    connection: str, driver: str, capfd: pytest.CaptureFixture[str]
) -> None:
    module = {"adbc": "adbc_driver_postgresql", "sqlalchemy": "sqlalchemy"}.get(
        driver, driver
    )
    pytest.importorskip(module)
    lf = pl.scan_database("t", connection, driver=driver)
    with pl.Config(verbose=True):
        assert SECRET not in repr(lf)
        with pytest.raises(Exception) as exc:
            lf.collect()
    out, err = capfd.readouterr()
    assert SECRET not in str(exc.value)
    assert SECRET not in out + err


def test_secret_from_credential_provider_absent(
    capfd: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("psycopg")
    lf = pl.scan_database(
        "t",
        "postgresql://user@nohost.invalid:1/db",
        driver="psycopg",
        credential_provider=lambda: {"password": SECRET},
    )
    with pl.Config(verbose=True), pytest.raises(Exception) as exc:
        lf.collect()
    out, err = capfd.readouterr()
    assert SECRET not in str(exc.value) + out + err


def test_plan_with_password_hides_it() -> None:
    lf = pl.scan_database(
        "t",
        f"postgresql://user:{SECRET}@h/db",
        schema={"x": pl.Int64},
        driver="sqlalchemy",
    )
    assert SECRET not in lf.explain()
    assert SECRET not in str(lf)


# driver resolution


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        ("sqlite:////tmp/a.db", "//tmp/a.db"),
        ("duckdb:///C:\\Users\\a\\t.duckdb", "C:\\Users\\a\\t.duckdb"),
        ("sqlite:///d:/a/b.db", "d:/a/b.db"),
        ("sqlite://", ":memory:"),
        ("duckdb:///:memory:", ":memory:"),
    ],
)
def test_file_path_of_uri(uri: str, path: str) -> None:
    # on Windows, `/C:/...` would be a network path
    from polars_database._bridge import _file_path

    assert _file_path(uri) == path


def _hide_modules(monkeypatch: pytest.MonkeyPatch, *hidden: str) -> None:
    from polars_database import _bridge

    real = _bridge._has_module
    monkeypatch.setattr(
        _bridge,
        "_has_module",
        lambda m: False if m.split(".")[0] in hidden or m in hidden else real(m),
    )


def test_resolution_picks_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("psycopg")
    _hide_modules(monkeypatch, "adbc_driver_postgresql", "connectorx")
    lf = pl.scan_database(
        "t", "postgresql://u@h/db", schema={"x": pl.Int64}, partition_on=None
    )
    plan = lf.explain()
    assert "via python:psycopg" in plan

    df = pl.io.database.show_drivers("postgresql")
    available = dict(zip(df["backend"], df["available"], strict=True))
    assert available["connectorx"] is False
    assert available["python:psycopg"] is True


def test_pin_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "connectorx")
    lf = pl.scan_database(
        "t", "postgresql://u@h/db", schema={"x": pl.Int64}, driver="connectorx"
    )
    with pytest.raises(pl.exceptions.ComputeError) as exc:
        lf.collect_schema()
    msg = str(exc.value)
    assert "pinned database driver 'connectorx' is not installed" in msg
    assert "pip install connectorx" in msg


def test_no_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "connectorx", "trino", "sqlalchemy", "adbc_driver_trino")
    lf = pl.scan_database("t", "trino://u@h/c", schema={"x": pl.Int64})
    with pytest.raises(pl.exceptions.ComputeError) as exc:
        lf.collect_schema()
    msg = re.sub(
        r"Checked ADBC manifests in .*",
        "Checked ADBC manifests in <dirs>",
        str(exc.value),
    )
    msg = msg.split("\n\nThis error occurred")[0]
    assert msg == (
        "no driver available for database dialect 'trino' (trino://u@h/c)\n"
        "  Install one of (best first):\n"
        "    dbc install trino\n"
        "    pip install connectorx\n"
        "    pip install trino\n"
        "    pip install sqlalchemy  (plus the dialect's DB-API driver)\n"
        "  Or pin one with driver=..., or pass a connection factory.\n"
        "  Checked ADBC manifests in <dirs>"
    )


def test_probing_imports_no_heavy_modules() -> None:
    code = (
        "import sys, polars as pl\n"
        "before = set(sys.modules)\n"
        "pl.io.database.show_drivers()\n"
        "new = set(sys.modules) - before\n"
        "heavy = {'sqlalchemy', 'pyarrow', 'pandas', 'psycopg', 'pymysql', 'oracledb',"
        " 'connectorx', 'duckdb', 'sqlglot', 'clickhouse_connect'}\n"
        "print(sorted({m.split('.')[0] for m in new} & heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


# 1.6, 1.7, 1.9: partitioning, concurrency and the reader


@pytest.fixture
def numbers_db(tmp_path: Path) -> str:
    db = tmp_path / "n.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE n (id INTEGER PRIMARY KEY, s TEXT)")
        conn.executemany(
            "INSERT INTO n VALUES (?, ?)", [(i, str(i)) for i in range(1000)]
        )
    return f"sqlite:///{db}"


def test_bounds_use_exact_filters(
    numbers_db: str, capfd: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("n", numbers_db, partition_on="id", partition_num=4)
    with pl.Config(verbose=True):
        out = lf.filter(pl.col("id") > 700).collect()
    _, err = capfd.readouterr()
    assert out.height == 299
    bounds = next(line for line in err.splitlines() if "partition bounds" in line)
    assert 'MAX("id")' in bounds
    assert '"id" > 700' in bounds


@pytest.mark.parametrize("cap", [1, 3])
def test_max_connections_caps_concurrency(
    numbers_db: str, cap: int, capfd: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=12, max_connections=cap
    )
    with pl.Config(verbose=True):
        assert lf.collect().height == 1000
    _, err = capfd.readouterr()
    assert f"max_concurrent_scans: {cap}," in err


@pytest.mark.parametrize(("env", "expected"), [("5", 2), ("1", 1)])
def test_env_override_lowers_but_never_raises_connection_cap(
    numbers_db: str,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    env: str,
    expected: int,
) -> None:
    # more readers than sessions could deadlock an ordered read
    pytest.importorskip("adbc_driver_sqlite")
    monkeypatch.setenv("POLARS_MAX_CONCURRENT_SCANS", env)
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=12, max_connections=2
    )
    with pl.Config(verbose=True):
        assert lf.collect().height == 1000
    _, err = capfd.readouterr()
    assert f"max_concurrent_scans: {expected}," in err


def test_env_override_above_connection_cap_in_order(
    numbers_db: str, tmp_path: Path
) -> None:
    # An override above the session count, and morsels small enough that a reader's
    # output fills before it is done: a later partition must never hold the session the
    # first one waits for. In a subprocess, so a deadlock fails instead of hanging.
    pytest.importorskip("adbc_driver_sqlite")
    out = tmp_path / "out.parquet"
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_database('n', {numbers_db!r}, partition_on='id', "
        "partition_num=12, max_connections=2)\n"
        f"lf.sink_parquet({str(out)!r})\n"
    )
    env = {
        **os.environ,
        "POLARS_MAX_CONCURRENT_SCANS": "5",
        "POLARS_IDEAL_MORSEL_SIZE": "4",
    }
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("the ordered read with more readers than sessions hung")
    assert pl.read_parquet(out)["id"].to_list() == list(range(1000))


def test_live_connection_reads_one_partition_at_a_time(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(100)")
    live = duckdb.connect(path, read_only=True)
    try:
        lf = pl.scan_database(
            "t", live, partition_on="id", partition_num=4, max_connections=4
        )
        with pl.Config(verbose=True):
            assert lf.collect().height == 100
        _, err = capfd.readouterr()
        assert "max_concurrent_scans: 1," in err
    finally:
        live.close()


def test_connect_is_retried(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(10)")
    fail = {"next": 0}

    def factory() -> Any:
        if fail["next"] > 0:
            fail["next"] -= 1
            msg = "transient"
            raise ConnectionError(msg)
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory, partition_on=None)
    lf.collect_schema()
    fail["next"] = 3  # fails three times, then works: within the retry budget
    assert lf.collect().height == 10
    fail["next"] = 4  # one more than the budget
    with pytest.raises(pl.exceptions.ComputeError, match="transient"):
        lf.collect()


def test_failed_login_is_not_retried(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    with duckdb.connect(path) as conn:
        conn.execute("CREATE TABLE t AS SELECT range AS id FROM range(10)")
    calls = {"n": 0, "fail": False}

    def factory() -> Any:
        calls["n"] += 1
        if calls["fail"]:
            msg = 'password authentication failed for user "polars"'
            raise PermissionError(msg)
        return duckdb.connect(path, read_only=True)

    lf = pl.scan_database("t", factory, partition_on=None)
    lf.collect_schema()
    calls.update(n=0, fail=True)
    with pytest.raises(pl.exceptions.ComputeError, match="password"):
        lf.collect()
    assert calls["n"] == 1

    # a query error fails the same way again: it isn't re-run either
    calls.update(n=0, fail=False)
    bad = pl.scan_database(
        "SELECT CAST('x' || id AS INTEGER) AS x FROM t", factory, partition_on=None
    )
    with pytest.raises(pl.exceptions.PolarsError):
        bad.collect()
    assert calls["n"] <= 3  # dialect detection, planning, then one read


def test_cast_error_names_everything(numbers_db: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database("n", numbers_db, schema={"id": pl.Int64, "s": pl.Date})
    with pytest.raises(pl.exceptions.SchemaError) as exc:
        lf.collect()
    msg = str(exc.value)
    assert "'s'" in msg
    assert "date" in msg
    assert "partition 0" in msg
    assert "adbc:sqlite" in msg


@pytest.mark.parametrize("n", [1, 2, 7, 64])
@pytest.mark.parametrize("table", ["with_nulls", "empty", "single"])
def test_every_row_read_once(tmp_path: Path, n: int, table: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    db = tmp_path / "p.db"
    all_rows: dict[str, list[tuple[int, int | None]]] = {
        "with_nulls": [(i, None if i % 9 == 0 else i * 3 - 50) for i in range(300)],
        "empty": [],
        "single": [(i, 7) for i in range(20)],
    }
    rows = all_rows[table]
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)")
        conn.executemany("INSERT INTO p VALUES (?, ?)", rows)
    uri = f"sqlite:///{db}"
    cases: list[dict[str, Any]] = [
        {"partition_on": "k", "partition_num": n},
        # bounds that don't cover the data: rows outside land in the end partitions
        {"partition_on": "k", "partition_num": n, "partition_range": (100, 200)},
    ]
    for kwargs in cases:
        out = pl.scan_database(
            "p", uri, schema={"id": pl.Int64, "k": pl.Int64}, **kwargs
        )
        ids = out.select("id").collect()["id"].to_list()
        assert sorted(ids) == [r[0] for r in rows], kwargs


# params and consistency


def test_params(numbers_db: str) -> None:
    lf = pl.scan_database(
        "SELECT id, s FROM n WHERE id < ? AND s != ?",
        numbers_db,
        params=[10, "3"],
        driver="sqlite3",
    )
    out = lf.filter(pl.col("id") > 5).collect()
    assert sorted(out["id"].to_list()) == [6, 7, 8, 9]
    # every partition binds the same parameters
    lf = pl.scan_database(
        "SELECT id FROM n WHERE id < ?",
        numbers_db,
        params=[100],
        driver="sqlite3",
        partition_on="id",
        partition_num=4,
    )
    assert lf.select(pl.len()).collect().item() == 100


def test_params_errors(numbers_db: str) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="query source"):
        pl.scan_database("n", numbers_db, params=[1], driver="sqlite3").collect()
    pytest.importorskip("adbc_driver_sqlite")
    with pytest.raises(pl.exceptions.InvalidOperationError, match="binds parameters"):
        pl.scan_database(
            "SELECT id FROM n WHERE id < ?", numbers_db, params=[1], driver="adbc"
        ).collect()


def test_params_change_the_scan(numbers_db: str) -> None:
    q = "SELECT id FROM n WHERE id < ?"
    a = pl.scan_database(q, numbers_db, params=[5], driver="sqlite3")
    b = pl.scan_database(q, numbers_db, params=[7], driver="sqlite3")
    out = pl.concat([a, b]).select(pl.len()).collect()
    assert out.item() == 12


def test_snapshot_consistency_with_partitions(numbers_db: str) -> None:
    pytest.importorskip("adbc_driver_sqlite")
    lf = pl.scan_database(
        "n", numbers_db, partition_on="id", partition_num=2, consistency="snapshot"
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="snapshot"):
        lf.collect()
    one = pl.scan_database("n", numbers_db, partition_on=None, consistency="snapshot")
    assert one.collect().height == 1000


def test_all_null_first_batch(tmp_path: Path) -> None:
    # sqlite3 types each batch on its own: the first one is all NULL
    path = tmp_path / "t.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
        conn.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [(i, None if i <= 500 else i) for i in range(1, 1001)],
        )
        conn.commit()
    lf = pl.scan_database(
        "SELECT id, v FROM t ORDER BY id",
        f"sqlite:///{path}",
        driver="sqlite3",
        batch_size=100,
    )
    assert lf.collect_schema()["v"] == pl.Int64
    assert lf.collect()["v"].null_count() == 500
