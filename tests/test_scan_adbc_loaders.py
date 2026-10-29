"""
The two ADBC loaders (`POLARS_DATABASE_ADBC_LOADER`) return the same data.

The whole suite can run on either loader by setting `POLARS_DATABASE_ADBC_LOADER` for
the run. These tests switch it per scan and compare the loaders directly: on SQLite
always, and on Postgres and MySQL when `POLARS_TEST_POSTGRES_URI` /
`POLARS_TEST_MYSQL_URI` are set.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytest.importorskip("sqlglot")
pytest.importorskip("adbc_driver_manager")
pytest.importorskip("adbc_driver_sqlite")

LOADERS = ["rust", "python"]


@pytest.fixture
def edge_db(tmp_path: Path) -> str:
    path = tmp_path / "edge.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, i INTEGER, f REAL, s TEXT, b BLOB)")
    rows: list[tuple[Any, ...]] = [
        (1, 2**63 - 1, float("inf"), "", b""),
        (2, -(2**63), -0.0, "émoji 🦆", b"\x00\xff"),
        (3, 0, 1.5, "it's %_\\", None),
        (4, None, None, None, b"x"),
    ]
    rows += [(i, i, i / 7, f"s{i}", None) for i in range(5, 20_005)]
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute("CREATE TABLE empty (a INTEGER, b TEXT)")
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


def _each_loader(
    monkeypatch: pytest.MonkeyPatch, read: Callable[[], pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    out = {}
    for loader in LOADERS:
        monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", loader)
        out[loader] = read()
    return out


def _assert_same(results: dict[str, pl.DataFrame], sort: bool = True) -> None:
    rust, python = results["rust"], results["python"]
    if sort:
        rust, python = rust.sort(rust.columns), python.sort(python.columns)
    assert_frame_equal(rust, python)


@pytest.mark.parametrize(
    "read",
    [
        lambda uri: pl.scan_database("t", uri, driver="adbc").collect(),
        lambda uri: (
            pl.scan_database(
                "t", uri, driver="adbc", partition_on="id", partition_num=4
            )
            .filter(pl.col("f") > 3, pl.col("s").str.starts_with("s1"))
            .select("id", "s")
            .collect(engine="streaming")
        ),
        lambda uri: pl.scan_database(
            "SELECT id, s FROM t WHERE id < 100", uri, driver="adbc"
        ).collect(),
        lambda uri: pl.scan_database("empty", uri, driver="adbc").collect(),
        lambda uri: (
            pl.scan_database("t", uri, driver="adbc").select(pl.len()).collect()
        ),
    ],
    ids=["table", "partitioned", "query", "empty", "count"],
)
def test_loaders_read_the_same_sqlite(
    edge_db: str, monkeypatch: pytest.MonkeyPatch, read: Callable[[str], pl.DataFrame]
) -> None:
    _assert_same(_each_loader(monkeypatch, lambda: read(edge_db)))


def test_loaders_stop_early(edge_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # an early stop drops the stream before its end: the statement and connection close
    out = _each_loader(
        monkeypatch,
        lambda: (
            pl.scan_database("t", edge_db, driver="adbc", batch_size=100)
            .head(3)
            .collect(engine="streaming")
        ),
    )
    assert out["rust"].height == out["python"].height == 3


def test_loaders_report_sql_errors(
    edge_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for loader in LOADERS:
        monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", loader)
        with pytest.raises(pl.exceptions.ComputeError, match="no_such_column"):
            pl.scan_database(
                "SELECT no_such_column FROM t", edge_db, driver="adbc"
            ).collect()


def test_loader_setting_errors(edge_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", "go")
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="must be 'rust' or 'python'"
    ):
        pl.scan_database("t", edge_db, driver="adbc").collect()

    # without adbc_driver_manager, the python loader names the package; no fallback
    monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", "python")
    monkeypatch.setitem(sys.modules, "adbc_driver_manager", None)
    with pytest.raises(pl.exceptions.ComputeError, match="pip install adbc-driver"):
        pl.scan_database("t", edge_db, driver="adbc").collect()


@pytest.mark.parametrize("loader", LOADERS)
def test_loaders_redact_rejected_credentials(
    edge_db: str, loader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the SQLite driver quotes a rejected option, value included
    monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", loader)
    with pytest.raises(pl.exceptions.ComputeError) as e:
        pl.scan_database(
            "t", edge_db, driver="adbc", connect_options={"password": "s3cr3t-pw"}
        ).collect()
    assert "s3cr3t-pw" not in str(e.value)


def test_python_loader_rejects_driver_manager_arguments(
    edge_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", "python")
    with pytest.raises(pl.exceptions.ComputeError, match="can't be used with"):
        pl.scan_database(
            "t", edge_db, driver="adbc", connect_options={"entrypoint": "x"}
        ).collect()


SERVERS = {
    "postgres": ("POLARS_TEST_POSTGRES_URI", "adbc_driver_postgresql"),
    "mysql": ("POLARS_TEST_MYSQL_URI", None),
}


@pytest.mark.parametrize("server", sorted(SERVERS))
@pytest.mark.parametrize("source", ["edge", "edge_types"])
@pytest.mark.xdist_group("adbc_loaders")
def test_loaders_read_the_same_servers(
    server: str, source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    var, module = SERVERS[server]
    uri = os.environ.get(var)
    if uri is None:
        pytest.skip(f"needs {var}")
    if module is not None:
        pytest.importorskip(module)
    try:
        pl.scan_database(source, uri, driver="adbc").collect_schema()
    except pl.exceptions.ComputeError as e:
        if "not found" in str(e):
            pytest.skip(f"no ADBC driver for {server}")
        raise
    # unconstrained `numeric` with 12 decimal places: text, as in the server suite
    overrides = (
        {"nu": pl.String} if (server, source) == ("postgres", "edge_types") else None
    )
    _assert_same(
        _each_loader(
            monkeypatch,
            lambda: pl.scan_database(
                source,
                uri,
                driver="adbc",
                partition_on="id",
                partition_num=3,
                schema_overrides=overrides,
            ).collect(),
        )
    )


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.xdist_group("adbc_loaders")
def test_loaders_reject_postgres_infinity(
    loader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the ADBC Postgres driver wraps ±infinity to dates around 290 000 BC
    uri = os.environ.get("POLARS_TEST_POSTGRES_URI")
    if uri is None:
        pytest.skip("needs POLARS_TEST_POSTGRES_URI")
    pytest.importorskip("adbc_driver_postgresql")
    monkeypatch.setenv("POLARS_DATABASE_ADBC_LOADER", loader)
    for value in [
        "'infinity'::timestamp",
        "'-infinity'::timestamptz",
        "'infinity'::date",
        "'294276-12-31 23:59:59'::timestamp",  # past what the driver's shift can hold
        "ARRAY['2020-01-01'::timestamp, 'infinity']",
        "ARRAY['-infinity'::date]",
    ]:
        query = f"SELECT 1 AS id, {value} AS v"
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="'v' holds a Postgres"
        ):
            pl.scan_database(query, uri, driver="adbc").collect()
    # the earliest and latest Postgres values read as they are
    finite = (
        "SELECT 1 AS id, '4714-11-24 00:00:00 BC'::timestamp AS lo, "
        "'4714-11-24 BC'::date AS dlo, '200000-12-31'::date AS dhi"
    )
    df = pl.scan_database(finite, uri, driver="adbc").collect()
    assert df["dlo"].dt.year().item() == -4713
    assert df["dhi"].dt.year().item() == 200000
