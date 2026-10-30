"""
The sqlglot SQL renderer behind the Rust `SqlRenderer` trait.

Rust sends a dialect-neutral `SqlQuery` as JSON; this module builds sqlglot
expressions from it and generates SQL for the target dialect. It is the only code
that imports sqlglot, and it only runs on the machine that plans the query.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from polars._dependencies import import_optional

sqlglot = import_optional(
    "sqlglot",
    err_prefix="scan_database needs",
    install_message="Install it with: pip install sqlglot",
)
exp = sqlglot.exp

_BINARY = {
    "Eq": exp.EQ,
    "NotEq": exp.NEQ,
    "Lt": exp.LT,
    "LtEq": exp.LTE,
    "Gt": exp.GT,
    "GtEq": exp.GTE,
    "And": exp.And,
    "Or": exp.Or,
}

# Temporal literals are cast with microsecond precision, so that no dialect rounds them.
_TIMESTAMP = "TIMESTAMP(6)"
_TIMESTAMPTZ = "TIMESTAMPTZ(6)"
_TIME = "TIME(6)"
# Spark SQL's timestamps are always microseconds and take no precision. sqlglot drops it
# for these dialects up to 30.24, but keeps it for Databricks from 30.25.
_NO_TEMPORAL_PRECISION = {"databricks", "spark", "hive"}


def version() -> str:
    return str(sqlglot.__version__)


def _ident(name: str) -> Any:
    return exp.to_identifier(name, quoted=True)


def _column(name: str) -> Any:
    return exp.Column(this=_ident(name))


def _raw(sql: str) -> Any:
    # inserted verbatim: user-written SQL is never reparsed or rewritten
    return exp.Var(this=sql)


def _cast(text: str, dtype: str) -> Any:
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build(dtype))


def _ansi(kind: str, text: str) -> Any:
    # `DATE '…'` / `TIMESTAMP '…'`: Oracle's CAST from text depends on NLS settings
    return _raw(f"{kind} '{text}'")


def _string(text: str, dialect: str | None) -> Any:
    if dialect == "tsql":
        # N'…': a plain literal is converted to the database's code page first, which can
        # turn characters it lacks into '?'
        return exp.National(this=text)
    if dialect == "postgres" and "\\" in text:
        # E'…' reads backslashes the same whatever `standard_conforming_strings` is; a
        # plain '…' holding one means something else when it is off
        escaped = text.replace("\\", "\\\\").replace("'", "''")
        return _raw(f"E'{escaped}'")
    return exp.Literal.string(text)


def _literal(v: Any, dialect: str | None) -> Any:
    if v == "Null":
        return exp.Null()
    ((kind, x),) = v.items()
    if kind == "Bool":
        return exp.Boolean(this=x)
    if kind == "Int":
        return exp.Literal.number(int(x))
    if kind == "Float":
        return exp.Literal.number(repr(float(x)))
    if kind == "Decimal":
        if dialect == "clickhouse":
            # a bare literal is parsed as a float, which overflows for wide decimals
            scale = len(x.partition(".")[2])
            return _cast(x, f"Decimal(38, {scale})")
        return exp.Literal.number(x)
    if kind == "String":
        return _string(x, dialect)
    if kind == "Date":
        if dialect == "oracle":
            return _ansi("DATE", x)
        if dialect == "clickhouse":
            # `Date` starts in 1970 and wraps silently; `Date32` covers 1900-2299
            return exp.Cast(
                this=exp.Literal.string(x),
                to=exp.DataType.build("Date32", dialect="clickhouse"),
            )
        return _cast(x, "DATE")
    if kind == "Timestamp":
        if dialect == "oracle":
            return _ansi("TIMESTAMP", x["text"] + (" +00:00" if x["with_tz"] else ""))
        if dialect == "clickhouse":
            return exp.Cast(
                this=exp.Literal.string(x["text"]),
                to=exp.DataType.build(
                    "DateTime64(6, 'UTC')" if x["with_tz"] else "DateTime64(6)",
                    dialect="clickhouse",
                ),
            )
        if dialect in _NO_TEMPORAL_PRECISION:
            text = x["text"] + ("+00:00" if x["with_tz"] else "")
            return _cast(text, "TIMESTAMP")
        if x["with_tz"]:
            return _cast(x["text"] + "+00:00", _TIMESTAMPTZ)
        return _cast(x["text"], _TIMESTAMP)
    if kind == "Time":
        return _cast(x, "TIME" if dialect in _NO_TEMPORAL_PRECISION else _TIME)
    msg = f"unknown SQL literal kind: {kind!r}"
    raise ValueError(msg)


def _expr(e: Any, d: str | None) -> Any:
    ((kind, x),) = e.items()
    if kind == "Column":
        return _column(x)
    if kind == "Literal":
        return _literal(x, d)
    if kind == "Binary":
        left, op, right = x
        node = _BINARY[op]
        lhs, rhs = _expr(left, d), _expr(right, d)
        if op in ("And", "Or"):
            lhs, rhs = exp.Paren(this=lhs), exp.Paren(this=rhs)
        return node(this=lhs, expression=rhs)
    if kind == "Not":
        return exp.Not(this=exp.Paren(this=_expr(x, d)))
    if kind == "IsNull":
        inner, negated = x
        is_null = exp.Is(this=_expr(inner, d), expression=exp.Null())
        return exp.Not(this=is_null) if negated else is_null
    if kind == "InList":
        inner, values, negated = x
        node = exp.In(
            this=_expr(inner, d), expressions=[_literal(v, d) for v in values]
        )
        return exp.Not(this=node) if negated else node
    if kind == "Between":
        inner, lo, hi = x
        return exp.Between(this=_expr(inner, d), low=_expr(lo, d), high=_expr(hi, d))
    if kind == "Like":
        pattern = _string(x["pattern"], d)
        like = exp.Like(this=_expr(x["expr"], d), expression=pattern)
        if d == "clickhouse" and x["escape"] == "\\":
            # no ESCAPE clause; backslash is its escape character already
            return like
        escape = (
            exp.Literal.string(x["escape"]) if d == "tsql" else _string(x["escape"], d)
        )
        return exp.Escape(this=like, expression=escape)
    if kind == "IsNotDistinctFrom":
        left, right, not_distinct = x
        node = exp.NullSafeEQ if not_distinct else exp.NullSafeNEQ
        return node(this=_expr(left, d), expression=_expr(right, d))
    if kind == "Raw":
        return exp.Paren(this=_raw(x))
    msg = f"unknown SQL expression kind: {kind!r}"
    raise ValueError(msg)


def _select_item(item: Any) -> Any:
    if item == "Star":
        return exp.Star()
    if item == "CountStar":
        return exp.Count(this=exp.Star())
    if item == "One":
        return exp.Literal.number(1)
    ((kind, x),) = item.items()
    if kind == "Column":
        return _column(x)
    if kind == "Min":
        return exp.Min(this=_column(x))
    if kind == "Max":
        return exp.Max(this=_column(x))
    if kind == "Expr":
        # dialect conversion SQL from Polars' rules, e.g. CAST("u" AS TEXT)
        return exp.Alias(this=_raw(x["sql"]), alias=_ident(x["alias"]))
    msg = f"unknown SQL select item: {item!r}"
    raise ValueError(msg)


def _from(f: Any) -> Any:
    ((kind, x),) = f.items()
    if kind == "Table":
        parts = [_ident(p) for p in x]
        name = parts[-1]
        db = parts[-2] if len(parts) >= 2 else None
        catalog = parts[-3] if len(parts) >= 3 else None
        return exp.Table(this=name, db=db, catalog=catalog)
    if kind == "Subquery":
        # a placeholder, replaced by the verbatim user query after generation
        return exp.Table(
            this=exp.to_identifier(_SUBQUERY_PLACEHOLDER),
            alias=exp.TableAlias(this=_ident("_pl")),
        )
    msg = f"unknown SQL FROM kind: {kind!r}"
    raise ValueError(msg)


_SUBQUERY_PLACEHOLDER = f"__polars_subquery_{secrets.token_hex(8)}__"


def _ordered(o: Any, d: str | None) -> list[Any]:
    # NULLs where the dialect puts them anyway, so no emulation is generated
    null_ordering = sqlglot.Dialect.get_or_raise(d).NULL_ORDERING
    desc = bool(o["descending"])
    nulls_first = (null_ordering == "nulls_are_small" and not desc) or (
        null_ordering == "nulls_are_large" and desc
    )
    this = _expr(o["expr"], d)
    if d is None:
        # an unknown database (FlightSQL, ODBC): its NULL order is unknown, and it may
        # not have `NULLS FIRST`. NULLs first on ASC, as range partitions expect
        # (`nulls_sort_first` in `partition.rs`)
        is_null = exp.Case(
            ifs=[
                exp.If(
                    this=exp.Is(this=this.copy(), expression=exp.Null()),
                    true=exp.Literal.number(0),
                )
            ],
            default=exp.Literal.number(1),
        )
        return [
            exp.Ordered(this=is_null, desc=desc, nulls_first=nulls_first),
            exp.Ordered(this=this, desc=desc, nulls_first=nulls_first),
        ]
    return [exp.Ordered(this=this, desc=desc, nulls_first=nulls_first)]


def build(query: dict[str, Any], dialect: str | None = None) -> Any:
    """Build the sqlglot expression of a `SqlQuery`."""
    d = dialect
    select = exp.Select(expressions=[_select_item(c) for c in query["columns"]])
    select = select.from_(_from(query["from"]), copy=False)
    if filters := query["filter"]:
        select = select.where(*(_expr(f, d) for f in filters), copy=False)
    if order_by := query["order_by"]:
        select = select.order_by(
            *(e for o in order_by for e in _ordered(o, d)), copy=False
        )
    if (limit := query["limit"]) is not None:
        select = select.limit(int(limit), copy=False)
    return select


def render(query_json: str, dialect: str | None) -> str:
    """Render a `SqlQuery` (as JSON) for a dialect (`None` is sqlglot's default)."""
    query = json.loads(query_json)
    sql = str(build(query, dialect).sql(dialect=dialect))
    from_ = query["from"]
    if "Subquery" in from_:
        sub = from_["Subquery"]
        # a newline ends a trailing `--` comment of the user's query
        end = "\n" if "--" in sub else ""
        sql = sql.replace(_SUBQUERY_PLACEHOLDER, f"({sub}{end})", 1)
    return sql


_WRITES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,
)


def _is_read_only(stmt: Any) -> bool:
    """A query that only reads: no DML or DDL (in CTEs either), no SELECT INTO."""
    if not isinstance(stmt, (exp.Query, exp.Values)):
        return False
    if stmt.find(*_WRITES) is not None:
        return False
    return not any(s.args.get("into") for s in stmt.find_all(exp.Select))


# Functions that return other values each time they run (or each time a query runs), by
# lower-cased name; sqlglot parses most of them as `Anonymous`. `random()` / `rand()` /
# `DBMS_RANDOM.VALUE` and the `uuid()` family parse as typed nodes (`_VOLATILE_NODES`).
_VOLATILE_FUNCTIONS = frozenset(
    {
        # Postgres (and DuckDB)
        "random",
        "random_normal",
        "setseed",
        "gen_random_uuid",
        "gen_random_bytes",
        "uuid_generate_v1",
        "uuid_generate_v1mc",
        "uuid_generate_v4",
        "uuidv4",
        "uuidv7",
        "nextval",
        # the time of the call or statement; `now()` / `CURRENT_TIMESTAMP` are the
        # transaction's (and SQL Server's `SYSDATETIME()` its statement's), so not listed
        "clock_timestamp",
        "statement_timestamp",
        "timeofday",
        # SQLite
        "randomblob",
        # SQL Server
        "newid",
        "newsequentialid",
        "crypt_gen_random",
        # Oracle (`DBMS_RANDOM.*` and `seq.NEXTVAL` are checked apart)
        "sys_guid",
        # MySQL / MariaDB
        "rand",
        "uuid",
        "uuid_short",
        "uuid_v4",
        "uuid_v7",
        "random_bytes",
        "sysdate",
        # ClickHouse (and its `rand*` family, see `_is_volatile`)
        "generateuuidv4",
        "generateuuidv7",
        "generateulid",
        "generateserialid",
        "generatesnowflakeid",
        "fuzzbits",
        # Snowflake
        "uuid_string",
        "seq1",
        "seq2",
        "seq4",
        "seq8",
        "randstr",
    }
)
_VOLATILE_NODES = tuple(
    node
    for node in (
        getattr(exp, name, None)
        for name in ("Rand", "Randn", "Uuid", "Seq1", "Seq2", "Seq4", "Seq8")
    )
    if node is not None
)


def _is_volatile(node: Any, dialect: str | None, relations: set[str]) -> bool:
    if isinstance(node, _VOLATILE_NODES):
        return True
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        return name in _VOLATILE_FUNCTIONS or (
            dialect == "clickhouse" and name.startswith("rand")
        )
    if isinstance(node, exp.Dot):
        # Oracle `DBMS_RANDOM.RANDOM()`, `DBMS_CRYPTO.RANDOMBYTES(…)`
        return node.this.name.lower() in ("dbms_random", "dbms_crypto")
    if isinstance(node, exp.Column) and node.table:
        # `seq.NEXTVAL` (Oracle, Snowflake), `DBMS_RANDOM.VALUE` without parentheses; a
        # qualifier that names a table or alias of the query is a column `nextval`
        qualifier = node.table.lower()
        if qualifier in relations:
            return False
        return node.name.lower() == "nextval" or qualifier in (
            "dbms_random",
            "dbms_crypto",
        )
    return False


def _relations(stmt: Any) -> set[str]:
    """Lower-cased names and aliases of the tables, subqueries and CTEs of a query."""
    names: set[str] = set()
    for node in stmt.find_all(exp.Table, exp.Subquery, exp.CTE):
        names.add(node.alias_or_name.lower())
        if isinstance(node, exp.Table):
            names.add(node.name.lower())
    names.discard("")
    return names


def inspect_query(sql: str, dialect: str | None) -> dict[str, Any]:
    """Facts about a user query that decide whether it can be wrapped."""
    facts: dict[str, Any] = {
        "parses": False,
        "single_select": False,
        "has_cte": False,
        "top_level_order_by_without_limit": False,
        "output_columns": None,
        "read_only": False,
        "repeatable": False,
    }
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.SqlglotError:
        return facts
    facts["parses"] = True
    facts["read_only"] = len(statements) == 1 and _is_read_only(statements[0])
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return facts

    stmt = statements[0]
    facts["single_select"] = True
    facts["has_cte"] = stmt.args.get("with_") is not None or (
        stmt.args.get("with") is not None
    )
    has_limit = any(stmt.args.get(k) is not None for k in ("limit", "fetch", "offset"))
    has_top = isinstance(stmt, exp.Select) and stmt.args.get("limit") is not None
    facts["top_level_order_by_without_limit"] = (
        stmt.args.get("order") is not None and not has_limit and not has_top
    )
    relations = _relations(stmt)
    # anywhere in the query: a nested LIMIT without a unique ORDER BY picks rows too
    facts["repeatable"] = stmt.find(
        exp.Limit, exp.Fetch, exp.Offset, exp.TableSample
    ) is None and not any(
        _is_volatile(node, dialect, relations)
        for node in stmt.find_all(exp.Func, exp.Dot, exp.Column)
    )
    selects = stmt.selects
    if selects and not any(isinstance(s, exp.Star) or s.is_star for s in selects):
        facts["output_columns"] = [s.alias_or_name for s in selects]
    return facts
