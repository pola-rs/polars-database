//! Planning-time work: resolving a scan (connection, schema, catalog) and compiling its SQL.
//!
//! Both run in the resolver (`crate::resolver`), which Polars calls on the planning machine: the
//! client for local runs, the scheduler for distributed runs. Workers only receive the finished
//! SQL.

use std::sync::Arc;

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

use super::backend::resolve::{detect_dialect, resolve_backend};
use super::backend::{BackendKind, DatabaseSession, open_backend, open_session_with_retries};
use super::catalog::{self, CatalogInfo};
use super::config::{ColumnMeta, CompiledPartition, CompiledScan, DatabaseScanConfig, TableStats};
use super::dialect::DialectName;
use super::options::{Consistency, DatabaseRelation, DatabaseScanOptions, PartitionSpec};
use super::partition::{self, Partition};
use super::schema::{declared_dtype, normalize_dtype};
use super::sql::{
    QueryFacts, SqlBinOp, SqlExpr, SqlFrom, SqlLiteral, SqlOrderBy, SqlQuery, SqlRenderer,
    SqlSelectItem, sql_renderer,
};

/// Below this estimated size, `partition_on="auto"` reads a table with one query.
const MIN_PARTITIONED_BYTES: u64 = 128 * 1024 * 1024;
/// Target bytes per partition for `partition_on="auto"`.
const BYTES_PER_PARTITION: u64 = 256 * 1024 * 1024;
/// Rows sampled to type columns a row-wise driver could not type from an empty result.
const TYPE_SAMPLE_ROWS: u64 = 1000;

fn verbose() -> bool {
    polars_core::config::verbose()
}

/// The `FROM` target of a relation, for the fixed discovery and catalog SQL.
fn quoted_relation(dialect: DialectName, relation: &DatabaseRelation) -> String {
    match relation {
        DatabaseRelation::Table { parts, .. } => parts
            .iter()
            .map(|p| dialect.quote_ident(p))
            .collect::<Vec<_>>()
            .join("."),
        DatabaseRelation::Query { sql } => format!("({}) pl_q", wrappable(sql)),
    }
}

fn strip_statement_end(sql: &str) -> &str {
    sql.trim().trim_end_matches(';').trim_end()
}

/// The query as it goes inside `(...)`: a newline ends a trailing `--` comment.
fn wrappable(sql: &str) -> String {
    let sql = strip_statement_end(sql);
    if sql.contains("--") {
        format!("{sql}\n")
    } else {
        sql.to_string()
    }
}

/// Statement keywords that write or change state. Used when the query can't be parsed.
const WRITE_KEYWORDS: &[&str] = &[
    "ALTER", "ANALYZE", "CALL", "COPY", "CREATE", "DELETE", "DO", "DROP", "EXEC", "EXECUTE",
    "GRANT", "INSERT", "LOCK", "MERGE", "REPLACE", "REVOKE", "TRUNCATE", "UPDATE", "UPSERT", "USE",
    "VACUUM",
];

/// `scan_database` only reads. A query source that writes is rejected, whichever driver runs
/// it; sessions are read-only too where the dialect allows it.
fn check_read_only(sql: &str, facts: Option<&QueryFacts>) -> PolarsResult<()> {
    if let Some(facts) = facts.filter(|f| f.parses) {
        polars_ensure!(
            facts.read_only,
            InvalidOperation:
            "scan_database only runs read queries: the source must be one SELECT statement \
            that doesn't write (no INSERT/UPDATE/DELETE/DDL, also not in a CTE, no SELECT INTO)"
        );
        return Ok(());
    }
    let first = first_keyword(sql).to_ascii_uppercase();
    polars_ensure!(
        !WRITE_KEYWORDS.contains(&first.as_str()),
        InvalidOperation: "{first} statements are not valid read queries for scan_database"
    );
    Ok(())
}

/// The first word of a statement, after comments.
fn first_keyword(sql: &str) -> &str {
    let mut s = sql.trim_start();
    loop {
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.split_once('\n').map_or("", |(_, r)| r).trim_start();
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.split_once("*/").map_or("", |(_, r)| r).trim_start();
        } else if let Some(rest) = s.strip_prefix('(') {
            s = rest.trim_start();
        } else {
            break;
        }
    }
    let end = s
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    &s[..end]
}

/// Why a user query can't be wrapped as a derived table, if it can't.
fn wrap_blocker(facts: &QueryFacts, dialect: DialectName) -> Option<&'static str> {
    if !facts.parses {
        return Some("the query does not parse in the dialect");
    }
    if !facts.single_select {
        return Some("the query is not a single SELECT statement");
    }
    if dialect == DialectName::Mssql && facts.has_cte {
        return Some("SQL Server does not allow a CTE inside a derived table");
    }
    if dialect == DialectName::Mssql && facts.top_level_order_by_without_limit {
        return Some("SQL Server does not allow ORDER BY without TOP inside a derived table");
    }
    if let Some(cols) = &facts.output_columns {
        let mut seen = PlHashSet::with_capacity(cols.len());
        if !cols.iter().all(|c| seen.insert(c)) {
            return Some("the query has duplicate output column names");
        }
    }
    None
}

/// Resolves a scan: dialect, backend, schema and catalog metadata. Runs once per resolver, when
/// Polars first needs the schema. Partitions and SQL are left to [`compile_scan`].
pub fn resolve_scan(
    options: Arc<DatabaseScanOptions>,
    user_schema: Option<SchemaRef>,
) -> PolarsResult<DatabaseScanConfig> {
    let dialect = detect_dialect(&options.connection, options.dialect)?;
    // `orders` names `ORDERS` on Oracle, as it would in SQL. A table that exists only as
    // written (created quoted, `"Orders"`) is still found: see the schema discovery below.
    // A user schema means no I/O, so nothing could check the folded name: the name is used as
    // written.
    let written = options.clone();
    let mut options = if user_schema.is_none() {
        options.with_relation_case_folded(dialect)
    } else {
        options
    };
    let as_written = (!Arc::ptr_eq(&written, &options)).then_some(written);
    let backend = resolve_backend(&options.connection, dialect, options.driver.as_deref())?;
    if options.connection.has_query_params() {
        polars_ensure!(
            matches!(options.relation, DatabaseRelation::Query { .. }),
            InvalidOperation: "params can only be used with a query source"
        );
        polars_ensure!(
            !matches!(backend.kind, BackendKind::Adbc { .. } | BackendKind::ConnectorX),
            InvalidOperation:
            "params need a driver that binds parameters itself, which {} doesn't do here; \
            pin a Python driver with driver=..., e.g. driver=\"psycopg\"",
            backend.kind
        );
    }
    if verbose() {
        eprintln!(
            "[scan_database]: dialect={dialect}, backend={backend} ({})",
            backend.reason
        );
    }

    // A query that can't be wrapped runs as given, without pushdown.
    let mut pushdown_disabled: Option<PlSmallStr> =
        (!options.pushdown).then(|| "pushdown=False".into());
    if let DatabaseRelation::Query { sql } = &options.relation {
        let facts = sql_renderer().and_then(|r| r.inspect_query(sql, dialect));
        check_read_only(sql, facts.as_ref().ok())?;
        // Each partition runs the query again: it must pick the same rows every time.
        if let Ok(facts) = &facts
            && facts.parses
            && !facts.repeatable
            && matches!(
                options.partitioning,
                PartitionSpec::Column { .. } | PartitionSpec::Predicates(_)
            )
        {
            polars_bail!(
                InvalidOperation:
                "this query can't be partitioned: each partition runs it again, and with \
                LIMIT, OFFSET, TABLESAMPLE or a random function it may return other rows \
                each time; pass partition_on=None, or partition a table"
            );
        }
        let blocker = match &facts {
            _ if pushdown_disabled.is_some() => None,
            Ok(facts) => wrap_blocker(facts, dialect).map(PlSmallStr::from),
            Err(e) => Some(format!("the query could not be inspected: {e}").into()),
        };
        if let Some(reason) = blocker {
            if verbose() {
                eprintln!("[scan_database]: pushdown disabled: {reason}");
            }
            pushdown_disabled = Some(reason);
        }
    }

    let mut columns: Vec<ColumnMeta>;
    let mut stats = TableStats::default();
    let schema: SchemaRef;

    if let Some(schema_) = user_schema {
        // No I/O: a user schema is authoritative, and column semantics stay conservative:
        // collations are unknown, and so are some types behind the dtypes: SQL Server's
        // legacy `datetime`, a MySQL `YEAR` (see the filter), and a SQLite column's
        // affinity (a `String` may hold numbers that compare as such). Otherwise the
        // user's `String` states a text column.
        columns = schema_
            .iter()
            .map(|(name, dtype)| ColumnMeta {
                name: name.clone(),
                dtype: dtype.clone(),
                db_type: None,
                strings: catalog::unknown_string_semantics(dialect),
                is_key: false,
                comparable: match dtype {
                    DataType::String => dialect != DialectName::Sqlite,
                    DataType::Datetime(_, _) => dialect != DialectName::Mssql,
                    _ => true,
                },
            })
            .collect();
        schema = schema_;
    } else {
        let backend_impl = open_backend(&backend, options.fetch.batch_size)?;
        let mut session =
            open_session_with_retries(backend_impl.as_ref(), &options.connection, dialect)?;
        // The catalog first: some declared types are read as text (`text_conversion`), and
        // discovery must select them the same way the scan will.
        let discover = |session: &mut dyn DatabaseSession, options: &DatabaseScanOptions| {
            let info = match &options.relation {
                DatabaseRelation::Table { parts, .. } => {
                    catalog::table_info(session, dialect, parts)
                },
                DatabaseRelation::Query { .. } => CatalogInfo::default(),
            };
            let select_list = converted_select_list(dialect, &backend.kind, &info);
            let discovered = discover_schema(
                session,
                options,
                dialect,
                pushdown_disabled.is_some(),
                select_list.as_deref(),
                &info,
            )?;
            PolarsResult::Ok((info, discovered))
        };
        let (mut info, mut discovered) = match discover(session.as_mut(), &options) {
            Ok(found) => found,
            // No table by the folded name: the name as written, on a new session (a failed
            // statement can abort the session's transaction).
            Err(e) => match as_written {
                Some(written) => {
                    session = open_session_with_retries(
                        backend_impl.as_ref(),
                        &written.connection,
                        dialect,
                    )?;
                    let found = discover(session.as_mut(), &written).map_err(|_| e)?;
                    options = written;
                    found
                },
                None => return Err(e),
            },
        };
        stats = info.stats;
        // A query's result types decide how the database compares its columns. Without
        // pushdown nothing is compared.
        if matches!(options.relation, DatabaseRelation::Query { .. }) && pushdown_disabled.is_none()
        {
            let names = discovered.iter_names().cloned().collect::<Vec<_>>();
            let from = quoted_relation(dialect, &options.relation);
            // Python drivers refuse DDL, ConnectorX runs each statement on its own connection.
            let temp_tables = matches!(backend.kind, BackendKind::Adbc { .. });
            info = catalog::query_info(session.as_mut(), dialect, &from, &names, temp_tables);
        }

        // A declared type fixes the dtype, whichever driver read it.
        for (name, dtype) in discovered.iter_mut() {
            let Some(t) = info.declared_type(name) else {
                continue;
            };
            if dialect.text_conversion(t, name).is_some() {
                // read as text by the dialect rules; the value stays text
                *dtype = DataType::String;
            } else if let Some(declared) = declared_dtype(dialect, t) {
                *dtype = declared;
            }
        }
        columns = discovered
            .iter()
            .map(|(name, dtype)| ColumnMeta {
                name: name.clone(),
                dtype: dtype.clone(),
                db_type: info.declared_type(name).map(Into::into).or_else(|| {
                    // A query's result type, for the filter's checks (MySQL `YEAR`), where
                    // the column is read as it is.
                    info.compare_types
                        .get(name.as_str())
                        .filter(|t| {
                            info.types_required
                                && column_conversion(
                                    dialect,
                                    &backend.kind,
                                    t,
                                    &dialect.quote_ident(name),
                                )
                                .is_none()
                        })
                        .map(|t| t.as_str().into())
                }),
                strings: info.string_semantics(dialect, name),
                is_key: info.key_column.as_ref() == Some(name),
                comparable: !info.incomplete
                    && info.compares_like_polars(dialect, name, dtype)
                    && info.declared_type(name).is_none_or(|t| {
                        dialect.text_conversion(t, name).is_none()
                            && !(alias_shadows_column(dialect)
                                && column_conversion(
                                    dialect,
                                    &backend.kind,
                                    t,
                                    &dialect.quote_ident(name),
                                )
                                .is_some())
                    }),
            })
            .collect();
        if let DatabaseRelation::Query { sql } = &options.relation
            && dialect == DialectName::Oracle
        {
            // With pushdown=False the query is still wrapped to read its time zone
            // columns as UTC, so the values don't depend on pushdown: that takes its types.
            if !options.pushdown
                && sql_renderer()
                    .and_then(|r| r.inspect_query(sql, dialect))
                    .is_ok_and(|f| wrap_blocker(&f, dialect).is_none())
            {
                let names = discovered.iter_names().cloned().collect::<Vec<_>>();
                let from = quoted_relation(dialect, &options.relation);
                info = catalog::query_info(session.as_mut(), dialect, &from, &names, false);
            }
            oracle_query_timestamps(&info, &mut columns, &mut discovered);
        }
        schema = Arc::new(discovered);
    }

    // ConnectorX writes every decimal as `Decimal(38, 10)`, cutting digits beyond 10
    // places. A table's declared types say which columns to read as text; a query's
    // decimal columns are read as text too, which needs the query wrapped.
    if matches!(backend.kind, BackendKind::ConnectorX)
        && matches!(options.relation, DatabaseRelation::Query { .. })
    {
        for c in columns.iter_mut().filter(|c| c.dtype.is_decimal()) {
            polars_ensure!(
                pushdown_disabled.is_none(),
                InvalidOperation:
                "ConnectorX cuts decimals to 10 decimal places, and this query can't be \
                wrapped to read column '{}' exactly ({}); pin another driver, or CAST the \
                column to text in the query",
                c.name, pushdown_disabled.as_deref().unwrap_or("")
            );
            c.db_type = Some(CONNECTORX_DECIMAL.into());
            c.comparable &= !alias_shadows_column(dialect);
        }
    }

    let schema = match &options.schema_overrides {
        Some(overrides) if !overrides.is_empty() => {
            let mut schema = (*schema).clone();
            for (name, dtype) in overrides.iter() {
                // Overrides may only retype columns; they don't add any.
                if let Some(existing) = schema.get_mut(name) {
                    let retyped = existing != dtype;
                    *existing = dtype.clone();
                    if let Some(c) = columns.iter_mut().find(|c| c.name == name) {
                        c.dtype = dtype.clone();
                        // The database still compares the stored values, not the cast ones.
                        c.comparable &= !retyped;
                    }
                } else {
                    polars_bail!(
                        ColumnNotFound:
                        "schema_overrides names column '{name}', which the database source doesn't have"
                    );
                }
            }
            Arc::new(schema)
        },
        _ => schema,
    };

    Ok(DatabaseScanConfig {
        fingerprint: options.fingerprint(),
        options,
        dialect,
        backend,
        schema,
        columns: columns.into(),
        stats,
        compiled: None,
        connection_share: None,
        pushdown_disabled,
        bounds_cache: Default::default(),
    })
}

/// An Oracle query's `TIMESTAMP WITH [LOCAL] TIME ZONE` columns, by its result types
/// (`CatalogInfo::compare_types`): read as UTC through their conversion, like a table's.
/// Drivers read them as naive wall-clock times otherwise, so a temporal column whose type
/// is unknown is not compared in the database. The types are unknown when the query can't
/// be wrapped (then it can't be converted either) or the lookup fails: such columns are
/// read as the driver returns them (naive, the offset dropped).
fn oracle_query_timestamps(info: &CatalogInfo, columns: &mut [ColumnMeta], schema: &mut Schema) {
    for c in columns.iter_mut() {
        let t = info
            .compare_types
            .get(&c.name)
            .map(|t| t.to_ascii_lowercase());
        match t {
            Some(t) if t.starts_with("timestamp") && t.contains("time zone") => {
                let Some(dtype) = declared_dtype(DialectName::Oracle, &t) else {
                    continue;
                };
                c.db_type = Some(t.into());
                c.dtype = dtype.clone();
                // read as UTC like a table's column, so compared like one
                c.comparable = !info.incomplete;
                if let Some(d) = schema.get_mut(&c.name) {
                    *d = dtype;
                }
            },
            Some(_) => {},
            None if c.dtype.is_temporal() => c.comparable = false,
            None => {},
        }
    }
}

/// ClickHouse resolves a select-list alias in `WHERE`: a filter on a column read through
/// a conversion (`toString(x) AS x`) would compare the converted value.
fn alias_shadows_column(dialect: DialectName) -> bool {
    dialect == DialectName::Clickhouse
}

/// The `db_type` of a query's decimal column read through ConnectorX: selected as text.
const CONNECTORX_DECIMAL: &str = "<connectorx decimal>";

/// SQL that converts a column of a declared type where the backend can't read it as is, or
/// `None` to select it as it is.
fn column_conversion(
    dialect: DialectName,
    backend: &BackendKind,
    db_type: &str,
    column: &str,
) -> Option<String> {
    if db_type == CONNECTORX_DECIMAL {
        return Some(dialect.cast_to_text(column));
    }
    if let Some(sql) = dialect.text_conversion(db_type, column) {
        return Some(sql);
    }
    // ClickHouse `DateTime` (seconds) reaches Arrow as a plain UInt32; `DateTime64` is a
    // timestamp, and compares like the column.
    if dialect == DialectName::Clickhouse {
        let t = super::schema::clickhouse_value_type(db_type);
        if t == "datetime" || t.starts_with("datetime(") {
            return Some(format!("toDateTime64({column}, 0)"));
        }
    }
    // Read as text and parse the exact value; the planned dtype stays the declared decimal.
    // ConnectorX reads decimals with at most 10 decimal places, dropping digits; an
    // unconstrained `numeric` / `NUMBER` reaches some drivers as a float (oracledb), and
    // its values may have more digits than the planned scale (an error then, not rounding).
    if let Some(DataType::Decimal(p, scale)) = declared_dtype(dialect, db_type)
        && ((matches!(backend, BackendKind::ConnectorX) && scale > 10)
            || DataType::Decimal(p, scale) == super::schema::UNCONSTRAINED_DECIMAL)
    {
        if dialect == DialectName::Oracle {
            // the session (a live connection's) may write a decimal comma
            return Some(format!(
                "TO_CHAR({column}, 'TM9', 'NLS_NUMERIC_CHARACTERS=''.,''')"
            ));
        }
        return Some(dialect.cast_to_text(column));
    }
    // Oracle returns `TIMESTAMP WITH LOCAL TIME ZONE` in the session's time zone (a live
    // connection's is the user's), and drivers drop the offset of `TIMESTAMP WITH TIME
    // ZONE`, keeping the wall-clock time: read both as UTC, which the planned dtype says
    // they are.
    if dialect == DialectName::Oracle {
        let t = db_type.trim().to_ascii_lowercase();
        if t.starts_with("timestamp") && t.contains("local time zone") {
            return Some(format!(
                "SYS_EXTRACT_UTC(CAST({column} AS TIMESTAMP WITH TIME ZONE))"
            ));
        }
        if t.starts_with("timestamp") && t.contains("time zone") {
            return Some(format!("SYS_EXTRACT_UTC({column})"));
        }
    }
    // The ADBC MySQL driver sizes an `UNSIGNED` decimal one digit short, failing on values
    // with every digit used.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && matches!(backend, BackendKind::Adbc { .. })
        && db_type.to_ascii_lowercase().contains("unsigned")
        && declared_dtype(dialect, db_type).is_some_and(|d| d.is_decimal())
    {
        return Some(dialect.cast_to_text(column));
    }
    // MySQL zero dates (`'0000-00-00'`, and dates with a zero month or day) are no date:
    // read them as NULL, as Connector/J's `zeroDateTimeBehavior=convertToNull` does. Drivers
    // fail on them otherwise. Pushed filters see them as NULL too (`zero_dates_as_null`).
    if mysql_zero_date_type(dialect, db_type) {
        return Some(format!("IF({}, NULL, {column})", mysql_zero_date(column)));
    }
    let _ = backend;
    None
}

/// A MySQL `DATE` / `DATETIME` / `TIMESTAMP`, which may hold zero dates.
fn mysql_zero_date_type(dialect: DialectName, db_type: &str) -> bool {
    let t = db_type.trim().to_ascii_lowercase();
    let base = t.split('(').next().unwrap_or(&t);
    matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && matches!(base, "date" | "datetime" | "timestamp")
}

/// Whether a MySQL date column (quoted) holds a zero date; NULL for NULL.
fn mysql_zero_date(column: &str) -> String {
    format!("MONTH({column}) = 0 OR DAYOFMONTH({column}) = 0")
}

/// A pushed filter on MySQL date columns, evaluated as if their zero dates were NULL, as
/// they are read: `(f AND NOT zero(c)) OR (zero(c) AND f[c := NULL])`. The database folds
/// the second branch to a constant (false, except for `IS NULL`), keeping index ranges on
/// `c` usable.
fn zero_dates_as_null(config: &DatabaseScanConfig, mut filter: SqlExpr) -> SqlExpr {
    for c in config.columns.iter() {
        if !references(&filter, &c.name) {
            continue;
        }
        let Some(zero) = zero_date_check(config, &c.name) else {
            continue;
        };
        let as_null = substitute_null(&filter, &c.name);
        filter = SqlExpr::binary(
            SqlExpr::binary(filter, SqlBinOp::And, SqlExpr::Not(Box::new(zero.clone()))),
            SqlBinOp::Or,
            SqlExpr::binary(zero, SqlBinOp::And, as_null),
        );
    }
    filter
}

/// `TRUE` where a MySQL date column holds a zero date (`FALSE` for NULL), if the column
/// may hold zero dates.
fn zero_date_check(config: &DatabaseScanConfig, column: &str) -> Option<SqlExpr> {
    let dialect = config.dialect;
    let t = config.column(column)?.db_type.as_deref()?;
    mysql_zero_date_type(dialect, t).then(|| {
        SqlExpr::Raw(format!(
            "IFNULL({}, FALSE)",
            mysql_zero_date(&dialect.quote_ident(column))
        ))
    })
}

/// An `ORDER BY` column. A MySQL date column is ordered by its values as read, zero
/// dates as NULL: a date with a zero month or day (`2020-00-01`) sorts among the real
/// dates otherwise. The conversion is written on the table's column, so it applies
/// whether or not the column is selected. (An index on the column no longer gives the
/// order then: the database sorts.)
fn order_by_expr(config: &DatabaseScanConfig, column: &PlSmallStr) -> SqlExpr {
    match &config.options.relation {
        relation @ DatabaseRelation::Table { .. } if zero_date_check(config, column).is_some() => {
            let dialect = config.dialect;
            let qualified = format!(
                "{}.{}",
                quoted_relation(dialect, relation),
                dialect.quote_ident(column)
            );
            SqlExpr::Raw(format!(
                "IF({}, NULL, {qualified})",
                mysql_zero_date(&qualified)
            ))
        },
        _ => SqlExpr::col(column.clone()),
    }
}

fn references(e: &SqlExpr, column: &str) -> bool {
    use SqlExpr as E;
    match e {
        E::Column(c) => c == column,
        E::Literal(_) | E::Raw(_) => false,
        E::Binary(l, _, r) | E::IsNotDistinctFrom(l, r, _) => {
            references(l, column) || references(r, column)
        },
        E::Not(x) | E::IsNull(x, _) | E::InList(x, _, _) => references(x, column),
        E::Between(x, lo, hi) => [x, lo, hi].iter().any(|x| references(x, column)),
        E::Like { expr, .. } => references(expr, column),
    }
}

/// `e` with `column` replaced by NULL.
fn substitute_null(e: &SqlExpr, column: &str) -> SqlExpr {
    use SqlExpr as E;
    let sub = |x: &SqlExpr| Box::new(substitute_null(x, column));
    match e {
        E::Column(c) if c == column => E::Literal(SqlLiteral::Null),
        E::Column(_) | E::Literal(_) | E::Raw(_) => e.clone(),
        E::Binary(l, op, r) => E::Binary(sub(l), *op, sub(r)),
        E::IsNotDistinctFrom(l, r, b) => E::IsNotDistinctFrom(sub(l), sub(r), *b),
        E::Not(x) => E::Not(sub(x)),
        E::IsNull(x, b) => E::IsNull(sub(x), *b),
        E::InList(x, list, b) => E::InList(sub(x), list.clone(), *b),
        E::Between(x, lo, hi) => E::Between(sub(x), sub(lo), sub(hi)),
        E::Like {
            expr,
            pattern,
            escape,
        } => E::Like {
            expr: sub(expr),
            pattern: pattern.clone(),
            escape: *escape,
        },
    }
}

/// The columns of a table as select items, with the conversions applied, if any column
/// needs one.
fn converted_select_list(
    dialect: DialectName,
    backend: &BackendKind,
    info: &CatalogInfo,
) -> Option<Vec<SqlSelectItem>> {
    let mut any = false;
    let items = info
        .types
        .iter()
        .map(
            |(name, t)| match column_conversion(dialect, backend, t, &dialect.quote_ident(name)) {
                Some(sql) => {
                    any = true;
                    SqlSelectItem::Expr {
                        sql,
                        alias: name.clone(),
                    }
                },
                None => SqlSelectItem::Column(name.clone()),
            },
        )
        .collect::<Vec<_>>();
    any.then_some(items)
}

/// Select items as fixed SQL, for the discovery queries that don't go through the renderer.
fn fixed_select_list(dialect: DialectName, items: &[SqlSelectItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            SqlSelectItem::Column(c) => dialect.quote_ident(c),
            SqlSelectItem::Expr { sql, alias } => {
                format!("{sql} AS {}", dialect.quote_ident(alias))
            },
            _ => "*".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The result schema of the relation, without reading rows.
fn discover_schema(
    session: &mut dyn DatabaseSession,
    options: &DatabaseScanOptions,
    dialect: DialectName,
    unwrapped: bool,
    select_list: Option<&[SqlSelectItem]>,
    info: &CatalogInfo,
) -> PolarsResult<Schema> {
    let sql = match (&options.relation, unwrapped) {
        // A query that can't be wrapped is run as is; only its first batch is read.
        (DatabaseRelation::Query { sql }, true) => strip_statement_end(sql).to_string(),
        (relation, _) => format!(
            "SELECT {} FROM {} WHERE 1 = 0",
            select_list.map_or_else(|| "*".to_string(), |s| fixed_select_list(dialect, s)),
            quoted_relation(dialect, relation)
        ),
    };
    if verbose() {
        eprintln!("[scan_database]: schema from: {sql}");
    }
    let first = {
        let mut batches = session.execute(&sql, Some(1))?;
        batches.next().transpose()?
    };
    let Some(first) = first else {
        polars_bail!(ComputeError: "database returned no result for the schema query: {sql}");
    };

    let mut schema: Schema = first
        .schema()
        .iter()
        .map(|(name, dtype)| (name.clone(), normalize_dtype(dtype)))
        .collect();

    // Row-wise drivers can't type columns of an empty result, and SQLite has no column
    // types at all (ADBC calls every column of an empty result Int64). Type them from a
    // sample.
    let sample_types = dialect == DialectName::Sqlite;
    let untyped = sample_types || schema.iter().any(|(_, dt)| dt.is_null());
    // An unwrapped query was typed from its first batch already.
    let ran_query = unwrapped && matches!(options.relation, DatabaseRelation::Query { .. });
    if untyped && !ran_query && dialect.supports_limit() {
        let renderer = sql_renderer()?;
        let items = select_list.map_or_else(|| vec![SqlSelectItem::Star], <[_]>::to_vec);
        let mut query = SqlQuery::select(items, from_clause(options));
        query.limit = Some(TYPE_SAMPLE_ROWS);
        let sample_sql = renderer.render(&query, dialect)?;
        if verbose() {
            eprintln!("[scan_database]: typing columns from a sample: {sample_sql}");
        }
        let sample = session.query(&sample_sql)?;
        for (name, dtype) in schema.iter_mut() {
            if (dtype.is_null() || (sample_types && sample.height() > 0))
                && let Ok(col) = sample.column(name)
            {
                *dtype = normalize_dtype(col.dtype());
            }
            // No values to type by (ADBC types an empty result's columns Int64): the
            // declared type's affinity tells what values the column holds.
            if sample_types
                && (sample.height() == 0 || dtype.is_null())
                && let Some(t) = info.compare_types.get(name)
            {
                *dtype = sqlite_affinity_dtype(t);
            }
        }
    }
    // Still untyped (every sampled value is NULL, or no rows to sample): a dtype any value
    // can be cast to, not `Null`, which no value read later would fit.
    for (_, dtype) in schema.iter_mut() {
        if dtype.is_null() {
            *dtype = DataType::String;
        }
    }
    Ok(schema)
}

/// The dtype a SQLite column of declared type `t` reads as, by its type affinity (the
/// SQLite rules). `NUMERIC` affinity (`DATE`, `DECIMAL`, `BOOLEAN`, …) holds integers,
/// floats or text alike: `String`, which they all cast to.
fn sqlite_affinity_dtype(t: &str) -> DataType {
    let t = t.to_ascii_uppercase();
    if t.contains("INT") {
        DataType::Int64
    } else if t.contains("CHAR") || t.contains("CLOB") || t.contains("TEXT") {
        DataType::String
    } else if t.contains("BLOB") {
        DataType::Binary
    } else if t.contains("REAL") || t.contains("FLOA") || t.contains("DOUB") {
        DataType::Float64
    } else {
        DataType::String
    }
}

fn from_clause(options: &DatabaseScanOptions) -> SqlFrom {
    match &options.relation {
        DatabaseRelation::Table { parts, .. } => SqlFrom::Table(parts.clone()),
        DatabaseRelation::Query { sql } => SqlFrom::Subquery(strip_statement_end(sql).into()),
    }
}

/// A pushed conjunct of the scan predicate.
#[derive(Clone, Debug)]
pub struct PushedFilter {
    pub expr: SqlExpr,
    /// The database returns exactly the rows Polars keeps. Otherwise a superset.
    pub exact: bool,
}

/// What Polars pushed into a scan (the resolver's projection, filters and limit).
#[derive(Clone, Debug, Default)]
pub struct ScanPushdown {
    /// Columns to read, in schema order. `None` reads every column.
    pub columns: Option<Vec<PlSmallStr>>,
    pub filters: Vec<PushedFilter>,
    /// Number of top-level conjuncts in the scan predicate, pushed or not.
    pub total_filters: usize,
    /// Rows needed from the start of the scan.
    pub limit: Option<u64>,
}

/// Resolves partitions and renders the SQL of every partition. Runs on every resolution, which
/// Polars does after predicate, projection and slice pushdown.
///
/// `target_partitions` is the distributed planner's target, which overrides automatic counts
/// (not a `partition_num` the user gave).
pub fn compile_scan(
    config: &DatabaseScanConfig,
    pushdown: ScanPushdown,
    target_partitions: Option<usize>,
) -> PolarsResult<CompiledScan> {
    let options = &config.options;
    let dialect = config.dialect;

    if let Some(reason) = &config.pushdown_disabled {
        return compile_unwrapped(config, reason.clone(), pushdown.total_filters);
    }

    let renderer = sql_renderer()?;
    let from = from_clause(options);

    let mut columns = pushdown
        .columns
        .clone()
        .unwrap_or_else(|| config.schema.iter_names().cloned().collect());
    for c in &options.order_by {
        polars_ensure!(
            config.schema.contains(c),
            ColumnNotFound: "order_by column '{c}' is not in the database source"
        );
    }
    columns.dedup();

    let (partitions, partitioning) = resolve_partitions(
        config,
        renderer.as_ref(),
        &from,
        &pushdown.filters,
        target_partitions,
    )?;
    if options.consistency == Consistency::Snapshot && partitions.len() > 1 {
        polars_bail!(
            InvalidOperation:
            "consistency='snapshot' is not supported for {dialect} with {} partitions yet; \
            pass partition_on=None to read with one consistent query",
            partitions.len()
        );
    }

    let select: Vec<SqlSelectItem> = if columns.is_empty() {
        vec![SqlSelectItem::One]
    } else {
        columns.iter().map(|c| select_item(config, c)).collect()
    };
    let order_by: Vec<SqlOrderBy> = options
        .order_by
        .iter()
        .map(|c| SqlOrderBy {
            expr: order_by_expr(config, c),
            descending: false,
        })
        .collect();
    let limit = pushdown.limit.filter(|_| dialect.supports_limit());

    let pushed: Vec<SqlExpr> = pushdown
        .filters
        .iter()
        .map(|f| zero_dates_as_null(config, f.expr.clone()))
        .collect();
    let mut compiled = Vec::with_capacity(partitions.len());
    for part in &partitions {
        let mut filter = Vec::with_capacity(pushdown.filters.len() + 1);
        // A zero date goes to the partition of the NULLs, as which it is read.
        filter.extend(
            part.predicate
                .clone()
                .map(|p| zero_dates_as_null(config, p)),
        );
        filter.extend(pushed.iter().cloned());

        let query = SqlQuery {
            columns: select.clone(),
            from: from.clone(),
            filter: filter.clone(),
            order_by: order_by.clone(),
            limit,
        };
        let count = SqlQuery {
            columns: vec![SqlSelectItem::CountStar],
            from: from.clone(),
            filter,
            order_by: vec![],
            limit: None,
        };
        compiled.push(CompiledPartition {
            sql: renderer.render(&query, dialect)?,
            count_sql: renderer.render(&count, dialect)?,
        });
    }

    let exact_filters = pushdown.filters.iter().filter(|f| f.exact).count();
    let out = CompiledScan {
        renderer: renderer.id(),
        all_exact: pushdown.total_filters == pushdown.filters.len()
            && exact_filters == pushdown.filters.len(),
        columns,
        partitioning,
        pushed_filters: pushdown.filters.len(),
        exact_filters,
        total_filters: pushdown.total_filters,
        limit,
        pushdown_disabled: None,
        repeatable: query_repeatable(config),
        partitions: compiled,
    };
    if verbose() {
        eprintln!(
            "[scan_database]: {} partition(s) ({}), sql[0]: {}",
            out.partitions.len(),
            out.partitioning,
            out.partitions[0].sql
        );
    }
    Ok(out)
}

/// A column as a select item, read through its conversion if it has one.
fn select_item(config: &DatabaseScanConfig, c: &PlSmallStr) -> SqlSelectItem {
    let dialect = config.dialect;
    let conversion = config
        .column(c)
        .and_then(|m| m.db_type.as_deref())
        .and_then(|t| column_conversion(dialect, &config.backend.kind, t, &dialect.quote_ident(c)));
    match conversion {
        Some(sql) => SqlSelectItem::Expr {
            sql,
            alias: c.clone(),
        },
        None => SqlSelectItem::Column(c.clone()),
    }
}

/// A scan without pushdown: the relation as given, one partition, everything else in Polars.
fn compile_unwrapped(
    config: &DatabaseScanConfig,
    reason: PlSmallStr,
    total_filters: usize,
) -> PolarsResult<CompiledScan> {
    let dialect = config.dialect;
    let order_by = &config.options.order_by;
    let (sql, count_sql) = match &config.options.relation {
        DatabaseRelation::Query { sql } => {
            polars_ensure!(
                order_by.is_empty(),
                InvalidOperation:
                "order_by needs pushdown, which is off for this scan ({reason}); \
                put the ORDER BY in the query instead"
            );
            let sql = strip_statement_end(sql);
            // Columns read through a conversion (Oracle time zones) need the query wrapped.
            let items = config
                .schema
                .iter_names()
                .map(|c| select_item(config, c))
                .collect::<Vec<_>>();
            let read = if items
                .iter()
                .any(|i| matches!(i, SqlSelectItem::Expr { .. }))
            {
                format!(
                    "SELECT {} FROM {}",
                    fixed_select_list(dialect, &items),
                    quoted_relation(dialect, &config.options.relation)
                )
            } else {
                sql.to_string()
            };
            (
                read,
                format!("SELECT COUNT(*) FROM ({}) pl_q", wrappable(sql)),
            )
        },
        relation => {
            let from = quoted_relation(dialect, relation);
            // the select list applies the column conversions, as a pushed-down scan does
            let items = config
                .schema
                .iter_names()
                .map(|c| select_item(config, c))
                .collect::<Vec<_>>();
            let list = if items
                .iter()
                .any(|i| matches!(i, SqlSelectItem::Expr { .. }))
            {
                fixed_select_list(dialect, &items)
            } else {
                "*".to_string()
            };
            let mut sql = format!("SELECT {list} FROM {from}");
            if !order_by.is_empty() {
                for c in order_by {
                    polars_ensure!(
                        config.schema.contains(c),
                        ColumnNotFound: "order_by column '{c}' is not in the database source"
                    );
                }
                let cols = order_by
                    .iter()
                    .map(|c| dialect.quote_ident(c))
                    .collect::<Vec<_>>()
                    .join(", ");
                sql.push_str(&format!(" ORDER BY {cols}"));
            }
            (sql, format!("SELECT COUNT(*) FROM {from}"))
        },
    };
    Ok(CompiledScan {
        renderer: "none".into(),
        partitions: vec![CompiledPartition { sql, count_sql }],
        all_exact: total_filters == 0,
        columns: config.schema.iter_names().cloned().collect(),
        partitioning: "1".into(),
        pushed_filters: 0,
        exact_filters: 0,
        total_filters,
        limit: None,
        pushdown_disabled: Some(reason),
        repeatable: query_repeatable(config),
    })
}

/// Whether the source returns the same rows each time it runs. A query that can't be
/// inspected is taken to be repeatable, as when it is partitioned.
fn query_repeatable(config: &DatabaseScanConfig) -> bool {
    match &config.options.relation {
        DatabaseRelation::Table { .. } => true,
        DatabaseRelation::Query { sql } => sql_renderer()
            .and_then(|r| r.inspect_query(sql, config.dialect))
            .map_or(true, |f| !f.parses || f.repeatable),
    }
}

/// Turns the scan's `PartitionSpec` into partitions, querying bounds where needed.
fn resolve_partitions(
    config: &DatabaseScanConfig,
    renderer: &dyn SqlRenderer,
    from: &SqlFrom,
    filters: &[PushedFilter],
    target_partitions: Option<usize>,
) -> PolarsResult<(Vec<Partition>, PlSmallStr)> {
    let options = &config.options;
    let one = || Ok((vec![Partition { predicate: None }], PlSmallStr::from("1")));
    let max_connections = config.max_connections();

    // A live connection reads partitions one after another on one connection; splitting
    // only adds queries.
    if config.options.connection.is_live() && target_partitions.is_none() {
        if let PartitionSpec::Auto { .. } = options.partitioning {
            return one();
        }
    }

    let (column, n, range, why) = match &options.partitioning {
        PartitionSpec::None => return one(),
        PartitionSpec::Predicates(preds) => {
            let parts = partition::predicate_partitions(preds)?;
            polars_ensure!(
                options.order_by.is_empty() || parts.len() <= 1,
                InvalidOperation:
                "order_by doesn't give a global order across {} partition_predicates; \
                partition on the first order_by column instead",
                parts.len()
            );
            let desc = format!("{} (partition_predicates)", parts.len());
            return Ok((parts, desc.into()));
        },
        spec @ PartitionSpec::Column {
            column,
            num_partitions,
            range,
            ..
        } => {
            polars_ensure!(
                options.order_by.is_empty() || options.order_by[0] == *column,
                InvalidOperation:
                "order_by={:?} needs partitioning on '{}', not on '{column}'",
                options.order_by, options.order_by[0]
            );
            // The user's `partition_num` (e.g. a cap on the database's load) wins over the
            // distributed target.
            let n = num_partitions
                .or(target_partitions)
                .unwrap_or_else(|| partition::target_count(spec, max_connections));
            (column.clone(), n, range.clone(), "partition_on")
        },
        PartitionSpec::Auto { num_partitions } => {
            // A snapshot is one query until partitions can share one: never split
            // automatically, so it doesn't start failing once the table grows.
            if matches!(options.relation, DatabaseRelation::Query { .. })
                || options.consistency == Consistency::Snapshot
            {
                return one();
            }
            // With order_by, a range on its first column also gives a global order.
            let column = match options.order_by.first() {
                Some(c) => Some(c.clone()),
                None => config
                    .columns
                    .iter()
                    .find(|c| c.is_key)
                    .map(|c| c.name.clone()),
            };
            let Some(column) = column else {
                if verbose() {
                    eprintln!(
                        "[scan_database]: one partition: no single-column key; \
                        pass partition_on= to read in parallel"
                    );
                }
                return one();
            };
            let n = match (*num_partitions, target_partitions, config.stats.bytes) {
                (Some(n), _, _) | (None, Some(n), _) => n,
                (None, None, Some(bytes)) if bytes < MIN_PARTITIONED_BYTES => return one(),
                (None, None, Some(bytes)) => {
                    (bytes.div_ceil(BYTES_PER_PARTITION) as usize).clamp(1, 4 * max_connections)
                },
                (None, None, None) => max_connections,
            };
            (column, n, None, "auto")
        },
    };

    // The column and the range are checked whatever the partition count, so a mistake
    // doesn't depend on the machine or the table size.
    let Some(dtype) = config.schema.get(&column) else {
        polars_bail!(ColumnNotFound: "partition column '{column}' is not in the database source");
    };
    let Some(kind) = partition::range_kind(dtype) else {
        if why == "auto" {
            return one();
        }
        polars_bail!(
            InvalidOperation:
            "partition column '{column}' has dtype {dtype}; range partitioning needs an integer, decimal, \
            float or date column (use partition_predicates for other types)"
        );
    };

    let range = match range {
        Some(range) => {
            let (lo, hi) = (
                partition::scalar_value(&range.0, kind)?,
                partition::scalar_value(&range.1, kind)?,
            );
            polars_ensure!(
                lo <= hi,
                InvalidOperation: "partition_range must be (low, high), got ({}, {})",
                range.0.value(), range.1.value()
            );
            Some((lo, hi))
        },
        None => None,
    };
    if n <= 1 {
        return one();
    }

    let (lo, hi, bounds_from) = match range {
        Some((lo, hi)) => (lo, hi, "partition_range"),
        None => {
            let (lo, hi) = query_bounds(config, renderer, from, &column, filters)?;
            match (lo, hi) {
                (Some(lo), Some(hi)) => (lo, hi, "MIN/MAX"),
                // Empty table, or only NULLs.
                _ => return one(),
            }
        },
    };

    let nulls_first = partition::nulls_sort_first(config.dialect);
    let parts = partition::range_partitions(&column, lo, hi, n, kind, nulls_first);
    let desc = format!(
        "{} (range on {}, bounds from {bounds_from})",
        parts.len(),
        config.dialect.quote_ident(&column)
    );
    Ok((parts, desc.into()))
}

/// `MIN` and `MAX` of a column, over the rows the exact filters keep.
fn query_bounds(
    config: &DatabaseScanConfig,
    renderer: &dyn SqlRenderer,
    from: &SqlFrom,
    column: &PlSmallStr,
    filters: &[PushedFilter],
) -> PolarsResult<(Option<f64>, Option<f64>)> {
    let mut query = SqlQuery::select(
        vec![
            SqlSelectItem::Min(column.clone()),
            SqlSelectItem::Max(column.clone()),
        ],
        from.clone(),
    );
    query.filter = filters
        .iter()
        .filter(|f| f.exact)
        .map(|f| f.expr.clone())
        .collect();
    // MySQL zero dates are read as NULL, and no driver reads them as a bound.
    if let Some(zero) = zero_date_check(config, column) {
        query.filter.push(SqlExpr::Not(Box::new(zero)));
    }
    let sql = renderer.render(&query, config.dialect)?;
    if let Some(bounds) = config.bounds_cache.lock().unwrap().get(&sql) {
        return Ok(*bounds);
    }
    if verbose() {
        eprintln!("[scan_database]: partition bounds: {sql}");
    }
    let backend = open_backend(&config.backend, None)?;
    let mut session =
        open_session_with_retries(backend.as_ref(), &config.options.connection, config.dialect)?;
    let df = session.query(&sql)?;
    polars_ensure!(
        df.width() == 2 && df.height() == 1,
        ComputeError: "unexpected result shape for the partition bounds query: {:?}", df.shape()
    );
    let bounds = (
        partition::bound_value(&df.columns()[0]),
        partition::bound_value(&df.columns()[1]),
    );
    config.bounds_cache.lock().unwrap().insert(sql, bounds);
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_guard() {
        assert_eq!(first_keyword("  -- x\n /* y */ (SELECT 1)"), "SELECT");
        assert!(check_read_only("DELETE FROM t", None).is_err());
        assert!(check_read_only("/* c */ insert into t values (1)", None).is_err());
        assert!(check_read_only("SELECT * FROM t", None).is_ok());
        let facts = QueryFacts {
            parses: true,
            read_only: false,
            ..Default::default()
        };
        assert!(check_read_only("WITH d AS (DELETE FROM t) SELECT 1", Some(&facts)).is_err());
    }
}
