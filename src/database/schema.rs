//! Schema normalisation. The planned schema is authoritative: every batch is cast to it, so
//! the output does not depend on which backend or machine produced it.

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail, polars_err};

use super::dialect::DialectName;

/// Normalises a dtype from a backend to the dtype Polars plans with.
pub fn normalize_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Datetime(tu, Some(tz)) if tz.as_str().is_empty() => DataType::Datetime(*tu, None),
        // The session time zone is UTC; drivers spell it differently.
        DataType::Datetime(tu, Some(tz))
            if matches!(
                tz.as_str(),
                "Etc/UTC" | "Etc/UCT" | "UCT" | "Universal" | "Etc/Universal" | "Zulu" | "Etc/Zulu"
            ) =>
        {
            DataType::Datetime(*tu, Some(TimeZone::UTC))
        },
        // Opaque driver types are read as their storage, except ADBC Postgres `NUMERIC`
        // (text), which is a decimal like with every other driver.
        DataType::Extension(ext, storage) => {
            if ext.name() == "arrow.opaque"
                && ext.serialize_metadata().is_some_and(|m| {
                    m.contains("\"type_name\": \"numeric\"")
                        || m.contains("\"type_name\":\"numeric\"")
                })
            {
                return UNCONSTRAINED_DECIMAL;
            }
            let _ = ext;
            normalize_dtype(storage)
        },
        dt => dt.clone(),
    }
}

/// The dtype of a `NUMERIC` without precision, whose values each have their own scale
/// (read as exact text). A value with more digits than the scale is an error; use
/// `schema_overrides` for another precision.
pub const UNCONSTRAINED_DECIMAL: DataType = DataType::Decimal(38, 9);

/// The dtype of a declared database type, where the declaration fixes it whatever the driver
/// returns. Drivers differ here: ADBC Postgres returns `NUMERIC` as text and ConnectorX as a
/// wider decimal; row-wise drivers return every integer as `Int64`.
///
/// - `NUMERIC(p, s)` / `DECIMAL(p, s)` with `p <= 38` is `Decimal(p, s)`;
/// - integer types map to their width (and signedness, for MySQL `UNSIGNED` and SQL Server
///   `TINYINT`); SQL Server `BIT` is `Boolean`.
pub fn declared_dtype(dialect: DialectName, db_type: &str) -> Option<DataType> {
    let mut t = db_type.trim().to_ascii_lowercase();
    // Arrays of a declared type: drivers pick the element width themselves otherwise.
    if matches!(dialect, DialectName::Postgresql | DialectName::Redshift)
        && let Some(inner) = t.strip_suffix("[]")
    {
        return declared_dtype(dialect, inner).map(|d| DataType::List(Box::new(d)));
    }
    if dialect == DialectName::Clickhouse {
        t = clickhouse_value_type(&t);
        if let Some(inner) = t.strip_prefix("array(").and_then(|r| r.strip_suffix(')')) {
            return declared_dtype(dialect, inner).map(|d| DataType::List(Box::new(d)));
        }
        if let Some(dt) = clickhouse_dtype(&t) {
            return Some(dt);
        }
    }
    if let Some(dt) = declared_integer(dialect, &t) {
        return Some(dt);
    }
    if (matches!(dialect, DialectName::Postgresql | DialectName::Redshift) && t == "numeric")
        || (dialect == DialectName::Oracle && t == "number")
    {
        return Some(UNCONSTRAINED_DECIMAL);
    }
    // Oracle `TIMESTAMP WITH [LOCAL] TIME ZONE`: values are normalised to the UTC session.
    if dialect == DialectName::Oracle && t.starts_with("timestamp") && t.contains("time zone") {
        return Some(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
    }
    // Drivers disagree on the width of MySQL `YEAR` and `FLOAT` (4 bytes).
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore) {
        if t == "year" || t.starts_with("year(") {
            return Some(DataType::Int16);
        }
        if t == "float" || t.starts_with("float ") {
            return Some(DataType::Float32);
        }
    }
    // MySQL `TIME` is read as a duration by some drivers.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && (t == "time" || t.starts_with("time("))
    {
        return Some(DataType::Time);
    }
    // MySQL `TIMESTAMP` is stored in UTC (the session time zone is UTC); `DATETIME` is naive.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && (t == "timestamp" || t.starts_with("timestamp("))
    {
        return Some(DataType::Datetime(
            TimeUnit::Microseconds,
            Some(TimeZone::UTC),
        ));
    }
    if dialect == DialectName::Oracle {
        // `DATE` has a time part; `NUMBER(p, 0)` with p <= 18 always fits an Int64.
        if t == "date" {
            return Some(DataType::Datetime(TimeUnit::Microseconds, None));
        }
        if let Some(args) = t.strip_prefix("number(").and_then(|r| r.strip_suffix(')'))
            && let Some((p, s)) = args.split_once(',')
            && s.trim() == "0"
            && p.trim().parse::<usize>().is_ok_and(|p| p <= 18)
        {
            return Some(DataType::Int64);
        }
    }
    // MySQL `decimal(30,15) unsigned zerofill`: the attributes don't change the values.
    let mut t = t.as_str();
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore) {
        while let Some(rest) = t
            .strip_suffix("unsigned")
            .or_else(|| t.strip_suffix("zerofill"))
        {
            t = rest.trim_end();
        }
    }
    let args = ["numeric(", "decimal(", "number("]
        .iter()
        .find_map(|p| t.strip_prefix(p))?;
    let args = args.strip_suffix(')')?;
    let (p, s) = match args.split_once(',') {
        Some((p, s)) => (
            p.trim().parse::<usize>().ok()?,
            s.trim().parse::<usize>().ok()?,
        ),
        None => (args.trim().parse::<usize>().ok()?, 0),
    };
    if !(1..=38).contains(&p) || s > p {
        return None;
    }
    Some(DataType::Decimal(p, s))
}

/// Casts a batch to the planned schema: selects the planned columns in order and casts each.
/// The columns in `overrides` (`schema_overrides`) are cast as the user asked, also when
/// a float becomes an integer.
pub fn cast_to_schema(
    df: &DataFrame,
    schema: &Schema,
    overrides: Option<&Schema>,
) -> PolarsResult<DataFrame> {
    let height = df.height();
    let mut columns = Vec::with_capacity(schema.len());
    for (name, dtype) in schema.iter() {
        let col = df.column(name).map_err(|_| {
            polars_err!(
                SchemaMismatch:
                "database result is missing column '{name}'"
            )
        })?;
        let storage;
        let col = if matches!(col.dtype(), DataType::Extension(_, _)) {
            storage = col
                .as_materialized_series()
                .to_storage()
                .clone()
                .into_column();
            &storage
        } else {
            col
        };
        let converted;
        let col = match duration_to_time(col, dtype)? {
            Some(c) => {
                converted = c;
                &converted
            },
            None => col,
        };
        let parsed;
        let col = match parse_temporal(col, dtype)? {
            Some(p) => {
                parsed = p;
                &parsed
            },
            None => col,
        };
        let col = if col.dtype() == dtype {
            col.clone()
        } else if col.dtype() == &DataType::Null {
            Column::full_null(name.clone(), height, dtype)
        } else {
            let cast = col
                .cast_with_options(dtype, CastOptions::Strict)
                .map_err(|e| {
                    polars_err!(
                        SchemaMismatch:
                        "database column '{name}' is {} but {dtype} was planned: {e}",
                        col.dtype()
                    )
                })?;
            check_decimal_exact(col, &cast)?;
            if overrides.is_none_or(|o| !o.contains(name)) {
                check_numeric_exact(col, &cast)?;
            }
            cast
        };
        columns.push(col);
    }
    DataFrame::new(height, columns)
}

/// A decimal cast must not round an exact value (text or a decimal of a larger scale):
/// unconstrained `numeric` / `NUMBER` are planned as `Decimal(38, 9)`, and a value with
/// more digits is an error, not silently another number. Floats only approximate a
/// decimal, so rounding them to the declared scale is their reading.
fn check_decimal_exact(src: &Column, cast: &Column) -> PolarsResult<()> {
    let DataType::Decimal(_, scale) = cast.dtype() else {
        return Ok(());
    };
    let bad = match src.dtype() {
        DataType::String => src
            .str()?
            .iter()
            .flatten()
            .find(|v| {
                let v = v.trim();
                !v.contains(['e', 'E'])
                    && v.split_once('.')
                        .is_some_and(|(_, frac)| frac.trim_end_matches('0').len() > *scale)
            })
            .map(str::to_string),
        DataType::Decimal(_, src_scale) if src_scale > scale => {
            let back = cast.cast(src.dtype())?;
            let same = src
                .as_materialized_series()
                .equal_missing(back.as_materialized_series())?;
            match same.iter().position(|v| v == Some(false)) {
                Some(i) => Some(src.get(i)?.to_string()),
                None => None,
            }
        },
        _ => None,
    };
    match bad {
        Some(v) => polars_bail!(
            SchemaMismatch:
            "database column '{}' has the value {v}, which {} would round; read the column \
            as text with schema_overrides={{'{}': pl.String}} (or as pl.Float64)",
            src.name(), cast.dtype(), src.name()
        ),
        None => Ok(()),
    }
}

/// A cast between integers and floats (or from a decimal to an integer) must not round
/// either: a strict cast truncates `1000.5` to `1000`, and `2**53 + 1` becomes
/// `9007199254740992.0`. Columns typed from a sample (SQLite, which has no column types)
/// may hold values of both kinds.
fn check_numeric_exact(src: &Column, cast: &Column) -> PolarsResult<()> {
    let (from, to) = (src.dtype(), cast.dtype());
    let lossy = (to.is_integer() && (from.is_float() || from.is_decimal()))
        || (to.is_float() && from.is_integer());
    if !lossy {
        return Ok(());
    }
    let back = cast.cast(from)?;
    let same = src
        .as_materialized_series()
        .equal_missing(back.as_materialized_series())?;
    let Some(i) = same.iter().position(|v| v != Some(true)) else {
        return Ok(());
    };
    let alt = if to.is_integer() {
        "pl.Float64"
    } else {
        "pl.Int64"
    };
    polars_bail!(
        SchemaMismatch:
        "database column '{}' has the value {}, which {to} would round; read the column with \
        schema_overrides={{'{}': {alt}}} (or as pl.String)",
        src.name(), src.get(i)?, src.name()
    )
}

/// A `TIME` read as a duration (PyMySQL returns `timedelta`s) as a time of day. MySQL `TIME`
/// may be negative or exceed 24 hours: such values are an error.
fn duration_to_time(col: &Column, dtype: &DataType) -> PolarsResult<Option<Column>> {
    if let (DataType::Duration(tu), DataType::Time) = (col.dtype(), dtype) {
        let ns_per = match tu {
            TimeUnit::Nanoseconds => 1,
            TimeUnit::Microseconds => 1_000,
            TimeUnit::Milliseconds => 1_000_000,
        };
        let phys = col.to_physical_repr();
        let values = phys.i64()?;
        let day = 86_400_000_000_000i64;
        if let Some(bad) = values
            .iter()
            .flatten()
            .find(|v| !(0..day).contains(&(v * ns_per)))
        {
            polars_bail!(
                SchemaMismatch:
                "database column '{}' has a duration of {bad} {tu}, which is not a time of day",
                col.name()
            );
        }
        let ns = values.apply_values(|v| v * ns_per).into_series();
        return Ok(Some(ns.cast(&DataType::Time)?.into_column()));
    }
    let _ = (col, dtype);
    Ok(None)
}

/// Text of a temporal column (SQLite stores dates as text) parsed to the planned dtype.
/// Polars 2.0 has no string-to-temporal cast. A value that doesn't parse is an error.
fn parse_temporal(col: &Column, dtype: &DataType) -> PolarsResult<Option<Column>> {
    use polars_core::chunked_array::temporal::string::StringMethods;

    if !col.dtype().is_string() {
        return Ok(None);
    }
    let ca = col.str()?;
    let parsed: Series = match dtype {
        DataType::Date => ca.as_date(Some("%Y-%m-%d"), true)?.into_series(),
        DataType::Datetime(tu, tz) => {
            let raise = StringChunked::from_slice(PlSmallStr::EMPTY, &["raise"]);
            ca.as_datetime(None, *tu, true, false, tz.as_ref(), &raise)?
                .into_series()
        },
        DataType::Time => ca.as_time(None, true)?.into_series(),
        _ => return Ok(None),
    };
    if parsed.null_count() != ca.null_count() {
        let bad = ca
            .iter()
            .zip(parsed.is_null().iter())
            .find_map(|(s, null)| (null == Some(true)).then_some(s).flatten())
            .unwrap_or_default();
        polars_bail!(
            SchemaMismatch:
            "database column '{}' has text '{bad}', which is not a {dtype}",
            col.name()
        );
    }
    Ok(Some(parsed.into_column()))
}

/// ClickHouse value types whose dtype the declaration fixes.
/// A ClickHouse type without its `Nullable(…)` and `LowCardinality(…)` wrappers, lower-cased.
pub fn clickhouse_value_type(db_type: &str) -> String {
    let mut t = db_type.trim().to_ascii_lowercase();
    loop {
        let inner = ["nullable(", "lowcardinality("]
            .iter()
            .find_map(|p| t.strip_prefix(p).and_then(|r| r.strip_suffix(')')));
        match inner {
            Some(inner) => t = inner.trim().to_string(),
            None => return t,
        }
    }
}

fn clickhouse_dtype(t: &str) -> Option<DataType> {
    use DataType as D;
    Some(match t {
        "int8" => D::Int8,
        "int16" => D::Int16,
        "int32" => D::Int32,
        "int64" => D::Int64,
        "uint8" => D::UInt8,
        "uint16" => D::UInt16,
        "uint32" => D::UInt32,
        "uint64" => D::UInt64,
        "float32" => D::Float32,
        "float64" => D::Float64,
        "bool" => D::Boolean,
        "date" | "date32" => D::Date,
        "string" => D::String,
        // read as text (`text_conversion`)
        _ if t.starts_with("enum8(") || t.starts_with("enum16(") => D::String,
        // seconds; read through `toDateTime64(…, 0)` (`column_conversion`)
        _ if t == "datetime" || t.starts_with("datetime(") => {
            let tz = t
                .strip_prefix("datetime(")
                .and_then(|r| r.strip_suffix(')'))
                .map(|tz| tz.trim().trim_matches('\'').to_string());
            let tz = match tz {
                Some(tz) if tz.eq_ignore_ascii_case("utc") => Some(TimeZone::UTC),
                tz => TimeZone::opt_try_new(tz).ok()?,
            };
            D::Datetime(TimeUnit::Milliseconds, tz)
        },
        _ => {
            if let Some(args) = t
                .strip_prefix("datetime64(")
                .and_then(|r| r.strip_suffix(')'))
            {
                let (precision, tz) = match args.split_once(',') {
                    Some((p, tz)) => (p, Some(tz.trim().trim_matches('\'').to_string())),
                    None => (args, None),
                };
                let unit = match precision.trim().parse::<u32>().ok()? {
                    0..=3 => TimeUnit::Milliseconds,
                    4..=6 => TimeUnit::Microseconds,
                    _ => TimeUnit::Nanoseconds,
                };
                let tz = match tz {
                    Some(tz) if tz.eq_ignore_ascii_case("utc") => Some(TimeZone::UTC),
                    // An unknown zone name keeps the driver's dtype.
                    tz => TimeZone::opt_try_new(tz).ok()?,
                };
                return Some(D::Datetime(unit, tz));
            }
            return None;
        },
    })
}

fn declared_integer(dialect: DialectName, t: &str) -> Option<DataType> {
    use DataType as D;
    let unsigned = t.contains("unsigned");
    // `int(11)`, `tinyint(1) unsigned`: the display width doesn't matter.
    let base = t.split(['(', ' ']).next()?;
    let (signed, unsigned_dt) = match (dialect, base) {
        (DialectName::Mssql, "tinyint") => return Some(D::UInt8),
        (DialectName::Mssql, "bit") => return Some(D::Boolean),
        (DialectName::Mysql | DialectName::Singlestore, "tinyint") => (D::Int8, D::UInt8),
        (_, "smallint" | "int2") => (D::Int16, D::UInt16),
        (DialectName::Mysql | DialectName::Singlestore, "mediumint") => (D::Int32, D::UInt32),
        (_, "integer" | "int" | "int4") => (D::Int32, D::UInt32),
        (_, "bigint" | "int8") => (D::Int64, D::UInt64),
        _ => return None,
    };
    let is_mysql = matches!(dialect, DialectName::Mysql | DialectName::Singlestore);
    if !matches!(
        dialect,
        DialectName::Postgresql | DialectName::Redshift | DialectName::Mssql
    ) && !is_mysql
    {
        return None;
    }
    Some(if unsigned && is_mysql {
        unsigned_dt
    } else {
        signed
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inexact_numeric_casts_fail() {
        let cast = |col: Column, dtype: DataType| {
            let df = DataFrame::new(col.len(), vec![col]).unwrap();
            cast_to_schema(
                &df,
                &Schema::from_iter([Field::new("x".into(), dtype)]),
                None,
            )
        };
        let floats = |v: &[f64]| Column::new("x".into(), v);
        assert!(cast(floats(&[1.0, 2.0]), DataType::Int64).is_ok());
        let e = cast(floats(&[1.0, 1000.5]), DataType::Int64).unwrap_err();
        assert!(e.to_string().contains("1000.5"), "{e}");
        let ints = |v: &[i64]| Column::new("x".into(), v);
        assert!(cast(ints(&[1, 1 << 53]), DataType::Float64).is_ok());
        assert!(cast(ints(&[1, (1 << 53) + 1]), DataType::Float64).is_err());
        let with_null = Column::new("x".into(), &[Some(1.0), None]);
        assert!(cast(with_null, DataType::Int64).is_ok());
    }

    #[test]
    fn declared_types() {
        {
            let pg = DialectName::Postgresql;
            assert_eq!(
                declared_dtype(pg, "numeric(12,2)"),
                Some(DataType::Decimal(12, 2))
            );
            assert_eq!(
                declared_dtype(pg, "DECIMAL(10, 0)"),
                Some(DataType::Decimal(10, 0))
            );
            assert_eq!(
                declared_dtype(DialectName::Oracle, "number(19,0)"),
                Some(DataType::Decimal(19, 0))
            );
            assert_eq!(
                declared_dtype(DialectName::Oracle, "number(10,0)"),
                Some(DataType::Int64)
            );
        }
        let pg = DialectName::Postgresql;
        assert_eq!(declared_dtype(pg, "numeric"), Some(UNCONSTRAINED_DECIMAL));
        assert_eq!(declared_dtype(pg, "numeric(50,2)"), None);
        assert_eq!(declared_dtype(pg, "text"), None);
        assert_eq!(declared_dtype(pg, "integer"), Some(DataType::Int32));
        let my = DialectName::Mysql;
        assert_eq!(declared_dtype(my, "tinyint(1)"), Some(DataType::Int8));
        assert_eq!(
            declared_dtype(my, "int(10) unsigned"),
            Some(DataType::UInt32)
        );
        assert_eq!(
            declared_dtype(my, "bigint unsigned"),
            Some(DataType::UInt64)
        );
        assert_eq!(
            declared_dtype(my, "decimal(30,15) unsigned"),
            Some(DataType::Decimal(30, 15))
        );
        assert_eq!(
            declared_dtype(my, "DECIMAL(20,5) UNSIGNED ZEROFILL"),
            Some(DataType::Decimal(20, 5))
        );
        let ms = DialectName::Mssql;
        assert_eq!(declared_dtype(ms, "tinyint"), Some(DataType::UInt8));
        assert_eq!(declared_dtype(ms, "bit"), Some(DataType::Boolean));
        assert_eq!(declared_dtype(DialectName::Sqlite, "integer"), None);
        let ch = DialectName::Clickhouse;
        assert_eq!(declared_dtype(ch, "Nullable(Int16)"), Some(DataType::Int16));
        assert_eq!(
            declared_dtype(ch, "LowCardinality(Nullable(String))"),
            Some(DataType::String)
        );
        assert_eq!(
            declared_dtype(ch, "Nullable(Decimal(18, 4))"),
            Some(DataType::Decimal(18, 4))
        );
        assert_eq!(
            declared_dtype(ch, "Nullable(DateTime64(6))"),
            Some(DataType::Datetime(TimeUnit::Microseconds, None))
        );
        assert_eq!(
            declared_dtype(ch, "DateTime64(3, 'UTC')"),
            Some(DataType::Datetime(
                TimeUnit::Milliseconds,
                Some(TimeZone::UTC)
            ))
        );
    }
}
