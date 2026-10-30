use std::fmt;

use polars_error::{PolarsResult, polars_bail};
use serde::{Deserialize, Serialize};
use strum_macros::{EnumIter, IntoStaticStr};

/// The SQL dialect of a database. Decides how SQL is written and how types map. Independent
/// of the backend (driver) that moves the data.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, EnumIter, IntoStaticStr, Serialize, Deserialize,
)]
#[strum(serialize_all = "lowercase")]
pub enum DialectName {
    Postgresql,
    Redshift,
    Mysql,
    Mssql,
    Oracle,
    Sqlite,
    Duckdb,
    Snowflake,
    Bigquery,
    Databricks,
    Clickhouse,
    Trino,
    Flightsql,
    Exasol,
    Singlestore,
    Teradata,
    Generic,
}

impl fmt::Display for DialectName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.into())
    }
}

/// How string comparisons behave in the database, relative to Polars' byte-wise comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringSemantics {
    /// `=` / `IN` / `LIKE` match exactly what Polars matches. Otherwise they match a superset
    /// (case- or accent-insensitive, or PAD SPACE collations).
    pub equality_exact: bool,
    /// `<`, `>` order strings like Polars (byte order). Otherwise string ranges are not pushed.
    pub ordering_exact: bool,
    /// The empty string is NULL (Oracle), so `''` literals are never pushed.
    pub empty_is_null: bool,
}

/// A statement of the dialect's session setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupStatement {
    pub sql: &'static str,
    /// Pushed literals or comparisons rely on it: a session where it fails is not used.
    /// Otherwise it only tightens session hygiene, and a server that rejects it is ignored.
    pub required: bool,
}

impl StringSemantics {
    pub const BINARY: Self = Self {
        equality_exact: true,
        ordering_exact: true,
        empty_is_null: false,
    };
    /// Unknown collation: assume case-insensitive and PAD SPACE.
    pub const UNKNOWN: Self = Self {
        equality_exact: false,
        ordering_exact: false,
        empty_is_null: false,
    };
}

impl DialectName {
    pub fn all() -> impl Iterator<Item = Self> {
        <Self as strum::IntoEnumIterator>::iter()
    }

    pub fn parse(s: &str) -> PolarsResult<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "postgresql" | "postgres" | "pgsql" | "psql" | "cockroachdb" | "cockroach"
            | "timescale" | "timescaledb" | "neon" | "supabase" | "alloydb" | "yugabyte" => {
                Self::Postgresql
            },
            "redshift" => Self::Redshift,
            "mysql" | "mariadb" | "tidb" | "vitess" => Self::Mysql,
            "mssql" | "sqlserver" | "tsql" | "azuresql" | "synapse" => Self::Mssql,
            "oracle" | "oracledb" => Self::Oracle,
            "sqlite" | "sqlite3" => Self::Sqlite,
            "duckdb" | "md" | "motherduck" => Self::Duckdb,
            "snowflake" => Self::Snowflake,
            "bigquery" => Self::Bigquery,
            "databricks" | "spark" => Self::Databricks,
            "clickhouse" | "chdb" => Self::Clickhouse,
            "trino" | "presto" => Self::Trino,
            "flightsql" | "grpc" | "dremio" => Self::Flightsql,
            "exasol" => Self::Exasol,
            "singlestore" | "memsql" => Self::Singlestore,
            "teradata" | "teradatasql" => Self::Teradata,
            "generic" | "ansi" => Self::Generic,
            other => polars_bail!(InvalidOperation: "unknown database dialect: '{other}'"),
        })
    }

    /// Dialect from a URI scheme. Unknown schemes map to [`DialectName::Generic`].
    pub fn from_scheme(scheme: &str) -> Self {
        let scheme = scheme.to_ascii_lowercase();
        // `grpc+tls` is a Flight SQL scheme, not a `dialect+driver` pair.
        if scheme.starts_with("grpc") {
            return Self::Flightsql;
        }
        let base = scheme.split('+').next().unwrap_or(&scheme);
        Self::parse(base).unwrap_or(Self::Generic)
    }

    /// Dialect from the `Driver=` name of an ODBC connection string.
    pub fn from_odbc_driver(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        let rules: &[(&str, Self)] = &[
            ("sql server", Self::Mssql),
            ("mssql", Self::Mssql),
            ("postgres", Self::Postgresql),
            ("psqlodbc", Self::Postgresql),
            ("redshift", Self::Redshift),
            ("mariadb", Self::Mysql),
            ("mysql", Self::Mysql),
            ("oracle", Self::Oracle),
            ("sqlite", Self::Sqlite),
            ("duckdb", Self::Duckdb),
            ("snowflake", Self::Snowflake),
            ("bigquery", Self::Bigquery),
            ("databricks", Self::Databricks),
            ("simba spark", Self::Databricks),
            ("clickhouse", Self::Clickhouse),
            ("trino", Self::Trino),
            ("presto", Self::Trino),
            ("exasol", Self::Exasol),
            ("teradata", Self::Teradata),
        ];
        rules
            .iter()
            .find(|(pat, _)| n.contains(pat))
            .map_or(Self::Generic, |(_, d)| *d)
    }

    /// Dialect from the root module of a Python connection object's type.
    pub fn from_python_module(module: &str) -> Self {
        let m = module.to_ascii_lowercase();
        let m = m.trim_start_matches('_');
        let root = m.split('.').next().unwrap_or(m);
        match root {
            "psycopg" | "psycopg2" | "pg8000" | "asyncpg" | "adbc_driver_postgresql" => {
                Self::Postgresql
            },
            "redshift_connector" => Self::Redshift,
            "pymysql" | "mysqldb" | "mysql" | "mariadb" | "aiomysql" => Self::Mysql,
            "pymssql" | "mssql_python" => Self::Mssql,
            "oracledb" | "cx_oracle" => Self::Oracle,
            "sqlite3" | "aiosqlite" | "adbc_driver_sqlite" => Self::Sqlite,
            "duckdb" | "adbc_driver_duckdb" => Self::Duckdb,
            "snowflake" | "adbc_driver_snowflake" => Self::Snowflake,
            "adbc_driver_bigquery" => Self::Bigquery,
            "databricks" => Self::Databricks,
            "clickhouse_connect" | "clickhouse_driver" => Self::Clickhouse,
            "trino" | "prestodb" => Self::Trino,
            "adbc_driver_flightsql" => Self::Flightsql,
            "pyexasol" => Self::Exasol,
            "teradatasql" => Self::Teradata,
            _ if m.starts_with("google.cloud.bigquery") => Self::Bigquery,
            _ => Self::Generic,
        }
    }

    /// Dialect name used by the SQL renderer (sqlglot / polyglot-sql naming). `None` means the
    /// renderer's default (ANSI-like) dialect.
    pub fn renderer_dialect(self) -> Option<&'static str> {
        Some(match self {
            Self::Postgresql => "postgres",
            Self::Redshift => "redshift",
            Self::Mysql => "mysql",
            Self::Mssql => "tsql",
            Self::Oracle => "oracle",
            Self::Sqlite => "sqlite",
            Self::Duckdb => "duckdb",
            Self::Snowflake => "snowflake",
            Self::Bigquery => "bigquery",
            Self::Databricks => "databricks",
            Self::Clickhouse => "clickhouse",
            Self::Trino => "trino",
            Self::Exasol => "exasol",
            Self::Singlestore => "singlestore",
            Self::Teradata => "teradata",
            Self::Flightsql | Self::Generic => return None,
        })
    }

    /// ADBC driver name (manifest slug and URI scheme) for this dialect.
    pub fn adbc_driver(self) -> Option<&'static str> {
        Some(match self {
            Self::Postgresql => "postgresql",
            Self::Redshift => "redshift",
            Self::Mysql => "mysql",
            Self::Mssql => "mssql",
            Self::Oracle => "oracle",
            Self::Sqlite => "sqlite",
            Self::Duckdb => "duckdb",
            Self::Snowflake => "snowflake",
            Self::Bigquery => "bigquery",
            Self::Databricks => "databricks",
            Self::Clickhouse => "clickhouse",
            Self::Trino => "trino",
            Self::Flightsql => "flightsql",
            Self::Exasol => "exasol",
            Self::Singlestore => "singlestore",
            Self::Teradata => "teradata",
            Self::Generic => return None,
        })
    }

    /// Whether ConnectorX can read this dialect.
    pub fn connectorx_supported(self) -> bool {
        matches!(
            self,
            Self::Postgresql
                | Self::Redshift
                | Self::Mysql
                | Self::Mssql
                | Self::Oracle
                | Self::Sqlite
                | Self::Bigquery
                | Self::Clickhouse
                | Self::Trino
        )
    }

    /// Whether float comparisons agree with Polars. Polars uses a total order (NaN equals NaN
    /// and sorts above all numbers). Postgres and DuckDB do the same; MySQL, SQL Server, Oracle
    /// `NUMBER` and SQLite can't store NaN. Other dialects are not verified, so float
    /// predicates are not pushed there.
    pub fn float_comparisons_exact(self) -> bool {
        matches!(
            self,
            Self::Postgresql
                | Self::Redshift
                | Self::Duckdb
                | Self::Mysql
                | Self::Mssql
                | Self::Oracle
                | Self::Sqlite
        )
    }

    /// Whether date and timestamp comparisons against typed literals agree with Polars. SQLite
    /// stores temporal values as text or numbers in any format, so they are not pushed there.
    pub fn temporal_comparisons_exact(self) -> bool {
        !matches!(self, Self::Sqlite | Self::Generic | Self::Flightsql)
    }

    /// Days since the epoch that a date or timestamp literal may fall in: ClickHouse's
    /// `Date32` and `DateTime64` cover 1900-2299 and wrap silently outside that; MySQL turns
    /// dates outside 1000-9999 into NULL; elsewhere the literal text needs years 1-9999.
    pub fn temporal_literal_days(self) -> (i64, i64) {
        match self {
            Self::Clickhouse => (-25_567, 120_529),
            Self::Mysql | Self::Singlestore => (-354_285, 2_932_896),
            _ => (-719_162, 2_932_896),
        }
    }

    /// SQL that reads a column of a declared type as text, for types that drivers can't
    /// export to Arrow consistently (or at all): Postgres `uuid`/`json(b)` (bytes or Python
    /// objects depending on the driver), ClickHouse `UUID` and `Enum*`, Oracle `JSON`. `column`
    /// is quoted.
    pub fn text_conversion(self, db_type: &str, column: &str) -> Option<String> {
        let t = if self == Self::Clickhouse {
            super::schema::clickhouse_value_type(db_type)
        } else {
            db_type.trim().to_ascii_lowercase()
        };
        let t = t.as_str();
        // Decimals wider than Polars' 38 digits: their exact text.
        if wide_decimal(t) {
            return Some(self.cast_to_text(column));
        }
        match self {
            // Types outside the ones every driver reads natively (uuid, json, interval,
            // inet, money, enums, ranges, citext, domains, …): their text form. Drivers
            // otherwise disagree (objects, bytes of the wire format) or fail outright
            // (ADBC on `interval`, ConnectorX on `macaddr`).
            Self::Postgresql | Self::Redshift if !postgres_native_type(t) => {
                Some(format!("CAST({column} AS TEXT)"))
            },
            // the server's own text form, so every driver returns the same string
            Self::Mysql | Self::Singlestore if t == "json" || t.starts_with("set(") => {
                Some(format!("CAST({column} AS CHAR)"))
            },
            // WKT, not the internal WKB bytes (which ConnectorX can't read at all)
            Self::Mysql | Self::Singlestore if MYSQL_GEOMETRY.contains(&t) => {
                Some(format!("ST_AsText({column})"))
            },
            // `'1010'`, as Postgres writes `bit(n)`; drivers return bytes or integers
            Self::Mysql | Self::Singlestore if t == "bit" || t.starts_with("bit(") => {
                let width = t
                    .strip_prefix("bit(")
                    .and_then(|r| r.strip_suffix(')'))
                    .and_then(|w| w.trim().parse::<u32>().ok())
                    .unwrap_or(1);
                Some(format!("LPAD(BIN({column}), {width}, '0')"))
            },
            // UUID, enums, and everything outside the types every driver reads natively
            // (Map, Tuple, IPv4, FixedString, 128/256-bit integers, …): ConnectorX returns
            // wrong values and even wrong row counts for some of them.
            Self::Clickhouse if !clickhouse_native_type(t) => Some(format!("toString({column})")),
            Self::Oracle if t == "json" => Some(format!("JSON_SERIALIZE({column} RETURNING CLOB)")),
            Self::Oracle if t == "rowid" || t == "urowid" => Some(format!("ROWIDTOCHAR({column})")),
            // `INTERVAL … TO …`: Arrow's month-day-nano intervals have no Polars dtype
            Self::Oracle if t.starts_with("interval") => Some(format!("TO_CHAR({column})")),
            _ => None,
        }
    }

    /// SQL that reads any column as text.
    pub fn cast_to_text(self, column: &str) -> String {
        match self {
            Self::Mysql | Self::Singlestore => format!("CAST({column} AS CHAR)"),
            Self::Mssql => format!("CAST({column} AS NVARCHAR(MAX))"),
            Self::Oracle => format!("TO_CHAR({column})"),
            Self::Clickhouse => format!("toString({column})"),
            Self::Postgresql | Self::Redshift | Self::Sqlite | Self::Duckdb => {
                format!("CAST({column} AS TEXT)")
            },
            _ => format!("CAST({column} AS VARCHAR)"),
        }
    }

    /// Whether `LIKE` ignores (ASCII) case whatever the collation: SQLite's default.
    pub fn like_case_insensitive(self) -> bool {
        matches!(self, Self::Sqlite)
    }

    /// Whether decimal comparisons agree with Polars. SQLite has no decimal type: values
    /// declared as decimals are stored as text or floats.
    pub fn decimal_comparisons_exact(self) -> bool {
        !matches!(self, Self::Sqlite)
    }

    /// Whether a `TIME` type (and literal) exists.
    pub fn supports_time(self) -> bool {
        self.temporal_comparisons_exact() && !matches!(self, Self::Oracle)
    }

    /// Whether `IS [NOT] DISTINCT FROM` is available. Otherwise it is emulated.
    pub fn supports_is_distinct_from(self) -> bool {
        matches!(
            self,
            Self::Postgresql
                | Self::Redshift
                | Self::Duckdb
                | Self::Snowflake
                | Self::Bigquery
                | Self::Databricks
                | Self::Trino
                | Self::Sqlite
        )
    }

    /// Whether boolean literals and columns exist. MSSQL uses `BIT`, older Oracle `NUMBER(1)`.
    pub fn supports_boolean(self) -> bool {
        !matches!(self, Self::Mssql | Self::Oracle | Self::Teradata)
    }

    /// Longest `IN (…)` list pushed as-is. Longer lists become a `BETWEEN min AND max`
    /// superset.
    pub fn max_in_list(self) -> usize {
        match self {
            Self::Oracle => 1000,
            Self::Mssql => 2000,
            _ => 10_000,
        }
    }

    /// Whether `LIMIT`/`TOP`/`FETCH FIRST` is known to work for this dialect.
    pub fn supports_limit(self) -> bool {
        !matches!(self, Self::Generic | Self::Flightsql)
    }

    /// Default string semantics when the catalog gives no collation for a column.
    pub fn default_string_semantics(self) -> StringSemantics {
        match self {
            // Deterministic default collations; ordering depends on the collation (only "C"
            // matches byte order), which schema discovery refines per column.
            Self::Postgresql | Self::Redshift => StringSemantics {
                equality_exact: true,
                ordering_exact: false,
                empty_is_null: false,
            },
            // Strings are byte sequences, compared byte-wise.
            Self::Duckdb | Self::Clickhouse => StringSemantics::BINARY,
            // ANSI padding: `'a ' = 'a'` is true for every collation.
            Self::Mssql => StringSemantics::UNKNOWN,
            Self::Oracle => StringSemantics {
                equality_exact: true,
                ordering_exact: false,
                empty_is_null: true,
            },
            // SQLite is BINARY unless a column declares `COLLATE`, which schema discovery checks.
            Self::Sqlite => StringSemantics::BINARY,
            _ => StringSemantics::UNKNOWN,
        }
    }

    /// String semantics for a column with the given catalog collation name.
    pub fn string_semantics_for_collation(self, collation: Option<&str>) -> StringSemantics {
        let Some(c) = collation.map(|c| c.to_ascii_lowercase()) else {
            return self.default_string_semantics();
        };
        match self {
            // Empty: a non-deterministic collation (see the catalog query).
            Self::Postgresql | Self::Redshift if c.is_empty() => StringSemantics::UNKNOWN,
            Self::Postgresql | Self::Redshift => StringSemantics {
                equality_exact: true,
                ordering_exact: matches!(c.as_str(), "c" | "posix" | "ucs_basic"),
                empty_is_null: false,
            },
            Self::Mysql | Self::Singlestore => {
                // Only NO PAD binary collations (`*_0900_bin`, `binary`) are byte-exact.
                let exact = c == "binary" || (c.ends_with("_bin") && c.contains("0900"));
                StringSemantics {
                    equality_exact: exact,
                    ordering_exact: exact,
                    empty_is_null: false,
                }
            },
            Self::Sqlite => {
                if c == "binary" {
                    StringSemantics::BINARY
                } else {
                    StringSemantics::UNKNOWN
                }
            },
            _ => self.default_string_semantics(),
        }
    }

    /// Statements run once per session, before any query.
    pub fn session_setup(self) -> &'static [SetupStatement] {
        const fn optional(sql: &'static str) -> SetupStatement {
            SetupStatement {
                sql,
                required: false,
            }
        }
        const fn required(sql: &'static str) -> SetupStatement {
            SetupStatement {
                sql,
                required: true,
            }
        }
        const POSTGRES: &[SetupStatement] = &[
            optional("SET application_name = 'polars'"),
            optional("SET TIME ZONE 'UTC'"),
            optional("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"),
        ];
        const MYSQL: &[SetupStatement] = &[
            // `TIMESTAMP` values and literals are read in the session time zone
            required("SET time_zone = '+00:00'"),
            // literals are written with backslash escapes
            required(
                "SET SESSION sql_mode = REPLACE(@@SESSION.sql_mode, 'NO_BACKSLASH_ESCAPES', '')",
            ),
            optional("SET SESSION TRANSACTION READ ONLY"),
        ];
        // string comparisons are pushed as byte-wise (see `default_string_semantics`)
        const ORACLE: &[SetupStatement] = &[required(
            "ALTER SESSION SET NLS_COMP = BINARY NLS_SORT = BINARY",
        )];
        match self {
            Self::Postgresql => POSTGRES,
            Self::Mysql => MYSQL,
            Self::Oracle => ORACLE,
            _ => &[],
        }
    }

    /// Quotes a string literal for the fixed metadata queries.
    pub fn quote_str(self, s: &str) -> String {
        let mut out = s.replace('\'', "''");
        if matches!(
            self,
            Self::Mysql | Self::Singlestore | Self::Bigquery | Self::Redshift
        ) {
            out = out.replace('\\', "\\\\");
        }
        // `E'…'` reads backslashes the same whatever `standard_conforming_strings` is.
        if self == Self::Postgresql && out.contains('\\') {
            return format!("E'{}'", out.replace('\\', "\\\\"));
        }
        format!("'{out}'")
    }

    /// Quotes an identifier for the fixed metadata and partitioning queries.
    pub fn quote_ident(self, s: &str) -> String {
        match self {
            Self::Mysql | Self::Singlestore | Self::Bigquery | Self::Databricks => {
                format!("`{}`", s.replace('`', "``"))
            },
            Self::Mssql => format!("[{}]", s.replace(']', "]]")),
            _ => format!("\"{}\"", s.replace('"', "\"\"")),
        }
    }
}

const MYSQL_GEOMETRY: &[&str] = &[
    "geometry",
    "point",
    "linestring",
    "polygon",
    "multipoint",
    "multilinestring",
    "multipolygon",
    "geometrycollection",
    "geomcollection",
];

/// `numeric(p, s)` / `decimal(p, s)` / `number(p, s)` with `p > 38`, ClickHouse
/// `Decimal256(s)`.
fn wide_decimal(t: &str) -> bool {
    if t.starts_with("decimal256(") {
        return true;
    }
    ["numeric(", "decimal(", "number("]
        .iter()
        .find_map(|p| t.strip_prefix(p))
        .and_then(|args| args.split([',', ')']).next())
        .and_then(|p| p.trim().parse::<u32>().ok())
        .is_some_and(|p| p > 38)
}

/// ClickHouse types (lower-cased, unwrapped) that every driver reads natively.
fn clickhouse_native_type(t: &str) -> bool {
    let base = t.split('(').next().unwrap_or(t);
    match base {
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
        | "float32" | "float64" | "bool" | "string" | "date" | "date32" | "datetime"
        | "datetime64" | "decimal" | "decimal32" | "decimal64" | "decimal128" => true,
        "array" => t
            .strip_prefix("array(")
            .and_then(|r| r.strip_suffix(')'))
            .is_some_and(|inner| {
                let inner = super::schema::clickhouse_value_type(inner);
                !matches!(
                    inner.split('(').next(),
                    Some("array" | "uuid" | "enum8" | "enum16")
                ) && clickhouse_native_type(&inner)
            }),
        _ => false,
    }
}

/// Postgres types (as `format_type` writes them) that every driver reads natively.
fn postgres_native_type(t: &str) -> bool {
    // `numeric(12,2)`, `timestamp(3) without time zone`, `integer[]`
    let mut base = String::with_capacity(t.len());
    let mut depth = 0;
    for c in t.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 => base.push(c),
            _ => {},
        }
    }
    let base = base
        .trim_end_matches("[]")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    matches!(
        base.as_str(),
        "smallint"
            | "integer"
            | "bigint"
            | "real"
            | "double precision"
            | "numeric"
            | "boolean"
            | "text"
            | "character varying"
            | "character"
            | "\"char\""
            | "date"
            | "timestamp without time zone"
            | "timestamp with time zone"
            | "time without time zone"
            | "bytea"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection() {
        assert_eq!(
            DialectName::from_scheme("postgres"),
            DialectName::Postgresql
        );
        assert_eq!(DialectName::from_scheme("mssql+pyodbc"), DialectName::Mssql);
        assert_eq!(DialectName::from_scheme("grpc+tls"), DialectName::Flightsql);
        assert_eq!(DialectName::from_scheme("db2"), DialectName::Generic);
        assert_eq!(
            DialectName::from_odbc_driver("ODBC Driver 18 for SQL Server"),
            DialectName::Mssql
        );
        assert_eq!(
            DialectName::from_python_module("psycopg.connection"),
            DialectName::Postgresql
        );
        assert_eq!(
            DialectName::from_python_module("_duckdb"),
            DialectName::Duckdb
        );
    }

    #[test]
    fn collations() {
        let my = DialectName::Mysql;
        assert!(
            !my.string_semantics_for_collation(Some("utf8mb4_0900_ai_ci"))
                .equality_exact
        );
        assert!(
            !my.string_semantics_for_collation(Some("utf8mb4_bin"))
                .equality_exact
        );
        assert!(
            my.string_semantics_for_collation(Some("utf8mb4_0900_bin"))
                .equality_exact
        );
        let pg = DialectName::Postgresql;
        assert!(pg.string_semantics_for_collation(Some("C")).ordering_exact);
        assert!(
            !pg.string_semantics_for_collation(Some("en_US.utf8"))
                .ordering_exact
        );
    }

    #[test]
    fn quoting() {
        assert_eq!(DialectName::Mysql.quote_str("a'b\\c"), "'a''b\\\\c'");
        assert_eq!(DialectName::Postgresql.quote_str("a'b"), "'a''b'");
        assert_eq!(DialectName::Postgresql.quote_str("a'b\\c"), "E'a''b\\\\c'");
        assert_eq!(DialectName::Postgresql.quote_ident("a\"b"), "\"a\"\"b\"");
        assert_eq!(DialectName::Mssql.quote_ident("a]b"), "[a]]b]");
    }
}
