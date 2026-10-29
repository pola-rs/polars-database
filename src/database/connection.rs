use std::fmt;
use std::hash::{Hash, Hasher};
#[cfg(feature = "python")]
use std::sync::Arc;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use serde::{Deserialize, Serialize};

/// Where a database scan connects to.
#[derive(Clone, Serialize, Deserialize)]
pub enum ConnectionTarget {
    /// `scheme://…`, including `scheme+driver://…` and `profile://name`.
    Uri(PlSmallStr),
    /// An ODBC connection string, e.g. `Driver={…};Server=…`.
    Odbc(PlSmallStr),
    /// A zero-argument Python callable that returns a DB-API connection.
    #[cfg(feature = "python")]
    Factory(Arc<PythonObject>),
    /// A live Python connection object. Usable on the local engines only.
    #[cfg(feature = "python")]
    Live(Arc<LiveConnection>),
}

/// A live connection object can't be sent anywhere, so it refuses to serialize.
#[cfg(feature = "python")]
#[derive(Clone)]
pub struct LiveConnection(pub PythonObject);

#[cfg(feature = "python")]
impl Serialize for LiveConnection {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "scan_database with a live connection object cannot be serialized; \
            pass a URI or a connection factory instead",
        ))
    }
}

#[cfg(feature = "python")]
impl<'de> Deserialize<'de> for LiveConnection {
    fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "scan_database with a live connection object cannot be deserialized",
        ))
    }
}

/// Everything needed to open a connection, as given by the user.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub target: ConnectionTarget,
    /// Driver-specific connect options (ADBC database options, ConnectorX `protocol`, …).
    pub connect_options: Vec<(PlSmallStr, PlSmallStr)>,
    /// Driver-specific statement options.
    pub execute_options: Vec<(PlSmallStr, PlSmallStr)>,
    /// Called where the connection is opened; returns `{user, password}` overrides.
    #[cfg(feature = "python")]
    pub credential_provider: Option<Arc<PythonObject>>,
    /// Bound parameters of a query source, passed to the driver's own binding. Never
    /// interpolated into the SQL.
    #[cfg(feature = "python")]
    pub query_params: Option<Arc<PythonObject>>,
    /// A stable identity of `query_params` (their `repr`), for scan deduplication.
    pub query_params_key: Option<PlSmallStr>,
    /// `(connect_options, execute_options)` with their Python values, for Python backends:
    /// driver keyword arguments aren't always strings.
    #[cfg(feature = "python")]
    pub python_options: Option<Arc<PythonObject>>,
}

impl ConnectionSpec {
    pub fn new(target: ConnectionTarget) -> Self {
        Self {
            target,
            connect_options: vec![],
            execute_options: vec![],
            #[cfg(feature = "python")]
            credential_provider: None,
            #[cfg(feature = "python")]
            query_params: None,
            query_params_key: None,
            #[cfg(feature = "python")]
            python_options: None,
        }
    }

    pub fn has_query_params(&self) -> bool {
        self.query_params_key.is_some()
    }

    pub fn uri(&self) -> Option<&str> {
        match &self.target {
            ConnectionTarget::Uri(uri) => Some(uri),
            _ => None,
        }
    }

    /// The URI scheme, lower-cased, without a `+driver` suffix.
    pub fn scheme(&self) -> Option<PlSmallStr> {
        let uri = self.uri()?;
        let (scheme, _) = uri.split_once(':')?;
        let base = scheme.split('+').next().unwrap_or(scheme);
        Some(base.to_ascii_lowercase().into())
    }

    /// The `+driver` suffix of a SQLAlchemy-style URI scheme, if any.
    pub fn scheme_driver(&self) -> Option<PlSmallStr> {
        let uri = self.uri()?;
        let (scheme, _) = uri.split_once(':')?;
        let (_, driver) = scheme.split_once('+')?;
        Some(driver.to_ascii_lowercase().into())
    }

    /// The URI without the `+driver` part of its scheme, for drivers that don't understand it.
    pub fn uri_without_scheme_driver(&self) -> Option<String> {
        let uri = self.uri()?;
        let (scheme, rest) = uri.split_once(':')?;
        let base = scheme.split('+').next().unwrap_or(scheme);
        Some(format!("{base}:{rest}"))
    }

    /// The `Driver=` name of an ODBC connection string.
    pub fn odbc_driver_name(&self) -> Option<PlSmallStr> {
        let ConnectionTarget::Odbc(s) = &self.target else {
            return None;
        };
        s.split(';').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            k.trim()
                .eq_ignore_ascii_case("driver")
                .then(|| v.trim().trim_matches(|c| c == '{' || c == '}').into())
        })
    }

    pub fn is_live(&self) -> bool {
        #[cfg(feature = "python")]
        {
            matches!(self.target, ConnectionTarget::Live(_))
        }
        #[cfg(not(feature = "python"))]
        {
            false
        }
    }

    /// A stable identity of the connection that never includes secrets. Used for scan
    /// deduplication and caching.
    pub fn fingerprint(&self, state: &mut impl Hasher) {
        match &self.target {
            ConnectionTarget::Uri(uri) => {
                0u8.hash(state);
                redact_uri(uri).hash(state);
            },
            ConnectionTarget::Odbc(s) => {
                1u8.hash(state);
                redact_odbc(s).hash(state);
            },
            #[cfg(feature = "python")]
            ConnectionTarget::Factory(f) => {
                2u8.hash(state);
                (Arc::as_ptr(f) as usize).hash(state);
            },
            #[cfg(feature = "python")]
            ConnectionTarget::Live(c) => {
                3u8.hash(state);
                (Arc::as_ptr(c) as usize).hash(state);
            },
        }
        // Values too: scans that differ only in an option (a search path, a role) read
        // different data and must not be merged.
        self.connect_options.hash(state);
        self.execute_options.hash(state);
        #[cfg(feature = "python")]
        self.credential_provider
            .as_ref()
            .map(|p| Arc::as_ptr(p) as usize)
            .hash(state);
        self.query_params_key.hash(state);
    }
}

/// Replaces the password of a `scheme://user:password@host` URI and the values of
/// secret-looking query parameters.
pub fn redact_uri(uri: &str) -> String {
    let mut out = uri.to_string();
    if let Some(scheme_end) = out.find("://") {
        let auth_start = scheme_end + 3;
        let rest = &out[auth_start..];
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if let Some(at) = rest[..auth_end].rfind('@') {
            let userinfo = &rest[..at];
            if let Some(colon) = userinfo.find(':') {
                let start = auth_start + colon + 1;
                let end = auth_start + at;
                out.replace_range(start..end, "***");
            }
        }
    }
    if let Some(q) = out.find('?') {
        let (head, query) = out.split_at(q + 1);
        let redacted = query
            .split('&')
            .map(|kv| match kv.split_once('=') {
                Some((k, _)) if is_secret_key(k) => format!("{k}=***"),
                _ => kv.to_string(),
            })
            .collect::<Vec<_>>()
            .join("&");
        out = format!("{head}{redacted}");
    }
    out
}

pub fn redact_odbc(s: &str) -> String {
    s.split(';')
        .map(|kv| match kv.split_once('=') {
            Some((k, _)) if is_secret_key(k.trim()) => format!("{k}=***"),
            _ => kv.to_string(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn is_secret_key(k: &str) -> bool {
    let k = k.to_ascii_lowercase();
    ["password", "pwd", "secret", "token", "key", "credential"]
        .iter()
        .any(|s| k.contains(s))
}

impl fmt::Debug for ConnectionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uri(uri) => write!(f, "Uri({})", redact_uri(uri)),
            Self::Odbc(s) => write!(f, "Odbc({})", redact_odbc(s)),
            #[cfg(feature = "python")]
            Self::Factory(_) => f.write_str("Factory(<callable>)"),
            #[cfg(feature = "python")]
            Self::Live(_) => f.write_str("Live(<connection>)"),
        }
    }
}

impl fmt::Display for ConnectionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uri(uri) => f.write_str(&redact_uri(uri)),
            Self::Odbc(s) => f.write_str(&redact_odbc(s)),
            #[cfg(feature = "python")]
            Self::Factory(_) => f.write_str("<connection factory>"),
            #[cfg(feature = "python")]
            Self::Live(_) => f.write_str("<live connection>"),
        }
    }
}

impl fmt::Debug for ConnectionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionSpec")
            .field("target", &self.target)
            .field(
                "connect_options",
                &self
                    .connect_options
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_passwords() {
        assert_eq!(
            redact_uri("postgresql://u:secret@h:5432/db?sslmode=require&password=x"),
            "postgresql://u:***@h:5432/db?sslmode=require&password=***"
        );
        assert_eq!(redact_uri("sqlite:///tmp/a.db"), "sqlite:///tmp/a.db");
        assert_eq!(redact_uri("postgresql://u@h/db"), "postgresql://u@h/db");
        assert_eq!(
            redact_odbc("Driver={X};Server=s;PWD=abc;UID=u"),
            "Driver={X};Server=s;PWD=***;UID=u"
        );
    }

    #[test]
    fn scheme_parts() {
        let spec = ConnectionSpec::new(ConnectionTarget::Uri("PostgreSQL+psycopg://u@h/db".into()));
        assert_eq!(spec.scheme().as_deref(), Some("postgresql"));
        assert_eq!(spec.scheme_driver().as_deref(), Some("psycopg"));
        assert_eq!(
            spec.uri_without_scheme_driver().as_deref(),
            Some("PostgreSQL://u@h/db")
        );
        let odbc = ConnectionSpec::new(ConnectionTarget::Odbc(
            "Driver={ODBC Driver 18 for SQL Server};Server=x".into(),
        ));
        assert_eq!(
            odbc.odbc_driver_name().as_deref(),
            Some("ODBC Driver 18 for SQL Server")
        );
    }
}
