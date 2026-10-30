//! The ADBC backend: loads ADBC driver shared libraries and reads query results as Arrow C
//! streams.
//!
//! Drivers are found through ADBC driver manifests (see [`manifest`]) or given as a library
//! path. Two loaders can then drive them, chosen by `POLARS_DATABASE_ADBC_LOADER`
//! ([`AdbcLoader`]):
//!
//! - `rust` (the default): no driver manager. The library is `dlopen`ed and initialised
//!   through its `AdbcDriverInit`-style entrypoint, and called through its C ABI.
//! - `python`: Python's `adbc_driver_manager`, through the Python bridge.
//!
//! Both get the same database and statement options ([`connect_options`],
//! [`batch_size_option`]) and import results the same way ([`import_stream`]), so they return
//! the same data. Drivers are never unloaded: Go drivers hang on unload.

mod ffi;
pub mod manifest;

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, LazyLock, Mutex};

use polars_arrow::array::{Array, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use self::ffi::*;
pub use self::manifest::{FoundDriver, find_driver, manifest_search_dirs};
use super::{
    BackendCapabilities, BackendChoice, BatchIter, CancelHandle, DatabaseBackend, DatabaseSession,
    PythonDriverBridge,
};
use crate::database::connection::{
    ConnectionSpec, ConnectionTarget, password_secrets, percent_decode, scrub_secrets,
    secret_userinfo_end, userinfo_end,
};
use crate::database::dialect::DialectName;

/// An initialised driver. Never unloaded.
pub struct LoadedDriver {
    /// Boxed so its address is stable: handles point at it through `private_driver`.
    table: Box<AdbcDriver>,
    /// The ADBC API version the driver was initialised with.
    api_version: c_int,
    library: PathBuf,
    entrypoint: String,
}

// SAFETY: the function table is immutable after init, and ADBC drivers must be usable from any
// thread (handles are not shared between threads, except for cancel).
unsafe impl Send for LoadedDriver {}
unsafe impl Sync for LoadedDriver {}

impl LoadedDriver {
    pub fn library(&self) -> &Path {
        &self.library
    }

    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    /// Whether the driver implements ADBC 1.1 (cancel, option getters, …).
    pub fn is_1_1(&self) -> bool {
        self.api_version >= ADBC_VERSION_1_1_0
    }

    fn table_ptr(&self) -> *mut AdbcDriver {
        &*self.table as *const AdbcDriver as *mut AdbcDriver
    }

    fn check(&self, status: AdbcStatusCode, error: &AdbcError, what: &str) -> PolarsResult<()> {
        if status == ADBC_STATUS_OK {
            return Ok(());
        }
        let message = error
            .describe()
            .unwrap_or_else(|| "no error message".into());
        Err(polars_err!(
            ComputeError: "ADBC {what} failed ({}) in '{}': {message}",
            status_name(status), self.library.display()
        ))
    }
}

macro_rules! driver_fn {
    ($driver:expr, $name:ident) => {
        $driver.table.$name.ok_or_else(|| {
            polars_err!(
                ComputeError: "ADBC driver '{}' does not implement {}",
                $driver.library.display(), stringify!($name)
            )
        })?
    };
}

/// Loaded drivers by (library, manifest entrypoint).
type DriverCache = polars_utils::aliases::PlHashMap<(PathBuf, Option<String>), Arc<LoadedDriver>>;

static LOADED_DRIVERS: LazyLock<Mutex<DriverCache>> = LazyLock::new(Default::default);

/// The init symbol derived from a library file name, as the ADBC driver manager does it:
/// `libadbc_driver_sqlite.so.1.0` → `AdbcDriverSqliteInit`, `proprietary_driver.dll` →
/// `AdbcProprietaryDriverInit`.
pub fn default_entrypoint(library: &str) -> String {
    let filename = library.rsplit(['/', '\\']).next().unwrap_or(library);
    let stem = filename.split('.').next().unwrap_or(filename);
    let stem = stem.strip_prefix("lib").unwrap_or(stem);
    let mut out: String = stem
        .split(['-', '_'])
        .filter(|t| !t.is_empty())
        .map(|t| {
            let mut c = t.chars();
            let first = c.next().unwrap().to_ascii_uppercase();
            std::iter::once(first).chain(c).collect::<String>()
        })
        .collect();
    if !out.starts_with("Adbc") {
        out.insert_str(0, "Adbc");
    }
    out.push_str("Init");
    out
}

/// Loads and initialises a driver library, or returns the one already loaded.
pub fn load_driver(found: &FoundDriver) -> PolarsResult<Arc<LoadedDriver>> {
    let key_path = found
        .library
        .canonicalize()
        .unwrap_or_else(|_| found.library.clone());
    let key = (key_path, found.entrypoint.clone());
    let mut cache = LOADED_DRIVERS.lock().unwrap();
    if let Some(d) = cache.get(&key) {
        return Ok(d.clone());
    }
    let driver = Arc::new(load_driver_uncached(found)?);
    cache.insert(key, driver.clone());
    Ok(driver)
}

fn load_driver_uncached(found: &FoundDriver) -> PolarsResult<LoadedDriver> {
    let path = &found.library;
    // SAFETY: loading a library runs its initialisers; the user asked for this driver.
    let lib = unsafe { libloading::Library::new(path.as_path()) }.map_err(|e| {
        polars_err!(ComputeError: "could not load ADBC driver library '{}': {e}", path.display())
    })?;

    let mut candidates = vec![];
    candidates.extend(found.entrypoint.clone());
    candidates.push(default_entrypoint(&path.to_string_lossy()));
    candidates.push("AdbcDriverInit".to_string());
    candidates.dedup();

    let mut init: Option<(AdbcDriverInitFunc, String)> = None;
    for name in &candidates {
        // SAFETY: the symbol is an `AdbcDriverInitFunc` by the ADBC convention.
        if let Ok(sym) = unsafe { lib.get::<AdbcDriverInitFunc>(name.as_str()) } {
            init = Some((*sym, name.clone()));
            break;
        }
    }
    let Some((init, entrypoint)) = init else {
        polars_bail!(
            ComputeError: "ADBC driver library '{}' exports none of the entrypoints {:?}",
            path.display(), candidates
        )
    };

    let mut api_version = ADBC_VERSION_1_1_0;
    let mut table = Box::new(AdbcDriver::empty());
    let mut error = AdbcError::new();
    let mut status = unsafe {
        init(
            api_version,
            &mut *table as *mut AdbcDriver as *mut c_void,
            &mut error,
        )
    };
    if status == ADBC_STATUS_NOT_IMPLEMENTED {
        api_version = ADBC_VERSION_1_0_0;
        *table = AdbcDriver::empty();
        error.reset();
        status = unsafe {
            init(
                api_version,
                &mut *table as *mut AdbcDriver as *mut c_void,
                &mut error,
            )
        };
    }
    if status != ADBC_STATUS_OK {
        let message = error.describe().unwrap_or_default();
        polars_bail!(
            ComputeError: "ADBC driver '{}' failed to initialise ({}) via '{entrypoint}': {message}",
            path.display(), status_name(status)
        );
    }
    drop(error);

    // Never unloaded: Go drivers hang on unload, and other threads may still run driver code.
    std::mem::forget(lib);

    Ok(LoadedDriver {
        table,
        api_version,
        library: path.clone(),
        entrypoint,
    })
}

/// Which code loads and calls ADBC drivers, from `POLARS_DATABASE_ADBC_LOADER`.
///
/// Not part of the plan: the data comes from the driver, whichever code calls it, so each
/// machine that opens a connection may use its own setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdbcLoader {
    /// Polars' own C-ABI binding.
    Rust,
    /// Python's `adbc_driver_manager`.
    Python,
}

impl AdbcLoader {
    /// Read whenever a backend is instantiated, so it can be switched between scans.
    pub fn from_env() -> PolarsResult<Self> {
        match std::env::var("POLARS_DATABASE_ADBC_LOADER").as_deref() {
            Err(_) | Ok("" | "rust") => Ok(Self::Rust),
            Ok("python") => Ok(Self::Python),
            Ok(other) => polars_bail!(
                InvalidOperation:
                "POLARS_DATABASE_ADBC_LOADER must be 'rust' or 'python', got '{other}'"
            ),
        }
    }
}

impl std::fmt::Display for AdbcLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rust => "the Rust ADBC loader",
            Self::Python => "adbc_driver_manager (Python)",
        })
    }
}

enum Loader {
    Rust(Arc<LoadedDriver>),
    Python(Arc<dyn PythonDriverBridge>),
}

pub struct AdbcBackend {
    loader: Loader,
    found: FoundDriver,
    choice: BackendChoice,
}

impl AdbcBackend {
    /// `driver`: an ADBC driver name (`"sqlite"`, `"postgresql"`, …), or a path to a shared
    /// library or a `.toml` manifest.
    /// `location_hint`: a library or manifest path found during planning, possibly on another
    /// machine. Tried first when it exists here.
    pub fn load(
        driver: &str,
        location_hint: Option<&str>,
        choice: BackendChoice,
    ) -> PolarsResult<Self> {
        let found = location_hint
            .and_then(find_driver)
            .or_else(|| find_driver(driver));
        let Some(found) = found else {
            let dirs = manifest_search_dirs()
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let hint = location_hint
                .map(|h| format!("'{h}' (from planning) does not exist here; "))
                .unwrap_or_default();
            if driver.contains(['/', '\\']) || driver.ends_with(".toml") {
                polars_bail!(ComputeError: "ADBC driver library or manifest '{driver}' does not exist");
            }
            polars_bail!(
                ComputeError:
                "ADBC driver '{driver}' not found: {hint}looked for '{driver}.toml' in [{dirs}]. \
                Install it with: {}  (or pass the path of the driver library or manifest)",
                super::resolve::adbc_install_hint(driver)
            )
        };
        let which = AdbcLoader::from_env()?;
        let loader = match which {
            AdbcLoader::Rust => Loader::Rust(load_driver(&found)?),
            AdbcLoader::Python => {
                let Some(bridge) = super::python_bridge() else {
                    polars_bail!(
                        ComputeError:
                        "POLARS_DATABASE_ADBC_LOADER=python needs Python, which is not available \
                        on this machine"
                    )
                };
                Loader::Python(bridge.clone())
            },
        };
        if polars_core::config::verbose() {
            eprintln!(
                "[scan_database]: {} ('{}') is loaded through {which}",
                choice.kind,
                found.library.display()
            );
        }
        Ok(Self {
            loader,
            found,
            choice,
        })
    }

    pub fn found(&self) -> &FoundDriver {
        &self.found
    }

    /// Opens a connection, like [`DatabaseBackend::connect`].
    pub fn open(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        let options = connect_options(spec, dialect)?;
        let execute_options = spec
            .execute_options
            .iter()
            .map(|(k, v)| Ok((cstring(k, "option key")?, cstring(v, k)?)))
            .collect::<PolarsResult<Vec<_>>>()?;
        let session: PolarsResult<Box<dyn DatabaseSession>> = match &self.loader {
            Loader::Rust(driver) => Conn::open(driver.clone(), &options).map(|conn| {
                Box::new(AdbcSession {
                    conn: Arc::new(conn),
                    dialect,
                    execute_options,
                }) as _
            }),
            Loader::Python(bridge) => {
                bridge.adbc_connect(&self.found, &options, dialect, &execute_options)
            },
        };
        // Some drivers quote the URI or rejected options, credentials included.
        session.map_err(|e| {
            let secrets = secrets(&options);
            e.wrap_msg(|m| redact(m, &secrets))
        })
    }
}

/// The credentials among the options of a connection: password-like options, and the
/// password of the URI as written and decoded.
fn secrets(options: &[(CString, CString)]) -> Vec<String> {
    let mut out = vec![];
    for (k, v) in options {
        let (k, v) = (
            k.to_string_lossy().to_ascii_lowercase(),
            v.to_string_lossy(),
        );
        if ["password", "passwd", "pwd", "passphrase", "secret", "token"]
            .iter()
            .any(|s| k.contains(s))
        {
            out.push(v.into_owned());
        } else if k == ADBC_OPTION_URI.to_string_lossy()
            && let Some((_, rest)) = v.split_once("://")
            && let Some(at) = secret_userinfo_end(rest)
            && let Some((_, password)) = rest[..at].split_once(':')
        {
            out.extend(password_secrets(password));
            out.push(percent_decode(password));
        }
    }
    out.retain(|s| !s.is_empty());
    // Longest first, so that one secret containing another is replaced whole.
    out.sort_by_key(|s| std::cmp::Reverse(s.len()));
    out.dedup();
    out
}

fn redact(message: &str, secrets: &[String]) -> String {
    scrub_secrets(message, secrets)
}

/// The database options for a connection, `uri` first. The same for every loader.
pub fn connect_options(
    spec: &ConnectionSpec,
    dialect: DialectName,
) -> PolarsResult<Vec<(CString, CString)>> {
    let mut uri = adbc_uri(spec, dialect)?;
    // Redshift's ADBC driver isn't libpq.
    if dialect == DialectName::Postgresql
        && let Some(u) = &mut uri
        && !u.contains("connect_timeout=")
    {
        // libpq otherwise waits for the OS TCP timeout (minutes), and connects are retried.
        u.push(if u.contains('?') { '&' } else { '?' });
        u.push_str("connect_timeout=10");
    }
    let mut options: Vec<(CString, CString)> = vec![];
    if dialect == DialectName::Duckdb
        && let Some(u) = uri.take()
    {
        // The DuckDB driver reads `duckdb:///d.db` as `/d.db`: it gets the path, as the
        // Python backends read it, and the parameters as settings.
        for (k, v) in duckdb_options(&u) {
            options.push((cstring(&k, "option key")?, cstring(&v, &k)?));
        }
    }
    let mut creds = credentials(spec)?;
    if dialect == DialectName::Clickhouse
        && let Some(u) = &uri
        && let Some(http) = clickhouse_http_uri(u)
    {
        // The ClickHouse driver takes the HTTP endpoint, with credentials as options.
        uri = Some(http.uri);
        let (user, password) = creds.take().unwrap_or_default();
        for (key, value) in [
            (ADBC_OPTION_USERNAME, user.or(http.user)),
            (ADBC_OPTION_PASSWORD, password.or(http.password)),
        ] {
            if let Some(v) = value {
                options.push((key.to_owned(), cstring(&v, "credential")?));
            }
        }
    }
    if let Some((user, password)) = creds {
        // Drivers that take a `scheme://` URI (libpq, MySQL, …) read credentials from it
        // and may reject the `username`/`password` options.
        match uri
            .as_deref()
            .and_then(|u| with_userinfo(u, user.as_deref(), password.as_deref()))
        {
            Some(with_creds) => uri = Some(with_creds),
            None => {
                if let Some(u) = user {
                    options.push((ADBC_OPTION_USERNAME.to_owned(), cstring(&u, "credential")?));
                }
                if let Some(p) = password {
                    options.push((ADBC_OPTION_PASSWORD.to_owned(), cstring(&p, "credential")?));
                }
            },
        }
    }
    if let Some(uri) = uri {
        options.insert(0, (ADBC_OPTION_URI.to_owned(), cstring(&uri, "uri")?));
    }
    for (k, v) in &spec.connect_options {
        options.push((cstring(k, "option key")?, cstring(v, k)?));
    }
    Ok(options)
}

impl DatabaseBackend for AdbcBackend {
    fn choice(&self) -> &BackendChoice {
        &self.choice
    }

    fn capabilities(&self) -> BackendCapabilities {
        let mut caps = BackendCapabilities::ARROW_NATIVE
            | BackendCapabilities::STREAMING
            | BackendCapabilities::NO_GIL
            | BackendCapabilities::SESSION_SETUP;
        match &self.loader {
            Loader::Rust(driver) => {
                if driver.is_1_1() {
                    caps |= BackendCapabilities::CANCEL;
                }
            },
            // The driver manager answers a cancel on an ADBC 1.0 driver with an error.
            Loader::Python(_) => {
                caps |= BackendCapabilities::NEEDS_PYTHON | BackendCapabilities::CANCEL
            },
        }
        caps
    }

    fn connect(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        self.open(spec, dialect)
    }
}

fn cstring(s: &str, what: &str) -> PolarsResult<CString> {
    CString::new(s).map_err(|_| polars_err!(ComputeError: "ADBC {what} contains a NUL byte"))
}

/// The `uri` database option for a connection target.
fn adbc_uri(spec: &ConnectionSpec, dialect: DialectName) -> PolarsResult<Option<String>> {
    match &spec.target {
        ConnectionTarget::Uri(_) => {
            let uri = spec.uri_without_scheme_driver().unwrap_or_default();
            if dialect == DialectName::Sqlite {
                return Ok(Some(sqlite_path(&uri)?));
            }
            Ok(Some(uri))
        },
        ConnectionTarget::Odbc(_) => polars_bail!(
            InvalidOperation: "an ODBC connection string can't be used with an ADBC driver"
        ),
        #[allow(unreachable_patterns)]
        _ => polars_bail!(
            InvalidOperation: "a Python connection object can't be used with an ADBC driver"
        ),
    }
}

struct ClickhouseHttp {
    uri: String,
    user: Option<String>,
    password: Option<String>,
}

/// `clickhouse://user:pass@host:8123/db` → `http://host:8123/?database=db` and the
/// credentials; `clickhouses://` (or `?secure=true`) → `https://`. `http(s)://` passes through.
fn clickhouse_http_uri(uri: &str) -> Option<ClickhouseHttp> {
    let (scheme, rest) = uri.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme == "http" || scheme == "https" {
        return None;
    }
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, host) = match authority.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, authority),
    };
    let (user, password) = match userinfo {
        Some(u) => match u.split_once(':') {
            Some((u, p)) => (Some(percent_decode(u)), Some(percent_decode(p))),
            None => (Some(percent_decode(u)), None),
        },
        None => (None, None),
    };
    let secure = scheme.ends_with('s')
        || query
            .split('&')
            .any(|kv| kv.eq_ignore_ascii_case("secure=true") || kv == "secure=1");
    let mut params: Vec<String> = query
        .split('&')
        .filter(|kv| !kv.is_empty() && !kv.to_ascii_lowercase().starts_with("secure="))
        .map(str::to_string)
        .collect();
    let database = path.trim_matches('/');
    if !database.is_empty() {
        params.insert(0, format!("database={database}"));
    }
    let mut out = format!("{}://{host}", if secure { "https" } else { "http" });
    if !params.is_empty() {
        out.push_str("/?");
        out.push_str(&params.join("&"));
    }
    Some(ClickhouseHttp {
        uri: out,
        user,
        password,
    })
}

/// The path and the parameters of a `sqlite:` or `duckdb:` URI, read as SQLAlchemy (and the
/// Python backends, `_file_target`) read it: `sqlite:///rel.db` → `rel.db`,
/// `sqlite:////abs.db` → `/abs.db`, `sqlite://` → `:memory:`. `None` for a `file:` URI after
/// the scheme, or a string without the scheme, which pass through.
fn file_target<'a>(uri: &'a str, schemes: &[&str]) -> Option<(String, Vec<(&'a str, String)>)> {
    let lower = uri.to_ascii_lowercase();
    let prefix = schemes.iter().find(|p| lower.starts_with(*p))?;
    let mut rest = &uri[prefix.len()..];
    if let Some(r) = rest.strip_prefix("//") {
        // An authority (`sqlite://rel.db`, ConnectorX's form) is read as part of the path.
        rest = r.strip_prefix('/').unwrap_or(r);
    }
    if rest.starts_with("file:") {
        return None;
    }
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let mut path = match percent_decode(path) {
        p if matches!(p.as_str(), "" | ":memory:" | "/:memory:") => ":memory:".to_string(),
        p => p,
    };
    // `sqlite:////C:/path` → `C:/path`: on Windows, `/C:/path` is a network path.
    let b = path.as_bytes();
    if b.len() >= 4
        && b[0] == b'/'
        && b[1].is_ascii_alphabetic()
        && b[2] == b':'
        && matches!(b[3], b'/' | b'\\')
    {
        path.remove(0);
    }
    let params = query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (k, percent_decode(&v.replace('+', " ")))
        })
        .collect();
    Some((path, params))
}

/// SQLite URI filename parameters: they make the path a `file:` URI.
const SQLITE_URI_PARAMS: [&str; 7] = [
    "cache",
    "immutable",
    "mode",
    "modeof",
    "nolock",
    "psow",
    "vfs",
];

/// The file of a `sqlite:` URI ([`file_target`]). URI filename parameters (`?mode=ro`) make a
/// `file:` URI; others are an error, as the ADBC driver has no option for them.
fn sqlite_path(uri: &str) -> PolarsResult<String> {
    let Some((path, params)) = file_target(uri, &["sqlite3:", "sqlite:"]) else {
        let lower = uri.to_ascii_lowercase();
        return Ok(
            match ["sqlite3:", "sqlite:"]
                .iter()
                .find(|p| lower.starts_with(*p))
            {
                // `sqlite:///file:a.db?mode=ro`
                Some(p) => uri[p.len()..].trim_start_matches('/').to_string(),
                None => uri.to_string(),
            },
        );
    };
    let mut file_params = vec![];
    let mut unknown = vec![];
    for (k, v) in &params {
        if SQLITE_URI_PARAMS.contains(k) {
            file_params.push(format!("{k}={}", percent_encode(v)));
        } else if !matches!(*k, "uri" | "check_same_thread") {
            // `uri=true` is implied by the file parameters (SQLAlchemy's form); the thread
            // check is a Python `sqlite3` option.
            unknown.push(*k);
        }
    }
    polars_ensure!(
        unknown.is_empty(),
        InvalidOperation:
        "the SQLite ADBC driver has no option for URI parameter(s) {unknown:?}; \
        pin another driver with driver=\"sqlite3\""
    );
    if file_params.is_empty() {
        return Ok(path);
    }
    let mut file = String::from("file:");
    for c in path.chars() {
        match c {
            '%' => file.push_str("%25"),
            '?' => file.push_str("%3F"),
            '#' => file.push_str("%23"),
            c => file.push(c),
        }
    }
    Ok(format!("{file}?{}", file_params.join("&")))
}

/// The DuckDB driver's database options for a `duckdb:` URI: `path`, and the URI parameters as
/// DuckDB settings (`read_only=true` is `access_mode=READ_ONLY`). DuckDB rejects an unknown
/// setting itself.
fn duckdb_options(uri: &str) -> Vec<(String, String)> {
    let Some((path, params)) = file_target(uri, &["duckdb:"]) else {
        return vec![("path".into(), uri.to_string())];
    };
    let mut out = vec![("path".to_string(), path)];
    for (k, v) in params {
        if k == "read_only" {
            let read_only = matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
            out.push((
                "access_mode".into(),
                if read_only { "READ_ONLY" } else { "READ_WRITE" }.into(),
            ));
        } else {
            out.push((k.to_string(), v));
        }
    }
    out
}

/// `username` / `password` database options from the connection's credential provider.
fn credentials(spec: &ConnectionSpec) -> PolarsResult<Option<(Option<String>, Option<String>)>> {
    #[cfg(feature = "python")]
    if spec.credential_provider.is_some() {
        let Some(bridge) = super::python_bridge() else {
            polars_bail!(ComputeError: "a credential provider needs Python, which is not available");
        };
        return bridge.credentials(spec);
    }
    let _ = spec;
    Ok(None)
}

/// `scheme://user:password@host/…` with the given credentials, replacing those in the URI.
/// `None` if the URI has no authority part.
fn with_userinfo(uri: &str, user: Option<&str>, password: Option<&str>) -> Option<String> {
    let (scheme, rest) = uri.split_once("://")?;
    // `host_and_tail`: the host, and the path and query after it.
    let (old_userinfo, host_and_tail) = match userinfo_end(rest) {
        Some(at) => (Some(&rest[..at]), &rest[at + 1..]),
        None => (None, rest),
    };
    let (old_user, old_password) = match old_userinfo.map(|u| u.split_once(':')) {
        Some(Some((u, p))) => (Some(u), Some(p)),
        Some(None) => (old_userinfo, None),
        None => (None, None),
    };
    let user = match user {
        Some(u) => Some(percent_encode(u)),
        None => old_user.map(str::to_string),
    };
    let mut userinfo = user.unwrap_or_default();
    // A provider that gives only the user keeps the URI's password.
    if let Some(p) = password
        .map(percent_encode)
        .or(old_password.map(str::to_string))
    {
        userinfo.push(':');
        userinfo.push_str(&p);
    }
    let at = if userinfo.is_empty() { "" } else { "@" };
    Some(format!("{scheme}://{userinfo}{at}{host_and_tail}"))
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A database and its one connection. Shared by the session, its result streams and its
/// cancel handle, so it lives until all of them are gone.
struct Conn {
    driver: Arc<LoadedDriver>,
    database: *mut AdbcDatabase,
    connection: *mut AdbcConnection,
    /// The statement currently executing, for cancel.
    active: Mutex<Option<StmtPtr>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StmtPtr(*mut AdbcStatement);
unsafe impl Send for StmtPtr {}

// SAFETY: the connection is used by one thread at a time (the session is `&mut`, and result
// streams are consumed in order); ADBC allows cancel from another thread.
unsafe impl Send for Conn {}
unsafe impl Sync for Conn {}

impl Conn {
    fn open(driver: Arc<LoadedDriver>, options: &[(CString, CString)]) -> PolarsResult<Self> {
        let d = &*driver;
        let mut error = AdbcError::new();

        let database = Box::into_raw(Box::new(AdbcDatabase::empty()));
        let connection = Box::into_raw(Box::new(AdbcConnection::empty()));
        // From here on, `Drop` cleans up whatever was initialised.
        let conn = Conn {
            driver: driver.clone(),
            database,
            connection,
            active: Mutex::new(None),
        };

        unsafe {
            let status = driver_fn!(d, DatabaseNew)(database, &mut error);
            d.check(status, &error, "DatabaseNew")?;
            (*database).private_driver = d.table_ptr();
            let set = driver_fn!(d, DatabaseSetOption);
            for (k, v) in options {
                error.reset();
                let status = set(database, k.as_ptr(), v.as_ptr(), &mut error);
                d.check(
                    status,
                    &error,
                    &format!("setting database option '{}'", k.to_string_lossy()),
                )?;
            }
            error.reset();
            let status = driver_fn!(d, DatabaseInit)(database, &mut error);
            d.check(status, &error, "DatabaseInit")?;

            error.reset();
            let status = driver_fn!(d, ConnectionNew)(connection, &mut error);
            d.check(status, &error, "ConnectionNew")?;
            (*connection).private_driver = d.table_ptr();
            error.reset();
            let status = driver_fn!(d, ConnectionInit)(connection, database, &mut error);
            d.check(status, &error, "ConnectionInit")?;
        }
        Ok(conn)
    }

    fn cancel(&self) -> PolarsResult<()> {
        let d = &*self.driver;
        let active = self.active.lock().unwrap();
        let mut error = AdbcError::new();
        let mut status = ADBC_STATUS_NOT_IMPLEMENTED;
        unsafe {
            if let (Some(stmt), Some(cancel)) = (*active, d.table.StatementCancel) {
                status = cancel(stmt.0, &mut error);
            }
            if status == ADBC_STATUS_NOT_IMPLEMENTED
                && let Some(cancel) = d.table.ConnectionCancel
            {
                error.reset();
                status = cancel(self.connection, &mut error);
            }
        }
        // Nothing running: nothing to cancel.
        if status == ADBC_STATUS_INVALID_STATE {
            return Ok(());
        }
        d.check(status, &error, "cancel")
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        let t = &self.driver.table;
        let mut error = AdbcError::new();
        unsafe {
            if !(*self.connection).private_data.is_null()
                && let Some(release) = t.ConnectionRelease
            {
                release(self.connection, &mut error);
                error.reset();
            }
            if !(*self.database).private_data.is_null()
                && let Some(release) = t.DatabaseRelease
            {
                release(self.database, &mut error);
            }
            drop(Box::from_raw(self.connection));
            drop(Box::from_raw(self.database));
        }
    }
}

/// An open statement. Released on drop.
struct Stmt {
    conn: Arc<Conn>,
    ptr: *mut AdbcStatement,
}

unsafe impl Send for Stmt {}

impl Stmt {
    fn new(conn: Arc<Conn>) -> PolarsResult<Self> {
        let d = &*conn.driver;
        let ptr = Box::into_raw(Box::new(AdbcStatement::empty()));
        let stmt = Stmt {
            conn: conn.clone(),
            ptr,
        };
        let mut error = AdbcError::new();
        unsafe {
            let status = driver_fn!(d, StatementNew)(conn.connection, ptr, &mut error);
            d.check(status, &error, "StatementNew")?;
            (*ptr).private_driver = d.table_ptr();
        }
        Ok(stmt)
    }

    fn driver(&self) -> &LoadedDriver {
        &self.conn.driver
    }

    fn set_option(&self, key: &CStr, value: &CStr) -> PolarsResult<()> {
        let d = self.driver();
        let mut error = AdbcError::new();
        let status = unsafe {
            driver_fn!(d, StatementSetOption)(self.ptr, key.as_ptr(), value.as_ptr(), &mut error)
        };
        d.check(
            status,
            &error,
            &format!("setting statement option '{}'", key.to_string_lossy()),
        )
    }

    fn set_sql(&self, sql: &str) -> PolarsResult<()> {
        let d = self.driver();
        let sql = cstring(sql, "query")?;
        let mut error = AdbcError::new();
        let status =
            unsafe { driver_fn!(d, StatementSetSqlQuery)(self.ptr, sql.as_ptr(), &mut error) };
        d.check(status, &error, "StatementSetSqlQuery")
    }

    /// Runs the statement. With `out` null, no result set is expected.
    fn execute(&self, out: *mut ArrowArrayStream) -> PolarsResult<()> {
        let d = self.driver();
        let execute = driver_fn!(d, StatementExecuteQuery);
        *self.conn.active.lock().unwrap() = Some(StmtPtr(self.ptr));
        let mut error = AdbcError::new();
        let mut rows_affected: i64 = -1;
        let status = unsafe { execute(self.ptr, out, &mut rows_affected, &mut error) };
        if out.is_null() {
            *self.conn.active.lock().unwrap() = None;
        }
        d.check(status, &error, "query")
    }
}

impl Drop for Stmt {
    fn drop(&mut self) {
        {
            let mut active = self.conn.active.lock().unwrap();
            if *active == Some(StmtPtr(self.ptr)) {
                *active = None;
            }
        }
        unsafe {
            if !(*self.ptr).private_data.is_null()
                && let Some(release) = self.conn.driver.table.StatementRelease
            {
                let mut error = AdbcError::new();
                release(self.ptr, &mut error);
            }
            drop(Box::from_raw(self.ptr));
        }
    }
}

/// A connection to a database through an ADBC driver.
pub struct AdbcSession {
    conn: Arc<Conn>,
    dialect: DialectName,
    execute_options: Vec<(CString, CString)>,
}

impl AdbcSession {
    fn prepare(&self, sql: &str, batch_size: Option<usize>) -> PolarsResult<Stmt> {
        let stmt = Stmt::new(self.conn.clone())?;
        if let Some(n) = batch_size
            && let Some((key, value)) = batch_size_option(self.dialect, n)
        {
            // A hint: drivers that reject it still work.
            let _ = stmt.set_option(key, &CString::new(value).unwrap());
        }
        for (k, v) in &self.execute_options {
            stmt.set_option(k, v)?;
        }
        stmt.set_sql(sql)?;
        Ok(stmt)
    }
}

/// The statement option that bounds the batch size, for drivers known to have one. A hint:
/// drivers that reject it still work.
pub fn batch_size_option(dialect: DialectName, rows: usize) -> Option<(&'static CStr, String)> {
    match dialect {
        DialectName::Sqlite => Some((c"adbc.sqlite.query.batch_rows", rows.to_string())),
        // A byte hint: assume narrow rows, so a batch never has far more rows than asked.
        DialectName::Postgresql | DialectName::Redshift => Some((
            c"adbc.postgresql.batch_size_hint_bytes",
            rows.saturating_mul(64).max(1024).to_string(),
        )),
        _ => None,
    }
}

impl DatabaseSession for AdbcSession {
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter> {
        let stmt = self.prepare(sql, batch_size)?;
        let stream = Box::into_raw(Box::new(ArrowArrayStream::empty()));
        let result = stmt.execute(stream);
        // SAFETY: allocated above; the driver only wrote into it.
        let stream = unsafe { Box::from_raw(stream) };
        result?;
        // SAFETY: the driver returned a valid C stream.
        let batches = unsafe { import_stream(stream, stmt) }?;
        Ok(check_batches(self.dialect, batches))
    }

    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        let stmt = self.prepare(sql, None)?;
        stmt.execute(ptr::null_mut())
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        if !self.conn.driver.is_1_1() {
            return None;
        }
        let conn = self.conn.clone();
        Some(Arc::new(move || conn.cancel()))
    }
}

/// What a result stream needs kept alive until it is released: its statement, and through
/// it the connection.
pub trait StreamOwner: Send + 'static {
    /// The driver's error for a failed stream operation, when it carries more than the
    /// stream's own `get_last_error`.
    fn stream_error(&self, _stream: *mut ArrowArrayStream) -> Option<polars_error::PolarsError> {
        None
    }
}

impl StreamOwner for Stmt {
    fn stream_error(&self, stream: *mut ArrowArrayStream) -> Option<polars_error::PolarsError> {
        stream_error(self, stream)
    }
}

/// The batches of a query's result stream, as `DataFrame`s. `owner` is dropped after the
/// stream is released.
///
/// # Safety
/// `stream` must be a valid Arrow C stream, usable from any thread.
pub unsafe fn import_stream(
    stream: Box<ArrowArrayStream>,
    owner: impl StreamOwner,
) -> PolarsResult<BatchIter> {
    // The heap address stays valid while the reader owns the box.
    let stream_ptr = &*stream as *const ArrowArrayStream as *mut ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::try_new(stream) }?;
    // Before any array is imported or built: those of these types would panic.
    check_arrow_types(&reader.field().dtype)?;
    Ok(Box::new(AdbcBatches {
        reader: Some(reader),
        stream_ptr,
        owner,
        yielded: false,
    }))
}

/// The driver's error for a failed stream operation (ADBC 1.1), which carries more than the
/// stream's own `get_last_error`.
fn stream_error(stmt: &Stmt, stream: *mut ArrowArrayStream) -> Option<polars_error::PolarsError> {
    let d = stmt.driver();
    let f = d.table.ErrorFromArrayStream?;
    let mut status = ADBC_STATUS_OK;
    let error = unsafe { f(stream, &mut status) };
    if error.is_null() {
        return None;
    }
    // Owned by the stream: borrowed, never released here.
    let message = unsafe { &*error }.describe()?;
    Some(polars_err!(
        ComputeError: "ADBC query failed ({}) in '{}': {message}",
        status_name(status), d.library.display()
    ))
}

/// The batches of one query, as `DataFrame`s.
struct AdbcBatches<O> {
    // Field order is drop order: the stream is released before its statement.
    reader: Option<ArrowArrayStreamReader<Box<ArrowArrayStream>>>,
    stream_ptr: *mut ArrowArrayStream,
    owner: O,
    yielded: bool,
}

// SAFETY: the stream and statement are only used from the thread that pulls the iterator.
unsafe impl<O: StreamOwner> Send for AdbcBatches<O> {}

impl<O: StreamOwner> Iterator for AdbcBatches<O> {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = self.reader.as_mut()?;
            // SAFETY: the driver produces arrays matching its schema.
            match unsafe { reader.next() } {
                Some(Ok(array)) => {
                    if array.is_empty() {
                        continue;
                    }
                    self.yielded = true;
                    return Some(batch_to_df(array));
                },
                Some(Err(e)) => {
                    let err = self.owner.stream_error(self.stream_ptr).unwrap_or(e);
                    self.reader = None;
                    return Some(Err(err));
                },
                None => {
                    let reader = self.reader.take()?;
                    if self.yielded {
                        return None;
                    }
                    // Empty result: one empty frame with the result's columns.
                    self.yielded = true;
                    let dtype = reader.field().dtype.clone();
                    return Some(batch_to_df(Box::new(StructArray::new_empty(dtype))));
                },
            }
        }
    }
}

/// Checks the batches of a result for values the driver could not convert.
pub fn check_batches(dialect: DialectName, batches: BatchIter) -> BatchIter {
    if dialect != DialectName::Postgresql {
        return batches;
    }
    Box::new(batches.map(|df| df.and_then(|df| check_postgres_infinity(&df).map(|_| df))))
}

/// The earliest Postgres date and timestamp, 4714-11-24 BC (Julian day 0), from the Unix epoch.
const PG_MIN_DAYS: i32 = -2_440_588;
const PG_MIN_US: i64 = PG_MIN_DAYS as i64 * 86_400_000_000;

/// An error for a Postgres date or timestamp the ADBC Postgres driver could not convert.
///
/// The driver shifts the server's values (µs or days since 2000) to the Unix epoch, and the
/// shift wraps for `infinity` / `-infinity` (`i64::MAX` / `i64::MIN`, `i32::MAX` / `i32::MIN`)
/// and for timestamps after about year 294 000. Wrapped, they read as values before the
/// earliest Postgres value, so anything earlier than that is an error, also nested in lists
/// and structs.
fn check_postgres_infinity(df: &DataFrame) -> PolarsResult<()> {
    for c in df.columns() {
        if wraps_postgres_temporal(c.as_materialized_series())? {
            polars_bail!(
                InvalidOperation:
                "column '{}' holds a Postgres 'infinity', '-infinity' or a timestamp after year \
                294 000, which Polars can't represent; filter it out or replace it in a query, \
                e.g. CASE WHEN isfinite(\"{}\") THEN \"{}\" END",
                c.name(), c.name(), c.name()
            );
        }
    }
    Ok(())
}

fn wraps_postgres_temporal(s: &polars_core::series::Series) -> PolarsResult<bool> {
    use polars_core::prelude::{ChunkAgg, DataType, TimeUnit};

    Ok(match s.dtype() {
        DataType::Datetime(TimeUnit::Microseconds, _) => s
            .to_physical_repr()
            .i64()?
            .min()
            .is_some_and(|v| v < PG_MIN_US),
        DataType::Date => s
            .to_physical_repr()
            .i32()?
            .min()
            .is_some_and(|v| v < PG_MIN_DAYS),
        DataType::List(inner) if has_temporal(inner) => {
            wraps_postgres_temporal(&s.list()?.get_inner())?
        },
        DataType::Struct(_) => {
            for f in s.struct_()?.fields_as_series() {
                if wraps_postgres_temporal(&f)? {
                    return Ok(true);
                }
            }
            false
        },
        _ => false,
    })
}

fn has_temporal(dtype: &polars_core::prelude::DataType) -> bool {
    use polars_core::prelude::DataType;
    match dtype {
        DataType::Date | DataType::Datetime(_, _) => true,
        DataType::List(inner) => has_temporal(inner),
        DataType::Struct(_) => true,
        _ => false,
    }
}

/// Arrow types Polars has no dtype for (importing them would panic).
fn unsupported_arrow_type(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Interval(_) | ArrowDataType::Union(_) | ArrowDataType::Decimal256(_, _) => {
            true
        },
        ArrowDataType::List(f)
        | ArrowDataType::LargeList(f)
        | ArrowDataType::FixedSizeList(f, _)
        | ArrowDataType::Map(f, _) => unsupported_arrow_type(f.dtype()),
        ArrowDataType::Struct(fields) => fields.iter().any(|f| unsupported_arrow_type(f.dtype())),
        ArrowDataType::Extension(ext) => unsupported_arrow_type(&ext.inner),
        _ => false,
    }
}

/// An error for a result column of a type Polars has no dtype for.
fn check_arrow_types(dtype: &ArrowDataType) -> PolarsResult<()> {
    let ArrowDataType::Struct(fields) = dtype else {
        polars_bail!(ComputeError: "ADBC driver returned a non-struct batch: {dtype:?}");
    };
    if let Some(f) = fields.iter().find(|f| unsupported_arrow_type(f.dtype())) {
        polars_bail!(
            InvalidOperation:
            "column '{}' has Arrow type {:?}, which Polars can't read; convert it in a query, \
            e.g. CAST(\"{}\" AS TEXT)",
            f.name, f.dtype(), f.name
        );
    }
    Ok(())
}

fn batch_to_df(array: Box<dyn Array>) -> PolarsResult<DataFrame> {
    check_arrow_types(array.dtype())?;
    // A record batch has no top-level nulls; drop a validity some drivers still attach.
    let array = array.with_validity(None);
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct dtype")
        .clone();
    DataFrame::try_from(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_credentials() {
        let opts = |kv: &[(&str, &str)]| {
            kv.iter()
                .map(|(k, v)| (CString::new(*k).unwrap(), CString::new(*v).unwrap()))
                .collect::<Vec<_>>()
        };
        let s = secrets(&opts(&[
            ("uri", "mysql://u:p%40ss@w/rd@h:3306/db?x=1"),
            ("password", "hunter2"),
            ("adbc.snowflake.sql.client_option.auth_token", "tok3n"),
            ("username", "u"),
        ]));
        let m = redact(
            "bad uri mysql://u:p%40ss@w/rd@h:3306/db: password='hunter2' p@ss@w/rd tok3n u",
            &s,
        );
        assert_eq!(
            m,
            "bad uri mysql://u:***@h:3306/db: password='***' *** *** u"
        );
        assert!(secrets(&opts(&[("uri", "postgresql://u@h/db")])).is_empty());
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("p%40ss%2F"), "p@ss/");
        assert_eq!(percent_decode("%41"), "A");
        // not a hex escape: kept as is, also before a multi-byte character
        assert_eq!(percent_decode("%aé"), "%aé");
        assert_eq!(percent_decode("%é1"), "%é1");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%4"), "%4");
    }

    #[test]
    fn postgres_infinity() {
        use polars_core::prelude::*;

        // what the ADBC Postgres driver reads for ±infinity
        let ts = |v: i64| {
            Series::new("t".into(), [Some(0i64), Some(v)])
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap()
        };
        let date = |v: i32| {
            Series::new("d".into(), [Some(0i32), Some(v)])
                .cast(&DataType::Date)
                .unwrap()
        };
        let df = |s: Series| DataFrame::new(s.len(), vec![s.into()]).unwrap();
        for v in [-9222425352054775809, -9222425352054775808] {
            assert!(check_postgres_infinity(&df(ts(v))).is_err());
        }
        for v in [-2147472692, -2147472691] {
            assert!(check_postgres_infinity(&df(date(v))).is_err());
        }
        // the earliest Postgres values are fine
        assert!(check_postgres_infinity(&df(ts(PG_MIN_US))).is_ok());
        assert!(check_postgres_infinity(&df(date(PG_MIN_DAYS))).is_ok());
        // nested
        let list = ts(-9222425352054775809).implode().unwrap().into_series();
        assert!(check_postgres_infinity(&df(list)).is_err());
    }

    #[test]
    fn entrypoints() {
        assert_eq!(
            default_entrypoint("/x/libadbc_driver_sqlite.so"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("libadbc_driver_sqlite.so.2.0.0"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("C:\\d\\adbc_driver_sqlite.dll"),
            "AdbcDriverSqliteInit"
        );
        assert_eq!(
            default_entrypoint("proprietary_driver.dll"),
            "AdbcProprietaryDriverInit"
        );
        assert_eq!(
            default_entrypoint("libadbc-driver-flight-sql.dylib"),
            "AdbcDriverFlightSqlInit"
        );
    }

    /// Reads through ADBC with no Python in the process: the Rust-only path workers use.
    /// Set `POLARS_TEST_ADBC_SQLITE_LIBRARY` (and for the servers
    /// `POLARS_TEST_ADBC_POSTGRES_LIBRARY` + `POLARS_TEST_POSTGRES_URI`,
    /// `POLARS_TEST_ADBC_MYSQL_LIBRARY` + `POLARS_TEST_MYSQL_URI`) to run it.
    #[test]
    fn reads_without_python() {
        use polars_core::prelude::DataType;

        use crate::database::backend::{BackendChoice, BackendKind};

        let read = |lib_var: &str, uri: &str, dialect: DialectName, sql: &str| {
            let Ok(lib) = std::env::var(lib_var) else {
                return None;
            };
            let choice = BackendChoice {
                kind: BackendKind::Adbc {
                    driver: lib.as_str().into(),
                },
                version: None,
                location: None,
                reason: "test".into(),
            };
            let backend = AdbcBackend::load(&lib, None, choice).unwrap();
            let spec = ConnectionSpec::new(ConnectionTarget::Uri(uri.into()));
            let mut session = backend.open(&spec, dialect).unwrap();
            Some(session.query(sql).unwrap())
        };

        if let Some(df) = read(
            "POLARS_TEST_ADBC_SQLITE_LIBRARY",
            "sqlite://",
            DialectName::Sqlite,
            "SELECT 1 AS a, 'x' AS b, 1.5 AS c, NULL AS d",
        ) {
            assert_eq!(df.shape(), (1, 4));
            assert_eq!(df.column("a").unwrap().dtype(), &DataType::Int64);
            assert_eq!(df.column("b").unwrap().dtype(), &DataType::String);
        }
        // Every column of `edge_types` (JSON, UUID, binary, NUMERIC, tz, time) imports.
        for (lib, uri, dialect) in [
            (
                "POLARS_TEST_ADBC_POSTGRES_LIBRARY",
                "POLARS_TEST_POSTGRES_URI",
                DialectName::Postgresql,
            ),
            (
                "POLARS_TEST_ADBC_MYSQL_LIBRARY",
                "POLARS_TEST_MYSQL_URI",
                DialectName::Mysql,
            ),
        ] {
            let Ok(uri) = std::env::var(uri) else {
                continue;
            };
            let edge = dialect.quote_ident("edge_types");
            if let Some(df) = read(lib, &uri, dialect, &format!("SELECT * FROM {edge}")) {
                assert_eq!(df.shape(), (3, 7), "{dialect}");
                for c in df.columns().iter().filter(|c| c.name() != "id") {
                    assert!(
                        c.null_count() >= 1,
                        "{dialect}: {} has no NULL row",
                        c.name()
                    );
                }
            }
        }
    }

    #[test]
    fn clickhouse_uris() {
        let h = clickhouse_http_uri("clickhouse://u:p%40ss@h:8123/db").unwrap();
        assert_eq!(h.uri, "http://h:8123/?database=db");
        assert_eq!(h.user.as_deref(), Some("u"));
        assert_eq!(h.password.as_deref(), Some("p@ss"));
        let h = clickhouse_http_uri("clickhouse://h:8443?secure=true").unwrap();
        assert_eq!(h.uri, "https://h:8443");
        assert!(clickhouse_http_uri("http://h:8123").is_none());
    }

    #[test]
    fn userinfo() {
        assert_eq!(
            with_userinfo("postgresql://u@h:5432/db", None, Some("p@ss:w/rd")).as_deref(),
            Some("postgresql://u:p%40ss%3Aw%2Frd@h:5432/db")
        );
        assert_eq!(
            with_userinfo("postgresql://old:x@h/db?a=1", Some("new"), Some("y")).as_deref(),
            Some("postgresql://new:y@h/db?a=1")
        );
        assert_eq!(
            with_userinfo("mysql://h", Some("u"), None).as_deref(),
            Some("mysql://u@h")
        );
        assert_eq!(with_userinfo("/tmp/a.db", Some("u"), None), None);
        // An `@` in a parameter isn't the end of the userinfo.
        assert_eq!(
            with_userinfo(
                "postgresql://polars:pw@127.0.0.1:5432/polars?application_name=a@b",
                Some("u"),
                None
            )
            .as_deref(),
            Some("postgresql://u:pw@127.0.0.1:5432/polars?application_name=a@b")
        );
        assert_eq!(
            with_userinfo("postgresql://u:12#Sekr3t@h:5432/db", None, Some("x")).as_deref(),
            Some("postgresql://u:x@h:5432/db")
        );
        // A path with an `@` isn't a userinfo.
        assert_eq!(
            with_userinfo("postgresql://localhost:5432/my@db", Some("u"), None).as_deref(),
            Some("postgresql://u@localhost:5432/my@db")
        );
        // Only a user from the provider: the URI's password stays.
        assert_eq!(
            with_userinfo("postgresql://old:p%40ss@h/db", Some("new"), None).as_deref(),
            Some("postgresql://new:p%40ss@h/db")
        );
        assert_eq!(
            with_userinfo("postgresql://old:pa#ss@h/db", Some("new"), Some("y")).as_deref(),
            Some("postgresql://new:y@h/db")
        );
    }

    #[test]
    fn sqlite_paths() {
        let sqlite_path = |uri: &str| super::sqlite_path(uri).unwrap();
        // As SQLAlchemy reads them: three slashes are relative, four absolute.
        assert_eq!(sqlite_path("sqlite:///tmp/a.db"), "tmp/a.db");
        assert_eq!(sqlite_path("sqlite:////tmp/a.db"), "/tmp/a.db");
        assert_eq!(sqlite_path("sqlite:rel.db"), "rel.db");
        assert_eq!(sqlite_path("sqlite:///a%20b.db"), "a b.db");
        assert_eq!(sqlite_path("sqlite://"), ":memory:");
        assert_eq!(sqlite_path("sqlite:///:memory:"), ":memory:");
        assert_eq!(sqlite_path("file:a.db?mode=ro"), "file:a.db?mode=ro");
        assert_eq!(
            sqlite_path("sqlite:///file:a.db?mode=ro"),
            "file:a.db?mode=ro"
        );
        assert_eq!(
            sqlite_path("sqlite:////t/a%3Fb.db?mode=ro"),
            "file:/t/a%3Fb.db?mode=ro"
        );
        assert_eq!(sqlite_path("sqlite:////C:/t/a.db"), "C:/t/a.db");
        // As the Python `sqlite3` backend: SQLAlchemy's `uri=true` is implied.
        assert_eq!(
            sqlite_path("sqlite:////t/a.db?mode=ro&uri=true"),
            "file:/t/a.db?mode=ro"
        );
        assert!(super::sqlite_path("sqlite:////t/a.db?timeout=5").is_err());
    }

    #[test]
    fn duckdb_paths() {
        let opts = |uri: &str| duckdb_options(uri);
        assert_eq!(
            opts("duckdb:///d.db"),
            [("path".to_string(), "d.db".to_string())]
        );
        assert_eq!(
            opts("duckdb:////t/d.db?read_only=true&threads=2"),
            [
                ("path".to_string(), "/t/d.db".to_string()),
                ("access_mode".to_string(), "READ_ONLY".to_string()),
                ("threads".to_string(), "2".to_string()),
            ]
        );
        assert_eq!(
            opts("duckdb://"),
            [("path".to_string(), ":memory:".to_string())]
        );
    }
}
