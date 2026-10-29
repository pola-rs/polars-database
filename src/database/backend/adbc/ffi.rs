//! The ADBC C ABI (`adbc.h`, API version 1.1.0), declared by hand. Only the types are declared
//! here; the driver's functions are reached through the [`AdbcDriver`] function table filled in
//! by the driver's init function.
//!
//! The Arrow C data and stream structs are Polars' own (`polars_arrow::ffi`), which have the
//! layout the C data interface prescribes.
#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use polars_arrow::ffi::{ArrowArray, ArrowArrayStream, ArrowSchema};

pub type AdbcStatusCode = u8;

pub const ADBC_STATUS_OK: AdbcStatusCode = 0;
pub const ADBC_STATUS_UNKNOWN: AdbcStatusCode = 1;
pub const ADBC_STATUS_NOT_IMPLEMENTED: AdbcStatusCode = 2;
pub const ADBC_STATUS_NOT_FOUND: AdbcStatusCode = 3;
pub const ADBC_STATUS_ALREADY_EXISTS: AdbcStatusCode = 4;
pub const ADBC_STATUS_INVALID_ARGUMENT: AdbcStatusCode = 5;
pub const ADBC_STATUS_INVALID_STATE: AdbcStatusCode = 6;
pub const ADBC_STATUS_INVALID_DATA: AdbcStatusCode = 7;
pub const ADBC_STATUS_INTEGRITY: AdbcStatusCode = 8;
pub const ADBC_STATUS_INTERNAL: AdbcStatusCode = 9;
pub const ADBC_STATUS_IO: AdbcStatusCode = 10;
pub const ADBC_STATUS_CANCELLED: AdbcStatusCode = 11;
pub const ADBC_STATUS_TIMEOUT: AdbcStatusCode = 12;
pub const ADBC_STATUS_UNAUTHENTICATED: AdbcStatusCode = 13;
pub const ADBC_STATUS_UNAUTHORIZED: AdbcStatusCode = 14;

pub const ADBC_VERSION_1_0_0: c_int = 1_000_000;
pub const ADBC_VERSION_1_1_0: c_int = 1_001_000;

pub const ADBC_ERROR_VENDOR_CODE_PRIVATE_DATA: i32 = i32::MIN;

pub const ADBC_OPTION_URI: &CStr = c"uri";
pub const ADBC_OPTION_USERNAME: &CStr = c"username";
pub const ADBC_OPTION_PASSWORD: &CStr = c"password";

pub fn status_name(status: AdbcStatusCode) -> &'static str {
    match status {
        ADBC_STATUS_OK => "OK",
        ADBC_STATUS_UNKNOWN => "UNKNOWN",
        ADBC_STATUS_NOT_IMPLEMENTED => "NOT_IMPLEMENTED",
        ADBC_STATUS_NOT_FOUND => "NOT_FOUND",
        ADBC_STATUS_ALREADY_EXISTS => "ALREADY_EXISTS",
        ADBC_STATUS_INVALID_ARGUMENT => "INVALID_ARGUMENT",
        ADBC_STATUS_INVALID_STATE => "INVALID_STATE",
        ADBC_STATUS_INVALID_DATA => "INVALID_DATA",
        ADBC_STATUS_INTEGRITY => "INTEGRITY",
        ADBC_STATUS_INTERNAL => "INTERNAL",
        ADBC_STATUS_IO => "IO",
        ADBC_STATUS_CANCELLED => "CANCELLED",
        ADBC_STATUS_TIMEOUT => "TIMEOUT",
        ADBC_STATUS_UNAUTHENTICATED => "UNAUTHENTICATED",
        ADBC_STATUS_UNAUTHORIZED => "UNAUTHORIZED",
        _ => "UNKNOWN_STATUS",
    }
}

#[repr(C)]
pub struct AdbcError {
    pub message: *mut c_char,
    pub vendor_code: i32,
    pub sqlstate: [c_char; 5],
    pub release: Option<unsafe extern "C" fn(*mut AdbcError)>,
    /// 1.1.0 only.
    pub private_data: *mut c_void,
    /// 1.1.0 only.
    pub private_driver: *mut AdbcDriver,
}

impl AdbcError {
    /// `ADBC_ERROR_INIT`.
    pub fn new() -> Self {
        Self {
            message: ptr::null_mut(),
            vendor_code: ADBC_ERROR_VENDOR_CODE_PRIVATE_DATA,
            sqlstate: [0; 5],
            release: None,
            private_data: ptr::null_mut(),
            private_driver: ptr::null_mut(),
        }
    }

    /// The error message and details, if the driver set any.
    pub fn describe(&self) -> Option<String> {
        if self.message.is_null() {
            return None;
        }
        let mut out = unsafe { CStr::from_ptr(self.message) }
            .to_string_lossy()
            .into_owned();
        if self.sqlstate.iter().any(|&c| c != 0) {
            let state: String = self
                .sqlstate
                .iter()
                .take_while(|&&c| c != 0)
                // `c_char` is `i8` on some platforms.
                .map(|&c| {
                    #[allow(clippy::unnecessary_cast)]
                    let c = c as u8;
                    c as char
                })
                .collect();
            out.push_str(&format!(" (SQLSTATE {state})"));
        }
        if self.vendor_code != 0 && self.vendor_code != ADBC_ERROR_VENDOR_CODE_PRIVATE_DATA {
            out.push_str(&format!(" (vendor code {})", self.vendor_code));
        }
        Some(out)
    }

    pub fn reset(&mut self) {
        self.release_inner();
        *self = Self::new();
    }

    fn release_inner(&mut self) {
        if let Some(release) = self.release.take() {
            unsafe { release(self) };
        }
    }
}

impl Default for AdbcError {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AdbcError {
    fn drop(&mut self) {
        self.release_inner();
    }
}

#[repr(C)]
pub struct AdbcErrorDetail {
    pub key: *const c_char,
    pub value: *const u8,
    pub value_length: usize,
}

#[repr(C)]
pub struct AdbcDatabase {
    pub private_data: *mut c_void,
    pub private_driver: *mut AdbcDriver,
}

#[repr(C)]
pub struct AdbcConnection {
    pub private_data: *mut c_void,
    pub private_driver: *mut AdbcDriver,
}

#[repr(C)]
pub struct AdbcStatement {
    pub private_data: *mut c_void,
    pub private_driver: *mut AdbcDriver,
}

macro_rules! empty_handle {
    ($($t:ty),*) => {$(
        impl $t {
            pub fn empty() -> Self {
                Self {
                    private_data: ptr::null_mut(),
                    private_driver: ptr::null_mut(),
                }
            }
        }
    )*};
}
empty_handle!(AdbcDatabase, AdbcConnection, AdbcStatement);

#[repr(C)]
pub struct AdbcPartitions {
    pub num_partitions: usize,
    pub partitions: *mut *const u8,
    pub partition_lengths: *const usize,
    pub private_data: *mut c_void,
    pub release: Option<unsafe extern "C" fn(*mut AdbcPartitions)>,
}

type Fp<Args> = Option<Args>;
type E = *mut AdbcError;
type S = AdbcStatusCode;

/// The driver's function table, in the exact order of `struct AdbcDriver` in `adbc.h`.
/// Entries after `StatementSetSubstraitPlan` exist from API 1.1.0 on; with a 1.0.0 driver they
/// stay `None`.
#[repr(C)]
pub struct AdbcDriver {
    pub private_data: *mut c_void,
    pub private_manager: *mut c_void,

    pub release: Fp<unsafe extern "C" fn(*mut AdbcDriver, E) -> S>,

    pub DatabaseInit: Fp<unsafe extern "C" fn(*mut AdbcDatabase, E) -> S>,
    pub DatabaseNew: Fp<unsafe extern "C" fn(*mut AdbcDatabase, E) -> S>,
    pub DatabaseSetOption:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *const c_char, E) -> S>,
    pub DatabaseRelease: Fp<unsafe extern "C" fn(*mut AdbcDatabase, E) -> S>,

    pub ConnectionCommit: Fp<unsafe extern "C" fn(*mut AdbcConnection, E) -> S>,
    pub ConnectionGetInfo: Fp<
        unsafe extern "C" fn(*mut AdbcConnection, *const u32, usize, *mut ArrowArrayStream, E) -> S,
    >,
    pub ConnectionGetObjects: Fp<
        unsafe extern "C" fn(
            *mut AdbcConnection,
            c_int,
            *const c_char,
            *const c_char,
            *const c_char,
            *const *const c_char,
            *const c_char,
            *mut ArrowArrayStream,
            E,
        ) -> S,
    >,
    pub ConnectionGetTableSchema: Fp<
        unsafe extern "C" fn(
            *mut AdbcConnection,
            *const c_char,
            *const c_char,
            *const c_char,
            *mut ArrowSchema,
            E,
        ) -> S,
    >,
    pub ConnectionGetTableTypes:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *mut ArrowArrayStream, E) -> S>,
    pub ConnectionInit: Fp<unsafe extern "C" fn(*mut AdbcConnection, *mut AdbcDatabase, E) -> S>,
    pub ConnectionNew: Fp<unsafe extern "C" fn(*mut AdbcConnection, E) -> S>,
    pub ConnectionSetOption:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *const c_char, E) -> S>,
    pub ConnectionReadPartition: Fp<
        unsafe extern "C" fn(*mut AdbcConnection, *const u8, usize, *mut ArrowArrayStream, E) -> S,
    >,
    pub ConnectionRelease: Fp<unsafe extern "C" fn(*mut AdbcConnection, E) -> S>,
    pub ConnectionRollback: Fp<unsafe extern "C" fn(*mut AdbcConnection, E) -> S>,

    pub StatementBind:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *mut ArrowArray, *mut ArrowSchema, E) -> S>,
    pub StatementBindStream:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *mut ArrowArrayStream, E) -> S>,
    pub StatementExecuteQuery:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *mut ArrowArrayStream, *mut i64, E) -> S>,
    pub StatementExecutePartitions: Fp<
        unsafe extern "C" fn(
            *mut AdbcStatement,
            *mut ArrowSchema,
            *mut AdbcPartitions,
            *mut i64,
            E,
        ) -> S,
    >,
    pub StatementGetParameterSchema:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *mut ArrowSchema, E) -> S>,
    pub StatementNew: Fp<unsafe extern "C" fn(*mut AdbcConnection, *mut AdbcStatement, E) -> S>,
    pub StatementPrepare: Fp<unsafe extern "C" fn(*mut AdbcStatement, E) -> S>,
    pub StatementRelease: Fp<unsafe extern "C" fn(*mut AdbcStatement, E) -> S>,
    pub StatementSetOption:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *const c_char, E) -> S>,
    pub StatementSetSqlQuery: Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, E) -> S>,
    pub StatementSetSubstraitPlan:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const u8, usize, E) -> S>,

    // --- 1.1.0 ---
    pub ErrorGetDetailCount: Fp<unsafe extern "C" fn(*const AdbcError) -> c_int>,
    pub ErrorGetDetail: Fp<unsafe extern "C" fn(*const AdbcError, c_int) -> AdbcErrorDetail>,
    pub ErrorFromArrayStream:
        Fp<unsafe extern "C" fn(*mut ArrowArrayStream, *mut AdbcStatusCode) -> *const AdbcError>,

    pub DatabaseGetOption:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *mut c_char, *mut usize, E) -> S>,
    pub DatabaseGetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *mut u8, *mut usize, E) -> S>,
    pub DatabaseGetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *mut f64, E) -> S>,
    pub DatabaseGetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *mut i64, E) -> S>,
    pub DatabaseSetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, *const u8, usize, E) -> S>,
    pub DatabaseSetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, f64, E) -> S>,
    pub DatabaseSetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcDatabase, *const c_char, i64, E) -> S>,

    pub ConnectionCancel: Fp<unsafe extern "C" fn(*mut AdbcConnection, E) -> S>,
    pub ConnectionGetOption: Fp<
        unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *mut c_char, *mut usize, E) -> S,
    >,
    pub ConnectionGetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *mut u8, *mut usize, E) -> S>,
    pub ConnectionGetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *mut f64, E) -> S>,
    pub ConnectionGetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *mut i64, E) -> S>,
    pub ConnectionGetStatistics: Fp<
        unsafe extern "C" fn(
            *mut AdbcConnection,
            *const c_char,
            *const c_char,
            *const c_char,
            c_char,
            *mut ArrowArrayStream,
            E,
        ) -> S,
    >,
    pub ConnectionGetStatisticNames:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *mut ArrowArrayStream, E) -> S>,
    pub ConnectionSetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, *const u8, usize, E) -> S>,
    pub ConnectionSetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, f64, E) -> S>,
    pub ConnectionSetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcConnection, *const c_char, i64, E) -> S>,

    pub StatementCancel: Fp<unsafe extern "C" fn(*mut AdbcStatement, E) -> S>,
    pub StatementExecuteSchema:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *mut ArrowSchema, E) -> S>,
    pub StatementGetOption: Fp<
        unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *mut c_char, *mut usize, E) -> S,
    >,
    pub StatementGetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *mut u8, *mut usize, E) -> S>,
    pub StatementGetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *mut f64, E) -> S>,
    pub StatementGetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *mut i64, E) -> S>,
    pub StatementSetOptionBytes:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, *const u8, usize, E) -> S>,
    pub StatementSetOptionDouble:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, f64, E) -> S>,
    pub StatementSetOptionInt:
        Fp<unsafe extern "C" fn(*mut AdbcStatement, *const c_char, i64, E) -> S>,
}

impl AdbcDriver {
    /// A table with every entry unset, to be filled in by the driver's init function.
    pub fn empty() -> Self {
        // SAFETY: every field is a raw pointer or an `Option` of a function pointer, for which
        // all-zero bytes are null / `None`.
        unsafe { std::mem::zeroed() }
    }
}

/// `AdbcDriverInitFunc`.
pub type AdbcDriverInitFunc =
    unsafe extern "C" fn(version: c_int, driver: *mut c_void, error: *mut AdbcError) -> S;

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;

    #[test]
    fn layouts() {
        let p = size_of::<*const ()>();
        assert_eq!(size_of::<AdbcDatabase>(), 2 * p);
        // 2 data pointers + `release` + 26 1.0.0 functions + 3 error functions + 7 database
        // option functions + 10 connection functions + 9 statement functions.
        assert_eq!(size_of::<AdbcDriver>(), (2 + 1 + 26 + 3 + 7 + 10 + 9) * p);
        assert_eq!(offset_of!(AdbcDriver, StatementSetSubstraitPlan), 28 * p);
        assert_eq!(offset_of!(AdbcDriver, ErrorGetDetailCount), 29 * p);
        assert_eq!(offset_of!(AdbcDriver, StatementSetOptionInt), 57 * p);
        if p == 8 {
            assert_eq!(offset_of!(AdbcError, release), 24);
            assert_eq!(size_of::<AdbcError>(), 48);
        }
    }
}
