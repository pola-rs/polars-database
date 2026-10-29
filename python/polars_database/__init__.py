"""
Native database scan plugin for Polars (`polars.scan_database`).

This package is a dependency of Polars and is not meant to be used directly. Polars
loads it through the versioned FFI contracts it exports (`_polars_io_plugin_ids`,
`_capsule(id)`).
"""

from polars_database._lib import (
    __version__,
    _capsule,
    _polars_io_plugin_ids,
    dialect_from_scheme,
    driver_statuses,
    find_adbc_driver,
    scan_resolver,
)

__all__ = [
    "__version__",
    "_capsule",
    "_polars_io_plugin_ids",
    "dialect_from_scheme",
    "driver_statuses",
    "find_adbc_driver",
    "scan_resolver",
]
