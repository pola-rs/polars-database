# polars-database

Native database scan plugin for [Polars](https://github.com/pola-rs/polars), released to PyPI as
`polars-database` (Python module `polars_database`). It implements `polars.scan_database`:
dialects, SQL pushdown, partitioning, driver resolution and the ADBC and Python driver backends.

Polars calls the plugin directly from its engine through the versioned FFI contracts in
[`polars-io-ext-ffi`](https://github.com/pola-rs/polars/tree/main/crates/polars-io-ext-ffi)
(`lazyframe_resolver.v1`, `file_reader.v1`); compatibility is decided by the plugin IDs the module
exports, not by the package version.

## Development

This crate is built as part of the Polars workspace: clone it (or add it as a worktree) to
`crates/polars-database` in the [Polars repository](https://github.com/pola-rs/polars). Then, from
the Polars root:

```bash
make -C py-polars build-polars-database  # build and install into the Polars venv
make -C py-polars test                   # also builds the plugin
```

## Branches and CI

- `dev`: development branch. Polars pull request and push CI clones it.
- `main`: release branch. Releases are tagged (e.g. `py-0.1.0`) from it.

The released version is `project.version` in `pyproject.toml`; `Cargo.toml`'s version is not
used for releases.

## Releasing

Releases are done from the Polars repository:

1. Bump `project.version` in `pyproject.toml` on `main` and push a tag for it.
2. Run Polars' `release-polars-database` workflow with that tag. It builds the sdist and abi3
   wheels (one per target) in the Polars workspace and publishes them to PyPI.
3. To make Polars require the release, update the `polars-database >= <version>, < <next major
   version>` requirement in the `database` extra of Polars' `py-polars/pyproject.toml`.

The PyPI project page is `README.pypi.md`.
