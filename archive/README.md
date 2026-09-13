# archive/

Read-only archive of superseded code, populated by W4 (`w4-archive`):
the versioned `final_experimentations/v1..v8` chain, the `legacy` and
`legacy_cleaned` families, and six `code_for_website/*` experiment subdirs.
Layout mirrors the origin tree. See [MANIFEST.md](MANIFEST.md) for one row per
moved family (origin path, what superseded it in `npdl/`, how to run the
archived copy) plus the deliberately-not-archived list (`v9`, `main_runs/`,
`npd_simulator/`, result dumps).

Nothing under `archive/` is imported by live code; it exists to preserve
research provenance while the canonical implementation lives in `npdl/`.
