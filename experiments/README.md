# experiments/

This directory will hold the canonical experiment system: a registry where each run records its seed, config hash, command line, and an artifact manifest, with result outputs written to ignored directories rather than committed. Runners migrated from `scripts/runners/` and the surviving versioned experiment families will live here as thin consumers of the `npdl` package, plus one small golden fixture kept under `tests/` for regression checks.
