"""Seeded experiment-run registry (W5).

Every experiment run creates a run directory holding:

- ``resolved_config.json``: the exact configuration the run executed,
- ``run_info.json``: experiment name, seed, config hash, command line,
- ``manifest.json``: every output artifact with its sha256 hash.

Re-running with the same seed and config reproduces the manifest hashes,
provided the runner itself is deterministic (no wall-clock timestamps or
unseeded randomness in artifact bytes). ``*.log`` files are excluded from
the manifest: log records carry wall-clock timestamps by nature.
"""

from npdl.experiments.registry import (
    ExperimentRun,
    compute_config_hash,
    create_run,
    verify_manifest,
)
from npdl.experiments.validate import (
    ValidationError,
    validate,
    validate_config_file,
    validate_file,
    validate_scenario_file,
)

__all__ = [
    "ExperimentRun",
    "ValidationError",
    "compute_config_hash",
    "create_run",
    "validate",
    "validate_config_file",
    "validate_file",
    "validate_scenario_file",
]
