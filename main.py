"""Deprecated root entry point for N-person IPD experiments.

The runner logic moved verbatim to :mod:`npdl.simulation.experiments`
(W2 entry-point unification). The canonical CLI is now::

    python run.py simulate [options]

This module only re-exports the moved names for backward compatibility
(``scripts/runners/*`` and ``tests/test_integration.py`` still import from
here) and emits a :class:`DeprecationWarning` on import and whenever
:func:`main` is called (the call-site warning is what CLI users see, since
an import-time warning alone is hidden by default when running a script).
It will be removed in a later release once the remaining importers migrate.
"""

import warnings

_DEPRECATION_MESSAGE = (
    "main.py is deprecated and will be removed in a later release; "
    "use 'python run.py simulate' or import from "
    "'npdl.simulation.experiments' instead."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

from npdl.simulation import experiments as _experiments
from npdl.simulation.experiments import (
    load_scenarios,
    print_comparative_summary,
    save_results,
    setup_experiment,
)

__all__ = [
    "load_scenarios",
    "main",
    "print_comparative_summary",
    "save_results",
    "setup_experiment",
]


def main():
    """Deprecated; delegates to :func:`npdl.simulation.experiments.main`."""
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    return _experiments.main()


if __name__ == "__main__":
    main()
