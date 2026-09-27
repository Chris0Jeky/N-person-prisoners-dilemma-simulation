"""
Visualization tools for N-Person Prisoner's Dilemma simulations.

This module contains components for visualizing simulation results,
agent behavior, and network structures using Flask and Dash.
"""

import sys
import types

try:
    from . import dashboard
except ModuleNotFoundError as exc:
    # The dashboard needs optional viz deps that minimal envs (and W6 CI)
    # do not install. Expose a stub submodule so that
    # `npdl.visualization.dashboard` stays importable and patchable;
    # calling run_dashboard raises an ImportError naming the real problem.
    # Only stub for the known optional deps -- anything else propagates.
    _OPTIONAL_VIZ_DEPS = {
        "dash",
        "dash_bootstrap_components",
        "plotly",
        "flask",
        "pandas",
    }
    if exc.name is None or exc.name.split(".")[0] not in _OPTIONAL_VIZ_DEPS:
        raise
    dashboard = types.ModuleType(__name__ + ".dashboard")

    def _missing_run_dashboard(*args, **kwargs):
        raise ImportError(
            "Visualization dashboard requires optional dependencies "
            "(dash, dash_bootstrap_components, plotly, flask, pandas); "
            f"import failed: {exc}"
        )

    dashboard.run_dashboard = _missing_run_dashboard
    sys.modules[__name__ + ".dashboard"] = dashboard
