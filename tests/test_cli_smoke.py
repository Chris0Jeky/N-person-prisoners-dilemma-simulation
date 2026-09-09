"""W2 CLI smoke tests: entry-point wiring without running full simulations.

Covers the W2 acceptance checks that do not need optional dependencies
(dash/plotly/pygame) or long simulation runs:

- ``python run.py simulate --help`` (and the other subcommand helps) exit 0.
- ``visualize`` / ``interactive`` dispatch wiring in :mod:`npdl.cli`.
- The deprecated root ``main.py`` shim warns and delegates to
  :mod:`npdl.simulation.experiments`.
"""

import os
import subprocess
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)

from unittest.mock import MagicMock, patch  # noqa: E402

import npdl.cli as cli  # noqa: E402


def run_entry(script, *args):
    """Run a repo entry-point script as a subprocess."""
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, script), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


class TestRunPyHelp:
    """``run.py`` help output works for every subcommand."""

    def test_simulate_help(self):
        proc = run_entry("run.py", "simulate", "--help")
        assert proc.returncode == 0
        assert "scenario_file" in proc.stdout
        assert "--results_dir" in proc.stdout

    def test_top_level_help_lists_commands(self):
        proc = run_entry("run.py", "--help")
        assert proc.returncode == 0
        for command in ("simulate", "visualize", "interactive"):
            assert command in proc.stdout

    def test_no_command_prints_help(self):
        proc = run_entry("run.py")
        assert proc.returncode == 0
        assert "simulate" in proc.stdout

    def test_visualize_help(self):
        proc = run_entry("run.py", "visualize", "--help")
        assert proc.returncode == 0

    def test_interactive_help(self):
        proc = run_entry("run.py", "interactive", "--help")
        assert proc.returncode == 0


class TestVisualizeWiring:
    """``run_visualization`` dispatches to the dashboard correctly."""

    def test_missing_deps_returns_1(self, capsys):
        with patch.object(cli, "check_dependencies", return_value=["dash"]):
            assert cli.run_visualization() == 1
        captured = capsys.readouterr()
        assert "Missing required dependencies" in captured.out
        assert "dash" in captured.out

    def test_success_dispatch(self, monkeypatch, capsys):
        fake_dashboard = MagicMock()
        # Inject the dashboard module instead of patching its import path:
        # the real module needs dash/plotly, which may not be installed.
        monkeypatch.setitem(
            sys.modules, "npdl.visualization.dashboard", fake_dashboard
        )
        monkeypatch.setattr(cli, "check_dependencies", lambda packages: [])
        assert cli.run_visualization() == 0
        fake_dashboard.run_dashboard.assert_called_once_with(debug=True)
        captured = capsys.readouterr()
        assert "127.0.0.1:8050" in captured.out

    def test_dashboard_error_returns_1(self, monkeypatch, capsys):
        fake_dashboard = MagicMock()
        fake_dashboard.run_dashboard.side_effect = RuntimeError("boom")
        monkeypatch.setitem(
            sys.modules, "npdl.visualization.dashboard", fake_dashboard
        )
        monkeypatch.setattr(cli, "check_dependencies", lambda packages: [])
        assert cli.run_visualization() == 1
        captured = capsys.readouterr()
        assert "Error starting visualization dashboard" in captured.out


class TestInteractiveWiring:
    """``run_interactive`` dispatches to the interactive game correctly."""

    def test_success_dispatch(self, capsys):
        with patch("npdl.interactive.game.main") as mock_game:
            assert cli.run_interactive() == 0
        mock_game.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Starting interactive game mode" in captured.out

    def test_import_error_returns_1(self, capsys):
        with patch(
            "npdl.interactive.game.main", side_effect=ImportError("No game module")
        ):
            assert cli.run_interactive() == 1
        captured = capsys.readouterr()
        assert "Could not import interactive game module" in captured.out


class TestMainShim:
    """Root ``main.py`` warns (deprecated) and delegates to the moved module."""

    def test_cli_help_warns_and_succeeds(self):
        proc = run_entry("main.py", "--help")
        assert proc.returncode == 0
        assert "deprecat" in proc.stderr.lower()
        assert "Run N-person IPD experiments" in proc.stdout

    def test_import_warns(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-W",
                "always::DeprecationWarning",
                "-c",
                "import main",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "deprecat" in proc.stderr.lower()

    def test_main_call_warns_and_delegates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import main
            from npdl.simulation import experiments

        with patch.object(experiments, "main") as mock_main:
            with pytest.warns(DeprecationWarning, match="deprecated"):
                main.main()
        mock_main.assert_called_once_with()

    def test_reexports_moved_names(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import main
            from npdl.simulation import experiments

        for name in (
            "load_scenarios",
            "setup_experiment",
            "save_results",
            "print_comparative_summary",
        ):
            assert getattr(main, name) is getattr(experiments, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
