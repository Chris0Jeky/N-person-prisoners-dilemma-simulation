"""
Tests for the command-line interface module.

This module tests the CLI functionality including:
- Command parsing
- Argument validation
- Error handling
- Command execution
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.cli import main, check_dependencies, run_visualization, run_simulation, run_interactive


class TestCLI:
    """Test suite for the command-line interface."""
    
    def test_check_dependencies(self):
        """Test dependency checking functionality."""
        # Test with packages that should exist in a standard Python install
        result = check_dependencies(["os", "sys", "json"])
        assert result == []
        
        # Test with packages that likely don't exist
        result = check_dependencies(["fake_package_xyz", "another_fake_package"])
        assert len(result) == 2
        assert "fake_package_xyz" in result
        assert "another_fake_package" in result
        
    @patch('sys.argv', ['npdl'])
    def test_main_no_command(self, capsys):
        """Test main function with no command shows help."""
        with patch('argparse.ArgumentParser.print_help') as mock_help:
            result = main()
            assert result == 0
            mock_help.assert_called_once()
            
    @patch('sys.argv', ['npdl', 'simulate'])
    @patch('npdl.cli.run_simulation')
    def test_main_simulate_command(self, mock_run_sim):
        """Test main function with simulate command."""
        mock_run_sim.return_value = 0
        result = main()
        assert result == 0
        mock_run_sim.assert_called_once()
        
    @patch('sys.argv', ['npdl', 'visualize'])
    @patch('npdl.cli.run_visualization')
    def test_main_visualize_command(self, mock_run_vis):
        """Test main function with visualize command."""
        mock_run_vis.return_value = 0
        result = main()
        assert result == 0
        mock_run_vis.assert_called_once()
        
    @patch('sys.argv', ['npdl', 'interactive'])
    @patch('npdl.cli.run_interactive')
    def test_main_interactive_command(self, mock_run_int):
        """Test main function with interactive command."""
        mock_run_int.return_value = 0
        result = main()
        assert result == 0
        mock_run_int.assert_called_once()
        
    @patch('sys.argv', ['npdl', 'simulate', '--enhanced', '--verbose', '--analyze'])
    @patch('npdl.cli.run_simulation')
    def test_simulate_with_arguments(self, mock_run_sim):
        """Test simulate command with various arguments."""
        mock_run_sim.return_value = 0
        result = main()
        assert result == 0
        
        # Check that run_simulation was called with correct args
        args = mock_run_sim.call_args[0][0]
        assert args.enhanced is True
        assert args.verbose is True
        assert args.analyze is True
        
    @patch('sys.argv', ['npdl', 'simulate', '--scenario_file', 'custom.json', '--results_dir', 'custom_results'])
    @patch('npdl.cli.run_simulation')
    def test_simulate_with_custom_paths(self, mock_run_sim):
        """Test simulate command with custom file paths."""
        mock_run_sim.return_value = 0
        result = main()
        assert result == 0
        
        args = mock_run_sim.call_args[0][0]
        assert args.scenario_file == 'custom.json'
        assert args.results_dir == 'custom_results'
        
    @patch('npdl.cli.check_dependencies')
    def test_run_visualization_missing_deps(self, mock_check_deps, capsys):
        """Test visualization with missing dependencies."""
        mock_check_deps.return_value = ['dash', 'plotly']
        
        result = run_visualization()
        assert result == 1
        
        captured = capsys.readouterr()
        assert "Missing required dependencies" in captured.out
        assert "dash" in captured.out
        assert "plotly" in captured.out
        
    @patch('npdl.cli.check_dependencies')
    @patch('npdl.visualization.dashboard.run_dashboard')
    def test_run_visualization_success(self, mock_dashboard, mock_check_deps, capsys):
        """Test successful visualization launch."""
        mock_check_deps.return_value = []
        
        result = run_visualization()
        assert result == 0
        
        mock_dashboard.assert_called_once_with(debug=True)
        captured = capsys.readouterr()
        assert "Starting visualization dashboard" in captured.out
        assert "http://127.0.0.1:8050/" in captured.out
        
    @patch('npdl.cli.check_dependencies')
    @patch('npdl.visualization.dashboard.run_dashboard')
    def test_run_visualization_error(self, mock_dashboard, mock_check_deps, capsys):
        """Test visualization with dashboard error."""
        mock_check_deps.return_value = []
        mock_dashboard.side_effect = Exception("Dashboard error")
        
        result = run_visualization()
        assert result == 1
        
        captured = capsys.readouterr()
        assert "Error starting visualization dashboard" in captured.out
        
    @patch('npdl.simulation.run_simulation')
    def test_run_simulation_success(self, mock_sim_runner):
        """Test successful simulation run."""
        mock_sim_runner.return_value = 0
        
        args = Mock()
        args.enhanced = True
        args.scenario_file = "test.json"
        args.results_dir = "test_results"
        args.log_dir = "test_logs"
        args.analyze = True
        args.verbose = True
        
        result = run_simulation(args)
        assert result == 0
        
        mock_sim_runner.assert_called_once()
        call_kwargs = mock_sim_runner.call_args[1]
        assert call_kwargs['enhanced'] is True
        assert call_kwargs['verbose'] is True
        assert call_kwargs['analyze'] is True
        
    @patch('npdl.simulation.run_simulation', side_effect=ImportError)
    @patch('main.main')
    def test_run_simulation_fallback(self, mock_main, mock_sim_runner, capsys):
        """Test simulation fallback to main.py."""
        args = Mock()
        args.enhanced = False
        args.scenario_file = "test.json"
        args.results_dir = "results"
        args.log_dir = "logs"
        args.analyze = False
        args.verbose = False
        
        with patch('sys.argv', ['test']):
            result = run_simulation(args)
            assert result == 0
            
        mock_main.assert_called_once()
        captured = capsys.readouterr()
        assert "Using legacy simulation runner" in captured.out
        
    @patch('npdl.simulation.run_simulation', side_effect=ImportError)
    @patch('main.main', side_effect=ImportError)
    def test_run_simulation_import_error(self, mock_main, mock_sim_runner, capsys):
        """Test simulation with import errors."""
        args = Mock()
        args.enhanced = False
        args.scenario_file = "test.json"
        args.results_dir = "results"
        args.log_dir = "logs"
        args.analyze = False
        args.verbose = False
        
        result = run_simulation(args)
        assert result == 1
        
        captured = capsys.readouterr()
        assert "Error: Could not import simulation module" in captured.out
        
    @patch('npdl.interactive.game.main')
    def test_run_interactive_success(self, mock_game, capsys):
        """Test successful interactive game launch."""
        result = run_interactive()
        assert result == 0
        
        mock_game.assert_called_once()
        captured = capsys.readouterr()
        assert "Starting interactive game mode" in captured.out
        
    @patch('npdl.interactive.game.main', side_effect=ImportError("No game module"))
    def test_run_interactive_import_error(self, mock_game, capsys):
        """Test interactive game with import error."""
        result = run_interactive()
        assert result == 1
        
        captured = capsys.readouterr()
        assert "Error: Could not import interactive game module" in captured.out
        
    def test_command_line_help(self):
        """Test that help text is properly formatted."""
        with patch('sys.argv', ['npdl', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    main()
            
            # argparse exits with 0 for help
            assert exc_info.value.code == 0
            
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        with patch('sys.argv', ['npdl', 'invalid_command']):
            with pytest.raises(SystemExit):
                main()


class TestArgumentParsing:
    """Test argument parsing edge cases."""
    
    @patch('sys.argv', ['npdl', 'simulate', '--log_dir', 'custom_logs'])
    @patch('npdl.cli.run_simulation')
    def test_log_dir_argument(self, mock_run_sim):
        """Test log directory argument parsing."""
        mock_run_sim.return_value = 0
        main()
        
        args = mock_run_sim.call_args[0][0]
        assert args.log_dir == 'custom_logs'
        
    @patch('sys.argv', ['npdl', 'simulate', '--scenario_file'])
    def test_missing_argument_value(self):
        """Test handling of missing argument values."""
        with pytest.raises(SystemExit):
            main()
            
    @patch('sys.argv', ['npdl', 'simulate', '--unknown_arg'])
    def test_unknown_argument(self):
        """Test handling of unknown arguments."""
        with pytest.raises(SystemExit):
            main()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])