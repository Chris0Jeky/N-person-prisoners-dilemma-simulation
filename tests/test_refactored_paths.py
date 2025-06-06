"""
Test script to verify that all paths are correct after project structure refactoring.
This script tests file existence and critical code paths to ensure refactoring didn't break functionality.
"""

import os
import sys
import json
import unittest
import importlib

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules that need to be tested
try:
    from npdl.core.agents import Agent
    from npdl.core.environment import Environment
    from npdl.core.utils import create_payoff_matrix
    from npdl.simulation.runner import load_scenarios
    from main import setup_experiment
except ImportError as e:
    print(f"Error importing core modules: {e}")


class TestRefactoredPaths(unittest.TestCase):
    """Test case for verifying path changes after refactoring."""
    
    def setUp(self):
        """Set up test environment."""
        # Define paths to critical files
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scenario_files = [
            os.path.join(self.project_root, 'scenarios', 'scenarios.json'),
            os.path.join(self.project_root, 'scenarios', 'enhanced_scenarios.json'),
            os.path.join(self.project_root, 'scenarios', 'pairwise_scenarios.json')
        ]
        self.config_files = [
            os.path.join(self.project_root, 'configs', 'multi_sweep_config.json'),
            os.path.join(self.project_root, 'configs', 'sweep_config.json')
        ]
        self.logger = None  # Will be set in tests if needed
    
    def test_file_existence(self):
        """Test that all required files exist in their new locations."""
        # Check scenario files
        for file_path in self.scenario_files:
            self.assertTrue(os.path.exists(file_path), f"Missing scenario file: {file_path}")
        
        # Check config files
        for file_path in self.config_files:
            self.assertTrue(os.path.exists(file_path), f"Missing config file: {file_path}")
        
        # Check test files were moved
        self.assertTrue(
            os.path.exists(os.path.join(self.project_root, 'tests', 'test_core_basic.py')),
            "test_core_basic.py not found in tests directory"
        )
        
        # Check analysis scripts
        self.assertTrue(
            os.path.exists(os.path.join(self.project_root, 'analysis_scripts', 'run_statistical_analysis.py')),
            "run_statistical_analysis.py not found in analysis_scripts directory"
        )
    
    def test_load_scenarios(self):
        """Test loading scenarios from new path."""
        # Test loading standard scenarios
        standard_scenarios_path = os.path.join(self.project_root, 'scenarios', 'scenarios.json')
        try:
            scenarios = load_scenarios(standard_scenarios_path)
            self.assertIsInstance(scenarios, list)
            self.assertGreater(len(scenarios), 0)
            print(f"Successfully loaded {len(scenarios)} standard scenarios")
        except Exception as e:
            self.fail(f"Failed to load standard scenarios: {e}")
        
        # Test loading enhanced scenarios
        enhanced_scenarios_path = os.path.join(self.project_root, 'scenarios', 'enhanced_scenarios.json')
        try:
            enhanced_scenarios = load_scenarios(enhanced_scenarios_path)
            self.assertIsInstance(enhanced_scenarios, list)
            self.assertGreater(len(enhanced_scenarios), 0)
            print(f"Successfully loaded {len(enhanced_scenarios)} enhanced scenarios")
        except Exception as e:
            self.fail(f"Failed to load enhanced scenarios: {e}")
            
    def test_load_config(self):
        """Test loading config files from new path."""
        # Test loading multi sweep config
        sweep_config_path = os.path.join(self.project_root, 'configs', 'multi_sweep_config.json')
        try:
            with open(sweep_config_path, 'r') as f:
                config = json.load(f)
            self.assertIsInstance(config, dict)
            self.assertIn('global_settings', config)
            print(f"Successfully loaded sweep config with {len(config)} sections")
        except Exception as e:
            self.fail(f"Failed to load sweep config: {e}")
    
    def test_setup_experiment(self):
        """Test setting up an experiment using a scenario."""
        import logging
        
        # Create a test logger
        test_logger = logging.getLogger('test')
        test_logger.setLevel(logging.ERROR)  # Keep it quiet
        
        # Load a scenario
        try:
            scenarios_path = os.path.join(self.project_root, 'scenarios', 'scenarios.json')
            scenarios = load_scenarios(scenarios_path)
            if scenarios:
                # Take the first scenario as a test
                test_scenario = scenarios[0]
                print(f"Testing setup_experiment with scenario: {test_scenario.get('scenario_name', 'unknown')}")
                
                # Try to set up an experiment
                try:
                    env, theoretical_scores = setup_experiment(test_scenario, test_logger)
                    self.assertIsInstance(env, Environment)
                    self.assertIsInstance(theoretical_scores, dict)
                    self.assertGreater(len(env.agents), 0)
                    print(f"Successfully set up experiment with {len(env.agents)} agents")
                except Exception as e:
                    self.fail(f"Failed to set up experiment: {e}")
            else:
                self.skipTest("No scenarios available for testing setup_experiment")
        except Exception as e:
            self.fail(f"Failed to load scenarios for setup_experiment test: {e}")
    
    def test_import_paths(self):
        """Test that all modules can be imported correctly."""
        # Test critical imports
        modules_to_test = [
            'npdl.core.agents',
            'npdl.core.environment',
            'npdl.core.utils',
            'npdl.core.logging_utils',
            'npdl.simulation.runner',
            'npdl.visualization.dashboard',
            'npdl.analysis.sweep_visualizer'
        ]
        
        for module_name in modules_to_test:
            try:
                imported_module = importlib.import_module(module_name)
                self.assertIsNotNone(imported_module)
                print(f"Successfully imported {module_name}")
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    print("Running path refactoring tests...")
    unittest.main()
