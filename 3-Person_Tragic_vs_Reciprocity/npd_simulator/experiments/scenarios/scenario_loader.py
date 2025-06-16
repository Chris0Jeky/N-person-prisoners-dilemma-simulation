"""
Scenario loading utilities
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union


class ScenarioLoader:
    """
    Loads and manages experiment scenarios.
    """
    
    def __init__(self, scenarios_dir: str = "scenarios"):
        """
        Initialize scenario loader.
        
        Args:
            scenarios_dir: Directory containing scenario files
        """
        self.scenarios_dir = Path(scenarios_dir)
        
    def load_scenario(self, scenario_name: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load a scenario by name.
        
        Args:
            scenario_name: Name of the scenario file (without extension)
            
        Returns:
            Scenario configuration(s)
        """
        # Try different extensions
        for ext in ['.json', '.yml', '.yaml']:
            scenario_path = self.scenarios_dir / f"{scenario_name}{ext}"
            if scenario_path.exists():
                return self._load_file(scenario_path)
        
        raise FileNotFoundError(f"Scenario not found: {scenario_name}")
    
    def load_all_scenarios(self) -> Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Load all scenarios from the scenarios directory.
        
        Returns:
            Dictionary mapping scenario names to configurations
        """
        scenarios = {}
        
        for file_path in self.scenarios_dir.glob("*.json"):
            scenario_name = file_path.stem
            scenarios[scenario_name] = self._load_file(file_path)
        
        return scenarios
    
    def _load_file(self, file_path: Path) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Load scenario from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_scenario(self, scenario: Union[Dict[str, Any], List[Dict[str, Any]]], 
                     name: str):
        """
        Save a scenario to file.
        
        Args:
            scenario: Scenario configuration(s)
            name: Name for the scenario file
        """
        file_path = self.scenarios_dir / f"{name}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(scenario, f, indent=2)
    
    def create_standard_scenarios(self):
        """Create a set of standard scenarios for testing."""
        scenarios = {}
        
        # Classic 3-agent scenarios
        scenarios['classic_3tft'] = {
            'name': 'classic_3tft',
            'agents': [
                {'id': 0, 'type': 'TFT', 'exploration_rate': 0.0},
                {'id': 1, 'type': 'TFT', 'exploration_rate': 0.0},
                {'id': 2, 'type': 'TFT', 'exploration_rate': 0.0}
            ],
            'num_rounds': 1000
        }
        
        scenarios['mixed_classic'] = {
            'name': 'mixed_classic',
            'agents': [
                {'id': 0, 'type': 'TFT', 'exploration_rate': 0.1},
                {'id': 1, 'type': 'AllD', 'exploration_rate': 0.0},
                {'id': 2, 'type': 'AllC', 'exploration_rate': 0.0}
            ],
            'num_rounds': 1000
        }
        
        # Q-learning scenarios
        scenarios['qlearning_vs_allc'] = {
            'name': 'qlearning_vs_allc',
            'agents': [
                {'id': 0, 'type': 'QLearning', 'exploration_rate': 0.0,
                 'learning_rate': 0.1, 'epsilon': 0.1},
                {'id': 1, 'type': 'AllC', 'exploration_rate': 0.0},
                {'id': 2, 'type': 'AllC', 'exploration_rate': 0.0}
            ],
            'num_rounds': 2000
        }
        
        scenarios['enhanced_qlearning_test'] = {
            'name': 'enhanced_qlearning_test',
            'agents': [
                {'id': 0, 'type': 'EnhancedQLearning', 'exploration_rate': 0.0,
                 'learning_rate': 0.1, 'epsilon': 0.1, 'epsilon_decay': 0.995,
                 'exclude_self': True, 'opponent_modeling': True},
                {'id': 1, 'type': 'AllC', 'exploration_rate': 0.0},
                {'id': 2, 'type': 'AllD', 'exploration_rate': 0.0}
            ],
            'num_rounds': 2000
        }
        
        # Larger scenarios
        scenarios['large_mixed_10'] = {
            'name': 'large_mixed_10',
            'agents': [],
            'num_rounds': 1000
        }
        
        # Create 10 agents with mixed strategies
        agent_types = ['TFT', 'AllD', 'AllC', 'Random', 'QLearning']
        for i in range(10):
            agent_type = agent_types[i % len(agent_types)]
            agent = {'id': i, 'type': agent_type, 'exploration_rate': 0.05}
            
            if agent_type == 'Random':
                agent['cooperation_probability'] = 0.5
            elif agent_type == 'QLearning':
                agent['learning_rate'] = 0.1
                agent['epsilon'] = 0.1
            
            scenarios['large_mixed_10']['agents'].append(agent)
        
        # Save all scenarios
        for name, scenario in scenarios.items():
            self.save_scenario(scenario, name)
        
        return scenarios