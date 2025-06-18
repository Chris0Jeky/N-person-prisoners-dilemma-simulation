"""
Scenario generation for experiments
"""

from typing import List, Dict, Any, Optional
import itertools
import random


class ScenarioGenerator:
    """
    Generates experiment scenarios with various agent compositions.
    """
    
    def __init__(self):
        """Initialize scenario generator."""
        self.base_agent_types = [
            "TFT", "pTFT", "pTFT-Threshold",
            "AllC", "AllD", "Random",
            "QLearning", "EnhancedQLearning"
        ]
        
    def generate_all_compositions(self, 
                                 num_agents: int,
                                 agent_types: Optional[List[str]] = None,
                                 exploration_rates: List[float] = [0.0, 0.1]) -> List[Dict[str, Any]]:
        """
        Generate all possible agent compositions.
        
        Args:
            num_agents: Number of agents
            agent_types: Types to include (uses defaults if None)
            exploration_rates: Exploration rates to test
            
        Returns:
            List of scenario configurations
        """
        if agent_types is None:
            agent_types = ["TFT", "AllC", "AllD"]
        
        scenarios = []
        
        # Generate all combinations with replacement
        for composition in itertools.combinations_with_replacement(agent_types, num_agents):
            for exp_rate in exploration_rates:
                scenario = self._create_scenario(composition, exp_rate, num_agents)
                scenarios.append(scenario)
        
        return scenarios
    
    def generate_balanced_scenarios(self,
                                  num_agents: int,
                                  num_scenarios: int = 10) -> List[Dict[str, Any]]:
        """
        Generate balanced scenarios with mixed strategies.
        
        Args:
            num_agents: Number of agents
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario configurations
        """
        scenarios = []
        
        for i in range(num_scenarios):
            # Create a balanced mix
            composition = []
            
            # Ensure at least one of each major type
            core_types = ["TFT", "AllC", "AllD"]
            composition.extend(core_types)
            
            # Fill remaining slots
            remaining = num_agents - len(core_types)
            for _ in range(remaining):
                agent_type = random.choice(self.base_agent_types)
                composition.append(agent_type)
            
            # Shuffle for variety
            random.shuffle(composition)
            
            # Create scenario
            exp_rate = random.choice([0.0, 0.05, 0.1])
            scenario = self._create_scenario(composition, exp_rate, num_agents)
            scenario['name'] = f"balanced_scenario_{i+1}"
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_qlearning_test_scenarios(self, num_agents: int = 3) -> List[Dict[str, Any]]:
        """
        Generate scenarios specifically for testing Q-learning.
        
        Args:
            num_agents: Number of agents
            
        Returns:
            List of Q-learning test scenarios
        """
        scenarios = []
        
        # Test against different opponent types
        opponent_types = ["AllC", "AllD", "TFT", "Random"]
        
        for opp_type in opponent_types:
            # Basic Q-learning
            scenario = {
                'name': f'qlearning_vs_{opp_type.lower()}',
                'agents': [
                    {'id': 0, 'type': 'QLearning', 'exploration_rate': 0.0,
                     'learning_rate': 0.1, 'epsilon': 0.1}
                ],
                'num_rounds': 1000
            }
            
            # Add opponents
            for i in range(1, num_agents):
                scenario['agents'].append({
                    'id': i, 
                    'type': opp_type,
                    'exploration_rate': 0.0
                })
            
            scenarios.append(scenario)
            
            # Enhanced Q-learning with improvements
            enhanced_scenario = {
                'name': f'enhanced_qlearning_vs_{opp_type.lower()}',
                'agents': [
                    {'id': 0, 'type': 'EnhancedQLearning', 'exploration_rate': 0.0,
                     'learning_rate': 0.1, 'epsilon': 0.1, 'epsilon_decay': 0.995,
                     'exclude_self': True, 'opponent_modeling': True}
                ],
                'num_rounds': 2000
            }
            
            # Add opponents
            for i in range(1, num_agents):
                enhanced_scenario['agents'].append({
                    'id': i,
                    'type': opp_type,
                    'exploration_rate': 0.0
                })
            
            scenarios.append(enhanced_scenario)
        
        return scenarios
    
    def generate_tournament_scenarios(self, 
                                    agent_types: List[str],
                                    rounds_per_pair: int = 100) -> Dict[str, Any]:
        """
        Generate tournament scenario for pairwise games.
        
        Args:
            agent_types: List of agent types
            rounds_per_pair: Rounds per pairwise interaction
            
        Returns:
            Tournament configuration
        """
        agents = []
        
        for i, agent_type in enumerate(agent_types):
            agent_config = {
                'id': i,
                'type': agent_type,
                'exploration_rate': 0.0
            }
            
            # Add specific parameters for RL agents
            if 'QLearning' in agent_type:
                agent_config.update({
                    'learning_rate': 0.1,
                    'epsilon': 0.1
                })
                
                if agent_type == 'EnhancedQLearning':
                    agent_config.update({
                        'epsilon_decay': 0.995,
                        'exclude_self': True
                    })
            
            agents.append(agent_config)
        
        return {
            'name': f'tournament_{len(agent_types)}_agents',
            'agents': agents,
            'rounds_per_pair': rounds_per_pair,
            'num_episodes': 1
        }
    
    def _create_scenario(self, 
                        composition: List[str],
                        exploration_rate: float,
                        num_agents: int) -> Dict[str, Any]:
        """
        Create a scenario configuration.
        
        Args:
            composition: List of agent types
            exploration_rate: Exploration rate for all agents
            num_agents: Total number of agents
            
        Returns:
            Scenario configuration
        """
        agents = []
        
        for i, agent_type in enumerate(composition):
            agent_config = {
                'id': i,
                'type': agent_type,
                'exploration_rate': exploration_rate
            }
            
            # Add type-specific parameters
            if agent_type == "Random":
                agent_config['cooperation_probability'] = 0.5
            elif agent_type == "pTFT-Threshold":
                agent_config['threshold'] = 0.5
            elif 'QLearning' in agent_type:
                agent_config['learning_rate'] = 0.1
                agent_config['epsilon'] = 0.1
                
                if agent_type == 'EnhancedQLearning':
                    agent_config['epsilon_decay'] = 0.995
                    agent_config['exclude_self'] = True
            
            agents.append(agent_config)
        
        # Create scenario name
        type_counts = {}
        for agent_type in composition:
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        name_parts = [f"{count}{t}" for t, count in type_counts.items()]
        exp_str = f"_exp{int(exploration_rate*100)}" if exploration_rate > 0 else ""
        name = f"{'_'.join(name_parts)}{exp_str}"
        
        return {
            'name': name,
            'agents': agents,
            'num_rounds': 1000
        }