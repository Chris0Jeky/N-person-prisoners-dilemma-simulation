"""
Adapter to integrate true pairwise mode with the existing simulation framework.

This module provides compatibility between the new true pairwise implementation
and the existing agent/environment infrastructure.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from .agents import Agent, TitForTat, GenerousTitForTat, Pavlov, QLearningAgent, \
    LeniencyRewardAdjustedQLearning, HystereticQLearning, WolfPHC, UCB1Agent
from .true_pairwise import (
    TruePairwiseAgent, TruePairwiseTFT, TruePairwiseGTFT, TruePairwisePavlov,
    TruePairwiseQLearning, TruePairwiseAdaptive, TruePairwiseEnvironment
)


class TruePairwiseAgentAdapter(TruePairwiseAgent):
    """Adapter that wraps existing agents for true pairwise interactions."""
    
    def __init__(self, base_agent: Agent, memory_length: int = 10):
        super().__init__(base_agent.agent_id, memory_length)
        self.base_agent = base_agent
        self.strategy_type = base_agent.strategy_type
        
        # Map strategy types to behavior
        self.strategy_map = {
            "tit_for_tat": self._tft_behavior,
            "generous_tit_for_tat": self._gtft_behavior,
            "suspicious_tit_for_tat": self._stft_behavior,
            "tit_for_two_tats": self._tf2t_behavior,
            "pavlov": self._pavlov_behavior,
            "always_cooperate": lambda _: "cooperate",
            "always_defect": lambda _: "defect",
            "random": lambda _: np.random.choice(["cooperate", "defect"])
        }
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        """Choose action based on the wrapped agent's strategy."""
        memory = self.get_opponent_memory(opponent_id)
        
        # For RL agents, create a special context
        if hasattr(self.base_agent, 'choose_move') and self.strategy_type in [
            "q_learning", "lra_q_learning", "hysteretic_q_learning", "wolf_phc", "ucb1"
        ]:
            # Create opponent-specific context
            context = {
                'opponent_coop_proportion': memory.get_cooperation_rate(),
                'specific_opponent_moves': {opponent_id: memory.get_last_move()} if memory.get_last_move() else {}
            }
            
            # Temporarily override the agent's memory
            original_memory = self.base_agent.memory
            self.base_agent.memory = [{'neighbor_moves': context}] if context else []
            
            # Get action
            action = self.base_agent.choose_move(self.base_agent.memory)
            
            # Restore original memory
            self.base_agent.memory = original_memory
            
            return action
            
        # For non-RL strategies, use mapped behavior
        if self.strategy_type in self.strategy_map:
            return self.strategy_map[self.strategy_type](memory)
            
        # Default behavior
        return "cooperate"
        
    def _tft_behavior(self, memory) -> str:
        """Tit-for-Tat behavior."""
        last_move = memory.get_last_move()
        if last_move is None:
            return "cooperate"
        return last_move
        
    def _gtft_behavior(self, memory) -> str:
        """Generous Tit-for-Tat behavior."""
        last_move = memory.get_last_move()
        if last_move is None:
            return "cooperate"
        if last_move == "defect":
            # Be generous 10% of the time
            if np.random.random() < 0.1:
                return "cooperate"
        return last_move
        
    def _stft_behavior(self, memory) -> str:
        """Suspicious Tit-for-Tat behavior."""
        last_move = memory.get_last_move()
        if last_move is None:
            return "defect"  # Start with defection
        return last_move
        
    def _tf2t_behavior(self, memory) -> str:
        """Tit-for-Two-Tats behavior."""
        if len(memory.interaction_history) < 2:
            return "cooperate"
            
        # Check last two opponent moves
        last_two = [memory.interaction_history[-2]['opponent_move'],
                    memory.interaction_history[-1]['opponent_move']]
        
        # Defect only if opponent defected twice in a row
        if all(move == "defect" for move in last_two):
            return "defect"
        return "cooperate"
        
    def _pavlov_behavior(self, memory) -> str:
        """Pavlov (Win-Stay/Lose-Shift) behavior."""
        if not memory.interaction_history:
            return "cooperate"
            
        last_interaction = memory.interaction_history[-1]
        my_last_move = last_interaction['my_move']
        reward = last_interaction['reward']
        
        # Win-stay: If got good reward (3 or 5), repeat move
        # Lose-shift: If got bad reward (0 or 1), switch move
        if reward >= 3:
            return my_last_move
        else:
            return "defect" if my_last_move == "cooperate" else "cooperate"


def create_true_pairwise_agent(agent_config: Dict[str, Any]) -> TruePairwiseAgent:
    """Factory function to create true pairwise agents from configuration."""
    
    agent_type = agent_config.get('type', 'tit_for_tat')
    agent_id = agent_config.get('id', f'agent_{np.random.randint(10000)}')
    
    # Direct true pairwise implementations
    if agent_type == 'true_pairwise_tft':
        return TruePairwiseTFT(
            agent_id,
            nice=agent_config.get('nice', True),
            forgiving_probability=agent_config.get('forgiving_probability', 0.0)
        )
    elif agent_type == 'true_pairwise_gtft':
        return TruePairwiseGTFT(
            agent_id,
            generosity=agent_config.get('generosity', 0.1)
        )
    elif agent_type == 'true_pairwise_pavlov':
        return TruePairwisePavlov(agent_id)
    elif agent_type == 'true_pairwise_q_learning':
        return TruePairwiseQLearning(
            agent_id,
            learning_rate=agent_config.get('learning_rate', 0.1),
            discount_factor=agent_config.get('discount_factor', 0.95),
            epsilon=agent_config.get('epsilon', 0.1),
            state_type=agent_config.get('state_representation', 'memory_enhanced')
        )
    elif agent_type == 'true_pairwise_adaptive':
        return TruePairwiseAdaptive(
            agent_id,
            assessment_period=agent_config.get('assessment_period', 10)
        )
    else:
        # Create wrapper for existing agent types
        base_config = agent_config.copy()
        base_config['type'] = agent_type
        
        # Import create_agent function
        from .agents import create_agent as create_base_agent
        base_agent = create_base_agent(base_config)
        
        return TruePairwiseAgentAdapter(base_agent, memory_length=10)


class TruePairwiseSimulationAdapter:
    """Adapter to run true pairwise simulations with existing infrastructure."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = []
        self.environment = None
        self.results = None
        
    def setup(self):
        """Set up the simulation from configuration."""
        # Create agents
        for agent_config in self.config['agents']:
            agent = create_true_pairwise_agent(agent_config)
            self.agents.append(agent)
            
        # Create environment
        self.environment = TruePairwiseEnvironment(
            agents=self.agents,
            episodes=self.config.get('episodes', 1),
            rounds_per_episode=self.config.get('rounds', 100),
            noise_level=self.config.get('noise_level', 0.0),
            reset_between_episodes=self.config.get('reset_between_episodes', True)
        )
        
    def run(self) -> Dict[str, Any]:
        """Run the simulation and return results."""
        if not self.environment:
            self.setup()
            
        self.results = self.environment.run_simulation()
        return self.results
        
    def get_results_for_analysis(self) -> Dict[str, Any]:
        """Convert results to format expected by existing analysis tools."""
        if not self.results:
            raise ValueError("No results available. Run simulation first.")
            
        # Convert to expected format
        converted_results = {
            'agents': {},
            'rounds': [],
            'network': {'type': 'fully_connected', 'interaction_mode': 'true_pairwise'}
        }
        
        # Agent data
        for agent_id, stats in self.results['final_statistics'].items():
            if agent_id != 'global':
                converted_results['agents'][agent_id] = {
                    'total_score': stats['total_score'],
                    'cooperation_rate': stats['overall_cooperation_rate'],
                    'strategy_type': self.agents[0].strategy_type if hasattr(self.agents[0], 'strategy_type') else 'unknown',
                    'opponent_relationships': stats['opponent_relationships']
                }
                
        # Round data
        for round_data in self.results['round_data']:
            converted_results['rounds'].append({
                'round': round_data['round'],
                'episode': round_data['episode'],
                'cooperation_rate': round_data['cooperation_rate'],
                'total_payoff': round_data['total_payoff']
            })
            
        return converted_results
        
    def export_to_csv(self, filepath: str):
        """Export results to CSV format."""
        import pandas as pd
        
        if not self.results:
            raise ValueError("No results available. Run simulation first.")
            
        # Create dataframes
        agent_df = pd.DataFrame.from_dict(
            self.results['final_statistics'], 
            orient='index'
        ).drop('global')
        
        round_df = pd.DataFrame(self.results['round_data'])
        
        # Save to CSV
        agent_df.to_csv(f"{filepath}_agents.csv")
        round_df.to_csv(f"{filepath}_rounds.csv")
        
        # Save detailed interaction data
        interactions = []
        for episode_data in self.results['episodes']:
            for interaction in episode_data['results']:
                interactions.append(interaction)
                
        interaction_df = pd.DataFrame(interactions)
        interaction_df.to_csv(f"{filepath}_interactions.csv")
        
        print(f"Results exported to {filepath}_*.csv")