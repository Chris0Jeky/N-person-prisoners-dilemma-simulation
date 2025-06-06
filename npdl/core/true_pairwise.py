"""
True Pairwise Implementation for N-Person Prisoner's Dilemma

This module implements a true pairwise interaction mode where agents can make
individual decisions for each opponent, maintaining separate relationships and
memories for each bilateral interaction.

Key differences from the aggregate pairwise mode:
- Agents make separate decisions for each opponent
- Agents maintain opponent-specific memories and learning states
- Supports true reciprocity and relationship-based strategies
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from abc import ABC, abstractmethod
import logging
import random

try:
    import numpy as np
except ImportError:
    # Fallback for systems without numpy
    class np:
        @staticmethod
        def random():
            return random.random()
        
        class random:
            @staticmethod
            def random():
                return random.random()
            
            @staticmethod
            def choice(seq):
                return random.choice(seq)
            
            @staticmethod
            def seed(s):
                random.seed(s)

# Define get_pairwise_payoffs locally to avoid numpy dependency in utils
def get_pairwise_payoffs(move1: str, move2: str, R=3, S=0, T=5, P=1) -> Tuple[float, float]:
    """Returns payoffs for player 1 and player 2 in a 2-player PD."""
    if move1 == "cooperate" and move2 == "cooperate":
        return R, R
    elif move1 == "cooperate" and move2 == "defect":
        return S, T
    elif move1 == "defect" and move2 == "cooperate":
        return T, S
    elif move1 == "defect" and move2 == "defect":
        return P, P
    else:
        raise ValueError(f"Invalid moves: {move1}, {move2}")

logger = logging.getLogger(__name__)


class OpponentSpecificMemory:
    """Manages memory for interactions with a specific opponent."""
    
    def __init__(self, opponent_id: str, memory_length: int = 10):
        self.opponent_id = opponent_id
        self.memory_length = memory_length
        self.interaction_history = []  # List of (my_move, opponent_move, reward) tuples
        self.cooperation_count = 0
        self.defection_count = 0
        self.total_interactions = 0
        self.cumulative_reward = 0
        
    def add_interaction(self, my_move: str, opponent_move: str, reward: float):
        """Record an interaction with this opponent."""
        self.interaction_history.append({
            'my_move': my_move,
            'opponent_move': opponent_move,
            'reward': reward,
            'round': self.total_interactions
        })
        
        # Keep only recent history
        if len(self.interaction_history) > self.memory_length:
            self.interaction_history.pop(0)
            
        # Update statistics
        if opponent_move == "cooperate":
            self.cooperation_count += 1
        else:
            self.defection_count += 1
            
        self.total_interactions += 1
        self.cumulative_reward += reward
        
    def get_cooperation_rate(self) -> float:
        """Get this opponent's cooperation rate."""
        if self.total_interactions == 0:
            return 0.5  # Neutral assumption for unknown opponents
        return self.cooperation_count / self.total_interactions
    
    def get_recent_cooperation_rate(self, window: int = 5) -> float:
        """Get cooperation rate over recent interactions."""
        if not self.interaction_history:
            return 0.5
            
        recent = self.interaction_history[-window:]
        coop_count = sum(1 for interaction in recent if interaction['opponent_move'] == "cooperate")
        return coop_count / len(recent)
        
    def get_last_move(self) -> Optional[str]:
        """Get opponent's last move."""
        if not self.interaction_history:
            return None
        return self.interaction_history[-1]['opponent_move']
    
    def get_reciprocity_score(self) -> float:
        """Calculate how much this opponent reciprocates our moves."""
        if len(self.interaction_history) < 2:
            return 0.5
            
        reciprocated = 0
        for i in range(1, len(self.interaction_history)):
            my_prev_move = self.interaction_history[i-1]['my_move']
            opp_curr_move = self.interaction_history[i]['opponent_move']
            if my_prev_move == opp_curr_move:
                reciprocated += 1
                
        return reciprocated / (len(self.interaction_history) - 1)
    
    def reset(self):
        """Reset memory for this opponent."""
        self.interaction_history = []
        self.cooperation_count = 0
        self.defection_count = 0
        self.total_interactions = 0
        self.cumulative_reward = 0


class TruePairwiseAgent(ABC):
    """Base class for agents that can make opponent-specific decisions."""
    
    def __init__(self, agent_id: str, memory_length: int = 10):
        self.agent_id = agent_id
        self.memory_length = memory_length
        self.opponent_memories: Dict[str, OpponentSpecificMemory] = {}
        self.total_score = 0
        self.round_number = 0
        
    def get_opponent_memory(self, opponent_id: str) -> OpponentSpecificMemory:
        """Get or create memory for a specific opponent."""
        if opponent_id not in self.opponent_memories:
            self.opponent_memories[opponent_id] = OpponentSpecificMemory(
                opponent_id, self.memory_length
            )
        return self.opponent_memories[opponent_id]
    
    @abstractmethod
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        """Choose an action specifically for this opponent."""
        pass
    
    def update_memory(self, opponent_id: str, my_move: str, opponent_move: str, reward: float):
        """Update memory for a specific opponent interaction."""
        memory = self.get_opponent_memory(opponent_id)
        memory.add_interaction(my_move, opponent_move, reward)
        self.total_score += reward
        
    def get_overall_cooperation_rate(self) -> float:
        """Get cooperation rate across all opponents."""
        total_coops = sum(mem.cooperation_count for mem in self.opponent_memories.values())
        total_interactions = sum(mem.total_interactions for mem in self.opponent_memories.values())
        
        if total_interactions == 0:
            return 0.5
        return total_coops / total_interactions
    
    def reset_episode_memory(self, opponent_id: Optional[str] = None):
        """Reset memory for specific opponent or all opponents."""
        if opponent_id:
            if opponent_id in self.opponent_memories:
                self.opponent_memories[opponent_id].reset()
        else:
            for memory in self.opponent_memories.values():
                memory.reset()
                
    def reset_all(self):
        """Complete reset of agent state."""
        self.opponent_memories.clear()
        self.total_score = 0
        self.round_number = 0


class TruePairwiseTFT(TruePairwiseAgent):
    """Tit-for-Tat agent with opponent-specific responses."""
    
    def __init__(self, agent_id: str, nice: bool = True, forgiving_probability: float = 0.0):
        super().__init__(agent_id)
        self.nice = nice  # Cooperate on first move
        self.forgiving_probability = forgiving_probability
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        memory = self.get_opponent_memory(opponent_id)
        last_move = memory.get_last_move()
        
        if last_move is None:
            # First interaction with this opponent
            return "cooperate" if self.nice else "defect"
        
        # Copy opponent's last move
        if last_move == "defect" and self.forgiving_probability > 0:
            # Possibly forgive
            if np.random.random() < self.forgiving_probability:
                return "cooperate"
                
        return last_move


class TruePairwiseGTFT(TruePairwiseAgent):
    """Generous Tit-for-Tat with opponent-specific generosity."""
    
    def __init__(self, agent_id: str, generosity: float = 0.1):
        super().__init__(agent_id)
        self.generosity = generosity
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        memory = self.get_opponent_memory(opponent_id)
        last_move = memory.get_last_move()
        
        if last_move is None:
            return "cooperate"
            
        if last_move == "defect":
            # Be generous sometimes
            if np.random.random() < self.generosity:
                return "cooperate"
            return "defect"
        
        return "cooperate"


class TruePairwisePavlov(TruePairwiseAgent):
    """Win-Stay/Lose-Shift with opponent-specific tracking."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        memory = self.get_opponent_memory(opponent_id)
        
        if not memory.interaction_history:
            return "cooperate"
            
        last_interaction = memory.interaction_history[-1]
        my_last_move = last_interaction['my_move']
        reward = last_interaction['reward']
        
        # Win-stay: If got good reward (3 or 5), repeat move
        # Lose-shift: If got bad reward (0 or 1), switch move
        if reward >= 3:  # Good outcome
            return my_last_move
        else:  # Bad outcome
            return "defect" if my_last_move == "cooperate" else "cooperate"


class TruePairwiseQLearning(TruePairwiseAgent):
    """Q-Learning agent with separate Q-tables for each opponent."""
    
    def __init__(self, agent_id: str, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1,
                 state_type: str = "memory_enhanced"):
        super().__init__(agent_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_type = state_type
        self.q_tables: Dict[str, Dict[str, Dict[str, float]]] = {}
        
    def get_state_for_opponent(self, opponent_id: str) -> str:
        """Get current state for interaction with specific opponent."""
        memory = self.get_opponent_memory(opponent_id)
        
        if self.state_type == "basic":
            # Simple state based on last move
            last_move = memory.get_last_move()
            if last_move is None:
                return "start"
            return f"last_{last_move}"
            
        elif self.state_type == "proportion":
            # State based on cooperation rate
            coop_rate = memory.get_cooperation_rate()
            if coop_rate < 0.33:
                return "mostly_defect"
            elif coop_rate < 0.67:
                return "mixed"
            else:
                return "mostly_cooperate"
                
        elif self.state_type == "memory_enhanced":
            # State based on recent history pattern
            if len(memory.interaction_history) < 2:
                return "start"
                
            recent_pattern = []
            for i in range(min(3, len(memory.interaction_history))):
                idx = -(i+1)
                recent_pattern.append(memory.interaction_history[idx]['opponent_move'][0])
            
            return ''.join(recent_pattern)
            
        elif self.state_type == "reciprocity":
            # State based on opponent's reciprocity
            reciprocity = memory.get_reciprocity_score()
            if reciprocity < 0.33:
                return "non_reciprocal"
            elif reciprocity < 0.67:
                return "somewhat_reciprocal"
            else:
                return "highly_reciprocal"
                
    def get_q_value(self, opponent_id: str, state: str, action: str) -> float:
        """Get Q-value for specific opponent, state, and action."""
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        if state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][state] = {"cooperate": 0.0, "defect": 0.0}
        return self.q_tables[opponent_id][state][action]
    
    def update_q_value(self, opponent_id: str, state: str, action: str, 
                       reward: float, next_state: str):
        """Update Q-value for specific opponent interaction."""
        current_q = self.get_q_value(opponent_id, state, action)
        
        # Get max Q-value for next state
        next_q_cooperate = self.get_q_value(opponent_id, next_state, "cooperate")
        next_q_defect = self.get_q_value(opponent_id, next_state, "defect")
        max_next_q = max(next_q_cooperate, next_q_defect)
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        if state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][state] = {"cooperate": 0.0, "defect": 0.0}
        
        self.q_tables[opponent_id][state][action] = new_q
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        state = self.get_state_for_opponent(opponent_id)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(["cooperate", "defect"])
            
        # Choose action with highest Q-value
        q_cooperate = self.get_q_value(opponent_id, state, "cooperate")
        q_defect = self.get_q_value(opponent_id, state, "defect")
        
        if q_cooperate > q_defect:
            return "cooperate"
        elif q_defect > q_cooperate:
            return "defect"
        else:
            # Break ties randomly
            return np.random.choice(["cooperate", "defect"])


class TruePairwiseAdaptive(TruePairwiseAgent):
    """Adaptive agent that switches strategies based on opponent behavior."""
    
    def __init__(self, agent_id: str, assessment_period: int = 10):
        super().__init__(agent_id)
        self.assessment_period = assessment_period
        self.opponent_strategies: Dict[str, str] = {}
        
    def assess_opponent_strategy(self, opponent_id: str) -> str:
        """Assess what strategy the opponent is using."""
        memory = self.get_opponent_memory(opponent_id)
        
        if memory.total_interactions < self.assessment_period:
            return "unknown"
            
        # Check for always cooperate/defect
        coop_rate = memory.get_cooperation_rate()
        if coop_rate > 0.95:
            return "always_cooperate"
        elif coop_rate < 0.05:
            return "always_defect"
            
        # Check for reciprocity
        reciprocity = memory.get_reciprocity_score()
        if reciprocity > 0.8:
            return "tit_for_tat"
        elif reciprocity < 0.2:
            return "anti_tit_for_tat"
            
        # Check for random
        recent_pattern = [h['opponent_move'] for h in memory.interaction_history[-10:]]
        switches = sum(1 for i in range(1, len(recent_pattern)) 
                      if recent_pattern[i] != recent_pattern[i-1])
        if switches > 6:  # High switching rate
            return "random"
            
        return "mixed"
        
    def choose_action_for_opponent(self, opponent_id: str, round_num: int) -> str:
        memory = self.get_opponent_memory(opponent_id)
        
        # Reassess strategy periodically
        if memory.total_interactions % self.assessment_period == 0:
            self.opponent_strategies[opponent_id] = self.assess_opponent_strategy(opponent_id)
            
        strategy = self.opponent_strategies.get(opponent_id, "unknown")
        
        # Adapt response based on opponent strategy
        if strategy == "always_cooperate":
            # Exploit cooperator
            return "defect"
        elif strategy == "always_defect":
            # Punish defector
            return "defect"
        elif strategy == "tit_for_tat":
            # Cooperate with reciprocator
            return "cooperate"
        elif strategy == "random":
            # Play cautiously
            return "defect" if memory.get_recent_cooperation_rate() < 0.5 else "cooperate"
        else:
            # Default to reciprocal behavior
            last_move = memory.get_last_move()
            return last_move if last_move else "cooperate"


class TruePairwiseEnvironment:
    """Environment for true pairwise interactions with individual decision-making."""
    
    def __init__(self, agents: List[TruePairwiseAgent], 
                 episodes: int = 1, rounds_per_episode: int = 100,
                 noise_level: float = 0.0, reset_between_episodes: bool = True):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.episodes = episodes
        self.rounds_per_episode = rounds_per_episode
        self.noise_level = noise_level
        self.reset_between_episodes = reset_between_episodes
        self.history = []
        self.round_data = []
        
    def add_noise(self, action: str) -> str:
        """Add noise to action (implementation error)."""
        if self.noise_level > 0 and np.random.random() < self.noise_level:
            return "defect" if action == "cooperate" else "cooperate"
        return action
        
    def play_single_game(self, agent1_id: str, agent2_id: str, round_num: int) -> Dict[str, Any]:
        """Play a single game between two agents."""
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]
        
        # Get current states before moves
        state1 = agent1.get_state_for_opponent(agent2_id) if hasattr(agent1, 'get_state_for_opponent') else None
        state2 = agent2.get_state_for_opponent(agent1_id) if hasattr(agent2, 'get_state_for_opponent') else None
        
        # Agents choose actions
        action1 = agent1.choose_action_for_opponent(agent2_id, round_num)
        action2 = agent2.choose_action_for_opponent(agent1_id, round_num)
        
        # Add noise
        actual_action1 = self.add_noise(action1)
        actual_action2 = self.add_noise(action2)
        
        # Calculate payoffs
        payoff1, payoff2 = get_pairwise_payoffs(actual_action1, actual_action2)
        
        # Update memories
        agent1.update_memory(agent2_id, actual_action1, actual_action2, payoff1)
        agent2.update_memory(agent1_id, actual_action2, actual_action1, payoff2)
        
        # Update Q-values for RL agents
        if isinstance(agent1, TruePairwiseQLearning):
            next_state1 = agent1.get_state_for_opponent(agent2_id)
            agent1.update_q_value(agent2_id, state1, action1, payoff1, next_state1)
            
        if isinstance(agent2, TruePairwiseQLearning):
            next_state2 = agent2.get_state_for_opponent(agent1_id)
            agent2.update_q_value(agent1_id, state2, action2, payoff2, next_state2)
            
        return {
            'round': round_num,
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'action1': actual_action1,
            'action2': actual_action2,
            'intended_action1': action1,
            'intended_action2': action2,
            'payoff1': payoff1,
            'payoff2': payoff2,
            'noise_applied': (action1 != actual_action1 or action2 != actual_action2)
        }
        
    def run_round(self, round_num: int) -> List[Dict[str, Any]]:
        """Run one round of all pairwise interactions."""
        round_results = []
        agent_ids = list(self.agents.keys())
        
        # Each pair plays once
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                result = self.play_single_game(agent_ids[i], agent_ids[j], round_num)
                round_results.append(result)
                
        return round_results
        
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run one episode of the game."""
        episode_results = []
        
        for round_num in range(self.rounds_per_episode):
            round_results = self.run_round(round_num)
            episode_results.extend(round_results)
            
            # Store round-level data
            cooperation_count = sum(1 for r in round_results 
                                  for action in [r['action1'], r['action2']] 
                                  if action == "cooperate")
            total_actions = len(round_results) * 2
            
            self.round_data.append({
                'episode': episode_num,
                'round': round_num,
                'cooperation_rate': cooperation_count / total_actions if total_actions > 0 else 0,
                'total_payoff': sum(r['payoff1'] + r['payoff2'] for r in round_results)
            })
            
        return {
            'episode': episode_num,
            'results': episode_results,
            'agent_scores': {aid: agent.total_score for aid, agent in self.agents.items()}
        }
        
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        simulation_results = []
        
        for episode in range(self.episodes):
            # Reset memories between episodes if configured
            if episode > 0 and self.reset_between_episodes:
                for agent in self.agents.values():
                    agent.reset_episode_memory()
                    
            episode_result = self.run_episode(episode)
            simulation_results.append(episode_result)
            
        # Compile final statistics
        final_stats = self.compile_statistics()
        
        return {
            'episodes': simulation_results,
            'round_data': self.round_data,
            'final_statistics': final_stats,
            'configuration': {
                'episodes': self.episodes,
                'rounds_per_episode': self.rounds_per_episode,
                'noise_level': self.noise_level,
                'reset_between_episodes': self.reset_between_episodes,
                'agents': [{'id': aid, 'type': type(agent).__name__} 
                          for aid, agent in self.agents.items()]
            }
        }
        
    def compile_statistics(self) -> Dict[str, Any]:
        """Compile comprehensive statistics from the simulation."""
        stats = {}
        
        # Agent-level statistics
        for agent_id, agent in self.agents.items():
            agent_stats = {
                'total_score': agent.total_score,
                'overall_cooperation_rate': agent.get_overall_cooperation_rate(),
                'opponent_relationships': {}
            }
            
            # Opponent-specific statistics
            for opp_id, memory in agent.opponent_memories.items():
                agent_stats['opponent_relationships'][opp_id] = {
                    'cooperation_rate': memory.get_cooperation_rate(),
                    'reciprocity_score': memory.get_reciprocity_score(),
                    'total_interactions': memory.total_interactions,
                    'cumulative_reward': memory.cumulative_reward,
                    'average_reward': (memory.cumulative_reward / memory.total_interactions 
                                     if memory.total_interactions > 0 else 0)
                }
                
            stats[agent_id] = agent_stats
            
        # Global statistics
        all_cooperations = sum(sum(m.cooperation_count for m in a.opponent_memories.values()) 
                              for a in self.agents.values())
        all_interactions = sum(sum(m.total_interactions for m in a.opponent_memories.values()) 
                              for a in self.agents.values())
        
        stats['global'] = {
            'overall_cooperation_rate': all_cooperations / all_interactions if all_interactions > 0 else 0,
            'total_interactions': all_interactions // 2,  # Divide by 2 as each interaction is counted twice
            'average_score': np.mean([a.total_score for a in self.agents.values()])
        }
        
        return stats