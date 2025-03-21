# agents.py
import random
import math

class Strategy:
    """Base Strategy class that all specific strategies inherit from."""
    
    def choose_move(self, agent, neighbors):
        """Choose the next move for the agent.
        
        Args:
            agent: The agent making the decision
            neighbors: List of neighbor agent IDs
            
        Returns:
            String: "cooperate" or "defect"
        """
        raise NotImplementedError("Subclasses must implement choose_move")
    
    def update(self, agent, action, reward, neighbor_moves):
        """Update internal state based on the results of the last action.
        
        Args:
            agent: The agent that performed the action
            action: The action that was taken ("cooperate" or "defect")
            reward: The reward received
            neighbor_moves: Dictionary of neighbor IDs to their moves
        """
        pass  # Default implementation does nothing


class RandomStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        return random.choice(["cooperate", "defect"])


class AlwaysCooperateStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        return "cooperate"


class AlwaysDefectStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        return "defect"


class TitForTatStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return "cooperate"  # Cooperate on first move
        
        # Get a random neighbor's move from the last round
        last_round = agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        if neighbor_moves:
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            return neighbor_moves[random_neighbor_id]
        return "cooperate"  # Default to cooperate if no neighbors


class GenerousTitForTatStrategy(Strategy):
    def __init__(self, generosity=0.1):
        self.generosity = generosity
        
    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return "cooperate"
        
        last_round = agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        if neighbor_moves:
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            move = neighbor_moves[random_neighbor_id]
            # Occasionally forgive defection
            if move == "defect" and random.random() < self.generosity:
                return "cooperate"
            return move
        return "cooperate"


class SuspiciousTitForTatStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return "defect"  # Start with defection
        
        last_round = agent.memory[-1]
        neighbor_moves = last_round['neighbor_moves']
        if neighbor_moves:
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            return neighbor_moves[random_neighbor_id]
        return "defect"  # Default to defect if no neighbors


class PavlovStrategy(Strategy):
    def __init__(self, initial_move="cooperate"):
        self.initial_move = initial_move
        
    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return self.initial_move
        
        last_round = agent.memory[-1]
        last_move = last_round['my_move']
        last_reward = last_round['reward']
        
        # Win-stay, lose-shift
        if last_reward >= 3:  # High reward threshold
            return last_move  # Keep the same move
        else:
            return "defect" if last_move == "cooperate" else "cooperate"  # Switch


class RandomProbStrategy(Strategy):
    def __init__(self, prob_coop=0.5):
        self.prob_coop = prob_coop
        
    def choose_move(self, agent, neighbors):
        return "cooperate" if random.random() < self.prob_coop else "defect"


class QLearningStrategy(Strategy):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
    def choose_move(self, agent, neighbors):
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(["cooperate", "defect"])
        
        # Exploitation - choose action with highest Q-value
        return "cooperate" if agent.q_values["cooperate"] >= agent.q_values["defect"] else "defect"
    
    def update(self, agent, action, reward, neighbor_moves):
        # Find the best next action
        best_next_action = "cooperate" if agent.q_values["cooperate"] > agent.q_values["defect"] else "defect"
        best_next_q = agent.q_values[best_next_action]
        
        # Update Q-value for the chosen action using the Q-learning formula
        agent.q_values[action] = (1 - self.learning_rate) * agent.q_values[action] + \
                                self.learning_rate * (reward + self.discount_factor * best_next_q)


class AdaptiveQLearningStrategy(QLearningStrategy):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, min_epsilon=0.01, decay_rate=0.99):
        super().__init__(learning_rate, discount_factor, epsilon)
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
    def update(self, agent, action, reward, neighbor_moves):
        super().update(agent, action, reward, neighbor_moves)
        
        # Decay epsilon over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


# Factory function to create strategy objects
def create_strategy(strategy_type, **kwargs):
    """Factory function to create strategy objects based on strategy type."""
    strategies = {
        "random": RandomStrategy,
        "always_cooperate": AlwaysCooperateStrategy,
        "always_defect": AlwaysDefectStrategy,
        "tit_for_tat": TitForTatStrategy,
        "generous_tit_for_tat": GenerousTitForTatStrategy,
        "suspicious_tit_for_tat": SuspiciousTitForTatStrategy,
        "pavlov": PavlovStrategy,
        "randomprob": RandomProbStrategy,
        "q_learning": QLearningStrategy,
        "q_learning_adaptive": AdaptiveQLearningStrategy,
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = strategies[strategy_type]
    return strategy_class(**kwargs)


class Agent:
    def __init__(self, agent_id, strategy="random", learning_rate=0.1, discount_factor=0.9, epsilon=0.1,
                 generosity=0.05, initial_move="cooperate", prob_coop=0.5):
        self.agent_id = agent_id
        self.strategy_type = strategy
        
        # Create strategy parameters dictionary
        strategy_params = {}
        if strategy == "generous_tit_for_tat":
            strategy_params["generosity"] = generosity
        elif strategy == "pavlov":
            strategy_params["initial_move"] = initial_move
        elif strategy == "randomprob":
            strategy_params["prob_coop"] = prob_coop
        elif strategy in ["q_learning", "q_learning_adaptive"]:
            strategy_params.update({
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon": epsilon
            })
            if strategy == "q_learning_adaptive":
                strategy_params.update({
                    "min_epsilon": 0.01,
                    "decay_rate": 0.99
                })
                
        # Create the strategy object
        self.strategy = create_strategy(strategy, **strategy_params)
        
        # Initialize agent state
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}
        
    def choose_move(self, neighbors):
        """Choose the next move based on the agent's strategy."""
        return self.strategy.choose_move(self, neighbors)
        
    def update_q_value(self, action, reward, next_state_actions):
        """Update Q-values if the agent uses Q-learning."""
        if self.strategy_type in ("q_learning", "q_learning_adaptive"):
            self.strategy.update(self, action, reward, next_state_actions)
            
    def update_memory(self, my_move, neighbor_moves, reward):
        """Update the agent's memory with the result of the last round."""
        self.memory.append({
            'my_move': my_move,
            'neighbor_moves': neighbor_moves,
            'reward': reward,
        })
        
    def reset(self):
        """Reset the agent's state for a new simulation."""
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}