# agents.py
import random
import math
from collections import deque

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
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, state_type="basic"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_type = state_type
        
    def _get_current_state(self, agent):
        """Get the current state representation for Q-learning.
        
        Args:
            agent: The agent making the decision
            
        Returns:
            A hashable state representation (string or tuple)
        """
        if not agent.memory:
            return 'initial'  # Special state for the first move
        
        if self.state_type == "basic":
            return 'standard'  # Default state for backward compatibility
            
        last_round_info = agent.memory[-1]  # Memory stores results of the round *leading* to this state
        neighbor_moves = last_round_info['neighbor_moves']
        
        num_neighbors = len(neighbor_moves)
        if num_neighbors == 0:
            return 'no_neighbors'  # Special state if isolated
            
        num_cooperating_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")
        coop_proportion = num_cooperating_neighbors / num_neighbors
        
        if self.state_type == "proportion":
            # Use the exact proportion as state
            return (coop_proportion,)
            
        elif self.state_type == "proportion_discretized":
            # Discretize the proportion into bins (5 bins)
            if coop_proportion <= 0.2:
                state_feature = 0.2
            elif coop_proportion <= 0.4:
                state_feature = 0.4
            elif coop_proportion <= 0.6:
                state_feature = 0.6
            elif coop_proportion <= 0.8:
                state_feature = 0.8
            else:
                state_feature = 1.0
            return (state_feature,)
            
        elif self.state_type == "count":
            # Use absolute count of cooperating neighbors
            return (num_cooperating_neighbors,)
            
        elif self.state_type == "threshold":
            # Binary feature indicating if majority cooperated
            return (coop_proportion > 0.5,)
            
        elif self.state_type == "memory_enhanced":
            # Include agent's own last move
            own_last_move = last_round_info['my_move']
            # Convert to binary for compactness (1 for cooperate, 0 for defect)
            own_move_binary = 1 if own_last_move == "cooperate" else 0
            
            # Discretize neighbor cooperation
            if coop_proportion <= 0.33:
                neighbor_state = 0  # Low cooperation
            elif coop_proportion <= 0.67:
                neighbor_state = 1  # Medium cooperation
            else:
                neighbor_state = 2  # High cooperation
                
            return (own_move_binary, neighbor_state)
            
        # Default fallback
        return 'standard'
    
    def choose_move(self, agent, neighbors):
        # Get current state
        current_state = self._get_current_state(agent)
        agent.last_state_representation = current_state
        
        # Ensure state exists in Q-table, initialize if not
        if current_state not in agent.q_values:
            agent.q_values[current_state] = {"cooperate": 0.0, "defect": 0.0}
        
        # Exploration (epsilon-greedy)
        if random.random() < self.epsilon:
            return random.choice(["cooperate", "defect"])
        
        # Exploitation - choose action with highest Q-value
        if agent.q_values[current_state]["cooperate"] >= agent.q_values[current_state]["defect"]:
            return "cooperate"
        else:
            return "defect"
    
    def update(self, agent, action, reward, neighbor_moves):
        # Use the state that was stored during choose_move
        state_executed = agent.last_state_representation
        if state_executed is None:
            # This should not happen after the first round, but just in case
            return
            
        # Calculate the state for the next step (based on the current memory after this round)
        next_state = self._get_current_state(agent)
        
        # Ensure next state exists in Q-table, initialize if not
        if next_state not in agent.q_values:
            agent.q_values[next_state] = {"cooperate": 0.0, "defect": 0.0}
            
        # Find max Q-value for the next state
        best_next_q = max(agent.q_values[next_state].values())
        
        # Update Q-value for the chosen action using the Q-learning formula
        current_q = agent.q_values[state_executed][action]
        agent.q_values[state_executed][action] = (1 - self.learning_rate) * current_q + \
                                self.learning_rate * (reward + self.discount_factor * best_next_q)


class AdaptiveQLearningStrategy(QLearningStrategy):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, 
                 min_epsilon=0.01, decay_rate=0.99, state_type="basic"):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
    def update(self, agent, action, reward, neighbor_moves):
        super().update(agent, action, reward, neighbor_moves)
        
        # Decay epsilon over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class LRAQLearningStrategy(QLearningStrategy):
    """Learning Rate Adjusting Q-Learning Strategy.
    
    Adjusts learning rate based on cooperation levels of neighbors.
    Increases learning rate after cooperative outcomes, decreases after defection.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, 
                 increase_rate=0.1, decrease_rate=0.05, state_type="proportion_discretized"):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.increase_rate = increase_rate  # For cooperation
        self.decrease_rate = decrease_rate  # For defection
        self.base_learning_rate = learning_rate
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.9
        
    def update(self, agent, action, reward, neighbor_moves):
        # Count cooperating neighbors
        coop_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")
        total_neighbors = len(neighbor_moves) if neighbor_moves else 1
        
        # Adjust learning rate based on cooperation level and agent's own action
        if action == "cooperate" and coop_neighbors / total_neighbors > 0.5:
            # Cooperating with cooperators - increase learning rate
            self.learning_rate = min(self.max_learning_rate, 
                                    self.learning_rate + self.increase_rate)
        elif action == "defect" and coop_neighbors / total_neighbors > 0.5:
            # Defecting against cooperators - decrease learning rate
            self.learning_rate = max(self.min_learning_rate, 
                                    self.learning_rate - self.decrease_rate)
        
        # Use the adjusted learning rate for the standard Q-update
        super().update(agent, action, reward, neighbor_moves)
        
        # Slowly return to base rate when not adjusted (regression to mean)
        if self.learning_rate > self.base_learning_rate:
            self.learning_rate = max(self.base_learning_rate, 
                                     self.learning_rate - 0.01)
        elif self.learning_rate < self.base_learning_rate:
            self.learning_rate = min(self.base_learning_rate, 
                                     self.learning_rate + 0.01)


class HystereticQLearningStrategy(QLearningStrategy):
    """Hysteretic Q-Learning Strategy.
    
    Uses different learning rates for positive and negative experiences.
    More optimistic updating that is resistant to occasional defections.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, 
                 beta=0.01, state_type="proportion_discretized"):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.beta = beta  # Lower learning rate for negative experiences
        
    def update(self, agent, action, reward, neighbor_moves):
        # Get the state representation that was used for the action
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
            
        # Calculate the next state
        next_state = self._get_current_state(agent)
        
        # Ensure next state exists in Q-table
        if next_state not in agent.q_values:
            agent.q_values[next_state] = {"cooperate": 0.0, "defect": 0.0}
            
        # Calculate the target Q-value
        best_next_q = max(agent.q_values[next_state].values())
        target = reward + self.discount_factor * best_next_q
        current = agent.q_values[state_executed][action]
        
        # Use different learning rates for positive and negative updates
        if target > current:  # Positive experience
            agent.q_values[state_executed][action] = (1 - self.learning_rate) * current + \
                                                     self.learning_rate * target
        else:  # Negative experience - learn more slowly
            agent.q_values[state_executed][action] = (1 - self.beta) * current + \
                                                     self.beta * target


class WolfPHCStrategy(QLearningStrategy):
    """Win or Learn Fast Policy Hill-Climbing.
    
    Adjusts learning rate based on whether the agent is "winning" or "losing"
    compared to its average historical performance.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1,
                 alpha_win=0.05, alpha_lose=0.2, alpha_avg=0.01, state_type="proportion_discretized"):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.alpha_win = alpha_win
        self.alpha_lose = alpha_lose
        self.alpha_avg = alpha_avg
        self.average_payoff = 0.0
        self.policy_counts = {}  # {state: {'cooperate': count, 'defect': count}}
        self.average_policy = {}  # {state: {'cooperate': prob, 'defect': prob}}
        
    def _update_average_policy(self, state):
        """Update the average policy for a given state."""
        if state not in self.policy_counts:
            return
            
        total_actions = sum(self.policy_counts[state].values())
        if total_actions == 0:
            return
            
        if state not in self.average_policy:
            self.average_policy[state] = {'cooperate': 0.5, 'defect': 0.5}
            
        for action in ['cooperate', 'defect']:
            action_count = self.policy_counts[state][action]
            action_prob = action_count / total_actions
            current_avg = self.average_policy[state][action]
            # Exponential moving average of the policy
            self.average_policy[state][action] = (1 - self.alpha_avg) * current_avg + \
                                               self.alpha_avg * action_prob
        
    def choose_move(self, agent, neighbors):
        # Get current state
        current_state = self._get_current_state(agent)
        agent.last_state_representation = current_state
        
        # Ensure state exists in Q-table, initialize if not
        if current_state not in agent.q_values:
            agent.q_values[current_state] = {"cooperate": 0.0, "defect": 0.0}
            
        # Initialize policy counts if needed
        if current_state not in self.policy_counts:
            self.policy_counts[current_state] = {'cooperate': 0, 'defect': 0}
            self.average_policy[current_state] = {'cooperate': 0.5, 'defect': 0.5}
            
        # Exploration (epsilon-greedy)
        if random.random() < self.epsilon:
            chosen_action = random.choice(["cooperate", "defect"])
        else:
            # Exploitation - choose action with highest Q-value
            if agent.q_values[current_state]["cooperate"] >= agent.q_values[current_state]["defect"]:
                chosen_action = "cooperate"
            else:
                chosen_action = "defect"
                
        # Increment count for the chosen action
        self.policy_counts[current_state][chosen_action] += 1
        
        return chosen_action
        
    def update(self, agent, action, reward, neighbor_moves):
        # Get the state that was used for the action
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
            
        # Update average payoff
        self.average_payoff = (1 - self.alpha_avg) * self.average_payoff + self.alpha_avg * reward
        
        # Update average policy for the state
        self._update_average_policy(state_executed)
        
        # Calculate V_pi and V_pi_avg for the state
        q_coop = agent.q_values[state_executed]['cooperate']
        q_def = agent.q_values[state_executed]['defect']
        
        # Estimate current policy (epsilon-greedy for simplicity)
        pi_coop = 1.0 if q_coop >= q_def else 0.0
        pi_def = 1.0 - pi_coop
        V_pi = pi_coop * q_coop + pi_def * q_def
        
        # Get average policy values
        avg_pi_coop = self.average_policy[state_executed]['cooperate']
        avg_pi_def = self.average_policy[state_executed]['defect']
        V_pi_avg = avg_pi_coop * q_coop + avg_pi_def * q_def
        
        # Determine learning rate - faster when losing, slower when winning
        current_alpha = self.alpha_lose if V_pi < V_pi_avg else self.alpha_win
        
        # Calculate next state and update Q-values
        next_state = self._get_current_state(agent)
        if next_state not in agent.q_values:
            agent.q_values[next_state] = {"cooperate": 0.0, "defect": 0.0}
            
        # Find max Q-value for next state
        best_next_q = max(agent.q_values[next_state].values())
        
        # Update Q-value using the dynamic learning rate
        current_q = agent.q_values[state_executed][action]
        agent.q_values[state_executed][action] = (1 - current_alpha) * current_q + \
                                               current_alpha * (reward + self.discount_factor * best_next_q)


class UCB1QLearningStrategy(QLearningStrategy):
    """Upper Confidence Bound Q-Learning Strategy.
    
    Uses UCB1 algorithm for smarter exploration based on uncertainty.
    """
    def __init__(self, exploration_constant=2.0, learning_rate=0.1, 
                 discount_factor=0.9, state_type="proportion_discretized"):
        super().__init__(learning_rate, discount_factor, 0.0, state_type)  # Set epsilon to 0, using UCB instead
        self.exploration_constant = exploration_constant
        self.action_counts = {}  # {state: {'cooperate': count, 'defect': count}}
        self.total_steps = 0
        
    def choose_move(self, agent, neighbors):
        self.total_steps += 1
        current_state = self._get_current_state(agent)
        agent.last_state_representation = current_state
        
        # Initialize Q-values and counts if state is new
        if current_state not in agent.q_values:
            agent.q_values[current_state] = {"cooperate": 0.0, "defect": 0.0}
        if current_state not in self.action_counts:
            self.action_counts[current_state] = {'cooperate': 0, 'defect': 0}
            
        # Calculate UCB values
        ucb_values = {}
        total_log = math.log(self.total_steps + 1)  # Pre-compute logarithm
        
        for action in ["cooperate", "defect"]:
            q_value = agent.q_values[current_state][action]
            count = self.action_counts[current_state][action]
            
            if count == 0:
                # If an action hasn't been tried, prioritize it
                ucb_values[action] = float('inf')
            else:
                # UCB formula: Q-value + C * sqrt(log(total_steps) / count)
                exploration_bonus = self.exploration_constant * math.sqrt(total_log / count)
                ucb_values[action] = q_value + exploration_bonus
                
        # Choose action with highest UCB value
        if ucb_values['cooperate'] == float('inf') and ucb_values['defect'] == float('inf'):
            # First time in this state, choose randomly
            chosen_action = random.choice(['cooperate', 'defect'])
        elif ucb_values['cooperate'] >= ucb_values['defect']:
            chosen_action = 'cooperate'
        else:
            chosen_action = 'defect'
            
        return chosen_action
        
    def update(self, agent, action, reward, neighbor_moves):
        state_executed = agent.last_state_representation
        if state_executed is None:
            return
            
        # Increment count for the executed action
        if state_executed not in self.action_counts:
            self.action_counts[state_executed] = {'cooperate': 0, 'defect': 0}
        self.action_counts[state_executed][action] += 1
        
        # Standard Q-learning update
        super().update(agent, action, reward, neighbor_moves)


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
        "lra_q": LRAQLearningStrategy,
        "hysteretic_q": HystereticQLearningStrategy,
        "wolf_phc": WolfPHCStrategy,
        "ucb1_q": UCB1QLearningStrategy,
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = strategies[strategy_type]
    return strategy_class(**kwargs)


class Agent:
    def __init__(self, agent_id, strategy="random", memory_length=10, 
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1,
                 state_type="proportion_discretized", generosity=0.05, 
                 initial_move="cooperate", prob_coop=0.5,
                 increase_rate=0.1, decrease_rate=0.05, beta=0.01,
                 alpha_win=0.05, alpha_lose=0.2, alpha_avg=0.01,
                 exploration_constant=2.0):
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
                "epsilon": epsilon,
                "state_type": state_type
            })
            if strategy == "q_learning_adaptive":
                strategy_params.update({
                    "min_epsilon": 0.01,
                    "decay_rate": 0.99
                })
        elif strategy == "lra_q":
            strategy_params.update({
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon": epsilon,
                "increase_rate": increase_rate,
                "decrease_rate": decrease_rate,
                "state_type": state_type
            })
        elif strategy == "hysteretic_q":
            strategy_params.update({
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon": epsilon,
                "beta": beta,
                "state_type": state_type
            })
        elif strategy == "wolf_phc":
            strategy_params.update({
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon": epsilon,
                "alpha_win": alpha_win,
                "alpha_lose": alpha_lose, 
                "alpha_avg": alpha_avg,
                "state_type": state_type
            })
        elif strategy == "ucb1_q":
            strategy_params.update({
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "exploration_constant": exploration_constant,
                "state_type": state_type
            })
                
        # Create the strategy object
        self.strategy = create_strategy(strategy, **strategy_params)
        
        # Initialize agent state
        self.score = 0
        self.memory = deque(maxlen=memory_length)  # Use deque with max length
        self.q_values = {}  # Change to empty dict to support state-based Q-values
        self.last_state_representation = None  # Track state for Q-learning
        
    def choose_move(self, neighbors):
        """Choose the next move based on the agent's strategy."""
        return self.strategy.choose_move(self, neighbors)
        
    def update_q_value(self, action, reward, next_state_actions):
        """Update Q-values if the agent uses Q-learning."""
        if self.strategy_type in ("q_learning", "q_learning_adaptive", 
                                 "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
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
        self.memory.clear()
        self.q_values = {}
        self.last_state_representation = None