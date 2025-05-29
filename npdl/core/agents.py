# agents.py
import random
import math
from collections import deque

from networkx.classes import common_neighbors
from typing import List, Dict, Tuple, Any, Optional, Union, Hashable


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

    def update(
        self, agent: "Agent", action: str, reward: float, neighbor_moves: Dict[int, str]
    ) -> None:
        """Update the agent's Q-values based on the received reward.

        Implements the Q-learning update rule with dynamic learning rate adjustment
        based on the cooperation levels of neighbors.

        Args:
            agent: The agent that performed the action
            action: The action that was taken ("cooperate" or "defect")
            reward: The reward received
            neighbor_moves: Dictionary mapping neighbor IDs to their moves
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
    def choose_move(self, agent, neighbors):  # neighbors arg kept for API consistency
        """Choose the move for a TitForTat agent, handling both interaction modes.
        This version prioritizes specific opponent moves for pairwise mode.
        """
        if not agent.memory:
            return "cooperate"  # Cooperate on first move

        last_round = agent.memory[-1]
        interaction_context = last_round.get('neighbor_moves',
                                             {})  # Using 'neighbor_moves' as the key for interaction context

        # CASE 1: Pairwise mode with specific opponent moves
        if isinstance(interaction_context, dict) and 'specific_opponent_moves' in interaction_context:
            specific_moves = interaction_context['specific_opponent_moves']
            if specific_moves:  # Check if there were any opponents
                # True TFT: defect if ANY specific opponent defected against this agent last round
                if any(move == "defect" for move in specific_moves.values()):
                    return "defect"
                return "cooperate"  # All specific opponents cooperated
            return "cooperate"  # No specific opponents, default to coop

        # CASE 2: Pairwise mode fallback (old format) or general aggregate if no specific
        elif isinstance(interaction_context, dict) and 'opponent_coop_proportion' in interaction_context:
            coop_proportion = interaction_context['opponent_coop_proportion']
            # For TFT, be strict: only cooperate if all (or almost all) opponents cooperated
            return "cooperate" if coop_proportion >= 0.99 else "defect"

        # CASE 3: Standard neighborhood mode
        elif isinstance(interaction_context, dict) and interaction_context:  # Ensure it's a non-empty dict
            # Original logic: pick a random neighbor's move
            # This is a simplification for N-IPD TFT. A more robust N-IPD TFT might
            # defect if *any* neighbor defected, or based on majority.
            # Keeping original random neighbor logic for now for neighborhood mode.
            random_neighbor_id = random.choice(list(interaction_context.keys()))
            return interaction_context[random_neighbor_id]

        # Default if no memory, or empty/unrecognized interaction_context
        return "cooperate"


class GenerousTitForTatStrategy(Strategy):
    def __init__(self, generosity=0.1):
        self.generosity = generosity

    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return "cooperate"

        last_round = agent.memory[-1]
        interaction_context = last_round.get('neighbor_moves', {})

        # CASE 1: Pairwise mode with specific opponent moves
        if isinstance(interaction_context, dict) and 'specific_opponent_moves' in interaction_context:
            specific_moves = interaction_context['specific_opponent_moves']
            if specific_moves:
                if any(move == "defect" for move in specific_moves.values()):
                    return "cooperate" if random.random() < self.generosity else "defect"
                return "cooperate"
            return "cooperate"

        # CASE 2: Pairwise mode fallback or general aggregate
        elif isinstance(interaction_context, dict) and 'opponent_coop_proportion' in interaction_context:
            coop_proportion = interaction_context['opponent_coop_proportion']
            if coop_proportion < 0.99:  # If not all cooperated
                return "cooperate" if random.random() < self.generosity else "defect"
            return "cooperate"

        # CASE 3: Standard neighborhood mode
        elif isinstance(interaction_context, dict) and interaction_context:
            random_neighbor_id = random.choice(list(interaction_context.keys()))
            move = interaction_context[random_neighbor_id]
            if move == "defect" and random.random() < self.generosity:
                return "cooperate"
            return move

        return "cooperate"


class SuspiciousTitForTatStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return "defect"  # Start with defection

        last_round = agent.memory[-1]
        interaction_context = last_round.get('neighbor_moves', {})

        # CASE 1: Pairwise mode with specific opponent moves
        if isinstance(interaction_context, dict) and 'specific_opponent_moves' in interaction_context:
            specific_moves = interaction_context['specific_opponent_moves']
            if specific_moves:
                if any(move == "defect" for move in specific_moves.values()):
                    return "defect"
                return "cooperate"
            return "defect"  # No specific opponents, maintain suspicion

        # CASE 2: Pairwise mode fallback or general aggregate
        elif isinstance(interaction_context, dict) and 'opponent_coop_proportion' in interaction_context:
            coop_proportion = interaction_context['opponent_coop_proportion']
            return "cooperate" if coop_proportion >= 0.99 else "defect"

        # CASE 3: Standard neighborhood mode
        elif isinstance(interaction_context, dict) and interaction_context:
            random_neighbor_id = random.choice(list(interaction_context.keys()))
            return interaction_context[random_neighbor_id]

        return "defect"


class TitForTwoTatsStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        if len(agent.memory) < 2:
            return "cooperate"

        last_round = agent.memory[-1]
        prev_round = agent.memory[-2]

        last_interaction_context = last_round.get('neighbor_moves', {})
        prev_interaction_context = prev_round.get('neighbor_moves', {})

        # CASE 1: Pairwise mode with specific opponent moves
        if isinstance(last_interaction_context, dict) and 'specific_opponent_moves' in last_interaction_context and \
                isinstance(prev_interaction_context, dict) and 'specific_opponent_moves' in prev_interaction_context:

            last_specific = last_interaction_context['specific_opponent_moves']
            prev_specific = prev_interaction_context['specific_opponent_moves']

            if last_specific and prev_specific:  # Ensure there are opponent records
                # Check common opponents or all opponents from last round if list changes
                # For simplicity, check if ANY opponent defected twice.
                # A more complex TF2T would track each opponent's last two moves individually.
                # This current logic for TF2T in pairwise will defect if any single opponent
                # has defected against it in both of the previous two rounds.
                for opp_id in last_specific:
                    if opp_id in prev_specific:
                        if last_specific[opp_id] == "defect" and prev_specific[opp_id] == "defect":
                            return "defect"
                return "cooperate"  # No such opponent found
            return "cooperate"  # Not enough specific history

        # CASE 2: Pairwise mode fallback or general aggregate
        elif isinstance(last_interaction_context, dict) and 'opponent_coop_proportion' in last_interaction_context and \
                isinstance(prev_interaction_context, dict) and 'opponent_coop_proportion' in prev_interaction_context:

            last_coop_prop = last_interaction_context['opponent_coop_proportion']
            prev_coop_prop = prev_interaction_context['opponent_coop_proportion']
            # Defect if aggregate cooperation was low (<0.5 implies majority defected) in both previous rounds
            if last_coop_prop < 0.5 and prev_coop_prop < 0.5:
                return "defect"
            return "cooperate"

        # CASE 3: Standard neighborhood mode
        elif isinstance(last_interaction_context, dict) and last_interaction_context and \
                isinstance(prev_interaction_context, dict) and prev_interaction_context:

            # Original logic: Pick a random common neighbor's two past moves
            common_neighbors = list(set(last_interaction_context.keys()) & set(prev_interaction_context.keys()))
            if not common_neighbors:
                return "cooperate"

            random_neighbor_id = random.choice(common_neighbors)
            last_opponent_move = last_interaction_context.get(random_neighbor_id, "cooperate")
            prev_opponent_move = prev_interaction_context.get(random_neighbor_id, "cooperate")

            if last_opponent_move == "defect" and prev_opponent_move == "defect":
                return "defect"
            return "cooperate"

        return "cooperate"


class PavlovStrategy(Strategy):
    def __init__(self, initial_move="cooperate"):
        self.initial_move = initial_move

    def choose_move(self, agent, neighbors):
        if not agent.memory:
            return self.initial_move

        last_round = agent.memory[-1]
        last_move = last_round["my_move"]
        last_reward = last_round["reward"]

        # Handle pairwise case where reward might be different scale
        # In pairwise mode, the reward might be averaged across multiple opponents
        if "opponent_coop_proportion" in last_round.get("neighbor_moves", {}):
            # Adjust threshold - if average reward is better than P, keep move
            # This should work with typical PD values (e.g., R=3, S=0, T=5, P=1)
            if last_reward > 1.5:  # Above midpoint between P(1) and R(3)
                return last_move  # Keep the same move
            else:
                return "defect" if last_move == "cooperate" else "cooperate"  # Switch

        # Standard win-stay, lose-shift
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
    def __init__(
        self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, state_type="basic"
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_type = state_type

    def _get_current_state(self, agent: "Agent") -> Hashable:
        """Get the current state representation for Q-learning.
        Handles both neighborhood and pairwise (aggregate) interaction context.
        """
        if not agent.memory:
            return "initial"

        last_round_info = agent.memory[-1]
        interaction_context = last_round_info.get('neighbor_moves', {}) # Use 'neighbor_moves' as key

        if self.state_type == "basic":
            return 'standard'

        # Check for pairwise mode aggregate information first
        if isinstance(interaction_context, dict) and 'opponent_coop_proportion' in interaction_context:
            coop_proportion = interaction_context['opponent_coop_proportion']

            if self.state_type == "proportion":
                # Use the exact proportion as state
                return (coop_proportion,)

            elif self.state_type == "proportion_discretized":
                if coop_proportion <= 0.2: state_feature = 0.2
                elif coop_proportion <= 0.4: state_feature = 0.4
                elif coop_proportion <= 0.6: state_feature = 0.6
                elif coop_proportion <= 0.8: state_feature = 0.8
                else: state_feature = 1.0
                return (state_feature,)
            elif self.state_type == "memory_enhanced":
                own_last_move = agent.memory[-1]['my_move']
                own_prev_move = agent.memory[-2]['my_move'] if len(agent.memory) >= 2 else own_last_move
                own_last_bin = 1 if own_last_move == "cooperate" else 0
                own_prev_bin = 1 if own_prev_move == "cooperate" else 0
                if coop_proportion <= 0.33: opponent_state_bin = 0
                elif coop_proportion <= 0.67: opponent_state_bin = 1
                else: opponent_state_bin = 2
                return (own_last_bin, own_prev_bin, opponent_state_bin)

            elif self.state_type == "count":
                # For pairwise mode, convert proportion to a discretized "count" representation
                # This helps maintain compatibility with the standard mode
                num_bins = 10  # Discretize into 10 bins
                discretized_count = int(coop_proportion * num_bins)
                return (discretized_count,)

            elif self.state_type == "threshold":
                # Binary feature indicating if majority cooperated
                return (coop_proportion > 0.5,)

            else:  # Fallback for unhandled pairwise state type
                return ('pairwise_agg', round(coop_proportion, 1))

        elif isinstance(interaction_context, dict):
            num_neighbors = len(interaction_context)
            if num_neighbors == 0:
                return 'no_neighbors'

            num_cooperating_neighbors = sum(1 for move in interaction_context.values() if move == "cooperate")
            coop_proportion = num_cooperating_neighbors / num_neighbors

        if self.state_type == "proportion":
            # Use the exact proportion as state
            return (coop_proportion,)

        elif self.state_type == "memory_enhanced":
            # Example: Own last 2 moves + discretized neighbor coop proportion
            # Determine own moves (default to C if not enough history)
            own_last_move = "cooperate"
            own_prev_move = "cooperate"
            if len(agent.memory) >= 1:
                own_last_move = agent.memory[-1]["my_move"]
            if len(agent.memory) >= 2:
                own_prev_move = agent.memory[-2]["my_move"]

            own_last_bin = 1 if own_last_move == "cooperate" else 0
            own_prev_bin = 1 if own_prev_move == "cooperate" else 0

            # Use discretized neighbor state (from previous state implementation)
            if coop_proportion <= 0.33:
                neighbor_state = 0  # Low
            elif coop_proportion <= 0.67:
                neighbor_state = 1  # Med
            else:
                neighbor_state = 2  # High

            # State includes own last 2 moves and neighbor summary
            return (own_last_bin, own_prev_bin, neighbor_state)

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

        # Default fallback
        return "standard"

    def _initialize_q_values_for_state(self, agent: "Agent", state: Hashable) -> None:
        """Initialize Q-values for a new state based on the agent's initialization type.

        Creates initial Q-values for a previously unseen state, using different
        initialization strategies (zero, optimistic, or random) as specified by
        the agent's q_init_type parameter.

        Args:
            agent: The agent whose Q-values are being initialized
            state: The state for which to initialize Q-values
        """

        if agent.q_init_type == "optimistic":
            init_val = agent.max_possible_payoff
        elif agent.q_init_type == "random":
            init_val = random.uniform(-0.1, 0.1)
        else:  # Default to zero initialization
            init_val = 0.0
        agent.q_values[state] = {"cooperate": init_val, "defect": init_val}

    def _ensure_state_exists(self, agent, state):
        """Checks if state exists in the Q-table and initializes if not."""
        if state not in agent.q_values:
            self._initialize_q_values_for_state(agent, state)

    def choose_move(self, agent, neighbors):
        # Get current state
        current_state = self._get_current_state(agent)
        agent.last_state_representation = current_state

        self._ensure_state_exists(agent, current_state)

        # Ensure state exists in Q-table, initialize if not
        if current_state not in agent.q_values:
            agent.q_values[current_state] = {"cooperate": 0.0, "defect": 0.0}

        # Exploration (epsilon-greedy)
        if random.random() < self.epsilon:
            return random.choice(["cooperate", "defect"])

        # Exploitation - choose action with highest Q-value
        if (
            agent.q_values[current_state]["cooperate"]
            >= agent.q_values[current_state]["defect"]
        ):
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

        self._ensure_state_exists(agent, next_state)

        # Ensure next state exists in Q-table, initialize if not
        if next_state not in agent.q_values:
            agent.q_values[next_state] = {"cooperate": 0.0, "defect": 0.0}

        # Find max Q-value for the next state
        best_next_q = max(agent.q_values[next_state].values())

        # Update Q-value for the chosen action using the Q-learning formula
        current_q = agent.q_values[state_executed][action]
        agent.q_values[state_executed][action] = (
            1 - self.learning_rate
        ) * current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q
        )


class AdaptiveQLearningStrategy(QLearningStrategy):
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.9,
        min_epsilon=0.01,
        decay_rate=0.99,
        state_type="basic",
    ):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def update(self, agent, action, reward, neighbor_moves):
        super().update(agent, action, reward, neighbor_moves)

        # Decay epsilon over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class LRAQLearningStrategy(QLearningStrategy):
    """Learning Rate Adjusting Q-Learning Strategy.

    This strategy dynamically adjusts its learning rate based on the cooperation
    levels of neighbors. It increases the learning rate after cooperative outcomes
    to reinforce successful cooperation patterns, and decreases it after defection
    to slow down learning from potentially suboptimal interactions.

    This approach helps overcome the tendency of standard Q-learning to converge
    to defection in social dilemma games by giving more weight to cooperative
    experiences.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        increase_rate=0.1,
        decrease_rate=0.05,
        state_type="proportion_discretized",
    ):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.increase_rate = increase_rate  # For cooperation
        self.decrease_rate = decrease_rate  # For defection
        self.base_learning_rate = learning_rate
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.9

    def update(self, agent, action, reward, neighbor_moves):
        # Handle both pairwise and neighborhood modes
        if (
            isinstance(neighbor_moves, dict)
            and "opponent_coop_proportion" in neighbor_moves
        ):
            # Pairwise mode - neighbor_moves contains opponent_coop_proportion
            coop_proportion = neighbor_moves["opponent_coop_proportion"]
            coop_neighbors = coop_proportion  # Using the proportion directly
            total_neighbors = 1  # In proportional terms
        else:
            # Standard neighborhood mode
            coop_neighbors = sum(
                1 for move in neighbor_moves.values() if move == "cooperate"
            )
            total_neighbors = len(neighbor_moves) if neighbor_moves else 1
            coop_proportion = (
                coop_neighbors / total_neighbors if total_neighbors > 0 else 0
            )

        # Adjust learning rate based on cooperation level and agent's own action
        if action == "cooperate" and coop_proportion > 0.5:
            # Cooperating with cooperators - increase learning rate
            self.learning_rate = min(
                self.max_learning_rate, self.learning_rate + self.increase_rate
            )
        elif action == "defect" and coop_proportion > 0.5:
            # Defecting against cooperators - decrease learning rate
            self.learning_rate = max(
                self.min_learning_rate, self.learning_rate - self.decrease_rate
            )

        # Use the adjusted learning rate for the standard Q-update
        super().update(agent, action, reward, neighbor_moves)

        # Slowly return to base rate when not adjusted (regression to mean)
        if self.learning_rate > self.base_learning_rate:
            self.learning_rate = max(self.base_learning_rate, self.learning_rate - 0.01)
        elif self.learning_rate < self.base_learning_rate:
            self.learning_rate = min(self.base_learning_rate, self.learning_rate + 0.01)


class HystereticQLearningStrategy(QLearningStrategy):
    """Hysteretic Q-Learning Strategy for resilient cooperation.

    This strategy uses different learning rates for positive and negative experiences.
    Positive updates (when the target Q-value is higher than current) use the standard
    learning rate, while negative updates use a slower rate (beta). This creates
    an optimistic agent that is more resistant to occasional defections.

    Hysteretic Q-learning helps maintain cooperation in unstable, non-stationary
    environments by being slower to learn negative outcomes.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        beta=0.01,
        state_type="proportion_discretized",
    ):
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
            agent.q_values[state_executed][action] = (
                1 - self.learning_rate
            ) * current + self.learning_rate * target
        else:  # Negative experience - learn more slowly
            agent.q_values[state_executed][action] = (
                1 - self.beta
            ) * current + self.beta * target


class WolfPHCStrategy(QLearningStrategy):
    """Win or Learn Fast Policy Hill-Climbing Strategy.

    This strategy adjusts learning rates based on whether the agent is "winning"
    or "losing" compared to its historical performance. It learns quickly
    when performing worse than average (losing) and cautiously when doing better
    than average (winning).

    WoLF-PHC helps achieve convergence in multi-agent settings by dynamically
    balancing between exploration and exploitation, improving stability in
    non-stationary environments.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        alpha_win=0.05,
        alpha_lose=0.2,
        alpha_avg=0.01,
        state_type="proportion_discretized",
    ):
        super().__init__(learning_rate, discount_factor, epsilon, state_type)
        self.alpha_win = alpha_win
        self.alpha_lose = alpha_lose
        self.alpha_avg = alpha_avg
        self.average_payoff = 0.0
        self.policy_counts = {}  # {state: {'cooperate': count, 'defect': count}}
        self.average_policy = {}  # {state: {'cooperate': prob, 'defect': prob}}

    def _update_average_policy(self, state: Hashable) -> None:
        """Update the average policy for a given state.

        Tracks the average policy (probability distribution over actions)
        for a state over time. This is used to determine whether the agent
        is "winning" or "losing" compared to historical performance.

        Args:
            state: The state for which to update the average policy
        """

        if state not in self.policy_counts:
            return

        total_actions = sum(self.policy_counts[state].values())
        if total_actions == 0:
            return

        if state not in self.average_policy:
            self.average_policy[state] = {"cooperate": 0.5, "defect": 0.5}

        for action in ["cooperate", "defect"]:
            action_count = self.policy_counts[state][action]
            action_prob = action_count / total_actions
            current_avg = self.average_policy[state][action]
            # Exponential moving average of the policy
            self.average_policy[state][action] = (
                1 - self.alpha_avg
            ) * current_avg + self.alpha_avg * action_prob

    def choose_move(self, agent, neighbors):
        # Get current state
        current_state = self._get_current_state(agent)
        agent.last_state_representation = current_state

        # Ensure state exists in Q-table, initialize if not
        if current_state not in agent.q_values:
            agent.q_values[current_state] = {"cooperate": 0.0, "defect": 0.0}

        # Initialize policy counts if needed
        if current_state not in self.policy_counts:
            self.policy_counts[current_state] = {"cooperate": 0, "defect": 0}
            self.average_policy[current_state] = {"cooperate": 0.5, "defect": 0.5}

        # Exploration (epsilon-greedy)
        if random.random() < self.epsilon:
            chosen_action = random.choice(["cooperate", "defect"])
        else:
            # Exploitation - choose action with highest Q-value
            if (
                agent.q_values[current_state]["cooperate"]
                >= agent.q_values[current_state]["defect"]
            ):
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
        self.average_payoff = (
            1 - self.alpha_avg
        ) * self.average_payoff + self.alpha_avg * reward

        # Update average policy for the state
        self._update_average_policy(state_executed)

        # Calculate V_pi and V_pi_avg for the state
        q_coop = agent.q_values[state_executed]["cooperate"]
        q_def = agent.q_values[state_executed]["defect"]

        # Estimate current policy pi using epsilon-greedy logic
        num_actions = 2.0  # Float for division
        if q_coop >= q_def:  # Cooperate is 'best' action
            pi_coop = 1.0 - self.epsilon + self.epsilon / num_actions
            pi_def = self.epsilon / num_actions
        else:  # Defect is 'best' action
            pi_def = 1.0 - self.epsilon + self.epsilon / num_actions
            pi_coop = self.epsilon / num_actions

        # Ensure probabilities sum roughly to 1 (optional sanity check)
        # assert math.isclose(pi_coop + pi_def, 1.0), f"Policy probs sum to {pi_coop + pi_def}"

        V_pi = pi_coop * q_coop + pi_def * q_def

        # Get average policy values (ensure state exists)
        if state_executed not in self.average_policy:
            self.average_policy[state_executed] = {"cooperate": 0.5, "defect": 0.5}
        avg_pi_coop = self.average_policy[state_executed]["cooperate"]
        avg_pi_def = self.average_policy[state_executed]["defect"]
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
        agent.q_values[state_executed][action] = (
            1 - current_alpha
        ) * current_q + current_alpha * (reward + self.discount_factor * best_next_q)


class UCB1QLearningStrategy(QLearningStrategy):
    """Upper Confidence Bound Q-Learning Strategy.

    This strategy replaces the standard ε-greedy exploration with UCB1 algorithm
    from multi-armed bandit research. It balances exploration and exploitation
    by considering both the estimated value of actions and the uncertainty in
    those estimates.

    UCB1 provides more informed exploration by prioritizing actions with high
    potential value or high uncertainty, which can lead to faster convergence to
    optimal policies compared to random exploration.
    """

    def __init__(
        self,
        exploration_constant=2.0,
        learning_rate=0.1,
        discount_factor=0.9,
        state_type="proportion_discretized",
    ):
        super().__init__(
            learning_rate, discount_factor, 0.0, state_type
        )  # Set epsilon to 0, using UCB instead
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
            self.action_counts[current_state] = {"cooperate": 0, "defect": 0}

        # Calculate UCB values
        ucb_values = {}
        total_log = math.log(self.total_steps + 1)  # Pre-compute logarithm

        for action in ["cooperate", "defect"]:
            q_value = agent.q_values[current_state][action]
            count = self.action_counts[current_state][action]

            if count == 0:
                # If an action hasn't been tried, prioritize it
                ucb_values[action] = float("inf")
            else:
                # UCB formula: Q-value + C * sqrt(log(total_steps) / count)
                exploration_bonus = self.exploration_constant * math.sqrt(
                    total_log / count
                )
                ucb_values[action] = q_value + exploration_bonus

        # Choose action with highest UCB value
        if ucb_values["cooperate"] == float("inf") and ucb_values["defect"] == float(
            "inf"
        ):
            # First time in this state, choose randomly
            chosen_action = random.choice(["cooperate", "defect"])
        elif ucb_values["cooperate"] >= ucb_values["defect"]:
            chosen_action = "cooperate"
        else:
            chosen_action = "defect"

        return chosen_action

    def update(self, agent, action, reward, neighbor_moves):
        state_executed = agent.last_state_representation
        if state_executed is None:
            return

        # Increment count for the executed action
        if state_executed not in self.action_counts:
            self.action_counts[state_executed] = {"cooperate": 0, "defect": 0}
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
        "tit_for_two_tats": TitForTwoTatsStrategy,
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
    def __init__(
        self,
        agent_id,
        strategy="random",
        memory_length=10,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        state_type="proportion_discretized",
        q_init_type="zero",
        max_possible_payoff=5.0,
        generosity=0.05,
        initial_move="cooperate",
        prob_coop=0.5,
        increase_rate=0.1,
        decrease_rate=0.05,
        beta=0.01,
        alpha_win=0.05,
        alpha_lose=0.2,
        alpha_avg=0.01,
        exploration_constant=2.0,
    ):
        self.agent_id = agent_id
        self.strategy_type = strategy
        self.q_init_type = q_init_type  # Store the initialization type
        self.max_possible_payoff = (
            max_possible_payoff  # Store max payoff for optimistic init
        )

        # Create strategy parameters dictionary
        strategy_params = {}
        if strategy == "generous_tit_for_tat":
            strategy_params["generosity"] = generosity
        elif strategy == "pavlov":
            strategy_params["initial_move"] = initial_move
        elif strategy == "randomprob":
            strategy_params["prob_coop"] = prob_coop
        elif strategy in ["q_learning", "q_learning_adaptive"]:
            strategy_params.update(
                {
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "epsilon": epsilon,
                    "state_type": state_type,
                }
            )
            if strategy in [
                "q_learning",
                "q_learning_adaptive",
                "lra_q",
                "hysteretic_q",
                "wolf_phc",
                "ucb1_q",
            ]:
                strategy_params.update(
                    {
                        "learning_rate": learning_rate,
                        "discount_factor": discount_factor,
                        "epsilon": epsilon,
                        "state_type": state_type,
                        # For simplicity, we'll handle it in the strategy's _ensure_state_exists method.
                    }
                )
        elif strategy == "lra_q":
            strategy_params.update(
                {
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "epsilon": epsilon,
                    "increase_rate": increase_rate,
                    "decrease_rate": decrease_rate,
                    "state_type": state_type,
                }
            )
        elif strategy == "hysteretic_q":
            strategy_params.update(
                {
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "epsilon": epsilon,
                    "beta": beta,
                    "state_type": state_type,
                }
            )
        elif strategy == "wolf_phc":
            strategy_params.update(
                {
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "epsilon": epsilon,
                    "alpha_win": alpha_win,
                    "alpha_lose": alpha_lose,
                    "alpha_avg": alpha_avg,
                    "state_type": state_type,
                }
            )
        elif strategy == "ucb1_q":
            strategy_params.update(
                {
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "exploration_constant": exploration_constant,
                    "state_type": state_type,
                }
            )

        # Create the strategy object
        self.strategy = create_strategy(strategy, **strategy_params)

        # Initialize agent state
        self.score = 0
        self.memory = deque(maxlen=memory_length)  # Use deque with max length
        self.q_values = {}  # Change to empty dict to support state-based Q-values
        self.last_state_representation = None  # Track state for Q-learning

    def choose_move(self, neighbors: List[int]) -> str:
        """Choose the next move for the agent using UCB1 exploration.

        Instead of randomly exploring with probability epsilon, this method
        calculates UCB values for each action that balance exploitation (high Q-value)
        with exploration (high uncertainty).

        Args:
            agent: The agent making the decision
            neighbors: List of neighbor agent IDs

        Returns:
            The selected action ("cooperate" or "defect")
        """
        return self.strategy.choose_move(self, neighbors)

    def update_q_value(self, action, reward, next_state_actions):
        """Update Q-values if the agent uses Q-learning."""
        if self.strategy_type in (
            "q_learning",
            "q_learning_adaptive",
            "lra_q",
            "hysteretic_q",
            "wolf_phc",
            "ucb1_q",
        ):
            self.strategy.update(self, action, reward, next_state_actions)

    def update_memory(self, my_move, neighbor_moves, reward):
        """Update the agent's memory with the result of the last round."""
        self.memory.append(
            {
                "my_move": my_move,
                "neighbor_moves": neighbor_moves,
                "reward": reward,
            }
        )

    def reset(self):
        """Reset the agent's state for a new simulation."""
        self.score = 0
        self.memory.clear()
        self.q_values = {}
        self.last_state_representation = None
