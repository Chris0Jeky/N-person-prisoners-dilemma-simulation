#!/usr/bin/env python3
"""
Debug Enhanced Q-Learning with features from enhanced_qlearning_agents.py
Shows individual choices each round to understand behavior
"""

import random
import sys
import os
from collections import defaultdict

# Constants
COOPERATE = 0
DEFECT = 1
PAYOFFS_2P = {
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_other_cooperators, total_agents):
    """Calculates N-Person payoff based on the linear formula."""
    if total_agents <= 1: return R if my_move == COOPERATE else P
    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / (total_agents - 1))
    else:  # Defect
        return P + (T - P) * (num_other_cooperators / (total_agents - 1))


class EnhancedQLearningAgent:
    """Enhanced Q-Learning with configurable improvements."""
    
    def __init__(self, agent_id, 
                 learning_rate=0.15, 
                 discount_factor=0.95,
                 epsilon=0.2,
                 epsilon_decay=0.995,
                 epsilon_min=0.05,
                 exclude_self=True,  # Exclude self from state
                 opponent_modeling=True,  # Model opponents
                 state_type="fine"):  # Use fine-grained states
        
        self.agent_id = agent_id
        self.strategy_name = "EnhancedQLearning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exclude_self = exclude_self
        self.opponent_modeling = opponent_modeling
        self.state_type = state_type
        
        # Q-tables
        self.q_table_pairwise = {}
        self.q_table_nperson = {}
        
        # Memory
        self.last_state_pairwise = {}
        self.last_action_pairwise = {}
        self.last_state_nperson = None
        self.last_action_nperson = None
        self.last_others_coop_ratio = None
        self.my_last_nperson_action = None
        
        # For opponent modeling
        if opponent_modeling:
            self.opponent_history = defaultdict(lambda: {'C': 0, 'D': 0})
            self.opponent_models = {}  # opponent_id -> predicted cooperation rate
            self.opponent_response_model = defaultdict(lambda: defaultdict(lambda: 0.5))
            # opponent_id -> {my_action -> their_cooperation_rate}
        
        # For pairwise
        self.opponent_last_moves = {}
    
    def _get_state_neighborhood(self, overall_coop_ratio, my_last_action=None):
        """Get state for neighborhood mode."""
        if self.exclude_self and my_last_action is not None and overall_coop_ratio is not None:
            # Calculate cooperation ratio excluding self
            total_agents = 3  # Fixed for 3-person game
            total_coops = overall_coop_ratio * total_agents
            
            # Remove self contribution
            if my_last_action == COOPERATE:
                total_coops -= 1
            
            # Calculate others' cooperation ratio
            others_coop_ratio = total_coops / (total_agents - 1) if total_agents > 1 else 0
        else:
            others_coop_ratio = overall_coop_ratio if overall_coop_ratio is not None else 0.5
        
        # Store for later use
        self.last_others_coop_ratio = others_coop_ratio
        
        if self.state_type == "basic":
            # Discretize into bins
            if others_coop_ratio <= 0.2:
                base_state = 'very_low'
            elif others_coop_ratio <= 0.4:
                base_state = 'low'
            elif others_coop_ratio <= 0.6:
                base_state = 'medium'
            elif others_coop_ratio <= 0.8:
                base_state = 'high'
            else:
                base_state = 'very_high'
        elif self.state_type == "fine":
            # Finer discretization (0.1 increments)
            base_state = f"coop_{int(others_coop_ratio * 10) / 10:.1f}"
        elif self.state_type == "coarse":
            # Coarser discretization
            if others_coop_ratio <= 0.33:
                base_state = 'low'
            elif others_coop_ratio <= 0.67:
                base_state = 'medium'
            else:
                base_state = 'high'
        else:
            base_state = 'default'
        
        # Add opponent model information if enabled
        if self.opponent_modeling and self.opponent_models:
            # Average predicted cooperation rates
            avg_pred = sum(self.opponent_models.values()) / len(self.opponent_models)
            
            # Predict responses for both possible actions
            pred_if_coop = []
            pred_if_defect = []
            for opp_id in self.opponent_response_model:
                pred_if_coop.append(self.opponent_response_model[opp_id]['C'])
                pred_if_defect.append(self.opponent_response_model[opp_id]['D'])
            
            avg_pred_coop = sum(pred_if_coop) / len(pred_if_coop) if pred_if_coop else 0.5
            avg_pred_defect = sum(pred_if_defect) / len(pred_if_defect) if pred_if_defect else 0.5
            
            model_state = f"_model{int(avg_pred * 10) / 10:.1f}_ifC{int(avg_pred_coop * 10) / 10:.1f}_ifD{int(avg_pred_defect * 10) / 10:.1f}"
            return base_state + model_state
        
        return base_state
    
    def _get_state_pairwise(self, opponent_id):
        """Get state for pairwise mode with opponent modeling."""
        # Basic state from opponent's last move
        if opponent_id not in self.opponent_last_moves:
            base_state = 'initial'
        else:
            last_move = self.opponent_last_moves[opponent_id]
            base_state = 'opp_coop' if last_move == COOPERATE else 'opp_defect'
        
        # Add opponent model if enabled
        if self.opponent_modeling and opponent_id in self.opponent_models:
            pred_coop = self.opponent_models[opponent_id]
            pred_if_c = self.opponent_response_model[opponent_id]['C']
            pred_if_d = self.opponent_response_model[opponent_id]['D']
            base_state += f"_model{pred_coop:.1f}_ifC{pred_if_c:.1f}_ifD{pred_if_d:.1f}"
        
        return base_state
    
    def _update_opponent_model(self, opponent_id, action, my_last_action=None):
        """Update opponent model based on observed action."""
        if not self.opponent_modeling:
            return
        
        # Update basic frequency model
        if action == COOPERATE:
            self.opponent_history[opponent_id]['C'] += 1
        else:
            self.opponent_history[opponent_id]['D'] += 1
        
        total = self.opponent_history[opponent_id]['C'] + self.opponent_history[opponent_id]['D']
        if total > 0:
            self.opponent_models[opponent_id] = self.opponent_history[opponent_id]['C'] / total
        
        # Update response model if we know what we played
        if my_last_action is not None:
            key = 'C' if my_last_action == COOPERATE else 'D'
            old_rate = self.opponent_response_model[opponent_id][key]
            # Exponential moving average
            if action == COOPERATE:
                self.opponent_response_model[opponent_id][key] = 0.9 * old_rate + 0.1
            else:
                self.opponent_response_model[opponent_id][key] = 0.9 * old_rate
    
    def _ensure_state_exists(self, state, q_table):
        """Initialize Q-values for new states with optimistic values."""
        if state not in q_table:
            # Optimistic initialization to encourage exploration of cooperation
            q_table[state] = {
                COOPERATE: 1.0,  # Very optimistic about cooperation
                DEFECT: 0.5
            }
    
    def _choose_action_epsilon_greedy(self, state, q_table):
        """Choose action using epsilon-greedy policy."""
        self._ensure_state_exists(state, q_table)
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice([COOPERATE, DEFECT])
        
        # Exploitation
        q_values = q_table[state]
        if q_values[COOPERATE] > q_values[DEFECT]:
            return COOPERATE
        elif q_values[DEFECT] > q_values[COOPERATE]:
            return DEFECT
        else:
            # Tie-breaking: favor cooperation
            return COOPERATE if random.random() > 0.4 else DEFECT
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise mode."""
        state = self._get_state_pairwise(opponent_id)
        action = self._choose_action_epsilon_greedy(state, self.q_table_pairwise)
        
        # Store for later update
        self.last_state_pairwise[opponent_id] = state
        self.last_action_pairwise[opponent_id] = action
        
        return action
    
    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for N-person mode."""
        # Get state (excluding self if configured)
        state = self._get_state_neighborhood(
            prev_round_group_coop_ratio, 
            self.my_last_nperson_action if self.exclude_self else None
        )
        
        action = self._choose_action_epsilon_greedy(state, self.q_table_nperson)
        
        # Store for later update
        self.last_state_nperson = state
        self.last_action_nperson = action
        
        return action
    
    def update_q_value_pairwise(self, opponent_id, opponent_move, my_move, my_payoff):
        """Update Q-value and models for pairwise interaction."""
        # Update opponent model
        if opponent_id in self.last_action_pairwise:
            self._update_opponent_model(opponent_id, opponent_move, self.last_action_pairwise[opponent_id])
        
        # Update history
        self.opponent_last_moves[opponent_id] = opponent_move
        
        if opponent_id in self.last_state_pairwise:
            state = self.last_state_pairwise[opponent_id]
            action = self.last_action_pairwise[opponent_id]
            next_state = self._get_state_pairwise(opponent_id)
            
            self._ensure_state_exists(state, self.q_table_pairwise)
            self._ensure_state_exists(next_state, self.q_table_pairwise)
            
            # Q-learning update
            current_q = self.q_table_pairwise[state][action]
            max_next_q = max(self.q_table_pairwise[next_state].values())
            new_q = current_q + self.learning_rate * (
                my_payoff + self.discount_factor * max_next_q - current_q
            )
            self.q_table_pairwise[state][action] = new_q
    
    def update_q_value_nperson(self, my_move, payoff, current_coop_ratio, opponent_moves):
        """Update Q-value and models for N-person game."""
        # Update opponent models
        if self.opponent_modeling:
            for opp_id, opp_move in opponent_moves.items():
                self._update_opponent_model(opp_id, opp_move, self.my_last_nperson_action)
        
        # Store my action for next round
        self.my_last_nperson_action = my_move
        
        if self.last_state_nperson is not None:
            state = self.last_state_nperson
            action = self.last_action_nperson
            
            # Get next state
            next_state = self._get_state_neighborhood(current_coop_ratio, my_move)
            
            self._ensure_state_exists(state, self.q_table_nperson)
            self._ensure_state_exists(next_state, self.q_table_nperson)
            
            # Q-learning update
            current_q = self.q_table_nperson[state][action]
            max_next_q = max(self.q_table_nperson[next_state].values())
            new_q = current_q + self.learning_rate * (
                payoff + self.discount_factor * max_next_q - current_q
            )
            self.q_table_nperson[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset(self):
        """Reset for new episode."""
        self.opponent_last_moves = {}
        self.last_state_pairwise = {}
        self.last_action_pairwise = {}
        self.last_state_nperson = None
        self.last_action_nperson = None
        self.last_others_coop_ratio = None
        self.my_last_nperson_action = None
        self.decay_epsilon()


class StaticAgent:
    """Static strategy agent (AllC, AllD, TFT)"""
    def __init__(self, agent_id, strategy_name):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.opponent_last_moves = {}

    def choose_pairwise_action(self, opponent_id):
        """Choose action for a 2-player game."""
        if self.strategy_name == "TFT":
            return self.opponent_last_moves.get(opponent_id, COOPERATE)
        elif self.strategy_name == "AllC":
            return COOPERATE
        elif self.strategy_name == "AllD":
            return DEFECT

    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for an N-player group game."""
        if self.strategy_name == "TFT":
            if prev_round_group_coop_ratio is None:
                return COOPERATE
            else:
                return COOPERATE if random.random() < prev_round_group_coop_ratio else DEFECT
        elif self.strategy_name == "AllC":
            return COOPERATE
        elif self.strategy_name == "AllD":
            return DEFECT

    def reset(self):
        self.opponent_last_moves = {}


def detailed_pairwise_simulation(agents, num_rounds=20, verbose=True):
    """Run pairwise simulation with detailed output."""
    for agent in agents:
        agent.reset()
    
    print("\n=== PAIRWISE SIMULATION ===")
    print(f"Agents: {[f'{a.agent_id} ({a.strategy_name})' for a in agents]}")
    print(f"Payoff matrix: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1)")
    print("-" * 80)
    
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    cooperation_counts = {agent.agent_id: 0 for agent in agents}
    total_interactions = {agent.agent_id: 0 for agent in agents}
    
    for round_num in range(num_rounds):
        if verbose and round_num < 10:
            print(f"\nRound {round_num + 1}:")
        
        round_moves = {}
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        
        # All pairs play
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Show states for enhanced QL agents
                if verbose and round_num < 10 and agent1.strategy_name == "EnhancedQLearning":
                    state1 = agent1._get_state_pairwise(agent2.agent_id)
                    q_vals1 = agent1.q_table_pairwise.get(state1, {})
                    print(f"  {agent1.agent_id} state vs {agent2.agent_id}: '{state1}'")
                    print(f"    Q-values: C={q_vals1.get(COOPERATE, 1.0):.3f}, D={q_vals1.get(DEFECT, 0.5):.3f}")
                    if agent1.opponent_modeling and agent2.agent_id in agent1.opponent_models:
                        print(f"    Opponent model: coop_rate={agent1.opponent_models[agent2.agent_id]:.2f}")
                
                # Make moves
                move1 = agent1.choose_pairwise_action(agent2.agent_id)
                move2 = agent2.choose_pairwise_action(agent1.agent_id)
                
                # Track cooperation
                if move1 == COOPERATE:
                    cooperation_counts[agent1.agent_id] += 1
                if move2 == COOPERATE:
                    cooperation_counts[agent2.agent_id] += 1
                total_interactions[agent1.agent_id] += 1
                total_interactions[agent2.agent_id] += 1
                
                # Store moves
                round_moves[(agent1.agent_id, agent2.agent_id)] = (move1, move2)
                
                # Update histories for static agents
                if hasattr(agent1, 'opponent_last_moves'):
                    agent1.opponent_last_moves[agent2.agent_id] = move2
                if hasattr(agent2, 'opponent_last_moves'):
                    agent2.opponent_last_moves[agent1.agent_id] = move1
                
                # Calculate payoffs
                payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
                round_payoffs[agent1.agent_id] += payoff1
                round_payoffs[agent2.agent_id] += payoff2
                
                # Update Q-values for enhanced QL agents
                if agent1.strategy_name == "EnhancedQLearning":
                    agent1.update_q_value_pairwise(agent2.agent_id, move2, move1, payoff1)
                if agent2.strategy_name == "EnhancedQLearning":
                    agent2.update_q_value_pairwise(agent1.agent_id, move1, move2, payoff2)
                
                if verbose and round_num < 10:
                    move1_str = "C" if move1 == COOPERATE else "D"
                    move2_str = "C" if move2 == COOPERATE else "D"
                    print(f"  {agent1.agent_id} vs {agent2.agent_id}: {move1_str} vs {move2_str}, payoffs: ({payoff1}, {payoff2})")
        
        # Update cumulative scores
        for agent_id in cumulative_scores:
            cumulative_scores[agent_id] += round_payoffs[agent_id]
        
        if verbose and round_num < 10:
            print(f"  Round scores: {round_payoffs}")
            print(f"  Cumulative: {cumulative_scores}")
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    for agent in agents:
        coop_rate = cooperation_counts[agent.agent_id] / total_interactions[agent.agent_id] if total_interactions[agent.agent_id] > 0 else 0
        print(f"{agent.agent_id}: Score={cumulative_scores[agent.agent_id]}, Cooperation rate={coop_rate:.2%}")
        
        # Show Q-table info for enhanced QL agents
        if agent.strategy_name == "EnhancedQLearning":
            print(f"  Total states learned: {len(agent.q_table_pairwise)}")
            if agent.opponent_modeling:
                print(f"  Opponent models: {dict(agent.opponent_models)}")


def detailed_nperson_simulation(agents, num_rounds=20, verbose=True):
    """Run N-person simulation with detailed output."""
    for agent in agents:
        agent.reset()
    
    print("\n=== N-PERSON SIMULATION ===")
    print(f"Agents: {[f'{a.agent_id} ({a.strategy_name})' for a in agents]}")
    print(f"Payoff formula: S=0, P=1, R=3, T=5")
    print("-" * 80)
    
    cumulative_scores = {agent.agent_id: 0 for agent in agents}
    cooperation_counts = {agent.agent_id: 0 for agent in agents}
    prev_round_coop_ratio = None
    num_total_agents = len(agents)
    
    for round_num in range(num_rounds):
        if verbose and round_num < 10:
            print(f"\nRound {round_num + 1}:")
            if prev_round_coop_ratio is not None:
                print(f"  Previous cooperation ratio: {prev_round_coop_ratio:.2f}")
        
        # Show states for enhanced QL agents
        for agent in agents:
            if verbose and round_num < 10 and agent.strategy_name == "EnhancedQLearning":
                state = agent._get_state_neighborhood(
                    prev_round_coop_ratio,
                    agent.my_last_nperson_action if agent.exclude_self else None
                )
                q_vals = agent.q_table_nperson.get(state, {})
                print(f"  {agent.agent_id} state: '{state}'")
                print(f"    Q-values: C={q_vals.get(COOPERATE, 1.0):.3f}, D={q_vals.get(DEFECT, 0.5):.3f}")
                if agent.exclude_self and agent.last_others_coop_ratio is not None:
                    print(f"    Others' coop ratio (excluding self): {agent.last_others_coop_ratio:.2f}")
        
        # All agents make moves
        moves = {}
        for agent in agents:
            move = agent.choose_nperson_action(prev_round_coop_ratio)
            moves[agent.agent_id] = move
            if move == COOPERATE:
                cooperation_counts[agent.agent_id] += 1
            if verbose and round_num < 10:
                move_str = "C" if move == COOPERATE else "D"
                print(f"  {agent.agent_id} plays: {move_str}")
        
        # Calculate cooperation ratio
        num_cooperators = list(moves.values()).count(COOPERATE)
        current_coop_ratio = num_cooperators / num_total_agents if num_total_agents > 0 else 0
        
        if verbose and round_num < 10:
            print(f"  Cooperation count: {num_cooperators}/{num_total_agents} = {current_coop_ratio:.2f}")
        
        # Calculate payoffs
        round_payoffs = {}
        for agent in agents:
            my_move = moves[agent.agent_id]
            num_other_cooperators = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, num_other_cooperators, num_total_agents)
            round_payoffs[agent.agent_id] = payoff
            cumulative_scores[agent.agent_id] += payoff
            
            # Update Q-values for enhanced QL agents
            if agent.strategy_name == "EnhancedQLearning":
                # Get opponent moves for modeling
                opponent_moves = {aid: move for aid, move in moves.items() if aid != agent.agent_id}
                agent.update_q_value_nperson(my_move, payoff, current_coop_ratio, opponent_moves)
        
        if verbose and round_num < 10:
            print(f"  Payoffs: {round_payoffs}")
            print(f"  Cumulative: {cumulative_scores}")
        
        prev_round_coop_ratio = current_coop_ratio
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    for agent in agents:
        coop_rate = cooperation_counts[agent.agent_id] / num_rounds if num_rounds > 0 else 0
        print(f"{agent.agent_id}: Score={cumulative_scores[agent.agent_id]}, Cooperation rate={coop_rate:.2%}")
        
        # Show info for enhanced QL agents
        if agent.strategy_name == "EnhancedQLearning":
            print(f"  Total states learned: {len(agent.q_table_nperson)}")
            if agent.opponent_modeling:
                print(f"  Opponent models: {dict(agent.opponent_models)}")


def main():
    """Run debugging experiments with enhanced Q-learning."""
    print("Enhanced Q-Learning Debugging Experiments")
    print("Features: Exclude self, Opponent modeling, Fine-grained states, Optimistic init")
    print("=" * 80)
    
    # Experiment 1: 1 Enhanced QL vs 2 TFT (Pairwise)
    print("\n### Experiment 1: 1 Enhanced QL vs 2 TFT (Pairwise) ###")
    agents1 = [
        EnhancedQLearningAgent(agent_id="QL_1", exclude_self=True, opponent_modeling=True),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents1, num_rounds=30, verbose=True)
    
    # Experiment 2: 1 Enhanced QL vs 2 TFT (N-person)
    print("\n\n### Experiment 2: 1 Enhanced QL vs 2 TFT (N-person) ###")
    agents2 = [
        EnhancedQLearningAgent(agent_id="QL_1", exclude_self=True, opponent_modeling=True),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents2, num_rounds=30, verbose=True)
    
    # Experiment 3: 2 Enhanced QL vs 1 TFT (Pairwise)
    print("\n\n### Experiment 3: 2 Enhanced QL vs 1 TFT (Pairwise) ###")
    agents3 = [
        EnhancedQLearningAgent(agent_id="QL_1", exclude_self=True, opponent_modeling=True),
        EnhancedQLearningAgent(agent_id="QL_2", exclude_self=True, opponent_modeling=True),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents3, num_rounds=30, verbose=True)
    
    # Experiment 4: Test without enhancements
    print("\n\n### Experiment 4: QL without enhancements vs with enhancements (Pairwise) ###")
    agents4 = [
        EnhancedQLearningAgent(agent_id="QL_Basic", exclude_self=False, opponent_modeling=False, state_type="basic"),
        EnhancedQLearningAgent(agent_id="QL_Enhanced", exclude_self=True, opponent_modeling=True, state_type="fine"),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents4, num_rounds=50, verbose=False)
    
    # Experiment 5: Long run to see convergence
    print("\n\n### Experiment 5: 2 Enhanced QL vs 1 TFT (N-person, 100 rounds) ###")
    agents5 = [
        EnhancedQLearningAgent(agent_id="QL_1", exclude_self=True, opponent_modeling=True),
        EnhancedQLearningAgent(agent_id="QL_2", exclude_self=True, opponent_modeling=True),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents5, num_rounds=100, verbose=False)


if __name__ == "__main__":
    main()