#!/usr/bin/env python3
"""
Improved Q-Learning with 2-round history as suggested by co-author
Shows individual choices each round to understand behavior
"""

import random
import sys
import os

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


class ImprovedQLearningAgent:
    """Q-Learning with 2-round history for richer state representation."""
    
    def __init__(self, agent_id, learning_rate=0.15, discount_factor=0.95, 
                 epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.05):
        self.agent_id = agent_id
        self.strategy_name = "ImprovedQLearning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  # Higher to value future more
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-tables
        self.q_table_pairwise = {}
        self.q_table_nperson = {}
        
        # History tracking for richer state
        self.my_history_pairwise = {}  # opponent_id -> [my last 2 moves]
        self.opp_history_pairwise = {}  # opponent_id -> [their last 2 moves]
        self.my_history_nperson = []  # my last 2 moves
        self.coop_ratio_history = []  # last 2 cooperation ratios
        
        # For current round
        self.last_state_pairwise = {}
        self.last_action_pairwise = {}
        self.last_state_nperson = None
        self.last_action_nperson = None
    
    def _get_state_pairwise_history(self, opponent_id):
        """Get state based on last 2 rounds of history."""
        if opponent_id not in self.my_history_pairwise:
            return 'initial'
        
        my_hist = self.my_history_pairwise[opponent_id]
        opp_hist = self.opp_history_pairwise[opponent_id]
        
        if len(my_hist) < 2 or len(opp_hist) < 2:
            # Only 1 round of history
            if len(my_hist) == 1 and len(opp_hist) == 1:
                return f"1round_M{'C' if my_hist[0] == COOPERATE else 'D'}_O{'C' if opp_hist[0] == COOPERATE else 'D'}"
            return 'initial'
        
        # Full 2-round history: (My_t-2, Opp_t-2, My_t-1, Opp_t-1)
        state = f"M{self._move_to_char(my_hist[0])}{self._move_to_char(my_hist[1])}_O{self._move_to_char(opp_hist[0])}{self._move_to_char(opp_hist[1])}"
        return state
    
    def _get_state_nperson_history(self):
        """Get state based on cooperation trends and my recent behavior."""
        if len(self.coop_ratio_history) == 0:
            return 'initial'
        
        if len(self.coop_ratio_history) == 1:
            # One round of history
            ratio = self.coop_ratio_history[0]
            my_last = 'C' if self.my_history_nperson[0] == COOPERATE else 'D' if len(self.my_history_nperson) > 0 else 'X'
            return f"1round_{self._ratio_to_category(ratio)}_M{my_last}"
        
        # Two rounds of history - look at trend
        ratio_t2 = self.coop_ratio_history[0]
        ratio_t1 = self.coop_ratio_history[1]
        trend = 'up' if ratio_t1 > ratio_t2 + 0.1 else 'down' if ratio_t1 < ratio_t2 - 0.1 else 'stable'
        
        # Include my recent behavior
        my_recent = ""
        if len(self.my_history_nperson) >= 2:
            my_recent = f"_M{self._move_to_char(self.my_history_nperson[0])}{self._move_to_char(self.my_history_nperson[1])}"
        
        return f"{self._ratio_to_category(ratio_t1)}_{trend}{my_recent}"
    
    def _move_to_char(self, move):
        """Convert move to character."""
        return 'C' if move == COOPERATE else 'D'
    
    def _ratio_to_category(self, ratio):
        """Convert ratio to category."""
        if ratio <= 0.33:
            return 'low'
        elif ratio <= 0.67:
            return 'medium'
        else:
            return 'high'
    
    def _ensure_state_exists(self, state, q_table):
        """Initialize Q-values for new states with optimistic values."""
        if state not in q_table:
            # Optimistic initialization to encourage exploration
            q_table[state] = {
                COOPERATE: 0.5,  # Start optimistic about cooperation
                DEFECT: 0.3
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
            # Tie-breaking: slightly favor cooperation
            return COOPERATE if random.random() > 0.45 else DEFECT
    
    def choose_pairwise_action(self, opponent_id):
        """Choose action for pairwise mode using history."""
        state = self._get_state_pairwise_history(opponent_id)
        action = self._choose_action_epsilon_greedy(state, self.q_table_pairwise)
        
        # Store for later update
        self.last_state_pairwise[opponent_id] = state
        self.last_action_pairwise[opponent_id] = action
        
        return action
    
    def choose_nperson_action(self, prev_round_group_coop_ratio):
        """Choose action for N-person mode using history."""
        # Update history if we have a new ratio
        if prev_round_group_coop_ratio is not None:
            self.coop_ratio_history.append(prev_round_group_coop_ratio)
            if len(self.coop_ratio_history) > 2:
                self.coop_ratio_history.pop(0)
        
        state = self._get_state_nperson_history()
        action = self._choose_action_epsilon_greedy(state, self.q_table_nperson)
        
        # Store for later update
        self.last_state_nperson = state
        self.last_action_nperson = action
        
        return action
    
    def update_pairwise_history(self, opponent_id, my_move, opponent_move):
        """Update history after a pairwise interaction."""
        # Initialize if needed
        if opponent_id not in self.my_history_pairwise:
            self.my_history_pairwise[opponent_id] = []
            self.opp_history_pairwise[opponent_id] = []
        
        # Add to history
        self.my_history_pairwise[opponent_id].append(my_move)
        self.opp_history_pairwise[opponent_id].append(opponent_move)
        
        # Keep only last 2 moves
        if len(self.my_history_pairwise[opponent_id]) > 2:
            self.my_history_pairwise[opponent_id].pop(0)
            self.opp_history_pairwise[opponent_id].pop(0)
    
    def update_nperson_history(self, my_move):
        """Update history after N-person round."""
        self.my_history_nperson.append(my_move)
        if len(self.my_history_nperson) > 2:
            self.my_history_nperson.pop(0)
    
    def update_q_value_pairwise(self, opponent_id, opponent_move, my_move, my_payoff):
        """Update Q-value for pairwise interaction."""
        # First update history
        self.update_pairwise_history(opponent_id, my_move, opponent_move)
        
        if opponent_id in self.last_state_pairwise:
            state = self.last_state_pairwise[opponent_id]
            action = self.last_action_pairwise[opponent_id]
            next_state = self._get_state_pairwise_history(opponent_id)
            
            self._ensure_state_exists(state, self.q_table_pairwise)
            self._ensure_state_exists(next_state, self.q_table_pairwise)
            
            # Q-learning update
            current_q = self.q_table_pairwise[state][action]
            max_next_q = max(self.q_table_pairwise[next_state].values())
            new_q = current_q + self.learning_rate * (
                my_payoff + self.discount_factor * max_next_q - current_q
            )
            self.q_table_pairwise[state][action] = new_q
    
    def update_q_value_nperson(self, my_move, payoff, current_coop_ratio):
        """Update Q-value for N-person game."""
        # Update history
        self.update_nperson_history(my_move)
        
        if self.last_state_nperson is not None:
            state = self.last_state_nperson
            action = self.last_action_nperson
            
            # Update ratio history for next state
            temp_history = self.coop_ratio_history.copy()
            temp_history.append(current_coop_ratio)
            if len(temp_history) > 2:
                temp_history.pop(0)
            self.coop_ratio_history = temp_history
            
            next_state = self._get_state_nperson_history()
            
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
        self.my_history_pairwise = {}
        self.opp_history_pairwise = {}
        self.my_history_nperson = []
        self.coop_ratio_history = []
        self.last_state_pairwise = {}
        self.last_action_pairwise = {}
        self.last_state_nperson = None
        self.last_action_nperson = None
        self.decay_epsilon()
    
    # Keep compatibility with original interface
    opponent_last_moves = {}  # Dummy for compatibility


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
        if verbose and round_num < 10:  # Only show first 10 rounds in detail
            print(f"\nRound {round_num + 1}:")
        
        round_moves = {}
        round_payoffs = {agent.agent_id: 0 for agent in agents}
        
        # All pairs play
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Show states for QL agents
                if verbose and round_num < 10 and agent1.strategy_name == "ImprovedQLearning":
                    state1 = agent1._get_state_pairwise_history(agent2.agent_id)
                    q_vals1 = agent1.q_table_pairwise.get(state1, {})
                    print(f"  {agent1.agent_id} state vs {agent2.agent_id}: '{state1}'")
                    print(f"    Q-values: C={q_vals1.get(COOPERATE, 0.5):.3f}, D={q_vals1.get(DEFECT, 0.3):.3f}")
                
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
                
                # Update Q-values for QL agents
                if agent1.strategy_name == "ImprovedQLearning":
                    agent1.update_q_value_pairwise(agent2.agent_id, move2, move1, payoff1)
                if agent2.strategy_name == "ImprovedQLearning":
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
        
        # Show some Q-table entries for QL agents
        if agent.strategy_name == "ImprovedQLearning":
            print(f"  Sample Q-table entries:")
            for i, (state, q_vals) in enumerate(list(agent.q_table_pairwise.items())[:5]):
                print(f"    '{state}': C={q_vals[COOPERATE]:.3f}, D={q_vals[DEFECT]:.3f}")
            print(f"  Total states learned: {len(agent.q_table_pairwise)}")


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
        
        # Show states for QL agents
        for agent in agents:
            if verbose and round_num < 10 and agent.strategy_name == "ImprovedQLearning":
                state = agent._get_state_nperson_history()
                q_vals = agent.q_table_nperson.get(state, {})
                print(f"  {agent.agent_id} state: '{state}'")
                print(f"    Q-values: C={q_vals.get(COOPERATE, 0.5):.3f}, D={q_vals.get(DEFECT, 0.3):.3f}")
        
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
            
            # Update Q-values for QL agents
            if agent.strategy_name == "ImprovedQLearning":
                agent.update_q_value_nperson(my_move, payoff, current_coop_ratio)
        
        if verbose and round_num < 10:
            print(f"  Payoffs: {round_payoffs}")
            print(f"  Cumulative: {cumulative_scores}")
        
        prev_round_coop_ratio = current_coop_ratio
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    for agent in agents:
        coop_rate = cooperation_counts[agent.agent_id] / num_rounds if num_rounds > 0 else 0
        print(f"{agent.agent_id}: Score={cumulative_scores[agent.agent_id]}, Cooperation rate={coop_rate:.2%}")
        
        # Show some Q-table entries for QL agents
        if agent.strategy_name == "ImprovedQLearning":
            print(f"  Sample Q-table entries:")
            for i, (state, q_vals) in enumerate(list(agent.q_table_nperson.items())[:5]):
                print(f"    '{state}': C={q_vals[COOPERATE]:.3f}, D={q_vals[DEFECT]:.3f}")
            print(f"  Total states learned: {len(agent.q_table_nperson)}")


def main():
    """Run debugging experiments with improved Q-learning."""
    print("Improved Q-Learning Debugging Experiments (2-Round History)")
    print("=" * 80)
    
    # Experiment 1: 1 QL vs 2 TFT (Pairwise)
    print("\n### Experiment 1: 1 Improved QL vs 2 TFT (Pairwise) ###")
    agents1 = [
        ImprovedQLearningAgent(agent_id="QL_1", epsilon=0.2),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents1, num_rounds=30, verbose=True)
    
    # Experiment 2: 1 QL vs 2 TFT (N-person)
    print("\n\n### Experiment 2: 1 Improved QL vs 2 TFT (N-person) ###")
    agents2 = [
        ImprovedQLearningAgent(agent_id="QL_1", epsilon=0.2),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
        StaticAgent(agent_id="TFT_2", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents2, num_rounds=30, verbose=True)
    
    # Experiment 3: 2 QL vs 1 TFT (Pairwise)
    print("\n\n### Experiment 3: 2 Improved QL vs 1 TFT (Pairwise) ###")
    agents3 = [
        ImprovedQLearningAgent(agent_id="QL_1", epsilon=0.2),
        ImprovedQLearningAgent(agent_id="QL_2", epsilon=0.2),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents3, num_rounds=30, verbose=True)
    
    # Experiment 4: 2 QL vs 1 TFT (N-person) 
    print("\n\n### Experiment 4: 2 Improved QL vs 1 TFT (N-person) ###")
    agents4 = [
        ImprovedQLearningAgent(agent_id="QL_1", epsilon=0.2),
        ImprovedQLearningAgent(agent_id="QL_2", epsilon=0.2),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_nperson_simulation(agents4, num_rounds=30, verbose=True)
    
    # Run longer simulation to see convergence
    print("\n\n### Experiment 5: 2 Improved QL vs 1 TFT (Pairwise, 100 rounds) ###")
    agents5 = [
        ImprovedQLearningAgent(agent_id="QL_1", epsilon=0.2),
        ImprovedQLearningAgent(agent_id="QL_2", epsilon=0.2),
        StaticAgent(agent_id="TFT_1", strategy_name="TFT")
    ]
    detailed_pairwise_simulation(agents5, num_rounds=100, verbose=False)


if __name__ == "__main__":
    main()