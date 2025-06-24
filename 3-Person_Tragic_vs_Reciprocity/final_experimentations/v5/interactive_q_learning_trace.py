#!/usr/bin/env python3
"""
Interactive Q-Learning Trace
Shows exactly what happens during Q-learning in your implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from final_agents import PairwiseAdaptiveQLearner, StaticAgent
from config import VANILLA_PARAMS

# Payoffs
T, R, P, S = 5, 3, 1, 0

print("=" * 80)
print("INTERACTIVE Q-LEARNING TRACE")
print("This shows EXACTLY what happens in your code during learning")
print("=" * 80)

# Create a modified Q-learner that prints everything
class VerboseQLearner(PairwiseAdaptiveQLearner):
    def __init__(self, agent_id, params):
        super().__init__(agent_id, params)
        self.round_num = 0
        self.verbose = True
    
    def choose_pairwise_action(self, opponent_id):
        self.round_num += 1
        state = self._get_state(opponent_id)
        
        # Initialize Q-table for opponent if not exists
        if opponent_id not in self.q_tables:
            self.q_tables[opponent_id] = {}
        
        if state not in self.q_tables[opponent_id]:
            self.q_tables[opponent_id][state] = self._make_q_dict()
        
        epsilon = self.epsilons.get(opponent_id, self._get_initial_eps())
        q_table = self.q_tables[opponent_id]
        
        if self.verbose and self.round_num <= 10:
            print(f"\n{'='*60}")
            print(f"ROUND {self.round_num} - CHOOSING ACTION")
            print(f"{'='*60}")
            print(f"Opponent: {opponent_id}")
            print(f"Current state: '{state}'")
            print(f"Q-values: Cooperate={q_table[state]['cooperate']:.3f}, Defect={q_table[state]['defect']:.3f}")
            print(f"Epsilon (exploration rate): {epsilon:.3f}")
        
        # Decision making
        rand_val = np.random.random()
        if rand_val < epsilon:
            action = 'cooperate' if np.random.random() < 0.5 else 'defect'
            if self.verbose and self.round_num <= 10:
                print(f"\nDECISION: EXPLORE (random={rand_val:.3f} < ε={epsilon:.3f})")
                print(f"Random choice: {action.upper()}")
        else:
            if q_table[state]['cooperate'] > q_table[state]['defect']:
                action = 'cooperate'
            elif q_table[state]['defect'] > q_table[state]['cooperate']:
                action = 'defect'
            else:
                action = 'cooperate' if np.random.random() < 0.5 else 'defect'
            
            if self.verbose and self.round_num <= 10:
                print(f"\nDECISION: EXPLOIT (random={rand_val:.3f} >= ε={epsilon:.3f})")
                print(f"Best Q-value action: {action.upper()}")
        
        self.last_contexts[opponent_id] = {'state': state, 'action': action}
        return 0 if action == 'cooperate' else 1
    
    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        super().record_pairwise_outcome(opponent_id, my_move, opponent_move, reward)
        
        if self.verbose and self.round_num <= 10:
            context = self.last_contexts.get(opponent_id)
            if context:
                # Get the values used in update
                q_table = self.q_tables[opponent_id]
                state = context['state']
                action = context['action']
                
                # Get next state
                next_state = self._get_state(opponent_id)
                if next_state not in q_table:
                    q_table[next_state] = self._make_q_dict()
                
                # Values for calculation
                old_q = q_table[state][action]
                max_next_q = max(q_table[next_state].values())
                lr = self.learning_rates.get(opponent_id, self._get_initial_lr())
                df = self.params.get('df', 0.9)
                
                # The actual update (already done by super())
                new_q = q_table[state][action]
                
                print(f"\nOUTCOME & Q-UPDATE:")
                print(f"My action: {'COOPERATE' if my_move == 0 else 'DEFECT'}")
                print(f"Opponent action: {'COOPERATE' if opponent_move == 0 else 'DEFECT'}")
                print(f"Reward received: {reward}")
                print(f"Next state: '{next_state}'")
                print(f"\nQ-LEARNING CALCULATION:")
                print(f"Old Q({state}, {action}) = {old_q:.3f}")
                print(f"Learning rate (α) = {lr:.3f}")
                print(f"Discount factor (γ) = {df:.3f}")
                print(f"Max Q(next_state) = {max_next_q:.3f}")
                print(f"\nTD Target = reward + γ × max_next_Q")
                print(f"         = {reward} + {df} × {max_next_q:.3f}")
                print(f"         = {reward + df * max_next_q:.3f}")
                print(f"\nTD Error = TD_Target - Old_Q")
                print(f"         = {reward + df * max_next_q:.3f} - {old_q:.3f}")
                print(f"         = {reward + df * max_next_q - old_q:.3f}")
                print(f"\nNew Q = Old_Q + α × TD_Error")
                print(f"      = {old_q:.3f} + {lr:.3f} × {reward + df * max_next_q - old_q:.3f}")
                print(f"      = {new_q:.3f}")
                print(f"\nQ-table after update:")
                for s in sorted(q_table.keys()):
                    print(f"  {s}: C={q_table[s]['cooperate']:.3f}, D={q_table[s]['defect']:.3f}")

# Test different scenarios
print("\n" + "="*80)
print("SCENARIO 1: Q-LEARNER vs ALWAYS COOPERATE")
print("="*80)

# Use your actual parameters
test_params = VANILLA_PARAMS.copy()
print(f"\nParameters: {test_params}")

ql1 = VerboseQLearner("QL", test_params)
allc = StaticAgent("AllC", "AllC")

# Simulate some rounds
for round in range(10):
    # QL chooses action
    ql_move = ql1.choose_pairwise_action("AllC")
    allc_move = allc.choose_pairwise_action("QL")
    
    # Calculate payoffs
    if ql_move == 0 and allc_move == 0:
        ql_payoff, allc_payoff = R, R
    elif ql_move == 0 and allc_move == 1:
        ql_payoff, allc_payoff = S, T
    elif ql_move == 1 and allc_move == 0:
        ql_payoff, allc_payoff = T, S
    else:
        ql_payoff, allc_payoff = P, P
    
    # Record outcomes
    ql1.record_pairwise_outcome("AllC", ql_move, allc_move, ql_payoff)
    allc.record_pairwise_outcome("QL", allc_move, ql_move, allc_payoff)

# Turn off verbose for remaining rounds
ql1.verbose = False

print("\n" + "="*80)
print("CONTINUING SILENTLY FOR 90 MORE ROUNDS...")
print("="*80)

for round in range(90):
    ql_move = ql1.choose_pairwise_action("AllC")
    allc_move = allc.choose_pairwise_action("QL")
    
    if ql_move == 0 and allc_move == 0:
        ql_payoff, allc_payoff = R, R
    elif ql_move == 0 and allc_move == 1:
        ql_payoff, allc_payoff = S, T
    elif ql_move == 1 and allc_move == 0:
        ql_payoff, allc_payoff = T, S
    else:
        ql_payoff, allc_payoff = P, P
    
    ql1.record_pairwise_outcome("AllC", ql_move, allc_move, ql_payoff)
    allc.record_pairwise_outcome("QL", allc_move, ql_move, allc_payoff)

print("\nFINAL Q-TABLE (after 100 rounds):")
q_table = ql1.q_tables["AllC"]
for state in sorted(q_table.keys()):
    q_c = q_table[state]['cooperate']
    q_d = q_table[state]['defect']
    best = 'COOPERATE' if q_c > q_d else 'DEFECT' if q_d > q_c else 'EQUAL'
    print(f"  {state}: C={q_c:.3f}, D={q_d:.3f} -> {best}")

print(f"\nTotal score: QL={ql1.total_score}, AllC={allc.total_score}")

# Create visualization of Q-table evolution
print("\n" + "="*80)
print("VISUALIZING Q-TABLE EVOLUTION")
print("="*80)

# Track Q-values over more rounds
ql2 = PairwiseAdaptiveQLearner("QL2", test_params)
allc2 = StaticAgent("AllC2", "AllC")

# Track specific Q-values
q_evolution = {
    'rounds': [],
    'Q(after_C, cooperate)': [],
    'Q(after_C, defect)': [],
}

for round in range(500):
    # Play
    ql_move = ql2.choose_pairwise_action("AllC2")
    allc_move = allc2.choose_pairwise_action("QL2")
    
    if ql_move == 0 and allc_move == 0:
        ql_payoff = R
    elif ql_move == 1 and allc_move == 0:
        ql_payoff = T
    else:
        ql_payoff = S  # Shouldn't happen vs AllC
    
    ql2.record_pairwise_outcome("AllC2", ql_move, allc_move, ql_payoff)
    allc2.record_pairwise_outcome("QL2", allc_move, 0, R)
    
    # Track Q-values
    if round % 5 == 0:  # Sample every 5 rounds
        q_table = ql2.q_tables.get("AllC2", {})
        if "('defect', 'cooperate')" in q_table:  # After we defect, they cooperate
            q_evolution['rounds'].append(round)
            q_evolution['Q(after_C, cooperate)'].append(q_table["('defect', 'cooperate')"]['cooperate'])
            q_evolution['Q(after_C, defect)'].append(q_table["('defect', 'cooperate')"]['defect'])

# Plot
if q_evolution['rounds']:
    plt.figure(figsize=(10, 6))
    plt.plot(q_evolution['rounds'], q_evolution['Q(after_C, cooperate)'], 
             label='Q(opponent_cooperated, cooperate)', color='green', linewidth=2)
    plt.plot(q_evolution['rounds'], q_evolution['Q(after_C, defect)'], 
             label='Q(opponent_cooperated, defect)', color='red', linewidth=2)
    
    plt.xlabel('Round')
    plt.ylabel('Q-Value')
    plt.title(f'Q-Value Evolution: Q-Learner vs Always Cooperate\n(γ={test_params["df"]}, α={test_params["lr"]}, ε={test_params["eps"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark when defection becomes preferred
    crossover = None
    for i in range(len(q_evolution['rounds'])):
        if q_evolution['Q(after_C, defect)'][i] > q_evolution['Q(after_C, cooperate)'][i]:
            crossover = q_evolution['rounds'][i]
            break
    
    if crossover:
        plt.axvline(x=crossover, color='black', linestyle='--', alpha=0.5)
        plt.text(crossover, plt.ylim()[1]*0.9, f'Defection preferred\nafter round {crossover}', 
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('q_value_evolution_trace.png', dpi=150)
    print("Saved Q-value evolution as: q_value_evolution_trace.png")

print("\n" + "="*80)
print("KEY INSIGHTS FROM THIS TRACE:")
print("="*80)
print("1. The agent starts with Q=0 for all state-action pairs")
print("2. Each experience updates exactly one Q-value")
print("3. The discount factor (γ) determines how much future matters")
print("4. Against AllC, Q(defect) grows faster than Q(cooperate)")
print("5. Eventually, the agent learns to exploit the cooperator")

plt.show()