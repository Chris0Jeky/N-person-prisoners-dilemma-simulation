#!/usr/bin/env python3
"""
Complete Numerical Example of Q-Learning
Shows exact calculations step by step
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Payoffs
T, R, P, S = 5, 3, 1, 0

print("=" * 80)
print("COMPLETE NUMERICAL Q-LEARNING EXAMPLE")
print("=" * 80)

# Parameters (using your settings)
learning_rate = 0.1
discount_factor = 0.6  # Your changed value
epsilon = 0.2

print(f"\nPARAMETERS:")
print(f"Learning rate (α) = {learning_rate}")
print(f"Discount factor (γ) = {discount_factor}")
print(f"Exploration rate (ε) = {epsilon}")

print(f"\nPAYOFF MATRIX:")
print(f"Both cooperate (R,R) = ({R},{R})")
print(f"I defect, they cooperate (T,S) = ({T},{S})")
print(f"I cooperate, they defect (S,T) = ({S},{T})")
print(f"Both defect (P,P) = ({P},{P})")

# Initialize Q-table
Q = {
    "start": {"C": 0.0, "D": 0.0},
    "opp_cooperated": {"C": 0.0, "D": 0.0},
    "opp_defected": {"C": 0.0, "D": 0.0}
}

print("\n" + "=" * 80)
print("STEP-BY-STEP Q-LEARNING PROCESS")
print("=" * 80)

# Simulate 5 detailed rounds
rounds = [
    {
        "round": 1,
        "state": "start",
        "epsilon_roll": 0.15,
        "action_choice": "explore",
        "random_action": "defect",
        "opponent_action": "cooperate",
        "my_payoff": T,
        "next_state": "opp_defected"  # Opponent will retaliate
    },
    {
        "round": 2,
        "state": "opp_defected",
        "epsilon_roll": 0.85,
        "action_choice": "exploit",
        "best_action": "defect",  # Both Q-values are 0, so could be either
        "opponent_action": "defect",
        "my_payoff": P,
        "next_state": "opp_defected"
    },
    {
        "round": 3,
        "state": "opp_defected",
        "epsilon_roll": 0.05,
        "action_choice": "explore",
        "random_action": "cooperate",
        "opponent_action": "defect",
        "my_payoff": S,
        "next_state": "opp_cooperated"  # We cooperated
    },
    {
        "round": 4,
        "state": "opp_cooperated",
        "epsilon_roll": 0.75,
        "action_choice": "exploit",
        "best_action": None,  # Will be determined by Q-values
        "opponent_action": "cooperate",
        "my_payoff": None,  # Depends on our action
        "next_state": None
    },
    {
        "round": 5,
        "state": None,  # Depends on round 4
        "epsilon_roll": 0.90,
        "action_choice": "exploit",
        "best_action": None,
        "opponent_action": None,
        "my_payoff": None,
        "next_state": None
    }
]

# Process each round
for r in rounds[:3]:  # First 3 rounds are predetermined
    print(f"\n{'='*60}")
    print(f"ROUND {r['round']}")
    print(f"{'='*60}")
    
    state = r['state']
    print(f"\n1. CURRENT STATE: '{state}'")
    print(f"   Q-values: C={Q[state]['C']:.3f}, D={Q[state]['D']:.3f}")
    
    print(f"\n2. ACTION SELECTION:")
    print(f"   Random roll: {r['epsilon_roll']:.3f}")
    print(f"   Since {r['epsilon_roll']:.3f} {'<' if r['epsilon_roll'] < epsilon else '>='} ε={epsilon}, we {r['action_choice'].upper()}")
    
    if r['action_choice'] == 'explore':
        action = r['random_action']
        print(f"   Random choice: {action.upper()}")
    else:
        if Q[state]['C'] > Q[state]['D']:
            action = 'cooperate'
        elif Q[state]['D'] > Q[state]['C']:
            action = 'defect'
        else:
            action = r.get('best_action', 'defect')  # Tie-breaker
        print(f"   Best Q-value: {action.upper()}")
    
    print(f"\n3. OUTCOME:")
    print(f"   My action: {action.upper()}")
    print(f"   Opponent action: {r['opponent_action'].upper()}")
    print(f"   My reward: {r['my_payoff']}")
    print(f"   Next state: '{r['next_state']}'")
    
    print(f"\n4. Q-VALUE UPDATE:")
    
    # Calculate update
    old_q = Q[state][action[0].upper()]
    next_state = r['next_state']
    max_next_q = max(Q[next_state]['C'], Q[next_state]['D'])
    
    print(f"   Old Q({state}, {action}) = {old_q:.3f}")
    print(f"   Max Q({next_state}, any) = {max_next_q:.3f}")
    
    print(f"\n   TD Target = r + γ × max_next_Q")
    print(f"            = {r['my_payoff']} + {discount_factor} × {max_next_q:.3f}")
    td_target = r['my_payoff'] + discount_factor * max_next_q
    print(f"            = {td_target:.3f}")
    
    print(f"\n   TD Error = TD_Target - Old_Q")
    print(f"           = {td_target:.3f} - {old_q:.3f}")
    td_error = td_target - old_q
    print(f"           = {td_error:.3f}")
    
    print(f"\n   New Q = Old_Q + α × TD_Error")
    print(f"        = {old_q:.3f} + {learning_rate} × {td_error:.3f}")
    new_q = old_q + learning_rate * td_error
    print(f"        = {new_q:.3f}")
    
    # Update Q-table
    Q[state][action[0].upper()] = new_q
    
    print(f"\n5. UPDATED Q-TABLE:")
    for s in ["start", "opp_cooperated", "opp_defected"]:
        print(f"   {s}: C={Q[s]['C']:.3f}, D={Q[s]['D']:.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Q-Learning Numerical Example: First 3 Rounds', fontsize=16, fontweight='bold')

# Round 1 visualization
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.text(5, 9, 'Round 1: First Interaction', ha='center', fontsize=14, fontweight='bold')

# State and Q-values
state_box = FancyBboxPatch((1, 7), 3, 1.5, boxstyle="round,pad=0.1",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(state_box)
ax1.text(2.5, 7.75, 'State: "start"', ha='center', fontsize=11)
ax1.text(2.5, 6.3, 'Q(C)=0.0, Q(D)=0.0', ha='center', fontsize=10)

# Action selection
ax1.text(5, 5.5, 'Roll=0.15 < ε=0.2 → EXPLORE', ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='orange'))
ax1.text(5, 4.8, 'Random: DEFECT', ha='center', fontsize=11, fontweight='bold', color='red')

# Outcome
ax1.text(5, 3.5, 'Outcome: I defect, Opp cooperates', ha='center', fontsize=11)
ax1.text(5, 2.8, 'Reward = T = 5', ha='center', fontsize=11, color='green')

# Q-update calculation
calc_text = f'Q-update: 0.0 + 0.1×(5 + 0.6×0.0 - 0.0) = 0.5'
ax1.text(5, 1.5, calc_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow'))

# Round 2 visualization
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.text(5, 9, 'Round 2: After Defection', ha='center', fontsize=14, fontweight='bold')

state_box = FancyBboxPatch((1, 7), 3, 1.5, boxstyle="round,pad=0.1",
                          facecolor='lightblue', edgecolor='black', linewidth=2)
ax2.add_patch(state_box)
ax2.text(2.5, 7.75, 'State: "opp_defected"', ha='center', fontsize=11)
ax2.text(2.5, 6.3, 'Q(C)=0.0, Q(D)=0.0', ha='center', fontsize=10)

ax2.text(5, 5.5, 'Roll=0.85 > ε=0.2 → EXPLOIT', ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax2.text(5, 4.8, 'Best: DEFECT (tie→defect)', ha='center', fontsize=11, fontweight='bold', color='red')

ax2.text(5, 3.5, 'Outcome: Both defect', ha='center', fontsize=11)
ax2.text(5, 2.8, 'Reward = P = 1', ha='center', fontsize=11)

calc_text = f'Q-update: 0.0 + 0.1×(1 + 0.6×0.0 - 0.0) = 0.1'
ax2.text(5, 1.5, calc_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow'))

# Q-table evolution
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.text(5, 9, 'Q-Table Evolution', ha='center', fontsize=14, fontweight='bold')

# Draw Q-tables after each round
tables_y = [7, 5, 3]
rounds_data = [
    {"title": "Initial", "values": {"start": (0, 0), "opp_coop": (0, 0), "opp_def": (0, 0)}},
    {"title": "After R1", "values": {"start": (0, 0.5), "opp_coop": (0, 0), "opp_def": (0, 0)}},
    {"title": "After R2", "values": {"start": (0, 0.5), "opp_coop": (0, 0), "opp_def": (0, 0.1)}},
]

for i, (y, data) in enumerate(zip(tables_y, rounds_data)):
    ax3.text(1, y + 1.3, data["title"], fontsize=11, fontweight='bold')
    
    # Table headers
    ax3.text(3, y + 0.8, 'State', ha='center', fontsize=9)
    ax3.text(5, y + 0.8, 'Q(C)', ha='center', fontsize=9)
    ax3.text(7, y + 0.8, 'Q(D)', ha='center', fontsize=9)
    
    # Table data
    states = [("start", "start"), ("opp_coop", "opp_C"), ("opp_def", "opp_D")]
    for j, (key, label) in enumerate(states):
        q_c, q_d = data["values"][key]
        ax3.text(3, y + 0.3 - j*0.3, label, ha='center', fontsize=8)
        ax3.text(5, y + 0.3 - j*0.3, f'{q_c:.1f}', ha='center', fontsize=8)
        ax3.text(7, y + 0.3 - j*0.3, f'{q_d:.1f}', ha='center', fontsize=8)
        
        # Highlight changed values
        if i > 0:
            prev_values = rounds_data[i-1]["values"][key]
            if q_c != prev_values[0]:
                ax3.add_patch(Rectangle((4.7, y + 0.15 - j*0.3), 0.6, 0.25, 
                                      facecolor='yellow', alpha=0.5))
            if q_d != prev_values[1]:
                ax3.add_patch(Rectangle((6.7, y + 0.15 - j*0.3), 0.6, 0.25, 
                                      facecolor='yellow', alpha=0.5))

# Key insights
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.text(5, 9, 'Key Insights', ha='center', fontsize=14, fontweight='bold')

insights = [
    "1. Q-values start at 0 (no prior knowledge)",
    "2. Only one Q-value updates per round",
    "3. The update depends on:",
    "   • Immediate reward (r)",
    "   • Future value (γ × max Q)",
    "   • Learning rate (α)",
    "",
    f"4. With γ={discount_factor}:",
    f"   • Future is worth {discount_factor:.0%} per step",
    f"   • Agent is somewhat myopic",
    "",
    "5. Against cooperators:",
    "   • Q(defect) grows faster",
    "   • T=5 > R=3 dominates"
]

for i, insight in enumerate(insights):
    y_pos = 7.5 - i*0.5
    if insight.startswith("   "):
        ax4.text(3, y_pos, insight, fontsize=9, style='italic')
    else:
        ax4.text(1, y_pos, insight, fontsize=10)

plt.tight_layout()
plt.savefig('numerical_q_learning_example.png', dpi=150, bbox_inches='tight')
print("\nSaved numerical example as: numerical_q_learning_example.png")

# Create a comparison chart showing effect of different gamma values
fig2, ax = plt.subplots(figsize=(10, 6))

# Calculate TD targets for different scenarios and gamma values
gammas = [0.0, 0.2, 0.6, 0.9]
scenarios = [
    ("Defect vs Cooperator", T, R),  # Immediate=5, Future≈3
    ("Cooperate vs Cooperator", R, R),  # Immediate=3, Future≈3
    ("Defect vs Defector", P, P),  # Immediate=1, Future≈1
    ("Cooperate vs Defector", S, P),  # Immediate=0, Future≈1
]

x = np.arange(len(scenarios))
width = 0.2

for i, gamma in enumerate(gammas):
    td_targets = []
    for _, immediate, future in scenarios:
        td_target = immediate + gamma * future
        td_targets.append(td_target)
    
    ax.bar(x + i*width - 1.5*width, td_targets, width, 
           label=f'γ={gamma}', alpha=0.8)

ax.set_xlabel('Scenario')
ax.set_ylabel('TD Target Value')
ax.set_title('How Discount Factor Affects Value Estimates')
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in scenarios], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, gamma in enumerate(gammas):
    for j, (_, immediate, future) in enumerate(scenarios):
        td_target = immediate + gamma * future
        ax.text(j + i*width - 1.5*width, td_target + 0.1, f'{td_target:.1f}', 
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('discount_factor_comparison_chart.png', dpi=150, bbox_inches='tight')
print("Saved discount factor comparison as: discount_factor_comparison_chart.png")

print("\n" + "=" * 80)
print("SUMMARY: HOW Q-LEARNING MAKES DECISIONS")
print("=" * 80)

print("\n1. STATE REPRESENTATION:")
print("   - Your implementation uses history: last 2 interactions")
print("   - States like ((C,D),(D,C)) = 'I cooperated, they defected, then I defected, they cooperated'")

print("\n2. ACTION SELECTION (ε-greedy):")
print(f"   - {epsilon:.0%} of the time: EXPLORE (random action)")
print(f"   - {1-epsilon:.0%} of the time: EXPLOIT (best Q-value)")

print("\n3. Q-VALUE UPDATE:")
print("   Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]")
print(f"   - α={learning_rate} controls learning speed")
print(f"   - γ={discount_factor} controls future importance")

print("\n4. CONVERGENCE:")
print("   - Q-values stabilize as agent learns")
print("   - Final values determine strategy")
print("   - Higher Q(defect) → exploitative behavior")
print("   - Higher Q(cooperate) → cooperative behavior")

plt.show()