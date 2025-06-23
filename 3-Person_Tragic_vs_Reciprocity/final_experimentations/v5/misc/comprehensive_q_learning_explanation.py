#!/usr/bin/env python3
"""
Comprehensive Q-Learning Explanation with Step-by-Step Visualization
Shows exactly how Q-values are updated and how agents make decisions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# Set up the payoff matrix
T, R, P, S = 5, 3, 1, 0

print("=" * 80)
print("COMPREHENSIVE Q-LEARNING EXPLANATION")
print("=" * 80)

# Create a large figure with multiple subplots
fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.2)

# Title
fig.suptitle('Complete Q-Learning Process in Prisoner\'s Dilemma', fontsize=20, fontweight='bold')

# ============= SUBPLOT 1: Q-TABLE STRUCTURE =============
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.text(5, 4.5, '1. Q-TABLE STRUCTURE', ha='center', fontsize=16, fontweight='bold')

# Draw Q-table
table_data = [
    ['State', 'Cooperate', 'Defect'],
    ['Start (first round)', 'Q=0.0', 'Q=0.0'],
    ['After opponent cooperated', 'Q=?', 'Q=?'],
    ['After opponent defected', 'Q=?', 'Q=?']
]

table_x, table_y = 2, 1
cell_width, cell_height = 2, 0.5

for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        color = 'lightgray' if i == 0 else 'white'
        rect = Rectangle((table_x + j*cell_width, table_y - i*cell_height), 
                        cell_width, cell_height, 
                        facecolor=color, edgecolor='black')
        ax1.add_patch(rect)
        ax1.text(table_x + j*cell_width + cell_width/2, 
                table_y - i*cell_height + cell_height/2, 
                cell, ha='center', va='center', fontsize=9)

ax1.text(5, 0.2, 'Q(state, action) stores the expected value of taking action in state', 
         ha='center', fontsize=11, style='italic')

# ============= SUBPLOT 2: ACTION SELECTION =============
ax2 = fig.add_subplot(gs[1, :])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')
ax2.text(5, 4.5, '2. HOW AGENT CHOOSES ACTIONS (ε-greedy)', ha='center', fontsize=16, fontweight='bold')

# Epsilon-greedy visualization
eps = 0.2  # Your current epsilon

# Draw probability bar
bar_x, bar_y = 1, 2
bar_width, bar_height = 8, 1

# Exploration part
explore_rect = Rectangle((bar_x, bar_y), bar_width * eps, bar_height,
                        facecolor='orange', edgecolor='black', linewidth=2)
ax2.add_patch(explore_rect)
ax2.text(bar_x + bar_width*eps/2, bar_y + bar_height/2, 
         f'EXPLORE\n{eps:.0%}', ha='center', va='center', fontweight='bold')

# Exploitation part
exploit_rect = Rectangle((bar_x + bar_width*eps, bar_y), bar_width * (1-eps), bar_height,
                        facecolor='lightgreen', edgecolor='black', linewidth=2)
ax2.add_patch(exploit_rect)
ax2.text(bar_x + bar_width*eps + bar_width*(1-eps)/2, bar_y + bar_height/2, 
         f'EXPLOIT\n{1-eps:.0%}', ha='center', va='center', fontweight='bold')

# Explanations
ax2.text(2, 1.3, 'Random choice\n(explore new actions)', ha='center', fontsize=9)
ax2.text(7, 1.3, 'Choose best Q-value\n(use learned knowledge)', ha='center', fontsize=9)

# Example
ax2.text(5, 0.5, 'Example: If Q(s,C)=2.5 and Q(s,D)=3.1, then:', ha='center', fontsize=10)
ax2.text(5, 0.1, '• 20% chance: random choice (C or D with 50/50)   • 80% chance: choose D (higher Q)', 
         ha='center', fontsize=9)

# ============= SUBPLOT 3: Q-LEARNING UPDATE FORMULA =============
ax3 = fig.add_subplot(gs[2, :])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 5)
ax3.axis('off')
ax3.text(5, 4.5, '3. Q-LEARNING UPDATE FORMULA', ha='center', fontsize=16, fontweight='bold')

# Main formula
formula_box = FancyBboxPatch((0.5, 2.5), 9, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='lightyellow',
                            edgecolor='black', linewidth=2)
ax3.add_patch(formula_box)
ax3.text(5, 3.1, r'Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s′,a′) - Q(s,a)]', 
         ha='center', va='center', fontsize=14, fontweight='bold')

# Component explanations
components = [
    ('Q(s,a)', 'Current Q-value', 1),
    ('α', 'Learning rate (0.1)', 2.5),
    ('r', 'Immediate reward', 4),
    ('γ', 'Discount factor (0.6)', 5.5),
    ('max Q(s′,a′)', 'Best future value', 7),
    ('TD Error', 'Temporal Difference', 8.5)
]

for comp, desc, x in components:
    ax3.text(x, 2, comp, ha='center', fontsize=11, fontweight='bold')
    ax3.text(x, 1.6, desc, ha='center', fontsize=9)

# TD Error explanation
td_box = FancyBboxPatch((0.5, 0.5), 9, 0.8,
                       boxstyle="round,pad=0.05",
                       facecolor='lightblue',
                       edgecolor='blue', linewidth=1)
ax3.add_patch(td_box)
ax3.text(5, 0.9, 'TD Error = [actual value] - [expected value] = [r + γ×future] - [current Q]', 
         ha='center', va='center', fontsize=10)

# ============= SUBPLOT 4: STEP-BY-STEP EXAMPLE =============
ax4 = fig.add_subplot(gs[3:5, :])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.text(5, 9.5, '4. COMPLETE EXAMPLE: Q-LEARNER vs TIT-FOR-TAT', ha='center', fontsize=16, fontweight='bold')

# Initialize values for example
state = "opponent_cooperated"
Q_values = {"cooperate": 1.8, "defect": 2.2}
epsilon = 0.2
learning_rate = 0.1
discount_factor = 0.6

# Step 1: Current situation
y_pos = 8.5
ax4.text(0.5, y_pos, 'STEP 1: Current Situation', fontsize=12, fontweight='bold')
ax4.text(0.5, y_pos-0.4, f'• State: "{state}" (opponent cooperated last round)', fontsize=10)
ax4.text(0.5, y_pos-0.8, f'• Current Q-values: Q(s,C)={Q_values["cooperate"]}, Q(s,D)={Q_values["defect"]}', fontsize=10)

# Step 2: Action selection
y_pos = 7.3
ax4.text(0.5, y_pos, 'STEP 2: Choose Action (ε-greedy)', fontsize=12, fontweight='bold')
ax4.text(0.5, y_pos-0.4, f'• Roll random number: 0.73', fontsize=10)
ax4.text(0.5, y_pos-0.8, f'• Since 0.73 > ε={epsilon}, we EXPLOIT (choose best Q-value)', fontsize=10)
ax4.text(0.5, y_pos-1.2, f'• Best action: DEFECT (Q={Q_values["defect"]} > Q={Q_values["cooperate"]})', fontsize=10, color='red')

# Step 3: Take action and observe
y_pos = 5.7
ax4.text(0.5, y_pos, 'STEP 3: Take Action and Observe Result', fontsize=12, fontweight='bold')
ax4.text(0.5, y_pos-0.4, f'• We DEFECT, opponent COOPERATES', fontsize=10)
ax4.text(0.5, y_pos-0.8, f'• Immediate reward: r = T = {T}', fontsize=10, color='green')
ax4.text(0.5, y_pos-1.2, f'• Next state: "opponent_defected" (TFT will retaliate)', fontsize=10)

# Step 4: Look up next state values
y_pos = 4.1
ax4.text(0.5, y_pos, 'STEP 4: Evaluate Future (Next State)', fontsize=12, fontweight='bold')
ax4.text(0.5, y_pos-0.4, f'• Next state Q-values: Q(opponent_defected, C)=0.5, Q(opponent_defected, D)=1.2', fontsize=10)
ax4.text(0.5, y_pos-0.8, f'• Best future value: max Q(s′,a′) = 1.2', fontsize=10)

# Step 5: Calculate update
y_pos = 2.7
ax4.text(0.5, y_pos, 'STEP 5: Calculate Q-Value Update', fontsize=12, fontweight='bold')

# Detailed calculation
calc_box = FancyBboxPatch((0.5, 0.3), 9, 2.2,
                         boxstyle="round,pad=0.1",
                         facecolor='lightgreen',
                         edgecolor='green', linewidth=2)
ax4.add_patch(calc_box)

calc_y = 2.2
ax4.text(5, calc_y, 'Q(opponent_cooperated, defect) update:', ha='center', fontsize=11, fontweight='bold')
ax4.text(5, calc_y-0.4, f'TD Target = r + γ × max Q(s′,a′) = {T} + {discount_factor} × 1.2 = {T + discount_factor * 1.2:.2f}', 
         ha='center', fontsize=10)
ax4.text(5, calc_y-0.8, f'TD Error = TD Target - Current Q = {T + discount_factor * 1.2:.2f} - {Q_values["defect"]} = {T + discount_factor * 1.2 - Q_values["defect"]:.2f}', 
         ha='center', fontsize=10)
ax4.text(5, calc_y-1.2, f'New Q = Old Q + α × TD Error = {Q_values["defect"]} + {learning_rate} × {T + discount_factor * 1.2 - Q_values["defect"]:.2f}', 
         ha='center', fontsize=10)
ax4.text(5, calc_y-1.6, f'New Q = {Q_values["defect"] + learning_rate * (T + discount_factor * 1.2 - Q_values["defect"]):.3f}', 
         ha='center', fontsize=11, fontweight='bold', color='green')

# ============= SUBPLOT 5: STATE REPRESENTATION =============
ax5 = fig.add_subplot(gs[5, 0])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 5)
ax5.axis('off')
ax5.text(5, 4.5, '5. STATE REPRESENTATION', ha='center', fontsize=14, fontweight='bold')

# Show how states are determined
ax5.text(5, 3.8, 'Pairwise Game States:', ha='center', fontsize=11, fontweight='bold')
ax5.text(5, 3.4, '• "start": First interaction with opponent', ha='center', fontsize=9)
ax5.text(5, 3.0, '• History-based: Last 2 moves (mine, theirs)', ha='center', fontsize=9)
ax5.text(5, 2.6, '• Example: ((C,D), (D,C)) = "I cooperated, they defected,', ha='center', fontsize=9)
ax5.text(5, 2.3, '           then I defected, they cooperated"', ha='center', fontsize=9)

ax5.text(5, 1.7, 'Neighborhood Game States:', ha='center', fontsize=11, fontweight='bold')
ax5.text(5, 1.3, '• Based on cooperation ratio', ha='center', fontsize=9)
ax5.text(5, 0.9, '• "low": ratio ≤ 0.33', ha='center', fontsize=9)
ax5.text(5, 0.5, '• "medium": 0.33 < ratio ≤ 0.67', ha='center', fontsize=9)
ax5.text(5, 0.1, '• "high": ratio > 0.67', ha='center', fontsize=9)

# ============= SUBPLOT 6: LEARNING PROGRESSION =============
ax6 = fig.add_subplot(gs[5, 1])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 5)
ax6.axis('off')
ax6.text(5, 4.5, '6. LEARNING PROGRESSION', ha='center', fontsize=14, fontweight='bold')

# Show how Q-values evolve
ax6.text(5, 3.8, 'Early Learning (Random Q-values):', ha='center', fontsize=10, fontweight='bold')
ax6.text(5, 3.4, '→ Random exploration, learning payoffs', ha='center', fontsize=9)

ax6.text(5, 2.8, 'Mid Learning (Patterns emerge):', ha='center', fontsize=10, fontweight='bold')
ax6.text(5, 2.4, '→ Q(s,defect) > Q(s,cooperate) against AllC', ha='center', fontsize=9)
ax6.text(5, 2.0, '→ Q(s,cooperate) ≈ Q(s,defect) against TFT', ha='center', fontsize=9)

ax6.text(5, 1.4, 'Late Learning (Converged):', ha='center', fontsize=10, fontweight='bold')
ax6.text(5, 1.0, '→ Stable strategy emerged', ha='center', fontsize=9)
ax6.text(5, 0.6, '→ Exploration still causes occasional changes', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('comprehensive_q_learning_explanation.png', dpi=150, bbox_inches='tight')
print("Saved comprehensive explanation as: comprehensive_q_learning_explanation.png")

# ============= SECOND FIGURE: Q-VALUE EVOLUTION =============
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Q-Value Evolution During Learning', fontsize=16, fontweight='bold')

# Simulate learning against different opponents
opponents = ['AllC', 'TFT', 'AllD', 'Random']
opponent_behaviors = {
    'AllC': lambda hist: 0,  # Always cooperate
    'TFT': lambda hist: hist[-1] if hist else 0,  # Copy last move
    'AllD': lambda hist: 1,  # Always defect
    'Random': lambda hist: np.random.randint(2)  # Random
}

for idx, (opponent, behavior) in enumerate(opponent_behaviors.items()):
    ax = axes2[idx // 2, idx % 2]
    
    # Initialize Q-table
    Q = {
        'start': {'C': 0.0, 'D': 0.0},
        'after_C': {'C': 0.0, 'D': 0.0},
        'after_D': {'C': 0.0, 'D': 0.0}
    }
    
    # Track Q-values over time
    q_history = {
        'after_C_C': [],
        'after_C_D': [],
        'after_D_C': [],
        'after_D_D': []
    }
    
    # Learning parameters
    lr = 0.1
    gamma = 0.6
    epsilon = 0.2
    
    # Simulate 200 rounds
    my_history = []
    opp_history = []
    state = 'start'
    
    for round in range(200):
        # Choose action (ε-greedy)
        if np.random.random() < epsilon:
            action = 'C' if np.random.random() < 0.5 else 'D'
        else:
            action = 'C' if Q[state]['C'] >= Q[state]['D'] else 'D'
        
        # Opponent responds
        opp_action_num = behavior(my_history)
        opp_action = 'C' if opp_action_num == 0 else 'D'
        
        # Get reward
        my_move = 0 if action == 'C' else 1
        opp_move = 0 if opp_action == 'C' else 1
        
        if my_move == 0 and opp_move == 0:
            reward = R
        elif my_move == 0 and opp_move == 1:
            reward = S
        elif my_move == 1 and opp_move == 0:
            reward = T
        else:
            reward = P
        
        # Determine next state
        next_state = 'after_C' if opp_action == 'C' else 'after_D'
        
        # Q-learning update
        max_next_q = max(Q[next_state].values())
        td_target = reward + gamma * max_next_q
        Q[state][action] = Q[state][action] + lr * (td_target - Q[state][action])
        
        # Record history
        if state == 'after_C':
            q_history['after_C_C'].append(Q['after_C']['C'])
            q_history['after_C_D'].append(Q['after_C']['D'])
        elif state == 'after_D':
            q_history['after_D_C'].append(Q['after_D']['C'])
            q_history['after_D_D'].append(Q['after_D']['D'])
        
        # Update history and state
        my_history.append(my_move)
        opp_history.append(opp_move)
        state = next_state
    
    # Plot Q-values
    ax.plot(q_history['after_C_C'], label='Q(after_C, Cooperate)', color='green', linewidth=2)
    ax.plot(q_history['after_C_D'], label='Q(after_C, Defect)', color='red', linewidth=2)
    ax.set_title(f'Learning vs {opponent}')
    ax.set_xlabel('Updates')
    ax.set_ylabel('Q-Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final strategy text
    final_strategy = "Cooperate" if Q['after_C']['C'] > Q['after_C']['D'] else "Defect"
    ax.text(0.95, 0.95, f'Final: {final_strategy}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('q_value_evolution_by_opponent.png', dpi=150, bbox_inches='tight')
print("Saved Q-value evolution as: q_value_evolution_by_opponent.png")

# ============= THIRD FIGURE: DECISION FLOW CHART =============
fig3, ax3 = plt.subplots(figsize=(12, 10))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

# Title
ax3.text(5, 9.5, 'Q-Learning Agent Decision Flow', ha='center', fontsize=18, fontweight='bold')

# Flow chart elements
elements = [
    # (x, y, width, height, text, color)
    (4, 8, 2, 0.8, '1. Observe State', 'lightblue'),
    (4, 6.8, 2, 0.8, '2. Look up Q-values', 'lightgreen'),
    (1.5, 5.3, 2.5, 0.8, 'Explore?', 'yellow'),
    (5.5, 5.3, 2.5, 0.8, 'Exploit?', 'yellow'),
    (1.5, 4, 2.5, 0.8, 'Random Action', 'orange'),
    (5.5, 4, 2.5, 0.8, 'Best Q Action', 'lightgreen'),
    (4, 2.7, 2, 0.8, '3. Execute Action', 'lightcoral'),
    (4, 1.5, 2, 0.8, '4. Observe Reward', 'lightblue'),
    (4, 0.3, 2, 0.8, '5. Update Q-value', 'lightgreen'),
]

for x, y, w, h, text, color in elements:
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black', linewidth=2)
    ax3.add_patch(box)
    ax3.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows
arrows = [
    (5, 7.8, 5, 7.2),  # 1 to 2
    (5, 6.4, 5, 5.7),  # 2 to decision
    (4.5, 5.3, 3.5, 5.3),  # to explore
    (5.5, 5.3, 6.5, 5.3),  # to exploit
    (2.75, 4.8, 2.75, 4.4),  # explore down
    (6.75, 4.8, 6.75, 4.4),  # exploit down
    (2.75, 3.6, 4, 3.1),  # explore to execute
    (6.75, 3.6, 5, 3.1),  # exploit to execute
    (5, 2.3, 5, 1.9),  # execute to reward
    (5, 1.1, 5, 0.7),  # reward to update
]

for x1, y1, x2, y2 in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           connectionstyle='arc3,rad=0',
                           linewidth=2, color='black')
    ax3.add_patch(arrow)

# Add probability labels
ax3.text(2.75, 5.5, f'ε = {epsilon}', ha='center', fontsize=10, style='italic')
ax3.text(6.75, 5.5, f'1-ε = {1-epsilon}', ha='center', fontsize=10, style='italic')

# Add loop back arrow
loop_arrow = FancyArrowPatch((5.5, 0.3), (8, 0.3), 
                            arrowstyle='->', 
                            connectionstyle='arc3,rad=.5',
                            linewidth=2, color='blue')
ax3.add_patch(loop_arrow)
ax3.text(8.5, 2, 'Repeat', ha='center', fontsize=10, color='blue', rotation=90)

plt.tight_layout()
plt.savefig('q_learning_decision_flow.png', dpi=150, bbox_inches='tight')
print("Saved decision flow as: q_learning_decision_flow.png")

print("\n" + "=" * 80)
print("SUMMARY OF KEY CONCEPTS:")
print("=" * 80)
print("\n1. Q-TABLE: Stores value estimates for each (state, action) pair")
print("2. ACTION SELECTION: ε-greedy balances exploration vs exploitation")
print("3. Q-UPDATE: Uses TD learning to improve estimates based on experience")
print("4. STATES: Represent game history or current situation")
print("5. CONVERGENCE: Q-values stabilize as agent learns optimal strategy")
print("\nAll visualizations have been saved as PNG files.")

plt.show()