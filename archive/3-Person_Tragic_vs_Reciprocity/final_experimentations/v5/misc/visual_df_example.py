#!/usr/bin/env python3
"""
Visual example of Q-learning calculations with discount factor
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle('Q-Learning Calculation Example: Effect of Discount Factor', fontsize=18, fontweight='bold')

# Payoffs
T, R, P, S = 5, 3, 1, 0

# Common parameters
lr = 0.1  # learning rate

# Example scenario: Agent is in state "opponent_cooperated", considering whether to C or D
state = "Opponent just Cooperated"
old_q_coop = 2.0  # Current Q-value for cooperation
old_q_defect = 2.5  # Current Q-value for defection

# Three different discount factors
discount_factors = [0.0, 0.6, 0.9]
scenarios = [
    "Myopic Agent (γ=0.0): Only cares about immediate reward",
    "Balanced Agent (γ=0.6): Your current setting",
    "Patient Agent (γ=0.9): Values future highly"
]

for idx, (ax, df, scenario) in enumerate(zip(axes, discount_factors, scenarios)):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9.5, scenario, ha='center', fontsize=14, fontweight='bold')
    
    # Draw state box
    state_box = FancyBboxPatch((0.5, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(state_box)
    ax.text(2, 7.75, f'State: {state}', ha='center', va='center', fontsize=11)
    
    # Current Q-values
    ax.text(0.5, 6, 'Current Q-values:', fontsize=11, fontweight='bold')
    ax.text(0.5, 5.5, f'Q(state, Cooperate) = {old_q_coop:.1f}', fontsize=10)
    ax.text(0.5, 5, f'Q(state, Defect) = {old_q_defect:.1f}', fontsize=10)
    
    # Action choices
    coop_box = FancyBboxPatch((0.5, 3), 4, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen',
                              edgecolor='green', linewidth=2)
    ax.add_patch(coop_box)
    ax.text(2.5, 3.75, 'Action: COOPERATE', ha='center', va='center', fontsize=11, fontweight='bold')
    
    defect_box = FancyBboxPatch((5.5, 3), 4, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightcoral',
                                edgecolor='red', linewidth=2)
    ax.add_patch(defect_box)
    ax.text(7.5, 3.75, 'Action: DEFECT', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Outcomes and calculations
    y_calc = 2.3
    
    # Cooperate calculation
    ax.text(0.5, y_calc, 'If Cooperate:', fontsize=10, fontweight='bold', color='green')
    ax.text(0.5, y_calc-0.4, f'• Immediate reward: R = {R}', fontsize=9)
    ax.text(0.5, y_calc-0.8, f'• Next state: "Both Cooperated"', fontsize=9)
    ax.text(0.5, y_calc-1.2, f'• Best next Q ≈ {R:.1f} (continue cooperating)', fontsize=9)
    
    td_target_c = R + df * R
    new_q_c = old_q_coop + lr * (td_target_c - old_q_coop)
    
    ax.text(0.5, y_calc-1.7, f'Q-update calculation:', fontsize=9, fontweight='bold')
    ax.text(0.5, y_calc-2.1, f'TD target = {R} + {df} × {R} = {td_target_c:.2f}', fontsize=9)
    ax.text(0.5, y_calc-2.5, f'New Q = {old_q_coop} + {lr} × ({td_target_c:.2f} - {old_q_coop}) = {new_q_c:.2f}', fontsize=9)
    
    # Defect calculation
    ax.text(5.5, y_calc, 'If Defect:', fontsize=10, fontweight='bold', color='red')
    ax.text(5.5, y_calc-0.4, f'• Immediate reward: T = {T}', fontsize=9)
    ax.text(5.5, y_calc-0.8, f'• Next state: "I Defected"', fontsize=9)
    ax.text(5.5, y_calc-1.2, f'• Best next Q ≈ {P:.1f} (mutual defection)', fontsize=9)
    
    td_target_d = T + df * P
    new_q_d = old_q_defect + lr * (td_target_d - old_q_defect)
    
    ax.text(5.5, y_calc-1.7, f'Q-update calculation:', fontsize=9, fontweight='bold')
    ax.text(5.5, y_calc-2.1, f'TD target = {T} + {df} × {P} = {td_target_d:.2f}', fontsize=9)
    ax.text(5.5, y_calc-2.5, f'New Q = {old_q_defect} + {lr} × ({td_target_d:.2f} - {old_q_defect}) = {new_q_d:.2f}', fontsize=9)
    
    # Decision
    if new_q_c > new_q_d:
        decision = "COOPERATE"
        decision_color = 'green'
        reason = f"Q(C)={new_q_c:.2f} > Q(D)={new_q_d:.2f}"
    else:
        decision = "DEFECT"
        decision_color = 'red'
        reason = f"Q(D)={new_q_d:.2f} > Q(C)={new_q_c:.2f}"
    
    decision_box = FancyBboxPatch((3, 0.1), 4, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='yellow',
                                  edgecolor=decision_color, linewidth=3)
    ax.add_patch(decision_box)
    ax.text(5, 0.5, f'Decision: {decision} ({reason})', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=decision_color)

plt.tight_layout()
plt.savefig('visual_discount_factor_example.png', dpi=150, bbox_inches='tight')
print("Saved visual example as: visual_discount_factor_example.png")

# Create a second figure showing value over time
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Calculate cumulative value over multiple rounds for different strategies
rounds = np.arange(20)
discount_factors_plot = [0.0, 0.2, 0.6, 0.9]
colors = ['red', 'orange', 'blue', 'green']

for df, color in zip(discount_factors_plot, colors):
    # Strategy: Mutual cooperation
    coop_values = [R * (df ** t) for t in rounds]
    cumulative_coop = np.cumsum(coop_values)
    
    # Strategy: Defect once then mutual defection
    defect_values = [T if t == 0 else P * (df ** t) for t in rounds]
    cumulative_defect = np.cumsum(defect_values)
    
    ax2.plot(rounds, cumulative_coop, '-', color=color, linewidth=2, 
             label=f'γ={df} Cooperate')
    ax2.plot(rounds, cumulative_defect, '--', color=color, linewidth=2, 
             label=f'γ={df} Defect', alpha=0.7)

ax2.set_xlabel('Round', fontsize=12)
ax2.set_ylabel('Cumulative Value', fontsize=12)
ax2.set_title('Long-term Value: Cooperation vs Defection Strategy', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', ncol=2)
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.axhline(y=R*20, color='gray', linestyle=':', alpha=0.5)
ax2.text(15, R*20 + 1, 'No discounting\n(20 rounds of cooperation)', 
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('cumulative_value_comparison.png', dpi=150, bbox_inches='tight')
print("Saved cumulative value comparison as: cumulative_value_comparison.png")

print("\nVisual examples created showing:")
print("1. Step-by-step Q-learning calculations with different discount factors")
print("2. How discount factor affects the cumulative value of different strategies")

plt.show()