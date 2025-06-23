#!/usr/bin/env python3
"""
Visualize the impact of different parameters on Q-learning cooperation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parameter Impact on Q-Learning Cooperation\n(2 QL vs 1 TFT-E)', 
             fontsize=16, fontweight='bold')

# Payoffs
T, R, P, S = 5, 3, 1, 0

# ============ Subplot 1: Discount Factor Impact ============
ax1 = axes[0, 0]
discount_factors = [0.5, 0.7, 0.9, 0.95, 0.99]
colors = plt.cm.viridis(np.linspace(0, 1, len(discount_factors)))

# Calculate value of sustained cooperation vs one-time defection
rounds = np.arange(0, 20)
for i, gamma in enumerate(discount_factors):
    # Value of always cooperating
    coop_values = [R * (gamma ** t) for t in rounds]
    cumulative_coop = np.cumsum(coop_values)
    
    # Value of defect once then punished
    defect_values = [T if t == 0 else P * (gamma ** t) for t in rounds]
    cumulative_defect = np.cumsum(defect_values)
    
    # When does cooperation become better?
    crossover = None
    for j, (c, d) in enumerate(zip(cumulative_coop, cumulative_defect)):
        if c > d:
            crossover = j
            break
    
    ax1.plot(rounds, cumulative_coop, '-', color=colors[i], 
             label=f'γ={gamma} (C)', linewidth=2)
    ax1.plot(rounds, cumulative_defect, '--', color=colors[i], 
             label=f'γ={gamma} (D)', alpha=0.7)
    
    if crossover:
        ax1.plot(crossover, cumulative_coop[crossover], 'o', 
                color=colors[i], markersize=8)

ax1.set_xlabel('Rounds into Future')
ax1.set_ylabel('Cumulative Value')
ax1.set_title('Impact of Discount Factor (γ)\nWhen does cooperation become valuable?')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# ============ Subplot 2: Learning Rate Impact ============
ax2 = axes[0, 1]

# Simulate learning with different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
noise_prob = 0.1  # TFT-E error rate

for i, lr in enumerate(learning_rates):
    Q_coop = []
    Q_defect = []
    q_c, q_d = 0, 0
    
    # Simulate 200 learning steps
    for step in range(200):
        # Simulate noisy rewards (TFT-E makes errors)
        if np.random.random() < noise_prob:
            # Error: unexpected outcome
            reward_c = S if np.random.random() < 0.5 else R
            reward_d = T if np.random.random() < 0.5 else P
        else:
            # Normal: cooperation maintained, defection punished
            reward_c = R
            reward_d = P
        
        # Q-learning updates
        q_c = q_c + lr * (reward_c + 0.95 * max(q_c, q_d) - q_c)
        q_d = q_d + lr * (reward_d + 0.95 * max(q_c, q_d) - q_d)
        
        Q_coop.append(q_c)
        Q_defect.append(q_d)
    
    color = colors[i]
    if lr == 0.05 or lr == 0.1:  # Highlight recommended values
        ax2.plot(Q_coop, color=color, linewidth=3, label=f'α={lr} (C)')
        ax2.plot(Q_defect, color=color, linewidth=3, linestyle='--', label=f'α={lr} (D)')
    else:
        ax2.plot(Q_coop, color=color, linewidth=1, alpha=0.5, label=f'α={lr} (C)')
        ax2.plot(Q_defect, color=color, linewidth=1, linestyle='--', alpha=0.5, label=f'α={lr} (D)')

ax2.set_xlabel('Learning Steps')
ax2.set_ylabel('Q-Value')
ax2.set_title('Impact of Learning Rate (α)\nLearning stability with 10% noise')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=R/(1-0.95), color='green', linestyle=':', alpha=0.5, 
           label='Optimal C value')
ax2.axhline(y=P/(1-0.95), color='red', linestyle=':', alpha=0.5,
           label='Optimal D value')

# ============ Subplot 3: Exploration Rate Impact ============
ax3 = axes[1, 0]

# Show how exploration affects stability
exploration_rates = [0.0, 0.05, 0.1, 0.2, 0.3]
rounds = 1000
window = 50

for i, eps in enumerate(exploration_rates):
    cooperation_rate = []
    current_strategy = 1  # Start cooperating
    
    for r in range(rounds):
        # With probability eps, explore (random)
        if np.random.random() < eps:
            action = np.random.randint(2)
        else:
            action = current_strategy
        
        # Simple learning: if good outcome, keep strategy
        if action == 0 and np.random.random() > 0.1:  # Cooperate, usually works
            current_strategy = 0
        elif action == 1:  # Defect usually bad in long run
            if np.random.random() < 0.3:  # Sometimes seems good
                current_strategy = 1
            else:
                current_strategy = 0
        
        cooperation_rate.append(1 - action)
    
    # Smooth the data
    smoothed = np.convolve(cooperation_rate, np.ones(window)/window, mode='valid')
    
    if eps == 0.1:  # Highlight recommended
        ax3.plot(smoothed, color=colors[i], linewidth=3, label=f'ε={eps}')
    else:
        ax3.plot(smoothed, color=colors[i], linewidth=1, alpha=0.7, label=f'ε={eps}')

ax3.set_xlabel('Round')
ax3.set_ylabel('Cooperation Rate (smoothed)')
ax3.set_title('Impact of Exploration Rate (ε)\nStability of cooperation')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.1, 1.1)

# ============ Subplot 4: Hysteretic vs Standard ============
ax4 = axes[1, 1]

# Compare standard vs hysteretic Q-learning
scenarios = ['Initial\nLearning', 'After\nCooperation', 'After\nDefection', 'After\nTFT Error']
x = np.arange(len(scenarios))
width = 0.35

# Standard Q-learning response
standard_changes = [0, 0.1, -0.1, -0.1]  # Q-value changes
hysteretic_changes = [0, 0.12, -0.002, -0.002]  # Much less affected by negative

ax4.bar(x - width/2, standard_changes, width, label='Standard QL', color='lightcoral')
ax4.bar(x + width/2, hysteretic_changes, width, label='Hysteretic QL', color='lightgreen')

ax4.set_xlabel('Scenario')
ax4.set_ylabel('Change in Q(cooperate)')
ax4.set_title('Standard vs Hysteretic Q-Learning\nResponse to different outcomes')
ax4.set_xticks(x)
ax4.set_xticklabels(scenarios)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linewidth=1)

# Add annotations
ax4.annotate('Optimistic\nbias helps\nmaintain\ncooperation', 
            xy=(2.4, -0.002), xytext=(3, 0.05),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('parameter_impact_visualization.png', dpi=150, bbox_inches='tight')

# Create summary recommendation figure
fig2, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Recommended Parameter Changes for Better Cooperation', 
        ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.90, '(2 Q-Learners vs 1 TFT-E Scenario)', 
        ha='center', fontsize=12, style='italic', transform=ax.transAxes)

# Create table
table_data = [
    ['Parameter', 'Current', 'Recommended', 'Reason'],
    ['Discount Factor (γ)', '0.9', '0.95-0.99', 'Value long-term cooperation more'],
    ['Learning Rate (α)', '0.1', '0.05-0.08', 'More stable with noisy TFT-E'],
    ['Exploration (ε)', '0.15', '0.08-0.12', 'Less disruption once learned'],
    ['Hysteretic β', '0.005', '0.001-0.002', 'More optimistic about cooperation'],
]

# Draw table
y_start = 0.75
row_height = 0.08
col_widths = [0.25, 0.15, 0.15, 0.35]
col_starts = [0.05, 0.35, 0.55, 0.75]

for i, row in enumerate(table_data):
    y = y_start - i * row_height
    
    for j, (cell, width, start) in enumerate(zip(row, col_widths, col_starts)):
        # Header row
        if i == 0:
            ax.add_patch(plt.Rectangle((start, y-0.02), width, row_height-0.01, 
                                     facecolor='lightgray', edgecolor='black'))
            ax.text(start + width/2, y + row_height/2 - 0.02, cell, 
                   ha='center', va='center', fontweight='bold', fontsize=11)
        else:
            ax.add_patch(plt.Rectangle((start, y-0.02), width, row_height-0.01, 
                                     facecolor='white', edgecolor='black'))
            # Color code improvements
            if j == 2:  # Recommended column
                ax.text(start + width/2, y + row_height/2 - 0.02, cell, 
                       ha='center', va='center', fontsize=10, color='green', fontweight='bold')
            else:
                ax.text(start + width/2, y + row_height/2 - 0.02, cell, 
                       ha='center', va='center', fontsize=10)

# Add key insights
insights_y = 0.35
ax.text(0.05, insights_y, 'Key Insights:', fontsize=12, fontweight='bold', transform=ax.transAxes)

insights = [
    '• Higher γ makes mutual cooperation more attractive than short-term exploitation',
    '• Lower α prevents overreacting to TFT-E\'s 10% error rate',
    '• Moderate ε maintains learned cooperation without too much disruption',
    '• Hysteretic QL with low β naturally maintains optimistic cooperation',
]

for i, insight in enumerate(insights):
    ax.text(0.05, insights_y - 0.05 - i*0.04, insight, fontsize=10, transform=ax.transAxes)

# Add example calculation
ax.text(0.05, 0.08, 'Example: With γ=0.95 instead of 0.9:', fontsize=11, 
        fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.04, 'Value of infinite cooperation: R/(1-γ) = 3/(1-0.95) = 60 (vs 30 with γ=0.9)', 
        fontsize=10, transform=ax.transAxes, style='italic')

plt.savefig('parameter_recommendations.png', dpi=150, bbox_inches='tight')
print("Saved parameter impact visualizations!")

plt.show()