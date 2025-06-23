#!/usr/bin/env python3
"""
Fixed Q-Learning Decision Flow Chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create figure with better layout
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(7, 11, 'Q-Learning Agent Decision Flow', ha='center', fontsize=20, fontweight='bold')

# Define colors for different types of elements
colors = {
    'observation': '#E8F4FD',  # Light blue
    'calculation': '#E8F5E9',  # Light green
    'decision': '#FFF9C4',     # Light yellow
    'action': '#FFEBEE',       # Light red
    'update': '#F3E5F5'        # Light purple
}

# Helper function to add a box with text
def add_box(x, y, width, height, text, color, ax, fontsize=11):
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

# Helper function to add an arrow
def add_arrow(x1, y1, x2, y2, ax, style='->', color='black', connectionstyle='arc3,rad=0'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           connectionstyle=connectionstyle,
                           linewidth=2, color=color)
    ax.add_patch(arrow)

# Main flow elements
# Step 1: Observe State
add_box(7, 9.5, 3, 0.8, '1. Observe State', colors['observation'], ax, 12)
ax.text(7, 8.8, 'e.g., "opponent cooperated"', ha='center', fontsize=9, style='italic')

# Arrow down
add_arrow(7, 9, 7, 8.3, ax)

# Step 2: Look up Q-values
add_box(7, 7.8, 3.5, 0.8, '2. Look up Q-values', colors['calculation'], ax, 12)
ax.text(7, 7.1, 'Q(state, cooperate) & Q(state, defect)', ha='center', fontsize=9, style='italic')

# Arrow down
add_arrow(7, 7.3, 7, 6.6, ax)

# Step 3: Action Selection (ε-greedy)
add_box(7, 6.1, 4, 0.8, '3. Action Selection', colors['decision'], ax, 12)
ax.text(7, 5.4, 'Generate random number r ∈ [0,1]', ha='center', fontsize=9, style='italic')

# Decision branches
add_arrow(6, 5.6, 4, 4.9, ax)  # Left branch
add_arrow(8, 5.6, 10, 4.9, ax)  # Right branch

# Exploration branch (left)
add_box(3, 4.5, 3, 0.7, f'r < ε (Explore)', colors['decision'], ax, 10)
ax.text(3, 3.95, 'ε = 0.2 (20%)', ha='center', fontsize=9)
add_arrow(3, 4.1, 3, 3.5, ax)
add_box(3, 3.1, 2.5, 0.6, 'Random Action', colors['action'], ax, 10)
ax.text(3, 2.65, '50% C, 50% D', ha='center', fontsize=8, style='italic')

# Exploitation branch (right)
add_box(11, 4.5, 3, 0.7, f'r ≥ ε (Exploit)', colors['decision'], ax, 10)
ax.text(11, 3.95, '1-ε = 0.8 (80%)', ha='center', fontsize=9)
add_arrow(11, 4.1, 11, 3.5, ax)
add_box(11, 3.1, 2.5, 0.6, 'Best Q Action', colors['action'], ax, 10)
ax.text(11, 2.65, 'argmax Q(s,a)', ha='center', fontsize=8, style='italic')

# Merge paths
add_arrow(3, 2.8, 6.5, 2.3, ax)
add_arrow(11, 2.8, 7.5, 2.3, ax)

# Step 4: Execute Action
add_box(7, 1.9, 3, 0.7, '4. Execute Action', colors['action'], ax, 12)
ax.text(7, 1.45, 'Play C or D', ha='center', fontsize=9, style='italic')

# Arrow down
add_arrow(7, 1.5, 7, 1.0, ax)

# Step 5: Observe Outcome
add_box(7, 0.6, 3, 0.7, '5. Observe Outcome', colors['observation'], ax, 12)
ax.text(7, 0.15, 'Get reward & next state', ha='center', fontsize=9, style='italic')

# Q-Update box (positioned to the right)
add_box(11, 0.6, 4, 1.2, '6. Q-Value Update', colors['update'], ax, 11)

# Add Q-update formula
formula_text = 'Q(s,a) ← Q(s,a) + α[r + γ·max Q(s\',a\') - Q(s,a)]'
ax.text(11, 0.4, formula_text, ha='center', fontsize=9, style='italic')

# Connect to Q-update
add_arrow(8.5, 0.6, 9, 0.6, ax)

# Loop back arrow
add_arrow(11, 1.2, 11, 9.5, ax, connectionstyle='arc3,rad=.3', color='blue')
ax.text(11.5, 5.5, 'Next\nRound', ha='center', fontsize=10, color='blue', rotation=90)

# Add legend for colors
legend_x = 0.5
legend_y = 10
legend_items = [
    ('Observation', colors['observation']),
    ('Calculation', colors['calculation']),
    ('Decision', colors['decision']),
    ('Action', colors['action']),
    ('Update', colors['update'])
]

ax.text(legend_x + 1, legend_y + 0.5, 'Legend:', fontsize=10, fontweight='bold')
for i, (label, color) in enumerate(legend_items):
    y = legend_y - i * 0.3
    rect = mpatches.Rectangle((legend_x, y - 0.1), 0.3, 0.2, 
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(legend_x + 0.4, y, label, fontsize=9, va='center')

# Add parameter box
param_box = FancyBboxPatch((0.2, 0.2), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='white',
                          edgecolor='black', linewidth=1)
ax.add_patch(param_box)
ax.text(1.7, 1.5, 'Parameters:', fontsize=10, fontweight='bold', ha='center')
ax.text(1.7, 1.2, 'α = 0.1 (learning rate)', fontsize=9, ha='center')
ax.text(1.7, 0.9, 'γ = 0.6 (discount factor)', fontsize=9, ha='center')
ax.text(1.7, 0.6, 'ε = 0.2 (exploration rate)', fontsize=9, ha='center')

# Add detailed example in bottom right
example_box = FancyBboxPatch((12.2, 0.2), 1.6, 3.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#F5F5F5',
                            edgecolor='black', linewidth=1)
ax.add_patch(example_box)
ax.text(13, 3.5, 'Example:', fontsize=9, fontweight='bold', ha='center')
ax.text(13, 3.2, 'State: "opp_C"', fontsize=8, ha='center')
ax.text(13, 2.9, 'Q(s,C) = 2.5', fontsize=8, ha='center')
ax.text(13, 2.6, 'Q(s,D) = 3.8', fontsize=8, ha='center')
ax.text(13, 2.3, 'Roll: 0.73', fontsize=8, ha='center')
ax.text(13, 2.0, '0.73 > 0.2', fontsize=8, ha='center')
ax.text(13, 1.7, '→ EXPLOIT', fontsize=8, ha='center', fontweight='bold')
ax.text(13, 1.4, 'Choose D', fontsize=8, ha='center', color='red')
ax.text(13, 1.1, '(higher Q)', fontsize=8, ha='center', style='italic')

# Don't use tight_layout - manually adjust instead
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
plt.savefig('q_learning_decision_flow_fixed.png', dpi=150, bbox_inches='tight')
print("Saved fixed decision flow as: q_learning_decision_flow_fixed.png")

# Create a second, simpler flow chart focusing on the key decision
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9, 'Simplified Q-Learning Decision Process', ha='center', fontsize=18, fontweight='bold')

# Central decision diamond
diamond_x, diamond_y = 5, 5
diamond_size = 1.5
diamond = mpatches.FancyBboxPatch((diamond_x - diamond_size/2, diamond_y - diamond_size/2), 
                                  diamond_size, diamond_size,
                                  boxstyle="round,pad=0.1",
                                  transform=ax2.transData,
                                  facecolor='yellow',
                                  edgecolor='black', linewidth=2)
# Rotate 45 degrees
from matplotlib.transforms import Affine2D
t = Affine2D().rotate_deg(45) + ax2.transData
diamond.set_transform(t)
ax2.add_patch(diamond)
ax2.text(diamond_x, diamond_y, 'Random\n< ε?', ha='center', va='center', fontsize=11, fontweight='bold')

# Current state (top)
add_box(5, 7.5, 2.5, 0.8, 'Current State\n& Q-values', '#E8F4FD', ax2)
add_arrow(5, 7.1, 5, 5.8, ax2)

# Explore path (left)
add_box(2, 5, 2, 0.8, 'EXPLORE\n(20%)', 'orange', ax2)
add_arrow(4.3, 5, 3, 5, ax2)
add_box(2, 3, 2.5, 0.8, 'Random\nAction', '#FFCDD2', ax2)
add_arrow(2, 4.6, 2, 3.4, ax2)

# Exploit path (right)
add_box(8, 5, 2, 0.8, 'EXPLOIT\n(80%)', 'lightgreen', ax2)
add_arrow(5.7, 5, 7, 5, ax2)
add_box(8, 3, 2.5, 0.8, 'Best Q\nAction', '#C8E6C9', ax2)
add_arrow(8, 4.6, 8, 3.4, ax2)

# Outcome (bottom)
add_box(5, 1, 3, 0.8, 'Get Reward\n& Update Q', '#E1BEE7', ax2)
add_arrow(2, 2.6, 4, 1.4, ax2)
add_arrow(8, 2.6, 6, 1.4, ax2)

# Add some example numbers
ax2.text(1, 7, 'Example:\nQ(s,C)=2.1\nQ(s,D)=3.5', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.text(9, 7, 'Roll: 0.73\n0.73 > 0.2\n→ Exploit!', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('q_learning_decision_simple.png', dpi=150, bbox_inches='tight')
print("Saved simplified decision flow as: q_learning_decision_simple.png")

print("\nFixed decision flow charts created successfully!")
print("The warning should be resolved now.")

plt.show()