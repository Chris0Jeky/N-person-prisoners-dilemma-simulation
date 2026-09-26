#!/usr/bin/env python3
"""
Detailed example showing how discount factor affects Q-learning decisions
in the Prisoner's Dilemma
"""

import numpy as np
import matplotlib.pyplot as plt

# Prisoner's Dilemma payoffs
T, R, P, S = 5, 3, 1, 0  # Temptation, Reward, Punishment, Sucker

print("=" * 80)
print("Q-LEARNING DISCOUNT FACTOR: PRACTICAL EXAMPLE")
print("=" * 80)

print("\nPayoff Matrix:")
print("              Opponent Cooperates    Opponent Defects")
print("I Cooperate:       R = 3                S = 0")
print("I Defect:          T = 5                P = 1")
print()

# Q-learning formula: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
# Where: α = learning rate, γ = discount factor, r = immediate reward

print("=" * 80)
print("SCENARIO: Q-Learning Agent vs Tit-for-Tat (TFT)")
print("=" * 80)

# Let's trace through several rounds of learning
# State representation: opponent's last move (C or D)
# Actions: Cooperate (C) or Defect (D)

def simulate_q_updates(discount_factor, num_updates=10):
    """Simulate Q-value updates with given discount factor"""
    
    print(f"\n--- Discount Factor γ = {discount_factor} ---")
    
    # Initialize Q-table
    # States: "start" (first round), "after_C" (opponent cooperated), "after_D" (opponent defected)
    Q = {
        "start": {"C": 0.0, "D": 0.0},
        "after_C": {"C": 0.0, "D": 0.0},
        "after_D": {"C": 0.0, "D": 0.0}
    }
    
    learning_rate = 0.1
    
    print("\nInitial Q-table (all zeros)")
    
    # Simulate a sequence of interactions with TFT
    # TFT starts with cooperation, then copies our last move
    
    updates = [
        # (current_state, action, reward, next_state, explanation)
        ("start", "D", T, "after_D", "Round 1: We defect against TFT's initial cooperation"),
        ("after_D", "C", S, "after_C", "Round 2: We cooperate, but TFT defects (copying our last move)"),
        ("after_C", "C", R, "after_C", "Round 3: Mutual cooperation"),
        ("after_C", "D", T, "after_D", "Round 4: We defect against cooperation"),
        ("after_D", "C", S, "after_C", "Round 5: We cooperate, TFT defects"),
        ("after_C", "C", R, "after_C", "Round 6: Mutual cooperation"),
        ("after_C", "C", R, "after_C", "Round 7: Mutual cooperation continues"),
    ]
    
    print("\nQ-Learning Updates:")
    print("-" * 70)
    
    for i, (state, action, reward, next_state, explanation) in enumerate(updates[:num_updates]):
        old_q = Q[state][action]
        max_next_q = max(Q[next_state].values())
        
        # Q-learning update
        td_target = reward + discount_factor * max_next_q
        td_error = td_target - old_q
        new_q = old_q + learning_rate * td_error
        
        Q[state][action] = new_q
        
        print(f"\n{explanation}")
        print(f"State: {state}, Action: {action}, Reward: {reward}")
        print(f"Old Q-value: {old_q:.3f}")
        print(f"TD Target = {reward} + {discount_factor} × {max_next_q:.3f} = {td_target:.3f}")
        print(f"TD Error = {td_target:.3f} - {old_q:.3f} = {td_error:.3f}")
        print(f"New Q-value = {old_q:.3f} + {learning_rate} × {td_error:.3f} = {new_q:.3f}")
    
    print("\n" + "-" * 70)
    print("Final Q-table:")
    for state in Q:
        print(f"{state:8s}: C={Q[state]['C']:.3f}, D={Q[state]['D']:.3f}")
        best = "C" if Q[state]['C'] > Q[state]['D'] else "D" if Q[state]['D'] > Q[state]['C'] else "Equal"
        print(f"          Best action: {best}")
    
    return Q

# Compare different discount factors
print("\n" + "=" * 80)
print("COMPARING DIFFERENT DISCOUNT FACTORS")
print("=" * 80)

discount_factors = [0.0, 0.2, 0.6, 0.9, 0.99]
final_q_tables = {}

for df in discount_factors:
    Q = simulate_q_updates(df, num_updates=7)
    final_q_tables[df] = Q

# Analyze the differences
print("\n" + "=" * 80)
print("ANALYSIS: How Discount Factor Affects Strategy")
print("=" * 80)

print("\n1. IMMEDIATE vs FUTURE REWARDS:")
print("-" * 50)
print("When deciding whether to cooperate with a cooperator:")
print("- Immediate reward for defection: T = 5")
print("- Immediate reward for cooperation: R = 3")
print("- But cooperation maintains the cooperative relationship!")

print("\n2. LONG-TERM VALUE CALCULATION:")
print("-" * 50)
print("If we maintain cooperation, expected future value per round ≈ R = 3")
print("If we defect, opponent will defect next, future value ≈ P = 1")

for df in [0.0, 0.6, 0.9]:
    # Calculate value of different strategies over multiple rounds
    rounds = 10
    
    # Strategy 1: Always cooperate with cooperator
    coop_value = R  # Immediate
    future_coop = sum([R * (df ** i) for i in range(1, rounds)])
    total_coop = coop_value + future_coop
    
    # Strategy 2: Defect once, then get punished
    defect_value = T  # Immediate gain
    future_defect = sum([P * (df ** i) for i in range(1, rounds)])
    total_defect = defect_value + future_defect
    
    print(f"\nγ = {df}:")
    print(f"  Cooperate strategy value: {R} + {future_coop:.2f} = {total_coop:.2f}")
    print(f"  Defect strategy value: {T} + {future_defect:.2f} = {total_defect:.2f}")
    print(f"  Better strategy: {'Cooperate' if total_coop > total_defect else 'Defect'}")

print("\n3. DECISION BOUNDARIES:")
print("-" * 50)
print("The agent prefers cooperation when:")
print("R / (1 - γ) > T + γP / (1 - γ)")
print("Simplifying: R > T(1 - γ) + γP")
print(f"With R={R}, T={T}, P={P}:")

for df in [0.0, 0.2, 0.6, 0.9]:
    threshold = T * (1 - df) + df * P
    cooperates = R > threshold
    print(f"  γ = {df}: {R} > {threshold:.2f} ? {cooperates} → {'Cooperate' if cooperates else 'Defect'}")

# Visualize Q-value evolution
print("\n" + "=" * 80)
print("VISUALIZING Q-VALUE EVOLUTION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Q-Value Evolution with Different Discount Factors', fontsize=16)

for idx, df in enumerate([0.0, 0.2, 0.6, 0.9]):
    ax = axes[idx // 2, idx % 2]
    
    # Track Q-values over many updates
    Q = {
        "start": {"C": 0.0, "D": 0.0},
        "after_C": {"C": 0.0, "D": 0.0},
        "after_D": {"C": 0.0, "D": 0.0}
    }
    
    q_history = {"after_C_C": [], "after_C_D": []}
    
    # Simulate longer sequence
    for round in range(100):
        # Simulate typical TFT interaction pattern
        if round % 4 == 0:  # Occasionally test defection
            state, action, reward, next_state = "after_C", "D", T, "after_D"
        elif round % 4 == 1:  # Get punished
            state, action, reward, next_state = "after_D", "C", S, "after_C"
        else:  # Mutual cooperation
            state, action, reward, next_state = "after_C", "C", R, "after_C"
        
        # Update Q-value
        old_q = Q[state][action]
        max_next_q = max(Q[next_state].values())
        Q[state][action] = old_q + 0.1 * (reward + df * max_next_q - old_q)
        
        # Track key Q-values
        q_history["after_C_C"].append(Q["after_C"]["C"])
        q_history["after_C_D"].append(Q["after_C"]["D"])
    
    # Plot
    ax.plot(q_history["after_C_C"], label="Q(after_C, Cooperate)", linewidth=2)
    ax.plot(q_history["after_C_D"], label="Q(after_C, Defect)", linewidth=2)
    ax.set_title(f'γ = {df}')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Q-Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark which action is preferred
    final_pref = "Cooperate" if Q["after_C"]["C"] > Q["after_C"]["D"] else "Defect"
    ax.text(0.02, 0.98, f"Final choice: {final_pref}", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('q_value_evolution.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization as: q_value_evolution.png")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)
print("\n1. With γ = 0.0 (no future consideration):")
print("   - Agent only cares about immediate reward")
print("   - Always defects against cooperators (T=5 > R=3)")
print("   - Cannot learn reciprocal strategies")

print("\n2. With γ = 0.6 (your current setting):")
print("   - Moderate future consideration")
print("   - May learn to cooperate if relationship is stable")
print("   - Still somewhat tempted by immediate defection")

print("\n3. With γ = 0.9 (standard setting):")
print("   - Strong future consideration")
print("   - Learns that cooperation maintains valuable relationships")
print("   - Resists temptation to defect for short-term gain")

print("\n4. Mathematical insight:")
print("   - The effective value of infinite cooperation: R/(1-γ)")
print(f"   - With R=3: γ=0.6 → value=7.5, γ=0.9 → value=30")
print("   - Higher γ makes sustained cooperation much more valuable!")

plt.show()