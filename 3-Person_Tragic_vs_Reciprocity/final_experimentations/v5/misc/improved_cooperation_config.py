#!/usr/bin/env python3
"""
Improved configurations for better Q-learning cooperation in multi-agent scenarios
Especially for 2 Q-learners vs 1 TFT-E situations
"""

# ========== PROBLEM ANALYSIS ==========
# When 2 Q-learners face 1 TFT-E:
# 1. They may fall into mutual defection cycles
# 2. They don't coordinate well with each other
# 3. TFT-E's 10% error rate disrupts learning
# 4. Standard Q-learning is too greedy/exploitative

# ========== SOLUTION STRATEGIES ==========

# Strategy 1: OPTIMISTIC INITIALIZATION
# Start with positive Q-values to encourage initial cooperation
OPTIMISTIC_VANILLA_PARAMS = {
    'lr': 0.1,
    'df': 0.95,  # Higher discount factor for longer-term thinking
    'eps': 0.15,
    'initial_q_value': 3.0,  # Start optimistic (mutual cooperation value)
}

# Strategy 2: SLOWER LEARNING WITH HIGH PATIENCE
# Learn more carefully to avoid overreacting to TFT errors
PATIENT_VANILLA_PARAMS = {
    'lr': 0.05,  # Slower learning rate
    'df': 0.99,  # Very patient (values future highly)
    'eps': 0.1,  # Less exploration
}

# Strategy 3: LENIENT Q-LEARNING
# Inspired by Leniency in multi-agent learning - forgive mistakes
LENIENT_PARAMS = {
    'lr': 0.1,
    'df': 0.95,
    'eps': 0.12,
    'leniency_rate': 0.8,  # 80% chance to ignore negative experiences initially
    'leniency_decay': 0.995,  # Slowly become less lenient
}

# Strategy 4: ENHANCED HYSTERETIC FOR MULTI-AGENT
# Hysteretic Q-learning is already good for cooperation, tune it better
IMPROVED_HYSTERETIC_PARAMS = {
    'lr': 0.15,      # Higher learning rate for positive updates
    'beta': 0.001,   # Much lower for negative (more optimistic)
    'df': 0.95,      # High patience
    'eps': 0.05,     # Low exploration (trust the optimism)
}

# Strategy 5: ADAPTIVE WITH COOPERATION BIAS
# Adaptive agent that's biased toward cooperation
COOPERATION_BIASED_ADAPTIVE = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.02,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.2,
    'adaptation_factor': 1.05,  # Slower adaptation
    'reward_window_size': 100,  # Longer memory
    'df': 0.95,
    'cooperation_bonus': 0.5,  # Add bonus to cooperation rewards
}

# Strategy 6: WIN-STAY LOSE-SHIFT INSPIRED
# More reactive to good outcomes, less to bad
WSLS_INSPIRED_PARAMS = {
    'lr': 0.2,       # Fast learning
    'df': 0.9,       
    'eps': 0.05,     # Low exploration
    'win_threshold': 3,  # If reward >= 3, it's a "win"
    'shift_probability': 0.3,  # Only 30% chance to change after loss
}

# ========== RECOMMENDED CONFIGURATIONS ==========

# Best for 2QL vs 1TFT-E scenarios
MULTI_QL_VS_TFT_CONFIG = {
    'VANILLA_PARAMS': {
        'lr': 0.08,      # Moderate learning rate
        'df': 0.95,      # High patience for sustained cooperation
        'eps': 0.1,      # Moderate exploration
    },
    
    'HYSTERETIC_PARAMS': {
        'lr': 0.12,      # Good news learning
        'beta': 0.002,   # Very slow bad news learning
        'df': 0.95,      
        'eps': 0.05,     # Low exploration
    },
    
    'ADAPTIVE_PARAMS': {
        'initial_lr': 0.1,
        'initial_eps': 0.15,
        'min_lr': 0.03,
        'max_lr': 0.15,
        'min_eps': 0.02,
        'max_eps': 0.15,
        'adaptation_factor': 1.08,
        'reward_window_size': 75,
        'df': 0.95,
    }
}

# ========== ANALYSIS & RECOMMENDATIONS ==========

print("=" * 80)
print("IMPROVING Q-LEARNING COOPERATION IN MULTI-AGENT SCENARIOS")
print("=" * 80)

print("\nPROBLEM: 2 Q-learners vs 1 TFT-E")
print("-" * 40)
print("1. Q-learners may exploit each other")
print("2. TFT-E's 10% errors disrupt learning")
print("3. No explicit coordination mechanism")
print("4. Standard parameters too exploitative")

print("\nSOLUTION APPROACHES:")
print("-" * 40)

print("\n1. INCREASE DISCOUNT FACTOR (γ)")
print("   Current: 0.9 → Recommended: 0.95-0.99")
print("   Why: Makes future cooperation more valuable")
print("   Effect: Reduces temptation to defect for short-term gain")

print("\n2. ADJUST LEARNING RATE (α)")
print("   Current: 0.1 → Recommended: 0.05-0.08")
print("   Why: Slower learning is more robust to noise")
print("   Effect: Less overreaction to TFT-E's errors")

print("\n3. OPTIMIZE EXPLORATION (ε)")
print("   Current: 0.15 → Recommended: 0.08-0.12")
print("   Why: Too much exploration disrupts cooperation")
print("   Effect: More stable strategies once learned")

print("\n4. USE HYSTERETIC Q-LEARNING")
print("   Current: β=0.005 → Recommended: β=0.001-0.002")
print("   Why: Even more optimistic about cooperation")
print("   Effect: Maintains cooperation despite occasional defections")

print("\n5. IMPLEMENT LENIENCY")
print("   Forgive early mistakes during learning")
print("   Gradually become less lenient")
print("   Helps establish mutual cooperation")

print("\nRECOMMENDED CHANGES TO CONFIG.PY:")
print("-" * 40)
print("""
# For better multi-agent cooperation
VANILLA_PARAMS = {
    'lr': 0.08,   # Slower learning (was 0.1)
    'df': 0.95,   # Higher patience (was 0.9)
    'eps': 0.1,   # Less exploration (was 0.15)
}

HYSTERETIC_PARAMS = {
    'lr': 0.12,     # Slightly higher positive learning
    'beta': 0.002,  # Much lower negative learning (was 0.005)
    'df': 0.95,     # Higher patience
    'eps': 0.05,    # Less exploration (was 0.08)
}
""")

print("\nEXPECTED IMPROVEMENTS:")
print("-" * 40)
print("1. Higher sustained cooperation between Q-learners")
print("2. Better resilience to TFT-E's errors")
print("3. Faster convergence to mutual cooperation")
print("4. More stable long-term performance")

print("\nADDITIONAL TECHNIQUES TO CONSIDER:")
print("-" * 40)
print("1. Optimistic Q-value initialization")
print("2. Decay exploration rate over time")
print("3. Add small cooperation bonus to rewards")
print("4. Use longer state memory for better context")

# Create a test configuration file
test_config = """# Improved configuration for multi-agent cooperation
# Optimized for 2 Q-learners vs 1 TFT-E scenarios

SIMULATION_CONFIG = {
    'num_rounds': 10000,
    'num_runs': 10,
}

# More cooperative vanilla Q-learning
VANILLA_PARAMS = {
    'lr': 0.08,    # Slower learning for stability
    'df': 0.95,    # High patience for long-term cooperation
    'eps': 0.1,    # Moderate exploration
}

# Highly optimistic hysteretic Q-learning
HYSTERETIC_PARAMS = {
    'lr': 0.12,     # Good news learning rate
    'beta': 0.002,  # Very low bad news learning rate
    'df': 0.95,     # High patience
    'eps': 0.05,    # Low exploration
}

# Adaptive with cooperation bias
ADAPTIVE_PARAMS = {
    'initial_lr': 0.1,
    'initial_eps': 0.15,
    'min_lr': 0.03,
    'max_lr': 0.15,
    'min_eps': 0.02,
    'max_eps': 0.15,
    'adaptation_factor': 1.08,
    'reward_window_size': 75,
    'df': 0.95,
}
"""

with open('improved_config.py', 'w') as f:
    f.write(test_config)

print("\nCreated 'improved_config.py' for testing!")
print("\nTo use: Replace config.py with improved_config.py and re-run experiments")