#!/usr/bin/env python3
"""
Test script to verify the setup for multi-agent scaling experiments.
This shows what scenarios will be run without actually executing them.
"""

# Group sizes to test
GROUP_SIZES = [3, 5, 7, 10, 15, 20, 25]

# Opponent types from v6
OPPONENT_TYPES = ["AllC", "AllD", "Random", "TFT"]

print("="*80)
print("MULTI-AGENT SCALING EXPERIMENT SETUP")
print("="*80)
print(f"\nGroup sizes to test: {GROUP_SIZES}")
print(f"Opponent types: {OPPONENT_TYPES}")

print("\n" + "="*80)
print("SCENARIOS TO BE TESTED:")
print("="*80)

scenario_count = 0

# Test 1: 2 QL agents vs varying numbers of opponents
print("\n1. TWO QL AGENTS vs VARYING OPPONENTS:")
print("-"*50)
for group_size in GROUP_SIZES:
    if group_size < 3:
        continue
    n_opponents = group_size - 2
    print(f"\nGroup size {group_size}:")
    for opp_type in OPPONENT_TYPES:
        scenario_count += 1
        print(f"  - 2 QL agents vs {n_opponents} {opp_type} agents")

# Test 2: Mixed QL types
print("\n\n2. MIXED QL TYPES:")
print("-"*50)
for group_size in GROUP_SIZES:
    if group_size < 4:
        continue
    n_legacy3 = group_size // 2
    n_nodecay = group_size - n_legacy3
    scenario_count += 1
    print(f"Group size {group_size}: {n_legacy3} Legacy3Round + {n_nodecay} QLNoDecay")

# Test 3: All same QL type
print("\n\n3. ALL SAME QL TYPE:")
print("-"*50)
for group_size in GROUP_SIZES:
    scenario_count += 2  # One for each QL type
    print(f"Group size {group_size}:")
    print(f"  - All {group_size} Legacy3Round agents")
    print(f"  - All {group_size} QLNoDecay agents")

print(f"\n\nTOTAL SCENARIOS: {scenario_count}")

# Show QL agent configurations
print("\n" + "="*80)
print("QL AGENT CONFIGURATIONS:")
print("="*80)
print("\n1. Legacy3Round QL (from LEGACY_3ROUND_PARAMS):")
print("   - lr: 0.15")
print("   - df: 0.99")
print("   - eps: 0.25 (with decay)")
print("   - epsilon_decay: 0.998")
print("   - epsilon_min: 0.01")
print("   - optimistic_init: -0.3")
print("   - history_length: 3 rounds")

print("\n2. QLNoDecay (custom configuration):")
print("   - lr: 0.1")
print("   - df: 0.95")
print("   - eps: 0.1 (NO decay)")
print("   - epsilon_decay: 1.0 (disabled)")
print("   - epsilon_min: 0.1")
print("   - optimistic_init: 1.0 (cooperative)")
print("   - history_length: 3 rounds")

print("\n" + "="*80)
print("KEY DIFFERENCES:")
print("="*80)
print("- Legacy3Round: Starts with high exploration (25%) that decays over time")
print("- QLNoDecay: Fixed 10% exploration throughout the game")
print("- Legacy3Round: Pessimistic initialization (-0.3)")
print("- QLNoDecay: Optimistic/cooperative initialization (1.0)")
print("- Both use 3-round history for sophisticated pattern detection")