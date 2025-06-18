"""
Quick analysis of parameter sweep issues
"""

import csv
import os

def analyze_sweep_results():
    """Analyze the parameter sweep results to understand issues."""
    
    results_dir = "parameter_sweep_results"
    
    # Read all sweep results
    with open(os.path.join(results_dir, "all_sweep_results.csv"), 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 50)
    
    # Find top performing parameters by cooperation
    sorted_by_coop = sorted(results, key=lambda x: float(x['avg_cooperation']), reverse=True)
    print("\nTop 10 by Average Cooperation:")
    print("-" * 30)
    for i, r in enumerate(sorted_by_coop[:10]):
        print(f"{i+1}. {r['sweep_type']} sweep:")
        if r['param1_name']:
            print(f"   {r['param1_name']}: {r['param1_value']}")
        if r['param2_name']:
            print(f"   {r['param2_name']}: {r['param2_value']}")
        if r['param3_name']:
            print(f"   {r['param3_name']}: {r['param3_value']}")
        print(f"   Cooperation: {float(r['avg_cooperation']):.3f}")
        print(f"   Score: {float(r['avg_score']):.1f}")
        print(f"   Stability: {float(r['cooperation_stability']):.3f}")
        print()
    
    # Analyze parameter ranges
    print("\nParameter Ranges Tested:")
    print("-" * 30)
    params = ['learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay', 'epsilon_min', 'memory_length']
    
    for param in params:
        values = []
        for r in results:
            if r['param1_name'] == param and r['param1_value']:
                values.append(float(r['param1_value']))
            if r['param2_name'] == param and r['param2_value']:
                values.append(float(r['param2_value']))
            if r['param3_name'] == param and r['param3_value']:
                values.append(float(r['param3_value']))
        
        if values:
            print(f"\n{param}:")
            print(f"  Min: {min(values):.4f}")
            print(f"  Max: {max(values):.4f}")
            print(f"  Unique values: {len(set(values))}")
    
    # Compare with "optimal" parameters
    print("\n\n'OPTIMAL' PARAMETERS vs SWEEP RESULTS")
    print("=" * 50)
    optimal = {
        'learning_rate': 0.282,
        'discount_factor': 0.881,
        'epsilon': 0.5,
        'epsilon_decay': 0.822,
        'epsilon_min': 0.0046
    }
    
    print("\nCurrently set as optimal:")
    for p, v in optimal.items():
        print(f"  {p}: {v}")
    
    # Find closest tested values
    print("\n\nClosest tested values and their performance:")
    print("-" * 40)
    
    for param, optimal_val in optimal.items():
        # Find single parameter sweeps for this param
        param_results = []
        for r in results:
            if r['sweep_type'] == 'single' and r['param1_name'] == param:
                val = float(r['param1_value'])
                dist = abs(val - optimal_val)
                param_results.append((dist, val, r))
        
        if param_results:
            param_results.sort(key=lambda x: x[0])
            closest = param_results[0]
            print(f"\n{param}:")
            print(f"  Optimal: {optimal_val:.4f}")
            print(f"  Closest tested: {closest[1]:.4f}")
            print(f"  Performance: Coop={float(closest[2]['avg_cooperation']):.3f}, Score={float(closest[2]['avg_score']):.1f}")
            
            # Also show best performing value for this parameter
            best = max([r for _, _, r in param_results], key=lambda x: float(x['avg_cooperation']))
            best_val = float(best['param1_value'])
            if best_val != closest[1]:
                print(f"  Best performing: {best_val:.4f}")
                print(f"  Best performance: Coop={float(best['avg_cooperation']):.3f}, Score={float(best['avg_score']):.1f}")
    
    # Analyze why extremes perform well
    print("\n\nEXTREME VALUES ANALYSIS")
    print("=" * 50)
    
    extreme_results = []
    for r in results:
        if r['param1_value']:
            val = float(r['param1_value'])
            param = r['param1_name']
            if param != 'memory_length':
                if val <= 0.01 or val >= 0.99:
                    extreme_results.append(r)
    
    if extreme_results:
        extreme_results.sort(key=lambda x: float(x['avg_cooperation']), reverse=True)
        print(f"\nFound {len(extreme_results)} results with extreme parameter values")
        print("\nTop extreme value performances:")
        for r in extreme_results[:5]:
            print(f"\n{r['param1_name']} = {r['param1_value']}")
            print(f"  Cooperation: {float(r['avg_cooperation']):.3f}")
            print(f"  Score: {float(r['avg_score']):.1f}")
            print(f"  Stability: {float(r['cooperation_stability']):.3f}")
    
    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("=" * 50)
    print("\n1. The parameter sweep shows extreme values (0.0, 1.0) often perform best")
    print("   - This suggests the agent may be exploiting specific opponent weaknesses")
    print("   - Extreme values are likely not robust across different opponents")
    
    print("\n2. The 'optimal' parameters are moderate values that may be:")
    print("   - More stable across different scenarios")
    print("   - Less prone to overfitting")
    print("   - But not optimal for any specific opponent")
    
    print("\n3. To improve parameter selection:")
    print("   - Test against multiple opponent strategies simultaneously")
    print("   - Consider parameter schedules (e.g., decay over time)")
    print("   - Use cross-validation across different game scenarios")
    print("   - Test stability and convergence, not just final performance")
    
    print("\n4. The limited ranges tested (discount_factor up to 0.99, epsilon_decay up to 0.999)")
    print("   may have missed important regions of the parameter space")
    
    print("\n5. Consider that:")
    print("   - epsilon_min = 0.0 means pure exploitation eventually")
    print("   - memory_length = 0 means no memory (reactive agent)")
    print("   - These extremes work well against predictable opponents but fail against adaptive ones")

if __name__ == "__main__":
    analyze_sweep_results()