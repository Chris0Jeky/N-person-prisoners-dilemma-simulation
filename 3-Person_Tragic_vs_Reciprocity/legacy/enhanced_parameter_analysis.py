"""
Enhanced Parameter Analysis for Q-Learning
Addresses issues with limited parameter ranges and provides deeper insights
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas as pd
from enhanced_qlearning_demo_generator import (
    EnhancedQLearningAgent, StaticAgent, 
    run_multiple_simulations_extended, run_pairwise_simulation_extended,
    run_nperson_simulation_extended
)
import os
from datetime import datetime
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedParameterAnalyzer:
    def __init__(self, output_dir="enhanced_sweep_results"):
        """Initialize enhanced parameter analyzer with wider ranges."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Base parameters for reference
        self.base_params = {
            'learning_rate': 0.282,
            'discount_factor': 0.881,
            'epsilon': 0.5,
            'epsilon_decay': 0.822,
            'epsilon_min': 0.0046,
            'memory_length': 50,
            'state_type': 'proportion_discretized',
            'q_init_type': 'optimistic'
        }
        
        # Extended parameter ranges including extremes and finer granularity
        self.param_ranges = {
            'learning_rate': np.concatenate([
                np.array([0.001, 0.01]),  # Very low
                np.linspace(0.05, 0.5, 10),  # Original range
                np.linspace(0.6, 0.95, 5),   # High values
                np.array([0.99, 0.999])      # Near 1.0
            ]),
            'discount_factor': np.concatenate([
                np.array([0.0, 0.1, 0.3]),   # Low values
                np.linspace(0.5, 0.95, 10),   # Original range
                np.array([0.98, 0.99, 0.995, 0.999])  # Very high
            ]),
            'epsilon': np.concatenate([
                np.array([0.001, 0.01]),     # Very low (exploitation heavy)
                np.linspace(0.05, 0.5, 10),  # Original range
                np.linspace(0.6, 0.9, 5),    # High exploration
                np.array([0.95, 0.99])       # Near pure exploration
            ]),
            'epsilon_decay': np.concatenate([
                np.array([0.5, 0.7]),        # Fast decay
                np.linspace(0.8, 0.95, 8),   # Moderate decay
                np.array([0.98, 0.99, 0.995, 0.999, 0.9999])  # Very slow decay
            ]),
            'epsilon_min': np.concatenate([
                np.array([0.0, 0.0001, 0.001]),  # Near zero
                np.logspace(-3, -1, 7),           # Original range
                np.array([0.05, 0.1])             # Higher minimums
            ]),
            'memory_length': np.array([1, 5, 10, 20, 50, 100, 200, 500], dtype=int)
        }
        
        self.results = []
        
    def test_parameter_combination(self, params, num_runs=30, num_rounds=100,
                                 game_type='pairwise', opponent_strategy='TFT',
                                 training_rounds=1000):
        """Test a specific parameter combination with multiple metrics."""
        agents = []
        
        # Create Q-learning agent
        q_agent = EnhancedQLearningAgent(
            agent_id=0,
            learning_rate=params.get('learning_rate', self.base_params['learning_rate']),
            discount_factor=params.get('discount_factor', self.base_params['discount_factor']),
            epsilon=params.get('epsilon', self.base_params['epsilon']),
            epsilon_decay=params.get('epsilon_decay', self.base_params['epsilon_decay']),
            epsilon_min=params.get('epsilon_min', self.base_params['epsilon_min']),
            memory_length=int(params.get('memory_length', self.base_params['memory_length'])),
            state_type=params.get('state_type', self.base_params['state_type']),
            q_init_type=params.get('q_init_type', self.base_params['q_init_type'])
        )
        agents.append(q_agent)
        
        # Add opponents
        if game_type == 'pairwise':
            agents.append(StaticAgent(agent_id=1, strategy_name=opponent_strategy))
            sim_func = run_pairwise_simulation_extended
        else:
            for i in range(1, 5):
                if i <= 2:
                    agents.append(StaticAgent(agent_id=i, strategy_name='TFT'))
                else:
                    agents.append(StaticAgent(agent_id=i, strategy_name='AllD'))
            sim_func = run_nperson_simulation_extended
        
        # Run simulations
        coop_runs, score_runs = run_multiple_simulations_extended(
            sim_func, agents, num_rounds, num_runs, training_rounds=training_rounds
        )
        
        # Calculate comprehensive metrics
        avg_coop = np.mean(coop_runs)
        avg_score = np.mean(score_runs)
        coop_stability = np.std(coop_runs)
        score_stability = np.std(score_runs)
        
        # Learning metrics
        early_coop = np.mean([run[:20] for run in coop_runs])
        late_coop = np.mean([run[-20:] for run in coop_runs])
        learning_improvement = late_coop - early_coop
        
        # Convergence speed (rounds to reach stable performance)
        convergence_rounds = []
        for run in coop_runs:
            rolling_mean = pd.Series(run).rolling(10).mean()
            rolling_std = pd.Series(run).rolling(10).std()
            stable_idx = np.where((rolling_std < 0.1) & (~np.isnan(rolling_std)))[0]
            if len(stable_idx) > 0:
                convergence_rounds.append(stable_idx[0])
            else:
                convergence_rounds.append(num_rounds)
        avg_convergence = np.mean(convergence_rounds)
        
        return {
            'avg_cooperation': avg_coop,
            'avg_score': avg_score,
            'coop_stability': coop_stability,
            'score_stability': score_stability,
            'early_cooperation': early_coop,
            'late_cooperation': late_coop,
            'learning_improvement': learning_improvement,
            'convergence_speed': avg_convergence,
            'final_coop': np.mean([run[-1] for run in coop_runs])
        }
    
    def analyze_parameter_sensitivity(self, param_name, other_params=None):
        """Analyze sensitivity of a single parameter with fixed others."""
        if other_params is None:
            other_params = self.base_params.copy()
        
        results = []
        param_values = self.param_ranges[param_name]
        
        print(f"\\nAnalyzing sensitivity for {param_name}...")
        for i, value in enumerate(param_values):
            print(f"  Testing {param_name}={value:.4f} ({i+1}/{len(param_values)})")
            test_params = other_params.copy()
            test_params[param_name] = value
            
            metrics = self.test_parameter_combination(test_params)
            metrics[param_name] = value
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_region(self, param_names, num_samples=50):
        """Use optimization to find optimal parameter regions."""
        print(f"\\nSearching for optimal regions in {param_names}...")
        
        def objective(x):
            params = self.base_params.copy()
            for i, param in enumerate(param_names):
                params[param] = x[i]
            
            # Negative because we want to maximize
            metrics = self.test_parameter_combination(params, num_runs=10)
            return -(metrics['avg_cooperation'] + metrics['avg_score']/200)
        
        # Define bounds
        bounds = []
        for param in param_names:
            if param in ['learning_rate', 'epsilon', 'epsilon_min']:
                bounds.append((0.001, 0.999))
            elif param == 'discount_factor':
                bounds.append((0.0, 0.999))
            elif param == 'epsilon_decay':
                bounds.append((0.5, 0.9999))
            elif param == 'memory_length':
                bounds.append((1, 500))
        
        # Multiple random starts
        best_result = None
        best_score = float('inf')
        
        for _ in range(num_samples):
            x0 = []
            for i, param in enumerate(param_names):
                if param == 'memory_length':
                    x0.append(np.random.randint(bounds[i][0], bounds[i][1]))
                else:
                    x0.append(np.random.uniform(bounds[i][0], bounds[i][1]))
            
            try:
                result = optimize.minimize(objective, x0, bounds=bounds, 
                                         method='L-BFGS-B', options={'maxiter': 20})
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
            except:
                continue
        
        if best_result:
            optimal_params = {}
            for i, param in enumerate(param_names):
                optimal_params[param] = best_result.x[i]
            
            # Test the optimal parameters more thoroughly
            test_params = self.base_params.copy()
            test_params.update(optimal_params)
            metrics = self.test_parameter_combination(test_params, num_runs=50)
            
            return optimal_params, metrics
        
        return None, None
    
    def analyze_parameter_interactions(self, param1, param2, grid_size=10):
        """Analyze interaction between two parameters."""
        print(f"\\nAnalyzing interaction between {param1} and {param2}...")
        
        # Create grid
        if param1 == 'memory_length':
            p1_values = self.param_ranges[param1]
        else:
            p1_min = min(self.param_ranges[param1])
            p1_max = max(self.param_ranges[param1])
            p1_values = np.linspace(p1_min, p1_max, grid_size)
        
        if param2 == 'memory_length':
            p2_values = self.param_ranges[param2]
        else:
            p2_min = min(self.param_ranges[param2])
            p2_max = max(self.param_ranges[param2])
            p2_values = np.linspace(p2_min, p2_max, grid_size)
        
        results = []
        total = len(p1_values) * len(p2_values)
        count = 0
        
        for v1 in p1_values:
            for v2 in p2_values:
                count += 1
                print(f"  Testing combination {count}/{total}")
                
                test_params = self.base_params.copy()
                test_params[param1] = v1
                test_params[param2] = v2
                
                metrics = self.test_parameter_combination(test_params, num_runs=20)
                metrics[param1] = v1
                metrics[param2] = v2
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def diagnose_poor_performance(self, params):
        """Diagnose why specific parameters perform poorly."""
        print("\\nDiagnosing parameter performance...")
        
        # Test with different opponents
        opponents = ['TFT', 'AllD', 'AllC', 'Random']
        opponent_results = {}
        
        for opp in opponents:
            metrics = self.test_parameter_combination(
                params, opponent_strategy=opp, num_runs=20
            )
            opponent_results[opp] = metrics
        
        # Test with different game types
        pairwise_metrics = self.test_parameter_combination(
            params, game_type='pairwise', num_runs=20
        )
        nperson_metrics = self.test_parameter_combination(
            params, game_type='nperson', num_runs=20
        )
        
        # Analyze Q-value evolution
        q_agent = EnhancedQLearningAgent(
            agent_id=0,
            learning_rate=params.get('learning_rate', self.base_params['learning_rate']),
            discount_factor=params.get('discount_factor', self.base_params['discount_factor']),
            epsilon=params.get('epsilon', self.base_params['epsilon']),
            epsilon_decay=params.get('epsilon_decay', self.base_params['epsilon_decay']),
            epsilon_min=params.get('epsilon_min', self.base_params['epsilon_min']),
            memory_length=int(params.get('memory_length', self.base_params['memory_length'])),
            state_type=params.get('state_type', self.base_params['state_type']),
            q_init_type=params.get('q_init_type', self.base_params['q_init_type'])
        )
        
        # Track Q-values over time
        q_snapshots = []
        opponent = StaticAgent(agent_id=1, strategy_name='TFT')
        
        for round in range(0, 1000, 100):
            if round > 0:
                # Simulate some rounds
                for _ in range(100):
                    actions = [q_agent.choose_action() for _ in range(2)]
                    q_agent.update(actions, [3, 3])  # Mutual cooperation payoff
            
            q_snapshot = {}
            for state in q_agent.q_table:
                q_snapshot[state] = q_agent.q_table[state].copy()
            q_snapshots.append((round, q_snapshot))
        
        diagnosis = {
            'opponent_performance': opponent_results,
            'pairwise_metrics': pairwise_metrics,
            'nperson_metrics': nperson_metrics,
            'q_evolution': q_snapshots
        }
        
        return diagnosis
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\\nGenerating comprehensive parameter analysis report...")
        
        # 1. Single parameter sensitivity
        sensitivity_results = {}
        for param in ['learning_rate', 'discount_factor', 'epsilon', 
                     'epsilon_decay', 'epsilon_min']:
            df = self.analyze_parameter_sensitivity(param)
            sensitivity_results[param] = df
            df.to_csv(os.path.join(self.output_dir, f'sensitivity_{param}.csv'))
        
        # 2. Key parameter interactions
        interactions = [
            ('learning_rate', 'discount_factor'),
            ('epsilon', 'epsilon_decay'),
            ('discount_factor', 'memory_length')
        ]
        
        interaction_results = {}
        for p1, p2 in interactions:
            df = self.analyze_parameter_interactions(p1, p2)
            interaction_results[f'{p1}__{p2}'] = df
            df.to_csv(os.path.join(self.output_dir, f'interaction_{p1}_{p2}.csv'))
        
        # 3. Find optimal regions
        optimal_regions = {}
        param_sets = [
            ['learning_rate', 'discount_factor'],
            ['epsilon', 'epsilon_decay', 'epsilon_min'],
            ['learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay']
        ]
        
        for param_set in param_sets:
            opt_params, metrics = self.find_optimal_region(param_set, num_samples=20)
            if opt_params:
                optimal_regions['_'.join(param_set)] = {
                    'parameters': opt_params,
                    'metrics': metrics
                }
        
        # 4. Diagnose current "optimal" parameters
        diagnosis = self.diagnose_poor_performance(self.base_params)
        
        # 5. Create visualizations
        self.create_visualizations(sensitivity_results, interaction_results, 
                                 optimal_regions, diagnosis)
        
        # 6. Generate summary report
        self.write_summary_report(sensitivity_results, interaction_results,
                                optimal_regions, diagnosis)
        
        return {
            'sensitivity': sensitivity_results,
            'interactions': interaction_results,
            'optimal_regions': optimal_regions,
            'diagnosis': diagnosis
        }
    
    def create_visualizations(self, sensitivity, interactions, optimal, diagnosis):
        """Create comprehensive visualizations."""
        # 1. Sensitivity plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (param, df) in enumerate(sensitivity.items()):
            if i < 6:
                ax = axes[i]
                ax.plot(df[param], df['avg_cooperation'], 'b-', label='Cooperation', linewidth=2)
                ax.plot(df[param], df['avg_score']/200, 'r--', label='Score (normalized)', linewidth=2)
                ax.fill_between(df[param], 
                               df['avg_cooperation'] - df['coop_stability'],
                               df['avg_cooperation'] + df['coop_stability'],
                               alpha=0.3, color='blue')
                ax.set_xlabel(param)
                ax.set_ylabel('Performance')
                ax.set_title(f'Sensitivity: {param}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_sensitivity.png'), dpi=300)
        plt.close()
        
        # 2. Interaction heatmaps
        for name, df in interactions.items():
            params = name.split('__')
            pivot = df.pivot_table(values='avg_cooperation', 
                                  index=params[0], columns=params[1])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, cmap='viridis', annot=True, fmt='.3f')
            plt.title(f'Parameter Interaction: {params[0]} vs {params[1]}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'interaction_{name}.png'), dpi=300)
            plt.close()
        
        # 3. Performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compare against different opponents
        opp_perf = diagnosis['opponent_performance']
        opponents = list(opp_perf.keys())
        coop_scores = [opp_perf[o]['avg_cooperation'] for o in opponents]
        avg_scores = [opp_perf[o]['avg_score'] for o in opponents]
        
        x = np.arange(len(opponents))
        width = 0.35
        
        ax1.bar(x - width/2, coop_scores, width, label='Cooperation', color='skyblue')
        ax1.bar(x + width/2, np.array(avg_scores)/200, width, label='Score (norm)', color='lightcoral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(opponents)
        ax1.set_ylabel('Performance')
        ax1.set_title('Performance vs Different Opponents')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning curves
        ax2.plot(range(100), [0.5] * 100, 'g-', label='Target', linewidth=2)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cooperation Rate')
        ax2.set_title('Learning Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_analysis.png'), dpi=300)
        plt.close()
    
    def write_summary_report(self, sensitivity, interactions, optimal, diagnosis):
        """Write comprehensive text summary."""
        with open(os.path.join(self.output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("ENHANCED PARAMETER ANALYSIS REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write("1. PARAMETER SENSITIVITY ANALYSIS\\n")
            f.write("-" * 30 + "\\n")
            for param, df in sensitivity.items():
                best_idx = df['avg_cooperation'].idxmax()
                best_row = df.iloc[best_idx]
                f.write(f"\\n{param}:\\n")
                f.write(f"  Best value: {best_row[param]:.4f}\\n")
                f.write(f"  Cooperation: {best_row['avg_cooperation']:.3f}\\n")
                f.write(f"  Score: {best_row['avg_score']:.1f}\\n")
                f.write(f"  Stability: {best_row['coop_stability']:.3f}\\n")
            
            f.write("\\n2. OPTIMAL PARAMETER REGIONS\\n")
            f.write("-" * 30 + "\\n")
            for region_name, region_data in optimal.items():
                f.write(f"\\n{region_name}:\\n")
                for param, value in region_data['parameters'].items():
                    f.write(f"  {param}: {value:.4f}\\n")
                metrics = region_data['metrics']
                f.write(f"  Performance: Coop={metrics['avg_cooperation']:.3f}, ")
                f.write(f"Score={metrics['avg_score']:.1f}\\n")
            
            f.write("\\n3. CURRENT PARAMETERS DIAGNOSIS\\n")
            f.write("-" * 30 + "\\n")
            f.write("\\nPerformance vs opponents:\\n")
            for opp, metrics in diagnosis['opponent_performance'].items():
                f.write(f"  {opp}: Coop={metrics['avg_cooperation']:.3f}, ")
                f.write(f"Score={metrics['avg_score']:.1f}\\n")
            
            f.write("\\n4. RECOMMENDATIONS\\n")
            f.write("-" * 30 + "\\n")
            f.write("- Consider using adaptive parameter schedules\\n")
            f.write("- Test parameters against diverse opponent strategies\\n")
            f.write("- Monitor convergence speed and stability\\n")
            f.write("- Use ensemble methods for robustness\\n")


if __name__ == "__main__":
    analyzer = EnhancedParameterAnalyzer()
    results = analyzer.generate_report()
    print("\\nAnalysis complete! Check enhanced_sweep_results/ for detailed results.")