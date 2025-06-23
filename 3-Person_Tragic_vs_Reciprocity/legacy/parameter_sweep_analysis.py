"""
Parameter Sweep Analysis for Enhanced Q-Learning in Multi-Agent Games
Analyzes the effects of hyperparameters on agent performance in both single 
and combined parameter variations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, combinations
import pandas as pd
from enhanced_qlearning_demo_generator import (
    EnhancedQLearningAgent, StaticAgent, 
    run_multiple_simulations_extended, run_pairwise_simulation_extended,
    run_nperson_simulation_extended
)
import os
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ParameterSweepAnalyzer:
    def __init__(self, base_params=None, output_dir=None):
        """Initialize the parameter sweep analyzer."""
        self.base_params = base_params or {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 0.25,
            'epsilon_decay': 0.9,
            'epsilon_min': 0.01,
            'memory_length': 50,
            'state_type': 'proportion_discretized',
            'q_init_type': 'optimistic'
        }
        
        # Define parameter ranges for sweeps
        self.param_ranges = {
            'learning_rate': np.linspace(0.01, 0.5, 10),
            'discount_factor': np.linspace(0.5, 0.99, 10),
            'epsilon': np.linspace(0.1, 0.5, 10),
            'epsilon_decay': np.linspace(0.8, 0.999, 10),
            'epsilon_min': np.logspace(-3, -1, 10),
            'memory_length': np.array([10, 20, 30, 50, 75, 100, 150, 200], dtype=int)
        }
        
        # Create output directory - static folder that gets overwritten
        if output_dir is None:
            output_dir = "parameter_sweep_results"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'single': {},
            'double': {},
            'triple': {}
        }
        
    def run_simulation_with_params(self, params, num_runs=50, num_rounds=100, 
                                 game_type='pairwise', opponent_strategy='TFT'):
        """Run simulation with specific parameters."""
        agents = []
        
        # Create Q-learning agent with specified parameters
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
        
        # Add opponents based on game type
        if game_type == 'pairwise':
            agents.append(StaticAgent(agent_id=1, strategy_name=opponent_strategy))
            sim_func = run_pairwise_simulation_extended
        else:  # n-person game
            for i in range(1, 5):  # 5-person game
                if i <= 2:
                    agents.append(StaticAgent(agent_id=i, strategy_name='TFT'))
                else:
                    agents.append(StaticAgent(agent_id=i, strategy_name='AllD'))
            sim_func = run_nperson_simulation_extended
        
        # Run simulations
        coop_runs, score_runs = run_multiple_simulations_extended(
            sim_func, agents, num_rounds, num_runs, training_rounds=1000
        )
        
        # Calculate metrics
        q_agent_id = 0
        avg_cooperation = np.mean([np.mean(run) for run in coop_runs[q_agent_id]])
        avg_score = np.mean([np.mean(run) for run in score_runs[q_agent_id]])
        final_cooperation = np.mean([run[-1] for run in coop_runs[q_agent_id]])
        cooperation_stability = np.mean([np.std(run) for run in coop_runs[q_agent_id]])
        
        return {
            'avg_cooperation': avg_cooperation,
            'avg_score': avg_score,
            'final_cooperation': final_cooperation,
            'cooperation_stability': cooperation_stability,
            'coop_runs': coop_runs[q_agent_id],
            'score_runs': score_runs[q_agent_id]
        }
    
    def single_parameter_sweep(self, param_name, param_values=None, **kwargs):
        """Sweep a single parameter while keeping others fixed."""
        if param_values is None:
            param_values = self.param_ranges[param_name]
        
        results = []
        for value in param_values:
            params = self.base_params.copy()
            params[param_name] = value
            
            result = self.run_simulation_with_params(params, **kwargs)
            result['param_value'] = value
            results.append(result)
        
        self.results['single'][param_name] = results
        return results
    
    def double_parameter_sweep(self, param1_name, param2_name, 
                             param1_values=None, param2_values=None, **kwargs):
        """Sweep two parameters simultaneously."""
        if param1_values is None:
            param1_values = self.param_ranges[param1_name][:5]  # Reduce for computational efficiency
        if param2_values is None:
            param2_values = self.param_ranges[param2_name][:5]
        
        results = []
        for val1, val2 in product(param1_values, param2_values):
            params = self.base_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            
            result = self.run_simulation_with_params(params, **kwargs)
            result['param1_value'] = val1
            result['param2_value'] = val2
            results.append(result)
        
        key = f"{param1_name}__{param2_name}"
        self.results['double'][key] = results
        return results
    
    def triple_parameter_sweep(self, param1_name, param2_name, param3_name,
                             param1_values=None, param2_values=None, 
                             param3_values=None, **kwargs):
        """Sweep three parameters simultaneously."""
        if param1_values is None:
            param1_values = self.param_ranges[param1_name][:3]  # Very reduced for efficiency
        if param2_values is None:
            param2_values = self.param_ranges[param2_name][:3]
        if param3_values is None:
            param3_values = self.param_ranges[param3_name][:3]
        
        results = []
        for val1, val2, val3 in product(param1_values, param2_values, param3_values):
            params = self.base_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            params[param3_name] = val3
            
            result = self.run_simulation_with_params(params, **kwargs)
            result['param1_value'] = val1
            result['param2_value'] = val2
            result['param3_value'] = val3
            results.append(result)
        
        key = f"{param1_name}__{param2_name}__{param3_name}"
        self.results['triple'][key] = results
        return results
    
    def visualize_single_parameter_sweep(self, param_name, results=None):
        """Create comprehensive visualizations for single parameter sweep."""
        if results is None:
            results = self.results['single'][param_name]
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        param_values = [r['param_value'] for r in results]
        
        # 1. Main metrics plot
        ax1 = fig.add_subplot(gs[0, :])
        metrics = ['avg_cooperation', 'avg_score', 'final_cooperation']
        colors = ['blue', 'green', 'red']
        
        for metric, color in zip(metrics, colors):
            values = [r[metric] for r in results]
            ax1.plot(param_values, values, 'o-', label=metric.replace('_', ' ').title(), 
                    color=color, linewidth=2, markersize=8)
        
        ax1.set_xlabel(f'{param_name.replace("_", " ").title()}', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'Effect of {param_name.replace("_", " ").title()} on Performance Metrics', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cooperation stability
        ax2 = fig.add_subplot(gs[1, 0])
        stability_values = [r['cooperation_stability'] for r in results]
        ax2.plot(param_values, stability_values, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.set_xlabel(f'{param_name.replace("_", " ").title()}', fontsize=12)
        ax2.set_ylabel('Cooperation Stability (Std Dev)', fontsize=12)
        ax2.set_title('Cooperation Stability Across Runs', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot of cooperation rates
        ax3 = fig.add_subplot(gs[1, 1])
        coop_data = []
        labels = []
        for i, r in enumerate(results[::max(1, len(results)//8)]):  # Sample for clarity
            coop_data.append([np.mean(run) for run in r['coop_runs']])
            labels.append(f'{r["param_value"]:.3f}')
        
        bp = ax3.boxplot(coop_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_xlabel(f'{param_name.replace("_", " ").title()}', fontsize=12)
        ax3.set_ylabel('Average Cooperation Rate', fontsize=12)
        ax3.set_title('Distribution of Cooperation Rates', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Learning curves
        ax4 = fig.add_subplot(gs[1, 2])
        sample_indices = np.linspace(0, len(results)-1, min(5, len(results)), dtype=int)
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
        
        for idx, color in zip(sample_indices, colors_gradient):
            r = results[idx]
            avg_coop_over_time = np.mean(r['coop_runs'], axis=0)
            ax4.plot(avg_coop_over_time, label=f'{param_name}={r["param_value"]:.3f}', 
                    color=color, alpha=0.7)
        
        ax4.set_xlabel('Round', fontsize=12)
        ax4.set_ylabel('Average Cooperation Rate', fontsize=12)
        ax4.set_title('Learning Curves for Different Parameter Values', fontsize=12)
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlation heatmap
        ax5 = fig.add_subplot(gs[2, 0])
        corr_data = pd.DataFrame({
            param_name: param_values,
            'Avg Cooperation': [r['avg_cooperation'] for r in results],
            'Avg Score': [r['avg_score'] for r in results],
            'Final Cooperation': [r['final_cooperation'] for r in results],
            'Stability': stability_values
        })
        corr_matrix = corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=ax5, cbar_kws={'label': 'Correlation'})
        ax5.set_title('Correlation Matrix', fontsize=12)
        
        # 6. Regression analysis
        ax6 = fig.add_subplot(gs[2, 1])
        x = param_values
        y = [r['avg_cooperation'] for r in results]
        
        # Fit polynomial
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(x), max(x), 100)
        
        ax6.scatter(x, y, color='blue', s=50, alpha=0.6, label='Data')
        ax6.plot(x_smooth, p(x_smooth), 'r-', linewidth=2, label='Polynomial Fit')
        ax6.set_xlabel(f'{param_name.replace("_", " ").title()}', fontsize=12)
        ax6.set_ylabel('Average Cooperation', fontsize=12)
        ax6.set_title('Regression Analysis', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance improvement
        ax7 = fig.add_subplot(gs[2, 2])
        baseline_coop = results[0]['avg_cooperation']
        improvements = [(r['avg_cooperation'] - baseline_coop) / baseline_coop * 100 
                       for r in results]
        
        colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax7.bar(range(len(improvements)), improvements, color=colors_imp, alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_xlabel('Parameter Value Index', fontsize=12)
        ax7.set_ylabel('% Improvement from Baseline', fontsize=12)
        ax7.set_title('Performance Improvement Relative to Baseline', fontsize=12)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Final round analysis
        ax8 = fig.add_subplot(gs[3, :])
        final_coops = []
        for r in results:
            final_coops.append([run[-1] for run in r['coop_runs']])
        
        positions = np.arange(len(results))
        width = 0.8
        
        violin_parts = ax8.violinplot(final_coops, positions=positions, widths=width,
                                     showmeans=True, showmedians=True)
        
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        
        ax8.set_xlabel(f'{param_name.replace("_", " ").title()}', fontsize=12)
        ax8.set_ylabel('Final Round Cooperation Rate', fontsize=12)
        ax8.set_title('Distribution of Final Round Cooperation Rates', fontsize=12)
        ax8.set_xticks(positions[::max(1, len(positions)//10)])
        ax8.set_xticklabels([f'{v:.3f}' for v in param_values[::max(1, len(param_values)//10)]])
        ax8.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Comprehensive Analysis: {param_name.replace("_", " ").title()} Parameter Sweep', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        filename = os.path.join(self.output_dir, f'single_sweep_{param_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved single parameter sweep visualization: {filename}")
        
        # Save results to CSV
        csv_filename = os.path.join(self.output_dir, f'single_sweep_{param_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV data: {csv_filename}")
    
    def visualize_double_parameter_sweep(self, param1_name, param2_name, results=None):
        """Create comprehensive visualizations for double parameter sweep."""
        key = f"{param1_name}__{param2_name}"
        if results is None:
            results = self.results['double'][key]
        
        # Reshape results for heatmaps
        param1_values = sorted(list(set(r['param1_value'] for r in results)))
        param2_values = sorted(list(set(r['param2_value'] for r in results)))
        
        # Create matrices for different metrics
        metrics = ['avg_cooperation', 'avg_score', 'final_cooperation', 'cooperation_stability']
        matrices = {metric: np.zeros((len(param2_values), len(param1_values))) for metric in metrics}
        
        for r in results:
            i = param2_values.index(r['param2_value'])
            j = param1_values.index(r['param1_value'])
            for metric in metrics:
                matrices[metric][i, j] = r[metric]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1-4. Heatmaps for each metric
        for idx, (metric, matrix) in enumerate(matrices.items()):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(param1_values)))
            ax.set_yticks(range(len(param2_values)))
            ax.set_xticklabels([f'{v:.3f}' for v in param1_values], rotation=45)
            ax.set_yticklabels([f'{v:.3f}' for v in param2_values])
            ax.set_xlabel(param1_name.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(param2_name.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(param2_values)):
                for j in range(len(param1_values)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        # 5. 3D surface plot
        ax5 = fig.add_subplot(gs[0, 2], projection='3d')
        X, Y = np.meshgrid(param1_values, param2_values)
        surf = ax5.plot_surface(X, Y, matrices['avg_cooperation'], cmap='viridis',
                               linewidth=0, antialiased=True, alpha=0.8)
        ax5.set_xlabel(param1_name.replace('_', ' ').title(), fontsize=10)
        ax5.set_ylabel(param2_name.replace('_', ' ').title(), fontsize=10)
        ax5.set_zlabel('Avg Cooperation', fontsize=10)
        ax5.set_title('3D Surface: Average Cooperation', fontsize=12)
        fig.colorbar(surf, ax=ax5, shrink=0.5, aspect=5)
        
        # 6. Contour plot
        ax6 = fig.add_subplot(gs[1, 2])
        contour = ax6.contour(X, Y, matrices['avg_cooperation'], levels=10, colors='black', alpha=0.4)
        contourf = ax6.contourf(X, Y, matrices['avg_cooperation'], levels=20, cmap='RdYlBu_r')
        ax6.clabel(contour, inline=True, fontsize=8)
        ax6.set_xlabel(param1_name.replace('_', ' ').title(), fontsize=12)
        ax6.set_ylabel(param2_name.replace('_', ' ').title(), fontsize=12)
        ax6.set_title('Contour Plot: Average Cooperation', fontsize=12)
        fig.colorbar(contourf, ax=ax6)
        
        # 7. Optimal region identification
        ax7 = fig.add_subplot(gs[2, :])
        
        # Find optimal regions (top 20% performance)
        threshold = np.percentile(matrices['avg_cooperation'], 80)
        optimal_mask = matrices['avg_cooperation'] >= threshold
        
        # Create custom colormap
        colors_opt = ['lightgray', 'darkgreen']
        n_bins = 2
        cmap_opt = plt.cm.colors.ListedColormap(colors_opt)
        
        im = ax7.imshow(optimal_mask.astype(int), cmap=cmap_opt, aspect='auto')
        ax7.set_xticks(range(len(param1_values)))
        ax7.set_yticks(range(len(param2_values)))
        ax7.set_xticklabels([f'{v:.3f}' for v in param1_values], rotation=45)
        ax7.set_yticklabels([f'{v:.3f}' for v in param2_values])
        ax7.set_xlabel(param1_name.replace('_', ' ').title(), fontsize=12)
        ax7.set_ylabel(param2_name.replace('_', ' ').title(), fontsize=12)
        ax7.set_title('Optimal Parameter Regions (Top 20% Performance)', fontsize=14)
        
        # Add legend
        patches = [mpatches.Patch(color='lightgray', label='Below 80th percentile'),
                  mpatches.Patch(color='darkgreen', label='Top 20% (Optimal)')]
        ax7.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.suptitle(f'Double Parameter Sweep: {param1_name} vs {param2_name}', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        filename = os.path.join(self.output_dir, f'double_sweep_{param1_name}__{param2_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved double parameter sweep visualization: {filename}")
        
        # Save results to CSV
        csv_filename = os.path.join(self.output_dir, f'double_sweep_{param1_name}__{param2_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV data: {csv_filename}")
    
    def visualize_triple_parameter_sweep(self, param1_name, param2_name, param3_name, results=None):
        """Create visualizations for triple parameter sweep."""
        key = f"{param1_name}__{param2_name}__{param3_name}"
        if results is None:
            results = self.results['triple'][key]
        
        # Get unique values for each parameter
        param3_values = sorted(list(set(r['param3_value'] for r in results)))
        
        # Create figure with subplots for each param3 value
        n_param3 = len(param3_values)
        fig = plt.figure(figsize=(20, 6 * n_param3))
        
        for idx, param3_val in enumerate(param3_values):
            # Filter results for this param3 value
            filtered_results = [r for r in results if r['param3_value'] == param3_val]
            
            # Create heatmap for this slice
            param1_values = sorted(list(set(r['param1_value'] for r in filtered_results)))
            param2_values = sorted(list(set(r['param2_value'] for r in filtered_results)))
            
            matrix = np.zeros((len(param2_values), len(param1_values)))
            for r in filtered_results:
                i = param2_values.index(r['param2_value'])
                j = param1_values.index(r['param1_value'])
                matrix[i, j] = r['avg_cooperation']
            
            # Create subplot
            ax = plt.subplot(n_param3, 2, idx * 2 + 1)
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(param1_values)))
            ax.set_yticks(range(len(param2_values)))
            ax.set_xticklabels([f'{v:.3f}' for v in param1_values], rotation=45)
            ax.set_yticklabels([f'{v:.3f}' for v in param2_values])
            ax.set_xlabel(param1_name.replace('_', ' ').title())
            ax.set_ylabel(param2_name.replace('_', ' ').title())
            ax.set_title(f'Avg Cooperation | {param3_name}={param3_val:.3f}')
            plt.colorbar(im, ax=ax)
            
            # Create 3D scatter plot
            ax3d = plt.subplot(n_param3, 2, idx * 2 + 2, projection='3d')
            x = [r['param1_value'] for r in filtered_results]
            y = [r['param2_value'] for r in filtered_results]
            z = [r['avg_cooperation'] for r in filtered_results]
            
            scatter = ax3d.scatter(x, y, z, c=z, cmap='viridis', s=50)
            ax3d.set_xlabel(param1_name.replace('_', ' ').title())
            ax3d.set_ylabel(param2_name.replace('_', ' ').title())
            ax3d.set_zlabel('Avg Cooperation')
            ax3d.set_title(f'3D View | {param3_name}={param3_val:.3f}')
            plt.colorbar(scatter, ax=ax3d)
        
        plt.suptitle(f'Triple Parameter Sweep: {param1_name} vs {param2_name} vs {param3_name}', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        filename = os.path.join(self.output_dir, 
                               f'triple_sweep_{param1_name}__{param2_name}__{param3_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved triple parameter sweep visualization: {filename}")
        
        # Save results to CSV
        csv_filename = os.path.join(self.output_dir, 
                                    f'triple_sweep_{param1_name}__{param2_name}__{param3_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV data: {csv_filename}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report of all parameter sweeps."""
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Single parameter effects summary
        ax1 = fig.add_subplot(gs[0, :])
        param_effects = {}
        
        for param_name, results in self.results['single'].items():
            if results:
                values = [r['avg_cooperation'] for r in results]
                param_effects[param_name] = {
                    'range': max(values) - min(values),
                    'std': np.std(values),
                    'max': max(values),
                    'min': min(values)
                }
        
        if param_effects:
            params = list(param_effects.keys())
            ranges = [param_effects[p]['range'] for p in params]
            
            bars = ax1.bar(params, ranges, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Parameter', fontsize=12)
            ax1.set_ylabel('Performance Range (Max - Min)', fontsize=12)
            ax1.set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, param in zip(bars, params):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Best parameter values
        ax2 = fig.add_subplot(gs[1, 0])
        best_values = {}
        
        for param_name, results in self.results['single'].items():
            if results:
                best_idx = np.argmax([r['avg_cooperation'] for r in results])
                best_values[param_name] = {
                    'value': results[best_idx]['param_value'],
                    'performance': results[best_idx]['avg_cooperation']
                }
        
        if best_values:
            y_pos = np.arange(len(best_values))
            params = list(best_values.keys())
            values = [best_values[p]['value'] for p in params]
            performances = [best_values[p]['performance'] for p in params]
            
            # Normalize values for display
            base_values = [self.base_params[p] for p in params]
            relative_values = [(v - b) / b * 100 for v, b in zip(values, base_values)]
            
            bars = ax2.barh(y_pos, relative_values, color=['green' if x > 0 else 'red' for x in relative_values])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(params)
            ax2.set_xlabel('% Change from Base Value', fontsize=12)
            ax2.set_title('Optimal Parameter Values (% Change)', fontsize=12)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add performance annotations
            for i, (bar, perf) in enumerate(zip(bars, performances)):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{perf:.3f}', ha='left', va='center', fontsize=9)
        
        # 3. Parameter importance ranking
        ax3 = fig.add_subplot(gs[1, 1])
        importance_scores = []
        param_names = []
        
        for param_name, results in self.results['single'].items():
            if results:
                values = [r['avg_cooperation'] for r in results]
                # Importance = range * correlation with performance
                param_vals = [r['param_value'] for r in results]
                if len(values) > 1:
                    corr = abs(np.corrcoef(param_vals, values)[0, 1])
                    importance = (max(values) - min(values)) * corr
                    importance_scores.append(importance)
                    param_names.append(param_name)
        
        if importance_scores:
            sorted_indices = np.argsort(importance_scores)[::-1]
            sorted_params = [param_names[i] for i in sorted_indices]
            sorted_scores = [importance_scores[i] for i in sorted_indices]
            
            colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_params)))
            bars = ax3.bar(range(len(sorted_params)), sorted_scores, color=colors_imp)
            ax3.set_xticks(range(len(sorted_params)))
            ax3.set_xticklabels(sorted_params, rotation=45)
            ax3.set_ylabel('Importance Score', fontsize=12)
            ax3.set_title('Parameter Importance Ranking', fontsize=12)
        
        # 4. Interaction effects summary
        ax4 = fig.add_subplot(gs[1, 2])
        if self.results['double']:
            interaction_data = []
            labels = []
            
            for key, results in self.results['double'].items():
                if results:
                    values = [r['avg_cooperation'] for r in results]
                    interaction_strength = np.std(values)
                    interaction_data.append(interaction_strength)
                    labels.append(key.replace('__', ' vs\n'))
            
            if interaction_data:
                bars = ax4.bar(range(len(interaction_data)), interaction_data, color='coral')
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, rotation=0, fontsize=9)
                ax4.set_ylabel('Interaction Strength (Std Dev)', fontsize=12)
                ax4.set_title('Parameter Interaction Strengths', fontsize=12)
        
        # 5. Performance distribution across all experiments
        ax5 = fig.add_subplot(gs[2, :])
        all_performances = []
        
        # Collect all performance values
        for results_dict in [self.results['single'], self.results['double'], self.results['triple']]:
            for results in results_dict.values():
                all_performances.extend([r['avg_cooperation'] for r in results])
        
        if all_performances:
            ax5.hist(all_performances, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(all_performances), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_performances):.3f}')
            ax5.axvline(np.median(all_performances), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(all_performances):.3f}')
            ax5.set_xlabel('Average Cooperation Rate', fontsize=12)
            ax5.set_ylabel('Frequency', fontsize=12)
            ax5.set_title('Distribution of Performance Across All Parameter Configurations', fontsize=14)
            ax5.legend()
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary statistics
        summary_data = []
        summary_data.append(['Metric', 'Value'])
        summary_data.append(['Total Experiments', str(len(all_performances))])
        summary_data.append(['Best Performance', f'{max(all_performances):.4f}' if all_performances else 'N/A'])
        summary_data.append(['Worst Performance', f'{min(all_performances):.4f}' if all_performances else 'N/A'])
        summary_data.append(['Average Performance', f'{np.mean(all_performances):.4f}' if all_performances else 'N/A'])
        summary_data.append(['Performance Std Dev', f'{np.std(all_performances):.4f}' if all_performances else 'N/A'])
        
        # Find best configuration
        best_config = None
        best_perf = -1
        for param_name, results in self.results['single'].items():
            for r in results:
                if r['avg_cooperation'] > best_perf:
                    best_perf = r['avg_cooperation']
                    best_config = f"{param_name}={r['param_value']:.3f}"
        
        summary_data.append(['Best Single Param Config', best_config or 'N/A'])
        
        table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('Parameter Sweep Analysis Summary Report', fontsize=18, fontweight='bold')
        
        # Save figure
        filename = os.path.join(self.output_dir, 'summary_report.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary report: {filename}")
        
        # Also save a text summary
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Parameter Sweep Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Base Parameters:\n")
            for param, value in self.base_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Single Parameter Sweep Results:\n")
            f.write("=" * 50 + "\n")
            
            for param_name, results in self.results['single'].items():
                if results:
                    values = [r['avg_cooperation'] for r in results]
                    f.write(f"\n{param_name}:\n")
                    f.write(f"  Range: {min(values):.4f} - {max(values):.4f}\n")
                    f.write(f"  Best value: {results[np.argmax(values)]['param_value']:.4f}\n")
                    f.write(f"  Best performance: {max(values):.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Key Insights:\n")
            f.write("=" * 50 + "\n")
            
            if all_performances:
                f.write(f"- Total configurations tested: {len(all_performances)}\n")
                f.write(f"- Performance range: {min(all_performances):.4f} - {max(all_performances):.4f}\n")
                f.write(f"- Average performance: {np.mean(all_performances):.4f} ± {np.std(all_performances):.4f}\n")
        
        # Save comprehensive CSV summary
        all_results = []
        
        # Collect single parameter sweep results
        for param_name, results in self.results['single'].items():
            for r in results:
                row = {
                    'sweep_type': 'single',
                    'param1_name': param_name,
                    'param1_value': r['param_value'],
                    'param2_name': None,
                    'param2_value': None,
                    'param3_name': None,
                    'param3_value': None,
                    'avg_cooperation': r['avg_cooperation'],
                    'avg_score': r['avg_score'],
                    'final_cooperation': r['final_cooperation'],
                    'cooperation_stability': r['cooperation_stability']
                }
                all_results.append(row)
        
        # Collect double parameter sweep results
        for key, results in self.results['double'].items():
            param1_name, param2_name = key.split('__')
            for r in results:
                row = {
                    'sweep_type': 'double',
                    'param1_name': param1_name,
                    'param1_value': r['param1_value'],
                    'param2_name': param2_name,
                    'param2_value': r['param2_value'],
                    'param3_name': None,
                    'param3_value': None,
                    'avg_cooperation': r['avg_cooperation'],
                    'avg_score': r['avg_score'],
                    'final_cooperation': r['final_cooperation'],
                    'cooperation_stability': r['cooperation_stability']
                }
                all_results.append(row)
        
        # Collect triple parameter sweep results
        for key, results in self.results['triple'].items():
            param1_name, param2_name, param3_name = key.split('__')
            for r in results:
                row = {
                    'sweep_type': 'triple',
                    'param1_name': param1_name,
                    'param1_value': r['param1_value'],
                    'param2_name': param2_name,
                    'param2_value': r['param2_value'],
                    'param3_name': param3_name,
                    'param3_value': r['param3_value'],
                    'avg_cooperation': r['avg_cooperation'],
                    'avg_score': r['avg_score'],
                    'final_cooperation': r['final_cooperation'],
                    'cooperation_stability': r['cooperation_stability']
                }
                all_results.append(row)
        
        # Save comprehensive results
        if all_results:
            df_all = pd.DataFrame(all_results)
            csv_filename = os.path.join(self.output_dir, 'all_sweep_results.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"Saved comprehensive CSV: {csv_filename}")


def main():
    """Run comprehensive parameter sweep analysis."""
    print("Starting parameter sweep analysis...")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ParameterSweepAnalyzer()
    
    # Parameters to sweep
    params_to_sweep = ['learning_rate', 'discount_factor', 'epsilon', 
                      'epsilon_decay', 'epsilon_min', 'memory_length']
    
    # Run single parameter sweeps
    print("\n1. Running single parameter sweeps...")
    for param in params_to_sweep:
        print(f"   Sweeping {param}...")
        results = analyzer.single_parameter_sweep(param, num_runs=30, num_rounds=100)
        analyzer.visualize_single_parameter_sweep(param)
    
    # Run double parameter sweeps for key combinations
    print("\n2. Running double parameter sweeps...")
    double_combinations = [
        ('learning_rate', 'discount_factor'),
        ('epsilon', 'epsilon_decay'),
        ('learning_rate', 'epsilon'),
        ('discount_factor', 'memory_length')
    ]
    
    for param1, param2 in double_combinations:
        print(f"   Sweeping {param1} vs {param2}...")
        results = analyzer.double_parameter_sweep(param1, param2, num_runs=20, num_rounds=100)
        analyzer.visualize_double_parameter_sweep(param1, param2)
    
    # Run triple parameter sweeps for critical combinations
    print("\n3. Running triple parameter sweeps...")
    triple_combinations = [
        ('learning_rate', 'discount_factor', 'epsilon'),
        ('epsilon', 'epsilon_decay', 'epsilon_min')
    ]
    
    for param1, param2, param3 in triple_combinations:
        print(f"   Sweeping {param1} vs {param2} vs {param3}...")
        results = analyzer.triple_parameter_sweep(param1, param2, param3, 
                                                num_runs=15, num_rounds=100)
        analyzer.visualize_triple_parameter_sweep(param1, param2, param3)
    
    # Create summary report
    print("\n4. Creating summary report...")
    analyzer.create_summary_report()
    
    print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()