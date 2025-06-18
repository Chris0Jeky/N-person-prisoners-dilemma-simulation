"""
Plot generation for experiment results
"""

from typing import Dict, List, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class PlotGenerator:
    """
    Generates various plots for experiment analysis.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_all_plots(self, results: Dict[str, Any]):
        """Generate all standard plots for experiment results."""
        self.plot_cooperation_evolution(results)
        self.plot_score_distribution(results)
        self.plot_agent_performance(results)
        
        if 'history' in results:
            self.plot_cooperation_heatmap(results)
    
    def plot_cooperation_evolution(self, results: Dict[str, Any]):
        """Plot cooperation rate over time."""
        if 'history' not in results:
            return
            
        history = results['history']
        rounds = [h['round'] for h in history]
        coop_rates = [h['cooperation_rate'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, coop_rates, linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Cooperation Rate')
        plt.title('Evolution of Cooperation')
        plt.ylim(0, 1)
        
        # Add rolling average
        window = min(50, len(rounds) // 10)
        if window > 1:
            rolling_avg = pd.Series(coop_rates).rolling(window).mean()
            plt.plot(rounds, rolling_avg, 'r--', alpha=0.7, label=f'{window}-round average')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooperation_evolution.png', dpi=300)
        plt.close()
    
    def plot_score_distribution(self, results: Dict[str, Any]):
        """Plot final score distribution."""
        if 'agent_stats' not in results:
            return
            
        agent_stats = results['agent_stats']
        scores = [stat['total_score'] for stat in agent_stats]
        agent_ids = [stat['agent_id'] for stat in agent_stats]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(agent_ids, scores)
        
        # Color by agent type if available
        if 'config' in results and 'agents' in results['config']:
            agent_types = [ag['type'] for ag in results['config']['agents']]
            unique_types = list(set(agent_types))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            
            for bar, agent_id in zip(bars, agent_ids):
                agent_type = agent_types[agent_id]
                color_idx = unique_types.index(agent_type)
                bar.set_color(colors[color_idx])
        
        plt.xlabel('Agent ID')
        plt.ylabel('Total Score')
        plt.title('Final Score Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png', dpi=300)
        plt.close()
    
    def plot_agent_performance(self, results: Dict[str, Any]):
        """Plot agent performance metrics."""
        if 'agent_stats' not in results:
            return
            
        agent_stats = results['agent_stats']
        
        # Prepare data
        data = pd.DataFrame(agent_stats)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cooperation rates
        ax1.bar(data['agent_id'], data['cooperation_rate'])
        ax1.set_xlabel('Agent ID')
        ax1.set_ylabel('Cooperation Rate')
        ax1.set_title('Agent Cooperation Rates')
        ax1.set_ylim(0, 1)
        
        # Score vs cooperation scatter
        ax2.scatter(data['cooperation_rate'], data['total_score'], s=100)
        ax2.set_xlabel('Cooperation Rate')
        ax2.set_ylabel('Total Score')
        ax2.set_title('Score vs Cooperation Rate')
        
        # Add agent labels
        for idx, row in data.iterrows():
            ax2.annotate(f"A{row['agent_id']}", 
                        (row['cooperation_rate'], row['total_score']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agent_performance.png', dpi=300)
        plt.close()
    
    def plot_cooperation_heatmap(self, results: Dict[str, Any]):
        """Plot cooperation patterns as heatmap."""
        if 'history' not in results:
            return
            
        history = results['history']
        num_agents = results.get('num_agents', 3)
        
        # Extract cooperation patterns over time
        cooperation_matrix = []
        
        for entry in history[-100:]:  # Last 100 rounds
            if 'actions' in entry:
                round_cooperation = [entry['actions'].get(i, 1) for i in range(num_agents)]
                cooperation_matrix.append(round_cooperation)
        
        if not cooperation_matrix:
            return
            
        cooperation_matrix = np.array(cooperation_matrix).T
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(1 - cooperation_matrix,  # Invert so cooperation is light
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Action (0=Cooperate, 1=Defect)'},
                   yticklabels=[f'Agent {i}' for i in range(num_agents)])
        plt.xlabel('Round (last 100)')
        plt.title('Agent Actions Over Time')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooperation_heatmap.png', dpi=300)
        plt.close()
    
    def generate_batch_comparison(self, results: List[Dict[str, Any]]):
        """Generate comparison plots for batch experiments."""
        # Extract summary statistics
        exp_names = []
        avg_coops = []
        avg_scores = []
        
        for i, result in enumerate(results):
            name = result.get('config', {}).get('name', f'Exp_{i+1}')
            exp_names.append(name)
            avg_coops.append(result.get('average_cooperation', 0))
            
            if 'agent_stats' in result:
                scores = [stat['total_score'] for stat in result['agent_stats']]
                avg_scores.append(sum(scores) / len(scores) if scores else 0)
            else:
                avg_scores.append(0)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cooperation rates
        ax1.bar(range(len(exp_names)), avg_coops)
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1.set_ylabel('Average Cooperation Rate')
        ax1.set_title('Cooperation Rates Across Experiments')
        ax1.set_ylim(0, 1)
        
        # Average scores
        ax2.bar(range(len(exp_names)), avg_scores)
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels(exp_names, rotation=45, ha='right')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Scores Across Experiments')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_comparison.png', dpi=300)
        plt.close()
    
    def generate_scaling_plots(self, analysis: Dict[str, Any]):
        """Generate plots for scaling analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        agent_counts = analysis['agent_counts']
        
        # Cooperation vs agent count
        ax1.plot(agent_counts, analysis['avg_cooperation'], 'o-', markersize=8)
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Average Cooperation Rate')
        ax1.set_title('Cooperation Scaling')
        ax1.grid(True)
        
        # Score vs agent count
        ax2.plot(agent_counts, analysis['avg_score'], 's-', markersize=8)
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Score Scaling')
        ax2.grid(True)
        
        # Score variance vs agent count
        ax3.plot(agent_counts, analysis['score_variance'], '^-', markersize=8)
        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Score Standard Deviation')
        ax3.set_title('Score Variance Scaling')
        ax3.grid(True)
        
        # Convergence time vs agent count
        ax4.plot(agent_counts, analysis['convergence_time'], 'd-', markersize=8)
        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Convergence Time (rounds)')
        ax4.set_title('Convergence Time Scaling')
        ax4.grid(True)
        
        plt.suptitle('Scaling Analysis: Effect of Agent Count', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_analysis.png', dpi=300)
        plt.close()
    
    def generate_sweep_plots(self, analysis: Dict[str, Any]):
        """Generate plots for parameter sweep analysis."""
        param_effects = analysis['parameter_effects']
        
        num_params = len(param_effects)
        fig, axes = plt.subplots(num_params, 2, figsize=(15, 5 * num_params))
        
        if num_params == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, effect) in enumerate(param_effects.items()):
            # Score vs parameter
            axes[i, 0].plot(effect['values'], effect['avg_scores'], 'o-', markersize=8)
            axes[i, 0].set_xlabel(param_name)
            axes[i, 0].set_ylabel('Average Score')
            axes[i, 0].set_title(f'Score vs {param_name}')
            axes[i, 0].grid(True)
            
            # Cooperation vs parameter
            axes[i, 1].plot(effect['values'], effect['avg_cooperation'], 's-', markersize=8)
            axes[i, 1].set_xlabel(param_name)
            axes[i, 1].set_ylabel('Average Cooperation Rate')
            axes[i, 1].set_title(f'Cooperation vs {param_name}')
            axes[i, 1].grid(True)
        
        plt.suptitle('Parameter Sweep Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_sweep.png', dpi=300)
        plt.close()