"""
Static Style Visualizer for NPD Simulator
Generates publication-ready figures matching the style of static_figure_generator.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class StaticStyleVisualizer:
    """Generate static_figure_generator.py style visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)
        
        # Color mapping for agent types
        self.agent_colors = {
            'TFT': '#2E86C1',
            'TFT-E': '#E74C3C',
            'AllD': '#E67E22',
            'AllC': '#27AE60',
            'pTFT': '#8E44AD',
            'pTFT-E': '#D35400'
        }
        
    def aggregate_multiple_runs(self, results_list: List[Dict]) -> Dict:
        """Aggregate results from multiple runs with statistics"""
        aggregated = defaultdict(lambda: defaultdict(list))
        
        for result in results_list:
            # Extract cooperation rates per round
            history = result.get('history', [])
            for round_data in history:
                round_num = round_data['round']
                
                # Overall cooperation
                coop_rate = round_data.get('cooperation_rate', 0)
                aggregated['cooperation_rates'][round_num].append(coop_rate)
                
                # Per-agent cooperation and scores
                for agent_id, agent_data in round_data.get('agents', {}).items():
                    agent_type = agent_data.get('type', 'Unknown')
                    
                    # Track by agent type
                    if 'cooperation_rate' in agent_data:
                        aggregated[f'{agent_type}_cooperation'][round_num].append(
                            agent_data['cooperation_rate']
                        )
                    
                    if 'cumulative_score' in agent_data:
                        aggregated[f'{agent_type}_scores'][round_num].append(
                            agent_data['cumulative_score']
                        )
        
        # Calculate statistics
        statistics = {}
        for key, round_data in aggregated.items():
            statistics[key] = {}
            for round_num, values in round_data.items():
                values = np.array(values)
                statistics[key][round_num] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'sem': np.std(values) / np.sqrt(len(values)),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'raw_values': values
                }
        
        return statistics
    
    def smooth_data(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply rolling window smoothing"""
        if len(data) < window:
            return data
        
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # Pad the beginning to maintain array length
        padding = np.full(window - 1, smoothed[0])
        return np.concatenate([padding, smoothed])
    
    def plot_2x2_grid(self, experiments: Dict[str, Dict], 
                      metric: str, title: str, 
                      ylabel: str, filename: str,
                      agent_type_filter: Optional[str] = None):
        """Create 2x2 grid plot comparing 4 experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        experiment_names = list(experiments.keys())[:4]  # Take first 4
        
        for idx, (exp_name, ax) in enumerate(zip(experiment_names, axes)):
            if exp_name not in experiments:
                ax.axis('off')
                continue
                
            stats = experiments[exp_name]
            
            # Extract data for the metric
            if agent_type_filter:
                metric_key = f'{agent_type_filter}_{metric}'
            else:
                metric_key = metric
                
            if metric_key not in stats:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(exp_name)
                continue
            
            metric_data = stats[metric_key]
            rounds = sorted(metric_data.keys())
            
            # Extract statistics
            means = [metric_data[r]['mean'] for r in rounds]
            ci_lower = [metric_data[r]['ci_lower'] for r in rounds]
            ci_upper = [metric_data[r]['ci_upper'] for r in rounds]
            
            # Plot individual runs (faded)
            for run_idx in range(len(metric_data[rounds[0]]['raw_values'])):
                run_values = [metric_data[r]['raw_values'][run_idx] 
                             for r in rounds if run_idx < len(metric_data[r]['raw_values'])]
                ax.plot(rounds[:len(run_values)], run_values, 
                       alpha=0.1, color='gray', linewidth=0.5)
            
            # Plot mean with CI
            ax.fill_between(rounds, ci_lower, ci_upper, 
                           alpha=0.3, color='blue', label='95% CI')
            
            # Plot smoothed mean
            smoothed_mean = self.smooth_data(np.array(means))
            ax.plot(rounds, smoothed_mean, 'b-', linewidth=2, label='Mean (smoothed)')
            
            # Plot raw mean
            ax.plot(rounds, means, 'b--', alpha=0.5, linewidth=1, label='Mean (raw)')
            
            ax.set_title(exp_name, fontsize=14, fontweight='bold')
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Only show legend on first subplot
                ax.legend(loc='best', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_figures(self, experiment_results: Dict[str, List[Dict]]):
        """Generate all 6 figure types matching static_figure_generator.py"""
        
        # Aggregate results for each experiment
        aggregated_experiments = {}
        for exp_name, results_list in experiment_results.items():
            aggregated_experiments[exp_name] = self.aggregate_multiple_runs(results_list)
        
        # Figure 1: Pairwise TFT Cooperation
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='cooperation_rates',
            title='Figure 1: Cooperation of TFT Agents Over Time (Pairwise)',
            ylabel='TFT Cooperation Rate',
            filename='figure1_pairwise_tft_cooperation.png',
            agent_type_filter='TFT'
        )
        
        # Figure 2: Neighbourhood TFT Cooperation
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='cooperation_rates',
            title='Figure 2: Cooperation of TFT Agents Over Time (Neighbourhood)',
            ylabel='TFT Cooperation Rate',
            filename='figure2_neighbourhood_tft_cooperation.png',
            agent_type_filter='TFT'
        )
        
        # Figure 3: Pairwise Agent Scores
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='scores',
            title='Figure 3: Cumulative Scores Over Time (Pairwise)',
            ylabel='Cumulative Score',
            filename='figure3_pairwise_agent_scores.png',
            agent_type_filter='TFT'
        )
        
        # Figure 4: Neighbourhood Agent Scores
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='scores',
            title='Figure 4: Cumulative Scores Over Time (Neighbourhood)',
            ylabel='Cumulative Score',
            filename='figure4_neighbourhood_agent_scores.png',
            agent_type_filter='TFT'
        )
        
        # Figure 5: Pairwise All Agent Cooperation
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='cooperation_rates',
            title='Figure 5: Overall Cooperation Over Time (Pairwise)',
            ylabel='Cooperation Rate',
            filename='figure5_pairwise_all_cooperation.png'
        )
        
        # Figure 6: Neighbourhood All Agent Cooperation
        self.plot_2x2_grid(
            aggregated_experiments,
            metric='cooperation_rates',
            title='Figure 6: Overall Cooperation Over Time (Neighbourhood)',
            ylabel='Cooperation Rate',
            filename='figure6_neighbourhood_all_cooperation.png'
        )
    
    def create_comparison_plots(self, results_path: Path):
        """Create comparison plots from existing results"""
        # Load results from JSON files
        experiment_results = defaultdict(list)
        
        # Map standard experiment names
        exp_name_mapping = {
            '3_tft': '3 TFT',
            '2_tft_e_1_alld': '2 TFT-E + 1 AllD',
            '2_tft_1_alld': '2 TFT + 1 AllD',
            '2_tft_e_1_allc': '2 TFT-E + 1 AllC'
        }
        
        # Find all result JSON files
        for json_file in results_path.rglob('*_results.json'):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                
                # Determine experiment type from config or filename
                exp_type = None
                for key, mapped_name in exp_name_mapping.items():
                    if key in str(json_file).lower():
                        exp_type = mapped_name
                        break
                
                if exp_type:
                    experiment_results[exp_type].append(result)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if experiment_results:
            self.generate_all_figures(experiment_results)
            print(f"Generated static-style figures in {self.output_dir}")
        else:
            print("No results found to process")