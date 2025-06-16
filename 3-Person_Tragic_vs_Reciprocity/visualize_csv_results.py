#!/usr/bin/env python3
"""
Standalone CSV visualizer for NPD simulation results
Reads data from CSV files and generates publication-ready figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import ast
from collections import defaultdict
import warnings
import sys

warnings.filterwarnings('ignore')


class CSVVisualizer:
    """Generate visualizations from CSV files"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        
    def read_csv_data(self, csv_dir: Path) -> Dict[str, pd.DataFrame]:
        """Read CSV files from a directory"""
        csv_data = {}
        
        # Look for history and summary CSV files
        history_file = None
        summary_file = None
        
        for csv_file in csv_dir.glob("*.csv"):
            if "history" in csv_file.name:
                history_file = csv_file
            elif "summary" in csv_file.name:
                summary_file = csv_file
        
        if history_file:
            csv_data['history'] = pd.read_csv(history_file)
            print(f"  - Loaded history from: {history_file}")
        if summary_file:
            csv_data['summary'] = pd.read_csv(summary_file)
            print(f"  - Loaded summary from: {summary_file}")
            
        return csv_data
    
    def parse_dict_column(self, series: pd.Series) -> pd.DataFrame:
        """Parse a column containing dictionary strings into separate columns"""
        parsed_data = []
        for item in series:
            try:
                # Convert string representation of dict to actual dict
                if isinstance(item, str):
                    parsed_dict = ast.literal_eval(item)
                else:
                    parsed_dict = item
                parsed_data.append(parsed_dict)
            except:
                parsed_data.append({})
        
        return pd.DataFrame(parsed_data)
    
    def plot_cooperation_evolution(self, history_df: pd.DataFrame) -> Path:
        """Plot cooperation rate over time"""
        plt.figure(figsize=(10, 6))
        
        # Plot cooperation rate
        plt.plot(history_df['round'], history_df['cooperation_rate'], 
                linewidth=2, label='Cooperation Rate', color='blue')
        
        # Add rolling average
        window = min(50, len(history_df) // 10)
        if window > 1:
            rolling_avg = history_df['cooperation_rate'].rolling(window).mean()
            plt.plot(history_df['round'], rolling_avg, 'r--', 
                    alpha=0.7, label=f'{window}-round average')
        
        plt.xlabel('Round')
        plt.ylabel('Cooperation Rate')
        plt.title('Evolution of Cooperation')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.figures_dir / 'cooperation_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_agent_actions_heatmap(self, history_df: pd.DataFrame) -> Path:
        """Plot agent actions as a heatmap"""
        # Parse actions column
        actions_df = self.parse_dict_column(history_df['actions'])
        
        if actions_df.empty:
            return None
            
        # Get number of agents
        num_agents = len(actions_df.columns)
        
        # Convert to numpy array and transpose for proper orientation
        actions_matrix = actions_df.values.T
        
        # Take last 100 rounds for clarity
        last_rounds = min(100, len(history_df))
        actions_matrix = actions_matrix[:, -last_rounds:]
        
        plt.figure(figsize=(12, 6))
        
        # Create heatmap (0 = cooperate = green, 1 = defect = red)
        sns.heatmap(actions_matrix, 
                   cmap='RdYlGn_r',  # Reversed so cooperation is green
                   cbar_kws={'label': 'Action (0=Cooperate, 1=Defect)'},
                   yticklabels=[f'Agent {i}' for i in range(num_agents)],
                   xticklabels=False)
        
        plt.xlabel(f'Round (last {last_rounds})')
        plt.title('Agent Actions Over Time')
        plt.tight_layout()
        
        output_path = self.figures_dir / 'cooperation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_score_distribution(self, summary_df: pd.DataFrame) -> Path:
        """Plot final score distribution from summary data"""
        # Find agent stats section
        agent_stats_start = None
        for idx, row in summary_df.iterrows():
            if pd.notna(row['Metric']) and 'Agent Stats' in str(row['Metric']):
                agent_stats_start = idx + 1
                break
        
        if agent_stats_start is None:
            return None
            
        # Extract agent data
        agent_data = []
        for idx in range(agent_stats_start, len(summary_df)):
            if pd.isna(summary_df.iloc[idx]['Metric']):
                break
            agent_data.append({
                'agent_id': int(summary_df.iloc[idx]['Metric']),
                'total_score': float(summary_df.iloc[idx]['Value']),
                'cooperation_rate': float(summary_df.iloc[idx].iloc[2]) if len(summary_df.columns) > 2 else 0
            })
        
        if not agent_data:
            return None
            
        agent_df = pd.DataFrame(agent_data)
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        bars = plt.bar(agent_df['agent_id'], agent_df['total_score'])
        
        # Color bars by cooperation rate
        for bar, coop_rate in zip(bars, agent_df['cooperation_rate']):
            bar.set_color(plt.cm.RdYlGn(coop_rate))
        
        plt.xlabel('Agent ID')
        plt.ylabel('Total Score')
        plt.title('Final Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar for cooperation rate
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Cooperation Rate')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'score_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_agent_performance(self, summary_df: pd.DataFrame, history_df: pd.DataFrame) -> Path:
        """Plot agent performance metrics"""
        # Extract agent stats from summary
        agent_stats_start = None
        for idx, row in summary_df.iterrows():
            if pd.notna(row['Metric']) and 'Agent Stats' in str(row['Metric']):
                agent_stats_start = idx + 1
                break
        
        if agent_stats_start is None:
            return None
            
        agent_data = []
        for idx in range(agent_stats_start, len(summary_df)):
            if pd.isna(summary_df.iloc[idx]['Metric']):
                break
            agent_data.append({
                'agent_id': int(summary_df.iloc[idx]['Metric']),
                'total_score': float(summary_df.iloc[idx]['Value']),
                'cooperation_rate': float(summary_df.iloc[idx].iloc[2]) if len(summary_df.columns) > 2 else 0
            })
        
        if not agent_data:
            return None
            
        agent_df = pd.DataFrame(agent_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cooperation rates
        ax1.bar(agent_df['agent_id'], agent_df['cooperation_rate'])
        ax1.set_xlabel('Agent ID')
        ax1.set_ylabel('Cooperation Rate')
        ax1.set_title('Agent Cooperation Rates')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Score vs cooperation scatter
        ax2.scatter(agent_df['cooperation_rate'], agent_df['total_score'], 
                   s=100, alpha=0.7)
        ax2.set_xlabel('Cooperation Rate')
        ax2.set_ylabel('Total Score')
        ax2.set_title('Score vs Cooperation Rate')
        ax2.grid(True, alpha=0.3)
        
        # Add agent labels
        for idx, row in agent_df.iterrows():
            ax2.annotate(f"A{row['agent_id']}", 
                        (row['cooperation_rate'], row['total_score']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'agent_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_plots(self, csv_dir: Path) -> Dict[str, Path]:
        """Generate all plots from CSV data"""
        # Read CSV data
        csv_data = self.read_csv_data(csv_dir)
        
        if not csv_data:
            print(f"No CSV data found in {csv_dir}")
            return {}
        
        generated_plots = {}
        
        # Generate plots based on available data
        if 'history' in csv_data:
            history_df = csv_data['history']
            
            # Cooperation evolution
            plot_path = self.plot_cooperation_evolution(history_df)
            if plot_path:
                generated_plots['cooperation_evolution'] = plot_path
                print(f"  ✓ Generated: {plot_path}")
            
            # Actions heatmap
            plot_path = self.plot_agent_actions_heatmap(history_df)
            if plot_path:
                generated_plots['cooperation_heatmap'] = plot_path
                print(f"  ✓ Generated: {plot_path}")
        
        if 'summary' in csv_data:
            summary_df = csv_data['summary']
            
            # Score distribution
            plot_path = self.plot_score_distribution(summary_df)
            if plot_path:
                generated_plots['score_distribution'] = plot_path
                print(f"  ✓ Generated: {plot_path}")
            
            # Agent performance
            if 'history' in csv_data:
                plot_path = self.plot_agent_performance(summary_df, csv_data['history'])
                if plot_path:
                    generated_plots['agent_performance'] = plot_path
                    print(f"  ✓ Generated: {plot_path}")
        
        return generated_plots
    
    def visualize_experiment(self, experiment_path: Path) -> Dict[str, Any]:
        """Visualize a single experiment from its CSV files"""
        csv_dir = experiment_path / "csv"
        
        if not csv_dir.exists():
            print(f"CSV directory not found: {csv_dir}")
            return {}
        
        print(f"\nGenerating visualizations for {experiment_path.name}...")
        generated_plots = self.generate_all_plots(csv_dir)
        
        print(f"\nSummary: Generated {len(generated_plots)} plots")
        
        return generated_plots
    
    def visualize_batch(self, batch_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Visualize all experiments in a batch directory"""
        results = {}
        
        print(f"\nProcessing batch directory: {batch_dir}")
        
        # Find all experiment directories with CSV subdirectories
        exp_dirs = []
        for exp_dir in batch_dir.rglob("csv"):
            if exp_dir.is_dir():
                exp_dirs.append(exp_dir.parent)
        
        print(f"Found {len(exp_dirs)} experiments to process")
        
        for exp_dir in exp_dirs:
            # Update output directory for this experiment
            self.output_dir = exp_dir
            self.figures_dir = exp_dir / "figures"
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            plots = self.visualize_experiment(exp_dir)
            if plots:
                results[exp_dir.name] = plots
        
        return results


def main():
    """Main function to run CSV visualizer"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_csv_results.py <path_to_experiment_or_batch>")
        print("\nExamples:")
        print("  python visualize_csv_results.py demo_results/basic_runs/demo_3agent_mixed")
        print("  python visualize_csv_results.py demo_results/batch_runs/batch_20250616_042128")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    # Create visualizer
    visualizer = CSVVisualizer(path)
    
    # Check if it's a batch directory or single experiment
    if (path / "csv").exists():
        # Single experiment
        visualizer.visualize_experiment(path)
    else:
        # Batch directory
        results = visualizer.visualize_batch(path)
        
        if results:
            print(f"\n{'='*50}")
            print(f"Batch visualization complete!")
            print(f"Processed {len(results)} experiments")
            print(f"{'='*50}")


if __name__ == "__main__":
    main()