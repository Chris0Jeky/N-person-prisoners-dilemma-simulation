"""
Experiment visualizer that creates figures in timestamped directories.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ExperimentVisualizer:
    """Creates visualizations for experiment results in proper directory structure."""
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        
    def create_experiment_figures(self, experiment_dir: Path):
        """Create figures for a single experiment directory."""
        figures_dir = experiment_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Check for CSV files
        csv_dir = experiment_dir / "csv"
        if csv_dir.exists():
            self._create_figures_from_csv(csv_dir, figures_dir, experiment_dir.name)
        
        # Check for JSON results
        json_files = list(experiment_dir.glob("*_results.json"))
        if json_files:
            self._create_figures_from_json(json_files[0], figures_dir)
    
    def _create_figures_from_csv(self, csv_dir: Path, figures_dir: Path, experiment_name: str):
        """Create figures from CSV data."""
        # Load history and summary CSVs
        history_file = csv_dir / f"{experiment_name}_history.csv"
        summary_file = csv_dir / f"{experiment_name}_summary.csv"
        
        if history_file.exists():
            history_df = pd.read_csv(history_file)
            self._create_cooperation_evolution_plot(history_df, figures_dir / "cooperation_evolution.png")
            self._create_agent_performance_plot(history_df, figures_dir / "agent_performance.png")
        
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            self._create_score_distribution_plot(summary_df, figures_dir / "score_distribution.png")
    
    def _create_cooperation_evolution_plot(self, df: pd.DataFrame, output_path: Path):
        """Create cooperation evolution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by round and calculate mean cooperation
        if 'round' in df.columns and 'cooperated' in df.columns:
            coop_by_round = df.groupby('round')['cooperated'].mean()
            
            ax.plot(coop_by_round.index, coop_by_round.values, 'b-', linewidth=2)
            ax.fill_between(coop_by_round.index, 0, coop_by_round.values, alpha=0.3)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate')
            ax.set_title('Cooperation Evolution Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_agent_performance_plot(self, df: pd.DataFrame, output_path: Path):
        """Create agent performance plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        if 'agent_id' in df.columns:
            # Cooperation rate by agent
            agent_coop = df.groupby('agent_id')['cooperated'].mean()
            
            ax1.bar(agent_coop.index, agent_coop.values, alpha=0.7)
            ax1.set_xlabel('Agent ID')
            ax1.set_ylabel('Cooperation Rate')
            ax1.set_title('Agent Cooperation Rates')
            ax1.grid(True, alpha=0.3)
            
            # Score evolution by agent
            for agent_id in df['agent_id'].unique():
                agent_data = df[df['agent_id'] == agent_id]
                if 'round' in agent_data.columns and 'score' in agent_data.columns:
                    rounds = agent_data['round']
                    scores = agent_data.groupby('round')['score'].sum().cumsum()
                    ax2.plot(scores.index, scores.values, label=f'Agent {agent_id}', linewidth=2)
            
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Cumulative Score')
            ax2.set_title('Agent Score Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_score_distribution_plot(self, df: pd.DataFrame, output_path: Path):
        """Create score distribution plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if 'final_score' in df.columns:
            scores = df['final_score']
            
            ax.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(scores.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {scores.mean():.2f}')
            ax.axvline(scores.median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {scores.median():.2f}')
            
            ax.set_xlabel('Final Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Final Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_figures_from_json(self, json_file: Path, figures_dir: Path):
        """Create figures from JSON results."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create cooperation heatmap
        self._create_cooperation_heatmap(data, figures_dir / "cooperation_heatmap.png")
    
    def _create_cooperation_heatmap(self, data: Dict, output_path: Path):
        """Create cooperation heatmap between agents."""
        if 'history' not in data:
            return
        
        history = data['history']
        if not history:
            return
        
        # Extract number of agents
        num_agents = len(data.get('final_scores', {}))
        if num_agents == 0:
            return
        
        # Create cooperation matrix
        coop_matrix = np.zeros((num_agents, num_agents))
        interaction_counts = np.zeros((num_agents, num_agents))
        
        for round_data in history:
            for interaction in round_data.get('interactions', []):
                agent1 = interaction['agent1_id']
                agent2 = interaction['agent2_id']
                
                if interaction['agent1_action'] == 'C':
                    coop_matrix[agent1][agent2] += 1
                if interaction['agent2_action'] == 'C':
                    coop_matrix[agent2][agent1] += 1
                
                interaction_counts[agent1][agent2] += 1
                interaction_counts[agent2][agent1] += 1
        
        # Normalize by interaction count
        with np.errstate(divide='ignore', invalid='ignore'):
            coop_rate_matrix = np.where(interaction_counts > 0, 
                                      coop_matrix / interaction_counts, 
                                      0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(coop_rate_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cooperation Rate', rotation=270, labelpad=15)
        
        # Set ticks
        ax.set_xticks(np.arange(num_agents))
        ax.set_yticks(np.arange(num_agents))
        ax.set_xticklabels([f'Agent {i}' for i in range(num_agents)])
        ax.set_yticklabels([f'Agent {i}' for i in range(num_agents)])
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(num_agents):
            for j in range(num_agents):
                if interaction_counts[i][j] > 0:
                    text = ax.text(j, i, f'{coop_rate_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Agent Cooperation Heatmap")
        ax.set_xlabel("Target Agent")
        ax.set_ylabel("Source Agent")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_all_experiments(self):
        """Process all experiments in the results directory."""
        # Find all timestamped directories
        timestamp_pattern = r'\d{8}_\d{6}'
        
        processed = 0
        for item in self.base_results_dir.iterdir():
            if item.is_dir() and len(item.name) == 15 and '_' in item.name:
                # Check if it matches timestamp pattern
                try:
                    # Verify it's a valid timestamp
                    datetime.strptime(item.name, '%Y%m%d_%H%M%S')
                    
                    print(f"Processing experiment: {item.name}")
                    self.create_experiment_figures(item)
                    processed += 1
                except ValueError:
                    # Not a timestamp directory
                    continue
        
        # Also process batch directories
        for batch_dir in self.base_results_dir.glob("batch_*"):
            if batch_dir.is_dir():
                print(f"Processing batch: {batch_dir.name}")
                for exp_dir in batch_dir.iterdir():
                    if exp_dir.is_dir():
                        print(f"  - {exp_dir.name}")
                        self.create_experiment_figures(exp_dir)
                        processed += 1
        
        print(f"\nProcessed {processed} experiments")
        return processed


def main():
    """Main function to process experiment visualizations."""
    import sys
    
    # Check dependencies
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Error: matplotlib is required")
        sys.exit(1)
    
    try:
        import pandas
    except ImportError:
        print("Error: pandas is required")
        sys.exit(1)
    
    # Process experiments
    visualizer = ExperimentVisualizer()
    processed = visualizer.process_all_experiments()
    
    if processed == 0:
        print("\nNo experiments found to process.")
        print("Make sure experiments have been run and results are in the 'results' directory.")


if __name__ == "__main__":
    main()