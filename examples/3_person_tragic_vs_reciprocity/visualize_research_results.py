#!/usr/bin/env python3
"""
Visualize research experiment results from CSV files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ResearchResultsVisualizer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_csv_data(self, filename):
        """Load data from CSV file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
        return pd.read_csv(filepath)
    
    def create_tft_cooperation_figures(self):
        """Create figures 1 and 2 for TFT cooperation (pairwise and neighbourhood)."""
        
        # Figure 1: Pairwise TFT Cooperation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        configs = [
            ("3_TFT", "3 TFT"),
            ("2_TFT-E__plus__1_AllD", "2 TFT-E + 1 AllD"),
            ("2_TFT__plus__1_AllD", "2 TFT + 1 AllD"),
            ("2_TFT-E__plus__1_AllC", "2 TFT-E + 1 AllC")
        ]
        
        for idx, (config_name, display_name) in enumerate(configs):
            ax = axes[idx]
            
            # Load aggregated data
            df = self.load_csv_data(f"pairwise_tft_cooperation_{config_name}_aggregated.csv")
            
            if df is not None and not df.empty:
                rounds = df['Round']
                mean_coop = df['Mean_Cooperation_Rate']
                lower_ci = df['Lower_95_CI']
                upper_ci = df['Upper_95_CI']
                
                # Plot confidence interval
                ax.fill_between(rounds, lower_ci, upper_ci, alpha=0.3, color='blue', label='95% CI')
                
                # Plot mean
                ax.plot(rounds, mean_coop, 'b-', linewidth=2, label='Mean')
                
                ax.set_xlabel('Round')
                ax.set_ylabel('Cooperation Rate')
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                
                if idx == 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"pairwise_{display_name}")
        
        plt.suptitle('Figure 1: Cooperation of TFT Agents Over Time (Pairwise)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure1_pairwise_tft_cooperation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Neighbourhood TFT Cooperation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (config_name, display_name) in enumerate(configs):
            ax = axes[idx]
            
            # Load aggregated data
            df = self.load_csv_data(f"neighbourhood_tft_cooperation_{config_name}_aggregated.csv")
            
            if df is not None and not df.empty:
                rounds = df['Round']
                mean_coop = df['Mean_Cooperation_Rate']
                lower_ci = df['Lower_95_CI']
                upper_ci = df['Upper_95_CI']
                
                # Plot confidence interval
                ax.fill_between(rounds, lower_ci, upper_ci, alpha=0.3, color='blue', label='95% CI')
                
                # Plot mean
                ax.plot(rounds, mean_coop, 'b-', linewidth=2, label='Mean')
                
                ax.set_xlabel('Round')
                ax.set_ylabel('Cooperation Rate')
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
                
                if idx == 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"neighbourhood_{display_name}")
        
        plt.suptitle('Figure 2: Cooperation of TFT Agents Over Time (Neighbourhood)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure2_neighbourhood_tft_cooperation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created figures 1 and 2: TFT cooperation plots")
    
    def create_ql_cooperation_figures(self):
        """Create figures 3 and 4 for QLearning agent scores."""
        
        # Load comparative analysis for QL data
        comp_analysis_path = self.results_dir / "comparative_analysis.json"
        if not comp_analysis_path.exists():
            print("Warning: comparative_analysis.json not found")
            return
            
        with open(comp_analysis_path, 'r') as f:
            comp_data = json.load(f)
        
        ql_data = comp_data.get('ql_analysis', {}).get('2QL', {})
        
        # Figure 3: Pairwise QL Agent Scores
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        configs = list(ql_data.keys())[:4]
        
        for idx, config in enumerate(configs):
            ax = axes[idx]
            
            if config in ql_data and 'final_scores' in ql_data[config]:
                scores_data = ql_data[config]['final_scores']
                
                agents = list(scores_data.keys())
                means = [scores_data[agent]['mean_score'] for agent in agents]
                stds = [scores_data[agent]['std_score'] for agent in agents]
                
                x = np.arange(len(agents))
                ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(agents, rotation=45)
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(config)
        
        plt.suptitle('Figure 3: QLearning Agent Final Scores', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure3_ql_agent_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created figure 3: QL agent scores")
    
    def create_cooperation_comparison_figures(self):
        """Create figures 5 and 6 for overall cooperation comparison."""
        
        # Figure 5: All Pairwise Cooperation
        summary_df = self.load_csv_data("pairwise_tft_cooperation_all_experiments_summary.csv")
        
        if summary_df is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each experiment
            for col in summary_df.columns:
                if col != 'Round' and '_mean' in col:
                    exp_name = col.replace('_mean', '')
                    mean_values = summary_df[col]
                    std_col = f"{exp_name}_std"
                    
                    if std_col in summary_df.columns:
                        std_values = summary_df[std_col]
                        ax.plot(summary_df['Round'], mean_values, label=exp_name, linewidth=2)
                        ax.fill_between(summary_df['Round'], 
                                      mean_values - std_values, 
                                      mean_values + std_values, 
                                      alpha=0.2)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate')
            ax.set_title('Figure 5: Overall Cooperation Over Time (Pairwise)', fontsize=16)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'figure5_pairwise_all_cooperation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 6: All Neighbourhood Cooperation
        summary_df = self.load_csv_data("neighbourhood_tft_cooperation_all_experiments_summary.csv")
        
        if summary_df is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each experiment
            for col in summary_df.columns:
                if col != 'Round' and '_mean' in col:
                    exp_name = col.replace('_mean', '')
                    mean_values = summary_df[col]
                    std_col = f"{exp_name}_std"
                    
                    if std_col in summary_df.columns:
                        std_values = summary_df[std_col]
                        ax.plot(summary_df['Round'], mean_values, label=exp_name, linewidth=2)
                        ax.fill_between(summary_df['Round'], 
                                      mean_values - std_values, 
                                      mean_values + std_values, 
                                      alpha=0.2)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate')
            ax.set_title('Figure 6: Overall Cooperation Over Time (Neighbourhood)', fontsize=16)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'figure6_neighbourhood_all_cooperation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Created figures 5 and 6: Overall cooperation comparison")
    
    def create_tragic_valley_visualization(self):
        """Create a figure showing the tragic valley vs reciprocity hill effect."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load final cooperation rates from comparative analysis
        comp_analysis_path = self.results_dir / "comparative_analysis.json"
        if comp_analysis_path.exists():
            with open(comp_analysis_path, 'r') as f:
                comp_data = json.load(f)
            
            tft_data = comp_data.get('tft_analysis', {})
            
            configs = ['3_TFT', '2_TFT-E__plus__1_AllD', '2_TFT__plus__1_AllD', '2_TFT-E__plus__1_AllC']
            display_names = ['3 TFT', '2 TFT-E + 1 AllD', '2 TFT + 1 AllD', '2 TFT-E + 1 AllC']
            
            pairwise_rates = []
            neighbourhood_rates = []
            
            for config in configs:
                pairwise_rate = tft_data.get('pairwise', {}).get(config, {}).get('mean_final_cooperation', 0)
                neighbourhood_rate = tft_data.get('neighbourhood', {}).get(config, {}).get('mean_final_cooperation', 0)
                pairwise_rates.append(pairwise_rate)
                neighbourhood_rates.append(neighbourhood_rate)
            
            x = np.arange(len(configs))
            width = 0.35
            
            # Bar plot comparison
            bars1 = ax1.bar(x - width/2, pairwise_rates, width, label='Pairwise', alpha=0.8)
            bars2 = ax1.bar(x + width/2, neighbourhood_rates, width, label='Neighbourhood', alpha=0.8)
            
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Final Cooperation Rate')
            ax1.set_title('Tragic Valley vs Reciprocity Hill')
            ax1.set_xticks(x)
            ax1.set_xticklabels(display_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            # Difference plot
            differences = [n - p for n, p in zip(neighbourhood_rates, pairwise_rates)]
            colors = ['green' if d > 0 else 'red' for d in differences]
            
            bars3 = ax2.bar(x, differences, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Cooperation Difference (Neighbourhood - Pairwise)')
            ax2.set_title('Reciprocity Hill Effect')
            ax2.set_xticks(x)
            ax2.set_xticklabels(display_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars3:
                height = bar.get_height()
                ax2.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height > 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.suptitle('Figure 7: Tragic Valley vs Reciprocity Hill Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure7_tragic_valley_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created figure 7: Tragic valley vs reciprocity hill analysis")
    
    def run(self):
        """Generate all visualizations."""
        print(f"Generating visualizations in {self.figures_dir}")
        
        # Create all figures
        self.create_tft_cooperation_figures()
        self.create_ql_cooperation_figures()
        self.create_cooperation_comparison_figures()
        self.create_tragic_valley_visualization()
        
        print(f"\nAll figures generated in: {self.figures_dir}")
        
        # List generated figures
        figures = list(self.figures_dir.glob("*.png"))
        if figures:
            print("\nGenerated figures:")
            for fig in sorted(figures):
                print(f"  - {fig.name}")
        else:
            print("\nNo figures were generated.")


def main():
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    # Check if pandas is available
    try:
        import pandas
    except ImportError:
        print("Error: pandas is required for data processing")
        print("Install with: pip install pandas")
        sys.exit(1)
    
    # Run visualizer
    visualizer = ResearchResultsVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()