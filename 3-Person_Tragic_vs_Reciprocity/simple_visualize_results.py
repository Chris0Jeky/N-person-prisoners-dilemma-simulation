#!/usr/bin/env python3
"""
Simple visualization of research results using only matplotlib and standard libraries.
"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SimpleResearchVisualizer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
    def read_csv(self, filename):
        """Read CSV file and return data as dict of lists."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        data = {}
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    if key not in data:
                        data[key] = []
                    try:
                        # Try to convert to float
                        data[key].append(float(value))
                    except ValueError:
                        # Keep as string if not a number
                        data[key].append(value)
        return data
    
    def create_tft_cooperation_figures(self):
        """Create TFT cooperation figures."""
        configs = [
            ("3_TFT", "3 TFT"),
            ("2_TFT-E__plus__1_AllD", "2 TFT-E + 1 AllD"),
            ("2_TFT__plus__1_AllD", "2 TFT + 1 AllD"),
            ("2_TFT-E__plus__1_AllC", "2 TFT-E + 1 AllC")
        ]
        
        # Figure 1: Pairwise
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (config_name, display_name) in enumerate(configs):
            ax = axes[idx]
            data = self.read_csv(f"pairwise_tft_cooperation_{config_name}_aggregated.csv")
            
            if data and 'Round' in data and 'Mean_Cooperation_Rate' in data:
                rounds = data['Round']
                mean_coop = data['Mean_Cooperation_Rate']
                lower_ci = data.get('Lower_95_CI', mean_coop)
                upper_ci = data.get('Upper_95_CI', mean_coop)
                
                ax.fill_between(rounds, lower_ci, upper_ci, alpha=0.3, color='blue', label='95% CI')
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
        plt.savefig(self.figures_dir / 'figure1_pairwise_tft_cooperation_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Neighbourhood
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (config_name, display_name) in enumerate(configs):
            ax = axes[idx]
            data = self.read_csv(f"neighbourhood_tft_cooperation_{config_name}_aggregated.csv")
            
            if data and 'Round' in data and 'Mean_Cooperation_Rate' in data:
                rounds = data['Round']
                mean_coop = data['Mean_Cooperation_Rate']
                lower_ci = data.get('Lower_95_CI', mean_coop)
                upper_ci = data.get('Upper_95_CI', mean_coop)
                
                ax.fill_between(rounds, lower_ci, upper_ci, alpha=0.3, color='blue', label='95% CI')
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
        plt.savefig(self.figures_dir / 'figure2_neighbourhood_tft_cooperation_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created fixed TFT cooperation figures")
    
    def create_comparison_figure(self):
        """Create tragic valley vs reciprocity hill comparison."""
        # Load comparative analysis
        comp_path = self.results_dir / "comparative_analysis.json"
        if not comp_path.exists():
            print("Warning: comparative_analysis.json not found")
            return
            
        with open(comp_path, 'r') as f:
            comp_data = json.load(f)
        
        tft_data = comp_data.get('tft_analysis', {})
        
        configs = ['3_TFT', '2_TFT-E__plus__1_AllD', '2_TFT__plus__1_AllD', '2_TFT-E__plus__1_AllC']
        display_names = ['3 TFT', '2 TFT-E\n+ 1 AllD', '2 TFT\n+ 1 AllD', '2 TFT-E\n+ 1 AllC']
        
        pairwise_rates = []
        neighbourhood_rates = []
        
        for config in configs:
            p_rate = tft_data.get('pairwise', {}).get(config, {}).get('mean_final_cooperation', 0)
            n_rate = tft_data.get('neighbourhood', {}).get(config, {}).get('mean_final_cooperation', 0)
            pairwise_rates.append(p_rate)
            neighbourhood_rates.append(n_rate)
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(configs))
        width = 0.35
        
        # Bar plot
        bars1 = ax1.bar(x - width/2, pairwise_rates, width, label='Pairwise', color='#ff7f0e', alpha=0.8)
        bars2 = ax1.bar(x + width/2, neighbourhood_rates, width, label='Neighbourhood', color='#2ca02c', alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Final Cooperation Rate')
        ax1.set_title('Tragic Valley vs Reciprocity Hill')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Difference plot
        differences = [n - p for n, p in zip(neighbourhood_rates, pairwise_rates)]
        colors = ['#2ca02c' if d > 0 else '#ff7f0e' for d in differences]
        
        bars3 = ax2.bar(x, differences, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Cooperation Difference\n(Neighbourhood - Pairwise)')
        ax2.set_title('Reciprocity Hill Effect')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, diff in zip(bars3, differences):
            ax2.annotate(f'{diff:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, diff),
                       xytext=(0, 3 if diff > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if diff > 0 else 'top', fontsize=9)
        
        plt.suptitle('Figure 3: Tragic Valley vs Reciprocity Hill Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure3_tragic_valley_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created tragic valley comparison figure")
    
    def create_summary_plots(self):
        """Create summary cooperation plots."""
        # Load summary CSVs
        pairwise_summary = self.read_csv("pairwise_tft_cooperation_all_experiments_summary.csv")
        neighbourhood_summary = self.read_csv("neighbourhood_tft_cooperation_all_experiments_summary.csv")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pairwise summary
        if pairwise_summary and 'Round' in pairwise_summary:
            rounds = pairwise_summary['Round']
            
            for col_name in pairwise_summary:
                if col_name.endswith('_mean') and col_name != 'Round':
                    exp_name = col_name.replace('_mean', '')
                    mean_vals = pairwise_summary[col_name]
                    ax1.plot(rounds, mean_vals, linewidth=2, label=exp_name)
            
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Cooperation Rate')
            ax1.set_title('Pairwise Game Cooperation')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
        
        # Neighbourhood summary
        if neighbourhood_summary and 'Round' in neighbourhood_summary:
            rounds = neighbourhood_summary['Round']
            
            for col_name in neighbourhood_summary:
                if col_name.endswith('_mean') and col_name != 'Round':
                    exp_name = col_name.replace('_mean', '')
                    mean_vals = neighbourhood_summary[col_name]
                    ax2.plot(rounds, mean_vals, linewidth=2, label=exp_name)
            
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Cooperation Rate')
            ax2.set_title('Neighbourhood Game Cooperation')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
        
        plt.suptitle('Figure 4: Cooperation Summary - All Experiments', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure4_cooperation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created cooperation summary plots")
    
    def run(self):
        """Generate all visualizations."""
        print(f"Generating visualizations in {self.figures_dir}")
        
        self.create_tft_cooperation_figures()
        self.create_comparison_figure()
        self.create_summary_plots()
        
        print(f"\nAll figures generated in: {self.figures_dir}")
        
        # List generated figures
        figures = list(self.figures_dir.glob("*.png"))
        if figures:
            print("\nGenerated figures:")
            for fig in sorted(figures):
                size_kb = fig.stat().st_size / 1024
                print(f"  - {fig.name} ({size_kb:.1f} KB)")


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Error: matplotlib is required")
        sys.exit(1)
    
    visualizer = SimpleResearchVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()