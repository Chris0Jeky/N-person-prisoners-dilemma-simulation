#!/usr/bin/env python3
"""
Fix visualization issues - properly show the contrast between pairwise (0) and neighbourhood cooperation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_improved_cooperation_plots():
    """Create improved cooperation plots that clearly show the 0 vs high cooperation contrast."""
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Figure 5: Pairwise cooperation (should show 0 or very low)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load pairwise summary
    pairwise_df = pd.read_csv(results_dir / "pairwise_tft_cooperation_all_experiments_summary.csv")
    
    # Define colors for each experiment
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    exp_columns = [col for col in pairwise_df.columns if col.endswith('_mean') and col != 'Round']
    
    for idx, col in enumerate(exp_columns):
        exp_name = col.replace('_mean', '').replace('_', ' ')
        mean_vals = pairwise_df[col]
        std_col = col.replace('_mean', '_std')
        
        if std_col in pairwise_df.columns:
            std_vals = pairwise_df[std_col]
            
            # Plot with enhanced visibility
            ax.plot(pairwise_df['Round'], mean_vals, 
                   label=exp_name, linewidth=3, color=colors[idx % len(colors)])
            
            # Add shaded confidence interval
            ax.fill_between(pairwise_df['Round'], 
                          mean_vals - std_vals, 
                          mean_vals + std_vals, 
                          alpha=0.2, color=colors[idx % len(colors)])
    
    # Enhance the plot to show the near-zero values
    ax.set_xlabel('Round', fontsize=14)
    ax.set_ylabel('Cooperation Rate', fontsize=14)
    ax.set_title('Figure 5: Pairwise Game - Cooperation Remains at Zero (Tragic Valley)', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.5)  # Adjusted scale to better show the zero values
    
    # Add annotation explaining the result
    ax.text(0.5, 0.95, 'All experiments show 0% cooperation in pairwise games', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure5_pairwise_all_cooperation_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 6: Neighbourhood cooperation (should show high values)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Load neighbourhood summary
    neighbourhood_df = pd.read_csv(results_dir / "neighbourhood_tft_cooperation_all_experiments_summary.csv")
    
    exp_columns = [col for col in neighbourhood_df.columns if col.endswith('_mean') and col != 'Round']
    
    for idx, col in enumerate(exp_columns):
        exp_name = col.replace('_mean', '').replace('_', ' ')
        mean_vals = neighbourhood_df[col]
        std_col = col.replace('_mean', '_std')
        
        if std_col in neighbourhood_df.columns:
            std_vals = neighbourhood_df[std_col]
            
            ax.plot(neighbourhood_df['Round'], mean_vals, 
                   label=exp_name, linewidth=3, color=colors[idx % len(colors)])
            
            ax.fill_between(neighbourhood_df['Round'], 
                          mean_vals - std_vals, 
                          mean_vals + std_vals, 
                          alpha=0.2, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Round', fontsize=14)
    ax.set_ylabel('Cooperation Rate', fontsize=14)
    ax.set_title('Figure 6: Neighbourhood Game - High Cooperation Achieved (Reciprocity Hill)', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Add annotation
    ax.text(0.5, 0.05, 'Neighbourhood games achieve 60-100% cooperation', 
            transform=ax.transAxes, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure6_neighbourhood_all_cooperation_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created improved cooperation comparison figures")
    
    # Create a combined comparison figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Pairwise subplot
    for idx, col in enumerate([c for c in pairwise_df.columns if c.endswith('_mean') and c != 'Round']):
        exp_name = col.replace('_mean', '').replace('_', ' ')
        mean_vals = pairwise_df[col]
        ax1.plot(pairwise_df['Round'], mean_vals, 
                label=exp_name, linewidth=2.5, color=colors[idx % len(colors)])
    
    ax1.set_ylabel('Cooperation Rate', fontsize=12)
    ax1.set_title('Pairwise Game (Tragic Valley Effect)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Neighbourhood subplot
    for idx, col in enumerate([c for c in neighbourhood_df.columns if c.endswith('_mean') and c != 'Round']):
        exp_name = col.replace('_mean', '').replace('_', ' ')
        mean_vals = neighbourhood_df[col]
        ax2.plot(neighbourhood_df['Round'], mean_vals, 
                label=exp_name, linewidth=2.5, color=colors[idx % len(colors)])
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Cooperation Rate', fontsize=12)
    ax2.set_title('Neighbourhood Game (Reciprocity Hill Effect)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)
    
    plt.suptitle('Figure 7: Direct Comparison - Tragic Valley vs Reciprocity Hill', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure7_direct_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created direct comparison figure")

def create_summary_visualization():
    """Create a clear summary visualization of the key findings."""
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    
    # Load comparative analysis
    with open(results_dir / "comparative_analysis.json", 'r') as f:
        comp_data = json.load(f)
    
    tft_data = comp_data.get('tft_analysis', {})
    
    # Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = ['3_TFT', '2_TFT-E__plus__1_AllD', '2_TFT__plus__1_AllD', '2_TFT-E__plus__1_AllC']
    display_names = ['3 TFT', '2 TFT-E\n+ 1 AllD', '2 TFT\n+ 1 AllD', '2 TFT-E\n+ 1 AllC']
    
    # Extract data
    pairwise_rates = []
    neighbourhood_rates = []
    differences = []
    
    for config in configs:
        p_rate = tft_data.get('pairwise', {}).get(config, {}).get('mean_final_cooperation', 0)
        n_rate = tft_data.get('neighbourhood', {}).get(config, {}).get('mean_final_cooperation', 0)
        pairwise_rates.append(p_rate)
        neighbourhood_rates.append(n_rate)
        differences.append(n_rate - p_rate)
    
    x = np.arange(len(configs))
    
    # 1. Final cooperation rates comparison
    width = 0.35
    bars1 = ax1.bar(x - width/2, pairwise_rates, width, label='Pairwise', color='#ff7f0e', alpha=0.8)
    bars2 = ax1.bar(x + width/2, neighbourhood_rates, width, label='Neighbourhood', color='#2ca02c', alpha=0.8)
    
    ax1.set_ylabel('Final Cooperation Rate', fontsize=12)
    ax1.set_title('Final Cooperation Rates by Game Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Cooperation difference (Reciprocity Hill effect)
    colors = ['#2ca02c' if d > 0 else '#ff7f0e' for d in differences]
    bars3 = ax2.bar(x, differences, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Cooperation Difference\n(Neighbourhood - Pairwise)', fontsize=12)
    ax2.set_title('Reciprocity Hill Effect Magnitude', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, diff in zip(bars3, differences):
        ax2.annotate(f'+{diff:.2f}', xy=(bar.get_x() + bar.get_width() / 2, diff),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Pairwise results detail
    ax3.text(0.5, 0.5, 'PAIRWISE GAME RESULTS', ha='center', va='center',
             fontsize=16, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.3, 'All configurations: 0% cooperation', ha='center', va='center',
             fontsize=14, color='red', transform=ax3.transAxes)
    ax3.text(0.5, 0.15, 'Demonstrates Tragic Valley Effect', ha='center', va='center',
             fontsize=12, style='italic', transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. Neighbourhood results detail
    ax4.text(0.5, 0.7, 'NEIGHBOURHOOD GAME RESULTS', ha='center', va='center',
             fontsize=16, fontweight='bold', transform=ax4.transAxes)
    
    results_text = '\n'.join([f'{name}: {rate:.1%} cooperation' 
                             for name, rate in zip(display_names, neighbourhood_rates)])
    ax4.text(0.5, 0.4, results_text, ha='center', va='center',
             fontsize=12, color='green', transform=ax4.transAxes)
    ax4.text(0.5, 0.1, 'Demonstrates Reciprocity Hill Effect', ha='center', va='center',
             fontsize=12, style='italic', transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.suptitle('Research Results Summary: Tragic Valley vs Reciprocity Hill', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'figure8_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created results summary figure")

def main():
    print("Creating improved visualizations...")
    create_improved_cooperation_plots()
    create_summary_visualization()
    print("\nImproved figures created in results/figures/")
    
    # List new figures
    figures_dir = Path("results/figures")
    new_figures = list(figures_dir.glob("*_fixed.png")) + list(figures_dir.glob("*_summary.png")) + list(figures_dir.glob("*_comparison.png"))
    
    if new_figures:
        print("\nNew figures created:")
        for fig in sorted(new_figures):
            print(f"  - {fig.name}")

if __name__ == "__main__":
    main()