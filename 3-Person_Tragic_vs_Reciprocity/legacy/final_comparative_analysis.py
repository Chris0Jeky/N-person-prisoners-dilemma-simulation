#!/usr/bin/env python3
"""
Final Comparative Analysis - Compares Basic QL vs Final Enhanced QL
Works with results from final_demo_full.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_summary_data(base_dir, ql_type):
    """Load summary CSV files for a given Q-learning type."""
    summary_files = {
        'pairwise_coop': f'{base_dir}/{ql_type}_experiments/csv/{ql_type}_pairwise_cooperation_summary.csv',
        'pairwise_scores': f'{base_dir}/{ql_type}_experiments/csv/{ql_type}_pairwise_scores_summary.csv',
        'nperson_coop': f'{base_dir}/{ql_type}_experiments/csv/{ql_type}_nperson_cooperation_summary.csv',
        'nperson_scores': f'{base_dir}/{ql_type}_experiments/csv/{ql_type}_nperson_scores_summary.csv'
    }
    
    data = {}
    for key, filepath in summary_files.items():
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
        else:
            print(f"Warning: File not found - {filepath}")
            data[key] = None
    
    return data


def load_detailed_data(base_dir, ql_type, experiment_name, game_mode):
    """Load detailed CSV file for a specific experiment."""
    clean_name = experiment_name.replace(' ', '_').replace('+', 'plus')
    filepath = f'{base_dir}/{ql_type}_experiments/csv/{ql_type}_{game_mode}_{clean_name}.csv'
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"Warning: File not found - {filepath}")
        return None


def get_ql_agent_columns(df, ql_type):
    """Get the QL agent column names from a dataframe."""
    if ql_type == '1QL':
        return ['QL_1']
    elif ql_type == '2QL':
        return ['QL_1', 'QL_2']
    elif ql_type == '1EQL':
        return ['EQL_1']
    elif ql_type == '2EQL':
        return ['EQL_1', 'EQL_2']
    return []


def create_comparison_plots_by_scenario(ql_data, eql_data, scenario_type="1 QL", results_dir="qlearning_results", eql_results_dir="final_enhanced_qlearning_results"):
    """Create comparison plots for QL vs Enhanced QL by scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{scenario_type} Comparison: Basic QL vs Final Enhanced QL', fontsize=16, weight='bold')
    
    # Define plot configurations
    plot_configs = [
        ('pairwise_coop', 'Pairwise Cooperation', axes[0, 0]),
        ('pairwise_scores', 'Pairwise Scores', axes[0, 1]),
        ('nperson_coop', 'Neighbourhood Cooperation', axes[1, 0]),
        ('nperson_scores', 'Neighbourhood Scores', axes[1, 1])
    ]
    
    for data_key, title, ax in plot_configs:
        if ql_data.get(data_key) is not None and eql_data.get(data_key) is not None:
            # Get the correct column name based on data type
            value_col = 'Avg_Cooperation' if 'coop' in data_key else 'Final_Score'
            
            # Filter data for the scenario type
            ql_scenario = scenario_type
            eql_scenario = scenario_type.replace('QL', 'EQL')
            
            ql_filtered = ql_data[data_key][ql_data[data_key]['Experiment'].str.contains(ql_scenario, regex=False)]
            eql_filtered = eql_data[data_key][eql_data[data_key]['Experiment'].str.contains(eql_scenario, regex=False)]
            
            # Get QL agent data only
            ql_agents = []
            eql_agents = []
            
            for _, row in ql_filtered.iterrows():
                if row['Agent'] in get_ql_agent_columns(None, scenario_type.split()[0]):
                    ql_agents.append(row)
            
            for _, row in eql_filtered.iterrows():
                if row['Agent'] in get_ql_agent_columns(None, scenario_type.split()[0].replace('QL', 'EQL')):
                    eql_agents.append(row)
            
            if ql_agents and eql_agents:
                ql_df = pd.DataFrame(ql_agents)
                eql_df = pd.DataFrame(eql_agents)
                
                # Extract experiment names for x-axis
                experiments = ql_df['Experiment'].str.replace(ql_scenario, '').str.strip()
                x_pos = np.arange(len(experiments))
                
                # Calculate means for bar plots
                ql_means = ql_df.groupby('Experiment')[value_col].mean().values
                eql_means = eql_df.groupby('Experiment')[value_col].mean().values
                
                # Create grouped bar plot
                width = 0.35
                ax.bar(x_pos - width/2, ql_means, width, label='Basic QL', alpha=0.8, color='blue')
                ax.bar(x_pos + width/2, eql_means, width, label='Final Enhanced QL', alpha=0.8, color='darkblue')
                
                ax.set_xlabel('Experiment Configuration')
                ax.set_ylabel(value_col.replace('_', ' '))
                ax.set_title(title)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(experiments, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_detailed_comparison(experiment_name, ql_type='1QL', results_dir="qlearning_results", eql_results_dir="final_enhanced_qlearning_results"):
    """Create detailed time-series comparison for a specific experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    experiment_display = experiment_name.replace('_', ' ').replace('plus', '+')
    fig.suptitle(f'Detailed Comparison: {experiment_display}', fontsize=16, weight='bold')
    
    # Load data for both QL types
    game_modes = ['pairwise_cooperation', 'pairwise_scores', 'nperson_cooperation', 'nperson_scores']
    titles = ['Pairwise Cooperation', 'Pairwise Scores', 'Neighbourhood Cooperation', 'Neighbourhood Scores']
    
    eql_type = ql_type.replace('QL', 'EQL')
    eql_experiment = experiment_name.replace('QL', 'EQL')
    
    for idx, (game_mode, title, ax) in enumerate(zip(game_modes, titles, axes.flatten())):
        # Load data
        ql_data = load_detailed_data(results_dir, ql_type, experiment_name, game_mode)
        eql_data = load_detailed_data(eql_results_dir, eql_type, eql_experiment, game_mode)
        
        if ql_data is not None and eql_data is not None:
            # Get agent columns
            ql_cols = get_ql_agent_columns(ql_data, ql_type)
            eql_cols = get_ql_agent_columns(eql_data, eql_type)
            
            # Plot mean values for each QL type
            for col in ql_cols:
                if f'{col}_mean' in ql_data.columns:
                    ax.plot(ql_data['Round'], ql_data[f'{col}_mean'], 
                           label=f'Basic {col}', linewidth=2, color='blue')
            
            for col in eql_cols:
                if f'{col}_mean' in eql_data.columns:
                    ax.plot(eql_data['Round'], eql_data[f'{col}_mean'], 
                           label=f'Final Enhanced {col}', linewidth=2, linestyle='--', color='darkblue')
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Cooperation Rate' if 'cooperation' in game_mode else 'Cumulative Score')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_performance_heatmap(results_dir="qlearning_results", eql_results_dir="final_enhanced_qlearning_results"):
    """Create a heatmap showing performance differences between QL and Enhanced QL."""
    # Load all summary data
    ql_data_1 = load_summary_data(results_dir, '1QL')
    ql_data_2 = load_summary_data(results_dir, '2QL')
    eql_data_1 = load_summary_data(eql_results_dir, '1EQL')
    eql_data_2 = load_summary_data(eql_results_dir, '2EQL')
    
    # Prepare data for heatmap
    scenarios = []
    metrics = ['Pairwise Coop', 'Pairwise Score', 'NPerson Coop', 'NPerson Score']
    
    # Process 1QL scenarios
    if ql_data_1['pairwise_coop'] is not None and eql_data_1['pairwise_coop'] is not None:
        for exp in ql_data_1['pairwise_coop']['Experiment'].unique():
            if '1 QL' in exp:
                scenarios.append(exp.replace('1 QL', '1QL'))
    
    # Process 2QL scenarios
    if ql_data_2['pairwise_coop'] is not None and eql_data_2['pairwise_coop'] is not None:
        for exp in ql_data_2['pairwise_coop']['Experiment'].unique():
            if '2 QL' in exp:
                scenarios.append(exp.replace('2 QL', '2QL'))
    
    # Create difference matrix
    diff_matrix = np.zeros((len(scenarios), len(metrics)))
    
    for i, scenario in enumerate(scenarios):
        # Determine QL type
        ql_type = '1QL' if '1QL' in scenario else '2QL'
        eql_type = ql_type.replace('QL', 'EQL')
        
        # Get the correct data
        ql_data = ql_data_1 if ql_type == '1QL' else ql_data_2
        eql_data = eql_data_1 if eql_type == '1EQL' else eql_data_2
        
        # Adjust scenario name for lookup
        lookup_scenario = scenario.replace('1QL', '1 QL').replace('2QL', '2 QL')
        lookup_scenario_eql = lookup_scenario.replace('QL', 'EQL')
        
        # Calculate differences for each metric
        for j, (data_key, metric) in enumerate(zip(['pairwise_coop', 'pairwise_scores', 
                                                    'nperson_coop', 'nperson_scores'], metrics)):
            if ql_data[data_key] is not None and eql_data[data_key] is not None:
                # Get QL agents only
                value_col = 'Avg_Cooperation' if 'coop' in data_key else 'Final_Score'
                
                ql_agents = get_ql_agent_columns(None, ql_type)
                eql_agents = get_ql_agent_columns(None, eql_type)
                
                ql_value = ql_data[data_key][
                    (ql_data[data_key]['Experiment'] == lookup_scenario) & 
                    (ql_data[data_key]['Agent'].isin(ql_agents))
                ][value_col].mean()
                
                eql_value = eql_data[data_key][
                    (eql_data[data_key]['Experiment'] == lookup_scenario_eql) & 
                    (eql_data[data_key]['Agent'].isin(eql_agents))
                ][value_col].mean()
                
                # Calculate percentage difference
                if not np.isnan(ql_value) and ql_value != 0:
                    diff_matrix[i, j] = ((eql_value - ql_value) / abs(ql_value)) * 100
                else:
                    diff_matrix[i, j] = 0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create the heatmap
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(scenarios)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% Difference (Enhanced - Basic)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{diff_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black" if abs(diff_matrix[i, j]) < 25 else "white")
    
    ax.set_title('Performance Difference: Final Enhanced QL vs Basic QL\n(Positive = Enhanced Better)')
    plt.tight_layout()
    
    return fig


def main():
    """Main function to run all comparative analyses."""
    # Check which results are available
    basic_results_dir = 'qlearning_results'
    enhanced_results_dir = 'final_enhanced_qlearning_results'
    
    has_basic = os.path.exists(basic_results_dir)
    has_enhanced = os.path.exists(enhanced_results_dir)
    
    if not has_basic or not has_enhanced:
        print("ERROR: Results directories not found!")
        print(f"Basic QL results ({basic_results_dir}): {'Found' if has_basic else 'Not found'}")
        print(f"Enhanced QL results ({enhanced_results_dir}): {'Found' if has_enhanced else 'Not found'}")
        print("\nPlease run final_demo_full.py with:")
        print("  use_enhanced = True  (for enhanced results)")
        print("  use_enhanced = False (for basic results)")
        return
    
    # Create output directory
    output_dir = 'final_comparative_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Final Q-Learning Comparative Analysis")
    print("=" * 50)
    print(f"Basic QL results from: {basic_results_dir}/")
    print(f"Enhanced QL results from: {enhanced_results_dir}/")
    
    # Load all data
    print("\nLoading data...")
    ql_data_1 = load_summary_data(basic_results_dir, '1QL')
    ql_data_2 = load_summary_data(basic_results_dir, '2QL')
    eql_data_1 = load_summary_data(enhanced_results_dir, '1EQL')
    eql_data_2 = load_summary_data(enhanced_results_dir, '2EQL')
    
    # Create comparison plots by scenario
    print("\nGenerating comparison plots...")
    
    # 1QL vs 1EQL comparison
    fig1 = create_comparison_plots_by_scenario(ql_data_1, eql_data_1, "1 QL", basic_results_dir, enhanced_results_dir)
    fig1.savefig(f'{output_dir}/1QL_vs_1EQL_comparison.png', dpi=300, bbox_inches='tight')
    print("  - Saved: 1QL_vs_1EQL_comparison.png")
    
    # 2QL vs 2EQL comparison
    fig2 = create_comparison_plots_by_scenario(ql_data_2, eql_data_2, "2 QL", basic_results_dir, enhanced_results_dir)
    fig2.savefig(f'{output_dir}/2QL_vs_2EQL_comparison.png', dpi=300, bbox_inches='tight')
    print("  - Saved: 2QL_vs_2EQL_comparison.png")
    
    # Create detailed comparisons for specific experiments
    print("\nGenerating detailed time-series comparisons...")
    
    detailed_experiments = [
        ('1 QL + 2 AllD', '1QL'),
        ('1 QL + 2 TFT', '1QL'),
        ('1 QL + 1 AllD + 1 AllC', '1QL'),
        ('2 QL + 1 AllD', '2QL'),
        ('2 QL + 1 TFT', '2QL'),
    ]
    
    for exp_name, ql_type in detailed_experiments:
        fig = create_detailed_comparison(exp_name, ql_type, basic_results_dir, enhanced_results_dir)
        clean_name = exp_name.replace(' ', '_').replace('+', 'plus')
        fig.savefig(f'{output_dir}/detailed_{clean_name}.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved: detailed_{clean_name}.png")
        plt.close(fig)
    
    # Create performance difference heatmap
    print("\nGenerating performance difference heatmap...")
    fig_heatmap = create_performance_heatmap(basic_results_dir, enhanced_results_dir)
    fig_heatmap.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print("  - Saved: performance_heatmap.png")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = []
    
    # Calculate overall statistics
    for ql_type, eql_type in [('1QL', '1EQL'), ('2QL', '2EQL')]:
        ql_data = ql_data_1 if ql_type == '1QL' else ql_data_2
        eql_data = eql_data_1 if eql_type == '1EQL' else eql_data_2
        
        for data_key in ['pairwise_coop', 'pairwise_scores', 'nperson_coop', 'nperson_scores']:
            if ql_data[data_key] is not None and eql_data[data_key] is not None:
                value_col = 'Avg_Cooperation' if 'coop' in data_key else 'Final_Score'
                
                # Get QL agents only
                ql_agents = get_ql_agent_columns(None, ql_type)
                eql_agents = get_ql_agent_columns(None, eql_type)
                
                ql_mean = ql_data[data_key][ql_data[data_key]['Agent'].isin(ql_agents)][value_col].mean()
                eql_mean = eql_data[data_key][eql_data[data_key]['Agent'].isin(eql_agents)][value_col].mean()
                
                improvement = ((eql_mean - ql_mean) / abs(ql_mean)) * 100 if ql_mean != 0 else 0
                
                summary_stats.append({
                    'Scenario': ql_type,
                    'Metric': data_key.replace('_', ' ').title(),
                    'Basic QL': f"{ql_mean:.3f}",
                    'Final Enhanced QL': f"{eql_mean:.3f}",
                    'Improvement': f"{improvement:+.1f}%"
                })
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
    print("  - Saved: summary_statistics.csv")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - FINAL ENHANCED Q-LEARNING")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    print(f"\nAll comparative analysis files saved to '{output_dir}/' directory")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()