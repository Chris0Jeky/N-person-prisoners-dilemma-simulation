#!/usr/bin/env python3
"""
Q-Learning Comparative Analysis
Compares performance between basic and enhanced Q-learning implementations
Generates visualization plots and statistical summaries
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


def load_detailed_data(ql_type, experiment_name, game_mode):
    """Load detailed CSV file for a specific experiment."""
    # For enhanced QL, we need to adjust the experiment name
    if 'EQL' in ql_type:
        # Replace "QL" with "EQL" in the experiment name
        adjusted_experiment_name = experiment_name.replace('QL', 'EQL')
        clean_name = adjusted_experiment_name.replace(' ', '_').replace('+', 'plus')
        results_dir = 'enhanced_qlearning_results'
    else:
        clean_name = experiment_name.replace(' ', '_').replace('+', 'plus')
        results_dir = 'qlearning_results'
    
    filepath = f'{results_dir}/{ql_type}_experiments/csv/{ql_type}_{game_mode}_{clean_name}.csv'
    
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


def create_comparison_plots_by_scenario(ql_data, eql_data, scenario_type="1 QL"):
    """Create comparison plots for QL vs Enhanced QL by scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{scenario_type} Comparison: Basic QL vs Enhanced QL', fontsize=16, weight='bold')
    
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
            ql_agent_data = ql_filtered[ql_filtered['Agent'].str.contains('QL')]
            eql_agent_data = eql_filtered[eql_filtered['Agent'].str.contains('EQL')]
            
            # Group by experiment and calculate mean across QL agents
            ql_exp_means = {}
            eql_exp_means = {}
            
            for exp in ql_filtered['Experiment'].unique():
                exp_data = ql_agent_data[ql_agent_data['Experiment'] == exp]
                if len(exp_data) > 0:
                    ql_exp_means[exp] = exp_data[value_col].mean()
            
            for exp in eql_filtered['Experiment'].unique():
                exp_data = eql_agent_data[eql_agent_data['Experiment'] == exp]
                if len(exp_data) > 0:
                    eql_exp_means[exp] = exp_data[value_col].mean()
            
            # Align experiments
            aligned_experiments = []
            ql_values = []
            eql_values = []
            
            for ql_exp in sorted(ql_exp_means.keys()):
                # Extract opponent configuration
                opponent_config = ql_exp.split(' + ', 1)[1] if ' + ' in ql_exp else ql_exp
                aligned_experiments.append(opponent_config)
                ql_values.append(ql_exp_means[ql_exp])
                
                # Find matching EQL experiment
                matching_eql = None
                for eql_exp in eql_exp_means:
                    if opponent_config in eql_exp:
                        matching_eql = eql_exp
                        break
                
                if matching_eql:
                    eql_values.append(eql_exp_means[matching_eql])
                else:
                    eql_values.append(0)
            
            if not aligned_experiments:
                continue
                
            x = np.arange(len(aligned_experiments))
            width = 0.35
            
            # Create bars
            bars1 = ax.bar(x - width/2, ql_values, width, label='Basic QL', color='lightblue', edgecolor='black')
            bars2 = ax.bar(x + width/2, eql_values, width, label='Enhanced QL', color='darkblue', edgecolor='black')
            
            # Customize plot
            ax.set_xlabel('Opponent Configuration')
            ax.set_ylabel('Average Cooperation Rate' if 'coop' in data_key else 'Average Final Score')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(aligned_experiments, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_detailed_comparison_plots(experiment_name, num_rounds=100):
    """Create detailed comparison plots for a specific experiment showing evolution over time."""
    # Determine QL type from experiment name
    if '2 QL' in experiment_name:
        ql_type = '2QL'
        eql_type = '2EQL'
    else:
        ql_type = '1QL'
        eql_type = '1EQL'
    
    # Load data for both QL types
    ql_pairwise = load_detailed_data(ql_type, experiment_name, 'pairwise_cooperation')
    eql_pairwise = load_detailed_data(eql_type, experiment_name, 'pairwise_cooperation')
    ql_nperson = load_detailed_data(ql_type, experiment_name, 'nperson_cooperation')
    eql_nperson = load_detailed_data(eql_type, experiment_name, 'nperson_cooperation')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Detailed Comparison: {experiment_name}', fontsize=16, weight='bold')
    
    # Pairwise Cooperation
    ax = axes[0, 0]
    if ql_pairwise is not None and eql_pairwise is not None:
        rounds = range(1, min(num_rounds + 1, len(ql_pairwise) + 1))
        
        # Get QL agent columns
        ql_agents = get_ql_agent_columns(ql_pairwise, ql_type)
        eql_agents = get_ql_agent_columns(eql_pairwise, eql_type)
        
        # Plot average of QL agents
        ql_means = []
        for col in ql_agents:
            if f'{col}_mean' in ql_pairwise.columns:
                ql_means.append(ql_pairwise[f'{col}_mean'].values)
        
        if ql_means:
            ql_avg = np.mean(ql_means, axis=0)
            ax.plot(rounds, ql_avg[:len(rounds)], label='Basic QL', color='lightblue', linewidth=2)
        
        # Plot average of EQL agents
        eql_means = []
        for col in eql_agents:
            if f'{col}_mean' in eql_pairwise.columns:
                eql_means.append(eql_pairwise[f'{col}_mean'].values)
        
        if eql_means:
            eql_avg = np.mean(eql_means, axis=0)
            ax.plot(rounds, eql_avg[:len(rounds)], label='Enhanced QL', color='darkblue', linewidth=2)
        
        ax.legend()
    
    ax.set_title('Pairwise Cooperation Rate')
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Neighbourhood Cooperation
    ax = axes[0, 1]
    if ql_nperson is not None and eql_nperson is not None:
        rounds = range(1, min(num_rounds + 1, len(ql_nperson) + 1))
        
        # Get QL agent columns
        ql_agents = get_ql_agent_columns(ql_nperson, ql_type)
        eql_agents = get_ql_agent_columns(eql_nperson, eql_type)
        
        # Plot average of QL agents
        ql_means = []
        for col in ql_agents:
            if f'{col}_mean' in ql_nperson.columns:
                ql_means.append(ql_nperson[f'{col}_mean'].values)
        
        if ql_means:
            ql_avg = np.mean(ql_means, axis=0)
            ax.plot(rounds, ql_avg[:len(rounds)], label='Basic QL', color='lightblue', linewidth=2)
        
        # Plot average of EQL agents
        eql_means = []
        for col in eql_agents:
            if f'{col}_mean' in eql_nperson.columns:
                eql_means.append(eql_nperson[f'{col}_mean'].values)
        
        if eql_means:
            eql_avg = np.mean(eql_means, axis=0)
            ax.plot(rounds, eql_avg[:len(rounds)], label='Enhanced QL', color='darkblue', linewidth=2)
        
        ax.legend()
    
    ax.set_title('Neighbourhood Cooperation Rate')
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Pairwise Scores
    ql_pairwise_scores = load_detailed_data(ql_type, experiment_name, 'pairwise_scores')
    eql_pairwise_scores = load_detailed_data(eql_type, experiment_name, 'pairwise_scores')
    
    ax = axes[1, 0]
    if ql_pairwise_scores is not None and eql_pairwise_scores is not None:
        rounds = range(1, min(num_rounds + 1, len(ql_pairwise_scores) + 1))
        
        # Get QL agent columns
        ql_agents = get_ql_agent_columns(ql_pairwise_scores, ql_type)
        eql_agents = get_ql_agent_columns(eql_pairwise_scores, eql_type)
        
        # Plot average of QL agents
        ql_means = []
        for col in ql_agents:
            if f'{col}_mean' in ql_pairwise_scores.columns:
                ql_means.append(ql_pairwise_scores[f'{col}_mean'].values)
        
        if ql_means:
            ql_avg = np.mean(ql_means, axis=0)
            ax.plot(rounds, ql_avg[:len(rounds)], label='Basic QL', color='lightblue', linewidth=2)
        
        # Plot average of EQL agents
        eql_means = []
        for col in eql_agents:
            if f'{col}_mean' in eql_pairwise_scores.columns:
                eql_means.append(eql_pairwise_scores[f'{col}_mean'].values)
        
        if eql_means:
            eql_avg = np.mean(eql_means, axis=0)
            ax.plot(rounds, eql_avg[:len(rounds)], label='Enhanced QL', color='darkblue', linewidth=2)
        
        ax.legend()
    
    ax.set_title('Pairwise Cumulative Scores')
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Score')
    ax.grid(True, alpha=0.3)
    
    # Neighbourhood Scores
    ql_nperson_scores = load_detailed_data(ql_type, experiment_name, 'nperson_scores')
    eql_nperson_scores = load_detailed_data(eql_type, experiment_name, 'nperson_scores')
    
    ax = axes[1, 1]
    if ql_nperson_scores is not None and eql_nperson_scores is not None:
        rounds = range(1, min(num_rounds + 1, len(ql_nperson_scores) + 1))
        
        # Get QL agent columns
        ql_agents = get_ql_agent_columns(ql_nperson_scores, ql_type)
        eql_agents = get_ql_agent_columns(eql_nperson_scores, eql_type)
        
        # Plot average of QL agents
        ql_means = []
        for col in ql_agents:
            if f'{col}_mean' in ql_nperson_scores.columns:
                ql_means.append(ql_nperson_scores[f'{col}_mean'].values)
        
        if ql_means:
            ql_avg = np.mean(ql_means, axis=0)
            ax.plot(rounds, ql_avg[:len(rounds)], label='Basic QL', color='lightblue', linewidth=2)
        
        # Plot average of EQL agents
        eql_means = []
        for col in eql_agents:
            if f'{col}_mean' in eql_nperson_scores.columns:
                eql_means.append(eql_nperson_scores[f'{col}_mean'].values)
        
        if eql_means:
            eql_avg = np.mean(eql_means, axis=0)
            ax.plot(rounds, eql_avg[:len(rounds)], label='Enhanced QL', color='darkblue', linewidth=2)
        
        ax.legend()
    
    ax.set_title('Neighbourhood Cumulative Scores')
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_heatmap(ql_data, eql_data):
    """Create a heatmap showing performance differences between QL and Enhanced QL."""
    # Prepare data for heatmap
    metrics = ['pairwise_coop', 'pairwise_scores', 'nperson_coop', 'nperson_scores']
    scenarios = ['1 QL', '2 QL']
    
    improvement_data = []
    
    for scenario in scenarios:
        row_data = []
        for metric in metrics:
            if ql_data.get(metric) is not None and eql_data.get(metric) is not None:
                # Get the correct column name based on data type
                value_col = 'Avg_Cooperation' if 'coop' in metric else 'Final_Score'
                
                # Filter for scenario
                ql_scenario = scenario
                eql_scenario = scenario.replace('QL', 'EQL')
                
                ql_filtered = ql_data[metric][ql_data[metric]['Experiment'].str.contains(ql_scenario, regex=False)]
                eql_filtered = eql_data[metric][eql_data[metric]['Experiment'].str.contains(eql_scenario, regex=False)]
                
                # Get QL agent data
                ql_agents = ql_filtered[ql_filtered['Agent'].str.contains('QL')]
                eql_agents = eql_filtered[eql_filtered['Agent'].str.contains('EQL')]
                
                if len(ql_agents) > 0 and len(eql_agents) > 0:
                    # Calculate average
                    ql_avg = ql_agents[value_col].mean()
                    eql_avg = eql_agents[value_col].mean()
                    
                    improvement = ((eql_avg - ql_avg) / (abs(ql_avg) + 0.001)) * 100
                    row_data.append(improvement)
                else:
                    row_data.append(0)
            else:
                row_data.append(0)
        
        improvement_data.append(row_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvement_df = pd.DataFrame(improvement_data, 
                                index=scenarios,
                                columns=['Pairwise\nCooperation', 'Pairwise\nScores', 
                                       'Neighbourhood\nCooperation', 'Neighbourhood\nScores'])
    
    sns.heatmap(improvement_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement %'}, ax=ax,
                vmin=-50, vmax=50)
    
    ax.set_title('Enhanced QL Performance Improvement over Basic QL (%)', fontsize=14, weight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Scenario')
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run comparative analysis."""
    print("=== Q-Learning Comparative Analysis ===")
    
    # Check if results directories exist
    if not os.path.exists('qlearning_results') or not os.path.exists('enhanced_qlearning_results'):
        print("Error: Results directories not found. Please run the experiments first.")
        return
    
    # Load summary data
    print("\nLoading summary data...")
    ql_data = {}
    ql_data.update(load_summary_data('qlearning_results', '1QL'))
    ql_data.update(load_summary_data('qlearning_results', '2QL'))
    
    eql_data = {}
    eql_data.update(load_summary_data('enhanced_qlearning_results', '1EQL'))
    eql_data.update(load_summary_data('enhanced_qlearning_results', '2EQL'))
    
    # Create output directory
    output_dir = 'comparative_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create comparison plots by scenario
    print("\nCreating scenario comparison plots...")
    try:
        fig_1ql = create_comparison_plots_by_scenario(ql_data, eql_data, "1 QL")
        fig_1ql.savefig(f'{output_dir}/1QL_vs_1EQL_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig_1ql)
        print("  - Created 1QL comparison plot")
    except Exception as e:
        print(f"  - Error creating 1QL comparison: {e}")
    
    try:
        fig_2ql = create_comparison_plots_by_scenario(ql_data, eql_data, "2 QL")
        fig_2ql.savefig(f'{output_dir}/2QL_vs_2EQL_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig_2ql)
        print("  - Created 2QL comparison plot")
    except Exception as e:
        print(f"  - Error creating 2QL comparison: {e}")
    
    # 2. Create detailed comparison for specific scenarios
    print("\nCreating detailed comparison plots...")
    detailed_scenarios = [
        "1 QL + 2 AllD",
        "1 QL + 2 TFT",
        "1 QL + 1 AllD + 1 AllC",  # Fixed order to match actual files
        "2 QL + 1 AllD",
        "2 QL + 1 TFT"
    ]
    
    for scenario in detailed_scenarios:
        try:
            fig = create_detailed_comparison_plots(scenario)
            clean_name = scenario.replace(' ', '_').replace('+', 'plus')
            fig.savefig(f'{output_dir}/detailed_{clean_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Created detailed plot for {scenario}")
        except Exception as e:
            print(f"  - Error creating detailed plot for {scenario}: {e}")
    
    # 3. Create summary heatmap
    print("\nCreating summary heatmap...")
    try:
        fig_heatmap = create_summary_heatmap(ql_data, eql_data)
        fig_heatmap.savefig(f'{output_dir}/performance_improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig_heatmap)
        print("  - Created summary heatmap")
    except Exception as e:
        print(f"  - Error creating heatmap: {e}")
    
    # 4. Generate summary statistics report
    print("\nGenerating summary report...")
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write("Q-Learning Comparative Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Compare average cooperation rates
        for scenario in ['1 QL', '2 QL']:
            f.write(f"\n{scenario} Scenario:\n")
            f.write("-" * 30 + "\n")
            
            for game_mode in ['pairwise', 'nperson']:
                for metric in ['coop', 'scores']:
                    key = f'{game_mode}_{metric}'
                    if ql_data.get(key) is not None and eql_data.get(key) is not None:
                        value_col = 'Avg_Cooperation' if metric == 'coop' else 'Final_Score'
                        
                        ql_scenario = scenario
                        eql_scenario = scenario.replace('QL', 'EQL')
                        
                        ql_filtered = ql_data[key][ql_data[key]['Experiment'].str.contains(ql_scenario, regex=False)]
                        eql_filtered = eql_data[key][eql_data[key]['Experiment'].str.contains(eql_scenario, regex=False)]
                        
                        ql_agents = ql_filtered[ql_filtered['Agent'].str.contains('QL')]
                        eql_agents = eql_filtered[eql_filtered['Agent'].str.contains('EQL')]
                        
                        if len(ql_agents) > 0 and len(eql_agents) > 0:
                            ql_avg = ql_agents[value_col].mean()
                            eql_avg = eql_agents[value_col].mean()
                            
                            if metric == 'coop':
                                f.write(f"\n{game_mode.capitalize()} Cooperation:\n")
                            else:
                                f.write(f"\n{game_mode.capitalize()} Final Score:\n")
                            
                            f.write(f"  Basic QL: {ql_avg:.3f}\n")
                            f.write(f"  Enhanced QL: {eql_avg:.3f}\n")
                            f.write(f"  Improvement: {((eql_avg - ql_avg) / (abs(ql_avg) + 0.001)) * 100:.1f}%\n")
                        else:
                            # Debug info
                            f.write(f"\n{game_mode.capitalize()} {metric}: No data found\n")
                            f.write(f"  QL agents found: {len(ql_agents)}\n")
                            f.write(f"  EQL agents found: {len(eql_agents)}\n")
    
    print(f"\nAnalysis complete! Results saved to '{output_dir}' directory.")
    print("\nKey findings:")
    print("- Check the comparison plots to see performance differences")
    print("- The heatmap shows percentage improvements across scenarios")
    print("- Detailed plots show evolution over time")


if __name__ == "__main__":
    main()