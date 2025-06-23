#!/usr/bin/env python3
"""
Compare Different Enhanced Q-Learning Versions
This script allows comparison between:
1. Basic QL vs Old Enhanced QL
2. Basic QL vs Unified Enhanced QL
3. Old Enhanced QL vs Unified Enhanced QL
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def check_available_results():
    """Check which result directories are available."""
    dirs = {
        'basic': 'qlearning_results',
        'old_enhanced': 'enhanced_qlearning_results',
        'unified_enhanced': 'enhanced_qlearning_results_unified'
    }
    
    available = {}
    for key, dir_path in dirs.items():
        available[key] = os.path.exists(dir_path)
    
    return available, dirs


def load_experiment_data(results_dir, ql_type, experiment_name, game_mode):
    """Load data for a specific experiment."""
    clean_name = experiment_name.replace(' ', '_').replace('+', 'plus')
    filepath = f'{results_dir}/{ql_type}_experiments/csv/{ql_type}_{game_mode}_{clean_name}.csv'
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def compare_pairwise_performance(exp_name="2 QL + 1 AllD"):
    """Compare pairwise performance across different versions."""
    available, dirs = check_available_results()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Pairwise Performance Comparison: {exp_name}', fontsize=16, weight='bold')
    
    # Prepare data
    versions = []
    if available['basic']:
        versions.append(('Basic QL', dirs['basic'], '2QL', 'blue', '-'))
    if available['old_enhanced']:
        versions.append(('Old Enhanced QL', dirs['old_enhanced'], '2EQL', 'red', '--'))
    if available['unified_enhanced']:
        versions.append(('Unified Enhanced QL', dirs['unified_enhanced'], '2EQL', 'green', ':'))
    
    if len(versions) < 2:
        print("Need at least 2 versions to compare!")
        return
    
    # Plot configurations
    plot_configs = [
        ('pairwise_cooperation', 'Cooperation Rate', axes[0, 0]),
        ('pairwise_scores', 'Cumulative Score', axes[0, 1])
    ]
    
    # Load and plot data
    for game_mode, ylabel, ax in plot_configs[:2]:
        for version_name, results_dir, ql_type, color, linestyle in versions:
            adjusted_exp = exp_name.replace('QL', 'EQL') if 'EQL' in ql_type else exp_name
            data = load_experiment_data(results_dir, ql_type, adjusted_exp, game_mode)
            
            if data is not None:
                # Plot QL agents only
                agent_cols = ['QL_1', 'QL_2'] if 'QL' in ql_type else ['EQL_1', 'EQL_2']
                
                for agent in agent_cols:
                    if f'{agent}_mean' in data.columns:
                        ax.plot(data['Round'], data[f'{agent}_mean'], 
                               label=f'{version_name} - {agent}', 
                               color=color, linestyle=linestyle, linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{game_mode.replace("_", " ").title()}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Add statistics comparison in bottom panels
    ax_stats = axes[1, 0]
    ax_diff = axes[1, 1]
    
    # Calculate final statistics
    stats_data = []
    for version_name, results_dir, ql_type, _, _ in versions:
        adjusted_exp = exp_name.replace('QL', 'EQL') if 'EQL' in ql_type else exp_name
        
        coop_data = load_experiment_data(results_dir, ql_type, adjusted_exp, 'pairwise_cooperation')
        score_data = load_experiment_data(results_dir, ql_type, adjusted_exp, 'pairwise_scores')
        
        if coop_data is not None and score_data is not None:
            agent_cols = ['QL_1', 'QL_2'] if 'QL' in ql_type else ['EQL_1', 'EQL_2']
            
            # Calculate average final values
            final_coop = np.mean([coop_data[f'{agent}_mean'].iloc[-1] for agent in agent_cols 
                                 if f'{agent}_mean' in coop_data.columns])
            final_score = np.mean([score_data[f'{agent}_mean'].iloc[-1] for agent in agent_cols 
                                  if f'{agent}_mean' in score_data.columns])
            
            stats_data.append({
                'Version': version_name,
                'Final Cooperation': final_coop,
                'Final Score': final_score
            })
    
    # Plot statistics
    stats_df = pd.DataFrame(stats_data)
    x = np.arange(len(stats_df))
    width = 0.35
    
    ax_stats.bar(x - width/2, stats_df['Final Cooperation'], width, label='Cooperation Rate')
    ax_stats.bar(x + width/2, stats_df['Final Score']/stats_df['Final Score'].max(), width, label='Normalized Score')
    ax_stats.set_xlabel('Version')
    ax_stats.set_ylabel('Value')
    ax_stats.set_title('Final Performance Metrics')
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(stats_df['Version'], rotation=45, ha='right')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)
    
    # Plot differences from basic
    if 'Basic QL' in stats_df['Version'].values:
        basic_idx = stats_df[stats_df['Version'] == 'Basic QL'].index[0]
        basic_coop = stats_df.iloc[basic_idx]['Final Cooperation']
        basic_score = stats_df.iloc[basic_idx]['Final Score']
        
        diff_data = []
        for _, row in stats_df.iterrows():
            if row['Version'] != 'Basic QL':
                diff_data.append({
                    'Version': row['Version'],
                    'Cooperation Diff': ((row['Final Cooperation'] - basic_coop) / basic_coop) * 100,
                    'Score Diff': ((row['Final Score'] - basic_score) / basic_score) * 100
                })
        
        if diff_data:
            diff_df = pd.DataFrame(diff_data)
            x = np.arange(len(diff_df))
            
            ax_diff.bar(x - width/2, diff_df['Cooperation Diff'], width, label='Cooperation %')
            ax_diff.bar(x + width/2, diff_df['Score Diff'], width, label='Score %')
            ax_diff.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_diff.set_xlabel('Version')
            ax_diff.set_ylabel('% Difference from Basic QL')
            ax_diff.set_title('Performance Improvement over Basic QL')
            ax_diff.set_xticks(x)
            ax_diff.set_xticklabels(diff_df['Version'], rotation=45, ha='right')
            ax_diff.legend()
            ax_diff.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_comprehensive_comparison():
    """Create a comprehensive comparison across all available versions."""
    available, dirs = check_available_results()
    
    # Key experiments to compare
    key_experiments = [
        '2 QL + 1 AllD',
        '2 QL + 1 TFT',
        '1 QL + 2 AllD',
        '1 QL + 1 AllD + 1 AllC'
    ]
    
    results = []
    
    for exp in key_experiments:
        for version_key, version_name in [('basic', 'Basic QL'), 
                                         ('old_enhanced', 'Old Enhanced'), 
                                         ('unified_enhanced', 'Unified Enhanced')]:
            if available[version_key]:
                ql_type = '2QL' if '2 QL' in exp else '1QL'
                if version_key != 'basic':
                    ql_type = ql_type.replace('QL', 'EQL')
                
                adjusted_exp = exp.replace('QL', 'EQL') if 'EQL' in ql_type else exp
                
                # Load cooperation data
                coop_data = load_experiment_data(dirs[version_key], ql_type, adjusted_exp, 'pairwise_cooperation')
                
                if coop_data is not None:
                    agent_cols = ['QL_1', 'QL_2'] if 'QL' in ql_type else ['EQL_1', 'EQL_2']
                    if '1QL' in ql_type or '1EQL' in ql_type:
                        agent_cols = ['QL_1'] if 'QL' in ql_type else ['EQL_1']
                    
                    # Calculate average final cooperation
                    final_coops = []
                    for agent in agent_cols:
                        if f'{agent}_mean' in coop_data.columns:
                            final_coops.append(coop_data[f'{agent}_mean'].iloc[-1])
                    
                    if final_coops:
                        results.append({
                            'Experiment': exp,
                            'Version': version_name,
                            'Final Cooperation': np.mean(final_coops)
                        })
    
    # Create visualization
    if results:
        results_df = pd.DataFrame(results)
        
        # Pivot for heatmap
        pivot_df = results_df.pivot(index='Experiment', columns='Version', values='Final Cooperation')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Final Cooperation Rates Across Versions')
        ax1.set_ylabel('Experiment')
        
        # Bar plot showing improvements
        if 'Basic QL' in pivot_df.columns:
            improvements = pd.DataFrame()
            for col in pivot_df.columns:
                if col != 'Basic QL':
                    improvements[col] = ((pivot_df[col] - pivot_df['Basic QL']) / pivot_df['Basic QL']) * 100
            
            improvements.plot(kind='bar', ax=ax2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('% Improvement over Basic QL')
            ax2.set_xlabel('Experiment')
            ax2.set_ylabel('% Improvement')
            ax2.legend(title='Version')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    return None


def main():
    """Main function to run comparisons."""
    available, dirs = check_available_results()
    
    print("Enhanced Q-Learning Version Comparison")
    print("=" * 50)
    print("\nAvailable results:")
    for key, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {key}: {dirs[key]}")
    
    if sum(available.values()) < 2:
        print("\nERROR: Need at least 2 versions to compare!")
        print("Please run the necessary demo generators first.")
        return
    
    # Create output directory
    output_dir = 'enhanced_version_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparisons
    print("\nGenerating comparison plots...")
    
    # Detailed comparison for a key experiment
    fig1 = compare_pairwise_performance("2 QL + 1 AllD")
    if fig1:
        fig1.savefig(f'{output_dir}/detailed_2QL_1AllD_comparison.png', dpi=300, bbox_inches='tight')
        print("  - Saved: detailed_2QL_1AllD_comparison.png")
    
    # Comprehensive comparison
    fig2 = create_comprehensive_comparison()
    if fig2:
        fig2.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        print("  - Saved: comprehensive_comparison.png")
    
    # Generate summary report
    print("\nGenerating summary report...")
    
    with open(f'{output_dir}/comparison_summary.txt', 'w') as f:
        f.write("Enhanced Q-Learning Version Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Available Versions:\n")
        for key, is_available in available.items():
            if is_available:
                f.write(f"  - {key}\n")
        
        f.write("\n\nKey Findings:\n")
        
        if available['unified_enhanced'] and available['old_enhanced']:
            f.write("  - Unified Enhanced QL uses corrected pairwise state representation\n")
            f.write("  - Old Enhanced QL had a bug in pairwise state calculation\n")
            f.write("  - Unified version should show better performance in pairwise scenarios\n")
        
        if available['unified_enhanced'] and available['basic']:
            f.write("  - Unified Enhanced QL adds epsilon decay and optional memory states\n")
            f.write("  - Performance improvements vary by scenario\n")
    
    print(f"\nAll comparison files saved to '{output_dir}/' directory")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()