"""
Simple Analysis of IPD Results
==============================
Basic analysis and visualization of the CSV export results.
"""

import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt


def load_all_results(results_dir="simple_results"):
    """Load all results from the results directory."""
    all_agents = []
    all_rounds = []
    
    for scenario_dir in glob(os.path.join(results_dir, "*")):
        if os.path.isdir(scenario_dir):
            for run_dir in glob(os.path.join(scenario_dir, "run_*")):
                # Load agents data
                agents_file = os.path.join(run_dir, "experiment_results_agents.csv")
                if os.path.exists(agents_file):
                    df = pd.read_csv(agents_file)
                    all_agents.append(df)
                
                # Load rounds data
                rounds_file = os.path.join(run_dir, "experiment_results_rounds.csv")
                if os.path.exists(rounds_file):
                    df = pd.read_csv(rounds_file)
                    all_rounds.append(df)
    
    if all_agents:
        agents_df = pd.concat(all_agents, ignore_index=True)
    else:
        agents_df = pd.DataFrame()
        
    if all_rounds:
        rounds_df = pd.concat(all_rounds, ignore_index=True)
    else:
        rounds_df = pd.DataFrame()
    
    return agents_df, rounds_df


def plot_average_scores(agents_df):
    """Plot average scores by strategy across all scenarios."""
    plt.figure(figsize=(10, 6))
    
    # Calculate average scores by strategy
    avg_scores = agents_df.groupby('strategy')['final_score'].agg(['mean', 'std'])
    
    # Create bar plot
    strategies = avg_scores.index
    means = avg_scores['mean']
    stds = avg_scores['std']
    
    plt.bar(strategies, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xlabel('Strategy')
    plt.ylabel('Average Final Score')
    plt.title('Average Scores by Strategy (All Scenarios)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('simple_results/average_scores_by_strategy.png')
    plt.close()


def plot_cooperation_rates(rounds_df):
    """Plot cooperation rates by strategy."""
    plt.figure(figsize=(10, 6))
    
    # Calculate cooperation rates
    coop_rates = rounds_df.groupby('strategy')['move'].apply(
        lambda x: (x == 'cooperate').mean()
    )
    
    # Create bar plot
    strategies = coop_rates.index
    rates = coop_rates.values
    
    plt.bar(strategies, rates, alpha=0.7, color='green')
    plt.xlabel('Strategy')
    plt.ylabel('Cooperation Rate')
    plt.title('Cooperation Rates by Strategy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, (strat, rate) in enumerate(zip(strategies, rates)):
        plt.text(i, rate + 0.02, f'{rate:.1%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('simple_results/cooperation_rates_by_strategy.png')
    plt.close()


def plot_scenario_comparison(agents_df):
    """Plot score comparison across scenarios."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get average scores by scenario and strategy
    scenario_scores = agents_df.groupby(['scenario_name', 'strategy'])['final_score'].mean().unstack()
    
    # Create grouped bar plot
    scenario_scores.plot(kind='bar', ax=ax)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Scores by Scenario and Strategy')
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('simple_results/scores_by_scenario.png')
    plt.close()


def plot_exploration_effect(agents_df, rounds_df):
    """Analyze the effect of exploration on cooperation."""
    # Focus on scenarios with exploration variations
    exploration_scenarios = ['TFT_vs_Defect_NoExploration', 'TFT_vs_Defect_WithExploration',
                           'Mixed_Strategies_LowExploration', 'Mixed_Strategies_HighExploration']
    
    filtered_rounds = rounds_df[rounds_df['scenario_name'].isin(exploration_scenarios)]
    
    if filtered_rounds.empty:
        print("No exploration scenarios found in data.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate cooperation rates by scenario
    coop_by_scenario = filtered_rounds.groupby(['scenario_name', 'strategy'])['move'].apply(
        lambda x: (x == 'cooperate').mean()
    ).unstack()
    
    # Create grouped bar plot
    coop_by_scenario.plot(kind='bar')
    plt.xlabel('Scenario')
    plt.ylabel('Cooperation Rate')
    plt.title('Effect of Exploration on Cooperation Rates')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('simple_results/exploration_effect.png')
    plt.close()


def plot_cooperation_over_time(rounds_df, scenario_name=None, max_rounds=50):
    """Plot cooperation rate over time for a specific scenario."""
    if scenario_name is None:
        # Use first scenario if none specified
        scenario_name = rounds_df['scenario_name'].iloc[0]
    
    scenario_data = rounds_df[rounds_df['scenario_name'] == scenario_name]
    
    if scenario_data.empty:
        print(f"No data found for scenario: {scenario_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate cooperation rate by round and strategy
    for strategy in scenario_data['strategy'].unique():
        strategy_data = scenario_data[scenario_data['strategy'] == strategy]
        coop_by_round = strategy_data.groupby('round')['move'].apply(
            lambda x: (x == 'cooperate').mean()
        )
        
        # Plot only up to max_rounds
        coop_by_round = coop_by_round[coop_by_round.index < max_rounds]
        plt.plot(coop_by_round.index, coop_by_round.values, label=strategy, marker='o', markersize=4)
    
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.title(f'Cooperation Rate Over Time - {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'simple_results/cooperation_over_time_{scenario_name}.png')
    plt.close()


def print_summary_statistics(agents_df, rounds_df):
    """Print summary statistics."""
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal simulations: {agents_df['scenario_name'].nunique()} scenarios")
    print(f"Total runs: {len(agents_df) // 3} (assuming 3 agents per run)")
    
    print("\n--- Average Scores by Strategy ---")
    avg_scores = agents_df.groupby('strategy')['final_score'].agg(['mean', 'std', 'min', 'max'])
    print(avg_scores.round(2))
    
    print("\n--- Cooperation Rates by Strategy ---")
    coop_rates = rounds_df.groupby('strategy')['move'].apply(
        lambda x: (x == 'cooperate').mean()
    )
    for strat, rate in coop_rates.items():
        print(f"{strat}: {rate:.1%}")
    
    print("\n--- Best Performing Strategy by Scenario ---")
    best_by_scenario = agents_df.groupby(['scenario_name', 'strategy'])['final_score'].mean()
    for scenario in agents_df['scenario_name'].unique():
        scenario_scores = best_by_scenario[scenario]
        best_strategy = scenario_scores.idxmax()
        best_score = scenario_scores.max()
        print(f"{scenario}: {best_strategy} (avg score: {best_score:.2f})")


def main():
    """Run all analyses."""
    print("Loading results...")
    agents_df, rounds_df = load_all_results()
    
    if agents_df.empty:
        print("No results found. Please run simulations first.")
        return
    
    print(f"Loaded {len(agents_df)} agent records and {len(rounds_df)} round records")
    
    # Create plots directory if it doesn't exist
    os.makedirs("simple_results", exist_ok=True)
    
    # Generate analyses
    print("\nGenerating plots...")
    plot_average_scores(agents_df)
    print("- Created: average_scores_by_strategy.png")
    
    plot_cooperation_rates(rounds_df)
    print("- Created: cooperation_rates_by_strategy.png")
    
    plot_scenario_comparison(agents_df)
    print("- Created: scores_by_scenario.png")
    
    plot_exploration_effect(agents_df, rounds_df)
    print("- Created: exploration_effect.png")
    
    # Plot cooperation over time for a few scenarios
    scenarios_to_plot = ['TFT_vs_Defect_NoExploration', 'Mixed_Strategies_HighExploration']
    for scenario in scenarios_to_plot:
        if scenario in agents_df['scenario_name'].values:
            plot_cooperation_over_time(rounds_df, scenario)
            print(f"- Created: cooperation_over_time_{scenario}.png")
    
    # Print summary statistics
    print_summary_statistics(agents_df, rounds_df)


if __name__ == "__main__":
    main()
