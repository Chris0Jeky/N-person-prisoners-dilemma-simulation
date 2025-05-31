"""
Run Simple IPD simulations and export to CSV
============================================
This script runs various scenarios and exports results in npdl-compatible format.
"""

from simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation
from csv_exporter import CSVExporter, export_batch_simulations


def run_single_export_example():
    """Run a single simulation and export it."""
    print("Running single export example...")
    
    # Create agents
    alice = Agent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    bob = Agent("Bob", Strategy.ALWAYS_DEFECT, exploration_rate=0.1)
    charlie = Agent("Charlie", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
    
    # Create game and simulation
    game = PrisonersDilemmaGame()
    sim = Simulation([alice, bob, charlie], game)
    
    # Create exporter and export
    exporter = CSVExporter(results_dir="simple_results")
    exporter.export_simulation(sim, "Simple_3Agent_Test", num_rounds=50, run_number=0)


def create_scenarios():
    """Create various scenarios for batch export."""
    scenarios = [
        {
            'name': 'Baseline_AllCooperate',
            'num_rounds': 100,
            'agents': [
                {'name': 'Agent0', 'strategy': Strategy.ALWAYS_COOPERATE},
                {'name': 'Agent1', 'strategy': Strategy.ALWAYS_COOPERATE},
                {'name': 'Agent2', 'strategy': Strategy.ALWAYS_COOPERATE},
            ],
            'payoffs': {'reward': 3, 'temptation': 5, 'punishment': 1, 'sucker': 0}
        },
        {
            'name': 'Baseline_AllDefect',
            'num_rounds': 100,
            'agents': [
                {'name': 'Agent0', 'strategy': Strategy.ALWAYS_DEFECT},
                {'name': 'Agent1', 'strategy': Strategy.ALWAYS_DEFECT},
                {'name': 'Agent2', 'strategy': Strategy.ALWAYS_DEFECT},
            ],
            'payoffs': {'reward': 3, 'temptation': 5, 'punishment': 1, 'sucker': 0}
        },
        {
            'name': 'TFT_vs_Defect_NoExploration',
            'num_rounds': 100,
            'agents': [
                {'name': 'TFT1', 'strategy': Strategy.TIT_FOR_TAT},
                {'name': 'TFT2', 'strategy': Strategy.TIT_FOR_TAT},
                {'name': 'Defector', 'strategy': Strategy.ALWAYS_DEFECT},
            ]
        },
        {
            'name': 'TFT_vs_Defect_WithExploration',
            'num_rounds': 100,
            'agents': [
                {'name': 'TFT1', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'TFT2', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'Defector', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.2},
            ]
        },
        {
            'name': 'Mixed_Strategies_LowExploration',
            'num_rounds': 150,
            'agents': [
                {'name': 'Cooperator', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.05},
                {'name': 'Defector', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.05},
                {'name': 'TitForTat', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.05},
            ]
        },
        {
            'name': 'Mixed_Strategies_HighExploration',
            'num_rounds': 150,
            'agents': [
                {'name': 'Cooperator', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.3},
                {'name': 'Defector', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.3},
                {'name': 'TitForTat', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.3},
            ]
        },
        {
            'name': 'TFT_Dominant',
            'num_rounds': 200,
            'agents': [
                {'name': 'TFT1', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.02},
                {'name': 'TFT2', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.02},
                {'name': 'TFT3', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.02},
            ]
        },
        {
            'name': 'Exploration_Effect_Test',
            'num_rounds': 100,
            'agents': [
                {'name': 'NoExplore', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0},
                {'name': 'LowExplore', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'HighExplore', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.5},
            ]
        },
        {
            'name': 'Asymmetric_Payoffs',
            'num_rounds': 100,
            'agents': [
                {'name': 'Agent0', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'Agent1', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.1},
                {'name': 'Agent2', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.1},
            ],
            'payoffs': {'reward': 4, 'temptation': 7, 'punishment': 0, 'sucker': -1}
        }
    ]
    
    return scenarios


def run_batch_export():
    """Run batch export with multiple scenarios and runs."""
    scenarios = create_scenarios()
    
    print("Starting batch export of simple IPD simulations...")
    print(f"Number of scenarios: {len(scenarios)}")
    print(f"Runs per scenario: 5")
    
    export_batch_simulations(scenarios, num_runs=5, results_dir="simple_results")


def create_summary_csv():
    """Create a summary CSV combining results from all scenarios."""
    import pandas as pd
    import os
    from glob import glob
    
    results_dir = "simple_results"
    
    # Collect all agent CSVs
    all_agents_data = []
    all_rounds_data = []
    
    for scenario_dir in glob(os.path.join(results_dir, "*")):
        if os.path.isdir(scenario_dir):
            scenario_name = os.path.basename(scenario_dir)
            
            # Process each run
            for run_dir in glob(os.path.join(scenario_dir, "run_*")):
                # Read agents data
                agents_file = os.path.join(run_dir, "experiment_results_agents.csv")
                if os.path.exists(agents_file):
                    df = pd.read_csv(agents_file)
                    all_agents_data.append(df)
                
                # Read rounds data  
                rounds_file = os.path.join(run_dir, "experiment_results_rounds.csv")
                if os.path.exists(rounds_file):
                    df = pd.read_csv(rounds_file)
                    all_rounds_data.append(df)
    
    # Combine and save
    if all_agents_data:
        combined_agents = pd.concat(all_agents_data, ignore_index=True)
        combined_agents.to_csv(os.path.join(results_dir, "all_agents_summary.csv"), index=False)
        print(f"\nCombined agents summary saved to: {os.path.join(results_dir, 'all_agents_summary.csv')}")
        
        # Create aggregate statistics
        print("\nAggregate Statistics by Scenario and Strategy:")
        print("="*60)
        agg_stats = combined_agents.groupby(['scenario_name', 'strategy'])['final_score'].agg(['mean', 'std', 'count'])
        print(agg_stats)
    
    if all_rounds_data:
        # Sample rounds data (too large to save all)
        combined_rounds = pd.concat(all_rounds_data, ignore_index=True)
        
        # Calculate cooperation rates
        print("\n\nCooperation Rates by Scenario:")
        print("="*60)
        coop_rates = combined_rounds.groupby(['scenario_name', 'strategy'])['move'].apply(
            lambda x: (x == 'cooperate').mean()
        ).round(3)
        print(coop_rates)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Run single export example
        run_single_export_example()
    elif len(sys.argv) > 1 and sys.argv[1] == "--summary":
        # Create summary of existing results
        create_summary_csv()
    else:
        # Run batch export (default)
        run_batch_export()
        print("\n\nCreating summary statistics...")
        create_summary_csv()
