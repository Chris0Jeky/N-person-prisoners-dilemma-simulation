"""
Custom Scenario Example
======================
Shows how to create custom scenarios for CSV export.
"""

from simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation
from csv_exporter import CSVExporter


def create_custom_scenarios():
    """Example of creating custom scenarios programmatically."""
    
    # Scenario 1: Testing different exploration rates
    exploration_test = {
        'name': 'Custom_Exploration_Test',
        'num_rounds': 200,
        'agents': [
            {'name': 'Conservative', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.01},
            {'name': 'Moderate', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.15},
            {'name': 'Adventurous', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.30},
        ]
    }
    
    # Scenario 2: Custom payoff matrix (Chicken game variant)
    chicken_game = {
        'name': 'Custom_Chicken_Game',
        'num_rounds': 150,
        'agents': [
            {'name': 'Player1', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.1},
            {'name': 'Player2', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.1},
            {'name': 'Player3', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
        ],
        'payoffs': {
            'reward': 3,      # Both cooperate (both swerve)
            'temptation': 5,  # I defect, you cooperate (I go straight, you swerve)
            'punishment': 0,  # Both defect (crash!)
            'sucker': 1       # I cooperate, you defect (I swerve, you go straight)
        }
    }
    
    # Scenario 3: Gradual exploration increase
    agents_gradual = []
    for i in range(3):
        agents_gradual.append({
            'name': f'Agent{i}',
            'strategy': Strategy.TIT_FOR_TAT,
            'exploration_rate': i * 0.1  # 0.0, 0.1, 0.2
        })
    
    gradual_exploration = {
        'name': 'Custom_Gradual_Exploration',
        'num_rounds': 100,
        'agents': agents_gradual
    }
    
    return [exploration_test, chicken_game, gradual_exploration]


def run_custom_single_simulation():
    """Run a single custom simulation with detailed output."""
    print("Running custom single simulation...")
    
    # Create custom agents
    agents = [
        Agent("Learner", Strategy.TIT_FOR_TAT, exploration_rate=0.15),
        Agent("Exploiter", Strategy.ALWAYS_DEFECT, exploration_rate=0.05),
        Agent("Nice Guy", Strategy.ALWAYS_COOPERATE, exploration_rate=0.25)
    ]
    
    # Custom payoff matrix (more forgiving)
    game = PrisonersDilemmaGame(
        reward=4,      # Cooperation bonus
        temptation=6,  # Still tempting to defect
        punishment=2,  # Less harsh punishment
        sucker=1       # Not too bad to be exploited
    )
    
    # Create simulation
    sim = Simulation(agents, game)
    
    # Export with custom name
    exporter = CSVExporter(results_dir="custom_results")
    exporter.export_simulation(
        sim, 
        scenario_name="Custom_Forgiving_Game",
        num_rounds=75,
        run_number=0
    )
    
    # Also run a visual simulation for first 10 rounds
    print("\nVisual simulation of first 10 rounds:")
    sim.run_simulation(num_rounds=10)


def run_custom_batch():
    """Run custom scenarios in batch mode."""
    from csv_exporter import export_batch_simulations
    
    scenarios = create_custom_scenarios()
    
    print("Running custom batch simulations...")
    print(f"Number of custom scenarios: {len(scenarios)}")
    
    # Run with fewer runs for quick testing
    export_batch_simulations(
        scenarios, 
        num_runs=3,
        results_dir="custom_results"
    )
    
    # Quick analysis
    print("\n\nQuick Analysis of Results:")
    analyze_custom_results()


def analyze_custom_results():
    """Quick analysis of custom results."""
    import pandas as pd
    import os
    from glob import glob
    
    results_dir = "custom_results"
    all_agents = []
    
    # Load all agent results
    for scenario_dir in glob(os.path.join(results_dir, "*")):
        if os.path.isdir(scenario_dir):
            for run_dir in glob(os.path.join(scenario_dir, "run_*")):
                agents_file = os.path.join(run_dir, "experiment_results_agents.csv")
                if os.path.exists(agents_file):
                    df = pd.read_csv(agents_file)
                    all_agents.append(df)
    
    if all_agents:
        combined = pd.concat(all_agents, ignore_index=True)
        
        # Show average scores by scenario
        print("\nAverage Scores by Scenario:")
        print("="*50)
        avg_by_scenario = combined.groupby(['scenario_name', 'strategy'])['final_score'].mean()
        for scenario in combined['scenario_name'].unique():
            print(f"\n{scenario}:")
            scenario_scores = avg_by_scenario[scenario]
            for strategy, score in scenario_scores.items():
                print(f"  {strategy}: {score:.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        run_custom_single_simulation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_custom_results()
    else:
        run_custom_batch()
