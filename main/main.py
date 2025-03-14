import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from agents import Agent
from environment import Environment
from utils import create_payoff_matrix


def load_scenarios(file_path):
    """Load scenario definitions from a JSON file."""
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    return scenarios


def run_experiment(scenario):
    """Runs a single experiment based on the provided scenario."""
    # 1. Create Agents based on the scenario's agent_strategies.
    agents = []
    agent_id_counter = 0
    for strategy, count in scenario["agent_strategies"].items():
        for _ in range(count):
            if strategy == "q_learning":
                agents.append(
                    Agent(
                        agent_id=agent_id_counter,
                        strategy=strategy,
                        learning_rate=scenario["learning_rate"],
                        discount_factor=scenario["discount_factor"],
                        epsilon=scenario["epsilon"],
                    )
                )
            else:
                agents.append(Agent(agent_id=agent_id_counter, strategy=strategy))
            agent_id_counter += 1

    # 2. Create Payoff Matrix
    payoff_matrix = create_payoff_matrix(scenario["num_agents"])

    # 3. Create Environment (this will also log network stats)
    env = Environment(
        agents,
        payoff_matrix,
        network_type=scenario["network_type"],
        network_params=scenario["network_params"],
    )

    # 4. Run Simulation for the specified number of rounds.
    results = env.run_simulation(scenario["num_rounds"])

    # 5. Return experiment results along with scenario info and final agent states.
    return {"scenario": scenario, "results": results, "agents": agents}


def save_results(all_results, base_filename="experiment_results", results_dir="results"):
    """Saves the experiment results to CSV files."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for result in all_results:
        scenario_name = result['scenario']['scenario_name']

        # Save agent summary data.
        agent_data = []
        for agent in result['agents']:
            agent_data.append({
                'scenario_name': scenario_name,
                'agent_id': agent.agent_id,
                'strategy': agent.strategy,
                'final_score': agent.score,
                'final_q_values': agent.q_values if agent.strategy == 'q_learning' else None,
            })
        df_agent = pd.DataFrame(agent_data)
        agent_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_agents.csv")
        df_agent.to_csv(agent_filename, index=False)

        # Save round-by-round data.
        round_data = []
        for round_result in result['results']:
            round_num = round_result['round']
            moves = round_result['moves']
            payoffs = round_result['payoffs']
            for agent_id, move in moves.items():
                payoff = payoffs.get(agent_id, None)
                round_data.append({
                    'scenario_name': scenario_name,
                    'round': round_num,
                    'agent_id': agent_id,
                    'move': move,
                    'payoff': payoff,
                    'strategy': next((ag.strategy for ag in result['agents'] if ag.agent_id == agent_id), None)
                })
        df_rounds = pd.DataFrame(round_data)
        rounds_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_rounds.csv")
        df_rounds.to_csv(rounds_filename, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run N-person IPD experiments")
    parser.add_argument('--scenario_file', type=str, default='scenarios.json',
                        help='Path to the JSON file containing scenario definitions.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save experiment results.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    all_results = []

    for scenario in scenarios:
        if args.verbose:
            print(f"Running scenario: {scenario['scenario_name']}")
        result = run_experiment(scenario)
        all_results.append(result)

    save_results(all_results, results_dir=args.results_dir)

    if args.verbose:
        print(f"Simulations complete. Results saved to '{args.results_dir}' directory.")


if __name__ == "__main__":
    main()
