#main.py
import argparse
import json
import os
import logging
import pandas as pd

from agents import Agent
from environment import Environment
from utils import create_payoff_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log", mode="w")
    ]
)


def load_scenarios(file_path):
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    return scenarios


def run_experiment(scenario):
    logging.info(f"Running scenario: {scenario['scenario_name']}")

    # Create agents, handling strategy-specific parameters.
    agents = []
    agent_id_counter = 0
    for strategy, count in scenario["agent_strategies"].items():
        for _ in range(count):
            agent_params = {"agent_id": agent_id_counter, "strategy": strategy}
            if strategy == "q_learning" or strategy == "q_learning_adaptive":
                agent_params.update({
                    "learning_rate": scenario.get("learning_rate", 0.1), #use .get to provide default values
                    "discount_factor": scenario.get("discount_factor", 0.9),
                    "epsilon": scenario.get("epsilon", 0.1),
                })
            elif strategy == "generous_tit_for_tat":
                agent_params["generosity"] = scenario.get("generosity", 0.05) #default value for generosity
            elif strategy == "pavlov" or strategy == "suspicious_tit_for_tat":
                agent_params["initial_move"] = scenario.get("initial_move", "cooperate") #default value
            elif strategy == "randomprob":
              agent_params["prob_coop"] = scenario.get("prob_coop", 0.5)

            agents.append(Agent(**agent_params))  # Use dictionary unpacking
            agent_id_counter += 1

    payoff_matrix = create_payoff_matrix(scenario["num_agents"])
    env = Environment(
        agents,
        payoff_matrix,
        network_type=scenario["network_type"],
        network_params=scenario["network_params"],
    )
    results = env.run_simulation(scenario["num_rounds"])
    logging.info(f"Completed scenario: {scenario['scenario_name']}")
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
        logging.info(f"Saved agent results for scenario '{scenario_name}' to {agent_filename}")

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
        logging.info(f"Saved round results for scenario '{scenario_name}' to {rounds_filename}")



def main():
    parser = argparse.ArgumentParser(description="Run N-person IPD experiments")
    parser.add_argument('--scenario_file', type=str, default='scenarios.json',
                        help='Path to the JSON file containing scenario definitions.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save experiment results.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)  # Set logging level to DEBUG

    scenarios = load_scenarios(args.scenario_file)
    all_results = []

    for scenario in scenarios:
        result = run_experiment(scenario)
        all_results.append(result)

    save_results(all_results, results_dir=args.results_dir)
    logging.info(f"Simulations complete.  Results saved to '{args.results_dir}' directory.")


if __name__ == "__main__":
    main()