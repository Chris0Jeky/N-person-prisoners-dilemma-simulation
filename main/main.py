# main.py
import os
import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from agents import Agent
from environment import Environment
from utils import create_payoff_matrix

scenarios = [
    {
        "scenario_name": "FullyConnected_QLearning",
        "num_agents": 20,
        "num_rounds": 100,
        "network_type": "fully_connected",
        "network_params": {},
        "agent_strategies": {  # Define strategies for each agent
            "q_learning": 10,  # 10 Q-learning agents
            "random": 5,      # 5 random agents
            "tit_for_tat": 5,    # 5 Tit-for-Tat agents
        },
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.1,
        "logging_interval": 10, #logging every 10 rounds
    },
    {
        "scenario_name": "SmallWorld_Mixed",
        "num_agents": 50,
        "num_rounds": 200,
        "network_type": "small_world",
        "network_params": {"k": 4, "beta": 0.3},
        "agent_strategies": {
            "q_learning": 25,
            "always_defect": 25,
        },
        "learning_rate": 0.2,
        "discount_factor": 0.8,
        "epsilon": 0.2,
        "logging_interval": 20,
    },
     {
        "scenario_name": "ScaleFree_HighEpsilon",
        "num_agents": 30,
        "num_rounds": 150,
        "network_type": "scale_free",
        "network_params": {"m": 2},
        "agent_strategies": {
            "q_learning": 20,
            "tit_for_tat": 10,
        },
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.5,  # High exploration rate
        "logging_interval": 10,
    },
]
# --- Experiment Runner Function ---
def run_experiment(scenario):
    """Runs a single experiment based on the provided scenario."""

    # 1. Create Agents
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

    # 3. Create Environment
    env = Environment(
        agents,
        payoff_matrix,
        network_type=scenario["network_type"],
        network_params=scenario["network_params"],
    )

    # 4. Run Simulation
    results = env.run_simulation(scenario["num_rounds"])

    # 5. Return Results (including scenario info)
    return {"scenario": scenario, "results": results, "agents": agents}

# --- Result Storage Function ---
def save_results(all_results, base_filename="experiment_results"):
    """Saves the experiment results to CSV files."""

    # Create a directory for storing results if one doesn't exist.
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for result in all_results:
        scenario_name = result['scenario']['scenario_name']

        # Create a DataFrame for agent data at the simulation's end
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

        # Create a DataFrame for round-by-round data
        round_data = []
        for round_result in result['results']:
            round_num = round_result['round']
            moves = round_result['moves']
            payoffs = round_result['payoffs']
            for agent_id, move in moves.items():
                payoff = payoffs.get(agent_id,None)
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

# --- Main Execution Block ---
if __name__ == "__main__":
    all_results = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario['scenario_name']}")
        result = run_experiment(scenario)
        all_results.append(result)

    save_results(all_results)

    print("Simulations complete.  Results saved to 'results' directory.")