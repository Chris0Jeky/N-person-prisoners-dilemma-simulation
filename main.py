# main.py
import argparse
import json
import os
import pandas as pd
import time
import logging
import random
import numpy as np

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix, linear_payoff_C, linear_payoff_D
from npdl.core.logging_utils import setup_logging, log_experiment_summary, generate_ascii_chart, log_network_stats

def load_scenarios(file_path):
    """Load scenario definitions from a JSON file."""
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    return scenarios


# --- Modified run_experiment to handle setup and return env details ---
def setup_experiment(scenario, logger):
    """Sets up agents and environment for a scenario run."""
    logger.debug(f"Setting up agents for scenario: {scenario['scenario_name']}")
    agents = []
    agent_id_counter = 0
    for strategy, count in scenario["agent_strategies"].items():
        for _ in range(count):
            agent_params = {
                "agent_id": agent_id_counter,
                "strategy": strategy,
                "memory_length": scenario.get("memory_length", 10),
                "q_init_type": scenario.get("q_init_type", "zero"),
                "max_possible_payoff": scenario.get("payoff_params", {}).get("T", 5.0)  # Pass T for optimistic init
            }

            # Add strategy-specific parameters
            agent_params.update(scenario.get("agent_params", {}).get(strategy, {}))  # Simplified generic param passing

            # Override specific params if needed (example for Q-learning base params)
            if strategy in ["q_learning", "q_learning_adaptive", "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"]:
                agent_params.update({
                    "learning_rate": scenario.get("learning_rate", 0.1),
                    "discount_factor": scenario.get("discount_factor", 0.9),
                    "epsilon": scenario.get("epsilon", 0.1),  # Note: UCB1 ignores this internally
                    "state_type": scenario.get("state_type", "proportion_discretized"),
                })
                # Further specific params (beta, rates, constant etc.)
                if strategy == "lra_q": agent_params.update({"increase_rate": scenario.get("increase_rate", 0.1),
                                                             "decrease_rate": scenario.get("decrease_rate", 0.05)})
                if strategy == "hysteretic_q": agent_params["beta"] = scenario.get("beta", 0.01)
                if strategy == "wolf_phc": agent_params.update(
                    {"alpha_win": scenario.get("alpha_win", 0.05), "alpha_lose": scenario.get("alpha_lose", 0.2),
                     "alpha_avg": scenario.get("alpha_avg", 0.01)})
                if strategy == "ucb1_q": agent_params["exploration_constant"] = scenario.get("exploration_constant",
                                                                                             2.0)

            elif strategy == "generous_tit_for_tat":
                agent_params["generosity"] = scenario.get("generosity", 0.05)
            elif strategy == "pavlov" or strategy == "suspicious_tit_for_tat":
                agent_params["initial_move"] = scenario.get("initial_move", "cooperate")
            elif strategy == "randomprob":
                agent_params["prob_coop"] = scenario.get("prob_coop", 0.5)

            agents.append(Agent(**agent_params))
            agent_id_counter += 1

    # Create payoff matrix
    payoff_type = scenario.get("payoff_type", "linear")
    payoff_params = scenario.get("payoff_params", {})
    payoff_matrix = create_payoff_matrix(
        scenario["num_agents"],
        payoff_type=payoff_type,
        params=payoff_params
    )

    # Calculate Theoretical Global Scores (using the created matrix's C/D lists)
    N_agents = scenario["num_agents"]
    C_payoffs = payoff_matrix['C']
    D_payoffs = payoff_matrix['D']

    # Payoff C(k) means payoff for coop when k *others* cooperate (so k+1 total cooperators)
    # Payoff D(k) means payoff for defect when k *others* cooperate
    max_coop_score_per_agent = C_payoffs[N_agents - 1] if N_agents > 0 else 0
    max_defect_score_per_agent = D_payoffs[0] if N_agents > 0 else 0

    theoretical_scores = {
        "max_cooperation": N_agents * max_coop_score_per_agent,
        "max_defection": N_agents * max_defect_score_per_agent
    }

    # Calculate half-half (approximate for odd N)
    if N_agents > 0:
        n_coop = N_agents // 2
        n_defect = N_agents - n_coop
        # Payoff for a cooperator: C(n_coop - 1) as n_coop-1 others cooperate
        # Payoff for a defector: D(n_coop) as n_coop others cooperate
        c_idx = max(0, n_coop - 1)  # Index for C payoff list
        d_idx = n_coop  # Index for D payoff list

        # Ensure indices are within bounds
        c_idx = min(c_idx, N_agents - 1)
        d_idx = min(d_idx, N_agents - 1)

        score_coop = C_payoffs[c_idx] if n_coop > 0 else 0
        score_defect = D_payoffs[d_idx] if n_defect > 0 else 0

        theoretical_scores["half_half"] = (n_coop * score_coop) + (n_defect * score_defect)
    else:
        theoretical_scores["half_half"] = 0

    # Create environment
    env = Environment(
        agents,
        payoff_matrix,
        network_type=scenario["network_type"],
        network_params=scenario["network_params"],
        logger=logger
    )

    return env, theoretical_scores

def save_results(scenario_name, run_number, agents, round_results, base_filename="experiment_results", results_dir="results", logger=None):
    """Save experiment results for a specific run to CSV files."""
    run_dir = os.path.join(results_dir, scenario_name, f"run_{run_number:02d}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    if logger is None:
        logger = logging.getLogger(__name__)

    # Save agent summary data
    agent_data = []
    for agent in agents:
        q_values_summary = None
        full_q_values_str = None  # Initialize
        if agent.strategy_type in ('q_learning', 'q_learning_adaptive', 'lra_q',
                                   'hysteretic_q', 'wolf_phc', 'ucb1_q'):
            try:
                full_q_values_str = str(agent.q_values)  # Save full table first
                avg_coop_q = 0.0
                avg_defect_q = 0.0
                state_count = 0
                if agent.q_values and isinstance(agent.q_values, dict):
                    for state, actions in agent.q_values.items():
                        # Check if 'actions' is a dict and has the keys
                        if isinstance(actions, dict):
                            avg_coop_q += actions.get("cooperate", 0.0)
                            avg_defect_q += actions.get("defect", 0.0)
                            state_count += 1
                        else:
                            # Log unexpected structure
                            logger.warning(
                                f"Agent {agent.agent_id}: Unexpected Q-value structure for state {state}: {actions}")
                if state_count > 0:
                    avg_coop_q /= state_count
                    avg_defect_q /= state_count
                    q_values_summary = {"avg_cooperate": avg_coop_q, "avg_defect": avg_defect_q}
                else:
                    # Handle case where agent might not have learned anything
                    q_values_summary = {"avg_cooperate": 0.0, "avg_defect": 0.0}
            except Exception as e:
                # Log the error and continue
                logger.error(f"Agent {agent.agent_id}: Error processing Q-values for saving: {e}",
                             exc_info=False)  # exc_info=False to avoid full stack trace unless needed
                q_values_summary = None  # Indicate error
        agent_data.append({
            'scenario_name': scenario_name,
            'run_number': run_number,
            'agent_id': agent.agent_id,
            'strategy': agent.strategy_type,
            'final_score': agent.score,
            'final_q_values_avg': q_values_summary,
            'full_q_values': full_q_values_str
        })
    df_agent = pd.DataFrame(agent_data)
    agent_filename = os.path.join(run_dir, f"{base_filename}_agents.csv")
    df_agent.to_csv(agent_filename, index=False)

    # Save round-by-round data
    round_data = []
    for round_result in round_results['results']:
        round_num = round_result['round']
        moves = round_result['moves']
        payoffs = round_result['payoffs']
        for agent_id, move in moves.items():
            payoff = payoffs.get(agent_id, None)
            agent = next((a for a in round_results['agents'] if a.agent_id == agent_id), None)
            round_data.append({
                'scenario_name': scenario_name,
                'run_number': run_number,
                'round': round_num,
                'agent_id': agent_id,
                'move': move,
                'payoff': payoff,
                'strategy': agent.strategy_type if agent else None
            })
    df_rounds = pd.DataFrame(round_data)
    rounds_filename = os.path.join(run_dir, f"{base_filename}_rounds.csv")
    df_rounds.to_csv(rounds_filename, index=False)

    # Save network structure if environment is available
    if 'environment' in round_result and hasattr(round_result['environment'], 'export_network_structure'):
        network_data = round_result['environment'].export_network_structure()
        network_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_network.json")

        with open(network_filename, 'w') as f:
            json.dump(network_data, f, indent=2, default=str)

def print_comparative_summary(scenario_results_agg, logger=None):
    """Print a comparative summary of all scenario results"""

    if logger is None:
        logger = logging.getLogger(__name__)

    print("\n=== AGGREGATED COMPARATIVE SCENARIO SUMMARY (Avg over runs) ===\n")

    # Extract key metrics
    scenarios = list(scenario_results_agg.keys())
    avg_final_coop_rates = []
    std_final_coop_rates = []
    avg_max_scores = []
    # Winning strategy might be less meaningful when averaged, maybe dominant strategy?

    for scenario_name, agg_data in scenario_results_agg.items():
        avg_final_coop_rates.append(agg_data['avg_final_coop_rate'])
        std_final_coop_rates.append(agg_data['std_final_coop_rate'])
        avg_max_scores.append(agg_data['avg_final_max_score'])  # Assuming max score per run is stored

    print(f"{'Scenario':<35} {'Avg Coop Rate':<15} {'Std Coop Rate':<15} {'Avg Max Score':<15}")
    print("-" * 85)

    for i in range(len(scenarios)):
        scenario_name = scenarios[i]
        # Ensure we have data before trying to access it
        coop_rate_str = f"{avg_final_coop_rates[i]:.3f}" if i < len(avg_final_coop_rates) else "N/A"
        coop_std_str = f"{std_final_coop_rates[i]:.3f}" if i < len(std_final_coop_rates) else "N/A"
        max_score_str = f"{avg_max_scores[i]:.2f}" if i < len(avg_max_scores) else "N/A"

        print(f"{scenario_name:<35} {coop_rate_str:<15} {coop_std_str:<15} {max_score_str:<15}")

    print("\n")

    # Print cooperation rate chart (using averages)
    if avg_final_coop_rates:
        chart = generate_ascii_chart(avg_final_coop_rates,
                                     title="Avg Final Cooperation Rates Across Scenarios",
                                     width=50, height=10)
        print(chart)

    print("\n=== END OF AGGREGATED COMPARATIVE SUMMARY ===\n")

def main():
    parser = argparse.ArgumentParser(description="Run N-person IPD experiments")
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced_scenarios.json instead of scenarios.json')
    parser.add_argument('--scenario_file', type=str, default='scenarios.json',
                        help='Path to the JSON file containing scenario definitions.')
    parser.add_argument('--num_runs', type=int, default=10, # New argument for number of runs
                        help='Number of runs (seeds) per scenario.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save experiment results.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save log files.')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis on results after experiments complete.')
    # Interactive mode removed from this script, keep it separate or use CLI
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    args = parser.parse_args()

    # Set up logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = os.path.join(args.log_dir, "experiment.log")
    logger = setup_logging(log_file, level=log_level)
    
    logger.info("Starting N-person IPD experiments")
    
    # Load scenarios
    if args.enhanced:
        scenario_file = 'enhanced_scenarios.json'
        logger.info("Using enhanced scenarios")
    else:
        scenario_file = args.scenario_file
    
    scenarios = load_scenarios(scenario_file)
    logger.info(f"Loaded {len(scenarios)} scenarios from {scenario_file}")
    
    # Run experiments
    all_results = []
    for scenario in scenarios:
        try:
            result = setup_experiment(scenario, logger)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error running scenario {scenario['scenario_name']}: {e}", exc_info=True)
    
    # Save results
    save_results(all_results, results_dir=args.results_dir)
    logger.info(f"Simulations complete. Results saved to '{args.results_dir}' directory.")
    
    # Print comparative summary
    print_comparative_summary(all_results)
    
    # Run analysis if requested
    if args.analyze:
        try:
            from npdl.analysis.analysis import analyze_multiple_scenarios
            logger.info("Running analysis on experiment results")
            scenario_names = [result['scenario']['scenario_name'] for result in all_results]
            analyze_multiple_scenarios(scenario_names, args.results_dir, "analysis_results")
            logger.info("Analysis complete. Results saved to 'analysis_results' directory.")
        except Exception as e:
            logger.error(f"Error running analysis: {e}", exc_info=True)

if __name__ == "__main__":
    main()