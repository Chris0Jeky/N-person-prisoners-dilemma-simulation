# main.py
import argparse
import json
import os
import pandas as pd
import time
import logging

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix
from npdl.core.logging_utils import setup_logging, log_experiment_summary, generate_ascii_chart

def load_scenarios(file_path):
    """Load scenario definitions from a JSON file."""
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    return scenarios

def run_experiment(scenario, logger):
    """Run a single experiment based on the given scenario."""
    logger.info(f"Running scenario: {scenario['scenario_name']}")
    start_time = time.time()
    
    # Create agents based on strategy configuration
    agents = []
    agent_id_counter = 0
    for strategy, count in scenario["agent_strategies"].items():
        for _ in range(count):
            agent_params = {"agent_id": agent_id_counter, "strategy": strategy}
            
            # Add common parameters for all agents
            memory_length = scenario.get("memory_length", 10)
            agent_params["memory_length"] = memory_length
            agent_params["q_init_type"] = scenario.get("q_init_type", "zero")

            # Determine max possible payoff (T value for optimistic init)
            payoff_params_for_init = scenario.get("payoff_params", {})
            max_payoff = payoff_params_for_init.get("T", 5.0)  # Default to T=5 if not specified
            agent_params["max_possible_payoff"] = max_payoff

            # Add strategy-specific parameters
            if strategy in ["q_learning", "q_learning_adaptive", "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"]:
                # Common Q-learning parameters
                agent_params.update({
                    "learning_rate": scenario.get("learning_rate", 0.1),
                    "discount_factor": scenario.get("discount_factor", 0.9),
                    "epsilon": scenario.get("epsilon", 0.1),
                    "state_type": scenario.get("state_type", "proportion_discretized"),
                })
                
                # Strategy-specific params
                if strategy == "lra_q":
                    agent_params.update({
                        "increase_rate": scenario.get("increase_rate", 0.1),
                        "decrease_rate": scenario.get("decrease_rate", 0.05)
                    })
                elif strategy == "hysteretic_q":
                    agent_params["beta"] = scenario.get("beta", 0.01)
                elif strategy == "wolf_phc":
                    agent_params.update({
                        "alpha_win": scenario.get("alpha_win", 0.05),
                        "alpha_lose": scenario.get("alpha_lose", 0.2),
                        "alpha_avg": scenario.get("alpha_avg", 0.01)
                    })
                elif strategy == "ucb1_q":
                    agent_params["exploration_constant"] = scenario.get("exploration_constant", 2.0)
            
            elif strategy == "generous_tit_for_tat":
                agent_params["generosity"] = scenario.get("generosity", 0.05)
            elif strategy == "pavlov" or strategy == "suspicious_tit_for_tat":
                agent_params["initial_move"] = scenario.get("initial_move", "cooperate")
            elif strategy == "randomprob":
                agent_params["prob_coop"] = scenario.get("prob_coop", 0.5)

            agents.append(Agent(**agent_params))
            agent_id_counter += 1
    
    # Create payoff matrix with the specified payoff function
    payoff_type = scenario.get("payoff_type", "linear")
    payoff_params = scenario.get("payoff_params", {})
    payoff_matrix = create_payoff_matrix(
        scenario["num_agents"], 
        payoff_type=payoff_type,
        params=payoff_params
    )
    
    # Create environment
    env = Environment(
        agents,
        payoff_matrix,
        network_type=scenario["network_type"],
        network_params=scenario["network_params"],
        logger=logger
    )
    
    # Run simulation
    logging_interval = scenario.get("logging_interval", 10)
    use_global_bonus = scenario.get("use_global_bonus", False)
    rewiring_interval = scenario.get("rewiring_interval", 0)
    rewiring_prob = scenario.get("rewiring_prob", 0.0)
    
    # Run simulation with enhanced parameters
    results = env.run_simulation(
        scenario["num_rounds"], 
        logging_interval=logging_interval,
        use_global_bonus=use_global_bonus,
        rewiring_interval=rewiring_interval,
        rewiring_prob=rewiring_prob
    )
    
    # Log summary statistics
    log_experiment_summary(scenario, results, agents, logger)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"Completed scenario: {scenario['scenario_name']} in {execution_time:.2f} seconds")
    
    return {"scenario": scenario, "results": results, "agents": agents, "environment": env}

def save_results(all_results, base_filename="experiment_results", results_dir="results", logger=None):
    """Save experiment results to CSV files."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for result in all_results:
        scenario_name = result['scenario']['scenario_name']

        # Save agent summary data
        agent_data = []
        for agent in result['agents']:
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
                'agent_id': agent.agent_id,
                'strategy': agent.strategy_type,
                'final_score': agent.score,
                'final_q_values_avg': q_values_summary,  # Renamed for clarity
                'full_q_values': full_q_values_str
            })
        df_agent = pd.DataFrame(agent_data)
        agent_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_agents.csv")
        df_agent.to_csv(agent_filename, index=False)
        
        # Save round-by-round data
        round_data = []
        for round_result in result['results']:
            round_num = round_result['round']
            moves = round_result['moves']
            payoffs = round_result['payoffs']
            for agent_id, move in moves.items():
                payoff = payoffs.get(agent_id, None)
                agent = next((a for a in result['agents'] if a.agent_id == agent_id), None)
                round_data.append({
                    'scenario_name': scenario_name,
                    'round': round_num,
                    'agent_id': agent_id,
                    'move': move,
                    'payoff': payoff,
                    'strategy': agent.strategy_type if agent else None
                })
        df_rounds = pd.DataFrame(round_data)
        rounds_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_rounds.csv")
        df_rounds.to_csv(rounds_filename, index=False)
        
        # Save network structure if environment is available
        if 'environment' in result and hasattr(result['environment'], 'export_network_structure'):
            network_data = result['environment'].export_network_structure()
            network_filename = os.path.join(results_dir, f"{base_filename}_{scenario_name}_network.json")
            
            with open(network_filename, 'w') as f:
                json.dump(network_data, f, indent=2, default=str)

def print_comparative_summary(all_results):
    """Print a comparative summary of all scenario results"""
    print("\n=== COMPARATIVE SCENARIO SUMMARY ===\n")
    
    # Extract key metrics
    scenarios = []
    final_coop_rates = []
    max_scores = []
    winning_strategies = []
    q_learning_preferences = []
    
    for result in all_results:
        scenario = result['scenario']
        scenario_name = scenario['scenario_name']
        scenarios.append(scenario_name)
        
        # Calculate final cooperation rate
        last_round = result['results'][-1]
        coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
        coop_rate = coop_count / len(last_round['moves'])
        final_coop_rates.append(coop_rate)
        
        # Find highest scoring strategy
        strategies = {}
        for agent in result['agents']:
            if agent.strategy_type not in strategies:
                strategies[agent.strategy_type] = []
            strategies[agent.strategy_type].append(agent)
        
        if strategies:
            avg_scores = [(strategy, sum(a.score for a in agents) / len(agents)) 
                          for strategy, agents in strategies.items()]
            winning_strategy = max(avg_scores, key=lambda x: x[1])
            winning_strategies.append(winning_strategy[0])
            max_scores.append(winning_strategy[1])
        else:
            winning_strategies.append("N/A")
            max_scores.append(0)
        
        # Analyze Q-learning outcomes
        q_learning_agents = [a for a in result['agents'] if a.strategy_type in (
            "q_learning", "q_learning_adaptive", "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"
        )]
        if q_learning_agents:
            # Handle complex Q-value structure (state-based)
            avg_q_coop = 0
            avg_q_defect = 0
            q_count = 0
            
            for agent in q_learning_agents:
                for state in agent.q_values:
                    if "cooperate" in agent.q_values[state] and "defect" in agent.q_values[state]:
                        avg_q_coop += agent.q_values[state]["cooperate"]
                        avg_q_defect += agent.q_values[state]["defect"]
                        q_count += 1
            
            if q_count > 0:
                avg_q_coop /= q_count
                avg_q_defect /= q_count
            
            if avg_q_defect > avg_q_coop + 3.0:
                preference = "Defect"
            elif avg_q_coop > avg_q_defect + 3.0:
                preference = "Cooperate"
            else:
                preference = "Mixed"
            
            q_learning_preferences.append(preference)
        else:
            q_learning_preferences.append("N/A")
    
    # Print the comparative table
    print(f"{'Scenario':<30} {'Coop Rate':<10} {'Max Score':<10} {'Winner':<20} {'Q-Learning':<10}")
    print("-" * 80)
    
    for i in range(len(scenarios)):
        print(f"{scenarios[i]:<30} {final_coop_rates[i]:.2f}       {max_scores[i]:<10.2f} {winning_strategies[i]:<20} {q_learning_preferences[i]:<10}")
    
    print("\n")
    
    # Print cooperation rate chart
    if final_coop_rates:
        chart = generate_ascii_chart(final_coop_rates, title="Cooperation Rates Across Scenarios", width=40, height=10)
        print(chart)
    
    print("\n=== END OF COMPARATIVE SUMMARY ===\n")

def main():
    parser = argparse.ArgumentParser(description="Run N-person IPD experiments")
    # Add a new argument for enhanced scenarios
    parser.add_argument('--enhanced', action='store_true', 
                        help='Use enhanced_scenarios.json instead of scenarios.json')
    parser.add_argument('--scenario_file', type=str, default='scenarios.json',
                        help='Path to the JSON file containing scenario definitions.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save experiment results.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save log files.')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis on results after experiments complete.')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (play against AI agents).')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging.')
    args = parser.parse_args()
    
    # Run in interactive mode if requested
    if args.interactive:
        try:
            from interactive_game import main as interactive_main
            interactive_main()
            return
        except ImportError:
            print("Error: interactive_game module not found.")
            return

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
            result = run_experiment(scenario, logger)
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