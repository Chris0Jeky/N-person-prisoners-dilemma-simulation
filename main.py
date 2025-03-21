# main.py
import argparse
import json
import os
import pandas as pd
import time

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
            
            # Add strategy-specific parameters
            if strategy == "q_learning" or strategy == "q_learning_adaptive":
                agent_params.update({
                    "learning_rate": scenario.get("learning_rate", 0.1),
                    "discount_factor": scenario.get("discount_factor", 0.9),
                    "epsilon": scenario.get("epsilon", 0.1),
                })
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
    results = env.run_simulation(scenario["num_rounds"], logging_interval)
    
    # Log summary statistics
    log_experiment_summary(scenario, results, agents, logger)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"Completed scenario: {scenario['scenario_name']} in {execution_time:.2f} seconds")
    
    return {"scenario": scenario, "results": results, "agents": agents}

def save_results(all_results, base_filename="experiment_results", results_dir="results"):
    """Save experiment results to CSV files."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for result in all_results:
        scenario_name = result['scenario']['scenario_name']

        # Save agent summary data
        agent_data = []
        for agent in result['agents']:
            agent_data.append({
                'scenario_name': scenario_name,
                'agent_id': agent.agent_id,
                'strategy': agent.strategy_type,
                'final_score': agent.score,
                'final_q_values': agent.q_values if agent.strategy_type in ('q_learning', 'q_learning_adaptive') else None,
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
        q_learning_agents = [a for a in result['agents'] if a.strategy_type in ("q_learning", "q_learning_adaptive")]
        if q_learning_agents:
            avg_q_coop = sum(a.q_values["cooperate"] for a in q_learning_agents) / len(q_learning_agents)
            avg_q_defect = sum(a.q_values["defect"] for a in q_learning_agents) / len(q_learning_agents)
            
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
    scenarios = load_scenarios(args.scenario_file)
    logger.info(f"Loaded {len(scenarios)} scenarios from {args.scenario_file}")
    
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