#!/usr/bin/env python3
# run_scenario_generator.py
import argparse
import json
import os
import itertools
import random
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Tuple

from main import setup_experiment, save_results  # Re-use setup and saving
from npdl.core.logging_utils import setup_logging

# --- Configuration for Scenario Generation ---

# Define pools of options to combine
AGENT_STRATEGY_POOL = [
    "hysteretic_q", "wolf_phc", "lra_q",  # Learning strategies
    "q_learning", "q_learning_adaptive", "ucb1_q",
    "tit_for_tat", "tit_for_two_tats", "pavlov",  # Reactive strategies
    "generous_tit_for_tat", "suspicious_tit_for_tat",
    "always_cooperate", "always_defect", "random"  # Fixed/Simple strategies
]

NETWORK_POOL = [
    ("small_world", {"k": 4, "beta": 0.3}),
    ("small_world", {"k": 6, "beta": 0.2}),
    ("scale_free", {"m": 2}),
    ("scale_free", {"m": 3}),
    ("fully_connected", {}),
    ("random", {"probability": 0.4}),
    ("random", {"probability": 0.7})
]

INTERACTION_MODE_POOL = ["neighborhood"]
NUM_AGENTS_POOL = [20, 30, 40, 50]
NUM_ROUNDS_POOL = [200, 300, 500]  # Use shorter rounds for evaluation initially
STATE_TYPE_POOL = ["proportion_discretized", "memory_enhanced", "count"]
MEMORY_LENGTH_POOL = [3, 5, 10, 20]
PAYOFF_PARAMS_POOL = [
    {"R": 3, "S": 0, "T": 5, "P": 1},  # Standard PD
    {"R": 3, "S": 0, "T": 6, "P": 1},  # Higher Temptation
    {"R": 4, "S": 0, "T": 5, "P": 1},  # Higher Reward
    {"R": 3, "S": -1, "T": 5, "P": 0}  # Negative Sucker's payoff
]

LEARNING_RATE_POOL = [0.05, 0.1, 0.2]
DISCOUNT_FACTOR_POOL = [0.8, 0.9, 0.99]
EPSILON_POOL = [0.05, 0.1, 0.2, 0.3]
Q_INIT_TYPE_POOL = ["zero", "optimistic", "random"]


# --- Evaluation Metric ---
def evaluate_scenario_run(env, round_results) -> Dict[str, float]:
    """Calculate metrics to judge if a scenario run is 'interesting'."""
    metrics = {
        'final_coop_rate': np.nan, 
        'coop_rate_change': np.nan,
        'score_variance': np.nan, 
        'strategy_dominance': np.nan,
        'network_clustering': np.nan
    }
    
    if not round_results or len(round_results) < 2:
        return metrics

    # 1. Final Cooperation Rate
    last_round = round_results[-1]
    if last_round['moves']:
        coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
        metrics['final_coop_rate'] = coop_count / len(last_round['moves'])
    
    # 2. Cooperation Rate Change (dynamics)
    first_round = round_results[0]
    if first_round['moves'] and last_round['moves']:
        first_coop_count = sum(1 for move in first_round['moves'].values() if move == "cooperate")
        first_coop_rate = first_coop_count / len(first_round['moves'])
        metrics['coop_rate_change'] = metrics['final_coop_rate'] - first_coop_rate

    # 3. Variance in Final Scores Across Strategies
    final_scores_by_strategy = {}
    for agent in env.agents:
        if agent.strategy_type not in final_scores_by_strategy:
            final_scores_by_strategy[agent.strategy_type] = []
        final_scores_by_strategy[agent.strategy_type].append(agent.score)

    avg_scores = [np.mean(scores) for scores in final_scores_by_strategy.values() if scores]
    if len(avg_scores) > 1:
        metrics['score_variance'] = np.var(avg_scores)
    else:
        metrics['score_variance'] = 0.0  # No variance if only one strategy type

    # 4. Strategy dominance (max score difference)
    if len(avg_scores) > 1:
        metrics['strategy_dominance'] = max(avg_scores) - min(avg_scores)
    
    # 5. Network characteristics
    try:
        metrics['network_clustering'] = nx.average_clustering(env.graph)
    except:
        metrics['network_clustering'] = 0.0

    return metrics


# --- Scenario Generation ---
def generate_random_scenario(scenario_id: int) -> Dict[str, Any]:
    """Generates a single random scenario configuration."""
    scenario = {}
    scenario['scenario_name'] = f"GeneratedScenario_{scenario_id:04d}"

    # Choose core parameters
    scenario['num_agents'] = random.choice(NUM_AGENTS_POOL)
    scenario['num_rounds'] = random.choice(NUM_ROUNDS_POOL)
    net_type, net_params = random.choice(NETWORK_POOL)
    scenario['network_type'] = net_type
    scenario['network_params'] = net_params
    scenario['interaction_mode'] = random.choice(INTERACTION_MODE_POOL)

    # Choose agent strategies (ensure at least 2 types, total num_agents)
    num_strategies = random.randint(2, 4)  # Mix 2-4 strategies
    chosen_strategies = random.sample(AGENT_STRATEGY_POOL, num_strategies)
    agent_counts = {}
    agents_assigned = 0
    for i, strat in enumerate(chosen_strategies):
        if i < num_strategies - 1:
            # Assign random count, ensuring at least 1 agent per strategy
            max_possible = scenario['num_agents'] - agents_assigned - (num_strategies - 1 - i)
            count = random.randint(1, max(1, max_possible // (num_strategies - i)))
        else:
            # Assign remaining agents to the last strategy
            count = scenario['num_agents'] - agents_assigned
        agent_counts[strat] = count
        agents_assigned += count
    scenario['agent_strategies'] = agent_counts

    # Choose other parameters
    scenario['payoff_type'] = "linear"  # Keep simple for now
    scenario['payoff_params'] = random.choice(PAYOFF_PARAMS_POOL)
    scenario['state_type'] = random.choice(STATE_TYPE_POOL)
    scenario['memory_length'] = random.choice(MEMORY_LENGTH_POOL)
    scenario['q_init_type'] = random.choice(Q_INIT_TYPE_POOL)

    # Learning parameters
    scenario['learning_rate'] = random.choice(LEARNING_RATE_POOL)
    scenario['discount_factor'] = random.choice(DISCOUNT_FACTOR_POOL)
    scenario['epsilon'] = random.choice(EPSILON_POOL)
    
    # Add special parameters for certain strategies
    if "hysteretic_q" in agent_counts:
        scenario['beta'] = round(random.uniform(0.005, 0.05), 4)
    if "lra_q" in agent_counts:
        scenario['increase_rate'] = round(random.uniform(0.05, 0.2), 3)
        scenario['decrease_rate'] = round(random.uniform(0.01, 0.1), 3)
    if "wolf_phc" in agent_counts:
        scenario['alpha_win'] = round(random.uniform(0.01, 0.1), 3)
        scenario['alpha_lose'] = round(random.uniform(0.1, 0.3), 3)
    if "ucb1_q" in agent_counts:
        scenario['exploration_constant'] = round(random.uniform(1.0, 3.0), 1)
    if "generous_tit_for_tat" in agent_counts:
        scenario['generosity'] = round(random.uniform(0.05, 0.2), 2)

    # Optional enhancements (less frequent)
    if random.random() < 0.3:  # 30% chance
        scenario['use_global_bonus'] = True
    if random.random() < 0.2:  # 20% chance
        scenario['rewiring_interval'] = random.choice([20, 50, 100])
        scenario['rewiring_prob'] = round(random.uniform(0.05, 0.2), 2)

    scenario['logging_interval'] = scenario['num_rounds'] + 1  # Disable logging during evaluation

    return scenario


def save_scenario_metadata(scenario_list, file_path="generated_scenarios_metadata.json"):
    """Save metadata about the generated scenarios for later analysis."""
    meta_data = {
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scenarios": len(scenario_list),
        "parameters": {
            "agent_strategies": list(set(strat for s in scenario_list for strat in s["config"]["agent_strategies"].keys())),
            "network_types": list(set(s["config"]["network_type"] for s in scenario_list)),
            "interaction_modes": list(set(s["config"]["interaction_mode"] for s in scenario_list))
        },
        "scenarios": [
            {
                "name": s["config"]["scenario_name"],
                "metrics": s["metrics"],
                "selection_score": s.get("selection_score", 0),
                "config_summary": {
                    "strategies": s["config"]["agent_strategies"],
                    "network": s["config"]["network_type"],
                    "interaction": s["config"]["interaction_mode"],
                    "agents": s["config"]["num_agents"],
                    "rounds": s["config"]["num_rounds"]
                }
            } 
            for s in scenario_list
        ]
    }
    
    with open(file_path, 'w') as f:
        json.dump(meta_data, f, indent=2)


# --- Main Generator Function ---
def run_scenario_generation(num_scenarios_to_generate: int,
                            num_eval_runs: int,
                            num_save_runs: int,
                            top_n_to_save: int,
                            results_dir: str,
                            log_level_str: str = 'INFO'):
    """Generates, evaluates, and saves interesting scenarios."""

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    gen_logger = setup_logging(level=log_level, console=True, log_file="scenario_generator.log")

    gen_logger.info(f"Starting Scenario Generation & Evaluation")
    gen_logger.info(f"Generating {num_scenarios_to_generate} scenarios...")
    gen_logger.info(f"Evaluating each with {num_eval_runs} runs...")
    gen_logger.info(f"Saving top {top_n_to_save} scenarios with {num_save_runs} runs each.")

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    evaluated_scenarios = []
    start_gen_time = time.time()

    for i in range(num_scenarios_to_generate):
        scenario_config = generate_random_scenario(i)
        gen_logger.info(f"Evaluating Scenario {i+1}/{num_scenarios_to_generate}: {scenario_config['scenario_name']}")

        run_metrics_list = []
        for run_num in range(num_eval_runs):
            seed = i * num_eval_runs + run_num  # Unique seed
            random.seed(seed)
            np.random.seed(seed)

            silent_logger = logging.getLogger(f'silent_run_{i}_{run_num}')
            silent_logger.setLevel(logging.ERROR)
            try:
                env, _ = setup_experiment(scenario_config, silent_logger)
                round_results = env.run_simulation(
                    scenario_config["num_rounds"],
                    logging_interval=scenario_config["num_rounds"] + 1,  # Disable logging during evaluation
                    use_global_bonus=scenario_config.get("use_global_bonus", False),
                    rewiring_interval=scenario_config.get("rewiring_interval", 0),
                    rewiring_prob=scenario_config.get("rewiring_prob", 0.0)
                )
                metrics = evaluate_scenario_run(env, round_results)
                run_metrics_list.append(metrics)
            except Exception as e:
                gen_logger.error(f"  Error evaluating run {run_num} for {scenario_config['scenario_name']}: {e}")

        # Aggregate evaluation metrics
        if run_metrics_list:
            avg_eval_metrics = {}
            valid_metrics = [m for m in run_metrics_list if isinstance(m, dict)]
            if valid_metrics:
                keys = valid_metrics[0].keys()
                for key in keys:
                    values = [m[key] for m in valid_metrics if key in m and not np.isnan(m[key])]
                    avg_eval_metrics[f'avg_{key}'] = np.mean(values) if values else np.nan
                evaluated_scenarios.append({"config": scenario_config, "metrics": avg_eval_metrics})
            else:
                gen_logger.warning(f"  No valid evaluation metrics for {scenario_config['scenario_name']}")
        else:
            gen_logger.warning(f"  No evaluation runs succeeded for {scenario_config['scenario_name']}")

    gen_logger.info(f"Finished evaluating {len(evaluated_scenarios)} scenarios.")

    if not evaluated_scenarios:
        gen_logger.error("No scenarios were successfully evaluated. Exiting.")
        return

    # --- Selection Criteria ---
    def calculate_interestingness_score(eval_result):
        """Calculate an 'interestingness' score based on multiple metrics."""
        # Extract metrics (handle potential NaN values)
        metrics = eval_result['metrics']
        
        # Get normalized metrics (0-1 range)
        coop = metrics.get('avg_final_coop_rate', 0)
        coop = 0 if pd.isna(coop) else coop
        
        # Absolute change in cooperation rate (dynamics are interesting)
        coop_change = abs(metrics.get('avg_coop_rate_change', 0))
        coop_change = 0 if pd.isna(coop_change) else min(coop_change, 1.0)  # Cap at 1.0
        
        # Variance in scores across strategies (interesting if different strategies have different outcomes)
        variance = metrics.get('avg_score_variance', 0) 
        variance = 0 if pd.isna(variance) else min(variance / 1000, 1.0)  # Normalize and cap
        
        # Strategy dominance (difference between best and worst strategy)
        dominance = metrics.get('avg_strategy_dominance', 0)
        dominance = 0 if pd.isna(dominance) else min(dominance / 100, 1.0)  # Normalize and cap
        
        # Network clustering (more interesting if there's clustering)
        clustering = metrics.get('avg_network_clustering', 0)
        clustering = 0 if pd.isna(clustering) else clustering
        
        # Avoid extreme cases (either all cooperation or all defection)
        extreme_penalty = 0.5 * (abs(coop - 0.5) ** 2)
        
        # Weighted combination of metrics
        # Higher weights for dynamics and variance, penalize extreme cooperation rates
        score = (
            0.2 * (1 - extreme_penalty) +  # Prefer intermediate cooperation rates
            0.3 * coop_change +            # Reward dynamics (changes over time)
            0.3 * variance +               # Reward variance across strategies
            0.15 * dominance +             # Reward clear strategy dominance
            0.05 * clustering              # Small reward for network clustering
        )
        
        return score

    # Apply the scoring function and sort
    for scenario in evaluated_scenarios:
        scenario["selection_score"] = calculate_interestingness_score(scenario)
    
    evaluated_scenarios.sort(key=lambda x: x["selection_score"], reverse=True)

    # Save metadata about all evaluated scenarios
    save_scenario_metadata(evaluated_scenarios, file_path=os.path.join(results_dir, "generated_scenarios_metadata.json"))

    gen_logger.info(f"\n--- Top {top_n_to_save} Evaluated Scenarios ---")
    for k in range(min(top_n_to_save, len(evaluated_scenarios))):
        scenario = evaluated_scenarios[k]
        gen_logger.info(f"Rank {k+1}: {scenario['config']['scenario_name']} (Score: {scenario['selection_score']:.3f})")
        gen_logger.info(f"  Metrics: {scenario['metrics']}")
        gen_logger.info(f"  Config: {scenario['config']['agent_strategies']}, "
                        f"{scenario['config']['network_type']}, "
                        f"Mode: {scenario['config']['interaction_mode']}")

    # --- Run and Save Selected Scenarios ---
    gen_logger.info(f"\nRunning full simulations ({num_save_runs} runs) and saving results for top {top_n_to_save} scenarios...")
    saved_count = 0
    
    # Save the full configurations of selected scenarios for later reference
    selected_configs = [evaluated_scenarios[k]['config'] for k in range(min(top_n_to_save, len(evaluated_scenarios)))]
    with open(os.path.join(results_dir, "selected_scenarios.json"), 'w') as f:
        json.dump(selected_configs, f, indent=2)
    
    for k in range(min(top_n_to_save, len(evaluated_scenarios))):
        selected_scenario = evaluated_scenarios[k]['config']
        scenario_name = selected_scenario['scenario_name']
        gen_logger.info(f"Running & Saving: {scenario_name} ({k+1}/{top_n_to_save})")

        # Set a reasonable logging interval for the final runs
        selected_scenario['logging_interval'] = max(1, selected_scenario["num_rounds"] // 20)

        # Run multiple times and save results
        for run_number in range(num_save_runs):
            seed = (k + num_scenarios_to_generate) * num_save_runs + run_number  # Unique seed
            random.seed(seed)
            np.random.seed(seed)

            silent_logger = logging.getLogger(f'silent_save_{k}_{run_number}')
            silent_logger.setLevel(logging.ERROR)
            try:
                env, theoretical_scores = setup_experiment(selected_scenario, silent_logger)
                round_results = env.run_simulation(
                    selected_scenario["num_rounds"],
                    logging_interval=selected_scenario["logging_interval"],
                    use_global_bonus=selected_scenario.get("use_global_bonus", False),
                    rewiring_interval=selected_scenario.get("rewiring_interval", 0),
                    rewiring_prob=selected_scenario.get("rewiring_prob", 0.0)
                )
                # Save full results
                save_results(scenario_name, run_number, env.agents, round_results,
                             results_dir=results_dir, logger=gen_logger, env=env)
            except Exception as e:
                gen_logger.error(f"  ERROR during save run {run_number} for {scenario_name}: {e}")
        saved_count += 1

    end_gen_time = time.time()
    gen_logger.info(f"\nScenario Generation & Selection complete. Saved {saved_count} scenarios.")
    gen_logger.info(f"Total time: {end_gen_time - start_gen_time:.2f} seconds.")
    gen_logger.info(f"Results saved in: {results_dir}")
    gen_logger.info(f"Metadata saved as: {os.path.join(results_dir, 'generated_scenarios_metadata.json')}")
    gen_logger.info(f"Selected configurations saved as: {os.path.join(results_dir, 'selected_scenarios.json')}")
    
    return evaluated_scenarios


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, evaluate, and save interesting N-IPD scenarios.")
    parser.add_argument('--num_generate', type=int, default=50,
                        help='Number of random scenarios to generate and evaluate.')
    parser.add_argument('--eval_runs', type=int, default=3,
                        help='Number of runs for the initial evaluation phase.')
    parser.add_argument('--save_runs', type=int, default=10,
                        help='Number of runs for the final saving phase of selected scenarios.')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top scenarios to save results for.')
    parser.add_argument('--results_dir', type=str, default='results/generated_scenarios',
                        help='Base directory to save the final results of selected scenarios.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level for the generator script.')
    args = parser.parse_args()

    run_scenario_generation(
        num_scenarios_to_generate=args.num_generate,
        num_eval_runs=args.eval_runs,
        num_save_runs=args.save_runs,
        top_n_to_save=args.top_n,
        results_dir=args.results_dir,
        log_level_str=args.log_level
    )
