#!/usr/bin/env python3
"""
Advanced Scenario Generator with:
1. Parallel processing for faster evaluation
2. Evolutionary algorithm for iterative improvement
3. Enhanced interestingness metrics

This script extends the basic scenario generator with more sophisticated
approaches to discover and refine interesting scenarios.
"""
import argparse
import json
import os
import random
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable
from functools import partial
import multiprocessing
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from scipy import stats

from main import setup_experiment, save_results
from npdl.core.logging_utils import setup_logging
from run_scenario_generator import (
    AGENT_STRATEGY_POOL, NETWORK_POOL, INTERACTION_MODE_POOL, 
    NUM_AGENTS_POOL, NUM_ROUNDS_POOL, STATE_TYPE_POOL, 
    MEMORY_LENGTH_POOL, PAYOFF_PARAMS_POOL,
    LEARNING_RATE_POOL, DISCOUNT_FACTOR_POOL, EPSILON_POOL,
    Q_INIT_TYPE_POOL, generate_random_scenario,
    save_scenario_metadata
)


# --- Enhanced Evaluation Metrics ---
def calculate_enhanced_metrics(env, round_results) -> Dict[str, float]:
    """Calculate enhanced metrics to judge if a scenario run is 'interesting'."""
    metrics = {
        # Basic metrics (from original implementation)
        'final_coop_rate': np.nan, 
        'coop_rate_change': np.nan,
        'score_variance': np.nan, 
        'strategy_dominance': np.nan,
        'network_clustering': np.nan,
        
        # Enhanced metrics
        'coop_volatility': np.nan,           # Measure of cooperation rate fluctuation
        'strategy_adaptation_rate': np.nan,   # How quickly learning strategies adapt
        'learning_convergence': np.nan,       # Whether Q-values converge
        'social_welfare': np.nan,             # Overall system wellbeing
        'strategy_countering': np.nan,        # Whether strategies effectively counter others
        'pattern_complexity': np.nan,         # Complexity of cooperation patterns
        'equilibrium_stability': np.nan       # Stability in final rounds
    }
    
    if not round_results or len(round_results) < 5:
        return metrics

    # Extract data
    coop_rates = []
    strategy_scores = defaultdict(lambda: defaultdict(list))  # round -> strategy -> scores
    q_differences = []  # Track Q-value differences for learning agents
    
    # Process round results
    for round_idx, round_data in enumerate(round_results):
        # Calculate cooperation rate for this round
        if round_data['moves']:
            coop_count = sum(1 for move in round_data['moves'].values() if move == "cooperate")
            coop_rates.append(coop_count / len(round_data['moves']))
        
        # Track scores by strategy
        for agent in env.agents:
            agent_id = agent.agent_id
            if agent_id in round_data['payoffs']:
                strategy_scores[round_idx][agent.strategy_type].append(round_data['payoffs'][agent_id])
        
        # Track Q-value changes for learning agents (if available)
        if round_idx > 0 and hasattr(env, 'q_value_history'):
            for agent_id, history in env.q_value_history.items():
                if len(history) > round_idx:
                    prev_q = history[round_idx-1]
                    curr_q = history[round_idx]
                    # Calculate average absolute difference in Q-values
                    if prev_q and curr_q:
                        diff = np.mean([abs(curr_q.get(s, {}).get(a, 0) - prev_q.get(s, {}).get(a, 0)) 
                                      for s in set(curr_q.keys()) | set(prev_q.keys())
                                      for a in ['cooperate', 'defect']])
                        q_differences.append(diff)
    
    # 1. Basic Metrics (from original implementation)
    if coop_rates:
        metrics['final_coop_rate'] = coop_rates[-1]
        metrics['coop_rate_change'] = coop_rates[-1] - coop_rates[0] if len(coop_rates) > 1 else 0
    
    # Calculate variance in final scores across strategies
    final_round_idx = len(round_results) - 1
    if final_round_idx in strategy_scores:
        avg_scores = [np.mean(scores) for scores in strategy_scores[final_round_idx].values() if scores]
        if len(avg_scores) > 1:
            metrics['score_variance'] = np.var(avg_scores)
            metrics['strategy_dominance'] = max(avg_scores) - min(avg_scores)
    
    # Network clustering (from original implementation)
    try:
        metrics['network_clustering'] = nx.average_clustering(env.graph)
    except:
        metrics['network_clustering'] = 0.0
    
    # 2. Enhanced Metrics
    
    # 2.1 Cooperation Volatility: Measure of how much cooperation rates fluctuate
    if len(coop_rates) > 5:
        # Calculate rolling standard deviation with window of 5 rounds
        rolling_std = pd.Series(coop_rates).rolling(5).std().dropna().mean()
        metrics['coop_volatility'] = rolling_std
    
    # 2.2 Strategy Adaptation Rate: How quickly learning strategies change their behavior
    if q_differences:
        metrics['strategy_adaptation_rate'] = np.mean(q_differences)
    
    # 2.3 Learning Convergence: Whether Q-values stabilize by the end
    if len(q_differences) > 10:
        early_diffs = q_differences[:len(q_differences)//2]
        late_diffs = q_differences[len(q_differences)//2:]
        # Ratio of early to late differences (higher means more convergence)
        if np.mean(late_diffs) > 0:
            metrics['learning_convergence'] = np.mean(early_diffs) / np.mean(late_diffs)
    
    # 2.4 Social Welfare: Total payoff across all agents (average per agent)
    total_final_payoff = sum(sum(payoffs) for payoffs in strategy_scores[final_round_idx].values())
    num_agents = len(env.agents)
    if num_agents > 0:
        metrics['social_welfare'] = total_final_payoff / num_agents
    
    # 2.5 Strategy Countering: Whether strategies effectively counter others
    # (measured by rank correlation of strategy scores over time)
    if len(strategy_scores) > 10:
        mid_round = len(strategy_scores) // 2
        late_round = len(strategy_scores) - 1
        
        # Get strategies present in both rounds
        common_strategies = set(strategy_scores[mid_round].keys()) & set(strategy_scores[late_round].keys())
        
        if common_strategies:
            # Calculate average scores for each strategy at mid and late rounds
            mid_scores = {s: np.mean(strategy_scores[mid_round][s]) for s in common_strategies}
            late_scores = {s: np.mean(strategy_scores[late_round][s]) for s in common_strategies}
            
            # Convert to lists maintaining order
            strats = list(common_strategies)
            mid_values = [mid_scores[s] for s in strats]
            late_values = [late_scores[s] for s in strats]
            
            # Calculate rank correlation
            if len(strats) > 1:
                try:
                    correlation, _ = stats.spearmanr(mid_values, late_values)
                    # Invert so that higher values mean more countering (rank changes)
                    metrics['strategy_countering'] = 1.0 - abs(correlation)
                except:
                    pass
    
    # 2.6 Pattern Complexity: Complexity of cooperation patterns
    if len(coop_rates) > 10:
        # Use approximate entropy as a measure of complexity/randomness
        try:
            from entropy import app_entropy
            metrics['pattern_complexity'] = app_entropy(coop_rates, 2)
        except ImportError:
            # Fallback: use standard deviation as a simple proxy for complexity
            metrics['pattern_complexity'] = np.std(coop_rates)
    
    # 2.7 Equilibrium Stability: Whether system stabilizes in final rounds
    if len(coop_rates) > 10:
        # Compare variance in first half vs second half of simulation
        first_half = coop_rates[:len(coop_rates)//2]
        second_half = coop_rates[len(coop_rates)//2:]
        
        if np.var(first_half) > 0:
            # Ratio of second half variance to first half (lower means more stable)
            stability_ratio = np.var(second_half) / np.var(first_half)
            metrics['equilibrium_stability'] = 1.0 - min(1.0, stability_ratio)
    
    return metrics


def calculate_enhanced_interestingness_score(eval_result):
    """Calculate a more sophisticated 'interestingness' score based on multiple metrics."""
    # Extract metrics (handle potential NaN values)
    metrics = eval_result['metrics']
    
    # Helper function to safely get metric values
    def get_metric(key, default=0.0):
        value = metrics.get(key, default)
        return default if pd.isna(value) else value
    
    # ---- Basic metrics ----
    # Cooperation rate (prefer intermediate values around 0.5)
    coop = get_metric('avg_final_coop_rate')
    # Absolute change in cooperation rate (dynamics are interesting)
    coop_change = abs(get_metric('avg_coop_rate_change'))
    # Variance in scores across strategies (interesting if different outcomes)
    variance = min(get_metric('avg_score_variance') / 1000, 1.0)  # Normalize and cap
    # Strategy dominance (difference between best and worst strategy)
    dominance = min(get_metric('avg_strategy_dominance') / 100, 1.0)  # Normalize and cap
    # Network clustering (more interesting if there's clustering)
    clustering = get_metric('avg_network_clustering')
    
    # ---- Enhanced metrics ----
    # Volatility in cooperation rates (fluctuations are interesting)
    volatility = min(get_metric('avg_coop_volatility') * 5, 1.0)  # Scale and cap
    # Adaptation rate (how quickly strategies adapt)
    adaptation = min(get_metric('avg_strategy_adaptation_rate') * 10, 1.0)  # Scale and cap
    # Learning convergence (whether learning stabilizes)
    convergence = get_metric('avg_learning_convergence') / 5  # Scale down
    # Social welfare (total system payoff)
    welfare = get_metric('avg_social_welfare') / 10  # Scale down
    # Strategy countering (whether rankings change over time)
    countering = get_metric('avg_strategy_countering')
    # Pattern complexity (complexity of cooperation patterns)
    complexity = min(get_metric('avg_pattern_complexity') * 2, 1.0)  # Scale and cap
    # Equilibrium stability (whether system stabilizes)
    stability = get_metric('avg_equilibrium_stability')
    
    # ---- Calculate weighted score ----
    
    # Avoid extreme cases (either all cooperation or all defection)
    extreme_penalty = 0.5 * (abs(coop - 0.5) ** 2)
    
    # Weighted combination of metrics
    score = (
        # Basic metrics (40%)
        0.10 * (1 - extreme_penalty) +     # Prefer intermediate cooperation rates
        0.10 * coop_change +               # Reward dynamics (changes over time)
        0.10 * variance +                  # Reward variance across strategies
        0.05 * dominance +                 # Reward clear strategy dominance
        0.05 * clustering +                # Small reward for network clustering
        
        # Enhanced metrics (60%)
        0.10 * volatility +                # Reward fluctuations in cooperation
        0.10 * adaptation +                # Reward strategy adaptation
        0.05 * (1 - convergence) +         # Reward non-convergence (ongoing learning)
        0.05 * welfare +                   # Reward high social welfare
        0.15 * countering +                # Heavily reward strategy countering
        0.10 * complexity +                # Reward complex patterns
        0.05 * (1 - stability)             # Reward ongoing dynamics vs equilibrium
    )
    
    return score


# --- Parallel Evaluation ---
def evaluate_scenario_parallel(scenario_config, num_runs, seed_offset=0):
    """Evaluate a single scenario with multiple runs in a worker process."""
    silent_logger = logging.getLogger(f'silent_eval_{scenario_config["scenario_name"]}')
    silent_logger.setLevel(logging.ERROR)
    
    run_metrics_list = []
    for run_num in range(num_runs):
        # Set unique seed for this run
        seed = seed_offset + run_num
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            env, _ = setup_experiment(scenario_config, silent_logger)
            round_results = env.run_simulation(
                scenario_config["num_rounds"],
                logging_interval=scenario_config["num_rounds"] + 1,  # Disable logging during evaluation
                use_global_bonus=scenario_config.get("use_global_bonus", False),
                rewiring_interval=scenario_config.get("rewiring_interval", 0),
                rewiring_prob=scenario_config.get("rewiring_prob", 0.0)
            )
            metrics = calculate_enhanced_metrics(env, round_results)
            run_metrics_list.append(metrics)
        except Exception as e:
            # Just log to console - worker process logging might not work properly
            print(f"Error in worker for {scenario_config['scenario_name']}, run {run_num}: {e}")
    
    # Aggregate metrics across runs
    avg_eval_metrics = {}
    valid_metrics = [m for m in run_metrics_list if isinstance(m, dict)]
    
    if valid_metrics:
        keys = set().union(*[set(m.keys()) for m in valid_metrics])
        for key in keys:
            values = [m[key] for m in valid_metrics if key in m and not np.isnan(m[key])]
            avg_eval_metrics[f'avg_{key}'] = np.mean(values) if values else np.nan
    
    return {
        "config": scenario_config,
        "metrics": avg_eval_metrics,
        "num_valid_runs": len(valid_metrics)
    }


# --- Evolutionary Algorithm Functions ---

def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """Create a new scenario by combining parameters from two parent scenarios."""
    child = {}
    
    # Basic scenario info
    child['scenario_name'] = f"Evo_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Randomly select parameters from either parent
    # Core parameters
    child['num_agents'] = parent1['num_agents'] if random.random() < 0.5 else parent2['num_agents']
    child['num_rounds'] = parent1['num_rounds'] if random.random() < 0.5 else parent2['num_rounds']
    
    # Network parameters - take as a set to maintain consistency
    if random.random() < 0.5:
        child['network_type'] = parent1['network_type']
        child['network_params'] = parent1['network_params'].copy()
    else:
        child['network_type'] = parent2['network_type']
        child['network_params'] = parent2['network_params'].copy()
    
    # Interaction mode
    child['interaction_mode'] = parent1['interaction_mode'] if random.random() < 0.5 else parent2['interaction_mode']
    
    # Strategy distribution - more complex crossover
    # Option 1: Take complete distribution from one parent
    if random.random() < 0.3:
        child['agent_strategies'] = parent1['agent_strategies'].copy()
    elif random.random() < 0.6:
        child['agent_strategies'] = parent2['agent_strategies'].copy()
    # Option 2: Mix strategies from both parents
    else:
        child['agent_strategies'] = {}
        # Get all unique strategies from both parents
        all_strategies = set(parent1['agent_strategies'].keys()) | set(parent2['agent_strategies'].keys())
        
        # Assign agents while keeping total count consistent
        remaining_agents = child['num_agents']
        strategies_list = list(all_strategies)
        random.shuffle(strategies_list)
        
        for i, strategy in enumerate(strategies_list):
            if i == len(strategies_list) - 1:
                # Last strategy gets all remaining agents
                if remaining_agents > 0:
                    child['agent_strategies'][strategy] = remaining_agents
            else:
                # Get count from either parent, or 0 if not present
                p1_count = parent1['agent_strategies'].get(strategy, 0)
                p2_count = parent2['agent_strategies'].get(strategy, 0)
                
                # Take weighted average with random weight
                weight = random.random()
                count = int(weight * p1_count + (1 - weight) * p2_count)
                
                # Ensure at least 1 agent if strategy is selected
                if count > 0:
                    count = min(max(1, count), remaining_agents - (len(strategies_list) - i - 1))
                    child['agent_strategies'][strategy] = count
                    remaining_agents -= count
        
        # If we somehow didn't assign all agents, add them to a random strategy
        if remaining_agents > 0 and child['agent_strategies']:
            random_strategy = random.choice(list(child['agent_strategies'].keys()))
            child['agent_strategies'][random_strategy] += remaining_agents
    
    # Other parameters - randomly select from either parent
    params_to_copy = [
        'payoff_type', 'payoff_params', 'state_type', 'memory_length',
        'learning_rate', 'discount_factor', 'epsilon', 'q_init_type',
        'logging_interval'
    ]
    
    for param in params_to_copy:
        if param in parent1 and param in parent2:
            # Both parents have the parameter
            if isinstance(parent1[param], dict) and isinstance(parent2[param], dict):
                # For dictionary parameters, do field-by-field selection
                child[param] = {}
                all_keys = set(parent1[param].keys()) | set(parent2[param].keys())
                for key in all_keys:
                    if key in parent1[param] and key in parent2[param]:
                        # Both have the key, randomly select
                        child[param][key] = parent1[param][key] if random.random() < 0.5 else parent2[param][key]
                    elif key in parent1[param]:
                        # Only parent1 has it
                        child[param][key] = parent1[param][key]
                    else:
                        # Only parent2 has it
                        child[param][key] = parent2[param][key]
            else:
                # For simple parameters, randomly select from either parent
                child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
        elif param in parent1:
            child[param] = parent1[param]
        elif param in parent2:
            child[param] = parent2[param]
    
    # Special RL parameters that need to be consistent with agent strategies
    rl_params = [
        'beta', 'increase_rate', 'decrease_rate', 'alpha_win', 
        'alpha_lose', 'alpha_avg', 'exploration_constant', 'generosity'
    ]
    
    for param in rl_params:
        if param in parent1 and param in parent2:
            child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
        elif param in parent1:
            child[param] = parent1[param]
        elif param in parent2:
            child[param] = parent2[param]
    
    # Optional parameters
    for param in ['use_global_bonus', 'rewiring_interval', 'rewiring_prob']:
        # 50% chance to inherit if either parent has it
        if (param in parent1 or param in parent2) and random.random() < 0.5:
            if param in parent1:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
    
    return child


def mutate(scenario: Dict, mutation_rate: float = 0.2) -> Dict:
    """Mutate parameters in a scenario with a given probability."""
    mutated = scenario.copy()
    
    # Function to decide whether to mutate a parameter
    def should_mutate():
        return random.random() < mutation_rate
    
    # 1. Mutate basic parameters
    if should_mutate():
        mutated['num_agents'] = random.choice(NUM_AGENTS_POOL)
    
    if should_mutate():
        mutated['num_rounds'] = random.choice(NUM_ROUNDS_POOL)
    
    # 2. Mutate network structure
    if should_mutate():
        net_type, net_params = random.choice(NETWORK_POOL)
        mutated['network_type'] = net_type
        mutated['network_params'] = net_params
    
    # 3. Mutate interaction mode
    if should_mutate():
        mutated['interaction_mode'] = random.choice(INTERACTION_MODE_POOL)
    
    # 4. Mutate agent strategies
    if should_mutate():
        # Different mutation approaches for strategies
        mutation_type = random.choice(['add', 'remove', 'increase', 'decrease', 'swap'])
        
        strategies = mutated['agent_strategies'].copy()
        
        if mutation_type == 'add' and len(strategies) < 5:
            # Add a new strategy
            available_strategies = [s for s in AGENT_STRATEGY_POOL if s not in strategies]
            if available_strategies:
                new_strategy = random.choice(available_strategies)
                # Take some agents from existing strategies
                total_agents = mutated['num_agents']
                agents_to_reassign = max(1, int(total_agents * 0.2))  # Reassign up to 20%
                
                # Remove agents from existing strategies
                remaining = agents_to_reassign
                for strategy in list(strategies.keys()):
                    if remaining <= 0:
                        break
                    agents_to_take = min(strategies[strategy] - 1, remaining)
                    if agents_to_take > 0:
                        strategies[strategy] -= agents_to_take
                        remaining -= agents_to_take
                
                # Assign agents to new strategy
                strategies[new_strategy] = agents_to_reassign - remaining
        
        elif mutation_type == 'remove' and len(strategies) > 2:
            # Remove a strategy
            strategy_to_remove = random.choice(list(strategies.keys()))
            agents_to_reassign = strategies[strategy_to_remove]
            del strategies[strategy_to_remove]
            
            # Redistribute agents
            remaining_strategies = list(strategies.keys())
            for i, agents in enumerate([agents_to_reassign // len(remaining_strategies)] * len(remaining_strategies)):
                strategies[remaining_strategies[i]] += agents
            
            # Handle any remainder
            remainder = agents_to_reassign % len(remaining_strategies)
            for i in range(remainder):
                strategies[remaining_strategies[i]] += 1
        
        elif mutation_type == 'increase':
            # Increase representation of one strategy
            if len(strategies) > 1:
                strategy_to_increase = random.choice(list(strategies.keys()))
                others = [s for s in strategies.keys() if s != strategy_to_increase]
                strategy_to_decrease = random.choice(others)
                
                # Transfer up to 25% of agents
                agents_to_transfer = max(1, min(
                    strategies[strategy_to_decrease] - 1,  # Leave at least 1
                    int(mutated['num_agents'] * 0.25)      # Up to 25% of total
                ))
                
                strategies[strategy_to_increase] += agents_to_transfer
                strategies[strategy_to_decrease] -= agents_to_transfer
        
        elif mutation_type == 'decrease':
            # Decrease representation of one strategy
            if len(strategies) > 1:
                strategy_to_decrease = random.choice(list(strategies.keys()))
                others = [s for s in strategies.keys() if s != strategy_to_decrease]
                strategy_to_increase = random.choice(others)
                
                # Transfer up to 25% of agents
                agents_to_transfer = max(1, min(
                    strategies[strategy_to_decrease] - 1,  # Leave at least 1
                    int(mutated['num_agents'] * 0.25)      # Up to 25% of total
                ))
                
                strategies[strategy_to_decrease] -= agents_to_transfer
                strategies[strategy_to_increase] += agents_to_transfer
        
        elif mutation_type == 'swap':
            # Swap one strategy for another
            if strategies:
                strategy_to_replace = random.choice(list(strategies.keys()))
                available_strategies = [s for s in AGENT_STRATEGY_POOL if s not in strategies]
                if available_strategies:
                    new_strategy = random.choice(available_strategies)
                    agent_count = strategies[strategy_to_replace]
                    del strategies[strategy_to_replace]
                    strategies[new_strategy] = agent_count
        
        mutated['agent_strategies'] = strategies
    
    # 5. Mutate other parameters
    params_to_potentially_mutate = [
        ('state_type', STATE_TYPE_POOL),
        ('memory_length', MEMORY_LENGTH_POOL),
        ('q_init_type', Q_INIT_TYPE_POOL),
        ('learning_rate', LEARNING_RATE_POOL),
        ('discount_factor', DISCOUNT_FACTOR_POOL),
        ('epsilon', EPSILON_POOL),
    ]
    
    for param, pool in params_to_potentially_mutate:
        if param in mutated and should_mutate():
            mutated[param] = random.choice(pool)
    
    # 6. Mutate payoff parameters
    if 'payoff_params' in mutated and should_mutate():
        mutated['payoff_params'] = random.choice(PAYOFF_PARAMS_POOL)
    
    # 7. Mutate special parameters (with small probability)
    if should_mutate() and random.random() < 0.3:
        if random.random() < 0.5:
            mutated['use_global_bonus'] = not mutated.get('use_global_bonus', False)
        else:
            # Add or modify rewiring
            mutated['rewiring_interval'] = random.choice([20, 50, 100])
            mutated['rewiring_prob'] = round(random.uniform(0.05, 0.2), 2)
    
    # Regenerate name to indicate mutation
    mutated['scenario_name'] = f"Evo_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return mutated


# --- Evolutionary Scenario Generation ---
def run_evolutionary_scenario_generation(
    pop_size: int = 20,
    num_generations: int = 5,
    eval_runs: int = 3,
    elitism: int = 2,
    crossover_fraction: float = 0.7,
    mutation_rate: float = 0.2,
    save_runs: int = 10,
    top_n_to_save: int = 5,
    results_dir: str = "results/evolved_scenarios",
    log_level_str: str = 'INFO'
):
    """Run evolutionary algorithm to discover interesting scenarios."""
    
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    gen_logger = setup_logging(level=log_level, console=True, log_file="evolutionary_generator.log")
    
    gen_logger.info(f"Starting Evolutionary Scenario Generation")
    gen_logger.info(f"Population Size: {pop_size}, Generations: {num_generations}")
    gen_logger.info(f"Evaluation Runs per Scenario: {eval_runs}")
    gen_logger.info(f"Using {min(cpu_count(), pop_size)} worker processes for parallel evaluation")
    
    # Create results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    start_time = time.time()
    
    # Generate initial population
    gen_logger.info("Generating initial population...")
    population = []
    for i in range(pop_size):
        scenario_config = generate_random_scenario(i)
        population.append(scenario_config)
    
    # Tracking best scenarios and their scores across generations
    best_scenarios = []
    generation_stats = []
    
    # Evolution loop
    for generation in range(num_generations):
        gen_logger.info(f"Generation {generation+1}/{num_generations}")
        
        # Evaluate all scenarios in parallel
        gen_logger.info(f"Evaluating {len(population)} scenarios in parallel...")
        
        # Prepare worker function with seed offset
        seed_offset = generation * 1000
        worker_fn = partial(evaluate_scenario_parallel, num_runs=eval_runs, seed_offset=seed_offset)
        
        # Use process pool for parallel execution
        with Pool(processes=min(cpu_count(), len(population))) as pool:
            evaluated_population = pool.map(worker_fn, population)
        
        # Add interestingness score to each scenario
        for scenario_result in evaluated_population:
            scenario_result["selection_score"] = calculate_enhanced_interestingness_score(scenario_result)
        
        # Sort by score
        evaluated_population.sort(key=lambda x: x["selection_score"], reverse=True)
        
        # Save generation stats
        gen_stats = {
            "generation": generation+1,
            "avg_score": np.mean([s["selection_score"] for s in evaluated_population]),
            "max_score": evaluated_population[0]["selection_score"] if evaluated_population else 0,
            "min_score": evaluated_population[-1]["selection_score"] if evaluated_population else 0,
            "top_scenario_name": evaluated_population[0]["config"]["scenario_name"] if evaluated_population else "none",
            "scenario_types": defaultdict(int)
        }
        
        # Count strategy types in this generation
        for scenario in evaluated_population:
            for strategy in scenario["config"].get("agent_strategies", {}).keys():
                gen_stats["scenario_types"][strategy] += 1
        
        generation_stats.append(gen_stats)
        
        # Track best scenario from this generation
        if evaluated_population:
            best_scenarios.append(evaluated_population[0])
        
        # Log top scenarios from this generation
        gen_logger.info(f"Top scenarios in generation {generation+1}:")
        for i, scenario in enumerate(evaluated_population[:3]):
            gen_logger.info(f"{i+1}. {scenario['config']['scenario_name']} "
                           f"Score: {scenario['selection_score']:.3f}")
            gen_logger.info(f"   Strategies: {scenario['config']['agent_strategies']}")
            gen_logger.info(f"   Metrics: {', '.join([f'{k[4:]}={v:.3f}' for k, v in scenario['metrics'].items() if k.startswith('avg_') and not pd.isna(v)])}")
        
        # If this is the last generation, break the loop
        if generation == num_generations - 1:
            break
        
        # Create next generation
        gen_logger.info("Creating next generation...")
        next_gen = []
        
        # 1. Elitism: Keep best individuals
        next_gen.extend([scenario["config"] for scenario in evaluated_population[:elitism]])
        
        # 2. Crossover: Create children by combining parents
        num_crossovers = int((pop_size - len(next_gen)) * crossover_fraction)
        for _ in range(num_crossovers):
            # Select parents using tournament selection
            parent1 = tournament_selection(evaluated_population, tournament_size=3)
            parent2 = tournament_selection(evaluated_population, tournament_size=3)
            # Create child
            child = crossover(parent1["config"], parent2["config"])
            next_gen.append(child)
        
        # 3. Mutation: Create new individuals by mutating existing ones
        while len(next_gen) < pop_size:
            # Select individual to mutate
            parent = tournament_selection(evaluated_population, tournament_size=2)
            # Mutate
            mutated = mutate(parent["config"], mutation_rate)
            next_gen.append(mutated)
        
        # Set as new population
        population = next_gen
    
    # Final evaluation complete, sort all-time best scenarios
    best_scenarios.sort(key=lambda x: x["selection_score"], reverse=True)
    
    # Save metadata about the evolutionary process
    evolution_metadata = {
        "parameters": {
            "population_size": pop_size,
            "generations": num_generations,
            "evaluation_runs": eval_runs,
            "elitism": elitism,
            "crossover_fraction": crossover_fraction,
            "mutation_rate": mutation_rate
        },
        "generation_stats": generation_stats,
        "best_scenarios_by_generation": [
            {
                "generation": i+1,
                "name": scenario["config"]["scenario_name"],
                "score": scenario["selection_score"],
                "strategies": scenario["config"]["agent_strategies"]
            }
            for i, scenario in enumerate(best_scenarios)
        ]
    }
    
    # Save evolution metadata
    with open(os.path.join(results_dir, "evolution_metadata.json"), 'w') as f:
        json.dump(evolution_metadata, f, indent=2)
    
    # Save all evaluated scenarios metadata
    all_scenarios = []
    for i, scenario in enumerate(best_scenarios):
        all_scenarios.append({
            "config": scenario["config"],
            "metrics": scenario["metrics"],
            "selection_score": scenario["selection_score"]
        })
    
    save_scenario_metadata(all_scenarios, file_path=os.path.join(results_dir, "all_evolved_scenarios.json"))
    
    # Select top N scenarios to save full results
    top_scenarios = best_scenarios[:top_n_to_save]
    
    # Save the full configurations for reference
    selected_configs = [scenario["config"] for scenario in top_scenarios]
    with open(os.path.join(results_dir, "selected_evolved_scenarios.json"), 'w') as f:
        json.dump(selected_configs, f, indent=2)
    
    # Run and save full results for top scenarios
    gen_logger.info(f"\nRunning full simulations ({save_runs} runs) and saving results for top {top_n_to_save} scenarios...")
    saved_count = 0
    
    for k, scenario in enumerate(top_scenarios):
        selected_scenario = scenario["config"]
        scenario_name = selected_scenario['scenario_name']
        gen_logger.info(f"Running & Saving: {scenario_name} ({k+1}/{top_n_to_save})")
        
        # Set a reasonable logging interval for final saved runs
        selected_scenario['logging_interval'] = max(1, selected_scenario["num_rounds"] // 20)
        
        # Run multiple times and save results
        for run_number in range(save_runs):
            seed = 10000 + (k * save_runs + run_number)  # Use a completely different seed range
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
    
    end_time = time.time()
    gen_logger.info(f"\nEvolutionary Scenario Generation & Selection complete.")
    gen_logger.info(f"Saved {saved_count} scenarios from {num_generations} generations.")
    gen_logger.info(f"Total time: {end_time - start_time:.2f} seconds.")
    gen_logger.info(f"Results saved in: {results_dir}")
    
    return best_scenarios, generation_stats


def tournament_selection(evaluated_population, tournament_size=3):
    """Select a scenario from the population using tournament selection."""
    tournament = random.sample(evaluated_population, min(tournament_size, len(evaluated_population)))
    return max(tournament, key=lambda x: x["selection_score"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced scenario generation with evolution and parallel processing")
    parser.add_argument("--pop_size", type=int, default=20,
                       help="Population size for evolutionary algorithm")
    parser.add_argument("--generations", type=int, default=5,
                       help="Number of generations to evolve")
    parser.add_argument("--eval_runs", type=int, default=3,
                       help="Number of evaluation runs per scenario")
    parser.add_argument("--elitism", type=int, default=2,
                       help="Number of best scenarios to keep unchanged in each generation")
    parser.add_argument("--crossover", type=float, default=0.7,
                       help="Fraction of new generation created through crossover")
    parser.add_argument("--mutation", type=float, default=0.2,
                       help="Mutation rate for scenario parameters")
    parser.add_argument("--save_runs", type=int, default=10,
                       help="Number of final runs for selected scenarios")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top scenarios to save")
    parser.add_argument("--results_dir", type=str, default="results/evolved_scenarios",
                       help="Directory to save results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    run_evolutionary_scenario_generation(
        pop_size=args.pop_size,
        num_generations=args.generations,
        eval_runs=args.eval_runs,
        elitism=args.elitism,
        crossover_fraction=args.crossover,
        mutation_rate=args.mutation,
        save_runs=args.save_runs,
        top_n_to_save=args.top_n,
        results_dir=args.results_dir,
        log_level_str=args.log_level
    )
