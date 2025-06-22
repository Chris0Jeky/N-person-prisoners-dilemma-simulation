#!/usr/bin/env python3
"""
Final Q-Learning Demo Generator using the robust Unified API
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# Import the clean, unified agents and simulation runner
from final_agents import StaticAgent, SimpleQLearningAgent, EnhancedQLearningAgent
from final_simulation_runner import run_pairwise_simulation, run_nperson_simulation


# --- Experiment Setup (Same as before, but using new class names) ---
def setup_experiments(agent_type='enhanced'):
    strategies = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
    experiments = {}

    ql_agent_class = EnhancedQLearningAgent if agent_type == 'enhanced' else SimpleQLearningAgent
    ql_agent_name = "EQL" if agent_type == 'enhanced' else "QL"

    # 1 QL Agent vs. various pairs
    for combo in combinations(strategies, 2):
        exp_name = f"1 {ql_agent_name} + 1 {combo[0]} + 1 {combo[1]}"
        experiments[exp_name] = [
            ql_agent_class(agent_id=f"{ql_agent_name}_1"),
            StaticAgent(agent_id=f"{combo[0]}_1", strategy_name=combo[0]),
            StaticAgent(agent_id=f"{combo[1]}_1", strategy_name=combo[1])
        ]
    for strat in strategies:
        exp_name = f"1 {ql_agent_name} + 2 {strat}"
        experiments[exp_name] = [
            ql_agent_class(agent_id=f"{ql_agent_name}_1"),
            StaticAgent(agent_id=f"{strat}_1", strategy_name=strat),
            StaticAgent(agent_id=f"{strat}_2", strategy_name=strat)
        ]
    return experiments


# --- Multi-run and Aggregation Logic (Simplified) ---
def run_multiple_simulations(simulation_func, agent_templates, num_rounds, num_runs):
    all_coop_runs = {agent.agent_id: [] for agent in agent_templates}
    all_score_runs = {agent.agent_id: [] for agent in agent_templates}

    for _ in range(num_runs):
        # Create fresh agent instances from templates
        fresh_agents = [type(a)(**a.__dict__) for a in agent_templates]

        coop_history, score_history = simulation_func(fresh_agents, num_rounds)
        for agent in fresh_agents:
            all_coop_runs[agent.agent_id].append(coop_history[agent.agent_id])
            all_score_runs[agent.agent_id].append(score_history[agent.agent_id])

    # Aggregate results
    agg_coop = {aid: {'mean': np.mean(runs, axis=0)} for aid, runs in all_coop_runs.items()}
    agg_scores = {aid: {'mean': np.mean(runs, axis=0)} for aid, runs in all_score_runs.items()}
    return agg_coop, agg_scores


# --- Plotting (Simplified for demonstration) ---
def create_comparison_plot(basic_data, enhanced_data, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f'Detailed Comparison: {title}', fontsize=20, weight='bold')

    plot_metrics = {
        'Pairwise Cooperation': ('pairwise_coop', axes[0, 0]),
        'Pairwise Scores': ('pairwise_scores', axes[0, 1]),
        'Neighbourhood Cooperation': ('nperson_coop', axes[1, 0]),
        'Neighbourhood Scores': ('nperson_scores', axes[1, 1]),
    }

    for label, (key, ax) in plot_metrics.items():
        if key in basic_data and key in enhanced_data:
            # Plot Basic QL
            for agent_id, data in basic_data[key].items():
                if 'QL' in agent_id:
                    ax.plot(data['mean'], label=f'Basic {agent_id}', color='C0', linestyle='-')
            # Plot Enhanced QL
            for agent_id, data in enhanced_data[key].items():
                if 'EQL' in agent_id:
                    ax.plot(data['mean'], label=f'Enhanced {agent_id}', color='C1', linestyle='--')

        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation Rate" if 'Cooperation' in label else "Cumulative Score", fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 15  # Keep low for faster demo runs

    # --- 1. Get Basic QL Data ---
    print("--- Running Basic QL Simulations ---")
    basic_experiments = setup_experiments(agent_type='simple')
    basic_results = {}
    exp_to_run = '1 QL + 2 TFT'
    print(f"Running experiment: {exp_to_run}")
    basic_templates = basic_experiments[exp_to_run]
    basic_results['pairwise_coop'], basic_results['pairwise_scores'] = run_multiple_simulations(run_pairwise_simulation,
                                                                                                basic_templates,
                                                                                                NUM_ROUNDS, NUM_RUNS)
    basic_results['nperson_coop'], basic_results['nperson_scores'] = run_multiple_simulations(run_nperson_simulation,
                                                                                              basic_templates,
                                                                                              NUM_ROUNDS, NUM_RUNS)

    # --- 2. Get Enhanced QL Data ---
    print("\n--- Running Enhanced QL Simulations ---")
    enhanced_experiments = setup_experiments(agent_type='enhanced')
    enhanced_results = {}
    exp_to_run = '1 EQL + 2 TFT'
    print(f"Running experiment: {exp_to_run}")
    enhanced_templates = enhanced_experiments[exp_to_run]
    enhanced_results['pairwise_coop'], enhanced_results['pairwise_scores'] = run_multiple_simulations(
        run_pairwise_simulation, enhanced_templates, NUM_ROUNDS, NUM_RUNS)
    enhanced_results['nperson_coop'], enhanced_results['nperson_scores'] = run_multiple_simulations(
        run_nperson_simulation, enhanced_templates, NUM_ROUNDS, NUM_RUNS)

    # --- 3. Generate Comparison Plot ---
    os.makedirs("final_comparison_charts", exist_ok=True)
    create_comparison_plot(
        basic_results,
        enhanced_results,
        title="1 QL vs 1 EQL (against 2 TFTs)",
        save_path="final_comparison_charts/final_tft_comparison.png"
    )