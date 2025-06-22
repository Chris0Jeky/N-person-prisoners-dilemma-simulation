import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations

# Import the new, robust agents and simulation runners
from final_agents import StaticAgent, EnhancedQLearningAgent, SimpleQLearningAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation


# --- Experiment Runner ---
def run_experiment_set(agent_templates, num_rounds, num_runs):
    """Runs multiple simulations and aggregates the results."""
    pairwise_runs = []
    nperson_runs = []

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...", end='\r')
        # Create fresh instances for each run
        fresh_agents_p = [type(a)(**a.__dict__) for a in agent_templates]
        fresh_agents_n = [type(a)(**a.__dict__) for a in agent_templates]

        pairwise_runs.append(run_pairwise_tournament(fresh_agents_p, num_rounds))
        nperson_runs.append(run_nperson_simulation(fresh_agents_n, num_rounds))
    print("\n  Done.")

    # Aggregate results
    def aggregate(runs):
        agg = {agent_id: {'coop_rate': [], 'score': []} for agent_id in runs[0]}
        for agent_id in agg:
            for metric in ['coop_rate', 'score']:
                all_metric_runs = [run[agent_id][metric] for run in runs]
                agg[agent_id][metric] = np.mean(all_metric_runs, axis=0)
        return agg

    return aggregate(pairwise_runs), aggregate(nperson_runs)


# --- Plotting ---
def plot_comparison(basic_data, enhanced_data, title):
    """Generates a 4-panel comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.suptitle(f"Detailed Comparison: {title}", fontsize=22, weight='bold')
    sns.set_style("whitegrid")

    metrics = [
        ('Pairwise Cooperation', 'coop_rate', 0, 0), ('Pairwise Scores', 'score', 0, 1),
        ('Neighbourhood Cooperation', 'coop_rate', 1, 0), ('Neighbourhood Scores', 'score', 1, 1)
    ]

    # Extract data for QL agents only
    b_data_p, b_data_n = basic_data
    e_data_p, e_data_n = enhanced_data

    b_ql_id = next(k for k in b_data_p if "QL" in k)
    e_ql_id = next(k for k in e_data_p if "QL" in k)

    data_map = {
        (0, 0): (b_data_p[b_ql_id], e_data_p[e_ql_id]), (0, 1): (b_data_p[b_ql_id], e_data_p[e_ql_id]),
        (1, 0): (b_data_n[b_ql_id], e_data_n[e_ql_id]), (1, 1): (b_data_n[b_ql_id], e_data_n[e_ql_id]),
    }

    for label, metric, r, c in metrics:
        ax = axes[r, c]
        basic_plot_data, enhanced_plot_data = data_map[(r, c)]

        ax.plot(basic_plot_data[metric], label='Basic QL_1', color='blue', linestyle='-')
        ax.plot(enhanced_plot_data[metric], label='Final Enhanced EQL_1', color='darkblue', linestyle='--')

        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation Rate" if 'Coop' in label else "Cumulative Score", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# --- Main ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 20  # Keep low for a quick demonstration

    # --- Experiment 1: Basic QL vs. AllD and AllC ---
    print("--- Running Basic QL vs. 1 AllD + 1 AllC ---")
    basic_templates = [
        SimpleQLearningAgent(agent_id="Basic_QL_1"),
        StaticAgent(agent_id="AllD_1", strategy_name="AllD"),
        StaticAgent(agent_id="AllC_1", strategy_name="AllC"),
    ]
    basic_data_allx = run_experiment_set(basic_templates, NUM_ROUNDS, NUM_RUNS)

    # --- Experiment 2: Enhanced QL vs. AllD and AllC ---
    print("\n--- Running Enhanced QL vs. 1 AllD + 1 AllC ---")
    enhanced_templates = [
        EnhancedQLearningAgent(agent_id="Enhanced_EQL_1"),
        StaticAgent(agent_id="AllD_1", strategy_name="AllD"),
        StaticAgent(agent_id="AllC_1", strategy_name="AllC"),
    ]
    enhanced_data_allx = run_experiment_set(enhanced_templates, NUM_ROUNDS, NUM_RUNS)

    # --- Generate and Save Plot ---
    os.makedirs("final_charts", exist_ok=True)
    fig = plot_comparison(basic_data_allx, enhanced_data_allx, "1 QL + 1 AllD + 1 AllC")
    fig.savefig("final_charts/final_comparison_AllDxAllC.png")
    plt.show()

    print("\nDemonstration complete. Chart saved to 'final_charts/'.")