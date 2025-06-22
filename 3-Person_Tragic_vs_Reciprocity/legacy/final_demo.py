import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# Import the new, robust agents and simulation runners
from final_agents import StaticAgent, EnhancedQLearningAgent, SimpleQLearningAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation


# --- Experiment Runner ---
def run_experiment_set(agent_templates, num_rounds, num_runs):
    """Runs multiple simulations and aggregates the results."""
    pairwise_runs = []
    nperson_runs = []

    for i in range(num_runs):
        print(f"    Run {i + 1}/{num_runs}...", end='\r')
        # Create fresh instances for each run to ensure statelessness
        fresh_agents_p = [type(a)(**a.__dict__) for a in agent_templates]
        fresh_agents_n = [type(a)(**a.__dict__) for a in agent_templates]

        pairwise_runs.append(run_pairwise_tournament(fresh_agents_p, num_rounds))
        nperson_runs.append(run_nperson_simulation(fresh_agents_n, num_rounds))
    print("\n    Done.")

    # Aggregate results by taking the mean across all runs
    def aggregate(runs):
        agg = {agent_id: {'coop_rate': [], 'score': []} for agent_id in runs[0]}
        for agent_id in agg:
            for metric in ['coop_rate', 'score']:
                all_metric_runs = [run[agent_id][metric] for run in runs]
                agg[agent_id][metric] = np.mean(all_metric_runs, axis=0)
        return agg

    return aggregate(pairwise_runs), aggregate(nperson_runs)


# --- Plotting ---
def plot_comparison(basic_data, enhanced_data, title, save_path):
    """Generates a 4-panel comparison plot for a single experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.suptitle(f"Detailed Comparison: {title}", fontsize=22, weight='bold')
    sns.set_style("whitegrid")

    metrics = [
        ('Pairwise Cooperation', 'coop_rate', 0, 0), ('Pairwise Scores', 'score', 0, 1),
        ('Neighbourhood Cooperation', 'coop_rate', 1, 0), ('Neighbourhood Scores', 'score', 1, 1)
    ]

    b_data_p, b_data_n = basic_data
    e_data_p, e_data_n = enhanced_data

    # Find all QL agent IDs in the results
    b_ql_ids = sorted([k for k in b_data_p if "QL" in k])
    e_ql_ids = sorted([k for k in e_data_p if "EQL" in k])

    data_map = {
        (0, 0): (b_data_p, e_data_p), (0, 1): (b_data_p, e_data_p),
        (1, 0): (b_data_n, e_data_n), (1, 1): (b_data_n, e_data_n),
    }

    for label, metric, r, c in metrics:
        ax = axes[r, c]
        basic_plot_data, enhanced_plot_data = data_map[(r, c)]

        # Plot all Basic QL agents
        for i, ql_id in enumerate(b_ql_ids):
            ax.plot(basic_plot_data[ql_id][metric], label=f'Basic {ql_id}', color=f'C{i}', linestyle='-')
        # Plot all Enhanced QL agents
        for i, ql_id in enumerate(e_ql_ids):
            ax.plot(enhanced_plot_data[ql_id][metric], label=f'Enhanced {ql_id}', color=f'C{i}', linestyle='--')

        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation Rate" if 'Coop' in label else "Cumulative Score", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved to {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 200  # Keep low for faster demo runs; increase for smoother final charts

    # Define all experiment configurations
    experiment_configs = {
        "1_QL_plus_1_AllD_plus_1_AllC": [("QL", {}), ("AllD", {}), ("AllC", {})],
        "1_QL_plus_2_AllD": [("QL", {}), ("AllD", {}), ("AllD", {})],
        "1_QL_plus_2_TFT": [("QL", {}), ("TFT", {}), ("TFT", {})],
        "2_QL_plus_1_AllD": [("QL", {}), ("QL", {}), ("AllD", {})],
        "2_QL_plus_1_TFT": [("QL", {}), ("TFT", {})]  # Corrected this line
    }

    # Ensure the output directory exists
    output_dir = "final_charts_all_scenarios"
    os.makedirs(output_dir, exist_ok=True)

    # --- Loop through experiments, run, and plot ---
    for name, config in experiment_configs.items():
        print(f"\n--- Running Experiment: {name} ---")

        # --- 1. Basic QL Version ---
        print("  Running Basic QL version...")
        basic_templates = []
        counts = defaultdict(int)
        for agent_type, params in config:
            counts[agent_type] += 1
            if agent_type == "QL":
                basic_templates.append(SimpleQLearningAgent(agent_id=f"Basic_QL_{counts[agent_type]}", **params))
            else:
                basic_templates.append(
                    StaticAgent(agent_id=f"{agent_type}_{counts[agent_type]}", strategy_name=agent_type, **params))

        basic_data = run_experiment_set(basic_templates, NUM_ROUNDS, NUM_RUNS)

        # --- 2. Enhanced QL Version ---
        print("  Running Enhanced QL version...")
        enhanced_templates = []
        counts = defaultdict(int)
        for agent_type, params in config:
            counts[agent_type] += 1
            if agent_type == "QL":
                enhanced_templates.append(
                    EnhancedQLearningAgent(agent_id=f"Enhanced_EQL_{counts[agent_type]}", **params))
            else:
                enhanced_templates.append(
                    StaticAgent(agent_id=f"{agent_type}_{counts[agent_type]}", strategy_name=agent_type, **params))

        enhanced_data = run_experiment_set(enhanced_templates, NUM_ROUNDS, NUM_RUNS)

        # --- 3. Generate and Save Plot ---
        print("  Generating plot...")
        plot_path = os.path.join(output_dir, f"comparison_{name}.png")
        plot_comparison(basic_data, enhanced_data, name.replace('_', ' '), plot_path)

    print(f"\nAll experiments complete. Charts saved to '{output_dir}'.")