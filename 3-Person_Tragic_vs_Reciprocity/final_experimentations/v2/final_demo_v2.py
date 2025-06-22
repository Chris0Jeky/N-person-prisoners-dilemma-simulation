import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# Import the new, robust agents and simulation runners
from final_agents_v2 import StaticAgent, SimpleQLearningAgent, VanillaEnhancedAgent, BetterEnhancedAgent
from final_simulation_v2 import run_pairwise_tournament, run_nperson_simulation


# --- Experiment Runner ---
def run_experiment_set(agent_templates, num_rounds, num_runs):
    """Runs multiple simulations and aggregates the results."""
    pairwise_runs, nperson_runs = [], []
    for i in range(num_runs):
        print(f"    Run {i + 1}/{num_runs}...", end='\r')
        fresh_p = [type(a)(**a.__dict__) for a in agent_templates]
        fresh_n = [type(a)(**a.__dict__) for a in agent_templates]
        pairwise_runs.append(run_pairwise_tournament(fresh_p, num_rounds))
        nperson_runs.append(run_nperson_simulation(fresh_n, num_rounds))
    print("\n    Done.")

    def aggregate(runs):
        agg = {aid: {m: [] for m in ['coop_rate', 'score']} for aid in runs[0]}
        for aid in agg:
            for m in agg[aid]:
                all_runs = [run[aid][m] for run in runs]
                agg[aid][m] = np.mean(all_runs, axis=0)
        return agg

    return aggregate(pairwise_runs), aggregate(nperson_runs)


# --- Plotting ---
def plot_trio_comparison(results, title, save_path):
    """Generates a 4-panel plot comparing three agent types."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.suptitle(f"Agent Comparison: {title}", fontsize=22, weight='bold')
    sns.set_style("whitegrid")

    agent_colors = {
        "SimpleQLearning": "#1f77b4",  # Muted Blue
        "VanillaEnhanced": "#2ca02c",  # Muted Green
        "BetterEnhanced": "#ff7f0e",  # Muted Orange
    }

    metrics = [
        ('Pairwise Cooperation', 'coop_rate', 0, 0), ('Pairwise Scores', 'score', 0, 1),
        ('Neighbourhood Cooperation', 'coop_rate', 1, 0), ('Neighbourhood Scores', 'score', 1, 1)
    ]

    for label, metric, r, c in metrics:
        ax = axes[r, c]
        for agent_name, data in results.items():
            data_p, data_n = data
            plot_data = data_p if 'Pairwise' in label else data_n

            ql_id = next((k for k in plot_data if agent_name in k), None)
            if ql_id:
                ax.plot(plot_data[ql_id][metric], label=agent_name, color=agent_colors[agent_name])

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
    NUM_RUNS = 20

    # Define the opponent configuration
    opponents = [StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
                 StaticAgent(agent_id="TFT_2", strategy_name="TFT")]

    experiment_name = "vs_2_TFT"
    output_dir = "final_trio_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Store results for each agent type
    all_results = {}

    # Define the three QL agents to test
    agents_to_test = {
        "SimpleQLearning": SimpleQLearningAgent(agent_id="SimpleQL_1"),
        "VanillaEnhanced": VanillaEnhancedAgent(agent_id="VanillaEnhanced_1"),
        "BetterEnhanced": BetterEnhancedAgent(agent_id="BetterEnhanced_1"),
    }

    for name, agent_instance in agents_to_test.items():
        print(f"\n--- Running Experiment for: {name} ---")
        templates = [agent_instance] + opponents
        all_results[name] = run_experiment_set(templates, NUM_ROUNDS, NUM_RUNS)

    print("\n--- Generating Final Plot ---")
    plot_path = os.path.join(output_dir, f"trio_comparison_{experiment_name}.png")
    plot_trio_comparison(all_results, "vs. 2 TFTs", plot_path)

    print(f"\nAll experiments complete. Chart saved to '{output_dir}'.")