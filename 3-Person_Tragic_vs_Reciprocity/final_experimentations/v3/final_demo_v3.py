import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

from final_agents_v3 import StaticAgent, VanillaQLearningAgent, AdaptiveAgent
from final_simulation_v3 import run_pairwise_tournament, run_nperson_simulation


def run_experiment_set(agent_templates, num_rounds, num_runs):
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


def plot_comparison(results, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.suptitle(f"Agent Comparison: {title}", fontsize=22, weight='bold')
    sns.set_style("whitegrid")

    agent_colors = {"Vanilla": "#1f77b4", "Adaptive": "#ff7f0e"}  # Blue, Orange
    metrics = [
        ('Pairwise Cooperation', 'coop_rate', 0, 0), ('Pairwise Scores', 'score', 0, 1),
        ('Neighbourhood Cooperation', 'coop_rate', 1, 0), ('Neighbourhood Scores', 'score', 1, 1)
    ]

    for label, metric, r, c in metrics:
        ax = axes[r, c]
        for agent_name, data in results.items():  # agent_name is now 'Vanilla' or 'Adaptive'
            data_p, data_n = data
            plot_data = data_p if 'Pairwise' in label else data_n

            # Corrected logic to find the agent ID
            ql_id = next((k for k in plot_data if k.startswith(agent_name)), None)

            if ql_id:
                ax.plot(plot_data[ql_id][metric], label=agent_name, color=agent_colors.get(agent_name))

        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Round"), ax.set_ylabel("Rate" if 'Coop' in label else "Score")
        ax.legend(), ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved to {save_path}")


if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 20

    opponents = [StaticAgent(agent_id="TFT_1", strategy_name="TFT"),
                 StaticAgent(agent_id="TFT_2", strategy_name="TFT")]

    experiment_name = "vs_2_TFTs"
    output_dir = "adaptive_comparison"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Simplified dictionary keys for robust lookup in the plot function
    agents_to_test = {
        "Vanilla": VanillaQLearningAgent(agent_id="Vanilla_1"),
        "Adaptive": AdaptiveAgent(agent_id="Adaptive_1"),
    }

    for name, agent_instance in agents_to_test.items():
        print(f"\n--- Running Experiment for: {name} ---")
        templates = [agent_instance] + opponents
        all_results[name] = run_experiment_set(templates, NUM_ROUNDS, NUM_RUNS)

    print("\n--- Generating Final Plot ---")
    plot_path = os.path.join(output_dir, f"adaptive_comparison_{experiment_name}.png")
    plot_comparison(all_results, f"Vanilla vs. Adaptive Agent {experiment_name}", plot_path)

    print(f"\nAll experiments complete. Chart saved to '{output_dir}'.")