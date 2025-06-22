import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict

# Import the robust agents and simulation runners
from final_agents import StaticAgent, EnhancedQLearningAgent, SimpleQLearningAgent
from final_simulation import run_pairwise_tournament, run_nperson_simulation


# --- Experiment Runner ---
def run_experiment_set(agent_templates, num_rounds, num_runs):
    """Runs multiple simulations and aggregates the results."""
    pairwise_runs = []
    nperson_runs = []

    for i in range(num_runs):
        print(f"    Run {i + 1}/{num_runs}...", end='\r')
        # Create fresh instances for each run
        fresh_agents_p = [type(a)(**a.__dict__) for a in agent_templates]
        fresh_agents_n = [type(a)(**a.__dict__) for a in agent_templates]

        pairwise_runs.append(run_pairwise_tournament(fresh_agents_p, num_rounds))
        nperson_runs.append(run_nperson_simulation(fresh_agents_n, num_rounds))
    print("\n    Done.")

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
    """Generates a 4-panel comparison plot with distinct colors."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.suptitle(f"Detailed Comparison: {title}", fontsize=22, weight='bold')
    sns.set_style("whitegrid")

    metrics = [
        ('Pairwise Cooperation', 'coop_rate', 0, 0), ('Pairwise Scores', 'score', 0, 1),
        ('Neighbourhood Cooperation', 'coop_rate', 1, 0), ('Neighbourhood Scores', 'score', 1, 1)
    ]

    b_data_p, b_data_n = basic_data
    e_data_p, e_data_n = enhanced_data

    b_ql_ids = sorted([k for k in b_data_p if "QL" in k])
    e_ql_ids = sorted([k for k in e_data_p if "EQL" in k])

    data_map = {
        (0, 0): (b_data_p, e_data_p), (0, 1): (b_data_p, e_data_p),
        (1, 0): (b_data_n, e_data_n), (1, 1): (b_data_n, e_data_n),
    }

    for label, metric, r, c in metrics:
        ax = axes[r, c]
        basic_plot_data, enhanced_plot_data = data_map[(r, c)]

        for ql_id in b_ql_ids:
            ax.plot(basic_plot_data[ql_id][metric], label=f'Basic {ql_id}', color="#1f77b4")  # Blue
        for ql_id in e_ql_ids:
            ax.plot(enhanced_plot_data[ql_id][metric], label=f'Enhanced {ql_id}', color="#ff7f0e")  # Orange

        ax.set_title(label, fontsize=16)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Cooperation Rate" if 'Coop' in label else "Cumulative Score", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Chart saved to {save_path}")


def create_performance_heatmap(all_results, save_path):
    """Creates a heatmap summarizing the performance difference."""
    heatmap_data = []
    experiment_names = []

    for name, results in all_results.items():
        experiment_names.append(name.replace('_', ' ').replace('plus', '+'))
        row_data = []

        # Helper to get average metric for QL agents
        def get_avg_metric(data, metric):
            ql_ids = [k for k in data if "QL" in k]
            if not ql_ids: return 0

            if metric == 'coop_rate':
                # Average of the mean cooperation rate over all rounds
                return np.mean([np.mean(data[qid]['coop_rate']) for qid in ql_ids])
            else:  # score
                # Average of the final score
                return np.mean([data[qid]['score'][-1] for qid in ql_ids])

        # Calculate metrics for pairwise and nperson
        basic_p, basic_n = results['basic']
        enh_p, enh_n = results['enhanced']

        metrics_to_calc = [
            (basic_p, enh_p, 'coop_rate'), (basic_p, enh_p, 'score'),
            (basic_n, enh_n, 'coop_rate'), (basic_n, enh_n, 'score')
        ]

        for basic_data, enh_data, metric_name in metrics_to_calc:
            basic_val = get_avg_metric(basic_data, metric_name)
            enh_val = get_avg_metric(enh_data, metric_name)

            if basic_val == 0:  # Avoid division by zero
                percent_diff = 0 if enh_val == 0 else 100.0
            else:
                percent_diff = ((enh_val - basic_val) / abs(basic_val)) * 100
            row_data.append(percent_diff)

        heatmap_data.append(row_data)

    df = pd.DataFrame(heatmap_data,
                      index=experiment_names,
                      columns=['Pairwise Coop', 'Pairwise Score', 'NPerson Coop', 'NPerson Score'])

    plt.figure(figsize=(12, 16))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                cbar_kws={'label': '% Difference (Enhanced - Basic)'})

    plt.title("Performance Difference: Final Enhanced QL vs Basic QL\n(Positive = Enhanced Better)", fontsize=16)
    plt.xlabel("Metric")
    plt.ylabel("Experiment Scenario")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 20

    experiment_configs = {
        "1_QL_plus_1_AllD_plus_1_AllC": [("QL", {}), ("AllD", {}), ("AllC", {})],
        "1_QL_plus_2_AllD": [("QL", {}), ("AllD", {}), ("AllD", {})],
        "1_QL_plus_2_TFT": [("QL", {}), ("TFT", {}), ("TFT", {})],
        "2_QL_plus_1_AllD": [("QL", {}), ("QL", {}), ("AllD", {})],
        "2_QL_plus_1_TFT": [("QL", {}), ("QL", {}), ("TFT", {})],
    }

    output_dir = "final_charts_all_scenarios"
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for name, config in experiment_configs.items():
        print(f"\n--- Running Experiment: {name} ---")

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

        all_results[name] = {'basic': basic_data, 'enhanced': enhanced_data}

        print("  Generating plot...")
        plot_path = os.path.join(output_dir, f"comparison_{name}.png")
        plot_comparison(basic_data, enhanced_data, name.replace('_', ' ').replace('plus', '+'), plot_path)

    print("\n--- Generating Summary Heatmap ---")
    heatmap_path = os.path.join(output_dir, "performance_heatmap.png")
    create_performance_heatmap(all_results, heatmap_path)

    print(f"\nAll experiments complete. Charts saved to '{output_dir}'.")