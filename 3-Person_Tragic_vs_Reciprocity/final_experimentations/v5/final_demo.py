import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS, SIMULATION_CONFIG

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_cooperators, total_agents):
    others_coop = num_cooperators - (1 - my_move)
    if my_move == 0:
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:
        return P + (T - P) * (others_coop / (total_agents - 1))


# --- Simulation Runners ---
def run_pairwise_tournament(agents, num_rounds):
    for agent in agents: agent.reset()
    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    agent_map = {a.agent_id: a for a in agents}
    for _ in range(num_rounds):
        moves = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                moves[(agent1.agent_id, agent2.agent_id)] = (
                    agent1.choose_pairwise_action(agent2.agent_id),
                    agent2.choose_pairwise_action(agent1.agent_id)
                )
        round_moves_by_agent = defaultdict(list)
        for (id1, id2), (move1, move2) in moves.items():
            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            agent_map[id1].record_pairwise_outcome(id2, move1, move2, payoff1)
            agent_map[id2].record_pairwise_outcome(id1, move2, move1, payoff2)
            round_moves_by_agent[id1].append(move1)
            round_moves_by_agent[id2].append(move2)
        for agent in agents:
            agent_moves = round_moves_by_agent[agent.agent_id]
            if not agent_moves: continue
            history[agent.agent_id]['coop_rate'].append(agent_moves.count(0) / len(agent_moves))
            history[agent.agent_id]['score'].append(agent.total_score)
    return history


def run_nperson_simulation(agents, num_rounds):
    for agent in agents: agent.reset()
    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    coop_ratio = None
    for _ in range(num_rounds):
        moves = {a.agent_id: a.choose_neighborhood_action(coop_ratio) for a in agents}
        num_cooperators = list(moves.values()).count(0)
        current_coop_ratio = num_cooperators / len(agents)
        for agent in agents:
            my_move = moves[agent.agent_id]
            payoff = nperson_payoff(my_move, num_cooperators, len(agents))
            agent.record_neighborhood_outcome(current_coop_ratio, payoff)
            history[agent.agent_id]['coop_rate'].append(1 - my_move)
            history[agent.agent_id]['score'].append(agent.total_score)
        coop_ratio = current_coop_ratio
    return history


def run_experiment_set(config, opponents, num_rounds, num_runs):
    """Runs simulations for one agent config against a set of opponents."""
    # Pairwise
    p_templates = [config["class"](agent_id=f"{config['name']}_1", params=config["params"])] + opponents
    p_runs = [run_pairwise_tournament([type(a)(**a.__dict__) for a in p_templates], num_rounds) for _ in
              range(num_runs)]
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs], axis=0) for m in ['coop_rate', 'score']} for aid in
             p_runs[0]}

    # Neighborhood
    n_templates = [config["class"](agent_id=f"{config['name']}_1", params=config["params"])] + opponents
    n_runs = [run_nperson_simulation([type(a)(**a.__dict__) for a in n_templates], num_rounds) for _ in range(num_runs)]
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs], axis=0) for m in ['coop_rate', 'score']} for aid in
             n_runs[0]}

    return p_agg, n_agg


def smooth_data(data, window_size=50):
    """Apply rolling average to smooth data"""
    if len(data) < window_size:
        window_size = max(1, len(data) // 10)
    
    # Use pandas for efficient rolling average
    series = pd.Series(data)
    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values

def plot_scenario_comparison(results, title, save_path, num_rounds=None):
    """Generates a 4-panel plot comparing Vanilla vs. Adaptive for one scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f"Comparison: {title}", fontsize=22)
    colors = {"Vanilla": "#1f77b4", "Adaptive": "#ff7f0e"}
    
    # Determine smoothing window based on number of rounds
    if num_rounds is None:
        num_rounds = SIMULATION_CONFIG['num_rounds']
    smooth_window = max(50, num_rounds // 100)  # At least 50, or 1% of total rounds

    for name, data in results.items():
        p_data, n_data = data
        ql_id = f"{name}_QL_1"
        
        # Apply smoothing to cooperation rates
        p_coop_smooth = smooth_data(p_data[ql_id]['coop_rate'], smooth_window)
        n_coop_smooth = smooth_data(n_data[ql_id]['coop_rate'], smooth_window)
        
        # Plot raw data with low alpha and smoothed data with high alpha
        axes[0, 0].plot(p_data[ql_id]['coop_rate'], color=colors[name], alpha=0.2, linewidth=0.5)
        axes[0, 0].plot(p_coop_smooth, label=name, color=colors[name], linewidth=2.5)
        
        axes[1, 0].plot(n_data[ql_id]['coop_rate'], color=colors[name], alpha=0.2, linewidth=0.5)
        axes[1, 0].plot(n_coop_smooth, label=name, color=colors[name], linewidth=2.5)
        
        # Scores don't need smoothing as they're cumulative
        axes[0, 1].plot(p_data[ql_id]['score'], label=name, color=colors[name], linewidth=2)
        axes[1, 1].plot(n_data[ql_id]['score'], label=name, color=colors[name], linewidth=2)

    axes[0, 0].set_title('Pairwise Cooperation'); axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Score'); axes[0, 1].set_ylabel('Cumulative Score')
    axes[1, 0].set_title('Neighborhood Cooperation'); axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Score'); axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat: 
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Round')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def create_heatmap(all_results, save_path):
    """Generates a heatmap comparing performance across all scenarios."""
    heatmap_data = []
    scenarios = sorted(all_results.keys())
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        v_p, v_n = results['Vanilla']
        a_p, a_n = results['Adaptive']
        v_id, a_id = "Vanilla_QL_1", "Adaptive_QL_1"

        metrics = [
            (np.mean(v_p[v_id]['coop_rate']), np.mean(a_p[a_id]['coop_rate'])),
            (v_p[v_id]['score'][-1], a_p[a_id]['score'][-1]),
            (np.mean(v_n[v_id]['coop_rate']), np.mean(a_n[a_id]['coop_rate'])),
            (v_n[v_id]['score'][-1], a_n[a_id]['score'][-1])
        ]

        row = [((a - v) / abs(v) * 100) if v != 0 else 0 for v, a in metrics]
        heatmap_data.append(row)

    df = pd.DataFrame(heatmap_data, index=scenarios, columns=['PW Coop', 'PW Score', 'N-Hood Coop', 'N-Hood Score'])
    plt.figure(figsize=(10, 8));
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title("Performance % Difference (Adaptive vs. Vanilla)", fontsize=16)
    plt.savefig(save_path);
    plt.close()


if __name__ == "__main__":
    NUM_ROUNDS = SIMULATION_CONFIG['num_rounds']
    NUM_RUNS = SIMULATION_CONFIG['num_runs']
    OUTPUT_DIR = "all_scenarios_comparison"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Running simulations with {NUM_ROUNDS} rounds and {NUM_RUNS} runs per scenario")

    # --- Define Scenarios ---
    scenario_opponents = {
        "vs_1_AllC": [StaticAgent("AllC_1", "AllC")],
        "vs_1_AllD": [StaticAgent("AllD_1", "AllD")],
        "vs_1_Random": [StaticAgent("Random_1", "Random")],
        "vs_1_TFT": [StaticAgent("TFT_1", "TFT")],
        "vs_1_TFT-E": [StaticAgent("TFT-E_1", "TFT", exploration_rate=0.1)],
    }

    # --- Define Agent Types to Test ---
    agent_types = {
        "Vanilla": {"class": PairwiseAdaptiveQLearner, "params": VANILLA_PARAMS},
        "Adaptive": {"class": PairwiseAdaptiveQLearner, "params": ADAPTIVE_PARAMS},
    }

    all_scenario_results = {}
    for scenario_name, opponents in scenario_opponents.items():
        print(f"\n===== Running Scenario: 2 QL Agents {scenario_name} =====")
        scenario_results = {}
        # We need two QL agents + the defined opponents
        full_opponents = [StaticAgent(**opponents[0].__dict__)]

        for agent_name, config in agent_types.items():
            print(f"  --- Testing {agent_name} agents ---")
            templates = [
                            config["class"](agent_id=f"{agent_name}_QL_1", params=config["params"]),
                            config["class"](agent_id=f"{agent_name}_QL_2", params=config["params"])
                        ] + full_opponents
            scenario_results[agent_name] = run_experiment_set(config, templates, NUM_ROUNDS, NUM_RUNS)

        all_scenario_results[f"2QL_{scenario_name}"] = scenario_results

        # Plot individual comparison for this scenario
        plot_path = os.path.join(OUTPUT_DIR, f"comparison_2QL_{scenario_name}.png")
        plot_scenario_comparison(scenario_results, f"2 QL Agents {scenario_name}", plot_path)

    # --- Generate Final Heatmap ---
    heatmap_path = os.path.join(OUTPUT_DIR, "summary_heatmap.png")
    create_heatmap(all_scenario_results, heatmap_path)

    print(f"\nAll experiments complete. Charts and heatmap saved to '{OUTPUT_DIR}'.")