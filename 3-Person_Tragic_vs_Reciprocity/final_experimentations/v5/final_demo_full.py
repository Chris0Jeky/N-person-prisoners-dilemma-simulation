import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_cooperators, total_agents):
    # num_cooperators includes all agents who cooperated (including self if applicable)
    # Calculate the number of OTHER agents who cooperated
    others_coop = num_cooperators - (1 - my_move)  # subtract 1 if I cooperated
    if my_move == 0:  # cooperate
        return S + (R - S) * (others_coop / (total_agents - 1))
    else:  # defect
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


def run_experiment_set(agents, num_rounds, num_runs):
    """Runs both pairwise and neighborhood simulations for a given agent configuration."""
    # Pairwise simulations
    p_runs = [run_pairwise_tournament([type(a)(**a.__dict__) for a in agents], num_rounds) for _ in range(num_runs)]
    p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in p_runs[0]}
    
    # Neighborhood simulations  
    n_runs = [run_nperson_simulation([type(a)(**a.__dict__) for a in agents], num_rounds) for _ in range(num_runs)]
    n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} 
             for aid in n_runs[0]}
    
    return p_agg, n_agg


def plot_scenario_comparison(results, title, save_path):
    """Generates a 4-panel plot comparing Vanilla vs. Adaptive for one scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f"Comparison: {title}", fontsize=22)
    colors = {"Vanilla": "#1f77b4", "Adaptive": "#ff7f0e"}
    
    for agent_type, (p_data, n_data) in results.items():
        # Find the QL agents in the results
        ql_agents = [aid for aid in p_data.keys() if agent_type in aid and 'QL' in aid]
        if not ql_agents:
            continue
            
        # Average across all QL agents of this type
        p_coop = np.mean([p_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
        p_score = np.mean([p_data[aid]['score'] for aid in ql_agents], axis=0)
        n_coop = np.mean([n_data[aid]['coop_rate'] for aid in ql_agents], axis=0)
        n_score = np.mean([n_data[aid]['score'] for aid in ql_agents], axis=0)
        
        axes[0, 0].plot(p_coop, label=agent_type, color=colors[agent_type], linewidth=2)
        axes[0, 1].plot(p_score, label=agent_type, color=colors[agent_type], linewidth=2)
        axes[1, 0].plot(n_coop, label=agent_type, color=colors[agent_type], linewidth=2)
        axes[1, 1].plot(n_score, label=agent_type, color=colors[agent_type], linewidth=2)
    
    axes[0, 0].set_title('Pairwise Cooperation Rate'); axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Cumulative Score'); axes[0, 1].set_ylabel('Cumulative Score')
    axes[1, 0].set_title('Neighborhood Cooperation Rate'); axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Cumulative Score'); axes[1, 1].set_ylabel('Cumulative Score')
    
    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def create_heatmap(all_results, save_path):
    """Generates a heatmap comparing performance across all scenarios."""
    heatmap_data = []
    scenario_names = []
    
    for scenario_name, results in sorted(all_results.items()):
        if 'Vanilla' not in results or 'Adaptive' not in results:
            continue
            
        v_p, v_n = results['Vanilla']
        a_p, a_n = results['Adaptive']
        
        # Find QL agents
        v_ql = [aid for aid in v_p.keys() if 'Vanilla' in aid and 'QL' in aid]
        a_ql = [aid for aid in a_p.keys() if 'Adaptive' in aid and 'QL' in aid]
        
        if not v_ql or not a_ql:
            continue
        
        # Calculate average metrics
        v_p_coop = np.mean([np.mean(v_p[aid]['coop_rate']) for aid in v_ql])
        a_p_coop = np.mean([np.mean(a_p[aid]['coop_rate']) for aid in a_ql])
        v_p_score = np.mean([v_p[aid]['score'][-1] for aid in v_ql])
        a_p_score = np.mean([a_p[aid]['score'][-1] for aid in a_ql])
        
        v_n_coop = np.mean([np.mean(v_n[aid]['coop_rate']) for aid in v_ql])
        a_n_coop = np.mean([np.mean(a_n[aid]['coop_rate']) for aid in a_ql])
        v_n_score = np.mean([v_n[aid]['score'][-1] for aid in v_ql])
        a_n_score = np.mean([a_n[aid]['score'][-1] for aid in a_ql])
        
        # Calculate percentage improvements
        metrics = [
            (v_p_coop, a_p_coop),
            (v_p_score, a_p_score),
            (v_n_coop, a_n_coop),
            (v_n_score, a_n_score)
        ]
        
        row = []
        for v, a in metrics:
            if v != 0:
                improvement = ((a - v) / abs(v)) * 100
            else:
                improvement = 100 if a > 0 else 0
            row.append(improvement)
        
        heatmap_data.append(row)
        scenario_names.append(scenario_name)
    
    if not heatmap_data:
        print("No data for heatmap")
        return
    
    df = pd.DataFrame(heatmap_data, 
                      index=scenario_names,
                      columns=['Pairwise\nCooperation', 'Pairwise\nScore', 
                               'Neighborhood\nCooperation', 'Neighborhood\nScore'])
    
    plt.figure(figsize=(10, len(scenario_names) * 0.5 + 2))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': '% Improvement'})
    plt.title("Performance Improvement: Adaptive vs. Vanilla Q-Learning\n(Positive = Adaptive Better)", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Scenarios", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 20
    OUTPUT_DIR = "final_comparison_charts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define opponent strategies
    opponent_configs = {
        "AllC": {"strategy": "AllC", "error_rate": 0.0},
        "AllD": {"strategy": "AllD", "error_rate": 0.0},
        "Random": {"strategy": "Random", "error_rate": 0.0},
        "TFT": {"strategy": "TFT", "error_rate": 0.0},
        "TFT-E": {"strategy": "TFT-E", "error_rate": 0.1},
    }
    
    # Define Q-learner configurations
    ql_configs = {
        "Vanilla": {"pairwise_class": PairwiseAdaptiveQLearner, 
                    "neighborhood_class": NeighborhoodAdaptiveQLearner,
                    "params": VANILLA_PARAMS},
        "Adaptive": {"pairwise_class": PairwiseAdaptiveQLearner,
                     "neighborhood_class": NeighborhoodAdaptiveQLearner, 
                     "params": ADAPTIVE_PARAMS},
    }
    
    all_scenario_results = {}
    
    # --- Scenario 1: 2 QL vs. 1 Opponent ---
    print("\n=== Running 2 QL vs. 1 Opponent Scenarios ===")
    for opp_name, opp_config in opponent_configs.items():
        scenario_name = f"2QL_vs_1{opp_name}"
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        
        for ql_name, ql_config in ql_configs.items():
            print(f"  Testing {ql_name} Q-learners...")
            agents = [
                ql_config["pairwise_class"](agent_id=f"{ql_name}_QL_1", params=ql_config["params"]),
                ql_config["pairwise_class"](agent_id=f"{ql_name}_QL_2", params=ql_config["params"]),
                StaticAgent(agent_id=f"{opp_name}_1", strategy_name=opp_config["strategy"], 
                           error_rate=opp_config["error_rate"])
            ]
            scenario_results[ql_name] = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS)
        
        all_scenario_results[scenario_name] = scenario_results
        plot_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.png")
        plot_scenario_comparison(scenario_results, scenario_name, plot_path)
        print(f"  Saved plot: {plot_path}")
    
    # --- Scenario 2: 1 QL vs. 2 Opponents ---
    print("\n=== Running 1 QL vs. 2 Opponents Scenarios ===")
    for opp_name, opp_config in opponent_configs.items():
        scenario_name = f"1QL_vs_2{opp_name}"
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        
        for ql_name, ql_config in ql_configs.items():
            print(f"  Testing {ql_name} Q-learner...")
            agents = [
                ql_config["pairwise_class"](agent_id=f"{ql_name}_QL_1", params=ql_config["params"]),
                StaticAgent(agent_id=f"{opp_name}_1", strategy_name=opp_config["strategy"],
                           error_rate=opp_config["error_rate"]),
                StaticAgent(agent_id=f"{opp_name}_2", strategy_name=opp_config["strategy"],
                           error_rate=opp_config["error_rate"])
            ]
            scenario_results[ql_name] = run_experiment_set(agents, NUM_ROUNDS, NUM_RUNS)
        
        all_scenario_results[scenario_name] = scenario_results
        plot_path = os.path.join(OUTPUT_DIR, f"{scenario_name}.png")
        plot_scenario_comparison(scenario_results, scenario_name, plot_path)
        print(f"  Saved plot: {plot_path}")
    
    # --- Generate Summary Heatmap ---
    print("\n=== Generating Summary Heatmap ===")
    heatmap_path = os.path.join(OUTPUT_DIR, "summary_heatmap_all_scenarios.png")
    create_heatmap(all_scenario_results, heatmap_path)
    print(f"Heatmap saved: {heatmap_path}")
    
    print(f"\nAll experiments complete! Results saved to '{OUTPUT_DIR}'")