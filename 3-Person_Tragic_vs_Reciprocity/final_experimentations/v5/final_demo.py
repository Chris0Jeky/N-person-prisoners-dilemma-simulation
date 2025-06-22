# final_demo.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

from final_agents import StaticAgent, PairwiseAdaptiveQLearner, NeighborhoodAdaptiveQLearner
from config import VANILLA_PARAMS, ADAPTIVE_PARAMS

# --- Payoff Logic ---
PAYOFFS_2P = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_other_cooperators, total_agents):
    # In this simplified payoff, move 0 is cooperate, 1 is defect.
    # So (1-my_move) is 1 if coop, 0 if defect.
    others_coop = num_other_cooperators - (1 - my_move)
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


# --- Main ---
if __name__ == "__main__":
    NUM_ROUNDS = 1000
    NUM_RUNS = 20
    OUTPUT_DIR = "final_comparison_charts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    opponents = [StaticAgent(agent_id="TFT_1"), StaticAgent(agent_id="TFT_2")]

    # Store results for plotting
    all_results = defaultdict(dict)

    # --- Agent Configurations to Test ---
    agent_configs = {
        "Vanilla": {"pairwise_class": PairwiseAdaptiveQLearner, "neighborhood_class": NeighborhoodAdaptiveQLearner,
                    "params": VANILLA_PARAMS},
        "Adaptive": {"pairwise_class": PairwiseAdaptiveQLearner, "neighborhood_class": NeighborhoodAdaptiveQLearner,
                     "params": ADAPTIVE_PARAMS},
    }

    for name, config in agent_configs.items():
        print(f"\n--- Running Scenario: {name} ---")

        # --- Pairwise Simulation ---
        print("  Running Pairwise Tournament...")
        p_templates = [config["pairwise_class"](agent_id=f"{name}_QL_1", params=config["params"])] + opponents
        p_runs = [run_pairwise_tournament([type(a)(**a.__dict__) for a in p_templates], NUM_ROUNDS) for _ in
                  range(NUM_RUNS)]
        p_agg = {aid: {m: np.mean([r[aid][m] for r in p_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} for
                 aid in p_runs[0]}
        all_results[name]['pairwise'] = p_agg

        # --- Neighborhood Simulation ---
        print("  Running Neighborhood Simulation...")
        n_templates = [config["neighborhood_class"](agent_id=f"{name}_QL_1", params=config["params"])] + opponents
        n_runs = [run_nperson_simulation([type(a)(**a.__dict__) for a in n_templates], NUM_ROUNDS) for _ in
                  range(NUM_RUNS)]
        n_agg = {aid: {m: np.mean([r[aid][m] for r in n_runs if aid in r], axis=0) for m in ['coop_rate', 'score']} for
                 aid in n_runs[0]}
        all_results[name]['neighborhood'] = n_agg

    # --- Plotting Results ---
    print("\n--- Generating Comparison Plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("Agent Performance vs. 2 TFTs", fontsize=22)

    colors = {"Vanilla": "#1f77b4", "Adaptive": "#ff7f0e"}

    for name, results in all_results.items():
        ql_id = f"{name}_QL_1"
        axes[0, 0].plot(results['pairwise'][ql_id]['coop_rate'], label=name, color=colors[name])
        axes[0, 1].plot(results['pairwise'][ql_id]['score'], label=name, color=colors[name])
        axes[1, 0].plot(results['neighborhood'][ql_id]['coop_rate'], label=name, color=colors[name])
        axes[1, 1].plot(results['neighborhood'][ql_id]['score'], label=name, color=colors[name])

    axes[0, 0].set_title('Pairwise Cooperation');
    axes[0, 0].set_ylabel('Cooperation Rate')
    axes[0, 1].set_title('Pairwise Score');
    axes[0, 1].set_ylabel('Cumulative Score')
    axes[1, 0].set_title('Neighborhood Cooperation');
    axes[1, 0].set_ylabel('Cooperation Rate')
    axes[1, 1].set_title('Neighborhood Score');
    axes[1, 1].set_ylabel('Cumulative Score')

    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(OUTPUT_DIR, "final_vanilla_vs_adaptive.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Chart saved to {save_path}")