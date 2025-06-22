from collections import defaultdict
from final_agents import COOPERATE, DEFECT

# --- Payoff Logic ---
PAYOFFS_2P = {
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}
T, R, P, S = 5, 3, 1, 0


def nperson_payoff(my_move, num_other_cooperators, total_agents):
    if total_agents <= 1: return R if my_move == COOPERATE else P
    if my_move == COOPERATE:
        return S + (R - S) * (num_other_cooperators / (total_agents - 1))
    else:
        return P + (T - P) * (num_other_cooperators / (total_agents - 1))


# --- Simulation Runners ---
def run_pairwise_tournament(agents, num_rounds):
    for agent in agents: agent.reset()

    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    agent_map = {a.agent_id: a for a in agents}
    opponent_last_moves = defaultdict(dict)

    for _ in range(num_rounds):
        moves = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                context1 = {'mode': 'pairwise', 'opponent_id': agent2.agent_id,
                            'last_opponent_move': opponent_last_moves[agent1.agent_id].get(agent2.agent_id)}
                context2 = {'mode': 'pairwise', 'opponent_id': agent1.agent_id,
                            'last_opponent_move': opponent_last_moves[agent2.agent_id].get(agent1.agent_id)}
                moves[(agent1.agent_id, agent2.agent_id)] = (agent1.choose_action(context1),
                                                             agent2.choose_action(context2))

        round_moves_by_agent = defaultdict(dict)
        for (id1, id2), (move1, move2) in moves.items():
            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]

            context1 = {'mode': 'pairwise', 'reward': payoff1, 'my_move': move1, 'opponent_id': id2,
                        'opponent_move': move2, 'last_opponent_move': move2}
            context2 = {'mode': 'pairwise', 'reward': payoff2, 'my_move': move2, 'opponent_id': id1,
                        'opponent_move': move1, 'last_opponent_move': move1}

            agent_map[id1].record_outcome(context1)
            agent_map[id2].record_outcome(context2)

            round_moves_by_agent[id1][id2] = move1
            round_moves_by_agent[id2][id1] = move2
            opponent_last_moves[id1][id2] = move2
            opponent_last_moves[id2][id1] = move1

        for agent in agents:
            if (len(agents) - 1) > 0:
                coop_moves = sum(1 for m in round_moves_by_agent[agent.agent_id].values() if m == COOPERATE)
                history[agent.agent_id]['coop_rate'].append(coop_moves / (len(agents) - 1))
            else:
                history[agent.agent_id]['coop_rate'].append(0)
            history[agent.agent_id]['score'].append(agent.total_score)

    return history


def run_nperson_simulation(agents, num_rounds):
    for agent in agents: agent.reset()
    history = {a.agent_id: {'coop_rate': [], 'score': []} for a in agents}
    coop_ratio = None

    for _ in range(num_rounds):
        context = {'mode': 'neighborhood', 'coop_ratio': coop_ratio}
        moves = {a.agent_id: a.choose_action(context) for a in agents}

        num_cooperators = sum(1 for m in moves.values() if m == COOPERATE)
        current_coop_ratio = num_cooperators / len(agents) if len(agents) > 0 else 0

        for agent in agents:
            my_move = moves[agent.agent_id]
            others_coop = num_cooperators - (1 if my_move == COOPERATE else 0)
            payoff = nperson_payoff(my_move, others_coop, len(agents))

            outcome_context = {'mode': 'neighborhood', 'reward': payoff, 'my_move': my_move,
                               'coop_ratio': current_coop_ratio}
            agent.record_outcome(outcome_context)

            history[agent.agent_id]['coop_rate'].append(1 if my_move == COOPERATE else 0)
            history[agent.agent_id]['score'].append(agent.total_score)

        coop_ratio = current_coop_ratio

    return history