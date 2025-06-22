import numpy as np
from final_agents import COOPERATE, DEFECT, PAYOFFS_2P, nperson_payoff


def run_pairwise_simulation(agents, num_rounds):
    """Runs a pairwise tournament using the clean, unified API."""
    for agent in agents: agent.reset()
    all_coop_history = {a.agent_id: [] for a in agents}
    score_history = {a.agent_id: [] for a in agents}

    for _ in range(num_rounds):
        # 1. All agents choose their moves against all opponents
        moves_this_round = {}
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                move1 = agent1.choose_action(opponent_id=agent2.agent_id)
                move2 = agent2.choose_action(opponent_id=agent1.agent_id)
                moves_this_round[(agent1.agent_id, agent2.agent_id)] = (move1, move2)

        # 2. Calculate payoffs and record outcomes
        round_moves_by_agent = {a.agent_id: {} for a in agents}
        for (id1, id2), (move1, move2) in moves_this_round.items():
            agent1 = next(a for a in agents if a.agent_id == id1)
            agent2 = next(a for a in agents if a.agent_id == id2)
            round_moves_by_agent[id1][id2] = move1
            round_moves_by_agent[id2][id1] = move2

            payoff1, payoff2 = PAYOFFS_2P[(move1, move2)]
            agent1.record_outcome(payoff=payoff1, my_move=move1, opponent_id=id2, opponent_move=move2)
            agent2.record_outcome(payoff=payoff2, my_move=move2, opponent_id=id1, opponent_move=move1)

        # 3. Log history for this round
        for agent in agents:
            coop_count = sum(1 for move in round_moves_by_agent[agent.agent_id].values() if move == COOPERATE)
            all_coop_history[agent.agent_id].append(coop_count / (len(agents) - 1))
            score_history[agent.agent_id].append(agent.total_score)

    return all_coop_history, score_history


def run_nperson_simulation(agents, num_rounds):
    """Runs an N-person simulation using the clean, unified API."""
    for agent in agents: agent.reset()
    all_coop_history = {a.agent_id: [] for a in agents}
    score_history = {a.agent_id: [] for a in agents}
    prev_round_coop_ratio = None

    for _ in range(num_rounds):
        moves = {a.agent_id: a.choose_action(prev_round_group_coop_ratio=prev_round_coop_ratio) for a in agents}
        num_cooperators = sum(1 for move in moves.values() if move == COOPERATE)
        current_coop_ratio = num_cooperators / len(agents)

        for agent in agents:
            my_move = moves[agent.agent_id]
            payoff = nperson_payoff(my_move, num_cooperators - (1 if my_move == COOPERATE else 0), len(agents))
            agent.record_outcome(payoff=payoff, my_move=my_move, prev_round_group_coop_ratio=current_coop_ratio)
            all_coop_history[agent.agent_id].append(1 if my_move == COOPERATE else 0)
            score_history[agent.agent_id].append(agent.total_score)

        prev_round_coop_ratio = current_coop_ratio

    return all_coop_history, score_history