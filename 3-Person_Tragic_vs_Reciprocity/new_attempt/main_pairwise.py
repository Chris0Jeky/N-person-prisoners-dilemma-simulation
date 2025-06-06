import random

# --- Part 1: Contents of main_pairwise.py (adapted with episodes) ---

PAIRWISE_COOPERATE = 0
PAIRWISE_DEFECT = 1
PAIRWISE_PAYOFFS = {
    (PAIRWISE_COOPERATE, PAIRWISE_COOPERATE): (3, 3), (PAIRWISE_COOPERATE, PAIRWISE_DEFECT): (0, 5),
    (PAIRWISE_DEFECT, PAIRWISE_COOPERATE): (5, 0), (PAIRWISE_DEFECT, PAIRWISE_DEFECT): (1, 1),
}

def pairwise_move_to_str(move):
    return "Cooperate" if move == PAIRWISE_COOPERATE else "Defect"

class PairwiseAgent:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        self.total_score = 0
        self.opponent_last_moves = {} # Key: opponent_id, Value: opponent's last move
        self.num_cooperations = 0
        self.num_defections = 0

    def choose_action(self, opponent_id, current_round_in_episode): # Changed param name
        intended_move = None
        if self.strategy_name == "TFT":
            # Cooperate on the first move of an episode or if no history with this opponent for current episode context
            if current_round_in_episode == 0 or opponent_id not in self.opponent_last_moves:
                intended_move = PAIRWISE_COOPERATE
            else:
                intended_move = self.opponent_last_moves[opponent_id]
        elif self.strategy_name == "AllD":
            intended_move = PAIRWISE_DEFECT
        else:
            raise ValueError(f"Unknown pairwise strategy: {self.strategy_name}")

        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        return intended_move, actual_move

    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                           my_intended_move, my_actual_move, round_num_in_episode):
        self.total_score += my_payoff
        self.opponent_last_moves[opponent_id] = opponent_actual_move # Store for next round in *this* episode
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1

    def clear_opponent_history(self, opponent_id):
        """Clears the history for a specific opponent, typically between episodes."""
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset_for_new_tournament(self): # Full reset for a new tournament run by the runner
        self.total_score = 0
        self.opponent_last_moves = {}
        self.num_cooperations = 0
        self.num_defections = 0

class PairwiseIteratedPrisonersDilemma:
    def __init__(self, agents, num_episodes, rounds_per_episode):
        self.agents = agents
        self.num_episodes = num_episodes
        self.rounds_per_episode = rounds_per_episode
        self.payoff_matrix = PAIRWISE_PAYOFFS

    def _play_single_round(self, agent1, agent2, current_round_in_episode):
        intended_move1, actual_move1 = agent1.choose_action(agent2.agent_id, current_round_in_episode)
        intended_move2, actual_move2 = agent2.choose_action(agent1.agent_id, current_round_in_episode)
        payoff1, payoff2 = self.payoff_matrix[(actual_move1, actual_move2)]
        agent1.record_interaction(agent2.agent_id, actual_move2, payoff1, intended_move1, actual_move1, current_round_in_episode)
        agent2.record_interaction(agent1.agent_id, actual_move1, payoff2, intended_move2, actual_move2, current_round_in_episode)

    def run_pairwise_interaction(self, agent1, agent2):
        for episode_num in range(self.num_episodes):
            for round_num_in_episode in range(self.rounds_per_episode):
                self._play_single_round(agent1, agent2, round_num_in_episode)
            
            # After an episode, if there are more episodes to come, reset memory for this pair
            if self.num_episodes > 1 and episode_num < self.num_episodes - 1:
                agent1.clear_opponent_history(agent2.agent_id)
                agent2.clear_opponent_history(agent1.agent_id)


    def run_tournament(self):
        for agent in self.agents: # Full reset before tournament starts
            agent.reset_for_new_tournament()
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                self.run_pairwise_interaction(self.agents[i], self.agents[j])
        self.print_tournament_results()

    def print_tournament_results(self):
        print("\n--- Pairwise Tournament Results ---")
        print(f"(Episodes: {self.num_episodes}, Rounds/Episode: {self.rounds_per_episode})")
        total_coops_all, total_moves_all = 0, 0
        sorted_agents = sorted(self.agents, key=lambda ag: ag.total_score, reverse=True)
        for agent in sorted_agents:
            print(f"Agent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate*100:.1f}%)")
            print(f"  Total Score: {agent.total_score}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")
            total_coops_all += agent.num_cooperations
            total_moves_all += total_agent_moves
        if total_moves_all > 0:
            print(f"Overall Pairwise Cooperation Rate: {total_coops_all/total_moves_all:.2f}")

def run_pairwise_experiment(agent_configurations, total_rounds_per_pair, 
                            episodic_mode, num_episodes_if_episodic):
    agents_for_experiment = []
    for config in agent_configurations:
        agents_for_experiment.append(
            PairwiseAgent(agent_id=config['id'], 
                          strategy_name=config['strategy'], 
                          exploration_rate=config['exploration_rate'])
        )
    
    _num_episodes = 1
    _rounds_per_episode = total_rounds_per_pair
    if episodic_mode:
        _num_episodes = num_episodes_if_episodic
        if total_rounds_per_pair % _num_episodes != 0:
            raise ValueError("Total rounds must be divisible by num_episodes for episodic mode.")
        _rounds_per_episode = total_rounds_per_pair // _num_episodes
    
    game_simulation = PairwiseIteratedPrisonersDilemma(
        agents=agents_for_experiment,
        num_episodes=_num_episodes,
        rounds_per_episode=_rounds_per_episode
    )
    game_simulation.run_tournament()
    return game_simulation