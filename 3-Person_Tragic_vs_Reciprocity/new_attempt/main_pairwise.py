# main_pairwise.py
import random

# --- Constants ---
COOPERATE = 0
DEFECT = 1
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3), (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0), (DEFECT, DEFECT): (1, 1),
}


def move_to_str(move):
    return "Cooperate" if move == COOPERATE else "Defect"


# --- Agent Class (Pairwise) ---
class Agent:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate
        self.total_score = 0
        self.opponent_last_moves = {}
        self.num_cooperations = 0
        self.num_defections = 0
        self.moves_history_detailed = []

    def choose_action(self, opponent_id, current_round_in_interaction):
        intended_move = None
        if self.strategy_name == "TFT":
            if current_round_in_interaction == 0 or opponent_id not in self.opponent_last_moves:
                intended_move = COOPERATE
            else:
                intended_move = self.opponent_last_moves[opponent_id]
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        return intended_move, actual_move

    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                           my_intended_move, my_actual_move, round_num_in_interaction):
        self.total_score += my_payoff
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        if my_actual_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        self.moves_history_detailed.append({
            "round_in_interaction": round_num_in_interaction, "opponent_id": opponent_id,
            "my_intended_move": move_to_str(my_intended_move), "my_actual_move": move_to_str(my_actual_move),
            "opponent_actual_move": move_to_str(opponent_actual_move), "my_payoff": my_payoff
        })

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset_for_new_tournament(self):
        self.total_score = 0
        self.opponent_last_moves = {}
        self.moves_history_detailed = []
        self.num_cooperations = 0
        self.num_defections = 0


# --- Game Class (Pairwise) ---
class IteratedPrisonersDilemma:
    def __init__(self, agents, num_rounds_pairwise):
        self.agents = agents
        self.num_rounds_pairwise = num_rounds_pairwise
        self.payoff_matrix = PAYOFFS
        self.tournament_log = []

    def _play_single_round(self, agent1, agent2, current_round_in_interaction):
        intended_move1, actual_move1 = agent1.choose_action(agent2.agent_id, current_round_in_interaction)
        intended_move2, actual_move2 = agent2.choose_action(agent1.agent_id, current_round_in_interaction)
        payoff1, payoff2 = self.payoff_matrix[(actual_move1, actual_move2)]
        agent1.record_interaction(agent2.agent_id, actual_move2, payoff1, intended_move1, actual_move1,
                                  current_round_in_interaction)
        agent2.record_interaction(agent1.agent_id, actual_move1, payoff2, intended_move2, actual_move2,
                                  current_round_in_interaction)
        return {
            "agent1_id": agent1.agent_id, "agent2_id": agent2.agent_id,
            "round_in_interaction": current_round_in_interaction,
            "agent1_strategy": agent1.strategy_name, "agent2_strategy": agent2.strategy_name,
            "agent1_intended_move": move_to_str(intended_move1), "agent1_actual_move": move_to_str(actual_move1),
            "agent2_intended_move": move_to_str(intended_move2), "agent2_actual_move": move_to_str(actual_move2),
            "agent1_payoff": payoff1, "agent2_payoff": payoff2
        }

    def run_pairwise_interaction(self, agent1, agent2):
        interaction_log = []
        for i in range(self.num_rounds_pairwise):
            interaction_log.append(self._play_single_round(agent1, agent2, i))
        return interaction_log

    def run_tournament(self):
        self.tournament_log = []
        for agent in self.agents:
            agent.reset_for_new_tournament()
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                self.tournament_log.extend(self.run_pairwise_interaction(self.agents[i], self.agents[j]))
        self.print_tournament_results()

    def print_tournament_results(self):
        print("\n--- Pairwise Tournament Results ---")
        total_coops_all, total_moves_all = 0, 0
        sorted_agents = sorted(self.agents, key=lambda ag: ag.total_score, reverse=True)
        for agent in sorted_agents:
            print(
                f"\nAgent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate * 100:.1f}%)")
            print(f"  Total Score: {agent.total_score}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")
            total_coops_all += agent.num_cooperations
            total_moves_all += total_agent_moves
        if total_moves_all > 0:
            print(f"\nOverall Cooperation Rate: {total_coops_all / total_moves_all:.2f}")
        self.analyze_cooperation_by_interaction_type()

    def analyze_cooperation_by_interaction_type(self):
        # (Simplified for brevity, can be expanded as in the original)
        print("\n--- Cooperation Analysis by Interaction Type (Pairwise) ---")
        stats = {"TFT_vs_TFT": {"coops": 0, "moves": 0}, "TFT_vs_AllD_TFT": {"coops": 0, "moves": 0},
                 "TFT_vs_AllD_AllD": {"coops": 0, "moves": 0}}
        for r in self.tournament_log:
            s1, s2 = r['agent1_strategy'], r['agent2_strategy']
            m1, m2 = r['agent1_actual_move'], r['agent2_actual_move']
            if s1 == "TFT" and s2 == "TFT":
                stats["TFT_vs_TFT"]["moves"] += 2
                if m1 == "Cooperate": stats["TFT_vs_TFT"]["coops"] += 1
                if m2 == "Cooperate": stats["TFT_vs_TFT"]["coops"] += 1
            elif (s1 == "TFT" and s2 == "AllD") or (s1 == "AllD" and s2 == "TFT"):
                tft_move = m1 if s1 == "TFT" else m2
                alld_move = m2 if s1 == "TFT" else m1
                stats["TFT_vs_AllD_TFT"]["moves"] += 1
                if tft_move == "Cooperate": stats["TFT_vs_AllD_TFT"]["coops"] += 1
                stats["TFT_vs_AllD_AllD"]["moves"] += 1
                if alld_move == "Cooperate": stats["TFT_vs_AllD_AllD"]["coops"] += 1

        if stats["TFT_vs_TFT"]["moves"] > 0: print(
            f"TFT vs TFT: {stats['TFT_vs_TFT']['coops'] / stats['TFT_vs_TFT']['moves']:.2f} coop rate")
        if stats["TFT_vs_AllD_TFT"]["moves"] > 0: print(
            f"TFT (vs AllD): {stats['TFT_vs_AllD_TFT']['coops'] / stats['TFT_vs_AllD_TFT']['moves']:.2f} coop rate")
        if stats["TFT_vs_AllD_AllD"]["moves"] > 0: print(
            f"AllD (vs TFT): {stats['TFT_vs_AllD_AllD']['coops'] / stats['TFT_vs_AllD_AllD']['moves']:.2f} coop rate")


def run_pairwise_experiment(agent_configurations, num_rounds_pairwise_param):
    agents_for_experiment = []
    for config in agent_configurations:
        agents_for_experiment.append(
            Agent(agent_id=config['id'],
                  strategy_name=config['strategy'],
                  exploration_rate=config['exploration_rate'])
        )

    # print(f"--- Initializing Pairwise Experiment ---") # Moved to runner for less verbosity per call
    # print(f"Agent Configurations:")
    # for config in agent_configurations:
    #     print(f"  - ID: {config['id']}, Strategy: {config['strategy']}, Exploration: {config['exploration_rate']*100:.1f}%")
    # print(f"Rounds per pairwise interaction: {num_rounds_pairwise_param}")

    game_simulation = IteratedPrisonersDilemma(
        agents=agents_for_experiment,
        num_rounds_pairwise=num_rounds_pairwise_param
    )
    game_simulation.run_tournament()
    # print(f"--- Pairwise Experiment Finished ---") # Moved to runner
    return game_simulation


if __name__ == "__main__":
    # Example test run for main_pairwise.py
    print("Running main_pairwise.py as standalone test...")
    default_agent_configs = [
        {'id': "TFT_1_pair_test", 'strategy': "TFT", 'exploration_rate': 0.05},
        {'id': "TFT_2_pair_test", 'strategy': "TFT", 'exploration_rate': 0.05},
        {'id': "AllD_1_pair_test", 'strategy': "AllD", 'exploration_rate': 0.05}
    ]
    run_pairwise_experiment(default_agent_configs, num_rounds_pairwise_param=50)