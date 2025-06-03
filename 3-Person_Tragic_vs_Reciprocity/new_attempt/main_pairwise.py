import random

# --- Constants ---
COOPERATE = 0  # Representing "Cooperate"
DEFECT = 1  # Representing "Defect"

# Payoff matrix: (Agent1_Payoff, Agent2_Payoff)
# R=3 (Reward), S=0 (Sucker), T=5 (Temptation), P=1 (Punishment)
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),  # Both cooperate
    (COOPERATE, DEFECT): (0, 5),  # Agent1 cooperates, Agent2 defects
    (DEFECT, COOPERATE): (5, 0),  # Agent1 defects, Agent2 cooperates
    (DEFECT, DEFECT): (1, 1),  # Both defect
}


def move_to_str(move):
    """Converts a move (0 or 1) to a string representation."""
    return "Cooperate" if move == COOPERATE else "Defect"


# --- Agent Class ---
class Agent:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate

        self.total_score = 0
        # Stores the last actual move made by an opponent against this agent.
        # Key: opponent_id, Value: opponent's last actual move (COOPERATE or DEFECT)
        self.opponent_last_moves = {}

        # For cooperation rate calculation and detailed history
        self.num_cooperations = 0
        self.num_defections = 0
        self.moves_history_detailed = []  # Stores dicts of interaction details

    def choose_action(self, opponent_id, current_round_in_interaction):
        """
        Determines the agent's intended move based on its strategy and opponent's history.
        Then, applies exploration/error to get the actual move.
        current_round_in_interaction is 0-indexed.
        """
        intended_move = None
        if self.strategy_name == "TFT":
            # Cooperate on the first move of a pairwise interaction or if no history with this opponent
            if current_round_in_interaction == 0 or opponent_id not in self.opponent_last_moves:
                intended_move = COOPERATE
            else:
                # Mimic the opponent's last actual move against this agent
                intended_move = self.opponent_last_moves[opponent_id]
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name} for agent {self.agent_id}")

        # Apply exploration/error
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move  # Flip the move (0 becomes 1, 1 becomes 0)

        return intended_move, actual_move

    def record_interaction(self, opponent_id, opponent_actual_move, my_payoff,
                           my_intended_move, my_actual_move, round_num_in_interaction):
        """Records the outcome of an interaction round."""
        self.total_score += my_payoff
        # Store the opponent's actual move for TFT's next decision against this opponent
        self.opponent_last_moves[opponent_id] = opponent_actual_move

        if my_actual_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1

        self.moves_history_detailed.append({
            "round_in_interaction": round_num_in_interaction,
            "opponent_id": opponent_id,
            "my_intended_move": move_to_str(my_intended_move),
            "my_actual_move": move_to_str(my_actual_move),
            "opponent_actual_move": move_to_str(opponent_actual_move),
            "my_payoff": my_payoff
        })

    def get_cooperation_rate(self):
        """Calculates the cooperation rate for this agent."""
        total_moves = self.num_cooperations + self.num_defections
        if total_moves == 0:
            return 0.0
        return self.num_cooperations / total_moves

    def reset_for_new_tournament(self):
        """Resets agent's state for a new tournament (if needed)."""
        self.total_score = 0
        self.opponent_last_moves = {}
        self.moves_history_detailed = []
        self.num_cooperations = 0
        self.num_defections = 0


# --- Game Class ---
class IteratedPrisonersDilemma:
    def __init__(self, agents, num_rounds_pairwise):
        self.agents = agents
        self.num_rounds_pairwise = num_rounds_pairwise
        self.payoff_matrix = PAYOFFS
        # Stores a log of all rounds in the tournament:
        # list of dicts like {'agent1_id': ..., 'agent2_id': ..., 'round': ..., ...}
        self.tournament_log = []

    def _play_single_round(self, agent1, agent2, current_round_in_interaction):
        """Simulates a single round of interaction between two agents."""
        # Agents decide their moves based on history with *this* opponent and round number
        intended_move1, actual_move1 = agent1.choose_action(agent2.agent_id, current_round_in_interaction)
        intended_move2, actual_move2 = agent2.choose_action(agent1.agent_id, current_round_in_interaction)

        # Determine payoffs based on actual moves
        payoff1, payoff2 = self.payoff_matrix[(actual_move1, actual_move2)]

        # Agents record the outcome (including the opponent's *actual* move)
        agent1.record_interaction(agent2.agent_id, actual_move2, payoff1,
                                  intended_move1, actual_move1, current_round_in_interaction)
        agent2.record_interaction(agent1.agent_id, actual_move1, payoff2,
                                  intended_move2, actual_move2, current_round_in_interaction)

        return {
            "agent1_id": agent1.agent_id, "agent2_id": agent2.agent_id,
            "round_in_interaction": current_round_in_interaction,
            "agent1_strategy": agent1.strategy_name, "agent2_strategy": agent2.strategy_name,
            "agent1_intended_move": move_to_str(intended_move1), "agent1_actual_move": move_to_str(actual_move1),
            "agent2_intended_move": move_to_str(intended_move2), "agent2_actual_move": move_to_str(actual_move2),
            "agent1_payoff": payoff1, "agent2_payoff": payoff2
        }

    def run_pairwise_interaction(self, agent1, agent2):
        """Runs a series of rounds between two specific agents."""
        # For TFT, starting fresh with a new opponent means cooperating.
        # The `current_round_in_interaction == 0` check in `agent.choose_action` handles this.
        # No explicit reset of `agent.opponent_last_moves` is needed here because it's keyed by opponent_id.

        interaction_log = []
        for i in range(self.num_rounds_pairwise):
            round_result = self._play_single_round(agent1, agent2, i)
            interaction_log.append(round_result)
        return interaction_log

    def run_tournament(self):
        """Runs interactions between all unique pairs of agents."""
        self.tournament_log = []
        # Reset agents if this tournament is part of multiple experiments
        for agent in self.agents:
            agent.reset_for_new_tournament()  # Clears scores and history for a fresh tournament

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]

                # Log interaction details
                # print(f"\n--- Starting Interaction: {agent1.agent_id} vs {agent2.agent_id} ---")

                interaction_log_for_pair = self.run_pairwise_interaction(agent1, agent2)
                self.tournament_log.extend(interaction_log_for_pair)

        self.print_tournament_results()

    def print_tournament_results(self):
        """Prints the summary of the tournament results."""
        print("\n--- Tournament Results ---")
        total_cooperations_all_agents = 0
        total_moves_all_agents = 0

        # Sort agents by total score for ranked display (optional)
        sorted_agents = sorted(self.agents, key=lambda ag: ag.total_score, reverse=True)

        for agent in sorted_agents:
            print(
                f"\nAgent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate * 100}%)")
            print(f"  Total Score: {agent.total_score}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")

            total_cooperations_all_agents += agent.num_cooperations
            total_moves_all_agents += total_agent_moves

            # To see detailed moves for a specific agent:
            # print("  Detailed Moves Log:")
            # for move_record in agent.moves_history_detailed:
            #     print(f"    vs {move_record['opponent_id']}, Rnd {move_record['round_in_interaction']}: Intended {move_record['my_intended_move']}, Actual {move_record['my_actual_move']}, Opponent Actual {move_record['opponent_actual_move']}, Payoff {move_record['my_payoff']}")

        if total_moves_all_agents > 0:
            overall_coop_rate = total_cooperations_all_agents / total_moves_all_agents
            print(f"\n--- Overall Tournament Summary ---")
            print(
                f"Overall Cooperation Rate (all agents): {overall_coop_rate:.2f} ({total_cooperations_all_agents} C / {total_moves_all_agents} total moves)")
        else:
            print("\nNo moves were made in the tournament.")

        self.analyze_cooperation_by_interaction_type()

    def analyze_cooperation_by_interaction_type(self):
        """Analyzes cooperation rates based on the types of strategies interacting."""
        print("\n--- Cooperation Analysis by Interaction Type ---")

        # Counters for specific interaction types
        # (move_type_agent1_coops, move_type_agent1_total_moves)
        interaction_stats = {
            "TFT_vs_TFT": {"coops": 0, "moves": 0},  # Moves by TFT agents in TFT vs TFT
            "TFT_vs_AllD": {"tft_coops": 0, "tft_moves": 0,  # TFT's moves when playing AllD
                            "alld_coops": 0, "alld_moves": 0}  # AllD's moves when playing TFT
        }

        for record in self.tournament_log:
            s1_name = record['agent1_strategy']
            s2_name = record['agent2_strategy']

            act1 = record['agent1_actual_move']
            act2 = record['agent2_actual_move']

            if s1_name == "TFT" and s2_name == "TFT":
                interaction_stats["TFT_vs_TFT"]["moves"] += 2
                if act1 == "Cooperate": interaction_stats["TFT_vs_TFT"]["coops"] += 1
                if act2 == "Cooperate": interaction_stats["TFT_vs_TFT"]["coops"] += 1

            elif s1_name == "TFT" and s2_name == "AllD":
                interaction_stats["TFT_vs_AllD"]["tft_moves"] += 1
                if act1 == "Cooperate": interaction_stats["TFT_vs_AllD"]["tft_coops"] += 1

                interaction_stats["TFT_vs_AllD"]["alld_moves"] += 1
                if act2 == "Cooperate": interaction_stats["TFT_vs_AllD"]["alld_coops"] += 1

            elif s1_name == "AllD" and s2_name == "TFT":
                interaction_stats["TFT_vs_AllD"]["alld_moves"] += 1
                if act1 == "Cooperate": interaction_stats["TFT_vs_AllD"]["alld_coops"] += 1

                interaction_stats["TFT_vs_AllD"]["tft_moves"] += 1
                if act2 == "Cooperate": interaction_stats["TFT_vs_AllD"]["tft_coops"] += 1

        # Report TFT vs TFT
        stats_tft_tft = interaction_stats["TFT_vs_TFT"]
        if stats_tft_tft["moves"] > 0:
            rate = stats_tft_tft["coops"] / stats_tft_tft["moves"]
            print(
                f"Cooperation Rate in TFT vs TFT interactions (moves by TFT agents): {rate:.2f} ({stats_tft_tft['coops']}/{stats_tft_tft['moves']})")
        else:
            print("No TFT vs TFT interactions logged.")

        # Report TFT's behavior when against AllD
        stats_tft_v_alld_tft = interaction_stats["TFT_vs_AllD"]
        if stats_tft_v_alld_tft["tft_moves"] > 0:
            rate = stats_tft_v_alld_tft["tft_coops"] / stats_tft_v_alld_tft["tft_moves"]
            print(
                f"TFT Cooperation Rate when playing against AllD: {rate:.2f} ({stats_tft_v_alld_tft['tft_coops']}/{stats_tft_v_alld_tft['tft_moves']})")
        else:
            print("No TFT moves in TFT vs AllD interactions logged.")

        # Report AllD's behavior when against TFT
        stats_tft_v_alld_alld = interaction_stats["TFT_vs_AllD"]
        if stats_tft_v_alld_alld["alld_moves"] > 0:
            rate = stats_tft_v_alld_alld["alld_coops"] / stats_tft_v_alld_alld["alld_moves"]
            print(
                f"AllD Cooperation Rate when playing against TFT (due to exploration): {rate:.2f} ({stats_tft_v_alld_alld['alld_coops']}/{stats_tft_v_alld_alld['alld_moves']})")
        else:
            print("No AllD moves in TFT vs AllD interactions logged.")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Set the global exploration rate (error chance) for all agents
    # You can also set this individually when creating agents if needed
    EXPLORATION_RATE = 0.05  # 5% chance of making an error

    # Number of rounds for each pairwise interaction
    NUM_ROUNDS_PAIRWISE = 100

    # Experiment setup: 2 TFT agents and 1 AllD agent
    agents_for_experiment = [
        Agent(agent_id="TFT_1", strategy_name="TFT", exploration_rate=EXPLORATION_RATE),
        Agent(agent_id="TFT_2", strategy_name="TFT", exploration_rate=EXPLORATION_RATE),
        Agent(agent_id="AllD_1", strategy_name="AllD", exploration_rate=EXPLORATION_RATE)
    ]

    print(f"Starting Iterated Prisoner's Dilemma Simulation")
    print(f"Number of Agents: {len(agents_for_experiment)}")
    print(f"Rounds per pairwise interaction: {NUM_ROUNDS_PAIRWISE}")
    print(f"Global Exploration Rate: {EXPLORATION_RATE * 100}%")
    print(
        f"Payoffs: CC: {PAYOFFS[(COOPERATE, COOPERATE)]}, CD: {PAYOFFS[(COOPERATE, DEFECT)]}, DC: {PAYOFFS[(DEFECT, COOPERATE)]}, DD: {PAYOFFS[(DEFECT, DEFECT)]}")
    print("---")

    # Initialize and run the game
    game_simulation = IteratedPrisonersDilemma(
        agents=agents_for_experiment,
        num_rounds_pairwise=NUM_ROUNDS_PAIRWISE
    )
    game_simulation.run_tournament()

    # Example: Accessing detailed log for the first agent's first recorded move
    # if agents_for_experiment[0].moves_history_detailed:
    #     print("\nExample of detailed move log for Agent TFT_1 (first recorded move):")
    #     print(agents_for_experiment[0].moves_history_detailed[0])

    # Example: Accessing the full tournament log
    # if game_simulation.tournament_log:
    #     print("\nExample from overall tournament log (first round played):")
    #     print(game_simulation.tournament_log[0])
    #     if len(game_simulation.tournament_log) > NUM_ROUNDS_PAIRWISE:
    #          print("\nExample from overall tournament log (first round of second pair interaction):")
    #          print(game_simulation.tournament_log[NUM_ROUNDS_PAIRWISE])
