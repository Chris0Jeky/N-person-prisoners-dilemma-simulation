import random

# --- Constants ---
COOPERATE = 0
DEFECT = 1

# Base Payoff values from classic IPD
R_REWARD = 3
S_SUCKER = 0
T_TEMPTATION = 5
P_PUNISHMENT = 1


def move_to_str(move):
    return "Cooperate" if move == COOPERATE else "Defect"


# --- Payoff Functions for N-Person Game ---
def linear_payoff_cooperator(num_other_cooperators, total_agents, R=R_REWARD, S=S_SUCKER):
    """
    Calculates the payoff for an agent who COOPERATED.
    n: Number of OTHER agents who cooperated.
    N: Total number of agents in the game.
    """
    if total_agents <= 1:  # Single agent scenario
        return R
        # num_other_agents = total_agents - 1
    # if num_other_agents == 0: # Should be covered by N <= 1, but for safety
    #     return R
    return S + (R - S) * (num_other_cooperators / (total_agents - 1))


def linear_payoff_defector(num_other_cooperators, total_agents, T=T_TEMPTATION, P=P_PUNISHMENT):
    """
    Calculates the payoff for an agent who DEFECTED.
    n: Number of OTHER agents who cooperated.
    N: Total number of agents in the game.
    """
    if total_agents <= 1:  # Single agent scenario
        return P
    # num_other_agents = total_agents - 1
    # if num_other_agents == 0: # Should be covered by N <= 1, but for safety
    #     return P
    return P + (T - P) * (num_other_cooperators / (total_agents - 1))


# --- Agent Class for N-Person Game ---
class AgentNPerson:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.exploration_rate = exploration_rate

        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.history = []  # Could store more detailed history if needed

    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        """
        Determines the agent's intended move.
        prev_round_overall_coop_ratio: Ratio of ALL agents cooperating in the previous round (0 to 1).
                                       None for the first round.
        current_round_num: 0-indexed.
        """
        intended_move = None
        if self.strategy_name == "pTFT":  # Probabilistic Tit-For-Tat
            if current_round_num == 0 or prev_round_overall_coop_ratio is None:
                intended_move = COOPERATE  # Cooperate on the first move
            else:
                if random.random() < prev_round_overall_coop_ratio:
                    intended_move = COOPERATE
                else:
                    intended_move = DEFECT
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name} for agent {self.agent_id}")

        # Apply exploration/error
        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move  # Flip the move

        return intended_move, actual_move

    def record_round_outcome(self, my_actual_move, payoff, round_details):
        self.total_score += payoff
        if my_actual_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        self.history.append({
            "round": round_details["round_num"],
            "my_actual_move": move_to_str(my_actual_move),
            "payoff": payoff,
            "details": round_details  # includes all moves, coop_ratio etc.
        })

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset(self):
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.history = []


# --- N-Person Game Simulation Class ---
class NPersonPrisonersDilemma:
    def __init__(self, agents, num_rounds, R=R_REWARD, S=S_SUCKER, T=T_TEMPTATION, P=P_PUNISHMENT):
        self.agents = agents
        self.num_rounds = num_rounds
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        self.round_history_log = []  # Logs details of each round for global analysis

    def run_simulation(self):
        for agent in self.agents:
            agent.reset()

        prev_round_overall_coop_ratio = None  # For the very first round
        num_total_agents = len(self.agents)

        if num_total_agents == 0:
            print("No agents to simulate.")
            return

        for i in range(self.num_rounds):
            current_round_num = i

            # 1. Agents choose actions
            intended_moves_map = {}  # agent_id -> intended_move
            actual_moves_map = {}  # agent_id -> actual_move

            for agent in self.agents:
                intended, actual = agent.choose_action(prev_round_overall_coop_ratio, current_round_num)
                intended_moves_map[agent.agent_id] = intended
                actual_moves_map[agent.agent_id] = actual

            # 2. Calculate round statistics based on actual moves
            num_actual_cooperators_this_round = sum(1 for move in actual_moves_map.values() if move == COOPERATE)
            current_overall_coop_ratio = num_actual_cooperators_this_round / num_total_agents if num_total_agents > 0 else 0

            round_details_for_log = {
                "round_num": current_round_num,
                "intended_moves": {aid: move_to_str(m) for aid, m in intended_moves_map.items()},
                "actual_moves": {aid: move_to_str(m) for aid, m in actual_moves_map.items()},
                "num_cooperators": num_actual_cooperators_this_round,
                "cooperation_ratio": current_overall_coop_ratio,
                "prev_round_coop_ratio_used_for_decision": prev_round_overall_coop_ratio
            }
            self.round_history_log.append(round_details_for_log)

            # 3. Calculate payoffs and record outcomes for each agent
            for agent in self.agents:
                my_actual_move = actual_moves_map[agent.agent_id]

                # Number of *other* agents who cooperated
                n_others_cooperated = num_actual_cooperators_this_round
                if my_actual_move == COOPERATE:
                    n_others_cooperated -= 1

                payoff = 0
                if my_actual_move == COOPERATE:
                    payoff = linear_payoff_cooperator(n_others_cooperated, num_total_agents, self.R, self.S)
                else:  # Agent defected
                    payoff = linear_payoff_defector(n_others_cooperated, num_total_agents, self.T, self.P)

                agent.record_round_outcome(my_actual_move, payoff, round_details_for_log)

            # 4. Update prev_round_overall_coop_ratio for the next round
            prev_round_overall_coop_ratio = current_overall_coop_ratio

            # Optional: Print round-by-round details
            # print(f"Round {current_round_num}: Coops={num_actual_cooperators_this_round}/{num_total_agents} (Ratio: {current_overall_coop_ratio:.2f}) "
            #       f"Moves: { {aid: move_to_str(m) for aid,m in actual_moves_map.items()} }")

        self.print_simulation_results()

    def print_simulation_results(self):
        print("\n--- N-Person Simulation Results ---")
        total_cooperations_all_agents = 0
        total_moves_all_agents = 0

        sorted_agents = sorted(self.agents, key=lambda ag: ag.total_score, reverse=True)

        for agent in sorted_agents:
            print(
                f"\nAgent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate * 100:.1f}%)")
            print(f"  Total Score: {agent.total_score:.2f}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")

            total_cooperations_all_agents += agent.num_cooperations
            total_moves_all_agents += total_agent_moves

        if total_moves_all_agents > 0:
            overall_coop_rate = total_cooperations_all_agents / total_moves_all_agents
            print(f"\n--- Overall Simulation Summary ---")
            print(f"Total Rounds Played: {self.num_rounds}")
            print(f"Overall Cooperation Rate (all agents, all rounds): {overall_coop_rate:.2f} "
                  f"({total_cooperations_all_agents} C / {total_moves_all_agents} total moves)")
        else:
            print("\nNo moves were made in the simulation.")

        # You can add more detailed analysis of self.round_history_log if needed
        # For example, plot cooperation_ratio over rounds:
        # coop_ratios_over_time = [log['cooperation_ratio'] for log in self.round_history_log]
        # print(f"\nCooperation ratios per round: {coop_ratios_over_time}")


# --- Main Execution ---
if __name__ == "__main__":
    EXPLORATION_RATE_N = 0.05  # 5% chance of making an error
    NUM_ROUNDS_N_PERSON = 200  # Number of rounds for the N-person game

    agents_n_person = [
        AgentNPerson(agent_id="TFT_1", strategy_name="pTFT", exploration_rate=EXPLORATION_RATE_N),
        AgentNPerson(agent_id="TFT_2", strategy_name="pTFT", exploration_rate=EXPLORATION_RATE_N),
        AgentNPerson(agent_id="AllD_1", strategy_name="AllD", exploration_rate=EXPLORATION_RATE_N)
    ]

    print(f"Starting N-Person Prisoner's Dilemma Simulation")
    print(f"Number of Agents: {len(agents_n_person)}")
    print(f"Number of Rounds: {NUM_ROUNDS_N_PERSON}")
    print(f"Global Exploration Rate: {EXPLORATION_RATE_N * 100:.1f}%")
    print(f"Base Payoffs: R={R_REWARD}, S={S_SUCKER}, T={T_TEMPTATION}, P={P_PUNISHMENT}")
    print(
        f"Cooperator Payoff Formula (example for N=3, n_others_coop=1): C_payoff = S + (R-S)*(n_others_coop / (N-1)) = {S_SUCKER + (R_REWARD - S_SUCKER) * (1 / (3 - 1))}")
    print(
        f"Defector Payoff Formula (example for N=3, n_others_coop=1): D_payoff = P + (T-P)*(n_others_coop / (N-1)) = {P_PUNISHMENT + (T_TEMPTATION - P_PUNISHMENT) * (1 / (3 - 1))}")
    print("---")

    simulation_n_person = NPersonPrisonersDilemma(
        agents=agents_n_person,
        num_rounds=NUM_ROUNDS_N_PERSON
    )
    simulation_n_person.run_simulation()

    # Example: Print cooperation ratio from the last round
    if simulation_n_person.round_history_log:
        last_round_log = simulation_n_person.round_history_log[-1]
        print(f"\nDetails from last round (Round {last_round_log['round_num']}):")
        print(f"  Cooperation Ratio: {last_round_log['cooperation_ratio']:.2f}")
        print(f"  Actual Moves: {last_round_log['actual_moves']}")

