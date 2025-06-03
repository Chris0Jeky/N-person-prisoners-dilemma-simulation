# main_neighbourhood.py
import random

# --- Constants ---
COOPERATE = 0
DEFECT = 1
R_REWARD, S_SUCKER, T_TEMPTATION, P_PUNISHMENT = 3, 0, 5, 1


def move_to_str(move):
    return "Cooperate" if move == COOPERATE else "Defect"


def linear_payoff_cooperator(n_others_coop, total_agents, R=R_REWARD, S=S_SUCKER):
    if total_agents <= 1: return R
    return S + (R - S) * (n_others_coop / (total_agents - 1))


def linear_payoff_defector(n_others_coop, total_agents, T=T_TEMPTATION, P=P_PUNISHMENT):
    if total_agents <= 1: return P
    return P + (T - P) * (n_others_coop / (total_agents - 1))


# --- Agent Class (N-Person) ---
class AgentNPerson:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name  # Expects "pTFT" or "AllD"
        self.exploration_rate = exploration_rate
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.history = []

    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        intended_move = None
        if self.strategy_name == "pTFT":
            if current_round_num == 0 or prev_round_overall_coop_ratio is None:
                intended_move = COOPERATE
            else:
                intended_move = COOPERATE if random.random() < prev_round_overall_coop_ratio else DEFECT
        elif self.strategy_name == "AllD":
            intended_move = DEFECT
        else:
            raise ValueError(f"Unknown N-person strategy: {self.strategy_name}")

        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        return intended_move, actual_move

    def record_round_outcome(self, my_actual_move, payoff, round_details):
        self.total_score += payoff
        if my_actual_move == COOPERATE:
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        # self.history.append(...) # Can add detailed history storage

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset(self):
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        self.history = []


# --- Game Class (N-Person) ---
class NPersonPrisonersDilemma:
    def __init__(self, agents, num_rounds, R=R_REWARD, S=S_SUCKER, T=T_TEMPTATION, P=P_PUNISHMENT):
        self.agents = agents
        self.num_rounds = num_rounds
        self.R, self.S, self.T, self.P = R, S, T, P
        self.round_history_log = []

    def run_simulation(self):
        for agent in self.agents: agent.reset()
        prev_overall_coop_ratio = None
        N = len(self.agents)
        if N == 0: return

        for i in range(self.num_rounds):
            intended_moves, actual_moves = {}, {}
            for agent in self.agents:
                intended, actual = agent.choose_action(prev_overall_coop_ratio, i)
                intended_moves[agent.agent_id], actual_moves[agent.agent_id] = intended, actual

            num_actual_coops = sum(1 for m in actual_moves.values() if m == COOPERATE)
            current_overall_coop_ratio = num_actual_coops / N if N > 0 else 0

            # Store log if needed, simplified here
            # self.round_history_log.append(...)

            for agent in self.agents:
                my_move = actual_moves[agent.agent_id]
                n_others_cooped = num_actual_coops - (1 if my_move == COOPERATE else 0)
                payoff = 0
                if my_move == COOPERATE:
                    payoff = linear_payoff_cooperator(n_others_cooped, N, self.R, self.S)
                else:
                    payoff = linear_payoff_defector(n_others_cooped, N, self.T, self.P)
                agent.record_round_outcome(my_move, payoff, {})  # Pass round details if stored

            prev_overall_coop_ratio = current_overall_coop_ratio
        self.print_simulation_results()

    def print_simulation_results(self):
        print("\n--- N-Person Simulation Results ---")
        total_coops_all, total_moves_all = 0, 0
        sorted_agents = sorted(self.agents, key=lambda ag: ag.total_score, reverse=True)
        for agent in sorted_agents:
            print(
                f"\nAgent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate * 100:.1f}%)")
            print(f"  Total Score: {agent.total_score:.2f}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")
            total_coops_all += agent.num_cooperations
            total_moves_all += total_agent_moves

        if total_moves_all > 0:
            print(f"\nOverall N-Person Cooperation Rate: {total_coops_all / total_moves_all:.2f}")


def run_n_person_experiment(agent_configurations, num_rounds_n_person_param):
    agents_for_experiment = []
    for config in agent_configurations:
        # Ensure strategy name is "pTFT" for N-person TFT-like agents
        strategy_name_n_person = config['strategy']
        if strategy_name_n_person == "TFT":  # Map generic TFT to pTFT for this model
            strategy_name_n_person = "pTFT"

        agents_for_experiment.append(
            AgentNPerson(agent_id=config['id'],
                         strategy_name=strategy_name_n_person,
                         exploration_rate=config['exploration_rate'])
        )

    # print(f"--- Initializing N-Person Experiment ---") # Moved to runner
    # print(f"Agent Configurations:")
    # for config in agent_configurations: # Original configs before mapping TFT to pTFT
    #     print(f"  - ID: {config['id']}, Strategy: {config['strategy']} (as {agents_for_experiment[agent_configurations.index(config)].strategy_name}), Exploration: {config['exploration_rate']*100:.1f}%")
    # print(f"Number of Rounds: {num_rounds_n_person_param}")

    simulation_n_person = NPersonPrisonersDilemma(
        agents=agents_for_experiment,
        num_rounds=num_rounds_n_person_param
    )
    simulation_n_person.run_simulation()
    # print(f"--- N-Person Experiment Finished ---") # Moved to runner
    return simulation_n_person


if __name__ == "__main__":
    print("Running main_neighbourhood.py as standalone test...")
    default_agent_configs_n = [
        {'id': "TFT_1_n_test", 'strategy': "TFT", 'exploration_rate': 0.05},  # Will be mapped to pTFT
        {'id': "TFT_2_n_test", 'strategy': "TFT", 'exploration_rate': 0.05},  # Will be mapped to pTFT
        {'id': "AllD_1_n_test", 'strategy': "AllD", 'exploration_rate': 0.05}
    ]
    run_n_person_experiment(default_agent_configs_n, num_rounds_n_person_param=50)