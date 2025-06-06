# --- Part 2: Contents of main_neighbourhood.py (adapted with pTFT-Threshold) ---

import random

NPERSON_COOPERATE = 0
NPERSON_DEFECT = 1
NPERSON_R_REWARD, NPERSON_S_SUCKER, NPERSON_T_TEMPTATION, NPERSON_P_PUNISHMENT = 3, 0, 5, 1

def nperson_move_to_str(move):
    return "Cooperate" if move == NPERSON_COOPERATE else "Defect"

def nperson_linear_payoff_cooperator(n_others_coop, total_agents, R=NPERSON_R_REWARD, S=NPERSON_S_SUCKER):
    if total_agents <= 1: return R
    # total_agents-1 could be 0 if total_agents is 1. Handled by the line above.
    # For N > 1, N-1 is the correct divisor.
    return S + (R - S) * (n_others_coop / (total_agents - 1))

def nperson_linear_payoff_defector(n_others_coop, total_agents, T=NPERSON_T_TEMPTATION, P=NPERSON_P_PUNISHMENT):
    if total_agents <= 1: return P
    return P + (T - P) * (n_others_coop / (total_agents - 1))

class NPersonAgent:
    def __init__(self, agent_id, strategy_name, exploration_rate): # strategy_name can be pTFT, pTFT-Threshold, AllD
        self.agent_id = agent_id
        self.strategy_name = strategy_name 
        self.exploration_rate = exploration_rate
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

    def choose_action(self, prev_round_overall_coop_ratio, current_round_num):
        intended_move = None
        if self.strategy_name == "pTFT":
            if current_round_num == 0 or prev_round_overall_coop_ratio is None:
                intended_move = NPERSON_COOPERATE
            else:
                intended_move = NPERSON_COOPERATE if random.random() < prev_round_overall_coop_ratio else NPERSON_DEFECT
        elif self.strategy_name == "pTFT-Threshold":
            if current_round_num == 0 or prev_round_overall_coop_ratio is None:
                intended_move = NPERSON_COOPERATE
            elif prev_round_overall_coop_ratio >= 0.5:
                intended_move = NPERSON_COOPERATE
            else: # prev_round_overall_coop_ratio < 0.5
                prob_coop = prev_round_overall_coop_ratio / 0.5 # Scales from 0 to just under 1
                intended_move = NPERSON_COOPERATE if random.random() < prob_coop else NPERSON_DEFECT
        elif self.strategy_name == "AllD":
            intended_move = NPERSON_DEFECT
        else:
            raise ValueError(f"Unknown N-person strategy: {self.strategy_name}")

        actual_move = intended_move
        if random.random() < self.exploration_rate:
            actual_move = 1 - intended_move
        return intended_move, actual_move

    def record_round_outcome(self, my_actual_move, payoff):
        self.total_score += payoff
        if my_actual_move == NPERSON_COOPERATE: self.num_cooperations += 1
        else: self.num_defections += 1

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset(self): # Full reset for a new simulation run
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0

class NPersonPrisonersDilemma:
    def __init__(self, agents, num_rounds, R=NPERSON_R_REWARD, S=NPERSON_S_SUCKER, T=NPERSON_T_TEMPTATION, P=NPERSON_P_PUNISHMENT):
        self.agents = agents
        self.num_rounds = num_rounds
        self.R, self.S, self.T, self.P = R, S, T, P

    def run_simulation(self):
        for agent in self.agents: agent.reset()
        prev_overall_coop_ratio = None
        N_total_agents = len(self.agents)
        if N_total_agents == 0: return

        for i in range(self.num_rounds):
            actual_moves = {}
            for agent in self.agents:
                _, actual = agent.choose_action(prev_overall_coop_ratio, i)
                actual_moves[agent.agent_id] = actual
            
            num_actual_coops_this_round = sum(1 for m in actual_moves.values() if m == NPERSON_COOPERATE)
            current_overall_coop_ratio = num_actual_coops_this_round / N_total_agents if N_total_agents > 0 else 0
            
            for agent in self.agents:
                my_actual_move = actual_moves[agent.agent_id]
                n_others_cooperated = num_actual_coops_this_round
                if my_actual_move == NPERSON_COOPERATE:
                    n_others_cooperated -= 1 
                
                payoff = 0
                if my_actual_move == NPERSON_COOPERATE:
                    payoff = nperson_linear_payoff_cooperator(n_others_cooperated, N_total_agents, self.R, self.S)
                else: # Agent defected
                    payoff = nperson_linear_payoff_defector(n_others_cooperated, N_total_agents, self.T, self.P)
                agent.record_round_outcome(my_actual_move, payoff)
            
            prev_overall_coop_ratio = current_overall_coop_ratio
        self.print_simulation_results()

    def print_simulation_results(self):
        # Determine the pTFT variant if applicable for display
        tft_variant_in_use = "N/A"
        for ag in self.agents:
            if ag.strategy_name in ["pTFT", "pTFT-Threshold"]:
                tft_variant_in_use = ag.strategy_name
                break

        print("\n--- N-Person Simulation Results ---")
        print(f"(pTFT variant: {tft_variant_in_use})") # Print which pTFT variant was used
        total_coops_all, total_moves_all = 0, 0
        sorted_agents = sorted(self.agents, key=lambda ag_sort: ag_sort.total_score, reverse=True)
        for agent in sorted_agents:
            print(f"Agent ID: {agent.agent_id} (Strategy: {agent.strategy_name}, Exploration: {agent.exploration_rate*100:.1f}%)")
            print(f"  Total Score: {agent.total_score:.2f}")
            coop_rate = agent.get_cooperation_rate()
            total_agent_moves = agent.num_cooperations + agent.num_defections
            print(f"  Cooperation Rate: {coop_rate:.2f} ({agent.num_cooperations} C / {total_agent_moves} total moves)")
            total_coops_all += agent.num_cooperations
            total_moves_all += total_agent_moves # Used to be agent_moves, fixed
        
        if total_moves_all > 0:
            print(f"Overall N-Person Cooperation Rate: {total_coops_all/total_moves_all:.2f}")

def run_n_person_experiment(agent_configurations, num_rounds_total, tft_variant_for_tft_agents="pTFT"):
    agents_for_experiment = []
    for config in agent_configurations:
        strategy_to_use = config['strategy']
        if config['strategy'] == "TFT": # If config says "TFT", use the specified variant
            strategy_to_use = tft_variant_for_tft_agents
        
        agents_for_experiment.append(
            NPersonAgent(agent_id=config['id'], 
                         strategy_name=strategy_to_use, 
                         exploration_rate=config['exploration_rate'])
        )
    
    simulation_n_person = NPersonPrisonersDilemma(
        agents=agents_for_experiment,
        num_rounds=num_rounds_total
    )
    simulation_n_person.run_simulation()
    return simulation_n_person