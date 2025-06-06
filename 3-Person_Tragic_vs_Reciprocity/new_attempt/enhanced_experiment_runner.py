import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# --- Part 1: Enhanced main_pairwise.py with comprehensive logging ---

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
        self.reset_for_new_tournament()

    def choose_action(self, opponent_id, current_round_in_episode):
        intended_move = None
        if self.strategy_name == "TFT":
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
                           my_intended_move, my_actual_move, round_num_in_episode,
                           global_round_number, episode_number):
        self.total_score += my_payoff
        self.opponent_last_moves[opponent_id] = opponent_actual_move
        
        if my_actual_move == PAIRWISE_COOPERATE:
            self.num_cooperations += 1
            self.log_is_coop_numeric.append(0)  # 0 for cooperate
        else:
            self.num_defections += 1
            self.log_is_coop_numeric.append(1)  # 1 for defect
            
        self.log_cumulative_score.append(self.total_score)
        self.log_interaction_details.append({
            "opponent_id": opponent_id,
            "my_actual_move": pairwise_move_to_str(my_actual_move),
            "payoff": my_payoff,
            "global_round": global_round_number,
            "episode": episode_number,
            "round_in_episode": round_num_in_episode
        })
        self.log_moves_by_episode[episode_number].append(my_actual_move)

    def clear_opponent_history(self, opponent_id):
        if opponent_id in self.opponent_last_moves:
            del self.opponent_last_moves[opponent_id]

    def get_cooperation_rate(self):
        total_moves = self.num_cooperations + self.num_defections
        return self.num_cooperations / total_moves if total_moves > 0 else 0.0

    def reset_for_new_tournament(self):
        self.total_score = 0
        self.opponent_last_moves = {}
        self.num_cooperations = 0
        self.num_defections = 0
        # Logging attributes
        self.log_cumulative_score = [0]  # Start with score 0 before any interactions
        self.log_is_coop_numeric = []    # Moves (0 for C, 1 for D)
        self.log_interaction_details = []
        self.log_moves_by_episode = {}  # Dictionary to track moves per episode


class PairwiseIteratedPrisonersDilemma:
    def __init__(self, agents, num_episodes, rounds_per_episode):
        self.agents = agents
        self.num_episodes = num_episodes
        self.rounds_per_episode = rounds_per_episode
        self.payoff_matrix = PAIRWISE_PAYOFFS
        self.full_interaction_log = []
        self.global_round_counter = 0

    def _play_single_round(self, agent1, agent2, episode_num, current_round_in_episode):
        intended_move1, actual_move1 = agent1.choose_action(agent2.agent_id, current_round_in_episode)
        intended_move2, actual_move2 = agent2.choose_action(agent1.agent_id, current_round_in_episode)
        payoff1, payoff2 = self.payoff_matrix[(actual_move1, actual_move2)]
        
        self.global_round_counter += 1
        
        agent1.record_interaction(agent2.agent_id, actual_move2, payoff1, intended_move1, actual_move1, 
                                  current_round_in_episode, self.global_round_counter, episode_num)
        agent2.record_interaction(agent1.agent_id, actual_move1, payoff2, intended_move2, actual_move2, 
                                  current_round_in_episode, self.global_round_counter, episode_num)
        
        self.full_interaction_log.append({
            "agent1_id": agent1.agent_id, "agent2_id": agent2.agent_id,
            "episode": episode_num, "round_in_episode": current_round_in_episode,
            "global_round": self.global_round_counter,
            "agent1_move": pairwise_move_to_str(actual_move1), 
            "agent2_move": pairwise_move_to_str(actual_move2),
            "agent1_payoff": payoff1, "agent2_payoff": payoff2
        })

    def run_pairwise_interaction(self, agent1, agent2):
        for episode_num in range(self.num_episodes):
            # Initialize episode tracking
            if episode_num not in agent1.log_moves_by_episode:
                agent1.log_moves_by_episode[episode_num] = []
            if episode_num not in agent2.log_moves_by_episode:
                agent2.log_moves_by_episode[episode_num] = []
                
            for round_num_in_episode in range(self.rounds_per_episode):
                self._play_single_round(agent1, agent2, episode_num, round_num_in_episode)
                
            if self.num_episodes > 1 and episode_num < self.num_episodes - 1:
                agent1.clear_opponent_history(agent2.agent_id)
                agent2.clear_opponent_history(agent1.agent_id)

    def run_tournament(self):
        self.full_interaction_log = []
        self.global_round_counter = 0
        for agent in self.agents:
            agent.reset_for_new_tournament()
        
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                self.run_pairwise_interaction(self.agents[i], self.agents[j])

    def get_final_results_and_logs(self, params):
        agent_final_stats = []
        agent_time_series = {}
        
        # Calculate cooperation rates by episode for each agent
        agent_episode_coop_rates = {}
        
        for agent in self.agents:
            agent_final_stats.append({
                "id": agent.agent_id, "strategy": agent.strategy_name,
                "final_score": agent.total_score, 
                "final_cooperation_rate": agent.get_cooperation_rate()
            })
            agent_time_series[agent.agent_id] = {
                "scores_vs_round": agent.log_cumulative_score[1:],  # remove initial 0
                "is_coop_numeric_vs_round": agent.log_is_coop_numeric,
                "interaction_details": agent.log_interaction_details
            }
            
            # Calculate episode cooperation rates
            episode_coop_rates = []
            for episode_num in sorted(agent.log_moves_by_episode.keys()):
                moves = agent.log_moves_by_episode[episode_num]
                if moves:
                    coop_rate = sum(1 for m in moves if m == PAIRWISE_COOPERATE) / len(moves)
                    episode_coop_rates.append(coop_rate)
            agent_episode_coop_rates[agent.agent_id] = episode_coop_rates
            
        # Calculate overall cooperation rate per round
        round_coop_rates = []
        if self.full_interaction_log:
            max_round = max(log["global_round"] for log in self.full_interaction_log)
            for round_num in range(1, max_round + 1):
                round_interactions = [log for log in self.full_interaction_log if log["global_round"] == round_num]
                if round_interactions:
                    total_coops = sum(1 for log in round_interactions 
                                      for move in [log["agent1_move"], log["agent2_move"]] 
                                      if move == "Cooperate")
                    total_moves = len(round_interactions) * 2
                    round_coop_rates.append(total_coops / total_moves if total_moves > 0 else 0)
                    
        return {
            "params": params,
            "agent_final_stats": agent_final_stats,
            "agent_time_series": agent_time_series,
            "agent_episode_coop_rates": agent_episode_coop_rates,
            "global_time_series": {
                "interaction_log": self.full_interaction_log,
                "round_cooperation_rates": round_coop_rates
            },
            "summary_metrics": {
                "overall_cooperation_rate": sum(s['final_cooperation_rate'] for s in agent_final_stats) / len(agent_final_stats) if agent_final_stats else 0,
                "episodes_data": {
                    "num_episodes": self.num_episodes,
                    "rounds_per_episode": self.rounds_per_episode
                }
            }
        }


def run_pairwise_experiment(agent_configurations, total_rounds_per_pair, 
                            episodic_mode, num_episodes_if_episodic, current_params):
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
            raise ValueError("Total rounds must be divisible by num_episodes.")
        _rounds_per_episode = total_rounds_per_pair // _num_episodes
    
    current_params.update({
        "num_episodes": _num_episodes, 
        "rounds_per_episode": _rounds_per_episode
    })

    game_simulation = PairwiseIteratedPrisonersDilemma(
        agents=agents_for_experiment,
        num_episodes=_num_episodes,
        rounds_per_episode=_rounds_per_episode
    )
    game_simulation.run_tournament()
    return game_simulation.get_final_results_and_logs(current_params)


# --- Part 2: Enhanced main_neighbourhood.py with comprehensive logging ---

NPERSON_COOPERATE = 0
NPERSON_DEFECT = 1
NPERSON_R_REWARD, NPERSON_S_SUCKER, NPERSON_T_TEMPTATION, NPERSON_P_PUNISHMENT = 3, 0, 5, 1

def nperson_move_to_str(move):
    return "Cooperate" if move == NPERSON_COOPERATE else "Defect"

def nperson_linear_payoff_cooperator(n_others_coop, total_agents, R=NPERSON_R_REWARD, S=NPERSON_S_SUCKER):
    if total_agents <= 1: return R
    return S + (R - S) * (n_others_coop / (total_agents - 1))

def nperson_linear_payoff_defector(n_others_coop, total_agents, T=NPERSON_T_TEMPTATION, P=NPERSON_P_PUNISHMENT):
    if total_agents <= 1: return P
    return P + (T - P) * (n_others_coop / (total_agents - 1))

class NPersonAgent:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        self.agent_id = agent_id
        self.strategy_name = strategy_name 
        self.exploration_rate = exploration_rate
        self.reset()

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
            else: 
                prob_coop = prev_round_overall_coop_ratio / 0.5
                intended_move = NPERSON_COOPERATE if random.random() < prob_coop else NPERSON_DEFECT
        elif self.strategy_name == "AllD": 
            intended_move = NPERSON_DEFECT
        else: 
            raise ValueError(f"Unknown N-person strategy: {self.strategy_name}")
            
        actual_move = intended_move
        if random.random() < self.exploration_rate: 
            actual_move = 1 - intended_move
        return intended_move, actual_move

    def record_round_outcome(self, my_actual_move, payoff, round_num):
        self.total_score += payoff
        if my_actual_move == NPERSON_COOPERATE: 
            self.num_cooperations += 1
        else:
            self.num_defections += 1
        self.log_is_coop_numeric.append(my_actual_move)  # 0 for C, 1 for D
        self.log_cumulative_score.append(self.total_score)
        self.log_round_details.append({
            "round": round_num,
            "move": nperson_move_to_str(my_actual_move),
            "payoff": payoff
        })

    def get_cooperation_rate(self):
        total_moves = len(self.log_is_coop_numeric)
        if total_moves == 0: return 0.0
        actual_coops = self.log_is_coop_numeric.count(NPERSON_COOPERATE)
        return actual_coops / total_moves

    def reset(self):
        self.total_score = 0
        self.num_cooperations = 0
        self.num_defections = 0
        # Logging attributes
        self.log_cumulative_score = [0]  # Start with score 0
        self.log_is_coop_numeric = []
        self.log_round_details = []

class NPersonPrisonersDilemma:
    def __init__(self, agents, num_rounds, R=NPERSON_R_REWARD, S=NPERSON_S_SUCKER, 
                 T=NPERSON_T_TEMPTATION, P=NPERSON_P_PUNISHMENT):
        self.agents = agents
        self.num_rounds = num_rounds
        self.R, self.S, self.T, self.P = R, S, T, P
        # Logging
        self.log_overall_coop_rate_vs_round = []
        self.log_round_details = []

    def run_simulation(self):
        for agent in self.agents: 
            agent.reset()
        self.log_overall_coop_rate_vs_round = []
        self.log_round_details = []

        prev_overall_coop_ratio = None
        N_total_agents = len(self.agents)
        if N_total_agents == 0: return

        for i in range(self.num_rounds):
            actual_moves_map = {}
            for agent in self.agents:
                _, actual = agent.choose_action(prev_overall_coop_ratio, i)
                actual_moves_map[agent.agent_id] = actual
            
            num_actual_coops_this_round = sum(1 for m in actual_moves_map.values() if m == NPERSON_COOPERATE)
            current_overall_coop_ratio = num_actual_coops_this_round / N_total_agents if N_total_agents > 0 else 0.0
            self.log_overall_coop_rate_vs_round.append(current_overall_coop_ratio)

            round_payoffs = {}
            for agent in self.agents:
                my_actual_move = actual_moves_map[agent.agent_id]
                n_others_cooperated = num_actual_coops_this_round - (1 if my_actual_move == NPERSON_COOPERATE else 0)
                payoff = 0
                if my_actual_move == NPERSON_COOPERATE: 
                    payoff = nperson_linear_payoff_cooperator(n_others_cooperated, N_total_agents, self.R, self.S)
                else: 
                    payoff = nperson_linear_payoff_defector(n_others_cooperated, N_total_agents, self.T, self.P)
                agent.record_round_outcome(my_actual_move, payoff, i)
                round_payoffs[agent.agent_id] = payoff
            
            self.log_round_details.append({
                "round_num": i, "overall_coop_ratio": current_overall_coop_ratio,
                "moves": {aid: nperson_move_to_str(m) for aid, m in actual_moves_map.items()},
                "payoffs": round_payoffs
            })
            prev_overall_coop_ratio = current_overall_coop_ratio

    def get_final_results_and_logs(self, params):
        agent_final_stats = []
        agent_time_series = {}
        for agent in self.agents:
            agent_final_stats.append({
                "id": agent.agent_id, "strategy": agent.strategy_name,
                "final_score": agent.total_score, 
                "final_cooperation_rate": agent.get_cooperation_rate()
            })
            agent_time_series[agent.agent_id] = {
                "scores_vs_round": agent.log_cumulative_score[1:],  # remove initial 0
                "is_coop_numeric_vs_round": agent.log_is_coop_numeric,
                "round_details": agent.log_round_details
            }
            
        # Calculate episode-based cooperation rates for N-Person (simulated episodes)
        # We'll divide the rounds into "episodes" of equal length for comparison with pairwise
        simulated_episodes = 20  # Same as pairwise episodes for comparison
        rounds_per_episode = self.num_rounds // simulated_episodes
        episode_coop_rates = []
        
        for ep in range(simulated_episodes):
            start_round = ep * rounds_per_episode
            end_round = (ep + 1) * rounds_per_episode if ep < simulated_episodes - 1 else self.num_rounds
            if start_round < len(self.log_overall_coop_rate_vs_round):
                episode_rates = self.log_overall_coop_rate_vs_round[start_round:end_round]
                if episode_rates:
                    episode_coop_rates.append(np.mean(episode_rates))
                    
        return {
            "params": params,
            "agent_final_stats": agent_final_stats,
            "agent_time_series": agent_time_series,
            "global_time_series": {
                "overall_coop_rate_vs_round": self.log_overall_coop_rate_vs_round,
                "detailed_round_logs": self.log_round_details,
                "episode_coop_rates": episode_coop_rates
            },
            "summary_metrics": {
                "final_overall_cooperation_rate": self.log_overall_coop_rate_vs_round[-1] if self.log_overall_coop_rate_vs_round else 0,
                "avg_overall_cooperation_rate": sum(self.log_overall_coop_rate_vs_round) / len(self.log_overall_coop_rate_vs_round) if self.log_overall_coop_rate_vs_round else 0
            }
        }

def run_n_person_experiment(agent_configurations, num_rounds_total, tft_variant_for_tft_agents, current_params):
    agents_for_experiment = []
    for config in agent_configurations:
        strategy_to_use = config['strategy']
        if config['strategy'] == "TFT": 
            strategy_to_use = tft_variant_for_tft_agents
        agents_for_experiment.append(
            NPersonAgent(agent_id=config['id'], strategy_name=strategy_to_use, exploration_rate=config['exploration_rate'])
        )
    current_params.update({"num_rounds_total_n_person": num_rounds_total})

    simulation_n_person = NPersonPrisonersDilemma(agents=agents_for_experiment, num_rounds=num_rounds_total)
    simulation_n_person.run_simulation()
    return simulation_n_person.get_final_results_and_logs(current_params)

# --- Part 3: Enhanced Experiment Runner with comprehensive logging ---

EXPLORATION_RATES_TO_TEST = [0.0, 0.10] 
AGENT_COMPOSITIONS_TO_TEST = [
    {"name": "3_TFTs", "config_templates": [{"id_prefix": "TFT", "strategy": "TFT"}] * 3 },
    {"name": "2_TFTs_1_AllD", "config_templates": [{"id_prefix": "TFT", "strategy": "TFT"}] * 2 + [{"id_prefix": "AllD", "strategy": "AllD"}]},
    {"name": "2_TFTs_3_AllD", "config_templates": [{"id_prefix": "TFT", "strategy": "TFT"}] * 2 + [{"id_prefix": "AllD", "strategy": "AllD"}] * 3}
]
NUM_ROUNDS_PAIRWISE_TOTAL_PER_PAIR = 100 
NUM_ROUNDS_N_PERSON_TOTAL = 200 
PAIRWISE_EPISODIC_MODES = [False, True]
PAIRWISE_NUM_EPISODES_IF_EPISODIC = 10
NPERSON_TFT_VARIANTS = ["pTFT", "pTFT-Threshold"]

def generate_agent_configs_runner(composition_template_entry, exploration_rate_param):
    configs, id_counts = [], {}
    for template in composition_template_entry["config_templates"]:
        prefix = template["id_prefix"]
        id_counts[prefix] = id_counts.get(prefix, 0) + 1
        agent_id = f"{prefix}_{id_counts[prefix]}"
        configs.append({"id": agent_id, "strategy": template["strategy"], "exploration_rate": exploration_rate_param})
    return configs

ALL_EXPERIMENT_RESULTS = []  # Global list to store detailed results

def run_all_experiments_and_log():
    global ALL_EXPERIMENT_RESULTS
    ALL_EXPERIMENT_RESULTS = []  # Clear for fresh run
    print("Starting Enhanced Experiment Runner with comprehensive logging...")

    for exp_rate in EXPLORATION_RATES_TO_TEST:
        for composition_entry in AGENT_COMPOSITIONS_TO_TEST:
            composition_name = composition_entry['name']
            num_agents = len(composition_entry["config_templates"])
            base_agent_configs = generate_agent_configs_runner(composition_entry, exp_rate)

            # Pairwise Model Experiments
            for is_episodic in PAIRWISE_EPISODIC_MODES:
                print(f"Running: Pairwise, {composition_name}, ExpRate: {exp_rate*100:.0f}%, Episodic: {is_episodic}")
                current_params = {
                    "model_type": "Pairwise", "exploration_rate": exp_rate,
                    "composition_name": composition_name, "num_agents": num_agents,
                    "episodic_mode": is_episodic, "tft_variant": None
                }
                detailed_log = run_pairwise_experiment(
                    agent_configurations=[dict(c) for c in base_agent_configs], 
                    total_rounds_per_pair=NUM_ROUNDS_PAIRWISE_TOTAL_PER_PAIR,
                    episodic_mode=is_episodic,
                    num_episodes_if_episodic=PAIRWISE_NUM_EPISODES_IF_EPISODIC,
                    current_params=current_params
                )
                ALL_EXPERIMENT_RESULTS.append(detailed_log)
            
            # N-Person Model Experiments
            if num_agents > 1:
                for tft_variant in NPERSON_TFT_VARIANTS:
                    print(f"Running: N-Person, {composition_name}, ExpRate: {exp_rate*100:.0f}%, TFT Variant: {tft_variant}")
                    current_params = {
                        "model_type": "N-Person", "exploration_rate": exp_rate,
                        "composition_name": composition_name, "num_agents": num_agents,
                        "tft_variant": tft_variant, "episodic_mode": None
                    }
                    detailed_log = run_n_person_experiment(
                        agent_configurations=[dict(c) for c in base_agent_configs], 
                        num_rounds_total=NUM_ROUNDS_N_PERSON_TOTAL,
                        tft_variant_for_tft_agents=tft_variant,
                        current_params=current_params
                    )
                    ALL_EXPERIMENT_RESULTS.append(detailed_log)
    print("\nAll experiments complete. Data logged for visualization.")


# --- Part 4: Comprehensive Analyzer and Advanced Visualizations ---

def smooth_data(data, window_length=None, polyorder=3):
    """Apply Savitzky-Golay filter for smoothing."""
    if len(data) < 5:
        return data
    if window_length is None:
        window_length = min(21, len(data) // 4)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(5, window_length)
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    try:
        return savgol_filter(data, window_length, polyorder)
    except:
        return data


def plot_evolution_of_cooperation_comparison(all_results):
    """Create comparison plot similar to the shared image: Pairwise vs N-Person evolution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # Select specific experiments for comparison
    # Focus on 2_TFTs_1_AllD with 10% exploration, episodic pairwise, and pTFT-Threshold N-Person
    pairwise_exp = next((r for r in all_results if 
                         r['params']['model_type'] == "Pairwise" and 
                         r['params']['composition_name'] == "2_TFTs_1_AllD" and 
                         r['params']['exploration_rate'] == 0.10 and 
                         r['params']['episodic_mode'] == True), None)
    
    nperson_exp = next((r for r in all_results if 
                        r['params']['model_type'] == "N-Person" and 
                        r['params']['composition_name'] == "2_TFTs_1_AllD" and 
                        r['params']['exploration_rate'] == 0.10 and 
                        r['params']['tft_variant'] == "pTFT-Threshold"), None)
    
    if not pairwise_exp or not nperson_exp:
        print("Could not find matching experiments for evolution comparison")
        return
    
    # --- First subplot: Evolution of Cooperation with raw and smoothed data ---
    ax1 = axes[0]
    
    # Pairwise cooperation rate
    pairwise_coop_rates = pairwise_exp['global_time_series']['round_cooperation_rates']
    rounds_pairwise = list(range(len(pairwise_coop_rates)))
    
    # N-Person cooperation rate
    nperson_coop_rates = nperson_exp['global_time_series']['overall_coop_rate_vs_round']
    rounds_nperson = list(range(len(nperson_coop_rates)))
    
    # Plot raw data with transparency
    ax1.plot(rounds_pairwise, pairwise_coop_rates, 'b-', alpha=0.3, linewidth=0.8, label='Pairwise (raw)')
    ax1.plot(rounds_nperson, nperson_coop_rates, 'r-', alpha=0.3, linewidth=0.8, label='N-Person (raw)')
    
    # Plot smoothed data
    pairwise_smoothed = smooth_data(pairwise_coop_rates, window_length=21)
    nperson_smoothed = smooth_data(nperson_coop_rates, window_length=21)
    
    ax1.plot(rounds_pairwise, pairwise_smoothed, 'b-', linewidth=3, label='Pairwise (smoothed)')
    ax1.plot(rounds_nperson, nperson_smoothed, 'r-', linewidth=3, label='N-Person (smoothed)')
    
    ax1.set_title('Evolution of Cooperation: Pairwise vs N-Person', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('TFT Cooperation Rate', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Second subplot: Episode-Average Cooperation Rates ---
    ax2 = axes[1]
    
    # Calculate episode averages for Pairwise
    pairwise_episode_rates = []
    if pairwise_exp and 'agent_episode_coop_rates' in pairwise_exp:
        # Average across all TFT agents
        tft_agents = [aid for aid, stats in 
                      zip(pairwise_exp['agent_episode_coop_rates'].keys(), pairwise_exp['agent_final_stats'])
                      if 'TFT' in stats['strategy']]
        if tft_agents:
            num_episodes = len(pairwise_exp['agent_episode_coop_rates'][tft_agents[0]])
            for ep in range(num_episodes):
                ep_rates = [pairwise_exp['agent_episode_coop_rates'][aid][ep] for aid in tft_agents]
                pairwise_episode_rates.append(np.mean(ep_rates))
    
    # Use pre-calculated episode rates for N-Person
    nperson_episode_rates = nperson_exp['global_time_series']['episode_coop_rates']
    
    # Plot episode averages
    if pairwise_episode_rates:
        episodes = list(range(len(pairwise_episode_rates)))
        ax2.plot(episodes, pairwise_episode_rates, 'bo-', markersize=8, linewidth=2, label='Pairwise')
        # Add trend line
        z = np.polyfit(episodes, pairwise_episode_rates, 1)
        p = np.poly1d(z)
        ax2.plot(episodes, p(episodes), 'b--', alpha=0.5, linewidth=2)
    
    if nperson_episode_rates:
        episodes = list(range(len(nperson_episode_rates)))
        ax2.plot(episodes, nperson_episode_rates, 'ro-', markersize=8, linewidth=2, label='N-Person')
        # Add trend line
        z = np.polyfit(episodes, nperson_episode_rates, 1)
        p = np.poly1d(z)
        ax2.plot(episodes, p(episodes), 'r--', alpha=0.5, linewidth=2)
    
    ax2.set_title('Episode-Average Cooperation Rates', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average TFT Cooperation Rate', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Third subplot: Conceptual Model ---
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 5)
    ax3.axis('off')
    
    # Title
    ax3.text(5, 4.5, 'Conceptual Model: Reciprocity Hill vs Tragic Valley', 
             fontsize=16, fontweight='bold', ha='center')
    
    # Reciprocity Hill (Pairwise)
    hill_box = FancyBboxPatch((0.5, 2.5), 3.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='blue',
                              linewidth=3)
    ax3.add_patch(hill_box)
    ax3.text(2.25, 3.5, 'Reciprocity Hill\n(Pairwise)', 
             fontsize=12, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='blue'))
    ax3.text(2.25, 2.8, 'Cooperation\nstabilizes', 
             fontsize=10, ha='center', va='center')
    
    # Tragic Valley (N-Person)
    valley_box = FancyBboxPatch((6, 0.5), 3.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='lightcoral',
                                edgecolor='red',
                                linewidth=3)
    ax3.add_patch(valley_box)
    ax3.text(7.75, 1.5, 'Tragic Valley\n(N-Person)', 
             fontsize=12, fontweight='bold', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red'))
    ax3.text(7.75, 0.8, 'Defection\ndominates', 
             fontsize=10, ha='center', va='center')
    
    # Add arrows showing the dynamics
    ax3.annotate('', xy=(2.25, 4.2), xytext=(2.25, 3.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax3.annotate('', xy=(7.75, 0.3), xytext=(7.75, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    plt.savefig('evolution_of_cooperation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_tft_performance_across_conditions(all_results):
    """Compare TFT agent performance across different experimental conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for analysis
    comparison_data = []
    for res in all_results:
        params = res['params']
        # Focus on TFT agents only
        tft_stats = [s for s in res['agent_final_stats'] if 'TFT' in s['strategy']]
        if tft_stats:
            avg_tft_score = np.mean([s['final_score'] for s in tft_stats])
            avg_tft_coop = np.mean([s['final_cooperation_rate'] for s in tft_stats])
            
            comparison_data.append({
                "Model": params['model_type'],
                "Composition": params['composition_name'],
                "Exploration Rate": f"{params['exploration_rate']*100:.0f}%",
                "Episodic": params.get('episodic_mode', 'N/A'),
                "TFT Variant": params.get('tft_variant', 'N/A'),
                "Avg TFT Score": avg_tft_score,
                "Avg TFT Cooperation": avg_tft_coop
            })
    
    df = pd.DataFrame(comparison_data)
    
    # --- Subplot 1: TFT Cooperation Rate by Model and Composition ---
    ax1 = axes[0, 0]
    df_pivot = df.pivot_table(values='Avg TFT Cooperation', 
                              index='Composition', 
                              columns='Model', 
                              aggfunc='mean')
    df_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('Average TFT Cooperation Rate by Model Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_ylim(0, 1.05)
    ax1.legend(title='Model Type')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # --- Subplot 2: Impact of Exploration Rate ---
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='Exploration Rate', y='Avg TFT Cooperation', 
                hue='Model', ax=ax2)
    ax2.set_title('TFT Cooperation vs Exploration Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cooperation Rate')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # --- Subplot 3: Episodic vs Non-Episodic (Pairwise only) ---
    ax3 = axes[1, 0]
    df_pairwise = df[df['Model'] == 'Pairwise']
    sns.barplot(data=df_pairwise, x='Composition', y='Avg TFT Cooperation', 
                hue='Episodic', ax=ax3)
    ax3.set_title('Pairwise: Episodic vs Non-Episodic Impact', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cooperation Rate')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # --- Subplot 4: TFT Variant Comparison (N-Person only) ---
    ax4 = axes[1, 1]
    df_nperson = df[df['Model'] == 'N-Person']
    sns.barplot(data=df_nperson, x='Composition', y='Avg TFT Cooperation', 
                hue='TFT Variant', ax=ax4)
    ax4.set_title('N-Person: pTFT vs pTFT-Threshold', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cooperation Rate')
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tft_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_cooperation_dynamics_heatmap(all_results):
    """Create heatmaps showing cooperation dynamics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Select experiments for detailed view
    experiments = [
        ("Pairwise Episodic", lambda r: r['params']['model_type'] == "Pairwise" and 
         r['params']['episodic_mode'] == True and r['params']['exploration_rate'] == 0.10),
        ("Pairwise Non-Episodic", lambda r: r['params']['model_type'] == "Pairwise" and 
         r['params']['episodic_mode'] == False and r['params']['exploration_rate'] == 0.10),
        ("N-Person pTFT", lambda r: r['params']['model_type'] == "N-Person" and 
         r['params']['tft_variant'] == "pTFT" and r['params']['exploration_rate'] == 0.10),
        ("N-Person pTFT-Threshold", lambda r: r['params']['model_type'] == "N-Person" and 
         r['params']['tft_variant'] == "pTFT-Threshold" and r['params']['exploration_rate'] == 0.10)
    ]
    
    for idx, (title, filter_func) in enumerate(experiments):
        ax = axes[idx // 2, idx % 2]
        
        # Get experiments matching criteria
        matching_exps = [r for r in all_results if filter_func(r)]
        
        if not matching_exps:
            ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center')
            ax.set_title(title)
            continue
        
        # Create matrix for heatmap (compositions x time)
        compositions = sorted(set(r['params']['composition_name'] for r in matching_exps))
        
        # For each composition, get cooperation over time
        heatmap_data = []
        for comp in compositions:
            comp_exp = next((r for r in matching_exps if r['params']['composition_name'] == comp), None)
            if comp_exp:
                if comp_exp['params']['model_type'] == 'Pairwise':
                    coop_rates = comp_exp['global_time_series']['round_cooperation_rates']
                else:
                    coop_rates = comp_exp['global_time_series']['overall_coop_rate_vs_round']
                
                # Resample to fixed length for visualization
                target_length = 50
                if len(coop_rates) > target_length:
                    indices = np.linspace(0, len(coop_rates)-1, target_length, dtype=int)
                    resampled = [coop_rates[i] for i in indices]
                else:
                    resampled = coop_rates
                
                heatmap_data.append(resampled)
        
        if heatmap_data:
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_yticks(range(len(compositions)))
            ax.set_yticklabels(compositions)
            ax.set_xlabel('Time (normalized)')
            ax.set_title(f'{title}: Cooperation Dynamics', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cooperation Rate')
    
    plt.tight_layout()
    plt.savefig('cooperation_dynamics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_strategy_emergence_patterns(all_results):
    """Analyze how cooperation patterns emerge differently across strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Focus on different agent compositions with 10% exploration
    compositions = ["3_TFTs", "2_TFTs_1_AllD", "2_TFTs_3_AllD"]
    
    for i, comp in enumerate(compositions):
        # Pairwise episodic
        ax_pair = axes[0, i]
        pairwise_exp = next((r for r in all_results if 
                             r['params']['model_type'] == "Pairwise" and 
                             r['params']['composition_name'] == comp and 
                             r['params']['exploration_rate'] == 0.10 and 
                             r['params']['episodic_mode'] == True), None)
        
        if pairwise_exp:
            # Plot individual TFT agent trajectories
            for agent_id, time_series in pairwise_exp['agent_time_series'].items():
                agent_stat = next(s for s in pairwise_exp['agent_final_stats'] if s['id'] == agent_id)
                if 'TFT' in agent_stat['strategy']:
                    coop_trajectory = [1 - m for m in time_series['is_coop_numeric_vs_round']]  # Convert to cooperation
                    rolling_avg = pd.Series(coop_trajectory).rolling(window=10, min_periods=1).mean()
                    ax_pair.plot(rolling_avg, label=agent_id, alpha=0.7)
            
            ax_pair.set_title(f'Pairwise: {comp}', fontsize=12, fontweight='bold')
            ax_pair.set_xlabel('Interactions')
            ax_pair.set_ylabel('Cooperation Rate (10-round avg)')
            ax_pair.set_ylim(-0.05, 1.05)
            ax_pair.grid(True, alpha=0.3)
            if i == 0:
                ax_pair.legend(fontsize=8)
        
        # N-Person pTFT-Threshold
        ax_nperson = axes[1, i]
        nperson_exp = next((r for r in all_results if 
                            r['params']['model_type'] == "N-Person" and 
                            r['params']['composition_name'] == comp and 
                            r['params']['exploration_rate'] == 0.10 and 
                            r['params']['tft_variant'] == "pTFT-Threshold"), None)
        
        if nperson_exp:
            # Plot overall cooperation rate
            coop_rates = nperson_exp['global_time_series']['overall_coop_rate_vs_round']
            ax_nperson.plot(coop_rates, 'k-', linewidth=2, alpha=0.3, label='Overall')
            
            # Smooth version
            smoothed = smooth_data(coop_rates)
            ax_nperson.plot(smoothed, 'k-', linewidth=3, label='Overall (smoothed)')
            
            # Add individual agent trajectories
            for agent_id, time_series in nperson_exp['agent_time_series'].items():
                agent_stat = next(s for s in nperson_exp['agent_final_stats'] if s['id'] == agent_id)
                if 'TFT' in agent_stat['strategy']:
                    coop_trajectory = [1 - m for m in time_series['is_coop_numeric_vs_round']]
                    rolling_avg = pd.Series(coop_trajectory).rolling(window=20, min_periods=1).mean()
                    ax_nperson.plot(rolling_avg, '--', alpha=0.5, label=agent_id)
            
            ax_nperson.set_title(f'N-Person: {comp}', fontsize=12, fontweight='bold')
            ax_nperson.set_xlabel('Rounds')
            ax_nperson.set_ylabel('Cooperation Rate (20-round avg)')
            ax_nperson.set_ylim(-0.05, 1.05)
            ax_nperson.grid(True, alpha=0.3)
            if i == 0:
                ax_nperson.legend(fontsize=8)
    
    plt.suptitle('Strategy Emergence Patterns Across Compositions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('strategy_emergence_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_statistics_table(all_results):
    """Create a comprehensive summary table of key metrics."""
    summary_data = []
    
    for res in all_results:
        params = res['params']
        
        # Calculate key metrics
        tft_agents = [s for s in res['agent_final_stats'] if 'TFT' in s['strategy']]
        
        if tft_agents:
            avg_tft_score = np.mean([s['final_score'] for s in tft_agents])
            avg_tft_coop = np.mean([s['final_cooperation_rate'] for s in tft_agents])
            
            # Calculate stability (std dev of cooperation in last 25% of rounds)
            if params['model_type'] == 'Pairwise':
                coop_rates = res['global_time_series']['round_cooperation_rates']
            else:
                coop_rates = res['global_time_series']['overall_coop_rate_vs_round']
            
            last_quarter = coop_rates[-(len(coop_rates)//4):]
            stability = 1 - np.std(last_quarter) if last_quarter else 0
            
            summary_data.append({
                'Model': params['model_type'],
                'Composition': params['composition_name'],
                'Exploration': f"{params['exploration_rate']*100:.0f}%",
                'Variant': params.get('episodic_mode', params.get('tft_variant', 'N/A')),
                'Avg TFT Score': f"{avg_tft_score:.1f}",
                'Avg TFT Cooperation': f"{avg_tft_coop:.2%}",
                'Final Overall Coop': f"{coop_rates[-1]:.2%}" if coop_rates else "N/A",
                'Stability': f"{stability:.2f}"
            })
    
    df = pd.DataFrame(summary_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color coding for cooperation rates
    for i in range(len(df)):
        coop_val = float(df.iloc[i]['Avg TFT Cooperation'].strip('%')) / 100
        if coop_val >= 0.8:
            color = 'lightgreen'
        elif coop_val >= 0.5:
            color = 'yellow'
        else:
            color = 'lightcoral'
        table[(i+1, 5)].set_facecolor(color)
    
    plt.title('Summary Statistics: All Experiments', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('summary_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_all_results(all_results_list):
    """Main visualization function that creates all plots."""
    print("\nGenerating comprehensive visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Evolution of Cooperation Comparison (like the shared image)
    print("Creating evolution of cooperation comparison...")
    plot_evolution_of_cooperation_comparison(all_results_list)
    
    # 2. TFT Performance Analysis
    print("Creating TFT performance analysis...")
    plot_tft_performance_across_conditions(all_results_list)
    
    # 3. Cooperation Dynamics Heatmap
    print("Creating cooperation dynamics heatmap...")
    plot_cooperation_dynamics_heatmap(all_results_list)
    
    # 4. Strategy Emergence Patterns
    print("Creating strategy emergence patterns...")
    plot_strategy_emergence_patterns(all_results_list)
    
    # 5. Summary Statistics Table
    print("Creating summary statistics table...")
    create_summary_statistics_table(all_results_list)
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    run_all_experiments_and_log()
    
    if ALL_EXPERIMENT_RESULTS:
        visualize_all_results(ALL_EXPERIMENT_RESULTS)
    else:
        print("No experiment results to visualize.")
