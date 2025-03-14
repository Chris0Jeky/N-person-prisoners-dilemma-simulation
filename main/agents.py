# agents.py
import random


class Agent:
    def __init__(self, agent_id, strategy="random", learning_rate=0.1, discount_factor=0.9, epsilon=0.1,
                 generosity=0.05, initial_move="cooperate", prob_coop=0.5):
        self.agent_id = agent_id
        self.strategy = strategy
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # Parameters for specific strategies
        self.generosity = generosity  # For GTFT
        self.initial_move = initial_move  # For STFT and Pavlov
        self.prob_coop = prob_coop  # For RandomProb

    def choose_move(self, neighbors):
        if self.strategy == "random":
            return random.choice(["cooperate", "defect"])
        elif self.strategy == "always_cooperate":
            return "cooperate"
        elif self.strategy == "always_defect":
            return "defect"
        elif self.strategy == "tit_for_tat":
            if not self.memory:
                return "cooperate"
            else:
                last_round = self.memory[-1]
                neighbor_moves = last_round['neighbor_moves']
                if neighbor_moves:
                    random_neighbor_move = random.choice(list(neighbor_moves.values()))
                    return random_neighbor_move
                else:
                    return "cooperate"  # Default to cooperate if no neighbors
        elif self.strategy == "generous_tit_for_tat":
            if not self.memory:
                return "cooperate"
            else:
                last_round = self.memory[-1]
                neighbor_moves = last_round['neighbor_moves']
                if neighbor_moves:
                    random_neighbor_move = random.choice(list(neighbor_moves.values()))
                    if random_neighbor_move == "defect" and random.random() < self.generosity:
                        return "cooperate"  # Cooperate with probability 'generosity'
                    return random_neighbor_move
                else:
                    return "cooperate"
        elif self.strategy == "suspicious_tit_for_tat":
            if not self.memory:
                return "defect"  # Defect on the first move
            else:
                last_round = self.memory[-1]
                neighbor_moves = last_round['neighbor_moves']
                if neighbor_moves:
                    random_neighbor_move = random.choice(list(neighbor_moves.values()))
                    return random_neighbor_move
                else:
                    return "cooperate"
        elif self.strategy == "pavlov":
            if not self.memory:
                return self.initial_move  # can be either cooperate or defect
            else:
                last_round = self.memory[-1]
                last_reward = last_round['reward']
                if last_reward >= 3:  # Assuming R=3 and T=5 as high payoffs
                    return self.memory[-1]['my_move']  # Repeat last move
                else:
                    return "defect" if self.memory[-1]['my_move'] == "cooperate" else "cooperate"  # Switch move
        elif self.strategy == "q_learning":
            if random.random() < self.epsilon:
                return random.choice(["cooperate", "defect"])
            else:
                return "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        elif self.strategy == "q_learning_adaptive":
            if random.random() < self.epsilon:
                return random.choice(["cooperate", "defect"])
            else:
                return "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        elif self.strategy == "randomprob":
            if random.random() < self.prob_coop:
                return "cooperate"
            else:
                return "defect"
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_q_value(self, action, reward, next_state_actions):
        if self.strategy not in ("q_learning", "q_learning_adaptive"):  # added q_learning_adaptive
            return

        best_next_action = "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        best_next_q = self.q_values[best_next_action]

        self.q_values[action] = (1 - self.learning_rate) * self.q_values[action] + \
                                self.learning_rate * (reward + self.discount_factor * best_next_q)

    def update_memory(self, my_move, neighbor_moves, reward):
        self.memory.append({
            'my_move': my_move,
            'neighbor_moves': neighbor_moves,
            'reward': reward,
        })

    def decrease_epsilon(self, decay_rate=0.99):
        # Apply a decay to epsilon
        self.epsilon *= decay_rate

    def reset(self):
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}