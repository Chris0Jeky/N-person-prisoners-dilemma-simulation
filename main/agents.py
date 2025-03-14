# agents.py
import random

class Agent:
    def __init__(self, agent_id, strategy="random", learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.strategy = strategy
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_move(self, neighbors):
        # (rest of the choose_move method as before)
        if self.strategy == "random":
            return random.choice(["cooperate", "defect"])
        elif self.strategy == "always_cooperate":
            return "cooperate"
        elif self.strategy == "always_defect":
            return "defect"
        elif self.strategy == "tit_for_tat":
            if not self.memory:  # First move, cooperate
                return "cooperate"
            else:
                # Get the last move of a random neighbor (simplification).
                last_round = self.memory[-1]
                neighbor_moves = last_round['neighbor_moves']
                if neighbor_moves:  # if the agent has neighbours
                    random_neighbor_move = random.choice(list(neighbor_moves.values()))
                    return random_neighbor_move
                else:
                    return "cooperate"
        elif self.strategy == "q_learning":
            # ε-greedy
            if random.random() < self.epsilon:
                return random.choice(["cooperate", "defect"])
            else:
                return "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_q_value(self, action, reward, next_state_actions):
        # (rest of the update_q_value method as before)
        if self.strategy != "q_learning":
            return

        best_next_action = "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        best_next_q = self.q_values[best_next_action]

        self.q_values[action] = (1 - self.learning_rate) * self.q_values[action] + \
                                self.learning_rate * (reward + self.discount_factor * best_next_q)

    def update_memory(self, my_move, neighbor_moves, reward):
        # (rest of update_memory as before)
        self.memory.append({
            'my_move': my_move,
            'neighbor_moves': neighbor_moves,
            'reward': reward,
        })

    def reset(self):
       # (rest of reset as before)
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}