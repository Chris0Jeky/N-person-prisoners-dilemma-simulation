import random

class Agent:
    def __init__(self, agent_id, strategy="random", learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id  # Unique identifier
        self.strategy = strategy  # Initial strategy (e.g., "random", "tit_for_tat", "always_defect")
        self.score = 0
        self.memory = []  # List to store history of interactions (optional, but useful)
        self.q_values = {"cooperate": 0.0, "defect": 0.0}  # Initialize Q-values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_move(self, neighbors):
        """Chooses an action based on the agent's strategy."""

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
                # Get the last move of a *random* neighbor.  This is a simplification.
                #  A more sophisticated approach would consider all neighbors.
                last_round = self.memory[-1]
                neighbor_moves = last_round['neighbor_moves']
                if neighbor_moves:  # if the agent has neighbours
                    random_neighbor_move = random.choice(list(neighbor_moves.values()))
                    return random_neighbor_move
                else:  # if the agent hasn't got neighbours
                    return "cooperate"

        elif self.strategy == "q_learning":
            if random.random() < self.epsilon:  # Explore
                return random.choice(["cooperate", "defect"])
            else:  # Exploit
                return "cooperate" if self.q_values["cooperate"] > self.q_values["defect"] else "defect"
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_q_value(self, action, reward, next_state_actions):
        """Updates the Q-value for the given action using the Q-learning update rule."""
        if self.strategy != "q_learning":
            return  # Only update Q-values if using Q-learning

        # Simplified:  Find the best possible action in the *next* state.  This assumes
        #  we know what the other agents *might* do.  In a true multi-agent setting,
        #  this is more complex.
        best_next_action = "cooperate" if self.q_values["cooperate"] > self.q_values[
            "defect"] else "defect"  # Simplified
        best_next_q = self.q_values[best_next_action]

        # Q-learning update rule
        self.q_values[action] = (1 - self.learning_rate) * self.q_values[action] + \
                                self.learning_rate * (reward + self.discount_factor * best_next_q)

    def update_memory(self, my_move, neighbor_moves, reward):
        """Adds interaction to memory."""
        self.memory.append({
            'my_move': my_move,
            'neighbor_moves': neighbor_moves,
            'reward': reward,
        })

    def reset(self):
        """Resets agent values for a new game."""
        self.score = 0
        self.memory = []
        self.q_values = {"cooperate": 0.0, "defect": 0.0}


class Environment:
    def __init__(self, agents, payoff_matrix, network_type="fully_connected"):
        self.agents = agents
        self.payoff_matrix = payoff_matrix
        self.network_type = network_type
        self.network = self._create_network()  # Create the network

    def _create_network(self):
        """Creates the network structure based on the specified type."""
        if self.network_type == "fully_connected":
            #  A dictionary where each agent is connected to all others.
            return {agent.agent_id: [other.agent_id for other in self.agents if other != agent]
                    for agent in self.agents}
        #  Add other network types here (e.g., "random", "small_world", "scale_free")
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")

    def get_neighbors(self, agent_id):
        """Returns the list of neighbors for a given agent."""
        return self.network[agent_id]

    def calculate_payoffs(self, moves):
        """Calculates payoffs for all agents based on their moves."""
        payoffs = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_move = moves[agent_id]
            neighbors = self.get_neighbors(agent_id)
            neighbor_moves = {neighbor_id: moves[neighbor_id] for neighbor_id in neighbors}
            num_cooperating_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")

            if agent_move == "cooperate":
                payoffs[agent_id] = self.payoff_matrix["C"][num_cooperating_neighbors]
            else:  # agent defects
                payoffs[agent_id] = self.payoff_matrix["D"][num_cooperating_neighbors]
        return payoffs

    def run_round(self):
        """Runs a single round of the IPD."""

        # 1. Collect moves from all agents.
        moves = {agent.agent_id: agent.choose_move(self.get_neighbors(agent.agent_id)) for agent in self.agents}

        # 2. Calculate payoffs.
        payoffs = self.calculate_payoffs(moves)
        # print(payoffs)

        # 3. Update agent scores and Q-values.
        for agent in self.agents:
            agent.score += payoffs[agent.agent_id]
            neighbor_moves = {neighbor_id: moves[neighbor_id] for neighbor_id in self.get_neighbors(agent.agent_id)}
            agent.update_q_value(moves[agent.agent_id], payoffs[agent.agent_id], neighbor_moves)
            agent.update_memory(moves[agent.agent_id], neighbor_moves, payoffs[agent.agent_id])

        return moves, payoffs  # returns this round's moves and payoffs

    def run_simulation(self, num_rounds):
        """Runs the simulation for a given number of rounds"""
        results = []
        for round_num in range(num_rounds):
            moves, payoffs = self.run_round()
            results.append({'round': round_num, 'moves': moves, 'payoffs': payoffs})
        return results

    # Example N-person payoff functions (you can customize these)
    def C(n, N, R=3, S=0):
        """Payoff for cooperating when n players cooperate."""
        return S + (R - S) * (n / (N - 1))  # Linear function

    def D(n, N, T=5, P=1):
        """Payoff for defecting when n players cooperate."""
        return P + (T - P) * (n / (N - 1))  # Linear function

    def create_payoff_matrix(N):
        """
        Creates the payoff matrix for an N-person IPD.

        Args:
            N: The number of players.

        Returns:
            A dictionary representing the payoff matrix.  The keys are "C" (cooperate)
            and "D" (defect).  The values are lists, where the i-th element represents
            the payoff when i *other* players cooperate.
        """
        payoff_matrix = {
            "C": [C(n, N) for n in range(N)],  # Payoffs for cooperation
            "D": [D(n, N) for n in range(N)],  # Payoffs for defection
        }
        return payoff_matrix

        # Example Usage

if __name__ == "__main__":
        num_agents = 5
        num_rounds = 100

        # Create agents with different strategies
        agents = [
            Agent(agent_id=i, strategy="random") for i in range(num_agents - 2)
        ]
        agents.append(Agent(agent_id=num_agents - 2, strategy="tit_for_tat"))
        agents.append(Agent(agent_id=num_agents - 1, strategy="q_learning", epsilon=0.3))

        # Create the payoff matrix
        payoff_matrix = create_payoff_matrix(num_agents)
        # print(payoff_matrix)

        # Create the environment
        env = Environment(agents, payoff_matrix)

        # Run the simulation
        results = env.run_simulation(num_rounds)

        # Print some results (I'll want to do more sophisticated analysis)
        # for round_data in results:
        #    print(f"Round: {round_data['round']}")
        # print(f"  Moves: {round_data['moves']}")
        # print(f"  Payoffs: {round_data['payoffs']}")

        for agent in agents:
            print(agent.agent_id, agent.strategy, agent.score, agent.q_values)
