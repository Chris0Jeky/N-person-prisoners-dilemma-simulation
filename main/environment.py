# environment.py
import random  # needed for network creation
import networkx as nx  # importing networkx


class Environment:
    def __init__(self, agents, payoff_matrix, network_type="fully_connected", network_params=None):
        self.agents = agents
        self.payoff_matrix = payoff_matrix
        self.network_type = network_type
        self.network_params = network_params or {}  # Use an empty dict if network_params is None
        self.network = self._create_network()

    def _create_network(self):
        """Creates the network structure using networkx."""
        num_agents = len(self.agents)
        graph = nx.Graph()  # Start with an empty graph
        graph.add_nodes_from(range(num_agents))  # adds nodes, one for each agent

        if self.network_type == "fully_connected":
            # Connect every agent to every other agent.
            graph.add_edges_from([(i, j) for i in range(num_agents) for j in range(i + 1, num_agents)])
        elif self.network_type == "random":
            # Use the Erdős-Rényi model (G(n, p)).
            probability = self.network_params.get("probability", 0.5)  # Default probability of 0.5
            graph = nx.erdos_renyi_graph(num_agents, probability)
        elif self.network_type == "small_world":
            # Use the Watts-Strogatz model.
            k = self.network_params.get("k", 4)  # Number of nearest neighbors
            beta = self.network_params.get("beta", 0.2)  # Rewiring probability
            graph = nx.watts_strogatz_graph(num_agents, k, beta)
        elif self.network_type == "scale_free":
            # Use the Barabási-Albert model.
            m = self.network_params.get("m", 2)  # Number of edges to attach
            graph = nx.barabasi_albert_graph(num_agents, m)
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")

        # Convert the networkx graph to a dictionary for easy neighbor lookup.
        return {agent.agent_id: [neighbor for neighbor in graph.neighbors(agent.agent_id)]
                for agent in self.agents}

    def get_neighbors(self, agent_id):
        """Returns the list of neighbors for a given agent."""
        return self.network[agent_id]

    def calculate_payoffs(self, moves):
        # (rest of the calculate_payoffs method as before)
        payoffs = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_move = moves[agent_id]
            neighbors = self.get_neighbors(agent_id)
            neighbor_moves = {n_id: moves[n_id] for n_id in neighbors}
            num_cooperating_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")

            if agent_move == "cooperate":
                payoffs[agent_id] = self.payoff_matrix["C"][num_cooperating_neighbors]
            else:
                payoffs[agent_id] = self.payoff_matrix["D"][num_cooperating_neighbors]
        return payoffs

    def run_round(self):
        # (rest of the run_round method as before)
        moves = {agent.agent_id: agent.choose_move(self.get_neighbors(agent.agent_id))
                 for agent in self.agents}

        payoffs = self.calculate_payoffs(moves)

        # Update scores, Q-values and memory
        for agent in self.agents:
            agent.score += payoffs[agent.agent_id]
            neighbor_moves = {n_id: moves[n_id] for n_id in self.get_neighbors(agent.agent_id)}
            agent.update_q_value(moves[agent.agent_id], payoffs[agent.agent_id], neighbor_moves)
            agent.update_memory(moves[agent.agent_id], neighbor_moves, payoffs[agent.agent_id])

        return moves, payoffs

    def run_simulation(self, num_rounds):
        results = []
        for round_num in range(num_rounds):
            moves, payoffs = self.run_round()
            results.append({'round': round_num, 'moves': moves, 'payoffs': payoffs})

            # LOGGING: print every 10 rounds (as an example)
            if (round_num + 1) % 10 == 0:
                # Calculate average cooperation rate
                coop_count = sum(1 for move in moves.values() if move == "cooperate")
                avg_coop_rate = coop_count / len(self.agents)

                # Calculate average score
                total_score = sum(agent.score for agent in self.agents)
                avg_score = total_score / len(self.agents)

                print(f"Round: {round_num + 1}, "
                      f"Avg Coop Rate: {avg_coop_rate:.2f}, "
                      f"Avg Score: {avg_score:.2f}")

        return results
