# environment.py
import random  # needed for network creation
import networkx as nx

def log_network_stats(graph, network_type):
    """
    Logs key statistics for a given NetworkX graph.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees.values()) if degrees else 0
    min_degree = min(degrees.values()) if degrees else 0

    print(f"Network Type: {network_type}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}")
    print(f"Degree - Avg: {avg_degree:.2f}, Min: {min_degree}, Max: {max_degree}")
    # Optionally, you could print the entire degree distribution:
    print("Degree Distribution:")
    for node, deg in degrees.items():
        print(f"  Node {node}: Degree {deg}")

# Example usage within the Environment class
class Environment:
    def __init__(self, agents, payoff_matrix, network_type="fully_connected", network_params=None):
        self.agents = agents
        self.payoff_matrix = payoff_matrix
        self.network_type = network_type
        self.network_params = network_params or {}
        # Store the graph (before converting to a dict) for logging
        self.graph = self._create_network_graph()
        # Log network statistics right after creation:
        log_network_stats(self.graph, self.network_type)
        # Convert graph to dict for neighbor lookup:
        self.network = {agent.agent_id: list(self.graph.neighbors(agent.agent_id))
                        for agent in self.agents}

    def _create_network_graph(self):
        num_agents = len(self.agents)
        if self.network_type == "fully_connected":
            graph = nx.complete_graph(num_agents)
        elif self.network_type == "random":
            probability = self.network_params.get("probability", 0.5)
            graph = nx.erdos_renyi_graph(num_agents, probability)
        elif self.network_type == "small_world":
            k = self.network_params.get("k", 4)
            beta = self.network_params.get("beta", 0.2)
            graph = nx.watts_strogatz_graph(num_agents, k, beta)
        elif self.network_type == "scale_free":
            m = self.network_params.get("m", 2)
            graph = nx.barabasi_albert_graph(num_agents, m)
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
        return graph

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

                # Collect all Q-learning agents
                q_learning_agents = [agent for agent in self.agents if agent.strategy == "q_learning"]
                if q_learning_agents:
                    avg_coop_q = sum(a.q_values["cooperate"] for a in q_learning_agents) / len(q_learning_agents)
                    avg_defect_q = sum(a.q_values["defect"] for a in q_learning_agents) / len(q_learning_agents)
                    print(f"Round: {round_num + 1}, "
                          f"Avg Coop Rate: {avg_coop_rate:.2f}, "
                          f"Avg Score: {avg_score:.2f}, "
                          f"Avg Q(c): {avg_coop_q:.2f}, "
                          f"Avg Q(d): {avg_defect_q:.2f}")
                else:
                    print(f"Round: {round_num + 1}, "
                          f"Avg Coop Rate: {avg_coop_rate:.2f}, "
                          f"Avg Score: {avg_score:.2f}")
        return results
