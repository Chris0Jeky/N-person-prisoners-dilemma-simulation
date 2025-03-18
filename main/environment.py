# environment.py
import networkx as nx
import numpy as np
import random
import logging
from logging_utils import log_network_stats, log_round_stats

class Environment:
    """Environment class that manages agent interactions and network structure."""
    
    def __init__(self, agents, payoff_matrix, network_type="fully_connected", 
                 network_params=None, logger=None):
        """Initialize the environment.
        
        Args:
            agents: List of Agent objects
            payoff_matrix: Dictionary mapping actions to payoff lists
            network_type: Type of network to create
            network_params: Parameters for network creation
            logger: Logger object (if None, uses the root logger)
        """
        self.agents = agents
        self.payoff_matrix = payoff_matrix
        self.network_type = network_type
        self.network_params = network_params or {}
        self.logger = logger or logging.getLogger()
        
        # Create network graph
        self.graph = self._create_network_graph()
        
        # Log network statistics
        log_network_stats(self.graph, self.network_type, self.logger)
        
        # Convert graph to neighbor dictionary for faster lookup
        self.network = {agent.agent_id: list(self.graph.neighbors(agent.agent_id))
                        for agent in self.agents}
        
        # Store history of states
        self.history = []

    def _create_network_graph(self):
        """Create a NetworkX graph based on the specified network type and parameters."""
        num_agents = len(self.agents)
        
        if self.network_type == "fully_connected":
            graph = nx.complete_graph(num_agents)
        elif self.network_type == "random":
            probability = self.network_params.get("probability", 0.5)
            graph = nx.erdos_renyi_graph(num_agents, probability)
            
            # Ensure the graph is connected
            if not nx.is_connected(graph) and num_agents > 1:
                # Get connected components
                components = list(nx.connected_components(graph))
                
                # Connect components by adding edges between them
                for i in range(len(components) - 1):
                    node1 = random.choice(list(components[i]))
                    node2 = random.choice(list(components[i + 1]))
                    graph.add_edge(node1, node2)
                
                self.logger.info("Random graph was disconnected. Added edges to connect components.")
        elif self.network_type == "small_world":
            k = min(self.network_params.get("k", 4), num_agents - 1)
            beta = self.network_params.get("beta", 0.2)
            
            if num_agents <= k:
                self.logger.warning(f"Not enough agents for Small World with k={k}. Creating complete graph instead.")
                graph = nx.complete_graph(num_agents)
            else:
                graph = nx.watts_strogatz_graph(num_agents, k, beta)
        elif self.network_type == "scale_free":
            m = min(self.network_params.get("m", 2), num_agents - 1)
            
            if num_agents <= m:
                self.logger.warning(f"Not enough agents for Scale Free with m={m}. Creating complete graph instead.")
                graph = nx.complete_graph(num_agents)
            else:
                graph = nx.barabasi_albert_graph(num_agents, m)
        elif self.network_type == "regular":
            k = min(self.network_params.get("k", 4), num_agents - 1)
            
            if num_agents <= k:
                self.logger.warning(f"Not enough agents for Regular with k={k}. Creating complete graph instead.")
                graph = nx.complete_graph(num_agents)
            # k must be even for nx.random_regular_graph
            elif k % 2 == 1 and num_agents % 2 == 1:
                self.logger.warning(f"Both k={k} and num_agents={num_agents} are odd. Using k-1 instead.")
                graph = nx.random_regular_graph(k - 1, num_agents)
            else:
                graph = nx.random_regular_graph(k, num_agents)
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
        
        return graph

    def get_neighbors(self, agent_id):
        """Get the neighbors of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of neighbor agent IDs
        """
        return self.network[agent_id]

    def calculate_payoffs(self, moves):
        """Calculate payoffs for all agents based on their moves and neighbors' moves.
        
        Args:
            moves: Dictionary mapping agent_id to move
            
        Returns:
            Dictionary mapping agent_id to payoff
        """
        payoffs = {}
        
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_move = moves[agent_id]
            neighbors = self.get_neighbors(agent_id)
            
            # Skip if agent has no neighbors (isolated node)
            if not neighbors:
                payoffs[agent_id] = 0
                continue
            
            # Count cooperating neighbors
            neighbor_moves = {n_id: moves[n_id] for n_id in neighbors}
            num_cooperating_neighbors = sum(1 for move in neighbor_moves.values() if move == "cooperate")
            
            # Calculate payoff based on move
            if agent_move == "cooperate":
                payoffs[agent_id] = self.payoff_matrix["C"][num_cooperating_neighbors]
            else:  # defect
                payoffs[agent_id] = self.payoff_matrix["D"][num_cooperating_neighbors]
        
        return payoffs

    def run_round(self):
        """Run a single round of the simulation.
        
        Returns:
            Tuple of (moves, payoffs) dictionaries
        """
        # Agents choose moves
        moves = {agent.agent_id: agent.choose_move(self.get_neighbors(agent.agent_id))
                 for agent in self.agents}
        
        # Calculate payoffs
        payoffs = self.calculate_payoffs(moves)
        
        # Update agent states
        for agent in self.agents:
            # Update score
            agent.score += payoffs[agent.agent_id]
            
            # Get neighbor moves
            neighbors = self.get_neighbors(agent.agent_id)
            neighbor_moves = {n_id: moves[n_id] for n_id in neighbors if n_id in moves}
            
            # Update Q-values and memory
            agent.update_q_value(moves[agent.agent_id], payoffs[agent.agent_id], neighbor_moves)
            agent.update_memory(moves[agent.agent_id], neighbor_moves, payoffs[agent.agent_id])
        
        return moves, payoffs

    def run_simulation(self, num_rounds, logging_interval=10):
        """Run a full simulation for the specified number of rounds.
        
        Args:
            num_rounds: Number of rounds to simulate
            logging_interval: Log statistics every N rounds
            
        Returns:
            List of round results
        """
        results = []
        
        for round_num in range(num_rounds):
            # Run a single round
            moves, payoffs = self.run_round()
            
            # Store results
            results.append({'round': round_num, 'moves': moves, 'payoffs': payoffs})
            
            # Log statistics
            log_round_stats(round_num, self.agents, moves, payoffs, 
                           self.logger, logging_interval)
        
        self.logger.info(f"Simulation completed: {num_rounds} rounds")
        return results
    
    def update_network(self, rewiring_prob=0.05):
        """Dynamically update the network structure during simulation.
        
        Args:
            rewiring_prob: Probability of rewiring an edge
            
        Returns:
            Number of edges rewired
        """
        if rewiring_prob <= 0:
            return 0
        
        # Create a copy of the graph
        G = self.graph.copy()
        
        # Count rewired edges
        rewired_count = 0
        
        # For each edge, decide whether to rewire
        edges = list(G.edges())
        for u, v in edges:
            if random.random() < rewiring_prob:
                # Remove the existing edge
                G.remove_edge(u, v)
                
                # Find potential new neighbors for u
                # (nodes that u is not already connected to)
                potential_nodes = [n for n in G.nodes() if n != u and n != v and not G.has_edge(u, n)]
                
                if potential_nodes:
                    # Rewire to a random new neighbor
                    new_neighbor = random.choice(potential_nodes)
                    G.add_edge(u, new_neighbor)
                    rewired_count += 1
                else:
                    # If no potential new neighbors, restore the original edge
                    G.add_edge(u, v)
        
        if rewired_count > 0:
            # Update the graph and network dictionary
            self.graph = G
            self.network = {agent.agent_id: list(G.neighbors(agent.agent_id))
                            for agent in self.agents}
            self.logger.info(f"Network updated: {rewired_count} edges rewired")
        
        return rewired_count
    
    def get_network_metrics(self):
        """Calculate and return network metrics.
        
        Returns:
            Dictionary of network metrics
        """
        G = self.graph
        
        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            "density": nx.density(G),
        }
        
        # Add additional metrics for connected graphs
        if G.number_of_nodes() > 1:
            try:
                if nx.is_connected(G):
                    metrics["avg_clustering"] = nx.average_clustering(G)
                    metrics["avg_path_length"] = nx.average_shortest_path_length(G)
                    metrics["diameter"] = nx.diameter(G)
                else:
                    # Get the largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    largest_cc_graph = G.subgraph(largest_cc)
                    
                    metrics["largest_cc_size"] = len(largest_cc)
                    metrics["avg_clustering"] = nx.average_clustering(G)  # This works for disconnected graphs
                    metrics["avg_path_length_largest_cc"] = nx.average_shortest_path_length(largest_cc_graph)
                    metrics["diameter_largest_cc"] = nx.diameter(largest_cc_graph)
            except Exception as e:
                self.logger.warning(f"Error calculating some network metrics: {e}")
        
        return metrics