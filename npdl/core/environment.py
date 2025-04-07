# environment.py
import networkx as nx
import numpy as np
import random
import logging
from npdl.core.logging_utils import log_network_stats, log_round_stats
from npdl.core.utils import get_pairwise_payoffs
from typing import List, Dict, Tuple, Any, Optional, Union, Hashable, Set

class Environment:
    """Environment class that manages agent interactions and network structure."""
    
    def __init__(self, agents, payoff_matrix, network_type="fully_connected", 
                 network_params=None, logger=None, interaction_mode="neighborhood", **kwargs):
        """Initialize the environment.
        
        Args:
            agents: List of Agent objects
            payoff_matrix: Dictionary mapping actions to payoff lists
            network_type: Type of network to create
            network_params: Parameters for network creation
            logger: Logger object (if None, uses the root logger)
            interaction_mode: Mode of interaction - "neighborhood" (original) or "pairwise"
            **kwargs: Additional parameters including 2-player PD payoff values (R, S, T, P)
        """
        self.agents = agents
        self.payoff_matrix = payoff_matrix
        self.network_type = network_type
        self.network_params = network_params or {}
        self.logger = logger or logging.getLogger()
        self.interaction_mode = interaction_mode
        
        # Store 2-player PD payoff values for pairwise interactions
        self.R = kwargs.get("R", 3)  # Reward for mutual cooperation
        self.S = kwargs.get("S", 0)  # Sucker's payoff
        self.T = kwargs.get("T", 5)  # Temptation to defect
        self.P = kwargs.get("P", 1)  # Punishment for mutual defection
        
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

    def calculate_payoffs(self, moves, include_global_bonus=True):
        """Calculate payoffs for all agents based on their moves and neighbors' moves.
        
        Args:
            moves: Dictionary mapping agent_id to move
            include_global_bonus: Whether to include a bonus for global cooperation
            
        Returns:
            Dictionary mapping agent_id to payoff
        """
        payoffs = {}
        
        # Calculate global cooperation rate if needed
        global_coop_rate = 0
        if include_global_bonus:
            global_coop_rate = sum(1 for move in moves.values() if move == "cooperate") / len(moves)
            # Quadratic bonus - grows faster as cooperation increases
            global_bonus = global_coop_rate ** 2 * 2  # Max bonus of 2 when all cooperate
        
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
            
            # Calculate base payoff based on move
            if agent_move == "cooperate":
                base_payoff = self.payoff_matrix["C"][num_cooperating_neighbors]
                # Add cooperation bonus only for cooperators if enabled
                if include_global_bonus:
                    payoffs[agent_id] = base_payoff + global_bonus
                else:
                    payoffs[agent_id] = base_payoff
            else:  # defect
                payoffs[agent_id] = self.payoff_matrix["D"][num_cooperating_neighbors]
        
        return payoffs

    def run_round(self, use_global_bonus=True, rewiring_prob=0.0):
        """Run a single round of the simulation.
        
        Args:
            use_global_bonus: Whether to include global cooperation bonus in payoffs
            rewiring_prob: Probability to rewire network edges after this round
            
        Returns:
            Tuple of (moves, payoffs) dictionaries
        """
        if self.interaction_mode == "pairwise":
            return self._run_pairwise_round(rewiring_prob)
        else:
            # Original neighborhood-based interaction
            return self._run_neighborhood_round(use_global_bonus, rewiring_prob)
    
    def _run_neighborhood_round(self, use_global_bonus=True, rewiring_prob=0.0):
        """Run a single round using the original neighborhood-based interaction model.
        
        Args:
            use_global_bonus: Whether to include global cooperation bonus in payoffs
            rewiring_prob: Probability to rewire network edges after this round
            
        Returns:
            Tuple of (moves, payoffs) dictionaries
        """
        # Agents choose moves
        moves = {agent.agent_id: agent.choose_move(self.get_neighbors(agent.agent_id))
                 for agent in self.agents}
        
        # Calculate payoffs
        payoffs = self.calculate_payoffs(moves, include_global_bonus=use_global_bonus)
        
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
        
        # Optionally update network structure
        if rewiring_prob > 0:
            self.update_network(rewiring_prob=rewiring_prob)
            
        return moves, payoffs
    
    def _run_pairwise_round(self, rewiring_prob=0.0):
        """Run a single round using pairwise interactions between all agents.
        
        In this mode, each agent plays a separate 2-player PD game against every other agent.
        The agent's score for the round is the sum of payoffs from all its pairwise games.
        
        How it works:
        1. Each agent chooses a single move (C or D) for the entire round
        2. Every agent plays a 2-player PD game with every other agent
        3. Payoffs are accumulated for each agent from all their pairwise games
        4. Information about opponents' moves is stored as proportion of cooperators
        
        Args:
            rewiring_prob: Probability to rewire network edges after this round
            
        Returns:
            Tuple of (moves, payoffs) dictionaries
        """
        # Step 1: Each agent chooses one move for the round (used against all opponents)
        # For special strategies like TitForTat, we need to ensure they can handle an empty neighbor list
        # or a neighbor_moves with opponent_coop_proportion
        agent_moves_for_round = {}
        for agent in self.agents:
            try:
                # First try with empty list for strategies that don't need neighbors
                move = agent.choose_move([])
            except:
                # If that fails, try with the agent's network neighbors
                # This ensures TitForTat and similar strategies can at least get neighbor IDs
                neighbors = self.get_neighbors(agent.agent_id)
                move = agent.choose_move(neighbors)
            agent_moves_for_round[agent.agent_id] = move
        
        # Step 2: Simulate Pairwise Games between all agent pairs
        pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}  # Store all payoffs per agent
        opponent_coop_counts = {agent.agent_id: 0 for agent in self.agents}  # Track how many opponents cooperated *against* this agent
        opponent_total_counts = {agent.agent_id: 0 for agent in self.agents}
        
        agent_ids = [a.agent_id for a in self.agents]
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i_id = agent_ids[i]
                agent_j_id = agent_ids[j]
                move_i = agent_moves_for_round[agent_i_id]
                move_j = agent_moves_for_round[agent_j_id]
                
                # Get 2-player payoffs
                payoff_i, payoff_j = get_pairwise_payoffs(move_i, move_j, self.R, self.S, self.T, self.P)
                
                pairwise_payoffs[agent_i_id].append(payoff_i)
                pairwise_payoffs[agent_j_id].append(payoff_j)
                
                # Track opponent cooperation for state calculation later
                opponent_total_counts[agent_i_id] += 1
                opponent_total_counts[agent_j_id] += 1
                if move_j == "cooperate":
                    opponent_coop_counts[agent_i_id] += 1
                if move_i == "cooperate":
                    opponent_coop_counts[agent_j_id] += 1
        
        # Step 3: Aggregate Results & Update Agents
        round_payoffs_avg = {}
        round_opponent_coop_proportion = {}
        round_payoffs_total = {}
        
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_total_payoff = sum(pairwise_payoffs[agent_id])
            num_games = len(pairwise_payoffs[agent_id])
            agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
            
            round_payoffs_avg[agent_id] = agent_avg_payoff
            round_payoffs_total[agent_id] = agent_total_payoff
            
            # Calculate aggregate opponent behavior for this round (used for *next* state)
            opp_total = opponent_total_counts[agent_id]
            opp_coop = opponent_coop_counts[agent_id]
            opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5  # Default if isolated
            round_opponent_coop_proportion[agent_id] = opp_coop_prop
            
            # Update agent score (using total payoff)
            agent.score += agent_total_payoff
            
            # Store the aggregate state info instead of individual neighbor moves
            neighbor_moves = {'opponent_coop_proportion': opp_coop_prop}
            
            # Update memory with average reward and aggregate opponent state
            agent.update_memory(
                my_move=agent_moves_for_round[agent_id],
                neighbor_moves=neighbor_moves,
                reward=agent_avg_payoff  # Use average reward for memory/learning signal
            )
        
        # Step 4: Q-Value Updates (after memory is updated for all agents)
        for agent in self.agents:
            if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                      "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
                # Pass opponent cooperation proportion for next state calculation
                neighbor_moves = {'opponent_coop_proportion': round_opponent_coop_proportion[agent.agent_id]}
                agent.update_q_value(
                    action=agent_moves_for_round[agent.agent_id],
                    reward=round_payoffs_avg[agent.agent_id],
                    next_state_actions=neighbor_moves
                )
        
        # Optionally update network structure (for visualization, not functional in pairwise mode)
        if rewiring_prob > 0:
            self.update_network(rewiring_prob=rewiring_prob)
        
        return agent_moves_for_round, round_payoffs_total  # Return total payoffs for consistency with original model

    def run_simulation(self, num_rounds, logging_interval=10, use_global_bonus=True, 
                   rewiring_interval=0, rewiring_prob=0.0):
        """Run a full simulation for the specified number of rounds.
        
        Args:
            num_rounds: Number of rounds to simulate
            logging_interval: Log statistics every N rounds
            use_global_bonus: Whether to include global cooperation bonus in payoffs
            rewiring_interval: Interval (in rounds) to perform network rewiring
            rewiring_prob: Probability of rewiring per edge when rewiring occurs
            
        Returns:
            List of round results
        """
        results = []
        
        for round_num in range(num_rounds):
            # Determine if we should rewire the network this round
            do_rewiring = (rewiring_interval > 0 and 
                          rewiring_prob > 0 and 
                          round_num > 0 and 
                          round_num % rewiring_interval == 0)
            
            # Run a single round
            moves, payoffs = self.run_round(use_global_bonus=use_global_bonus,
                                          rewiring_prob=rewiring_prob if do_rewiring else 0.0)
            
            # Store results
            results.append({'round': round_num, 'moves': moves, 'payoffs': payoffs})
            
            # Log statistics
            log_round_stats(round_num, self.agents, moves, payoffs, 
                           self.logger, logging_interval)
        
        self.logger.info(f"Simulation completed: {num_rounds} rounds")
        return results
    
    def update_network(self, rewiring_prob=0.05, cooperation_bias=0.7):
        """Dynamically update the network structure during simulation.
        
        Args:
            rewiring_prob: Probability of rewiring an edge
            cooperation_bias: Tendency to prefer connections with similar cooperation level
            
        Returns:
            Number of edges rewired
        """
        if rewiring_prob <= 0:
            return 0
        
        # Create a copy of the graph
        G = self.graph.copy()
        
        # Count rewired edges
        rewired_count = 0
        
        # Calculate cooperation rates for all agents
        coop_rates = {}
        for agent in self.agents:
            if agent.memory:
                # Look at last 10 moves (or fewer if not available)
                recent_moves = [entry['my_move'] for entry in agent.memory]
                coop_count = sum(1 for move in recent_moves if move == "cooperate")
                coop_rates[agent.agent_id] = coop_count / len(recent_moves) if recent_moves else 0.5
            else:
                coop_rates[agent.agent_id] = 0.5  # Default if no memory
        
        # For each edge, decide whether to rewire
        edges = list(G.edges())
        for u, v in edges:
            if random.random() < rewiring_prob:
                # Get cooperation rates
                u_coop_rate = coop_rates.get(u, 0.5)
                v_coop_rate = coop_rates.get(v, 0.5)
                
                # Higher probability to disconnect if cooperation rates differ significantly
                # or if both agents are defectors
                disconnect_prob = abs(u_coop_rate - v_coop_rate)
                if u_coop_rate < 0.3 and v_coop_rate < 0.3:  # Both are defectors
                    disconnect_prob += 0.2  # Additional bias to break defector links
                
                if random.random() < disconnect_prob:
                    # Remove the existing edge
                    G.remove_edge(u, v)
                    
                    # Find potential new neighbors for u
                    potential_nodes = []
                    for node in G.nodes():
                        if node != u and node != v and not G.has_edge(u, node):
                            node_coop_rate = coop_rates.get(node, 0.5)
                            
                            # Calculate similarity score
                            similarity = 1 - abs(u_coop_rate - node_coop_rate)
                            
                            # Higher similarity means higher chance to be selected
                            # Add node multiple times based on similarity
                            weight = int(similarity * 10) + 1
                            potential_nodes.extend([node] * weight)
                    
                    if potential_nodes:
                        # Rewire to a new neighbor (weighted by similarity)
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
        
    def export_network_structure(self):
        """Export the network structure as a dictionary for visualization.
        
        Returns:
            Dictionary containing nodes and edges information
        """
        G = self.graph
        
        # Create dictionary representation of the network
        export_data = {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "network_type": self.network_type,
            "network_params": self.network_params,
        }
        
        # Add node attributes if available
        node_attrs = {}
        for node in G.nodes():
            # Check if node has any attributes
            if G.nodes[node]:
                node_attrs[node] = G.nodes[node].copy()
                
        if node_attrs:
            export_data["node_attributes"] = node_attrs
            
        # Add network metrics
        export_data["metrics"] = self.get_network_metrics()
        
        return export_data