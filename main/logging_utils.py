# logging_utils.py
import logging
import os
import networkx as nx

def setup_logging(log_file="experiment.log", level=logging.INFO, console=True):
    """Set up logging configuration.
    
    Args:
        log_file: Path to the log file
        level: Logging level
        console: Whether to also log to console
    """
    handlers = []
    
    if console:
        handlers.append(logging.StreamHandler())
    
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    handlers.append(logging.FileHandler(log_file, mode="w"))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger()

def log_network_stats(graph, network_type, logger=None):
    """Log key statistics for a given NetworkX graph.
    
    Args:
        graph: NetworkX graph object
        network_type: String describing the network type
        logger: Logger object (if None, uses the root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
    max_degree = max(degrees.values()) if degrees else 0
    min_degree = min(degrees.values()) if degrees else 0
    
    # Compute additional network metrics if there are nodes
    if num_nodes > 0:
        try:
            # Calculate clustering coefficient if the graph is not empty
            clustering = nx.average_clustering(graph)
            diameter = nx.diameter(graph) if nx.is_connected(graph) else float('inf')
            avg_path_length = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
        except:
            clustering = "N/A"
            diameter = "N/A"
            avg_path_length = "N/A"
    else:
        clustering = "N/A"
        diameter = "N/A"
        avg_path_length = "N/A"
    
    # Log basic network statistics
    logger.info(f"Network Type: {network_type}")
    logger.info(f"Nodes: {num_nodes}, Edges: {num_edges}")
    logger.info(f"Degree - Avg: {avg_degree:.2f}, Min: {min_degree}, Max: {max_degree}")
    logger.info(f"Clustering Coefficient: {clustering}")
    logger.info(f"Network Diameter: {diameter}")
    logger.info(f"Average Path Length: {avg_path_length}")
    
    # Log detailed degree distribution if verbose
    if logger.level <= logging.DEBUG:
        logger.debug("Degree Distribution:")
        for node, deg in sorted(degrees.items()):
            logger.debug(f"  Node {node}: Degree {deg}")

def log_round_stats(round_num, agents, moves, payoffs, logger=None, logging_interval=10):
    """Log statistics for a simulation round.
    
    Args:
        round_num: Current round number
        agents: List of Agent objects
        moves: Dictionary mapping agent_id to move
        payoffs: Dictionary mapping agent_id to payoff
        logger: Logger object (if None, uses the root logger)
        logging_interval: Only log on rounds that are a multiple of this number
    """
    # Skip logging if not a multiple of logging_interval
    if (round_num + 1) % logging_interval != 0:
        return
    
    if logger is None:
        logger = logging.getLogger()
    
    # Calculate cooperation rate
    coop_count = sum(1 for move in moves.values() if move == "cooperate")
    total_agents = len(moves)
    coop_rate = coop_count / total_agents if total_agents > 0 else 0
    
    # Calculate average score
    total_score = sum(agent.score for agent in agents)
    avg_score = total_score / len(agents) if agents else 0
    
    # Build the log message
    msg = f"Round {round_num + 1}: Cooperation Rate: {coop_rate:.2f}, Avg Score: {avg_score:.2f}"
    
    # Add Q-value information if any agents use Q-learning
    q_learning_agents = [a for a in agents if a.strategy_type in ("q_learning", "q_learning_adaptive")]
    if q_learning_agents:
        avg_q_coop = sum(a.q_values["cooperate"] for a in q_learning_agents) / len(q_learning_agents)
        avg_q_defect = sum(a.q_values["defect"] for a in q_learning_agents) / len(q_learning_agents)
        msg += f", Avg Q(c): {avg_q_coop:.2f}, Avg Q(d): {avg_q_defect:.2f}"
        
        # Log epsilon value for adaptive Q-learning agents
        adaptive_agents = [a for a in agents if a.strategy_type == "q_learning_adaptive"]
        if adaptive_agents:
            avg_epsilon = sum(a.strategy.epsilon for a in adaptive_agents) / len(adaptive_agents)
            msg += f", Avg ε: {avg_epsilon:.3f}"
    
    logger.info(msg)
    
    # If in debug mode, log strategy-specific statistics
    if logger.level <= logging.DEBUG:
        # Group agents by strategy
        strategies = {}
        for agent in agents:
            if agent.strategy_type not in strategies:
                strategies[agent.strategy_type] = []
            strategies[agent.strategy_type].append(agent)
        
        # Log cooperation rate by strategy
        for strategy, strategy_agents in strategies.items():
            strategy_agent_ids = [a.agent_id for a in strategy_agents]
            strategy_moves = [moves[aid] for aid in strategy_agent_ids if aid in moves]
            strategy_coop_rate = sum(1 for m in strategy_moves if m == "cooperate") / len(strategy_moves) if strategy_moves else 0
            logger.debug(f"  {strategy} cooperation rate: {strategy_coop_rate:.2f}")

def log_experiment_summary(scenario, results, agents, logger=None):
    """Log a summary of the experiment results.
    
    Args:
        scenario: Scenario dictionary
        results: List of round results
        agents: List of Agent objects
        logger: Logger object (if None, uses the root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"Experiment Summary for scenario: {scenario['scenario_name']}")
    logger.info(f"Total rounds: {scenario['num_rounds']}")
    
    # Calculate final cooperation rate from the last round
    last_round = results[-1]
    coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
    coop_rate = coop_count / len(last_round['moves'])
    logger.info(f"Final cooperation rate: {coop_rate:.2f}")
    
    # Calculate average scores by strategy
    strategies = {}
    for agent in agents:
        if agent.strategy_type not in strategies:
            strategies[agent.strategy_type] = []
        strategies[agent.strategy_type].append(agent)
    
    logger.info("Average scores by strategy:")
    for strategy, strategy_agents in strategies.items():
        avg_score = sum(a.score for a in strategy_agents) / len(strategy_agents)
        logger.info(f"  {strategy}: {avg_score:.2f}")
    
    # Q-learning specific information
    q_learning_agents = [a for a in agents if a.strategy_type in ("q_learning", "q_learning_adaptive")]
    if q_learning_agents:
        avg_q_coop = sum(a.q_values["cooperate"] for a in q_learning_agents) / len(q_learning_agents)
        avg_q_defect = sum(a.q_values["defect"] for a in q_learning_agents) / len(q_learning_agents)
        logger.info(f"Final average Q-values - Q(c): {avg_q_coop:.2f}, Q(d): {avg_q_defect:.2f}")