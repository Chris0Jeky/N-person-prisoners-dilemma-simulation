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
        # Ensure console handler doesn't fail on special characters
        console_handler = logging.StreamHandler()
        # Add a formatter that replaces problematic characters
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Use UTF-8 encoding for the file handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding='utf-8')
    handlers.append(file_handler)
    
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
    q_learning_agents = [a for a in agents if a.strategy_type in ("q_learning", "q_learning_adaptive", "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q")]
    if q_learning_agents:
        # Calculate average Q-values across all states
        avg_q_coop = 0
        avg_q_defect = 0
        q_count = 0
        
        for agent in q_learning_agents:
            for state in agent.q_values:
                if "cooperate" in agent.q_values[state] and "defect" in agent.q_values[state]:
                    avg_q_coop += agent.q_values[state]["cooperate"]
                    avg_q_defect += agent.q_values[state]["defect"]
                    q_count += 1
        
        if q_count > 0:
            avg_q_coop /= q_count
            avg_q_defect /= q_count
            msg += f", Avg Q(c): {avg_q_coop:.2f}, Avg Q(d): {avg_q_defect:.2f}"
        
        # Log epsilon value for adaptive Q-learning agents (using 'eps' instead of Greek letter)
        adaptive_agents = [a for a in agents if a.strategy_type == "q_learning_adaptive"]
        if adaptive_agents:
            avg_epsilon = sum(a.strategy.epsilon for a in adaptive_agents) / len(adaptive_agents)
            msg += f", Avg eps: {avg_epsilon:.3f}"
    
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

def log_experiment_summary(scenario, run_number, agents, round_results, theoretical_scores, logger=None):
    """Log a summary of the experiment results.
    
    Args:
        scenario: Scenario dictionary
        run_number: Number of the current run
        agents: List of Agent objects
        round_results: List of round results
        theoretical_scores: Dictionary with theoretical max scores
        logger: Logger object (if None, uses the root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"Experiment Summary for scenario: {scenario['scenario_name']}")
    logger.info(f"Total rounds: {scenario['num_rounds']}")

    # Calculate actual final global score for this run
    total_actual_score = sum(agent.score for agent in agents)
    
    # Calculate final cooperation rate from the last round
    last_round = round_results[-1]
    coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
    coop_rate = coop_count / len(last_round['moves'])
    logger.info(f"Final cooperation rate: {coop_rate:.2f}")
    logger.info(f"Run {run_number} Final cooperation rate: {coop_rate:.2f}")
    logger.info(f"Run {run_number} Final Global Score: {total_actual_score:.2f}")
    logger.info(f"  Theoretical Max Coop Score: {theoretical_scores['max_cooperation']:.2f}")
    logger.info(f"  Theoretical Max Defect Score: {theoretical_scores['max_defection']:.2f}")
    logger.info(f"  Theoretical Half-Half Score: {theoretical_scores['half_half']:.2f}")
    
    # Calculate average scores by strategy
    strategies = {}
    for agent in agents:
        if agent.strategy_type not in strategies:
            strategies[agent.strategy_type] = []
        strategies[agent.strategy_type].append(agent)
    
    logger.info(f"Run {run_number} Average scores by strategy:")
    for strategy, strategy_agents in strategies.items():
        avg_score = sum(a.score for a in strategy_agents) / len(strategy_agents)
        logger.info(f"  {strategy}: {avg_score:.2f}")
    
    # Q-learning specific information
    q_learning_agents = [a for a in agents if a.strategy_type in ("q_learning", "q_learning_adaptive", "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q")]
    if q_learning_agents:
        # Calculate average Q-values across all states
        avg_q_coop = 0
        avg_q_defect = 0
        q_count = 0
        
        for agent in q_learning_agents:
            for state in agent.q_values:
                if "cooperate" in agent.q_values[state] and "defect" in agent.q_values[state]:
                    avg_q_coop += agent.q_values[state]["cooperate"]
                    avg_q_defect += agent.q_values[state]["defect"]
                    q_count += 1
        
        if q_count > 0:
            avg_q_coop /= q_count
            avg_q_defect /= q_count
            logger.info(f"Final average Q-values - Q(c): {avg_q_coop:.2f}, Q(d): {avg_q_defect:.2f}")
        
    # Create a more detailed summary
    logger.info("\n--- DETAILED TOURNAMENT SUMMARY ---")
    
    # 1. Cooperation rate trend analysis
    if len(round_results) >= 3:  # Need at least 3 points to analyze a trend
        start_coop = sum(1 for move in round_results[0]['moves'].values() if move == "cooperate") / len(round_results[0]['moves'])
        middle_idx = len(round_results) // 2
        middle_coop = sum(1 for move in round_results[middle_idx]['moves'].values() if move == "cooperate") / len(round_results[middle_idx]['moves'])
        end_coop = coop_rate
        
        if end_coop > start_coop + 0.1:
            trend = "INCREASING"
        elif end_coop < start_coop - 0.1:
            trend = "DECREASING"
        else:
            trend = "STABLE"
            
        logger.info(f"Cooperation Trend: {trend}")
        logger.info(f"  Initial: {start_coop:.2f}, Middle: {middle_coop:.2f}, Final: {end_coop:.2f}")
    
    # 2. Strategy performance comparison
    if len(strategies) > 1:
        # Sort strategies by average score (descending)
        sorted_strategies = sorted(
            [(strategy, sum(a.score for a in agents) / len(agents)) 
             for strategy, agents in strategies.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info("Strategy Ranking (by score):")
        for i, (strategy, avg_score) in enumerate(sorted_strategies):
            logger.info(f"  {i+1}. {strategy}: {avg_score:.2f}")
        
        # Find most and least cooperative strategies
        strategy_coop_rates = {}
        for strategy, strategy_agents in strategies.items():
            strategy_agent_ids = [a.agent_id for a in strategy_agents]
            strategy_moves = [last_round['moves'][aid] for aid in strategy_agent_ids if aid in last_round['moves']]
            strategy_coop_rate = sum(1 for m in strategy_moves if m == "cooperate") / len(strategy_moves) if strategy_moves else 0
            strategy_coop_rates[strategy] = strategy_coop_rate
        
        most_coop = max(strategy_coop_rates.items(), key=lambda x: x[1])
        least_coop = min(strategy_coop_rates.items(), key=lambda x: x[1])
        
        logger.info(f"Most cooperative strategy: {most_coop[0]} ({most_coop[1]:.2f})")
        logger.info(f"Least cooperative strategy: {least_coop[0]} ({least_coop[1]:.2f})")
    
    # 3. Network effects
    logger.info(f"Network type: {scenario['network_type']}")
    if scenario['network_type'] != "fully_connected":
        logger.info(f"Network parameters: {scenario['network_params']}")
    
    # 4. Learning analysis for Q-learning agents
    if q_learning_agents and len(round_results) > 1:
        # Check if Q-values have converged
        if 'avg_q_coop' in locals() and 'avg_q_defect' in locals() and abs(avg_q_coop - avg_q_defect) > 3.0:
            if avg_q_defect > avg_q_coop:
                logger.info("Q-learning outcome: Strong preference for DEFECTION")
            else:
                logger.info("Q-learning outcome: Strong preference for COOPERATION")
        else:
            logger.info("Q-learning outcome: Mixed strategy (no strong preference)")
    
    logger.info(f"--- End Run {run_number} Summary ---")
        
def generate_ascii_chart(values, title="", width=50, height=10):
    """Generate a simple ASCII chart for displaying data trends.
    
    Args:
        values: List of numeric values to plot
        title: Chart title
        width: Width of the chart in characters
        height: Height of the chart in characters
        
    Returns:
        String containing ASCII chart
    """
    if not values:
        return "No data to plot"
    
    # Find min and max values
    min_val = min(values)
    max_val = max(values)
    
    # Avoid division by zero
    if max_val == min_val:
        max_val = min_val + 1
    
    # Create chart
    chart = [" " * width for _ in range(height)]
    
    # Plot points
    for i, val in enumerate(values):
        # Scale x-coordinate to fit width
        x = int((i / (len(values) - 1 if len(values) > 1 else 1)) * (width - 1))
        
        # Scale y-coordinate to fit height, and invert (0 at bottom)
        y = height - 1 - int(((val - min_val) / (max_val - min_val)) * (height - 1))
        
        # Plot point
        if 0 <= y < height:
            chart[y] = chart[y][:x] + "*" + chart[y][x+1:]
    
    # Add title
    result = title + "\n" if title else ""
    
    # Add y-axis labels
    max_str = f"{max_val:.2f}"
    mid_str = f"{(min_val + max_val) / 2:.2f}"
    min_str = f"{min_val:.2f}"
    
    # Add chart with axes
    result += max_str + " ┌" + "─" * width + "┐\n"
    for i, line in enumerate(chart):
        if i == 0:
            result += " " * len(max_str) + "│" + line + "│\n"
        elif i == height // 2:
            result += mid_str + "│" + line + "│\n"
        else:
            result += " " * len(max_str) + "│" + line + "│\n"
    result += min_str + " └" + "─" * width + "┘\n"
    
    return result