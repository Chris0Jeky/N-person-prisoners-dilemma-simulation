# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union

def load_results(scenario_name: str, results_dir: str = "results", base_filename: str = "experiment_results"):
    """Load experiment results for a given scenario.
    
    Args:
        scenario_name: Name of the scenario
        results_dir: Directory containing result files
        base_filename: Base filename prefix for result files
        
    Returns:
        Tuple of (agents_df, rounds_df)
        
    Raises:
        FileNotFoundError: If the scenario directory or result files are not found
    """
    scenario_dir = os.path.join(results_dir, scenario_name)
    
    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    
    # Directory structure with multiple runs
    all_agents = []
    all_rounds = []
    
    # Look for run directories
    run_dirs = [d for d in os.listdir(scenario_dir) if d.startswith('run_')]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in scenario: {scenario_name}")
        
    for run_dir in run_dirs:
        run_path = os.path.join(scenario_dir, run_dir)
        agents_file = os.path.join(run_path, f"{base_filename}_agents.csv")
        rounds_file = os.path.join(run_path, f"{base_filename}_rounds.csv")
        
        if os.path.exists(agents_file) and os.path.exists(rounds_file):
            try:
                agents_df = pd.read_csv(agents_file)
                rounds_df = pd.read_csv(rounds_file)
                all_agents.append(agents_df)
                all_rounds.append(rounds_df)
            except Exception as e:
                logging.error(f"Error reading files from {run_path}: {e}")
    if not all_agents or not all_rounds:
        raise FileNotFoundError(f"No valid run data found for scenario: {scenario_name}")
        
    agents_df = pd.concat(all_agents, ignore_index=True)
    rounds_df = pd.concat(all_rounds, ignore_index=True)
    return agents_df, rounds_df

def plot_cooperation_rate(rounds_df, title=None, figsize=(10, 6), save_path=None):
    """Plot cooperation rate over time.
    
    Args:
        rounds_df: DataFrame containing round-by-round data
        title: Plot title (if None, a default title is used)
        figsize: Figure size tuple
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Convert move to binary (1 for cooperate, 0 for defect)
    rounds_df['cooperation'] = (rounds_df['move'] == 'cooperate').astype(int)
    
    # Group by round and calculate average cooperation rate
    cooperation_by_round = rounds_df.groupby('round')['cooperation'].mean()
    
    # Plot the cooperation rate
    plt.plot(cooperation_by_round.index, cooperation_by_round.values)
    
    # Set title and labels
    plt.title(title or "Cooperation Rate Over Time")
    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_cooperation_by_strategy(rounds_df, title=None, figsize=(10, 6), save_path=None):
    """Plot cooperation rate over time by strategy.
    
    Args:
        rounds_df: DataFrame containing round-by-round data
        title: Plot title (if None, a default title is used)
        figsize: Figure size tuple
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Convert move to binary (1 for cooperate, 0 for defect)
    rounds_df['cooperation'] = (rounds_df['move'] == 'cooperate').astype(int)
    
    # Get list of strategies
    strategies = rounds_df['strategy'].unique()
    
    # Plot cooperation rate for each strategy
    for strategy in strategies:
        strategy_df = rounds_df[rounds_df['strategy'] == strategy]
        cooperation_by_round = strategy_df.groupby('round')['cooperation'].mean()
        plt.plot(cooperation_by_round.index, cooperation_by_round.values, label=strategy)
    
    # Set title and labels
    plt.title(title or "Cooperation Rate by Strategy")
    plt.xlabel("Round")
    plt.ylabel("Cooperation Rate")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_payoffs_by_strategy(rounds_df, title=None, figsize=(10, 6), save_path=None):
    """Plot average payoffs over time by strategy.
    
    Args:
        rounds_df: DataFrame containing round-by-round data
        title: Plot title (if None, a default title is used)
        figsize: Figure size tuple
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Get list of strategies
    strategies = rounds_df['strategy'].unique()
    
    # Plot average payoff for each strategy
    for strategy in strategies:
        strategy_df = rounds_df[rounds_df['strategy'] == strategy]
        payoff_by_round = strategy_df.groupby('round')['payoff'].mean()
        plt.plot(payoff_by_round.index, payoff_by_round.values, label=strategy)
    
    # Set title and labels
    plt.title(title or "Average Payoff by Strategy")
    plt.xlabel("Round")
    plt.ylabel("Average Payoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_score_distribution(agents_df, title=None, figsize=(10, 6), save_path=None):
    """Plot distribution of final scores by strategy.
    
    Args:
        agents_df: DataFrame containing agent-level data
        title: Plot title (if None, a default title is used)
        figsize: Figure size tuple
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create boxplot of scores by strategy
    sns.boxplot(x='strategy', y='final_score', data=agents_df)
    
    # Set title and labels
    plt.title(title or "Distribution of Final Scores by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Final Score")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def visualize_network(environment, metric=None, title=None, figsize=(10, 10), save_path=None):
    """Visualize the agent network with optional node coloring based on metrics.
    
    Args:
        environment: Environment object containing the network graph
        metric: Dictionary mapping agent_id to a metric value for coloring nodes
                (if None, nodes are not colored)
        title: Plot title (if None, a default title is used)
        figsize: Figure size tuple
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    G = environment.graph
    
    # Create layout based on network type
    if environment.network_type == "small_world":
        pos = nx.circular_layout(G)
    elif environment.network_type == "fully_connected":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with optional coloring
    if metric:
        node_colors = [metric.get(node, 0) for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            cmap=plt.cm.viridis,
            node_size=100
        )
        plt.colorbar(nodes)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=100)
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Set title
    plt.title(title or f"{environment.network_type.title()} Network")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def create_analysis_report(scenario_name: str, results_dir: str = "results", output_dir: str = "analysis_results") -> Dict[str, Any]:
    """Generate a comprehensive analysis report for a scenario.
    
    Args:
        scenario_name: Name of the scenario to analyze
        results_dir: Directory containing result files
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing analysis metrics
    
    Raises:
        FileNotFoundError: If the scenario results cannot be loaded
        IOError: If there's an error writing output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    agents_df, rounds_df = load_results(scenario_name, results_dir)
    
    # Generate plots
    cooperation_plot = plot_cooperation_rate(
        rounds_df, 
        title=f"Cooperation Rate Over Time - {scenario_name}",
        save_path=os.path.join(output_dir, f"{scenario_name}_cooperation_rate.png")
    )
    plt.close()
    
    strategy_coop_plot = plot_cooperation_by_strategy(
        rounds_df,
        title=f"Cooperation Rate by Strategy - {scenario_name}",
        save_path=os.path.join(output_dir, f"{scenario_name}_strategy_cooperation.png")
    )
    plt.close()
    
    payoff_plot = plot_payoffs_by_strategy(
        rounds_df,
        title=f"Average Payoff by Strategy - {scenario_name}",
        save_path=os.path.join(output_dir, f"{scenario_name}_strategy_payoffs.png")
    )
    plt.close()
    
    score_dist_plot = plot_score_distribution(
        agents_df,
        title=f"Distribution of Final Scores - {scenario_name}",
        save_path=os.path.join(output_dir, f"{scenario_name}_score_distribution.png")
    )
    plt.close()
    
    # Calculate key metrics
    metrics = {}
    
    # Overall metrics
    metrics['total_rounds'] = rounds_df['round'].max() + 1
    metrics['total_agents'] = len(agents_df)
    
    # Final cooperation rate
    final_round = rounds_df[rounds_df['round'] == rounds_df['round'].max()]
    final_coop_rate = (final_round['move'] == 'cooperate').mean()
    metrics['final_cooperation_rate'] = final_coop_rate
    
    # Cooperation rate by strategy
    strategy_coop = {}
    for strategy in agents_df['strategy'].unique():
        strategy_agents = agents_df[agents_df['strategy'] == strategy]['agent_id'].tolist()
        strategy_final_moves = final_round[final_round['agent_id'].isin(strategy_agents)]
        strategy_coop[strategy] = (strategy_final_moves['move'] == 'cooperate').mean()
    metrics['strategy_cooperation_rates'] = strategy_coop
    
    # Final scores by strategy
    strategy_scores = {}
    for strategy in agents_df['strategy'].unique():
        strategy_agents = agents_df[agents_df['strategy'] == strategy]
        strategy_scores[strategy] = strategy_agents['final_score'].mean()
    metrics['strategy_average_scores'] = strategy_scores
    
    # For Q-learning agents, calculate final Q-values
    q_learning_strategies = ['q_learning', 'q_learning_adaptive', 'lra_q', 'hysteretic_q', 'wolf_phc', 'ucb1_q']
    q_agents = agents_df[agents_df['strategy'].isin(q_learning_strategies)]
    
    if not q_agents.empty:
        # Check which Q-values column exists in the dataframe
        q_value_cols = [col for col in ['final_q_values', 'final_q_values_avg', 'full_q_values'] if col in q_agents.columns]
        
        if q_value_cols:
            q_value_col = q_value_cols[0]  # Use the first found column
            
            # Only process if the column is not empty
            if not q_agents[q_value_col].isna().all():
                # Check if the column contains dictionaries or strings
                first_valid_value = q_agents[q_value_col].dropna().iloc[0] if not q_agents[q_value_col].dropna().empty else None
                
                # This column might contain any of these formats:
                # 1. JSON string of whole Q-table
                # 2. String representation of Python dict for Q-values
                # 3. JSON string of Q-value averages (e.g., {"avg_cooperate": 1.2, "avg_defect": 0.8})
                # 4. Direct dict object (if reading from in-memory data)
                
                if isinstance(first_valid_value, str):
                    # Try to extract Q-values from string representations
                    try:
                        if 'avg_cooperate' in first_valid_value and 'avg_defect' in first_valid_value:
                            # This is likely a summary dict
                            q_values_list = []
                            for val in q_agents[q_value_col].dropna():
                                try:
                                    q_dict = json.loads(val.replace("'", "\""))
                                    q_values_list.append(q_dict)
                                except:
                                    continue
                            
                            if q_values_list:
                                avg_q_coop = np.mean([d.get('avg_cooperate', 0) for d in q_values_list])
                                avg_q_defect = np.mean([d.get('avg_defect', 0) for d in q_values_list])
                                metrics['average_final_q_cooperate'] = avg_q_coop
                                metrics['average_final_q_defect'] = avg_q_defect
                        else:
                            # This might be a full Q-table string or direct Q-values
                            # Let's try to be conservative and only extract if we clearly see cooperate/defect
                            if '"cooperate"' in first_valid_value or "'cooperate'" in first_valid_value:
                                metrics['average_final_q_cooperate'] = "Found but complex format"
                                metrics['average_final_q_defect'] = "Found but complex format"
                    except Exception as e:
                        print(f"Could not process Q-values in {q_value_col}: {e}")
                        metrics['average_final_q_cooperate'] = "Error processing"
                        metrics['average_final_q_defect'] = "Error processing"
                elif isinstance(first_valid_value, dict):
                    # Direct dictionary access
                    if 'avg_cooperate' in first_valid_value and 'avg_defect' in first_valid_value:
                        avg_q_coop = np.mean([d.get('avg_cooperate', 0) for d in q_agents[q_value_col].dropna()])
                        avg_q_defect = np.mean([d.get('avg_defect', 0) for d in q_agents[q_value_col].dropna()])
                    else:
                        # Try direct cooperate/defect keys
                        avg_q_coop = np.mean([d.get('cooperate', 0) for d in q_agents[q_value_col].dropna()])
                        avg_q_defect = np.mean([d.get('defect', 0) for d in q_agents[q_value_col].dropna()])
                    
                    metrics['average_final_q_cooperate'] = avg_q_coop
                    metrics['average_final_q_defect'] = avg_q_defect
                else:
                    # Unknown format, provide a placeholder
                    metrics['average_final_q_cooperate'] = "Unknown format"
                    metrics['average_final_q_defect'] = "Unknown format"
            else:
                # Column exists but all values are None/NaN
                metrics['average_final_q_cooperate'] = "No data"
                metrics['average_final_q_defect'] = "No data"
        else:
            # No Q-value columns found
            metrics['average_final_q_cooperate'] = "No Q-value columns"
            metrics['average_final_q_defect'] = "No Q-value columns"
    else:
        # No Q-learning agents found
        metrics['average_final_q_cooperate'] = "No Q-learning agents"
        metrics['average_final_q_defect'] = "No Q-learning agents"
    
    # Convert NumPy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # Convert metrics to JSON-serializable format
    serializable_metrics = convert_to_serializable(metrics)
    
    # Save metrics to a JSON file
    with open(os.path.join(output_dir, f"{scenario_name}_metrics.json"), 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    return metrics

def analyze_multiple_scenarios(scenarios: List[str], results_dir: str = "results", output_dir: str = "analysis_results") -> Optional[pd.DataFrame]:
    """Analyze multiple scenarios and generate a comparative report.
    
    Args:
        scenarios: List of scenario names to analyze
        results_dir: Directory containing result files
        output_dir: Directory to save analysis results
        
    Returns:
        DataFrame containing comparative metrics, or None if no scenarios were successfully analyzed
    
    Raises:
        IOError: If there's an error creating the output directory
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze each scenario
    all_metrics = []
    for scenario in scenarios:
        try:
            metrics = create_analysis_report(scenario, results_dir, output_dir)
            metrics['scenario_name'] = scenario
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error analyzing scenario {scenario}: {e}")
    
    # Create comparative DataFrame
    if all_metrics:
        comparative_df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        comparative_csv = os.path.join(output_dir, "comparative_analysis.csv")
        comparative_df.to_csv(comparative_csv, index=False)
        
        # Create comparative plots
        plt.figure(figsize=(10, 6))
        plt.bar(comparative_df['scenario_name'], comparative_df['final_cooperation_rate'])
        plt.title("Final Cooperation Rate by Scenario")
        plt.ylabel("Cooperation Rate")
        plt.xlabel("Scenario")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparative_cooperation_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparative_df
    
    return None