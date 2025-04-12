#!/usr/bin/env python3
"""
Visualization tools for evolutionary scenario generation.

This module provides functions to visualize and analyze the results of
evolutionary scenario generation, including generation trends, diversity
analysis, and comparative metrics.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from collections import defaultdict


def load_evolution_metadata(file_path="results/evolved_scenarios/evolution_metadata.json") -> Dict:
    """Load the metadata about the evolutionary process."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_evolved_scenarios(file_path="results/evolved_scenarios/all_evolved_scenarios.json") -> List[Dict]:
    """Load the data about all evaluated scenarios."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check if we have a list of scenarios or a different format
        if isinstance(data, list):
            # Already in the expected format
            return data
        elif isinstance(data, dict) and 'scenarios' in data:
            # Extract scenarios from a metadata format
            return data['scenarios']
        else:
            # Try to parse the format - handling string content
            scenarios = []
            
            if isinstance(data, dict):
                # Add the entire object as a single scenario
                return [data]
            elif isinstance(data, str):
                print(f"Warning: File contains string data instead of JSON. Attempting to parse...")
                # Try to parse string content if it's a serialized JSON string
                try:
                    parsed_data = json.loads(data)
                    if isinstance(parsed_data, list):
                        return parsed_data
                    elif isinstance(parsed_data, dict):
                        return [parsed_data]
                except json.JSONDecodeError:
                    print("Error: Could not parse string data as JSON")
                    # Return a minimal stub that won't crash the visualization
                    return [{"config": {"scenario_name": "Unknown"}, "metrics": {}, "selection_score": 0}]
            
            print(f"Warning: Unexpected data format in {file_path}")
            return [{"config": {"scenario_name": "Unknown"}, "metrics": {}, "selection_score": 0}]
    except Exception as e:
        print(f"Error loading scenarios file: {e}")
        # Return a minimal stub that won't crash the visualization
        return [{"config": {"scenario_name": "Unknown"}, "metrics": {}, "selection_score": 0}]


def plot_evolution_progress(metadata: Dict, save_path: Optional[str] = None):
    """Plot the progress of the evolutionary algorithm across generations."""
    generations = [gen["generation"] for gen in metadata["generation_stats"]]
    avg_scores = [gen["avg_score"] for gen in metadata["generation_stats"]]
    max_scores = [gen["max_score"] for gen in metadata["generation_stats"]]
    min_scores = [gen["min_score"] for gen in metadata["generation_stats"]]
    
    plt.figure(figsize=(12, 6))
    plt.plot(generations, avg_scores, 'o-', label='Average Score', color='blue')
    plt.plot(generations, max_scores, 's-', label='Best Score', color='green')
    plt.plot(generations, min_scores, '^-', label='Worst Score', color='red')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Interestingness Score', fontsize=12)
    plt.title('Evolution of Scenario Scores Across Generations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Mark the best individual from each generation
    for i, gen in enumerate(metadata["best_scenarios_by_generation"]):
        plt.annotate(
            f"G{i+1}",
            xy=(generations[i], max_scores[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_strategy_evolution(metadata: Dict, save_path: Optional[str] = None):
    """Plot how strategy composition evolved across generations."""
    # Extract strategy counts from each generation
    generations = [gen["generation"] for gen in metadata["generation_stats"]]
    strategy_evolution = defaultdict(list)
    
    # Collect all unique strategies
    all_strategies = set()
    for gen in metadata["generation_stats"]:
        for strategy in gen["scenario_types"].keys():
            all_strategies.add(strategy)
    
    # Create a dataframe with strategy counts for each generation
    for gen in metadata["generation_stats"]:
        for strategy in all_strategies:
            strategy_evolution[strategy].append(gen["scenario_types"].get(strategy, 0))
    
    # Convert to DataFrame for easier plotting
    evolution_df = pd.DataFrame(strategy_evolution, index=generations)
    
    # Normalize to show percentage of strategies
    normalized_df = evolution_df.div(evolution_df.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(14, 8))
    
    # Plot a stacked area chart
    ax = normalized_df.plot.area(alpha=0.8, colormap='tab20')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Strategy Proportion (%)', fontsize=12)
    plt.title('Evolution of Strategy Composition Across Generations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    return normalized_df


def plot_metric_evolution(scenarios: List[Dict], generations: int, 
                         metrics: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """Plot how different metrics evolved across generations."""
    # Organize scenarios by generation
    scenarios_by_gen = defaultdict(list)
    
    # Calculate scenarios per generation, ensuring it's at least 1
    scenarios_per_gen = max(1, len(scenarios) // generations)
    
    # Assign each scenario to its generation based on index in the list
    for i, scenario in enumerate(scenarios):
        gen = min(i // scenarios_per_gen + 1, generations)
        scenarios_by_gen[gen].append(scenario)
    
    # If no metrics specified, find common metrics
    if not metrics:
        metrics = []
        for scenario in scenarios:
            if "metrics" in scenario:
                for key in scenario["metrics"].keys():
                    if key.startswith("avg_") and key not in metrics:
                        metrics.append(key)
        # Take top 6 most common metrics to avoid overcrowding
        metrics = metrics[:6]
    
    # Calculate average metric values per generation
    metric_evolution = {metric: [] for metric in metrics}
    for gen in range(1, generations + 1):
        gen_scenarios = scenarios_by_gen.get(gen, [])
        for metric in metrics:
            values = [s["metrics"].get(metric, np.nan) for s in gen_scenarios]
            # Filter out NaN values
            values = [v for v in values if not pd.isna(v)]
            if values:
                metric_evolution[metric].append(np.mean(values))
            else:
                metric_evolution[metric].append(np.nan)
    
    # Create a dataframe for plotting
    evolution_df = pd.DataFrame(metric_evolution, index=range(1, generations + 1))
    
    # Plot each metric
    plt.figure(figsize=(14, 10))
    
    # Create one subplot for each metric
    num_metrics = len(metrics)
    cols = 2
    rows = (num_metrics + 1) // cols
    
    for i, metric in enumerate(metrics):
        if i < rows * cols:  # Make sure we don't exceed the grid
            ax = plt.subplot(rows, cols, i + 1)
            
            # Clean up metric name for display
            display_name = metric.replace('avg_', '').replace('_', ' ').title()
            
            # Plot the metric evolution with error handling
            try:
                evolution_df[metric].plot(ax=ax, marker='o', linestyle='-', linewidth=2)
                
                ax.set_xlabel('Generation')
                ax.set_ylabel(display_name)
                ax.set_title(f'Evolution of {display_name}')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error plotting {display_name}: {e}")
                ax.text(0.5, 0.5, f"No data available for {display_name}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()  # Close instead of show to avoid displaying in scripts


def plot_top_scenarios_comparison(scenarios: List[Dict], top_n: int = 5, 
                                 save_path: Optional[str] = None):
    """Compare metrics of the top N scenarios."""
    # Sort scenarios by score and take top N
    sorted_scenarios = sorted(scenarios, key=lambda x: x.get("selection_score", 0), reverse=True)
    top_scenarios = sorted_scenarios[:top_n]
    
    # Extract metrics for comparison
    comparison_data = []
    for scenario in top_scenarios:
        scenario_data = {
            "name": scenario["config"].get("scenario_name", "Unknown"),
            "score": scenario.get("selection_score", 0)
        }
        # Add metrics
        for metric, value in scenario.get("metrics", {}).items():
            if metric.startswith("avg_"):
                scenario_data[metric] = value
        
        comparison_data.append(scenario_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Melt the DataFrame for easier plotting
    id_vars = ["name", "score"]
    value_vars = [col for col in df.columns if col.startswith("avg_")]
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, 
                        var_name="metric", value_name="value")
    
    # Clean up metric names
    melted_df["metric"] = melted_df["metric"].str.replace("avg_", "").str.replace("_", " ").str.title()
    
    # Create a plot
    plt.figure(figsize=(14, 10))
    
    # Create a heatmap of metric values
    pivot_df = melted_df.pivot(index="name", columns="metric", values="value")
    
    # Normalize each column to 0-1 scale
    normalized_df = pivot_df.copy()
    for col in normalized_df.columns:
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        if col_max > col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(normalized_df, annot=pivot_df.round(3), fmt=".3f", 
                    cmap="viridis", linewidths=.5, cbar_kws={"label": "Normalized Value"})
    
    plt.title("Comparison of Top Scenario Metrics", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    
    # Return the DataFrame for further analysis
    return df


def create_radar_chart(scenarios: List[Dict], metrics: Optional[List[str]] = None,
                      save_path: Optional[str] = None):
    """Create a radar chart to compare multiple scenarios across different metrics."""
    # Check if we have any scenarios
    if not scenarios:
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, "No scenario data available for radar chart", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Sort scenarios by score
    sorted_scenarios = sorted(scenarios, key=lambda x: x.get("selection_score", 0), reverse=True)
    
    # Use top 5 scenarios or fewer if not enough
    top_scenarios = sorted_scenarios[:min(5, len(sorted_scenarios))]
    
    # If no metrics specified, find common metrics
    if not metrics:
        all_metrics = set()
        for scenario in top_scenarios:
            if "metrics" in scenario:
                for key in scenario.get("metrics", {}).keys():
                    if key.startswith("avg_"):
                        all_metrics.add(key)
        
        # Select a subset of metrics for readability
        important_metrics = [
            "avg_final_coop_rate",
            "avg_coop_rate_change",
            "avg_score_variance", 
            "avg_coop_volatility",
            "avg_strategy_adaptation_rate",
            "avg_pattern_complexity",
            "avg_equilibrium_stability"
        ]
        
        # Use important metrics if available, otherwise use all
        metrics = [m for m in important_metrics if m in all_metrics]
        if not metrics:
            metrics = list(all_metrics)[:8] if all_metrics else []  # Limit to 8 for readability
    
    # Check if we have any metrics to plot
    if not metrics:
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, "No metrics available for radar chart", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Extract scenario data
    labels = []
    data = []
    
    for scenario in top_scenarios:
        if "config" not in scenario:
            continue
            
        name = scenario["config"].get("scenario_name", "Unknown").replace("Evo_", "S")
        labels.append(name)
        
        # Extract metric values, handle missing metrics
        scenario_data = []
        for metric in metrics:
            value = scenario.get("metrics", {}).get(metric, 0)
            scenario_data.append(value)
        
        data.append(scenario_data)
    
    # Check if we have any data to plot
    if not data or not labels:
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, "No data available for radar chart", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Clean metric names for display
    display_metrics = [m.replace("avg_", "").replace("_", " ").title() for m in metrics]
    
    # Create the radar chart
    plt.figure(figsize=(10, 10))
    
    # Compute angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    # Close the polygon by repeating the first angle
    angles += angles[:1]
    
    # Set up subplot
    ax = plt.subplot(111, polar=True)
    
    # Plot each scenario
    for i, scenario_data in enumerate(data):
        # Check for all-zero data
        if sum(abs(v) for v in scenario_data) < 1e-10:
            continue
            
        try:
            # Normalize data for better visualization
            # Avoid division by zero by adding a small epsilon
            max_vals = np.max(np.abs(data), axis=0)
            max_vals = np.where(max_vals < 1e-10, 1.0, max_vals)  # Replace zeros with ones
            
            normalized_data = np.array(scenario_data) / max_vals
            
            # Close the polygon by repeating the first value
            values = normalized_data.tolist()
            values += values[:1]
            
            # Plot the scenario
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=labels[i])
            ax.fill(angles, values, alpha=0.1)
        except Exception as e:
            print(f"Error plotting scenario {labels[i]} in radar chart: {e}")
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Scenario Comparison Radar Chart", size=15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()


def analyze_strategy_performance(scenarios: List[Dict], save_path: Optional[str] = None):
    """Analyze which strategies performed well across all scenarios."""
    # Check if we have any valid scenarios
    valid_scenarios = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if "config" not in scenario or not isinstance(scenario["config"], dict):
            continue
        if "selection_score" not in scenario:
            continue
        valid_scenarios.append(scenario)
    
    if not valid_scenarios:
        print("No valid scenarios found for strategy performance analysis")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No valid scenario data for strategy analysis", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return pd.DataFrame()
    
    # Extract strategy performance data
    strategy_data = []
    
    for scenario in valid_scenarios:
        scenario_score = scenario["selection_score"]
        try:
            strategies = scenario["config"].get("agent_strategies", {})
            if not isinstance(strategies, dict):
                continue
                
            for strategy, count in strategies.items():
                strategy_data.append({
                    "strategy": strategy,
                    "count": count,
                    "scenario_score": scenario_score,
                    "scenario_name": scenario["config"].get("scenario_name", "Unknown")
                })
        except Exception as e:
            print(f"Error processing scenario strategies: {e}")
    
    if not strategy_data:
        print("No strategy data found after processing")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No strategy data available for analysis", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return pd.DataFrame()
        
    # Convert to DataFrame
    df = pd.DataFrame(strategy_data)
    
    # Group by strategy and calculate statistics
    try:
        strategy_stats = df.groupby("strategy").agg(
            count=("count", "sum"),
            avg_scenario_score=("scenario_score", "mean"),
            max_scenario_score=("scenario_score", "max"),
            num_scenarios=("scenario_name", "nunique")
        ).reset_index()
        
        # Sort by average scenario score
        strategy_stats = strategy_stats.sort_values("avg_scenario_score", ascending=False)
    except Exception as e:
        print(f"Error calculating strategy statistics: {e}")
        return pd.DataFrame()
    
    # Plot strategy performance
    plt.figure(figsize=(12, 8))
    
    try:
        # Create a bar plot with error bars
        ax = sns.barplot(
            x="strategy", 
            y="avg_scenario_score", 
            data=strategy_stats,
            palette="viridis"
        )
        
        # Add number of scenarios as text
        for i, row in strategy_stats.iterrows():
            ax.text(
                i, 
                row["avg_scenario_score"] + 0.01, 
                f"n={row['num_scenarios']}", 
                ha='center'
            )
        
        plt.title("Average Scenario Score by Strategy", fontsize=14)
        plt.xlabel("Strategy", fontsize=12)
        plt.ylabel("Average Scenario Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
    except Exception as e:
        print(f"Error creating strategy performance plot: {e}")
        plt.text(0.5, 0.5, "Error creating plot", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return strategy_stats


def create_evolution_report(metadata_file="results/evolved_scenarios/evolution_metadata.json",
                           scenarios_file="results/evolved_scenarios/all_evolved_scenarios.json",
                           output_dir="evolution_analysis"):
    """Create a comprehensive report on the evolutionary process."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        metadata = load_evolution_metadata(metadata_file)
        scenarios = load_evolved_scenarios(scenarios_file)
    except FileNotFoundError as e:
        print(f"Error: Could not load required files: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON format issue in data files: {e}")
        return
    
    print(f"Generating evolution analysis report in {output_dir}...")
    
    # Extract basic information
    try:
        generations = metadata["parameters"]["generations"]
        population_size = metadata["parameters"]["population_size"]
    except KeyError as e:
        print(f"Error: Missing required parameter in metadata: {e}")
        generations = 5  # Default fallback value
        population_size = 20
    
    # Check if we have valid data
    if not scenarios:
        print("Warning: No scenario data available in the provided file")
    
    if not metadata.get("generation_stats"):
        print("Warning: No generation statistics found in metadata")
    
    # 1. Plot evolution progress
    try:
        plot_evolution_progress(metadata, save_path=os.path.join(output_dir, "evolution_progress.png"))
    except Exception as e:
        print(f"Error generating evolution progress plot: {e}")
    
    # 2. Plot strategy evolution
    try:
        strategy_df = plot_strategy_evolution(metadata, save_path=os.path.join(output_dir, "strategy_evolution.png"))
        if strategy_df is not None:
            strategy_df.to_csv(os.path.join(output_dir, "strategy_evolution.csv"))
    except Exception as e:
        print(f"Error generating strategy evolution plot: {e}")
    
    # 3. Plot metric evolution
    try:
        plot_metric_evolution(scenarios, generations, save_path=os.path.join(output_dir, "metric_evolution.png"))
    except Exception as e:
        print(f"Error generating metric evolution plot: {e}")
    
    # 4. Compare top scenarios
    try:
        top_scenarios_df = plot_top_scenarios_comparison(scenarios, 
                                           save_path=os.path.join(output_dir, "top_scenarios_comparison.png"))
        if top_scenarios_df is not None:
            top_scenarios_df.to_csv(os.path.join(output_dir, "top_scenarios.csv"))
    except Exception as e:
        print(f"Error generating top scenarios comparison: {e}")
        
    # Create radar chart separately since it can fail independently
    try:
        create_radar_chart(scenarios[:5] if len(scenarios) >= 5 else scenarios, 
                         save_path=os.path.join(output_dir, "top_scenarios_radar.png"))
    except Exception as e:
        print(f"Error generating radar chart: {e}")
    
    # 5. Analyze strategy performance
    try:
        strategy_stats = analyze_strategy_performance(scenarios, save_path=os.path.join(output_dir, "strategy_performance.png"))
        if strategy_stats is not None:
            strategy_stats.to_csv(os.path.join(output_dir, "strategy_stats.csv"))
    except Exception as e:
        print(f"Error analyzing strategy performance: {e}")
    
    # 6. Generate summary report
    try:
        generate_summary_html(metadata, scenarios, output_dir)
    except Exception as e:
        print(f"Error generating HTML summary: {e}")
    
    print(f"Evolution analysis report generated in {output_dir}")


def generate_summary_html(metadata, scenarios, output_dir):
    """Generate an HTML summary report of the evolutionary process."""
    # Extract key information
    params = metadata.get("parameters", {})
    gen_stats = metadata.get("generation_stats", [])
    best_scenarios = sorted(scenarios, key=lambda x: x.get("selection_score", 0), reverse=True)[:5]
    
    # Check for required files
    has_progress_plot = os.path.exists(os.path.join(output_dir, "evolution_progress.png"))
    has_strategy_plot = os.path.exists(os.path.join(output_dir, "strategy_evolution.png"))
    has_metric_plot = os.path.exists(os.path.join(output_dir, "metric_evolution.png"))
    has_comparison_plot = os.path.exists(os.path.join(output_dir, "top_scenarios_comparison.png"))
    has_radar_plot = os.path.exists(os.path.join(output_dir, "top_scenarios_radar.png"))
    has_performance_plot = os.path.exists(os.path.join(output_dir, "strategy_performance.png"))
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evolutionary Scenario Generation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .chart-container {{ margin: 20px 0; }}
            .scenario {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .scenario h3 {{ margin-top: 0; }}
        </style>
    </head>
    <body>
        <h1>Evolutionary Scenario Generation Report</h1>
        
        <h2>Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Population Size</td><td>{params.get("population_size", "N/A")}</td></tr>
            <tr><td>Generations</td><td>{params.get("generations", "N/A")}</td></tr>
            <tr><td>Evaluation Runs</td><td>{params.get("evaluation_runs", "N/A")}</td></tr>
            <tr><td>Elitism</td><td>{params.get("elitism", "N/A")}</td></tr>
            <tr><td>Crossover Fraction</td><td>{params.get("crossover_fraction", "N/A")}</td></tr>
            <tr><td>Mutation Rate</td><td>{params.get("mutation_rate", "N/A")}</td></tr>
        </table>
    """
    
    # Add evolution progress chart if available
    if has_progress_plot:
        html_content += f"""
        <h2>Evolution Progress</h2>
        <div class="chart-container">
            <img src="evolution_progress.png" alt="Evolution Progress" style="max-width: 100%;">
        </div>
        """
    
    # Add strategy evolution chart if available
    if has_strategy_plot:
        html_content += f"""
        <h2>Strategy Evolution</h2>
        <div class="chart-container">
            <img src="strategy_evolution.png" alt="Strategy Evolution" style="max-width: 100%;">
        </div>
        """
    
    # Add metric evolution chart if available
    if has_metric_plot:
        html_content += f"""
        <h2>Metric Evolution</h2>
        <div class="chart-container">
            <img src="metric_evolution.png" alt="Metric Evolution" style="max-width: 100%;">
        </div>
        """
    
    # Add top scenarios comparison chart if available
    if has_comparison_plot:
        html_content += f"""
        <h2>Top Scenarios Comparison</h2>
        <div class="chart-container">
            <img src="top_scenarios_comparison.png" alt="Top Scenarios Comparison" style="max-width: 100%;">
        </div>
        """
    
    # Add radar chart if available
    if has_radar_plot:
        html_content += f"""
        <div class="chart-container">
            <img src="top_scenarios_radar.png" alt="Top Scenarios Radar Chart" style="max-width: 100%;">
        </div>
        """
    
    # Add strategy performance chart if available
    if has_performance_plot:
        html_content += f"""
        <h2>Strategy Performance</h2>
        <div class="chart-container">
            <img src="strategy_performance.png" alt="Strategy Performance" style="max-width: 100%;">
        </div>
        """
    
    # Add top 5 scenarios details
    html_content += "<h2>Top Scenarios</h2>\n"
    
    # Add top 5 scenarios details only if we have scenarios
    if best_scenarios:
        for i, scenario in enumerate(best_scenarios[:5]):
            if "config" not in scenario:
                continue
                
            config = scenario["config"]
            metrics = scenario.get("metrics", {})
            score = scenario.get("selection_score", 0)
            
            html_content += f"""
            <div class="scenario">
                <h3>{i+1}. {config.get("scenario_name", "Unknown")} (Score: {score:.3f})</h3>
                
                <h4>Configuration:</h4>
                <ul>
                    <li><b>Agents:</b> {config.get("num_agents", "N/A")}</li>
                    <li><b>Rounds:</b> {config.get("num_rounds", "N/A")}</li>
                    <li><b>Network:</b> {config.get("network_type", "N/A")}</li>
                    <li><b>Interaction:</b> {config.get("interaction_mode", "N/A")}</li>
                    <li><b>Strategies:</b> {config.get("agent_strategies", {})}</li>
                </ul>
                
                <h4>Key Metrics:</h4>
                <ul>
            """
            
            # Add top metrics
            top_metrics = [
                "avg_final_coop_rate", "avg_coop_rate_change", "avg_score_variance",
                "avg_coop_volatility", "avg_strategy_adaptation_rate", "avg_pattern_complexity"
            ]
            
            metrics_added = False
            for metric in top_metrics:
                if metric in metrics and not pd.isna(metrics[metric]):
                    display_name = metric.replace("avg_", "").replace("_", " ").title()
                    html_content += f"""
                    <li><span class="metric">{display_name}:</span> {metrics[metric]:.3f}</li>
                    """
                    metrics_added = True
            
            if not metrics_added:
                html_content += "<li>No metrics available</li>"
            
            html_content += """
                </ul>
            </div>
            """
    else:
        html_content += "<p>No scenario data available</p>"
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(os.path.join(output_dir, "evolution_report.html"), "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and visualize evolutionary scenario generation")
    parser.add_argument("--metadata", type=str, default="results/evolved_scenarios/evolution_metadata.json",
                       help="Path to the evolution metadata JSON file")
    parser.add_argument("--scenarios", type=str, default="results/evolved_scenarios/all_evolved_scenarios.json",
                       help="Path to the all evaluated scenarios JSON file")
    parser.add_argument("--output", type=str, default="evolution_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    create_evolution_report(args.metadata, args.scenarios, args.output)
