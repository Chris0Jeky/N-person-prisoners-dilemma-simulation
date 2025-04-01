"""
Flask+Dash dashboard for visualizing N-Person Prisoner's Dilemma simulations.

This module contains the implementation of the visualization dashboard
using Flask and Dash.
"""

import os
import sys
import json
import dash
import logging
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
import pandas as pd
import subprocess
from pathlib import Path
import importlib.util

from npdl.visualization.data_loader import (
    get_available_scenarios,
    load_scenario_results,
    get_cooperation_rates,
    get_strategy_cooperation_rates,
    load_network_structure,
    get_available_runs
)
from npdl.visualization.data_processor import (
    get_payoffs_by_strategy,
    get_strategy_scores,
    get_strategy_colors,
    prepare_network_data
)
from npdl.visualization.network_viz import create_network_figure
from npdl.visualization.matplotlib_config import configure_matplotlib

# Configure matplotlib to reduce debug output
configure_matplotlib()

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Get strategy colors
strategy_colors = get_strategy_colors()

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("N-Person Prisoner's Dilemma Visualization",
                   className="text-center mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Controls", className="mb-3"),
            html.Label("Select Scenario:"),
            dcc.Dropdown(
                id="scenario-dropdown",
                options=[],  # Will be populated on load
                value=None,
                className="mb-3"
            ),
            html.Label("Select Run:"),
            dcc.Dropdown(
                id="run-dropdown",
                options=[],  # Will be populated based on scenario
                value=None,
                className="mb-3"
            ),
            html.Label("Filter Strategies:"),
            dcc.Checklist(
                id="strategy-checklist",
                options=[],  # Will be populated based on scenario
                value=[],
                className="mb-3"
            ),
            html.Label("Round Range:"),
            dcc.RangeSlider(
                id="round-slider",
                min=0,
                max=100,
                step=1,
                value=[0, 100],
                marks=None,
                className="mb-4"
            ),
            html.Button(
                "Load Data",
                id="load-data-button",
                className="btn btn-primary mb-3"
            )
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Loading(
                        id="loading-cooperation",
                        type="circle",
                        children=[
                            dcc.Graph(id="cooperation-graph")
                        ]
                    )
                ], label="Cooperation Rates"),
                
                dbc.Tab([
                    dcc.Loading(
                        id="loading-payoffs",
                        type="circle",
                        children=[
                            dcc.Graph(id="payoff-graph")
                        ]
                    )
                ], label="Payoffs"),
                
                dbc.Tab([
                    dcc.Loading(
                        id="loading-scores",
                        type="circle",
                        children=[
                            dcc.Graph(id="score-graph")
                        ]
                    )
                ], label="Final Scores"),
                
                dbc.Tab([
                    html.Div([
                        html.Label("Select Round:"),
                        dcc.Slider(
                            id="network-round-slider",
                            min=0,
                            max=100,
                            step=1,
                            value=0,
                            marks=None
                        ),
                    ], className="mt-2 mb-2"),
                    dcc.Loading(
                        id="loading-network",
                        type="circle",
                        children=[
                            dcc.Graph(id="network-graph")
                        ]
                    )
                ], label="Network")
            ])
        ], width=9)
    ])
], fluid=True)


# Populate scenario dropdown on page load
@app.callback(
    Output("scenario-dropdown", "options"),
    Input("scenario-dropdown", "options")  # Dummy input to trigger on load
)
def populate_scenarios(dummy):
    try:
        scenarios = get_available_scenarios()
        return [{"label": s, "value": s} for s in scenarios]
    except Exception as e:
        print(f"Error loading scenarios: {e}")
        return []


# Update run dropdown when scenario is selected
@app.callback(
    [Output("run-dropdown", "options"),
     Output("run-dropdown", "value")],
    [Input("scenario-dropdown", "value")]
)
def update_run_dropdown(scenario):
    if scenario is None:
        return [], None
    
    try:
        runs = get_available_runs(scenario)
        options = [{"label": f"Run {r}", "value": r} for r in runs]
        
        # Add "All Runs" option
        if len(runs) > 1:
            options.insert(0, {"label": "All Runs (Aggregated)", "value": "all"})
            return options, "all"
        
        return options, runs[0] if runs else None
    except Exception as e:
        print(f"Error loading runs: {e}")
        return [], None


# Update controls based on selected scenario
@app.callback(
    [Output("strategy-checklist", "options"),
     Output("strategy-checklist", "value"),
     Output("round-slider", "min"),
     Output("round-slider", "max"),
     Output("round-slider", "value"),
     Output("round-slider", "marks"),
     Output("network-round-slider", "min"),
     Output("network-round-slider", "max"),
     Output("network-round-slider", "value"),
     Output("network-round-slider", "marks")],
    [Input("scenario-dropdown", "value"),
     Input("run-dropdown", "value"),
     Input("load-data-button", "n_clicks")]
)
def update_controls(scenario, run_value, n_clicks):
    if scenario is None:
        return [], [], 0, 100, [0, 100], None, 0, 100, 0, None
    
    try:
        # Convert 'all' to None for aggregated data
        run_number = None if run_value == 'all' else run_value
        
        # Load scenario data
        agents_df, rounds_df = load_scenario_results(scenario, run_number=run_number)
        
        # Get unique strategies
        strategies = agents_df['strategy'].unique().tolist()
        
        # Get min and max rounds
        min_round = rounds_df['round'].min()
        max_round = rounds_df['round'].max()
        
        # Create strategy options with corresponding colors
        strategy_options = []
        for strategy in strategies:
            color = strategy_colors.get(strategy, "#000000")
            strategy_options.append({
                "label": html.Span([
                    html.Div(style={"backgroundColor": color, 
                                   "width": "12px", 
                                   "height": "12px",
                                   "display": "inline-block",
                                   "marginRight": "5px"}),
                    strategy
                ]),
                "value": strategy
            })
        
        # Create round marks (show every 20th round)
        marks = {i: str(i) for i in range(min_round, max_round + 1, 20)}
        
        return (strategy_options, strategies, min_round, max_round, 
                [min_round, max_round], marks, min_round, max_round, 
                min_round, marks)
    except Exception as e:
        print(f"Error updating controls: {e}")
        return [], [], 0, 100, [0, 100], None, 0, 100, 0, None


# Update cooperation rate graph
@app.callback(
    Output("cooperation-graph", "figure"),
    [Input("scenario-dropdown", "value"),
     Input("run-dropdown", "value"),
     Input("strategy-checklist", "value"),
     Input("round-slider", "value"),
     Input("load-data-button", "n_clicks")]
)
def update_cooperation_graph(scenario, run_value, selected_strategies, round_range, n_clicks):
    if scenario is None or selected_strategies is None or len(selected_strategies) == 0:
        # Return empty figure with informational message
        fig = go.Figure()
        fig.add_annotation(
            text="Select a scenario and strategies to view cooperation rates",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Cooperation Rate Over Time",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    try:
        # Convert 'all' to None for aggregated data
        run_number = None if run_value == 'all' else run_value
        
        # Load scenario data
        agents_df, rounds_df = load_scenario_results(scenario, run_number=run_number)
        
        # Filter by round range
        min_round, max_round = round_range
        filtered_rounds = rounds_df[(rounds_df['round'] >= min_round) & 
                                    (rounds_df['round'] <= max_round)]
        
        # Filter by selected strategies
        filtered_rounds = filtered_rounds[filtered_rounds['strategy'].isin(selected_strategies)]
        
        # Calculate cooperation rates by strategy
        coop_rates = get_strategy_cooperation_rates(filtered_rounds)
        
        # Create figure
        fig = px.line(
            coop_rates, 
            x="round", 
            y="cooperation_rate", 
            color="strategy",
            title="Cooperation Rate Over Time by Strategy",
            labels={"cooperation_rate": "Cooperation Rate", "round": "Round"},
            color_discrete_map=strategy_colors
        )
        
        # Add overall cooperation rate
        overall_coop = get_cooperation_rates(filtered_rounds)
        fig.add_trace(
            go.Scatter(
                x=overall_coop["round"],
                y=overall_coop["cooperation_rate"],
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                name="Overall"
            )
        )
        
        # Handle run information in title
        run_info = f" (Run {run_value})" if run_value != 'all' and run_value is not None else " (All Runs)"
        fig.update_layout(
            title=f"Cooperation Rate Over Time by Strategy{run_info}",
            xaxis_title="Round",
            yaxis_title="Cooperation Rate",
            yaxis=dict(range=[0, 1]),
            legend_title="Strategy",
            hovermode="x unified"
        )
        
        return fig
    except Exception as e:
        print(f"Error updating cooperation graph: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Error Loading Cooperation Rate Data",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


# Update payoff graph
@app.callback(
    Output("payoff-graph", "figure"),
    [Input("scenario-dropdown", "value"),
     Input("run-dropdown", "value"),
     Input("strategy-checklist", "value"),
     Input("round-slider", "value"),
     Input("load-data-button", "n_clicks")]
)
def update_payoff_graph(scenario, run_value, selected_strategies, round_range, n_clicks):
    if scenario is None or selected_strategies is None or len(selected_strategies) == 0:
        # Return empty figure with informational message
        fig = go.Figure()
        fig.add_annotation(
            text="Select a scenario and strategies to view payoffs",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Average Payoff Over Time",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    try:
        # Convert 'all' to None for aggregated data
        run_number = None if run_value == 'all' else run_value
        
        # Load scenario data
        agents_df, rounds_df = load_scenario_results(scenario, run_number=run_number)
        
        # Filter by round range
        min_round, max_round = round_range
        filtered_rounds = rounds_df[(rounds_df['round'] >= min_round) & 
                                    (rounds_df['round'] <= max_round)]
        
        # Filter by selected strategies
        filtered_rounds = filtered_rounds[filtered_rounds['strategy'].isin(selected_strategies)]
        
        # Calculate payoffs by strategy
        payoffs = get_payoffs_by_strategy(filtered_rounds)
        
        # Handle run information in title
        run_info = f" (Run {run_value})" if run_value != 'all' and run_value is not None else " (All Runs)"
        
        # Create figure
        fig = px.line(
            payoffs, 
            x="round", 
            y="payoff", 
            color="strategy",
            title=f"Average Payoff Over Time by Strategy{run_info}",
            labels={"payoff": "Average Payoff", "round": "Round"},
            color_discrete_map=strategy_colors
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Average Payoff",
            legend_title="Strategy",
            hovermode="x unified"
        )
        
        return fig
    except Exception as e:
        print(f"Error updating payoff graph: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Error Loading Payoff Data",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


# Update score graph
@app.callback(
    Output("score-graph", "figure"),
    [Input("scenario-dropdown", "value"),
     Input("run-dropdown", "value"),
     Input("strategy-checklist", "value"),
     Input("load-data-button", "n_clicks")]
)
def update_score_graph(scenario, run_value, selected_strategies, n_clicks):
    if scenario is None or selected_strategies is None or len(selected_strategies) == 0:
        # Return empty figure with informational message
        fig = go.Figure()
        fig.add_annotation(
            text="Select a scenario and strategies to view final scores",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Distribution of Final Scores",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    try:
        # Convert 'all' to None for aggregated data
        run_number = None if run_value == 'all' else run_value
        
        # Load scenario data
        agents_df, rounds_df = load_scenario_results(scenario, run_number=run_number)
        
        # Filter by selected strategies
        filtered_agents = agents_df[agents_df['strategy'].isin(selected_strategies)]
        
        # Handle run information in title
        run_info = f" (Run {run_value})" if run_value != 'all' and run_value is not None else " (All Runs)"
        
        # Create figure
        fig = px.box(
            filtered_agents, 
            x="strategy", 
            y="final_score",
            color="strategy",
            title=f"Distribution of Final Scores by Strategy{run_info}",
            labels={"final_score": "Final Score", "strategy": "Strategy"},
            color_discrete_map=strategy_colors
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Final Score",
            showlegend=False,
            hovermode="closest"
        )
        
        return fig
    except Exception as e:
        print(f"Error updating score graph: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Error Loading Score Data",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


# Update network graph
@app.callback(
    Output("network-graph", "figure"),
    [Input("scenario-dropdown", "value"),
     Input("run-dropdown", "value"),
     Input("network-round-slider", "value"),
     Input("load-data-button", "n_clicks")]
)
def update_network_graph(scenario, run_value, round_num, n_clicks):
    if scenario is None:
        # Return empty figure with informational message
        fig = go.Figure()
        fig.add_annotation(
            text="Select a scenario to view network structure",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Network Visualization",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    try:
        # Use run 0 if 'all' is selected for network visualization
        run_number = 0 if run_value == 'all' else (run_value or 0)
        
        # Load network structure
        G, network_data = load_network_structure(scenario, run_number=run_number)
        
        # If network data is not available
        if not network_data:
            fig = go.Figure()
            fig.add_annotation(
                text="Network data not found for this scenario.<br>Run new simulations with network export enabled.",
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Network Data Not Available",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return fig
        
        # Load agent and round data for additional information
        agents_df, rounds_df = load_scenario_results(scenario, run_number=run_number)
        
        # Filter rounds data for the specific round
        if round_num is not None:
            round_data = rounds_df[rounds_df['round'] == round_num]
        else:
            # Use the last round if round_num is not specified
            round_num = rounds_df['round'].max()
            round_data = rounds_df[rounds_df['round'] == round_num]
        
        # Prepare node data for visualization
        node_data = {}
        
        for _, agent_row in agents_df.iterrows():
            agent_id = agent_row['agent_id']
            
            # Get the agent's data for the current round
            agent_round_data = round_data[round_data['agent_id'] == agent_id]
            
            if not agent_round_data.empty:
                move = agent_round_data.iloc[0]['move']
                payoff = agent_round_data.iloc[0]['payoff']
                
                node_data[agent_id] = {
                    'strategy': agent_row['strategy'],
                    'score': agent_row['final_score'],
                    'move': move,
                    'payoff': payoff
                }
        
        # Handle run information in title
        run_info = f" (Run {run_value})" if run_value != 'all' and run_value is not None else " (Run 0)"
        
        # Create network visualization
        network_title = f"Network Visualization - Round {round_num}{run_info}"
        fig = create_network_figure(
            G, 
            node_data=node_data,
            color_by='strategy',
            size_by='payoff',
            layout_type='spring',
            strategy_colors=get_strategy_colors()
        )
        
        # Update layout
        fig.update_layout(
            title=network_title,
            showlegend=False,
            height=600,
        )
        
        # Add network type info
        network_type = network_data.get('network_type', 'Unknown')
        network_params = network_data.get('network_params', {})
        params_str = ', '.join([f"{k}={v}" for k, v in network_params.items()])
        
        fig.add_annotation(
            text=f"Network Type: {network_type.title()} ({params_str})",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating network graph: {e}")
        
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error visualizing network: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Network Visualization Error",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig


def run_dashboard(debug: bool = True, port: int = 8050) -> None:
    """Run the Dash visualization dashboard.
    
    Args:
        debug: Whether to run the app in debug mode
        port: The port to run the dashboard on
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    
    try:
        # Use app.run() for newer Dash versions
        app.run(debug=debug, port=port)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        logger.info("Make sure all required dependencies are installed:")
        logger.info("  pip install dash dash-bootstrap-components plotly flask")


if __name__ == "__main__":
    run_dashboard()
