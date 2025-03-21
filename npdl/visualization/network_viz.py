"""
Network visualization utilities for N-Person Prisoner's Dilemma simulations.

This module contains functions for visualizing agent networks using Plotly.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional


def generate_network_positions(G: nx.Graph, layout_type: str = "spring") -> Dict[int, List[float]]:
    """Generate node positions for a NetworkX graph using the specified layout algorithm.
    
    Args:
        G: NetworkX graph
        layout_type: Type of layout algorithm to use
        
    Returns:
        Dictionary mapping node IDs to [x, y] positions
    """
    if layout_type == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == "spectral":
        pos = nx.spectral_layout(G)
    elif layout_type == "random":
        pos = nx.random_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Convert positions to dictionary of [x, y] lists
    return {node: [x, y] for node, (x, y) in pos.items()}


def create_network_figure(G: nx.Graph,
                         node_data: Optional[Dict[int, Dict[str, Any]]] = None,
                         color_by: str = "strategy",
                         size_by: str = "score",
                         layout_type: str = "spring",
                         strategy_colors: Optional[Dict[str, str]] = None) -> go.Figure:
    """Create a Plotly figure for visualizing a network of agents.
    
    Args:
        G: NetworkX graph representing agent connections
        node_data: Dictionary mapping node IDs to node attributes
        color_by: Node attribute to use for coloring
        size_by: Node attribute to use for sizing
        layout_type: Type of layout algorithm to use
        strategy_colors: Dictionary mapping strategy names to colors
        
    Returns:
        Plotly figure
    """
    # Generate node positions
    pos = generate_network_positions(G, layout_type)
    
    # Extract node positions
    x_pos = []
    y_pos = []
    node_ids = []
    node_colors = []
    node_sizes = []
    node_labels = []
    
    # Default strategy colors if not provided
    if strategy_colors is None:
        strategy_colors = {
            'always_cooperate': '#2ca02c',
            'always_defect': '#d62728',
            'tit_for_tat': '#1f77b4',
            'generous_tit_for_tat': '#17becf',
            'suspicious_tit_for_tat': '#ff7f0e',
            'pavlov': '#9467bd',
            'random': '#7f7f7f',
            'randomprob': '#8c564b',
            'q_learning': '#e377c2',
            'q_learning_adaptive': '#bcbd22',
        }
    
    default_color = "#7f7f7f"  # Default gray
    default_size = 10
    
    # Process nodes
    for node in G.nodes():
        node_ids.append(node)
        x_pos.append(pos[node][0])
        y_pos.append(pos[node][1])
        
        if node_data and node in node_data:
            data = node_data[node]
            
            # Set node color based on attribute
            if color_by in data and data[color_by] in strategy_colors:
                node_colors.append(strategy_colors[data[color_by]])
            else:
                node_colors.append(default_color)
            
            # Set node size based on attribute
            if size_by in data:
                # Scale size between 5 and 20
                size = 5 + (data[size_by] / 100) * 15  # Assuming score is roughly 0-100
                node_sizes.append(size)
            else:
                node_sizes.append(default_size)
            
            # Create node label
            label = f"ID: {node}<br>"
            for key, value in data.items():
                label += f"{key}: {value}<br>"
            node_labels.append(label)
        else:
            node_colors.append(default_color)
            node_sizes.append(default_size)
            node_labels.append(f"ID: {node}")
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Add line
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='#000')
        ),
        text=node_labels
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title='Agent Network',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                  ))
    
    return fig
