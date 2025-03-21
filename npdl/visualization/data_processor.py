"""
Data processing utilities for visualization dashboard.

This module contains functions for processing simulation results
for visualization purposes.
"""

import pandas as pd
from typing import Dict, List


def get_payoffs_by_strategy(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average payoffs by round and strategy."""
    # Group by round and strategy, calculate average payoff
    payoffs = rounds_df.groupby(['round', 'strategy'])['payoff'].mean().reset_index()
    return payoffs


def get_strategy_scores(agents_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final scores by strategy."""
    # Group by strategy, calculate average final score
    scores = agents_df.groupby('strategy')['final_score'].agg(['mean', 'std', 'min', 'max']).reset_index()
    return scores


def prepare_network_data(agents_df: pd.DataFrame, 
                         rounds_df: pd.DataFrame, 
                         round_num: int = None) -> Dict:
    """Prepare data for network visualization."""
    # If round_num is not specified, use the last round
    if round_num is None:
        round_num = rounds_df['round'].max()
    
    # Filter rounds data for the specific round
    round_data = rounds_df[rounds_df['round'] == round_num]
    
    # Prepare nodes data
    nodes = []
    for _, agent in agents_df.iterrows():
        agent_id = agent['agent_id']
        agent_round_data = round_data[round_data['agent_id'] == agent_id]
        
        # Skip if no data for this agent in the selected round
        if agent_round_data.empty:
            continue
            
        move = agent_round_data.iloc[0]['move']
        payoff = agent_round_data.iloc[0]['payoff']
        
        nodes.append({
            'id': agent_id,
            'strategy': agent['strategy'],
            'score': agent['final_score'],
            'move': move,
            'payoff': payoff
        })
    
    # Note: We don't have network structure in results currently
    # This would need to be stored or reconstructed
    
    return {
        'nodes': nodes,
        'edges': [],  # Placeholder
        'round': round_num
    }


def get_strategy_colors() -> Dict[str, str]:
    """Return a mapping of strategy names to colors."""
    return {
        'always_cooperate': '#2ca02c',  # Green
        'always_defect': '#d62728',     # Red
        'tit_for_tat': '#1f77b4',       # Blue
        'generous_tit_for_tat': '#17becf',  # Cyan
        'suspicious_tit_for_tat': '#ff7f0e',  # Orange
        'pavlov': '#9467bd',            # Purple
        'random': '#7f7f7f',            # Gray
        'randomprob': '#8c564b',        # Brown
        'q_learning': '#e377c2',        # Pink
        'q_learning_adaptive': '#bcbd22',  # Olive
    }
