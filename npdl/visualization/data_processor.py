"""
Data processing utilities for visualization dashboard.

This module contains functions for processing simulation results
for visualization purposes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union


def get_payoffs_by_strategy(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average payoffs by round and strategy.
    
    Args:
        rounds_df: DataFrame containing round-by-round data
        
    Returns:
        DataFrame with columns: 'round', 'strategy', and 'payoff'
    """
    # Make a copy to avoid modifying the original
    if rounds_df is None or rounds_df.empty:
        return pd.DataFrame(columns=['round', 'strategy', 'payoff'])
        
    df = rounds_df.copy()
    
    # Ensure required columns exist
    required_columns = ['round', 'strategy', 'payoff']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.warning(f"Missing required columns in rounds_df: {missing}")
        return pd.DataFrame(columns=required_columns)
    
    # Ensure payoff column is numeric
    df['payoff'] = pd.to_numeric(df['payoff'], errors='coerce')
        
    # Group by round and strategy, calculate average payoff
    # Use dropna=True to handle NaN values
    try:
        payoffs = df.groupby(['round', 'strategy'], dropna=True)['payoff'].mean().reset_index()
        
        # Check if result is empty after groupby (can happen with all NaN)
        if payoffs.empty:
            logging.warning("Empty result after groupby operation in get_payoffs_by_strategy")
            return pd.DataFrame(columns=required_columns)
            
        return payoffs
    except Exception as e:
        logging.error(f"Error in get_payoffs_by_strategy: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=required_columns)


def get_strategy_scores(agents_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final scores by strategy.
    
    Args:
        agents_df: DataFrame containing agent-level data
        
    Returns:
        DataFrame with columns: 'strategy', 'mean', 'std', 'min', 'max'
    """
    # Check for empty or None input
    if agents_df is None or agents_df.empty:
        logging.warning("Empty agents_df provided to get_strategy_scores")
        return pd.DataFrame(columns=['strategy', 'mean', 'std', 'min', 'max'])
    
    # Make a copy to avoid modifying the original
    df = agents_df.copy()
    
    # Ensure required columns exist
    required_columns = ['strategy', 'final_score']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.warning(f"Missing required columns in agents_df: {missing}")
        return pd.DataFrame(columns=['strategy', 'mean', 'std', 'min', 'max'])
    
    # Ensure final_score column is numeric
    df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
    
    try:
        # Group by strategy, calculate average final score
        scores = df.groupby('strategy')['final_score'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        # Check if result is empty after groupby (can happen with all NaN)
        if scores.empty:
            logging.warning("Empty result after groupby operation in get_strategy_scores")
            return pd.DataFrame(columns=['strategy', 'mean', 'std', 'min', 'max'])
            
        # Fill any NaN values with 0
        scores = scores.fillna(0)
        return scores
    except Exception as e:
        logging.error(f"Error in get_strategy_scores: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['strategy', 'mean', 'std', 'min', 'max'])


def prepare_network_data(agents_df: pd.DataFrame, 
                         rounds_df: pd.DataFrame, 
                         round_num: int = None) -> Dict:
    """Prepare data for network visualization."""
    # Handle empty dataframes
    if agents_df.empty or rounds_df.empty:
        return {
            'nodes': [],
            'edges': [],
            'round': round_num or 0
        }
    
    try:
        # If round_num is not specified, use the last round
        if round_num is None:
            round_num = rounds_df['round'].max()
        
        # Filter rounds data for the specific round
        round_data = rounds_df[rounds_df['round'] == round_num]
        
        # Prepare nodes data
        nodes = []
        for _, agent in agents_df.iterrows():
            # Safely get agent_id
            if 'agent_id' not in agent:
                continue
                
            agent_id = agent['agent_id']
            agent_round_data = round_data[round_data['agent_id'] == agent_id]
            
            # Skip if no data for this agent in the selected round
            if agent_round_data.empty:
                continue
                
            # Safely get values with defaults
            move = agent_round_data.iloc[0].get('move', 'unknown')
            payoff = agent_round_data.iloc[0].get('payoff', 0)
            strategy = agent.get('strategy', 'unknown')
            final_score = agent.get('final_score', 0)
            
            # Convert payoff to float if possible
            try:
                payoff = float(payoff)
            except (ValueError, TypeError):
                payoff = 0
                
            # Convert final_score to float if possible
            try:
                final_score = float(final_score)
            except (ValueError, TypeError):
                final_score = 0
            
            nodes.append({
                'id': agent_id,
                'strategy': strategy,
                'score': final_score,
                'move': move,
                'payoff': payoff
            })
        
        return {
            'nodes': nodes,
            'edges': [],  # Placeholder
            'round': round_num
        }
    except Exception as e:
        print(f"Error in prepare_network_data: {e}")
        return {
            'nodes': [],
            'edges': [],
            'round': round_num or 0
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
        'lra_q': '#e377c2',             # Same as q_learning (pink)
        'hysteretic_q': '#9edae5',      # Light blue
        'wolf_phc': '#c49c94',          # Light brown
        'ucb1_q': '#dbdb8d',            # Light yellow-green
        'tit_for_two_tats': '#98df8a',  # Light green
    }
