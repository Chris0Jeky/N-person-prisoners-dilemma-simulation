"""
Agent implementations for N-Person Prisoner's Dilemma
"""

from .base.agent import Agent, NPDAgent, PairwiseAgent
from .strategies.tft import TFTAgent, pTFTAgent, pTFTThresholdAgent
from .strategies.always import AllCAgent, AllDAgent
from .strategies.random import RandomAgent
from .rl.qlearning import QLearningAgent
from .rl.enhanced_qlearning import EnhancedQLearningAgent

__all__ = [
    "Agent",
    "NPDAgent", 
    "PairwiseAgent",
    "TFTAgent",
    "pTFTAgent",
    "pTFTThresholdAgent",
    "AllCAgent",
    "AllDAgent",
    "RandomAgent",
    "QLearningAgent",
    "EnhancedQLearningAgent"
]