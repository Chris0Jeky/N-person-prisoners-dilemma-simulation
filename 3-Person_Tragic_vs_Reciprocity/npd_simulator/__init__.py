"""
N-Person Prisoner's Dilemma Simulator

A modular framework for simulating and analyzing N-Person Prisoner's Dilemma games
with support for various agent strategies including reinforcement learning.
"""

__version__ = "2.0.0"
__author__ = "N-Person PD Research Team"

from .core import NPDGame, PairwiseGame
from .agents import Agent, QLearningAgent, TFTAgent, AllCAgent, AllDAgent
from .experiments import ExperimentRunner, ScenarioGenerator
from .analysis import ResultsAnalyzer, Visualizer

__all__ = [
    "NPDGame",
    "PairwiseGame", 
    "Agent",
    "QLearningAgent",
    "TFTAgent",
    "AllCAgent",
    "AllDAgent",
    "ExperimentRunner",
    "ScenarioGenerator",
    "ResultsAnalyzer",
    "Visualizer"
]