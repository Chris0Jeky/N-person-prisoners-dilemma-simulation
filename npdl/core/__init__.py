"""
Core components for N-Person Prisoner's Dilemma simulation.

This module contains the fundamental classes and functions for
agents, environment, and utility functions.
"""

# Make key components available at package level
from .agents import Agent, Strategy, create_strategy
from .environment import Environment

# W3-consolidated tournament agents (ported exactly from the versioned
# 3-person experiments); imported as submodules to keep this surface additive.
from . import tournament_agents, modular_agents, modular_strategies

# N-person RL strategies available separately to avoid circular imports
__all__ = ['Agent', 'Strategy', 'create_strategy', 'Environment',
           'tournament_agents', 'modular_agents', 'modular_strategies']
