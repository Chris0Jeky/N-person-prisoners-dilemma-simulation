"""
Simulation module for N-Person Prisoner's Dilemma.

This module provides functions for running simulations of the N-Person Prisoner's Dilemma
game with various agent strategies, network structures, and parameters.
"""

# Only import the name without loading the full module
__all__ = ['run_simulation']

# Use import in a function to avoid circular imports
def run_simulation(*args, **kwargs):
    """Run N-person IPD simulation."""
    from npdl.simulation.runner import run_simulation as _run_simulation
    return _run_simulation(*args, **kwargs)
