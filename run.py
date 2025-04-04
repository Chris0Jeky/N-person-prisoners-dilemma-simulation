#!/usr/bin/env python3
"""
Runner script for N-Person Prisoner's Dilemma simulation.

This script provides a simple entry point to the CLI interface.
"""

import sys
import os

# Add the current directory to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from npdl.cli import main

if __name__ == "__main__":
    sys.exit(main())
