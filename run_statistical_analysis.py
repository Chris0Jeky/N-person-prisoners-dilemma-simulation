import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from npdl.analysis.analysis import compare_scenarios_stats
from npdl.core.logging_utils import setup_logging # For logging configuration

if __name__ == "__main__":
    # --- Configuration ---
    # List the scenarios (folder names in 'results/') you want to compare
    scenarios_to_compare = [
        "Baseline_QL_BasicState",      # Original Baseline
        "Baseline_QL_PropDiscr",       # Intermediate Baseline
        "HysQ_Opt_PropDiscr_vs_TFT",   # Your best optimized Hysteretic-Q run
        "Wolf_Opt_PropDiscr_vs_TFT"    # Your best optimized Wolf-PHC run
    ]