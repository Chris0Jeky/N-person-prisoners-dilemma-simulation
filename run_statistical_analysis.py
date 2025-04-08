import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from npdl.analysis.analysis import compare_scenarios_stats
from npdl.core.logging_utils import setup_logging # For logging configuration