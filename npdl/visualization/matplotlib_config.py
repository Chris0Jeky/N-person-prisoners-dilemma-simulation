"""
Configuration utilities for matplotlib visualization.
Used to control logging levels and other settings.
"""

import logging

def configure_matplotlib():
    """Configure matplotlib to reduce debug noise."""
    # Set matplotlib logging level to INFO to reduce debug messages
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)
    
    # Additional matplotlib libraries that might be noisy
    for lib in ['matplotlib.font_manager', 'matplotlib.backends', 'matplotlib.pyplot']:
        logging.getLogger(lib).setLevel(logging.INFO)
