"""
Logging utilities for NPD Simulator
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Specialized logger for experiment tracking.
    """
    
    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main logger
        log_file = self.output_dir / f"{experiment_name}.log"
        self.logger = setup_logger(experiment_name, str(log_file))
        
        # Create metrics logger
        metrics_file = self.output_dir / f"{experiment_name}_metrics.csv"
        self.metrics_file = open(metrics_file, 'w')
        self.metrics_writer = None
        
    def log_start(self, config: dict):
        """Log experiment start."""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")
        
    def log_round(self, round_num: int, metrics: dict):
        """Log round metrics."""
        if self.metrics_writer is None:
            import csv
            fieldnames = ['round'] + list(metrics.keys())
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=fieldnames)
            self.metrics_writer.writeheader()
        
        row = {'round': round_num}
        row.update(metrics)
        self.metrics_writer.writerow(row)
        
    def log_end(self, results: dict):
        """Log experiment end."""
        self.logger.info(f"Experiment completed: {self.experiment_name}")
        self.logger.info(f"Final results: {results}")
        
    def close(self):
        """Close log files."""
        if hasattr(self, 'metrics_file'):
            self.metrics_file.close()