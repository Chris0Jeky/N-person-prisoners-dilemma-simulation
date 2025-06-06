"""
Tests for the logging utilities module.

This module tests the logging functionality including:
- Logging setup and configuration
- Network statistics logging
- Round statistics logging
- Experiment summary logging
- ASCII chart generation
"""

import pytest
import logging
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.core.logging_utils import (
    setup_logging, log_network_stats, log_round_stats,
    log_experiment_summary, generate_ascii_chart
)
from npdl.core.agents import Agent


class TestSetupLogging:
    """Test the logging setup functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        # Reset logging configuration
        logging.getLogger().handlers = []
        
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        log_file = os.path.join(self.temp_dir, "test.log")
        logger = setup_logging(log_file=log_file)
        
        assert isinstance(logger, logging.Logger)
        assert os.path.exists(log_file)
        
        # Test that logging works
        logger.info("Test message")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test message" in content
            
    def test_setup_logging_creates_directory(self):
        """Test that logging creates directory if it doesn't exist."""
        log_file = os.path.join(self.temp_dir, "subdir", "test.log")
        logger = setup_logging(log_file=log_file)
        
        assert os.path.exists(os.path.dirname(log_file))
        assert os.path.exists(log_file)
        
    def test_setup_logging_console_only(self):
        """Test logging with console output only."""
        log_file = os.path.join(self.temp_dir, "test.log")
        logger = setup_logging(log_file=log_file, console=True)
        
        # Check that there are both file and console handlers
        handlers = logger.handlers
        handler_types = [type(h).__name__ for h in handlers]
        assert 'StreamHandler' in handler_types
        assert 'FileHandler' in handler_types
        
    def test_setup_logging_no_console(self):
        """Test logging without console output."""
        log_file = os.path.join(self.temp_dir, "test.log")
        
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        logger = setup_logging(log_file=log_file, console=False)
        
        # Check that there's only a file handler
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
        assert len(handlers) >= 1
        
    def test_setup_logging_levels(self):
        """Test different logging levels."""
        log_file = os.path.join(self.temp_dir, "test.log")
        
        # Test DEBUG level
        logger = setup_logging(log_file=log_file, level=logging.DEBUG)
        assert logger.level == logging.DEBUG or logging.getLogger().level == logging.DEBUG
        
        # Test ERROR level
        logging.getLogger().handlers = []
        logger = setup_logging(log_file=log_file, level=logging.ERROR)
        assert logger.level == logging.ERROR or logging.getLogger().level == logging.ERROR


class TestLogNetworkStats:
    """Test network statistics logging."""
    
    def test_log_network_stats_fully_connected(self, caplog):
        """Test logging stats for a fully connected network."""
        # Create a simple fully connected graph
        graph = nx.complete_graph(5)
        
        with caplog.at_level(logging.INFO):
            log_network_stats(graph, "fully_connected")
            
        assert "Network Type: fully_connected" in caplog.text
        assert "Nodes: 5, Edges: 10" in caplog.text
        assert "Degree - Avg: 4.00" in caplog.text
        assert "Clustering Coefficient:" in caplog.text
        
    def test_log_network_stats_small_world(self, caplog):
        """Test logging stats for a small world network."""
        # Create a small world graph
        graph = nx.watts_strogatz_graph(10, 4, 0.3)
        
        with caplog.at_level(logging.INFO):
            log_network_stats(graph, "small_world")
            
        assert "Network Type: small_world" in caplog.text
        assert "Nodes: 10" in caplog.text
        assert "Degree - Avg:" in caplog.text
        
    def test_log_network_stats_empty_graph(self, caplog):
        """Test logging stats for an empty graph."""
        graph = nx.Graph()
        
        with caplog.at_level(logging.INFO):
            log_network_stats(graph, "empty")
            
        assert "Nodes: 0, Edges: 0" in caplog.text
        assert "Clustering Coefficient: N/A" in caplog.text
        
    def test_log_network_stats_disconnected_graph(self, caplog):
        """Test logging stats for a disconnected graph."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components
        
        with caplog.at_level(logging.INFO):
            log_network_stats(graph, "disconnected")
            
        assert "Network Diameter: inf" in caplog.text
        assert "Average Path Length: inf" in caplog.text
        
    def test_log_network_stats_debug_mode(self, caplog):
        """Test detailed logging in debug mode."""
        graph = nx.path_graph(3)  # Simple path: 0-1-2
        
        with caplog.at_level(logging.DEBUG):
            log_network_stats(graph, "path")
            
        assert "Degree Distribution:" in caplog.text
        assert "Node 0: Degree 1" in caplog.text
        assert "Node 1: Degree 2" in caplog.text
        assert "Node 2: Degree 1" in caplog.text


class TestLogRoundStats:
    """Test round statistics logging."""
    
    def setup_method(self):
        """Set up test agents."""
        self.agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat"),
            Agent(agent_id=3, strategy="q_learning", epsilon=0.1)
        ]
        
        # Set some scores
        self.agents[0].score = 10.0
        self.agents[1].score = 15.0
        self.agents[2].score = 12.0
        self.agents[3].score = 8.0
        
    def test_log_round_stats_basic(self, caplog):
        """Test basic round statistics logging."""
        moves = {0: "cooperate", 1: "defect", 2: "cooperate", 3: "defect"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0, 3: 5.0}
        
        with caplog.at_level(logging.INFO):
            log_round_stats(9, self.agents, moves, payoffs, logging_interval=10)
            
        assert "Round 10: Cooperation Rate: 0.50" in caplog.text
        assert "Avg Score: 11.25" in caplog.text
        
    def test_log_round_stats_skip_interval(self, caplog):
        """Test that logging is skipped when not at interval."""
        moves = {0: "cooperate", 1: "defect", 2: "cooperate", 3: "defect"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0, 3: 5.0}
        
        with caplog.at_level(logging.INFO):
            log_round_stats(5, self.agents, moves, payoffs, logging_interval=10)
            
        # Should not log anything
        assert "Round" not in caplog.text
        
    def test_log_round_stats_with_qlearning(self, caplog):
        """Test logging with Q-learning agents."""
        # Set up Q-values for the Q-learning agent
        self.agents[3].q_values = {
            "state1": {"cooperate": 2.5, "defect": 3.0},
            "state2": {"cooperate": 1.0, "defect": 4.0}
        }
        
        moves = {0: "cooperate", 1: "defect", 2: "cooperate", 3: "defect"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0, 3: 5.0}
        
        with caplog.at_level(logging.INFO):
            log_round_stats(9, self.agents, moves, payoffs, logging_interval=10)
            
        assert "Avg Q(c):" in caplog.text
        assert "Avg Q(d):" in caplog.text
        
    def test_log_round_stats_debug_mode(self, caplog):
        """Test detailed logging in debug mode."""
        moves = {0: "cooperate", 1: "defect", 2: "cooperate", 3: "defect"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0, 3: 5.0}
        
        with caplog.at_level(logging.DEBUG):
            log_round_stats(9, self.agents, moves, payoffs, logging_interval=10)
            
        assert "always_cooperate cooperation rate:" in caplog.text
        assert "always_defect cooperation rate:" in caplog.text
        
    def test_log_round_stats_adaptive_qlearning(self, caplog):
        """Test logging with adaptive Q-learning agents."""
        # Add an adaptive Q-learning agent
        adaptive_agent = Agent(agent_id=4, strategy="q_learning_adaptive", epsilon=0.5)
        self.agents.append(adaptive_agent)
        
        moves = {0: "cooperate", 1: "defect", 2: "cooperate", 3: "defect", 4: "cooperate"}
        payoffs = {0: 3.0, 1: 5.0, 2: 3.0, 3: 5.0, 4: 3.0}
        
        with caplog.at_level(logging.INFO):
            log_round_stats(9, self.agents, moves, payoffs, logging_interval=10)
            
        assert "Avg eps:" in caplog.text


class TestLogExperimentSummary:
    """Test experiment summary logging."""
    
    def setup_method(self):
        """Set up test scenario and agents."""
        self.scenario = {
            'scenario_name': 'Test Scenario',
            'num_rounds': 100,
            'network_type': 'small_world',
            'network_params': {'k': 4, 'beta': 0.3}
        }
        
        self.agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat"),
            Agent(agent_id=3, strategy="q_learning")
        ]
        
        # Set scores
        self.agents[0].score = 250.0
        self.agents[1].score = 400.0
        self.agents[2].score = 300.0
        self.agents[3].score = 280.0
        
        # Set Q-values for Q-learning agent
        self.agents[3].q_values = {
            "state1": {"cooperate": 2.5, "defect": 4.0}
        }
        
        self.round_results = [
            {"moves": {0: "cooperate", 1: "defect", 2: "cooperate", 3: "cooperate"}},
            {"moves": {0: "cooperate", 1: "defect", 2: "defect", 3: "defect"}},
            {"moves": {0: "cooperate", 1: "defect", 2: "defect", 3: "defect"}}
        ]
        
        self.theoretical_scores = {
            'max_cooperation': 1200.0,
            'max_defection': 1500.0,
            'half_half': 1000.0
        }
        
    def test_log_experiment_summary_basic(self, caplog):
        """Test basic experiment summary logging."""
        with caplog.at_level(logging.INFO):
            log_experiment_summary(
                self.scenario, 1, self.agents, self.round_results,
                self.theoretical_scores
            )
            
        assert "Experiment Summary for scenario: Test Scenario" in caplog.text
        assert "Total rounds: 100" in caplog.text
        assert "Final cooperation rate:" in caplog.text
        assert "Run 1 Final Global Score:" in caplog.text
        assert "Theoretical Max Coop Score: 1200.00" in caplog.text
        
    def test_log_experiment_summary_strategy_scores(self, caplog):
        """Test strategy-specific score logging."""
        with caplog.at_level(logging.INFO):
            log_experiment_summary(
                self.scenario, 1, self.agents, self.round_results,
                self.theoretical_scores
            )
            
        assert "Average scores by strategy:" in caplog.text
        assert "always_cooperate:" in caplog.text
        assert "always_defect:" in caplog.text
        assert "tit_for_tat:" in caplog.text
        assert "q_learning:" in caplog.text
        
    def test_log_experiment_summary_cooperation_trend(self, caplog):
        """Test cooperation trend analysis."""
        # Create more round results for trend analysis
        self.round_results = [
            {"moves": {0: "cooperate", 1: "cooperate", 2: "cooperate", 3: "cooperate"}},
            {"moves": {0: "cooperate", 1: "defect", 2: "cooperate", 3: "cooperate"}},
            {"moves": {0: "defect", 1: "defect", 2: "defect", 3: "defect"}}
        ]
        
        with caplog.at_level(logging.INFO):
            log_experiment_summary(
                self.scenario, 1, self.agents, self.round_results,
                self.theoretical_scores
            )
            
        assert "Cooperation Trend:" in caplog.text
        assert "Initial:" in caplog.text
        assert "Middle:" in caplog.text
        assert "Final:" in caplog.text
        
    def test_log_experiment_summary_qlearning_outcome(self, caplog):
        """Test Q-learning outcome analysis."""
        # Set strong preference for defection
        self.agents[3].q_values = {
            "state1": {"cooperate": 1.0, "defect": 5.0},
            "state2": {"cooperate": 0.5, "defect": 6.0}
        }
        
        with caplog.at_level(logging.INFO):
            log_experiment_summary(
                self.scenario, 1, self.agents, self.round_results,
                self.theoretical_scores
            )
            
        assert "Q-learning outcome:" in caplog.text
        
    def test_log_experiment_summary_strategy_ranking(self, caplog):
        """Test strategy ranking by score."""
        with caplog.at_level(logging.INFO):
            log_experiment_summary(
                self.scenario, 1, self.agents, self.round_results,
                self.theoretical_scores
            )
            
        assert "Strategy Ranking (by score):" in caplog.text
        assert "1. always_defect:" in caplog.text  # Highest score


class TestGenerateAsciiChart:
    """Test ASCII chart generation."""
    
    def test_generate_ascii_chart_basic(self):
        """Test basic ASCII chart generation."""
        values = [1, 2, 3, 4, 3, 2, 1]
        chart = generate_ascii_chart(values, title="Test Chart")
        
        assert "Test Chart" in chart
        assert "+" in chart
        assert "-" in chart
        assert "|" in chart
        assert "*" in chart
        
    def test_generate_ascii_chart_empty_data(self):
        """Test chart generation with empty data."""
        chart = generate_ascii_chart([])
        assert chart == "No data to plot"
        
    def test_generate_ascii_chart_single_value(self):
        """Test chart generation with single value."""
        chart = generate_ascii_chart([5.0])
        assert "5.00" in chart or "6.00" in chart  # Min or max value
        
    def test_generate_ascii_chart_constant_values(self):
        """Test chart generation with constant values."""
        values = [3.0, 3.0, 3.0, 3.0]
        chart = generate_ascii_chart(values)
        
        # Should handle the case where all values are the same
        assert "3.00" in chart
        assert "4.00" in chart  # max_val = min_val + 1
        
    def test_generate_ascii_chart_dimensions(self):
        """Test chart with custom dimensions."""
        values = [1, 2, 3, 2, 1]
        chart = generate_ascii_chart(values, width=20, height=5)
        
        lines = chart.strip().split('\n')
        # Title + border + height + border = height + 3 lines minimum
        assert len(lines) >= 8  # 5 + 3
        
    def test_generate_ascii_chart_large_values(self):
        """Test chart with large value ranges."""
        values = [0, 100, 200, 150, 50]
        chart = generate_ascii_chart(values, title="Large Values")
        
        assert "Large Values" in chart
        assert "200.00" in chart  # Max value
        assert "0.00" in chart    # Min value
        
    def test_generate_ascii_chart_error_handling(self):
        """Test error handling in chart generation."""
        # Test with invalid input that might cause errors
        with patch('npdl.core.logging_utils.min', side_effect=ValueError):
            chart = generate_ascii_chart([1, 2, 3])
            assert "Data summary" in chart or "No data" in chart


if __name__ == '__main__':
    pytest.main([__file__, '-v'])