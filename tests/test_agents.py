"""
Tests for the Agent class and its various strategies.
"""
import pytest
import random
from collections import deque

from npdl.core.agents import Agent, create_strategy, Strategy
from npdl.core.agents import (
    TitForTatStrategy, AlwaysCooperateStrategy, AlwaysDefectStrategy,
    RandomStrategy, PavlovStrategy, QLearningStrategy
)


class TestAgentBasics:
    """Test basic agent functionality."""

    def test_agent_initialization(self):
        """Test that agents initialize with correct default values."""
        agent = Agent(agent_id=0, strategy="random")
        assert agent.agent_id == 0
        assert agent.strategy_type == "random"
        assert agent.score == 0
        assert len(agent.memory) == 0
        assert isinstance(agent.strategy, RandomStrategy)

    def test_agent_with_different_strategies(self):
        """Test creating agents with different strategies."""
        agents = [
            Agent(agent_id=0, strategy="always_cooperate"),
            Agent(agent_id=1, strategy="always_defect"),
            Agent(agent_id=2, strategy="tit_for_tat"),
            Agent(agent_id=3, strategy="pavlov"),
            Agent(agent_id=4, strategy="q_learning", learning_rate=0.1)
        ]
        
        assert isinstance(agents[0].strategy, AlwaysCooperateStrategy)
        assert isinstance(agents[1].strategy, AlwaysDefectStrategy)
        assert isinstance(agents[2].strategy, TitForTatStrategy)
        assert isinstance(agents[3].strategy, PavlovStrategy)
        assert isinstance(agents[4].strategy, QLearningStrategy)

    def test_agent_memory_management(self):
        """Test that agent memory is properly managed."""
        agent = Agent(agent_id=0, strategy="random", memory_length=3)
        
        # Add more items than memory length
        agent.update_memory("cooperate", {1: "defect"}, 1.0)
        agent.update_memory("defect", {1: "cooperate"}, 2.0)
        agent.update_memory("cooperate", {1: "cooperate"}, 3.0)
        agent.update_memory("defect", {1: "defect"}, 4.0)
        
        # Verify memory has correct length and contains most recent items
        assert len(agent.memory) == 3
        assert agent.memory[0]["my_move"] == "defect"
        assert agent.memory[0]["reward"] == 2.0
        assert agent.memory[1]["my_move"] == "cooperate"
        assert agent.memory[2]["my_move"] == "defect"
        assert agent.memory[2]["reward"] == 4.0

    def test_agent_reset(self):
        """Test that agent reset properly clears state."""
        agent = Agent(agent_id=0, strategy="q_learning", learning_rate=0.1)
        
        # Set up some state
        agent.score = 100
        agent.update_memory("cooperate", {1: "defect"}, 5.0)
        agent.q_values = {"state1": {"cooperate": 1.0, "defect": 0.5}}
        agent.last_state_representation = "state1"
        
        # Reset the agent
        agent.reset()
        
        # Verify state is reset
        assert agent.score == 0
        assert len(agent.memory) == 0
        assert agent.q_values == {}
        assert agent.last_state_representation is None


class TestAgentStrategies:
    """Test agent strategy behaviors."""

    def test_always_cooperate_strategy(self):
        """Test AlwaysCooperate strategy always returns 'cooperate'."""
        agent = Agent(agent_id=0, strategy="always_cooperate")
        
        for _ in range(5):
            assert agent.choose_move([1, 2]) == "cooperate"
            agent.update_memory("cooperate", {1: "defect", 2: "defect"}, 0)

    def test_always_defect_strategy(self):
        """Test AlwaysDefect strategy always returns 'defect'."""
        agent = Agent(agent_id=0, strategy="always_defect")
        
        for _ in range(5):
            assert agent.choose_move([1, 2]) == "defect"
            agent.update_memory("defect", {1: "cooperate", 2: "cooperate"}, 5)

    def test_tit_for_tat_strategy(self, seed):
        """Test TitForTat strategy correctly responds to neighbor's previous move."""
        agent = Agent(agent_id=0, strategy="tit_for_tat")
        
        # First move should be cooperate
        assert agent.choose_move([1]) == "cooperate"
        
        # Simulate rounds with different neighbor moves
        scenarios = [
            {"neighbor_moves": {1: "cooperate"}, "expected_next": "cooperate"},
            {"neighbor_moves": {1: "defect"}, "expected_next": "defect"},
            {"neighbor_moves": {1: "cooperate"}, "expected_next": "cooperate"}
        ]
        
        for scenario in scenarios:
            agent.update_memory("cooperate", scenario["neighbor_moves"], 1)
            assert agent.choose_move([1]) == scenario["expected_next"]

    def test_pavlov_strategy(self):
        """Test Pavlov strategy implements win-stay, lose-shift correctly."""
        agent = Agent(agent_id=0, strategy="pavlov")
        
        # First move should be cooperate by default
        assert agent.choose_move([1]) == "cooperate"
        
        # Test win-stay (high reward)
        agent.update_memory("cooperate", {1: "cooperate"}, 4.0)  # High reward
        assert agent.choose_move([1]) == "cooperate"  # Stay with cooperate
        
        # Test lose-shift (low reward)
        agent.update_memory("cooperate", {1: "defect"}, 0.0)  # Low reward
        assert agent.choose_move([1]) == "defect"  # Shift to defect
        
        # Another win-stay (high reward)
        agent.update_memory("defect", {1: "defect"}, 3.5)  # High reward
        assert agent.choose_move([1]) == "defect"  # Stay with defect


class TestQLearningAgents:
    """Test Q-learning agent behaviors."""

    def test_q_learning_initialization(self):
        """Test Q-learning agent initializes correctly."""
        agent = Agent(
            agent_id=0,
            strategy="q_learning",
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.2
        )
        
        assert agent.strategy_type == "q_learning"
        assert isinstance(agent.strategy, QLearningStrategy)
        assert agent.strategy.learning_rate == 0.1
        assert agent.strategy.discount_factor == 0.9
        assert agent.strategy.epsilon == 0.2

    def test_q_learning_exploration_exploitation(self, monkeypatch):
        """Test Q-learning agent explores and exploits correctly."""
        agent = Agent(
            agent_id=0,
            strategy="q_learning",
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.5  # High epsilon for testing
        )
        
        # Mock random.random to test exploration vs exploitation
        monkeypatch.setattr(random, "random", lambda: 0.6)  # > epsilon, should exploit
        monkeypatch.setattr(random, "choice", lambda x: "cooperate")  # For exploration
        
        # Setup a state with known Q-values
        agent.last_state_representation = "test_state"
        agent.q_values["test_state"] = {"cooperate": 0.5, "defect": 1.0}
        
        # Should exploit and choose defect (higher Q-value)
        assert agent.choose_move([1]) == "defect"
        
        # Change mock to trigger exploration
        monkeypatch.setattr(random, "random", lambda: 0.4)  # < epsilon, should explore
        
        # Should explore and choose cooperate (from mocked random.choice)
        assert agent.choose_move([1]) == "cooperate"

    def test_q_value_update(self):
        """Test Q-value updates correctly after actions."""
        agent = Agent(
            agent_id=0,
            strategy="q_learning",
            learning_rate=0.5,  # High learning rate for noticeable updates
            discount_factor=0.9,
            epsilon=0
        )
        
        # Set initial state and Q-values
        state = "initial_state"
        agent.last_state_representation = state
        agent.q_values[state] = {"cooperate": 0.0, "defect": 0.0}
        
        # Set up next state to have higher Q-values
        next_state = "next_state"
        agent.q_values[next_state] = {"cooperate": 1.0, "defect": 0.0}
        
        # Update Q-value for cooperate action with reward 5
        agent.strategy._ensure_state_exists(agent, next_state)
        
        # Manually update Q-value using the formula:
        # Q(s,a) = (1-α)Q(s,a) + α(r + γ max(Q(s',a')))
        # With α=0.5, γ=0.9, r=5, max(Q(s',a'))=1.0
        agent.update_q_value("cooperate", 5, {})
        
        # Check math: 0.0 + 0.5 * (5 + 0.9 * 1.0) = 0.0 + 0.5 * 5.9 = 2.95
        expected_q_value = 0.0 + 0.5 * (5 + 0.9 * 1.0)
        assert agent.q_values[state]["cooperate"] == pytest.approx(expected_q_value)
