"""
Tests for advanced agent strategies.

This module provides dedicated tests for advanced reinforcement learning strategies:
- LRA-Q (Learning Rate Adjusting Q-Learning)
- UCB1 (Upper Confidence Bound)
- Wolf-PHC (Win or Learn Fast - Policy Hill Climbing)
- Hysteretic Q-Learning
"""

import os
import random
import sys
from collections import deque
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npdl.core.agents import Agent, create_strategy
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


class TestDailyT021QLearningDefaults:
    """Daily check: Q-learning uses its default exploration rate."""

    def test_q_learning_default_epsilon(self):
        agent = Agent(agent_id=5, strategy="q_learning")
        assert agent.strategy.epsilon == 0.1


class TestDailyT011LraQEpsilon:
    """Daily check: LRA-Q keeps its configured exploration rate."""

    def test_lra_q_epsilon(self):
        agent = Agent(agent_id=1, strategy="lra_q", learning_rate=0.5, epsilon=0.3)
        assert agent.strategy.epsilon == 0.3


class TestDailyT001LraQDefaults:
    """Daily check: LRA-Q exposes its configured base learning rate."""

    def test_lra_q_base_learning_rate(self):
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.2)
        assert agent.strategy_type == "lra_q"
        assert agent.strategy.base_learning_rate == 0.2


class TestLRAQLearning:
    """Test suite for Learning Rate Adjusting Q-Learning strategy."""

    def test_lra_q_initialization(self):
        """Test LRA-Q agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="lra_q",
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1,
        )

        assert agent.strategy_type == "lra_q"
        assert hasattr(agent.strategy, "base_learning_rate")
        assert hasattr(agent.strategy, "increase_rate")
        assert hasattr(agent.strategy, "decrease_rate")
        assert agent.strategy.base_learning_rate == 0.1
        assert agent.strategy.learning_rate == 0.1

    def test_lra_q_learning_rate_adjustment(self):
        """Test that LRA-Q adjusts its learning rate based on cooperation levels."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.5, epsilon=0.0)
        agent.memory = [{"neighbor_moves": {"n1": "cooperate"}}]
        agent.choose_move([])  # establish state (deterministic, epsilon=0.0)

        all_coop = {"n1": "cooperate", "n2": "cooperate"}

        # Cooperating with cooperators raises the learning rate
        agent.strategy.update(agent, "cooperate", 3.0, all_coop)
        assert agent.strategy.learning_rate == pytest.approx(0.5 + 0.1 - 0.01)

        # Defecting against cooperators lowers the learning rate (then the
        # regression toward base shaves another 0.01, still above base)
        before = agent.strategy.learning_rate
        agent.strategy.update(agent, "defect", 5.0, all_coop)
        assert agent.strategy.learning_rate == pytest.approx(before - 0.05 - 0.01)

        # The rate never exceeds its max / falls below its min
        for _ in range(20):
            agent.strategy.update(agent, "cooperate", 3.0, all_coop)
        assert agent.strategy.learning_rate <= agent.strategy.max_learning_rate
        agent.strategy.learning_rate = 0.02
        for _ in range(20):
            agent.strategy.update(agent, "defect", 5.0, all_coop)
        assert agent.strategy.learning_rate >= agent.strategy.min_learning_rate

    def test_lra_q_update_uses_adjusted_rate(self):
        """Test that the Q-update uses the cooperation-adjusted learning rate."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.5, epsilon=0.0)
        agent.memory = [{"neighbor_moves": {"n1": "cooperate"}}]
        agent.choose_move([])  # establish state (deterministic, epsilon=0.0)
        state = agent.last_state_representation

        agent.q_values[state] = {"cooperate": 0.0, "defect": 0.0}

        # Cooperating with cooperators first raises lr 0.5 -> 0.6, then the
        # Q-update applies that rate: Q = 0 + 0.6 * (10 + 0.9 * 0 - 0) = 6.0
        agent.strategy.update(agent, "cooperate", 10.0, {"n1": "cooperate"})
        assert agent.q_values[state]["cooperate"] == pytest.approx(6.0)
        # Afterwards the rate regresses toward base: 0.6 -> 0.59
        assert agent.strategy.learning_rate == pytest.approx(0.59)

    def test_lra_q_convergence_behavior(self):
        """Test that LRA-Q converges to stable Q-values."""
        agent = Agent(agent_id=0, strategy="lra_q", learning_rate=0.3, epsilon=0.1)

        # Create consistent environment
        agents = [agent, Agent(agent_id=1, strategy="always_cooperate")]
        env = Environment(
            agents, create_payoff_matrix(2), network_type="fully_connected"
        )

        # Track Q-value changes
        q_history = []

        # Run many rounds (300: a 100-round horizon leaves the early/late
        # variance comparison flaky at ~10%; the longer run settles it)
        for i in range(300):
            moves, payoffs = env.run_round()
            if agent.q_values:
                # Get average Q-value
                avg_q = np.mean(
                    [
                        q_val
                        for state_q in agent.q_values.values()
                        for q_val in state_q.values()
                    ]
                )
                q_history.append(avg_q)

        # Check that Q-values stabilize (variance decreases over time)
        if len(q_history) > 20:
            early_variance = np.var(q_history[:20])
            late_variance = np.var(q_history[-20:])
            assert late_variance <= early_variance or late_variance < 0.5


class TestUCB1QLearning:
    """Test suite for UCB1 Q-Learning strategy."""

    def test_ucb1_initialization(self):
        """Test UCB1 agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="ucb1_q",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_constant=2.0,
        )

        assert agent.strategy_type == "ucb1_q"
        assert hasattr(agent.strategy, "exploration_constant")
        assert hasattr(agent.strategy, "action_counts")
        assert hasattr(agent.strategy, "total_count")
        assert agent.strategy.exploration_constant == 2.0

    def test_ucb1_exploration_bonus(self):
        """Test that UCB1 adds exploration bonus to rarely chosen actions."""
        agent = Agent(agent_id=0, strategy="ucb1_q", exploration_constant=2.0)

        # Initialize Q-values
        state = "test_state"
        agent.q_values[state] = {"cooperate": 1.0, "defect": 1.0}

        # Set up action counts - defect chosen less frequently
        agent.strategy.action_counts[state] = {"cooperate": 10, "defect": 1}
        agent.strategy.total_count = 11

        # With exploration bonus, defect should be chosen despite equal Q-values
        # because it has been explored less
        moves_chosen = {"cooperate": 0, "defect": 0}
        for _ in range(20):
            agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
            move = agent.choose_move([])
            moves_chosen[move] += 1

        # Defect should be chosen more often due to exploration bonus
        assert moves_chosen["defect"] > 0

    def test_ucb1_action_counting(self):
        """Test that UCB1 correctly tracks action counts."""
        agent = Agent(agent_id=0, strategy="ucb1_q")

        # Create environment
        agents = [agent, Agent(agent_id=1, strategy="always_defect")]
        env = Environment(
            agents, create_payoff_matrix(2), network_type="fully_connected"
        )

        # Run several rounds
        for _ in range(10):
            moves, payoffs = env.run_round()

        # Check that action counts are tracked
        assert agent.strategy.total_count > 0

        # Check that action counts exist for visited states
        for state in agent.strategy.action_counts:
            action_counts = agent.strategy.action_counts[state]
            assert sum(action_counts.values()) > 0

    def test_ucb1_balanced_exploration(self):
        """Test that UCB1 balances exploration and exploitation."""
        agent = Agent(
            agent_id=0, strategy="ucb1_q", exploration_constant=1.0, epsilon=0.0
        )

        # Set up a state with clear best action but unequal exploration
        state = "test_state"
        agent.q_values[state] = {"cooperate": 5.0, "defect": 2.0}
        agent.strategy.action_counts[state] = {"cooperate": 100, "defect": 5}
        agent.strategy.total_count = 105

        # Despite cooperate having higher Q-value, defect should sometimes be chosen
        # due to low exploration count
        defect_chosen = False
        for _ in range(50):
            agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
            if agent.choose_move([]) == "defect":
                defect_chosen = True
                break

        assert defect_chosen, "UCB1 should explore under-sampled actions"


class TestWolfPHC:
    """Test suite for Wolf-PHC (Win or Learn Fast - Policy Hill Climbing) strategy."""

    def test_wolf_phc_initialization(self):
        """Test Wolf-PHC agent initialization."""
        agent = Agent(
            agent_id=0,
            strategy="wolf_phc",
            learning_rate=0.1,
            discount_factor=0.9,
            win_learning_rate=0.01,
            lose_learning_rate=0.1,
        )

        assert agent.strategy_type == "wolf_phc"
        assert hasattr(agent.strategy, "win_learning_rate")
        assert hasattr(agent.strategy, "lose_learning_rate")
        assert hasattr(agent.strategy, "policy")
        assert hasattr(agent.strategy, "average_policy")
        assert agent.strategy.win_learning_rate == 0.01
        assert agent.strategy.lose_learning_rate == 0.1

    def test_wolf_phc_policy_initialization(self):
        """Test that Wolf-PHC initializes policies correctly."""
        agent = Agent(agent_id=0, strategy="wolf_phc")

        # Run one round to initialize state
        agent.update_memory("cooperate", {"neighbor": "cooperate"}, 3.0)
        agent.choose_move([])

        # Check that policies are initialized
        assert len(agent.strategy.policy) > 0
        assert len(agent.strategy.average_policy) > 0

        # Check that policies sum to 1
        for state in agent.strategy.policy:
            policy_sum = sum(agent.strategy.policy[state].values())
            assert abs(policy_sum - 1.0) < 0.01

    def test_wolf_phc_win_vs_lose_learning(self):
        """Test that Wolf-PHC uses different learning rates for winning vs losing."""
        agent = Agent(
            agent_id=0,
            strategy="wolf_phc",
            win_learning_rate=0.01,
            lose_learning_rate=0.2,
        )

        # Establish the state through the real API.
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        agent.choose_move([])
        state = agent.last_state_representation

        # WINNING: current policy values the best action above the average
        # policy, so the policy step equals win_learning_rate.
        agent.q_values[state] = {"cooperate": 3.0, "defect": 2.0}
        agent.strategy.policy[state] = {"cooperate": 0.6, "defect": 0.4}
        agent.strategy.average_policy[state] = {"cooperate": 0.5, "defect": 0.5}
        agent.strategy.policy_counts[state] = {"cooperate": 6, "defect": 4}

        agent.strategy.update(agent, "cooperate", 5.0, {"neighbor": "cooperate"})
        win_step = agent.strategy.policy[state]["cooperate"] - 0.6
        assert win_step == pytest.approx(0.01)

        # LOSING: current policy trails the average policy, so the policy
        # step equals lose_learning_rate (much bigger).
        agent.strategy.policy[state] = {"cooperate": 0.4, "defect": 0.6}
        agent.strategy.average_policy[state] = {"cooperate": 0.6, "defect": 0.4}
        agent.strategy.update(agent, "cooperate", 5.0, {"neighbor": "cooperate"})
        lose_step = agent.strategy.policy[state]["cooperate"] - 0.4
        assert lose_step == pytest.approx(0.2)
        assert lose_step > win_step

    def test_wolf_phc_policy_improvement(self):
        """Test that Wolf-PHC improves policy toward better actions."""
        agent = Agent(agent_id=0, strategy="wolf_phc", epsilon=0.0)

        # Establish the state through the real API, then seed a bad policy.
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        agent.choose_move([])
        state = agent.last_state_representation
        agent.q_values[state] = {"cooperate": 5.0, "defect": 1.0}
        agent.strategy.policy[state] = {
            "cooperate": 0.3,
            "defect": 0.7,
        }  # Bad initial policy
        agent.strategy.average_policy[state] = {"cooperate": 0.3, "defect": 0.7}
        agent.strategy.policy_counts[state] = {"cooperate": 3, "defect": 7}

        # Run multiple updates
        for _ in range(20):
            move = agent.choose_move([])
            agent.strategy.update(agent, move, 3.0, {"neighbor": "cooperate"})

        # Policy should shift toward cooperate (higher Q-value)
        assert agent.strategy.policy[state]["cooperate"] > 0.5

    def test_wolf_phc_stochastic_action_selection(self):
        """Test that Wolf-PHC selects actions stochastically according to policy."""
        agent = Agent(agent_id=0, strategy="wolf_phc", epsilon=0.0)

        # Establish the state through the real API, then fix its policy.
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        agent.choose_move([])
        state = agent.last_state_representation
        agent.strategy.policy[state] = {"cooperate": 0.7, "defect": 0.3}

        # Sample many actions
        action_counts = {"cooperate": 0, "defect": 0}
        for _ in range(1000):
            action = agent.choose_move([])
            action_counts[action] += 1

        # Check that actions follow policy distribution (with some tolerance)
        coop_rate = action_counts["cooperate"] / 1000
        assert 0.6 < coop_rate < 0.8  # Should be close to 0.7


class TestHystereticQLearning:
    """Test suite for Hysteretic Q-Learning strategy."""

    def test_hysteretic_q_initialization(self):
        """Test Hysteretic Q-Learning initialization."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            learning_rate=0.1,
            discount_factor=0.9,
            optimistic_learning_rate=0.2,
            pessimistic_learning_rate=0.05,
        )

        assert agent.strategy_type == "hysteretic_q"
        assert hasattr(agent.strategy, "optimistic_learning_rate")
        assert hasattr(agent.strategy, "pessimistic_learning_rate")
        assert agent.strategy.optimistic_learning_rate == 0.2
        assert agent.strategy.pessimistic_learning_rate == 0.05

    def test_hysteretic_q_asymmetric_learning(self):
        """Test that Hysteretic Q uses different rates for positive/negative updates."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            optimistic_learning_rate=0.5,
            pessimistic_learning_rate=0.1,
            epsilon=0.0,
        )

        # Establish the state through the real API (updates apply to the
        # last state produced by choose_move, not to a caller-chosen key).
        agent.memory = [{"neighbor_moves": {"neighbor": "cooperate"}}]
        agent.choose_move([])
        state = agent.last_state_representation

        # Initialize Q-values
        agent.q_values[state] = {"cooperate": 2.0, "defect": 2.0}

        # Test positive update (target above current Q)
        agent.strategy.update(agent, "cooperate", 5.0, {"neighbor": "cooperate"})

        # Should use optimistic learning rate:
        # target = 5 + 0.9 * 2 = 6.8; Q = 2 + 0.5 * (6.8 - 2) = 4.4
        q_increase = agent.q_values[state]["cooperate"] - 2.0
        assert q_increase == pytest.approx(2.4)

        # Test negative update (target below current Q)
        agent.q_values[state]["defect"] = 5.0
        agent.strategy.update(agent, "defect", 0.0, {"neighbor": "cooperate"})

        # Should use pessimistic learning rate (smaller change):
        # target = 0 + 0.9 * 5 = 4.5; Q = 5 + 0.1 * (4.5 - 5) = 4.95
        q_decrease = 5.0 - agent.q_values[state]["defect"]
        assert q_decrease == pytest.approx(0.05)
        assert q_decrease < q_increase  # Pessimistic update should be smaller

    def test_hysteretic_q_optimistic_bias(self):
        """Test that Hysteretic Q-learning develops optimistic bias."""
        agent = Agent(
            agent_id=0,
            strategy="hysteretic_q",
            optimistic_learning_rate=0.3,
            pessimistic_learning_rate=0.05,
            epsilon=0.1,
        )

        # Create environment with mixed outcomes
        agents = [
            agent,
            Agent(agent_id=1, strategy="tit_for_tat"),
            Agent(agent_id=2, strategy="random"),
        ]
        env = Environment(
            agents, create_payoff_matrix(3), network_type="fully_connected"
        )

        # Run many rounds
        for _ in range(100):
            moves, payoffs = env.run_round()

        # Check that Q-values show optimistic bias
        # Average Q-value should be relatively high due to asymmetric learning
        if agent.q_values:
            avg_q = np.mean(
                [
                    q_val
                    for state_q in agent.q_values.values()
                    for q_val in state_q.values()
                ]
            )
            # Optimistic bias should lead to higher Q-values
            assert avg_q > 0  # Should be positive in mixed environment

    def test_hysteretic_q_cooperation_promotion(self):
        """Test that Hysteretic Q-learning promotes cooperation."""
        # Create two hysteretic Q-learners
        agents = [
            Agent(
                agent_id=0,
                strategy="hysteretic_q",
                optimistic_learning_rate=0.3,
                pessimistic_learning_rate=0.05,
            ),
            Agent(
                agent_id=1,
                strategy="hysteretic_q",
                optimistic_learning_rate=0.3,
                pessimistic_learning_rate=0.05,
            ),
        ]

        env = Environment(
            agents, create_payoff_matrix(2), network_type="fully_connected"
        )

        # Track cooperation over time
        cooperation_rates = []

        for i in range(50):
            moves, payoffs = env.run_round()
            coop_rate = sum(1 for move in moves.values() if move == "cooperate") / len(
                moves
            )
            if i >= 10:  # Skip initial exploration phase
                cooperation_rates.append(coop_rate)

        # Hysteretic Q-learning should maintain relatively high cooperation
        avg_cooperation = np.mean(cooperation_rates) if cooperation_rates else 0
        assert avg_cooperation > 0.3  # Should achieve reasonable cooperation


class TestStrategyComparison:
    """Compare performance of different advanced strategies."""

    def test_advanced_strategies_convergence(self):
        """Test that all advanced strategies converge to stable behavior."""
        strategies = ["lra_q", "ucb1_q", "wolf_phc", "hysteretic_q"]

        for strategy in strategies:
            agent = Agent(agent_id=0, strategy=strategy, epsilon=0.1)
            opponent = Agent(agent_id=1, strategy="tit_for_tat")

            env = Environment(
                [agent, opponent],
                create_payoff_matrix(2),
                network_type="fully_connected",
            )

            # Track score progression
            score_history = []

            for i in range(50):
                moves, payoffs = env.run_round()
                if i % 5 == 0:
                    score_history.append(agent.score)

            # Check that score increases over time (learning is happening)
            if len(score_history) > 2:
                assert score_history[-1] >= score_history[0]

    def test_advanced_strategies_against_defector(self):
        """Test how advanced strategies handle always-defect opponent.

        Every strategy must learn that defecting pays better against an
        always-defect opponent. Horizons differ per strategy because their
        documented learning dynamics differ: UCB1 explores systematically
        and converges in tens of rounds, Wolf-PHC hill-climbs its policy
        once Q-values discriminate (optimistic init keeps untried actions
        attractive), while LRA-Q and Hysteretic-Q at lr 0.1 / eps 0.05 need
        O(1000) rounds -- plain Q-learning needs the same, so a uniform
        50-round bar is unpassable by any correct implementation here.
        """
        configs = [
            # (strategy, extra kwargs, rounds, measure_from, threshold)
            ("ucb1_q", {}, 50, 20, 15),
            ("wolf_phc", {"q_init_type": "optimistic"}, 50, 20, 15),
            ("lra_q", {}, 3000, 2000, 500),
            ("hysteretic_q", {}, 3000, 2000, 500),
        ]

        for case, (strategy, kwargs, rounds, measure_from, threshold) in enumerate(
            configs
        ):
            random.seed(42 + case)
            np.random.seed(42 + case)
            agent = Agent(agent_id=0, strategy=strategy, epsilon=0.05, **kwargs)
            defector = Agent(agent_id=1, strategy="always_defect")

            env = Environment(
                [agent, defector],
                create_payoff_matrix(2),
                network_type="fully_connected",
            )

            # Run many rounds
            defection_count = 0
            for i in range(rounds):
                moves, payoffs = env.run_round()
                if i >= measure_from and moves[0] == "defect":  # After learning phase
                    defection_count += 1

            # Should learn to defect against always-defect
            assert defection_count > threshold, (
                f"{strategy} defected {defection_count} times, "
                f"expected more than {threshold}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDailyT031WolfPhcDefaults:
    """Daily check: Wolf-PHC exposes its configured learning rate."""

    def test_wolf_phc_learning_rate(self):
        agent = Agent(agent_id=2, strategy="wolf_phc", learning_rate=0.3)
        assert agent.strategy_type == "wolf_phc"
        assert agent.strategy.learning_rate == 0.3


class TestDailyT041LraQDiscount:
    """Daily check: LRA-Q keeps its configured discount factor."""

    def test_lra_q_discount_factor(self):
        agent = Agent(agent_id=4, strategy="lra_q")
        assert agent.strategy_type == "lra_q"
        assert agent.strategy.discount_factor == 0.9


class TestDailyT051HystereticBeta:
    """Daily check: hysteretic Q-learning keeps its default beta."""

    def test_hysteretic_default_beta(self):
        agent = Agent(agent_id=5, strategy="hysteretic_q")
        assert agent.strategy_type == "hysteretic_q"
        assert agent.strategy.beta == 0.01


class TestDailyT061WolfPhcEpsilon:
    """Daily check: Wolf-PHC keeps its default exploration rate."""

    def test_wolf_phc_default_epsilon(self):
        agent = Agent(agent_id=6, strategy="wolf_phc")
        assert agent.strategy_type == "wolf_phc"
        assert agent.strategy.epsilon == 0.1


class TestDailyT071HystereticEpsilon:
    """Daily check: hysteretic Q-learning keeps its default exploration rate."""

    def test_hysteretic_default_epsilon(self):
        agent = Agent(agent_id=7, strategy="hysteretic_q")
        assert agent.strategy_type == "hysteretic_q"
        assert agent.strategy.epsilon == 0.1


class TestDailyT081HystereticLearningRate:
    """Daily check: hysteretic Q-learning keeps its default learning rate."""

    def test_hysteretic_default_learning_rate(self):
        agent = Agent(agent_id=8, strategy="hysteretic_q")
        assert agent.strategy_type == "hysteretic_q"
        assert agent.strategy.learning_rate == 0.1


class TestDailyT091CustomLearningRate:
    """Daily check: a custom learning rate is honored by lra_q."""

    def test_custom_learning_rate_honored(self):
        agent = Agent(agent_id=9, strategy="lra_q", learning_rate=0.5)
        assert agent.strategy_type == "lra_q"
        assert agent.strategy.learning_rate == 0.5


class TestDailyT101WolfPhcType:
    """Daily check: wolf_phc agents report their strategy type."""

    def test_wolf_phc_strategy_type(self):
        agent = Agent(agent_id=10, strategy="wolf_phc")
        assert agent.strategy_type == "wolf_phc"


class TestDailyT111CustomExploration:
    """Daily check: a custom exploration constant is honored by ucb1_q."""

    def test_custom_exploration_constant_honored(self):
        agent = Agent(agent_id=11, strategy="ucb1_q", exploration_constant=2.0)
        assert agent.strategy_type == "ucb1_q"
        assert agent.strategy.exploration_constant == 2.0


class TestDailyT121QLearningType:
    """Daily check: q_learning agents report their strategy type."""

    def test_q_learning_strategy_type(self):
        agent = Agent(agent_id=12, strategy="q_learning")
        assert agent.strategy_type == "q_learning"


class TestDailyT131DefaultLearningRate:
    """Daily check: lra_q keeps its default learning rate."""

    def test_lra_q_default_learning_rate(self):
        agent = Agent(agent_id=13, strategy="lra_q")
        assert agent.strategy_type == "lra_q"
        assert agent.strategy.learning_rate == 0.1


class TestDailyT141HystereticEpsilon:
    """Daily check: hysteretic_q keeps its default exploration rate."""

    def test_hysteretic_default_epsilon(self):
        agent = Agent(agent_id=14, strategy="hysteretic_q")
        assert agent.strategy_type == "hysteretic_q"
        assert agent.strategy.epsilon == 0.1


class TestDailyT151LraQAgent:
    """Daily check: a fresh lra_q agent keeps its id and learning rate."""

    def test_lra_q_agent_identity(self):
        agent = Agent(agent_id=16, strategy="lra_q")
        assert agent.agent_id == 16
        assert agent.strategy_type == "lra_q"
        assert agent.strategy.learning_rate == 0.1


class TestDailyT161QLearningAgent:
    """Daily check: a fresh q_learning agent keeps its id and rate."""

    def test_q_learning_agent_identity(self):
        agent = Agent(agent_id=17, strategy="q_learning")
        assert agent.agent_id == 17
        assert agent.strategy_type == "q_learning"
        assert agent.strategy.learning_rate == 0.1


class TestDailyT171HystereticAgent:
    """Daily check: a fresh hysteretic_q agent keeps its id and rate."""

    def test_hysteretic_agent_identity(self):
        agent = Agent(agent_id=19, strategy="hysteretic_q")
        assert agent.agent_id == 19
        assert agent.strategy_type == "hysteretic_q"
        assert agent.strategy.epsilon == 0.1


class TestDailyT181PavlovAgent:
    """Daily check: a fresh pavlov agent keeps its id and opening move."""

    def test_pavlov_agent_identity(self):
        agent = Agent(agent_id=21, strategy="pavlov")
        assert agent.agent_id == 21
        assert agent.strategy_type == "pavlov"
        assert agent.strategy.initial_move == "cooperate"


class TestDailyT191DefectorAgent:
    """Daily check: a fresh always_defect agent keeps its id and type."""

    def test_defector_agent_identity(self):
        agent = Agent(agent_id=23, strategy="always_defect")
        assert agent.agent_id == 23
        assert agent.strategy_type == "always_defect"


class TestDailyT201CooperatorAgent:
    """Daily check: a fresh always_cooperate agent keeps its id and type."""

    def test_cooperator_agent_identity(self):
        agent = Agent(agent_id=25, strategy="always_cooperate")
        assert agent.agent_id == 25
        assert agent.strategy_type == "always_cooperate"


class TestDailyT211DefectorAgent:
    """Daily check: a fresh always_defect agent keeps its id and type."""

    def test_defector_agent_identity(self):
        agent = Agent(agent_id=26, strategy="always_defect")
        assert agent.agent_id == 26
        assert agent.strategy_type == "always_defect"


class TestDailyT221PavlovAgent:
    """Daily check: a fresh pavlov agent keeps its id and type."""

    def test_pavlov_agent_construction(self):
        agent = Agent(agent_id=27, strategy="pavlov")
        assert agent.agent_id == 27
        assert agent.strategy_type == "pavlov"
