"""Hand-worked characterisations of the five least-covered simulation core modules."""

import random

import networkx as nx
import numpy as np
import pytest

from npdl.core.agents import (
    AdaptiveQLearningStrategy,
    Agent,
    ProportionalTitForTatStrategy,
)
from npdl.core.environment import Environment
from npdl.core.n_person_rl import NPersonHystereticQ, NPersonQLearning
from npdl.core.true_pairwise import (
    OpponentSpecificMemory,
    TruePairwiseEnvironment,
    TruePairwiseQLearning,
    TruePairwiseTFT,
)
from npdl.core.true_pairwise_adapter import TruePairwiseAgentAdapter
from npdl.core.utils import create_payoff_matrix


@pytest.fixture(autouse=True)
def fixed_random_state():
    """Seed both RNGs without changing the surrounding suite's random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(42)
    np.random.seed(42)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


# true_pairwise_adapter: decisions must remain specific to each opponent.
@pytest.mark.parametrize(
    "strategy,opponent_moves,expected",
    [
        ("tit_for_tat", [], "cooperate"),
        ("tit_for_tat", ["defect"], "defect"),
        ("tit_for_tat", ["cooperate"], "cooperate"),
        ("suspicious_tit_for_tat", [], "defect"),
        ("suspicious_tit_for_tat", ["cooperate"], "cooperate"),
        ("generous_tit_for_tat", [], "cooperate"),
        ("generous_tit_for_tat", ["cooperate"], "cooperate"),
        ("tit_for_two_tats", [], "cooperate"),
        ("tit_for_two_tats", ["defect"], "cooperate"),
        ("tit_for_two_tats", ["defect", "defect"], "defect"),
        ("tit_for_two_tats", ["defect", "cooperate"], "cooperate"),
        ("tit_for_two_tats", ["cooperate", "defect"], "cooperate"),
    ],
)
def test_adapter_reciprocal_moves(strategy, opponent_moves, expected):
    adapter = TruePairwiseAgentAdapter(Agent(10, strategy=strategy))
    for move in opponent_moves:
        adapter.update_memory("known", "cooperate", move, 0)
    assert adapter.choose_action_for_opponent("known", 2) == expected
    opening = "defect" if strategy == "suspicious_tit_for_tat" else "cooperate"
    assert adapter.choose_action_for_opponent("unseen", 2) == opening


@pytest.mark.parametrize("draw,expected", [(0.05, "cooperate"), (0.1, "defect")])
def test_adapter_generosity_uses_fixed_ten_percent(monkeypatch, draw, expected):
    adapter = TruePairwiseAgentAdapter(
        Agent(10, strategy="generous_tit_for_tat", generosity=1.0)
    )
    adapter.update_memory("other", "cooperate", "defect", 0)
    monkeypatch.setattr(np.random, "random", lambda: draw)
    # characterises current behaviour; see report
    # The adapter ignores the base agent's generosity and uses a strict < 0.1.
    assert adapter.choose_action_for_opponent("other", 1) == expected


@pytest.mark.parametrize(
    "my_move,reward,expected",
    [
        ("cooperate", 3, "cooperate"),
        ("defect", 3, "defect"),
        ("cooperate", 2.999, "defect"),
        ("defect", 2.999, "cooperate"),
    ],
)
def test_adapter_pavlov_reward_boundary(my_move, reward, expected):
    adapter = TruePairwiseAgentAdapter(Agent(10, strategy="pavlov"))
    assert adapter.choose_action_for_opponent("other", 0) == "cooperate"
    adapter.update_memory("other", my_move, "defect", reward)
    assert adapter.choose_action_for_opponent("other", 1) == expected


def test_adapter_q_learning_uses_opponent_context_and_restores_base_memory():
    base = Agent(10, strategy="q_learning", epsilon=0, state_type="proportion")
    base.update_memory("cooperate", {99: "cooperate"}, 3)
    original_memory = base.memory
    base.q_values = {
        (0.0,): {"cooperate": 0, "defect": 4},
        (1.0,): {"cooperate": 4, "defect": 0},
    }
    adapter = TruePairwiseAgentAdapter(base)
    adapter.update_memory("defector", "cooperate", "defect", 0)
    adapter.update_memory("cooperator", "cooperate", "cooperate", 3)
    assert adapter.choose_action_for_opponent("defector", 1) == "defect"
    assert base.last_state_representation == (0.0,)
    assert base.memory is original_memory
    assert adapter.choose_action_for_opponent("cooperator", 1) == "cooperate"
    assert base.last_state_representation == (1.0,)
    assert base.memory is original_memory
    assert list(base.memory) == [
        {"my_move": "cooperate", "neighbor_moves": {99: "cooperate"}, "reward": 3}
    ]


# true_pairwise: independent Q-tables, state boundaries, and noisy outcomes.
@pytest.mark.parametrize(
    "state_type,moves,expected",
    [
        ("basic", [], "start"),
        ("basic", ["defect"], "last_defect"),
        ("basic", ["cooperate"], "last_cooperate"),
        ("proportion", [], "mixed"),
        ("proportion", ["defect"], "mostly_defect"),
        ("proportion", ["cooperate", "defect"], "mixed"),
        ("proportion", ["cooperate"], "mostly_cooperate"),
        ("reciprocity", [], "somewhat_reciprocal"),
        ("reciprocity", ["cooperate", "defect"], "non_reciprocal"),
        ("reciprocity", ["defect", "cooperate"], "highly_reciprocal"),
        ("reciprocity", ["cooperate", "cooperate", "defect"], "somewhat_reciprocal"),
        ("memory_enhanced", ["defect"], "start"),
        ("memory_enhanced", ["cooperate", "defect", "defect"], "ddc"),
    ],
)
def test_true_pairwise_states_have_known_answers(state_type, moves, expected):
    agent = TruePairwiseQLearning("learner", state_type=state_type)
    for move in moves:
        agent.update_memory("other", "cooperate", move, 0)
    assert agent.get_state_for_opponent("other") == expected


@pytest.mark.parametrize(
    "learning_rate,discount,expected", [(0, 1, 2), (1, 0, 3), (1, 1, 9), (0.25, 0.5, 3)]
)
def test_true_pairwise_q_update_known_answer(learning_rate, discount, expected):
    agent = TruePairwiseQLearning(
        "learner", learning_rate=learning_rate, discount_factor=discount
    )
    agent.q_tables = {
        "other": {
            "old": {"cooperate": 2, "defect": 7},
            "new": {"cooperate": 4, "defect": 6},
        },
        "untouched": {"old": {"cooperate": 99, "defect": 98}},
    }
    # 2 + alpha * (3 + gamma * 6 - 2).
    agent.update_q_value("other", "old", "cooperate", 3, "new")
    assert agent.q_tables["other"]["old"] == {"cooperate": expected, "defect": 7}
    assert agent.q_tables["other"]["new"] == {"cooperate": 4, "defect": 6}
    assert agent.q_tables["untouched"]["old"] == {"cooperate": 99, "defect": 98}


def test_true_pairwise_full_noise_learns_intended_action():
    learner = TruePairwiseQLearning(
        "learner", learning_rate=0.5, discount_factor=0, epsilon=0, state_type="basic"
    )
    env = TruePairwiseEnvironment([learner, TruePairwiseTFT("other")], noise_level=1)
    result = env.play_single_game("learner", "other", 0)
    assert result == {
        "round": 0,
        "agent1_id": "learner",
        "agent2_id": "other",
        "action1": "defect",
        "action2": "defect",
        "intended_action1": "cooperate",
        "intended_action2": "cooperate",
        "payoff1": 1,
        "payoff2": 1,
        "noise_applied": True,
    }
    # characterises current behaviour; see report
    # Reward for the executed D is learned against intended C: 0 + 0.5 * 1.
    assert learner.q_tables["other"]["start"] == {"cooperate": 0.5, "defect": 0}
    assert learner.total_score == 1
    assert learner.get_opponent_memory("other").get_last_move() == "defect"


def test_pairwise_memory_separates_lifetime_and_retained_cooperation():
    memory = OpponentSpecificMemory("other", memory_length=2)
    for move, reward in [("cooperate", 3), ("defect", 0), ("defect", 1)]:
        memory.add_interaction("cooperate", move, reward)
    assert [entry["round"] for entry in memory.interaction_history] == [1, 2]
    assert memory.get_cooperation_rate() == pytest.approx(1 / 3)
    assert memory.get_recent_cooperation_rate() == 0
    assert memory.cumulative_reward == 4


# n_person_rl: exact group-size scaling and strict cooperation thresholds.
@pytest.mark.parametrize(
    "context,action,expected",
    [
        ({}, "cooperate", 3),
        ({1: "cooperate"}, "cooperate", 3.5),
        ({1: "cooperate", 2: "defect"}, "cooperate", 3),
        ({"opponent_coop_proportion": 0.5}, "cooperate", 3),
        ({"opponent_coop_proportion": 0.5001}, "cooperate", 3.5),
        ({"opponent_coop_proportion": 0.7}, "defect", 3),
        ({"opponent_coop_proportion": 0.7001}, "defect", 2.5),
    ],
)
def test_n_person_reward_shaping_boundaries(context, action, expected):
    strategy = NPersonQLearning(N=4)
    assert strategy._shape_reward(3, action, (), context) == expected


@pytest.mark.parametrize("scaled,expected", [(True, 3.125), (False, 4.25)])
def test_n_person_scaled_update_known_answer(scaled, expected):
    strategy = NPersonQLearning(
        N=4, learning_rate=0.5, discount_factor=0.5, scale_learning=scaled
    )
    agent = Agent(0, strategy="always_cooperate")
    agent.last_state_representation = ("initial", 4)
    agent.q_values = {("initial", 4): {"cooperate": 2, "defect": 6}}
    # Shaped reward 3.5, bootstrap 0.5 * 6, alpha 0.25 (scaled) or 0.5.
    strategy.update(agent, "cooperate", 3, {1: "cooperate"})
    assert agent.q_values[("initial", 4)] == {"cooperate": expected, "defect": 6}
    assert strategy.learning_rate == 0.5
    assert strategy.group_baseline == pytest.approx(0.03)


def test_n_person_update_without_selected_state_keeps_scaled_rate():
    strategy = NPersonQLearning(N=4, learning_rate=0.4)
    agent = Agent(0, strategy="always_cooperate")
    strategy.update(agent, "cooperate", 3, {})
    # characterises current behaviour; see report
    # Returning before a selected state exists skips restoration of the rate.
    assert strategy.learning_rate == 0.2
    assert strategy.group_baseline == pytest.approx(0.03)
    assert agent.q_values == {}


@pytest.mark.parametrize(
    "action,reward,context,expected",
    [
        ("cooperate", 4, {"opponent_coop_proportion": 0.7}, 3.8),
        ("cooperate", 4, {"opponent_coop_proportion": 0.6}, 3.6),
        ("cooperate", 0, {}, 1.8),
        ("defect", 0, {}, 1.6),
        ("cooperate", 2, {}, 2),
    ],
)
def test_n_person_hysteretic_update_known_answer(action, reward, context, expected):
    strategy = NPersonHystereticQ(
        N=8, learning_rate=0.8, beta=0.2, discount_factor=0, scale_optimism=False
    )
    agent = Agent(0, strategy="always_cooperate")
    agent.last_state_representation = ("initial", 8)
    agent.q_values = {("initial", 8): {"cooperate": 2, "defect": 2}}
    # Positive cooperative alpha caps at .9; negative cooperative beta halves to .1.
    strategy.update(agent, action, reward, context)
    assert agent.q_values[("initial", 8)][action] == pytest.approx(expected)
    other_action = "defect" if action == "cooperate" else "cooperate"
    assert agent.q_values[("initial", 8)][other_action] == 2


@pytest.mark.parametrize(
    "state_type,group_size,expected",
    [
        ("n_person_rich", 4, (0.5, 0, False, 0.0, False, 4)),
        ("n_person_adaptive", 5, (0.5, 1.0, 0)),
        ("n_person_adaptive", 6, (False, 0, False)),
        ("unrecognised", 4, (0.5, 0, 4)),
    ],
)
def test_n_person_state_representation_at_even_split(state_type, group_size, expected):
    strategy = NPersonQLearning(N=group_size, state_type=state_type)
    agent = Agent(0, strategy="always_cooperate")
    agent.update_memory("cooperate", {1: "cooperate", 2: "defect"}, 0)
    assert strategy._get_current_state(agent) == expected


# agents: memory-dependent states and epsilon decay, including tie behaviour.
@pytest.mark.parametrize(
    "state_type,context,expected",
    [
        ("memory_enhanced", {"opponent_coop_proportion": 0.33}, (1, 1, 0)),
        ("memory_enhanced", {"opponent_coop_proportion": 0.67}, (1, 1, 1)),
        ("memory_enhanced", {"opponent_coop_proportion": 1}, (1, 1, 2)),
        ("memory_enhanced", {1: "defect"}, (1, 1, 0)),
        ("memory_enhanced", {1: "cooperate", 2: "defect"}, (1, 1, 1)),
        ("memory_enhanced", {1: "cooperate"}, (1, 1, 2)),
        ("count", {"opponent_coop_proportion": 0.5}, (2,)),
        ("count", {1: "cooperate"}, (1,)),
        ("threshold", {"opponent_coop_proportion": 0.5}, (False,)),
        ("threshold", {1: "cooperate"}, (True,)),
        ("threshold", {1: "defect"}, (False,)),
        ("threshold", {}, "unknown_context_fallback"),
        ("unrecognised", {"opponent_coop_proportion": 0.5}, ("pairwise_agg", 0.5)),
        ("unrecognised", {1: "cooperate"}, "standard_neighborhood"),
    ],
)
def test_agent_q_states_with_one_memory_entry(state_type, context, expected):
    agent = Agent(0, strategy="q_learning", memory_length=5, state_type=state_type)
    agent.update_memory("cooperate", context, 3)
    assert agent.strategy._get_current_state(agent) == expected


def test_agent_memory_enhanced_state_keeps_two_distinct_own_moves():
    agent = Agent(0, strategy="q_learning", state_type="memory_enhanced")
    agent.update_memory("defect", {1: "defect"}, 1)
    agent.update_memory("cooperate", {1: "cooperate"}, 3)
    assert agent.strategy._get_current_state(agent) == (1, 0, 2)


def test_random_q_initialisation_gives_both_actions_same_seeded_value():
    agent = Agent(0, strategy="q_learning", q_init_type="random", epsilon=0)
    agent.strategy._ensure_state_exists(agent, "initial")
    # First random.uniform(-.1, .1) at seed 42; both actions share that draw.
    assert agent.q_values["initial"] == pytest.approx(
        {"cooperate": 0.02788535969157674, "defect": 0.02788535969157674}
    )
    assert agent.choose_move([]) == "cooperate"


def test_adaptive_epsilon_decay_stops_at_floor_with_exact_q_update():
    strategy = AdaptiveQLearningStrategy(
        learning_rate=1, discount_factor=0, epsilon=0.8, decay_rate=0.5, min_epsilon=0.3
    )
    agent = Agent(0, strategy="always_cooperate")
    agent.last_state_representation = "initial"
    agent.q_values = {"initial": {"cooperate": 0, "defect": 0}}
    for expected_epsilon in [0.4, 0.3, 0.3]:
        strategy.update(agent, "cooperate", 3, {})
        assert strategy.epsilon == expected_epsilon
        assert agent.q_values["initial"] == {"cooperate": 3, "defect": 0}


@pytest.mark.parametrize("draw,expected", [(0.49, "cooperate"), (0.5, "defect")])
def test_proportional_tft_specific_moves_win_over_aggregate(
    monkeypatch, draw, expected
):
    agent = Agent(0, strategy="always_cooperate")
    agent.update_memory(
        "cooperate",
        {
            "opponent_coop_proportion": 1,
            "specific_opponent_moves": {1: "cooperate", 2: "defect"},
        },
        0,
    )
    monkeypatch.setattr(random, "random", lambda: draw)
    assert ProportionalTitForTatStrategy().choose_move(agent, [1, 2]) == expected


# environment: population-sized payoff indexing, isolated nodes, and step order.
@pytest.mark.parametrize("bonus,cooperator_payoff", [(False, 0), (True, 8 / 9)])
def test_sparse_neighborhood_payoffs_use_population_matrix(bonus, cooperator_payoff):
    agents = [Agent(i, strategy="always_cooperate") for i in [10, 20, 30]]
    graph = nx.Graph([(10, 20)])
    graph.add_node(30)
    env = Environment(agents, create_payoff_matrix(3), "custom", {"graph": graph})
    # One cooperative neighbor indexes D[1] = 1 + (5 - 1) / (3 - 1) = 3.
    # Global C rate is 2/3, giving connected C a bonus 2 * (2/3)**2.
    payoffs = env.calculate_payoffs(
        {10: "cooperate", 20: "defect", 30: "cooperate"}, include_global_bonus=bonus
    )
    assert payoffs == pytest.approx({10: cooperator_payoff, 20: 3, 30: 0})
    assert set(graph.edges()) == {(10, 20)}
    assert env.graph is not graph


def test_single_agent_round_has_zero_payoff_even_with_global_bonus():
    agent = Agent(17, strategy="always_cooperate")
    env = Environment([agent], create_payoff_matrix(1))
    assert env.run_round() == ({17: "cooperate"}, {17: 0})
    assert agent.score == 0
    assert list(agent.memory) == [
        {"my_move": "cooperate", "neighbor_moves": {}, "reward": 0}
    ]


@pytest.mark.parametrize(
    "probability,edges",
    [(0, {(10, 20), (20, 30)}), (1, {(10, 20), (10, 30), (20, 30)})],
)
def test_random_network_extremes_relabel_ids_and_connect_components(probability, edges):
    agents = [Agent(i, strategy="always_cooperate") for i in [10, 20, 30]]
    env = Environment(
        agents, create_payoff_matrix(3), "random", {"probability": probability}
    )
    # At probability zero, the three singleton components are joined in a chain.
    assert {tuple(sorted(edge)) for edge in env.graph.edges()} == edges
    assert set(env.network) == {10, 20, 30}
    for agent_id in env.network:
        assert set(env.get_neighbors(agent_id)) == set(env.graph.neighbors(agent_id))


def test_odd_regular_network_reduces_degree_by_one():
    agents = [Agent(i, strategy="always_cooperate") for i in range(5)]
    env = Environment(agents, create_payoff_matrix(5), "regular", {"k": 3})
    assert dict(env.graph.degree()) == dict.fromkeys(range(5), 2)
    assert env.graph.number_of_edges() == 5
    assert nx.is_connected(env.graph)


def test_two_agent_small_world_with_k_one_has_no_edges():
    agents = [Agent(i, strategy="always_cooperate") for i in [10, 20]]
    env = Environment(
        agents, create_payoff_matrix(2), "small_world", {"k": 1, "beta": 0}
    )
    # characterises current behaviour; see report
    # k=1 reaches Watts-Strogatz, which constructs no ring edges for this case.
    assert env.network == {10: [], 20: []}
    assert env.run_round() == ({10: "cooperate", 20: "cooperate"}, {10: 0, 20: 0})


def test_neighborhood_q_update_bootstraps_before_current_memory_is_appended():
    learner = Agent(
        10, strategy="q_learning", learning_rate=0.5, discount_factor=0.5, epsilon=0
    )
    learner.q_values = {"initial": {"cooperate": 4, "defect": 2}}
    defector = Agent(20, strategy="always_defect")
    env = Environment([learner, defector], create_payoff_matrix(2))
    assert env.run_round(use_global_bonus=False) == (
        {10: "cooperate", 20: "defect"},
        {10: 0, 20: 5},
    )
    # characterises current behaviour; see report
    # The bootstrap uses initial Q=4, not a fresh state from this round's D.
    assert learner.q_values == {"initial": {"cooperate": 3, "defect": 2}}
    assert learner.strategy._get_current_state(learner) == (0.2,)
    assert learner.score == 0
    assert defector.score == 5
    assert list(learner.memory) == [
        {"my_move": "cooperate", "neighbor_moves": {20: "defect"}, "reward": 0}
    ]


@pytest.mark.parametrize("probability", [0, 1])
def test_rewiring_two_agent_edge_cannot_find_new_neighbor(probability):
    agents = [
        Agent(10, strategy="always_cooperate"),
        Agent(20, strategy="always_defect"),
    ]
    env = Environment(agents, create_payoff_matrix(2))
    env.run_round(use_global_bonus=False)
    assert env.update_network(rewiring_prob=probability) == 0
    assert set(env.graph.edges()) == {(10, 20)}
    assert env.network == {10: [20], 20: [10]}
