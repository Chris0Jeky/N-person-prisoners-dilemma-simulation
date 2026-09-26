"""W3 golden-equivalence tests: npdl modular agents vs legacy v9.

Compares npdl.core.modular_strategies / npdl.core.modular_agents against
v9 strategies.py / modular_agents.py on identical seeded inputs.
Legacy modules are loaded read-only from the versioned experiment tree;
the legacy top-level imports (final_agents/strategies/config) are satisfied
via sys.modules pre-registration so nothing is copied or mutated.
"""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
V9 = REPO_ROOT / "3-Person_Tragic_vs_Reciprocity" / "final_experimentations" / "v9"


def load_legacy(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


legacy_agents = load_legacy("w3_mod_legacy_final_agents", V9 / "final_agents.py")
legacy_strategies = load_legacy("w3_mod_legacy_strategies", V9 / "strategies.py")
legacy_config = load_legacy("w3_mod_legacy_config", V9 / "config.py")
# Satisfy legacy `from final_agents/strategies/config import ...` statements.
sys.modules.setdefault("final_agents", legacy_agents)
sys.modules.setdefault("strategies", legacy_strategies)
sys.modules.setdefault("config", legacy_config)
legacy_modular = load_legacy("w3_mod_legacy_modular_agents", V9 / "modular_agents.py")

from npdl.core import modular_strategies as ported_strat  # noqa: E402
from npdl.core import modular_agents as ported_mod  # noqa: E402

C, D = 0, 1
SEED = 777
MOVE_SCRIPT = [C, D, D, C, C, C, D, C, D, D]
Q_SCRIPT = [
    {"cooperate": 0.0, "defect": 0.0},
    {"cooperate": 1.5, "defect": 0.5},
    {"cooperate": -1.0, "defect": 2.0},
    {"cooperate": 3.0, "defect": 3.0},
    {"cooperate": 0.2, "defect": -0.7},
]


def test_simple_state_strategy_equivalence():
    old, new = legacy_strategies.SimpleStateStrategy(), ported_strat.SimpleStateStrategy()
    states_old, states_new = [], []
    for i, mv in enumerate(MOVE_SCRIPT * 3):
        opp = i % 3
        states_old.append(old.get_state(None, opp))
        states_new.append(new.get_state(None, opp))
        old.update_history(opp, mv, MOVE_SCRIPT[(i + 1) % len(MOVE_SCRIPT)])
        new.update_history(opp, mv, MOVE_SCRIPT[(i + 1) % len(MOVE_SCRIPT)])
    assert states_old == states_new
    assert old.histories.keys() == new.histories.keys()
    for k in old.histories:
        assert list(old.histories[k]) == list(new.histories[k])
    old.reset()
    new.reset()
    assert old.histories == new.histories == {}


def test_statistical_summary_strategy_equivalence():
    old = legacy_strategies.StatisticalSummaryStrategy()
    new = ported_strat.StatisticalSummaryStrategy()
    assert old.get_state(None, 9) == new.get_state(None, 9) == "Opponent_Disposition_Unknown"
    # Sweep opp 0 from all-defect to mostly-cooperate to cross every band.
    sweep = [D, D, D, D] + [C] * 16
    states_old, states_new = [], []
    for i, mv in enumerate(sweep):
        old.update_stats(0, mv)
        new.update_stats(0, mv)
        states_old.append(old.get_state(None, 0))
        states_new.append(new.get_state(None, 0))
        # Second opponent keeps a mixed history in parallel.
        om = MOVE_SCRIPT[i % len(MOVE_SCRIPT)]
        old.update_stats(1, om)
        new.update_stats(1, om)
        states_old.append(old.get_state(None, 1))
        states_new.append(new.get_state(None, 1))
    assert states_old == states_new
    # All five disposition bands are exercised by the script.
    assert set(states_old) >= {
        "Opponent_Disposition_VeryLow", "Opponent_Disposition_Low",
        "Opponent_Disposition_Medium", "Opponent_Disposition_High",
        "Opponent_Disposition_VeryHigh",
    }
    assert old.opponent_stats == new.opponent_stats
    old.reset()
    new.reset()
    assert old.opponent_stats == new.opponent_stats == {}


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 1.0])
def test_epsilon_greedy_equivalence(epsilon):
    old = legacy_strategies.EpsilonGreedyStrategy(epsilon=epsilon)
    new = ported_strat.EpsilonGreedyStrategy(epsilon=epsilon)
    random.seed(SEED)
    acts_old = [old.choose_action(dict(q)) for q in Q_SCRIPT * 6]
    random.seed(SEED)
    acts_new = [new.choose_action(dict(q)) for q in Q_SCRIPT * 6]
    assert acts_old == acts_new
    old.set_epsilon(0.5)
    new.set_epsilon(0.5)
    random.seed(SEED)
    phase_old = [old.choose_action(dict(q)) for q in Q_SCRIPT]
    random.seed(SEED)
    phase_new = [new.choose_action(dict(q)) for q in Q_SCRIPT]
    assert phase_old == phase_new


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"temperature": 2.0, "min_temperature": 0.01, "decay_rate": 0.98}],
)
def test_softmax_equivalence(kwargs):
    old = legacy_strategies.SoftmaxStrategy(**kwargs)
    new = ported_strat.SoftmaxStrategy(**kwargs)
    random.seed(SEED)
    np.random.seed(SEED)
    acts_old = [old.choose_action(dict(q)) for q in Q_SCRIPT * 6]
    temp_old, steps_old = old.temperature, old.step_count
    random.seed(SEED)
    np.random.seed(SEED)
    acts_new = [new.choose_action(dict(q)) for q in Q_SCRIPT * 6]
    assert acts_old == acts_new
    assert (new.temperature, new.step_count) == (temp_old, steps_old)
    # 30 steps crossed the every-10-steps decay at least twice.
    assert steps_old == 30 and temp_old < old.initial_temperature
    old.reset()
    new.reset()
    assert (old.temperature, old.step_count) == (new.temperature, new.step_count)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        ("StandardQLearning", {}),
        ("StandardQLearning", {"learning_rate": 0.15, "discount_factor": 0.99}),
        ("HystereticQLearning", {}),
        ("HystereticQLearning", {"lr_positive": 0.2, "lr_negative": 0.005, "discount_factor": 0.95}),
    ],
)
def test_learning_strategy_equivalence(cls, kwargs):
    old = getattr(legacy_strategies, cls)(**kwargs)
    new = getattr(ported_strat, cls)(**kwargs)
    cases = [
        (1.0, 3.0, 2.0),   # positive delta
        (5.0, 0.0, 1.0),   # negative delta
        (0.0, 0.0, 0.0),   # zero delta
        (-2.0, 5.0, -1.0),
    ]
    for current_q, reward, next_max_q in cases:
        assert old.update_q_value(current_q, reward, next_max_q) == new.update_q_value(
            current_q, reward, next_max_q
        )


# --- Modular agents + factories ---

OPP_SCRIPT = [C, D, D, C, C, D, C, D]
REWARD_SCRIPT = [3.0, 0.0, 5.0, 1.0, 2.5, 4.0, 1.5, 3.5]
COOP_RATIOS = [None, 0.0, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0]

ADAPTIVE_PARAMS = {
    "initial_lr": 0.1,
    "initial_eps": 0.15,
    "min_lr": 0.03,
    "max_lr": 0.15,
    "min_eps": 0.02,
    "max_eps": 0.15,
    "adaptation_factor": 1.08,
    "reward_window_size": 10,
    "df": 0.95,
    "beta": 0.01,
}


def run_pairwise_script(agent, opponents=(7, 13), rounds=30):
    random.seed(SEED)
    np.random.seed(SEED)
    actions = []
    k = 0
    for r in range(rounds):
        for opp in opponents:
            a = agent.choose_pairwise_action(opp)
            actions.append(a)
            om = OPP_SCRIPT[(r + opp) % len(OPP_SCRIPT)]
            rw = REWARD_SCRIPT[k % len(REWARD_SCRIPT)]
            k += 1
            agent.record_pairwise_outcome(opp, a, om, rw)
    return actions, agent.total_score


def run_neighborhood_script(agent, rounds=30):
    random.seed(SEED)
    np.random.seed(SEED)
    actions = []
    for r in range(rounds):
        ratio_in = COOP_RATIOS[r % len(COOP_RATIOS)]
        a = agent.choose_neighborhood_action(ratio_in)
        actions.append(a)
        ratio_out = COOP_RATIOS[(r + 3) % len(COOP_RATIOS)]
        rw = REWARD_SCRIPT[r % len(REWARD_SCRIPT)]
        agent.record_neighborhood_outcome(ratio_out if ratio_out is not None else 0.5, rw)
    return actions, agent.total_score


def normalize(v):
    from collections import deque

    if isinstance(v, deque):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    return v


def modular_snapshot(agent):
    snap = {"total_score": agent.total_score, "strategy_name": agent.strategy_name}
    for attr in ("q_tables", "neighborhood_q_table", "last_contexts",
                 "last_neighborhood_context", "learning_rates", "epsilons",
                 "reward_windows", "neighborhood_lr", "neighborhood_epsilon",
                 "neighborhood_reward_window"):
        if hasattr(agent, attr):
            snap[attr] = normalize(getattr(agent, attr))
    for sub in ("state_strategy", "action_strategy", "learning_strategy"):
        sub_agent = getattr(agent, sub, None)
        if sub_agent is not None:
            snap[sub] = normalize(
                {k: v for k, v in vars(sub_agent).items()}
            )
    return snap


def test_default_param_dicts_match_config():
    assert ported_mod.MODULAR_BASE_PARAMS == legacy_config.MODULAR_BASE_PARAMS
    assert ported_mod.SOFTMAX_PARAMS == legacy_config.SOFTMAX_PARAMS
    assert ported_mod.HYSTERETIC_PARAMS == legacy_config.HYSTERETIC_PARAMS


PLAIN_FACTORIES = [
    "create_vanilla_qlearner",
    "create_statistical_qlearner",
    "create_softmax_qlearner",
    "create_statistical_softmax_qlearner",
    "create_hysteretic_statistical_qlearner",
]

ADAPTIVE_FACTORIES = [
    "create_adaptive_baseline",
    "create_adaptive_statistical",
    "create_adaptive_softmax",
    "create_adaptive_statistical_softmax",
    "create_adaptive_hysteretic_statistical",
]


@pytest.mark.parametrize("factory", PLAIN_FACTORIES)
@pytest.mark.parametrize("use_default_params", [True, False])
def test_plain_factory_pairwise_equivalence(factory, use_default_params):
    params = None if use_default_params else {"lr": 0.2, "df": 0.9, "eps": 0.3,
                                              "temperature": 1.5, "min_temperature": 0.05,
                                              "decay_rate": 0.99, "beta": 0.02}
    old = getattr(legacy_modular, factory)(1, params=None if params is None else dict(params))
    new = getattr(ported_mod, factory)(1, params=None if params is None else dict(params))
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert modular_snapshot(old) == modular_snapshot(new)


@pytest.mark.parametrize("factory", PLAIN_FACTORIES)
@pytest.mark.parametrize("use_default_params", [True, False])
def test_plain_factory_neighborhood_equivalence(factory, use_default_params):
    params = None if use_default_params else {"lr": 0.2, "df": 0.9, "eps": 0.3,
                                              "temperature": 1.5, "min_temperature": 0.05,
                                              "decay_rate": 0.99, "beta": 0.02}
    old = getattr(legacy_modular, factory)(1, params=None if params is None else dict(params))
    new = getattr(ported_mod, factory)(1, params=None if params is None else dict(params))
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert modular_snapshot(old) == modular_snapshot(new)


@pytest.mark.parametrize("factory", ADAPTIVE_FACTORIES)
def test_adaptive_factory_pairwise_equivalence(factory):
    old = getattr(legacy_modular, factory)(1, dict(ADAPTIVE_PARAMS))
    new = getattr(ported_mod, factory)(1, dict(ADAPTIVE_PARAMS))
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert modular_snapshot(old) == modular_snapshot(new)


@pytest.mark.parametrize("factory", ADAPTIVE_FACTORIES)
def test_adaptive_factory_neighborhood_equivalence(factory):
    old = getattr(legacy_modular, factory)(1, dict(ADAPTIVE_PARAMS))
    new = getattr(ported_mod, factory)(1, dict(ADAPTIVE_PARAMS))
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert modular_snapshot(old) == modular_snapshot(new)


def test_direct_composition_equivalence():
    """Hand-composed agents (not via factories) also match exactly."""
    old = legacy_modular.ModularQLearner(
        1, legacy_strategies.StatisticalSummaryStrategy(),
        legacy_strategies.SoftmaxStrategy(temperature=1.2),
        legacy_strategies.HystereticQLearning(lr_positive=0.15, lr_negative=0.02))
    new = ported_mod.ModularQLearner(
        1, ported_strat.StatisticalSummaryStrategy(),
        ported_strat.SoftmaxStrategy(temperature=1.2),
        ported_strat.HystereticQLearning(lr_positive=0.15, lr_negative=0.02))
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert modular_snapshot(old) == modular_snapshot(new)

    old2 = legacy_modular.ModularAdaptiveQLearner(
        2, legacy_strategies.SimpleStateStrategy(),
        legacy_strategies.EpsilonGreedyStrategy(epsilon=0.2),
        legacy_strategies.StandardQLearning(learning_rate=0.1),
        params=dict(ADAPTIVE_PARAMS))
    new2 = ported_mod.ModularAdaptiveQLearner(
        2, ported_strat.SimpleStateStrategy(),
        ported_strat.EpsilonGreedyStrategy(epsilon=0.2),
        ported_strat.StandardQLearning(learning_rate=0.1),
        params=dict(ADAPTIVE_PARAMS))
    assert run_neighborhood_script(old2) == run_neighborhood_script(new2)
    assert modular_snapshot(old2) == modular_snapshot(new2)
