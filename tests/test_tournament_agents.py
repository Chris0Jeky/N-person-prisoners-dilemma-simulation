"""W3 golden-equivalence tests: npdl.core.tournament_agents vs legacy v9 final_agents.

Each test runs the legacy implementation and the ported implementation on
identical seeded inputs (same scripted opponents, rewards, RNG seeds) and
asserts identical action sequences, scores, and learning state.
Legacy modules are loaded read-only from the versioned experiment tree.
"""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
V9 = REPO_ROOT / "3-Person_Tragic_vs_Reciprocity" / "final_experimentations" / "v9"
V7SUB = REPO_ROOT / "3-Person_Tragic_vs_Reciprocity" / "final_experimentations" / "v7" / "submittable_code"


def load_legacy(mod_name, path):
    """Load a legacy module read-only under a unique name."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


legacy_v9 = load_legacy("w3_legacy_v9_final_agents", V9 / "final_agents.py")
legacy_v2 = load_legacy("w3_legacy_v7sub_v2_final_agents", V7SUB / "final_agents_v2.py")

from npdl.core import tournament_agents as ported  # noqa: E402

C, D = 0, 1  # COOPERATE, DEFECT in both implementations

# Scripted fixtures shared by old/new runs
OPP_SCRIPT = [C, D, D, C, C, D, C, D]  # opponent moves, cycled
REWARD_SCRIPT = [3.0, 0.0, 5.0, 1.0, 2.5, 4.0, 1.5, 3.5]  # rewards, cycled
COOP_RATIOS = [None, 0.0, 0.2, 0.33, 0.5, 0.67, 0.8, 1.0]  # neighborhood ratios, cycled
SEED = 1234


def run_pairwise_script(agent, opponents=(7, 13), rounds=40):
    """Drive pairwise choose/record loop; return (actions, score)."""
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


def run_neighborhood_script(agent, rounds=40):
    """Drive neighborhood choose/record loop; return (actions, score)."""
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


def snapshot(agent):
    """Return comparable learning-state snapshot (plain containers only)."""
    snap = {"total_score": agent.total_score}
    for attr in ("q_tables", "neighborhood_q_table", "n_q_table", "q_table",
                 "histories", "last_contexts", "last_neighborhood_context",
                 "last_context", "learning_rates", "epsilons", "reward_windows",
                 "neighborhood_lr", "neighborhood_epsilon", "lr", "epsilon",
                 "my_history_pairwise", "opp_history_pairwise",
                 "my_history_nperson", "coop_ratio_history",
                 "reward_window", "neighborhood_reward_window",
                 "current_epsilon", "episode_count",
                 "opponent_last_moves", "last_neighborhood_move",
                 "error_rate"):
        if hasattr(agent, attr):
            v = getattr(agent, attr)
            snap[attr] = normalize(v)
    return snap


def normalize(v):
    from collections import deque

    if isinstance(v, deque):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    return v


# --- StaticAgent ---

STATIC_CASES = ["AllC", "AllD", "Random", "TFT", "TFT-E", "Unknown"]


@pytest.mark.parametrize("strategy", STATIC_CASES)
@pytest.mark.parametrize("error_rate", [0.0, 0.15])
def test_static_pairwise_equivalence(strategy, error_rate):
    old = legacy_v9.StaticAgent(1, strategy_name=strategy, error_rate=error_rate)
    new = ported.StaticAgent(1, strategy_name=strategy, error_rate=error_rate)
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert snapshot(old) == snapshot(new)


@pytest.mark.parametrize("strategy", STATIC_CASES)
@pytest.mark.parametrize("error_rate", [0.0, 0.15])
def test_static_neighborhood_equivalence(strategy, error_rate):
    old = legacy_v9.StaticAgent(1, strategy_name=strategy, error_rate=error_rate)
    new = ported.StaticAgent(1, strategy_name=strategy, error_rate=error_rate)
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert snapshot(old) == snapshot(new)


def test_static_tfte_decay_matches_v2():
    """Opt-in error_decay_rate=0.9995 reproduces v7sub final_agents_v2 exactly."""
    old = legacy_v2.StaticAgent(1, strategy_name="TFT-E", error_rate=0.2)
    new = ported.StaticAgent(1, strategy_name="TFT-E", error_rate=0.2,
                             error_decay_rate=0.9995)
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert snapshot(old) == snapshot(new)
    assert old.round_count == new.round_count
    old2 = legacy_v2.StaticAgent(2, strategy_name="TFT-E", error_rate=0.2)
    new2 = ported.StaticAgent(2, strategy_name="TFT-E", error_rate=0.2,
                              error_decay_rate=0.9995)
    assert run_neighborhood_script(old2) == run_neighborhood_script(new2)
    assert snapshot(old2) == snapshot(new2)


def test_static_constants():
    assert (ported.COOPERATE, ported.DEFECT) == (legacy_v9.COOPERATE, legacy_v9.DEFECT) == (0, 1)


# --- PairwiseAdaptiveQLearner ---

ADAPTIVE_PARAMS = {
    "initial_lr": 0.1,
    "initial_eps": 0.15,
    "min_lr": 0.03,
    "max_lr": 0.15,
    "min_eps": 0.02,
    "max_eps": 0.15,
    "adaptation_factor": 1.08,
    "reward_window_size": 10,  # small so adaptation triggers in the script
    "df": 0.95,
}

ADAPTIVE_PARAMS_NO_WINDOW = {"lr": 0.1, "eps": 0.1, "df": 0.9}


@pytest.mark.parametrize("params", [ADAPTIVE_PARAMS, ADAPTIVE_PARAMS_NO_WINDOW, {}])
def test_pairwise_adaptive_pairwise_equivalence(params):
    old = legacy_v9.PairwiseAdaptiveQLearner(1, dict(params))
    new = ported.PairwiseAdaptiveQLearner(1, dict(params))
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert snapshot(old) == snapshot(new)


@pytest.mark.parametrize("params", [ADAPTIVE_PARAMS, ADAPTIVE_PARAMS_NO_WINDOW, {}])
def test_pairwise_adaptive_neighborhood_equivalence(params):
    old = legacy_v9.PairwiseAdaptiveQLearner(1, dict(params))
    new = ported.PairwiseAdaptiveQLearner(1, dict(params))
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert snapshot(old) == snapshot(new)


def test_pairwise_adaptive_adaptation_triggers():
    """Sanity: the scripted run actually exercises parameter adaptation."""
    agent = ported.PairwiseAdaptiveQLearner(1, dict(ADAPTIVE_PARAMS))
    run_pairwise_script(agent)
    assert agent.learning_rates[7] != 0.1 or agent.epsilons[7] != 0.15
    agent2 = ported.PairwiseAdaptiveQLearner(1, dict(ADAPTIVE_PARAMS))
    run_neighborhood_script(agent2)
    assert agent2.neighborhood_lr != 0.1 or agent2.neighborhood_epsilon != 0.15


# --- HystereticQLearner ---

HYSTERETIC_PARAMS = {"lr": 0.12, "beta": 0.002, "df": 0.95, "eps": 0.05}


@pytest.mark.parametrize("params", [HYSTERETIC_PARAMS, {"lr": 0.1}, {}])
def test_hysteretic_pairwise_equivalence(params):
    old = legacy_v9.HystereticQLearner(1, dict(params))
    new = ported.HystereticQLearner(1, dict(params))
    assert run_pairwise_script(old) == run_pairwise_script(new)
    assert snapshot(old) == snapshot(new)


@pytest.mark.parametrize("params", [HYSTERETIC_PARAMS, {"lr": 0.1}, {}])
def test_hysteretic_neighborhood_equivalence(params):
    old = legacy_v9.HystereticQLearner(1, dict(params))
    new = ported.HystereticQLearner(1, dict(params))
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert snapshot(old) == snapshot(new)


# --- NeighborhoodAdaptiveQLearner (neighborhood-only API) ---

@pytest.mark.parametrize("params", [ADAPTIVE_PARAMS, ADAPTIVE_PARAMS_NO_WINDOW, {}])
def test_neighborhood_adaptive_equivalence(params):
    old = legacy_v9.NeighborhoodAdaptiveQLearner(1, dict(params))
    new = ported.NeighborhoodAdaptiveQLearner(1, dict(params))
    assert run_neighborhood_script(old) == run_neighborhood_script(new)
    assert snapshot(old) == snapshot(new)
