#!/usr/bin/env python3
"""W0 golden-fixture generator.

Regenerates byte-identical JSON fixtures from current `npdl` behavior.
Usage: python tests/fixtures/w0_golden/generate.py [--check]

With --check, regenerates to a temp dir and diffs against committed fixtures.
Seed: 42 everywhere. No external services, no wall-clock input.
"""

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from npdl.core.agents import Agent  # noqa: E402
from npdl.core.environment import Environment  # noqa: E402
from npdl.core.utils import create_payoff_matrix  # noqa: E402

SEED = 42
OUT_DIR = Path(__file__).resolve().parent


def norm(value):
    if isinstance(value, dict):
        return {
            str(k): norm(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


def write(name, payload):
    (OUT_DIR / name).write_text(
        json.dumps(norm(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def payoff_fixtures():
    return {
        "linear_N4_default": create_payoff_matrix(N=4, payoff_type="linear"),
        "linear_N6_default": create_payoff_matrix(N=6, payoff_type="linear"),
        "threshold_N5_default": create_payoff_matrix(N=5, payoff_type="threshold"),
        "exponential_N5_default": create_payoff_matrix(N=5, payoff_type="exponential"),
    }


# Scripted neighbor behavior across 6 rounds (3 neighbors).
PATTERNS = [
    {1: "cooperate", 2: "cooperate", 3: "cooperate"},
    {1: "cooperate", 2: "cooperate", 3: "cooperate"},
    {1: "defect", 2: "defect", 3: "defect"},
    {1: "cooperate", 2: "defect", 3: "cooperate"},
    {1: "defect", 2: "cooperate", 3: "defect"},
    {1: "cooperate", 2: "cooperate", 3: "cooperate"},
]

STRATEGIES = [
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "generous_tit_for_tat",
    "suspicious_tit_for_tat",
    "tit_for_two_tats",
    "pavlov",
    "randomprob",
    "q_learning",
]


def strategy_fixtures():
    out = {}
    for name in STRATEGIES:
        random.seed(SEED)
        np.random.seed(SEED)
        agent = Agent(agent_id=0, strategy=name)
        moves = []
        for step, pattern in enumerate(PATTERNS):
            move = agent.choose_move([1, 2, 3])
            moves.append(move)
            agent.update_memory(
                move, dict(pattern), reward=1.0 if move == "cooperate" else 2.0
            )
        out[name] = moves
    return {"seed": SEED, "patterns": PATTERNS, "moves": out}


def e2e_fixture():
    random.seed(SEED)
    np.random.seed(SEED)
    mix = [
        ("always_cooperate", 0),
        ("always_cooperate", 1),
        ("always_defect", 2),
        ("always_defect", 3),
        ("tit_for_tat", 4),
        ("q_learning", 5),
    ]
    agents = [Agent(agent_id=i, strategy=s) for s, i in mix]
    matrix = create_payoff_matrix(N=6, payoff_type="linear")
    env = Environment(agents, matrix, "fully_connected", {})
    rounds = []
    for _ in range(20):
        moves, payoffs = env.run_round(use_global_bonus=False, rewiring_prob=0.0)
        coop = sum(1 for m in moves.values() if m == "cooperate") / len(moves)
        rounds.append({"moves": moves, "payoffs": payoffs, "cooperation_rate": coop})
    return {
        "seed": SEED,
        "rounds": 20,
        "use_global_bonus": False,
        "per_round": rounds,
        "final_scores": {a.agent_id: a.score for a in agents},
    }


def generate(out_dir: Path):
    global OUT_DIR
    OUT_DIR = out_dir
    write("payoff_matrices.json", payoff_fixtures())
    write("strategy_moves.json", strategy_fixtures())
    write("e2e_small_run.json", e2e_fixture())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    if "--check" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            generate(Path(tmp))
            ok = True
            for name in (
                "payoff_matrices.json",
                "strategy_moves.json",
                "e2e_small_run.json",
            ):
                a = sha256_of(Path(tmp) / name)
                b = sha256_of(OUT_DIR / name)
                status = "IDENTICAL" if a == b else "DIFFERS"
                if a != b:
                    ok = False
                print(f"{name}: {status} ({b[:12]})")
            sys.exit(0 if ok else 1)
    generate(OUT_DIR)
    print("wrote fixtures to", OUT_DIR)
