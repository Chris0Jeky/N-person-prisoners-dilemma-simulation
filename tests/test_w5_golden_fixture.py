"""W5 golden fixture: the one kept result-dump sample.

``tests/fixtures/w5_golden/strategy_stats.csv`` is the smallest committed
result dump (relocated from ``evolution_analysis/`` when the dumps were
uncommitted); it pins the strategy-stats CSV shape for tests.
"""

import csv
import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "w5_golden", "strategy_stats.csv")

EXPECTED_HEADER = [
    "",
    "strategy",
    "count",
    "avg_scenario_score",
    "max_scenario_score",
    "num_scenarios",
]


def test_golden_fixture_shape():
    with open(FIXTURE, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == EXPECTED_HEADER
    assert len(rows) == 6  # header + 5 strategy rows
    strategies = {row[1] for row in rows[1:]}
    assert strategies == {
        "ucb1_q",
        "tit_for_two_tats",
        "always_defect",
        "tit_for_tat",
        "random",
    }
    for row in rows[1:]:
        assert int(row[2]) >= 1  # count
        assert float(row[3]) <= float(row[4])  # avg <= max


def test_golden_fixture_bytes_stable():
    with open(FIXTURE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == "fe2116c761d3b9808c33c45687867a35f55f32e1763691ef8ab317391e4cbad9"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
