# Orchestration Log — Research Repo Overhaul

Live swarm ledger (tracker T2). Companion to the live plan tracker:
`.agents/plans/2026-09-26-research-repo-overhaul.md` (T1).
Only these two files track the overhaul.

Conventions:

- One row per lane execution. Verdicts: `PASS`, `FAIL`, `PENDING`.
- QA agent is always a different agent from the implementer.
- Merge requires a recorded QA `PASS`. No direct pushes to `main` except the merge itself.
  Tracker-only commits (`.agents/*`) by the parent are exempt and go straight to `main`.
- Commit series per unit stay separate per the plan's structural commitments.

## Waves

| Wave | Units | Status |
|------|-------|--------|
| A (baseline) | T1, T2, W0 | DONE (PR #43 merged as da30f844) |
| B (core) | W1, then W2 + W3 parallel | W1 DONE (PR #44 merged as 694fd46e); W2+W3 DISPATCHED |
| C (breadth) | W4 + W5 + W6 parallel, then W7 + W8 parallel | PENDING |
| D (ship) | W9, W10, W11, then W12 (optional, needs reconfirmation) | PENDING |

## Ledger

| Unit | Lane | Worktree/Branch | Implementer | PR | QA agent | QA verdict | Merge commit | Notes |
|------|------|-----------------|-------------|----|----------|------------|--------------|-------|
| T1 | parent | main | parent | — | — | — | — | Plan saved 2026-09-26; user said `go` |
| T2 | parent | main | parent | — | — | — | — | This file created on `go` |
| W0 | parent | w0-baseline | parent | #43 | qa-reviewer subagent | PASS | da30f844 | Tag pushed; claude-review red = bot credit balance (infra); Codex clean; Qodana clean; self-approval impossible (same-user auth), QA recorded as comment review |
| W1 | swarm | w1-skeleton | implementer subagent | #44 | qa-reviewer subagent | PASS | 694fd46e | Caches were never tracked (no-op untrack); lock pins installed, floors missing; claude-review red = bot credit again |
| W2 | swarm | w2-entrypoints | PENDING | — | PENDING | PENDING | — | Dispatched with W3 in isolated worktrees |
| W3 | swarm | w3-consolidation | PENDING | — | PENDING | PENDING | — | Highest-risk unit; golden-fixture gated |

## Baseline record (W0)

- Snapshot tag: `pre-overhaul-baseline` (created on `go`, at then-current `HEAD`).
- Working-tree note at tag time: `code_for_website/main_runs/final_agents.py` and
  `npdl/analysis/analysis.py` had pre-existing uncommitted modifications; left untouched.
- Failing-tests baseline: reproduced below in "W0 evidence".
- File inventory: see `W0 evidence`.

## W0 evidence

- Tag `pre-overhaul-baseline` at `21c3174`; 2 pre-existing modified files left untouched.
- Stock `pytest tests/ -q`: collection error (`conftest.py:133` mark-on-fixture vs pytest 9).
- Waived + `--ignore=tests/test_visualization.py`: **36 failed, 440 passed, 1 error**
  (stale `.pytest_cache` said 11; authoritative list in `.agents/W0_BASELINE.md`).
- `tests/test_visualization.py` uncollectable: `plotly` not installed.
- Golden fixtures regenerate byte-identically (`generate.py --check` passes, 3/3 IDENTICAL).
- Full record: `.agents/W0_BASELINE.md`.

## QA findings

(None yet.)
