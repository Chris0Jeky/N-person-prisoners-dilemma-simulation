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
| B (core) | W1, then W2 + W3 parallel | DONE (W3 PR #46 merged as b73f10f3; lanes ran serialized — no isolation) |
| C (breadth) | W4 + W5 + W6 serialized, then W7 + W8 | W4 DONE (#47); W5 DONE (#48); W6 DONE (#49); W7 DONE (PR #50 merged as 1685efc8); W8 DISPATCHED |
| D (ship) | W9, W10, W11, then W12 (optional, needs reconfirmation) | PENDING |

## Ledger

| Unit | Lane | Worktree/Branch | Implementer | PR | QA agent | QA verdict | Merge commit | Notes |
|------|------|-----------------|-------------|----|----------|------------|--------------|-------|
| T1 | parent | main | parent | — | — | — | — | Plan saved 2026-09-26; user said `go` |
| T2 | parent | main | parent | — | — | — | — | This file created on `go` |
| W0 | parent | w0-baseline | parent | #43 | qa-reviewer subagent | PASS | da30f844 | Tag pushed; claude-review red = bot credit balance (infra); Codex clean; Qodana clean; self-approval impossible (same-user auth), QA recorded as comment review |
| W1 | swarm | w1-skeleton | implementer subagent | #44 | qa-reviewer subagent | PASS | 694fd46e | Caches were never tracked (no-op untrack); lock pins installed, floors missing; claude-review red = bot credit again |
| W2 | swarm | w2-entrypoints | implementer subagent | #45 | qa-reviewer subagent | PASS | 52c7503d | main.py logic moved verbatim to npdl/simulation/experiments.py; shim warns+delegates; run_npd_simulator.py removed (only served archived code); smoke tests pass; no new failures |
| W3 | swarm | w3-consolidation | implementer subagent | #46 | qa-reviewer subagent | PASS | b73f10f3 | 3 modules + additive export, 98 equivalence tests green, fixtures identical, no new failures; adversarial audit of 2 classes clean; lane went quiet after pushing — parent verified + cancelled, report arrived on cancel |
| W4 | swarm | w4-archive | implementer subagent | #47 | qa-reviewer subagent | PASS | 285f6a72 | 185 files, all R100 renames + manifest/README/1-line loader; v9 + main_runs kept runnable; main_runs user file untouched |
| W5 | swarm | w5-experiments | implementer subagent | #48 | qa-reviewer subagent | PASS | 8c497d98 | Registry + 4 runners + schemas + 40 tests; -86k dump lines; parent added 1 fixup (demo_results untrack) after followup successor lacked tools |
| W6 | swarm | w6-green | implementer subagent | #49 | qa-reviewer subagent | PASS | e127778b | Suite green: 628 passed + 1 documented plotly skip; adversarial audit: zero weakening in 10 test files; fixtures identical |
| W7 | swarm | w7-docs | implementer subagent | #50 | qa-reviewer subagent | PASS | 1685efc8 | README rewrite, link checker + test (zero dead links), docs fold, Paper Resources→paper/ + artifacts untracked; suite 641+1 |
| W8 | swarm | w8-dashboards | implementer subagent | — | PENDING | PENDING | — | Dispatched after W7 merge |

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

## Environment notes

- Worktree isolation unavailable (`workspace_git_probe_failed`): runtime-side probe
  cannot confirm the Git repo, likely the OneDrive path (spaces + `\\?\` prefix).
  Git itself works fine from the shell. Mitigation: lanes run serialized in the
  shared checkout (one writer at a time); W2 then W3, same for Wave C.
