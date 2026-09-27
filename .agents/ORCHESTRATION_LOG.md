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
| C (breadth) | W4 + W5 + W6 serialized, then W7 + W8 | DONE (W8 PR #51 merged as 3c25c676) |
| D (ship) | W9, W10, W11, then W12 (optional, needs reconfirmation) | W9 DONE (#52); chore DONE (#53); W10 DONE (zero findings); W11 DONE (PR #54 merged as 8d5101e0, v1.0.0 tagged+pushed); W12 AWAITING RECONFIRMATION |

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
| W8 | swarm | w8-dashboards | implementer subagent | #51 | qa-reviewer subagent | PASS | 3c25c676 | web-dashboard + static pages → archive (R100, MANIFEST rows); canonical Dash docs; smoke test (documented skip without Dash stack); suite 641+2, links OK |
| W9 | swarm | w9-gates | implementer subagent | #52 | qa-reviewer subagent | PASS | 98ad8cc7 | 4 gates (tests/lint/docs-links/coverage) live-proven green on CI; format commit audited pure; mypy exclusion + scipy + CSV eol justified; report arrived on cancel |
| CHORE | parent | chore/remove-claude-review | parent | #53 | parent (gate-absence proof) | PASS | 484651cb | Removed credit-dead claude-review gate per owner request; PR #53 itself proved absence + all other gates green; claude.yml mention-trigger kept |
| W10 | swarm | main (read-only) | 3 sweep lenses | — | self-verifying lenses | PASS | — | Correctness + quality + surface sweeps all PASS, no findings; no fix PRs required |
| W11 | swarm | w11-release | implementer subagent | #54 | qa-reviewer subagent | PASS | 8d5101e0 | CHANGELOG + MIGRATION + v1.0.0; changelog/migration spot-audited accurate; tag v1.0.0 pushed |
| W12 | — | dedicated branch (planned) | — | — | — | — | — | OPTIONAL: backdated 1–5/day replay Aug 2025→today. Needs explicit reconfirmation (fabricates provenance). Never touches pre-existing ~2449 commits. |

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
