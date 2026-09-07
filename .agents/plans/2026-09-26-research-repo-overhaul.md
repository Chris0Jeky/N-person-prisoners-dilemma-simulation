## Goal

Turn this repository into a clean, reproducible research project: one canonical package and CLI, one experiment system with seeded runs and artifact manifests, consolidated docs, a green test suite with automated gates, and legacy variants archived with provenance instead of scattered as live code. Delivery runs as a swarm of subagents coordinated through two live tracker files, with independent retroactive QA on every unit before merge.

## Success Criteria

- One canonical import surface (`npdl.*`) and one CLI entry (`run.py`); every other entry point (`main.py`, `run_npd_simulator.py`) is a thin shim or removed, with a migration note.
- Versioned duplicates (`3-Person_Tragic_vs_Reciprocity/final_experimentations/v1..v9`, `legacy*`, `code_for_website/*`, `simple_models/*`) reduced to a single archived snapshot plus a manifest; no duplicated agent implementations outside the archive.
- All scenario/config files (`scenarios/*.json`, `configs/*.json`) covered by checked-in schema files and validation tests; every experiment run records seed, config hash, command line, and an artifact manifest.
- Test suite green from today's baseline (11 failing entries: TFT ecosystem behavior, logging utilities, TFT agent defaults, visualization collection); generated artifacts (CSV/PNG, LaTeX build outputs, caches) untracked and ignored.
- Docs: one README path from root, one index under `docs/` with no dead links, paper sources separated from build artifacts.
- Automated gates on every PR: full test suite, format/type checks per repo config, docs-link check; coverage reported.
- Both live trackers current at all times; every merged unit carries a recorded independent QA verdict (correctness + quality).
- (Separate optional phase) Work published as small per-unit commits; backdated dissemination only if reconfirmed at that phase.

## Approach

Keep the good core, archive the sprawl, and make experiments reproducible. `npdl/` (core, simulation, analysis, visualization, interactive, CLI) stays canonical because the test suite, `run.py`, and README already target it. Everything else is either consolidated into it, converted to a declared experiment that consumes it, or moved to a read-only archive with a manifest. No behavior change without a golden-output comparison against the pre-change implementation.

Key decisions:

- Canonical package: `npdl/`; the versioned `final_experimentations/v1..v9` chain, `legacy*`, and `code_for_website/*` copies become one tagged pre-overhaul snapshot plus an `archive/` tree with an ownership manifest, not live imports. Rationale: 391 files under `3-Person_Tragic_vs_Reciprocity/` plus 40 under `code_for_website/` duplicate agent logic; deleting silently would destroy research provenance, so archive-first.
- Canonical entry: `run.py` (thin, delegates to `npdl.cli`); `main.py` (19 KB of runner logic) is folded into `npdl/simulation/` with `main.py` left as a deprecated shim for one release; `run_npd_simulator.py` (imports a package path that does not exist at root) is fixed or removed.
- Configs: `scenarios/` and `configs/` stay as data, but gain checked-in schema files and a validation test so bad parameters fail fast instead of mid-run.
- Experiments: one registry (seed, config hash, command, environment note, manifest) with results written to ignored directories; committed result dumps (`demo_results/`, `parameter_sweep_results/`, `evolution_analysis/` outputs) move to artifact storage or the archive, with one small golden fixture kept for tests.
- Dashboards: `npdl/visualization/dashboard.py` is the canonical dashboard; `web-dashboard/` (standalone pages) is either a thin consumer of exported data or archived, not a second implementation of the same charts.
- Tooling: reuse the checks already declared in `requirements.txt` and repo config (pytest, black, isort, mypy, Qodana workflow); any additional tool adoption is proposed by the swarm during execution and recorded in the orchestration log, never silently introduced.
- Swarm shape: parent orchestrates via the Workflow tool in waves; parallel writers run in isolated worktrees; each lane has an implementer and a separate retroactive QA agent (correctness: behavior equivalence; quality: tests, style, docs). Merge requires a QA pass recorded in the log.
- History: new commits only; the existing ~2449 commits are never rewritten. Backdated dissemination is quarantined as the last optional phase.

## Steps

Tracker setup (part of execution, structural commitment):

- T1. This plan file stays the live plan tracker: each step below gains status, owner lane, branch, PR, and QA verdict as work proceeds.
- T2. Create `.agents/ORCHESTRATION_LOG.md` on `go`: the swarm ledger (wave, lane, worktree/branch, dispatched task, resulting PR, reviewer comments, QA verdict, merge commit). Only these two files track the overhaul.

Work units (each is its own branch, PR, review, QA verdict, and commit series, in this order; dependencies in brackets):

- W0. Baseline and safety [none]: tag the pre-overhaul snapshot; record the failing-tests baseline (11 entries) and the full file inventory; define golden-output fixtures from current behavior for agents, payoffs, and one small end-to-end run. Commits: snapshot tag + baseline notes + fixtures.
- W1. Packaging and repo skeleton [W0]: add project metadata and a dependency lock next to `requirements.txt`; declare the canonical layout (`npdl/`, `tests/`, `configs/`, `scenarios/`, `experiments/`, `docs/`, `archive/`); remove committed caches (`__pycache__/`, `.pytest_cache/`) from tracking and tighten ignores. Commits: metadata, layout, ignore hygiene.
- W2. Entry-point unification [W1]: fold `main.py` runner logic into `npdl/simulation/`; leave a deprecated `main.py` shim; fix or remove `run_npd_simulator.py`; add CLI smoke tests (`simulate --help`, `visualize`, `interactive` wiring). Commits: simulation move, shim, CLI tests.
- W3. Agent/environment consolidation [W1]: diff `npdl/core/` against `3-Person_Tragic_vs_Reciprocity/*/final_agents.py`, `modular_agents.py`, `strategies.py`, and `simple_models/simple_*.py`; port any missing behavior into `npdl/core/` behind golden-output tests; redirect live code to `npdl.*`. This is the highest-risk unit. Commits: one per strategy/behavior port, each with its test.
- W4. Archive pass [W3]: move superseded versions (`v1..v8` at minimum; `v9` only if fully ported), `legacy*`, and `code_for_website/*` duplicates into `archive/` with a manifest (origin path, what superseded it, how to run the archived copy); keep exactly one runnable reference path per experiment family. Commits: one per archived family.
- W5. Experiment system [W2, W3]: introduce the run registry (seed, config hash, command, manifest); migrate `scripts/runners/*` and the surviving `3-Person`/`code_for_website` runners onto it; uncommit result dumps, keeping one small golden fixture; add schema files plus validation tests for `scenarios/` and `configs/`. Commits: registry, runner migrations one by one, schemas.
- W6. Test suite green [W3, W5]: fix the 11 failing entries (TFT ecosystem defaults, logging utilities, TFT agent defaults, visualization collection) against the real contracts in code and existing tests; add missing coverage for consolidated behavior; set the suite as a required gate. Commits: one fix per failing area, then coverage additions.
- W7. Docs and paper [W4, W5]: rewrite root README around the canonical layout; fix `docs/README.md` (it references a `testing/` dir that does not exist) and dead links; fold `COMPREHENSIVE_NPD_DOCUMENTATION.md` and `DEEP_RESEARCH_DIGEST.md` into `docs/` with clear ownership; split `Paper Resources/` into sources vs build artifacts (artifacts untracked/ignored). Commits: README, docs index, paper split.
- W8. Dashboards [W5]: make `npdl/visualization/dashboard.py` canonical; convert or archive `web-dashboard/` and the root `npd_cooperation_dashboard.html`; add a dashboard smoke test on fixture data. Commits: dashboard consolidation, smoke test.
- W9. Gates and monitoring [W6, W7, W8]: add CI workflows for tests, format/type checks, and docs-link checks on every PR; add coverage reporting and experiment-run manifests as build artifacts; keep the existing Qodana workflow. Commits: one per gate.
- W10. Retroactive QA sweep [W9]: dedicated QA agents re-verify every merged unit (behavior equivalence via golden outputs, test honesty, style, docs accuracy, no stray artifacts); findings become fix PRs with their own verdicts before release. Commits: fix PRs only.
- W11. Release [W10]: changelog, version tag, migration guide for the old entry points and moved paths, final tracker close-out. Commits: changelog, guide, tag.
- W12. Backdated dissemination (optional, separate, last) [W11]: only on explicit reconfirmation; replay the per-unit commits as 1–5 small commits per day with random times from Aug 2025 to today on a dedicated branch, then review and merge. Never rewrites the pre-existing ~2449 commits.

Swarm orchestration:

- Wave A: W0 alone (baseline must exist before anything moves).
- Wave B: W1, then W2+W3 in parallel lanes once W1 merges.
- Wave C: W4+W5+W6 in parallel lanes; W7+W8 in parallel once W4/W5 merge.
- Wave D: W9, then W10, then W11, then optionally W12.
- Every lane: implementer agent opens a PR; a different QA agent reviews for correctness (golden outputs, edge cases) and quality (tests, style, docs, no extra scope); the parent merges only on a recorded pass, and the verdict goes in the orchestration log.

## Validation Plan

- W0: `git tag` snapshot exists; `pytest tests/ -q` reproduces the 11-entry failing baseline; golden fixtures regenerate byte-identically twice from the same seed.
- W1: fresh checkout installs from the lock; `python -c "import npdl.cli"` passes; `git status --short` shows no tracked `__pycache__/` or `.pytest_cache/`; ignore rules verified with `git check-ignore` on sample artifact paths.
- W2: `python run.py simulate --help`, `visualize`, and `interactive` wiring smoke-tested; deprecated `main.py` shim emits its warning and delegates; no import of removed root-relative paths remains.
- W3: golden-output comparisons for every ported strategy pass (pre-change vs post-change fixtures); full `pytest tests/ -q` shows no new failures vs the W0 baseline.
- W4: every archived family has a manifest row; no live import resolves into `archive/` (verified by import scan); one runnable reference path per experiment family documented and smoke-run.
- W5: a fresh seeded run writes seed, config hash, command, and manifest; re-running with the same seed reproduces the manifest hashes; invalid scenario/config files fail the new validation tests.
- W6: `pytest tests/ -q` fully green; per-area suites (`test_agents.py`, TFT ecosystem, logging utilities, visualization) pass individually.
- W7: docs-link check passes with zero dead links; `docs/README.md` matches the real tree; paper sources build from sources alone with no committed build artifacts.
- W8: dashboard smoke test renders from fixture data; no duplicate chart logic remains outside the canonical dashboard path.
- W9: a trial PR demonstrates all gates running and a failing check blocking merge.
- W10: every unit has a recorded QA verdict; all findings closed or carried as explicit follow-ups.
- W11: changelog, migration guide, and tag reviewed; trackers marked complete.
- W12: branch history shows the planned per-day distribution with no changes to pre-existing commits; final diff vs W11 is history-shape only.

## Risks / Open Questions

- Behavior drift in W3 (highest risk): agent logic is duplicated across `npdl/core/`, versioned `final_agents.py` copies, and `simple_models/`; mitigation is golden-output fixtures from W0 and one-strategy-per-commit ports with tests.
- Provenance loss: researchers may still cite versioned paths; mitigation is archive-first with a manifest plus a migration guide, never silent deletion.
- Backdating (W12) fabricates provenance in a research repo and conflicts with reproducibility claims; that is why it is quarantined, optional, and needs reconfirmation before running.
- OneDrive workspace: long paths and file sync can interfere with caches and worktrees; mitigation is short branch/worktree names and verifying `git status` cleanliness per lane.
- Open questions: (1) archive `v9` and the `submittable_code` bundles too, or keep one runnable? (default: keep one runnable reference per family). (2) Is the W12 backdated dissemination still wanted given the provenance risk? (default: no; honest dates unless reconfirmed). (3) Should PRs target `main` directly or an integration branch? (default: short-lived lanes into `main` with required QA verdict).
