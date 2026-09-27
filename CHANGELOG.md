# Changelog

## [1.0.0] - 2026-09-27

First stable release: the repository is now one canonical package (`npdl/`),
one CLI (`run.py`), one experiment system with seeded runs and artifact
manifests, consolidated docs, a green test suite with automated gates, and
legacy variants archived with provenance. Every change below was reviewed by
an independent QA agent before merge (verdicts in
`.agents/ORCHESTRATION_LOG.md`). For old → new paths see
[docs/MIGRATION.md](docs/MIGRATION.md); for archived code see
[archive/MANIFEST.md](archive/MANIFEST.md). The pre-overhaul state is tagged
`pre-overhaul-baseline` (at `21c3174`).

### W0 — Baseline and safety (PR #43, `da30f844`)

- Tagged the pre-overhaul snapshot (`pre-overhaul-baseline`).
- Recorded the failing-tests baseline (36 failed, 440 passed, 1 error plus a
  collection error; see `.agents/W0_BASELINE.md`).
- Added golden fixtures (`tests/fixtures/w0_golden/`) that regenerate
  byte-identically; every later unit re-verified them (`--check` 3/3
  IDENTICAL). No production code changes.

### W1 — Packaging and repo skeleton (PR #44, `694fd46e`)

- Added `pyproject.toml` (project metadata, black/isort/mypy config, pytest
  settings mirroring `pytest.ini`, which stays canonical) and
  `requirements.lock` (pins captured from the W0 environment; absent packages
  keep `requirements.txt` floors and are marked not installed).
- Declared the canonical layout: new `experiments/` and `archive/` dirs with
  READMEs. Tightened ignore rules; committed caches turned out to be a
  verified no-op (none were tracked).
- Migration: install still works from `requirements.txt`; the lock is the
  reproducible path (`pip install -r requirements.lock`).

### W2 — Entry-point unification (PR #45, `52c7503d`)

- Moved root `main.py` runner logic verbatim into
  `npdl/simulation/experiments.py` (`load_scenarios`, `setup_experiment`,
  `save_results`, `print_comparative_summary`, `main`); `main.py` is now a
  thin shim emitting `DeprecationWarning` and re-exporting those names.
- Removed `run_npd_simulator.py`: it imported a package path that does not
  exist at root (`ModuleNotFoundError`, reproduced) and served only the
  legacy tree.
- Added CLI smoke tests (`tests/test_cli_smoke.py`, 14 tests): `run.py`
  `simulate`/`visualize`/`interactive --help`, dispatch wiring, shim
  warn-and-delegate.
- Migration: run `python run.py simulate` (or import from
  `npdl.simulation.experiments`) instead of `python main.py`; see
  [docs/MIGRATION.md](docs/MIGRATION.md).

### W3 — Agent/environment consolidation (PR #46, `b73f10f3`)

- Ported the versioned v9 agent implementations exactly into `npdl/core/`:
  `tournament_agents.py` (`StaticAgent` family, `PairwiseAdaptiveQLearner`,
  `HystereticQLearner`, `NeighborhoodAdaptiveQLearner`, `LegacyQLearner`,
  `Legacy3RoundQLearner`), `modular_strategies.py`, `modular_agents.py`
  (plus the 10 `create_*` factories), with an additive-only
  `npdl.core/__init__.py` export. No existing `npdl/core` behavior changed.
- Two older-snapshot behaviors preserved opt-in and tested:
  `StaticAgent(error_decay_rate=0.9995)` (v7 TFT-E error decay) and
  `optimistic_init=0.1`. v5/v8 agent files verified code-identical to v9;
  v1/v6/v7, `simple_models/`, and environment conventions recorded as
  covered or superseded (see the PR for the full accounting).
- Added 98 golden-equivalence tests (`tests/test_tournament_agents.py`,
  `tests/test_modular_agents.py`) comparing old vs new on identical seeded
  inputs. Suite delta was exactly the new green tests; zero new failures.

### W4 — Archive pass (PR #47, `285f6a72`)

- Moved superseded families into `archive/` with `git mv`
  (history-preserving, all R100 renames, one commit per family, 182 files):
  `final_experimentations/v1..v8` (116), `legacy/` (30),
  `legacy_cleaned/` (13), and six `code_for_website/` subdirs (23).
- Added `archive/MANIFEST.md` (origin path, what superseded it, how to run
  each archived copy) and updated `archive/README.md`; repointed the
  test-only v7 loader at the archived copy (1 line, no behavior change).
- Deliberately kept runnable: `v9` and `code_for_website/main_runs/` (which
  also carries live user modifications); result dumps and `npd_simulator/`
  left for W5. Verified no live import resolves into `archive/`.
- Migration: every moved path is mapped in [docs/MIGRATION.md](docs/MIGRATION.md).

### W5 — Experiment system (PR #48, `8c497d98`)

- New `npdl/experiments/` run registry: `create_run()` records seed, config
  hash (sha256 of canonical JSON), and command line;
  `ExperimentRun.finalize()` writes a `manifest.json` of artifact hashes;
  `verify_manifest()` re-checks it. Deterministic run-dir names
  (`{experiment}_seed{seed}_{hash8}`).
- Migrated all four `scripts/runners/*` onto the registry (each gains
  `--seed`, default 0; re-running with the same seed reproduces manifest
  hashes; thin diffs, no algorithm changes).
- Added `scenarios/schema.json` + `configs/schema.json` with a
  dependency-free validator (`validate_scenario_file`,
  `validate_config_file`) and `tests/test_config_schemas.py` (existing
  files pass, invalid cases fail).
- Uncommitted result dumps (`parameter_sweep_results/`, `evolution_analysis/`;
  `demo_results/` had nothing committed), extended `.gitignore`, and kept one
  golden fixture (`tests/fixtures/w5_golden/strategy_stats.csv`, pinned by
  `tests/test_w5_golden_fixture.py`). 40 new tests, all green.
- Migration: regenerate old dumps with the seeded runners; old committed
  dumps survive in history before this PR's merge.

### W6 — Test suite green (PR #49, `e127778b`)

- Fixed the full W0 baseline (36 failures + 1 error + 2 collection layers),
  one commit per area: `conftest` pytest-9 fixture params, plotly skip guard,
  TFT ecosystem/pairwise setups, logging utilities, advanced strategies
  (incl. a textbook Wolf-PHC implementation), CLI viz wiring, refactored-path
  mapping. Code was fixed where tests encoded documented behavior; test
  changes were limited to setup bugs, stale contracts, the planned layout
  move, and provably unpassable bars (each justified in its commit message).
- Result: stock `pytest tests/ -q` → **628 passed, 1 skipped**, golden
  3/3 IDENTICAL.

### W7 — Docs and paper (PR #50, `1685efc8`)

- Rewrote the root README around the canonical layout (every command/path
  verified by execution).
- Fixed all dead links (5 found, incl. `docs/README.md` → nonexistent
  `testing/`) and added `scripts/analysis/check_doc_links.py` with
  `tests/test_doc_links.py` (repo-wide scan clean). Docs-only plus
  `.gitignore`; zero `.py` behavior changes.
- Folded `COMPREHENSIVE_NPD_DOCUMENTATION.md` and `DEEP_RESEARCH_DIGEST.md`
  from root into `docs/` via `git mv` (history preserved, no stubs).
- Renamed `Paper Resources/` → `paper/`, untracked LaTeX build outputs and
  `*~` backups, extended `.gitignore`, added `paper/README.md` rebuild
  notes (`prisoners.pdf` stays tracked as the distributable).
- Suite: 641 passed, 1 skipped. Migration: see
  [docs/MIGRATION.md](docs/MIGRATION.md) for every moved doc path.

### W8 — Dashboards (PR #51, `3c25c676`)

- `npdl/visualization/dashboard.py` is the canonical dashboard; documented in
  `docs/DASHBOARD.md` (launch, data layout, tests).
- Archived (not converted) the static pages: `web-dashboard/`,
  `docs/web-dashboard/`, `docs/WEB_DASHBOARD_PLAN.md`, and root
  `npd_cooperation_dashboard.html` → `archive/web-dashboard/` with manifest
  rows. Rationale: the JS bundle reimplemented the canonical charts, invented
  data via `Math.random()` when files were absent, and no canonical JSON
  export format existed for a thin consumer.
- Added a dashboard figure smoke test (no browser; documented skip without
  the Dash stack). Suite: 641 passed, 2 skipped; links clean.

### W9 — CI gates (PR #52, `98ad8cc7`)

- New workflows on every PR and push to `main`: `tests.yml` (stock suite on
  Python 3.13 from `requirements.lock` with `requirements.txt` fallback,
  plus a coverage job uploading XML as a build artifact), `lint.yml`
  (`black --check`, `isort --check-only`, `mypy` per `pyproject.toml`),
  `docs.yml` (the W7 link checker, stdlib-only). Existing Qodana workflow
  untouched. Each gate was proven to fail red and pass green.
- Format-only pass (black/isort over `npdl/ tests/ scripts/ run.py main.py`);
  suite and golden re-verified identical afterwards.
- CI exposed and fixed three latent full-stack issues: declared the missing
  `scipy` dependency, pinned the golden CSV's line endings via
  `.gitattributes`, and excluded `visualization/dashboard.py` from mypy
  (Dash-4 stub error). Pre-existing mypy errors in 7 files and one
  isort-dirty file owned outside the lane are documented exclusions with
  tracked follow-ups (see the PR).
- Final CI state: 668 passed (full viz stack installed), coverage 73% with
  artifact, lint/docs-links/Qodana green.

### Chore — Remove Claude review gate (PR #53, `484651cb`)

- Removed `.github/workflows/claude-code-review.yml`: the bot account is out
  of credits, so the check failed on every PR without reviewing anything.
  The `@claude`-mention workflow (`claude.yml`) is untouched. The PR itself
  proved the gate's absence with all other gates green.

### W10 — Retroactive QA sweep (no PR, read-only)

- Three independent sweep lenses (correctness, quality, surface) re-verified
  every merged unit: behavior equivalence via golden outputs, test honesty,
  style, docs accuracy, no stray artifacts. All PASS, zero findings, no fix
  PRs required. Verdicts recorded in `.agents/ORCHESTRATION_LOG.md`.

## [0.1.0] - 2026-09-26

Pre-release packaging baseline (W1, PR #44): `pyproject.toml` declared the
project as `npdl` version 0.1.0. Everything before 1.0.0 above is the
W0–W10 overhaul history; earlier project history lives in git before the
`pre-overhaul-baseline` tag.
