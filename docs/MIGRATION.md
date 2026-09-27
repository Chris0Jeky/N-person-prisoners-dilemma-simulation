# Migration guide: pre-overhaul paths to 1.0.0

This guide maps every entry point and path that moved or was removed during
the W0–W10 overhaul (release [1.0.0](../CHANGELOG.md)). All moves used
`git mv`, so history follows the new paths; the pre-overhaul state is tagged
`pre-overhaul-baseline`. Each row below was verified against the current
tree. Per-unit detail lives in [../CHANGELOG.md](../CHANGELOG.md); archived
code provenance lives in [../archive/MANIFEST.md](../archive/MANIFEST.md).

## Entry points

| Old | New | Notes |
|-----|-----|-------|
| `python main.py ...` | `python run.py simulate ...` ([run.py](../run.py)) | `main.py` ([main.py](../main.py)) still works but is a deprecated shim: it emits `DeprecationWarning` and delegates to [`npdl/simulation/experiments.py`](../npdl/simulation/experiments.py). New code should call `run.py` or import from `npdl.simulation.experiments`. Canonical since W2. |
| `from main import load_scenarios, setup_experiment, ...` | `from npdl.simulation.experiments import ...` | The shim re-exports the moved names, so old imports keep working with a warning (W2). |
| `python run_npd_simulator.py` | Removed; no direct replacement | The script imported a package path that never existed at root (`ModuleNotFoundError`) and served only the legacy 3-person tree. Run the archived legacy copies per [archive/MANIFEST.md](../archive/MANIFEST.md), or the staying `v9` reference below (W2). |
| `python run.py ...` | Unchanged ([run.py](../run.py)) | Still the canonical CLI (`simulate` / `visualize` / `interactive`); smoke-tested in `tests/test_cli_smoke.py` (W2). |

## Versioned experiment code moved to archive/

| Old | New | Notes |
|-----|-----|-------|
| `3-Person_Tragic_vs_Reciprocity/final_experimentations/v1` … `v8` | `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v1` … `v8` | Superseded by `npdl.core.tournament_agents`, `npdl.core.modular_strategies`, `npdl.core.modular_agents` (W3 ports). Each family has a manifest row with run instructions (W4). |
| `3-Person_Tragic_vs_Reciprocity/legacy/` | `archive/3-Person_Tragic_vs_Reciprocity/legacy/` | Pre-versioned experiments; see manifest (W4). |
| `3-Person_Tragic_vs_Reciprocity/legacy_cleaned/` | `archive/3-Person_Tragic_vs_Reciprocity/legacy_cleaned/` | Cleaned `src/`-layout snapshot; see manifest (W4). |
| `3-Person_Tragic_vs_Reciprocity/final_experimentations/v9` | Unchanged (staying runnable reference) | W3 port source; kept runnable per the plan default (W4). |
| `3-Person_Tragic_vs_Reciprocity/npd_simulator/` | Unchanged (staying, still imported by staying runners) | Left in place by W4; W5 kept the surviving runners as-is. |

## Website experiment bundles moved to archive/

| Old | New | Notes |
|-----|-----|-------|
| `code_for_website/cooperation_focussed/`, `df_sensitivity/`, `multi_agent/`, `qlearning_demo/`, `static_figure_generator/`, `static_figure_generator_2tfte_allc_alld/` | `archive/code_for_website/<same names>/` | Agent logic superseded by `npdl.core.tournament_agents`; run instructions per family in the manifest (W4). |
| `code_for_website/main_runs/` | Unchanged (staying runnable reference) | Carries live user modifications; never moved by the overhaul (W4). |
| `code_for_website/run_all_experiments.py` entries for the six moved subdirs | Same script, graceful "Script not found" | The runner stays in place; paths were deliberately not repointed into `archive/` because no live code may resolve into the archive (W4). |

## Paper sources

| Old | New | Notes |
|-----|-----|-------|
| `Paper Resources/` | `paper/` ([paper/](../paper/)) | Renamed (W7). Sources (`.tex`/`.bib`/`.eps`/`.plt`/data) stay tracked; build outputs (`.pdf` intermediates, `.aux`, `.log`, `*-eps-converted-to.pdf`, `*~` backups) were untracked and are now ignored — except `prisoners.pdf`, which stays tracked as the distributable. Rebuild pipeline (`gnuplot` → `pdflatex`/`bibtex`) in [paper/README.md](../paper/README.md). |

## Dashboards and their docs

| Old | New | Notes |
|-----|-----|-------|
| `web-dashboard/` (standalone HTML/JS) | `archive/web-dashboard/` ([archive/web-dashboard/](../archive/web-dashboard/)) | Archived, not converted: it reimplemented the canonical charts and fabricated data when files were absent. Serve from the archive dir if needed (W8). |
| `docs/web-dashboard/` | `archive/web-dashboard/docs/` | Moved with the app so relative links still resolve (W8). |
| `docs/WEB_DASHBOARD_PLAN.md` | `archive/web-dashboard/docs/WEB_DASHBOARD_PLAN.md` | Static-site design plan, archived with the family (W8). |
| `npd_cooperation_dashboard.html` (root) | `archive/web-dashboard/npd_cooperation_dashboard.html` | Single static file, referenced by nothing live (W8). |
| (canonical) | `npdl/visualization/dashboard.py` + [DASHBOARD.md](DASHBOARD.md) | The Dash dashboard is the one supported dashboard (W8). |

## Docs folded into docs/

| Old | New | Notes |
|-----|-----|-------|
| `COMPREHENSIVE_NPD_DOCUMENTATION.md` (root) | `docs/COMPREHENSIVE_NPD_DOCUMENTATION.md` ([COMPREHENSIVE_NPD_DOCUMENTATION.md](COMPREHENSIVE_NPD_DOCUMENTATION.md)) | `git mv`, no stub left (W7). |
| `DEEP_RESEARCH_DIGEST.md` (root) | `docs/DEEP_RESEARCH_DIGEST.md` ([DEEP_RESEARCH_DIGEST.md](DEEP_RESEARCH_DIGEST.md)) | `git mv`, no stub left (W7). |
| `docs/README.md` → `testing/` | `tests/README.md` + `tests/TEST_PLAN.md` | The old link target never existed; the index now points at the real test docs (W7). |

## Result dumps now untracked

| Old | New | Notes |
|-----|-----|-------|
| Committed `parameter_sweep_results/`, `evolution_analysis/` dumps | Untracked and ignored (`.gitignore`); regenerate with the seeded runners | Removed from tracking in W5 (`-86k` dump lines). Old dumps survive in history before the W5 merge (`8c497d98`). `demo_results/` had nothing committed and is ignored too. |
| (golden reference) | `tests/fixtures/w5_golden/strategy_stats.csv` ([strategy_stats.csv](../tests/fixtures/w5_golden/strategy_stats.csv)) | The one kept fixture, pinned by shape + sha256 in `tests/test_w5_golden_fixture.py` (W5). |
| `scripts/runners/*.py` invocation | Same paths, new `--seed` flag (default 0), run from root with `PYTHONPATH=.` | All four runners register a run dir with `manifest.json`; same seed reproduces manifest hashes. `*.log` files are excluded from manifests by design (W5). |

## New experiment-system paths

| Path | Purpose |
|------|---------|
| `npdl/experiments/` ([npdl/experiments/](../npdl/experiments/)) | Run registry: `create_run`, `finalize`, `verify_manifest` (W5). |
| `scenarios/schema.json`, `configs/schema.json` | Checked-in schemas; enforced by `npdl.experiments.validate` (`ValidationError` with JSON paths) and `tests/test_config_schemas.py` (W5). |
| `tests/fixtures/w0_golden/` | Payoff, strategy-move, and end-to-end golden fixtures; `generate.py --check` must print 3/3 IDENTICAL (W0). |
| `experiments/` ([experiments/](../experiments/)) | Notes dir for the experiment system; the registry itself lives in `npdl/` (W1). |

## Overhaul process records (not runtime)

| Path | Purpose |
|------|---------|
| `.agents/plans/2026-09-26-research-repo-overhaul.md` | Live plan tracker (T1): goal, per-unit steps, validation plan. |
| `.agents/ORCHESTRATION_LOG.md` | Swarm ledger (T2): per-unit branch, PR, QA verdict, merge commit. |
| `.agents/W0_BASELINE.md` | W0 evidence: failing-tests baseline, file inventory, golden hashes. |

These are process records owned by the overhaul parent lane; normal users
can ignore them.
