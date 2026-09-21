# archive/ manifest (W4)

Superseded duplicate families moved here with `git mv` (history-preserving) on
branch `w4-archive`. Layout mirrors the origin tree so relative paths inside the
archived code (e.g. v1's `../../legacy/...` output dir) still resolve.
Nothing under `archive/` is imported by live code.

Conventions: run commands assume the repo root as cwd unless noted; archived
scripts use sibling imports, so `cd` into the listed dir first. Dependencies are
per-directory `requirements.txt` where present, else the root `requirements.txt`
(numpy/matplotlib/pandas/seaborn).

| Archived path | Origin path | Contents | Superseded in `npdl/` by | How to run the archived copy |
|---|---|---|---|---|
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v1` | `3-Person_Tragic_vs_Reciprocity/final_experimentations/v1` | Early context-dict-API agents (`BaseAgent`, `StaticAgent`, `SimpleQLearningAgent`, `EnhancedQLearningAgent`) + demo/simulation | `npdl.core.tournament_agents.StaticAgent`; v1 neighborhood states ≡ `proportion_discretized`, pairwise state ≡ `TruePairwiseQLearning` `basic`, enhanced state subsumed by `memory_enhanced` (W3 finding) | `cd archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v1 && python final_demo.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v2` | `.../final_experimentations/v2` | `*_v2` trio: `SimpleQLearningAgent`, `VanillaEnhancedAgent`, `BetterEnhancedAgent` | Same lineage as v1; canonical static/Q behavior in `npdl.core.tournament_agents` (v2–v4 pre-date the W3 diff, preserved here only) | `cd .../v2 && python final_demo_v2.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v3` | `.../final_experimentations/v3` | `*_v3` trio: `VanillaQLearningAgent`, `AdaptiveAgent` | Same as v2 | `cd .../v3 && python final_demo_v3.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v4` | `.../final_experimentations/v4` | `*_v4` trio: `VanillaQLearningAgent`, `TrulyAdaptiveAgent` | Same as v2 | `cd .../v4 && python final_demo_v4.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v5` | `.../final_experimentations/v5` | Modern `final_agents.py`/`modular_agents.py`/`strategies.py` + demos + `submittable_code/{main_runs,cooperation_focussed}` | `npdl.core.tournament_agents`, `npdl.core.modular_strategies`, `npdl.core.modular_agents` — v5 agent files are code-identical to v9 (W3 verified) | `cd .../v5 && python final_demo_full.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v6` | `.../final_experimentations/v6` | Same set as v5 (str action keys; some demos `.disabled`) | Same ports as v5; v9 int-key form is canonical (W3) | `cd .../v6 && python final_demo_full.py.disabled` is disabled; use `multi_agent_demo.py`: `python multi_agent_demo.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v7` | `.../final_experimentations/v7` | Same set as v5 (str keys) + `submittable_code/` (`final_agents_v2.py` with `VanillaQLearner`/old `EnhancedQLearner`, TFT experiments) | Same ports as v5; TFT-E error decay preserved opt-in via `StaticAgent(error_decay_rate=0.9995)`; fixed 0.1 optimistic init via `optimistic_init=0.1` (both W3-tested) | `cd .../v7 && python final_demo_full.py`; `cd .../v7/submittable_code && python tft_experiment_standalone.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/final_experimentations/v8` | `.../final_experimentations/v8` | Same set as v5 + `df_sensitivity_analysis.py` + `submittable_code/df_sensitivity` | Same ports as v5; v8 agent files code-identical to v9 (W3 verified) | `cd .../v8 && python final_demo_full.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/legacy` | `3-Person_Tragic_vs_Reciprocity/legacy` | Pre-versioned experiments: `main_pairwise.py`, `main_neighbourhood.py`, `qlearning_agents.py`, enhanced variants, figure generators, `submittable_code/` bundles, sweep results | Static/Q-learning behavior covered by `npdl.core.tournament_agents` + `npdl` strategies; pairwise/neighborhood runners pre-date W3, preserved here only | `cd archive/3-Person_Tragic_vs_Reciprocity/legacy && python run_all_experiments.py` (orchestrator) or `python main_pairwise.py` |
| `archive/3-Person_Tragic_vs_Reciprocity/legacy_cleaned` | `3-Person_Tragic_vs_Reciprocity/legacy_cleaned` | Cleaned `src/`-layout package snapshot (`agents.py`, `game_environments.py`, `experiment_runner.py`) + examples + `main.py` | Same as `legacy/`; `src/agents.py` lineage (`StaticAgent`, `SimpleQLearningAgent`, `NPDLQLearningAgent`) covered by `npdl.core.tournament_agents` | `cd archive/3-Person_Tragic_vs_Reciprocity/legacy_cleaned && python main.py` (see `run.sh`, `examples/`) |
| `archive/code_for_website/cooperation_focussed` | `code_for_website/cooperation_focussed` | Cooperation-measurement analysis bundle | Agent logic in `npdl.core.tournament_agents`; staying reference: `code_for_website/main_runs/` | `cd archive/code_for_website/cooperation_focussed && python cooperation_measurement.py` |
| `archive/code_for_website/df_sensitivity` | `code_for_website/df_sensitivity` | Discount-factor sensitivity analysis bundle | Same as above | `cd archive/code_for_website/df_sensitivity && python df_sensitivity_analysis.py` |
| `archive/code_for_website/multi_agent` | `code_for_website/multi_agent` | Standalone multi-agent scaling run | Same as above | `cd archive/code_for_website/multi_agent && python multi_agent_run_standalone.py` |
| `archive/code_for_website/qlearning_demo` | `code_for_website/qlearning_demo` | Q-learning demo bundle | Same as above | `cd archive/code_for_website/qlearning_demo && python run_qlearning_demo.py` |
| `archive/code_for_website/static_figure_generator` | `code_for_website/static_figure_generator` | Static figure generator (full) | Same as above (figures, not agents) | `cd archive/code_for_website/static_figure_generator && python static_figure_generator.py` |
| `archive/code_for_website/static_figure_generator_2tfte_allc_alld` | `code_for_website/static_figure_generator_2tfte_allc_alld` | Static figure generator (2 TFT-E + AllC/AllD) | Same as above | `cd archive/code_for_website/static_figure_generator_2tfte_allc_alld && python static_figure_generator_2tfte_allc_alld.py` |
| `archive/web-dashboard/` (app files) | `web-dashboard/` | Standalone static dashboard: `index.html`, `css/`, `js/` (~160KB reimplementing cooperation/payoff/strategy/network charts; `simulateScenarioResults`/`generateSyntheticData` fabricate curves when files are missing), `test-*.html` pages, README/SETUP_GUIDE | `npdl.visualization.dashboard` + [docs/DASHBOARD.md](../docs/DASHBOARD.md) (W8: archived, not converted -- no canonical export format exists for a thin consumer, and the synthetic-data fallback conflicts with reproducibility) | `cd archive/web-dashboard && python -m http.server 8000`, open `index.html` (CDN deps: Chart.js, D3, Plotly, Lucide) |
| `archive/web-dashboard/docs/` | `docs/web-dashboard/` | Dashboard docs hub: comprehensive doc, next-steps plan, fix summary, project status | [docs/DASHBOARD.md](../docs/DASHBOARD.md) (W8) | Read the `.md` files in place; the `../../web-dashboard/*.html` links resolve inside the archive |
| `archive/web-dashboard/docs/WEB_DASHBOARD_PLAN.md` | `docs/WEB_DASHBOARD_PLAN.md` | Static-site design plan (GitHub Pages vision) | Same as above | Same as above |
| `archive/web-dashboard/npd_cooperation_dashboard.html` | `npd_cooperation_dashboard.html` | Single-file static cooperation dashboard (Chart.js/D3 via CDN; referenced by no live doc or code) | `npdl.visualization.dashboard` (W8) | Open in a browser (needs network for the CDN scripts) |

## Deliberately NOT archived

- `3-Person_Tragic_vs_Reciprocity/final_experimentations/v9` — staying runnable
  reference for the versioned family (W3 port source; W5 owns runner migration).
- `code_for_website/main_runs/` — staying runnable reference; contains live user
  modifications, never to be moved/staged/committed by the overhaul.
- `3-Person_Tragic_vs_Reciprocity/npd_simulator/` — left in place: still imported
  by staying runners (`run_research_experiments.py`, `run_quick_test.py`,
  `check_registry.py`, `test_csv_visualizer.py`, `run_from_parent.py`). W5 owns it.
- `demo_results/`, `parameter_sweep_results/`, `evolution_analysis/` dumps — W5 owns.
- `3-Person_Tragic_vs_Reciprocity/code_for_website/` (single
  `static_figure_generator` copy) — out of W4 scope; leftover for W5/W7.
- `code_for_website/run_all_experiments.py` + `README_RUN_ALL.md` stay in place;
  the runner's entries for the six archived subdirs now report "Script not
  found" gracefully (exit path unchanged, no crash). W5 owns runner migration;
  paths were deliberately NOT repointed into `archive/` (no live code may
  resolve into the archive).

## Test-only loader update

`tests/test_tournament_agents.py` loads `v7/submittable_code/final_agents_v2.py`
read-only for W3 equivalence tests; its `V7SUB` loader path now points at the
archived location. `tests/test_modular_agents.py` needs no change (v9 stays).
No behavior change; all 98 W3 tests still pass.
