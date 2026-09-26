# W0 Baseline Record

Captured 2026-09-26 on `go`. Tag: `pre-overhaul-baseline` at
`21c31746be938bb564b77b55443f51f7cae1f528`.

Working-tree note at tag time: `code_for_website/main_runs/final_agents.py`
and `npdl/analysis/analysis.py` had pre-existing uncommitted modifications;
left untouched (not part of the baseline).

Environment: win32, Python 3.13.15, pytest 9.0.3. Note: `plotly` (and possibly
other visualization deps from `requirements.txt`) is not installed, and the
repo predates pytest 9.

## Test baseline (three layers)

1. Stock command `pytest tests/ -q`:
   COLLECTION ERROR. `tests/conftest.py:133` applies
   `@pytest.mark.parametrize` to the `diverse_env` fixture, which pytest 9
   rejects at import (`PytestRemovedIn9Warning` raised as error).
2. With `-W "ignore::pytest.PytestRemovedIn9Warning"`:
   476 items collected, then collection ERROR in
   `tests/test_visualization.py` (`ModuleNotFoundError: No module named 'plotly'`).
3. With the warning waived AND `--ignore=tests/test_visualization.py`:
   **36 failed, 440 passed, 1 error.**
   (The stale `.pytest_cache` `lastfailed` listing 11 entries predates this
   run and understates the breakage; the figures below are authoritative.)

### Failing entries (authoritative list)

- test_advanced_strategies.py: 15 (all LRA-Q, UCB1, Wolf-PHC, Hysteretic-Q
  init/learning tests + strategy comparison vs defector)
- test_cli.py: 2 (visualization success + error paths, AttributeError)
- test_integration.py: 1 (WolfOpt_vs_TFT_SW expected cooperation range)
- test_interactive_game.py: 2 (agent creation, game summary)
- test_logging_utils.py: 6 failed + 1 error (setup_logging variants, ascii
  chart dimensions + error handling, file-handler presence)
- test_pairwise.py: 1 (explicit TFT behavior)
- test_pairwise_strategies.py: 1 (TFT fallback to proportion)
- test_refactored_paths.py: 2 (file existence, import paths)
- test_tft_ecosystem_behavior.py: 6 (entire TFT ecosystem class)
- test_visualization.py: whole module uncollectable (missing plotly)

## File inventory (tracked, at tag)

- `3-Person_Tragic_vs_Reciprocity/`: 391 files (v1..v9 chains, legacy,
  legacy_cleaned, npd_simulator, demo_results with committed CSV/PNG/JSON)
- `npdl/`: 40 files (canonical package candidate)
- `code_for_website/`: 40 files (duplicated agent/runner copies)
- `tests/`: 58 files (20 test modules + caches)
- `simple_models/`: 15, `web-dashboard/`: 16, `docs/`: 12, `scripts/`: 9,
  `utilities/`: 11, `configs/`: 2, `scenarios/`: 6,
  `parameter_sweep_results/`: 12, `evolution_analysis/`: 10
- Root entry points: `main.py` (19 KB), `run.py` (thin CLI), `run_npd_simulator.py`
- Total commits at tag: ~2449; recent history is docstring micro-commits.

## Golden fixtures

`tests/fixtures/w0_golden/`: `generate.py` plus three JSON fixtures,
all regenerating byte-identically (`--check` passes):

- `payoff_matrices.json` sha256 `7c1240f632df…` (linear N=4/N=6,
  threshold N=5, exponential N=5, default params)
- `strategy_moves.json` sha256 `ee55c1e92d5a…` (9 strategies x 6 scripted
  rounds, seed 42)
- `e2e_small_run.json` sha256 `666c8f686ece…` (6 agents, fully connected,
  20 rounds, no global bonus, seed 42; defectors outscore cooperators,
  cooperation 0.667 -> 0.500)

Regenerate: `python tests/fixtures/w0_golden/generate.py --check`.
