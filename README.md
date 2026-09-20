# N-Person Prisoner's Dilemma Learning (NPDL)

A framework for simulating, analyzing, and visualizing agent behavior in
N-Person Prisoner's Dilemma scenarios: classic game-theory strategies and
reinforcement-learning agents playing on complex networks, with seeded,
reproducible experiment runs.

One package (`npdl/`), one CLI (`run.py`), one experiment registry
(`npdl.experiments`). Superseded versioned copies live read-only under
`archive/` — see [archive/MANIFEST.md](archive/MANIFEST.md).

## Requirements

Python 3.7+ and the packages in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python run.py simulate --help      # simulation options
python run.py simulate --scenario_file scenarios/scenarios.json
python run.py visualize             # dashboard at http://127.0.0.1:8050/
python run.py interactive           # play against AI agents
```

Point `--scenario_file` at any file in [scenarios/](scenarios/)
(`scenarios.json`, `enhanced_scenarios.json`, `pairwise_scenarios.json`,
`true_pairwise_scenarios.json`, `express_scenarios.json`,
`chris_testing_scenarios.json`); use `--enhanced` for the enhanced set,
`--results_dir` / `--log_dir` to redirect outputs, `--analyze` to analyze
after the run, and `--verbose` for debug logging.

`main.py` at the root is a deprecated shim (it emits a
`DeprecationWarning`); new code should call `python run.py simulate` or
import from `npdl.simulation.experiments`.

## Project layout

```
run.py                  # canonical CLI (delegates to npdl.cli)
main.py                 # deprecated shim over npdl.simulation.experiments
npdl/
├── cli.py              # simulate / visualize / interactive commands
├── core/               # agents, environment, payoff utils, logging
├── simulation/         # experiment runners (setup/run/save)
├── experiments/        # run registry: seed, config hash, manifest + validators
├── analysis/           # sweep visualizer, cooperation-pattern analysis
├── visualization/      # dashboard, data loading/processing, network viz
└── interactive/        # playable game vs AI agents
scenarios/              # scenario JSON files + schema.json
configs/                # sweep configs + schema.json
scripts/
├── runners/            # scenario generation, parameter sweep, evolution
├── demos/              # compare_rl_strategies, TFT ecosystem, true pairwise
└── analysis/           # cooperation-pattern analysis + docs-link checker
experiments/            # experiment-system notes (registry lives in npdl/)
tests/                  # test suite + golden fixtures (w0_golden, w5_golden)
docs/                   # documentation index and guides
archive/                # superseded versioned copies (read-only) + MANIFEST.md
Paper Resources/        # paper sources (.tex/.bib/.eps) + reference PDFs
```

## Running simulations

`python run.py simulate` reads a scenario file (default
`scenarios/scenarios.json`) and writes per-run results plus logs. Scenario
files are validated against [scenarios/schema.json](scenarios/schema.json)
before anything runs, so bad parameters fail fast:

```bash
python -c "from npdl.experiments import validate_scenario_file; validate_scenario_file('scenarios/scenarios.json'); print('valid')"
```

Minimal scenario (neighborhood mode):

```json
{
  "scenario_name": "Example_Neighborhood",
  "interaction_mode": "neighborhood",
  "num_agents": 30,
  "num_rounds": 500,
  "network_type": "small_world",
  "network_params": {"k": 4, "beta": 0.3},
  "agent_strategies": { "q_learning": 15, "tit_for_tat": 15 },
  "payoff_type": "linear",
  "payoff_params": {"R": 3, "S": 0, "T": 5, "P": 1},
  "state_type": "proportion_discretized"
}
```

For pairwise play set `"interaction_mode": "pairwise"` (see
[scenarios/pairwise_scenarios.json](scenarios/pairwise_scenarios.json)) or
`"true_pairwise"` (see
[scenarios/true_pairwise_scenarios.json](scenarios/true_pairwise_scenarios.json)
and [docs/implementation/PAIRWISE_MODE.md](docs/implementation/PAIRWISE_MODE.md)).

## Experiment registry

Every registry-backed run records its seed, config hash (sha256 of the
resolved config), command line, and a `manifest.json` of artifact hashes,
so a re-run with the same seed reproduces the manifest:

```python
from npdl.experiments import create_run, verify_manifest

run = create_run("results", "my_experiment", config, seed=7)
...  # write artifacts under run.run_dir
run.finalize()
assert verify_manifest(run.run_dir) == []
```

`npdl.experiments.validate` (`validate_scenario_file`,
`validate_config_file`) checks scenario/config files against the
checked-in schemas. Sweep configs live in [configs/](configs/) and are
consumed by `scripts/runners/run_parameter_sweep.py --config <file>
--seed <n>` (run with `PYTHONPATH=.` from the repo root).

## Scenario generation and sweeps

```bash
PYTHONPATH=. python scripts/runners/run_scenario_generator.py --num_generate 50 --eval_runs 3 --save_runs 10 --top_n 5
PYTHONPATH=. python scripts/runners/run_sweep_analysis.py --num_generate 30 --top_n 5
PYTHONPATH=. python scripts/runners/run_evolutionary_generator.py --pop_size 20 --generations 5 --eval_runs 3
PYTHONPATH=. python scripts/runners/run_parameter_sweep.py --config configs/sweep_config.json --seed 7
```

The generator samples random scenarios, scores them by "interestingness"
(cooperation dynamics, cross-strategy variance, volatility), and runs full
simulations of the winners; the evolutionary variant improves a population
of scenarios over generations. See
[docs/SCENARIO_GENERATION.md](docs/SCENARIO_GENERATION.md) for the full
guide, including the parameter pools and scoring weights.

## Visualization and interactive play

```bash
python run.py visualize      # Dash dashboard at http://127.0.0.1:8050/
python run.py interactive    # play against AI strategies yourself
```

The dashboard shows cooperation rates over time, strategy comparisons,
network structure, and payoffs. Dashboard internals are documented under
[docs/web-dashboard/](docs/web-dashboard/) (plan:
[docs/WEB_DASHBOARD_PLAN.md](docs/WEB_DASHBOARD_PLAN.md)).

## Demos

```bash
PYTHONPATH=. python scripts/demos/demonstrate_tft_ecosystem.py
PYTHONPATH=. python scripts/demos/demonstrate_true_pairwise.py
PYTHONPATH=. python scripts/demos/compare_rl_strategies.py
```

The TFT demo shows the ecosystem-aware Tit-for-Tat: instead of copying one
opponent's last move, TFT cooperates when the cooperation proportion among
its connected neighbors meets a configurable threshold (default 0.5), with
a probabilistic `ProportionalTitForTat` variant.

## Agent strategies

Classical (`npdl/core/agents.py`): Always Cooperate, Always Defect,
Tit for Tat (ecosystem-aware), Proportional Tit for Tat, Generous TFT,
Suspicious TFT, Tit for Two Tats, Pavlov, Random. Learning:
Q-Learning, Adaptive Q-Learning, LRA-Q, Hysteretic Q, Wolf-PHC, UCB1.
Tournament variants live in `npdl/core/tournament_agents.py` with modular
pieces in `npdl/core/modular_strategies.py` /
`npdl/core/modular_agents.py`.

Networks: fully connected, small world (Watts-Strogatz), scale-free
(Barabasi-Albert), random (Erdos-Renyi), regular. Interaction modes:
`neighborhood` (play neighbors only, default), `pairwise` (one move vs
everyone), `true_pairwise` (a separate move per opponent).

## Tests

```bash
pytest tests/ -q
# 628 passed, 1 skipped

python tests/fixtures/w0_golden/generate.py --check
# payoff_matrices.json: IDENTICAL
# strategy_moves.json: IDENTICAL
# e2e_small_run.json: IDENTICAL
```

The golden fixtures pin payoff matrices, scripted strategy moves, and one
small end-to-end run; `--check` regenerates and byte-compares. Test notes
live in [tests/README.md](tests/README.md) and
[tests/TEST_PLAN.md](tests/TEST_PLAN.md). Docs links are checked by
`scripts/analysis/check_doc_links.py` (run with `PYTHONPATH=.`).

## Documentation

Start at [docs/README.md](docs/README.md):

- [docs/SCENARIO_GENERATION.md](docs/SCENARIO_GENERATION.md) — scenario
  generation and sweep analysis
- [docs/implementation/PAIRWISE_MODE.md](docs/implementation/PAIRWISE_MODE.md)
  — pairwise / true-pairwise interaction modes
- [docs/N_PERSON_RL_ANALYSIS.md](docs/N_PERSON_RL_ANALYSIS.md),
  [docs/N_PERSON_RL_COMPARISON_PLAN.md](docs/N_PERSON_RL_COMPARISON_PLAN.md),
  [docs/N_PERSON_RL_IMPLEMENTATION_SUMMARY.md](docs/N_PERSON_RL_IMPLEMENTATION_SUMMARY.md)
  — N-person RL analysis track
- [docs/web-dashboard/](docs/web-dashboard/) — dashboard docs hub
- [archive/MANIFEST.md](archive/MANIFEST.md) — what was archived, what
  superseded it, how to run the archived copies

## License

This project is available under the MIT License.
