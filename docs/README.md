# NPDL Documentation

## Overview

This directory contains documentation for the N-Person Prisoner's Dilemma Learning (NPDL) framework.
Start at the [main README](../README.md) for installation and usage.

## Documentation Structure

- **[implementation/](implementation/)** - Technical implementation details
  - [`PAIRWISE_MODE.md`](implementation/PAIRWISE_MODE.md) - Pairwise
    interaction modes (aggregate and true pairwise)
- **[web-dashboard/](web-dashboard/)** - Dashboard docs hub
  - [`README.md`](web-dashboard/README.md) - Hub index: comprehensive
    documentation, next-steps plan, fix summary, project status
- **[SCENARIO_GENERATION.md](SCENARIO_GENERATION.md)** - Guide to scenario
  generation and analysis tools
- **[WEB_DASHBOARD_PLAN.md](WEB_DASHBOARD_PLAN.md)** - Web dashboard design
  and visualization plan
- **[N_PERSON_RL_ANALYSIS.md](N_PERSON_RL_ANALYSIS.md)** - Analysis of
  N-person RL experiments
- **[N_PERSON_RL_COMPARISON_PLAN.md](N_PERSON_RL_COMPARISON_PLAN.md)** -
  Comparison plan for N-person RL experiments
- **[N_PERSON_RL_IMPLEMENTATION_SUMMARY.md](N_PERSON_RL_IMPLEMENTATION_SUMMARY.md)** -
  Implementation summary for N-person RL experiments

Test documentation lives with the suite: [tests/README.md](../tests/README.md)
(runner notes) and [tests/TEST_PLAN.md](../tests/TEST_PLAN.md) (coverage plan).

## Key Concepts

### Interaction Modes

1. **Neighborhood Mode** (default) - Agents interact only with network neighbors
2. **Aggregate Pairwise Mode** - Agents play against all others with one decision per round
3. **True Pairwise Mode** - Agents make individual decisions for each opponent

### Agent Types

- **Reactive Strategies**: TFT, GTFT, STFT, Pavlov, TF2T
- **Learning Agents**: Q-Learning, Hysteretic Q, LRA-Q, Wolf-PHC, UCB1
- **Simple Strategies**: Always Cooperate, Always Defect, Random

### Network Types

- Fully Connected
- Small World (Watts-Strogatz)
- Scale-Free (Barabási-Albert)

## Quick Start

See the [main README](../README.md) for installation and usage instructions.
