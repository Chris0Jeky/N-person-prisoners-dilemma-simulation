# NPDL Documentation

## Overview

This directory contains documentation for the N-Person Prisoner's Dilemma Learning (NPDL) framework.

## Documentation Structure

- **[implementation/](implementation/)** - Technical implementation details
  - `PAIRWISE_MODE.md` - Pairwise interaction modes (aggregate and true pairwise)
  
- **[testing/](testing/)** - Testing documentation
  - Test suite overview and coverage reports
  
- **[SCENARIO_GENERATION.md](SCENARIO_GENERATION.md)** - Guide to scenario generation and analysis tools

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