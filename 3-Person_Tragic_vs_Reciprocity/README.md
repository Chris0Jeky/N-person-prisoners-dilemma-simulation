# 3-Person Tragic vs Reciprocity

This directory contains the **NPD Simulator** - a modular framework for N-Person Prisoner's Dilemma experiments.

## Directory Contents

- **`npd_simulator/`** - The main framework for running N-Person PD experiments
  - Supports any number of agents (not limited to 3)
  - Modular architecture with clear separation of concerns
  - Configuration-based experiments
  - Advanced features: batch processing, comparative analysis, parameter sweeps
  - See `npd_simulator/README.md` for full documentation

- **`pictures/`** - Historical results and figures from previous experiments

- **`MIGRATION_GUIDE.md`** - Guide for migrating from the old flat structure to the new framework

- **`REORGANIZATION_SUMMARY.md`** - Summary of the reorganization process

## Quick Start

```bash
# Run the demo
python npd_simulator/demo.py

# Run a single experiment
python npd_simulator/main.py single -c npd_simulator/configs/example_config.json

# Run Q-learning tests
python npd_simulator/main.py batch -c npd_simulator/configs/qlearning_tests.json

# Compare across different agent counts
python npd_simulator/main.py compare -c npd_simulator/configs/comparative_analysis.json -n 3 5 10 20
```

## Key Features

- **Scalable**: Test with any number of agents
- **Modular**: Easy to add new agent types and strategies
- **Configurable**: JSON-based experiment configuration
- **Comprehensive**: Built-in visualization and reporting
- **Advanced**: Parameter optimization and comparative analysis

For detailed documentation, see `npd_simulator/README.md`.