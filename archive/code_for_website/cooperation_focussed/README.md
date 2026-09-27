# Cooperation-Focused Analysis

This package contains scripts for measuring cooperation between Q-Learning agents and Tit-for-Tat (TFT) agents.

## Overview

The main script `cooperation_measurement.py` runs experiments focusing specifically on cooperation dynamics between:
- Legacy Q-learners (2-round history)
- Legacy3Round Q-learners (3-round history)
- TFT agents

## Scenarios

1. **2 QL vs 1 TFT**: Two Q-learners compete against one TFT agent
2. **1 QL vs 2 TFT**: One Q-learner competes against two TFT agents

Both Legacy and Legacy3Round variants are tested in each scenario.

## Usage

To run the cooperation measurement experiments:

```bash
python cooperation_measurement.py
```

## Output

The script generates:
- Individual plots for each scenario showing cooperation rates over time
- A combined plot showing all scenarios together
- CSV files with detailed time series data
- Summary statistics CSV file

All outputs are saved to the `cooperation_measurement_results/` directory.

## Configuration

Simulation parameters can be adjusted in `config.py`:
- `num_rounds`: Number of rounds per simulation (default: 20,000)
- `num_runs`: Number of runs to average over (default: 200)

## Requirements

- numpy
- matplotlib
- pandas
- Python 3.6+