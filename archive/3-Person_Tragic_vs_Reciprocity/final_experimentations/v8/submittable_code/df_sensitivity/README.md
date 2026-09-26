# Discount Factor Sensitivity Analysis

This code performs a sensitivity analysis on the effect of different discount factor (DF) values on cooperation rates between Q-Learning agents and Tit-for-Tat (TFT) agents in both pairwise and n-person game settings.

## Overview

The analysis tests how varying the discount factor affects cooperation dynamics in the following scenarios:
- 2 Q-Learning agents vs 1 TFT agent
- 1 Q-Learning agent vs 2 TFT agents

Two variants of Q-Learning agents are tested:
- **LegacyQL**: Uses 2-round history tracking
- **Legacy3RoundQL**: Uses 3-round history tracking

Discount factor values tested: 0.99, 0.95, 0.9, 0.7, 0.4, 0.0

## Files

- `agents.py`: Agent implementations (TFT, LegacyQL, Legacy3RoundQL)
- `config.py`: Configuration parameters for agents and simulations
- `df_sensitivity_analysis.py`: Main script that runs the analysis
- `README.md`: This file

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib

Install dependencies:
```bash
pip install numpy pandas matplotlib
```

## Running the Analysis

Simply run the main script:

```bash
python df_sensitivity_analysis.py
```

## Output

The script will create a `df_sensitivity_results/` directory containing:

1. **CSV Files**:
   - Individual CSV files for each DF value and scenario in subdirectories (`df_0.99/`, `df_0.95/`, etc.)
   - A summary CSV file with final cooperation rates across all DF values

2. **Figures**:
   - Comparison plots for each scenario showing all DF values
   - A special figure focusing on the "2 Legacy3Round QL vs 1 TFT" scenario

## Configuration

You can modify simulation parameters in `config.py`:
- `num_rounds`: Number of rounds per simulation (default: 20000)
- `num_runs`: Number of runs to average over (default: 200)

## What the Code Does

1. For each discount factor value (0.99, 0.95, 0.9, 0.7, 0.4, 0.0):
   - Runs 4 scenarios (2QL vs 1TFT and 1QL vs 2TFT for both Legacy and Legacy3Round)
   - Each scenario is run for multiple simulations and averaged
   - Tracks cooperation rates separately for QL and TFT groups

2. Generates visualizations showing:
   - How cooperation rates evolve over time for different DF values
   - Both pairwise and neighborhood (n-person) game results
   - Smoothed data for clearer trends

3. Saves detailed results:
   - Time series data for each round
   - Summary statistics (final cooperation rates)

## Expected Runtime

With default settings (20000 rounds, 200 runs), the full analysis takes approximately 30-60 minutes depending on your hardware.

## Understanding the Results

- **Higher DF values** (closer to 1) mean agents value future rewards more
- **Lower DF values** (closer to 0) mean agents focus on immediate rewards
- The figures show how this affects cooperation dynamics between Q-Learning and TFT agents