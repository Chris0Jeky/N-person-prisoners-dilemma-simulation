# Main Experimental Framework

This package contains the comprehensive simulation framework for comparing Q-learning agent variants in prisoner's dilemma games.

## Overview

The main script `final_demo_full.py` runs extensive experiments comparing different Q-learning strategies:
- Hysteretic Q-learning (asymmetric learning rates)
- Legacy Q-learning (2-round history with sophisticated state representation)
- Legacy3Round Q-learning (3-round history with extended state tracking)

## Game Types

1. **Pairwise Games**: Direct 1-on-1 interactions between agents
2. **N-Person Games**: Neighborhood interactions where all agents play simultaneously

## Scenarios

The framework tests Q-learners against various opponent strategies:
- Always Cooperate (AllC)
- Always Defect (AllD)
- Random (50/50 cooperation)
- Tit-for-Tat (TFT)
- TFT with Error (TFT-E, 10% error rate)

Each scenario is run in two configurations:
1. **2 QL vs 1 Opponent**: Two Q-learners compete against one static agent
2. **1 QL vs 2 Opponents**: One Q-learner competes against two static agents

## Usage

To run the full experimental suite:

```bash
python final_demo_full.py
```

The script supports parallel processing and will automatically use available CPU cores.

## Output

The script generates:
- Individual plots for each scenario (4-panel comparison)
- Summary heatmap comparing all agents
- CSV files with detailed results
- Configuration files documenting all parameters

All outputs are saved to the `final_comparison_charts/` directory.

## Configuration

Key parameters can be adjusted in `config.py`:
- `num_rounds`: Number of rounds per simulation (default: 20,000)
- `num_runs`: Number of runs to average over (default: 200)
- Learning parameters for each Q-learner variant

## Requirements

- numpy
- matplotlib
- seaborn
- pandas
- Python 3.6+
- multiprocessing support for parallel execution