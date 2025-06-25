# Running All Experiments

This directory contains multiple standalone experiments for analyzing game theory strategies in the Prisoner's Dilemma. A master script is provided to run all experiments in sequence.

## Quick Start

To run all experiments:
```bash
python run_all_experiments.py
```

## What Gets Run

The master script runs the following experiments in order:

1. **Static Figure Generator (Full)** - Analyzes cooperation dynamics with TFT variants
2. **Static Figure Generator (2 TFT-E Only)** - Focused analysis on 2 TFT-E agents
3. **Q-Learning Demo** - Q-learning agents in various configurations
4. **Save Configuration** - Saves experiment configurations
5. **Final Demo Full** - Comprehensive final analysis
6. **Cooperation Measurement** - Detailed cooperation analysis
7. **Discount Factor Sensitivity** - Analyzes sensitivity to discount factor

## Time Estimates

Running all experiments may take considerable time:
- Static experiments: ~10-30 minutes each
- Q-learning experiments: ~30-60 minutes
- Analysis scripts: ~5-15 minutes each

**Total estimated time: 1-3 hours** (depending on your hardware)

## Output Locations

Each experiment creates its own output directory:
- `static_figure_generator/results/`
- `static_figure_generator_2tfte_allc_alld/results_2tfte_only/`
- `qlearning_demo/qlearning_results/`
- `main_runs/results/`
- `cooperation_focussed/results/`
- `df_sensitivity/results/`

## Running Individual Experiments

You can also run experiments individually by navigating to their directories:

```bash
cd static_figure_generator
python static_figure_generator.py
```

## Requirements

Make sure you have installed all requirements:
```bash
pip install numpy matplotlib seaborn pandas
```

Note: Some experiments will work with reduced functionality if certain packages are missing.

## Troubleshooting

If an experiment fails:
1. Check the error message in the console output
2. Ensure all dependencies are installed
3. Try running the failed experiment individually
4. Check that you have write permissions for output directories

## Customizing Parameters

To modify experiment parameters:
1. Run experiments individually instead of using the master script
2. Edit the configuration variables at the top of each script
3. Common parameters to adjust:
   - `NUM_ROUNDS`: Number of game rounds
   - `NUM_RUNS`: Number of simulation runs
   - `TRAINING_ROUNDS`: Pre-training rounds (for Q-learning)