# Static Style Implementation for NPD Simulator

## Overview
This implementation adds static_figure_generator.py style visualization capabilities to the NPD simulator, enabling publication-ready figures with statistical aggregation across multiple runs.

## Key Components Added

### 1. **static_style_visualizer.py**
- Main visualization module that generates figures matching static_figure_generator.py style
- Features:
  - 2x2 grid layouts comparing 4 experiments per figure
  - Statistical aggregation (mean, std, 95% CI)
  - Smoothed trend lines overlaid on raw data
  - Individual run traces shown faded in background
  - Generates all 6 figure types from static generator

### 2. **enhanced_game_state.py**
- Extended game state tracking for agent-type specific metrics
- Tracks:
  - Per-agent-type cooperation rates
  - Per-agent-type cumulative scores
  - Round-by-round history with agent type information
  - Enhanced results format for visualization

### 3. **static_style_runner.py**
- Experiment runner designed for static-style experiments
- Features:
  - Runs standard experiment configurations (3 TFT, 2 TFT-E + 1 AllD, etc.)
  - Executes multiple runs (default 20) per configuration
  - Supports both pairwise and N-person games
  - Automatic visualization generation
  - Batch result saving

### 4. **test_static_style.py**
- Testing script to verify implementation
- Two test modes:
  - Uses existing demo results
  - Runs new small-scale experiments

## Usage

### Running Standard Experiments
```python
from experiments.runners.static_style_runner import StaticStyleRunner

# Create runner with desired parameters
runner = StaticStyleRunner(
    output_dir="results/static_style",
    num_runs=20,  # Number of runs per experiment
    num_rounds=100,  # Rounds for N-person games
    rounds_per_pair=100  # Rounds for pairwise games
)

# Run all standard experiments
results = runner.run_all_experiments()
```

### Using Existing Results
```python
from static_style_visualizer import StaticStyleVisualizer

# Create visualizer
visualizer = StaticStyleVisualizer(output_dir)

# Prepare results dictionary
# Keys: experiment names ("3 TFT", "2 TFT-E + 1 AllD", etc.)
# Values: list of result dictionaries from multiple runs
experiment_results = {
    "3 TFT": [result1, result2, ...],
    "2 TFT-E + 1 AllD": [result1, result2, ...],
    # ...
}

# Generate all figures
visualizer.generate_all_figures(experiment_results)
```

## Figure Types Generated

1. **Figure 1**: Pairwise TFT Cooperation
2. **Figure 2**: Neighbourhood TFT Cooperation  
3. **Figure 3**: Pairwise Agent Scores
4. **Figure 4**: Neighbourhood Agent Scores
5. **Figure 5**: Pairwise All Agent Cooperation
6. **Figure 6**: Neighbourhood All Agent Cooperation

Each figure contains 4 subplots in a 2x2 grid showing different experimental conditions.

## Key Differences from Original NPD Simulator

1. **Multiple Runs**: Runs each experiment 20 times for statistical significance
2. **Statistical Visualization**: Shows confidence intervals and smoothed trends
3. **Agent-Type Tracking**: Separately tracks metrics for different agent types
4. **Publication Style**: Uses whitegrid theme with professional formatting
5. **Comparative Layout**: 2x2 grid for easy comparison across conditions

## Integration with Existing Code

The implementation is designed to work alongside existing NPD simulator functionality:
- Uses existing agent implementations
- Compatible with current game mechanics
- Extends rather than replaces existing visualization
- Maintains backward compatibility

## Output Structure

```
results/static_style_TIMESTAMP/
├── figures/
│   ├── figure1_pairwise_tft_cooperation.png
│   ├── figure2_neighbourhood_tft_cooperation.png
│   ├── figure3_pairwise_agent_scores.png
│   ├── figure4_neighbourhood_agent_scores.png
│   ├── figure5_pairwise_all_cooperation.png
│   ├── figure6_neighbourhood_all_cooperation.png
│   └── nperson/
│       └── ... (N-person specific figures)
├── batches/
│   ├── pairwise_3 TFT/
│   │   ├── run_000.json
│   │   ├── run_001.json
│   │   └── ...
│   └── ...
└── csv/
    └── ... (summary statistics)
```