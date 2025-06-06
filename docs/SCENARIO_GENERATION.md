# Scenario Generation and Analysis Guide

This document provides detailed information about the scenario generation and analysis tools in the NPDL framework.

## Overview

The scenario generation system allows you to:

1. Automatically generate diverse scenarios by sampling from parameter pools
2. Evaluate scenarios based on metrics that measure "interestingness"
3. Select the most promising scenarios for deeper analysis
4. Run and save detailed results for selected scenarios
5. Visualize and compare results across different scenarios

This approach helps you efficiently explore the vast parameter space of Prisoner's Dilemma simulations and identify scenarios that exhibit interesting dynamics or behaviors.

## Getting Started

### Quick Start

To run the complete workflow (generation, evaluation, selection, analysis):

```bash
python run_sweep_analysis.py
```

This will:
1. Generate 30 random scenarios
2. Evaluate each with 3 quick runs
3. Select the top 5 most interesting scenarios
4. Run each selected scenario 10 times with full logging
5. Generate visualization and analysis in the `analysis_results` directory

### Custom Run

For more control over the process:

```bash
python run_sweep_analysis.py --num_generate 50 --eval_runs 3 --save_runs 15 --top_n 8 --results_dir "results/my_scenario_sweep" --analysis_dir "my_analysis_results"
```

## Components

### 1. Scenario Generator (`run_scenario_generator.py`)

This script:
- Generates random scenarios by sampling from parameter pools
- Runs brief simulations to evaluate each scenario
- Calculates metrics to measure "interestingness"
- Ranks scenarios by a composite score
- Selects and saves the top N scenarios for detailed analysis

Key functions:
- `generate_random_scenario()`: Creates a single random scenario configuration
- `evaluate_scenario_run()`: Calculates metrics for a single simulation run
- `calculate_interestingness_score()`: Computes a composite score based on multiple metrics
- `run_scenario_generation()`: Main function that orchestrates the entire process

### 2. Sweep Visualizer (`analysis/sweep_visualizer.py`)

This module:
- Loads metadata about generated scenarios
- Creates visualizations comparing different scenarios
- Generates a comprehensive analysis report

Key functions:
- `load_scenario_metadata()`: Loads the JSON metadata file
- `extract_scenario_data()`: Converts metadata to a DataFrame for analysis
- `plot_scenario_scores()`: Visualizes scenario rankings
- `plot_metric_comparison()`: Compares specific metrics across scenarios
- `plot_metric_correlation()`: Shows correlations between different metrics
- `plot_strategy_distribution()`: Visualizes strategy distributions
- `create_scenario_comparison_report()`: Generates a complete report

### 3. Complete Workflow (`run_sweep_analysis.py`)

Combines both tools into a seamless workflow for scenario exploration.

## Evaluation Metrics

The generator evaluates scenarios based on several metrics:

1. **Final Cooperation Rate**: The proportion of agents that cooperate in the final round
2. **Cooperation Rate Change**: How much the cooperation rate changes from first to last round
3. **Score Variance**: Variance in average scores across different strategies
4. **Strategy Dominance**: Difference between the best and worst performing strategies
5. **Network Clustering**: Clustering coefficient of the network structure

These metrics are combined into an "interestingness score" that favors:
- Scenarios with intermediate cooperation levels (not all cooperate or all defect)
- Dynamic scenarios where cooperation rates change significantly over time
- Scenarios where different strategies achieve different outcomes

## Customizing the Process

### Modifying Parameter Pools

To explore different parts of the parameter space, modify the parameter pools at the top of `run_scenario_generator.py`:

```python
AGENT_STRATEGY_POOL = [
    "hysteretic_q", "wolf_phc", "lra_q",  # Add or remove strategies
    "q_learning", "q_learning_adaptive", "ucb1_q",
    # ...
]

NETWORK_POOL = [
    ("small_world", {"k": 4, "beta": 0.3}),  # Add or modify network parameters
    # ...
]

NUM_AGENTS_POOL = [20, 30, 40, 50]  # Modify number of agents
# ...
```

### Customizing Interestingness Metrics

To change what makes a scenario "interesting," modify the `calculate_interestingness_score()` function in `run_scenario_generator.py`:

```python
def calculate_interestingness_score(eval_result):
    """Calculate an 'interestingness' score based on multiple metrics."""
    # Extract metrics
    metrics = eval_result['metrics']
    
    # Get normalized metrics (0-1 range)
    coop = metrics.get('avg_final_coop_rate', 0)
    coop_change = abs(metrics.get('avg_coop_rate_change', 0))
    variance = metrics.get('avg_score_variance', 0) / 1000
    
    # Adjust weights to prioritize different aspects
    score = (
        0.2 * (1 - abs(coop - 0.5)) +  # Intermediate cooperation levels
        0.4 * coop_change +            # Dynamic behavior (change over time)
        0.4 * variance                 # Strategy performance differences
    )
    
    return score
```

### Adding New Evaluation Metrics

To add new metrics, modify the `evaluate_scenario_run()` function in `run_scenario_generator.py`:

```python
def evaluate_scenario_run(env, round_results) -> Dict[str, float]:
    """Calculate metrics to judge if a scenario run is 'interesting'."""
    metrics = {
        'final_coop_rate': np.nan, 
        'coop_rate_change': np.nan,
        'score_variance': np.nan, 
        'my_new_metric': np.nan,  # Add your new metric
        # ...
    }
    
    # Calculate your new metric
    if round_results:
        # Implementation of your metric calculation
        metrics['my_new_metric'] = calculate_my_metric(env, round_results)
    
    return metrics
```

Then make sure to include it in the interestingness score calculation.

## Output Files

The scenario generation process produces several files:

1. **Metadata File** (`generated_scenarios_metadata.json`):
   - Contains information about all generated scenarios
   - Includes evaluation metrics and scores
   - Used by the visualization tools for analysis

2. **Selected Scenarios File** (`selected_scenarios.json`):
   - Contains the full configurations of the top N selected scenarios
   - Can be used to rerun these specific scenarios later

3. **Detailed Results** (in subdirectories):
   - For each selected scenario, detailed CSV files with round-by-round data
   - Network structure information (JSON)
   - Agent-level statistics

4. **Analysis Results** (in analysis directory):
   - Various visualizations comparing scenarios
   - CSV file with processed comparison data

## Advanced Usage

### Running Specific Scenarios

To run specific scenarios after selection:

```python
import json
from main import setup_experiment, save_results
import logging

# Load selected scenarios
with open("results/generated_scenarios/selected_scenarios.json", "r") as f:
    scenarios = json.load(f)

# Run a specific scenario
selected_scenario = scenarios[0]  # First selected scenario
logger = logging.getLogger("scenario_runner")
logger.setLevel(logging.INFO)

env, _ = setup_experiment(selected_scenario, logger)
results = env.run_simulation(
    selected_scenario["num_rounds"],
    logging_interval=20
)

# Analyze results
# ...
```

### Parallel Scenario Evaluation

For large scenario sweeps, you may want to parallelize the evaluation process. This requires adding multiprocessing to the `run_scenario_generator.py` script.

### Evolutionary Scenario Selection

Instead of just generating random scenarios, you could implement an evolutionary approach that:
1. Generates an initial population of scenarios
2. Evaluates and selects the most interesting ones
3. Creates new scenarios by combining and mutating the selected ones
4. Repeats the process for multiple generations

This would require modifying the `run_scenario_generator.py` script to add mutation, crossover, and generation tracking.

## Example Analysis Workflow

1. **Generate a diverse set of scenarios**:
   ```bash
   python run_scenario_generator.py --num_generate 100 --eval_runs 3 --top_n 10
   ```

2. **Examine the metadata to understand what makes scenarios interesting**:
   ```python
   import json
   with open("results/generated_scenarios/generated_scenarios_metadata.json", "r") as f:
       metadata = json.load(f)
   
   for scenario in metadata["scenarios"][:10]:  # Look at top 10
       print(f"Name: {scenario['name']}")
       print(f"Score: {scenario.get('selection_score', 0):.3f}")
       print(f"Metrics: {scenario['metrics']}")
       print(f"Config: {scenario['config_summary']}")
       print()
   ```

3. **Generate visualizations**:
   ```bash
   python -m analysis.sweep_visualizer --metadata "results/generated_scenarios/generated_scenarios_metadata.json"
   ```

4. **Refine parameter pools based on findings**:
   - Update the parameter pools in `run_scenario_generator.py` to focus on the most promising areas
   - Adjust weights in the interestingness score calculation

5. **Run a more focused sweep**:
   ```bash
   python run_sweep_analysis.py --num_generate 50 --results_dir "results/refined_sweep"
   ```

## Troubleshooting

### Scenarios Not Diverse Enough

If your generated scenarios are too similar:
- Expand the parameter pools to include more options
- Add randomization to parameter values rather than just selecting from pools
- Modify the scoring function to penalize similarity to previously selected scenarios

### No Scenarios Meet Criteria

If no scenarios are deemed "interesting":
- Lower the threshold for selection
- Adjust the interestingness scoring function
- Check if the evaluation metrics are being calculated correctly
- Increase the number of scenarios to generate

### Slow Evaluation Process

If the evaluation is too slow:
- Reduce the number of evaluation runs per scenario
- Reduce the number of rounds per evaluation run
- Implement parallel evaluation using multiprocessing
- Create a more efficient version of the evaluation function

## Best Practices

1. **Start small and iterate**:
   - Begin with a small number of scenarios and runs
   - Analyze the results and refine your approach
   - Gradually increase the scale of your exploration

2. **Balance exploration and exploitation**:
   - Initially, use wide parameter ranges to explore the space
   - As you learn more, focus on promising regions
   - Keep some randomness to avoid getting stuck in local optima

3. **Save intermediate results**:
   - Save metadata and results at each stage
   - This allows you to resume exploration if needed
   - You can also compare results across different sweeps

4. **Document your findings**:
   - Take notes on what parameter combinations lead to interesting behaviors
   - Look for patterns across successful scenarios
   - Use these insights to inform future experiments