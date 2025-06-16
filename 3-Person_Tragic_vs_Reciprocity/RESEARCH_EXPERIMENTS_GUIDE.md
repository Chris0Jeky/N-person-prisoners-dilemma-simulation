# Research Experiments Guide

## Overview

This guide explains the comprehensive research experiments implementation for analyzing:
1. **TFT-centric tests**: Comparing pairwise vs neighbourhood models to demonstrate tragic valley vs reciprocity hill
2. **QLearning-centric tests**: Testing Q-learning agents in various 3-agent combinations

## Implementation Details

### Core Components

1. **`npd_simulator/experiments/research_experiments.py`**
   - Main experiment runner class `ResearchExperiments`
   - Handles both TFT and QLearning experiments
   - Generates CSV data in exact format matching existing results
   - Creates comparative analysis

2. **Agent Configurations**

#### TFT-Centric Experiments
- `3_TFT`: Three pure TFT agents
- `2_TFT-E__plus__1_AllD`: Two TFT with exploration + one defector
- `2_TFT__plus__1_AllD`: Two pure TFT + one defector
- `2_TFT-E__plus__1_AllC`: Two TFT with exploration + one cooperator

#### QLearning-Centric Experiments

**2 QL Agent Combinations:**
- `2_QL__plus__1_AllD`: Two Q-learners + always defect
- `2_QL__plus__1_AllC`: Two Q-learners + always cooperate
- `2_QL__plus__1_TFT`: Two Q-learners + TFT
- `2_QL__plus__1_TFT-E`: Two Q-learners + TFT with exploration
- `2_QL__plus__1_Random`: Two Q-learners + random
- Similar combinations with Enhanced Q-Learning (`2_EQL__plus__...`)

**1 QL Agent Combinations:**
- All possible combinations of 1 Q-learner with 2 other agents
- Example: `1_QL__plus__1_TFT__plus__1_AllD`
- Includes both basic and enhanced Q-learning variants

## Running the Experiments

### Full Research Suite
```bash
python3 run_research_experiments.py
```

This runs:
- 20 runs per experiment configuration
- 200 rounds per run
- Both pairwise and neighbourhood models for TFT experiments
- All QLearning combinations
- Saves results to `results/` directory

### Quick Test
```bash
python3 run_quick_test.py
```

This runs a reduced version:
- 3 runs per experiment
- 20 rounds per run
- Limited configurations
- Saves to `test_results/` directory

## Output Format

### CSV Files

1. **Aggregated Data Files**
   ```
   results/pairwise_tft_cooperation_3_TFT_aggregated.csv
   ```
   Columns: `Round, Mean_Cooperation_Rate, Std_Dev, Lower_95_CI, Upper_95_CI, Run_1, Run_2, ..., Run_20`

2. **Summary Files**
   ```
   results/pairwise_tft_cooperation_all_experiments_summary.csv
   ```
   Columns: `Round, 3_TFT_mean, 3_TFT_std, 2_TFT-E__plus__1_AllD_mean, ...`

3. **QLearning Files**
   ```
   results/qlearning_cooperation_2_QL__plus__1_AllD_aggregated.csv
   results/qlearning_2ql_cooperation_all_experiments_summary.csv
   results/qlearning_1ql_cooperation_all_experiments_summary.csv
   ```

### Analysis Files

**`results/comparative_analysis.json`**
Contains:
- TFT analysis showing tragic valley vs reciprocity hill
- QLearning adaptation success metrics
- Key findings and insights

### Visualization Files

Figures are generated using the static style visualizer:
- `figure1_pairwise_tft_cooperation.png`
- `figure2_neighbourhood_tft_cooperation.png`
- Additional figures for scores and overall cooperation

## Key Metrics Tracked

1. **Cooperation Rates**
   - Per-round cooperation rates
   - Agent-type specific cooperation (TFT agents only)
   - Statistical measures (mean, std, 95% CI)

2. **Learning Metrics**
   - Early vs late cooperation (first 50 vs last 50 rounds)
   - Learning improvement over time
   - Final agent scores

3. **Comparative Metrics**
   - Pairwise vs neighbourhood differences
   - Tragic valley identification (cooperation drop in N-person)
   - Reciprocity hill identification (cooperation rise in N-person)

## Understanding the Results

### Tragic Valley vs Reciprocity Hill

- **Tragic Valley**: Cooperation is lower in N-person games compared to pairwise
  - Often seen with mixed strategies (e.g., 2 TFT + 1 AllD)
  - Indicates difficulty of maintaining cooperation in groups

- **Reciprocity Hill**: Cooperation is higher in N-person games
  - Can occur with certain agent combinations
  - Indicates emergent cooperation in groups

### QLearning Adaptation

The analysis tracks:
- How quickly QL agents learn to cooperate/defect
- Whether they exploit other strategies effectively
- Differences between basic and enhanced Q-learning

## Customization

To modify experiments:

1. **Change parameters** in `ResearchExperiments.__init__`:
   ```python
   self.num_runs = 30  # More runs
   self.num_rounds = 500  # Longer experiments
   ```

2. **Add new agent combinations**:
   ```python
   self.other_agent_types = ["AllD", "AllC", "TFT", "TFT-E", "Random", "MyNewAgent"]
   ```

3. **Modify TFT configurations**:
   ```python
   self.tft_configs["new_config"] = [
       {"type": "TFT", "exploration_rate": 0.2},
       # ...
   ]
   ```

## Troubleshooting

1. **Import errors**: Run from the main directory, not from npd_simulator/
2. **Missing agents**: Ensure all agents are registered in the registry
3. **Memory issues**: Reduce num_runs or num_rounds
4. **File permissions**: Check write permissions for results directory

## Next Steps

After running experiments:
1. Review CSV files for detailed round-by-round data
2. Check comparative_analysis.json for key insights
3. Examine visualization figures for patterns
4. Use the data for further statistical analysis or publication