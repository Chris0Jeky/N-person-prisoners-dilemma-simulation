# Q-Learning vs TFT Comprehensive Experiment

This experiment tests Vanilla and Adaptive Q-learning agents against Tit-for-Tat (TFT) agents in various configurations, exploring the impact of different discount factors on cooperation and performance.

## Overview

The experiment includes 16 scenarios testing:
- **Agent Types**: Vanilla Q-Learning and Adaptive Q-Learning
- **Discount Factors**: 0.6 (short-term focus) and 0.95 (long-term focus)
- **Configurations**: 
  - 1 QL vs 2 TFT
  - 1 QL vs 2 TFT-E (10% error rate)
  - 2 QL vs 1 TFT
  - 2 QL vs 1 TFT-E

## Requirements

- Python 3.7+
- numpy
- matplotlib
- seaborn
- pandas

Install requirements:
```bash
pip install numpy matplotlib seaborn pandas
```

## Running the Experiment

Simply run:
```bash
python tft_experiment.py
```

The experiment will:
1. Run all 16 scenarios (25 runs of 10,000 rounds each)
2. Generate individual plots for each scenario
3. Save CSV files with detailed results
4. Create a comparison heatmap
5. Generate discount factor comparison plots
6. Save summary statistics in JSON format

## Output Structure

```
tft_experiment_results/
├── figures/                    # Individual scenario plots
│   ├── 1_Vanilla_DF06_vs_2_TFT.png
│   ├── 1_Vanilla_DF095_vs_2_TFT.png
│   └── ... (16 total plots)
├── csv_files/                  # Detailed results
│   ├── 1_Vanilla_DF06_vs_2_TFT_pairwise.csv
│   ├── 1_Vanilla_DF06_vs_2_TFT_neighborhood.csv
│   └── ... (32 total CSV files)
├── comparison_heatmap.png      # Overall performance comparison
├── discount_factor_comparison.png  # DF effect visualization
└── summary_statistics.json     # Aggregated statistics
```

## Key Insights

The experiment reveals:
1. **Discount Factor Impact**: Higher discount factors (0.95) generally lead to more cooperation
2. **Agent Adaptability**: Adaptive Q-learners perform better against noisy TFT-E agents
3. **Configuration Effects**: 2 QL agents often struggle to coordinate compared to 1 QL agent

## Configuration Details

### Vanilla Q-Learning Parameters
- Learning Rate (α): 0.08
- Exploration Rate (ε): 0.1
- Discount Factor (γ): 0.6 or 0.95

### Adaptive Q-Learning Parameters
- Initial Learning Rate: 0.1 (adapts between 0.03-0.15)
- Initial Exploration Rate: 0.15 (adapts between 0.02-0.15)
- Adaptation Factor: 1.08
- Reward Window Size: 75
- Discount Factor (γ): 0.6 or 0.95

### Game Parameters
- Prisoner's Dilemma Payoffs: T=5, R=3, P=1, S=0
- TFT-E Error Rate: 10%

## Interpreting Results

Each plot shows:
- **Top Left**: Pairwise cooperation rates
- **Top Right**: Pairwise cumulative scores
- **Bottom Left**: Neighborhood cooperation rates
- **Bottom Right**: Neighborhood cumulative scores

The heatmap shows normalized performance across all scenarios, making it easy to compare different configurations.

## Customization

To modify the experiment:
- Adjust `NUM_ROUNDS` for shorter/longer simulations
- Change `NUM_RUNS` for more/fewer averaging runs
- Modify agent parameters in the configuration dictionaries
- Add new scenarios in the `scenarios` dictionary

## Citation

If you use this code, please cite:
```
Q-Learning vs TFT Experiment
3-Person Tragic vs Reciprocity Project
2024
```