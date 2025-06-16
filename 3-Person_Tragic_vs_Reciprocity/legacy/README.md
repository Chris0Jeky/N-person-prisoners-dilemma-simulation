# N-Person Prisoner's Dilemma: Q-Learning vs Traditional Strategies

This project analyzes cooperation dynamics in multi-agent prisoner's dilemma games, comparing Q-learning agents with traditional strategies like Tit-for-Tat.

## Overview

The simulation framework tests different agent strategies in both pairwise (2-player) and N-person (3+ player) prisoner's dilemma games. It includes:

- **Traditional Strategies**: TFT (Tit-for-Tat), TFT-E (TFT with exploration), AllC (Always Cooperate), AllD (Always Defect)
- **Learning Agents**: Basic Q-Learning and Enhanced Q-Learning (with optimistic initialization and better state representations)
- **Game Modes**: Pairwise interactions and N-person neighborhood games

## How to Run

### Step 1: Run Static Strategy Experiments
```bash
python static_figure_generator.py
```
This creates results in the `results/` folder showing cooperation dynamics for traditional strategies.

### Step 2: Run Basic Q-Learning Experiments
```bash
python run_qlearning_demo.py
```
Or directly:
```bash
python qlearning_demo_generator.py
```
This creates results in the `qlearning_results/` folder.

### Step 3: Run Enhanced Q-Learning Experiments
```bash
python enhanced_qlearning_demo_generator.py
```
This creates results in the `enhanced_qlearning_results/` folder.

### Step 4: Generate Comparative Analysis (Optional)
After running both Q-learning experiments:
```bash
python qlearning_comparative_analysis.py
```
This creates comparison plots in the `comparative_analysis/` folder.

## Results Structure

### `results/` - Static Strategy Results
- **CSV files**: Cooperation rates and scores over time
- **Figure 1-2**: TFT cooperation dynamics in pairwise vs neighborhood settings
- **Figure 3-4**: Agent scores over time
- **Figure 5-6**: All agent cooperation rates

### `qlearning_results/` - Basic Q-Learning
- **1QL_experiments/**: Single Q-learning agent vs various opponent pairs
- **2QL_experiments/**: Two Q-learning agents vs one opponent
- Each contains CSV data and PNG plots

### `enhanced_qlearning_results/` - Enhanced Q-Learning
- Same structure as basic Q-learning but with enhanced agents
- Enhanced agents use optimistic initialization and better state representations

### `comparative_analysis/` - Q-Learning Comparison
- Bar charts comparing basic vs enhanced Q-learning performance
- Heatmaps showing improvement percentages
- Summary statistics report

## Key Files

### Core Experiment Scripts
- `static_figure_generator.py` - Runs experiments with traditional strategies
- `qlearning_demo_generator.py` - Basic Q-learning experiments  
- `enhanced_qlearning_demo_generator.py` - Enhanced Q-learning experiments
- `qlearning_comparative_analysis.py` - Compares Q-learning variants

### Agent Implementations
- `qlearning_agents.py` - Basic Q-learning agents (SimpleQLearning, NPDLQLearning)
- `enhanced_qlearning_agents.py` - Enhanced Q-learning with advanced features

### Game Logic
- `main_pairwise.py` - 2-player prisoner's dilemma implementation
- `main_neighbourhood.py` - N-person prisoner's dilemma with linear payoffs

## Parameters

Default parameters (can be modified in the scripts):
- **Rounds**: 100 rounds per game
- **Runs**: 15 simulation runs for statistical averaging
- **Training**: 1000 rounds of pre-training for Q-learning agents
- **Players**: 3-agent groups (can be modified)

## Dependencies

- Python 3.7+
- numpy
- matplotlib
- seaborn
- pandas

## Understanding the Results

- **Cooperation Rate**: Percentage of cooperative actions (0.0 to 1.0)
- **Scores**: Cumulative payoffs based on prisoner's dilemma matrix
- **Confidence Intervals**: Shown as shaded regions (95% CI from 15 runs)
- **Q-Learning Performance**: Enhanced version typically shows ~20-40% improvement in cooperation stability

## Quick Start Example

To run a complete analysis:
```bash
# 1. Static strategies
python static_figure_generator.py

# 2. Q-learning experiments  
python run_qlearning_demo.py
python enhanced_qlearning_demo_generator.py

# 3. Compare results
python qlearning_comparative_analysis.py
```

All results will be saved in their respective directories with both data (CSV) and visualizations (PNG).