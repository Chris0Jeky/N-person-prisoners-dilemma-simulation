# Enhanced Experiment Runner with Comprehensive Logging and Advanced Visualizations

## Overview
This enhanced version of the experiment runner adds comprehensive data logging and sophisticated visualization capabilities to analyze cooperation dynamics in the N-Person Iterated Prisoner's Dilemma.

## Key Enhancements

### 1. Comprehensive Data Logging
- **Agent-Level Logging**: Each agent now tracks:
  - Cumulative scores over time
  - Individual moves (cooperation/defection) for each interaction
  - Episode-based cooperation rates
  - Detailed interaction logs with opponents, payoffs, and round numbers

- **Global-Level Logging**: The simulation tracks:
  - Overall cooperation rates per round
  - Detailed interaction logs for all agent pairs
  - Episode-based cooperation summaries
  - Round-by-round cooperation dynamics

### 2. Advanced Visualizations

#### Evolution of Cooperation Comparison
- Shows raw and smoothed cooperation trajectories for Pairwise vs N-Person models
- Includes episode-average cooperation rates with trend lines
- Features a conceptual model diagram illustrating "Reciprocity Hill" vs "Tragic Valley"

#### TFT Performance Analysis
- Compares TFT agent performance across different experimental conditions
- Shows impact of exploration rates on cooperation
- Analyzes episodic vs non-episodic modes (Pairwise)
- Compares pTFT vs pTFT-Threshold variants (N-Person)

#### Cooperation Dynamics Heatmap
- Visualizes cooperation patterns over time for different compositions
- Shows how cooperation evolves differently in various experimental conditions
- Uses color coding to highlight cooperation levels

#### Strategy Emergence Patterns
- Tracks individual agent trajectories within each composition
- Shows how cooperation patterns emerge differently across agent compositions
- Includes both individual and overall cooperation dynamics

#### Summary Statistics Table
- Comprehensive table of all experimental results
- Includes average TFT scores, cooperation rates, and stability metrics
- Color-coded cells for easy identification of high/low cooperation scenarios

## Usage

Run the enhanced experiments with:
```bash
python run_enhanced_experiments.py
```

Or directly:
```bash
python enhanced_experiment_runner.py
```

## Output Files
The script generates the following visualization files:
- `evolution_of_cooperation_comparison.png`: Main comparison plot similar to your reference image
- `tft_performance_analysis.png`: Multi-panel analysis of TFT performance
- `cooperation_dynamics_heatmap.png`: Heatmaps showing cooperation over time
- `strategy_emergence_patterns.png`: Individual agent trajectory analysis
- `summary_statistics_table.png`: Comprehensive results table

## Key Differences from Original Implementation

1. **No AllD Score Tracking**: As requested, visualizations focus on TFT agents only
2. **Time Series Focus**: Emphasis on temporal dynamics rather than just final outcomes
3. **Episode Analysis**: Special attention to episodic structure in Pairwise experiments
4. **Smoothing Techniques**: Uses Savitzky-Golay filtering for cleaner trend visualization
5. **Comparative Framework**: Direct comparisons between Pairwise and N-Person models

## Technical Notes

- Uses pandas for data manipulation
- matplotlib and seaborn for visualizations
- scipy for data smoothing
- numpy for numerical operations

## Experiment Parameters

- **Exploration Rates**: 0%, 10%
- **Agent Compositions**: 
  - 3 TFTs
  - 2 TFTs + 1 AllD
  - 2 TFTs + 3 AllD
- **Pairwise Settings**: 100 rounds total, episodic (10 episodes) and non-episodic
- **N-Person Settings**: 200 rounds, pTFT and pTFT-Threshold variants
