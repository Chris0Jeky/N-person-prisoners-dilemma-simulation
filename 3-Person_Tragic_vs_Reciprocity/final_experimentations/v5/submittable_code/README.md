# Submittable Code Packages

This directory contains two clean, independently runnable code packages for prisoner's dilemma experiments.

## Packages

### 1. cooperation_focussed/
A focused analysis package for measuring cooperation between Q-Learning agents and Tit-for-Tat agents.

**Main script**: `cooperation_measurement.py`

**Purpose**: Specifically analyzes cooperation dynamics between Legacy/Legacy3Round Q-learners and TFT agents in various configurations.

**To run**:
```bash
cd cooperation_focussed
python cooperation_measurement.py
```

### 2. main_runs/
The comprehensive experimental framework for comparing multiple Q-learning variants.

**Main script**: `final_demo_full.py`

**Purpose**: Runs extensive experiments comparing Hysteretic, Legacy, and Legacy3Round Q-learners against various opponent strategies in both pairwise and n-person games.

**To run**:
```bash
cd main_runs
python final_demo_full.py
```

## Requirements

Both packages require:
- Python 3.6+
- numpy
- matplotlib
- pandas
- seaborn (for main_runs only)

## Installation

```bash
pip install numpy matplotlib pandas seaborn
```

## Structure

Each package is self-contained with:
- All necessary agent implementations
- Configuration files
- Main execution scripts
- README documentation

The code has been cleaned and organized for clarity and professional presentation.