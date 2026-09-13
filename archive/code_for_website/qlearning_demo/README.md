# Q-Learning Demo for N-Person Prisoner's Dilemma

This package contains Q-learning agents that learn to play the Prisoner's Dilemma game in both pairwise and N-person settings.

## Overview

The Q-learning implementation features:
- **Richer state space**: Tracks last 2 rounds of history for both self and opponents
- **Optimistic initialization**: Q-values start at 0.5 for cooperation, 0.3 for defection
- **Trend detection**: In N-person mode, tracks if cooperation is increasing/decreasing
- **Two experiment types**:
  - 2QL experiments: 2 Q-learners vs 1 other agent (AllC, AllD, TFT, TFT-E, Random)
  - 1QL experiments: 1 Q-learner in all possible 3-agent combinations

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The code will work without all dependencies but with reduced functionality:
- Without numpy: Uses built-in functions for calculations
- Without pandas: Basic CSV output format
- Without matplotlib/seaborn: Skips plot generation

## Files

- `run_qlearning_demo.py` - Main script to run experiments
- `qlearning_demo_generator.py` - Core simulation and Q-learning logic
- `qlearning_agents.py` - Additional Q-learning agent implementations
- `requirements.txt` - Python dependencies

## Running the Demo

Basic run:
```bash
python run_qlearning_demo.py
```

The script uses these default parameters:
- NUM_ROUNDS = 1000 (rounds per simulation)
- NUM_RUNS = 500 (simulations per experiment)
- TRAINING_ROUNDS = 0 (pre-training rounds)

To modify parameters, edit the variables at the top of `run_qlearning_demo.py`.

## Output

The script creates a `qlearning_results/` directory with:

```
qlearning_results/
├── 2QL_experiments/
│   ├── csv/              # Detailed cooperation and score data
│   └── figures/          # Plots (if matplotlib available)
└── 1QL_experiments/
    ├── csv/              # Detailed cooperation and score data
    └── figures/          # Plots (if matplotlib available)
```

## Q-Learning Parameters

Default Q-learning parameters (in `qlearning_demo_generator.py`):
- learning_rate = 0.15
- discount_factor = 0.95
- epsilon = 0.2 (exploration rate)
- epsilon_decay = 0.995
- epsilon_min = 0.05

## State Representation

**Pairwise mode**: 
- Format: "MCD_OCC" = "My moves: Cooperate then Defect, Opponent: Cooperate twice"
- Tracks last 2 rounds of interaction history

**N-person mode**:
- Categorizes cooperation ratio (low/medium/high)
- Detects trends (up/down/stable)
- Includes agent's recent behavior