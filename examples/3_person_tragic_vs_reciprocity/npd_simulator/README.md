# N-Person Prisoner's Dilemma Simulator

A modular, extensible framework for simulating and analyzing N-Person Prisoner's Dilemma games with support for various agent strategies including reinforcement learning.

## Features

- **Flexible Agent Support**: Supports any number of agents (not limited to 3)
- **Multiple Game Modes**: N-Person (neighborhood) and Pairwise interactions
- **Rich Agent Types**: TFT variants, Always Cooperate/Defect, Random, Q-Learning, Enhanced Q-Learning
- **Experiment Management**: Single runs, batch experiments, comparative analysis, parameter sweeps
- **Comprehensive Analysis**: Automatic visualization, detailed reports, CSV exports
- **Modular Architecture**: Easy to extend with new agent types and analysis tools

## Quick Start

### Basic Usage

```bash
# Run a single experiment
python main.py single -c configs/example_config.json

# Run batch experiments
python main.py batch -c configs/qlearning_tests.json

# Run comparative analysis across different agent counts
python main.py compare -c configs/comparative_analysis.json -n 3 5 10 20

# Run parameter sweep
python main.py sweep -c configs/parameter_sweep.json
```

### Python API

```python
from npd_simulator import NPDSimulator
from npd_simulator.utils import load_config

# Create simulator
simulator = NPDSimulator(base_output_dir="results")

# Load and run configuration
config = load_config("configs/example_config.json")
results = simulator.run_single_experiment(config, experiment_type="npd")

# Run comparative analysis
analysis = simulator.run_comparative_analysis(config, agent_counts=[3, 5, 10, 20])
```

## Project Structure

```
npd_simulator/
├── core/                    # Core game mechanics
│   ├── game/               # Game implementations
│   │   ├── npd_game.py    # N-Person PD game
│   │   └── pairwise_game.py # Pairwise PD game
│   └── models/             # Data models
│       ├── payoff_matrix.py
│       └── game_state.py
├── agents/                  # Agent implementations
│   ├── base/               # Base agent classes
│   ├── strategies/         # Strategy-based agents
│   │   ├── tft.py         # TFT variants
│   │   ├── always.py      # AllC, AllD
│   │   └── random.py      # Random strategy
│   └── rl/                 # Reinforcement learning agents
│       ├── qlearning.py
│       └── enhanced_qlearning.py
├── experiments/             # Experiment management
│   ├── runners/            # Experiment runners
│   └── scenarios/          # Scenario generation
├── analysis/               # Analysis tools
│   ├── visualizers/        # Plot generation
│   └── reporters/          # Report generation
├── utils/                  # Utilities
├── configs/                # Configuration files
└── results/                # Output directory (created at runtime)
    ├── basic_runs/
    ├── qlearning_tests/
    ├── comparative_analysis/
    └── parameter_sweeps/
```

## Configuration Format

### Basic NPD Experiment

```json
{
  "name": "example_experiment",
  "agents": [
    {
      "id": 0,
      "type": "TFT",
      "exploration_rate": 0.1
    },
    {
      "id": 1,
      "type": "QLearning",
      "exploration_rate": 0.0,
      "learning_rate": 0.1,
      "epsilon": 0.1
    }
  ],
  "num_rounds": 1000
}
```

### Pairwise Tournament

```json
{
  "name": "pairwise_tournament",
  "agents": [...],
  "rounds_per_pair": 100,
  "num_episodes": 1
}
```

## Agent Types

### Strategy-Based Agents

- **TFT**: Classic Tit-for-Tat (pairwise only)
- **pTFT**: Probabilistic TFT for N-Person games
- **pTFT-Threshold**: pTFT with cooperation threshold
- **AllC**: Always Cooperate
- **AllD**: Always Defect
- **Random**: Cooperates with fixed probability

### Reinforcement Learning Agents

- **QLearning**: Basic Q-learning implementation
  - Configurable state representation
  - Epsilon-greedy exploration
  
- **EnhancedQLearning**: Advanced Q-learning with:
  - State representation excluding self (solves aliasing)
  - Decaying epsilon
  - Opponent modeling
  - Adaptive learning rates

## Results Organization

Results are automatically organized by experiment type:

```
results/
├── basic_runs/           # Standard experiments
│   ├── experiment_name/
│   │   ├── csv/         # Data exports
│   │   └── figures/     # Visualizations
├── qlearning_tests/      # Q-learning specific tests
├── comparative_analysis/ # Agent count comparisons
└── parameter_sweeps/     # Parameter optimization
```

## Extending the Framework

### Adding a New Agent Type

1. Create agent class in appropriate module:
```python
# agents/strategies/my_agent.py
from ..base.agent import NPDAgent

class MyAgent(NPDAgent):
    def choose_action(self, cooperation_ratio, round_number):
        # Implementation
        return intended_action, actual_action
```

2. Register in `agents/__init__.py`

3. Update experiment runner to handle the new type

### Adding Analysis Tools

1. Create analyzer in `analysis/` directory
2. Integrate with experiment runner or main orchestrator
3. Update visualization/reporting as needed

## Advanced Features

### Comparative Analysis

Compare performance across different agent counts:

```bash
python main.py compare -c configs/base_config.json -n 3 5 10 20 50
```

Automatically generates:
- Scaling plots (cooperation, scores, convergence)
- Comparative statistics
- Performance trends

### Parameter Sweeps

Optimize agent parameters:

```json
{
  "base_config": {...},
  "parameters": {
    "agents.0.learning_rate": [0.01, 0.1, 0.3],
    "agents.0.epsilon": [0.01, 0.1, 0.2]
  }
}
```

### Batch Processing

Run multiple experiments with different configurations:

```bash
python main.py batch -c configs/batch_scenarios.json --parallel --max-workers 4
```

## Output Files

### CSV Files
- `experiment_summary.csv`: Key metrics
- `experiment_history.csv`: Round-by-round data
- `agent_stats.csv`: Per-agent statistics

### Visualizations
- `cooperation_evolution.png`: Cooperation over time
- `score_distribution.png`: Final scores
- `agent_performance.png`: Performance metrics
- `cooperation_heatmap.png`: Action patterns

### Reports
- `report.html`: Comprehensive experiment report
- `batch_report.html`: Batch experiment summary

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- Pandas
- (Optional) YAML support: `pip install pyyaml`

## Installation

```bash
# Clone the repository
git clone <repository>

# Install dependencies
pip install -r requirements.txt

# Run example
cd npd_simulator
python main.py single -c configs/example_config.json
```

## Citation

If you use this simulator in your research, please cite:
[Citation information]

## License

[License information]