# N-Person Prisoner's Dilemma Simulation Framework

A comprehensive Python framework for simulating and analyzing cooperation dynamics in multi-agent prisoner's dilemma games. This toolkit supports both pairwise (2-player) and N-person game variants with various agent strategies including Q-learning.

## Overview

This framework investigates the emergence of cooperation in social dilemmas by simulating interactions between agents using different strategies:

- **Static Strategies**: TFT (Tit-for-Tat), AllD (Always Defect), AllC (Always Cooperate)
- **Adaptive Strategies**: Simple Q-Learning and NPDL-based Q-Learning
- **Game Modes**: Pairwise interactions and N-person neighborhood games
- **Analysis Tools**: Automated experiment runners, data collection, and visualization

## Key Features

- **Modular Design**: Clean separation between agents, game environments, and experiment runners
- **Flexible Configuration**: Easy to add new agent types or modify game parameters
- **No Dependencies**: Core functionality works with Python standard library only
- **Optional Visualizations**: Enhanced plots available with matplotlib/seaborn
- **Comprehensive Results**: Detailed CSV outputs and statistical summaries

## Installation

### Basic Installation (No Dependencies)
```bash
# Clone or download the repository
cd legacy_cleaned/

# Run with Python 3.7+
python -m src.experiment_runner
```

### Full Installation (With Visualization)
```bash
# Install optional dependencies
pip install -r requirements.txt

# Run experiments
python -m src.experiment_runner
```

## Quick Start

### Running Default Experiments

```python
from src import ExperimentRunner

# Create and run experiment runner
runner = ExperimentRunner(base_output_dir="results")
runner.run_all_experiments()
```

### Custom Agent Composition

```python
from src import create_agent, NPersonGame

# Create agents
agents = [
    create_agent("QL_1", "QL", epsilon=0.1),
    create_agent("TFT_1", "TFT"),
    create_agent("AllD_1", "AllD")
]

# Run N-person game
game = NPersonGame(agents, num_rounds=200)
results = game.run_simulation()
```

## Agent Types

### Static Agents

1. **TFT (Tit-for-Tat)**
   - Pairwise: Copies opponent's last move
   - N-person: Uses pTFT (probabilistic TFT) or pTFT-Threshold

2. **AllD (Always Defect)**
   - Always chooses to defect

3. **AllC (Always Cooperate)**
   - Always chooses to cooperate

### Q-Learning Agents

1. **Simple Q-Learning (`QL` or `SimpleQL`)**
   - Basic Q-learning implementation
   - Simple state discretization
   - Epsilon-greedy action selection

2. **NPDL Q-Learning (`NPDLQL`)**
   - Advanced state representations
   - Epsilon decay scheduling
   - Opponent modeling capability
   - Memory-enhanced states

## Game Environments

### Pairwise Game
- Classic 2-player Prisoner's Dilemma
- Payoff matrix: T=5, R=3, P=1, S=0
- Supports episodic and non-episodic modes

### N-Person Game
- Linear public goods game
- Payoffs scale with cooperation ratio
- Supports different group sizes

## Usage Examples

### Example 1: Simple Pairwise Tournament

```python
from src import run_pairwise_experiment

# Define agent configurations
configs = [
    {"id": "TFT_1", "strategy": "TFT", "exploration_rate": 0.0},
    {"id": "TFT_2", "strategy": "TFT", "exploration_rate": 0.0},
    {"id": "AllD_1", "strategy": "AllD", "exploration_rate": 0.0}
]

# Run experiment
results, game = run_pairwise_experiment(configs, total_rounds=100)

# Print results
print(f"Overall cooperation rate: {results['overall']['cooperation_rate']:.3f}")
for agent_id, stats in results['agents'].items():
    print(f"{agent_id}: Score={stats['total_score']}, Coop={stats['cooperation_rate']:.3f}")
```

### Example 2: Q-Learning Comparison

```python
from src import ExperimentRunner

# Custom Q-learning compositions
ql_compositions = [
    {
        "name": "SimpleQL_vs_NPDLQL",
        "agents": [
            {"id": "SimpleQL_1", "strategy": "SimpleQL"},
            {"id": "NPDLQL_1", "strategy": "NPDLQL"},
            {"id": "TFT_1", "strategy": "TFT"}
        ]
    }
]

runner = ExperimentRunner()
runner.set_configurations(
    exploration_rates=[0.0],
    agent_compositions=ql_compositions
)
runner.run_all_experiments()
```

### Example 3: Custom N-Person Simulation

```python
from src import create_agent, NPersonGame

# Create mixed strategy group
agents = []
for i in range(2):
    agents.append(create_agent(f"QL_{i}", "NPDLQL", 
                              epsilon=0.1, epsilon_decay=0.995))
for i in range(3):
    agents.append(create_agent(f"TFT_{i}", "pTFT-Threshold"))

# Configure and run game
game = NPersonGame(agents, num_rounds=500)
results = game.run_simulation()

# Analyze cooperation dynamics
cooperation_history = results['cooperation_history']
final_coop_rate = results['overall']['cooperation_rate']
print(f"Final cooperation rate: {final_coop_rate:.3f}")
```

## Output Structure

Results are saved in the following structure:
```
results/
├── exp_0/                    # 0% exploration rate
│   ├── 3_TFTs/
│   │   ├── pairwise_nonepisodic/
│   │   │   ├── results.json
│   │   │   └── summary.csv
│   │   └── nperson_pTFT/
│   └── 2_TFTs_1_AllD/
└── exp_10/                   # 10% exploration rate
```

## Configuration Options

### Q-Learning Parameters
- `learning_rate`: Learning rate (α) [0.0-1.0]
- `discount_factor`: Discount factor (γ) [0.0-1.0]
- `epsilon`: Exploration rate for ε-greedy [0.0-1.0]
- `epsilon_decay`: Decay rate for epsilon
- `epsilon_min`: Minimum epsilon value

### Game Parameters
- `num_rounds`: Total rounds per simulation
- `num_episodes`: Number of episodes (pairwise only)
- `rounds_per_episode`: Rounds per episode
- `num_runs`: Number of simulation runs for averaging

## Advanced Usage

### Creating Custom Agents

```python
from src.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_id, **kwargs):
        super().__init__(agent_id)
        # Initialize your agent
    
    def choose_action(self, context, mode='neighborhood'):
        # Implement decision logic
        return intended_move, actual_move
    
    def update_neighborhood(self, my_move, payoff):
        # Update after neighborhood game
        pass
    
    def update_pairwise(self, opponent_id, opponent_move, my_move, payoff):
        # Update after pairwise game
        pass
```

### Batch Experiments

```python
# Run experiments with different parameters
for lr in [0.05, 0.1, 0.2]:
    for epsilon in [0.05, 0.1, 0.15]:
        agents = [
            create_agent("QL_1", "NPDLQL", 
                        learning_rate=lr, epsilon=epsilon)
            # ... more agents
        ]
        # Run experiment and collect results
```

## Troubleshooting

1. **Import Errors**: Ensure you're running from the correct directory
2. **No Plots Generated**: Install matplotlib and seaborn: `pip install matplotlib seaborn`
3. **Memory Issues**: Reduce `num_runs` or `num_rounds` for large experiments

## Citation

If you use this framework in your research, please cite:
```
@software{npd_simulation,
  title={N-Person Prisoner's Dilemma Simulation Framework},
  author={Research Team},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.