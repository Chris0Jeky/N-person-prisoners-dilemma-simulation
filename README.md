# N-Person Prisoner's Dilemma Learning (NPDL)

A framework for simulating and analyzing agent behavior in N-Person Prisoner's Dilemma scenarios.

## Overview

This package provides tools for studying cooperation and defection in multi-agent systems using the prisoner's dilemma game. It includes:

- Various agent strategies from classic game theory and reinforcement learning
- Network structures for agent interactions
- Analysis tools for understanding cooperative behavior
- Interactive gameplay mode
- Visualization dashboard

## Requirements

The package requires Python 3.7+ and several dependencies. Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
npdl/
├── analysis/        # Analysis tools for experiment results
├── core/            # Core components (agents, environment, utils)
├── interactive/     # Interactive gameplay mode
├── simulation/      # Simulation runners
└── visualization/   # Data visualization tools and dashboard
```

## Usage

### Running Simulations

You can run simulations with different scenarios using the command:

```bash
python run.py simulate [options]
```

Options:
- `--enhanced`: Use enhanced scenarios from enhanced_scenarios.json
- `--scenario_file FILE`: Path to the JSON file containing scenario definitions
- `--results_dir DIR`: Directory to save experiment results
- `--log_dir DIR`: Directory to save log files
- `--analyze`: Run analysis on results after experiments complete
- `--verbose`: Enable verbose logging

### Visualization Dashboard

Launch the visualization dashboard to explore simulation results:

```bash
python run.py visualize
```

Then open your browser at http://127.0.0.1:8050/

### Interactive Mode

Play against AI agents in an interactive game:

```bash
python run.py interactive
```

## Strategies

The package includes multiple agent strategies:

- **Classical Strategies**
  - Always Cooperate
  - Always Defect
  - Tit for Tat
  - Generous Tit for Tat
  - Suspicious Tit for Tat
  - Tit for Two Tats
  - Pavlov (Win-Stay, Lose-Shift)
  - Random

- **Reinforcement Learning Strategies**
  - Q-Learning
  - Adaptive Q-Learning
  - Learning Rate Adjusting Q-Learning (LRA-Q)
  - Hysteretic Q-Learning
  - Win or Learn Fast Policy Hill-Climbing (WOLF-PHC)
  - Upper Confidence Bound (UCB1) Q-Learning

## Network Structures

Agents can interact in different network topologies:

- Fully Connected
- Small World
- Scale-Free
- Random 
- Regular

## Defining Scenarios

Scenarios are defined in JSON format with parameters like:

```json
{
  "scenario_name": "Example",
  "num_agents": 30,
  "num_rounds": 500,
  "network_type": "small_world",
  "network_params": {"k": 4, "beta": 0.3},
  "agent_strategies": { "q_learning": 15, "tit_for_tat": 15 },
  "payoff_type": "linear",
  "payoff_params": {"R": 3, "S": 0, "T": 5, "P": 1},
  "state_type": "proportion_discretized",
  "learning_rate": 0.1,
  "discount_factor": 0.9,
  "epsilon": 0.1,
  "logging_interval": 20
}
```

## Development

For development and testing:

```bash
# Run basic tests
python -m npdl.core.test_basic

# Lint code
black npdl/

# Type checking
mypy npdl/
```

## License

This project is available under the MIT License.
