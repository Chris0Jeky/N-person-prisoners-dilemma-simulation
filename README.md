# N-Person Prisoner's Dilemma Learning (NPDL)

A comprehensive framework for simulating, analyzing, and visualizing agent behavior in N-Person Prisoner's Dilemma scenarios.

## Overview

The NPDL package provides tools for studying cooperation and defection dynamics in multi-agent systems through the lens of the prisoner's dilemma game. It's designed for both research and educational purposes, allowing users to explore how different strategies interact in complex social networks.

Key features:
- 12+ agent strategies from classic game theory and reinforcement learning
- Multiple network structures for agent interactions
- Support for both neighborhood-based and pairwise interaction modes
- Detailed analysis tools for understanding cooperation patterns
- Interactive gameplay mode to compete against AI agents
- Visualization dashboard for exploring simulation results
- Comprehensive test suite for code validation

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
│   ├── agents.py    # Agent classes and strategy implementations
│   ├── environment.py # Environment and network simulation
│   ├── utils.py     # Utility functions and payoff calculations
│   └── logging_utils.py # Logging and data collection utilities
├── interactive/     # Interactive gameplay mode
├── simulation/      # Simulation runners
└── visualization/   # Data visualization tools and dashboard
    ├── dashboard.py # Web-based visualization dashboard 
    ├── data_loader.py # Functions for loading simulation data
    ├── data_processor.py # Data processing utilities
    └── network_viz.py # Network visualization components
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
- `--num_runs`: Number of simulation runs (default: 10)

### Visualization Dashboard

Launch the visualization dashboard to explore simulation results:

```bash
python run.py visualize
```

Then open your browser at http://127.0.0.1:8050/

The dashboard provides interactive visualizations of:
- Cooperation rates over time
- Strategy performance comparisons
- Network structure and dynamics
- Agent payoffs and scores

### Interactive Mode

Play against AI agents in an interactive game:

```bash
python run.py interactive
```

This allows you to experience firsthand how different AI strategies respond to your decisions.

## Interaction Modes

NPDL supports two interaction modes that define how agents play with each other:

### 1. Neighborhood-Based Mode (Default)

In this traditional mode, agents interact only with their immediate neighbors in the network. The payoff for each agent depends on how many of its neighbors cooperate or defect. This mode simulates local interactions where an agent's decisions affect only those directly connected to it.

### 2. Pairwise Mode

In pairwise mode, each agent plays a separate 2-player Prisoner's Dilemma game with every other agent in the system, regardless of network connections. This mode is useful for studying the evolution of cooperation in fully mixed populations, where each agent can interact with all others.

Key aspects of pairwise mode:
- Each agent chooses one move (cooperate/defect) for the entire round
- The agent plays that move against every other agent in a standard 2-player PD game
- An agent's score is the sum of payoffs from all its pairwise games
- For learning purposes, agents track the proportion of opponents who cooperated

To use pairwise mode, set the `interaction_mode` parameter in your scenario definition:

```json
{
  "scenario_name": "Pairwise_Example",
  "interaction_mode": "pairwise",
  "num_agents": 30,
  "num_rounds": 100,
  "network_type": "fully_connected",
  "agent_strategies": { "q_learning": 15, "tit_for_tat": 15 },
  "payoff_params": { "R": 3, "S": 0, "T": 5, "P": 1 }
}
```

The `network_type` is still used for visualization purposes in pairwise mode, but doesn't affect the interaction pattern.

## Agent Strategies

The package includes multiple agent strategies:

### Classical Strategies
- **Always Cooperate**: Always chooses to cooperate
- **Always Defect**: Always chooses to defect
- **Tit for Tat**: Mimics the opponent's previous move
- **Generous Tit for Tat**: Like Tit for Tat, but occasionally forgives defection
- **Suspicious Tit for Tat**: Starts with defection, then follows Tit for Tat
- **Tit for Two Tats**: Only defects after the opponent defects twice in a row
- **Pavlov (Win-Stay, Lose-Shift)**: Repeats previous move if successful, switches if not
- **Random**: Chooses randomly between cooperation and defection

### Reinforcement Learning Strategies
- **Q-Learning**: Standard Q-learning with epsilon-greedy exploration
- **Adaptive Q-Learning**: Q-learning with decaying exploration rate
- **Learning Rate Adjusting Q-Learning (LRA-Q)**: Dynamically adjusts learning rate based on cooperation levels
- **Hysteretic Q-Learning**: Uses different learning rates for positive and negative experiences
- **Win or Learn Fast Policy Hill-Climbing (WOLF-PHC)**: Adjusts learning rates based on performance
- **Upper Confidence Bound (UCB1) Q-Learning**: Uses UCB1 algorithm for exploration

## Strategy Behavior in Pairwise Mode

In pairwise mode, strategies behave slightly differently:

- **Tit for Tat variants**: Base their decisions on the majority behavior of opponents in the previous round
- **Pavlov**: Responds based on the average reward received
- **Reinforcement Learning**: Uses the proportion of cooperating opponents as state representation
- **LRA-Q**: Adjusts learning rate based on aggregate opponent cooperation

## Network Structures

Agents can interact in different network topologies:

- **Fully Connected**: Every agent connects to every other agent
- **Small World**: High clustering with short average path lengths (Watts-Strogatz model)
- **Scale-Free**: Power-law degree distribution with hub nodes (Barabási-Albert model)
- **Random**: Random connections with specified probability (Erdős–Rényi model)
- **Regular**: Each node has the same number of connections

## Defining Scenarios

Scenarios are defined in JSON format. Example with neighborhood-based interaction:

```json
{
  "scenario_name": "Example_Neighborhood",
  "interaction_mode": "neighborhood",
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

Example with pairwise interaction:

```json
{
  "scenario_name": "Example_Pairwise",
  "interaction_mode": "pairwise",
  "num_agents": 30,
  "num_rounds": 100,
  "network_type": "fully_connected",
  "agent_strategies": { "lra_q": 15, "tit_for_tat": 15 },
  "payoff_params": { "R": 3, "S": 0, "T": 5, "P": 1 },
  "state_type": "proportion_discretized",
  "learning_rate": 0.1,
  "discount_factor": 0.9,
  "epsilon": 0.1,
  "logging_interval": 20
}
```

The project includes both `scenarios.json` with basic scenarios and `enhanced_scenarios.json` with more advanced configurations. You can find pairwise examples in `pairwise_scenarios.json`.

## Testing

The project includes a comprehensive test suite with over 40 tests covering all major components. To run the tests:

```bash
pytest
```

Note: While all tests are currently passing, some test cases are still being refined to better validate the system behavior.

## Development

For development:

```bash
# Run tests with coverage report
pytest --cov=npdl

# Run basic functionality tests
python -m npdl.core.test_basic

# Lint code
black npdl/

# Type checking
mypy npdl/
```

## Future Work

Planned enhancements include:
- Additional learning strategies
- Dynamic payoff structures
- More sophisticated network rewiring mechanisms
- Enhanced visualization components
- Expanded test coverage
- Per-opponent learning in pairwise mode

## License

This project is available under the MIT License.
