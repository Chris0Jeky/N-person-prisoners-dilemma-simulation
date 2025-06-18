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
- Scenario generation and analysis tools for discovering interesting parameter combinations
- Comprehensive test suite for code validation

## What's New

### Improved Tit-for-Tat Implementation (v2.0)
The TFT strategy has been redesigned to be ecosystem-aware:
- **Network-aware**: TFT now considers the cooperation proportion among connected neighbors
- **Configurable threshold**: Set the minimum cooperation proportion required (default: 50%)
- **Proportional variant**: New ProportionalTitForTat strategy for probabilistic responses
- **Consistent behavior**: Works properly in both neighborhood and pairwise modes

See `scripts/demos/demonstrate_tft_ecosystem.py` for a demonstration of the new behavior.

## Requirements

The package requires Python 3.7+ and several dependencies. Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
npdl/
├── analysis/                # Analysis tools for experiment results
│   └── sweep_visualizer.py  # Visualization for scenario sweep results
├── core/                    # Core components (agents, environment, utils)
│   ├── agents.py            # Agent classes and strategy implementations
│   ├── environment.py       # Environment and network simulation
│   ├── utils.py             # Utility functions and payoff calculations
│   └── logging_utils.py     # Logging and data collection utilities
├── interactive/             # Interactive gameplay mode
├── simulation/              # Simulation runners
├── visualization/           # Data visualization tools and dashboard
│   ├── dashboard.py         # Web-based visualization dashboard 
│   ├── data_loader.py       # Functions for loading simulation data
│   ├── data_processor.py    # Data processing utilities
│   └── network_viz.py       # Network visualization components
├── scripts/
│   └── runners/
│       ├── run_scenario_generator.py    # Scenario generation and evaluation
│       ├── run_sweep_analysis.py        # Complete sweep workflow
│       └── run_evolutionary_generator.py # Evolution-based scenario generation
```

## Usage

### Running Simulations

You can run simulations with different scenarios using the command:

```bash
python run.py simulate [options]
```

Options:
- `--enhanced`: Use enhanced scenarios from enhanced_scenarios.json
- `--scenario_file FILE`: Path to the JSON file containing scenario definitions (try with pairwise_scenarios.json for pairwise examples)
- `--results_dir DIR`: Directory to save experiment results
- `--log_dir DIR`: Directory to save log files
- `--analyze`: Run analysis on results after experiments complete
- `--verbose`: Enable verbose logging
- `--num_runs`: Number of simulation runs (default: 10)

### Automatic Scenario Generation & Analysis

For systematic exploration of the parameter space, NPDL includes tools to generate, evaluate, and analyze diverse scenarios:

#### Basic Random Scenario Generation

```bash
python scripts/runners/run_sweep_analysis.py [options]
```

Options:
- `--num_generate INT`: Number of random scenarios to generate (default: 30)
- `--eval_runs INT`: Number of quick evaluation runs per scenario (default: 3)
- `--save_runs INT`: Number of full runs for selected scenarios (default: 10)
- `--top_n INT`: Number of top scenarios to save and analyze (default: 5)
- `--results_dir DIR`: Directory to save scenario results (default: results/generated_scenarios)
- `--analysis_dir DIR`: Directory to save analysis results (default: analysis_results)
- `--log_level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

This workflow:
1. Generates random scenarios by sampling from diverse parameter combinations
2. Briefly simulates each scenario to evaluate its "interestingness"
3. Selects the most interesting scenarios based on metrics like:
   - Cooperation rate dynamics (changes over time)
   - Variance in performance across different strategies
   - Strategy dominance (difference between best and worst strategies)
4. Runs more thorough simulations of the selected scenarios
5. Generates visualizations comparing the selected scenarios

#### Advanced Evolutionary Scenario Generation

For more sophisticated scenario discovery, the evolutionary algorithm approach iteratively improves scenarios through selection, crossover, and mutation:

```bash
python scripts/runners/run_evolutionary_generator.py [options]
```

Options:
- `--pop_size INT`: Population size for evolutionary algorithm (default: 20)
- `--generations INT`: Number of generations to evolve (default: 5)
- `--eval_runs INT`: Number of evaluation runs per scenario (default: 3)
- `--elitism INT`: Number of best scenarios to keep unchanged in each generation (default: 2)
- `--crossover FLOAT`: Fraction of new generation created through crossover (default: 0.7)
- `--mutation FLOAT`: Mutation rate for scenario parameters (default: 0.3)
- `--save_runs INT`: Number of full runs for selected scenarios (default: 10)
- `--top_n INT`: Number of top scenarios to save (default: 5)
- `--results_dir DIR`: Directory to save results (default: results/evolved_scenarios)
- `--log_level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

The evolutionary approach offers several advantages:
1. **Parallel evaluation** using multiple CPU cores for faster processing
2. **Enhanced interestingness metrics** including:
   - Cooperation volatility (fluctuations in cooperation rates)
   - Strategy adaptation rate (how quickly learning agents adapt)
   - Pattern complexity (complexity of cooperation dynamics)
   - Equilibrium stability (whether the system reaches stable states)
3. **Iterative improvement** through:
   - Tournament selection of promising scenarios
   - Crossover (combining parameters from high-scoring scenarios)
   - Mutation (random variations to explore new possibilities)
   - Generational improvement tracking

To visualize and analyze results, inspect the files generated in the
`evolution_analysis` directory after the run. These include CSV data and
several plots summarizing progress across generations.

#### Interestingness Metrics

The tool ranks scenarios by a composite "interestingness score" that considers:
- Whether cooperation rates are changing over time (dynamic scenarios)
- Whether different strategies achieve significantly different outcomes
- Avoiding extreme cases where all agents cooperate or all defect
- Volatility and complexity in cooperation patterns
- Strategy adaptation and learning characteristics

Example output graphs include:
- Scenario ranking by interestingness score
- Comparison of cooperation rates across scenarios
- Strategy performance distribution
- Parameter frequency analysis
- Correlation between different metrics
- Radar charts for multidimensional scenario comparison

For more targeted exploration, you can also run only the basic scenario generation step:

```bash
python scripts/runners/run_scenario_generator.py --num_generate 50 --eval_runs 3 --save_runs 10 --top_n 5
```

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

## How to Use Pairwise Mode

There are two main ways to use the pairwise interaction mode:

### 1. Using JSON Scenario Files

Create a JSON scenario file with `"interaction_mode": "pairwise"`:

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

Then run the simulation with:

```bash
python run.py simulate --scenario_file your_scenarios.json
```

For quick testing, you can use the included pairwise examples:

```bash
python run.py simulate --scenario_file pairwise_scenarios.json
```

### 2. Using the Environment API Directly

For more programmatic control, you can create an Environment with pairwise interaction mode:

```python
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix

# Create agents
agents = [
    Agent(agent_id=0, strategy="tit_for_tat"),
    Agent(agent_id=1, strategy="q_learning"),
    # Add more agents as needed
]

# Create payoff matrix
payoff_matrix = create_payoff_matrix(len(agents), "linear")

# Create environment with pairwise interaction
env = Environment(
    agents,
    payoff_matrix,
    network_type="fully_connected",  # Network still used for visualization
    interaction_mode="pairwise",     # Key parameter for pairwise mode
    R=3, S=0, T=5, P=1              # 2-player PD payoff values
)

# Run simulation
results = env.run_simulation(num_rounds=100)

# Analyze results
final_round = results[-1]
coop_rate = sum(1 for move in final_round['moves'].values() 
              if move == "cooperate") / len(final_round['moves'])
print(f"Final cooperation rate: {coop_rate:.2f}")
```

## Customizing Scenario Generation

When using the scenario generator, you can customize the parameter space to explore by modifying the parameter pools at the top of `run_scenario_generator.py`:

```python
# Define pools of options to combine
AGENT_STRATEGY_POOL = [
    "hysteretic_q", "wolf_phc", "lra_q",  # Learning strategies
    "q_learning", "q_learning_adaptive", "ucb1_q",
    "tit_for_tat", "tit_for_two_tats", "pavlov",  # Reactive strategies
    "generous_tit_for_tat", "suspicious_tit_for_tat",
    "always_cooperate", "always_defect", "random"  # Fixed/Simple strategies
]

NETWORK_POOL = [
    ("small_world", {"k": 4, "beta": 0.3}),
    ("scale_free", {"m": 2}),
    ("fully_connected", {}),
    # Add your custom network configurations
]

# Add other parameter pools
INTERACTION_MODE_POOL = ["neighborhood", "pairwise"]
NUM_AGENTS_POOL = [20, 30, 40, 50]
# ...
```

You can also adjust the scenario evaluation function to focus on different aspects of "interestingness":

```python
def calculate_interestingness_score(eval_result):
    """Calculate an 'interestingness' score based on multiple metrics."""
    # Extract metrics (handle potential NaN values)
    metrics = eval_result['metrics']
    
    # Get normalized metrics (0-1 range)
    coop = metrics.get('avg_final_coop_rate', 0)
    coop_change = abs(metrics.get('avg_coop_rate_change', 0))
    variance = metrics.get('avg_score_variance', 0) / 1000  # Normalize
    
    # Weighted combination of metrics - adjust weights to your preference 
    score = (
        0.2 * (1 - abs(coop - 0.5)) +  # Prefer intermediate cooperation rates
        0.4 * coop_change +            # Reward dynamics (changes over time)
        0.4 * variance                 # Reward variance across strategies
    )
    
    return score
```

## Agent Strategies

The package includes multiple agent strategies:

### Classical Strategies
- **Always Cooperate**: Always chooses to cooperate
- **Always Defect**: Always chooses to defect
- **Tit for Tat**: Cooperates based on the proportion of neighbors who cooperated (ecosystem-aware)
  - Threshold-based: Cooperates if cooperation proportion ≥ threshold (default 0.5)
  - Network-aware: Only considers connected neighbors in the calculation
- **Proportional Tit for Tat**: Cooperates with probability equal to neighborhood cooperation proportion
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

## Strategy Behavior in Different Modes

### Neighborhood Mode
In neighborhood mode, strategies consider only their connected neighbors:
- **Tit for Tat**: Cooperates if the proportion of cooperating neighbors meets its threshold
- **Proportional TFT**: Cooperates with probability = proportion of cooperating neighbors
- **Learning strategies**: Use neighborhood cooperation proportion as state representation

### Pairwise Mode
In pairwise mode, strategies consider all opponents:
- **Tit for Tat**: 
  - With specific opponent tracking: Defects if ANY opponent defected
  - With aggregate tracking: Uses overall cooperation proportion with threshold
- **Proportional TFT**: Cooperates with probability = overall cooperation proportion
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

In pairwise mode, the network structure is used only for visualization purposes, as all agents interact with all other agents regardless of network connections.

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

# Generate and analyze random scenarios
python scripts/runners/run_sweep_analysis.py --num_generate 10 --top_n 3

# Run evolutionary scenario generation with parallel processing
python scripts/runners/run_evolutionary_generator.py --pop_size 20 --generations 5 --eval_runs 3

# Analyze evolutionary results
# (visualization script removed - view data in the `evolution_analysis` directory)

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
- Multiple rounds per pairing in pairwise mode
- Further enhancements to evolutionary algorithms:
  - Multi-objective optimization for scenario generation
  - Diversity preservation mechanisms
  - Interactive evolutionary computation with user feedback
- Reinforcement learning for automatic scenario optimization
- Integration with external machine learning frameworks

## License

This project is available under the MIT License.
