# Migration Guide: Moving to NPD Simulator

This guide helps you migrate from the old flat file structure to the new modular NPD Simulator framework.

## Overview of Changes

### Old Structure
```
3-Person_Tragic_vs_Reciprocity/
├── main_neighbourhood.py
├── main_pairwise.py
├── extended_agents.py
├── qlearning_agents.py
├── enhanced_qlearning_agents.py
├── experiment_runner.py
├── static_figure_generator.py
└── pictures/
```

### New Structure
```
3-Person_Tragic_vs_Reciprocity/
├── npd_simulator/          # New modular framework
│   ├── core/              # Game mechanics
│   ├── agents/            # All agent types
│   ├── experiments/       # Experiment management
│   ├── analysis/          # Visualization & reporting
│   ├── configs/           # Configuration files
│   ├── utils/             # Utilities
│   └── main.py           # Central orchestrator
└── [old files]           # Original files preserved
```

## Key Improvements

1. **N-Agent Support**: No longer limited to 3 agents
2. **Modular Architecture**: Clear separation of concerns
3. **Configuration-Based**: JSON/YAML configs instead of hardcoded experiments
4. **Better Results Organization**: Type-based output directories
5. **Advanced Features**: Comparative analysis, parameter sweeps, batch processing

## Migration Steps

### 1. Using Existing Agent Code

If you have custom agents in the old structure, you can:

a) **Import directly** (temporary solution):
```python
# In your new code
import sys
sys.path.append('../')  # Path to old files
from extended_agents import ExtendedNPersonAgent
```

b) **Migrate to new structure** (recommended):
```python
# Old style (extended_agents.py)
class ExtendedNPersonAgent:
    def __init__(self, agent_id, strategy_name, exploration_rate):
        # ...

# New style (npd_simulator/agents/strategies/extended.py)
from ..base.agent import NPDAgent

class ExtendedAgent(NPDAgent):
    def __init__(self, agent_id: int, exploration_rate: float = 0.0):
        super().__init__(agent_id, exploration_rate)
        # ...
```

### 2. Running Old Experiments

#### Old Way:
```python
# In experiment_runner.py
agents = [
    NPersonAgent(0, "pTFT", 0.1),
    NPersonAgent(1, "AllD", 0.0),
    NPersonAgent(2, "AllC", 0.0)
]
simulation = NPersonPrisonersDilemma(agents, 1000)
simulation.run_simulation()
```

#### New Way:
```json
// config.json
{
  "name": "my_experiment",
  "agents": [
    {"id": 0, "type": "pTFT", "exploration_rate": 0.1},
    {"id": 1, "type": "AllD", "exploration_rate": 0.0},
    {"id": 2, "type": "AllC", "exploration_rate": 0.0}
  ],
  "num_rounds": 1000
}
```

```bash
python npd_simulator/main.py single -c config.json
```

### 3. Accessing Results

#### Old Way:
- Results in `pictures/` directory
- Manual file naming
- Mixed output types

#### New Way:
- Organized by experiment type:
  - `results/basic_runs/`
  - `results/qlearning_tests/`
  - `results/comparative_analysis/`
- Automatic naming with timestamps
- Separate `csv/` and `figures/` subdirectories

### 4. Q-Learning Migration

#### Old Q-Learning Usage:
```python
from qlearning_agents import SimpleQLearningAgent
from extended_agents import QLearningNPersonWrapper

ql_agent = SimpleQLearningAgent(0, learning_rate=0.15, epsilon=0.1)
wrapper = QLearningNPersonWrapper(0, ql_agent)
```

#### New Q-Learning Usage:
```json
{
  "agents": [{
    "id": 0,
    "type": "QLearning",
    "learning_rate": 0.15,
    "epsilon": 0.1,
    "state_type": "basic"
  }]
}
```

### 5. Enhanced Q-Learning Features

The new framework preserves all enhanced Q-learning improvements:

```json
{
  "type": "EnhancedQLearning",
  "exclude_self": true,        // State aliasing fix
  "epsilon_decay": 0.995,      // Decaying exploration
  "opponent_modeling": true,    // Opponent tracking
  "state_type": "basic"        // State discretization
}
```

## Quick Start Examples

### Example 1: Reproduce Old 3-Agent Experiment
```bash
# Create config file matching old experiment
cat > old_experiment.json << EOF
{
  "name": "classic_3agent",
  "agents": [
    {"id": 0, "type": "pTFT", "exploration_rate": 0.1},
    {"id": 1, "type": "pTFT", "exploration_rate": 0.1},
    {"id": 2, "type": "AllD", "exploration_rate": 0.0}
  ],
  "num_rounds": 1000
}
EOF

# Run it
python npd_simulator/main.py single -c old_experiment.json
```

### Example 2: Scale to More Agents
```bash
# Test same strategies with 10 agents
python npd_simulator/main.py compare -c old_experiment.json -n 3 5 10
```

### Example 3: Batch Q-Learning Tests
```bash
# Run all Q-learning scenarios
python npd_simulator/main.py batch -c npd_simulator/configs/qlearning_tests.json
```

## Advanced Features Not Available in Old Structure

### 1. Comparative Analysis
```bash
# Compare performance across different group sizes
python npd_simulator/main.py compare -c config.json -n 3 5 10 20 50
```

### 2. Parameter Optimization
```json
{
  "base_config": {...},
  "parameters": {
    "agents.0.learning_rate": [0.01, 0.1, 0.3],
    "agents.0.epsilon": [0.01, 0.1, 0.2]
  }
}
```

### 3. Parallel Batch Processing
```python
simulator = NPDSimulator()
results = simulator.run_batch_experiments(configs, parallel=True, max_workers=4)
```

## Compatibility Layer

For gradual migration, you can create a compatibility wrapper:

```python
# compatibility.py
from npd_simulator import NPDSimulator
from npd_simulator.utils import load_config

def run_old_style_experiment(agents, num_rounds):
    """Run experiment using old-style agent list."""
    # Convert to new config format
    config = {
        "name": "legacy_experiment",
        "agents": [],
        "num_rounds": num_rounds
    }
    
    for agent in agents:
        agent_config = {
            "id": agent.agent_id,
            "type": agent.strategy_name,
            "exploration_rate": agent.exploration_rate
        }
        config["agents"].append(agent_config)
    
    # Run with new system
    simulator = NPDSimulator()
    return simulator.run_single_experiment(config)
```

## Benefits of Migration

1. **Scalability**: Test with any number of agents
2. **Reproducibility**: Configuration files document experiments
3. **Automation**: Batch processing and parameter sweeps
4. **Analysis**: Built-in visualization and reporting
5. **Extensibility**: Easy to add new agent types and features

## Getting Help

- See `npd_simulator/README.md` for full documentation
- Run `python npd_simulator/demo.py` for examples
- Check `npd_simulator/configs/` for configuration templates

## Preserving Old Results

The new structure doesn't affect old files. You can:
1. Keep pictures/ directory as-is
2. Reference old results in new reports
3. Import old data for comparison

## Next Steps

1. Install new requirements: `pip install -r npd_simulator/requirements.txt`
2. Try the demo: `python npd_simulator/demo.py`
3. Create configs for your experiments
4. Gradually migrate custom code to new structure

The old files remain functional, so you can migrate at your own pace while taking advantage of new features immediately.