# Pairwise Interaction Mode

## Overview

The pairwise interaction mode is an alternative way of running N-Person Prisoner's Dilemma simulations where each agent plays a separate 2-player PD game with every other agent, rather than a single move against all neighbors at once.

This implementation offers several advantages:
- More realistic dyadic interactions between agents
- Direct comparison with classic 2-player PD results
- Clearer strategy dynamics between agent types

## How It Works

In pairwise mode:

1. Each agent chooses a single move (Cooperate or Defect) for the round
2. The agent plays separate 2-player PD games with each other agent using that move
3. Payoffs from all pairwise interactions are summed to give the agent's score for the round
4. Information about opponent moves is stored in the agent's memory

## Using Pairwise Mode

To use pairwise mode in your simulations, specify `interaction_mode="pairwise"` when creating an Environment:

```python
env = Environment(
    agents,
    payoff_matrix,
    network_type="fully_connected",  # Network structure still used for visualization
    interaction_mode="pairwise",     # Use pairwise interaction mode
    R=3, S=0, T=5, P=1               # Standard 2-player PD payoff values
)
```

You can also specify pairwise mode in scenario files:

```json
{
  "scenario_name": "Pairwise_Example",
  "num_agents": 20,
  "num_rounds": 100,
  "network_type": "fully_connected",
  "network_params": {},
  "interaction_mode": "pairwise",
  "agent_strategies": {
    "q_learning": 10,
    "tit_for_tat": 10
  },
  "payoff_params": { 
    "R": 3,
    "S": 0,
    "T": 5,
    "P": 1
  }
}
```

## Strategy Considerations

Strategies behave slightly differently in pairwise mode:

1. **TitForTat strategies**: In pairwise mode, TFT strategies defect if ANY opponent defected in the previous round, whereas in neighborhood mode they respond to a randomly selected neighbor.

2. **Learning strategies**: Q-learning and similar strategies adapt to the aggregate cooperation rate across all opponents, rather than the specific neighborhood structure.

3. **Memory**: Agent memory in pairwise mode includes specific opponent moves as well as an aggregate cooperation rate.

## Testing and Validation

You can test the pairwise implementation using:

```bash
python test_pairwise.py
```

If you encounter issues with the TitForTat strategies in pairwise mode, you can apply the runtime fixes:

```bash
python run_patched_tests.py
```

Or apply permanent fixes to the codebase:

```bash
python apply_permanent_fixes.py
```

## Implementation Notes

The pairwise mode still uses network structures for visualization purposes, but the actual interactions are not limited by network connections. Every agent interacts with every other agent, regardless of network structure.

The payoff parameters (R, S, T, P) are used directly for 2-player games, rather than through the N-person payoff functions.

## Comparing Pairwise and Neighborhood Modes

You can compare the two interaction modes using the test script:

```bash
python test_pairwise.py
```

Key differences in results typically include:
- Higher overall cooperation rates in pairwise mode (due to more targeted reciprocity)
- Different relative performance of strategies
- More pronounced effects of specific strategy interactions
