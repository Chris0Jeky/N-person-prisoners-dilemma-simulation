# Q-Learning in 3-Person Prisoner's Dilemma

This directory contains Q-learning implementations for the 3-person Prisoner's Dilemma game, demonstrating how reinforcement learning agents adapt to different environments.

## Overview

Two Q-learning implementations are provided:

1. **SimpleQLearningAgent**: A basic, standard Q-learning implementation with simple state representations
2. **NPDLQLearningAgent**: Based on the NPDL framework with more sophisticated state handling

Both work seamlessly with the existing neighborhood and pairwise game modes.

## Files

- `qlearning_agents.py`: Core Q-learning implementations
- `extended_agents.py`: Extended agents with AllC strategy and Q-learning wrappers
- `simple_qlearning_demo.py`: Focused demonstration script
- `test_qlearning_scenarios.py`: Comprehensive testing of all scenarios

## Key Features

### Simple Q-Learning
- **State Representation**: Discretized cooperation levels (very_low, low, medium, high, very_high)
- **Learning**: Standard Q-learning with ε-greedy exploration
- **Strengths**: Fast adaptation, clear Q-table interpretation
- **Best for**: Understanding basic Q-learning behavior

### NPDL Q-Learning
- **State Representation**: Multiple options including proportion-based and memory-enhanced states
- **Learning**: NPDL-style updates with optimistic initialization
- **Strengths**: More nuanced state tracking, better in complex environments
- **Best for**: Achieving better performance in mixed populations

## Running the Demonstrations

### Quick Demo
```bash
python simple_qlearning_demo.py
```

This runs four key scenarios:
1. QL vs AllD vs AllD - Q-learning learns to defect
2. QL vs AllC vs AllC - Q-learning learns to cooperate
3. QL vs TFT vs AllD - Q-learning in mixed environment
4. QL vs QL vs TFT - Multiple learning agents

### Full Test Suite
```bash
python test_qlearning_scenarios.py
```

This runs all seven scenarios in both neighborhood and pairwise modes:
- QL vs TFT vs AllD
- QL vs TFT vs AllC
- QL vs AllC vs AllC
- QL vs AllD vs AllD
- QL vs AllD vs AllC
- QL vs TFT vs TFT
- QL vs QL vs TFT

## How Q-Learning Works Here

### Neighborhood Mode
In neighborhood mode, all agents make simultaneous decisions based on the overall cooperation rate from the previous round.

**State**: Cooperation rate of the group
**Action**: Cooperate or Defect
**Reward**: Linear payoff based on number of cooperators

Example learning process:
```
Round 1: No history → Explore (maybe cooperate)
Round 2: If others cooperated → Learn cooperation is good
Round 3: Update Q-values, exploit best action
...
Eventually: Converges to best response given environment
```

### Pairwise Mode
In pairwise mode, agents play bilateral games with each other, tracking opponent-specific information.

**State**: Opponent's last action (or aggregate statistics)
**Action**: Cooperate or Defect  
**Reward**: Standard PD payoffs from each bilateral game

## Key Observations

### Against All Defectors (AllD)
- Both Q-learning types quickly learn to defect
- Cooperation rate drops to near 0%
- Q-values for defection become much higher than cooperation

### Against All Cooperators (AllC)
- Q-learning agents learn to cooperate for mutual benefit
- Some exploitation occurs initially, but cooperation stabilizes
- Demonstrates that Q-learning can learn reciprocity

### In Mixed Environments
- Simple QL: More volatile, reacts strongly to recent experiences
- NPDL QL: More stable, better at finding balanced strategies
- Both can achieve reasonable performance with proper parameters

### Multiple Q-Learning Agents
- Can lead to interesting dynamics as agents learn simultaneously
- May converge to mutual cooperation or defection depending on initial exploration
- Non-stationary environment challenges standard Q-learning assumptions

## Parameters and Tuning

Key parameters that affect Q-learning behavior:

```python
learning_rate = 0.1      # How quickly to update Q-values (0.1-0.3 typical)
discount_factor = 0.9    # How much to value future rewards (0.9-0.99 typical)
epsilon = 0.1           # Exploration rate for ε-greedy (0.05-0.2 typical)
```

### Tuning Tips
- **High learning rate**: Faster adaptation but less stable
- **Low learning rate**: More stable but slower to learn
- **High epsilon**: More exploration, better for changing environments
- **Low epsilon**: More exploitation, better when optimal strategy is clear

## Extending the Code

To add a new Q-learning variant:

1. Create a new class inheriting from the base structure
2. Implement `choose_action()` and `record_round_outcome()` for neighborhood mode
3. Implement `choose_action_pairwise()` and `record_interaction()` for pairwise mode
4. Wrap with appropriate wrapper class

Example:
```python
class CustomQLearning:
    def __init__(self, agent_id, **params):
        # Initialize Q-table and parameters
        
    def choose_action(self, prev_coop_ratio, round_num):
        # Implement action selection
        
    def record_round_outcome(self, action, payoff):
        # Update Q-values
```

## Theoretical Background

Q-learning aims to learn the optimal action-value function Q*(s,a) through repeated interactions:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- s = current state
- a = action taken
- r = reward received
- s' = next state
- α = learning rate
- γ = discount factor

In the N-person Prisoner's Dilemma:
- States represent the cooperation level of other agents
- Actions are binary (Cooperate/Defect)
- Rewards depend on collective behavior

The challenge is that the environment is non-stationary when multiple agents are learning simultaneously, violating standard Q-learning assumptions. Despite this, Q-learning often finds reasonable policies through exploration and adaptation.

## Conclusions

The Q-learning implementations demonstrate that:

1. **Adaptability**: Q-learning agents successfully adapt to their environment
2. **Context Matters**: Performance heavily depends on the population composition
3. **State Design**: More sophisticated state representations (NPDL) can improve performance
4. **Learning Dynamics**: Multiple learning agents create complex, sometimes unpredictable dynamics

These implementations provide a foundation for exploring how reinforcement learning behaves in multi-agent social dilemmas, showing both its potential and limitations.