# Pairwise Interaction Mode

## Overview

The pairwise interaction mode is an alternative way of simulating agent interactions in the N-Person Prisoner's Dilemma (NPDL) framework. Unlike the default neighborhood-based mode where agents only interact with their network neighbors, in pairwise mode each agent plays a separate 2-player Prisoner's Dilemma game with every other agent in the system.

This approach models a fully mixed population where each individual can interact with any other individual, regardless of network structure. The network structure is still maintained for visualization purposes, but doesn't influence the interaction pattern.

## Key Features

- Each agent chooses one move (cooperate/defect) for the entire round
- The agent plays that move against every other agent in the system
- Each pair plays a standard 2-player Prisoner's Dilemma game with payoffs (R, S, T, P)
- An agent's total score is the sum of all its pairwise interactions
- Learning agents use the proportion of cooperating opponents as input to their learning process

## Using Pairwise Mode

### In Scenario JSON Files

To use pairwise mode in a scenario configuration, add `"interaction_mode": "pairwise"` to your JSON file:

```json
{
  "scenario_name": "Pairwise_Example",
  "interaction_mode": "pairwise",
  "num_agents": 20,
  "num_rounds": 100,
  "network_type": "fully_connected",
  "agent_strategies": {
    "q_learning": 10,
    "tit_for_tat": 10
  },
  "payoff_params": { 
    "R": 3, "S": 0, "T": 5, "P": 1
  }
}
```

Run the simulation with:
```bash
python run.py simulate --scenario_file your_scenarios.json
```

### In Code

For programmatic usage, specify the interaction mode when creating the Environment:

```python
env = Environment(
    agents,
    payoff_matrix,
    network_type="fully_connected",
    interaction_mode="pairwise",  # Key parameter
    R=3, S=0, T=5, P=1           # 2-player PD payoff values
)
```

## How Strategies Behave in Pairwise Mode

In pairwise mode, strategies have been adapted to handle aggregate opponent behavior:

### Memory-Based Strategies

Memory-based strategies like Tit for Tat store and use the `opponent_coop_proportion` value instead of individual neighbor moves:

- **Tit for Tat**: Cooperates if the majority of opponents cooperated in the previous round (proportion > 0.5), otherwise defects
- **Generous Tit for Tat**: Same as TFT, but has a chance to cooperate even when majority defected
- **Suspicious Tit for Tat**: Same as TFT but starts with defection
- **Tit for Two Tats**: Defects only if the majority of opponents defected in two consecutive rounds

### Reinforcement Learning Strategies

RL strategies have modified state representations to work with aggregate opponent data:

- **QLearningStrategy**: Creates state representations based on the proportion of cooperating opponents
- **LRAQLearningStrategy**: Adjusts learning rate based on the proportion of cooperating opponents
- **HystereticQLearningStrategy**: Uses different learning rates for positive/negative experiences based on aggregate outcomes
- **WolfPHCStrategy**: Adjusts learning rates dynamically based on performance against the population
- **UCB1QLearningStrategy**: Uses UCB1 algorithm to balance exploration and exploitation in mixed populations

## Example Scenarios

### 1. Comparing Strategy Performance

This scenario tests how different strategy types perform against each other in a fully mixed population:

```json
{
  "scenario_name": "Mixed_Strategy_Tournament",
  "interaction_mode": "pairwise",
  "num_agents": 35,
  "num_rounds": 200,
  "network_type": "fully_connected",
  "agent_strategies": {
    "always_cooperate": 5,
    "always_defect": 5,
    "tit_for_tat": 5,
    "generous_tit_for_tat": 5,
    "suspicious_tit_for_tat": 5,
    "pavlov": 5,
    "q_learning": 5
  },
  "payoff_params": { 
    "R": 3, "S": 0, "T": 5, "P": 1
  }
}
```

### 2. Testing Reinforcement Learning Variants

This scenario compares different RL strategies in a pairwise setting:

```json
{
  "scenario_name": "RL_Comparison_Pairwise",
  "interaction_mode": "pairwise",
  "num_agents": 40,
  "num_rounds": 200,
  "network_type": "fully_connected",
  "agent_strategies": {
    "q_learning": 10,
    "lra_q": 10,
    "hysteretic_q": 10,
    "wolf_phc": 10
  },
  "payoff_params": { 
    "R": 3, "S": 0, "T": 5, "P": 1
  },
  "learning_rate": 0.1,
  "discount_factor": 0.9,
  "epsilon": 0.1
}
```

### 3. Investigating the Effect of Payoff Values

Change the R, S, T, P values to study their effects on cooperation:

```json
{
  "scenario_name": "Payoff_Effect_Study",
  "interaction_mode": "pairwise",
  "num_agents": 30,
  "num_rounds": 150,
  "network_type": "fully_connected",
  "agent_strategies": {
    "tit_for_tat": 15,
    "q_learning": 15
  },
  "payoff_params": { 
    "R": 4, "S": -1, "T": 6, "P": 0
  }
}
```

## Analysis Tips

When analyzing pairwise simulation results, consider:

1. **Strategy Dominance**: In pairwise mode, more aggressive strategies might initially perform better, but cooperative strategies often dominate in the long run if they can coordinate.

2. **Learning Convergence**: RL strategies might take longer to converge in pairwise mode because they're learning from a more complex aggregate signal.

3. **Score Distribution**: The absolute scores will be larger in pairwise mode (since each agent plays more games), so focus on relative performance.

4. **Cooperation Rate**: The cooperation rate is a key metric to understand how well strategies achieve mutual cooperation.

## Implementation Details

The pairwise mode is implemented through:

1. `Environment.run_round()` method that dispatches to `_run_pairwise_round()` when `interaction_mode="pairwise"`

2. `_run_pairwise_round()` method that:
   - Collects moves from all agents
   - Simulates pairwise games between all agent pairs
   - Calculates payoffs and opponent cooperation proportions
   - Updates agent memories and learning models

3. Strategy adaptations in `agents.py` to handle the `opponent_coop_proportion` data

4. State representation methods in `QLearningStrategy` that handle both interaction modes

## Limitations and Future Improvements

Current limitations of the pairwise mode:

1. All agents play the same move against all opponents in a round (unlike real-world scenarios where strategies might be opponent-specific)

2. Agents learn from aggregate opponent behavior rather than learning separate policies for each opponent

Planned improvements:

1. **Per-opponent learning**: Allow agents to learn specific strategies for different opponent types

2. **Multiple rounds per pairing**: Enable multiple interactions between the same agent pair within a round

3. **Reputation mechanisms**: Add reputation tracking to influence decision-making