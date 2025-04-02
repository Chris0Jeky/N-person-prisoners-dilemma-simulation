# Pairwise Interaction Model for N-Person Prisoner's Dilemma

This feature adds a pairwise interaction model to the existing N-Person Prisoner's Dilemma simulation framework. In this mode, agents play separate 2-player PD games against each agent in the population, with their total score being the sum of all pairwise games.

## Key Differences from Neighborhood Model

In the **standard neighborhood model**:
- Each agent makes one move (cooperate/defect)
- The payoff depends on the proportion of neighbors who cooperate
- Network structure is highly influential

In the **pairwise model**:
- Each agent makes one move (cooperate/defect) that's used in all their games
- Separate 2-player payoffs calculated for each pair of agents
- Network structure is mainly used for visualization

## Running Pairwise Simulations

1. **Basic Usage**:
   ```bash
   python main.py --scenario_file pairwise_scenarios.json --num_runs 3
   ```

2. **Testing the Implementation**:
   ```bash
   python test_pairwise.py
   ```

3. **Available Pairwise Scenarios**:

   The `pairwise_scenarios.json` file includes several pre-configured scenarios:
   - **Pairwise_Mixed**: A mix of all basic strategies
   - **Pairwise_GTFT_vs_QL**: Generous Tit-for-Tat vs. Q-Learning
   - **Pairwise_LRA_vs_TFT**: Learning Rate Adjusting Q-Learning vs. Tit-for-Tat
   - **Evolution_Of_Trust_Simulation**: Simulates the strategies from "The Evolution of Trust" game

## Creating Custom Pairwise Scenarios

To create a custom pairwise scenario, use the following template:

```json
{
  "scenario_name": "My_Pairwise_Scenario",
  "num_agents": 20,
  "num_rounds": 100,
  "network_type": "fully_connected",
  "network_params": {},
  "interaction_mode": "pairwise",  // This flag enables pairwise mode
  "agent_strategies": {
    "strategy1": count1,
    "strategy2": count2
  },
  "payoff_params": { 
    "R": 3,  // Reward for mutual cooperation
    "S": 0,  // Sucker's payoff
    "T": 5,  // Temptation to defect
    "P": 1   // Punishment for mutual defection
  },
  "learning_rate": 0.1,  // For RL agents
  "discount_factor": 0.9,  // For RL agents
  "epsilon": 0.1,  // For RL agents
  "state_type": "proportion_discretized",  // For RL agents
  "logging_interval": 10
}
```

## Implementation Details

1. **Environment Class**:
   - Added `interaction_mode` parameter
   - Added `_run_pairwise_round` method 
   - Modified `run_round` to call the appropriate method

2. **QLearningStrategy**:
   - Modified `_get_current_state` to handle pairwise data format
   - Uses opponent cooperation proportion to determine state

3. **LRAQLearningStrategy**:
   - Updated learning rate adjustment logic to work with aggregate opponent data

4. **Utility Functions**:
   - Added `get_pairwise_payoffs` function to calculate 2-player payoffs

## Interpretation of Results

When analyzing pairwise simulation results, note that:

1. **Scores**: Will generally be higher than in neighborhood model as agents play against all other agents
2. **Cooperation Rate**: Measures global cooperation in the population
3. **Strategy Performance**: Reflects how well each strategy does when playing against all other strategies
