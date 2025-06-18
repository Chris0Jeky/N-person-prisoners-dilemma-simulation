# N-Person Iterated Prisoner's Dilemma Implementation Summary

## Overview
I've created a complete N-Person Iterated Prisoner's Dilemma simulation that integrates with your existing framework. The implementation includes three key components:

1. **simple_npd.py** - Core N-Person simulation engine
2. **run_npd_simulation.py** - Interactive examples with verbose output
3. **run_npd_csv_export.py** - Batch simulation with CSV export for analysis

## Key Features

### N-Person Interaction Model
- All 3 agents interact simultaneously in each round (not pairwise)
- Uses the linear payoff functions from `npdl/core/utils.py`:
  - Cooperation: `Payoff_C(n) = S + (R-S) × (n/(N-1))`
  - Defection: `Payoff_D(n) = P + (T-P) × (n/(N-1))`
  - Where `n` = number of cooperating neighbors (excluding self)

### Strategies Implemented
1. **Always Cooperate** - Always plays C
2. **Always Defect** - Always plays D
3. **Tit-for-Tat** - Cooperates first round, then follows majority behavior from previous round

### Exploration Parameter
- Each agent has an exploration rate (0.0 to 1.0)
- With probability = exploration_rate, the agent does the OPPOSITE of their strategy
- Adds unpredictability and helps escape bad equilibria

### Verbose Output
The simulation provides extremely detailed round-by-round information:
- Decision phase showing each agent's strategy suggestion and actual choice
- Exploration notifications when agents deviate from their strategy
- Detailed payoff calculations with formulas
- Round interpretations (full cooperation, exploitation, etc.)
- Running scores and cooperation statistics

## Usage Examples

### Interactive Simulation
```bash
python examples/simple_models/run_npd_simulation.py
```
Runs 5 pre-configured examples demonstrating different scenarios.

### Batch Export for Analysis
```bash
python examples/simple_models/run_npd_csv_export.py
```
Runs multiple scenarios with 10 runs each, exporting to CSV format compatible with your analysis tools.

### Custom Usage
```python
from examples.simple_models.simple_npd import NPDAgent, Strategy, NPrisonersDilemmaGame, NPersonSimulation

# Create agents
alice = NPDAgent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
bob = NPDAgent("Bob", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
charlie = NPDAgent("Charlie", Strategy.ALWAYS_DEFECT, exploration_rate=0.2)

# Create game
game = NPrisonersDilemmaGame(N=3, R=3.0, S=0.0, T=5.0, P=1.0)

# Run simulation
sim = NPersonSimulation([alice, bob, charlie], game)
sim.run_simulation(num_rounds=10)
```

## Key Insights from Examples

1. **Without exploration**: Always Defect dominates, exploiting cooperators
2. **With exploration**: Outcomes become more variable, sometimes breaking defection spirals
3. **Modified payoffs**: When cooperation reward is higher relative to temptation, stable cooperation can emerge
4. **Tit-for-Tat dynamics**: In N-Person, TFT can get stuck in defection if majority defects

## Integration with Main Framework

The N-Person simulation is designed to be compatible with your existing tools:
- CSV export format matches the main npdl framework
- Can be analyzed using `analyze_results.py`
- Network structure is marked as "fully_connected" with "n_person" interaction mode
- Results can be compared directly with pairwise simulations

## Next Steps

You can now:
1. Run comparative experiments between pairwise and N-Person dynamics
2. Analyze how cooperation emerges differently in group vs dyadic interactions
3. Test how different network structures might affect N-Person games
4. Integrate more sophisticated strategies from the main framework

The implementation provides a solid foundation for exploring the "Tragedy Valley" vs "Reciprocity Hill" dynamics in group settings!