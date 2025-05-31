# Simple Iterated Prisoner's Dilemma Simulations

This directory contains simple, standalone implementations of the Iterated Prisoner's Dilemma (IPD) with 3 agents.

## Files

- `simple_ipd.py` - Core simulation logic with Agent, Strategy, and Game classes
- `run_simple_simulation.py` - Pre-configured example simulations
- `interactive_simulation.py` - Interactive configuration tool
- `csv_exporter.py` - Export results in npdl-compatible CSV format
- `run_csv_export.py` - Run batch simulations with CSV export
- `analyze_results.py` - Analyze and visualize CSV results
- `requirements.txt` - Python dependencies for analysis

## Features

1. **Three Simple Strategies:**
   - Always Cooperate: Always plays C
   - Always Defect: Always plays D
   - Tit-for-Tat: Starts with C, then copies opponent's last move

2. **Exploration Parameter:** 
   - Each agent can have an exploration rate (0.0 to 1.0)
   - With probability = exploration_rate, the agent does the OPPOSITE of what their strategy dictates
   - This adds unpredictability and can help escape bad equilibria

3. **Verbose Output:**
   - Shows detailed information for each round
   - Displays when exploration occurs
   - Tracks scores and cooperation rates

## Usage

### Install Requirements (for CSV export and analysis)

```bash
pip install -r simple_models/requirements.txt
```

### Run Example Simulations

```bash
python simple_models/run_simple_simulation.py
```

This runs three pre-configured examples:
1. Basic strategies with no exploration
2. Strategies with exploration (10-20%)
3. Longer simulation (10 rounds) with Tit-for-Tat agents

### Interactive Mode

```bash
python simple_models/interactive_simulation.py
```

This allows you to:
- Choose strategies for each agent
- Set exploration rates
- Configure number of rounds
- Optionally customize payoff matrix

### Custom Usage

```python
from simple_models.simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation

# Create agents
alice = Agent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
bob = Agent("Bob", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
charlie = Agent("Charlie", Strategy.ALWAYS_DEFECT, exploration_rate=0.2)

# Create game with standard payoffs
game = PrisonersDilemmaGame()  # T=5, R=3, P=1, S=0

# Run simulation
sim = Simulation([alice, bob, charlie], game)
sim.run_simulation(num_rounds=10)
```

## Payoff Matrix

Default payoffs follow the standard IPD structure:
- T (Temptation) = 5: I defect, you cooperate
- R (Reward) = 3: Both cooperate
- P (Punishment) = 1: Both defect  
- S (Sucker) = 0: I cooperate, you defect

Where T > R > P > S and 2R > T + S

## Example Output

The simulation provides detailed round-by-round information:

```
ROUND 1
===========
--- Alice vs Bob ---
  Alice (Tit-for-Tat):
    Base action: C, Actual: C
  Bob (Always Cooperate):
    Base action: C, Actual: D [EXPLORED! (rate=10.0%)]
  Result: Alice C vs Bob D
  Payoffs: Alice gets 0, Bob gets 5
  → Bob exploited Alice!
```

## CSV Export for Analysis

The simulation can export results in CSV format compatible with the main npdl analysis tools:

### Basic Export

```bash
# Run batch simulations with CSV export
python simple_models/run_csv_export.py

# Run a single export example
python simple_models/run_csv_export.py --single

# Create summary statistics from existing results
python simple_models/run_csv_export.py --summary
```

### Export Format

Results are saved in the same format as the main framework:
- `experiment_results_agents.csv` - Agent summary with final scores
- `experiment_results_rounds.csv` - Round-by-round moves and payoffs
- `experiment_results_network.json` - Network structure (fully connected)

### Run Analysis

After running CSV export, analyze the results:

```bash
# Generate plots and statistics
python simple_models/analyze_results.py
```

This creates several visualizations in the `simple_results` directory:
- `average_scores_by_strategy.png` - Bar chart of average scores
- `cooperation_rates_by_strategy.png` - Cooperation rates for each strategy
- `scores_by_scenario.png` - Comparison across different scenarios
- `exploration_effect.png` - Impact of exploration on cooperation
- `cooperation_over_time_*.png` - Time series of cooperation rates

### Using with Main Analysis Tools

The exported CSV files are compatible with the main npdl analysis tools:

```python
# After exporting from simple models
from npdl.analysis.analysis import analyze_multiple_scenarios

# Analyze simple model results
scenario_names = ['TFT_vs_Defect_NoExploration', 'Mixed_Strategies_HighExploration']
analyze_multiple_scenarios(scenario_names, 'simple_results', 'simple_analysis_results')
```

## Insights

- **Always Defect** typically dominates in short games without exploration
- **Tit-for-Tat** performs well when paired with other cooperative strategies
- **Exploration** can help cooperative strategies occasionally "test" defectors
- Longer games tend to favor reciprocal strategies like Tit-for-Tat
