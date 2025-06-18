# Simple Iterated Prisoner's Dilemma Simulations

This directory contains simple, standalone implementations of the Iterated Prisoner's Dilemma (IPD) with 3 agents, supporting both pairwise and N-Person interactions.

## Files

### Pairwise Version (Original)
- `simple_ipd.py` - Core pairwise simulation logic with Agent, Strategy, and Game classes
- `run_simple_simulation.py` - Pre-configured pairwise example simulations
- `interactive_simulation.py` - Interactive configuration tool for pairwise games
- `csv_exporter.py` - Export pairwise results in npdl-compatible CSV format
- `run_csv_export.py` - Run batch pairwise simulations with CSV export

### N-Person Version (New)
- `simple_npd.py` - N-Person simulation where all agents interact simultaneously
- `run_npd_simulation.py` - Pre-configured N-Person example simulations
- `run_npd_csv_export.py` - Run batch N-Person simulations with CSV export

### Analysis Tools
- `analyze_results.py` - Analyze and visualize CSV results from both versions
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
pip install -r examples/simple_models/requirements.txt
```

### Run Example Simulations

```bash
python examples/simple_models/run_simple_simulation.py
```

This runs three pre-configured examples:
1. Basic strategies with no exploration
2. Strategies with exploration (10-20%)
3. Longer simulation (10 rounds) with Tit-for-Tat agents

### Interactive Mode

```bash
python examples/simple_models/interactive_simulation.py
```

This allows you to:
- Choose strategies for each agent
- Set exploration rates
- Configure number of rounds
- Optionally customize payoff matrix

### Custom Usage

```python
from examples.simple_models.simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation

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
python examples/simple_models/run_csv_export.py

# Run a single export example
python examples/simple_models/run_csv_export.py --single

# Create summary statistics from existing results
python examples/simple_models/run_csv_export.py --summary
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
python examples/simple_models/analyze_results.py
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

## N-Person Iterated Prisoner's Dilemma

The N-Person version (`simple_npd.py`) implements a true multi-agent interaction where all agents play simultaneously.

### Key Differences from Pairwise

1. **Simultaneous Interaction**: All 3 agents interact in a single game each round
2. **Payoff Calculation**: Uses linear payoff functions from `npdl.core.utils`:
   - Cooperation: `Payoff_C(n) = S + (R-S) × (n/(N-1))`
   - Defection: `Payoff_D(n) = P + (T-P) × (n/(N-1))`
   - Where `n` = number of cooperating neighbors (excluding self)

3. **Tit-for-Tat Strategy**: In N-Person, TFT cooperates if majority cooperated last round

### Running N-Person Simulations

#### Interactive Examples
```bash
python examples/simple_models/run_npd_simulation.py
```

This runs 5 pre-configured N-Person examples:
1. Classic strategies with no exploration
2. Strategies with exploration
3. All Tit-for-Tat with varying exploration
4. Modified payoff structure (higher cooperation reward)
5. Longer simulation to observe patterns

#### Batch Export for Analysis
```bash
python examples/simple_models/run_npd_csv_export.py
```

This runs multiple N-Person scenarios with 10 runs each and exports to CSV.

### N-Person Usage Example

```python
from examples.simple_models.simple_npd import NPDAgent, Strategy, NPrisonersDilemmaGame, NPersonSimulation

# Create 3 agents
alice = NPDAgent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
bob = NPDAgent("Bob", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
charlie = NPDAgent("Charlie", Strategy.ALWAYS_DEFECT, exploration_rate=0.2)

# Create N-Person game
game = NPrisonersDilemmaGame(N=3, R=3.0, S=0.0, T=5.0, P=1.0)

# Run simulation
sim = NPersonSimulation([alice, bob, charlie], game)
sim.run_simulation(num_rounds=10)
```

### N-Person Output Example

The N-Person simulation provides extremely detailed output:

```
ROUND 1
======
--- DECISION PHASE ---
  Alice (Tit-for-Tat):
    Strategy suggests: C
    Final decision: C

  Bob (Always Cooperate):
    Strategy suggests: C
    EXPLORED to: D

  Charlie (Always Defect):
    Strategy suggests: D
    Final decision: D

--- PAYOFF CALCULATION ---
  Total cooperators: 1/3
  Total defectors: 2/3

  Individual payoffs:
    Alice: C → 0.00 points
      (Cooperated with 0 cooperating neighbors)
      Payoff = S + (R-S) × (n/(N-1)) = 0 + 3 × (0/2) = 0.00

    Bob: D → 1.00 points
      (Defected against 0 cooperating neighbors)
      Payoff = P + (T-P) × (n/(N-1)) = 1 + 4 × (0/2) = 1.00
```

### Comparing Pairwise vs N-Person

The framework allows direct comparison between pairwise and N-Person dynamics:

- **Pairwise**: 3 agents play 3 separate 2-player games per round
- **N-Person**: 3 agents play 1 shared game per round

This can reveal how cooperation dynamics differ between dyadic and group interactions.
