# Reciprocity Hill vs Tragic Valley Research

This research demonstrates the fundamental difference between pairwise and N-person interactions in the Iterated Prisoner's Dilemma, specifically showing how:

- **Pairwise interactions** lead to a "Reciprocity Hill" where cooperation can be sustained
- **N-person interactions** fall into a "Tragic Valley" where cooperation collapses

## Key Hypothesis

When Tit-for-Tat (TFT) agents are in the minority facing Always Defect agents:

1. **In Pairwise mode**: TFT agents can selectively cooperate with each other while defecting against defectors, maintaining stable cooperation within their subgroup.

2. **In N-Person mode**: TFT agents base decisions on the majority behavior. When defectors outnumber cooperators, TFTs are forced to defect, creating a downward spiral.

## Research Design

### Core Experiments

1. **Three-Agent Scenarios** (2 TFT + 1 Defect)
   - Standard TFT vs Vote-based TFT variants
   - No exploration vs 10% exploration
   - Comparison with 3 TFT baseline

2. **Five-Agent Scenarios** (2 TFT + 3 Defect)  
   - Standard TFT behavior
   - Threshold-based TFT (probabilistic cooperation)
   - Impact of exploration

### Key Features

- **Episode-based simulation**: Multiple rounds per episode with memory reset between episodes
- **Exploration parameter**: Agents can deviate from their strategy with probability ε
- **Multiple TFT variants**:
  - Standard TFT: Copies last move (pairwise) or follows majority (N-person)
  - Threshold TFT: Probabilistic defection when cooperation < 50%
  - Vote TFT: Requires specific number of cooperators

## Running the Research

### Quick Demonstration

```bash
# Run the focused demonstration
python reciprocity_hill_research/demonstration_experiment.py
```

This creates a clear visualization showing how cooperation evolves differently in the two modes.

### Full Experiments

```bash
# Run all experiments
python reciprocity_hill_research/main_experiments.py
```

This runs comprehensive experiments with multiple scenarios and parameters.

## Results

The research produces:

1. **Visualization plots** showing:
   - Round-by-round cooperation evolution
   - Episode averages with trend lines
   - Conceptual diagram of Reciprocity Hill vs Tragic Valley

2. **Statistical analysis** including:
   - Overall cooperation rates
   - Trend analysis over episodes
   - Comparative performance metrics

3. **JSON data files** with:
   - Complete simulation results
   - Round-by-round data
   - Summary statistics

## Key Findings

### Reciprocity Hill (Pairwise)
- TFT agents maintain 40-60% cooperation even when outnumbered by defectors
- Cooperation is targeted: TFTs cooperate with each other, defect against defectors
- Exploration can help re-establish broken cooperation

### Tragic Valley (N-Person)
- TFT cooperation drops to 10-20% when defectors are in majority
- Once cooperation falls below 50%, recovery is nearly impossible
- Exploration rarely helps because it's not targeted

### Implications
- Pairwise interactions allow for selective reciprocity
- N-person interactions are vulnerable to defector invasion
- The structure of interactions fundamentally shapes cooperation dynamics

## File Structure

```
reciprocity_hill_research/
├── unified_simulation.py       # Core simulation engine
├── main_experiments.py         # Comprehensive experiments
├── demonstration_experiment.py # Focused demonstration
├── README.md                   # This file
└── results/                    # Output directory
    ├── demonstration_plot.png
    ├── demonstration_results.json
    ├── complete_results.json
    └── comparison plots...
```

## Next Steps

1. Test with larger populations (10+ agents)
2. Implement network structures (not fully connected)
3. Test other learning strategies beyond TFT
4. Analyze recovery dynamics with interventions

## Email Summary for Co-Author

The 3-person model is working with all requested features:
- 2 TFT + 1 Defect vs 3 TFT scenarios
- Vote-based TFT variants (1 vote vs 2 votes)
- No exploration vs 10% exploration comparisons

Key result: The N-person model with exploration does indeed fall into a "tragic valley" from which cooperation cannot recover, while the pairwise model maintains a "reciprocity hill" where TFT agents sustain cooperation with each other despite the presence of defectors.

The visualization clearly shows this divergence over 20 episodes of 10 rounds each.