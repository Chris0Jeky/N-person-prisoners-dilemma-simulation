# True Pairwise Implementation Guide

## Overview

The true pairwise implementation addresses a fundamental limitation in the original pairwise mode: agents can now make **individual decisions for each opponent** rather than being forced to use one action for all opponents in a round.

## Key Differences

### Original Pairwise Mode (Aggregate)
- Agents make **one decision per round** that applies to all opponents
- Agents track aggregate cooperation statistics across all opponents
- Example: If TFT faces 5 cooperators and 5 defectors, it must choose to either cooperate or defect with ALL of them

### True Pairwise Mode (Individual)
- Agents make **separate decisions for each opponent**
- Agents maintain **opponent-specific memories and learning states**
- Example: TFT can cooperate with the 5 cooperators while defecting against the 5 defectors

## Implementation Structure

### Core Module: `npdl/core/true_pairwise.py`

Contains the main implementation:

1. **OpponentSpecificMemory**: Tracks interactions with individual opponents
   - Stores interaction history per opponent
   - Calculates opponent-specific statistics (cooperation rate, reciprocity score)
   - Maintains limited memory window for efficiency

2. **TruePairwiseAgent**: Base class for agents with individual decision-making
   - Abstract method `choose_action_for_opponent(opponent_id, round_num)`
   - Manages collection of opponent-specific memories
   - Provides methods for episode resets and statistics

3. **Strategy Implementations**:
   - `TruePairwiseTFT`: Tit-for-Tat with per-opponent tracking
   - `TruePairwiseGTFT`: Generous TFT with opponent-specific generosity
   - `TruePairwisePavlov`: Win-Stay/Lose-Shift per opponent
   - `TruePairwiseQLearning`: Q-learning with separate Q-tables per opponent
   - `TruePairwiseAdaptive`: Identifies and adapts to opponent strategies

4. **TruePairwiseEnvironment**: Manages true pairwise simulations
   - Handles individual games between agent pairs
   - Supports noise, episodes, and memory resets
   - Compiles detailed statistics on bilateral relationships

### Adapter Module: `npdl/core/true_pairwise_adapter.py`

Provides integration with existing framework:

1. **TruePairwiseAgentAdapter**: Wraps existing agents for true pairwise mode
2. **create_true_pairwise_agent**: Factory function for creating agents
3. **TruePairwiseSimulationAdapter**: Runs simulations and converts results

### Integration: Modified `npdl/simulation/runner.py`

Added support for `pairwise_mode` configuration:
- `"aggregate"`: Uses original implementation (default)
- `"individual"`: Uses true pairwise implementation

## Usage

### Configuration

Add to your scenario JSON:

```json
{
  "name": "True_Pairwise_Example",
  "interaction_mode": "pairwise",
  "pairwise_mode": "individual",  // Enable true pairwise
  "agent_strategies": [
    {"type": "true_pairwise_tft", "count": 10},
    {"type": "true_pairwise_adaptive", "count": 5},
    {"type": "always_cooperate", "count": 5}
  ],
  "network_type": "fully_connected",
  "num_rounds": 100
}
```

### Running Simulations

```bash
# Using the standard runner
python run.py scenarios/true_pairwise_scenarios.json

# Using the demonstration script
python demonstrate_true_pairwise.py
```

### Programmatic Usage

```python
from npdl.core.true_pairwise import TruePairwiseTFT, TruePairwiseEnvironment

# Create agents
agents = [
    TruePairwiseTFT("agent1"),
    TruePairwiseTFT("agent2"),
    # ... more agents
]

# Create environment
env = TruePairwiseEnvironment(
    agents=agents,
    rounds_per_episode=100,
    noise_level=0.05
)

# Run simulation
results = env.run_simulation()
```

## Advanced Features

### 1. State Representations for Q-Learning

The true pairwise Q-learning implementation supports multiple state representations:

- **"basic"**: Based on opponent's last move
- **"proportion"**: Based on opponent's cooperation rate
- **"memory_enhanced"**: Based on recent interaction pattern
- **"reciprocity"**: Based on opponent's reciprocity score

### 2. Adaptive Strategy

The adaptive agent can identify opponent strategies:
- Always cooperate/defect
- Tit-for-Tat variants
- Random behavior
- Mixed strategies

### 3. Episode Support

Simulations can be divided into episodes with optional memory resets between episodes, useful for studying:
- Learning dynamics
- Strategy evolution
- Forgiveness and reputation rebuilding

### 4. Noise Handling

Implementation errors (noise) can be added to study robustness:
- Actions may be flipped with specified probability
- Agents must adapt to noisy opponent signals

## Benefits of True Pairwise Mode

1. **Realistic Reciprocity**: Agents can maintain different relationships with different partners
2. **Better Performance**: Agents can optimize strategies per opponent rather than using one-size-fits-all
3. **Richer Dynamics**: More complex social structures emerge from individual relationships
4. **Learning Efficiency**: RL agents can learn faster with opponent-specific Q-tables
5. **Strategy Diversity**: New strategies become viable that exploit individual tracking

## Testing

Comprehensive test suite in `tests/test_true_pairwise.py`:
- Unit tests for memory and agent components
- Integration tests for environment
- Comparison tests between modes
- Performance benchmarks

Run tests:
```bash
pytest tests/test_true_pairwise.py -v
```

## Migration Guide

To convert existing scenarios to true pairwise:

1. Add `"pairwise_mode": "individual"` to scenarios with `"interaction_mode": "pairwise"`
2. Optionally replace agent types with `true_pairwise_*` variants for better performance
3. Consider adding `"episodes"` and `"reset_between_episodes"` for episodic structure
4. Adjust `num_rounds` as true pairwise may converge differently

## Performance Considerations

- Memory usage scales with O(n²) for n agents (each tracks all others)
- Computation is still O(n²) per round (same as aggregate pairwise)
- Use memory_length parameter to limit history storage
- Consider using fewer agents or episodes for large-scale experiments

## Future Extensions

Potential enhancements:
1. Network effects: Reputation spreading between agents
2. Group dynamics: Coalition formation based on bilateral relationships  
3. Communication: Agents sharing information about opponents
4. Meta-strategies: Agents reasoning about opponent's individual tracking