# State Space Analysis: 2-Round vs 3-Round History

## Current State Space (2-round history)

### Vanilla/Adaptive QL:
- State format: `((move1, opp_move1), (move2, opp_move2))`
- Possible states: 2^4 = 16 states
- Example: `((0,1), (1,0))`

### Legacy QL:
- State format: `"M{my_t-2}{my_t-1}_O{opp_t-2}{opp_t-1}"`
- Possible states: 2^4 = 16 main states + special states
- Example: `"MCC_ODD"`

## Proposed 3-Round History

### State Space Growth:
- 2-round: 2^4 = 16 states
- 3-round: 2^6 = 64 states
- **4x increase** in state space

### Benefits:
1. **Better Pattern Detection**: Can detect patterns like "Tit-for-Two-Tats" or alternating strategies
2. **Revenge/Forgiveness Detection**: Can see if opponent holds grudges for 2+ rounds
3. **Noise Filtering**: Can better distinguish between errors and intentional strategy changes

### Drawbacks:
1. **Slower Learning**: 4x more states to explore and learn Q-values for
2. **More Exploration Needed**: Higher epsilon or longer training time required
3. **Generalization Issues**: Less data per state, harder to learn robust policies

## Discount Factor Recommendations

### For 2-Round History (Current):
- **Standard**: γ = 0.95 (good balance)
- **Short-sighted**: γ = 0.9 (focuses on immediate rewards)
- **Long-sighted**: γ = 0.99 (values future cooperation)

### For 3-Round History:
- **Recommended**: γ = 0.97-0.99
- **Reasoning**: With more detailed history, you want to value future rewards more highly because:
  1. The agent can make more informed decisions
  2. Complex patterns take time to establish
  3. Higher γ helps learn stable cooperation strategies

## Specific Recommendations for TFT Scenarios

Against TFT opponents with 3-round history:

1. **Use γ = 0.98-0.99**: TFT has perfect memory, so long-term thinking is crucial
2. **Increase initial epsilon**: Start with ε = 0.25-0.3 for better exploration
3. **Slower epsilon decay**: Use 0.999 instead of 0.995
4. **Higher learning rate initially**: α = 0.15-0.2 to learn patterns faster

## Implementation Suggestion

```python
# For 3-round history variant
LEGACY_3ROUND_PARAMS = {
    'lr': 0.15,                  # Higher initial learning rate
    'df': 0.98,                  # Higher discount for long-term thinking
    'eps': 0.25,                 # More exploration needed
    'epsilon_decay': 0.999,      # Slower decay for larger state space
    'epsilon_min': 0.02,         # Slightly higher minimum
    'history_length': 3          # New parameter
}
```

## Expected Behavior Changes

With 3-round history against 2 TFT:
- **Early game**: More exploration, possibly more defection initially
- **Mid game**: Should discover mutual cooperation faster once patterns are learned
- **Late game**: More stable cooperation due to better pattern recognition

## Verdict

**Worth trying if**:
- You have enough rounds (40,000 should be sufficient)
- You suspect complex multi-round patterns matter
- You can afford longer training time

**Not recommended if**:
- Quick adaptation is more important than perfect play
- Running many short simulations
- Opponents use simple strategies (AllC, AllD, Random)