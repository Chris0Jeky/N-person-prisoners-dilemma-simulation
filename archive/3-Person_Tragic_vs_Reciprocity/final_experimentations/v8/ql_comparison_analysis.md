# Q-Learning Implementation Comparison Analysis

## Key Differences Between Legacy QL and Vanilla/Adaptive QL

### 1. State Representation

**Vanilla QL (PairwiseAdaptiveQLearner):**
- Simple tuple representation: `str(tuple(history))` with last 2 moves
- Example: `"((0, 1), (1, 0))"` 
- Total possible states: 16 (4^2 combinations)

**Legacy QL:**
- Detailed string representation: `"MCC_ODD"` format
- Includes MY moves and OPPONENT moves explicitly
- Example: `"MCD_OCC"` = "My: Cooperate then Defect, Opp: Cooperate then Cooperate"
- Total possible states: 16 + special states like "initial", "1round_MC_OC"

### 2. Q-Value Storage Format

**Vanilla QL:**
- Dictionary with string keys: `{'cooperate': 0.0, 'defect': 0.0}`
- Action lookup: `q_table[state]['cooperate']`

**Legacy QL:**
- Dictionary with integer keys: `{COOPERATE: 0.1, DEFECT: 0.1}`
- Action lookup: `q_table[state][COOPERATE]` where COOPERATE = 0
- **CRITICAL: Optimistic initialization at 0.1 instead of 0.0**

### 3. Action Selection Logic

**Vanilla QL:**
```python
if q_table[state]['cooperate'] >= q_table[state]['defect']:
    action = 'cooperate'
else:
    action = 'defect'
```

**Legacy QL:**
```python
if q_values[COOPERATE] >= q_values[DEFECT]:
    return COOPERATE
else:
    return DEFECT
```

### 4. Critical Bug in Vanilla QL Implementation

The Vanilla QL (PairwiseAdaptiveQLearner) has a **type mismatch issue**:
- Stores Q-values with string keys: `'cooperate'` and `'defect'`
- But returns integer actions: `COOPERATE` (0) and `DEFECT` (1)
- In `record_pairwise_outcome`, it tries to update using string keys from `context['action']`

### 5. Neighborhood State Differences

**Vanilla QL:**
- Simple categories: 'low', 'medium', 'high'
- No history tracking

**Legacy QL:**
- Tracks cooperation trends: 'up', 'down', 'stable'
- Includes own history: `"high_up_MCC"`
- More context-aware

## Why Legacy QL Defects Against TFT

1. **Optimistic Initialization (0.1)**: Starting both actions at 0.1 means the agent explores more initially, but against TFT's retaliation, defection might quickly become dominant.

2. **Detailed State Space**: The more granular state representation means:
   - Less generalization across similar situations
   - Needs more exploration to learn good policies
   - May get stuck in defection cycles with specific state patterns

3. **Epsilon Decay**: Legacy starts with ε=0.15 and decays to 0.01, while Vanilla uses fixed ε=0.1. The decay might lock in early defection patterns.

4. **Learning Against TFT**:
   - TFT retaliates immediately to defection
   - With optimistic initialization, Legacy explores defection early
   - Gets punished by TFT (both defect = reward 1)
   - But cooperation after defection gets exploited (S=0)
   - This creates a defection trap

## Behavioral Prediction

Against 2 TFT agents:
- **Vanilla/Adaptive**: Learns to cooperate because simpler state space generalizes better
- **Legacy**: Gets trapped in defection due to:
  - More complex state representation requiring more exploration
  - Optimistic initialization encouraging early defection
  - Epsilon decay locking in learned defection patterns
  - TFT's retaliation creating negative reinforcement for cooperation attempts

## Recommendations

1. For Legacy QL to cooperate better:
   - Use pessimistic initialization (0.0 or negative)
   - Increase initial epsilon for more exploration
   - Slower epsilon decay
   - Possibly simplify state representation

2. Fix Vanilla QL's type mismatch between string/integer actions