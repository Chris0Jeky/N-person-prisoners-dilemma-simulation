# Pairwise Interaction Mode Implementation Summary

## Overview
This document summarizes the implementation of robust pairwise interaction mode for reactive strategies in the N-Person Iterated Prisoner's Dilemma simulation framework.

## Changes Made

### 1. Environment (`npdl/core/environment.py`)

#### Modified: `_run_pairwise_round` method
- **Added**: Storage of specific opponent moves for each agent
- **Structure**: Each agent's memory now contains:
  ```python
  neighbor_moves = {
      "opponent_coop_proportion": float,  # For RL agents (unchanged)
      "specific_opponent_moves": {        # NEW: For reactive strategies
          opponent_id: "cooperate"/"defect",
          ...
      }
  }
  ```
- **Benefit**: Reactive strategies can now make decisions based on specific opponent history

### 2. Agent Strategies (`npdl/core/agents.py`)

#### Modified: TitForTatStrategy
- **Behavior**: Now defects if ANY opponent defected (true TFT behavior)
- **Priority**: Uses `specific_opponent_moves` when available, falls back to proportion
- **Threshold**: When using proportion fallback, only cooperates if proportion >= 0.99

#### Modified: GenerousTitForTatStrategy
- **Behavior**: Checks if any opponent defected, then applies generosity
- **Priority**: Uses specific moves when available
- **Consistency**: Maintains backward compatibility with proportion-based decisions

#### Modified: SuspiciousTitForTatStrategy
- **Behavior**: Only cooperates if ALL opponents cooperated
- **Priority**: Uses specific moves when available
- **Default**: Defects when no opponent information available

#### Modified: TitForTwoTatsStrategy
- **Behavior**: Tracks specific opponents across two rounds
- **Logic**: Defects only if a specific opponent defected in both previous rounds
- **Robustness**: Handles cases where opponents change between rounds

### 3. Tests Added

#### New File: `tests/test_pairwise_strategies.py`
Comprehensive unit tests for pairwise TFT variants including:
- First round behavior
- Response to any defection
- Response to all cooperation
- Fallback to proportion-based decisions
- Generous forgiveness behavior
- Suspicious initial defection
- Two-round memory for TF2T
- Edge cases (empty opponent lists, changing opponents)

#### New File: `tests/test_pairwise_integration.py`
Integration tests verifying:
- Correct storage of specific opponent moves
- TFT behavior across multiple rounds
- Payoff calculations in pairwise mode
- Comparison between neighborhood and pairwise modes

## Key Design Decisions

1. **Backward Compatibility**: Strategies gracefully fall back to proportion-based decisions when specific moves aren't available
2. **True TFT Behavior**: In pairwise mode, TFT defects if ANY opponent defected (not majority-based)
3. **Memory Structure**: Both aggregate and specific data stored to support all strategy types
4. **Q-Learning Unchanged**: RL strategies continue using `opponent_coop_proportion` without modification

## Testing Results
- All new tests pass (14 tests total)
- Existing tests remain unaffected
- Integration tests confirm correct end-to-end behavior

## Benefits

1. **Mechanistic Distinction**: Pairwise and neighborhood modes now produce distinctly different dynamics
2. **True Reactive Behavior**: TFT variants behave according to their canonical definitions
3. **Research Value**: Can now properly study the emergence of cooperation under different interaction structures
4. **Extensibility**: Easy to add new reactive strategies that use specific opponent history

## Future Considerations

1. **Performance**: Storing specific moves increases memory usage linearly with number of agents
2. **Analysis**: New analysis tools could leverage the richer interaction data
3. **Visualization**: Could visualize specific pairwise relationships and their evolution
