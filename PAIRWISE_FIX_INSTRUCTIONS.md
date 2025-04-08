# Pairwise Implementation Fix Instructions

## Problem Diagnosis

The pairwise interaction model has several issues with the TitForTat strategy implementation:

1. **TitForTat Memory Format**: The current implementation stores only an aggregate `opponent_coop_proportion` in the agent's memory, instead of tracking specific opponent moves.

2. **TitForTat Decision Logic**: The current implementation makes TFT defect only when a majority of opponents defect, but true TFT should defect if ANY opponent defected.

3. **Memory Structure**: The format used to update agent memory in pairwise mode lacks specific opponent information needed for proper TFT behavior.

## Complete Solution

### Step 1: Update Environment._run_pairwise_round

Replace the current implementation with the improved version:

```python
def _run_pairwise_round(self, rewiring_prob=0.0):
    """Run a single round using pairwise interactions with improved TFT support.
    
    This method fixes issues with TFT strategy in pairwise mode by:
    1. Tracking per-opponent moves in memory
    2. Using specific opponent moves for TFT strategies
    3. Maintaining compatibility with all strategies
    """
    # Step 1: Each agent chooses one move for the round
    agent_moves_for_round = {}
    
    # Special handling for TFT-like strategies
    for agent in self.agents:
        if agent.strategy_type in ["tit_for_tat", "generous_tit_for_tat", 
                                "suspicious_tit_for_tat", "tit_for_two_tats"]:
            # These strategies need special handling
            if agent.memory:
                last_round = agent.memory[-1]
                neighbor_moves = last_round.get('neighbor_moves', {})
                
                # Check if we have specific opponent moves
                if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
                    specific_moves = neighbor_moves['specific_opponent_moves']
                    # For standard TFT, defect if ANY opponent defected
                    if any(move == "defect" for move in specific_moves.values()):
                        move = "defect"
                    else:
                        move = "cooperate"
                elif agent.strategy_type == "tit_for_tat":
                    # Default to cooperate for first round or if no specific data
                    move = agent.choose_move([])
                else:
                    # For other strategies just use normal choose_move
                    move = agent.choose_move([])
            else:
                # First round - use strategy's default behavior
                move = agent.choose_move([])
        else:
            # Non-TFT strategies - use normal choose_move
            move = agent.choose_move([])
        
        agent_moves_for_round[agent.agent_id] = move
    
    # Step 2: Simulate Pairwise Games and track specific opponent moves
    pairwise_payoffs = {agent.agent_id: [] for agent in self.agents}
    specific_opponent_moves = {agent.agent_id: {} for agent in self.agents}
    opponent_coop_counts = {agent.agent_id: 0 for agent in self.agents}
    opponent_total_counts = {agent.agent_id: 0 for agent in self.agents}
    
    agent_ids = [a.agent_id for a in self.agents]
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            agent_i_id = agent_ids[i]
            agent_j_id = agent_ids[j]
            move_i = agent_moves_for_round[agent_i_id]
            move_j = agent_moves_for_round[agent_j_id]
            
            # Calculate payoffs
            payoff_i, payoff_j = get_pairwise_payoffs(move_i, move_j, self.R, self.S, self.T, self.P)
            
            # Store payoffs
            pairwise_payoffs[agent_i_id].append(payoff_i)
            pairwise_payoffs[agent_j_id].append(payoff_j)
            
            # Track specific opponent moves (crucial for TFT)
            specific_opponent_moves[agent_i_id][agent_j_id] = move_j
            specific_opponent_moves[agent_j_id][agent_i_id] = move_i
            
            # Also track aggregate cooperation stats
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Update agent memories with BOTH specific and aggregate data
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        # Store total payoffs for return value
        round_payoffs_total[agent_id] = agent_total_payoff
        
        # Calculate aggregate opponent behavior
        opp_total = opponent_total_counts[agent_id]
        opp_coop = opponent_coop_counts[agent_id]
        opp_coop_prop = opp_coop / opp_total if opp_total > 0 else 0.5
        
        # Update agent score
        agent.score += agent_total_payoff
        
        # Create a hybrid memory format with both aggregate and specific data
        neighbor_moves = {
            'opponent_coop_proportion': opp_coop_prop,
            'specific_opponent_moves': specific_opponent_moves[agent_id]
        }
        
        # Update memory with both data formats
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff
        )
    
    # Step 4: Q-Value Updates (using aggregate format for RL strategies)
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Create the expected format for Q-learning
            neighbor_moves = {'opponent_coop_proportion': 
                             opponent_coop_counts[agent.agent_id] / opponent_total_counts[agent.agent_id] 
                             if opponent_total_counts[agent.agent_id] > 0 else 0.5}
            
            # Update Q-values
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_total[agent.agent_id] / max(1, len(specific_opponent_moves[agent.agent_id])),
                next_state_actions=neighbor_moves
            )
    
    # Handle network visualization (has no functional effect in pairwise mode)
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total
```

### Step 2: Update TitForTatStrategy.choose_move

Replace the current implementation with the improved version:

```python
def choose_move(self, agent, neighbors):
    """Choose the move for a TitForTat agent in both interaction modes.
    
    This version handles both standard and pairwise interaction formats.
    """
    # First round or no memory - default to cooperate
    if not agent.memory:
        return "cooperate"
    
    # Get the last round's information
    last_round = agent.memory[-1]
    neighbor_moves = last_round.get('neighbor_moves', {})
    
    # CASE 1: Check for specific opponent moves format (improved pairwise mode)
    if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
        specific_moves = neighbor_moves['specific_opponent_moves']
        if specific_moves:
            # For true TFT, defect if ANY specific opponent defected
            if any(move == "defect" for move in specific_moves.values()):
                return "defect"
            return "cooperate"
    
    # CASE 2: Check for simple opponent_coop_proportion (old pairwise format)
    elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        # In standard TFT, we want to defect if ANY opponent defected
        # So we only cooperate if the cooperation rate is 100% (or very close)
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        if coop_proportion < 0.99:  # Allow for floating point imprecision
            return "defect"
        return "cooperate"
    
    # CASE 3: Standard neighborhood format - original implementation
    elif neighbor_moves:
        # Get any neighbor's move from the last round
        random_neighbor_id = random.choice(list(neighbor_moves.keys()))
        return neighbor_moves[random_neighbor_id]
    
    # Default to cooperate if no neighbors or memory is in an unexpected format
    return "cooperate"
```

### Step 3: Update Other TFT-like Strategies

Apply similar updates to:
- GenerousTitForTatStrategy
- SuspiciousTitForTatStrategy
- TitForTwoTatsStrategy

Each of these strategies needs the same memory format handling, with their specific strategy variations.

### Step 4: Create a Temporary Workaround

Until the core library is updated, the simplest workaround is to modify TitForTatStrategy in memory by directly accessing and modifying the class in the agents module:

```python
import importlib
import types
from npdl.core import agents

# Define the improved choose_move method
def improved_tft_choose_move(self, agent, neighbors):
    # [implementation as shown above]
    pass

# Apply the patch
agents.TitForTatStrategy.choose_move = improved_tft_choose_move
```

## Recommendations for Project Structure

Based on the analysis of the codebase, here are some recommendations for better organization:

1. **Separate Interaction Modes**: Create separate modules for different interaction modes, with clear interfaces between them.

2. **Strategy Adapters**: Implement a strategy adapter pattern that allows strategies to work with different interaction modes without modification.

3. **Unit Tests**: Expand test coverage to ensure all strategies work correctly in all interaction modes.

4. **Memory Standardization**: Define a clear standard for agent memory format that works across interaction modes.

5. **Documentation**: Improve documentation to clarify how different interaction modes work and how strategies should handle them.

## Quick Fix for Testing

For immediate testing, just update the test_pairwise.py file to use the improved_tft_behavior function we provided. This allows you to validate the behavior without changing the core library.

## Full Implementation Plan

1. Create proper backups of all files before making changes
2. Update Environment._run_pairwise_round method with the improved implementation
3. Update TitForTatStrategy.choose_move method with the improved implementation
4. Update other TFT-like strategies with corresponding improvements
5. Run all tests to verify the changes work correctly
6. Document the changes and update any relevant documentation

The full implementation should be done as a coordinated change, not piecemeal, to ensure consistency across the codebase.
