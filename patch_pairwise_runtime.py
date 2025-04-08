#!/usr/bin/env python3
"""
Runtime patcher for the pairwise interaction model.

This script applies runtime patches to fix issues with the TitForTat strategy in pairwise mode
without modifying the source files. It's a good way to test the fixes before making permanent changes.

Usage:
    # Import this before loading any NPDL modules
    import patch_pairwise_runtime

    # Or apply the patch later
    import patch_pairwise_runtime
    patch_pairwise_runtime.apply_patches()

    # Then run tests normally
    import test_pairwise
    test_pairwise.main()
"""

import sys
import importlib
import types
import logging

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def improved_tft_choose_move(self, agent, neighbors):
    """Improved TitForTat strategy that handles pairwise mode correctly.
    
    This implementation properly checks for specific opponent moves in memory
    and defects if ANY opponent defected in the previous round.
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
        try:
            # Get any neighbor's move from the last round
            import random
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            return neighbor_moves[random_neighbor_id]
        except (KeyError, IndexError, TypeError):
            # If there's any issue, default to cooperate
            return "cooperate"
    
    # Default to cooperate if no neighbors or memory is in an unexpected format
    return "cooperate"

def improved_generous_tft_choose_move(self, agent, neighbors):
    """Improved GenerousTitForTat strategy that handles pairwise mode correctly.
    
    This implementation properly checks for specific opponent moves in memory
    and occasionally forgives defection based on the generosity parameter.
    """
    if not agent.memory:
        return "cooperate"
    
    last_round = agent.memory[-1]
    neighbor_moves = last_round.get('neighbor_moves', {})
    
    # Handle improved pairwise format with specific opponent moves
    if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
        specific_moves = neighbor_moves['specific_opponent_moves']
        if specific_moves:
            # Check if any opponent defected
            if any(move == "defect" for move in specific_moves.values()):
                # Apply generosity - sometimes cooperate even when should defect
                import random
                if random.random() < self.generosity:
                    return "cooperate"
                return "defect"
            return "cooperate"
    
    # Handle old pairwise format with opponent_coop_proportion
    elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        # If not all cooperated, consider defecting but apply generosity
        if coop_proportion < 0.99:
            import random
            if random.random() < self.generosity:
                return "cooperate"
            return "defect"
        return "cooperate"
    
    # Handle standard neighborhood format
    elif neighbor_moves:
        try:
            import random
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            move = neighbor_moves[random_neighbor_id]
            # Occasionally forgive defection
            if move == "defect" and random.random() < self.generosity:
                return "cooperate"
            return move
        except (KeyError, IndexError, TypeError):
            return "cooperate"
    
    return "cooperate"

def improved_suspicious_tft_choose_move(self, agent, neighbors):
    """Improved SuspiciousTitForTat strategy that handles pairwise mode correctly.
    
    This implementation starts with defection and properly responds to opponent moves.
    """
    if not agent.memory:
        return "defect"  # Start with defection
    
    last_round = agent.memory[-1]
    neighbor_moves = last_round.get('neighbor_moves', {})
    
    # Handle improved pairwise format with specific opponent moves
    if isinstance(neighbor_moves, dict) and 'specific_opponent_moves' in neighbor_moves:
        specific_moves = neighbor_moves['specific_opponent_moves']
        if specific_moves:
            # Standard TFT behavior after first round
            if any(move == "defect" for move in specific_moves.values()):
                return "defect"
            return "cooperate"
    
    # Handle old pairwise format with opponent_coop_proportion
    elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
        coop_proportion = neighbor_moves['opponent_coop_proportion']
        # Only cooperate if ALL cooperated
        return "cooperate" if coop_proportion >= 0.99 else "defect"
    
    # Handle standard neighborhood format
    elif neighbor_moves:
        try:
            import random
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            return neighbor_moves[random_neighbor_id]
        except (KeyError, IndexError, TypeError):
            return "defect"
    
    return "defect"  # Default to defect if no info

def improved_tft2t_choose_move(self, agent, neighbors):
    """Improved TitForTwoTats strategy that handles pairwise mode correctly.
    
    This implementation defects only if opponents defected twice in a row.
    """
    # Cooperate for the first two rounds or if insufficient memory
    if len(agent.memory) < 2:
        return "cooperate"

    last_round = agent.memory[-1]
    prev_round = agent.memory[-2]

    # Handle improved pairwise format with specific opponent moves
    last_specific_moves = last_round.get('neighbor_moves', {}).get('specific_opponent_moves', {})
    prev_specific_moves = prev_round.get('neighbor_moves', {}).get('specific_opponent_moves', {})
    
    if last_specific_moves and prev_specific_moves:
        # Find common opponents present in both rounds
        common_opponents = set(last_specific_moves.keys()) & set(prev_specific_moves.keys())
        # Defect only if any opponent defected twice in a row
        for opp_id in common_opponents:
            if last_specific_moves[opp_id] == "defect" and prev_specific_moves[opp_id] == "defect":
                return "defect"
        return "cooperate"
    
    # Handle old pairwise format with opponent_coop_proportion
    last_coop_prop = last_round.get('neighbor_moves', {}).get('opponent_coop_proportion', 1.0)
    prev_coop_prop = prev_round.get('neighbor_moves', {}).get('opponent_coop_proportion', 1.0)
    
    # Since we don't have specific opponent tracking, use a more conservative approach
    # Only defect if the cooperation proportion was very low (< 0.5) twice in a row
    if last_coop_prop < 0.5 and prev_coop_prop < 0.5:
        return "defect"
    
    # For standard neighborhood format, use the original implementation
    try:
        last_neighbor_moves = last_round.get('neighbor_moves', {})
        prev_neighbor_moves = prev_round.get('neighbor_moves', {})
        
        if not isinstance(last_neighbor_moves, dict) or not isinstance(prev_neighbor_moves, dict):
            return "cooperate"
        
        # Try to find a neighbor present in both rounds
        import random
        common_neighbors = list(set(last_neighbor_moves.keys()) & set(prev_neighbor_moves.keys()))
        
        if not common_neighbors:
            # If no common neighbors, pick randomly from last round's neighbors
            if not last_neighbor_moves:
                return "cooperate"
            random_neighbor_id = random.choice(list(last_neighbor_moves.keys()))
            last_opponent_move = last_neighbor_moves.get(random_neighbor_id, "cooperate")
            prev_opponent_move = "cooperate"  # Default
        else:
            random_neighbor_id = random.choice(common_neighbors)
            last_opponent_move = last_neighbor_moves.get(random_neighbor_id, "cooperate")
            prev_opponent_move = prev_neighbor_moves.get(random_neighbor_id, "cooperate")
        
        # Defect only if the chosen opponent defected twice in a row
        if last_opponent_move == "defect" and prev_opponent_move == "defect":
            return "defect"
    except Exception:
        # In case of any error, default to cooperate
        pass
        
    return "cooperate"

def improved_run_pairwise_round(self, rewiring_prob=0.0):
    """Improved implementation of pairwise round with proper TFT support.
    
    This implementation correctly tracks per-opponent moves and stores them
    in the agent's memory for proper strategy decisions.
    """
    # Step 1: Each agent chooses one move for the round
    agent_moves_for_round = {}
    
    for agent in self.agents:
        try:
            move = agent.choose_move([])
        except Exception:
            # Fallback to standard approach with network neighbors
            try:
                neighbors = self.get_neighbors(agent.agent_id)
                move = agent.choose_move(neighbors)
            except Exception:
                # Ultimate fallback - default to cooperate
                move = "cooperate"
        
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
            from npdl.core.utils import get_pairwise_payoffs
            payoff_i, payoff_j = get_pairwise_payoffs(move_i, move_j, self.R, self.S, self.T, self.P)
            
            # Store payoffs
            pairwise_payoffs[agent_i_id].append(payoff_i)
            pairwise_payoffs[agent_j_id].append(payoff_j)
            
            # Track specific opponent moves (crucial for TFT)
            specific_opponent_moves[agent_i_id][agent_j_id] = move_j
            specific_opponent_moves[agent_j_id][agent_i_id] = move_i
            
            # Also track aggregated opponent cooperation stats
            opponent_total_counts[agent_i_id] += 1
            opponent_total_counts[agent_j_id] += 1
            if move_j == "cooperate":
                opponent_coop_counts[agent_i_id] += 1
            if move_i == "cooperate":
                opponent_coop_counts[agent_j_id] += 1
    
    # Step 3: Update agent memories with BOTH specific and aggregate data
    round_payoffs_avg = {}
    round_payoffs_total = {}
    
    for agent in self.agents:
        agent_id = agent.agent_id
        agent_total_payoff = sum(pairwise_payoffs[agent_id])
        num_games = len(pairwise_payoffs[agent_id])
        agent_avg_payoff = agent_total_payoff / num_games if num_games > 0 else 0
        
        # Store payoff summaries
        round_payoffs_avg[agent_id] = agent_avg_payoff
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
        
        # Update memory with both aggregate and specific opponent data
        agent.update_memory(
            my_move=agent_moves_for_round[agent_id],
            neighbor_moves=neighbor_moves,
            reward=agent_avg_payoff
        )
    
    # Step 4: Q-Value Updates (using the format RL strategies expect)
    for agent in self.agents:
        if agent.strategy_type in ("q_learning", "q_learning_adaptive", 
                                  "lra_q", "hysteretic_q", "wolf_phc", "ucb1_q"):
            # Q-learning strategies use the aggregate format
            neighbor_moves = {'opponent_coop_proportion': 
                            opponent_coop_counts[agent.agent_id] / opponent_total_counts[agent.agent_id] 
                            if opponent_total_counts[agent.agent_id] > 0 else 0.5}
            
            agent.update_q_value(
                action=agent_moves_for_round[agent.agent_id],
                reward=round_payoffs_avg[agent.agent_id],
                next_state_actions=neighbor_moves
            )
    
    # Optional network visualization (not functional in pairwise mode)
    if rewiring_prob > 0:
        self.update_network(rewiring_prob=rewiring_prob)
    
    return agent_moves_for_round, round_payoffs_total

def apply_patches():
    """Apply runtime patches to fix pairwise mode issues."""
    try:
        # Import the modules we need to patch
        from npdl.core import agents, environment
        
        # Patch TitForTat strategy
        agents.TitForTatStrategy.choose_move = improved_tft_choose_move
        logger.info("✓ Patched TitForTatStrategy.choose_move")
        
        # Patch GenerousTitForTat strategy
        agents.GenerousTitForTatStrategy.choose_move = improved_generous_tft_choose_move
        logger.info("✓ Patched GenerousTitForTatStrategy.choose_move")
        
        # Patch SuspiciousTitForTat strategy
        agents.SuspiciousTitForTatStrategy.choose_move = improved_suspicious_tft_choose_move
        logger.info("✓ Patched SuspiciousTitForTatStrategy.choose_move")
        
        # Patch TitForTwoTats strategy
        agents.TitForTwoTatsStrategy.choose_move = improved_tft2t_choose_move
        logger.info("✓ Patched TitForTwoTatsStrategy.choose_move")
        
        # Patch Environment._run_pairwise_round
        environment.Environment._run_pairwise_round = improved_run_pairwise_round
        logger.info("✓ Patched Environment._run_pairwise_round")
        
        logger.info("All patches applied successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to apply patches: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Apply patches automatically when the module is imported
success = apply_patches()
if success:
    logger.info("Pairwise interaction model patched successfully!")
else:
    logger.warning("Failed to patch pairwise interaction model.")
