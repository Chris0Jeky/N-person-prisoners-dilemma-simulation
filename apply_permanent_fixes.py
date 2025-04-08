#!/usr/bin/env python3
"""
Apply permanent fixes to the pairwise implementation.

This script will:
1. Make backup copies of the files to be modified
2. Update the TitForTat strategies in agents.py
3. Update the Environment._run_pairwise_round method in environment.py
4. Run tests to verify the changes work

IMPORTANT: This script modifies source files. Run at your own risk!
"""

import os
import sys
import re
import shutil
import importlib
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def backup_file(filepath, backup_dir="backups"):
    """Create a backup of a file."""
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(filepath)
    backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
    
    # Copy the file
    shutil.copy2(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")
    return backup_path

def read_file(filepath):
    """Read a file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Write content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Updated file: {filepath}")

def update_tft_class(agents_content):
    """Update the TitForTatStrategy class in agents.py."""
    tft_class_pattern = r'class TitForTatStrategy\(Strategy\):.*?(?=class)'
    replacement = """class TitForTatStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        \"\"\"Choose the move for a TitForTat agent, handling both interaction modes.
        
        This version fixes issues in pairwise mode by:
        1. Properly checking for specific opponent moves
        2. Using the most conservative approach (defect if ANY opponent defected)
        3. Maintaining compatibility with the original format
        
        Args:
            agent: The agent making the decision
            neighbors: List of neighbor agent IDs
            
        Returns:
            String: "cooperate" or "defect"
        \"\"\"
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

"""
    # Replace the TitForTatStrategy class
    updated_content = re.sub(tft_class_pattern, replacement, agents_content, flags=re.DOTALL)
    
    return updated_content

def update_generous_tft_class(content):
    """Update the GenerousTitForTatStrategy class in agents.py."""
    generous_tft_class_pattern = r'class GenerousTitForTatStrategy\(Strategy\):.*?(?=class)'
    replacement = """class GenerousTitForTatStrategy(Strategy):
    def __init__(self, generosity=0.1):
        self.generosity = generosity
        
    def choose_move(self, agent, neighbors):
        \"\"\"Choose move for GenerousTitForTat agent, handling both interaction modes.
        
        Like TitForTat but occasionally forgives defection based on generosity parameter.
        \"\"\"
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
                    if random.random() < self.generosity:
                        return "cooperate"
                    return "defect"
                return "cooperate"
        
        # Handle old pairwise format with opponent_coop_proportion
        elif isinstance(neighbor_moves, dict) and 'opponent_coop_proportion' in neighbor_moves:
            coop_proportion = neighbor_moves['opponent_coop_proportion']
            # If not all cooperated, consider defecting but apply generosity
            if coop_proportion < 0.99:
                if random.random() < self.generosity:
                    return "cooperate"
                return "defect"
            return "cooperate"
        
        # Handle standard neighborhood format
        elif neighbor_moves:
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            move = neighbor_moves[random_neighbor_id]
            # Occasionally forgive defection
            if move == "defect" and random.random() < self.generosity:
                return "cooperate"
            return move
        
        return "cooperate"

"""
    # Replace the GenerousTitForTatStrategy class
    updated_content = re.sub(generous_tft_class_pattern, replacement, content, flags=re.DOTALL)
    
    return updated_content

def update_suspicious_tft_class(content):
    """Update the SuspiciousTitForTatStrategy class in agents.py."""
    suspicious_tft_class_pattern = r'class SuspiciousTitForTatStrategy\(Strategy\):.*?(?=class)'
    replacement = """class SuspiciousTitForTatStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        \"\"\"Choose move for SuspiciousTitForTat agent, handling both interaction modes.
        
        Starts with defection and then mimics opponent's previous move.
        \"\"\"
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
            random_neighbor_id = random.choice(list(neighbor_moves.keys()))
            return neighbor_moves[random_neighbor_id]
        
        return "defect"  # Default to defect if no info

"""
    # Replace the SuspiciousTitForTatStrategy class
    updated_content = re.sub(suspicious_tft_class_pattern, replacement, content, flags=re.DOTALL)
    
    return updated_content

def update_tit_for_two_tats_class(content):
    """Update the TitForTwoTatsStrategy class in agents.py."""
    tft2t_class_pattern = r'class TitForTwoTatsStrategy\(Strategy\):.*?(?=class)'
    replacement = """class TitForTwoTatsStrategy(Strategy):
    def choose_move(self, agent, neighbors):
        \"\"\"Choose move for TitForTwoTats agent, handling both interaction modes.
        
        Defects only after opponent defects twice in a row.
        \"\"\"
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
        last_neighbor_moves = last_round.get('neighbor_moves', {})
        prev_neighbor_moves = prev_round.get('neighbor_moves', {})

        # Handle cases where neighbors might not exist in both rounds
        if not last_neighbor_moves or not prev_neighbor_moves:
            return "cooperate" # Default to cooperate if neighbors info are missing

        # Try to find a neighbor present in both rounds
        common_neighbors = list(set(last_neighbor_moves.keys()) & set(prev_neighbor_moves.keys()))

        if not common_neighbors:
            # If no common neighbors, pick randomly from last round's neighbors
            if not last_neighbor_moves: return "cooperate" # Should not happen if reached here
            random_neighbor_id = random.choice(list(last_neighbor_moves.keys()))
            last_opponent_move = last_neighbor_moves.get(random_neighbor_id, "cooperate")
            prev_opponent_move = "cooperate" # Default to cooperate if no memory
        else:
            random_neighbor_id = random.choice(common_neighbors)
            last_opponent_move = last_neighbor_moves.get(random_neighbor_id, "cooperate")
            prev_opponent_move = prev_neighbor_moves.get(random_neighbor_id, "cooperate")

        # Defect only if the chosen opponent defected twice in a row
        if last_opponent_move == "defect" and prev_opponent_move == "defect":
            return "defect"
        else:
            return "cooperate"

"""
    # Replace the TitForTwoTatsStrategy class
    updated_content = re.sub(tft2t_class_pattern, replacement, content, flags=re.DOTALL)
    
    return updated_content

def update_pairwise_round_method(content):
    """Update the _run_pairwise_round method in environment.py."""
    pairwise_round_pattern = r'def _run_pairwise_round\(self, rewiring_prob=0\.0\):.*?(?=def [^_]|\Z)'
    replacement = """def _run_pairwise_round(self, rewiring_prob=0.0):
        \"\"\"Run a single round using pairwise interactions with improved TFT support.
        
        This method fixes issues with TFT strategy in pairwise mode by:
        1. Tracking per-opponent moves in memory
        2. Using specific opponent moves for TFT strategies
        3. Maintaining compatibility with all strategies
        
        Args:
            rewiring_prob: Probability to rewire network edges
            
        Returns:
            Tuple of (moves, payoffs) dictionaries
        \"\"\"
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
        
        # Handle network visualization (not functional in pairwise mode)
        if rewiring_prob > 0:
            self.update_network(rewiring_prob=rewiring_prob)
        
        return agent_moves_for_round, round_payoffs_total
        
"""
    # Replace the _run_pairwise_round method
    updated_content = re.sub(pairwise_round_pattern, replacement, content, flags=re.DOTALL)
    
    return updated_content

def run_tests():
    """Run tests to verify the changes."""
    try:
        logger.info("Running pairwise tests...")
        result = subprocess.run(
            [sys.executable, "test_pairwise.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Tests completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e.stderr}")
        return False

def main():
    print("==================================================")
    print("  Applying Permanent Fixes to Pairwise Implementation")
    print("==================================================")
    print("\nThis script will modify source files to fix issues with")
    print("the pairwise interaction model. Backups will be created.")
    
    # Ask for confirmation
    response = input("\nContinue with permanent fixes? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Locate the files to modify
    core_dir = os.path.join("npdl", "core")
    agents_path = os.path.join(core_dir, "agents.py")
    env_path = os.path.join(core_dir, "environment.py")
    
    if not os.path.exists(agents_path) or not os.path.exists(env_path):
        logger.error(f"Required files not found. Make sure you're in the project root directory.")
        return
    
    # Create backups
    backup_file(agents_path)
    backup_file(env_path)
    
    # Read the files
    agents_content = read_file(agents_path)
    env_content = read_file(env_path)
    
    # Apply updates to agents.py
    logger.info("Updating TitForTat strategies...")
    updated_agents = update_tft_class(agents_content)
    updated_agents = update_generous_tft_class(updated_agents)
    updated_agents = update_suspicious_tft_class(updated_agents)
    updated_agents = update_tit_for_two_tats_class(updated_agents)
    
    # Apply updates to environment.py
    logger.info("Updating Environment._run_pairwise_round...")
    updated_env = update_pairwise_round_method(env_content)
    
    # Write the updated files
    write_file(agents_path, updated_agents)
    write_file(env_path, updated_env)
    
    # Run tests to verify the changes
    if run_tests():
        print("\n==================================================")
        print("  ✓ Permanent fixes applied successfully!")
        print("  ✓ All tests passed.")
        print("==================================================")
    else:
        print("\n==================================================")
        print("  ⚠ Fixes applied but tests failed.")
        print("  ⚠ Check the logs for details.")
        print("==================================================")

if __name__ == "__main__":
    main()
