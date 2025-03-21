# game.py
"""
Interactive gameplay for N-Person Prisoner's Dilemma.

This module allows human players to participate in simulations
against AI agents with a text-based interface.
"""

import random
import time
import os

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


class InteractiveGame:
    """Class for managing interactive gameplay."""
    
    def __init__(self, num_agents=5, num_rounds=10, network_type="fully_connected", 
                 opponents=None, payoff_type="linear", payoff_params=None):
        """Initialize the interactive game.
        
        Args:
            num_agents: Total number of agents including the human player
            num_rounds: Number of rounds to play
            network_type: Type of network structure
            opponents: List of opponent strategy types
            payoff_type: Type of payoff function
            payoff_params: Parameters for the payoff function
        """
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.network_type = network_type
        self.opponents = opponents or ["tit_for_tat", "q_learning", "always_defect"]
        self.payoff_type = payoff_type
        self.payoff_params = payoff_params or {}
        
        # Create agents
        self.agents = []
        self.human_id = 0  # Human player is always agent 0
        
        self._setup_game()
    
    def _setup_game(self):
        """Set up the game environment and agents."""
        # Create human player agent
        self.agents.append(Agent(agent_id=self.human_id, strategy="always_cooperate"))
        
        # Create AI opponents
        for i in range(1, self.num_agents):
            strategy = random.choice(self.opponents)
            agent_params = {
                "agent_id": i,
                "strategy": strategy,
            }
            
            # Add strategy-specific parameters
            if strategy == "q_learning":
                agent_params.update({
                    "learning_rate": 0.1,
                    "discount_factor": 0.9,
                    "epsilon": 0.2,
                })
            elif strategy == "generous_tit_for_tat":
                agent_params["generosity"] = 0.1
            
            self.agents.append(Agent(**agent_params))
        
        # Create payoff matrix
        self.payoff_matrix = create_payoff_matrix(
            self.num_agents, 
            payoff_type=self.payoff_type,
            params=self.payoff_params
        )
        
        # Create environment
        self.env = Environment(
            self.agents,
            self.payoff_matrix,
            network_type=self.network_type
        )
        
        # Track game history
        self.history = []
        
        # Get human player's neighbors
        self.human_neighbors = self.env.get_neighbors(self.human_id)
    
    def _get_human_move(self):
        """Get move from human player."""
        while True:
            choice = input("Your move (c for cooperate, d for defect): ").strip().lower()
            if choice == 'c':
                return "cooperate"
            elif choice == 'd':
                return "defect"
            else:
                print("Invalid choice. Please enter 'c' or 'd'.")
    
    def _display_round_results(self, round_num, moves, payoffs):
        """Display the results of a round."""
        print(f"\n--- Round {round_num + 1} Results ---")
        
        # Show player's move and payoff
        player_move = moves[self.human_id]
        player_payoff = payoffs[self.human_id]
        print(f"Your move: {player_move.upper()}")
        print(f"Your payoff: {player_payoff:.2f}")
        
        # Show neighbors' moves
        print("\nNeighbor moves:")
        for neighbor_id in self.human_neighbors:
            neighbor = next(a for a in self.agents if a.agent_id == neighbor_id)
            neighbor_move = moves[neighbor_id]
            neighbor_payoff = payoffs[neighbor_id]
            print(f"Agent {neighbor_id} ({neighbor.strategy_type}): {neighbor_move.upper()} (payoff: {neighbor_payoff:.2f})")
        
        # Show score summary
        print("\nCurrent scores:")
        print(f"Your score: {self.agents[self.human_id].score:.2f}")
        
        # Show cooperation rate
        coop_count = sum(1 for move in moves.values() if move == "cooperate")
        coop_rate = coop_count / len(moves)
        print(f"Cooperation rate: {coop_rate:.2f}")
        
        print("\nPress Enter to continue...")
        input()
    
    def _display_game_summary(self):
        """Display summary of the game."""
        clear_screen()
        print("\n=== Game Summary ===\n")
        
        # Calculate final scores
        print("Final Scores:")
        
        # Sort agents by score
        sorted_agents = sorted(self.agents, key=lambda a: a.score, reverse=True)
        
        for i, agent in enumerate(sorted_agents):
            if agent.agent_id == self.human_id:
                print(f"{i+1}. YOU: {agent.score:.2f}")
            else:
                print(f"{i+1}. Agent {agent.agent_id} ({agent.strategy_type}): {agent.score:.2f}")
        
        # Calculate cooperation rates by agent
        print("\nCooperation Rates:")
        
        for agent in self.agents:
            agent_moves = [round_result['moves'][agent.agent_id] for round_result in self.history]
            coop_rate = sum(1 for move in agent_moves if move == "cooperate") / len(agent_moves)
            
            if agent.agent_id == self.human_id:
                print(f"YOU: {coop_rate:.2f}")
            else:
                print(f"Agent {agent.agent_id} ({agent.strategy_type}): {coop_rate:.2f}")
        
        # Show trend over time
        if len(self.history) > 2:
            coop_rates = []
            for round_result in self.history:
                moves = round_result['moves']
                coop_count = sum(1 for move in moves.values() if move == "cooperate")
                coop_rates.append(coop_count / len(moves))
            
            print("\nCooperation Rate Trend:")
            for i, rate in enumerate(coop_rates):
                print(f"Round {i+1}: {'#' * int(rate * 20)} {rate:.2f}")
    
    def run_game(self):
        """Run the interactive game."""
        clear_screen()
        print("Welcome to the Interactive N-Person Prisoner's Dilemma Simulation!")
        print(f"You'll be playing against {self.num_agents - 1} AI agents over {self.num_rounds} rounds.")
        print(f"Network type: {self.network_type}")
        print(f"Your neighbors: {self.human_neighbors}")
        print("\nPress Enter to start the game...")
        input()
        
        for round_num in range(self.num_rounds):
            clear_screen()
            print(f"Round {round_num + 1} of {self.num_rounds}")
            print(f"Your score so far: {self.agents[self.human_id].score:.2f}")
            
            # Show neighbors' last moves if not the first round
            if round_num > 0:
                print("\nNeighbors' previous moves:")
                for neighbor_id in self.human_neighbors:
                    neighbor = next(a for a in self.agents if a.agent_id == neighbor_id)
                    prev_move = self.history[-1]['moves'][neighbor_id]
                    print(f"Agent {neighbor_id} ({neighbor.strategy_type}): {prev_move.upper()}")
            
            # Get human player's move
            human_move = self._get_human_move()
            
            # Get AI moves and override human player agent's default choice
            all_moves = {agent.agent_id: agent.choose_move(self.env.get_neighbors(agent.agent_id))
                        for agent in self.agents if agent.agent_id != self.human_id}
            all_moves[self.human_id] = human_move
            
            # Calculate payoffs
            payoffs = self.env.calculate_payoffs(all_moves)
            
            # Update agent states
            for agent in self.agents:
                # Update score
                agent.score += payoffs[agent.agent_id]
                
                # Get neighbor moves
                neighbors = self.env.get_neighbors(agent.agent_id)
                neighbor_moves = {n_id: all_moves[n_id] for n_id in neighbors if n_id in all_moves}
                
                # Update Q-values and memory (only for AI agents)
                if agent.agent_id != self.human_id:
                    agent.update_q_value(all_moves[agent.agent_id], payoffs[agent.agent_id], neighbor_moves)
                agent.update_memory(all_moves[agent.agent_id], neighbor_moves, payoffs[agent.agent_id])
            
            # Store round results
            self.history.append({'round': round_num, 'moves': all_moves, 'payoffs': payoffs})
            
            # Display round results
            self._display_round_results(round_num, all_moves, payoffs)
        
        # Display game summary
        self._display_game_summary()
        print("\nThanks for playing!")


def main():
    """Main function for the interactive game."""
    clear_screen()
    print("=== N-Person Prisoner's Dilemma Interactive Mode ===\n")
    
    # Get game parameters
    try:
        num_agents = int(input("Number of agents (including you) [5]: ") or "5")
        num_rounds = int(input("Number of rounds [10]: ") or "10")
        
        # Choose network type
        print("\nNetwork types:")
        print("1. Fully connected (everyone connects to everyone)")
        print("2. Small world (more realistic social network)")
        print("3. Scale-free (hub-based network)")
        print("4. Random network")
        
        network_choice = input("Choose network type [1]: ") or "1"
        if network_choice == "1":
            network_type = "fully_connected"
            network_params = {}
        elif network_choice == "2":
            network_type = "small_world"
            network_params = {"k": 4, "beta": 0.2}
        elif network_choice == "3":
            network_type = "scale_free"
            network_params = {"m": 2}
        elif network_choice == "4":
            network_type = "random"
            network_params = {"probability": 0.3}
        else:
            print("Invalid choice. Using fully connected network.")
            network_type = "fully_connected"
            network_params = {}
        
        # Choose opponent types
        print("\nOpponent types:")
        print("1. Mixed (various strategies)")
        print("2. All Tit-for-Tat")
        print("3. All Q-learning (adaptive)")
        print("4. All Always Defect")
        
        opponent_choice = input("Choose opponent types [1]: ") or "1"
        if opponent_choice == "1":
            opponents = ["tit_for_tat", "q_learning", "always_defect", "random", "pavlov"]
        elif opponent_choice == "2":
            opponents = ["tit_for_tat"]
        elif opponent_choice == "3":
            opponents = ["q_learning"]
        elif opponent_choice == "4":
            opponents = ["always_defect"]
        else:
            print("Invalid choice. Using mixed opponents.")
            opponents = ["tit_for_tat", "q_learning", "always_defect", "random", "pavlov"]
        
        # Choose payoff type
        print("\nPayoff types:")
        print("1. Linear (standard)")
        print("2. Exponential (synergistic)")
        print("3. Threshold (critical mass)")
        
        payoff_choice = input("Choose payoff type [1]: ") or "1"
        if payoff_choice == "1":
            payoff_type = "linear"
            payoff_params = {}
        elif payoff_choice == "2":
            payoff_type = "exponential"
            payoff_params = {"exponent": 2}
        elif payoff_choice == "3":
            payoff_type = "threshold"
            payoff_params = {"threshold": 0.6}
        else:
            print("Invalid choice. Using linear payoffs.")
            payoff_type = "linear"
            payoff_params = {}
        
        # Create and run game
        game = InteractiveGame(
            num_agents=num_agents,
            num_rounds=num_rounds,
            network_type=network_type,
            opponents=opponents,
            payoff_type=payoff_type,
            payoff_params=payoff_params
        )
        game.run_game()
        
    except KeyboardInterrupt:
        print("\nGame aborted. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Game aborted.")


if __name__ == "__main__":
    main()
