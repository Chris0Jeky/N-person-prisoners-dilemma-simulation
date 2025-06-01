"""
Simple N-Person Iterated Prisoner's Dilemma Simulation
======================================================
A simple simulation of 3 agents playing N-Person iterated prisoner's dilemma.
All agents interact simultaneously in each round.
Features simple strategies and an exploration parameter.
"""

import random
from typing import Dict, List, Tuple
from enum import Enum
import sys
import os

# Add parent directory to path to import from npdl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npdl.core.utils import linear_payoff_C, linear_payoff_D


class Action(Enum):
    COOPERATE = "C"
    DEFECT = "D"


class Strategy(Enum):
    ALWAYS_COOPERATE = "Always Cooperate"
    ALWAYS_DEFECT = "Always Defect"
    TIT_FOR_TAT = "Tit-for-Tat"


class NPDAgent:
    """Simple agent for N-Person PD with a strategy and exploration parameter."""
    
    def __init__(self, name: str, strategy: Strategy, exploration_rate: float = 0.0):
        self.name = name
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        self.action_history: List[Action] = []  # My own action history
        self.group_history: List[Dict[str, Action]] = []  # History of all agents' actions
        self.payoff_history: List[float] = []
        self.total_score = 0.0
        
    def decide_action(self) -> Action:
        """Decide action based on strategy and exploration."""
        # Determine base action according to strategy
        if self.strategy == Strategy.ALWAYS_COOPERATE:
            base_action = Action.COOPERATE
        elif self.strategy == Strategy.ALWAYS_DEFECT:
            base_action = Action.DEFECT
        elif self.strategy == Strategy.TIT_FOR_TAT:
            base_action = self._tit_for_tat_action()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Apply exploration (do opposite with exploration_rate probability)
        if random.random() < self.exploration_rate:
            # Do opposite action
            explored_action = Action.DEFECT if base_action == Action.COOPERATE else Action.COOPERATE
            print(f"      → {self.name} EXPLORES! (rate={self.exploration_rate:.1%})")
            print(f"        Base action: {base_action.value}, Explored to: {explored_action.value}")
            return explored_action
        else:
            return base_action
    
    def _tit_for_tat_action(self) -> Action:
        """Tit-for-tat: cooperate first, then follow majority from last round."""
        if len(self.group_history) == 0:
            # First round - cooperate
            return Action.COOPERATE
        else:
            # Look at last round's actions (excluding self)
            last_round = self.group_history[-1]
            others_actions = [action for name, action in last_round.items() if name != self.name]
            
            # Count cooperations
            coop_count = sum(1 for action in others_actions if action == Action.COOPERATE)
            
            # If majority cooperated, cooperate; otherwise defect
            if coop_count > len(others_actions) / 2:
                return Action.COOPERATE
            else:
                return Action.DEFECT
    
    def update_history(self, my_action: Action, all_actions: Dict[str, Action], payoff: float):
        """Update history with actions and payoff."""
        self.action_history.append(my_action)
        self.group_history.append(all_actions.copy())
        self.payoff_history.append(payoff)
        self.total_score += payoff


class NPrisonersDilemmaGame:
    """N-Person IPD game using linear payoff functions."""
    
    def __init__(self, N: int = 3, R: float = 3.0, S: float = 0.0, T: float = 5.0, P: float = 1.0):
        """
        Initialize N-Person game with payoff parameters.
        Uses linear payoff functions from npdl.core.utils.
        
        Args:
            N: Number of agents
            R: Reward for mutual cooperation
            S: Sucker's payoff
            T: Temptation payoff
            P: Punishment for mutual defection
        """
        self.N = N
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        
    def calculate_payoffs(self, actions: Dict[str, Action]) -> Dict[str, float]:
        """Calculate payoffs for all agents given their actions."""
        # Count cooperators (excluding self for each agent)
        num_cooperators = sum(1 for action in actions.values() if action == Action.COOPERATE)
        
        payoffs = {}
        for agent_name, agent_action in actions.items():
            # Number of cooperating neighbors (excluding self)
            n = num_cooperators - (1 if agent_action == Action.COOPERATE else 0)
            
            # Calculate payoff using linear functions
            if agent_action == Action.COOPERATE:
                payoffs[agent_name] = linear_payoff_C(n, self.N, self.R, self.S)
            else:  # DEFECT
                payoffs[agent_name] = linear_payoff_D(n, self.N, self.T, self.P)
                
        return payoffs


class NPersonSimulation:
    """Run the N-Person IPD simulation with verbose output."""
    
    def __init__(self, agents: List[NPDAgent], game: NPrisonersDilemmaGame):
        self.agents = agents
        self.game = game
        self.round_number = 0
        
    def run_round(self):
        """Run one round where all agents interact simultaneously."""
        self.round_number += 1
        print(f"\n{'='*60}")
        print(f"ROUND {self.round_number}")
        print(f"{'='*60}")
        
        # Phase 1: Decision Making
        print(f"\n--- DECISION PHASE ---")
        actions = {}
        base_actions = {}  # Store base actions for comparison
        
        for agent in self.agents:
            print(f"\n  {agent.name} ({agent.strategy.value}):")
            
            # Get base action without exploration
            old_exploration = agent.exploration_rate
            agent.exploration_rate = 0.0
            base_action = agent.decide_action()
            agent.exploration_rate = old_exploration
            base_actions[agent.name] = base_action
            
            # Get actual action (with exploration)
            actual_action = agent.decide_action()
            actions[agent.name] = actual_action
            
            print(f"    Strategy suggests: {base_action.value}")
            if actual_action != base_action:
                print(f"    EXPLORED to: {actual_action.value}")
            else:
                print(f"    Final decision: {actual_action.value}")
        
        # Phase 2: Calculate Payoffs
        print(f"\n--- PAYOFF CALCULATION ---")
        payoffs = self.game.calculate_payoffs(actions)
        
        # Count cooperators for context
        num_cooperators = sum(1 for action in actions.values() if action == Action.COOPERATE)
        print(f"\n  Total cooperators: {num_cooperators}/{self.game.N}")
        print(f"  Total defectors: {self.game.N - num_cooperators}/{self.game.N}")
        
        # Show individual payoffs and explanation
        print(f"\n  Individual payoffs:")
        for agent in self.agents:
            agent_action = actions[agent.name]
            agent_payoff = payoffs[agent.name]
            
            # Calculate n (cooperating neighbors)
            n = num_cooperators - (1 if agent_action == Action.COOPERATE else 0)
            
            print(f"\n    {agent.name}: {agent_action.value} → {agent_payoff:.2f} points")
            if agent_action == Action.COOPERATE:
                print(f"      (Cooperated with {n} cooperating neighbors)")
                print(f"      Payoff = S + (R-S) × (n/(N-1)) = {self.game.S} + {self.game.R - self.game.S} × ({n}/{self.game.N-1}) = {agent_payoff:.2f}")
            else:
                print(f"      (Defected against {n} cooperating neighbors)")
                print(f"      Payoff = P + (T-P) × (n/(N-1)) = {self.game.P} + {self.game.T - self.game.P} × ({n}/{self.game.N-1}) = {agent_payoff:.2f}")
        
        # Phase 3: Update History
        for agent in self.agents:
            agent.update_history(actions[agent.name], actions, payoffs[agent.name])
        
        # Phase 4: Interpretation
        print(f"\n--- ROUND INTERPRETATION ---")
        if num_cooperators == self.game.N:
            print("  → Full cooperation achieved! Everyone benefits from mutual cooperation.")
        elif num_cooperators == 0:
            print("  → Complete defection! Everyone receives the punishment payoff.")
        else:
            defectors = [name for name, action in actions.items() if action == Action.DEFECT]
            cooperators = [name for name, action in actions.items() if action == Action.COOPERATE]
            print(f"  → Mixed outcome: {', '.join(defectors)} exploited {', '.join(cooperators)}")
        
        # Show current scores
        self._show_scores()
    
    def _show_scores(self):
        """Show current total scores."""
        print(f"\n--- CUMULATIVE SCORES (after round {self.round_number}) ---")
        sorted_agents = sorted(self.agents, key=lambda x: x.total_score, reverse=True)
        for i, agent in enumerate(sorted_agents, 1):
            avg_score = agent.total_score / self.round_number if self.round_number > 0 else 0
            print(f"  {i}. {agent.name} ({agent.strategy.value}): {agent.total_score:.2f} total (avg: {avg_score:.2f}/round)")
    
    def run_simulation(self, num_rounds: int):
        """Run the complete simulation."""
        print(f"\n{'='*80}")
        print(f"N-PERSON ITERATED PRISONER'S DILEMMA SIMULATION")
        print(f"{'='*80}")
        
        print(f"\nGame Configuration:")
        print(f"  Number of agents: {len(self.agents)}")
        print(f"  Number of rounds: {num_rounds}")
        print(f"  Interaction type: N-Person (all agents interact simultaneously)")
        
        print(f"\nAgents:")
        for agent in self.agents:
            print(f"  - {agent.name}: {agent.strategy.value} (exploration={agent.exploration_rate:.1%})")
        
        print(f"\nPayoff Parameters:")
        print(f"  Reward (R) - mutual cooperation: {self.game.R}")
        print(f"  Temptation (T) - defect vs cooperators: {self.game.T}")
        print(f"  Punishment (P) - mutual defection: {self.game.P}")
        print(f"  Sucker (S) - cooperate vs defectors: {self.game.S}")
        print(f"\nPayoff Functions:")
        print(f"  Cooperation: Payoff_C(n) = S + (R-S) × (n/(N-1))")
        print(f"  Defection: Payoff_D(n) = P + (T-P) × (n/(N-1))")
        print(f"  where n = number of cooperating neighbors")
        
        # Run all rounds
        for _ in range(num_rounds):
            self.run_round()
        
        # Final results
        self._show_final_results()
    
    def _show_final_results(self):
        """Show comprehensive final results."""
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        
        # Sort agents by final score
        sorted_agents = sorted(self.agents, key=lambda x: x.total_score, reverse=True)
        
        print("\nFinal Rankings:")
        for i, agent in enumerate(sorted_agents, 1):
            avg_score = agent.total_score / self.round_number
            print(f"  {i}. {agent.name} ({agent.strategy.value}):")
            print(f"     Total score: {agent.total_score:.2f}")
            print(f"     Average per round: {avg_score:.2f}")
            print(f"     Exploration rate: {agent.exploration_rate:.1%}")
        
        print("\nCooperation Statistics:")
        for agent in self.agents:
            total_actions = len(agent.action_history)
            if total_actions > 0:
                coop_count = sum(1 for action in agent.action_history if action == Action.COOPERATE)
                coop_rate = coop_count / total_actions
                print(f"  {agent.name}: {coop_rate:.1%} cooperation rate ({coop_count}/{total_actions} actions)")
                
                # Show cooperation trend
                if total_actions >= 10:
                    first_half = agent.action_history[:total_actions//2]
                    second_half = agent.action_history[total_actions//2:]
                    first_coop = sum(1 for a in first_half if a == Action.COOPERATE) / len(first_half)
                    second_coop = sum(1 for a in second_half if a == Action.COOPERATE) / len(second_half)
                    trend = "↑" if second_coop > first_coop else ("↓" if second_coop < first_coop else "→")
                    print(f"     Trend: {first_coop:.0%} → {second_coop:.0%} {trend}")
        
        print("\nStrategy Performance Summary:")
        strategy_scores = {}
        strategy_counts = {}
        for agent in self.agents:
            strat = agent.strategy.value
            if strat not in strategy_scores:
                strategy_scores[strat] = 0
                strategy_counts[strat] = 0
            strategy_scores[strat] += agent.total_score
            strategy_counts[strat] += 1
        
        for strat, total in strategy_scores.items():
            avg = total / strategy_counts[strat]
            print(f"  {strat}: {avg:.2f} average score")


# Compatibility wrapper for CSV export
class Agent:
    """Wrapper class for compatibility with csv_exporter.py"""
    def __init__(self, npd_agent: NPDAgent):
        self.name = npd_agent.name
        self.strategy = npd_agent.strategy
        self.score = npd_agent.total_score
        # For CSV export compatibility
        self.history = {}
        self.opponent_history = {}


class Simulation:
    """Wrapper class for compatibility with csv_exporter.py"""
    def __init__(self, agents: List[Agent], game):
        self.agents = agents
        self.game = game
