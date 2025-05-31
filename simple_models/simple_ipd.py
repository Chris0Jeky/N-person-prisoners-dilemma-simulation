"""
Simple Iterated Prisoner's Dilemma Simulation
=============================================
A simple simulation of 3 agents playing iterated prisoner's dilemma in pairs.
Features simple strategies and an exploration parameter.
"""

import random
from typing import Dict, List, Tuple
from enum import Enum


class Action(Enum):
    COOPERATE = "C"
    DEFECT = "D"


class Strategy(Enum):
    ALWAYS_COOPERATE = "Always Cooperate"
    ALWAYS_DEFECT = "Always Defect"
    TIT_FOR_TAT = "Tit-for-Tat"


class Agent:
    """Simple agent with a strategy and exploration parameter."""
    
    def __init__(self, name: str, strategy: Strategy, exploration_rate: float = 0.0):
        self.name = name
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        self.history: Dict[str, List[Action]] = {}  # History of actions against each opponent
        self.opponent_history: Dict[str, List[Action]] = {}  # History of opponent's actions
        
    def decide_action(self, opponent_name: str) -> Action:
        """Decide action based on strategy and exploration."""
        # Determine base action according to strategy
        if self.strategy == Strategy.ALWAYS_COOPERATE:
            base_action = Action.COOPERATE
        elif self.strategy == Strategy.ALWAYS_DEFECT:
            base_action = Action.DEFECT
        elif self.strategy == Strategy.TIT_FOR_TAT:
            base_action = self._tit_for_tat_action(opponent_name)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Apply exploration (do opposite with exploration_rate probability)
        if random.random() < self.exploration_rate:
            # Do opposite action
            explored_action = Action.DEFECT if base_action == Action.COOPERATE else Action.COOPERATE
            return explored_action
        else:
            return base_action
    
    def _tit_for_tat_action(self, opponent_name: str) -> Action:
        """Tit-for-tat: cooperate first, then copy opponent's last move."""
        if opponent_name not in self.opponent_history or len(self.opponent_history[opponent_name]) == 0:
            # First interaction - cooperate
            return Action.COOPERATE
        else:
            # Copy opponent's last action
            return self.opponent_history[opponent_name][-1]
    
    def update_history(self, opponent_name: str, my_action: Action, opponent_action: Action):
        """Update history of actions."""
        if opponent_name not in self.history:
            self.history[opponent_name] = []
            self.opponent_history[opponent_name] = []
        
        self.history[opponent_name].append(my_action)
        self.opponent_history[opponent_name].append(opponent_action)


class PrisonersDilemmaGame:
    """Simple IPD game with customizable payoffs."""
    
    def __init__(self, reward: int = 3, temptation: int = 5, punishment: int = 1, sucker: int = 0):
        """
        Initialize game with payoff matrix.
        Default values: T=5, R=3, P=1, S=0
        Where T > R > P > S and 2R > T + S
        """
        self.reward = reward  # Both cooperate
        self.temptation = temptation  # I defect, opponent cooperates
        self.punishment = punishment  # Both defect
        self.sucker = sucker  # I cooperate, opponent defects
        
    def get_payoffs(self, action1: Action, action2: Action) -> Tuple[int, int]:
        """Get payoffs for both players given their actions."""
        if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
            return self.reward, self.reward
        elif action1 == Action.COOPERATE and action2 == Action.DEFECT:
            return self.sucker, self.temptation
        elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
            return self.temptation, self.sucker
        else:  # Both defect
            return self.punishment, self.punishment


class Simulation:
    """Run the IPD simulation with verbose output."""
    
    def __init__(self, agents: List[Agent], game: PrisonersDilemmaGame):
        self.agents = agents
        self.game = game
        self.scores = {agent.name: 0 for agent in agents}
        self.round_number = 0
        
    def run_round(self):
        """Run one round of all pairwise interactions."""
        self.round_number += 1
        print(f"\n{'='*60}")
        print(f"ROUND {self.round_number}")
        print(f"{'='*60}")
        
        # Generate all pairs
        pairs = []
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                pairs.append((self.agents[i], self.agents[j]))
        
        # Play each pair
        for agent1, agent2 in pairs:
            self._play_pair(agent1, agent2)
        
        # Show current scores
        self._show_scores()
    
    def _play_pair(self, agent1: Agent, agent2: Agent):
        """Play one interaction between two agents."""
        print(f"\n--- {agent1.name} vs {agent2.name} ---")
        
        # Get base decisions
        base_action1 = self._get_base_action(agent1, agent2.name)
        base_action2 = self._get_base_action(agent2, agent1.name)
        
        # Agents decide actions (with possible exploration)
        action1 = agent1.decide_action(agent2.name)
        action2 = agent2.decide_action(agent1.name)
        
        # Check if exploration happened
        explored1 = action1 != base_action1
        explored2 = action2 != base_action2
        
        # Get payoffs
        payoff1, payoff2 = self.game.get_payoffs(action1, action2)
        
        # Update scores
        self.scores[agent1.name] += payoff1
        self.scores[agent2.name] += payoff2
        
        # Update histories
        agent1.update_history(agent2.name, action1, action2)
        agent2.update_history(agent1.name, action2, action1)
        
        # Verbose output
        print(f"  {agent1.name} ({agent1.strategy.value}):")
        print(f"    Base action: {base_action1.value}, Actual: {action1.value}", end="")
        if explored1:
            print(f" [EXPLORED! (rate={agent1.exploration_rate:.1%})]")
        else:
            print()
        
        print(f"  {agent2.name} ({agent2.strategy.value}):")
        print(f"    Base action: {base_action2.value}, Actual: {action2.value}", end="")
        if explored2:
            print(f" [EXPLORED! (rate={agent2.exploration_rate:.1%})]")
        else:
            print()
        
        print(f"  Result: {agent1.name} {action1.value} vs {agent2.name} {action2.value}")
        print(f"  Payoffs: {agent1.name} gets {payoff1}, {agent2.name} gets {payoff2}")
        
        # Show game outcome interpretation
        if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
            print("  → Both cooperated! Win-win situation.")
        elif action1 == Action.DEFECT and action2 == Action.DEFECT:
            print("  → Both defected! Mutual punishment.")
        elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
            print(f"  → {agent1.name} exploited {agent2.name}!")
        else:
            print(f"  → {agent2.name} exploited {agent1.name}!")
    
    def _get_base_action(self, agent: Agent, opponent_name: str) -> Action:
        """Get the base action without exploration."""
        old_exploration = agent.exploration_rate
        agent.exploration_rate = 0.0
        base_action = agent.decide_action(opponent_name)
        agent.exploration_rate = old_exploration
        return base_action
    
    def _show_scores(self):
        """Show current scores."""
        print(f"\nScores after round {self.round_number}:")
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_scores:
            agent = next(a for a in self.agents if a.name == name)
            print(f"  {name} ({agent.strategy.value}): {score} points")
    
    def run_simulation(self, num_rounds: int):
        """Run the complete simulation."""
        print(f"\nStarting Iterated Prisoner's Dilemma Simulation")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Number of rounds: {num_rounds}")
        print(f"\nAgents:")
        for agent in self.agents:
            print(f"  - {agent.name}: {agent.strategy.value} (exploration={agent.exploration_rate:.1%})")
        
        print(f"\nPayoff Matrix:")
        print(f"  Both Cooperate (R): {self.game.reward}")
        print(f"  Temptation (T): {self.game.temptation}")
        print(f"  Punishment (P): {self.game.punishment}")
        print(f"  Sucker (S): {self.game.sucker}")
        
        for _ in range(num_rounds):
            self.run_round()
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        self._show_final_results()
    
    def _show_final_results(self):
        """Show final results with detailed statistics."""
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFinal Scores:")
        for i, (name, score) in enumerate(sorted_scores, 1):
            agent = next(a for a in self.agents if a.name == name)
            avg_score = score / (self.round_number * (len(self.agents) - 1))
            print(f"  {i}. {name} ({agent.strategy.value}): {score} points (avg: {avg_score:.2f} per interaction)")
        
        print("\nCooperation Statistics:")
        for agent in self.agents:
            total_actions = sum(len(actions) for actions in agent.history.values())
            if total_actions > 0:
                coop_count = sum(1 for actions in agent.history.values() 
                               for action in actions if action == Action.COOPERATE)
                coop_rate = coop_count / total_actions
                print(f"  {agent.name}: {coop_rate:.1%} cooperation rate ({coop_count}/{total_actions} actions)")
