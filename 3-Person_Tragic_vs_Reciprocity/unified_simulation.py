"""
Unified Simulation Engine for Reciprocity Hill vs Tragic Valley Research
========================================================================
This module provides a unified framework to compare pairwise and N-person
dynamics in the Iterated Prisoner's Dilemma.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import defaultdict, deque
import json
import os
from datetime import datetime


class Action(Enum):
    COOPERATE = "C"
    DEFECT = "D"


class InteractionMode(Enum):
    PAIRWISE = "pairwise"
    N_PERSON = "n_person"


class Strategy(Enum):
    ALWAYS_COOPERATE = "Always Cooperate"
    ALWAYS_DEFECT = "Always Defect"
    TIT_FOR_TAT = "Tit-for-Tat"
    MAJORITY_TFT = "Majority-TFT"  # N-person: follows majority
    THRESHOLD_TFT = "Threshold-TFT"  # Probabilistic based on cooperation rate
    VOTE_TFT_1 = "Vote-TFT-1"  # Needs 1 cooperator to cooperate
    VOTE_TFT_2 = "Vote-TFT-2"  # Needs 2 cooperators to cooperate


class Agent:
    """Unified agent that can work in both pairwise and N-person modes."""
    
    def __init__(self, agent_id: int, name: str, strategy: Strategy, 
                 exploration_rate: float = 0.0):
        self.id = agent_id
        self.name = name
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        
        # Memory structures
        self.episode_memory = []  # Current episode memory
        self.all_episodes_memory = []  # All episodes for analysis
        
        # For pairwise mode - track specific opponents
        self.pairwise_history = defaultdict(list)  # {opponent_id: [actions]}
        self.pairwise_opponent_history = defaultdict(list)  # {opponent_id: [opponent_actions]}
        
        # For N-person mode - track group behavior
        self.group_history = []  # List of {agent_id: action} dicts
        self.my_history = []  # My own actions
        
        # Scores
        self.episode_score = 0.0
        self.total_score = 0.0
        self.episode_scores = []  # Score per episode
        
    def reset_episode(self):
        """Reset memory for new episode, but keep cumulative score."""
        self.episode_memory = []
        self.pairwise_history.clear()
        self.pairwise_opponent_history.clear()
        self.group_history = []
        self.my_history = []
        self.episode_scores.append(self.episode_score)
        self.episode_score = 0.0
        
    def decide_action(self, mode: InteractionMode, 
                     opponent_id: Optional[int] = None,
                     group_state: Optional[Dict[int, Action]] = None) -> Action:
        """Decide action based on strategy and mode."""
        
        # Get base action
        if self.strategy == Strategy.ALWAYS_COOPERATE:
            base_action = Action.COOPERATE
        elif self.strategy == Strategy.ALWAYS_DEFECT:
            base_action = Action.DEFECT
        elif self.strategy in [Strategy.TIT_FOR_TAT, Strategy.MAJORITY_TFT]:
            base_action = self._tft_action(mode, opponent_id)
        elif self.strategy == Strategy.THRESHOLD_TFT:
            base_action = self._threshold_tft_action(mode, opponent_id)
        elif self.strategy == Strategy.VOTE_TFT_1:
            base_action = self._vote_tft_action(mode, opponent_id, threshold=1)
        elif self.strategy == Strategy.VOTE_TFT_2:
            base_action = self._vote_tft_action(mode, opponent_id, threshold=2)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Apply exploration
        if random.random() < self.exploration_rate:
            return Action.DEFECT if base_action == Action.COOPERATE else Action.COOPERATE
        return base_action
    
    def choose_actions_for_current_round_pairwise(self, all_opponent_ids: List[int]) -> Dict[int, Action]:
        """Decide actions for all opponents in pairwise mode simultaneously."""
        actions = {}
        
        for opponent_id in all_opponent_ids:
            # For non-adaptive strategies, use the same action for all
            if self.strategy in [Strategy.ALWAYS_COOPERATE, Strategy.ALWAYS_DEFECT]:
                action = self.decide_action(InteractionMode.PAIRWISE, opponent_id)
            else:
                # For adaptive strategies, decide based on specific opponent history
                action = self.decide_action(InteractionMode.PAIRWISE, opponent_id)
            
            actions[opponent_id] = action
        
        return actions
    
    def _tft_action(self, mode: InteractionMode, opponent_id: Optional[int]) -> Action:
        """Standard TFT: copy last move (pairwise) or strict rule (N-person)."""
        if mode == InteractionMode.PAIRWISE:
            # Pairwise: copy opponent's last move
            if opponent_id is None:
                return Action.COOPERATE
            if opponent_id not in self.pairwise_opponent_history:
                return Action.COOPERATE
            if len(self.pairwise_opponent_history[opponent_id]) == 0:
                return Action.COOPERATE
            return self.pairwise_opponent_history[opponent_id][-1]
        else:
            # N-person: STRICT RULE - defect if ANY other agent defected
            if len(self.group_history) == 0:
                return Action.COOPERATE
            
            last_round = self.group_history[-1]
            others_actions = [action for aid, action in last_round.items() if aid != self.id]
            if not others_actions:
                return Action.COOPERATE
                
            # Cooperate only if ALL other agents cooperated
            if all(action == Action.COOPERATE for action in others_actions):
                return Action.COOPERATE
            else:
                return Action.DEFECT
    
    def _threshold_tft_action(self, mode: InteractionMode, opponent_id: Optional[int]) -> Action:
        """Threshold TFT: probabilistic defection based on cooperation rate."""
        cooperation_rate = self._calculate_cooperation_rate(mode, opponent_id)
        
        if cooperation_rate >= 0.5:
            return Action.COOPERATE
        else:
            # Defection probability = 2 * (0.5 - cooperation_rate)
            # At 0% cooperation -> 100% defection
            # At 50% cooperation -> 0% defection
            defect_prob = 2 * (0.5 - cooperation_rate)
            if random.random() < defect_prob:
                return Action.DEFECT
            else:
                return Action.COOPERATE
    
    def _vote_tft_action(self, mode: InteractionMode, opponent_id: Optional[int], 
                        threshold: int) -> Action:
        """Vote-based TFT: need 'threshold' cooperators to cooperate."""
        if mode == InteractionMode.PAIRWISE:
            # In pairwise, threshold doesn't make sense, use standard TFT
            return self._tft_action(mode, opponent_id)
        else:
            # N-person: need 'threshold' cooperators
            if len(self.group_history) == 0:
                return Action.COOPERATE
            
            last_round = self.group_history[-1]
            others_actions = [action for aid, action in last_round.items() if aid != self.id]
            if not others_actions:
                return Action.COOPERATE
                
            coop_count = sum(1 for action in others_actions if action == Action.COOPERATE)
            if coop_count >= threshold:
                return Action.COOPERATE
            else:
                return Action.DEFECT
    
    def _calculate_cooperation_rate(self, mode: InteractionMode, 
                                   opponent_id: Optional[int]) -> float:
        """Calculate cooperation rate based on history."""
        if mode == InteractionMode.PAIRWISE and opponent_id is not None:
            if opponent_id not in self.pairwise_opponent_history:
                return 0.5  # Neutral starting point
            history = self.pairwise_opponent_history[opponent_id]
            if not history:
                return 0.5
            return sum(1 for action in history if action == Action.COOPERATE) / len(history)
        else:
            # N-person: overall cooperation rate
            total_actions = 0
            total_cooperations = 0
            for round_actions in self.group_history:
                for aid, action in round_actions.items():
                    if aid != self.id:
                        total_actions += 1
                        if action == Action.COOPERATE:
                            total_cooperations += 1
            if total_actions == 0:
                return 0.5
            return total_cooperations / total_actions
    
    def update_history(self, mode: InteractionMode, 
                      my_action: Action,
                      opponent_id: Optional[int] = None,
                      opponent_action: Optional[Action] = None,
                      group_actions: Optional[Dict[int, Action]] = None,
                      payoff: float = 0.0):
        """Update history based on interaction mode."""
        self.episode_score += payoff
        self.total_score += payoff
        self.my_history.append(my_action)
        
        if mode == InteractionMode.PAIRWISE:
            if opponent_id is not None and opponent_action is not None:
                self.pairwise_history[opponent_id].append(my_action)
                self.pairwise_opponent_history[opponent_id].append(opponent_action)
        else:
            if group_actions is not None:
                self.group_history.append(group_actions.copy())


class UnifiedSimulation:
    """Simulation that can run both pairwise and N-person games."""
    
    def __init__(self, agents: List[Agent], mode: InteractionMode,
                 R: float = 3.0, S: float = 0.0, T: float = 5.0, P: float = 1.0):
        self.agents = agents
        self.mode = mode
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        
        # For tracking results
        self.episode_results = []
        self.round_data = []
        
    def run_episode(self, rounds_per_episode: int, verbose: bool = True):
        """Run one episode of interactions."""
        episode_data = {
            'mode': self.mode.value,
            'rounds': rounds_per_episode,
            'round_details': []
        }
        
        for round_num in range(rounds_per_episode):
            if verbose and round_num == 0:
                print(f"\n{'='*60}")
                print(f"Episode Round {round_num + 1}")
                print(f"{'='*60}")
            
            round_result = self._run_round(round_num, verbose=(verbose and round_num < 3))
            episode_data['round_details'].append(round_result)
        
        # Store episode results
        episode_summary = {
            'mode': self.mode.value,
            'agent_scores': {agent.name: agent.episode_score for agent in self.agents},
            'cooperation_rates': self._calculate_episode_cooperation_rates()
        }
        self.episode_results.append(episode_summary)
        
        return episode_data
    
    def _run_round(self, round_num: int, verbose: bool = True) -> Dict:
        """Run one round of interaction."""
        if self.mode == InteractionMode.PAIRWISE:
            return self._run_pairwise_round(round_num, verbose)
        else:
            return self._run_nperson_round(round_num, verbose)
    
    def _run_pairwise_round(self, round_num: int, verbose: bool = True) -> Dict:
        """Run pairwise interactions between all agent pairs."""
        round_actions = {}
        round_payoffs = defaultdict(float)
        interactions = []
        
        # Generate all pairs
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1, agent2 = self.agents[i], self.agents[j]
                
                # Agents decide
                action1 = agent1.decide_action(self.mode, opponent_id=agent2.id)
                action2 = agent2.decide_action(self.mode, opponent_id=agent1.id)
                
                # Calculate payoffs
                if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
                    payoff1, payoff2 = self.R, self.R
                elif action1 == Action.COOPERATE and action2 == Action.DEFECT:
                    payoff1, payoff2 = self.S, self.T
                elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
                    payoff1, payoff2 = self.T, self.S
                else:
                    payoff1, payoff2 = self.P, self.P
                
                # Update histories
                agent1.update_history(self.mode, action1, agent2.id, action2, payoff=payoff1)
                agent2.update_history(self.mode, action2, agent1.id, action1, payoff=payoff2)
                
                # Track for this round
                round_payoffs[agent1.id] += payoff1
                round_payoffs[agent2.id] += payoff2
                
                interactions.append({
                    'agent1': agent1.name,
                    'agent2': agent2.name,
                    'action1': action1.value,
                    'action2': action2.value,
                    'payoff1': payoff1,
                    'payoff2': payoff2
                })
                
                if verbose:
                    print(f"\n{agent1.name} vs {agent2.name}: {action1.value} vs {action2.value}")
                    print(f"Payoffs: {agent1.name}={payoff1}, {agent2.name}={payoff2}")
        
        # Store round data
        round_result = {
            'round': round_num,
            'mode': 'pairwise',
            'interactions': interactions,
            'total_payoffs': {agent.name: round_payoffs[agent.id] for agent in self.agents}
        }
        
        return round_result
    
    def _run_nperson_round(self, round_num: int, verbose: bool = True) -> Dict:
        """Run N-person interaction where all agents play simultaneously."""
        # All agents decide simultaneously
        actions = {}
        for agent in self.agents:
            action = agent.decide_action(self.mode)
            actions[agent.id] = action
        
        # Calculate payoffs using linear functions
        payoffs = {}
        num_cooperators = sum(1 for action in actions.values() if action == Action.COOPERATE)
        
        for agent in self.agents:
            agent_action = actions[agent.id]
            # n = cooperating neighbors (excluding self)
            n = num_cooperators - (1 if agent_action == Action.COOPERATE else 0)
            N = len(self.agents)
            
            if agent_action == Action.COOPERATE:
                # Linear cooperation payoff
                payoff = self.S + (self.R - self.S) * (n / (N - 1)) if N > 1 else self.R
            else:
                # Linear defection payoff
                payoff = self.P + (self.T - self.P) * (n / (N - 1)) if N > 1 else self.P
            
            payoffs[agent.id] = payoff
            
            # Update agent history
            agent.update_history(self.mode, agent_action, group_actions=actions, payoff=payoff)
        
        if verbose:
            print(f"\nN-Person Round {round_num + 1}:")
            print(f"Actions: {', '.join(f'{a.name}={actions[a.id].value}' for a in self.agents)}")
            print(f"Cooperators: {num_cooperators}/{len(self.agents)}")
            print(f"Payoffs: {', '.join(f'{a.name}={payoffs[a.id]:.2f}' for a in self.agents)}")
        
        # Store round data
        round_result = {
            'round': round_num,
            'mode': 'n_person',
            'actions': {agent.name: actions[agent.id].value for agent in self.agents},
            'payoffs': {agent.name: payoffs[agent.id] for agent in self.agents},
            'num_cooperators': num_cooperators
        }
        
        return round_result
    
    def _calculate_episode_cooperation_rates(self) -> Dict[str, float]:
        """Calculate cooperation rates for the current episode."""
        rates = {}
        for agent in self.agents:
            if agent.my_history:
                coop_count = sum(1 for action in agent.my_history if action == Action.COOPERATE)
                rates[agent.name] = coop_count / len(agent.my_history)
            else:
                rates[agent.name] = 0.0
        return rates
    
    def reset_episode(self):
        """Reset all agents for a new episode."""
        for agent in self.agents:
            agent.reset_episode()
    
    def run_simulation(self, episodes: int, rounds_per_episode: int, 
                      verbose: bool = True) -> Dict:
        """Run complete simulation with multiple episodes."""
        simulation_results = {
            'mode': self.mode.value,
            'episodes': episodes,
            'rounds_per_episode': rounds_per_episode,
            'episode_data': [],
            'final_scores': {},
            'cooperation_evolution': []
        }
        
        print(f"\nRunning {self.mode.value} simulation:")
        print(f"Episodes: {episodes}, Rounds per episode: {rounds_per_episode}")
        print(f"Agents: {', '.join(f'{a.name} ({a.strategy.value})' for a in self.agents)}")
        
        for episode in range(episodes):
            if verbose:
                print(f"\n{'='*80}")
                print(f"EPISODE {episode + 1}")
                print(f"{'='*80}")
            
            episode_data = self.run_episode(rounds_per_episode, verbose=(episode < 2))
            simulation_results['episode_data'].append(self.episode_results[-1])
            
            # Track cooperation evolution
            coop_rates = self._calculate_episode_cooperation_rates()
            simulation_results['cooperation_evolution'].append(coop_rates)
            
            # Reset for next episode
            self.reset_episode()
        
        # Final scores
        for agent in self.agents:
            simulation_results['final_scores'][agent.name] = agent.total_score
        
        return simulation_results


def create_agents(config: Dict) -> List[Agent]:
    """Create agents based on configuration."""
    agents = []
    agent_id = 0
    
    for strategy_name, count in config.items():
        strategy = Strategy[strategy_name]
        for i in range(count):
            name = f"{strategy.value}-{i+1}" if count > 1 else strategy.value
            exploration = config.get(f'{strategy_name}_exploration', 0.0)
            agent = Agent(agent_id, name, strategy, exploration)
            agents.append(agent)
            agent_id += 1
    
    return agents


def save_results(results: Dict, filename: str):
    """Save simulation results to JSON file."""
    # Ensure directory exists
    if '/' in filename:
        # If filename contains a path, use it directly
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        filepath = filename
    else:
        # Otherwise, save to results directory
        os.makedirs('results', exist_ok=True)
        filepath = f'results/{filename}'
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
