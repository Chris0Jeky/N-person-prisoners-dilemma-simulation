"""
Game environment implementations for Prisoner's Dilemma simulations.

This module contains:
- Pairwise (2-player) Prisoner's Dilemma
- N-Person Prisoner's Dilemma with linear payoffs
"""

import random
from typing import List, Dict, Tuple, Optional

# Game constants
COOPERATE = 0
DEFECT = 1

# Pairwise payoff matrix
PAIRWISE_PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}

# N-Person payoff constants
R_REWARD = 3
S_SUCKER = 0
T_TEMPTATION = 5
P_PUNISHMENT = 1


class PairwiseGame:
    """2-Player Iterated Prisoner's Dilemma."""
    
    def __init__(self, agents, num_episodes=1, rounds_per_episode=100):
        self.agents = agents
        self.num_episodes = num_episodes
        self.rounds_per_episode = rounds_per_episode
        self.payoff_matrix = PAIRWISE_PAYOFFS
        self.history = []
        
    def play_round(self, agent1, agent2, round_num):
        """Play a single round between two agents."""
        # Get actions
        _, move1 = agent1.choose_action(agent2.agent_id, mode='pairwise')
        _, move2 = agent2.choose_action(agent1.agent_id, mode='pairwise')
        
        # Get payoffs
        payoff1, payoff2 = self.payoff_matrix[(move1, move2)]
        
        # Update agents
        agent1.update_pairwise(agent2.agent_id, move2, move1, payoff1)
        agent2.update_pairwise(agent1.agent_id, move1, move2, payoff2)
        
        # Record history
        self.history.append({
            'episode': self.current_episode,
            'round': round_num,
            'agent1_id': agent1.agent_id,
            'agent2_id': agent2.agent_id,
            'move1': move1,
            'move2': move2,
            'payoff1': payoff1,
            'payoff2': payoff2
        })
        
        return move1, move2, payoff1, payoff2
    
    def run_episode(self, agent1, agent2, episode_num):
        """Run a single episode between two agents."""
        self.current_episode = episode_num
        
        for round_num in range(self.rounds_per_episode):
            self.play_round(agent1, agent2, round_num)
        
        # Clear history between episodes if needed
        if self.num_episodes > 1 and episode_num < self.num_episodes - 1:
            agent1.clear_opponent_history(agent2.agent_id)
            agent2.clear_opponent_history(agent1.agent_id)
    
    def run_tournament(self):
        """Run full round-robin tournament."""
        # Reset all agents
        for agent in self.agents:
            if hasattr(agent, 'reset_full'):
                agent.reset_full()
            else:
                agent.reset()
        
        # Play all pairs
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                for episode in range(self.num_episodes):
                    self.run_episode(self.agents[i], self.agents[j], episode)
        
        return self.get_results()
    
    def get_results(self):
        """Get tournament results."""
        results = {
            'agents': {},
            'overall': {
                'total_cooperations': 0,
                'total_moves': 0,
                'cooperation_rate': 0.0
            }
        }
        
        for agent in self.agents:
            results['agents'][agent.agent_id] = {
                'total_score': agent.total_score,
                'num_cooperations': agent.num_cooperations,
                'num_defections': agent.num_defections,
                'cooperation_rate': agent.get_cooperation_rate(),
                'strategy': agent.strategy_name if hasattr(agent, 'strategy_name') else 'QLearning'
            }
            
            results['overall']['total_cooperations'] += agent.num_cooperations
            results['overall']['total_moves'] += agent.num_cooperations + agent.num_defections
        
        if results['overall']['total_moves'] > 0:
            results['overall']['cooperation_rate'] = (
                results['overall']['total_cooperations'] / results['overall']['total_moves']
            )
        
        return results


class NPersonGame:
    """N-Person Prisoner's Dilemma with linear payoffs."""
    
    def __init__(self, agents, num_rounds=200, 
                 R=R_REWARD, S=S_SUCKER, T=T_TEMPTATION, P=P_PUNISHMENT):
        self.agents = agents
        self.num_rounds = num_rounds
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        self.history = []
        
    def linear_payoff_cooperator(self, n_others_coop, total_agents):
        """Calculate payoff for cooperator."""
        if total_agents <= 1:
            return self.R
        return self.S + (self.R - self.S) * (n_others_coop / (total_agents - 1))
    
    def linear_payoff_defector(self, n_others_coop, total_agents):
        """Calculate payoff for defector."""
        if total_agents <= 1:
            return self.P
        return self.P + (self.T - self.P) * (n_others_coop / (total_agents - 1))
    
    def play_round(self, round_num, prev_coop_ratio):
        """Play a single round with all agents."""
        n_agents = len(self.agents)
        if n_agents == 0:
            return 0.0
        
        # Get all actions
        moves = {}
        for agent in self.agents:
            _, actual_move = agent.choose_action(prev_coop_ratio, mode='neighborhood')
            moves[agent.agent_id] = actual_move
        
        # Count cooperators
        num_cooperators = sum(1 for move in moves.values() if move == COOPERATE)
        current_coop_ratio = num_cooperators / n_agents if n_agents > 0 else 0.0
        
        # Calculate payoffs and update agents
        round_data = {
            'round': round_num,
            'cooperation_ratio': current_coop_ratio,
            'num_cooperators': num_cooperators,
            'moves': {},
            'payoffs': {}
        }
        
        for agent in self.agents:
            my_move = moves[agent.agent_id]
            n_others_coop = num_cooperators
            if my_move == COOPERATE:
                n_others_coop -= 1
            
            # Calculate payoff
            if my_move == COOPERATE:
                payoff = self.linear_payoff_cooperator(n_others_coop, n_agents)
            else:
                payoff = self.linear_payoff_defector(n_others_coop, n_agents)
            
            # Update agent
            agent.update_neighborhood(my_move, payoff)
            
            # Record data
            round_data['moves'][agent.agent_id] = my_move
            round_data['payoffs'][agent.agent_id] = payoff
        
        self.history.append(round_data)
        return current_coop_ratio
    
    def run_simulation(self):
        """Run full simulation."""
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        
        prev_coop_ratio = None
        
        # Run rounds
        for round_num in range(self.num_rounds):
            prev_coop_ratio = self.play_round(round_num, prev_coop_ratio)
        
        return self.get_results()
    
    def get_results(self):
        """Get simulation results."""
        results = {
            'agents': {},
            'overall': {
                'total_cooperations': 0,
                'total_moves': 0,
                'cooperation_rate': 0.0,
                'final_cooperation_ratio': self.history[-1]['cooperation_ratio'] if self.history else 0.0
            },
            'cooperation_history': [h['cooperation_ratio'] for h in self.history]
        }
        
        # Determine TFT variant used
        tft_variant = "N/A"
        for agent in self.agents:
            if hasattr(agent, 'strategy_name') and agent.strategy_name in ["pTFT", "pTFT-Threshold"]:
                tft_variant = agent.strategy_name
                break
        results['tft_variant'] = tft_variant
        
        # Agent results
        for agent in self.agents:
            results['agents'][agent.agent_id] = {
                'total_score': agent.total_score,
                'num_cooperations': agent.num_cooperations,
                'num_defections': agent.num_defections,
                'cooperation_rate': agent.get_cooperation_rate(),
                'strategy': agent.strategy_name if hasattr(agent, 'strategy_name') else 'QLearning',
                'avg_score_per_round': agent.total_score / self.num_rounds if self.num_rounds > 0 else 0
            }
            
            results['overall']['total_cooperations'] += agent.num_cooperations
            results['overall']['total_moves'] += agent.num_cooperations + agent.num_defections
        
        if results['overall']['total_moves'] > 0:
            results['overall']['cooperation_rate'] = (
                results['overall']['total_cooperations'] / results['overall']['total_moves']
            )
        
        return results


def run_pairwise_experiment(agent_configs, total_rounds=100, episodic=False, num_episodes=10):
    """Run a pairwise experiment with given agent configurations."""
    from .agents import create_agent
    
    # Create agents
    agents = []
    for config in agent_configs:
        agent = create_agent(
            config['id'],
            config['strategy'],
            exploration_rate=config.get('exploration_rate', 0.0)
        )
        agents.append(agent)
    
    # Setup game
    if episodic:
        if total_rounds % num_episodes != 0:
            raise ValueError("Total rounds must be divisible by num_episodes")
        rounds_per_episode = total_rounds // num_episodes
    else:
        num_episodes = 1
        rounds_per_episode = total_rounds
    
    # Run game
    game = PairwiseGame(agents, num_episodes, rounds_per_episode)
    results = game.run_tournament()
    
    return results, game


def run_nperson_experiment(agent_configs, num_rounds=200, tft_variant="pTFT"):
    """Run an N-person experiment with given agent configurations."""
    from .agents import create_agent
    
    # Create agents
    agents = []
    for config in agent_configs:
        strategy = config['strategy']
        # Convert TFT to specified variant for N-person
        if strategy == "TFT":
            strategy = tft_variant
        
        agent = create_agent(
            config['id'],
            strategy,
            exploration_rate=config.get('exploration_rate', 0.0)
        )
        agents.append(agent)
    
    # Run game
    game = NPersonGame(agents, num_rounds)
    results = game.run_simulation()
    
    return results, game