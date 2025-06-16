"""
Main Experiment Runner for N-Person Prisoner's Dilemma

Supports running experiments with configurable parameters and agent compositions.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from core import NPDGame, PairwiseGame
from agents import (
    Agent, TFTAgent, pTFTAgent, pTFTThresholdAgent,
    AllCAgent, AllDAgent, RandomAgent,
    QLearningAgent, EnhancedQLearningAgent,
    AgentRegistry
)
from utils.logging import setup_logger


class ExperimentRunner:
    """
    Runs experiments with N-Person Prisoner's Dilemma games.
    
    Features:
    - Configurable agent compositions
    - Multiple game modes (NPD, Pairwise)
    - Automatic result saving
    - Progress tracking
    """
    
    def __init__(self, 
                 output_dir: str = "results",
                 experiment_name: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Base directory for results
            experiment_name: Name for this experiment run
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(f"ExperimentRunner_{self.experiment_name}")
        
        # Create output directories
        self.setup_output_dirs()
        
    def setup_output_dirs(self):
        """Create output directory structure."""
        self.experiment_dir = self.output_dir / self.experiment_name
        self.csv_dir = self.experiment_dir / "csv"
        self.figures_dir = self.experiment_dir / "figures"
        
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def create_agent(self, agent_config: Dict[str, Any]) -> Agent:
        """
        Create an agent based on configuration.
        
        Args:
            agent_config: Configuration dict with 'type', 'id', and parameters
            
        Returns:
            Configured agent instance
        """
        agent_type = agent_config['type']
        agent_id = agent_config['id']
        exploration_rate = agent_config.get('exploration_rate', 0.0)
        
        # Strategy-based agents
        if agent_type == "TFT":
            return TFTAgent(agent_id, exploration_rate)
        elif agent_type == "pTFT":
            return pTFTAgent(agent_id, exploration_rate)
        elif agent_type == "pTFT-Threshold":
            threshold = agent_config.get('threshold', 0.5)
            return pTFTThresholdAgent(agent_id, exploration_rate, threshold)
        elif agent_type == "AllC":
            return AllCAgent(agent_id, exploration_rate)
        elif agent_type == "AllD":
            return AllDAgent(agent_id, exploration_rate)
        elif agent_type == "Random":
            coop_prob = agent_config.get('cooperation_probability', 0.5)
            return RandomAgent(agent_id, coop_prob, exploration_rate)
        
        # Q-Learning agents
        elif agent_type == "QLearning":
            params = {
                'learning_rate': agent_config.get('learning_rate', 0.1),
                'discount_factor': agent_config.get('discount_factor', 0.9),
                'epsilon': agent_config.get('epsilon', 0.1),
                'state_type': agent_config.get('state_type', 'basic')
            }
            return QLearningAgent(agent_id, exploration_rate=exploration_rate, **params)
        
        elif agent_type == "EnhancedQLearning":
            params = {
                'learning_rate': agent_config.get('learning_rate', 0.1),
                'discount_factor': agent_config.get('discount_factor', 0.9),
                'epsilon': agent_config.get('epsilon', 0.1),
                'epsilon_decay': agent_config.get('epsilon_decay', 1.0),
                'epsilon_min': agent_config.get('epsilon_min', 0.01),
                'exclude_self': agent_config.get('exclude_self', False),
                'opponent_modeling': agent_config.get('opponent_modeling', False),
                'state_type': agent_config.get('state_type', 'basic')
            }
            agent = EnhancedQLearningAgent(agent_id, exploration_rate=exploration_rate, **params)
            return agent
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def run_npd_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an N-Person Prisoner's Dilemma experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Results dictionary
        """
        self.logger.info(f"Starting NPD experiment: {config.get('name', 'unnamed')}")
        
        # Create agents
        agents = []
        for agent_config in config['agents']:
            agent = self.create_agent(agent_config)
            agents.append(agent)
            
        # Set number of agents for enhanced Q-learning
        num_agents = len(agents)
        for agent in agents:
            if hasattr(agent, 'set_num_agents'):
                agent.set_num_agents(num_agents)
        
        # Create game
        game = NPDGame(
            num_agents=num_agents,
            num_rounds=config['num_rounds']
        )
        
        # Run simulation
        for round_num in range(config['num_rounds']):
            # Collect actions
            actions = {}
            coop_ratio = game.get_cooperation_ratio() if round_num > 0 else None
            
            for agent in agents:
                _, actual_action = agent.choose_action(coop_ratio, round_num)
                actions[agent.agent_id] = actual_action
            
            # Play round
            payoffs = game.play_round(actions)
            
            # Update agents
            for agent in agents:
                agent.record_outcome(actions[agent.agent_id], payoffs[agent.agent_id])
                
                # Update opponent models for enhanced Q-learning
                if hasattr(agent, 'update_from_game_state'):
                    agent.update_from_game_state(actions)
            
            # Apply episode decay for enhanced Q-learning
            if round_num > 0 and round_num % config.get('episode_length', 100) == 0:
                for agent in agents:
                    if hasattr(agent, 'start_new_episode'):
                        agent.start_new_episode()
        
        # Get results
        results = game.get_results()
        results['config'] = config
        results['agent_stats'] = [agent.get_stats() for agent in agents]
        
        # Save results
        self.save_results(results, config.get('name', 'npd_experiment'))
        
        return results
    
    def run_pairwise_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a Pairwise Prisoner's Dilemma tournament.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Results dictionary
        """
        self.logger.info(f"Starting Pairwise experiment: {config.get('name', 'unnamed')}")
        
        # Create agents
        agents = []
        for agent_config in config['agents']:
            agent = self.create_agent(agent_config)
            agents.append(agent)
        
        # Create game
        game = PairwiseGame(
            num_agents=len(agents),
            rounds_per_pair=config['rounds_per_pair'],
            num_episodes=config.get('num_episodes', 1)
        )
        
        # Run tournament
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Run episodes
                for episode in range(game.num_episodes):
                    # Run rounds
                    for round_num in range(game.rounds_per_pair):
                        # Get actions
                        _, action1 = agent1.choose_action(agent2.agent_id, round_num)
                        _, action2 = agent2.choose_action(agent1.agent_id, round_num)
                        
                        # Play round
                        payoff1, payoff2 = game.play_pairwise_round(
                            agent1.agent_id, agent2.agent_id, action1, action2
                        )
                        
                        # Update agents
                        agent1.record_interaction(agent2.agent_id, action2, action1, payoff1)
                        agent2.record_interaction(agent1.agent_id, action1, action2, payoff2)
                        
                        # Record interaction
                        game.record_pairwise_interaction(
                            agent1.agent_id, agent2.agent_id,
                            action1, action2, payoff1, payoff2,
                            round_num, episode
                        )
                    
                    # Clear history between episodes
                    if episode < game.num_episodes - 1:
                        agent1.clear_opponent_history(agent2.agent_id)
                        agent2.clear_opponent_history(agent1.agent_id)
        
        # Get results
        results = game.get_tournament_results()
        results['config'] = config
        results['agent_stats'] = [agent.get_stats() for agent in agents]
        
        # Save results
        self.save_results(results, config.get('name', 'pairwise_experiment'))
        
        return results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save experiment results to CSV and JSON."""
        # Save full results as JSON
        json_path = self.experiment_dir / f"{experiment_name}_results.json"
        with open(json_path, 'w') as f:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save summary to CSV
        csv_path = self.csv_dir / f"{experiment_name}_summary.csv"
        self._save_summary_csv(results, csv_path)
        
        # Save history to CSV if available
        if 'history' in results:
            history_csv_path = self.csv_dir / f"{experiment_name}_history.csv"
            self._save_history_csv(results['history'], history_csv_path)
        
        self.logger.info(f"Results saved to {self.experiment_dir}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects for JSON saving."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj
    
    def _save_summary_csv(self, results: Dict[str, Any], csv_path: Path):
        """Save summary statistics to CSV."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers
            writer.writerow(['Metric', 'Value'])
            
            # Write basic stats
            writer.writerow(['num_agents', results.get('num_agents', 'N/A')])
            writer.writerow(['num_rounds', results.get('num_rounds', 'N/A')])
            writer.writerow(['average_cooperation', results.get('average_cooperation', 'N/A')])
            
            # Write agent stats
            if 'agent_stats' in results:
                writer.writerow([])
                writer.writerow(['Agent Stats'])
                writer.writerow(['agent_id', 'total_score', 'cooperation_rate'])
                
                for agent_stat in results['agent_stats']:
                    writer.writerow([
                        agent_stat['agent_id'],
                        agent_stat['total_score'],
                        agent_stat['cooperation_rate']
                    ])
    
    def _save_history_csv(self, history: List[Dict], csv_path: Path):
        """Save game history to CSV."""
        if not history:
            return
            
        with open(csv_path, 'w', newline='') as f:
            # Get all unique keys from history
            all_keys = set()
            for entry in history:
                all_keys.update(entry.keys())
            
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(history)