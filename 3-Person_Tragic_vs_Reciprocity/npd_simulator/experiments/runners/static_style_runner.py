"""
Static Style Experiment Runner
Runs experiments matching the configurations in static_figure_generator.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
from tqdm import tqdm

from npd_simulator.experiments.runners.experiment_runner import ExperimentRunner
from npd_simulator.core.models.enhanced_game_state import EnhancedGameState
from npd_simulator.core import NPDGame, PairwiseGame
from npd_simulator.static_style_visualizer import StaticStyleVisualizer
from npd_simulator.utils.logging import setup_logger


class StaticStyleRunner(ExperimentRunner):
    """
    Runs experiments in the style of static_figure_generator.py with multiple runs per configuration.
    """
    
    # Standard experiment configurations matching static_figure_generator.py
    STANDARD_EXPERIMENTS = {
        "3 TFT": {
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.0},
                {"id": 1, "type": "TFT", "exploration_rate": 0.0},
                {"id": 2, "type": "TFT", "exploration_rate": 0.0}
            ]
        },
        "2 TFT-E + 1 AllD": {
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.1},
                {"id": 1, "type": "TFT", "exploration_rate": 0.1},
                {"id": 2, "type": "AllD", "exploration_rate": 0.0}
            ]
        },
        "2 TFT + 1 AllD": {
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.0},
                {"id": 1, "type": "TFT", "exploration_rate": 0.0},
                {"id": 2, "type": "AllD", "exploration_rate": 0.0}
            ]
        },
        "2 TFT-E + 1 AllC": {
            "agents": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.1},
                {"id": 1, "type": "TFT", "exploration_rate": 0.1},
                {"id": 2, "type": "AllC", "exploration_rate": 0.0}
            ]
        }
    }
    
    def __init__(self, 
                 output_dir: str = "results/static_style",
                 num_runs: int = 20,
                 num_rounds: int = 100,
                 rounds_per_pair: int = 100):
        """
        Initialize static style runner.
        
        Args:
            output_dir: Base directory for results
            num_runs: Number of runs per experiment configuration
            num_rounds: Number of rounds for N-person games
            rounds_per_pair: Number of rounds per pair for pairwise games
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        super().__init__(output_dir, f"static_style_{timestamp}")
        
        self.num_runs = num_runs
        self.num_rounds = num_rounds
        self.rounds_per_pair = rounds_per_pair
        self.visualizer = StaticStyleVisualizer(self.figures_dir)
        
    def run_npd_experiment_enhanced(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run an N-Person experiment with enhanced tracking.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Results dictionary with enhanced metrics
        """
        # Create agents
        agents = []
        for agent_config in config['agents']:
            agent = self.create_agent(agent_config)
            agents.append(agent)
        
        # Create enhanced game state
        game_state = EnhancedGameState(len(agents), agents)
        
        # Create game
        game = NPDGame(
            num_agents=len(agents),
            num_rounds=config['num_rounds']
        )
        
        # Override game state
        game.game_state = game_state
        
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
        
        # Get enhanced results
        results = game_state.get_enhanced_results()
        results['config'] = config
        results['experiment_type'] = 'nperson'
        
        return results
    
    def run_pairwise_experiment_enhanced(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a Pairwise experiment with enhanced tracking.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Results dictionary with enhanced metrics
        """
        # Create agents
        agents = []
        for agent_config in config['agents']:
            agent = self.create_agent(agent_config)
            agents.append(agent)
        
        # Create enhanced game state
        game_state = EnhancedGameState(len(agents), agents)
        
        # Create game
        game = PairwiseGame(
            num_agents=len(agents),
            rounds_per_pair=config['rounds_per_pair']
        )
        
        # Track round-by-round for visualization
        for round_num in range(config['rounds_per_pair']):
            round_actions = {}
            round_payoffs = {}
            
            # Initialize payoffs
            for agent in agents:
                round_payoffs[agent.agent_id] = 0
            
            # Run all pairwise interactions for this round
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent1, agent2 = agents[i], agents[j]
                    
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
                    
                    # Accumulate payoffs
                    round_payoffs[agent1.agent_id] += payoff1
                    round_payoffs[agent2.agent_id] += payoff2
                    
                    # Track actions (use most recent for each agent)
                    round_actions[agent1.agent_id] = action1
                    round_actions[agent2.agent_id] = action2
            
            # Update game state
            game_state.update(round_actions, round_payoffs)
        
        # Get enhanced results
        results = game_state.get_enhanced_results()
        results['config'] = config
        results['experiment_type'] = 'pairwise'
        
        return results
    
    def run_single_experiment(self, exp_name: str, exp_config: Dict[str, Any], 
                            game_type: str) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        config = exp_config.copy()
        config['name'] = exp_name
        config['num_rounds'] = self.num_rounds
        config['rounds_per_pair'] = self.rounds_per_pair
        
        if game_type == 'nperson':
            return self.run_npd_experiment_enhanced(config)
        else:
            return self.run_pairwise_experiment_enhanced(config)
    
    def run_experiment_batch(self, exp_name: str, exp_config: Dict[str, Any], 
                           game_type: str, num_runs: int) -> List[Dict[str, Any]]:
        """Run multiple runs of an experiment configuration."""
        results = []
        
        for run in range(num_runs):
            self.logger.info(f"Running {exp_name} - Run {run + 1}/{num_runs}")
            result = self.run_single_experiment(exp_name, exp_config, game_type)
            results.append(result)
        
        return results
    
    def run_all_experiments(self, game_types: List[str] = ['pairwise', 'nperson']):
        """
        Run all standard experiments with multiple runs.
        
        Args:
            game_types: List of game types to run ('pairwise', 'nperson')
        """
        all_results = {}
        
        for game_type in game_types:
            self.logger.info(f"\nRunning {game_type} experiments...")
            
            for exp_name, exp_config in self.STANDARD_EXPERIMENTS.items():
                self.logger.info(f"\nExperiment: {exp_name}")
                
                # Run multiple times
                results = self.run_experiment_batch(
                    exp_name, exp_config, game_type, self.num_runs
                )
                
                # Store results
                key = f"{game_type}_{exp_name}"
                all_results[key] = results
                
                # Save individual experiment results
                self.save_batch_results(results, key)
        
        # Generate static-style visualizations
        self.generate_visualizations(all_results)
        
        return all_results
    
    def save_batch_results(self, results: List[Dict[str, Any]], experiment_key: str):
        """Save batch results for an experiment."""
        batch_dir = self.experiment_dir / "batches" / experiment_key
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each run
        for i, result in enumerate(results):
            run_file = batch_dir / f"run_{i:03d}.json"
            with open(run_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save aggregated summary
        summary = {
            'experiment_key': experiment_key,
            'num_runs': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = batch_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_visualizations(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Generate static-style visualizations for all results."""
        # Separate pairwise and nperson results
        pairwise_results = {k.replace('pairwise_', ''): v 
                          for k, v in all_results.items() if k.startswith('pairwise_')}
        nperson_results = {k.replace('nperson_', ''): v 
                         for k, v in all_results.items() if k.startswith('nperson_')}
        
        # Generate figures for each game type
        if pairwise_results:
            self.logger.info("\nGenerating pairwise visualizations...")
            self.visualizer.generate_all_figures(pairwise_results)
        
        if nperson_results:
            self.logger.info("\nGenerating N-person visualizations...")
            # Save in separate directory
            nperson_viz = StaticStyleVisualizer(self.figures_dir / 'nperson')
            nperson_viz.generate_all_figures(nperson_results)
    
    def run_custom_experiment(self, name: str, agent_configs: List[Dict], 
                            game_types: List[str] = ['pairwise', 'nperson']):
        """
        Run a custom experiment configuration.
        
        Args:
            name: Experiment name
            agent_configs: List of agent configurations
            game_types: Game types to run
        """
        config = {"agents": agent_configs}
        all_results = {}
        
        for game_type in game_types:
            results = self.run_experiment_batch(name, config, game_type, self.num_runs)
            key = f"{game_type}_{name}"
            all_results[key] = results
            self.save_batch_results(results, key)
        
        self.generate_visualizations(all_results)
        return all_results


def main():
    """Run static-style experiments."""
    runner = StaticStyleRunner(
        output_dir="results/static_style",
        num_runs=20,
        num_rounds=100,
        rounds_per_pair=100
    )
    
    # Run all standard experiments
    results = runner.run_all_experiments()
    
    print(f"\nExperiments completed. Results saved to: {runner.experiment_dir}")


if __name__ == "__main__":
    main()