"""
Unified experiment runner for systematic testing of agent compositions.

This module handles:
- Running experiments with different agent compositions
- Testing various exploration rates
- Comparing pairwise vs N-person dynamics
- Generating comprehensive results and visualizations
"""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from .agents import create_agent
from .game_environments import run_pairwise_experiment, run_nperson_experiment


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, base_output_dir="results"):
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Default configurations
        self.exploration_rates = [0.0, 0.1]
        self.agent_compositions = [
            {
                "name": "3_TFTs",
                "agents": [
                    {"id": "TFT_1", "strategy": "TFT"},  # Will be converted to pTFT for N-person
                    {"id": "TFT_2", "strategy": "TFT"},
                    {"id": "TFT_3", "strategy": "TFT"}
                ]
            },
            {
                "name": "2_TFTs_1_AllD",
                "agents": [
                    {"id": "TFT_1", "strategy": "TFT"},  # Will be converted to pTFT for N-person
                    {"id": "TFT_2", "strategy": "TFT"},
                    {"id": "AllD_1", "strategy": "AllD"}
                ]
            },
            {
                "name": "1_QL_2_TFTs",
                "agents": [
                    {"id": "QL_1", "strategy": "QL"},
                    {"id": "TFT_1", "strategy": "TFT"},
                    {"id": "TFT_2", "strategy": "TFT"}
                ]
            },
            {
                "name": "2_QL_1_TFT",
                "agents": [
                    {"id": "QL_1", "strategy": "QL"},
                    {"id": "QL_2", "strategy": "QL"},
                    {"id": "TFT_1", "strategy": "TFT"}
                ]
            }
        ]
        
        # Game parameters
        self.pairwise_rounds = 100
        self.nperson_rounds = 200
        self.num_runs = 10
        self.training_rounds = 1000
        
    def set_configurations(self, exploration_rates=None, agent_compositions=None):
        """Set custom configurations."""
        if exploration_rates is not None:
            self.exploration_rates = exploration_rates
        if agent_compositions is not None:
            self.agent_compositions = agent_compositions
    
    def run_single_experiment(self, agent_configs, game_type, exploration_rate, 
                            episodic=False, num_episodes=10, tft_variant="pTFT"):
        """Run a single experiment configuration."""
        # Add exploration rate to configs
        configs_with_exploration = []
        for config in agent_configs:
            new_config = config.copy()
            new_config['exploration_rate'] = exploration_rate
            configs_with_exploration.append(new_config)
        
        if game_type == "pairwise":
            return run_pairwise_experiment(
                configs_with_exploration,
                self.pairwise_rounds,
                episodic,
                num_episodes
            )
        else:  # nperson
            return run_nperson_experiment(
                configs_with_exploration,
                self.nperson_rounds,
                tft_variant
            )
    
    def run_multiple_simulations(self, agent_configs, game_type, exploration_rate,
                               num_runs=None, **kwargs):
        """Run multiple simulations and aggregate results."""
        if num_runs is None:
            num_runs = self.num_runs
        
        all_results = []
        all_games = []
        
        for run in range(num_runs):
            results, game = self.run_single_experiment(
                agent_configs, game_type, exploration_rate, **kwargs
            )
            all_results.append(results)
            all_games.append(game)
        
        # Aggregate results
        aggregated = self.aggregate_results(all_results)
        aggregated['num_runs'] = num_runs
        
        return aggregated, all_games
    
    def aggregate_results(self, results_list):
        """Aggregate results from multiple runs."""
        if not results_list:
            return {}
        
        aggregated = {
            'agents': {},
            'overall': {
                'cooperation_rate_mean': 0.0,
                'cooperation_rate_std': 0.0,
                'cooperation_rates': []
            }
        }
        
        # Collect data for each agent
        agent_data = {}
        for results in results_list:
            for agent_id, agent_results in results['agents'].items():
                if agent_id not in agent_data:
                    agent_data[agent_id] = {
                        'scores': [],
                        'cooperation_rates': [],
                        'strategy': agent_results.get('strategy', 'Unknown')
                    }
                agent_data[agent_id]['scores'].append(agent_results['total_score'])
                agent_data[agent_id]['cooperation_rates'].append(agent_results['cooperation_rate'])
            
            aggregated['overall']['cooperation_rates'].append(results['overall']['cooperation_rate'])
        
        # Calculate aggregated statistics
        if HAS_NUMPY:
            for agent_id, data in agent_data.items():
                aggregated['agents'][agent_id] = {
                    'strategy': data['strategy'],
                    'score_mean': np.mean(data['scores']),
                    'score_std': np.std(data['scores']),
                    'cooperation_rate_mean': np.mean(data['cooperation_rates']),
                    'cooperation_rate_std': np.std(data['cooperation_rates'])
                }
            
            aggregated['overall']['cooperation_rate_mean'] = np.mean(
                aggregated['overall']['cooperation_rates']
            )
            aggregated['overall']['cooperation_rate_std'] = np.std(
                aggregated['overall']['cooperation_rates']
            )
        else:
            # Fallback without numpy
            for agent_id, data in agent_data.items():
                scores = data['scores']
                coop_rates = data['cooperation_rates']
                aggregated['agents'][agent_id] = {
                    'strategy': data['strategy'],
                    'score_mean': sum(scores) / len(scores),
                    'score_std': 0.0,  # Simplified
                    'cooperation_rate_mean': sum(coop_rates) / len(coop_rates),
                    'cooperation_rate_std': 0.0  # Simplified
                }
            
            overall_rates = aggregated['overall']['cooperation_rates']
            aggregated['overall']['cooperation_rate_mean'] = sum(overall_rates) / len(overall_rates)
            aggregated['overall']['cooperation_rate_std'] = 0.0  # Simplified
        
        return aggregated
    
    def run_all_experiments(self):
        """Run all configured experiments."""
        print(f"Starting experiments at {self.timestamp}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"Exploration rates: {self.exploration_rates}")
        print(f"Number of runs per experiment: {self.num_runs}")
        print("-" * 60)
        
        for exploration_rate in self.exploration_rates:
            print(f"\n{'='*50}")
            print(f"EXPLORATION RATE: {exploration_rate*100:.0f}%")
            print(f"{'='*50}")
            
            for composition in self.agent_compositions:
                comp_name = composition['name']
                agent_configs = composition['agents']
                
                print(f"\n{'-'*40}")
                print(f"Composition: {comp_name}")
                print(f"Agents: {len(agent_configs)}")
                print(f"{'-'*40}")
                
                # Pairwise experiments
                print("\n>> Pairwise (Non-episodic)")
                results_pw_non, _ = self.run_multiple_simulations(
                    agent_configs, "pairwise", exploration_rate, 
                    episodic=False
                )
                self.save_results(results_pw_non, comp_name, exploration_rate, 
                                "pairwise_nonepisodic")
                self.print_summary(results_pw_non)
                
                print("\n>> Pairwise (Episodic)")
                results_pw_ep, _ = self.run_multiple_simulations(
                    agent_configs, "pairwise", exploration_rate,
                    episodic=True, num_episodes=10
                )
                self.save_results(results_pw_ep, comp_name, exploration_rate,
                                "pairwise_episodic")
                self.print_summary(results_pw_ep)
                
                # N-Person experiments
                if len(agent_configs) > 1:
                    print("\n>> N-Person (pTFT)")
                    results_np_ptft, _ = self.run_multiple_simulations(
                        agent_configs, "nperson", exploration_rate,
                        tft_variant="pTFT"
                    )
                    self.save_results(results_np_ptft, comp_name, exploration_rate,
                                    "nperson_pTFT")
                    self.print_summary(results_np_ptft)
                    
                    print("\n>> N-Person (pTFT-Threshold)")
                    results_np_thresh, _ = self.run_multiple_simulations(
                        agent_configs, "nperson", exploration_rate,
                        tft_variant="pTFT-Threshold"
                    )
                    self.save_results(results_np_thresh, comp_name, exploration_rate,
                                    "nperson_pTFT-Threshold")
                    self.print_summary(results_np_thresh)
        
        print(f"\n{'='*60}")
        print("All experiments completed!")
        print(f"Results saved in: {self.base_output_dir}")
    
    def print_summary(self, results):
        """Print summary of results."""
        print(f"Overall cooperation rate: {results['overall']['cooperation_rate_mean']:.3f} "
              f"(±{results['overall']['cooperation_rate_std']:.3f})")
        
        # Sort agents by score
        sorted_agents = sorted(
            results['agents'].items(),
            key=lambda x: x[1]['score_mean'],
            reverse=True
        )
        
        for agent_id, agent_results in sorted_agents[:3]:  # Top 3
            print(f"  {agent_id} ({agent_results['strategy']}): "
                  f"Score={agent_results['score_mean']:.1f}, "
                  f"Coop={agent_results['cooperation_rate_mean']:.3f}")
    
    def save_results(self, results, composition_name, exploration_rate, experiment_type):
        """Save results to file."""
        # Create directory structure
        exp_dir = os.path.join(
            self.base_output_dir,
            f"exp_{exploration_rate*100:.0f}",
            composition_name,
            experiment_type
        )
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(exp_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_path = os.path.join(exp_dir, "summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['agent_id', 'strategy', 'score_mean', 'score_std',
                           'cooperation_rate_mean', 'cooperation_rate_std'])
            for agent_id, agent_results in results['agents'].items():
                writer.writerow([
                    agent_id,
                    agent_results['strategy'],
                    agent_results['score_mean'],
                    agent_results['score_std'],
                    agent_results['cooperation_rate_mean'],
                    agent_results['cooperation_rate_std']
                ])
    
    def generate_plots(self):
        """Generate visualization plots if matplotlib is available."""
        if not HAS_PLOTTING:
            print("Matplotlib not available, skipping plots")
            return
        
        # Implementation would go here
        print("Plot generation not yet implemented")


def run_qlearning_experiments(output_dir="qlearning_results", num_runs=15, 
                            training_rounds=1000):
    """Run specific Q-learning experiments."""
    runner = ExperimentRunner(output_dir)
    runner.num_runs = num_runs
    runner.training_rounds = training_rounds
    
    # Q-learning specific compositions
    ql_compositions = [
        {
            "name": "1QL_2TFT",
            "agents": [
                {"id": "QL_1", "strategy": "QL"},
                {"id": "TFT_1", "strategy": "TFT"},
                {"id": "TFT_2", "strategy": "TFT"}
            ]
        },
        {
            "name": "1QL_1TFT_1AllD",
            "agents": [
                {"id": "QL_1", "strategy": "QL"},
                {"id": "TFT_1", "strategy": "TFT"},
                {"id": "AllD_1", "strategy": "AllD"}
            ]
        },
        {
            "name": "2QL_1TFT",
            "agents": [
                {"id": "QL_1", "strategy": "QL"},
                {"id": "QL_2", "strategy": "QL"},
                {"id": "TFT_1", "strategy": "TFT"}
            ]
        },
        {
            "name": "2QL_1AllD",
            "agents": [
                {"id": "QL_1", "strategy": "QL"},
                {"id": "QL_2", "strategy": "QL"},
                {"id": "AllD_1", "strategy": "AllD"}
            ]
        }
    ]
    
    runner.set_configurations(
        exploration_rates=[0.0],  # Q-learning uses epsilon instead
        agent_compositions=ql_compositions
    )
    
    runner.run_all_experiments()
    return runner


if __name__ == "__main__":
    # Run default experiments
    runner = ExperimentRunner()
    runner.run_all_experiments()