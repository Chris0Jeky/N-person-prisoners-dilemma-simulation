#!/usr/bin/env python3
"""
Basic experiment example - Running a simple tournament
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import run_pairwise_experiment, run_nperson_experiment

def main():
    print("=== Basic Prisoner's Dilemma Experiment ===\n")
    
    # Define agent configurations
    agent_configs = [
        {"id": "TFT_1", "strategy": "TFT"},
        {"id": "TFT_2", "strategy": "TFT"},
        {"id": "AllD_1", "strategy": "AllD"}
    ]
    
    # Run pairwise experiment
    print("1. Running Pairwise Tournament...")
    pairwise_results, _ = run_pairwise_experiment(
        agent_configs, 
        total_rounds=100,
        episodic=False
    )
    
    print(f"\nPairwise Results:")
    print(f"Overall cooperation rate: {pairwise_results['overall']['cooperation_rate']:.3f}")
    for agent_id, stats in pairwise_results['agents'].items():
        print(f"  {agent_id}: Score={stats['total_score']}, "
              f"Cooperation={stats['cooperation_rate']:.3f}")
    
    # Run N-person experiment
    print("\n2. Running N-Person Game...")
    nperson_results, _ = run_nperson_experiment(
        agent_configs,
        num_rounds=200,
        tft_variant="pTFT"
    )
    
    print(f"\nN-Person Results:")
    print(f"Overall cooperation rate: {nperson_results['overall']['cooperation_rate']:.3f}")
    for agent_id, stats in nperson_results['agents'].items():
        print(f"  {agent_id}: Score={stats['total_score']:.1f}, "
              f"Cooperation={stats['cooperation_rate']:.3f}")

if __name__ == "__main__":
    main()