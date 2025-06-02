"""
Main Experiments: Reciprocity Hill vs Tragic Valley
===================================================
Proving the difference between pairwise and N-person dynamics.
"""

from unified_simulation import (
    create_agents, UnifiedSimulation, InteractionMode, 
    Strategy, Agent, save_results
)
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json


def experiment_1_three_agents():
    """
    Experiment 1: 3 agents with binary options
    - 2 TFT + 1 Defect vs 3 TFT
    - Standard TFT vs Vote-based TFT (1 vote vs 2 votes)
    - No exploration vs 10% exploration
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Three Agent Scenarios")
    print("="*80)
    
    results = {}
    
    # Scenario 1a: 2 TFT + 1 Defect, No exploration
    print("\n--- Scenario 1a: 2 TFT + 1 Defect (No exploration) ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['2tft_1d_noexp_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['2tft_1d_noexp_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # Scenario 1b: 2 TFT + 1 Defect, 10% exploration
    print("\n--- Scenario 1b: 2 TFT + 1 Defect (10% exploration) ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['2tft_1d_exp_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['2tft_1d_exp_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # Scenario 2: 3 TFT (for comparison)
    print("\n--- Scenario 2: 3 TFT (No exploration) ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "TFT-3", Strategy.TIT_FOR_TAT, 0.0)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['3tft_noexp_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "TFT-3", Strategy.TIT_FOR_TAT, 0.0)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['3tft_noexp_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # Scenario 3: Vote-based TFT variants
    print("\n--- Scenario 3: Vote-based TFT (1-vote vs 2-vote) ---")
    
    # N-Person with Vote-TFT-1 (needs only 1 cooperator)
    agents_vote1 = [
        Agent(0, "VoteTFT1-1", Strategy.VOTE_TFT_1, 0.0),
        Agent(1, "VoteTFT1-2", Strategy.VOTE_TFT_1, 0.0),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_vote1 = UnifiedSimulation(agents_vote1, InteractionMode.N_PERSON)
    results['vote1_tft_nperson'] = sim_vote1.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person with Vote-TFT-2 (needs 2 cooperators)
    agents_vote2 = [
        Agent(0, "VoteTFT2-1", Strategy.VOTE_TFT_2, 0.0),
        Agent(1, "VoteTFT2-2", Strategy.VOTE_TFT_2, 0.0),
        Agent(2, "Defector", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_vote2 = UnifiedSimulation(agents_vote2, InteractionMode.N_PERSON)
    results['vote2_tft_nperson'] = sim_vote2.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    return results


def experiment_2_five_agents():
    """
    Experiment 2: 5 agents (2 TFT + 3 Defect)
    Testing the hypothesis that N-person leads to tragic valley
    while pairwise maintains reciprocity hill.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Five Agent Scenarios (2 TFT + 3 Defect)")
    print("="*80)
    
    results = {}
    
    # Scenario 1: Standard TFT, No exploration
    print("\n--- Scenario 1: Standard TFT (No exploration) ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['5agent_noexp_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.0),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.0),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['5agent_noexp_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # Scenario 2: Threshold TFT
    print("\n--- Scenario 2: Threshold TFT (50% cooperation threshold) ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "ThresholdTFT-1", Strategy.THRESHOLD_TFT, 0.0),
        Agent(1, "ThresholdTFT-2", Strategy.THRESHOLD_TFT, 0.0),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['5agent_threshold_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "ThresholdTFT-1", Strategy.THRESHOLD_TFT, 0.0),
        Agent(1, "ThresholdTFT-2", Strategy.THRESHOLD_TFT, 0.0),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.0),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.0)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['5agent_threshold_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # Scenario 3: With exploration
    print("\n--- Scenario 3: Standard TFT with 10% exploration ---")
    
    # Pairwise
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    results['5agent_exp_pairwise'] = sim_pw.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    # N-Person
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    results['5agent_exp_nperson'] = sim_np.run_simulation(
        episodes=10, rounds_per_episode=10, verbose=False
    )
    
    return results


def visualize_results(results: Dict, title: str):
    """Create visualizations comparing pairwise vs N-person dynamics."""
    # Extract cooperation evolution for TFT agents
    scenarios = []
    pairwise_coop = []
    nperson_coop = []
    
    for key, data in results.items():
        if 'tft' in key.lower() or 'threshold' in key.lower():
            # Get average TFT cooperation across episodes
            tft_names = [name for name in data['cooperation_evolution'][0].keys() 
                        if 'TFT' in name or 'Threshold' in name]
            
            if tft_names:
                avg_coop_episodes = []
                for episode in data['cooperation_evolution']:
                    avg_coop = np.mean([episode[name] for name in tft_names if name in episode])
                    avg_coop_episodes.append(avg_coop)
                
                avg_coop = np.mean(avg_coop_episodes)
                
                scenario_name = key.replace('_pairwise', '').replace('_nperson', '')
                if scenario_name not in scenarios:
                    scenarios.append(scenario_name)
                
                if 'pairwise' in key:
                    pairwise_coop.append(avg_coop)
                else:
                    nperson_coop.append(avg_coop)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot comparison
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, pairwise_coop, width, label='Pairwise', alpha=0.8)
    ax1.bar(x + width/2, nperson_coop, width, label='N-Person', alpha=0.8)
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Average TFT Cooperation Rate')
    ax1.set_title(f'{title}: Cooperation Rates')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score comparison
    pairwise_scores = []
    nperson_scores = []
    
    for scenario in scenarios:
        pw_key = f"{scenario}_pairwise"
        np_key = f"{scenario}_nperson"
        
        if pw_key in results and np_key in results:
            # Get TFT scores
            pw_data = results[pw_key]
            np_data = results[np_key]
            
            pw_tft_scores = [score for name, score in pw_data['final_scores'].items() 
                           if 'TFT' in name or 'Threshold' in name]
            np_tft_scores = [score for name, score in np_data['final_scores'].items() 
                           if 'TFT' in name or 'Threshold' in name]
            
            if pw_tft_scores:
                pairwise_scores.append(np.mean(pw_tft_scores))
            if np_tft_scores:
                nperson_scores.append(np.mean(np_tft_scores))
    
    ax2.bar(x - width/2, pairwise_scores, width, label='Pairwise', alpha=0.8)
    ax2.bar(x + width/2, nperson_scores, width, label='N-Person', alpha=0.8)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Average TFT Total Score')
    ax2.set_title(f'{title}: TFT Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reciprocity_hill_research/results/{title.lower().replace(" ", "_")}_comparison.png')
    plt.close()


def main():
    """Run all experiments and save results."""
    print("="*80)
    print("RECIPROCITY HILL VS TRAGIC VALLEY RESEARCH")
    print("="*80)
    
    # Run experiments
    exp1_results = experiment_1_three_agents()
    exp2_results = experiment_2_five_agents()
    
    # Combine results
    all_results = {**exp1_results, **exp2_results}
    
    # Save detailed results
    save_results(all_results, 'complete_results.json')
    
    # Create visualizations
    visualize_results(exp1_results, "Three Agent Experiments")
    visualize_results(exp2_results, "Five Agent Experiments")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    # Compare key scenarios
    key_comparisons = [
        ('2tft_1d_noexp', '2 TFT + 1 Defect (No exploration)'),
        ('2tft_1d_exp', '2 TFT + 1 Defect (10% exploration)'),
        ('5agent_noexp', '2 TFT + 3 Defect (No exploration)'),
        ('5agent_threshold', '2 Threshold-TFT + 3 Defect')
    ]
    
    for scenario_key, scenario_name in key_comparisons:
        pw_key = f"{scenario_key}_pairwise"
        np_key = f"{scenario_key}_nperson"
        
        if pw_key in all_results and np_key in all_results:
            print(f"\n{scenario_name}:")
            
            # Get TFT cooperation rates
            pw_coop = all_results[pw_key]['cooperation_evolution']
            np_coop = all_results[np_key]['cooperation_evolution']
            
            # Average across episodes
            pw_tft_avg = []
            np_tft_avg = []
            
            for episode in pw_coop:
                tft_rates = [rate for name, rate in episode.items() if 'TFT' in name or 'Threshold' in name]
                if tft_rates:
                    pw_tft_avg.append(np.mean(tft_rates))
            
            for episode in np_coop:
                tft_rates = [rate for name, rate in episode.items() if 'TFT' in name or 'Threshold' in name]
                if tft_rates:
                    np_tft_avg.append(np.mean(tft_rates))
            
            if pw_tft_avg and np_tft_avg:
                print(f"  Pairwise - TFT cooperation: {np.mean(pw_tft_avg):.2%}")
                print(f"  N-Person - TFT cooperation: {np.mean(np_tft_avg):.2%}")
                print(f"  Difference: {np.mean(pw_tft_avg) - np.mean(np_tft_avg):.2%} " +
                      f"({'Reciprocity Hill' if np.mean(pw_tft_avg) > np.mean(np_tft_avg) else 'Tragic Valley'})")
    
    print("\n" + "="*80)
    print("RESEARCH COMPLETE - Results saved to reciprocity_hill_research/results/")
    print("="*80)


if __name__ == "__main__":
    main()
