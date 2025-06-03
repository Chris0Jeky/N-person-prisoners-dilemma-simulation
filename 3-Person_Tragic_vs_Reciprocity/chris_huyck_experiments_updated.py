"""
Chris Huyck's Core Scenarios: Reciprocity Hill vs Tragic Valley (Updated)
=========================================================================
This script implements the specific 3-agent experiments requested to demonstrate
the behavioral differences between Neighbourhood (N-Person) and Pairwise models.
Updated to use the modified simultaneous decision-making in pairwise mode.
"""

from unified_simulation import (
    Agent, UnifiedSimulation, InteractionMode, 
    Strategy, Action, save_results
)
import json
import os
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


def run_scenario(scenario_name: str, agents: List[Agent], mode: InteractionMode,
                num_episodes: int = 50, rounds_per_episode: int = 100) -> Dict:
    """Run a single scenario and return detailed results."""
    
    print(f"\nRunning {scenario_name} in {mode.value} mode...")
    
    # Create simulation
    sim = UnifiedSimulation(agents, mode)
    
    # Data collection structures
    all_rounds_data = []
    all_episode_data = []
    agent_move_history = {agent.name: [] for agent in agents}
    agent_score_history = {agent.name: [] for agent in agents}
    cooperation_rate_history = []
    
    # For pairwise mode, track specific dyad interactions
    dyad_cooperation_history = defaultdict(list)  # {(agent1, agent2): [cooperation_rates]}
    
    # Run episodes
    for episode in range(num_episodes):
        # Run the episode using the simulation engine
        episode_data = sim.run_episode(rounds_per_episode, verbose=(episode == 0))
        
        # Extract data from the episode
        episode_moves = {agent.name: [] for agent in agents}
        episode_scores = {agent.name: [] for agent in agents}
        episode_cooperation_rates = []
        episode_dyad_cooperation = defaultdict(list)
        
        # Process each round's data
        for round_detail in episode_data['round_details']:
            round_num = round_detail['round']
            
            if mode == InteractionMode.PAIRWISE:
                # Extract actions from pre_round_actions
                pre_round_actions = round_detail.get('pre_round_actions', {})
                
                # Calculate representative action for each agent (majority vote)
                agent_round_actions = {}
                for agent_name, opponent_actions in pre_round_actions.items():
                    actions = list(opponent_actions.values())
                    coop_count = sum(1 for a in actions if a == 'C')
                    agent_round_actions[agent_name] = 'C' if coop_count > len(actions) / 2 else 'D'
                
                # Record moves
                for agent_name in episode_moves:
                    episode_moves[agent_name].append(agent_round_actions.get(agent_name, 'D'))
                
                # Process interactions for dyad analysis
                for interaction in round_detail['interactions']:
                    agent1 = interaction['agent1']
                    agent2 = interaction['agent2']
                    action1 = interaction['action1']
                    action2 = interaction['action2']
                    
                    # Track dyad cooperation
                    dyad_key = tuple(sorted([agent1, agent2]))
                    both_cooperated = (action1 == 'C' and action2 == 'C')
                    episode_dyad_cooperation[dyad_key].append(1 if both_cooperated else 0)
                
                # Record scores
                for agent_name, score in round_detail['total_payoffs'].items():
                    episode_scores[agent_name].append(score)
                
                # Calculate overall cooperation rate
                coop_count = sum(1 for action in agent_round_actions.values() if action == 'C')
                coop_rate = coop_count / len(agents)
                episode_cooperation_rates.append(coop_rate)
                
            else:  # N_PERSON mode
                # Extract actions and scores directly
                actions = round_detail['actions']
                for agent_name, action in actions.items():
                    episode_moves[agent_name].append(action)
                
                for agent_name, score in round_detail['payoffs'].items():
                    episode_scores[agent_name].append(score)
                
                # Calculate cooperation rate
                coop_count = round_detail['num_cooperators']
                coop_rate = coop_count / len(agents)
                episode_cooperation_rates.append(coop_rate)
            
            # Store round data
            round_data = {
                'episode': episode,
                'round': round_num,
                'cooperation_rate': coop_rate,
                'details': round_detail
            }
            all_rounds_data.append(round_data)
        
        # Store episode data
        episode_summary = {
            'episode': episode,
            'moves': episode_moves,
            'scores': episode_scores,
            'cooperation_rates': episode_cooperation_rates,
            'total_scores': {agent.name: agent.episode_score for agent in agents}
        }
        all_episode_data.append(episode_summary)
        
        # Update histories
        for agent_name in agent_move_history:
            agent_move_history[agent_name].extend(episode_moves[agent_name])
            agent_score_history[agent_name].extend(episode_scores[agent_name])
        cooperation_rate_history.extend(episode_cooperation_rates)
        
        # Update dyad cooperation history (for pairwise mode)
        for dyad_key, cooperation_list in episode_dyad_cooperation.items():
            dyad_cooperation_history[dyad_key].extend(cooperation_list)
    
    # Calculate summary statistics
    summary = {
        'scenario_name': scenario_name,
        'mode': mode.value,
        'num_episodes': num_episodes,
        'rounds_per_episode': rounds_per_episode,
        'agents': [{'name': agent.name, 'strategy': agent.strategy.value, 
                   'exploration_rate': agent.exploration_rate} for agent in agents],
        'final_total_scores': {agent.name: agent.total_score for agent in agents},
        'average_scores_per_round': {
            agent_name: sum(scores) / len(scores) if scores else 0
            for agent_name, scores in agent_score_history.items()
        },
        'cooperation_rates': {
            agent_name: sum(1 for move in moves if move == 'C') / len(moves) if moves else 0
            for agent_name, moves in agent_move_history.items()
        },
        'overall_cooperation_rate': sum(cooperation_rate_history) / len(cooperation_rate_history) if cooperation_rate_history else 0
    }
    
    # Add dyad-specific cooperation rates for pairwise mode
    if mode == InteractionMode.PAIRWISE and dyad_cooperation_history:
        dyad_cooperation_rates = {}
        for dyad_key, cooperation_list in dyad_cooperation_history.items():
            if cooperation_list:
                dyad_cooperation_rates[f"{dyad_key[0]}-{dyad_key[1]}"] = sum(cooperation_list) / len(cooperation_list)
        summary['dyad_cooperation_rates'] = dyad_cooperation_rates
    
    # Full results
    results = {
        'summary': summary,
        'episode_data': all_episode_data,
        'round_data': all_rounds_data
    }
    
    return results


def analyze_pairwise_interactions(results: Dict) -> Dict:
    """Analyze pairwise interactions to separate TFT-TFT from TFT-AllD cooperation."""
    if results['summary']['mode'] != 'pairwise':
        return {}
    
    analysis = {
        'tft_tft_cooperation': [],
        'tft_alld_cooperation': [],
        'round_by_round_tft_tft': [],
        'round_by_round_tft_alld': []
    }
    
    # Process each round's interactions
    for round_data in results['round_data']:
        round_details = round_data['details']
        
        tft_tft_round_coop = []
        tft_alld_round_coop = []
        
        for interaction in round_details['interactions']:
            agent1 = interaction['agent1']
            agent2 = interaction['agent2']
            action1 = interaction['action1']
            action2 = interaction['action2']
            
            # Identify interaction type
            if 'TFT' in agent1 and 'TFT' in agent2:
                # TFT vs TFT
                coop = 1 if (action1 == 'C' and action2 == 'C') else 0
                tft_tft_round_coop.append(coop)
            elif ('TFT' in agent1 and 'AllD' in agent2) or ('AllD' in agent1 and 'TFT' in agent2):
                # TFT vs AllD
                if 'TFT' in agent1:
                    tft_coop = 1 if action1 == 'C' else 0
                else:
                    tft_coop = 1 if action2 == 'C' else 0
                tft_alld_round_coop.append(tft_coop)
        
        # Calculate round averages
        if tft_tft_round_coop:
            analysis['round_by_round_tft_tft'].append(sum(tft_tft_round_coop) / len(tft_tft_round_coop))
        if tft_alld_round_coop:
            analysis['round_by_round_tft_alld'].append(sum(tft_alld_round_coop) / len(tft_alld_round_coop))
    
    # Calculate overall rates
    if analysis['round_by_round_tft_tft']:
        analysis['tft_tft_cooperation'] = sum(analysis['round_by_round_tft_tft']) / len(analysis['round_by_round_tft_tft'])
    if analysis['round_by_round_tft_alld']:
        analysis['tft_alld_cooperation'] = sum(analysis['round_by_round_tft_alld']) / len(analysis['round_by_round_tft_alld'])
    
    return analysis


def run_chris_huyck_experiments():
    """Run all 8 core scenarios requested by Chris Huyck."""
    
    print("="*80)
    print("CHRIS HUYCK'S CORE SCENARIOS (UPDATED)")
    print("3-Person Tragic Valley vs Reciprocity Hill Experiments")
    print("Using Simultaneous Decision-Making in Pairwise Mode")
    print("="*80)
    
    # Configuration
    num_episodes = 50
    rounds_per_episode = 100
    
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/chris_huyck_updated_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    # Scenario Set A: 2 TFT + 1 Always Defect
    print("\n" + "="*60)
    print("SCENARIO SET A: 2 TFT Agents + 1 Always Defect Agent")
    print("="*60)
    
    # A1: No Exploration/Mistakes
    print("\nA1: No Exploration/Mistakes")
    
    # Pairwise
    agents_a1_pw = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    ]
    results_a1_pw = run_scenario("2TFT_1AllD_NoExpl", agents_a1_pw, 
                                InteractionMode.PAIRWISE, num_episodes, rounds_per_episode)
    
    # Analyze pairwise interactions
    pairwise_analysis = analyze_pairwise_interactions(results_a1_pw)
    results_a1_pw['pairwise_analysis'] = pairwise_analysis
    
    all_results['A1_Pairwise'] = results_a1_pw
    save_results(results_a1_pw, f"{results_dir}/results_2TFT_1AllD_NoExpl_Pairwise.json")
    
    # N-Person
    agents_a1_np = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    ]
    results_a1_np = run_scenario("2TFT_1AllD_NoExpl", agents_a1_np, 
                                InteractionMode.N_PERSON, num_episodes, rounds_per_episode)
    all_results['A1_NPerson'] = results_a1_np
    save_results(results_a1_np, f"{results_dir}/results_2TFT_1AllD_NoExpl_NPerson.json")
    
    # Print immediate analysis for A1
    print("\nA1 Results Summary:")
    print(f"Pairwise - TFT overall cooperation: {results_a1_pw['summary']['cooperation_rates'].get('TFT1', 0):.2%}")
    if pairwise_analysis:
        print(f"  - TFT vs TFT cooperation: {pairwise_analysis.get('tft_tft_cooperation', 0):.2%}")
        print(f"  - TFT vs AllD cooperation: {pairwise_analysis.get('tft_alld_cooperation', 0):.2%}")
    print(f"N-Person - TFT overall cooperation: {results_a1_np['summary']['cooperation_rates'].get('TFT1', 0):.2%}")
    
    # A2: 10% Exploration/Mistakes for ALL agents
    print("\nA2: 10% Exploration/Mistakes for ALL agents")
    
    # Pairwise
    agents_a2_pw = [
        Agent(0, "TFT1_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(1, "TFT2_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(2, "AllD_Err", Strategy.ALWAYS_DEFECT, exploration_rate=0.1)
    ]
    results_a2_pw = run_scenario("2TFT_1AllD_10Expl", agents_a2_pw, 
                                InteractionMode.PAIRWISE, num_episodes, rounds_per_episode)
    
    # Analyze pairwise interactions
    pairwise_analysis_a2 = analyze_pairwise_interactions(results_a2_pw)
    results_a2_pw['pairwise_analysis'] = pairwise_analysis_a2
    
    all_results['A2_Pairwise'] = results_a2_pw
    save_results(results_a2_pw, f"{results_dir}/results_2TFT_1AllD_10Expl_Pairwise.json")
    
    # N-Person
    agents_a2_np = [
        Agent(0, "TFT1_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(1, "TFT2_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(2, "AllD_Err", Strategy.ALWAYS_DEFECT, exploration_rate=0.1)
    ]
    results_a2_np = run_scenario("2TFT_1AllD_10Expl", agents_a2_np, 
                                InteractionMode.N_PERSON, num_episodes, rounds_per_episode)
    all_results['A2_NPerson'] = results_a2_np
    save_results(results_a2_np, f"{results_dir}/results_2TFT_1AllD_10Expl_NPerson.json")
    
    # Scenario Set B: 3 TFT Agents (Baseline Comparison)
    print("\n" + "="*60)
    print("SCENARIO SET B: 3 TFT Agents (Baseline Comparison)")
    print("="*60)
    
    # B1: No Exploration/Mistakes
    print("\nB1: No Exploration/Mistakes")
    
    # Pairwise
    agents_b1_pw = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "TFT3", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    ]
    results_b1_pw = run_scenario("3TFT_NoExpl", agents_b1_pw, 
                                InteractionMode.PAIRWISE, num_episodes, rounds_per_episode)
    all_results['B1_Pairwise'] = results_b1_pw
    save_results(results_b1_pw, f"{results_dir}/results_3TFT_NoExpl_Pairwise.json")
    
    # N-Person
    agents_b1_np = [
        Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
        Agent(2, "TFT3", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    ]
    results_b1_np = run_scenario("3TFT_NoExpl", agents_b1_np, 
                                InteractionMode.N_PERSON, num_episodes, rounds_per_episode)
    all_results['B1_NPerson'] = results_b1_np
    save_results(results_b1_np, f"{results_dir}/results_3TFT_NoExpl_NPerson.json")
    
    # B2: 10% Exploration/Mistakes for ALL agents
    print("\nB2: 10% Exploration/Mistakes for ALL agents")
    
    # Pairwise
    agents_b2_pw = [
        Agent(0, "TFT1_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(1, "TFT2_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(2, "TFT3_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
    ]
    results_b2_pw = run_scenario("3TFT_10Expl", agents_b2_pw, 
                                InteractionMode.PAIRWISE, num_episodes, rounds_per_episode)
    all_results['B2_Pairwise'] = results_b2_pw
    save_results(results_b2_pw, f"{results_dir}/results_3TFT_10Expl_Pairwise.json")
    
    # N-Person
    agents_b2_np = [
        Agent(0, "TFT1_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(1, "TFT2_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1),
        Agent(2, "TFT3_Err", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
    ]
    results_b2_np = run_scenario("3TFT_10Expl", agents_b2_np, 
                                InteractionMode.N_PERSON, num_episodes, rounds_per_episode)
    all_results['B2_NPerson'] = results_b2_np
    save_results(results_b2_np, f"{results_dir}/results_3TFT_10Expl_NPerson.json")
    
    # Save combined results
    save_results(all_results, f"{results_dir}/all_results_combined.json")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for scenario_key, results in all_results.items():
        summary = results['summary']
        print(f"\n{scenario_key}: {summary['scenario_name']} ({summary['mode']})")
        print(f"  Overall cooperation rate: {summary['overall_cooperation_rate']:.2%}")
        print(f"  Agent cooperation rates:")
        for agent_name, coop_rate in summary['cooperation_rates'].items():
            print(f"    {agent_name}: {coop_rate:.2%}")
        
        # Print pairwise analysis if available
        if 'pairwise_analysis' in results and results['pairwise_analysis']:
            analysis = results['pairwise_analysis']
            if 'tft_tft_cooperation' in analysis:
                print(f"  Pairwise interaction analysis:")
                print(f"    TFT vs TFT cooperation: {analysis.get('tft_tft_cooperation', 0):.2%}")
                print(f"    TFT vs AllD cooperation: {analysis.get('tft_alld_cooperation', 0):.2%}")
        
        print(f"  Average scores per round:")
        for agent_name, avg_score in summary['average_scores_per_round'].items():
            print(f"    {agent_name}: {avg_score:.2f}")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*80)
    
    return results_dir


if __name__ == "__main__":
    results_dir = run_chris_huyck_experiments()
    print(f"\nExperiments complete! Results directory: {results_dir}")
