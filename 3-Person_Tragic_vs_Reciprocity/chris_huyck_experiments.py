"""
Chris Huyck's Core Scenarios: Reciprocity Hill vs Tragic Valley
================================================================
This script implements the specific 3-agent experiments requested to demonstrate
the behavioral differences between Neighbourhood (N-Person) and Pairwise models.
"""

from unified_simulation import (
    Agent, UnifiedSimulation, InteractionMode, 
    Strategy, Action, save_results
)
import json
import os
from datetime import datetime
from typing import Dict, List


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
    
    # Run episodes
    for episode in range(num_episodes):
        # Reset agents for new episode
        sim.reset_episode()
        
        episode_moves = {agent.name: [] for agent in agents}
        episode_scores = {agent.name: [] for agent in agents}
        episode_cooperation_rates = []
        
        # Run rounds
        for round_num in range(rounds_per_episode):
            # Store actions before round
            pre_round_actions = {}
            
            # Run the round
            if mode == InteractionMode.PAIRWISE:
                # Pairwise interactions
                round_actions = {}
                round_scores = {agent.name: 0.0 for agent in agents}
                
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        agent1, agent2 = agents[i], agents[j]
                        
                        # Agents decide
                        action1 = agent1.decide_action(mode, opponent_id=agent2.id)
                        action2 = agent2.decide_action(mode, opponent_id=agent1.id)
                        
                        # Store actions
                        if agent1.name not in round_actions:
                            round_actions[agent1.name] = []
                        if agent2.name not in round_actions:
                            round_actions[agent2.name] = []
                        
                        round_actions[agent1.name].append(action1)
                        round_actions[agent2.name].append(action2)
                        
                        # Calculate payoffs
                        if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
                            payoff1, payoff2 = sim.R, sim.R
                        elif action1 == Action.COOPERATE and action2 == Action.DEFECT:
                            payoff1, payoff2 = sim.S, sim.T
                        elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
                            payoff1, payoff2 = sim.T, sim.S
                        else:
                            payoff1, payoff2 = sim.P, sim.P
                        
                        # Update histories
                        agent1.update_history(mode, action1, agent2.id, action2, payoff=payoff1)
                        agent2.update_history(mode, action2, agent1.id, action1, payoff=payoff2)
                        
                        # Track scores
                        round_scores[agent1.name] += payoff1
                        round_scores[agent2.name] += payoff2
                
                # For pairwise, we'll use the most common action as the representative action
                for agent_name, actions in round_actions.items():
                    # Count cooperations
                    coop_count = sum(1 for a in actions if a == Action.COOPERATE)
                    # Use majority action as representative
                    pre_round_actions[agent_name] = Action.COOPERATE if coop_count > len(actions) / 2 else Action.DEFECT
                    
            else:  # N_PERSON
                # All agents decide simultaneously
                actions = {}
                for agent in agents:
                    action = agent.decide_action(mode)
                    actions[agent.id] = action
                    pre_round_actions[agent.name] = action
                
                # Calculate payoffs
                round_scores = {}
                num_cooperators = sum(1 for action in actions.values() if action == Action.COOPERATE)
                
                for agent in agents:
                    agent_action = actions[agent.id]
                    n = num_cooperators - (1 if agent_action == Action.COOPERATE else 0)
                    N = len(agents)
                    
                    if agent_action == Action.COOPERATE:
                        payoff = sim.S + (sim.R - sim.S) * (n / (N - 1)) if N > 1 else sim.R
                    else:
                        payoff = sim.P + (sim.T - sim.P) * (n / (N - 1)) if N > 1 else sim.P
                    
                    round_scores[agent.name] = payoff
                    
                    # Update agent history
                    agent.update_history(mode, agent_action, group_actions=actions, payoff=payoff)
            
            # Record data for this round
            for agent_name in episode_moves:
                episode_moves[agent_name].append(pre_round_actions[agent_name].value)
                episode_scores[agent_name].append(round_scores[agent_name])
            
            # Calculate cooperation rate for this round
            coop_count = sum(1 for action in pre_round_actions.values() if action == Action.COOPERATE)
            coop_rate = coop_count / len(agents)
            episode_cooperation_rates.append(coop_rate)
            
            # Store round data
            round_data = {
                'episode': episode,
                'round': round_num,
                'actions': {name: action.value for name, action in pre_round_actions.items()},
                'scores': round_scores,
                'cooperation_rate': coop_rate
            }
            all_rounds_data.append(round_data)
        
        # Store episode data
        episode_data = {
            'episode': episode,
            'moves': episode_moves,
            'scores': episode_scores,
            'cooperation_rates': episode_cooperation_rates,
            'total_scores': {agent.name: agent.episode_score for agent in agents}
        }
        all_episode_data.append(episode_data)
        
        # Update histories
        for agent_name in agent_move_history:
            agent_move_history[agent_name].extend(episode_moves[agent_name])
            agent_score_history[agent_name].extend(episode_scores[agent_name])
        cooperation_rate_history.extend(episode_cooperation_rates)
    
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
    
    # Full results
    results = {
        'summary': summary,
        'episode_data': all_episode_data,
        'round_data': all_rounds_data
    }
    
    return results


def run_chris_huyck_experiments():
    """Run all 8 core scenarios requested by Chris Huyck."""
    
    print("="*80)
    print("CHRIS HUYCK'S CORE SCENARIOS")
    print("3-Person Tragic Valley vs Reciprocity Hill Experiments")
    print("="*80)
    
    # Configuration
    num_episodes = 50
    rounds_per_episode = 100
    
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"3-Person_Tragic_vs_Reciprocity/results/chris_huyck_{timestamp}"
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
        print(f"  TFT cooperation rates:")
        for agent_name, coop_rate in summary['cooperation_rates'].items():
            if 'TFT' in agent_name:
                print(f"    {agent_name}: {coop_rate:.2%}")
        print(f"  Average scores per round:")
        for agent_name, avg_score in summary['average_scores_per_round'].items():
            print(f"    {agent_name}: {avg_score:.2f}")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*80)
    
    return results_dir


if __name__ == "__main__":
    results_dir = run_chris_huyck_experiments()
    print(f"\nExperiments complete! Results directory: {results_dir}")
