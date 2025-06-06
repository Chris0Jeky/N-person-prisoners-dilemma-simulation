"""
Demonstration Experiment: Clear Proof of Reciprocity Hill vs Tragic Valley
==========================================================================
A focused experiment showing how pairwise maintains cooperation (reciprocity hill)
while N-person falls into defection (tragic valley).
"""

from unified_simulation import (
    Agent, UnifiedSimulation, InteractionMode, Strategy, Action
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import json


def run_detailed_simulation(agents, mode, episodes=20, rounds_per_episode=10):
    """Run simulation and track detailed round-by-round data."""
    sim = UnifiedSimulation(agents, mode)
    
    # Track cooperation per round across all episodes
    all_rounds_data = []
    tft_cooperation_by_round = []
    
    for episode in range(episodes):
        # Run episode
        for round_num in range(rounds_per_episode):
            round_result = sim._run_round(round_num, verbose=False)
            
            # Calculate TFT cooperation rate for this round
            if mode == InteractionMode.PAIRWISE:
                # For pairwise, look at interactions
                tft_coop_count = 0
                tft_total = 0
                for interaction in round_result.get('interactions', []):
                    if 'TFT' in interaction['agent1']:
                        tft_total += 1
                        if interaction['action1'] == 'C':
                            tft_coop_count += 1
                    if 'TFT' in interaction['agent2']:
                        tft_total += 1
                        if interaction['action2'] == 'C':
                            tft_coop_count += 1
                tft_coop_rate = tft_coop_count / tft_total if tft_total > 0 else 0
            else:
                # For N-person
                actions = round_result.get('actions', {})
                tft_actions = [(name, action) for name, action in actions.items() if 'TFT' in name]
                tft_coop_rate = sum(1 for _, action in tft_actions if action == 'C') / len(tft_actions) if tft_actions else 0
            
            all_rounds_data.append({
                'episode': episode,
                'round': round_num,
                'tft_cooperation': tft_coop_rate,
                'total_round': episode * rounds_per_episode + round_num
            })
        
        # Reset for next episode
        sim.reset_episode()
    
    return all_rounds_data


def create_demonstration():
    """Create the main demonstration of reciprocity hill vs tragic valley."""
    
    print("="*80)
    print("DEMONSTRATION: Reciprocity Hill vs Tragic Valley")
    print("="*80)
    
    # Scenario: 2 TFT + 3 Always Defect with 10% exploration
    print("\nScenario: 2 Tit-for-Tat + 3 Always Defect (10% exploration)")
    print("This demonstrates how cooperation evolves differently in the two modes.")
    
    # Create agents for both simulations
    exploration_rate = 0.1
    
    # Pairwise agents
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, exploration_rate),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, exploration_rate),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, exploration_rate),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, exploration_rate),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, exploration_rate)
    ]
    
    # N-Person agents
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, exploration_rate),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, exploration_rate),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, exploration_rate),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, exploration_rate),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, exploration_rate)
    ]
    
    # Run simulations
    print("\nRunning pairwise simulation...")
    pw_data = run_detailed_simulation(agents_pw, InteractionMode.PAIRWISE, 
                                     episodes=20, rounds_per_episode=10)
    
    print("Running N-person simulation...")
    np_data = run_detailed_simulation(agents_np, InteractionMode.N_PERSON,
                                     episodes=20, rounds_per_episode=10)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data for plotting
    pw_rounds = [d['total_round'] for d in pw_data]
    pw_coop = [d['tft_cooperation'] for d in pw_data]
    
    np_rounds = [d['total_round'] for d in np_data]
    np_coop = [d['tft_cooperation'] for d in np_data]
    
    # Plot 1: Raw cooperation rates
    ax1.plot(pw_rounds, pw_coop, 'b-', alpha=0.5, linewidth=1, label='Pairwise (raw)')
    ax1.plot(np_rounds, np_coop, 'r-', alpha=0.5, linewidth=1, label='N-Person (raw)')
    
    # Add moving average
    window = 10
    pw_smooth = np.convolve(pw_coop, np.ones(window)/window, mode='valid')
    np_smooth = np.convolve(np_coop, np.ones(window)/window, mode='valid')
    
    ax1.plot(pw_rounds[window-1:], pw_smooth, 'b-', linewidth=3, label='Pairwise (smoothed)')
    ax1.plot(np_rounds[window-1:], np_smooth, 'r-', linewidth=3, label='N-Person (smoothed)')
    
    # Add shaded regions for episodes
    for i in range(0, 200, 20):
        if (i // 20) % 2 == 0:
            ax1.axvspan(i, i+10, alpha=0.1, color='gray')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('TFT Cooperation Rate')
    ax1.set_title('Evolution of Cooperation: Pairwise vs N-Person')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Episode averages
    episodes = list(range(20))
    pw_episode_avg = []
    np_episode_avg = []
    
    for ep in episodes:
        ep_pw = [d['tft_cooperation'] for d in pw_data if d['episode'] == ep]
        ep_np = [d['tft_cooperation'] for d in np_data if d['episode'] == ep]
        pw_episode_avg.append(np.mean(ep_pw))
        np_episode_avg.append(np.mean(ep_np))
    
    ax2.plot(episodes, pw_episode_avg, 'bo-', linewidth=2, markersize=8, 
             label='Pairwise', markerfacecolor='lightblue')
    ax2.plot(episodes, np_episode_avg, 'ro-', linewidth=2, markersize=8,
             label='N-Person', markerfacecolor='lightcoral')
    
    # Add trend lines
    z_pw = np.polyfit(episodes, pw_episode_avg, 1)
    p_pw = np.poly1d(z_pw)
    z_np = np.polyfit(episodes, np_episode_avg, 1)
    p_np = np.poly1d(z_np)
    
    ax2.plot(episodes, p_pw(episodes), "b--", alpha=0.8, linewidth=2)
    ax2.plot(episodes, p_np(episodes), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average TFT Cooperation Rate')
    ax2.set_title('Episode-Average Cooperation Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Plot 3: Conceptual diagram
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Reciprocity Hill (Pairwise)
    hill_x = np.linspace(0.5, 4.5, 100)
    hill_y = 7 + 2 * np.exp(-((hill_x - 2.5)**2) / 0.5)
    ax3.fill_between(hill_x, 5, hill_y, alpha=0.3, color='blue')
    ax3.plot(hill_x, hill_y, 'b-', linewidth=3)
    ax3.text(2.5, 8.5, 'Reciprocity Hill\n(Pairwise)', ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Tragic Valley (N-Person)
    valley_x = np.linspace(5.5, 9.5, 100)
    valley_y = 3 - 1.5 * np.exp(-((valley_x - 7.5)**2) / 0.5)
    ax3.fill_between(valley_x, 5, valley_y, alpha=0.3, color='red')
    ax3.plot(valley_x, valley_y, 'r-', linewidth=3)
    ax3.text(7.5, 2, 'Tragic Valley\n(N-Person)', ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Add arrows showing dynamics
    ax3.arrow(2.5, 6, 0, 1.5, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    ax3.arrow(7.5, 4, 0, -1.5, head_width=0.2, head_length=0.2, fc='red', ec='red')
    
    ax3.text(2.5, 5.5, 'Cooperation\nstabilizes', ha='center', fontsize=10)
    ax3.text(7.5, 4.5, 'Defection\ndominates', ha='center', fontsize=10)
    
    ax3.set_title('Conceptual Model: Reciprocity Hill vs Tragic Valley')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('3-Person_Tragic_vs_Reciprocity/results/demonstration_plot.png', dpi=150)
    plt.close()
    
    # Calculate and print statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Overall averages
    pw_overall = np.mean(pw_coop)
    np_overall = np.mean(np_coop)
    
    print(f"\nOverall TFT Cooperation Rates:")
    print(f"  Pairwise: {pw_overall:.2%}")
    print(f"  N-Person: {np_overall:.2%}")
    print(f"  Difference: {pw_overall - np_overall:.2%}")
    
    # First vs Last episode comparison
    first_pw = np.mean([d['tft_cooperation'] for d in pw_data if d['episode'] == 0])
    first_np = np.mean([d['tft_cooperation'] for d in np_data if d['episode'] == 0])
    last_pw = np.mean([d['tft_cooperation'] for d in pw_data if d['episode'] == 19])
    last_np = np.mean([d['tft_cooperation'] for d in np_data if d['episode'] == 19])
    
    print(f"\nFirst Episode TFT Cooperation:")
    print(f"  Pairwise: {first_pw:.2%}")
    print(f"  N-Person: {first_np:.2%}")
    
    print(f"\nLast Episode TFT Cooperation:")
    print(f"  Pairwise: {last_pw:.2%}")
    print(f"  N-Person: {last_np:.2%}")
    
    print(f"\nChange over time:")
    print(f"  Pairwise: {last_pw - first_pw:+.2%}")
    print(f"  N-Person: {last_np - first_np:+.2%}")
    
    # Detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    print("\nKey Finding:")
    print("In the PAIRWISE model, TFT agents can maintain selective cooperation")
    print("with each other while defecting against Always Defect agents.")
    print("This creates a 'Reciprocity Hill' where cooperation is stable.")
    
    print("\nIn the N-PERSON model, TFT agents base decisions on the majority.")
    print("With 3 defectors out of 5 agents, TFTs are pushed to defect,")
    print("creating a 'Tragic Valley' where cooperation cannot recover.")
    
    print("\nThe 10% exploration rate occasionally allows cooperation attempts,")
    print("but in N-Person these are not targeted and quickly fail.")
    print("In Pairwise, exploration can help re-establish mutual cooperation.")
    
    # Save detailed results
    results = {
        'scenario': '2 TFT + 3 Defect with 10% exploration',
        'pairwise': {
            'overall_cooperation': pw_overall,
            'first_episode': first_pw,
            'last_episode': last_pw,
            'trend': 'stable' if abs(last_pw - first_pw) < 0.1 else ('increasing' if last_pw > first_pw else 'decreasing')
        },
        'n_person': {
            'overall_cooperation': np_overall,
            'first_episode': first_np,
            'last_episode': last_np,
            'trend': 'stable' if abs(last_np - first_np) < 0.1 else ('increasing' if last_np > first_np else 'decreasing')
        },
        'raw_data': {
            'pairwise': pw_data,
            'n_person': np_data
        }
    }
    
    with open('3-Person_Tragic_vs_Reciprocity/results/demonstration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("Results saved to 3-Person_Tragic_vs_Reciprocity/results/")
    print("="*60)


if __name__ == "__main__":
    create_demonstration()
