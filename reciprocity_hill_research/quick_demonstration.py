"""
Quick Demonstration: Reciprocity Hill vs Tragic Valley
======================================================
A simplified demonstration that clearly shows the difference.
"""

from unified_simulation import (
    Agent, UnifiedSimulation, InteractionMode, Strategy
)
import os


def run_quick_demo():
    """Run a quick demonstration of the key phenomenon."""
    
    print("="*80)
    print("RECIPROCITY HILL VS TRAGIC VALLEY - QUICK DEMONSTRATION")
    print("="*80)
    print("\nScenario: 2 Tit-for-Tat + 3 Always Defect agents with 10% exploration")
    print("Episodes: 5, Rounds per episode: 10")
    
    # Create results directory
    os.makedirs('reciprocity_hill_research/results', exist_ok=True)
    
    # Run Pairwise simulation
    print("\n" + "-"*60)
    print("PAIRWISE INTERACTION MODE")
    print("-"*60)
    
    agents_pw = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    
    sim_pw = UnifiedSimulation(agents_pw, InteractionMode.PAIRWISE)
    
    # Track TFT cooperation
    pw_tft_coop = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        for round_num in range(10):
            sim_pw._run_round(round_num, verbose=False)
        
        # Calculate TFT cooperation rate this episode
        tft_coop_rates = []
        for agent in agents_pw[:2]:  # First two are TFT
            if agent.my_history:
                coop_rate = sum(1 for a in agent.my_history if a.value == 'C') / len(agent.my_history)
                tft_coop_rates.append(coop_rate)
        
        avg_tft_coop = sum(tft_coop_rates) / len(tft_coop_rates) if tft_coop_rates else 0
        pw_tft_coop.append(avg_tft_coop)
        print(f"  TFT average cooperation: {avg_tft_coop:.1%}")
        
        # Reset for next episode
        sim_pw.reset_episode()
    
    print(f"\nPairwise - Overall TFT cooperation trend: {pw_tft_coop[0]:.1%} → {pw_tft_coop[-1]:.1%}")
    
    # Run N-Person simulation
    print("\n" + "-"*60)
    print("N-PERSON INTERACTION MODE")
    print("-"*60)
    
    agents_np = [
        Agent(0, "TFT-1", Strategy.TIT_FOR_TAT, 0.1),
        Agent(1, "TFT-2", Strategy.TIT_FOR_TAT, 0.1),
        Agent(2, "Defector-1", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(3, "Defector-2", Strategy.ALWAYS_DEFECT, 0.1),
        Agent(4, "Defector-3", Strategy.ALWAYS_DEFECT, 0.1)
    ]
    
    sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
    
    # Track TFT cooperation
    np_tft_coop = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        for round_num in range(10):
            sim_np._run_round(round_num, verbose=False)
        
        # Calculate TFT cooperation rate this episode
        tft_coop_rates = []
        for agent in agents_np[:2]:  # First two are TFT
            if agent.my_history:
                coop_rate = sum(1 for a in agent.my_history if a.value == 'C') / len(agent.my_history)
                tft_coop_rates.append(coop_rate)
        
        avg_tft_coop = sum(tft_coop_rates) / len(tft_coop_rates) if tft_coop_rates else 0
        np_tft_coop.append(avg_tft_coop)
        print(f"  TFT average cooperation: {avg_tft_coop:.1%}")
        
        # Reset for next episode
        sim_np.reset_episode()
    
    print(f"\nN-Person - Overall TFT cooperation trend: {np_tft_coop[0]:.1%} → {np_tft_coop[-1]:.1%}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    print("\nPAIRWISE (Reciprocity Hill):")
    print(f"  - Initial cooperation: {pw_tft_coop[0]:.1%}")
    print(f"  - Final cooperation: {pw_tft_coop[-1]:.1%}")
    print(f"  - Average: {sum(pw_tft_coop)/len(pw_tft_coop):.1%}")
    print("  → TFT agents can maintain selective cooperation with each other")
    
    print("\nN-PERSON (Tragic Valley):")
    print(f"  - Initial cooperation: {np_tft_coop[0]:.1%}")
    print(f"  - Final cooperation: {np_tft_coop[-1]:.1%}")
    print(f"  - Average: {sum(np_tft_coop)/len(np_tft_coop):.1%}")
    print("  → TFT agents are forced to defect by the majority")
    
    print("\nKEY INSIGHT:")
    print("In pairwise interactions, TFT agents can cooperate with each other while")
    print("defecting against Always Defect agents. In N-person interactions, the")
    print("majority of defectors forces TFT agents into a defection spiral.")
    
    # Save results
    import json
    results = {
        'pairwise_cooperation': pw_tft_coop,
        'nperson_cooperation': np_tft_coop,
        'summary': {
            'pairwise_avg': sum(pw_tft_coop)/len(pw_tft_coop),
            'nperson_avg': sum(np_tft_coop)/len(np_tft_coop),
            'difference': sum(pw_tft_coop)/len(pw_tft_coop) - sum(np_tft_coop)/len(np_tft_coop)
        }
    }
    
    with open('reciprocity_hill_research/results/quick_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to reciprocity_hill_research/results/quick_demo_results.json")


if __name__ == "__main__":
    run_quick_demo()
