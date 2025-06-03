"""
Quick test to verify the Chris Huyck experiments are working
"""

from unified_simulation import Agent, UnifiedSimulation, InteractionMode, Strategy, save_results
from datetime import datetime
import os

# Create a simple test
agents = [
    Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
    Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
    Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
]

# Run in pairwise mode
sim_pw = UnifiedSimulation(agents, InteractionMode.PAIRWISE)
print("Running pairwise simulation...")
results_pw = sim_pw.run_simulation(episodes=2, rounds_per_episode=5, verbose=False)

# Run in n-person mode
agents_np = [
    Agent(0, "TFT1", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
    Agent(1, "TFT2", Strategy.TIT_FOR_TAT, exploration_rate=0.0),
    Agent(2, "AllD", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
]
sim_np = UnifiedSimulation(agents_np, InteractionMode.N_PERSON)
print("Running n-person simulation...")
results_np = sim_np.run_simulation(episodes=2, rounds_per_episode=5, verbose=False)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/test_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

save_results(results_pw, f"{results_dir}/test_pairwise.json")
save_results(results_np, f"{results_dir}/test_nperson.json")

print(f"\nTest complete! Results saved to: {results_dir}")
print("\nPairwise final scores:", results_pw['final_scores'])
print("N-Person final scores:", results_np['final_scores'])
