"""
Run N-Person IPD simulations with 3 agents.
"""

from simple_npd import NPDAgent, Strategy, NPrisonersDilemmaGame, NPersonSimulation


def run_example_1():
    """Run a basic example with no exploration - classic strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Classic Strategies (No Exploration)")
    print("="*80)
    
    # Create agents
    alice = NPDAgent("Alice", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
    bob = NPDAgent("Bob", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    charlie = NPDAgent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    
    # Create game and simulation
    game = NPrisonersDilemmaGame(N=3)
    sim = NPersonSimulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=5)


def run_example_2():
    """Run an example with exploration - strategies can occasionally deviate."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Strategies with Exploration")
    print("="*80)
    
    # Create agents with exploration
    alice = NPDAgent("Alice", Strategy.ALWAYS_COOPERATE, exploration_rate=0.1)  # 10% exploration
    bob = NPDAgent("Bob", Strategy.ALWAYS_DEFECT, exploration_rate=0.2)  # 20% exploration
    charlie = NPDAgent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.15)  # 15% exploration
    
    # Create game and simulation
    game = NPrisonersDilemmaGame(N=3)
    sim = NPersonSimulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=5)


def run_example_3():
    """Run all Tit-for-Tat agents with different exploration rates."""
    print("\n" + "="*80)
    print("EXAMPLE 3: All Tit-for-Tat with Different Exploration Rates")
    print("="*80)
    
    # Create agents
    alice = NPDAgent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    bob = NPDAgent("Bob", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
    charlie = NPDAgent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.2)
    
    # Create game and simulation
    game = NPrisonersDilemmaGame(N=3)
    sim = NPersonSimulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=10)


def run_example_4():
    """Run with modified payoff parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Modified Payoff Structure (Higher Cooperation Reward)")
    print("="*80)
    
    # Create agents
    alice = NPDAgent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    bob = NPDAgent("Bob", Strategy.ALWAYS_COOPERATE, exploration_rate=0.1)
    charlie = NPDAgent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    
    # Create game with modified payoffs (higher cooperation reward)
    game = NPrisonersDilemmaGame(N=3, R=4.0, S=0.5, T=4.5, P=1.0)
    sim = NPersonSimulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=8)


def run_example_5():
    """Run a longer simulation to see emergent patterns."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Longer Simulation - Emergent Patterns")
    print("="*80)
    
    # Create agents
    alice = NPDAgent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    bob = NPDAgent("Bob", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    charlie = NPDAgent("Charlie", Strategy.ALWAYS_DEFECT, exploration_rate=0.15)
    
    # Create game and simulation
    game = NPrisonersDilemmaGame(N=3)
    sim = NPersonSimulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=15)


def run_custom_simulation():
    """Template for custom simulations - modify as needed."""
    print("\n" + "="*80)
    print("CUSTOM N-PERSON SIMULATION")
    print("="*80)
    
    # Customize your agents here
    agent1 = NPDAgent("Player1", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    agent2 = NPDAgent("Player2", Strategy.ALWAYS_COOPERATE, exploration_rate=0.2)
    agent3 = NPDAgent("Player3", Strategy.ALWAYS_DEFECT, exploration_rate=0.1)
    
    # Customize payoff parameters
    game = NPrisonersDilemmaGame(N=3, R=3.0, S=0.0, T=5.0, P=1.0)
    
    # Create and run simulation
    sim = NPersonSimulation([agent1, agent2, agent3], game)
    sim.run_simulation(num_rounds=10)


if __name__ == "__main__":
    # Run all examples
    run_example_1()
    input("\nPress Enter to continue to Example 2...")
    
    run_example_2()
    input("\nPress Enter to continue to Example 3...")
    
    run_example_3()
    input("\nPress Enter to continue to Example 4...")
    
    run_example_4()
    input("\nPress Enter to continue to Example 5...")
    
    run_example_5()
    
    # Uncomment to run custom simulation
    # input("\nPress Enter to run custom simulation...")
    # run_custom_simulation()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
