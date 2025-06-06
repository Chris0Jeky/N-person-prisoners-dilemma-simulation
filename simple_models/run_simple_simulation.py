"""
Run a simple IPD simulation with 3 agents.
"""

from simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation


def run_example_1():
    """Run a basic example with no exploration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Strategies (No Exploration)")
    print("="*80)
    
    # Create agents
    alice = Agent("Alice", Strategy.ALWAYS_COOPERATE, exploration_rate=0.0)
    bob = Agent("Bob", Strategy.ALWAYS_DEFECT, exploration_rate=0.0)
    charlie = Agent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    
    # Create game and simulation
    game = PrisonersDilemmaGame()
    sim = Simulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=5)


def run_example_2():
    """Run an example with exploration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: With Exploration")
    print("="*80)
    
    # Create agents with exploration
    alice = Agent("Alice", Strategy.ALWAYS_COOPERATE, exploration_rate=0.1)  # 10% exploration
    bob = Agent("Bob", Strategy.ALWAYS_DEFECT, exploration_rate=0.2)  # 20% exploration
    charlie = Agent("Charlie", Strategy.TIT_FOR_TAT, exploration_rate=0.15)  # 15% exploration
    
    # Create game and simulation
    game = PrisonersDilemmaGame()
    sim = Simulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=5)


def run_example_3():
    """Run a longer simulation to see patterns emerge."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Longer Simulation (10 rounds)")
    print("="*80)
    
    # Create agents
    alice = Agent("Alice", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    bob = Agent("Bob", Strategy.TIT_FOR_TAT, exploration_rate=0.05)
    charlie = Agent("Charlie", Strategy.ALWAYS_DEFECT, exploration_rate=0.1)
    
    # Create game and simulation
    game = PrisonersDilemmaGame()
    sim = Simulation([alice, bob, charlie], game)
    
    # Run simulation
    sim.run_simulation(num_rounds=10)


def run_custom_simulation():
    """Run a custom simulation with user-defined parameters."""
    print("\n" + "="*80)
    print("CUSTOM SIMULATION")
    print("="*80)
    
    # You can customize this function to create your own scenarios
    # Example: Three Tit-for-Tat agents with different exploration rates
    agent1 = Agent("Player1", Strategy.TIT_FOR_TAT, exploration_rate=0.0)
    agent2 = Agent("Player2", Strategy.TIT_FOR_TAT, exploration_rate=0.1)
    agent3 = Agent("Player3", Strategy.TIT_FOR_TAT, exploration_rate=0.2)
    
    # Custom payoff matrix (you can modify these values)
    game = PrisonersDilemmaGame(reward=3, temptation=5, punishment=1, sucker=0)
    
    # Create and run simulation
    sim = Simulation([agent1, agent2, agent3], game)
    sim.run_simulation(num_rounds=7)


if __name__ == "__main__":
    # Run all examples
    run_example_1()
    run_example_2()
    run_example_3()
    
    # Uncomment to run custom simulation
    # run_custom_simulation()
