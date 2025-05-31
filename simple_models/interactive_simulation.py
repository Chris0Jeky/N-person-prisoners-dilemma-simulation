"""
Interactive IPD Simulation
=========================
Configure and run your own IPD simulation with custom parameters.
"""

from simple_ipd import Agent, Strategy, PrisonersDilemmaGame, Simulation


def get_strategy_choice(agent_name: str) -> Strategy:
    """Get strategy choice for an agent."""
    print(f"\nChoose strategy for {agent_name}:")
    print("1. Always Cooperate")
    print("2. Always Defect")
    print("3. Tit-for-Tat")
    
    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if choice == 1:
                return Strategy.ALWAYS_COOPERATE
            elif choice == 2:
                return Strategy.ALWAYS_DEFECT
            elif choice == 3:
                return Strategy.TIT_FOR_TAT
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_exploration_rate(agent_name: str) -> float:
    """Get exploration rate for an agent."""
    while True:
        try:
            rate = float(input(f"Enter exploration rate for {agent_name} (0.0-1.0, e.g., 0.1 for 10%): "))
            if 0.0 <= rate <= 1.0:
                return rate
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a decimal number.")


def run_interactive():
    """Run an interactive IPD simulation."""
    print("="*60)
    print("Interactive Iterated Prisoner's Dilemma Simulation")
    print("="*60)
    
    # Get number of rounds
    while True:
        try:
            num_rounds = int(input("\nHow many rounds to play? (e.g., 5, 10, 20): "))
            if num_rounds > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Configure three agents
    agents = []
    agent_names = ["Alice", "Bob", "Charlie"]
    
    for name in agent_names:
        print(f"\n--- Configuring {name} ---")
        strategy = get_strategy_choice(name)
        exploration_rate = get_exploration_rate(name)
        agents.append(Agent(name, strategy, exploration_rate))
    
    # Ask about custom payoffs
    use_custom = input("\nUse custom payoff matrix? (y/n, default=n): ").lower() == 'y'
    
    if use_custom:
        print("\nEnter payoff values (press Enter for defaults):")
        try:
            reward = int(input("Reward (both cooperate, default=3): ") or "3")
            temptation = int(input("Temptation (I defect, you cooperate, default=5): ") or "5")
            punishment = int(input("Punishment (both defect, default=1): ") or "1")
            sucker = int(input("Sucker (I cooperate, you defect, default=0): ") or "0")
            game = PrisonersDilemmaGame(reward, temptation, punishment, sucker)
        except ValueError:
            print("Invalid input. Using default payoffs.")
            game = PrisonersDilemmaGame()
    else:
        game = PrisonersDilemmaGame()
    
    # Create and run simulation
    sim = Simulation(agents, game)
    sim.run_simulation(num_rounds)
    
    # Ask if they want to run another
    if input("\nRun another simulation? (y/n): ").lower() == 'y':
        run_interactive()


if __name__ == "__main__":
    run_interactive()
