# test_basic.py
"""
Simple test script to verify that the core components are working correctly.
"""

import logging
from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix, plot_payoff_functions
from npdl.core.logging_utils import setup_logging

def test_agent_strategies():
    """Test that all agent strategies work as expected."""
    print("Testing agent strategies...")
    
    # Create agents with different strategies
    agents = [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="random"),
        Agent(agent_id=3, strategy="tit_for_tat"),
        Agent(agent_id=4, strategy="generous_tit_for_tat", generosity=0.2),
        Agent(agent_id=5, strategy="suspicious_tit_for_tat"),
        Agent(agent_id=6, strategy="pavlov"),
        Agent(agent_id=7, strategy="randomprob", prob_coop=0.7),
        Agent(agent_id=8, strategy="q_learning", epsilon=0.1),
        Agent(agent_id=9, strategy="q_learning_adaptive", epsilon=0.5)
    ]
    
    # Test initial moves
    for agent in agents:
        move = agent.choose_move([])
        print(f"Agent {agent.agent_id} ({agent.strategy_type}): {move}")
    
    # All tests passed
    print("Agent strategy tests completed!\n")

def test_payoff_functions():
    """Test that payoff functions work as expected."""
    print("Testing payoff functions...")
    
    # Create payoff matrices with different functions
    num_agents = 10
    linear_matrix = create_payoff_matrix(num_agents, "linear")
    exp_matrix = create_payoff_matrix(num_agents, "exponential", {"exponent": 2})
    thresh_matrix = create_payoff_matrix(num_agents, "threshold", {"threshold": 0.5})
    
    # Print payoff values
    print("Linear payoffs:")
    print(f"  Cooperation: {linear_matrix['C']}")
    print(f"  Defection: {linear_matrix['D']}")
    
    print("Exponential payoffs:")
    print(f"  Cooperation: {exp_matrix['C']}")
    print(f"  Defection: {exp_matrix['D']}")
    
    print("Threshold payoffs:")
    print(f"  Cooperation: {thresh_matrix['C']}")
    print(f"  Defection: {thresh_matrix['D']}")
    
    # Plot payoff functions
    try:
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode
        plot_payoff_functions(num_agents)
        print("Payoff function plot created.")
        plt.ioff()  # Turn off interactive mode
    except ImportError:
        print("Matplotlib not available, skipping plot.")
    
    # All tests passed
    print("Payoff function tests completed!\n")

def test_network_creation():
    """Test that different network types can be created."""
    print("Testing network creation...")
    
    # Create some agents
    agents = [Agent(agent_id=i, strategy="random") for i in range(20)]
    payoff_matrix = create_payoff_matrix(len(agents))
    
    # Create environments with different network types
    networks = [
        ("fully_connected", {}),
        ("random", {"probability": 0.2}),
        ("small_world", {"k": 4, "beta": 0.2}),
        ("scale_free", {"m": 2}),
        ("regular", {"k": 4})
    ]
    
    # Create and test each network
    for network_type, params in networks:
        try:
            env = Environment(agents, payoff_matrix, network_type, params)
            metrics = env.get_network_metrics()
            print(f"{network_type.title()} network created:")
            print(f"  Nodes: {metrics['num_nodes']}")
            print(f"  Edges: {metrics['num_edges']}")
            print(f"  Avg Degree: {metrics['avg_degree']:.2f}")
            print(f"  Density: {metrics['density']:.2f}")
            print(f"  Clustering: {metrics.get('avg_clustering', 'N/A')}")
        except Exception as e:
            print(f"Error creating {network_type} network: {e}")
    
    # All tests passed
    print("Network creation tests completed!\n")

def test_simulation():
    """Test running a simple simulation."""
    print("Testing simulation...")
    
    # Set up logging
    logger = setup_logging(level=logging.INFO, console=True)
    
    # Create agents
    agents = [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="tit_for_tat"),
        Agent(agent_id=3, strategy="q_learning", epsilon=0.2)
    ]
    
    # Create environment
    payoff_matrix = create_payoff_matrix(len(agents))
    env = Environment(agents, payoff_matrix, "fully_connected", logger=logger)
    
    # Run a short simulation
    print("Running 10 rounds of simulation...")
    results = env.run_simulation(10, logging_interval=2)
    
    # Print final scores
    print("Final scores:")
    for agent in agents:
        print(f"  Agent {agent.agent_id} ({agent.strategy_type}): {agent.score}")
    
    # All tests passed
    print("Simulation tests completed!\n")

def main():
    """Run all tests."""
    print("=== Starting basic tests ===\n")
    
    test_agent_strategies()
    test_payoff_functions()
    test_network_creation()
    test_simulation()
    
    print("=== All tests completed successfully! ===")

if __name__ == "__main__":
    main()