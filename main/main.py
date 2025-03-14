# main.py
import random
from agents import Agent  # Import the Agent class
from environment import Environment  # Import the Environment class
from utils import create_payoff_matrix

if __name__ == "__main__":
    num_agents = 20
    num_rounds = 500
    # Example of using different network types:
    # network_type = "fully_connected"
    # network_type = "random"
    # network_params = {"probability": 0.2}  # 20% chance of connection
    # network_type = "small_world"
    # network_params = {"k": 4, "beta": 0.3}  # 4 neighbors, 30% rewiring
    network_type = "scale_free"
    network_params = {"m": 3} #each new node creates 3 edges

    # Create agents
    agents = [Agent(agent_id=i, strategy="random") for i in range(num_agents - 6)]
    agents.append(Agent(agent_id=num_agents - 4, strategy="q_learning", epsilon=0.7))
    agents.append(Agent(agent_id=num_agents - 3, strategy="q_learning", epsilon=0.5))
    agents.append(Agent(agent_id=num_agents - 2, strategy="tit_for_tat"))
    agents.append(Agent(agent_id=num_agents - 5, strategy="q_learning", epsilon=0.4))
    agents.append(Agent(agent_id=num_agents - 6, strategy="q_learning", epsilon=0.2))
    agents.append(Agent(agent_id=num_agents - 7, strategy="q_learning", epsilon=0.9))

    # Create the payoff matrix
    payoff_matrix = create_payoff_matrix(num_agents)

    # Create the environment
    env = Environment(agents, payoff_matrix, network_type=network_type, network_params=network_params)

    # Run the simulation
    results = env.run_simulation(num_rounds)

    # Basic results printing (replace with proper logging/analysis)
    #for round_data in results:
    #    print(f"Round: {round_data['round']}")
    #    print(f"  Moves: {round_data['moves']}")
    #    print(f"  Payoffs: {round_data['payoffs']}")

    for agent in agents:
      print(agent.agent_id, agent.strategy, agent.score, agent.q_values)