"""
Demonstration of TFT ecosystem-aware behavior.

This script shows how TFT agents respond to cooperation proportions
in their neighborhood rather than just mimicking random neighbors.
"""

from npdl.core.agents import Agent
from npdl.core.environment import Environment
import networkx as nx


def demonstrate_tft_behavior():
    """Demonstrate TFT behavior in different cooperation environments."""
    
    print("=" * 70)
    print("TFT ECOSYSTEM-AWARE BEHAVIOR DEMONSTRATION")
    print("=" * 70)
    
    # Scenario 1: Cooperative neighborhood
    print("\n1. COOPERATIVE NEIGHBORHOOD (75% cooperation)")
    print("-" * 50)
    
    agents_coop = [
        Agent(agent_id="TFT", strategy="tit_for_tat", cooperation_threshold=0.5),
        Agent(agent_id="Coop1", strategy="always_cooperate"),
        Agent(agent_id="Coop2", strategy="always_cooperate"),
        Agent(agent_id="Coop3", strategy="always_cooperate"),
        Agent(agent_id="Defect1", strategy="always_defect"),
    ]
    
    network = nx.complete_graph(5)
    env_coop = Environment(agents_coop, network, interaction_mode="neighborhood")
    
    print("Initial round to establish history...")
    env_coop.run(rounds=1)
    
    tft = agents_coop[0]
    last_round = tft.memory[-1]
    neighbor_moves = last_round['neighbor_moves']
    
    coop_count = sum(1 for move in neighbor_moves.values() if move == "cooperate")
    total_neighbors = len(neighbor_moves)
    coop_proportion = coop_count / total_neighbors
    
    print(f"TFT observed: {coop_count}/{total_neighbors} cooperators ({coop_proportion:.1%} cooperation)")
    print(f"Neighbor moves: {neighbor_moves}")
    
    print("\nSecond round - TFT decision based on cooperation proportion...")
    env_coop.run(rounds=1)
    
    tft_decision = tft.memory[-1]['my_move']
    print(f"TFT chose to: {tft_decision.upper()}")
    print(f"(Threshold: {tft.strategy.cooperation_threshold}, Observed: {coop_proportion:.2f})")
    
    # Scenario 2: Defective neighborhood
    print("\n\n2. DEFECTIVE NEIGHBORHOOD (25% cooperation)")
    print("-" * 50)
    
    agents_defect = [
        Agent(agent_id="TFT", strategy="tit_for_tat", cooperation_threshold=0.5),
        Agent(agent_id="Defect1", strategy="always_defect"),
        Agent(agent_id="Defect2", strategy="always_defect"),
        Agent(agent_id="Defect3", strategy="always_defect"),
        Agent(agent_id="Coop1", strategy="always_cooperate"),
    ]
    
    env_defect = Environment(agents_defect, nx.complete_graph(5), interaction_mode="neighborhood")
    
    print("Initial round to establish history...")
    env_defect.run(rounds=1)
    
    tft2 = agents_defect[0]
    last_round2 = tft2.memory[-1]
    neighbor_moves2 = last_round2['neighbor_moves']
    
    coop_count2 = sum(1 for move in neighbor_moves2.values() if move == "cooperate")
    total_neighbors2 = len(neighbor_moves2)
    coop_proportion2 = coop_count2 / total_neighbors2
    
    print(f"TFT observed: {coop_count2}/{total_neighbors2} cooperators ({coop_proportion2:.1%} cooperation)")
    print(f"Neighbor moves: {neighbor_moves2}")
    
    print("\nSecond round - TFT decision based on cooperation proportion...")
    env_defect.run(rounds=1)
    
    tft_decision2 = tft2.memory[-1]['my_move']
    print(f"TFT chose to: {tft_decision2.upper()}")
    print(f"(Threshold: {tft2.strategy.cooperation_threshold}, Observed: {coop_proportion2:.2f})")
    
    # Scenario 3: Network topology matters
    print("\n\n3. NETWORK TOPOLOGY COMPARISON")
    print("-" * 50)
    
    # Same agents for both networks
    agent_ids = ["TFT", "Coop1", "Coop2", "Defect1", "Defect2", "Defect3"]
    
    # Fully connected
    print("\nFully Connected Network:")
    agents_fc = [
        Agent(agent_id="TFT", strategy="tit_for_tat"),
        Agent(agent_id="Coop1", strategy="always_cooperate"),
        Agent(agent_id="Coop2", strategy="always_cooperate"),
        Agent(agent_id="Defect1", strategy="always_defect"),
        Agent(agent_id="Defect2", strategy="always_defect"),
        Agent(agent_id="Defect3", strategy="always_defect"),
    ]
    
    env_fc = Environment(agents_fc, nx.complete_graph(6), interaction_mode="neighborhood")
    env_fc.run(rounds=2)
    
    tft_fc = agents_fc[0]
    fc_neighbors = tft_fc.memory[-2]['neighbor_moves']
    fc_coop_rate = sum(1 for m in fc_neighbors.values() if m == "cooperate") / len(fc_neighbors)
    
    print(f"  TFT sees {len(fc_neighbors)} neighbors")
    print(f"  Cooperation rate: {fc_coop_rate:.1%}")
    print(f"  TFT decision: {tft_fc.memory[-1]['my_move']}")
    
    # Small world - TFT only connected to defectors
    print("\nSmall World Network (TFT connected only to defectors):")
    agents_sw = [
        Agent(agent_id="TFT", strategy="tit_for_tat"),
        Agent(agent_id="Coop1", strategy="always_cooperate"),
        Agent(agent_id="Coop2", strategy="always_cooperate"),
        Agent(agent_id="Defect1", strategy="always_defect"),
        Agent(agent_id="Defect2", strategy="always_defect"),
        Agent(agent_id="Defect3", strategy="always_defect"),
    ]
    
    network_sw = nx.Graph()
    network_sw.add_edges_from([
        (0, 3),  # TFT to Defect1
        (0, 4),  # TFT to Defect2
        (0, 5),  # TFT to Defect3
        (1, 2),  # Coop1 to Coop2
        (3, 4),  # Defect1 to Defect2
        (4, 5),  # Defect2 to Defect3
    ])
    
    env_sw = Environment(agents_sw, network_sw, interaction_mode="neighborhood")
    env_sw.run(rounds=2)
    
    tft_sw = agents_sw[0]
    sw_neighbors = tft_sw.memory[-2]['neighbor_moves']
    sw_coop_rate = sum(1 for m in sw_neighbors.values() if m == "cooperate") / len(sw_neighbors)
    
    print(f"  TFT sees {len(sw_neighbors)} neighbors")
    print(f"  Cooperation rate: {sw_coop_rate:.1%}")
    print(f"  TFT decision: {tft_sw.memory[-1]['my_move']}")
    
    # Scenario 4: Proportional TFT
    print("\n\n4. PROPORTIONAL TFT (Probabilistic)")
    print("-" * 50)
    
    print("Running 20 trials with 60% cooperation rate...")
    
    cooperation_count = 0
    for i in range(20):
        agents_prob = [
            Agent(agent_id="PTFT", strategy="proportional_tit_for_tat"),
            Agent(agent_id="Coop1", strategy="always_cooperate"),
            Agent(agent_id="Coop2", strategy="always_cooperate"),
            Agent(agent_id="Coop3", strategy="always_cooperate"),
            Agent(agent_id="Defect1", strategy="always_defect"),
            Agent(agent_id="Defect2", strategy="always_defect"),
        ]
        
        env_prob = Environment(agents_prob, nx.complete_graph(6), interaction_mode="neighborhood")
        env_prob.run(rounds=2)
        
        ptft = agents_prob[0]
        if ptft.memory[-1]['my_move'] == "cooperate":
            cooperation_count += 1
    
    print(f"PTFT cooperated {cooperation_count}/20 times ({cooperation_count/20:.1%})")
    print("Expected: ~60% (matching neighborhood cooperation rate)")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tft_behavior()