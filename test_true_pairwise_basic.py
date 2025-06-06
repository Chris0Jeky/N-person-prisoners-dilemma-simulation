#!/usr/bin/env python3
"""
Basic test to verify true pairwise implementation works.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from npdl.core.true_pairwise import (
        OpponentSpecificMemory, TruePairwiseTFT, TruePairwiseEnvironment
    )
    print("✓ Core true_pairwise imports successful")
    
    from npdl.core.true_pairwise_adapter import (
        TruePairwiseAgentAdapter, create_true_pairwise_agent
    )
    print("✓ Adapter imports successful")
    
    print("\nTesting basic functionality...")
    
    # Create simple agents
    agent1 = TruePairwiseTFT("agent1")
    agent2 = TruePairwiseTFT("agent2")
    print("✓ Created TruePairwiseTFT agents")
    
    # Test opponent-specific decision
    action = agent1.choose_action_for_opponent("agent2", 0)
    print(f"✓ Agent1 chose action '{action}' for agent2")
    
    # Test memory update
    agent1.update_memory("agent2", "cooperate", "cooperate", 3)
    print("✓ Updated memory successfully")
    
    # Test memory retrieval
    memory = agent1.get_opponent_memory("agent2")
    print(f"✓ Retrieved memory: coop_rate={memory.get_cooperation_rate()}")
    
    # Test environment
    env = TruePairwiseEnvironment([agent1, agent2], rounds_per_episode=5)
    print("✓ Created TruePairwiseEnvironment")
    
    # Test single game
    result = env.play_single_game("agent1", "agent2", 0)
    print(f"✓ Played single game: {result['action1']} vs {result['action2']}")
    
    # Test agent creation through adapter
    config = {'type': 'tit_for_tat', 'id': 'test_agent'}
    adapted_agent = create_true_pairwise_agent(config)
    print(f"✓ Created adapted agent: {adapted_agent.agent_id}")
    
    print("\n✅ All basic tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)