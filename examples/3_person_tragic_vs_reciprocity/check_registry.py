#!/usr/bin/env python3
"""
Check if agent registry is working properly
"""

import sys
import os

# Add npd_simulator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'npd_simulator'))

from agents import AgentRegistry

print("Checking Agent Registry...")
print("=" * 50)

# List all registered agents
agents = AgentRegistry.list_agents()
print(f"Total registered agents: {len(agents)}")
print("\nRegistered agents:")
for agent in agents:
    metadata = AgentRegistry.get_metadata(agent)
    print(f"  - {agent}: {metadata['description']}")

print("\nCategories:")
categories = AgentRegistry.get_categories()
for cat in categories:
    agents_in_cat = AgentRegistry.list_agents(category=cat)
    print(f"  - {cat}: {agents_in_cat}")

# Test creating an agent
print("\nTesting agent creation:")
try:
    agent = AgentRegistry.create("TFT", agent_id=0)
    print(f"✓ Successfully created TFT agent: {agent}")
except Exception as e:
    print(f"✗ Failed to create TFT agent: {e}")

print("\nDone!")