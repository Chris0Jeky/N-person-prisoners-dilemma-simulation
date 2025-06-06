#!/usr/bin/env python3
"""Test just the imports"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing step by step...")

try:
    print("1. Importing utils...")
    from npdl.core.utils import get_pairwise_payoffs
    print("✓ utils imported")
except Exception as e:
    print(f"✗ utils failed: {e}")
    
try:
    print("2. Importing agents...")
    from npdl.core.agents import Agent
    print("✓ agents imported")
except Exception as e:
    print(f"✗ agents failed: {e}")

try:
    print("3. Importing true_pairwise...")
    from npdl.core import true_pairwise
    print("✓ true_pairwise imported")
except Exception as e:
    print(f"✗ true_pairwise failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Importing specific classes...")
    from npdl.core.true_pairwise import TruePairwiseTFT
    print("✓ TruePairwiseTFT imported")
except Exception as e:
    print(f"✗ TruePairwiseTFT failed: {e}")