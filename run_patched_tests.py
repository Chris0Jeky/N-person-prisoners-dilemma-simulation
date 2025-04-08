#!/usr/bin/env python3
"""
Run the pairwise tests with the patched implementation.

This script applies the runtime patches and then runs the tests.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    print("==================================================")
    print("   Running Pairwise Tests with Runtime Patches")
    print("==================================================")
    
    # Apply patches first
    print("\n[1] Applying runtime patches...")
    import patch_pairwise_runtime
    
    # Then run the tests
    print("\n[2] Running pairwise tests...")
    import test_pairwise
    test_pairwise.main()
    
    print("\n[3] All tests completed!")
    print("==================================================")
    print("If the tests pass, you can apply the permanent fixes")
    print("described in PAIRWISE_FIX_INSTRUCTIONS.md.")
    print("==================================================")

if __name__ == "__main__":
    main()
