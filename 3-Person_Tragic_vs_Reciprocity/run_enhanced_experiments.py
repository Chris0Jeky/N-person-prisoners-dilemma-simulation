#!/usr/bin/env python3
"""
Run the enhanced Q-learning experiments to test exploitation improvements.
"""

from enhanced_experiment_runner import main

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED Q-LEARNING EXPLOITATION EXPERIMENTS")
    print("Testing four improvements to Q-learning exploitation")
    print("=" * 80)
    print()
    
    # Run the enhanced Q-learning experiments
    results = main()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE!")
    print("Check the generated JSON file for detailed results.")
    print("=" * 80)
