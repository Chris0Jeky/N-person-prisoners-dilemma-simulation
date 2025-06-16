"""
Test script for static style visualization using existing demo results
"""

import json
from pathlib import Path
from static_style_visualizer import StaticStyleVisualizer


def test_with_demo_results():
    """Test static style visualizer with existing demo results."""
    
    # Find demo results
    demo_results_dir = Path("demo_results")
    if not demo_results_dir.exists():
        demo_results_dir = Path("../demo_results")
    
    if not demo_results_dir.exists():
        print("Demo results directory not found!")
        return
    
    # Create output directory
    output_dir = Path("results/static_style_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = StaticStyleVisualizer(output_dir)
    
    # Collect results from demo runs
    experiment_results = {}
    
    # Map demo experiment names to standard names
    name_mapping = {
        "demo_3agent_mixed": "3 TFT",
        "scaling_test_3_agents": "3 TFT"
    }
    
    # Search for result files
    for json_file in demo_results_dir.rglob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Get experiment name from path
            exp_dir = json_file.parent.name
            
            # Map to standard name if possible
            standard_name = name_mapping.get(exp_dir, exp_dir)
            
            # Add to results (simulate multiple runs by duplicating)
            if standard_name not in experiment_results:
                experiment_results[standard_name] = []
            
            # Add the same result multiple times to simulate multiple runs
            for _ in range(5):  # Simulate 5 runs
                experiment_results[standard_name].append(result)
            
            print(f"Loaded results from {json_file}")
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if experiment_results:
        print(f"\nGenerating static-style figures for {len(experiment_results)} experiments...")
        visualizer.generate_all_figures(experiment_results)
        print(f"Figures saved to {output_dir}")
    else:
        print("No results found to visualize!")


def test_with_static_runner():
    """Test by running a small experiment with static runner."""
    from experiments.runners.static_style_runner import StaticStyleRunner
    
    # Run a quick test with fewer rounds and runs
    runner = StaticStyleRunner(
        output_dir="results/static_style_test_run",
        num_runs=3,  # Just 3 runs for testing
        num_rounds=20,  # Just 20 rounds for testing
        rounds_per_pair=20
    )
    
    print("Running test experiments...")
    
    # Run just one experiment type for testing
    test_config = {
        "3 TFT": runner.STANDARD_EXPERIMENTS["3 TFT"]
    }
    
    # Override standard experiments for testing
    runner.STANDARD_EXPERIMENTS = test_config
    
    # Run experiments
    results = runner.run_all_experiments(game_types=['nperson'])
    
    print(f"\nTest completed. Results saved to: {runner.experiment_dir}")


if __name__ == "__main__":
    print("Testing static style visualization...")
    
    # First try with existing demo results
    print("\n1. Testing with existing demo results:")
    test_with_demo_results()
    
    # Then run a small test experiment
    print("\n2. Testing with new experiment run:")
    test_with_static_runner()