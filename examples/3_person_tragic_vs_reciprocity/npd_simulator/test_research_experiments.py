"""
Test the research experiments setup
"""

import logging
from experiments.research_experiments import ResearchExperiments

def test_basic_setup():
    """Test basic experiment setup."""
    logging.basicConfig(level=logging.INFO)
    
    # Create experiments with small parameters for testing
    experiments = ResearchExperiments(
        num_runs=2,  # Just 2 runs for testing
        num_rounds=10,  # Just 10 rounds for testing
        output_dir="test_results"
    )
    
    print("Testing TFT configurations...")
    # Test one TFT configuration
    test_config = experiments.tft_configs["3_TFT"]
    data = experiments._run_multiple_experiments(
        test_config, "nperson", "test_3_TFT"
    )
    
    print(f"Generated data with {len(data['cooperation_rates'])} runs")
    print(f"Each run has {len(data['cooperation_rates'][0])} rounds")
    
    # Test CSV saving
    print("\nTesting CSV generation...")
    experiments._save_aggregated_csv(
        data, "test", "cooperation", "test_3_TFT"
    )
    
    print("\nTest completed successfully!")
    
    # Clean up
    import shutil
    shutil.rmtree("test_results", ignore_errors=True)

if __name__ == "__main__":
    test_basic_setup()