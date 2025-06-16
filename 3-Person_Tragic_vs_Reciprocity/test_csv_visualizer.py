#!/usr/bin/env python3
"""Test script for CSV visualizer"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from npd_simulator.analysis.visualizers.csv_visualizer import CSVVisualizer

def test_visualizer():
    """Test the CSV visualizer on demo data"""
    # Test on demo_3agent_mixed
    demo_path = Path("demo_results/basic_runs/demo_3agent_mixed")
    
    if not demo_path.exists():
        print(f"Demo path not found: {demo_path}")
        return
    
    print(f"Testing CSV visualizer on: {demo_path}")
    
    # Create visualizer
    visualizer = CSVVisualizer(demo_path)
    
    # Generate visualizations
    plots = visualizer.visualize_experiment(demo_path)
    
    print(f"\nGenerated {len(plots)} plots successfully!")
    for name, path in plots.items():
        print(f"  - {name}: {path}")
    
    # Test on batch directory
    batch_path = Path("demo_results/batch_runs/batch_20250616_042128")
    if batch_path.exists():
        print(f"\nTesting on batch directory: {batch_path}")
        visualizer = CSVVisualizer(batch_path)
        results = visualizer.visualize_batch(batch_path)
        print(f"Processed {len(results)} experiments in batch")

if __name__ == "__main__":
    test_visualizer()