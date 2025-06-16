#!/usr/bin/env python3
"""
Process experiment results - convert JSON to CSV if needed and generate visualizations
"""

import json
import csv
from pathlib import Path
import sys
import ast
from typing import Dict, List, Any


class ExperimentProcessor:
    """Process experiment results from JSON or CSV files"""
    
    def __init__(self):
        self.processed_count = 0
    
    def find_json_files(self, directory: Path) -> List[Path]:
        """Find all JSON result files in a directory"""
        json_files = []
        for json_file in directory.rglob("*_results.json"):
            json_files.append(json_file)
        return json_files
    
    def json_to_csv(self, json_path: Path) -> bool:
        """Convert JSON results to CSV files"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Create CSV directory
            csv_dir = json_path.parent / "csv"
            csv_dir.mkdir(exist_ok=True)
            
            # Extract base name
            base_name = json_path.stem.replace('_results', '')
            
            # Write summary CSV
            summary_path = csv_dir / f"{base_name}_summary.csv"
            self.write_summary_csv(data, summary_path)
            
            # Write history CSV
            history_path = csv_dir / f"{base_name}_history.csv"
            self.write_history_csv(data, history_path)
            
            print(f"  ✓ Converted {json_path} to CSV files")
            return True
            
        except Exception as e:
            print(f"  ✗ Error converting {json_path}: {e}")
            return False
    
    def write_summary_csv(self, data: Dict[str, Any], output_path: Path):
        """Write summary data to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write main metrics
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['num_agents', data.get('num_agents', 0)])
            writer.writerow(['num_rounds', data.get('num_rounds', 0)])
            writer.writerow(['average_cooperation', data.get('average_cooperation', 0)])
            writer.writerow([])
            
            # Write agent stats
            if 'agent_stats' in data:
                writer.writerow(['Agent Stats'])
                writer.writerow(['agent_id', 'total_score', 'cooperation_rate'])
                for agent in data['agent_stats']:
                    writer.writerow([
                        agent['agent_id'],
                        agent['total_score'],
                        agent['cooperation_rate']
                    ])
    
    def write_history_csv(self, data: Dict[str, Any], output_path: Path):
        """Write history data to CSV"""
        if 'history' not in data:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['actions', 'cooperation_rate', 'payoffs', 'round'])
            
            # Write history data
            for entry in data['history']:
                actions = entry.get('actions', {})
                payoffs = entry.get('payoffs', {})
                writer.writerow([
                    str(actions),
                    entry.get('cooperation_rate', 0),
                    str(payoffs),
                    entry.get('round', 0)
                ])
    
    def check_and_create_directories(self, experiment_path: Path):
        """Ensure proper directory structure exists"""
        # Create figures directory if it doesn't exist
        figures_dir = experiment_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create csv directory if it doesn't exist
        csv_dir = experiment_path / "csv"
        csv_dir.mkdir(exist_ok=True)
    
    def process_experiment(self, experiment_path: Path) -> bool:
        """Process a single experiment directory"""
        self.check_and_create_directories(experiment_path)
        
        # Check if CSV files already exist
        csv_dir = experiment_path / "csv"
        csv_files = list(csv_dir.glob("*.csv"))
        
        if csv_files:
            print(f"  ✓ CSV files already exist in {experiment_path}")
            return True
        
        # Look for JSON files to convert
        json_files = list(experiment_path.glob("*_results.json"))
        
        if not json_files:
            print(f"  ⚠ No data files found in {experiment_path}")
            return False
        
        # Convert JSON to CSV
        for json_file in json_files:
            if self.json_to_csv(json_file):
                self.processed_count += 1
        
        return True
    
    def process_batch(self, batch_dir: Path):
        """Process all experiments in a batch directory"""
        print(f"\nProcessing batch directory: {batch_dir}")
        
        # Find all experiment directories
        experiment_dirs = []
        
        # Look for directories that might contain experiments
        for subdir in batch_dir.rglob("*"):
            if subdir.is_dir():
                # Check if it has JSON files or a csv subdirectory
                if list(subdir.glob("*_results.json")) or (subdir / "csv").exists():
                    experiment_dirs.append(subdir)
        
        print(f"Found {len(experiment_dirs)} experiment directories")
        
        for exp_dir in experiment_dirs:
            print(f"\nProcessing: {exp_dir}")
            self.process_experiment(exp_dir)
        
        print(f"\n{'='*60}")
        print(f"Processing complete. Converted {self.processed_count} experiments.")
        print(f"{'='*60}")
    
    def process_path(self, path: Path):
        """Process a path - could be batch or single experiment"""
        if not path.exists():
            print(f"Error: Path not found: {path}")
            return
        
        # Check if it's a single experiment directory
        if list(path.glob("*_results.json")) or (path / "csv").exists():
            print(f"Processing single experiment: {path}")
            self.process_experiment(path)
        else:
            # It's a batch directory
            self.process_batch(path)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python process_experiment_results.py <path_to_experiment_or_batch>")
        print("\nThis script will:")
        print("  1. Check for existing CSV files")
        print("  2. Convert JSON results to CSV if needed")
        print("  3. Create necessary directory structure")
        print("\nExamples:")
        print("  python process_experiment_results.py demo_results/basic_runs/demo_3agent_mixed")
        print("  python process_experiment_results.py demo_results/batch_runs/batch_20250616_042128")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    processor = ExperimentProcessor()
    processor.process_path(path)
    
    print("\nNext steps:")
    print("1. If CSV files were created, you can now generate visualizations")
    print("2. Use visualize_csv_results.py (requires matplotlib/seaborn)")
    print("3. Or use simple_csv_analyzer.py for text-based analysis")


if __name__ == "__main__":
    main()