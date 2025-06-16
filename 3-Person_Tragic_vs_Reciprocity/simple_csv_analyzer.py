#!/usr/bin/env python3
"""
Simple CSV analyzer that reads NPD simulation CSV files and displays analysis
without requiring external dependencies like numpy/matplotlib
"""

import csv
import json
import ast
from pathlib import Path
from typing import Dict, List, Any
import statistics
import sys


class SimpleCSVAnalyzer:
    """Analyze CSV files and display results"""
    
    def __init__(self):
        self.results = {}
    
    def read_csv_data(self, csv_dir: Path) -> Dict[str, List[Dict]]:
        """Read CSV files from a directory"""
        csv_data = {}
        
        # Look for history and summary CSV files
        for csv_file in csv_dir.glob("*.csv"):
            if "history" in csv_file.name:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    csv_data['history'] = list(reader)
                print(f"  - Loaded history from: {csv_file}")
            elif "summary" in csv_file.name:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    csv_data['summary'] = list(reader)
                print(f"  - Loaded summary from: {csv_file}")
        
        return csv_data
    
    def analyze_cooperation(self, history_data: List[Dict]) -> Dict[str, Any]:
        """Analyze cooperation patterns from history data"""
        if not history_data:
            return {}
        
        coop_rates = []
        rounds = []
        
        for row in history_data:
            try:
                coop_rate = float(row['cooperation_rate'])
                round_num = int(row['round'])
                coop_rates.append(coop_rate)
                rounds.append(round_num)
            except (ValueError, KeyError):
                continue
        
        if not coop_rates:
            return {}
        
        # Calculate statistics
        analysis = {
            'total_rounds': len(rounds),
            'average_cooperation': statistics.mean(coop_rates),
            'min_cooperation': min(coop_rates),
            'max_cooperation': max(coop_rates),
            'stdev_cooperation': statistics.stdev(coop_rates) if len(coop_rates) > 1 else 0,
            'final_cooperation': coop_rates[-1] if coop_rates else 0,
            'initial_cooperation': coop_rates[0] if coop_rates else 0
        }
        
        # Analyze trends
        if len(coop_rates) > 10:
            first_quarter = coop_rates[:len(coop_rates)//4]
            last_quarter = coop_rates[3*len(coop_rates)//4:]
            analysis['early_avg_cooperation'] = statistics.mean(first_quarter)
            analysis['late_avg_cooperation'] = statistics.mean(last_quarter)
            analysis['cooperation_change'] = analysis['late_avg_cooperation'] - analysis['early_avg_cooperation']
        
        return analysis
    
    def analyze_agents(self, history_data: List[Dict], summary_data: List[Dict]) -> Dict[str, Any]:
        """Analyze individual agent performance"""
        agent_analysis = {}
        
        # Parse agent stats from summary if available
        if summary_data:
            # Check if we have the agent stats header row
            for i, row in enumerate(summary_data):
                if 'agent_id' in row and 'total_score' in row and 'cooperation_rate' in row:
                    # This is the header row for agent stats
                    # The next rows contain the actual agent data
                    for j in range(i + 1, len(summary_data)):
                        agent_row = summary_data[j]
                        if agent_row.get('agent_id', '').strip():
                            try:
                                agent_id = int(agent_row['agent_id'])
                                agent_analysis[f'agent_{agent_id}'] = {
                                    'total_score': float(agent_row['total_score']),
                                    'cooperation_rate': float(agent_row['cooperation_rate'])
                                }
                            except (ValueError, KeyError):
                                break
                    break
        
        # Analyze actions from history
        if history_data and not agent_analysis:
            agent_actions = {}
            
            for row in history_data:
                try:
                    actions = ast.literal_eval(row['actions'])
                    for agent_id, action in actions.items():
                        if agent_id not in agent_actions:
                            agent_actions[agent_id] = []
                        agent_actions[agent_id].append(action)
                except:
                    continue
            
            # Calculate cooperation rates
            for agent_id, actions in agent_actions.items():
                coop_count = sum(1 for a in actions if a == 0)
                agent_analysis[f'agent_{agent_id}'] = {
                    'cooperation_rate': coop_count / len(actions) if actions else 0,
                    'total_actions': len(actions)
                }
        
        return agent_analysis
    
    def display_analysis(self, experiment_name: str, csv_data: Dict[str, List[Dict]]):
        """Display analysis results"""
        print(f"\n{'='*60}")
        print(f"Analysis for: {experiment_name}")
        print(f"{'='*60}")
        
        if 'history' in csv_data:
            coop_analysis = self.analyze_cooperation(csv_data['history'])
            
            if coop_analysis:
                print("\nCooperation Analysis:")
                print(f"  - Total rounds: {coop_analysis['total_rounds']}")
                print(f"  - Average cooperation rate: {coop_analysis['average_cooperation']:.3f}")
                print(f"  - Min cooperation rate: {coop_analysis['min_cooperation']:.3f}")
                print(f"  - Max cooperation rate: {coop_analysis['max_cooperation']:.3f}")
                print(f"  - Std dev: {coop_analysis['stdev_cooperation']:.3f}")
                print(f"  - Initial cooperation: {coop_analysis['initial_cooperation']:.3f}")
                print(f"  - Final cooperation: {coop_analysis['final_cooperation']:.3f}")
                
                if 'cooperation_change' in coop_analysis:
                    change = coop_analysis['cooperation_change']
                    trend = "increased" if change > 0 else "decreased"
                    print(f"  - Cooperation {trend} by {abs(change):.3f} from early to late game")
        
        agent_analysis = self.analyze_agents(
            csv_data.get('history', []), 
            csv_data.get('summary', [])
        )
        
        if agent_analysis:
            print("\nAgent Analysis:")
            for agent_name, stats in sorted(agent_analysis.items()):
                print(f"  {agent_name}:")
                for stat_name, value in stats.items():
                    if isinstance(value, float):
                        print(f"    - {stat_name}: {value:.3f}")
                    else:
                        print(f"    - {stat_name}: {value}")
    
    def analyze_experiment(self, experiment_path: Path):
        """Analyze a single experiment"""
        csv_dir = experiment_path / "csv"
        
        if not csv_dir.exists():
            print(f"CSV directory not found: {csv_dir}")
            return
        
        csv_data = self.read_csv_data(csv_dir)
        if csv_data:
            self.display_analysis(experiment_path.name, csv_data)
    
    def analyze_batch(self, batch_dir: Path):
        """Analyze all experiments in a batch"""
        print(f"\nAnalyzing batch directory: {batch_dir}")
        
        experiments_found = 0
        for exp_dir in batch_dir.rglob("csv"):
            if exp_dir.is_dir():
                parent_dir = exp_dir.parent
                self.analyze_experiment(parent_dir)
                experiments_found += 1
        
        print(f"\n{'='*60}")
        print(f"Batch analysis complete. Analyzed {experiments_found} experiments.")
        print(f"{'='*60}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python simple_csv_analyzer.py <path_to_experiment_or_batch>")
        print("\nExamples:")
        print("  python simple_csv_analyzer.py demo_results/basic_runs/demo_3agent_mixed")
        print("  python simple_csv_analyzer.py demo_results/batch_runs/batch_20250616_042128")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    analyzer = SimpleCSVAnalyzer()
    
    # Check if it's a batch directory or single experiment
    if (path / "csv").exists():
        # Single experiment
        analyzer.analyze_experiment(path)
    else:
        # Batch directory
        analyzer.analyze_batch(path)


if __name__ == "__main__":
    main()