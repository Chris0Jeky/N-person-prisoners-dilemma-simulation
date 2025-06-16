#!/usr/bin/env python3
"""
Analyze CSV results without external dependencies.
"""

import csv
import json
from pathlib import Path


def read_csv_data(filepath):
    """Read CSV and extract key information."""
    data = {'rounds': [], 'mean_cooperation': [], 'all_runs': {}}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['rounds'].append(int(row['Round']))
            data['mean_cooperation'].append(float(row['Mean_Cooperation_Rate']))
            
            # Get individual run data
            for key in row:
                if key.startswith('Run_'):
                    run_num = key
                    if run_num not in data['all_runs']:
                        data['all_runs'][run_num] = []
                    data['all_runs'][run_num].append(float(row[key]))
    
    return data


def analyze_experiment_data():
    """Analyze the experiment results."""
    results_dir = Path("results")
    
    print("=" * 60)
    print("RESEARCH EXPERIMENT RESULTS ANALYSIS")
    print("=" * 60)
    
    # Configurations to analyze
    configs = [
        ("3_TFT", "3 TFT"),
        ("2_TFT-E__plus__1_AllD", "2 TFT-E + 1 AllD"),
        ("2_TFT__plus__1_AllD", "2 TFT + 1 AllD"),
        ("2_TFT-E__plus__1_AllC", "2 TFT-E + 1 AllC")
    ]
    
    # Analyze TFT experiments
    print("\n1. TFT EXPERIMENT RESULTS")
    print("-" * 40)
    
    pairwise_results = {}
    neighbourhood_results = {}
    
    for config_name, display_name in configs:
        print(f"\n{display_name}:")
        
        # Pairwise
        pairwise_file = results_dir / f"pairwise_tft_cooperation_{config_name}_aggregated.csv"
        if pairwise_file.exists():
            data = read_csv_data(pairwise_file)
            final_coop = data['mean_cooperation'][-1] if data['mean_cooperation'] else 0
            pairwise_results[config_name] = final_coop
            print(f"  Pairwise final cooperation: {final_coop:.3f}")
        
        # Neighbourhood
        neighbourhood_file = results_dir / f"neighbourhood_tft_cooperation_{config_name}_aggregated.csv"
        if neighbourhood_file.exists():
            data = read_csv_data(neighbourhood_file)
            final_coop = data['mean_cooperation'][-1] if data['mean_cooperation'] else 0
            neighbourhood_results[config_name] = final_coop
            print(f"  Neighbourhood final cooperation: {final_coop:.3f}")
        
        # Calculate difference
        if config_name in pairwise_results and config_name in neighbourhood_results:
            diff = neighbourhood_results[config_name] - pairwise_results[config_name]
            print(f"  Difference (N - P): {diff:.3f} ({'Reciprocity Hill' if diff > 0 else 'Tragic Valley'})")
    
    # Analyze comparative results
    print("\n\n2. COMPARATIVE ANALYSIS")
    print("-" * 40)
    
    comp_file = results_dir / "comparative_analysis.json"
    if comp_file.exists():
        with open(comp_file, 'r') as f:
            comp_data = json.load(f)
        
        # TFT Analysis
        tft_analysis = comp_data.get('tft_analysis', {})
        differences = tft_analysis.get('differences', {})
        
        print("\nTragic Valley vs Reciprocity Hill:")
        for config, diff_data in differences.items():
            print(f"\n{config}:")
            print(f"  Cooperation difference: {diff_data['cooperation_difference']:.3f}")
            print(f"  Shows tragic valley: {diff_data['shows_tragic_valley']}")
            print(f"  Shows reciprocity hill: {diff_data['shows_reciprocity_hill']}")
        
        # QL Analysis summary
        ql_analysis = comp_data.get('ql_analysis', {})
        
        print("\n\n3. QLEARNING RESULTS SUMMARY")
        print("-" * 40)
        
        # 2QL experiments
        ql_2ql = ql_analysis.get('2QL', {})
        print("\n2QL Experiments:")
        for exp_name, exp_data in ql_2ql.items():
            print(f"\n{exp_name}:")
            print(f"  Early cooperation: {exp_data.get('mean_early_cooperation', 0):.3f}")
            print(f"  Late cooperation: {exp_data.get('mean_late_cooperation', 0):.3f}")
            print(f"  Learning improvement: {exp_data.get('learning_improvement', 0):.3f}")
            
            # Agent scores
            if 'final_scores' in exp_data:
                print("  Final scores:")
                for agent, score_data in exp_data['final_scores'].items():
                    print(f"    {agent}: {score_data['mean_score']:.1f} ± {score_data['std_score']:.1f}")
    
    # Summary statistics
    print("\n\n4. SUMMARY STATISTICS")
    print("-" * 40)
    
    # Count CSV files
    csv_files = list(results_dir.glob("*.csv"))
    print(f"Total CSV files generated: {len(csv_files)}")
    
    # Check figures
    figure_files = list(results_dir.glob("figure*.png"))
    print(f"Total figure files: {len(figure_files)}")
    
    if figure_files:
        print("\nFigures generated:")
        for fig in sorted(figure_files):
            size_kb = fig.stat().st_size / 1024
            print(f"  - {fig.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    analyze_experiment_data()