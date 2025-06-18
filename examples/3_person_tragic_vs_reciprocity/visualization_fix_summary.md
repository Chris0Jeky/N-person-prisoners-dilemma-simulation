# Visualization Code Fix Summary

## Issues Identified

1. **Empty Data Directory**: The batch directory `demo_results/batch_runs/batch_20250616_042128` contains no data files (neither JSON nor CSV), which is why visualizations show "No data".

2. **CSV Reading Implementation**: The original visualization code was designed to read from JSON files, not CSV files.

3. **Directory Structure**: The expected directory structure is:
   ```
   results/TIMESTAMP/
   ├── experiment_name/
   │   ├── csv/
   │   │   ├── experiment_name_history.csv
   │   │   └── experiment_name_summary.csv
   │   └── figures/
   │       ├── cooperation_evolution.png
   │       ├── cooperation_heatmap.png
   │       ├── score_distribution.png
   │       └── agent_performance.png
   ```

## Solutions Implemented

### 1. CSV-Based Visualizer (`npd_simulator/analysis/visualizers/csv_visualizer.py`)
- Reads data from CSV files instead of JSON
- Generates figures in the correct directory structure
- Handles both single experiments and batch directories
- Creates four standard visualizations:
  - Cooperation evolution over time
  - Agent actions heatmap
  - Score distribution
  - Agent performance metrics

### 2. Simple CSV Analyzer (`simple_csv_analyzer.py`)
- Standalone script that doesn't require external dependencies
- Provides text-based analysis of CSV data
- Shows cooperation statistics and agent performance
- Works without matplotlib/seaborn

### 3. Experiment Processor (`process_experiment_results.py`)
- Checks for existing data files
- Converts JSON to CSV format if needed
- Creates necessary directory structure
- Handles both single experiments and batch processing

## Usage

### For experiments with CSV data:
```bash
# Analyze CSV data (text output)
python3 simple_csv_analyzer.py demo_results/basic_runs/demo_3agent_mixed

# Generate visualizations (requires matplotlib/seaborn)
python3 visualize_csv_results.py demo_results/basic_runs/demo_3agent_mixed
```

### For experiments with JSON data:
```bash
# Convert JSON to CSV first
python3 process_experiment_results.py path/to/experiment

# Then analyze or visualize as above
```

### For batch processing:
```bash
# Process entire batch directory
python3 process_experiment_results.py demo_results/batch_runs/batch_name

# Analyze all experiments
python3 simple_csv_analyzer.py demo_results/batch_runs/batch_name
```

## Key Findings

- The batch `batch_20250616_042128` has no data files, which explains the "No data" errors
- Other batches like `batch_20250616_034507` have proper CSV data and can be visualized
- The CSV format stores:
  - Summary: overall metrics and per-agent statistics
  - History: round-by-round actions, cooperation rates, and payoffs

## Next Steps

1. **Run experiments** to generate data in the empty batch directory
2. **Use the CSV visualizer** on directories with data
3. **Install dependencies** (numpy, pandas, matplotlib, seaborn) for full visualization capabilities
4. **Use the simple analyzer** for quick text-based analysis without dependencies