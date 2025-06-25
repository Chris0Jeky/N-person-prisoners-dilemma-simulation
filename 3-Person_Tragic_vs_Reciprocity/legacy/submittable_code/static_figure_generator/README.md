# Static Figure Generator for N-Person Prisoner's Dilemma

This code analyzes cooperation dynamics with traditional strategies (TFT, TFT-E, AllC, AllD) in both pairwise and N-person prisoner's dilemma settings.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

Run the simulation:
```bash
python static_figure_generator.py
```

## Output

The script will:
1. Run 50 simulations per experiment for statistical aggregation
2. Test 4 different agent compositions:
   - 3 TFT-E agents
   - 2 TFT-E + 1 AllD
   - 2 TFT + 1 AllD  
   - 2 TFT-E + 1 AllC
3. Generate results in a `results/` directory including:
   - CSV files with aggregated cooperation and score data
   - PNG figures showing cooperation dynamics and agent scores
   - Both pairwise and neighbourhood (N-person) voting scenarios

## Configuration

- NUM_ROUNDS = 500 (rounds per simulation)
- NUM_RUNS = 50 (number of simulation runs for statistical aggregation)
- Exploration rates and decay parameters can be modified in the `setup_experiments()` function