# Static Figure Generator - 2 TFT-E with AllC/AllD Only

This is a modified version of the static figure generator that only generates results for two specific agent compositions:
- 2 TFT-E + 1 AllC
- 2 TFT-E + 1 AllD

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

Run the simulation:
```bash
python static_figure_generator_2tfte_allc_alld.py
```

## Output

The script will:
1. Run 50 simulations per experiment for statistical aggregation
2. Test only 2 agent compositions:
   - 2 TFT-E + 1 AllC
   - 2 TFT-E + 1 AllD
3. Generate results in a `results_2tfte_only/` directory including:
   - CSV files with aggregated cooperation and score data
   - PNG figures showing cooperation dynamics and agent scores
   - Both pairwise and neighbourhood (N-person) voting scenarios

## Configuration

- NUM_ROUNDS = 200 (rounds per simulation - shorter than the full version)
- NUM_RUNS = 50 (number of simulation runs for statistical aggregation)
- TFT-E exploration rate = 0.1 for both experiments