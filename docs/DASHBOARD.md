# Canonical dashboard (W8)

The canonical dashboard is the Dash app in
[`npdl/visualization/dashboard.py`](../npdl/visualization/dashboard.py),
launched through the CLI. It is the only live chart implementation.

## Launch

```bash
pip install -r requirements.txt   # adds dash, dash-bootstrap-components, plotly, flask
python run.py simulate            # writes results/<scenario>/run_XX/*.csv (+ network JSON)
python run.py visualize           # Dash dashboard at http://127.0.0.1:8050/
```

Run both commands from the repo root: the dashboard reads the `results/`
tree that `run.py simulate` writes (`experiment_results_agents.csv`,
`experiment_results_rounds.csv`, `experiment_results_network.json` per
`run_XX/` directory). A custom port is available programmatically:

```python
from npdl.visualization.dashboard import run_dashboard

run_dashboard(port=8051)
```

## Tabs

- **Cooperation Rates**: per-strategy cooperation rate over time plus the
  overall rate (from `npdl.visualization.data_loader` /
  `npdl.visualization.data_processor`).
- **Payoffs**: average payoff over time by strategy.
- **Final Scores**: distribution of final scores by strategy.
- **Network**: per-round network structure colored by strategy.

## Tests

```bash
pytest tests/test_dashboard_smoke.py -q   # figure smoke test, no browser
pytest tests/test_visualization.py -q     # loader/processor/network-figure tests
```

Both modules skip with a documented reason when the dashboard stack
(`dash`, `dash-bootstrap-components`, `plotly`, `flask`) is not
installed; the smoke test calls the Dash callbacks directly against a
small fixture `results/` tree and never opens a browser.
