# NPDL Test Suite

This directory contains comprehensive tests for the N-Person Prisoner's Dilemma Learning (NPDL) simulation framework.

## Overview

The test suite verifies the functionality of all components of the NPDL framework, including:
- Agent behavior and strategies
- Environment simulation
- Network structures
- Payoff calculations
- Visualization components
- Integration between components

## Running Tests

To run the entire test suite:

```bash
# From the project root directory
pytest
```

To run specific test files:

```bash
# Run agent tests
pytest tests/test_agents.py

# Run environment tests
pytest tests/test_environment.py

# Run utility function tests
pytest tests/test_utils.py

# Run integration tests
pytest tests/test_integration.py

# Run visualization tests
pytest tests/test_visualization.py
```

To run tests with coverage reporting:

```bash
pytest --cov=npdl
```

## Test Organization

The test suite is organized by component:

### Core Components
- `test_agents.py`: Tests for the Agent class and various strategy implementations
- `test_environment.py`: Tests for the Environment class, network creation, and simulation
- `test_utils.py`: Tests for utility functions, payoff calculations, and helper functions
- `test_logging_utils.py`: Tests for logging setup, statistics, and report generation
- `test_advanced_strategies.py`: Dedicated tests for LRA-Q, UCB1, Wolf-PHC, and Hysteretic Q-learning

### User Interfaces
- `test_cli.py`: Tests for command-line interface, argument parsing, and command execution
- `test_interactive_game.py`: Tests for interactive game mode, player input, and game flow

### Integration and Modes
- `test_integration.py`: Tests for end-to-end simulation workflows
- `test_pairwise.py`: Tests for pairwise interaction mode
- `test_pairwise_integration.py`: Integration tests for pairwise mode
- `test_pairwise_strategies.py`: Tests for strategies in pairwise mode
- `test_neighborhood_vs_pairwise.py`: Comparative tests between interaction modes
- `test_true_pairwise.py`: Tests for true pairwise implementation with individual decisions

### Other Tests
- `test_visualization.py`: Tests for data processing and visualization components
- `test_core_basic.py`: Basic functionality tests
- `test_environment_fixed.py`: Tests for environment fixes
- `test_refactored_paths.py`: Tests for path refactoring

## Test Fixtures

Common test fixtures are defined in `conftest.py` and include:
- `simple_agents`: A set of agents with different strategies
- `simple_environment`: A basic environment with a fully connected network
- `test_scenario`: A sample scenario configuration for testing
- Various data fixtures for visualization testing

## Network Visualization Troubleshooting

If you're experiencing issues with the network visualization in the dashboard:

1. **Check that network data is being saved properly**:
   - After running a simulation, check that `*_network.json` files exist in the results directory
   - These files should contain network structure data

2. **Run simulations with enhanced output**:
   ```bash
   python run.py simulate --enhanced --verbose
   ```

3. **Verify JSON structure**:
   - Open one of the network JSON files
   - It should contain "nodes", "edges", "network_type", and "network_params" fields

4. **Update visualization code**:
   - If necessary, run the fix script to ensure network data is properly exported:
   ```bash
   python fix_network_visualization.py
   ```

## Extending the Test Suite

When adding new features to the NPDL framework, please add corresponding tests:

1. For new agent strategies, add tests to `test_agents.py` or `test_advanced_strategies.py`
2. For new environment features, add tests to `test_environment.py`
3. For new utility functions, add tests to `test_utils.py`
4. For new visualization components, add tests to `test_visualization.py`
5. For workflow changes, update `test_integration.py`
6. For CLI changes, update `test_cli.py`
7. For interactive mode changes, update `test_interactive_game.py`
8. For logging features, update `test_logging_utils.py`
9. For pairwise mode changes, update relevant pairwise test files

See the comprehensive test plan in `TEST_PLAN.md` for more information on future test development.
