# NPDL Comprehensive Testing Plan

## Overview

This document outlines the testing strategy for the N-Person Prisoner's Dilemma Learning (NPDL) project. It includes information about the current test suite, future test development plans, and best practices for maintaining and expanding the test coverage.

## Testing Philosophy

The NPDL testing strategy follows these principles:

1. **Comprehensive Coverage**: Test all components and features of the system.
2. **Isolation**: Test components in isolation where possible to identify issues precisely.
3. **Integration**: Test interactions between components to ensure they work together correctly.
4. **Automation**: Automate tests to ensure consistency and enable continuous integration.
5. **Documentation**: Document test cases to clarify expected behavior and assist future development.

## Current Test Suite

The current test suite includes the following categories:

### Core Component Tests
- `test_agents.py`: Tests for Agent class and various strategies
- `test_environment.py`: Tests for Environment class and network structures
- `test_utils.py`: Tests for utility functions and payoff calculations
- `test_logging_utils.py`: Tests for logging setup, statistics, and report generation ✓ NEW
- `test_advanced_strategies.py`: Dedicated tests for LRA-Q, UCB1, Wolf-PHC, and Hysteretic Q-learning ✓ NEW

### User Interface Tests
- `test_cli.py`: Tests for command-line interface, argument parsing, and command execution ✓ NEW
- `test_interactive_game.py`: Tests for interactive game mode, player input, and game flow ✓ NEW

### Integration Tests
- `test_integration.py`: Tests for end-to-end simulation workflows
- `test_pairwise_integration.py`: Integration tests for pairwise interaction mode

### Visualization Tests
- `test_visualization.py`: Tests for data processing and visualization components

### Mode-Specific Tests
- `test_pairwise.py`: Tests for pairwise interaction mode (updated to remove workarounds) ✓ FIXED
- `test_pairwise_strategies.py`: Tests for strategies in pairwise mode
- `test_neighborhood_vs_pairwise.py`: Comparative tests between interaction modes
- `test_true_pairwise.py`: Tests for true pairwise implementation with individual decisions

### Other Tests
- `test_core_basic.py`: Basic functionality tests
- `test_environment_fixed.py`: Tests for environment fixes
- `test_refactored_paths.py`: Tests for path refactoring

## Test Coverage Map

| Module | Components | Coverage Status | Priority Areas |
|--------|------------|----------------|----------------|
| `npdl.core.agents` | Agent class, Strategy classes | Excellent ✓ | All strategies now tested |
| `npdl.core.environment` | Environment class, Network creation | Good | Network rewiring, Dynamic environments |
| `npdl.core.utils` | Payoff functions, Matrix creation | Good | Custom payoff functions |
| `npdl.core.logging_utils` | Logging setup, Statistics | Good ✓ | Fully tested |
| `npdl.core.true_pairwise` | Individual decision-making | Good | Recently added with tests |
| `npdl.visualization` | Data loading, Processing, Visualization | Partial | Dashboard components, Interactive elements |
| `npdl.interactive` | Game class, Interactive mode | Good ✓ | Fully tested |
| `npdl.cli` | Command-line interface | Good ✓ | Fully tested |

## Planned Test Development

### Completed (Previously High Priority) ✓
1. **Command-line Interface** - COMPLETED
   - ✓ Parameter parsing
   - ✓ Error handling
   - ✓ Output formatting
   - ✓ Command execution

2. **Interactive Mode** - COMPLETED
   - ✓ Game initialization
   - ✓ Player moves and feedback
   - ✓ AI opponent behavior
   - ✓ Score calculation

3. **Logging and Reporting** - COMPLETED
   - ✓ Log formatting
   - ✓ Report generation
   - ✓ Statistics calculation
   - ✓ ASCII chart generation

4. **Advanced Strategy Tests** - COMPLETED
   - ✓ Learning Rate Adjusting Q-Learning
   - ✓ Wolf-PHC
   - ✓ UCB1
   - ✓ Hysteretic Q-Learning
   - ✓ Strategic behavior in specific scenarios

### Current High Priority
1. **Dashboard and Visualization**
   - Dashboard component rendering tests
   - Network visualization accuracy
   - Data processing edge cases
   - Response to malformed data

### Medium Priority
1. **Enhanced Network Tests**
   - Dynamic network adaptation
   - Network metrics calculation
   - Impact of network structure on cooperation

2. **Performance Tests**
   - Simulation scaling
   - Memory usage profiling
   - Large-scale agent interactions

### Low Priority
1. **Performance Tests**
   - Simulation scaling
   - Memory usage
   - Optimization validation

2. **Configuration Management**
   - Parameter validation
   - Configuration file parsing
   - Default values

## Recent Test Additions (2024)

### New Test Files Created
1. **test_cli.py** - Comprehensive CLI testing including:
   - Command parsing and argument validation
   - Subcommand execution (simulate, visualize, interactive)
   - Error handling and fallback mechanisms
   - Dependency checking

2. **test_interactive_game.py** - Full interactive mode testing:
   - Game initialization with various parameters
   - Player input handling and validation
   - AI opponent behavior
   - Score calculation and game flow
   - UI display functions

3. **test_logging_utils.py** - Complete logging system tests:
   - Logging setup and configuration
   - Network statistics logging
   - Round statistics with Q-learning metrics
   - Experiment summary generation
   - ASCII chart generation

4. **test_advanced_strategies.py** - Dedicated advanced strategy tests:
   - LRA-Q learning rate adjustment
   - UCB1 exploration bonus calculation
   - Wolf-PHC policy improvement
   - Hysteretic Q-learning asymmetric updates
   - Strategy comparison tests

### Test Fixes
- **test_pairwise.py** - Removed manual workarounds for TFT behavior
  - Updated to work with current TFT implementation
  - Tests now properly verify cooperation proportion thresholds
  - Added tests for specific_opponent_moves format

## Test Implementation Guidelines

### Unit Tests
- Focus on testing individual functions and methods in isolation
- Use mocks for dependencies
- Test both normal operation and edge cases
- Verify both return values and side effects

### Integration Tests
- Test workflows that involve multiple components
- Verify that components work together correctly
- Use realistic test data

### System Tests
- Test the entire system end-to-end
- Verify that all components work together correctly
- Test with realistic scenarios and data

### Test Fixtures and Utilities
- Create reusable test fixtures for common setup
- Develop test utilities for repetitive tasks
- Document fixtures and utilities clearly

## Test Development Process

1. **Identify Gaps**: Review the test coverage map to identify areas that need additional testing
2. **Prioritize**: Focus on high-priority areas first
3. **Implement**: Write tests following the guidelines
4. **Review**: Review tests to ensure they are effective and maintainable
5. **Integrate**: Integrate tests into the continuous integration pipeline
6. **Monitor**: Monitor test results and address failures promptly

## Best Practices

### Test Organization
- Group tests logically by module and functionality
- Use descriptive test names that explain what is being tested
- Organize test files to mirror the structure of the code being tested

### Test Design
- Test one thing per test function
- Make tests independent and idempotent
- Minimize setup and teardown code
- Use fixtures for common setup

### Test Maintenance
- Update tests when code changes
- Remove obsolete tests
- Refactor tests to keep them maintainable
- Document complex test logic

## Continuous Integration

The test suite should be integrated into a continuous integration (CI) pipeline that:

1. Runs all tests on every commit
2. Reports test coverage
3. Fails the build if tests fail
4. Notifies developers of test failures

## Test-Driven Development

For new features, consider following a test-driven development (TDD) approach:

1. Write a failing test that describes the desired behavior
2. Implement the minimal code needed to pass the test
3. Refactor the code while keeping the test passing

## Conclusion

A comprehensive test suite is essential for maintaining the quality and reliability of the NPDL project. By following this testing plan, we can ensure that the project remains stable and functional as it evolves.

## Appendix: Test Command Reference

### Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_agents.py

# Run tests with coverage report
pytest --cov=npdl

# Run only unit tests
pytest -m unit

# Run tests and generate HTML coverage report
pytest --cov=npdl --cov-report=html
```

### Test Markers

- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow tests
- `visualization`: Visualization component tests
- `network`: Network functionality tests
