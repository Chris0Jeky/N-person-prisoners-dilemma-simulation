# Modularization Guide

This guide explains the modular architecture of the NPD Simulator and how to work with its components.

## Overview

The NPD Simulator has been reorganized into a modular architecture that promotes:
- **Separation of Concerns**: Each module has a specific responsibility
- **Extensibility**: Easy to add new features without modifying core code
- **Reusability**: Components can be used independently
- **Maintainability**: Clear structure makes code easier to understand and modify

## Architecture Overview

```
npd_simulator/
├── agents/          # Agent implementations and registry
├── core/            # Core game mechanics
├── experiments/     # Experiment runners and managers
├── analysis/        # Visualization and analysis tools
├── configs/         # Configuration management
├── profiles/        # Experiment profiles
├── utils/           # Shared utilities
└── examples/        # Example usage
```

## Key Components

### 1. Agent Registry System

The agent registry provides a plugin-style system for managing agents:

```python
from agents.registry import AgentRegistry

# Register a new agent type
@AgentRegistry.register(
    name="MyAgent",
    category="custom",
    description="My custom agent"
)
class MyAgent(NPDAgent):
    def choose_action(self, cooperation_ratio, round_number):
        # Implementation
        pass

# Create agents dynamically
agent = AgentRegistry.create("MyAgent", agent_id=0)

# List available agents
available = AgentRegistry.list_agents(category="basic")
```

**Benefits:**
- Dynamic agent registration
- Automatic discovery
- Centralized agent management
- Easy plugin development

### 2. Experiment Profiles

Profiles allow you to define reusable experiment configurations:

```yaml
# profiles/my_experiment.yaml
name: my_experiment
description: Test different strategies

base_agents:
  - type: TFT
    exploration_rate: 0.0
  - type: TFT
    exploration_rate: 0.0
  - type: AllD
    exploration_rate: 0.0

variations:
  - name: cooperative
    replace:
      2: {type: AllC}
  - name: mixed
    replace:
      1: {type: Random, cooperation_probability: 0.5}
```

**Usage:**
```python
from profiles.profile_loader import ProfileLoader

loader = ProfileLoader()
profile = loader.load_profile("my_experiment")
configs = loader.expand_variations(profile)
```

**Benefits:**
- Reusable experiment definitions
- Parameterized configurations
- Easy variation testing
- Version control friendly

### 3. Results Manager

Centralized results management with consistent organization:

```python
from utils.results_manager import ResultsManager

manager = ResultsManager()

# Save results
path = manager.save_experiment(
    results, 
    "my_experiment",
    category="testing"
)

# Search results
matches = manager.search_experiments(
    agent_types=["TFT", "AllD"],
    date_from="20250101"
)

# Compare experiments
comparison = manager.compare_experiments(
    ["exp1_20250115", "exp2_20250116"]
)
```

**Benefits:**
- Consistent result organization
- Easy result retrieval
- Built-in comparison tools
- Automatic indexing

### 4. Enhanced Game State

Track detailed metrics by agent type:

```python
from core.models.enhanced_game_state import EnhancedGameState

# Create with agent tracking
game_state = EnhancedGameState(num_agents, agents)

# Get type-specific metrics
tft_cooperation = game_state.get_type_cooperation_rates()["TFT"]
tft_scores = game_state.get_type_average_scores()["TFT"]
```

**Benefits:**
- Agent-type specific tracking
- Detailed round-by-round history
- Ready for statistical analysis
- Compatible with visualization tools

### 5. Static Style Visualization

Generate publication-ready figures:

```python
from static_style_visualizer import StaticStyleVisualizer

visualizer = StaticStyleVisualizer(output_dir)

# Generate all standard figures
visualizer.generate_all_figures(experiment_results)
```

**Benefits:**
- Professional figure quality
- Statistical aggregation
- Consistent styling
- Batch processing support

## Working with Modules

### Creating a Custom Agent

1. Create your agent class:
```python
# agents/strategies/my_strategy.py
from ..base.agent import NPDAgent
from ..registry import AgentRegistry

@AgentRegistry.register(
    name="Adaptive",
    category="advanced"
)
class AdaptiveAgent(NPDAgent):
    def __init__(self, agent_id, threshold=0.5, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.threshold = threshold
    
    def choose_action(self, cooperation_ratio, round_number):
        if cooperation_ratio > self.threshold:
            return 0, 0  # Cooperate
        return 1, 1  # Defect
```

2. Use in experiments:
```yaml
# profiles/adaptive_test.yaml
agents:
  - type: Adaptive
    threshold: 0.6
  - type: TFT
  - type: AllD
```

### Running Batch Experiments

1. Create a profile with variations:
```yaml
# profiles/parameter_sweep.yaml
name: threshold_sweep

variations:
  - name: low_threshold
    agents:
      - {type: Adaptive, threshold: 0.3}
      - {type: TFT}
      - {type: TFT}
  
  - name: high_threshold
    agents:
      - {type: Adaptive, threshold: 0.7}
      - {type: TFT}
      - {type: TFT}
```

2. Run the experiments:
```python
from experiments.runners.static_style_runner import StaticStyleRunner
from profiles.profile_loader import ProfileLoader

# Load profile
loader = ProfileLoader()
profile = loader.load_profile("parameter_sweep")
configs = loader.expand_variations(profile)

# Run experiments
runner = StaticStyleRunner(num_runs=20)
for config in configs:
    results = runner.run_experiment_batch(
        config['name'], 
        config, 
        'nperson', 
        20
    )
```

### Analyzing Results

1. Load and compare results:
```python
from utils.results_manager import ResultsManager
from analysis.comparative_analyzer import ComparativeAnalyzer

manager = ResultsManager()

# Find experiments
experiments = manager.search_experiments(
    name_pattern="threshold_sweep"
)

# Load and analyze
analyzer = ComparativeAnalyzer()
for exp in experiments:
    results = manager.load_experiment(exp['key'])
    analyzer.add_experiment(exp['name'], results)

# Generate comparison report
report = analyzer.generate_report()
```

## Best Practices

### 1. Use the Registry for Agents
- Always register new agents with the registry
- Provide clear descriptions and parameter documentation
- Use appropriate categories

### 2. Profile-Driven Experiments
- Define experiments in profiles, not code
- Use variations for parameter sweeps
- Keep profiles version controlled

### 3. Consistent Result Management
- Always use ResultsManager for saving results
- Add meaningful categories and names
- Use the search functionality

### 4. Modular Development
- Keep modules focused on single responsibilities
- Use dependency injection where possible
- Write tests for new components

### 5. Documentation
- Document new agents and their parameters
- Add examples for new features
- Update relevant guides

## Extension Points

The architecture provides several extension points:

1. **Custom Agents**: Extend base agent classes
2. **New Visualizations**: Add to analysis module
3. **Game Variants**: Extend core game classes
4. **Analysis Pipelines**: Create custom analyzers
5. **Export Formats**: Add to results manager

## Migration from Legacy Code

If you have legacy code using the old structure:

1. Move standalone scripts to `legacy/`
2. Extract reusable components
3. Reimplement using the modular structure
4. Use the migration guide for specific patterns

## Troubleshooting

### Common Issues

1. **Agent not found**: Ensure agent is registered
2. **Profile errors**: Validate YAML syntax
3. **Import errors**: Check module paths
4. **Result not saved**: Verify ResultsManager usage

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

The modular architecture supports future additions:

- Real-time experiment monitoring
- Distributed computing support
- Web-based result viewer
- Advanced statistical analysis
- Machine learning integration

## Summary

The modular architecture provides a flexible, extensible framework for N-Person Prisoner's Dilemma research. By following the patterns and practices outlined in this guide, you can easily extend the simulator with new features while maintaining code quality and organization.