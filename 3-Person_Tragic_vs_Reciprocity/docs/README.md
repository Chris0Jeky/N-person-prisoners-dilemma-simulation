# 3-Person Tragic vs Reciprocity Documentation

Welcome to the comprehensive documentation for the N-Person Prisoner's Dilemma Simulator project.

## Documentation Structure

### 📚 Guides
- [Migration Guide](guides/MIGRATION_GUIDE.md) - Transitioning from legacy code to npd_simulator
- [Reorganization Summary](guides/REORGANIZATION_SUMMARY.md) - Overview of the codebase reorganization
- [Quick Start Guide](guides/quick_start.md) - Get started with npd_simulator quickly
- [Modularization Guide](guides/modularization.md) - Understanding the modular architecture

### 🔧 API Reference
- [Agents API](api/agents.md) - Agent classes and interfaces
- [Game Mechanics API](api/games.md) - Core game implementations
- [Analysis API](api/analysis.md) - Visualization and analysis tools
- [Configuration API](api/configuration.md) - Configuration schemas and options

### 📖 Tutorials
- [Creating Custom Agents](tutorials/custom_agents.md) - Build your own agent strategies
- [Running Experiments](tutorials/running_experiments.md) - Design and execute experiments
- [Analyzing Results](tutorials/analyzing_results.md) - Visualize and interpret results
- [Batch Processing](tutorials/batch_processing.md) - Run large-scale experiments

## Key Concepts

### N-Person Prisoner's Dilemma
The N-Person Prisoner's Dilemma extends the classic 2-player game to groups of 3 or more players. In each round, players choose to cooperate or defect, with payoffs determined by the collective choices of all players.

### Agent Strategies
The simulator supports various agent strategies:
- **TFT (Tit-for-Tat)**: Cooperates initially, then mimics previous behavior
- **pTFT (Probabilistic TFT)**: Uses group cooperation rate for N-person games
- **AllC/AllD**: Always cooperates or always defects
- **Q-Learning**: Learns optimal strategies through reinforcement learning
- **Custom Agents**: Extend base classes to implement new strategies

### Experiment Types
- **Pairwise Games**: Traditional 2-player prisoner's dilemma tournaments
- **N-Person Games**: Multi-player simultaneous interactions
- **Parameter Sweeps**: Systematic exploration of parameter spaces
- **Scaling Tests**: Performance analysis with varying group sizes

## Project Structure

```
3-Person_Tragic_vs_Reciprocity/
├── npd_simulator/          # Main simulation framework
│   ├── agents/            # Agent implementations
│   ├── core/              # Game mechanics
│   ├── experiments/       # Experiment runners
│   ├── analysis/          # Visualization tools
│   ├── configs/           # Configuration files
│   └── utils/             # Utility functions
├── legacy/                # Archived standalone scripts
├── docs/                  # This documentation
└── results/               # Experiment results
```

## Getting Started

1. **Installation**: See the [main README](../README.md) for setup instructions
2. **Quick Start**: Follow the [Quick Start Guide](guides/quick_start.md)
3. **Examples**: Check the `npd_simulator/examples/` directory
4. **Configuration**: Learn about [experiment configuration](api/configuration.md)

## Recent Updates

- **Static Style Visualization**: Added publication-ready figure generation matching legacy styles
- **Enhanced Game State**: Agent-type specific tracking for detailed analysis
- **Modular Architecture**: Complete reorganization for better maintainability
- **Plugin System**: Dynamic agent registration for easy extensions

## Contributing

When contributing to this project:
1. Follow the modular architecture patterns
2. Add appropriate documentation for new features
3. Include tests for new functionality
4. Update relevant documentation sections

## Support

- **Issues**: Report bugs on the GitHub repository
- **Discussions**: Join project discussions for questions and ideas
- **Documentation**: This documentation is continuously updated

---

Last updated: 2025