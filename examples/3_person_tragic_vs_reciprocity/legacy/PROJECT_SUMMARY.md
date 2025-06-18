# Project Summary: N-Person Prisoner's Dilemma with Q-Learning

## What This Code Does

This project simulates multi-agent prisoner's dilemma games to study cooperation emergence. It compares traditional game theory strategies (like Tit-for-Tat) with machine learning approaches (Q-learning).

### Key Components

1. **Game Environments**
   - **Pairwise**: Classic 2-player prisoner's dilemma
   - **N-Person**: 3+ player games with linear public goods payoffs

2. **Agent Types**
   - **Traditional**: TFT (Tit-for-Tat), AllC (Always Cooperate), AllD (Always Defect)
   - **Learning**: Basic Q-learning and Enhanced Q-learning agents that learn optimal strategies

3. **Experiments**
   - Tests different combinations of agents (e.g., 2 Q-learners vs 1 defector)
   - Runs multiple simulations (default: 15) for statistical validity
   - Compares cooperation rates and cumulative scores

### Key Findings

The simulations typically show:
- Traditional TFT maintains stable cooperation among similar agents
- Q-learning agents can learn to cooperate or defect based on opponents
- Enhanced Q-learning (with optimistic initialization) achieves 20-40% better cooperation
- Mixed groups show complex dynamics depending on strategy composition

### Technical Details

- **State Representation**: Q-learners use discretized cooperation ratios as states
- **Learning**: Standard Q-learning with ε-greedy exploration
- **Payoff Matrix**: Standard prisoner's dilemma (T=5, R=3, P=1, S=0)
- **Visualization**: Plots show cooperation evolution over 100 rounds

### Research Applications

This framework can be used to:
- Study emergence of cooperation in social dilemmas
- Compare learning algorithms in multi-agent settings
- Analyze stability of different strategy combinations
- Explore effects of exploration vs exploitation in social learning