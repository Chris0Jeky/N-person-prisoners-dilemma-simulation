# N-Person Reinforcement Learning Agents: Analysis and Implementation Plan

## 1. Theoretical Framework for N-Person RL

### Core Challenges in N-Person RL

#### 1.1 State Space Representation
In 2-player games, states can be simple (e.g., last action of opponent). In N-person games:
- **Option 1: Aggregate State** - Summarize all opponents (e.g., cooperation rate)
- **Option 2: Full State** - Track each opponent individually (exponential growth)
- **Option 3: Hybrid State** - Key statistics + recent history

#### 1.2 Credit Assignment Problem
- In pairwise: Clear who contributed to your payoff
- In N-person: Diffuse responsibility - hard to attribute reward to specific actions
- Solution: Use shaped rewards or difference rewards

#### 1.3 Non-Stationarity
- N-1 other agents are potentially learning simultaneously
- Environment becomes highly non-stationary
- Requires adaptive learning rates or opponent modeling

### Q-Learning in N-Person Games

#### Standard Q-Learning Adaptation
```
State representation options:
1. s = (cooperation_rate_last_round)
2. s = (num_cooperators, num_defectors)
3. s = (my_last_action, group_cooperation_rate)
4. s = (history_vector) - memory of k rounds
```

#### Expected Performance
- **Tragedy Valley (Group)**: Likely to converge to defection due to:
  - Diluted feedback signals
  - Difficulty learning cooperative equilibria
  - Exploration leading to exploitation by others
  
- **Collaborative Hill (Pairwise)**: Should perform well:
  - Clear credit assignment
  - Can learn different policies for different partners
  - Direct feedback loops

### Wolf-PHC in N-Person Games

#### Key Adaptations Needed
1. **Win/Lose Definition**: 
   - Original: Compare to expected value
   - N-person: Compare to group average or Nash equilibrium
   
2. **Learning Rate Adjustment**:
   - Win: Learn slowly (exploit current success)
   - Lose: Learn fast (adapt to group dynamics)

#### Expected Performance
- Should be more robust than standard Q-learning
- Variable learning rates help with non-stationarity
- May achieve better cooperation through cautious winning

### Hysteretic Q-Learning in N-Person Games

#### Core Principle Adaptation
- Optimistic learning: Update more on positive experiences
- In N-person: Encourages trying cooperation despite occasional defection

#### Key Parameters
```
α+ > α- (learning rates for positive/negative TD errors)
In groups: May need even larger α+/α- ratio due to noise
```

#### Expected Performance
- Best suited for overcoming Tragedy Valley
- Optimism bias helps escape defection equilibria
- May sustain cooperation better than standard Q-learning

## 2. State Representation Design

### Proposed State Representations

#### Level 1: Minimal State (Memory-1)
```python
state = (my_last_action, group_cooperation_rate)
# Example: ('C', 0.6) - I cooperated, 60% of group cooperated
```

#### Level 2: Enhanced State (Memory-k)
```python
state = (
    my_last_k_actions,
    group_cooperation_rates_last_k,
    current_round_number  # For learning scheduling
)
```

#### Level 3: Statistical State
```python
state = (
    recent_cooperation_trend,  # increasing/stable/decreasing
    my_recent_payoff_vs_average,
    cooperation_volatility  # how stable is cooperation
)
```

## 3. Implementation Architecture

### Base N-Person RL Agent
```python
class NPerson_RL_Agent:
    def __init__(self, n_agents, state_type='minimal'):
        self.n_agents = n_agents
        self.state_type = state_type
        self.q_table = {}
        
    def get_state(self, history, neighborhood_history):
        """Convert game history to RL state"""
        if self.state_type == 'minimal':
            return self._get_minimal_state(history, neighborhood_history)
        # ... other state types
        
    def update(self, state, action, reward, next_state):
        """Update Q-values based on N-person dynamics"""
        pass
```

### Neighborhood vs Pairwise Modes

#### Neighborhood Mode (True N-Person)
- State based on group statistics
- Single policy for group interaction
- Reward from group payoff

#### Pairwise Mode (Decomposed)
- Separate Q-tables per partner OR
- State includes partner identity OR
- Single policy with partner features

## 4. Performance Predictions

### Q-Learning
- **Neighborhood**: 20-40% cooperation (converges to defection)
- **Pairwise**: 60-80% cooperation (learns reciprocity)

### Wolf-PHC
- **Neighborhood**: 30-50% cooperation (cautious cooperation)
- **Pairwise**: 70-85% cooperation (adaptive reciprocity)

### Hysteretic Q
- **Neighborhood**: 40-60% cooperation (optimism helps)
- **Pairwise**: 75-90% cooperation (forgives and cooperates)

## 5. Key Differences from Current Implementation

### Current Issues
1. State representation not adapted for N-person
2. No group-specific reward shaping
3. Learning parameters not tuned for group dynamics
4. No handling of non-stationarity from multiple learners

### Required Changes
1. Implement group-aware state representations
2. Add shaped rewards for group cooperation
3. Tune learning parameters for N-person dynamics
4. Add opponent modeling or awareness

## 6. Testing Plan

### Phase 1: Unit Tests
- State representation correctness
- Q-value updates in group settings
- Learning rate adjustments (Wolf-PHC)

### Phase 2: Simple Scenarios
1. **All Cooperators**: RL should learn to cooperate
2. **All Defectors**: RL should learn to defect
3. **Mixed Static**: RL should find best response
4. **Learning Opponents**: Test convergence

### Phase 3: Complex Scenarios
- Multiple RL agents learning simultaneously
- Mixed strategy populations
- Noisy environments
- Various group sizes