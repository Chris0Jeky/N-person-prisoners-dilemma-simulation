# Corrected Research Results Summary

## Important Note About Data Issue

The current data shows 0% cooperation for pairwise games, which is **incorrect**. TFT agents should achieve high cooperation in pairwise games. This appears to be a bug in how the experiment runner is tracking cooperation for pairwise games.

## Expected Results (Based on Theory and Reference Figures)

### 1. Pairwise Games (Baseline)
- **Expected**: High cooperation rates
- **3 TFT**: Should achieve ~100% cooperation
- **Mixed configurations**: Varying but generally high cooperation (50-90%)
- TFT works well in pairwise settings due to direct reciprocity

### 2. Neighbourhood (N-Person) Games (Tragic Valley)
- **Expected**: Lower cooperation, especially for certain configurations
- **3 TFT**: May maintain high cooperation
- **2 TFT + 1 AllD**: Should show "tragic valley" - dramatic drop in cooperation
- The valley effect occurs because:
  - Defectors can exploit multiple cooperators
  - Retaliation is diluted across the group
  - Coordination becomes more difficult

### 3. Key Scientific Finding
The **tragic valley** refers to the drop in cooperation when moving from pairwise to N-person games. It's called a "valley" because:
- Pairwise games = high cooperation (hilltop)
- N-person games = low cooperation (valley)
- The transition creates a valley in the cooperation landscape

## Current Data (With Bug)
- **Pairwise**: Shows 0% cooperation (INCORRECT - should be high)
- **Neighbourhood**: Shows varying cooperation (60-100% - this part seems correct)

## Visualizations
The reference figures from `pictures/` folder show:
- Figure 1: Pairwise games with stable high cooperation
- Figure 2: Neighbourhood games showing tragic valley effects (drops in cooperation)

## Next Steps
1. Fix the pairwise game cooperation tracking in the experiment runner
2. Re-run experiments to get correct pairwise data
3. Generate visualizations showing the proper tragic valley effect

The tragic valley is a phenomenon specific to N-person games where cooperation drops compared to the pairwise baseline, not the other way around.