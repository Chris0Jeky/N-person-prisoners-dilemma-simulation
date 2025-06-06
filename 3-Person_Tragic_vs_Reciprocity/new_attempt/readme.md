# Iterated Prisoner's Dilemma: Pairwise vs. N-Person Models

This project explores cooperative dynamics in the Iterated Prisoner's Dilemma (IPD) using two distinct interaction models: a traditional Pairwise model and an N-Person (or "neighbourhood") model. It investigates how different agent strategies, population compositions, and environmental factors like noise (exploration rate) and memory resets (episodes) affect cooperation levels and agent scores.

## Models Implemented

1.  **Pairwise Model:**
    *   Agents interact in pairs over multiple rounds.
    *   Each agent plays against every other agent in the simulation.
    *   **Episodic Gameplay (Optional):** Interactions can be divided into episodes. At the end of each episode, Tit-For-Tat (TFT) agents reset their memory of their opponent's last move for that specific pairing, allowing for a "fresh start" in the next episode.

2.  **N-Person (Neighbourhood) Model:**
    *   All agents in the simulation play in a single group simultaneously.
    *   An agent's payoff depends on their own move and the number of other agents in the group who chose to cooperate in that round.
    *   The payoff functions are linear, based on the proportion of cooperators among the *other* agents:
        *   Cooperator's Payoff: `S + (R - S) * (n_others_cooperated / (N - 1))`
        *   Defector's Payoff: `P + (T - P) * (n_others_cooperated / (N - 1))`

## Agent Strategies

1.  **Tit-For-Tat (TFT) - Pairwise Model:**
    *   Cooperates on the first move against an opponent (or the first move of an episode).
    *   Subsequently, mimics the opponent's last actual move against it.

2.  **Always Defect (AllD) - Both Models:**
    *   Always chooses to defect.

3.  **Probabilistic Tit-For-Tat (pTFT) - N-Person Model:**
    *   Cooperates on the first round of the simulation.
    *   Subsequently, cooperates with a probability equal to the overall cooperation ratio of *all* agents in the previous round.

4.  **Probabilistic Tit-For-Tat with Threshold (pTFT-Threshold) - N-Person Model:**
    *   Cooperates on the first round.
    *   If the previous round's overall cooperation ratio was >= 50%, it cooperates.
    *   If the ratio was < 50%, it cooperates with a probability linearly scaled from 0% (at 0% overall coop) to 100% (at just under 50% overall coop). The formula used is `Prob_Cooperate = Previous_Coop_Ratio / 0.5`.

## Key Parameters & Features

*   **Payoff Matrix (Classic):**
    *   Mutual Cooperation (R,R): (3,3)
    *   Cooperate-Defect (S,T): (0,5) (Cooperator gets Sucker's payoff S=0, Defector gets Temptation T=5)
    *   Defect-Cooperate (T,S): (5,0)
    *   Mutual Defection (P,P): (1,1)
*   **Exploration Rate (Error Chance):** A configurable probability (e.g., 0%, 10%) that an agent will make the opposite move to what its strategy dictates. This introduces noise into the system.
*   **Agent Compositions:** Simulations were run with different mixes of TFT (or its N-Person variants) and AllD agents (e.g., 3 TFTs; 2 TFTs & 1 AllD; 2 TFTs & 3 AllD).
*   **Number of Rounds:**
    *   Pairwise: 100 total rounds per pair (e.g., 1 episode of 100 rounds, or 10 episodes of 10 rounds).
    *   N-Person: 200 total rounds.

## Experiments Conducted

A series of experiments were run to cover combinations of:
*   **Exploration Rates:** 0%, 10%
*   **Agent Compositions:**
    *   3 TFTs
    *   2 TFTs, 1 AllD
    *   2 TFTs, 3 AllD
*   **Pairwise Model Settings:**
    *   Non-Episodic (1 episode of 100 rounds)
    *   Episodic (10 episodes of 10 rounds)
*   **N-Person Model TFT Variants:**
    *   pTFT (standard probabilistic)
    *   pTFT-Threshold

## Summary of Key Findings

### 1. Impact of Exploration Rate (Noise)
*   **0% Exploration:**
    *   Pure TFT groups achieve perfect cooperation in both Pairwise and N-Person models (with either pTFT variant).
    *   With AllD agents:
        *   **Pairwise:** TFTs effectively isolate and punish AllD(s) after an initial cooperative move while maintaining cooperation among themselves.
        *   **N-Person (pTFT):** The presence of even one AllD quickly leads to a collapse of cooperation, as pTFT agents' probability of cooperating drops with the overall cooperation ratio.
        *   **N-Person (pTFT-Threshold):** Showed remarkable resilience. In a 2 TFTs/1 AllD scenario, TFTs maintained perfect cooperation with each other because the initial 2/3 cooperation ratio kept them above the 50% threshold for deterministic cooperation. AllD scored very high by exploiting this. This effect diminished with more AllDs.
*   **10% Exploration:**
    *   Noise generally degrades cooperation across all models by causing unintended defections, which can trigger retaliation.
    *   AllD agents often benefit from noise as it can disrupt established cooperation among TFTs.

### 2. Pairwise Model: Episodic vs. Non-Episodic
*   **No Exploration:** Episodic resets have little impact on groups of pure TFTs (they remain cooperative). Against AllD, episodic resets lead to AllD scoring higher as TFTs repeatedly "forgive" and offer initial cooperation at the start of each episode.
*   **With Exploration (10%):**
    *   **Episodic mode significantly boosted cooperation and scores for TFTs when playing among themselves.** The memory reset helps break cycles of retaliation caused by errors, allowing cooperation to be re-established more easily.
    *   Against AllD agents, the benefit is less clear-cut for TFTs, as AllD continues to exploit the reset.

### 3. N-Person Model: pTFT vs. pTFT-Threshold
*   **pTFT (Standard):** Highly sensitive to the overall cooperation level. Prone to a rapid decline in cooperation if defectors are present or if noise pushes the cooperation ratio down.
*   **pTFT-Threshold:**
    *   **Significantly more robust in maintaining cooperation, especially among groups of reciprocal agents, even with noise.** The 50% threshold for deterministic cooperation acts as a strong stabilizing force.
    *   In the 2 TFTs/1 AllD (0% exploration) scenario, it led to sustained cooperation between the TFTs despite the AllD.
    *   With 10% exploration and 3 TFTs, it achieved very high cooperation (~86-87%).
    *   Its effectiveness decreases as the proportion of AllD agents increases, making it harder for the group to maintain the >=50% cooperation threshold.

### 4. Impact of Agent Composition
*   **More AllD Agents:** Unsurprisingly, increasing the number of AllD agents makes it harder to establish and maintain cooperation in all models and settings.
    *   In Pairwise, TFTs still try to cooperate with each other, but their overall scores and cooperation rates decrease due to more interactions with AllDs.
    *   In N-Person, a higher proportion of AllDs makes it very difficult for either pTFT or pTFT-Threshold to sustain cooperation, as the baseline group cooperation ratio is pulled down significantly.

### 5. General Observations
*   **Pairwise TFT** is effective at targeted reciprocity but can get locked into long defection cycles by noise without episodic resets.
*   **N-Person pTFT-Threshold** emerges as a surprisingly strong strategy for promoting group cooperation in the N-Person model, especially when the majority of the group is willing to cooperate or when noise is present.
*   The N-Person model, due to its global feedback mechanism (overall cooperation ratio influencing individual decisions), can lead to more rapid shifts in group behavior (either widespread cooperation or collapse) compared to the more localized interactions of the Pairwise model.

## How to Run

The experiments are driven by a main runner script (`experiment_runner.py` if saved as suggested) which imports functionalities from `main_pairwise.py` and `main_neighbourhood.py`. Ensure all three files are in the same directory.

To run the simulations:
```bash
python experiment_runner.py