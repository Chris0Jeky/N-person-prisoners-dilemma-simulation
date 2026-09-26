"""Tournament agents ported exactly from the versioned 3-person experiments.

Provenance (W3 consolidation): this module is a line-faithful port of
``3-Person_Tragic_vs_Reciprocity/final_experimentations/v9/final_agents.py``
(identical across v5/v8/v9; v6/v7 differ only in str-vs-int action keys).
Class names, method names, defaults, state encodings, update rules, and RNG
call order are preserved exactly so behavior matches the legacy
implementation on identical seeded inputs. Only the module docstring and
these provenance notes were added.

One deliberate extension: ``StaticAgent`` accepts an optional
``error_decay_rate`` keyword (default ``None`` = v9 behavior) reproducing
the TFT-E error decay from
``v7/submittable_code/final_agents_v2.py`` (``decay_rate=0.9995``).

Conventions (legacy, kept as-is): moves are ints (``COOPERATE=0``,
``DEFECT=1``); randomness uses the ``random`` module.
"""

import random
from collections import defaultdict, deque

import numpy as np

# --- Constants ---
COOPERATE, DEFECT = 0, 1


# --- Base Agents ---
class BaseAgent:
    """Minimal shared agent interface for tournament entries."""
    def __init__(self, agent_id, strategy_name):
        self.agent_id, self.strategy_name = agent_id, strategy_name
        self.total_score = 0

    def reset(self):
        """Reset the cumulative score for a new run."""
        self.total_score = 0


class StaticAgent(BaseAgent):
    """Static baseline agent with a fixed strategy and execution noise."""
    def __init__(self, agent_id, strategy_name="TFT", error_rate=0.0, **kwargs):
        super().__init__(agent_id, strategy_name)
        self.strategy_name = strategy_name
        self.error_rate = error_rate
        self.opponent_last_moves = {}
        self.last_neighborhood_move = COOPERATE
        # TFT-E error-decay extension (v7/submittable_code/final_agents_v2.py).
        # None (default) preserves v9 behavior exactly; 0.9995 reproduces v2.
        self.initial_error_rate = error_rate
        self.round_count = 0
        self.error_decay_rate = kwargs.get("error_decay_rate", None)

    def _apply_error(self, intended_move):
        """Apply error rate to the intended move"""
        if (self.strategy_name == "TFT-E" and self.initial_error_rate > 0
                and self.error_decay_rate is not None):
            self.round_count += 1
            # Exponential decay: error_rate = initial_rate * decay_rate^round
            self.error_rate = self.initial_error_rate * (self.error_decay_rate ** self.round_count)
            # Stop decaying when we get very close to 0
            if self.error_rate < 0.001:
                self.error_rate = 0.0
        if random.random() < self.error_rate:
            return random.choice([COOPERATE, DEFECT])
        return intended_move

    def choose_pairwise_action(self, opponent_id):
        """Choose the fixed-strategy move for a pairwise encounter."""
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)
        else:  # Default TFT
            intended = self.opponent_last_moves.get(opponent_id, COOPERATE)

        return self._apply_error(intended)

    def choose_neighborhood_action(self, coop_ratio):
        """Choose the static-strategy move for the neighborhood ratio."""
        if self.strategy_name == "AllC":
            intended = COOPERATE
        elif self.strategy_name == "AllD":
            intended = DEFECT
        elif self.strategy_name == "Random":
            intended = random.choice([COOPERATE, DEFECT])
        elif self.strategy_name == "TFT" or self.strategy_name == "TFT-E":
            # TFT in neighborhood: probabilistic cooperation based on cooperation ratio
            if coop_ratio is None:
                intended = COOPERATE  # Cooperate on first round
            else:
                # Cooperate with probability equal to cooperation ratio
                intended = COOPERATE if random.random() < coop_ratio else DEFECT
        else:  # Default TFT
            if coop_ratio is None:
                intended = COOPERATE
            else:
                # Probabilistic cooperation based on cooperation ratio
                intended = COOPERATE if random.random() < coop_ratio else DEFECT

        return self._apply_error(intended)

    def record_pairwise_outcome(self, opponent_id, my_move, opponent_move, reward):
        """Bank the reward and note the opponent move for a new run."""
        self.total_score += reward
        self.opponent_last_moves[opponent_id] = opponent_move

    def record_neighborhood_outcome(self, coop_ratio, reward):
        """Bank the reward and note the neighborhood move for a new run."""
        self.total_score += reward
        self.last_neighborhood_move = COOPERATE if coop_ratio and coop_ratio >= 0.5 else DEFECT

    def reset(self):
        """Clear tracked opponent moves and restore the default move."""
        super().reset()
        self.opponent_last_moves.clear()
        self.last_neighborhood_move = COOPERATE
        self.round_count = 0
        self.error_rate = self.initial_error_rate
