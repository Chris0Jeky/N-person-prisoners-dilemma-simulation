"""
Payoff Matrix for Prisoner's Dilemma Games
"""

from typing import Tuple


class PayoffMatrix:
    """
    Configurable payoff matrix for Prisoner's Dilemma games.
    
    Standard values:
    - R (Reward): Both cooperate
    - S (Sucker): I cooperate, opponent defects
    - T (Temptation): I defect, opponent cooperates
    - P (Punishment): Both defect
    
    Must satisfy: T > R > P > S and 2R > T + S
    """
    
    def __init__(self, 
                 R: float = 3.0,
                 S: float = 0.0, 
                 T: float = 5.0,
                 P: float = 1.0):
        """
        Initialize payoff matrix.
        
        Args:
            R: Reward for mutual cooperation
            S: Sucker's payoff (cooperate vs defect)
            T: Temptation to defect (defect vs cooperate)
            P: Punishment for mutual defection
        """
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        
        # Validate payoff constraints
        if not (T > R > P > S):
            raise ValueError(f"Payoff values must satisfy T > R > P > S. Got T={T}, R={R}, P={P}, S={S}")
        if not (2 * R > T + S):
            raise ValueError(f"Payoff values must satisfy 2R > T + S. Got 2R={2*R}, T+S={T+S}")
    
    def linear_cooperator_payoff(self, cooperation_ratio: float) -> float:
        """
        Calculate payoff for a cooperator in N-person game.
        
        Linear interpolation based on proportion of other cooperators.
        
        Args:
            cooperation_ratio: Proportion of other agents cooperating (0 to 1)
            
        Returns:
            Payoff value
        """
        return self.S + (self.R - self.S) * cooperation_ratio
    
    def linear_defector_payoff(self, cooperation_ratio: float) -> float:
        """
        Calculate payoff for a defector in N-person game.
        
        Linear interpolation based on proportion of other cooperators.
        
        Args:
            cooperation_ratio: Proportion of other agents cooperating (0 to 1)
            
        Returns:
            Payoff value
        """
        return self.P + (self.T - self.P) * cooperation_ratio
    
    def get_pairwise_payoff(self, my_action: int, opponent_action: int) -> float:
        """
        Get payoff for pairwise (2-player) interaction.
        
        Args:
            my_action: 0 for cooperate, 1 for defect
            opponent_action: 0 for cooperate, 1 for defect
            
        Returns:
            My payoff
        """
        if my_action == 0:  # I cooperate
            return self.R if opponent_action == 0 else self.S
        else:  # I defect
            return self.T if opponent_action == 0 else self.P
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'R': self.R,
            'S': self.S,
            'T': self.T,
            'P': self.P
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PayoffMatrix':
        """Create from dictionary representation."""
        return cls(**data)