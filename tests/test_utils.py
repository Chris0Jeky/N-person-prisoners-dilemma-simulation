"""
Enhanced tests for utility functions in the N-Person Prisoner's Dilemma simulation.
"""
import pytest
import numpy as np

# Assuming npdl structure allows direct import like this from tests/ dir
from npdl.core.utils import (
    linear_payoff_C, linear_payoff_D,
    exponential_payoff_C, exponential_payoff_D,
    threshold_payoff_C, threshold_payoff_D,
    create_payoff_matrix,
    get_pairwise_payoffs # Assuming this function exists if pairwise model is implemented
)


@pytest.mark.unit
class TestPayoffFunctions:
    """Test individual N-Person payoff functions."""

    @pytest.mark.parametrize("func, n, N, params, expected", [
        (linear_payoff_C, 0, 10, {"R": 3, "S": 0}, 0.0),      # C(0) = S
        (linear_payoff_C, 9, 10, {"R": 3, "S": 0}, 3.0),      # C(N-1) = R
        (linear_payoff_C, 4, 10, {"R": 3, "S": 0}, 1.3333),   # C(4/9)
        (linear_payoff_D, 0, 10, {"T": 5, "P": 1}, 1.0),      # D(0) = P
        (linear_payoff_D, 9, 10, {"T": 5, "P": 1}, 5.0),      # D(N-1) = T
        (linear_payoff_D, 4, 10, {"T": 5, "P": 1}, 2.7777),   # D(4/9)
        (exponential_payoff_C, 9, 10, {"R": 3, "S": 0, "exponent": 2}, 3.0),
        (exponential_payoff_C, 4, 10, {"R": 3, "S": 0, "exponent": 2}, 0.5925), # S+(R-S)*(4/9)^2
        (exponential_payoff_D, 9, 10, {"T": 5, "P": 1, "exponent": 2}, 5.0),
        (exponential_payoff_D, 4, 10, {"T": 5, "P": 1, "exponent": 2}, 1.7901), # P+(T-P)*(4/9)^2
        (threshold_payoff_C, 9, 10, {"R": 3, "S": 0, "threshold": 0.6}, 3.0),
        (threshold_payoff_C, 5, 10, {"R": 3, "S": 0, "threshold": 0.6}, 0.8333), # Below thresh: S+(R-S)*( (5/9)/0.6 )*0.3
        (threshold_payoff_C, 6, 10, {"R": 3, "S": 0, "threshold": 0.6}, 1.15), # Above thresh: S+(R-S)*(0.3 + 0.7*((6/9-0.6)/(1-0.6)))
        (threshold_payoff_D, 9, 10, {"T": 5, "P": 1, "threshold": 0.6}, 5.0),
        (threshold_payoff_D, 5, 10, {"T": 5, "P": 1, "threshold": 0.6}, 2.1111), # Below thresh P+(T-P)*((5/9)/0.6)*0.3
        (threshold_payoff_D, 6, 10, {"T": 5, "P": 1, "threshold": 0.6}, 2.9), # Above thresh P+(T-P)*(0.3 + 0.7*((6/9-0.6)/(1-0.6)))
        # Edge Case N=1
        (linear_payoff_C, 0, 1, {"R": 3, "S": 0}, 3.0), # Should return R
        (linear_payoff_D, 0, 1, {"T": 5, "P": 1}, 1.0), # Should return P
    ])
    def test_n_person_payoffs(self, func, n, N, params, expected):
        """Test various N-person payoff functions with specific inputs."""
        assert func(n, N, **params) == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
class TestPayoffMatrix:
    """Test payoff matrix creation."""

    @pytest.mark.parametrize("N, payoff_type, params", [
        (5, "linear", {}),
        (10, "exponential", {"exponent": 1.5}),
        (12, "threshold", {"threshold": 0.7}),
        (1, "linear", {}), # Edge case N=1
    ])
    def test_create_payoff_matrix_structure(self, N, payoff_type, params):
        """Test creating payoff matrices for different types and sizes."""
        matrix = create_payoff_matrix(N, payoff_type=payoff_type, params=params)

        assert "C" in matrix
        assert "D" in matrix
        assert isinstance(matrix["C"], list)
        assert isinstance(matrix["D"], list)
        assert len(matrix["C"]) == N
        assert len(matrix["D"]) == N
        # Check if values are numeric
        if N > 0:
             assert isinstance(matrix["C"][0], (int, float))
             assert isinstance(matrix["D"][0], (int, float))

    def test_custom_payoff_parameters(self):
        """Test payoff matrix with custom R, S, T, P parameters."""
        N = 5
        params = {"R": 4, "S": 1, "T": 6, "P": 2}
        matrix = create_payoff_matrix(N, payoff_type="linear", params=params)

        assert matrix["C"][N-1] == pytest.approx(params["R"]) # All other N-1 cooperate
        assert matrix["C"][0] == pytest.approx(params["S"])    # 0 others cooperate
        assert matrix["D"][N-1] == pytest.approx(params["T"]) # All other N-1 cooperate
        assert matrix["D"][0] == pytest.approx(params["P"])    # 0 others cooperate


# Optional: Add tests for pairwise payoff function if implemented
@pytest.mark.unit
class TestPairwisePayoffs:
    """Tests for the 2-player payoff utility function."""

    @pytest.mark.parametrize("move1, move2, params, expected_p1, expected_p2", [
        ("cooperate", "cooperate", {"R": 3, "S": 0, "T": 5, "P": 1}, 3, 3),
        ("cooperate", "defect",    {"R": 3, "S": 0, "T": 5, "P": 1}, 0, 5),
        ("defect",    "cooperate", {"R": 3, "S": 0, "T": 5, "P": 1}, 5, 0),
        ("defect",    "defect",    {"R": 3, "S": 0, "T": 5, "P": 1}, 1, 1),
        # Custom params
        ("cooperate", "cooperate", {"R": 10, "S": -1, "T": 12, "P": 0}, 10, 10),
        ("cooperate", "defect",    {"R": 10, "S": -1, "T": 12, "P": 0}, -1, 12),
        ("defect",    "cooperate", {"R": 10, "S": -1, "T": 12, "P": 0}, 12, -1),
        ("defect",    "defect",    {"R": 10, "S": -1, "T": 12, "P": 0}, 0, 0),
    ])
    def test_get_pairwise_payoffs(self, move1, move2, params, expected_p1, expected_p2):
        # Check if function exists before testing
        try:
            from npdl.core.utils import get_pairwise_payoffs
        except ImportError:
            pytest.skip("get_pairwise_payoffs function not found in utils. Skipping test.")

        p1, p2 = get_pairwise_payoffs(move1, move2, **params)
        assert p1 == expected_p1
        assert p2 == expected_p2

    def test_get_pairwise_payoffs_invalid_move(self):
         try:
            from npdl.core.utils import get_pairwise_payoffs
         except ImportError:
            pytest.skip("get_pairwise_payoffs function not found in utils. Skipping test.")

         with pytest.raises(ValueError):
             get_pairwise_payoffs("cooperate", "invalid")