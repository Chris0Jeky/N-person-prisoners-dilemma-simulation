# utils.py

def C(n, N, R=3, S=0):
    """Payoff for cooperating when n players cooperate."""
    return S + (R - S) * (n / (N - 1))

def D(n, N, T=5, P=1):
    """Payoff for defecting when n players cooperate."""
    return P + (T - P) * (n / (N - 1))

def create_payoff_matrix(N):
    """
    Creates the payoff matrix for an N-person IPD.
    """
    payoff_matrix = {
        "C": [C(n, N) for n in range(N)],
        "D": [D(n, N) for n in range(N)],
    }
    return payoff_matrix