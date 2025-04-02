# utils.py
import numpy as np

def linear_payoff_C(n, N, R=3, S=0):
    """Linear payoff function for cooperation when n players cooperate.
    
    Payoff increases linearly with the proportion of cooperating neighbors.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        R: Reward for mutual cooperation
        S: Sucker's payoff
        
    Returns:
        Payoff value
    """
    return S + (R - S) * (n / (N - 1)) if N > 1 else R

def linear_payoff_D(n, N, T=5, P=1):
    """Linear payoff function for defection when n players cooperate.
    
    Payoff increases linearly with the proportion of cooperating neighbors.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        T: Temptation payoff
        P: Punishment for mutual defection
        
    Returns:
        Payoff value
    """
    return P + (T - P) * (n / (N - 1)) if N > 1 else P

def exponential_payoff_C(n, N, R=3, S=0, exponent=2):
    """Exponential payoff function for cooperation when n players cooperate.
    
    Payoff increases exponentially with the proportion of cooperating neighbors,
    modeling potential synergistic effects of cooperation.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        R: Reward for mutual cooperation
        S: Sucker's payoff
        exponent: Exponent controlling the curve shape
        
    Returns:
        Payoff value
    """
    if N <= 1:
        return R
    prop = n / (N - 1)
    return S + (R - S) * (prop ** exponent)

def exponential_payoff_D(n, N, T=5, P=1, exponent=2):
    """Exponential payoff function for defection when n players cooperate.
    
    Payoff increases exponentially with the proportion of cooperating neighbors.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        T: Temptation payoff
        P: Punishment for mutual defection
        exponent: Exponent controlling the curve shape
        
    Returns:
        Payoff value
    """
    if N <= 1:
        return P
    prop = n / (N - 1)
    return P + (T - P) * (prop ** exponent)

def threshold_payoff_C(n, N, R=3, S=0, threshold=0.5):
    """Threshold payoff function for cooperation when n players cooperate.
    
    Payoff increases sharply after a threshold proportion of cooperating neighbors is reached,
    modeling critical mass effects in collective action problems.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        R: Reward for mutual cooperation
        S: Sucker's payoff
        threshold: Proportion of cooperators needed for significant benefit
        
    Returns:
        Payoff value
    """
    if N <= 1:
        return R
    prop = n / (N - 1)
    if prop <= threshold:
        # Below threshold, payoff increases slowly
        return S + (R - S) * (prop / threshold) * 0.3
    else:
        # Above threshold, payoff increases rapidly
        normalized_prop = (prop - threshold) / (1 - threshold)
        return S + (R - S) * (0.3 + 0.7 * normalized_prop)

def threshold_payoff_D(n, N, T=5, P=1, threshold=0.5):
    """Threshold payoff function for defection when n players cooperate.
    
    Payoff increases sharply after a threshold proportion of cooperating neighbors is reached.
    
    Args:
        n: Number of cooperating neighbors
        N: Total number of agents
        T: Temptation payoff
        P: Punishment for mutual defection
        threshold: Proportion of cooperators needed for significant benefit
        
    Returns:
        Payoff value
    """
    if N <= 1:
        return P
    prop = n / (N - 1)
    if prop <= threshold:
        # Below threshold, payoff increases slowly
        return P + (T - P) * (prop / threshold) * 0.3
    else:
        # Above threshold, payoff increases rapidly
        normalized_prop = (prop - threshold) / (1 - threshold)
        return P + (T - P) * (0.3 + 0.7 * normalized_prop)

def get_pairwise_payoffs(move1, move2, R=3, S=0, T=5, P=1):
    """Returns payoffs for player 1 and player 2 in a 2-player PD."""
    if move1 == "cooperate" and move2 == "cooperate":
        return R, R
    elif move1 == "cooperate" and move2 == "defect":
        return S, T
    elif move1 == "defect" and move2 == "cooperate":
        return T, S
    elif move1 == "defect" and move2 == "defect":
        return P, P
    else:
        raise ValueError("Invalid moves") 

def create_payoff_matrix(N, payoff_type="linear", params=None):
    """Create a payoff matrix for an N-person IPD with the specified payoff function.
    
    Args:
        N: Number of agents
        payoff_type: Type of payoff function ("linear", "exponential", "threshold")
        params: Dictionary of payoff function parameters
        
    Returns:
        Dictionary mapping actions to lists of payoffs
    """
    params = params or {}
    
    # Default parameter values
    R = params.get("R", 3)  # Reward for mutual cooperation
    S = params.get("S", 0)  # Sucker's payoff
    T = params.get("T", 5)  # Temptation to defect
    P = params.get("P", 1)  # Punishment for mutual defection
    
    # Select payoff functions based on type
    if payoff_type == "exponential":
        exponent = params.get("exponent", 2)
        C_func = lambda n: exponential_payoff_C(n, N, R, S, exponent)
        D_func = lambda n: exponential_payoff_D(n, N, T, P, exponent)
    elif payoff_type == "threshold":
        threshold = params.get("threshold", 0.5)
        C_func = lambda n: threshold_payoff_C(n, N, R, S, threshold)
        D_func = lambda n: threshold_payoff_D(n, N, T, P, threshold)
    else:  # default to linear
        C_func = lambda n: linear_payoff_C(n, N, R, S)
        D_func = lambda n: linear_payoff_D(n, N, T, P)
    
    # Create payoff matrix
    payoff_matrix = {
        "C": [C_func(n) for n in range(N)],
        "D": [D_func(n) for n in range(N)],
    }
    
    return payoff_matrix

def plot_payoff_functions(N, payoff_types=None, params=None, figsize=(10, 6)):
    """Plot the payoff functions for visualization.
    
    Args:
        N: Number of agents
        payoff_types: List of payoff function types to plot
        params: Dictionary of parameters for each payoff type
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if payoff_types is None:
        payoff_types = ["linear", "exponential", "threshold"]
    
    params = params or {}
    
    plt.figure(figsize=figsize)
    
    # Plot cooperation payoff functions
    plt.subplot(1, 2, 1)
    x = np.linspace(0, N-1, 100)
    
    for payoff_type in payoff_types:
        payoff_matrix = create_payoff_matrix(N, payoff_type, params.get(payoff_type, {}))
        
        # Interpolate between discrete payoff values for smoother curves
        y = np.interp(x, range(N), payoff_matrix["C"])
        plt.plot(x/(N-1), y, label=f"{payoff_type}")
    
    plt.title("Cooperation Payoff Functions")
    plt.xlabel("Proportion of Cooperating Neighbors")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot defection payoff functions
    plt.subplot(1, 2, 2)
    
    for payoff_type in payoff_types:
        payoff_matrix = create_payoff_matrix(N, payoff_type, params.get(payoff_type, {}))
        
        # Interpolate between discrete payoff values for smoother curves
        y = np.interp(x, range(N), payoff_matrix["D"])
        plt.plot(x/(N-1), y, label=f"{payoff_type}")
    
    plt.title("Defection Payoff Functions")
    plt.xlabel("Proportion of Cooperating Neighbors")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt