import numpy as np

def update_running_mean_var(
    mean: np.ndarray,
    var: np.ndarray,
    count: np.ndarray,
    state: int,
    action: int,
    reward: float,
    multiplier: int = 1
):
    """Online update of reward mean and variance using Welford's algorithm."""
    count[state, action] += multiplier
    n = count[state, action]
    delta = reward - mean[state, action]
    mean[state, action] += delta / n
    var[state, action] += delta * (reward - mean[state, action])
