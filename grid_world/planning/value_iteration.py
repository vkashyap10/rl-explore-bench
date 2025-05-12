from typing import Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# MDP: value iteration
# -----------------------------------------------------------------------------


def bellman(
    old_val: np.ndarray,
    probs: np.ndarray,
    rewards: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bellman operator for a fixed transition tensor and reward table.

    Parameters
    ----------
    old_val : (S,)
        Previous value estimate *v*.
    probs   : (S, S, A)
        Transition probabilities `P[s', s, a].
    rewards : (S, A)
        Immediate rewards `R[s, a].
    rng     : optional
        RNG for the tiny symmetry‑breaking perturbation.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Expected future return:   E_{s'}[v(s') | s, a]
    expect = np.einsum("ijk,i->jk", probs, old_val)  # shape (S, A)
    q_vals = rewards + expect

    # Add infinitesimal noise to break ties deterministically
    q_vals += 1.0e-8 * rng.standard_normal(q_vals.shape)

    new_val = q_vals.max(axis=1)
    new_pol = q_vals.argmax(axis=1)
    return new_val, new_pol, q_vals


def dp_value_iteration(
    probs: np.ndarray,
    rewards: np.ndarray,
    tau: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite‑horizon value iteration (τ steps) for a deterministic policy.

    Parameters
    ----------
    probs   : (S, S, A) array
        Transition probabilities with MATLAB‑style indexing `P[next, current, a].
    rewards : (S, A) array
        Immediate reward for each state–action pair.
    tau     : int
        Number of Bellman backups to perform.
    rng     : optional
        NumPy RNG used by the Bellman operator.

    Returns
    -------
    value  : (S,) ndarray – final value estimate after *τ* iterations.
    policy : (S,) ndarray – greedy action indices corresponding to *value*.
    """

    S, A = rewards.shape  # noqa: F841 – A is unused but kept for clarity
    old_val = np.zeros(S)

    value: np.ndarray  # for the type checker
    policy: np.ndarray
    for _ in range(int(tau)):
        value, policy, q_vals = bellman(old_val, probs, rewards, rng)
        old_val = value

    return value, policy, q_vals
