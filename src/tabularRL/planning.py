import numpy as np
from typing import Tuple, Optional
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
        Transition probabilities ``P[s', s, a]``.
    rewards : (S, A)
        Immediate rewards ``R[s, a]``.
    rng     : optional
        RNG for the tiny symmetry‑breaking perturbation.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Expected future return:   E_{s'}[v(s') | s, a]
    expect = np.einsum("ijk,i->jk", probs, old_val)  # shape (S, A)
    q_vals = rewards + expect

    # Add infinitesimal noise to break ties deterministically (as MATLAB did)
    q_vals += 1.0e-8 * rng.standard_normal(q_vals.shape)

    new_val = q_vals.max(axis=1)
    new_pol = q_vals.argmax(axis=1)
    return new_val, new_pol


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
        Transition probabilities with MATLAB‑style indexing ``P[next, current, a]``.
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
        value, policy = bellman(old_val, probs, rewards, rng)
        old_val = value

    return value, policy


# --------------------------------------------------------------
#  Solve the VAPOR‑lite convex program and return λ*, π*
# --------------------------------------------------------------
# import numpy as np
# import cvxpy as cp
# from typing import List, Tuple

# def _solve_vapor_lite(
#     self,
#     reward_mean: List[np.ndarray],    # E[r] for each layer
#     reward_sigma: List[np.ndarray],   # epistemic std‑dev for each (s,a)
# ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     """
#     Solve the VAPOR‑lite convex program (known‑P, tabular).

#     Returns
#     -------
#     occupancies : list of optimal lambda arrays, one per layer
#     policies    : list of optimal policy arrays, one per layer
#     """
#     num_layers   = self.L
#     num_actions  = self.A
#     layer_sizes  = self.S
#     trans_kernel = self.mdp.P        # list of P_l  shape (S_l, A, S_{l+1})
#     init_state   = self.mdp.rho      # rho(s)

#     # ------------------------------------------------------------------
#     # 1. Decision variables  lambda_l(s,a) >= 0
#     # ------------------------------------------------------------------
#     lam = [cp.Variable((layer_sizes[l], num_actions), nonneg=True)
#            for l in range(num_layers)]

#     constraints = []

#     # 2a. Initial‑flow constraint:  sum_a lambda_0(s,a) == rho(s)
#     constraints.append(cp.sum(lam[0], axis=1) == init_state)

#     # 2b. Flow conservation between layers
#     for l in range(num_layers - 1):
#         rhs_per_state = []
#         for s_next in range(layer_sizes[l + 1]):
#             inflow = [
#                 trans_kernel[l][s, a, s_next] * lam[l][s, a]
#                 for s in range(layer_sizes[l])
#                 for a in range(num_actions)
#             ]
#             rhs_per_state.append(cp.sum(cp.hstack(inflow)))
#         constraints.append(cp.sum(lam[l + 1], axis=1) == cp.hstack(rhs_per_state))

#     # ------------------------------------------------------------------
#     # 3. Objective = optimism   +   weighted entropy
#     # ------------------------------------------------------------------
#     objective = 0
#     for l in range(num_layers):
#         lam_l = lam[l]
#         r_mean_l = reward_mean[l]
#         sigma_l  = reward_sigma[l]

#         # optimistic reward   sum lambda * (r_bar + sigma)
#         objective += cp.sum(cp.multiply(lam_l, r_mean_l + sigma_l))

#         # weighted entropy    sum sigma * (-lambda log lambda)
#         objective += cp.sum(cp.multiply(sigma_l, cp.entr(lam_l)))   # cp.entr(x) = -x*log x

#     problem = cp.Problem(cp.Maximize(objective), constraints)
#     problem.solve(solver=cp.ECOS, abstol=1e-7, reltol=1e-7, feastol=1e-7)

#     # ------------------------------------------------------------------
#     # 4. Extract lambda* and derive policy* by row‑normalising
#     # ------------------------------------------------------------------
#     occupancies = [lam_var.value for lam_var in lam]
#     policies = []
#     for l, lam_val in enumerate(occupancies):
#         row_sum = lam_val.sum(axis=1, keepdims=True)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             pi = lam_val / row_sum       # normalise each state row
#         zero_rows = (row_sum.flatten() == 0)
#         if np.any(zero_rows):
#             pi[zero_rows] = 1.0 / num_actions   # uniform fallback
#         policies.append(pi)

#     return occupancies, policies
