import numpy as np
from typing import Tuple, Optional
import numpy as np
from tabularRL.env import flatten_grid_state

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
        value, policy = bellman(old_val, probs, rewards, rng)
        old_val = value

    return value, policy


# --------------------------------------------------------------
#  Solve the VAPOR‑lite convex program and return λ*, π*
# --------------------------------------------------------------
import numpy as np
import cvxpy as cp
from typing import Tuple

# ------------------------------------------------------------------
#  Flow‑constraint helper
# ------------------------------------------------------------------

def _build_flow_matrix(P_hat, rho, L, S, A):
    """
    Build the linear system   G @ Lambda = h   that enforces every
    **occupancy‑measure flow constraint** used by VAPOR.

    --------------------------------------------------------------
    Notation
    --------------------------------------------------------------
    • L ........ planning horizon (number of time‑steps)
    • S ........ number of flattened grid states
    • A ........ number of discrete actions
    • Lambda ... stacked occupancy vector of length  L*S*A
                 (we flatten the 3‑D tensor  Lambda[l,s,a]  in
                  row‑major order:   time‑slice ➔ state ➔ action)

    • P_hat[s_next, s, a]  ... posterior *mean* transition matrix
                               P(s_next | s,a)
    • rho[s] .............. probability of starting in state s

    --------------------------------------------------------------
    Constraints encoded
    --------------------------------------------------------------
    1)  Initial‑state occupancy (time‑slice l = 0):
        For every state s and each action a,
            Lambda_0(s,a)  must sum to  rho(s)

        Mathematically:
            ∑_a  Lambda[0,s,a] = rho[s]

        We create one row per state s.
        In that row the entries corresponding to
        0‑slice variables   Lambda[0,s,a]  are set to 1.0.

    2)  Flow conservation between consecutive slices
        (for l = 0 .. L-2):

        Incoming flow into state  s_next  at slice l+1
        equals the expected outgoing flow from slice l:

            ∑_a  Lambda_{l+1}(s_next,a)
            =  ∑_{s,a} P_hat(s_next | s,a) * Lambda_l(s,a)

        •  Left side (“incoming”): we add +1.0 on all
           variables  Lambda[l+1, s_next, a]  (one row
           for each state s_next).

        •  Right side (“outgoing”): we subtract
              P_hat[s_next, s, a]
           from every variable  Lambda[l, s, a].

        The row’s RHS constant is zero.

    --------------------------------------------------------------
    Returned values
    --------------------------------------------------------------
    G : ndarray, shape  (  S           +          (L-1)*S  ,   L*S*A )
        First  S  rows  → initial‑distribution equalities.
        Remaining rows → flow equalities for slices 1 .. L-1.

    h : ndarray, shape  (S + (L-1)*S ,)
        RHS vector containing  rho  followed by zeros.
    """
    n_var = L * S * A
    rows = []
    rhs = []

    # timestep 0 (initial distribution)
    for s in range(S):
        row = np.zeros(n_var)
        for a in range(A):
            row[s * A + a] = 1.0
        rows.append(row)
        rhs.append(rho[s])

    # flow constraints for l = 1 .. L-1
    for l in range(L - 1):
        idx_curr = l * S * A
        idx_next = (l + 1) * S * A

        for s_next in range(S):
            row = np.zeros(n_var)

            # outgoing flow from previous slice
            for s in range(S):
                for a in range(A):
                    row[idx_curr + s * A + a] -= P_hat[s_next, s, a]

            # incoming occupancy at next slice
            for a in range(A):
                row[idx_next + s_next * A + a] += 1.0

            rows.append(row)
            rhs.append(0.0)

    G = np.stack(rows, axis=0)
    h = np.array(rhs)
    return G, h


# ------------------------------------------------------------------
#  Solve VAPOR convex program
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  Solve VAPOR convex program  (DCP‑compliant version)
# ------------------------------------------------------------------
def solve_vapor(mu, sigma, P_hat, rho, L, S, A, *, eps: float = 1e-9):
    """
    Exact tabular VAPOR optimiser.

    Parameters
    ----------
    mu, sigma : ndarray, shape (S, A)
        Posterior mean and s.d. of the rewards.
    P_hat      : ndarray, shape (S, S, A)
        Posterior mean transition probabilities  P(s' | s,a).
    rho        : ndarray, shape (S,)
        Start‑state distribution.
    L, S, A    : ints
        Horizon, number of (flattened) states and actions.
    eps        : float
        Small constant to keep log(·) well‑defined.

    Returns
    -------
    Lambda_opt : ndarray, shape (L, S, A)
        Optimal occupancy measure.
    """
    n_var   = L * S * A
    Lambda  = cp.Variable(n_var, nonneg=True)      # flattened λ
    t       = cp.Variable(n_var)                   # epigraph helper
    mu_vec  = np.tile(mu, (L, 1)).flatten(order="C")     # shape (L*S*A,)
    sig_vec = np.tile(sigma, (L, 1)).flatten(order="C")  # shape (L*S*A,)

    # --------------------------- #
    #   Epigraph‑cone constraint  #
    # --------------------------- #
    # Requirement:   t_i ≥ λ_i * sqrt(‑2 log λ_i)
    #
    # Equivalent convex inequality:
    #   –λ_i log λ_i  ≥  t_i² / (2 λ_i)
    #
    # The RHS is represented with the SOC atom
    #       quad_over_lin(t_i , 2 λ_i)
    # which is DCP‑valid because  λ ↦ quad_over_lin(t, 2λ)  is convex
    # and affine in its second argument.
    # build element‑wise RHS  u_i = t_i² / (2 λ_i)
    u = cp.hstack([
            cp.quad_over_lin(t[i], 2 * (Lambda[i] + eps))
            for i in range(n_var)
        ])

    constraints = [
        u <= cp.entr(Lambda + eps)     # ← **note: no leading minus!**
    ]

    # --------------------------- #
    #   Flow conservation         #
    # --------------------------- #
    G, h = _build_flow_matrix(P_hat, rho, L, S, A)
    constraints += [G @ Lambda == h]

    # --------------------------- #
    #   Objective                 #
    # --------------------------- #
    objective = cp.Maximize(mu_vec @ Lambda + sig_vec @ t)

    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.ECOS,            # exponential‑cone capable
        abstol=1e-8,
        reltol=1e-8,
        feastol=1e-8,
        verbose=True
    )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"VAPOR optimisation failed")

    # reshape back to (L,S,A)
    return np.maximum(Lambda.value, 0.0).reshape((L, S, A))