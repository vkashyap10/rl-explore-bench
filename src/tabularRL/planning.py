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

def get_unknown_dynamics_reward_var(reward_std_flat, steps_per_episode, alpha, num_states, num_actions):
    """
    transition dynamics and reward distributions are time independent.
    """
    # Broadcast reward variance across timesteps: (L, S, A)
    reward_var_stack = np.tile(reward_std_flat[None, :, :], (steps_per_episode, 1, 1))  # shape (L, S, A)

    # print("reward_var_stack", reward_var_stack.shape)
    # print("reward_variance", reward_variance.shape)

    # Compute ∑_{s'} α(s′, s, a) over next states
    alpha_sum = alpha.sum(axis=0)  # shape (S, A)

    # print("alpha_sum", alpha_sum.shape)

    # Now compute σ_p^2 for each timestep l
    reward_var_augmented = np.zeros((steps_per_episode, num_states, num_actions))

    for l in range(steps_per_episode):
        discount_term = (steps_per_episode - l) ** 2
        reward_var_augmented[l] = 3.6**2 * np.square(reward_var_stack[l]) + discount_term * alpha_sum

    return reward_var_augmented


# ------------------------------------------------------------------
#  Solve VAPOR convex program  (DCP‑compliant version)
# ------------------------------------------------------------------
def solve_vapor(reward_mean, reward_std, alpha, initial_state_distribution, steps_per_episode, num_states, num_actions, eps: float = 1e-6):
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

    Returnstotal_timesteps
    -------
    Lambda_opt : ndarray, shape (L, S, A)
        Optimal occupancy measure.
    """
    num_vars = steps_per_episode * num_states * num_actions
    occupancy = cp.Variable(num_vars, nonneg=True)      # flattened λ
    occupancy_safe = occupancy + eps
    t_aux = cp.Variable(num_vars)                   # epigraph helper
    z_aux = cp.Variable(num_vars)

    reward_mean_flat  = np.tile(reward_mean, (steps_per_episode, 1)).flatten(order="C")     # shape (L*S*A,)
    transition_probability = alpha / alpha.sum(axis=0, keepdims=True) # posterior
    reward_std_flat = np.tile(reward_std, (steps_per_episode, 1)).flatten(order="C")  # shape (L*S*A,)
    # reward_std_flat = get_unknown_dynamics_reward_var(reward_std, steps_per_episode, alpha, num_states, num_actions)
    # reward_std_flat = cp.Constant(reward_std_flat.flatten(order="C"))

    # epigraph: t_i² ≤ 2 λ_i z_i
    expr = cp.vstack([cp.sqrt(2) * t_aux, occupancy_safe - z_aux])   # (2, n)
    soc_constraint  = cp.SOC(occupancy_safe + z_aux, expr, axis=0)

    # entropy part stays the same
    entropy_constraint = z_aux <= cp.entr(occupancy_safe)

    # entropy side
    constraints = [soc_constraint, entropy_constraint]

    # --------------------------- #
    #   Flow conservation         #
    # --------------------------- #
    flow_matrix, flow_rhs = _build_flow_matrix(transition_probability, initial_state_distribution, steps_per_episode, num_states, num_actions)
    constraints += [flow_matrix @ occupancy == flow_rhs]

    # --------------------------- #
    #   Objective                 #
    # --------------------------- #
    print("reward mean", reward_mean_flat)
    print("reward_std_flat", reward_std_flat)
    print("transition_probability", transition_probability[1,1,:])
    objective = cp.Maximize(reward_mean_flat @ occupancy + reward_std_flat @ t_aux)

    optimization_problem = cp.Problem(objective, constraints)
    optimization_problem.solve(
        solver=cp.ECOS,            # exponential‑cone capable
        abstol=1e-8,
        warm_start=True,
        verbose=False
    )

    if optimization_problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"VAPOR optimisation failed")

    # reshape back to (L,S,A)
    return np.maximum(occupancy.value, 0.0).reshape((steps_per_episode, num_states, num_actions))