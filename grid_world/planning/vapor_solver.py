# --------------------------------------------------------------
#  Solve the VAPOR‑lite convex program and return λ*, π*
# --------------------------------------------------------------
import cvxpy as cp
import numpy as np


# ------------------------------------------------------------------
#  Flow‑constraint helper
# ------------------------------------------------------------------
def get_occupancy_constraints(
    transition_probability,
    initial_state_distribution,
    steps_per_episode,
    num_states,
    num_actions,
):
    n_var = steps_per_episode * num_states * num_actions
    rows = []
    rhs = []

    # timestep 0 (initial distribution)
    for s in range(num_states):
        row = np.zeros(n_var)
        for a in range(num_actions):
            row[s * num_actions + a] = 1.0
        rows.append(row)
        rhs.append(initial_state_distribution[s])

    # occupancy constraints for l = 1 .. L-1
    for time_step in range(steps_per_episode - 1):
        idx_curr = time_step * num_states * num_actions
        idx_next = (time_step + 1) * num_states * num_actions

        for s_next in range(num_states):
            row = np.zeros(n_var)

            # outgoing flow from previous slice
            for s in range(num_states):
                for a in range(num_actions):
                    row[idx_curr + s * num_actions + a] -= transition_probability[
                        s_next, s, a
                    ]

            # incoming occupancy at next slice
            for a in range(num_actions):
                row[idx_next + s_next * num_actions + a] += 1.0

            rows.append(row)
            rhs.append(0.0)

    G = np.stack(rows, axis=0)
    h = np.array(rhs)
    return G, h


# ------------------------------------------------------------------
#  Solve VAPOR convex program
# ------------------------------------------------------------------


def get_unknown_dynamics_reward_var(
    reward_std_flat, steps_per_episode, alpha, num_states, num_actions
):
    """
    transition dynamics and reward distributions are time independent.
    """
    # Broadcast reward variance across timesteps: (L, S, A)
    reward_var_stack = np.tile(reward_std_flat[None, :, :], (steps_per_episode, 1, 1))

    # Compute ∑_{s'} α(s′, s, a) over next states
    alpha_sum = alpha.sum(axis=0)  # shape (S, A)

    # Now compute σ_p^2 for each timestep l
    reward_std_augmented = np.zeros((steps_per_episode, num_states, num_actions))

    for time_step in range(steps_per_episode):
        discount_term = (steps_per_episode - time_step) ** 2
        reward_std_augmented[time_step] = np.sqrt(
            3.6**2 * np.square(reward_var_stack[time_step])
            + discount_term / alpha_sum
        )

    return reward_std_augmented


# ------------------------------------------------------------------
#  Solve VAPOR convex program  (DCP‑compliant version)
# ------------------------------------------------------------------
def solve_vapor(
    reward_mean,
    reward_std,
    alpha,
    initial_state_distribution,
    steps_per_episode,
    num_states,
    num_actions,
    unknown_transition_dynamics=True,
    eps: float = 1e-6,
):
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
    occupancy = cp.Variable(num_vars, nonneg=True)  # flattened λ
    occupancy_safe = occupancy + eps
    t_aux = cp.Variable(num_vars)  # epigraph helper
    z_aux = cp.Variable(num_vars)

    # shape (L*S*A,)
    reward_mean_flat = np.tile(reward_mean, (steps_per_episode, 1)).flatten(order="C")

    if unknown_transition_dynamics:
        reward_std_flat = get_unknown_dynamics_reward_var(
            reward_std, steps_per_episode, alpha, num_states, num_actions
        )
        reward_std_flat = cp.Constant(reward_std_flat.flatten(order="C"))
    else:
        reward_std_flat = np.tile(reward_std, (steps_per_episode, 1)).flatten(order="C")

    transition_probability = alpha / alpha.sum(axis=0, keepdims=True)  # posterior

    # epigraph: t_i² ≤ 2 λ_i z_i
    expr = cp.vstack([cp.sqrt(2) * t_aux, occupancy_safe - z_aux])  # (2, n)
    soc_constraint = cp.SOC(occupancy_safe + z_aux, expr, axis=0)

    # entropy part stays the same
    entropy_constraint = z_aux <= cp.entr(occupancy_safe)

    # entropy side
    constraints = [soc_constraint, entropy_constraint]

    # --------------------------- #
    #   Occupancy constriant      #
    # --------------------------- #
    occ_constraint_matrix, occ_constraint_rhs = get_occupancy_constraints(
        transition_probability,
        initial_state_distribution,
        steps_per_episode,
        num_states,
        num_actions,
    )
    constraints += [occ_constraint_matrix @ occupancy == occ_constraint_rhs]

    # --------------------------- #
    #   Objective                 #
    # --------------------------- #
    objective = cp.Maximize(reward_mean_flat @ occupancy + reward_std_flat @ t_aux)

    optimization_problem = cp.Problem(objective, constraints)
    optimization_problem.solve(
        solver=cp.ECOS,
        abstol=1e-8,
        warm_start=True,
        verbose=False,
    )

    if optimization_problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("VAPOR optimisation failed")

    # reshape back to (L,S,A)
    return np.maximum(occupancy.value, 0.0).reshape(
        (steps_per_episode, num_states, num_actions)
    )
