import numpy as np
from gymnasium import Env
from tqdm import trange
from tabularRL.env import flatten_grid_state
from tabularRL.sampling import sample_normal_gamma_mat
from tabularRL.planning import solve_vapor

# ------------------------------------------------------------------
#  Main VAPOR routine
# ------------------------------------------------------------------

def vapor(
    env: Env,
    num_episodes: int,
    steps_per_episode: int,
    reward_mean_prior: np.ndarray,
    reward_mean_strength: np.ndarray,
    reward_precision_prior: np.ndarray,
    reward_precision_strength: np.ndarray,
    transition_dirichlet_prior: np.ndarray,
    seed: int | None = None,
):
    """
    Tabular VAPOR (Tarbouriech et al., NeurIPS 2023)

    Returns the same tuple as your psrl(...) helper for easy drop‑in.
    """

    if seed is not None:
        np.random.seed(seed)

    S = 4 * env.unwrapped.width * env.unwrapped.height
    A = 3                               # MiniGrid action count
    L = steps_per_episode
    T = num_episodes * steps_per_episode

    # tallies
    n_sa = np.zeros((S, A), dtype=int)
    rew_mean_run = np.zeros((S, A))
    rew_var_run = np.zeros((S, A))
    trans_counts = np.zeros_like(transition_dirichlet_prior, dtype=int)

    # logs
    rewards = np.zeros(T)
    states  = np.zeros(T, dtype=int)
    actions = np.zeros(T, dtype=int)
    values_log   = np.zeros((num_episodes, S))
    policies_log = np.zeros((num_episodes, S), dtype=int)

    global_step = 0
    rho = np.zeros(S)          # start‑state distribution
    start_state_idx = None

    for ep in trange(num_episodes, desc="Training episodes", unit="ep"):

        obs, _ = env.reset(seed=seed)
        s = flatten_grid_state(env)
        if start_state_idx is None:
            start_state_idx = s
            rho[start_state_idx] = 1.0
        else:
            assert s == start_state_idx

        # Posterior expectations
        alpha = transition_dirichlet_prior + trans_counts
        P_mean = alpha / alpha.sum(axis=0, keepdims=True)

        mu_hat, var_hat = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            n_sa,
            rew_mean_run,
            rew_var_run,
            # draw_sample=False
        )
        sigma_hat = np.sqrt(var_hat)

        # Solve VAPOR
        # print("mu_hat shape", mu_hat.shape)
        Lambda_opt = solve_vapor(mu_hat, sigma_hat, P_mean, rho, L, S, A)

        # Greedy deterministic policy from first slice
        pi_star = np.argmax(Lambda_opt[0], axis=-1)
        # policies_log[ep] = pi_star

        # Execute episode
        ep_return = 0.0
        for t in range(steps_per_episode):

            pi_star = np.argmax(Lambda_opt[t], axis=-1)
            a = pi_star[s]
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            s_next = flatten_grid_state(env)

            # logs
            actions[global_step] = a
            rewards[global_step] = r
            states[global_step]  = s

            # online reward stats (Welford)
            n_sa[s, a] += 1
            n = n_sa[s, a]
            delta = r - rew_mean_run[s, a]
            rew_mean_run[s, a] += delta / n
            rew_var_run[s, a]  += delta * (r - rew_mean_run[s, a])

            # transition counts (except last step)
            if t < steps_per_episode - 1:
                trans_counts[s_next, s, a] += 1
                s = s_next

            global_step += 1
            if terminated or truncated:
                break

        print(f"Episode {ep+1:3d}/{num_episodes} | total reward: {ep_return:6.3f}")

    return rewards, states, actions, values_log, policies_log, n_sa