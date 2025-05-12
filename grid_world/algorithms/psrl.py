import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from tqdm import trange

from grid_world.envs.grid_utils import extract_env_metadata, flatten_grid_state
from grid_world.planning.value_iteration import dp_value_iteration
from grid_world.sampling.distributions import (sample_action_from_scores,
                                               sample_dirichlet_mat,
                                               sample_normal_gamma_mat,
                                               update_running_mean_var)
from grid_world.utils.viz import init_reward_heatmaps, update_reward_heatmaps

# ------------------------------------------------------------------
#  Main PSRL routine
# ------------------------------------------------------------------


def psrl(
    env: Env,
    num_episodes: int,
    steps_per_episode: int,
    reward_mean_prior: np.ndarray,
    reward_mean_strength: np.ndarray,
    reward_precision_prior: np.ndarray,
    reward_precision_strength: np.ndarray,
    transition_dirichlet_prior: np.ndarray,
    rng: np.random.Generator,
    seed: int | None = None,
    experience_multiplier: int = 1,
    plotting: bool = False,
):
    """
    Posterior Sampling for Reinforcement Learning (PSRL).

    At the start of every episode, the agent draws a complete Markov
    Decision Process (MDP) from its Bayesian posterior, plans an optimal
    *deterministic* policy for that sample, and then follows the policy
    for ``steps_per_episode`` steps.  After each episode the environment resets
    deterministically to state ``s1``.

    Parameters
    ----------
    num_episodes : Number of episodes to run.
    steps_per_episode : Length of each episode (time-steps per episode).
    s1 : Index of the start state at the beginning of every episode
    reward_mean_prior : (S, A) Prior mean of rewards.
    reward_mean_strength : (S, A) Equivalent sample size that ``reward_mean_prior`` represents.
    reward_precision_prior : (S, A) Prior precision (inverse variance) of rewards.
    reward_precision_strength : (S, A) Equivalent sample size that ``reward_precision_prior`` represents.
    transition_dirichlet_prior : (S, S, A) Dirichlet hyper-parameters for the transition model.
    seed : Seed for NumPy’s random number generator.  If ``None`` (default), the global RNG state is left untouched.

    References
    ----------
    Ian Osband, Daniel Russo & Benjamin Van Roy,
    *“(More) Efficient Reinforcement Learning via Posterior Sampling.”*
    NeurIPS 2013.
    """

    width, height, num_states, num_actions = extract_env_metadata(env)

    # Empirical tallies
    total_visits = np.zeros((num_states, num_actions), dtype=int)
    transition_counts_obs = np.zeros_like(
        transition_dirichlet_prior, dtype=int
    )  # empirical transition count
    reward_running_mean = np.zeros((num_states, num_actions))
    reward_running_var = np.zeros((num_states, num_actions))

    global_step = 0
    rewards = []
    reward_stds = []

    if plotting:
        plt.ion()
        figR, _, imgsR, cbarsR = init_reward_heatmaps(width, height)

    for episode in trange(num_episodes, desc="Training episodes", unit="ep"):
        obs, _ = env.reset(seed=seed)
        state = flatten_grid_state(env, obs)

        # ---------- Posterior sample ----------
        alpha = transition_dirichlet_prior + transition_counts_obs
        p_sample = sample_dirichlet_mat(alpha, rng=rng)
        reward_mean_sample, _ = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            total_visits,
            reward_running_mean,
            reward_running_var,
            rng=rng,
        )

        reward_mean, reward_std = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            total_visits,
            reward_running_mean,
            reward_running_var,
            draw_sample=False,
            rng=rng,
        )

        # ---------- Plan ----------
        state_values, policy, q_vals = dp_value_iteration(
            p_sample, reward_mean_sample, steps_per_episode
        )
        if plotting:
            update_reward_heatmaps(
                imgsR, reward_mean, reward_std, width, height, episode + 1, figR, cbarsR
            )

        # ---------- Act ----------
        episode_rewards = 0
        for step_in_episode in range(steps_per_episode):
            action = sample_action_from_scores(scores=q_vals[state], rng=rng)

            obs, vector_reward, terminated, truncated, info = env.step(action)
            reward = sum(vector_reward)
            episode_rewards += reward
            new_state = flatten_grid_state(env, obs)

            # Online reward mean/variance (Welford)
            # total_visits[state, action] += 1 * experience_multiplier
            # n = total_visits[state, action]

            # delta = reward - reward_running_mean[state, action]
            # reward_running_mean[state, action] += delta / n
            # reward_running_var[state, action]  = ((n - 1) * reward_running_var[state, action] + delta * (reward - reward_running_mean[state, action])) / n

            update_running_mean_var(
                mean=reward_running_mean,
                var=reward_running_var,
                count=total_visits,
                state=state,
                action=action,
                reward=reward,
                multiplier=experience_multiplier,
            )

            if terminated:
                transition_counts_obs[new_state, new_state, :] += (
                    1 * experience_multiplier
                )  # all actions lead to termination

            # Transition counts (skip the forced reset at the end)
            if step_in_episode != steps_per_episode - 1:
                transition_counts_obs[new_state, state, action] += (
                    1 * experience_multiplier
                )
                state = new_state

            global_step += 1

            if terminated or truncated:
                break

        rewards.append(episode_rewards)
        reward_stds.append(reward_std)

        print(
            f"Episode {episode+1:>3}/{num_episodes} | Episode reward: {episode_rewards:.4f}"
        )

    if plotting:
        plt.ioff()
        plt.show()  # Keeps final plot open

    return np.array(rewards), np.stack(reward_stds)
