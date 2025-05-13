import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from tqdm import trange

from grid_world.envs.grid_utils import extract_env_metadata, flatten_grid_state
from grid_world.planning.value_iteration import dp_value_iteration
from grid_world.sampling.distributions import (sample_dirichlet_mat,
                                               sample_normal_gamma_mat,
                                               update_obs_reward_stats)
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
    transition_counts_obs = np.zeros_like(transition_dirichlet_prior, dtype=int)
    reward_mean_obs = np.zeros((num_states, num_actions))
    reward_var_obs = np.zeros((num_states, num_actions))

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
        transition_prob_sample = sample_dirichlet_mat(alpha, rng=rng)
        reward_mean_sample, _ = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            total_visits,
            reward_mean_obs,
            reward_var_obs,
            rng=rng,
        )

        # get mean statistics
        reward_mean, reward_std = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            total_visits,
            reward_mean_obs,
            reward_var_obs,
            draw_sample=False,
            rng=rng,
        )

        # ---------- Plan ----------
        value, policy, q_vals = dp_value_iteration(
            transition_prob_sample, reward_mean_sample, steps_per_episode
        )
        if plotting:
            update_reward_heatmaps(
                imgsR, reward_mean, reward_std, width, height, episode + 1, figR, cbarsR
            )

        # ---------- Act ----------
        episode_rewards = 0
        for step_in_episode in range(steps_per_episode):
            # action = sample_action_from_scores(scores=q_vals[state], rng=rng)
            action = policy[state]

            obs, vector_reward, terminated, truncated, info = env.step(action)
            reward = sum(vector_reward)
            episode_rewards += reward
            new_state = flatten_grid_state(env, obs)

            update_obs_reward_stats(
                mean=reward_mean_obs,
                var=reward_var_obs,
                total_visits=total_visits,
                state=state,
                action=action,
                reward=reward,
                multiplier=experience_multiplier,
            )

            # transition counts (except last step)
            if step_in_episode != steps_per_episode - 1:
                transition_counts_obs[new_state, state, action] += experience_multiplier
                state = new_state

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
