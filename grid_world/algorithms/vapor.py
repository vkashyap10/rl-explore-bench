# TODO: vectorise outer seed loop so that we can run multiple seeds at once. 100 episodes should take 100 minutes.
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from tqdm import trange

from grid_world.envs.grid_utils import extract_env_metadata, flatten_grid_state
from grid_world.planning.vapor_solver import solve_vapor
from grid_world.sampling.distributions import (sample_action_from_scores,
                                               sample_normal_gamma_mat,
                                               update_obs_reward_stats)
from grid_world.utils.viz import init_reward_heatmaps, update_reward_heatmaps

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
    initial_state_distribution,
    rng: np.random.Generator,
    seed: int | None = None,
    experience_multiplier: int = 1,
    plotting: bool = False,
):
    """
    Tabular VAPOR (Tarbouriech et al., NeurIPS 2023)

    Returns the same tuple as your psrl(...) helper for easy drop‑in.
    """

    width, height, num_states, num_actions = extract_env_metadata(env)

    # tallies
    total_visits = np.zeros((num_states, num_actions), dtype=int)
    reward_mean_obs = np.zeros((num_states, num_actions))
    reward_var_obs = np.zeros((num_states, num_actions))
    transition_counts_obs = np.zeros_like(transition_dirichlet_prior, dtype=int)

    rewards = []
    reward_stds = []

    if plotting:
        plt.ion()
        figR, _, imgsR, cbarsR = init_reward_heatmaps(width, height)

    for ep in trange(num_episodes, desc="Training episodes", unit="ep"):
        obs, _ = env.reset(seed=seed)
        state = flatten_grid_state(env, obs)

        # ---------- Posterior sample ----------
        alpha = transition_dirichlet_prior + transition_counts_obs
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
        Lambda_opt = solve_vapor(
            reward_mean,
            reward_std,
            alpha,
            initial_state_distribution,
            steps_per_episode,
            num_states,
            num_actions,
        )
        if plotting:
            update_reward_heatmaps(
                imgsR, reward_mean, reward_std, width, height, ep + 1, figR, cbarsR
            )

        # ---------- Act ----------
        episode_rewards = 0.0
        for step_in_episode in range(steps_per_episode):
            action = sample_action_from_scores(
                scores=Lambda_opt[step_in_episode, state], rng=rng
            )

            obs, vector_reward, terminated, truncated, _ = env.step(action)
            reward = sum(vector_reward)
            episode_rewards += reward
            next_state = flatten_grid_state(env, obs)

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
            if step_in_episode < steps_per_episode - 1:
                transition_counts_obs[next_state, state, action] += (
                    1 * experience_multiplier
                )
                state = next_state

            if terminated or truncated:
                break

        rewards.append(episode_rewards)
        reward_stds.append(reward_std)

        print(
            f"Episode {ep+1:3d}/{num_episodes} | total reward: {episode_rewards:6.3f}"
        )

    if plotting:
        plt.ioff()
        plt.show()  # Keeps final plot open

    return np.array(rewards), np.stack(reward_stds)
