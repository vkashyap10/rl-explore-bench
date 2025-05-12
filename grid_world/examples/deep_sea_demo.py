from typing import Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from grid_world.algorithms.psrl import psrl
from grid_world.algorithms.vapor import vapor
from grid_world.envs.grid_utils import extract_env_metadata, make_env
from grid_world.utils.viz import (plot_exploration_uncertainty,
                                  plot_reward_curves)


def run_experiment(
    seed: Optional[int] = None,
    n_iters: int = 10,
    num_episodes: int = 100,
    steps_per_episode: int = 20,
    experience_multiplier: int = 1,
    plotting: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a Bayesian RL experiment using PSRL or VAPOR in a tabular environment.

    Args:
        env (Env): Gymnasium environment instance.
        seed (int, optional): Random seed.
        n_iters (int): Number of independent runs.
        num_episodes (int): Number of episodes per run.
        steps_per_episode (int): Maximum steps per episode.
        learning_algorithm (str): Learning algorithm, either "psrl" or "vapor".

    Returns:
        Tuple of:
            - r_psrl: Array of rewards.
            - true_val: Placeholder for value function.
            - rho_list: Placeholder array.
    """

    # Setup shapes
    rewards_psrl_all = np.zeros((n_iters, num_episodes))
    rewards_vapor_all = np.zeros((n_iters, num_episodes))
    stds_psrl_all = np.zeros((n_iters, num_episodes))
    stds_vapor_all = np.zeros((n_iters, num_episodes))

    for i in range(n_iters):
        rng = np.random.default_rng(seed + i)

        env_psrl = make_env(cfg=cfg, rng=rng)
        env_vapor = make_env(cfg=cfg, rng=rng)

        # set env parameters
        _, _, num_states, num_actions = extract_env_metadata(env_psrl)

        # Prior hyperparameters
        transition_prior = (1 / np.sqrt(num_states)) * np.ones(
            (num_states, num_states, num_actions)
        )
        reward_mean_prior = np.ones((num_states, num_actions))
        reward_mean_strength = np.ones((num_states, num_actions))
        reward_precision_prior = (1 / 20) * np.ones((num_states, num_actions))
        reward_precision_strength = np.ones((num_states, num_actions))
        initial_state_distribution = np.ones(num_states) / num_states  # uniform
        # initial_state_distribution = np.zeros(num_states)
        # initial_state_distribution[0] = 1.0

        print(f"Running PSRL, iteration {i + 1}")
        rewards_psrl, reward_stds_psrl = psrl(
            env=env_psrl,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            reward_mean_prior=reward_mean_prior,
            reward_mean_strength=reward_mean_strength,
            reward_precision_prior=reward_precision_prior,
            reward_precision_strength=reward_precision_strength,
            transition_dirichlet_prior=transition_prior,
            seed=seed,
            experience_multiplier=experience_multiplier,
            rng=rng,
            plotting=plotting,
        )
        rewards_psrl_all[i] = rewards_psrl
        stds_psrl_all[i] = reward_stds_psrl.mean(axis=(1, 2))

        print(f"Running VAPOR, iteration {i + 1}")
        rewards_vapor, reward_stds_vapor = vapor(
            env=env_vapor,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            reward_mean_prior=reward_mean_prior,
            reward_mean_strength=reward_mean_strength,
            reward_precision_prior=reward_precision_prior,
            reward_precision_strength=reward_precision_strength,
            transition_dirichlet_prior=transition_prior,
            initial_state_distribution=initial_state_distribution,
            seed=seed,
            experience_multiplier=experience_multiplier,
            rng=rng,
            plotting=plotting,
        )
        rewards_vapor_all[i] = rewards_vapor
        stds_vapor_all[i] = reward_stds_vapor.mean(axis=(1, 2))

    return rewards_psrl_all, rewards_vapor_all, stds_psrl_all, stds_vapor_all


if __name__ == "__main__":
    # Load config
    cfg = OmegaConf.load("grid_world/configs/default.yaml")

    # Run experiment
    rewards_psrl_all, rewards_vapor_all, stds_psrl_all, stds_vapor_all = run_experiment(
        seed=cfg.seed,
        n_iters=cfg.n_iters,
        num_episodes=cfg.num_episodes,
        steps_per_episode=cfg.steps_per_episode,
        experience_multiplier=cfg.experience_multiplier,
        plotting=cfg.plotting,
    )

    plot_reward_curves(rewards_psrl_all, rewards_vapor_all)
    plot_exploration_uncertainty(stds_psrl_all, stds_vapor_all)
