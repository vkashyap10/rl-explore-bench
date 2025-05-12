import numpy as np
from gymnasium import Env
from typing import Tuple, Optional
from tabularRL.learning_algorithms.psrl import psrl
from tabularRL.learning_algorithms.vapor import vapor
import mo_gymnasium as mo_gym
from tabularRL.env import IgnoreTermination, RandomStartPosition, extract_env_metadata


def run_experiment(
    env: Env,
    seed: Optional[int] = None,
    n_iters: int = 10,
    num_episodes: int = 100,
    steps_per_episode: int = 20,
    learning_algorithm: str = "psrl",
    experience_multiplier:int = 1
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

    if seed is not None:
        np.random.seed(seed)

    # set env parameters
    _, _, num_states, num_actions = extract_env_metadata(env)

    
    total_timesteps = num_episodes * steps_per_episode

    # Prior hyperparameters
    transition_prior = (1 / np.sqrt(num_states)) * np.ones((num_states, num_states, num_actions))
    reward_mean_prior = np.ones((num_states, num_actions))
    reward_mean_strength = np.ones((num_states, num_actions))
    reward_precision_prior = (1 / 20) * np.ones((num_states, num_actions))
    reward_precision_strength = np.ones((num_states, num_actions))

    # Logging arrays
    r_psrl = np.zeros((total_timesteps, n_iters))

    for i in range(n_iters):
        if learning_algorithm == "psrl":
            psrl(
                env,
                num_episodes,
                steps_per_episode,
                reward_mean_prior,
                reward_mean_strength,
                reward_precision_prior,
                reward_precision_strength,
                transition_prior,
                seed=seed,
                experience_multiplier=experience_multiplier
            )
        elif learning_algorithm == "vapor":
            vapor(
                env,
                num_episodes,
                steps_per_episode,
                reward_mean_prior,
                reward_mean_strength,
                reward_precision_prior,
                reward_precision_strength,
                transition_prior,
                seed=seed,
                experience_multiplier=experience_multiplier
            )
        else:
            raise ValueError("Invalid learning algorithm. Choose 'psrl' or 'vapor'.")

        r_psrl[:, i] = rewards

    return r_psrl


if __name__ == "__main__":
    SEED = 42
    N_ITERS = 1
    N_EPISODES = 30000
    STEPS_PER_EPISODE = 20
    ENV_NAME = "deep-sea-treasure-v0"
    EXPERIENCE_MULTIPLIER = 100

    # Environment setup
    env = mo_gym.make(ENV_NAME, render_mode="human")
    env = IgnoreTermination(env)
    env = RandomStartPosition(env)

    # Run the experiment
    rewards, values, rhos = run_experiment(
        env,
        seed=SEED,
        n_iters=N_ITERS,
        num_episodes=N_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        learning_algorithm="vapor",
        experience_multiplier=EXPERIENCE_MULTIPLIER
    )
