from typing import Tuple

import mo_gymnasium as mo_gym
import numpy as np
from gymnasium import Env

from grid_world.envs.wrappers import IgnoreTermination, RandomStartPosition


def make_env(cfg, rng: np.random.Generator):
    env = mo_gym.make(cfg.env_name, render_mode=cfg.render_mode)
    env = IgnoreTermination(env)
    env = RandomStartPosition(env, rng=rng)
    return env


def flatten_grid_state(env: Env, obs):
    x, y = obs  # [x, y] position
    grid_size = env.observation_space.high - env.observation_space.low
    width, _ = grid_size[0], grid_size[1]
    return y * width + x


def extract_env_metadata(env: Env) -> Tuple[int, int, int, int]:
    """
    Extract grid dimensions, number of states, and number of actions from a tabular environment.

    Returns:
        width (int): Grid width.
        height (int): Grid height.
        num_states (int): Total number of discrete states.
        num_actions (int): Number of actions in the action space.
    """
    grid_size = env.observation_space.high - env.observation_space.low
    width, height = int(grid_size[0]), int(grid_size[1])
    num_states = width * height
    num_actions = env.action_space.n
    return width, height, num_states, num_actions
