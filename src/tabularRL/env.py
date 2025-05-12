from gymnasium import Env
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from typing import Tuple


def flatten_grid_state(env: Env, obs):
    x, y = obs  # [x, y] position
    grid_size = env.observation_space.high - env.observation_space.low
    width, height = grid_size[0], grid_size[1]
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


class IgnoreTermination(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Force `terminated` to False so it never ends early
        return obs, reward, False, truncated, info
    

class RandomStartPosition(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.valid_positions = self._get_valid_start_positions()

    def _get_valid_start_positions(self):
        valid_positions = []
        grid = self.unwrapped.sea_map  # <-- FIXED HERE
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0.0:
                    valid_positions.append(np.array([y, x]))
        return valid_positions

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        start_pos = self.valid_positions[np.random.randint(len(self.valid_positions))]
        self.unwrapped.current_state = start_pos.copy()
        obs = self.unwrapped._get_state()
        return obs, info
