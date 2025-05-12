import numpy as np
from gymnasium import Wrapper


class IgnoreTermination(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Force `terminated` to False so it never ends early
        return obs, reward, False, truncated, info


class RandomStartPosition(Wrapper):
    def __init__(self, env, rng: np.random.Generator):
        super().__init__(env)
        self.valid_positions = self._get_valid_start_positions()
        self.rng = rng

    def _get_valid_start_positions(self):
        valid_positions = []
        grid = self.unwrapped.sea_map
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0.0:
                    valid_positions.append(np.array([y, x]))
        return valid_positions

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        start_pos = self.valid_positions[self.rng.integers(len(self.valid_positions))]
        self.unwrapped.current_state = start_pos.copy()
        obs = self.unwrapped._get_state()
        return obs, info
