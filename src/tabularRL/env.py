from gymnasium import Env

def flatten_grid_state(env:Env):
    x, y = env.unwrapped.agent_pos       # 2D position
    d     = env.unwrapped.agent_dir      # orientation
    W     = env.unwrapped.width
    return (y * W + x) * 4 + d # flatten (x, y), add orientation offset


def flatten_grid_state_mo(env: Env, obs):
    x, y = obs  # [x, y] position
    grid_size = env.observation_space.high - env.observation_space.low
    width, height = grid_size[0], grid_size[1]
    return y * width + x