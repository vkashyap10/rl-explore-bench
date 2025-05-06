from gymnasium import Env

def flatten_grid_state(env:Env):
    x, y = env.unwrapped.agent_pos       # 2D position
    d     = env.unwrapped.agent_dir      # orientation
    W     = env.unwrapped.width
    return (y * W + x) * 4 + d # flatten (x, y), add orientation offset
