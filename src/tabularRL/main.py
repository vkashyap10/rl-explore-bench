import numpy as np
import gymnasium as gym
import minigrid           # <- registers all MiniGrid envs
from gymnasium import Env
from tabularRL.learning_algorithms.psrl import psrl
from tabularRL.learning_algorithms.vapor import vapor

def run_psrl_experiment(
    env:Env,
    seed=None,
    n_iters: int = 10,
    num_episodes: int = 100, # number of episodes per iteration
    # steps_per_episode: int = 20, # length of each episode
):
    num_states = 4 * env.unwrapped.width * env.unwrapped.height
    num_actions = 3 # env.action_space.n
    
    env.unwrapped.max_steps = 10
    steps_per_episode = env.unwrapped.max_steps
    print("max episode steps", steps_per_episode)

    if seed is not None:
        np.random.seed(seed)

    total_timesteps = num_episodes * steps_per_episode

    # Prior hyperparameters
    transition_dirichlet_prior = 1/num_states * np.ones((num_states, num_states, num_actions))  # Dirichlet prior
    reward_mean_prior = 0.5*np.ones((num_states, num_actions))         # assume no reward by default
    reward_mean_strength = np.ones((num_states, num_actions))      # weak prior – lets data dominate
    reward_precision_prior   = 4*np.ones((num_states, num_actions))     # prior precision
    reward_precision_strength = np.ones((num_states, num_actions))     # prior strength for variance

    # Logs
    r_psrl = np.zeros((total_timesteps, n_iters))
    true_val = np.zeros((n_iters, num_states))
    rho_list = np.zeros(n_iters)

    for i in range(n_iters):

        # rewards, states, actions, values_log, policies_log, total_visits = psrl(env,
        #     num_episodes, steps_per_episode,
        #     reward_mean_prior, reward_mean_strength, reward_precision_prior, reward_precision_strength, transition_dirichlet_prior, seed=seed
        # )

        rewards, states, actions, values_log, policies_log, total_visits = vapor(env,
            num_episodes, steps_per_episode,
            reward_mean_prior, reward_mean_strength, reward_precision_prior, reward_precision_strength, transition_dirichlet_prior, seed=seed
        )

        r_psrl[:, i] = rewards

        # rho_list[i] = V_opt[s1] * num_episodes  # cumulative value from start state

    return r_psrl, true_val, rho_list


seed = 42
# env = gym.make("MiniGrid-SimpleCrossingS9N1-v0", render_mode="human")

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")



# initial_observation = env.reset(seed = seed)
run_psrl_experiment(env, seed=42, n_iters=1, num_episodes=30000)