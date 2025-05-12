# TODO: fix variable names and add "we accelerate learning by imagining that each experienced transition (s, a, s′, r) is repeated 100 times.""
# TODO: vectorise outer seed loop so that we can run multiple seeds at once. 100 episodes should take 100 minutes.
# TODO: warm start? compile problem once?
# It feels like reward var is too little (look at plots)

import numpy as np
from gymnasium import Env
from tqdm import trange
from tabularRL.env import flatten_grid_state, extract_env_metadata
from tabularRL.sampling import sample_normal_gamma_mat, sample_action_from_scores
from tabularRL.planning import solve_vapor
from tabularRL.viz import update_reward_heatmaps, init_reward_heatmaps
import matplotlib.pyplot as plt
from tabularRL.common import update_running_mean_var

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
    seed: int | None = None,
    experience_multiplier:int =1
):
    """
    Tabular VAPOR (Tarbouriech et al., NeurIPS 2023)

    Returns the same tuple as your psrl(...) helper for easy drop‑in.
    """

    if seed is not None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed)

    width, height, num_states, num_actions = extract_env_metadata(env)

    # tallies
    total_visits = np.zeros((num_states, num_actions), dtype=int)
    reward_running_mean = np.zeros((num_states, num_actions))
    reward_running_var = np.zeros((num_states, num_actions))
    transition_counts_obs = np.zeros_like(transition_dirichlet_prior, dtype=int)

    global_step = 0
    initial_state_distribution = np.ones(num_states)/num_states          #uniform

    plt.ion()
    figR, axesR, imgsR, cbarsR = init_reward_heatmaps(width, height)

    
    for ep in trange(num_episodes, desc="Training episodes", unit="ep"):

        obs, _ = env.reset(seed=seed)
        state = flatten_grid_state(env, obs)

        # Posterior expectations
        alpha = transition_dirichlet_prior + transition_counts_obs
        reward_mean, reward_std = sample_normal_gamma_mat(
            reward_mean_prior,
            reward_mean_strength,
            reward_precision_prior,
            reward_precision_strength,
            total_visits,
            reward_running_mean,
            reward_running_var,
            draw_sample=False
        )

        Lambda_opt = solve_vapor(reward_mean, reward_std, alpha, initial_state_distribution, steps_per_episode, num_states, num_actions)        
        update_reward_heatmaps(imgsR, reward_mean, reward_std, width, height, ep + 1, figR, cbarsR)

        # Execute episode
        episode_rewards = 0.0
        for t in range(steps_per_episode):

            action = sample_action_from_scores(scores = Lambda_opt[t, state], rng=rng)

            obs, vector_reward, terminated, truncated, _ = env.step(action)
            reward = sum(vector_reward)
            episode_rewards += reward
            next_state = flatten_grid_state(env, obs)

            # online reward stats (Welford)
            # total_visits[state, action] += 1 * experience_multiplier
            # delta = reward - reward_running_mean[state, action]
            # reward_running_mean[state, action] += delta / total_visits[state, action]
            # reward_running_var[state, action]  += delta * (reward - reward_running_mean[state, action])

            update_running_mean_var(
                                    mean=reward_running_mean,
                                    var=reward_running_var,
                                    count=total_visits,
                                    state=state,                # current state index
                                    action=action,              # action just executed
                                    reward=reward,              # scalar reward you observed
                                    multiplier=experience_multiplier   # usually 1, >1 if you’re “imagining” extra repeats
                                )

            # the transition probabilities should update if episode is terminated. 0 for all states.
            if terminated:
                transition_counts_obs[next_state, next_state, :] += 1 * experience_multiplier # all actions lead to termination

            # transition counts (except last step)
            if t < steps_per_episode - 1:
                transition_counts_obs[next_state, state, action] += 1 * experience_multiplier
                state = next_state

            global_step += 1
            if terminated or truncated:
                break

        print(f"Episode {ep+1:3d}/{num_episodes} | total reward: {episode_rewards:6.3f}")

    return