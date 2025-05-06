import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from tqdm import trange
from tabularRL.sampling import sample_dirichlet_mat, sample_normal_gamma_mat
from tabularRL.planning import dp_value_iteration
from tabularRL.viz import *
from tabularRL.env import flatten_grid_state

# ------------------------------------------------------------------
#  Main PSRL routine
# ------------------------------------------------------------------

def psrl(
    env:Env,
    num_episodes: int,
    steps_per_episode: int,
    reward_mean_prior: np.ndarray,
    reward_mean_strength: np.ndarray,
    reward_precision_prior: np.ndarray,
    reward_precision_strength: np.ndarray,
    transition_dirichlet_prior: np.ndarray,
    seed: int | None = None,
):
    """
    Posterior Sampling for Reinforcement Learning (PSRL).

    At the start of every episode, the agent draws a complete Markov
    Decision Process (MDP) from its Bayesian posterior, plans an optimal
    *deterministic* policy for that sample, and then follows the policy
    for ``steps_per_episode`` steps.  After each episode the environment resets
    deterministically to state ``s1``.

    Parameters
    ----------
    num_episodes : Number of episodes to run.
    steps_per_episode : Length of each episode (time-steps per episode).
    s1 : Index of the start state at the beginning of every episode
    reward_mean_prior : (S, A) Prior mean of rewards.
    reward_mean_strength : (S, A) Equivalent sample size that ``reward_mean_prior`` represents.
    reward_precision_prior : (S, A) Prior precision (inverse variance) of rewards.
    reward_precision_strength : (S, A) Equivalent sample size that ``reward_precision_prior`` represents.
    transition_dirichlet_prior : (S, S, A) Dirichlet hyper-parameters for the transition model.
    seed : Seed for NumPy’s random number generator.  If ``None`` (default), the global RNG state is left untouched.

    Returns
    -------
    rewards : (T,) Scalar reward obtained at every time-step, where ``T = num_episodes * steps_per_episode``.
    states : (T,) State visited at every time-step.
    actions : (T,) Action taken at every time-step.
    values_log : (num_episodes, S) State-value function ``V`` computed for the sampled MDP at the start of each episode.
    policies_log : (num_episodes, S) Deterministic policy used in each episode (``policies_log[ep, s]`` is the action index chosen in state *s*).
    total_visits : (S, A) Cumulative state–action visit counts over the entire run.


    References
    ----------
    Ian Osband, Daniel Russo & Benjamin Van Roy,
    *“(More) Efficient Reinforcement Learning via Posterior Sampling.”*
    NeurIPS 2013.
    """

    if seed is not None:
        np.random.seed(seed)

    num_states = 4 * env.unwrapped.width * env.unwrapped.height
    num_actions = 3 # env.action_space.n
    total_timesteps = num_episodes * steps_per_episode # total number of steps

    # Empirical tallies
    episode_visits  = np.zeros((num_states, num_actions), dtype=int)
    total_visits  = np.zeros((num_states, num_actions), dtype=int)
    transition_counts_obs  = np.zeros_like(transition_dirichlet_prior, dtype=int) # empirical transition count
    reward_running_mean = np.zeros((num_states, num_actions))
    reward_running_var  = np.zeros((num_states, num_actions))

    # Logs
    rewards  = np.zeros(total_timesteps)
    states   = np.zeros(total_timesteps, dtype=int)
    actions  = np.zeros(total_timesteps, dtype=int)
    values_log   = np.zeros((num_episodes, num_states))
    policies_log = np.zeros((num_episodes, num_states), dtype=int)

    global_step = 0

    # plt.ion()
    # figV, axesV, imgsV = init_value_heatmaps(env.unwrapped.width, env.unwrapped.height)
    # figR, axesR, imgsR, cbarsR = init_reward_heatmaps(env.unwrapped.width, env.unwrapped.height)
    # figP, axesP, imgsP = init_transition_heatmaps(env.unwrapped.width, env.unwrapped.height)

    print("total episodes: ", num_episodes)
    print("episode length: ", steps_per_episode)
    print("initial pos:", env.unwrapped.agent_pos)
    print("grid: ", env.unwrapped.width)
    print("grid: ", env.unwrapped.height)
    for episode in trange(num_episodes, desc="Training episodes", unit="ep"):

        
        obs, _ = env.reset(seed=seed)
        state = flatten_grid_state(env)

        episode_visits.fill(0)     # reset per-episode counts

        # ---------- Posterior sample ----------
        alpha = transition_dirichlet_prior + transition_counts_obs
        p_sample = sample_dirichlet_mat(alpha)
        mu_sample, _ = sample_normal_gamma_mat(reward_mean_prior, reward_mean_strength, reward_precision_prior,
                                               reward_precision_strength, total_visits, reward_running_mean, reward_running_var)

        # ---------- Plan ----------
        state_values, policy = dp_value_iteration(p_sample, mu_sample, steps_per_episode)
        values_log[episode]   = state_values
        policies_log[episode] = policy

        # update VALUE figure (already present)
        # update_value_heatmaps(imgsV, values_log[episode], env.unwrapped.width, env.unwrapped.height,
        #                       episode+1, figV)

        # update REWARD & VARIANCE figure
        # update_reward_heatmaps(imgsR, reward_running_mean, reward_running_var, env.unwrapped.width, env.unwrapped.height,
        #                        episode+1, figR)
        
        # update_reward_heatmaps(imgsR, reward_running_mean, reward_running_var, env.unwrapped.width, env.unwrapped.height,
        #                episode + 1, figR, cbarsR)

        # update TRANSITION figure **using the sampled P of this episode**
        # update_transition_heatmaps(imgsP, p_sample,
        #                            40, env.unwrapped.width, env.unwrapped.height,
        #                            episode+1, figP)



        # ---------- Act ----------
        episode_rewards = 0
        for step_in_episode in range(steps_per_episode):
            action = policy[state]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            done = terminated or truncated
            new_state = flatten_grid_state(env)

            # Logs
            actions[global_step] = action
            rewards[global_step] = reward
            states[global_step]  = state

            # Online reward mean/variance (Welford)
            episode_visits[state, action] += 1
            total_visits[state, action] += 1
            n = total_visits[state, action]

            delta = reward - reward_running_mean[state, action]
            reward_running_mean[state, action] += delta / n
            reward_running_var[state, action]  = ((n - 1) * reward_running_var[state, action] +
                                     delta * (reward - reward_running_mean[state, action])) / n

            # Transition counts (skip the forced reset at the end)
            if step_in_episode != steps_per_episode - 1:
                transition_counts_obs[new_state, state, action] += 1
                state = new_state

            global_step += 1

            # TODO: update transition_counts_obs here?
            if done:
                break  # stop this episode early

        print(f"Episode {episode+1:>3}/{num_episodes} | Episode reward: {episode_rewards:.4f}")

    # plt.ioff()
    # plt.show()  # Keeps final plot open
    return rewards, states, actions, values_log, policies_log, total_visits
