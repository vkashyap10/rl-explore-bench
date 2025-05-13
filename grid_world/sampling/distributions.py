# ------------------------------------------------------------------
#  Posterior-sampling helpers
# ------------------------------------------------------------------
from typing import Optional, Tuple

import numpy as np


def sample_dirichlet_mat(
    alpha: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw a transition‑probability tensor from a Dirichlet prior

    Parameters
    ----------
    alpha : np.ndarray
        Dirichlet concentration parameters with shape ``(S, S, A)``—typically
        interpreted as counts.
    rng : np.random.Generator, optional
        NumPy random generator; if *None*, ``np.random.default_rng()`` is used.

    Returns
    -------
    theta : np.ndarray
        Sampled transition probabilities with the same shape as ``alpha``.
    """

    # Gamma draw followed by normalisation along the *first* axis
    theta = rng.gamma(shape=alpha, scale=1.0)
    theta /= theta.sum(axis=0, keepdims=True)
    return theta


def sample_normal_gamma_mat(
    reward_mean_prior: np.ndarray,
    reward_mean_strength: np.ndarray,
    reward_precision_prior: np.ndarray,
    reward_precision_strength: np.ndarray,
    total_visits: np.ndarray,
    reward_mean_obs: np.ndarray,
    reward_var_obs: np.ndarray,
    rng: np.random.Generator,
    draw_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute or sample from the Normal-Gamma posterior distribution.

    Parameters
    ----------
    draw_sample : bool
        If True, samples from the posterior. If False, returns posterior mean and expected precision.

    Returns
    -------
    mu  : Sampled or expected means (shape S×A)
    std_n : Sampled or expected std_n (shape S×A)

    References
    -------
    http://en.wikipedia.org/wiki/Normal-gamma_distribution
    http://www.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf
    """

    alpha0 = reward_precision_strength / 2.0
    beta0 = alpha0 / reward_precision_prior

    lambda_n = reward_mean_strength + total_visits
    mu_n = (
        reward_mean_strength * reward_mean_prior + total_visits * reward_mean_obs
    ) / lambda_n
    alpha_n = alpha0 + total_visits / 2.0
    beta_n = beta0 + 0.5 * (
        total_visits * reward_var_obs
        + reward_mean_strength
        * total_visits
        * (reward_mean_obs - reward_mean_prior) ** 2
        / lambda_n
    )

    if draw_sample:
        tau = rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        std_n = np.sqrt(1.0 / (lambda_n * tau))
        mu = rng.normal(loc=mu_n, scale=std_n)
    else:
        tau = alpha_n / beta_n  # Expected precision of Gamma(alpha, beta)
        mu = mu_n  # Posterior mean of Normal
        std_n = np.sqrt(1.0 / (lambda_n * tau))

    return mu, std_n


def sample_action_from_scores(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Samples an action from unnormalized score vector using softmax-like logic."""
    scores = np.maximum(scores, 0)
    norm = scores.sum()
    if norm == 0:
        return rng.integers(len(scores))
    probs = scores / norm
    return rng.choice(len(scores), p=probs)


def update_obs_reward_stats(
    mean: np.ndarray,
    var: np.ndarray,
    total_visits: np.ndarray,
    state: int,
    action: int,
    reward: float,
    multiplier: int = 1,
):
    """Online update of reward mean and variance using Welford's algorithm."""
    total_visits[state, action] += multiplier
    delta = reward - mean[state, action]
    mean[state, action] += delta / total_visits[state, action]
    var[state, action] += delta * (reward - mean[state, action])
