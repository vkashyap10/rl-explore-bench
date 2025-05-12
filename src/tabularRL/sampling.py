# ------------------------------------------------------------------
#  Posterior-sampling helpers
# ------------------------------------------------------------------
import numpy as np
from typing import Tuple, Optional
import numpy as np

def sample_dirichlet_mat(
    alpha: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw a transition‑probability tensor from a Dirichlet prior.

    A faithful vectorised port of MATLAB's ``sampleDirichletMat``.  Each slice
    along the *second* dimension is treated as an independent Dirichlet whose
    concentration parameters are given by ``alpha``.

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

    if rng is None:
        rng = np.random.default_rng()

    # Gamma draw followed by normalisation along the *first* axis
    theta = rng.gamma(shape=alpha, scale=1.0)
    theta /= theta.sum(axis=0, keepdims=True)
    return theta


import numpy as np
from typing import Tuple, Optional

def sample_normal_gamma_mat(
    reward_mean_prior: np.ndarray,
    reward_mean_strength: np.ndarray,
    reward_precision_prior: np.ndarray,
    reward_precision_strength: np.ndarray,
    total_visits: np.ndarray,
    reward_running_mean: np.ndarray,
    reward_running_var: np.ndarray,
    draw_sample: bool = True,
    rng: Optional[np.random.Generator] = None,
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
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha0 = reward_precision_strength / 2.0
    beta0 = alpha0 / reward_precision_prior

    lambda_n = reward_mean_strength + total_visits
    mu_n = (reward_mean_strength * reward_mean_prior + total_visits * reward_running_mean) / lambda_n
    alpha_n = alpha0 + total_visits / 2.0
    beta_n = beta0 + 0.5 * (total_visits * reward_running_var + reward_mean_strength * total_visits * (reward_running_mean - reward_mean_prior) ** 2 / lambda_n)

    if draw_sample:
        tau = rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        std_n = np.sqrt(1.0 / (lambda_n * tau))
        mu = rng.normal(loc=mu_n, scale=std_n)
    else:
        tau = alpha_n / beta_n   # Expected precision of Gamma(alpha, beta)
        mu = mu_n                # Posterior mean of Normal
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