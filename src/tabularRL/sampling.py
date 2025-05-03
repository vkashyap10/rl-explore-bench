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

    # Gamma draw followed by normalisation along the *first* axis (MATLAB sum(theta,1))
    theta = rng.gamma(shape=alpha, scale=1.0)
    theta /= theta.sum(axis=0, keepdims=True)
    return theta


def sample_normal_gamma_mat(
    mu0: np.ndarray,
    nMu0: np.ndarray,
    tau0: np.ndarray,
    nTau0: np.ndarray,
    nObs: np.ndarray,
    muObs: np.ndarray,
    varObs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (mu, tau) from the Normal–Gamma posterior distribution.

    This is a direct translation of the MATLAB function ``sampleNormalGammaMat``.

    Parameters
    ----------
    mu0 : np.ndarray
        Prior estimate of the mean (shape S×A).
    nMu0 : np.ndarray
        Effective sample size associated with ``mu0`` (same shape as ``mu0``).
    tau0 : np.ndarray
        Prior estimate of the precision (sigma⁻²) (shape S×A).
    nTau0 : np.ndarray
        Effective sample size associated with ``tau0`` (same shape as ``tau0``).
    nObs : np.ndarray
        Empirical number of observations (shape S×A).
    muObs : np.ndarray
        Empirical mean of observations (shape S×A).
    varObs : np.ndarray
        Empirical variance of observations (shape S×A).
    rng : np.random.Generator, optional
        Random number generator to use.  If ``None`` (default), ``np.random.default_rng()`` is used.

    Returns
    -------
    mu : np.ndarray
        Sampled means (shape S×A).
    tau : np.ndarray
        Sampled precisions (sigma⁻²) (shape S×A).
    """

    if rng is None:
        rng = np.random.default_rng()

    # Conjugate prior hyperparameters
    lambda0 = nMu0  # often denoted as kappa₀
    alpha0 = nTau0 / 2.0  # shape parameter for the Gamma prior
    beta0 = alpha0 / tau0  # scale parameter for the Gamma prior (because beta = alpha / tau)

    # Posterior hyperparameters after observing data
    mu_n = (lambda0 * mu0 + nObs * muObs) / (lambda0 + nObs)
    lambda_n = lambda0 + nObs
    alpha_n = alpha0 + nObs / 2.0
    beta_n = beta0 + 0.5 * (nObs*varObs + lambda0*nObs*(muObs - mu0)**2/(lambda0 + nObs))

    # Sample precision (tau) from the Gamma distribution
    # NumPy's Gamma uses shape (alpha) and scale (theta) parameters.
    tau = rng.gamma(shape=alpha_n, scale=1.0 / beta_n)

    # Sample mean (mu) conditional on tau from the Normal distribution
    std_n = np.sqrt(1.0 / (lambda_n * tau))
    mu = rng.normal(loc=mu_n, scale=std_n)

    return mu, tau

