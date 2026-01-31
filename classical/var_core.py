"""Classical Monte Carlo VaR utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GaussianModel:
    """Gaussian return model for a single asset/portfolio."""

    mu: float
    sigma: float

    def sample_returns(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        return rng.normal(self.mu, self.sigma, size=n)


def theoretical_var_gaussian(
    model: GaussianModel,
    confidence: float,
    portfolio_value: float = 1.0,
) -> float:
    """Closed-form VaR for Gaussian returns.

    VaR at confidence c is defined as the c-quantile of loss L = -P&L.
    For returns r ~ N(mu, sigma), VaR = -(mu + sigma * z_{1-c}) * V.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    z = NormalDist().inv_cdf(1.0 - confidence)
    return -portfolio_value * (model.mu + model.sigma * z)


def monte_carlo_var(
    model: GaussianModel,
    confidence: float,
    n_samples: int,
    portfolio_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo estimate of VaR using Gaussian sampling."""

    if rng is None:
        rng = np.random.default_rng()
    returns = model.sample_returns(rng, n_samples)
    losses = -portfolio_value * returns
    return float(np.quantile(losses, confidence))


def var_trials(
    model: GaussianModel,
    confidence: float,
    n_samples: int,
    trials: int,
    portfolio_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Repeated Monte Carlo VaR estimates for error analysis."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if rng is None:
        rng = np.random.default_rng()

    estimates = np.empty(trials, dtype=float)
    for i in range(trials):
        returns = model.sample_returns(rng, n_samples)
        losses = -portfolio_value * returns
        estimates[i] = np.quantile(losses, confidence)
    return estimates


def summarize_trials(estimates: Iterable[float], true_var: float) -> dict[str, float]:
    """Compute summary statistics for a set of VaR estimates."""

    estimates = np.asarray(list(estimates), dtype=float)
    mean = float(np.mean(estimates))
    std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    abs_err = float(np.mean(np.abs(estimates - true_var)))
    rmse = float(np.sqrt(np.mean((estimates - true_var) ** 2)))
    return {
        "mean": mean,
        "std": std,
        "abs_err": abs_err,
        "rmse": rmse,
    }
