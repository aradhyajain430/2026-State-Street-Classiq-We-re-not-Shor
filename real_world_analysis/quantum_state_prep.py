import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

from classiq import QArray, QBit, QNum, inplace_prepare_state, qfunc

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from real_world_analysis.real_world_analysis import (
    DoublePoissonJumpModel,
    fetch_returns,
    fit_double_poisson,
)

# Global state used by Classiq state-prep functions.
DOUBLE_POISSON_PROBS: List[float] = []
DOUBLE_POISSON_GRID: List[float] = []
DOUBLE_POISSON_PREP_BOUND: float = 0.0


def double_poisson_grid(
    model: DoublePoissonJumpModel,
    num_qubits: int,
    mc_samples: int = 200_000,
    tail_prob: float = 1e-3,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize the fitted model on a grid via Monte Carlo sampling.

    We sample the fitted model, choose a bounded range (via tail quantiles),
    and histogram the samples into 2^n bins to get a discrete PMF.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")

    rng = np.random.default_rng(seed)
    samples = model.sample_returns(rng, mc_samples)

    if tail_prob is not None:
        low, high = np.quantile(samples, [tail_prob, 1.0 - tail_prob])
    else:
        mu = float(np.mean(samples))
        sigma = float(np.std(samples, ddof=1))
        low = mu - 4.0 * sigma
        high = mu + 4.0 * sigma

    n = 2**num_qubits
    # Bin the samples into 2^n buckets for amplitude encoding.
    bins = np.linspace(low, high, n + 1)
    clipped = np.clip(samples, low, high)
    counts, _ = np.histogram(clipped, bins=bins)
    probs = counts / counts.sum()
    grid = (bins[:-1] + bins[1:]) / 2.0
    return grid, probs


def set_double_poisson_state_params(
    model: DoublePoissonJumpModel,
    num_qubits: int,
    mc_samples: int = 200_000,
    tail_prob: float = 1e-3,
    prep_bound: float = 0.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[float]]:
    """Compute and store the PMF for Classiq state preparation."""
    grid, probs = double_poisson_grid(
        model=model,
        num_qubits=num_qubits,
        mc_samples=mc_samples,
        tail_prob=tail_prob,
        seed=seed,
    )
    global DOUBLE_POISSON_PROBS, DOUBLE_POISSON_GRID, DOUBLE_POISSON_PREP_BOUND
    DOUBLE_POISSON_PROBS = probs.tolist()
    DOUBLE_POISSON_GRID = grid.tolist()
    DOUBLE_POISSON_PREP_BOUND = float(prep_bound)
    return grid, DOUBLE_POISSON_PROBS


def set_double_poisson_state_params_from_ticker(
    ticker: str,
    period: str,
    interval: str,
    threshold_std: float,
    num_qubits: int,
    mc_samples: int = 200_000,
    tail_prob: float = 1e-3,
    prep_bound: float = 0.0,
    seed: int | None = None,
) -> Tuple[DoublePoissonJumpModel, np.ndarray, List[float]]:
    """Fit the model from data and prepare its discrete PMF."""
    returns = fetch_returns(ticker, period, interval)
    model = fit_double_poisson(returns, threshold_std=threshold_std)
    grid, probs = set_double_poisson_state_params(
        model=model,
        num_qubits=num_qubits,
        mc_samples=mc_samples,
        tail_prob=tail_prob,
        prep_bound=prep_bound,
        seed=seed,
    )
    return model, grid, probs


@qfunc(synthesize_separately=True)
def load_double_poisson_state(asset: QNum):
    """Load the precomputed PMF into amplitudes (QNum view)."""
    inplace_prepare_state(
        probabilities=DOUBLE_POISSON_PROBS,
        bound=DOUBLE_POISSON_PREP_BOUND,
        target=asset,
    )


@qfunc(synthesize_separately=True)
def load_double_poisson_state_qarray(asset: QArray[QBit]):
    """Load the precomputed PMF into amplitudes (QArray view)."""
    inplace_prepare_state(
        probabilities=DOUBLE_POISSON_PROBS,
        bound=DOUBLE_POISSON_PREP_BOUND,
        target=asset,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI for fitting and preparing the double-Poisson state."""
    parser = argparse.ArgumentParser(
        description="Prepare a double-Poisson state from real data."
    )
    parser.add_argument("--ticker", type=str, default="GC=F")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--threshold-std", type=float, default=3.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--tail-prob", type=float, default=1e-3)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, grid, _ = set_double_poisson_state_params_from_ticker(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        threshold_std=args.threshold_std,
        num_qubits=args.num_qubits,
        mc_samples=args.mc_samples,
        tail_prob=args.tail_prob,
        prep_bound=args.prep_bound,
        seed=args.seed,
    )
    print("Built state params.")
    print(f"Model: {model}")
    print(f"Grid min={grid.min():.6f} max={grid.max():.6f} size={len(grid)}")


if __name__ == "__main__":
    main()
