import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np

from classiq import Const, QArray, QBit, QNum, qfunc, qperm
from classiq.applications.iqae.iqae import IQAE
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.generator.model import Constraints, Preferences

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from real_world_analysis.quantum_state_prep import (
    DOUBLE_POISSON_GRID,
    set_double_poisson_state_params_from_ticker,
    load_double_poisson_state,
)

LOGGER = logging.getLogger(__name__)
GLOBAL_INDEX = 0  # Set by the classical bisection index.


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    """Prepare the distribution and mark tail events via the payoff oracle."""
    load_double_poisson_state(asset)
    payoff(asset=asset, ind=ind)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    """Flag assets below the chosen threshold index."""
    ind ^= asset < GLOBAL_INDEX


def total_queries(iterations_data) -> int:
    """Compute total oracle queries across all IQAE iterations."""
    total = 0
    for it in iterations_data:
        k = int(it.grover_iterations)
        shots = it.sample_results.num_shots
        if shots is None:
            shots = sum(it.sample_results.counts.values())
        total += int(shots) * (2 * k + 1)
    return total


def classical_bisection_index(required_alpha: float, probs: List[float]) -> int:
    """Find the smallest index whose CDF exceeds the tail probability."""
    cdf = np.cumsum(probs)
    left = 0
    right = len(cdf) - 1
    while left < right:
        mid = (left + right) // 2
        if cdf[mid] < required_alpha:
            left = mid + 1
        else:
            right = mid
    return left


def invert_cdf(grid: np.ndarray, cdf: np.ndarray, alpha: float) -> float:
    """Linear interpolation inverse CDF on a discretized grid."""
    idx = int(np.searchsorted(cdf, alpha))
    if idx <= 0:
        return float(grid[0])
    if idx >= len(grid):
        return float(grid[-1])
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    if y1 == y0:
        return float(x1)
    t = (alpha - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def run_iqae_real_world(
    ticker: str,
    period: str,
    interval: str,
    threshold_std: float,
    num_qubits: int,
    alpha: float,
    epsilon: float,
    confidence_alpha: float,
    num_shots: int | None,
    mc_samples: int,
    tail_prob: float,
    truth_samples: int,
    max_width: int | None,
    machine_precision: int | None,
    seed: int | None,
) -> Dict[str, float]:
    """Run IQAE on a fitted double-Poisson model for one ticker."""
    model, grid, probs = set_double_poisson_state_params_from_ticker(
        ticker=ticker,
        period=period,
        interval=interval,
        threshold_std=threshold_std,
        num_qubits=num_qubits,
        mc_samples=mc_samples,
        tail_prob=tail_prob,
        prep_bound=0.0,
        seed=seed,
    )

    cdf = np.cumsum(probs)
    index = classical_bisection_index(alpha, probs)
    global GLOBAL_INDEX
    GLOBAL_INDEX = int(index)

    constraints = Constraints(max_width=max_width) if max_width else None
    preferences = (
        Preferences(machine_precision=machine_precision)
        if machine_precision
        else None
    )
    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=constraints,
        preferences=preferences,
    )
    exec_prefs = (
        ExecutionPreferences(num_shots=num_shots) if num_shots is not None else None
    )
    # IQAE estimates tail probability for the chosen index.
    res = iqae.run(
        epsilon=epsilon, alpha=confidence_alpha, execution_preferences=exec_prefs
    )
    alpha_est = float(np.clip(res.estimation, 1e-12, 1 - 1e-12))
    var_est = invert_cdf(np.asarray(grid), cdf, alpha_est)

    # Reference VaR is approximated by high-sample Monte Carlo.
    rng = np.random.default_rng(seed)
    returns = model.sample_returns(rng, truth_samples)
    confidence = 1.0 - alpha
    reference_var = float(np.quantile(-returns, confidence))

    return {
        "alpha_est": float(alpha_est),
        "reference_var": float(reference_var),
        "var_est": float(var_est),
        "error": float(abs(var_est - reference_var)),
        "total_queries": float(total_queries(res.iterations_data)),
        "index": float(index),
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
    }


def main() -> None:
    """CLI entry point for a single real-world IQAE VaR run."""
    parser = argparse.ArgumentParser(
        description="Run IQAE VaR once for a real-world double-Poisson model."
    )
    parser.add_argument("--ticker", type=str, default="GC=F")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--threshold-std", type=float, default=3.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--confidence-alpha", type=float, default=0.01)
    parser.add_argument("--num-shots", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--tail-prob", type=float, default=1e-3)
    parser.add_argument("--truth-samples", type=int, default=200_000)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    res = run_iqae_real_world(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        threshold_std=args.threshold_std,
        num_qubits=args.num_qubits,
        alpha=args.alpha,
        epsilon=args.epsilon,
        confidence_alpha=args.confidence_alpha,
        num_shots=args.num_shots,
        mc_samples=args.mc_samples,
        tail_prob=args.tail_prob,
        truth_samples=args.truth_samples,
        max_width=args.max_width,
        machine_precision=args.machine_precision,
        seed=args.seed,
    )
    print(
        "alpha_est={alpha_est:.6f} var_est={var_est:.6f} "
        "reference_var={reference_var:.6f} error={error:.6f} "
        "total_queries={total_queries:.0f}".format(**res)
    )


if __name__ == "__main__":
    main()
