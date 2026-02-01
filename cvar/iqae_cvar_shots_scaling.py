import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from classiq.applications.iqae.iqae import IQAE
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.generator.model import Constraints, Preferences

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.var_state_prep import set_gaussian_state_params
import quantum.iqae_var_run as iqae_var_run

LOGGER = logging.getLogger(__name__)


def geom_series(start: float, ratio: float, count: int) -> List[int]:
    values: List[int] = []
    current = float(start)
    for _ in range(count):
        values.append(int(round(current)))
        current *= ratio
    return values


def gaussian_cvar_loss(mu: float, sigma: float, alpha: float) -> float:
    """Closed-form CVaR for Gaussian returns (loss domain)."""
    z = scipy.stats.norm.ppf(alpha)
    pdf = scipy.stats.norm.pdf(z)
    return float(-mu + sigma * pdf / alpha)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run IQAE across shot counts and map tail probability to CVaR."""
    parser = argparse.ArgumentParser(
        description="IQAE CVaR error vs num_shots."
    )
    parser.add_argument("--mu", type=float, default=0.0005)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.142857)
    parser.add_argument("--confidence-alpha", type=float, default=0.102041)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--shots-start", type=int, default=64)
    parser.add_argument("--shots-ratio", type=float, default=1.4)
    parser.add_argument("--shots-count", type=int, default=12)
    parser.add_argument("--out", type=str, default="writeup/cvar_iqae_shots_scaling.png")
    parser.add_argument("--csv", type=str, default="cvar/cvar_iqae_shots_scaling.csv")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    alpha = 1.0 - args.confidence
    true_cvar = gaussian_cvar_loss(args.mu, args.sigma, alpha)

    # Build a discretized Gaussian state for the payoff oracle.
    grid, probs = set_gaussian_state_params(
        mu=args.mu,
        sigma=args.sigma,
        num_qubits=args.num_qubits,
        num_sigmas=args.num_sigmas,
        prep_bound=args.prep_bound,
    )
    cdf = np.cumsum(probs)
    index = int(np.searchsorted(cdf, alpha))
    iqae_var_run.GLOBAL_INDEX = index
    LOGGER.info("Using index=%s for alpha=%.6f", index, alpha)

    constraints = Constraints(max_width=args.max_width) if args.max_width else None
    preferences = (
        Preferences(machine_precision=args.machine_precision)
        if args.machine_precision
        else None
    )
    iqae = IQAE(
        state_prep_op=iqae_var_run.state_preparation,
        problem_vars_size=args.num_qubits,
        constraints=constraints,
        preferences=preferences,
    )

    shots = geom_series(args.shots_start, args.shots_ratio, args.shots_count)
    rows = []

    for n_shots in shots:
        LOGGER.info("Running IQAE for num_shots=%s", n_shots)
        exec_prefs = ExecutionPreferences(num_shots=n_shots)
        res = iqae.run(
            epsilon=args.epsilon,
            alpha=args.confidence_alpha,
            execution_preferences=exec_prefs,
        )
        alpha_est = float(np.clip(res.estimation, 1e-12, 1 - 1e-12))
        cvar_est = gaussian_cvar_loss(args.mu, args.sigma, alpha_est)
        err = abs(cvar_est - true_cvar)
        rows.append({"shots": int(n_shots), "cvar_error": float(err)})

    write_csv(Path(args.csv), rows)

    shots_arr = np.array([r["shots"] for r in rows], dtype=float)
    errors_arr = np.array([r["cvar_error"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(shots_arr, errors_arr, "o-", label="data")
    coeffs = np.polyfit(np.log(shots_arr), np.log(errors_arr), 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    fit_errors = np.exp(intercept) * (shots_arr**slope)
    ax.loglog(
        shots_arr,
        fit_errors,
        "--",
        label=f"fit ~ shots^{slope:.2f}",
    )
    ax.set_xlabel("num_shots (log)")
    ax.set_ylabel("CVaR abs error (log)")
    ax.set_title("IQAE CVaR error vs num_shots")
    ax.legend()
    ax.grid(True, which="both")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
