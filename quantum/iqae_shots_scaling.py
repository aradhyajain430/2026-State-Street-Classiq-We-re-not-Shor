import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.iqae_var_run import run_iqae_var

LOGGER = logging.getLogger(__name__)


def geom_series(start: float, ratio: float, count: int) -> List[int]:
    values: List[int] = []
    current = float(start)
    for _ in range(count):
        values.append(int(round(current)))
        current *= ratio
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Plot IQAE VaR error vs num_shots."
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--confidence-alpha", type=float, default=0.01)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--shots-start", type=int, default=64)
    parser.add_argument("--shots-ratio", type=float, default=1.4)
    parser.add_argument("--shots-count", type=int, default=12)
    parser.add_argument("--out", type=str, default="quantum/iqae_shots_scaling.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    shots = geom_series(args.shots_start, args.shots_ratio, args.shots_count)
    errors = []

    for n_shots in shots:
        LOGGER.info("Running IQAE VaR for num_shots=%s", n_shots)
        res = run_iqae_var(
            num_shots=n_shots,
            mu=args.mu,
            sigma=args.sigma,
            num_qubits=args.num_qubits,
            alpha=args.alpha,
            epsilon=args.epsilon,
            confidence_alpha=args.confidence_alpha,
            num_sigmas=args.num_sigmas,
            prep_bound=args.prep_bound,
            max_width=args.max_width,
            machine_precision=args.machine_precision,
        )
        errors.append(res["error"])

    shots_arr = np.array(shots, dtype=float)
    errors_arr = np.array(errors, dtype=float)

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
    ax.set_ylabel("VaR abs error (log)")
    ax.set_title("IQAE VaR error vs num_shots")
    ax.legend()
    ax.grid(True, which="both")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
