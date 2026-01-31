import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.iqae_var_run import run_iqae_var

LOGGER = logging.getLogger(__name__)


def geom_series(start: float, ratio: float, count: int) -> List[float]:
    values: List[float] = []
    current = float(start)
    for _ in range(count):
        values.append(current)
        current *= ratio
    return values


def parse_list(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Tune IQAE epsilon and confidence alpha with fixed num_shots."
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=64)
    parser.add_argument(
        "--epsilons",
        type=str,
        default="auto",
        help="Comma list or 'auto' for 8 points /1.4 starting at 0.2.",
    )
    parser.add_argument(
        "--confidence-alphas",
        type=str,
        default="auto",
        help="Comma list or 'auto' for 6 points /1.4 starting at 0.2.",
    )
    parser.add_argument("--out", type=str, default="quantum/iqae_eps_alpha.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.epsilons == "auto":
        epsilons = geom_series(start=0.2, ratio=1 / 1.4, count=8)
    else:
        epsilons = parse_list(args.epsilons)

    if args.confidence_alphas == "auto":
        confidence_alphas = geom_series(start=0.2, ratio=1 / 1.4, count=6)
    else:
        confidence_alphas = parse_list(args.confidence_alphas)

    results: list[Tuple[float, float, float]] = []
    best = (float("inf"), None, None)

    for eps in epsilons:
        for conf_alpha in confidence_alphas:
            LOGGER.info("Running IQAE with epsilon=%.6f, confidence_alpha=%.6f", eps, conf_alpha)
            res = run_iqae_var(
                num_shots=args.num_shots,
                mu=args.mu,
                sigma=args.sigma,
                num_qubits=args.num_qubits,
                alpha=args.alpha,
                epsilon=eps,
                confidence_alpha=conf_alpha,
                num_sigmas=args.num_sigmas,
                prep_bound=args.prep_bound,
                max_width=args.max_width,
                machine_precision=args.machine_precision,
            )
            error = res["error"]
            results.append((eps, conf_alpha, error))
            if error < best[0]:
                best = (error, eps, conf_alpha)

    best_error, best_eps, best_conf_alpha = best
    print(
        "Best: error={:.6f} epsilon={:.6f} confidence_alpha={:.6f}".format(
            best_error, best_eps, best_conf_alpha
        )
    )

    eps_arr = np.array([r[0] for r in results], dtype=float)
    conf_arr = np.array([r[1] for r in results], dtype=float)
    err_arr = np.array([r[2] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(
        eps_arr,
        conf_arr,
        c=err_arr,
        cmap="viridis",
        norm=LogNorm(),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epsilon (log)")
    ax.set_ylabel("confidence_alpha (log)")
    ax.set_title("VaR error vs IQAE hyperparameters (num_shots=64)")
    fig.colorbar(sc, ax=ax, label="VaR abs error")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
