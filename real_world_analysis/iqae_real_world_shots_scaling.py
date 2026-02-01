import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from real_world_analysis.iqae_real_world_run import run_iqae_real_world

LOGGER = logging.getLogger(__name__)


def geom_series(start: float, ratio: float, count: int) -> List[int]:
    values: List[int] = []
    current = float(start)
    for _ in range(count):
        values.append(int(round(current)))
        current *= ratio
    return values


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQAE shot scaling for real-world double-Poisson model."
    )
    parser.add_argument("--ticker", type=str, default="GC=F")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--threshold-std", type=float, default=3.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.142857)
    parser.add_argument("--confidence-alpha", type=float, default=0.102041)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--tail-prob", type=float, default=1e-3)
    parser.add_argument("--truth-samples", type=int, default=200_000)
    parser.add_argument("--shots-start", type=int, default=64)
    parser.add_argument("--shots-ratio", type=float, default=1.4)
    parser.add_argument("--shots-count", type=int, default=12)
    parser.add_argument("--out", type=str, default="real_world_analysis/results/iqae_real_world_shots.png")
    parser.add_argument("--csv", type=str, default="real_world_analysis/results/iqae_real_world_shots.csv")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    shots = geom_series(args.shots_start, args.shots_ratio, args.shots_count)
    rows = []

    for n_shots in shots:
        LOGGER.info("Running num_shots=%s", n_shots)
        res = run_iqae_real_world(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            threshold_std=args.threshold_std,
            num_qubits=args.num_qubits,
            alpha=args.alpha,
            epsilon=args.epsilon,
            confidence_alpha=args.confidence_alpha,
            num_shots=n_shots,
            mc_samples=args.mc_samples,
            tail_prob=args.tail_prob,
            truth_samples=args.truth_samples,
            max_width=28,
            machine_precision=None,
            seed=None,
        )
        rows.append({"shots": float(n_shots), "var_error": float(res["error"])})

    write_csv(Path(args.csv), rows)

    shots_arr = np.array([r["shots"] for r in rows], dtype=float)
    errors_arr = np.array([r["var_error"] for r in rows], dtype=float)

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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
