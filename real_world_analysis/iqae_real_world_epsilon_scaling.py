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
        description="IQAE epsilon scaling for real-world double-Poisson model."
    )
    parser.add_argument("--ticker", type=str, default="GC=F")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--threshold-std", type=float, default=3.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-shots", type=int, default=64)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--tail-prob", type=float, default=1e-3)
    parser.add_argument("--truth-samples", type=int, default=200_000)
    parser.add_argument("--epsilons", type=str, default="auto")
    parser.add_argument("--out", type=str, default="real_world_analysis/results/iqae_real_world_eps.png")
    parser.add_argument("--csv", type=str, default="real_world_analysis/results/iqae_real_world_eps.csv")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.epsilons == "auto":
        epsilons = geom_series(start=0.2, ratio=1 / 1.4, count=12)
    else:
        epsilons = parse_list(args.epsilons)

    rows = []
    for eps in epsilons:
        LOGGER.info("Running epsilon=%.6f", eps)
        res = run_iqae_real_world(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            threshold_std=args.threshold_std,
            num_qubits=args.num_qubits,
            alpha=args.alpha,
            epsilon=eps,
            confidence_alpha=0.01,
            num_shots=args.num_shots,
            mc_samples=args.mc_samples,
            tail_prob=args.tail_prob,
            truth_samples=args.truth_samples,
            max_width=28,
            machine_precision=None,
            seed=None,
        )
        rows.append(
            {
                "epsilon": float(eps),
                "queries": float(res["total_queries"]),
                "var_error": float(res["error"]),
            }
        )

    write_csv(Path(args.csv), rows)

    eps_arr = np.array([r["epsilon"] for r in rows], dtype=float)
    queries_arr = np.array([r["queries"] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(1.0 / eps_arr, queries_arr, "o-")
    ax[0].set_xlabel("1 / epsilon")
    ax[0].set_ylabel("Total oracle queries")
    ax[0].set_title("Queries vs 1/epsilon")
    ax[0].grid(True)

    ax[1].loglog(eps_arr, queries_arr, "o-", label="data")
    coeffs = np.polyfit(np.log(eps_arr), np.log(queries_arr), 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    fit_queries = np.exp(intercept) * (eps_arr**slope)
    ax[1].loglog(
        eps_arr,
        fit_queries,
        "--",
        label=f"fit ~ eps^{slope:.2f}",
    )
    ax[1].set_xlabel("epsilon (log)")
    ax[1].set_ylabel("queries (log)")
    ax[1].set_title("Log-log slope (expect ~ -1)")
    ax[1].legend()
    ax[1].grid(True, which="both")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
