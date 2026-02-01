import argparse
import csv
import logging
import os
import sys
from typing import List

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


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heatmap of IQAE error vs confidence_alpha and num_shots."
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05, help="Tail probability.")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument(
        "--confidence-alphas",
        type=str,
        default="auto",
        help="Comma list or 'auto' for 8 points /1.4 starting at 0.2.",
    )
    parser.add_argument(
        "--shots",
        type=str,
        default="auto",
        help="Comma list or 'auto' for 10 points /1.4 starting at 32.",
    )
    parser.add_argument("--out", type=str, default="quantum/iqae_alpha_shots_heatmap.png")
    parser.add_argument("--csv", type=str, default="quantum/iqae_alpha_shots_heatmap.csv")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.confidence_alphas == "auto":
        confidence_alphas = geom_series(start=0.2, ratio=1 / 1.4, count=8)
    else:
        confidence_alphas = parse_list(args.confidence_alphas)

    if args.shots == "auto":
        shots = geom_series(start=32, ratio=1.4, count=10)
    else:
        shots = parse_list(args.shots)
    shots = [int(round(s)) for s in shots]

    rows: list[dict] = []
    error_matrix = np.zeros((len(confidence_alphas), len(shots)), dtype=float)

    for i, conf_alpha in enumerate(confidence_alphas):
        for j, n_shots in enumerate(shots):
            LOGGER.info(
                "Running IQAE: confidence_alpha=%.6f shots=%s", conf_alpha, n_shots
            )
            res = run_iqae_var(
                num_shots=n_shots,
                mu=args.mu,
                sigma=args.sigma,
                num_qubits=args.num_qubits,
                alpha=args.alpha,
                epsilon=args.epsilon,
                confidence_alpha=conf_alpha,
                num_sigmas=args.num_sigmas,
                prep_bound=args.prep_bound,
                max_width=args.max_width,
                machine_precision=args.machine_precision,
            )
            error = float(res["error"])
            error_matrix[i, j] = error
            rows.append(
                {
                    "confidence_alpha": float(conf_alpha),
                    "shots": int(n_shots),
                    "error": float(error),
                }
            )

    write_csv(args.csv, rows)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(
        error_matrix,
        origin="lower",
        aspect="auto",
        norm=LogNorm(),
        cmap="viridis",
    )
    ax.set_xticks(np.arange(len(shots)))
    ax.set_xticklabels(shots, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(confidence_alphas)))
    ax.set_yticklabels([f"{v:.3g}" for v in confidence_alphas])
    ax.set_xlabel("num_shots")
    ax.set_ylabel("confidence_alpha")
    ax.set_title("IQAE error heatmap (confidence_alpha vs shots)")
    fig.colorbar(im, ax=ax, label="VaR abs error (log scale)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
