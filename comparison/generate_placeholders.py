from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_queries(output_dir: Path) -> None:
    queries = np.logspace(2, 5, 10)
    mc_error = 1.0 / np.sqrt(queries)
    iqae_error = 0.4 / queries

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(queries, mc_error, "o-", label="MC ~ 1/sqrt(N)")
    ax.loglog(queries, iqae_error, "s-", label="IQAE ~ 1/N")
    ax.set_xlabel("Total probability queries")
    ax.set_ylabel("VaR error (placeholder)")
    ax.set_title("Placeholder: Error vs Queries (log-log)")
    _style_axes(ax)
    ax.legend()
    _save(fig, output_dir, "comparison_error_vs_queries.png")


def plot_queries_vs_inv_eps(output_dir: Path) -> None:
    eps = np.logspace(-1, -3, 10)
    inv_eps = 1.0 / eps
    queries_mc = 15.0 / (eps**2)
    queries_iqae = 50.0 / eps

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(inv_eps, queries_mc, "o-", label="MC ~ 1/eps^2")
    ax.loglog(inv_eps, queries_iqae, "s-", label="IQAE ~ 1/eps")
    ax.set_xlabel("1/epsilon")
    ax.set_ylabel("Total probability queries (placeholder)")
    ax.set_title("Placeholder: Queries vs 1/epsilon (log-log)")
    _style_axes(ax)
    ax.legend()
    _save(fig, output_dir, "comparison_queries_vs_inv_eps.png")


def plot_shots_vs_eps(output_dir: Path) -> None:
    shots = np.logspace(2, 5, 10)
    mc_error = 1.0 / np.sqrt(shots)
    iqae_error = 0.9 / np.sqrt(shots)

    eps = np.logspace(-1, -3, 10)
    inv_eps = 1.0 / eps
    iqae_queries = 50.0 / eps

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_left.loglog(shots, mc_error, "o-", label="MC shots")
    ax_left.loglog(shots, iqae_error, "s-", label="IQAE shots")
    ax_left.set_xlabel("Shots")
    ax_left.set_ylabel("VaR error (placeholder)")
    ax_left.set_title("Placeholder: Error vs Shots")
    _style_axes(ax_left)
    ax_left.legend()

    ax_right.loglog(inv_eps, iqae_queries, "s-", label="IQAE ~ 1/eps")
    ax_right.set_xlabel("1/epsilon")
    ax_right.set_ylabel("Total queries (placeholder)")
    ax_right.set_title("Placeholder: Queries vs 1/epsilon")
    _style_axes(ax_right)
    ax_right.legend()

    fig.suptitle("Placeholder: Shot Noise vs Query Scaling", y=1.02)
    _save(fig, output_dir, "comparison_shots_vs_eps.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate placeholder comparison plots for the writeup."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "writeup",
        help="Output directory for placeholder PNGs.",
    )
    args = parser.parse_args()

    plot_error_vs_queries(args.out)
    plot_queries_vs_inv_eps(args.out)
    plot_shots_vs_eps(args.out)


if __name__ == "__main__":
    main()
