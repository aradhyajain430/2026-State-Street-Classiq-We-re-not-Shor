from __future__ import annotations

"""Run classical + quantum experiments and build comparison plots.

This script orchestrates the existing pipelines, aggregates their CSV outputs,
and writes the final comparison figures into writeup/.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str]) -> None:
    """Execute a subprocess and fail fast on errors."""
    subprocess.run(cmd, check=True)


def run_classical(
    outdir: Path,
    mu: float,
    sigma: float,
    confidence: float,
    trials: int,
    seed: int,
    sample_sizes: list[int] | None,
) -> None:
    """Run the classical Monte Carlo experiment and write CSV outputs."""
    cmd = [
        sys.executable,
        str(ROOT / "classical" / "run_classical.py"),
        "--outdir",
        str(outdir),
        "--mu",
        str(mu),
        "--sigma",
        str(sigma),
        "--confidence",
        str(confidence),
        "--trials",
        str(trials),
        "--seed",
        str(seed),
    ]
    if sample_sizes:
        cmd.extend(["--sample-sizes", ",".join(str(v) for v in sample_sizes)])
    run_command(cmd)


def run_quantum_epsilon(
    out_csv: Path,
    out_plot: Path,
    mu: float,
    sigma: float,
    alpha: float,
    num_shots: int,
    num_qubits: int,
) -> None:
    """Run IQAE epsilon scaling and write CSV outputs."""
    cmd = [
        sys.executable,
        str(ROOT / "quantum" / "iqae_epsilon_scaling.py"),
        "--out",
        str(out_plot),
        "--csv",
        str(out_csv),
        "--mu",
        str(mu),
        "--sigma",
        str(sigma),
        "--alpha",
        str(alpha),
        "--num-shots",
        str(num_shots),
        "--num-qubits",
        str(num_qubits),
    ]
    run_command(cmd)


def run_quantum_shots(
    out_csv: Path,
    out_plot: Path,
    mu: float,
    sigma: float,
    alpha: float,
    num_qubits: int,
) -> None:
    """Run IQAE shots scaling and write CSV outputs."""
    cmd = [
        sys.executable,
        str(ROOT / "quantum" / "iqae_shots_scaling.py"),
        "--out",
        str(out_plot),
        "--csv",
        str(out_csv),
        "--mu",
        str(mu),
        "--sigma",
        str(sigma),
        "--alpha",
        str(alpha),
        "--num-qubits",
        str(num_qubits),
    ]
    run_command(cmd)


def load_csv(path: Path) -> list[dict[str, float]]:
    """Load a numeric CSV into a list of dicts."""
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def plot_error_vs_queries(
    mc_samples: np.ndarray,
    mc_error: np.ndarray,
    iqae_queries: np.ndarray,
    iqae_error: np.ndarray,
    output_path: Path,
) -> None:
    """Plot MC error vs samples alongside IQAE error vs queries."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(mc_samples, mc_error, "o-", label="MC error vs samples")
    ax.loglog(iqae_queries, iqae_error, "s-", label="IQAE error vs queries")
    ax.set_xlabel("Total probability queries")
    ax.set_ylabel("VaR absolute error")
    ax.set_title("Error vs Total Queries (log-log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_queries_vs_inv_eps(
    mc_samples: np.ndarray,
    mc_error: np.ndarray,
    iqae_eps: np.ndarray,
    iqae_queries: np.ndarray,
    output_path: Path,
) -> None:
    """Plot query scaling vs 1/epsilon for MC and IQAE."""
    inv_eps_mc = 1.0 / mc_error
    inv_eps_iqae = 1.0 / iqae_eps

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(inv_eps_mc, mc_samples, "o-", label="MC (epsilon ≈ error)")
    ax.loglog(inv_eps_iqae, iqae_queries, "s-", label="IQAE")
    ax.set_xlabel("1 / epsilon")
    ax.set_ylabel("Total probability queries")
    ax.set_title("Queries vs 1/epsilon (log-log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_shots_vs_eps(
    mc_samples: np.ndarray,
    mc_error: np.ndarray,
    iqae_shots: np.ndarray,
    iqae_shot_error: np.ndarray,
    iqae_eps: np.ndarray,
    iqae_queries: np.ndarray,
    output_path: Path,
) -> None:
    """Plot shot noise vs query complexity in a two-panel figure."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_left.loglog(mc_samples, mc_error, "o-", label="MC samples")
    ax_left.loglog(iqae_shots, iqae_shot_error, "s-", label="IQAE shots")
    ax_left.set_xlabel("Shots / Samples")
    ax_left.set_ylabel("VaR absolute error")
    ax_left.set_title("Error vs Shots")
    ax_left.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_left.legend()

    ax_right.loglog(1.0 / iqae_eps, iqae_queries, "s-", label="IQAE")
    ax_right.set_xlabel("1 / epsilon")
    ax_right.set_ylabel("Total queries")
    ax_right.set_title("IQAE Queries vs 1/epsilon")
    ax_right.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_right.legend()

    fig.suptitle("Shot Noise vs Query Scaling", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entry point for comparison plot generation."""
    parser = argparse.ArgumentParser(
        description="Run classical + quantum experiments and generate comparison plots."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "writeup",
        help="Output directory for comparison PNGs.",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=ROOT / "comparison" / "results",
        help="Working directory for intermediate CSV/plots.",
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--num-shots", type=int, default=64)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="",
        help="Optional comma-separated list of MC sample sizes.",
    )
    args = parser.parse_args()

    sample_sizes = None
    if args.sample_sizes:
        sample_sizes = [int(float(v.strip())) for v in args.sample_sizes.split(",") if v.strip()]

    work_dir = args.work
    classical_dir = work_dir / "classical"
    quantum_dir = work_dir / "quantum"
    classical_dir.mkdir(parents=True, exist_ok=True)
    quantum_dir.mkdir(parents=True, exist_ok=True)

    run_classical(
        outdir=classical_dir,
        mu=args.mu,
        sigma=args.sigma,
        confidence=args.confidence,
        trials=args.trials,
        seed=args.seed,
        sample_sizes=sample_sizes,
    )

    run_quantum_epsilon(
        out_csv=quantum_dir / "iqae_epsilon_scaling.csv",
        out_plot=quantum_dir / "iqae_epsilon_scaling.png",
        mu=args.mu,
        sigma=args.sigma,
        alpha=args.alpha,
        num_shots=args.num_shots,
        num_qubits=args.num_qubits,
    )

    run_quantum_shots(
        out_csv=quantum_dir / "iqae_shots_scaling.csv",
        out_plot=quantum_dir / "iqae_shots_scaling.png",
        mu=args.mu,
        sigma=args.sigma,
        alpha=args.alpha,
        num_qubits=args.num_qubits,
    )

    mc_rows = load_csv(classical_dir / "error_scaling.csv")
    mc_samples = np.array([row["samples"] for row in mc_rows], dtype=float)
    mc_error = np.array([row["abs_err"] for row in mc_rows], dtype=float)

    iqae_eps_rows = load_csv(quantum_dir / "iqae_epsilon_scaling.csv")
    iqae_eps = np.array([row["epsilon"] for row in iqae_eps_rows], dtype=float)
    iqae_queries = np.array([row["queries"] for row in iqae_eps_rows], dtype=float)
    iqae_var_error = np.array([row["var_error"] for row in iqae_eps_rows], dtype=float)

    iqae_shots_rows = load_csv(quantum_dir / "iqae_shots_scaling.csv")
    iqae_shots = np.array([row["shots"] for row in iqae_shots_rows], dtype=float)
    iqae_shot_error = np.array([row["var_error"] for row in iqae_shots_rows], dtype=float)

    args.out.mkdir(parents=True, exist_ok=True)

    plot_error_vs_queries(
        mc_samples,
        mc_error,
        iqae_queries,
        iqae_var_error,
        args.out / "comparison_error_vs_queries.png",
    )
    plot_queries_vs_inv_eps(
        mc_samples,
        mc_error,
        iqae_eps,
        iqae_queries,
        args.out / "comparison_queries_vs_inv_eps.png",
    )
    plot_shots_vs_eps(
        mc_samples,
        mc_error,
        iqae_shots,
        iqae_shot_error,
        iqae_eps,
        iqae_queries,
        args.out / "comparison_shots_vs_eps.png",
    )


if __name__ == "__main__":
    main()
