"""Classical CVaR Monte Carlo experiments."""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from classical.var_core import GaussianModel, summarize_trials


def default_sample_sizes() -> list[int]:
    raw = np.logspace(2, 5, num=25)
    sizes: list[int] = []
    for value in raw:
        candidate = int(round(value))
        if sizes:
            candidate = max(candidate, sizes[-1] + 1)
        sizes.append(candidate)
    sizes[-1] = 100_000
    return sizes


def parse_sample_sizes(raw: str) -> list[int]:
    if not raw:
        return default_sample_sizes()
    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(float(part)))
    return sizes


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cvar_from_returns(returns: np.ndarray, confidence: float) -> float:
    """Compute CVaR as the mean of losses beyond the VaR threshold."""
    losses = -returns
    var = np.quantile(losses, confidence)
    tail = losses[losses >= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))


def cvar_trials(
    model: GaussianModel,
    confidence: float,
    n_samples: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Repeated CVaR estimates for error analysis."""
    estimates = np.empty(trials, dtype=float)
    for i in range(trials):
        returns = model.sample_returns(rng, n_samples)
        estimates[i] = cvar_from_returns(returns, confidence)
    return estimates


def plot_convergence(
    output_path: Path,
    sample_sizes: list[int],
    mean_estimates: list[float],
    std_estimates: list[float],
    true_cvar: float,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(sample_sizes, mean_estimates, yerr=std_estimates, fmt="o-")
    plt.axhline(true_cvar, color="black", linestyle="--", label="Reference CVaR")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("CVaR")
    plt.title("Monte Carlo CVaR Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_scaling(
    output_path: Path,
    sample_sizes: list[int],
    abs_errors: list[float],
    fit_slope: float,
    fit_intercept: float,
) -> None:
    sizes_np = np.asarray(sample_sizes, dtype=float)
    errors_np = np.asarray(abs_errors, dtype=float)
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes_np, errors_np, "o-", label="Mean absolute error")
    fit_line = np.exp(fit_intercept) * sizes_np ** fit_slope
    plt.loglog(sizes_np, fit_line, "--", label=f"Fit slope {fit_slope:.2f}")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("Mean absolute error (log scale)")
    plt.title("Monte Carlo CVaR Error Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run(args: argparse.Namespace) -> None:
    model = GaussianModel(mu=args.mu, sigma=args.sigma)

    seed = args.seed
    if seed is None:
        seed = int(time.time() * 1_000_000) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    # High-sample reference CVaR for error measurement.
    reference_returns = model.sample_returns(rng, args.reference_samples)
    true_cvar = cvar_from_returns(reference_returns, args.confidence)

    results = []
    mean_estimates = []
    std_estimates = []
    abs_errors = []

    for n_samples in args.sample_sizes:
        estimates = cvar_trials(
            model, args.confidence, n_samples, args.trials, rng
        )
        summary = summarize_trials(estimates, true_cvar)
        results.append(
            {
                "samples": n_samples,
                "mean_cvar": summary["mean"],
                "std_cvar": summary["std"],
                "abs_err": summary["abs_err"],
                "rmse": summary["rmse"],
            }
        )
        mean_estimates.append(summary["mean"])
        std_estimates.append(summary["std"])
        abs_errors.append(summary["abs_err"])

    log_n = np.log(np.asarray(args.sample_sizes, dtype=float))
    log_err = np.log(np.asarray(abs_errors, dtype=float))
    fit_slope, fit_intercept = np.polyfit(log_n, log_err, 1)

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "cvar_error_scaling.csv",
        "cvar_summary.json",
        "cvar_convergence.png",
        "cvar_error_scaling.png",
    ):
        target = output_dir / filename
        if target.exists():
            target.unlink()

    write_csv(output_dir / "cvar_error_scaling.csv", results)

    plot_convergence(
        output_dir / "cvar_convergence.png",
        args.sample_sizes,
        mean_estimates,
        std_estimates,
        true_cvar,
    )
    plot_error_scaling(
        output_dir / "cvar_error_scaling.png",
        args.sample_sizes,
        abs_errors,
        fit_slope,
        fit_intercept,
    )

    summary = {
        "model": asdict(model),
        "confidence": args.confidence,
        "true_cvar": true_cvar,
        "trials": args.trials,
        "sample_sizes": args.sample_sizes,
        "seed": seed,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
    }
    (output_dir / "cvar_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("CVaR run complete")
    print(f"Reference CVaR: {true_cvar:.6f}")
    print(f"Fit slope (log-log error vs N): {fit_slope:.3f}")
    print(f"Seed: {seed}")
    print(f"Outputs: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical Monte Carlo CVaR experiments")
    parser.add_argument("--mu", type=float, default=0.0005)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--reference-samples", type=int, default=200_000)
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=default_sample_sizes(),
        help="Comma-separated list of sample sizes",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "writeup"),
        help="Output directory for figures and CSV.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
