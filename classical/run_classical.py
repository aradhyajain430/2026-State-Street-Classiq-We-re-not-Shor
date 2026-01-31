"""Run classical Monte Carlo VaR experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from var_core import GaussianModel, monte_carlo_var, theoretical_var_gaussian, var_trials, summarize_trials


def default_sample_sizes() -> list[int]:
    # 30 logarithmically spaced points from 100 to 100,000 (inclusive).
    raw = np.logspace(2, 5, num=30)
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


def plot_convergence(
    output_path: Path,
    sample_sizes: list[int],
    mean_estimates: list[float],
    std_estimates: list[float],
    true_var: float,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(sample_sizes, mean_estimates, yerr=std_estimates, fmt="o-", label="MC mean +/- 1 std")
    plt.axhline(true_var, color="black", linestyle="--", label="Theoretical VaR")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("VaR")
    plt.title("Monte Carlo VaR Convergence")
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
    sample_sizes_np = np.asarray(sample_sizes, dtype=float)
    abs_errors_np = np.asarray(abs_errors, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.loglog(sample_sizes_np, abs_errors_np, "o-", label="Mean absolute error")

    fit_line = np.exp(fit_intercept) * sample_sizes_np ** fit_slope
    plt.loglog(sample_sizes_np, fit_line, "--", label=f"Fit slope {fit_slope:.2f}")

    theory_line = abs_errors_np[0] * (sample_sizes_np[0] / sample_sizes_np) ** 0.5
    plt.loglog(sample_sizes_np, theory_line, ":", label="O(1/sqrt(N)) reference")

    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("Mean absolute error (log scale)")
    plt.title("Monte Carlo Error Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run(args: argparse.Namespace) -> None:
    model = GaussianModel(mu=args.mu, sigma=args.sigma)
    confidence = args.confidence
    portfolio_value = args.portfolio

    seed = args.seed
    if seed is None:
        # Millisecond-resolution seed folded into uint32 range for NumPy.
        seed = int(time.time() * 1_000_000) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    true_var = theoretical_var_gaussian(model, confidence, portfolio_value)
    single_run = monte_carlo_var(model, confidence, args.single_samples, portfolio_value, rng)

    results = []
    mean_estimates = []
    std_estimates = []
    abs_errors = []

    for n_samples in args.sample_sizes:
        estimates = var_trials(model, confidence, n_samples, args.trials, portfolio_value, rng)
        summary = summarize_trials(estimates, true_var)
        results.append(
            {
                "samples": n_samples,
                "mean_var": summary["mean"],
                "std_var": summary["std"],
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
    # Ensure deterministic overwrite for common outputs.
    for filename in (
        "error_scaling.csv",
        "summary.json",
        "var_convergence.png",
        "error_scaling.png",
    ):
        target = output_dir / filename
        if target.exists():
            target.unlink()

    write_csv(output_dir / "error_scaling.csv", results)

    plot_convergence(output_dir / "var_convergence.png", args.sample_sizes, mean_estimates, std_estimates, true_var)
    plot_error_scaling(output_dir / "error_scaling.png", args.sample_sizes, abs_errors, fit_slope, fit_intercept)

    summary = {
        "model": asdict(model),
        "confidence": confidence,
        "portfolio_value": portfolio_value,
        "true_var": true_var,
        "single_run_var": single_run,
        "single_run_samples": args.single_samples,
        "trials": args.trials,
        "sample_sizes": args.sample_sizes,
        "seed": seed,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Classical VaR run complete")
    print(f"Theoretical VaR: {true_var:.6f}")
    print(f"Single-run VaR ({args.single_samples} samples): {single_run:.6f}")
    print(f"Fit slope (log-log error vs N): {fit_slope:.3f}")
    print(f"Seed: {seed}")
    print(f"Outputs: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical Monte Carlo VaR experiments")
    parser.add_argument("--mu", type=float, default=0.0005, help="Mean daily return")
    parser.add_argument("--sigma", type=float, default=0.02, help="Daily return volatility")
    parser.add_argument("--confidence", type=float, default=0.95, help="VaR confidence level")
    parser.add_argument("--portfolio", type=float, default=1.0, help="Portfolio value")
    parser.add_argument("--trials", type=int, default=200, help="Trials per sample size")
    parser.add_argument("--single-samples", type=int, default=10000, help="Samples for a single VaR estimate")
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=default_sample_sizes(),
        help="Comma-separated list of sample sizes",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: time-based)")
    parser.add_argument("--outdir", type=str, default="classical/results", help="Output directory")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
