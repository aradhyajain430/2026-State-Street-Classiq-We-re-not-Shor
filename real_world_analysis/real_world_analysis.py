from __future__ import annotations

"""Real-world VaR analysis using a double-Poisson jump model.

This module fits a two-sided Poisson jump-diffusion model to historical
returns and then runs Monte Carlo VaR convergence + error scaling studies.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DoublePoissonJumpModel:
    """Two-sided Poisson jump-diffusion model for daily returns."""
    mu: float
    sigma: float
    lambda_pos: float
    lambda_neg: float
    jump_mu_pos: float
    jump_sigma_pos: float
    jump_mu_neg: float
    jump_sigma_neg: float

    def sample_returns(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")

        # Continuous diffusion component.
        diffusion = rng.normal(self.mu, self.sigma, size=n)
        num_pos = rng.poisson(self.lambda_pos, size=n)
        num_neg = rng.poisson(self.lambda_neg, size=n)

        jump_impact = np.zeros(n)
        for i in range(n):
            if num_pos[i] > 0:
                # Positive jump magnitudes; enforce positive contribution.
                pos_jumps = rng.normal(
                    self.jump_mu_pos, self.jump_sigma_pos, size=num_pos[i]
                )
                jump_impact[i] += np.abs(pos_jumps).sum()
            if num_neg[i] > 0:
                # Negative jump magnitudes; enforce negative contribution.
                neg_jumps = rng.normal(
                    self.jump_mu_neg, self.jump_sigma_neg, size=num_neg[i]
                )
                jump_impact[i] -= np.abs(neg_jumps).sum()

        return diffusion + jump_impact


def fetch_returns(ticker: str, period: str, interval: str) -> np.ndarray:
    """Download daily log returns for a ticker via yfinance."""
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )
    if data.empty or "Close" not in data:
        raise ValueError(f"No data found for ticker {ticker}")
    prices = data["Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns.to_numpy()


def fit_double_poisson(
    returns: np.ndarray, threshold_std: float
) -> DoublePoissonJumpModel:
    """Fit a double-Poisson jump model by thresholding outlier returns."""
    mu_total = float(np.mean(returns))
    sigma_total = float(np.std(returns, ddof=1))
    if sigma_total == 0.0:
        raise ValueError("Zero volatility in returns; cannot fit model.")

    # Identify jumps as extreme moves relative to total volatility.
    is_jump = np.abs(returns - mu_total) > (threshold_std * sigma_total)
    jumps = returns[is_jump]
    diffusion = returns[~is_jump]

    mu = float(np.mean(diffusion))
    sigma = float(np.std(diffusion, ddof=1)) if diffusion.size > 1 else 0.0

    # Split jump sizes by sign to capture skew.
    pos_jumps = jumps[jumps > 0.0]
    neg_jumps = -jumps[jumps < 0.0]

    lambda_pos = float(len(pos_jumps) / len(returns))
    lambda_neg = float(len(neg_jumps) / len(returns))

    jump_mu_pos = float(np.mean(pos_jumps)) if pos_jumps.size > 0 else 0.0
    jump_sigma_pos = (
        float(np.std(pos_jumps, ddof=1)) if pos_jumps.size > 1 else 0.0
    )
    jump_mu_neg = float(np.mean(neg_jumps)) if neg_jumps.size > 0 else 0.0
    jump_sigma_neg = (
        float(np.std(neg_jumps, ddof=1)) if neg_jumps.size > 1 else 0.0
    )

    return DoublePoissonJumpModel(
        mu=mu,
        sigma=sigma,
        lambda_pos=lambda_pos,
        lambda_neg=lambda_neg,
        jump_mu_pos=jump_mu_pos,
        jump_sigma_pos=jump_sigma_pos,
        jump_mu_neg=jump_mu_neg,
        jump_sigma_neg=jump_sigma_neg,
    )


def monte_carlo_var(
    model: DoublePoissonJumpModel,
    confidence: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo VaR for the fitted model."""
    returns = model.sample_returns(rng, n_samples)
    losses = -returns
    return float(np.quantile(losses, confidence))


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_fit_overlay(
    output_path: Path, historical: np.ndarray, simulated: np.ndarray, title: str
) -> None:
    """Overlay empirical returns with fitted model samples."""
    plt.figure(figsize=(8, 5))
    plt.hist(
        historical,
        bins=80,
        density=True,
        alpha=0.6,
        label="Historical returns",
    )
    plt.hist(
        simulated,
        bins=80,
        density=True,
        alpha=0.5,
        label="Double-Poisson fit",
    )
    plt.xlabel("Daily log return")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_convergence(
    output_path: Path,
    sample_sizes: list[int],
    mean_estimates: list[float],
    std_estimates: list[float],
    true_var: float,
    title: str,
) -> None:
    """Plot MC convergence with error bars around the mean estimate."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(sample_sizes, mean_estimates, yerr=std_estimates, fmt="o-")
    plt.axhline(true_var, color="black", linestyle="--", label="Reference VaR")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("VaR")
    plt.title(title)
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
    title: str,
) -> None:
    """Plot log-log error scaling and fitted slope."""
    sizes_np = np.asarray(sample_sizes, dtype=float)
    errors_np = np.asarray(abs_errors, dtype=float)
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes_np, errors_np, "o-", label="Mean absolute error")
    fit_line = np.exp(fit_intercept) * sizes_np ** fit_slope
    plt.loglog(sizes_np, fit_line, "--", label=f"Fit slope {fit_slope:.2f}")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("Mean absolute error (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def default_sample_sizes() -> list[int]:
    raw = np.logspace(2, 5, num=20)
    sizes: list[int] = []
    for value in raw:
        candidate = int(round(value))
        if sizes:
            candidate = max(candidate, sizes[-1] + 1)
        sizes.append(candidate)
    sizes[-1] = 100_000
    return sizes


def run_for_ticker(
    ticker: str,
    args: argparse.Namespace,
    out_root: Path,
) -> None:
    """Run the full fit + MC pipeline for a single ticker."""
    returns = fetch_returns(ticker, args.period, args.interval)

    # Fit the two-sided jump model from historical data.
    model = fit_double_poisson(returns, args.threshold_std)

    rng = np.random.default_rng(args.seed)

    # High-sample Monte Carlo reference for error analysis.
    true_var = monte_carlo_var(model, args.confidence, args.truth_samples, rng)

    results = []
    mean_estimates = []
    std_estimates = []
    abs_errors = []

    for n_samples in args.sample_sizes:
        estimates = []
        for _ in range(args.trials):
            estimates.append(
                monte_carlo_var(model, args.confidence, n_samples, rng)
            )
        estimates_np = np.asarray(estimates, dtype=float)
        mean_estimates.append(float(np.mean(estimates_np)))
        std_estimates.append(float(np.std(estimates_np, ddof=1)) if len(estimates) > 1 else 0.0)
        abs_errors.append(float(np.mean(np.abs(estimates_np - true_var))))
        results.append(
            {
                "samples": n_samples,
                "mean_var": float(np.mean(estimates_np)),
                "std_var": float(np.std(estimates_np, ddof=1)) if len(estimates) > 1 else 0.0,
                "abs_err": float(np.mean(np.abs(estimates_np - true_var))),
            }
        )

    log_n = np.log(np.asarray(args.sample_sizes, dtype=float))
    log_err = np.log(np.asarray(abs_errors, dtype=float))
    fit_slope, fit_intercept = np.polyfit(log_n, log_err, 1)

    out_dir = out_root / ticker.replace("=", "").replace("^", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "error_scaling.csv", results)

    # Diagnostic: compare empirical returns vs model samples.
    simulated = model.sample_returns(rng, args.fit_samples)
    plot_fit_overlay(
        out_dir / "fit_overlay.png",
        returns,
        simulated,
        f"{ticker} returns: empirical vs double-Poisson fit",
    )
    plot_convergence(
        out_dir / "var_convergence.png",
        args.sample_sizes,
        mean_estimates,
        std_estimates,
        true_var,
        f"{ticker} VaR convergence (double-Poisson)",
    )
    plot_error_scaling(
        out_dir / "error_scaling.png",
        args.sample_sizes,
        abs_errors,
        fit_slope,
        fit_intercept,
        f"{ticker} MC error scaling (double-Poisson)",
    )

    summary = {
        "ticker": ticker,
        "period": args.period,
        "interval": args.interval,
        "confidence": args.confidence,
        "threshold_std": args.threshold_std,
        "truth_samples": args.truth_samples,
        "fit_samples": args.fit_samples,
        "trials": args.trials,
        "sample_sizes": args.sample_sizes,
        "seed": args.seed,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "model": asdict(model),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done: {ticker} -> {out_dir}")


def parse_sample_sizes(raw: str) -> list[int]:
    """Parse comma-separated sample sizes (fallback to default grid)."""
    if not raw:
        return default_sample_sizes()
    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(float(part)))
    return sizes


def build_parser() -> argparse.ArgumentParser:
    """CLI for the real-world analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Real-world VaR analysis using a double-Poisson jump model."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="GC=F,^GSPC",
        help="Comma-separated tickers (default: GC=F,^GSPC).",
    )
    parser.add_argument("--period", type=str, default="5y", help="yfinance period")
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--threshold-std", type=float, default=3.0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--truth-samples", type=int, default=200_000)
    parser.add_argument("--fit-samples", type=int, default=200_000)
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=default_sample_sizes(),
        help="Comma-separated list of MC sample sizes.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = int(time.time() * 1_000_000) % (2**32 - 1)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        run_for_ticker(ticker, args, out_root)


if __name__ == "__main__":
    main()
