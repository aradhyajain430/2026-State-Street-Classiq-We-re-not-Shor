import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from var_smart import JumpDiffusionModel, calculate_jump_var
from var_core import summarize_trials

def fit_jump_diffusion(ticker, threshold_std=3.0):
    df = yf.download(ticker, period="2y", progress=False, multi_level_index=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    mu_total, sigma_total = returns.mean(), returns.std()
    is_jump = np.abs(returns - mu_total) > (threshold_std * sigma_total)
    jumps, diffusion = returns[is_jump], returns[~is_jump]
    
    return JumpDiffusionModel(
        mu=float(diffusion.mean()),
        sigma=float(diffusion.std()),
        lamb=float(len(jumps) / len(returns)),
        jump_mu=float(jumps.mean() if len(jumps) > 0 else 0),
        jump_sigma=float(jumps.std() if len(jumps) > 0 else 0)
    )

def plot_convergence(path, sizes, means, stds, truth):
    plt.figure(figsize=(8, 5))
    plt.errorbar(sizes, means, yerr=stds, fmt="o-", label="Poisson MC mean +/- 1 std")
    plt.axhline(truth, color="red", linestyle="--", label="Ground Truth (1M Samples)")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("VaR")
    plt.title("Poisson VaR Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_error_scaling(path, sizes, errors, slope, intercept):
    sizes_np, errors_np = np.array(sizes), np.array(errors)
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes_np, errors_np, "o-", label="Mean absolute error")
    fit_line = np.exp(intercept) * sizes_np ** slope
    plt.loglog(sizes_np, fit_line, "--", label=f"Fit slope {slope:.2f}")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("MAE (log scale)")
    plt.title("Poisson Error Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    # Settings
    ticker = "AAPL"
    confidence = 0.95
    trials = 100
    sample_sizes = np.logspace(start=2, stop=5, num=20, base=10, dtype=int)
    outdir = Path("smart/results")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Fit and setup
    model = fit_jump_diffusion(ticker)
    rng = np.random.default_rng()
    
    # 2. Establish "True VaR" via high-sample MC
    print(f"Generating Poisson Ground Truth for {ticker}...")
    true_var = calculate_jump_var(model, confidence, 1_000_000)

    # 3. Trial Loop (Convergence logic)
    results_csv = []
    means, stds, errors = [], [], []

    for n in sample_sizes:
        print(f"Running {trials} trials for N={n}...")
        estimates = []
        for _ in range(trials):
            rets = model.sample_returns(rng, n)
            estimates.append(np.quantile(-rets, confidence))
        
        summary = summarize_trials(estimates, true_var)
        results_csv.append({"samples": n, "mean_var": summary["mean"], "abs_err": summary["abs_err"]})
        means.append(summary["mean"]); stds.append(summary["std"]); errors.append(summary["abs_err"])

    # 4. Error Scaling Fit
    log_n, log_err = np.log(sample_sizes), np.log(errors)
    slope, intercept = np.polyfit(log_n, log_err, 1)

    # 5. Generate Files
    # CSV
    with open(outdir / "error_scaling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["samples", "mean_var", "abs_err"])
        writer.writeheader(); writer.writerows(results_csv)
    
    # PNGs
    plot_convergence(outdir / "var_convergence.png", sample_sizes, means, stds, true_var)
    plot_error_scaling(outdir / "error_scaling.png", sample_sizes, errors, slope, intercept)
    
    # JSON
    summary_json = {
        "model": asdict(model),
        "confidence": confidence,
        "true_var": true_var,
        "fit_slope": float(slope),
        "trials": trials
    }
    (outdir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    print(f"\nDone! Files generated in {outdir}")

if __name__ == "__main__":
    main()
