import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from classiq.applications.iqae.iqae import IQAE
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.generator.model import Constraints, Preferences

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.var_state_prep import set_gaussian_state_params
import quantum.iqae_var_run as iqae_var_run

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


def total_queries(iterations_data) -> int:
    total = 0
    for it in iterations_data:
        k = int(it.grover_iterations)
        shots = it.sample_results.num_shots
        if shots is None:
            shots = sum(it.sample_results.counts.values())
        total += int(shots) * (2 * k + 1)
    return total


def gaussian_cvar_loss(mu: float, sigma: float, alpha: float) -> float:
    z = scipy.stats.norm.ppf(alpha)
    pdf = scipy.stats.norm.pdf(z)
    return float(-mu + sigma * pdf / alpha)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQAE CVaR scaling vs epsilon (queries ~ 1/epsilon)."
    )
    parser.add_argument("--mu", type=float, default=0.0005)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=64)
    parser.add_argument(
        "--epsilons",
        type=str,
        default="auto",
        help="Comma list or 'auto' for 12 points /1.4 starting at 0.2.",
    )
    parser.add_argument("--out", type=str, default="writeup/cvar_iqae_eps_scaling.png")
    parser.add_argument("--csv", type=str, default="cvar/cvar_iqae_eps_scaling.csv")
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

    alpha = 1.0 - args.confidence
    true_cvar = gaussian_cvar_loss(args.mu, args.sigma, alpha)

    grid, probs = set_gaussian_state_params(
        mu=args.mu,
        sigma=args.sigma,
        num_qubits=args.num_qubits,
        num_sigmas=args.num_sigmas,
        prep_bound=args.prep_bound,
    )
    cdf = np.cumsum(probs)
    index = int(np.searchsorted(cdf, alpha))
    iqae_var_run.GLOBAL_INDEX = index
    LOGGER.info("Using index=%s for alpha=%.6f", index, alpha)

    constraints = Constraints(max_width=args.max_width) if args.max_width else None
    preferences = (
        Preferences(machine_precision=args.machine_precision)
        if args.machine_precision
        else None
    )
    iqae = IQAE(
        state_prep_op=iqae_var_run.state_preparation,
        problem_vars_size=args.num_qubits,
        constraints=constraints,
        preferences=preferences,
    )
    exec_prefs = ExecutionPreferences(num_shots=args.num_shots)

    rows = []
    for eps in epsilons:
        LOGGER.info("Running IQAE for epsilon=%.6f", eps)
        res = iqae.run(
            epsilon=eps,
            alpha=0.01,
            execution_preferences=exec_prefs,
        )
        alpha_est = float(np.clip(res.estimation, 1e-12, 1 - 1e-12))
        cvar_est = gaussian_cvar_loss(args.mu, args.sigma, alpha_est)
        err = abs(cvar_est - true_cvar)
        queries = total_queries(res.iterations_data)
        rows.append(
            {
                "epsilon": float(eps),
                "queries": int(queries),
                "alpha_est": float(alpha_est),
                "cvar_est": float(cvar_est),
                "cvar_error": float(err),
            }
        )

    write_csv(Path(args.csv), rows)

    eps_arr = np.array([r["epsilon"] for r in rows], dtype=float)
    queries_arr = np.array([r["queries"] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(1.0 / eps_arr, queries_arr, "o-")
    ax[0].set_xlabel("1 / epsilon")
    ax[0].set_ylabel("Total oracle queries")
    ax[0].set_title("CVaR IQAE: Queries vs 1/epsilon")
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
