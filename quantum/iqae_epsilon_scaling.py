import argparse
import csv
import logging
import os
import sys
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


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="IQAE scaling vs epsilon (queries ~ 1/epsilon)."
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
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
    parser.add_argument("--out", type=str, default="quantum/iqae_epsilon_scaling.png")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--show", action="store_true")
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

    grid, probs = set_gaussian_state_params(
        mu=args.mu,
        sigma=args.sigma,
        num_qubits=args.num_qubits,
        num_sigmas=args.num_sigmas,
        prep_bound=args.prep_bound,
    )
    cdf = np.cumsum(probs)
    index = int(np.searchsorted(cdf, args.alpha))
    true_alpha = float(cdf[index])
    iqae_var_run.GLOBAL_INDEX = index
    LOGGER.info("Using index=%s true_alpha=%.6f", index, true_alpha)
    analytic_var = float(args.mu + args.sigma * scipy.stats.norm.ppf(args.alpha))

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

    results = []
    csv_rows = []
    for eps in epsilons:
        LOGGER.info("Running IQAE for epsilon=%.6f", eps)
        res = iqae.run(
            epsilon=eps,
            alpha=0.01,
            execution_preferences=exec_prefs,
        )
        queries = total_queries(res.iterations_data)
        alpha_est = float(res.estimation)
        alpha_est = float(np.clip(alpha_est, 1e-12, 1 - 1e-12))
        alpha_error = abs(alpha_est - true_alpha)
        var_est = float(args.mu + args.sigma * scipy.stats.norm.ppf(alpha_est))
        var_error = abs(var_est - analytic_var)
        results.append((eps, queries, alpha_error, var_error))
        csv_rows.append(
            {
                "epsilon": float(eps),
                "queries": int(queries),
                "alpha_est": float(alpha_est),
                "alpha_error": float(alpha_error),
                "var_est": float(var_est),
                "var_error": float(var_error),
                "true_alpha": float(true_alpha),
            }
        )

    eps_arr = np.array([r[0] for r in results], dtype=float)
    queries_arr = np.array([r[1] for r in results], dtype=float)
    error_arr = np.array([r[2] for r in results], dtype=float)

    if args.csv:
        write_csv(args.csv, csv_rows)

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
    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
