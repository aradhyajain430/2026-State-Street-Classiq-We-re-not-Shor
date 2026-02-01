import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import scipy.stats

from classiq import Const, QArray, QBit, QNum, qfunc, qperm
from classiq.applications.iqae.iqae import IQAE
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.generator.model import Constraints, Preferences

import classiq

classiq.authenticate()

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.var_state_prep import set_gaussian_state_params, load_gaussian_state

LOGGER = logging.getLogger(__name__)
GLOBAL_INDEX = 0


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    load_gaussian_state(asset)
    payoff(asset=asset, ind=ind)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    ind ^= asset < GLOBAL_INDEX


def calc_alpha_quantum(
    index: int,
    num_qubits: int,
    epsilon: float,
    confidence_alpha: float,
    num_shots: int | None,
    max_width: int | None,
    machine_precision: int | None,
) -> float:
    global GLOBAL_INDEX
    GLOBAL_INDEX = int(index)

    constraints = Constraints(max_width=max_width) if max_width else None
    preferences = (
        Preferences(machine_precision=machine_precision)
        if machine_precision
        else None
    )
    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=constraints,
        preferences=preferences,
    )
    exec_prefs = (
        ExecutionPreferences(num_shots=num_shots) if num_shots is not None else None
    )
    res = iqae.run(
        epsilon=epsilon, alpha=confidence_alpha, execution_preferences=exec_prefs
    )
    return float(res.estimation)


def classical_bisection_index(required_alpha: float, probs: List[float]) -> int:
    cdf = np.cumsum(probs)
    left = 0
    right = len(cdf) - 1
    while left < right:
        mid = (left + right) // 2
        if cdf[mid] < required_alpha:
            left = mid + 1
        else:
            right = mid
    return left


def run_iqae_var(
    num_shots: int | None,
    mu: float = 0.0,
    sigma: float = 1.0,
    num_qubits: int = 7,
    alpha: float = 0.05,
    epsilon: float = 0.05,
    confidence_alpha: float = 0.01,
    num_sigmas: float = 3.0,
    prep_bound: float = 0.0,
    max_width: int | None = 28,
    machine_precision: int | None = None,
) -> Dict[str, float]:
    grid, probs = set_gaussian_state_params(
        mu=mu,
        sigma=sigma,
        num_qubits=num_qubits,
        num_sigmas=num_sigmas,
        prep_bound=prep_bound,
    )
    probs_list = probs
    classical_index = classical_bisection_index(alpha, probs_list)
    classical_var = float(grid[classical_index])

    alpha_est = calc_alpha_quantum(
        index=classical_index,
        num_qubits=num_qubits,
        epsilon=epsilon,
        confidence_alpha=confidence_alpha,
        num_shots=num_shots,
        max_width=max_width,
        machine_precision=machine_precision,
    )
    alpha_est = float(np.clip(alpha_est, 1e-12, 1 - 1e-12))
    quantum_var = float(mu + sigma * scipy.stats.norm.ppf(alpha_est))

    analytic_var = float(mu + sigma * scipy.stats.norm.ppf(alpha))
    error = abs(quantum_var - analytic_var)

    return {
        "quantum_var": quantum_var,
        "analytic_var": analytic_var,
        "classical_var": classical_var,
        "alpha_est": alpha_est,
        "error": error,
        "num_shots": float(num_shots) if num_shots is not None else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run IQAE VaR once with a specified number of shots."
    )
    parser.add_argument("--num-shots", type=int, default=1024)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--confidence-alpha", type=float, default=0.01)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument("--max-width", type=int, default=28)
    parser.add_argument("--machine-precision", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    res = run_iqae_var(
        num_shots=args.num_shots,
        mu=args.mu,
        sigma=args.sigma,
        num_qubits=args.num_qubits,
        alpha=args.alpha,
        epsilon=args.epsilon,
        confidence_alpha=args.confidence_alpha,
        num_sigmas=args.num_sigmas,
        prep_bound=args.prep_bound,
        max_width=args.max_width,
        machine_precision=args.machine_precision,
    )
    print(
        "quantum_var={quantum_var:.6f} analytic_var={analytic_var:.6f} "
        "error={error:.6f} num_shots={num_shots}".format(**res)
    )


if __name__ == "__main__":
    main()
