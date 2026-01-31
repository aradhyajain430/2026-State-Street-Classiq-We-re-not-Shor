import argparse
import logging
import os
import sys
from typing import Callable, Tuple, List

import numpy as np

from classiq import Const, QArray, QBit, QNum, qfunc, qperm
from classiq.applications.iqae.iqae import IQAE
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.generator.model import Constraints, Preferences

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quantum.var_state_prep import set_gaussian_state_params, load_gaussian_state

GLOBAL_INDEX = 0
LOGGER = logging.getLogger(__name__)
_IQAE_CACHE: dict[tuple[int, int | None, int | None], IQAE] = {}
_EXEC_PREFS: ExecutionPreferences | None = None


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    # Prepare the Gaussian amplitude distribution and mark tail events.
    load_gaussian_state(asset)
    payoff(asset=asset, ind=ind)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    # Mark outcomes where the asset index is below the threshold.
    ind ^= asset < GLOBAL_INDEX


def calc_alpha_classical(index: int, probs: List[float]) -> float:
    return float(np.sum(probs[:index]))


def value_at_risk(
    required_alpha: float,
    index: int,
    probs: List[float],
    calc_alpha_func: Callable[[int, List[float]], float],
    tolerance: float,
) -> int:
    alpha_v = calc_alpha_func(index, probs)
    search_size = max(1, index // 2)
    LOGGER.info(
        "Starting bisection: index=%s alpha_v=%.6f search_size=%s",
        index,
        alpha_v,
        search_size,
    )

    while not np.isclose(alpha_v, required_alpha, atol=tolerance) and search_size > 0:
        if alpha_v < required_alpha:
            index += search_size
        else:
            index -= search_size
        search_size //= 2
        alpha_v = calc_alpha_func(index, probs)
        LOGGER.info(
            "Bisection step: index=%s alpha_v=%.6f next_search_size=%s",
            index,
            alpha_v,
            search_size,
        )

    return index


def _get_iqae(
    num_qubits: int,
    max_width: int | None,
    machine_precision: int | None,
) -> IQAE:
    key = (num_qubits, max_width, machine_precision)
    cached = _IQAE_CACHE.get(key)
    if cached is not None:
        return cached

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
    iqae.get_qprog()
    _IQAE_CACHE[key] = iqae
    return iqae


def run_iqae_result(
    index: int,
    num_qubits: int,
    epsilon: float,
    confidence_alpha: float,
    max_width: int | None,
    machine_precision: int | None,
    num_shots: int | None = None,
):
    global GLOBAL_INDEX
    GLOBAL_INDEX = int(index)

    iqae = _get_iqae(
        num_qubits=num_qubits,
        max_width=max_width,
        machine_precision=machine_precision,
    )

    LOGGER.info(
        "Running IQAE: index=%s epsilon=%.4f confidence_alpha=%.4f",
        index,
        epsilon,
        confidence_alpha,
    )
    exec_prefs = None
    if num_shots is not None:
        global _EXEC_PREFS
        if _EXEC_PREFS is None:
            _EXEC_PREFS = ExecutionPreferences(num_shots=num_shots)
        else:
            _EXEC_PREFS.num_shots = num_shots
        exec_prefs = _EXEC_PREFS
    iqae_res = iqae.run(
        epsilon=epsilon,
        alpha=confidence_alpha,
        execution_preferences=exec_prefs,
    )
    LOGGER.info(
        "IQAE done: estimation=%.6f CI=[%.6f, %.6f]",
        iqae_res.estimation,
        float(iqae_res.confidence_interval[0]),
        float(iqae_res.confidence_interval[1]),
    )
    return iqae_res


def calc_alpha_quantum(
    index: int,
    num_qubits: int,
    epsilon: float,
    confidence_alpha: float,
    max_width: int | None,
    machine_precision: int | None,
    num_shots: int | None = None,
) -> float:
    iqae_res = run_iqae_result(
        index=index,
        num_qubits=num_qubits,
        epsilon=epsilon,
        confidence_alpha=confidence_alpha,
        max_width=max_width,
        machine_precision=machine_precision,
        num_shots=num_shots,
    )
    return float(iqae_res.estimation)


def run_iqae_var(
    mu: float,
    sigma: float,
    num_qubits: int,
    alpha: float,
    epsilon: float,
    confidence_alpha: float,
    num_sigmas: float,
    prep_bound: float,
    max_width: int | None,
    machine_precision: int | None,
    num_shots: int | None = None,
) -> Tuple[float, float]:
    grid, probs = set_gaussian_state_params(
        mu=mu,
        sigma=sigma,
        num_qubits=num_qubits,
        num_sigmas=num_sigmas,
        prep_bound=prep_bound,
    )

    probs_list = probs
    tolerance = alpha / 10

    # Classical baseline
    classical_index = int(np.searchsorted(np.cumsum(probs_list), alpha))
    classical_var = float(grid[classical_index])
    LOGGER.info("Classical VaR index=%s value=%.6f", classical_index, classical_var)

    # Quantum-assisted bisection using IQAE
    initial_index = max(1, (2**num_qubits) // 4)
    quantum_index = value_at_risk(
        required_alpha=alpha,
        index=initial_index,
        probs=probs_list,
        calc_alpha_func=lambda idx, _: calc_alpha_quantum(
            idx,
            num_qubits=num_qubits,
            epsilon=epsilon,
            confidence_alpha=confidence_alpha,
            max_width=max_width,
            machine_precision=machine_precision,
            num_shots=num_shots,
        ),
        tolerance=tolerance,
    )
    quantum_var = float(grid[quantum_index])
    LOGGER.info("Quantum VaR index=%s value=%.6f", quantum_index, quantum_var)

    return classical_var, quantum_var


def main():
    parser = argparse.ArgumentParser(
        description="IQAE-based VaR estimation for a Gaussian distribution."
    )
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
    parser.add_argument("--num-shots", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--no-quantum",
        action="store_true",
        help="Skip IQAE run and only print the classical VaR.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    grid, probs = set_gaussian_state_params(
        mu=args.mu,
        sigma=args.sigma,
        num_qubits=args.num_qubits,
        num_sigmas=args.num_sigmas,
        prep_bound=args.prep_bound,
    )

    cdf = np.cumsum(probs)
    classical_index = int(np.searchsorted(cdf, args.alpha))
    classical_var = float(grid[classical_index])
    print(f"Classical VaR (alpha={args.alpha:.4f}): {classical_var:.6f}")

    if args.no_quantum:
        return

    classical_var, quantum_var = run_iqae_var(
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
        num_shots=args.num_shots,
    )
    print(f"IQAE VaR (alpha={args.alpha:.4f}): {quantum_var:.6f}")


if __name__ == "__main__":
    main()
