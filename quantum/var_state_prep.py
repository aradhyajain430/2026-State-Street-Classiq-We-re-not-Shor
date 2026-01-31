import argparse
import math
from typing import Tuple, List

import numpy as np

from classiq import (
    Output,
    QArray,
    QBit,
    QNum,
    inplace_prepare_state,
    prepare_state,
    qfunc,
    synthesize,
)

GAUSSIAN_PROBS: List[float] = []
GAUSSIAN_PREP_BOUND: float = 0.0


def gaussian_grid(
    mu: float,
    sigma: float,
    num_qubits: int,
    num_sigmas: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return grid points and normalized probabilities for a Gaussian."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    n = 2**num_qubits
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma
    grid = np.linspace(low, high, n)

    z = (grid - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
    probs = pdf / np.sum(pdf)
    return grid, probs


def set_gaussian_state_params(
    mu: float,
    sigma: float,
    num_qubits: int,
    num_sigmas: float = 3.0,
    prep_bound: float = 0.0,
) -> Tuple[np.ndarray, List[float]]:
    """Compute the Gaussian PMF and store it for state preparation."""
    grid, probs = gaussian_grid(mu, sigma, num_qubits, num_sigmas=num_sigmas)
    global GAUSSIAN_PROBS, GAUSSIAN_PREP_BOUND
    GAUSSIAN_PROBS = probs.tolist()
    GAUSSIAN_PREP_BOUND = float(prep_bound)
    return grid, GAUSSIAN_PROBS


@qfunc(synthesize_separately=True)
def load_gaussian_state(asset: QNum):
    # QNum is a numeric view of a qubit array, suitable for comparisons later.
    inplace_prepare_state(
        probabilities=GAUSSIAN_PROBS, bound=GAUSSIAN_PREP_BOUND, target=asset
    )


@qfunc(synthesize_separately=True)
def load_gaussian_state_qarray(asset: QArray[QBit]):
    # Same as load_gaussian_state, but typed explicitly as a qubit array.
    inplace_prepare_state(
        probabilities=GAUSSIAN_PROBS, bound=GAUSSIAN_PREP_BOUND, target=asset
    )


@qfunc
def main(out: Output[QArray[QBit]]) -> None:
    # End-to-end entry point: allocate and prepare the Gaussian state in one step.
    prepare_state(probabilities=GAUSSIAN_PROBS, bound=GAUSSIAN_PREP_BOUND, out=out)


def build_state_prep_qprog(
    mu: float = 0.0,
    sigma: float = 1.0,
    num_qubits: int = 7,
    num_sigmas: float = 3.0,
    prep_bound: float = 0.0,
):
    model = build_state_prep_model(
        mu=mu,
        sigma=sigma,
        num_qubits=num_qubits,
        num_sigmas=num_sigmas,
        prep_bound=prep_bound,
    )
    return synthesize(model.get_model())


def build_state_prep_model(
    mu: float = 0.0,
    sigma: float = 1.0,
    num_qubits: int = 7,
    num_sigmas: float = 3.0,
    prep_bound: float = 0.0,
):
    set_gaussian_state_params(
        mu=mu,
        sigma=sigma,
        num_qubits=num_qubits,
        num_sigmas=num_sigmas,
        prep_bound=prep_bound,
    )
    return main.create_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a Gaussian state using Classiq."
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--num-sigmas", type=float, default=3.0)
    parser.add_argument("--prep-bound", type=float, default=0.0)
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Call Classiq synthesis (requires network access).",
    )
    args = parser.parse_args()

    if args.synthesize:
        qprog = build_state_prep_qprog(
            mu=args.mu,
            sigma=args.sigma,
            num_qubits=args.num_qubits,
            num_sigmas=args.num_sigmas,
            prep_bound=args.prep_bound,
        )
        prog = qprog.to_program()
        print(f"Synthesized program id: {qprog.program_id}")
        print(f"Program syntax: {prog.syntax}")
        print(prog.code[:1000])
    else:
        model = build_state_prep_model(
            mu=args.mu,
            sigma=args.sigma,
            num_qubits=args.num_qubits,
            num_sigmas=args.num_sigmas,
            prep_bound=args.prep_bound,
        )
        model_json = model.get_model()
        print("Model built locally (no synthesis).")
        print(model_json[:1000])
