"""Run the full CVaR pipeline (classical + IQAE + combined panel)."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    python = sys.executable
    run([python, str(ROOT / "cvar" / "run_cvar.py")])
    run([python, str(ROOT / "cvar" / "iqae_cvar_epsilon_scaling.py")])
    run([python, str(ROOT / "cvar" / "iqae_cvar_shots_scaling.py")])
    run([python, str(ROOT / "cvar" / "combine_cvar_panels.py")])


if __name__ == "__main__":
    main()
