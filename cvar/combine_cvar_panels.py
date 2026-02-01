import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine CVaR plots into a single multi-panel figure."
    )
    parser.add_argument(
        "--writeup-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "writeup",
        help="Directory containing the CVaR PNGs (writeup/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "writeup" / "cvar_multi_panel.png",
        help="Output combined PNG path.",
    )
    args = parser.parse_args()

    writeup_dir = args.writeup_dir
    images = [
        load_image(writeup_dir / "cvar_convergence.png"),
        load_image(writeup_dir / "cvar_error_scaling.png"),
        load_image(writeup_dir / "cvar_iqae_eps_scaling.png"),
        load_image(writeup_dir / "cvar_iqae_shots_scaling.png"),
    ]
    titles = [
        "CVaR convergence (MC)",
        "CVaR error scaling (MC)",
        "CVaR IQAE: queries vs 1/epsilon",
        "CVaR IQAE: error vs shots",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved combined figure to: {args.out}")


if __name__ == "__main__":
    main()
