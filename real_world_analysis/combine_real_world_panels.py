import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine real-world plots into a single multi-panel figure."
    )
    parser.add_argument(
        "--writeup-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "writeup",
        help="Directory containing the real-world PNGs (writeup/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "writeup"
        / "real_world_multi_panel.png",
        help="Output combined PNG path.",
    )
    args = parser.parse_args()

    writeup_dir = args.writeup_dir
    images = {
        "Gold fit": load_image(writeup_dir / "real_world_gold_fit.png"),
        "S&P 500 fit": load_image(writeup_dir / "real_world_spx_fit.png"),
        "Gold convergence": load_image(writeup_dir / "real_world_gold_convergence.png"),
        "S&P 500 convergence": load_image(writeup_dir / "real_world_spx_convergence.png"),
        "Gold error scaling": load_image(writeup_dir / "real_world_gold_error_scaling.png"),
        "S&P 500 error scaling": load_image(writeup_dir / "real_world_spx_error_scaling.png"),
        "IQAE epsilon scaling (Gold)": load_image(writeup_dir / "real_world_iqae_eps.png"),
        "IQAE epsilon scaling (S&P 500)": load_image(
            writeup_dir / "real_world_iqae_eps_spx.png"
        ),
    }

    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    titles = [
        "Gold: empirical vs fit",
        "S&P 500: empirical vs fit",
        "Gold: VaR convergence",
        "S&P 500: VaR convergence",
        "Gold: error scaling",
        "S&P 500: error scaling",
    ]
    keys = [
        "Gold fit",
        "S&P 500 fit",
        "Gold convergence",
        "S&P 500 convergence",
        "Gold error scaling",
        "S&P 500 error scaling",
    ]

    for ax, title, key in zip(axes, titles, keys):
        ax.imshow(images[key])
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax_iqae_gold = fig.add_subplot(gs[3, 0])
    ax_iqae_gold.imshow(images["IQAE epsilon scaling (Gold)"])
    ax_iqae_gold.set_title("IQAE scaling (Gold): queries vs 1/epsilon", fontsize=11)
    ax_iqae_gold.axis("off")

    ax_iqae_spx = fig.add_subplot(gs[3, 1])
    ax_iqae_spx.imshow(images["IQAE epsilon scaling (S&P 500)"])
    ax_iqae_spx.set_title("IQAE scaling (S&P 500): queries vs 1/epsilon", fontsize=11)
    ax_iqae_spx.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved combined figure to: {args.out}")


if __name__ == "__main__":
    main()
