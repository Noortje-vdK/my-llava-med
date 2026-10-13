
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.interpolate import griddata


PARAMETER_COLUMNS = ["alpha", "beta", "noise_step"]
TARGET_BETA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RadGraph ER contours for beta = 0.05."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="radgraph_grid_scores.csv",
        help="Scored per-row CSV or parameter summary CSV.",
    )

    parser.add_argument(
        "--output",
        default="PROJECT_RESULTS/radgraph_er_contour_beta_005.png",
        help="Path for the output contour plot.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )

    return parser.parse_args()


def load_grid_scores(path):
    df = pd.read_csv(path)

    score_column = (
        "radgraph_er"
        if "radgraph_er" in df.columns
        else "radgraph_er_mean"
    )

    missing_columns = set(
        PARAMETER_COLUMNS + [score_column]
    ).difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing_columns)}. "
            "Use a scored RadGraph grid CSV or summary CSV."
        )

    for column in PARAMETER_COLUMNS + [score_column]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=PARAMETER_COLUMNS + [score_column]
    )

    if df.empty:
        raise ValueError(
            f"No valid parameter rows with RadGraph ER scores found in {path}"
        )

    return (
        df.groupby(
            PARAMETER_COLUMNS,
            as_index=False,
            dropna=False,
        )[score_column]
        .mean()
        .rename(columns={score_column: "radgraph_er"})
    )


def plot_grid(scores, output_path, show=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only beta = 0.05
    plot_scores = scores[
        np.isclose(scores["beta"], TARGET_BETA, atol=1e-10)
    ].copy()

    if plot_scores.empty:
        available_betas = sorted(scores["beta"].unique())
        raise ValueError(
            f"No data found for beta = {TARGET_BETA}. "
            f"Available beta values: {available_betas}"
        )

    x = plot_scores["alpha"].to_numpy()
    y = plot_scores["noise_step"].to_numpy()
    score = plot_scores["radgraph_er"].to_numpy()

    # Determine the plotting limits and add a little white space.
    x_range = np.ptp(x)
    x_margin = 0.05 * x_range if x_range > 0 else 0.05

    x_min = x.min() - x_margin
    x_max = x.max() + x_margin

    # Requested y-axis range
    y_min = 0
    y_max = 1000

    # Interpolation grid
    Xi = np.linspace(x.min(), x.max(), 250)
    Yi = np.linspace(y.min(), y.max(), 250)
    Xi, Yi = np.meshgrid(Xi, Yi)

    # Use linear interpolation for a robust contour plot.
    # Cubic interpolation can fail with sparse parameter grids.
    Zi = griddata(
        (x, y),
        score,
        (Xi, Yi),
        method="cubic",
    )

    # Color scale
    vmin = score.min()
    vmax = score.max()

    if np.isclose(vmin, vmax):
        # Avoid identical contour levels when all scores are equal.
        vmin -= 1e-6
        vmax += 1e-6

    levels = np.linspace(vmin, vmax, 15)

    # Figure and axes
    fig, ax = plt.subplots(figsize=(11, 8))

    fig.subplots_adjust(
        left=0.15,
        right=0.82,
        bottom=0.16,
        top=0.84,
    )

    fig.suptitle(
        "Mean RadGraph Entity Relation (ER) Score Across\n"
        "Visual Contrastive Decoding (VCD) parameters \n "
        r"$\beta = 0.05$",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    # Filled contours
    contour = ax.contourf(
        Xi,
        Yi,
        Zi,
        levels=levels,
        cmap="viridis",
        extend="both",
    )

    # Contour lines
    ax.contour(
        Xi,
        Yi,
        Zi,
        levels=levels,
        colors="black",
        linewidths=0.8,
        alpha=0.7,
    )

    # # Optional: show all sampled parameter combinations
    # ax.scatter(
    #     x,
    #     y,
    #     color="white",
    #     edgecolor="black",
    #     s=45,
    #     zorder=5,
    # )

    # Optimal point for beta = 0.05
    best_idx = plot_scores["radgraph_er"].idxmax()
    best = plot_scores.loc[best_idx]

    ax.scatter(
        best["alpha"],
        best["noise_step"],
        marker="*",
        s=300,
        color="red",
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
    )

    ax.annotate(
        f'Optimal point:{best["radgraph_er"]:.4f}',
        xy=(best["alpha"], best["noise_step"]),
        xytext=(-20, 14),
        textcoords="offset points",
        fontsize=15,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    # Axis labels
    ax.set_xlabel(
        r"$\alpha$ [unitless]",
        fontsize=19,
        fontweight="bold",
        labelpad=12,
    )

    ax.set_ylabel(
        "Noise step [unitless]",
        fontsize=19,
        fontweight="bold",
        labelpad=12,
    )

    # Add white space around the grid
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # At most 8 major y-axis tick labels
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))

    # Larger, inward-facing ticks
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=8,
        width=1.6,
        labelsize=15,
        pad=8,
        top=True,
        right=True,
    )

    # Bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Thicker frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    # Legend explaining the star
    # optimal_legend = Line2D(
    #     [0],
    #     [0],
    #     marker="*",
    #     color="red",
    #     markeredgecolor="white",
    #     markersize=18,
    #     linestyle="None",
    #     label="Optimal point",
    # )

    # ax.legend(
    #     handles=[optimal_legend],
    #     loc="upper right",
    #     fontsize=15,
    #     frameon=True,
    #     framealpha=0.95,
    #     prop={"weight": "bold", "size": 15},
    #     borderpad=0.8,
    # )

    # Colorbar
    cbar_ax = fig.add_axes([0.86, 0.16, 0.035, 0.68])
    cbar = fig.colorbar(contour, cax=cbar_ax)

    cbar.ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

    cbar.set_label(
        "Mean RadGraph ER Score",
        fontsize=17,
        fontweight="bold",
        labelpad=15,
    )

    cbar.ax.tick_params(
        labelsize=14,
        width=1.4,
        length=5,
    )

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    cbar.outline.set_linewidth(1.5)

    # Save and optionally display
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    if show:
        plt.show()

    plt.close(fig)

    print(f"Saved contour plot to {output_path}")
    print(
        f"Highest mean RadGraph ER: {best['radgraph_er']:.4f} "
        f"at alpha={best['alpha']}, "
        f"beta={best['beta']}, "
        f"noise_step={best['noise_step']}"
    )


def main():
    args = parse_args()
    scores = load_grid_scores(args.input)

    plot_grid(
        scores,
        args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
