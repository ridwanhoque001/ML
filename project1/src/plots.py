"""
Figures for the Project 1 write-up.

Produces:
    fig1_data_overview.png      the six two-dimensional datasets as given
    fig2_decision_regions.png   learned decision regions, 2D datasets
    fig3_decision_3d.png        learned decision surface, 3D datasets
    fig4_learning_curves.png    training and validation loss per dataset
    fig5_accuracy.png           test accuracy across all nine datasets

Colour assignments follow the roles the data plays: Class 0 and Class 1 are a
categorical pair (identity, no order), while Wide/Narrow/Overlap is an ordinal
ramp in a single hue (ordered by how far apart the classes sit). Both were
checked with the palette validator before use.

CS 4033/5033 Machine Learning Fundamentals, Project 1.
Author: Ridwan Hoque.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data import DATASETS, load_dataset
from experiment import run_single

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# --- palette -------------------------------------------------------------
# Categorical slots 1 and 2: identity, no ordering implied.
C0, C1 = "#2a78d6", "#eb6834"
# Ordinal blue ramp for the separation level, light (easiest) to dark (hardest).
ORDINAL = {"Wide": "#86b6ef", "Narrow": "#2a78d6", "Overlap": "#104281"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d7d6d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

TWO_D = [n for n in DATASETS if "2D" in n]
THREE_D = [n for n in DATASETS if "3D" in n]


def _scatter_classes(ax, X, y, size=18, alpha=0.9):
    """Plot the two classes. A white ring separates overlapping marks."""
    for label, colour in ((0, C0), (1, C1)):
        m = y == label
        ax.scatter(X[m, 0], X[m, 1], s=size, c=colour, alpha=alpha,
                   edgecolors="white", linewidths=0.6, zorder=3)


def fig_data_overview():
    """The raw two-dimensional data, before any model is involved."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for ax, name in zip(axes.ravel(), TWO_D):
        X, y = load_dataset(name)
        _scatter_classes(ax, X, y)
        ax.set_title(name.replace("_", " "), fontsize=9.5)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    handles = [Line2D([], [], marker="o", ls="", markersize=7, color=C0,
                      markeredgecolor="white", label="Class 0"),
               Line2D([], [], marker="o", ls="", markersize=7, color=C1,
                      markeredgecolor="white", label="Class 1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The six two-dimensional datasets as provided", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_data_overview.png"))
    plt.close(fig)


def fig_decision_regions(seed=0):
    """Decision regions learned on each 2D dataset, with the test split shown.

    Only test points are drawn. Showing the training points would make the
    picture look better than the model is: the interesting question is what
    the boundary does to data the network never saw.
    """
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    for ax, name in zip(axes.ravel(), TWO_D):
        res, net, scaler, splits = run_single(name, seed=seed, return_model=True)
        _, _, _, _, Xte, yte = splits

        pad = 0.35
        x_min, x_max = Xte[:, 0].min() - pad, Xte[:, 0].max() + pad
        y_min, y_max = Xte[:, 1].min() - pad, Xte[:, 1].max() + pad
        gx, gy = np.meshgrid(np.linspace(x_min, x_max, 320),
                             np.linspace(y_min, y_max, 320))
        grid = np.c_[gx.ravel(), gy.ravel()]
        proba = net.predict_proba(scaler.transform(grid)).reshape(gx.shape)

        # Filled regions, then the p = 0.5 boundary itself.
        ax.contourf(gx, gy, proba, levels=[0.0, 0.5, 1.0],
                    colors=[C0, C1], alpha=0.16, zorder=0)
        ax.contour(gx, gy, proba, levels=[0.5], colors=[INK],
                   linewidths=1.6, zorder=2)

        _scatter_classes(ax, Xte, yte, size=26)

        # Ring the errors so they are identifiable without relying on colour.
        wrong = net.predict(scaler.transform(Xte)).ravel() != yte
        if wrong.any():
            ax.scatter(Xte[wrong, 0], Xte[wrong, 1], s=110, facecolors="none",
                       edgecolors=INK, linewidths=1.4, zorder=4)

        acc = res["test_accuracy"] * 100
        ax.set_title(f"{name.replace('_', ' ')}\ntest accuracy {acc:.1f}%  "
                     f"({int(wrong.sum())} errors)", fontsize=9.5)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    handles = [Line2D([], [], marker="o", ls="", markersize=7, color=C0,
                      markeredgecolor="white", label="Class 0"),
               Line2D([], [], marker="o", ls="", markersize=7, color=C1,
                      markeredgecolor="white", label="Class 1"),
               Line2D([], [], color=INK, lw=1.6, label="decision boundary"),
               Line2D([], [], marker="o", ls="", markersize=10,
                      markerfacecolor="none", markeredgecolor=INK,
                      label="misclassified")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Learned decision regions, test split only", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_decision_regions.png"))
    plt.close(fig)


def fig_decision_3d(seed=0):
    """Test points and the learned boundary surface for the 3D datasets.

    The boundary is the set where the network outputs 0.5. For each point on a
    grid over (x1, x2) the output is scanned along x3 and the crossing located
    by linear interpolation, which traces out the surface wherever one exists.
    """
    fig = plt.figure(figsize=(12.5, 4.4))
    for i, name in enumerate(THREE_D, start=1):
        res, net, scaler, splits = run_single(name, seed=seed, return_model=True)
        _, _, _, _, Xte, yte = splits
        ax = fig.add_subplot(1, 3, i, projection="3d")

        pad = 0.3
        lo, hi = Xte.min(axis=0) - pad, Xte.max(axis=0) + pad
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 44),
                             np.linspace(lo[1], hi[1], 44))
        zs = np.linspace(lo[2], hi[2], 90)

        # Evaluate the network on the full (x1, x2, x3) lattice at once.
        pts = np.stack([np.repeat(gx.ravel(), zs.size),
                        np.repeat(gy.ravel(), zs.size),
                        np.tile(zs, gx.size)], axis=1)
        p = net.predict_proba(scaler.transform(pts)).reshape(gx.size, zs.size)

        surface = np.full(gx.size, np.nan)
        crosses = (p[:, :-1] - 0.5) * (p[:, 1:] - 0.5) < 0
        for row in range(gx.size):
            k = np.flatnonzero(crosses[row])
            if k.size:
                j = k[0]
                p0, p1 = p[row, j], p[row, j + 1]
                t = (0.5 - p0) / (p1 - p0)
                surface[row] = zs[j] + t * (zs[j + 1] - zs[j])
        surface = surface.reshape(gx.shape)

        ax.plot_surface(gx, gy, surface, color="#9aa3ad", alpha=0.35,
                        linewidth=0, antialiased=True, shade=False)
        for label, colour in ((0, C0), (1, C1)):
            m = yte == label
            ax.scatter(Xte[m, 0], Xte[m, 1], Xte[m, 2], s=26, c=colour,
                       edgecolors="white", linewidths=0.5, depthshade=False)

        ax.set_title(f"{name.replace('_', ' ')}\n"
                     f"test accuracy {res['test_accuracy']*100:.1f}%", fontsize=9.5)
        ax.set_xlabel("$x_1$", labelpad=-4)
        ax.set_ylabel("$x_2$", labelpad=-4)
        ax.set_zlabel("$x_3$", labelpad=-4)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=18, azim=-58)

    handles = [Line2D([], [], marker="o", ls="", markersize=7, color=C0,
                      markeredgecolor="white", label="Class 0"),
               Line2D([], [], marker="o", ls="", markersize=7, color=C1,
                      markeredgecolor="white", label="Class 1"),
               Line2D([], [], marker="s", ls="", markersize=8, color="#9aa3ad",
                      alpha=0.6, label="decision surface ($p = 0.5$)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Learned decision surface, three-dimensional datasets "
                 "(test split)", fontsize=11, y=1.06)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(os.path.join(FIG_DIR, "fig3_decision_3d.png"))
    plt.close(fig)


def fig_learning_curves(seed=0):
    """Training and validation loss against epoch, one panel per dataset."""
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex=False)
    for ax, name in zip(axes.ravel(), DATASETS):
        _, net, _, _ = run_single(name, seed=seed, return_model=True)
        tr = net.history["train_loss"]
        va = net.history["val_loss"]
        epochs = np.arange(1, len(tr) + 1)
        ax.plot(epochs, tr, color=C0, lw=1.6, label="training")
        ax.plot(epochs, va, color=C1, lw=1.6, label="validation")
        ax.axvline(net.best_epoch, color=INK2, lw=1.0, ls="--")
        ax.annotate(f"best epoch {net.best_epoch}", xy=(net.best_epoch, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -11), textcoords="offset points",
                    fontsize=7.5, color=INK2)
        ax.set_title(name.replace("_", " "), fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")

    handles = [Line2D([], [], color=C0, lw=2, label="training loss"),
               Line2D([], [], color=C1, lw=2, label="validation loss"),
               Line2D([], [], color=INK2, lw=1, ls="--",
                      label="best epoch (weights restored here)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curves, one representative run per dataset",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_learning_curves.png"))
    plt.close(fig)


def fig_accuracy(summary_rows):
    """Mean test accuracy per dataset, as a dot plot with one standard deviation.

    A dot plot rather than bars. Every value sits between 92 and 100 percent,
    so bars drawn from zero would be nine near-identical blocks, and bars drawn
    from 80 would exaggerate the differences by truncating the baseline that
    gives a bar its meaning. A dot encodes position only, so the axis is free
    to span just the range the data occupies.
    """
    by_name = {r["dataset"]: r for r in summary_rows}
    families = ["Gaussian_2D", "Gaussian_3D", "Moons_2D"]
    levels = ["Wide", "Narrow", "Overlap"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ypos, ylabels, row = [], [], 0

    for fi, fam in enumerate(families):
        for lvl in levels:                      # Wide at the top of each group
            r = by_name[f"{fam}_{lvl}"]
            mean, std = r["test_accuracy_mean"] * 100, r["test_accuracy_std"] * 100
            ax.errorbar(mean, row, xerr=std, fmt="o", markersize=9,
                        color=ORDINAL[lvl], ecolor=INK2, elinewidth=1.1,
                        capsize=3, markeredgecolor="white", markeredgewidth=1.0,
                        zorder=3)
            ax.text(mean + std + 0.35, row, f"{mean:.1f}%", va="center",
                    fontsize=8.5, color=INK)
            ypos.append(row)
            ylabels.append(f"{fam.replace('_', ' ')}  {lvl}")
            row -= 1
        row -= 0.6                              # gap between families

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(88.8, 102.8)
    ax.set_xlabel("mean test accuracy over 30 repetitions (%), "
                  "bars show one standard deviation")
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # No legend box: each row is already named on the y axis, so identity is
    # directly labelled rather than carried by colour alone. The ramp is there
    # to make the ordering visible at a glance, not to encode the label.
    ax.set_title("Test accuracy falls as the classes move closer together",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_accuracy.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    import csv
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "summary.csv")) as fh:
        summary = [{k: (v if k == "dataset" else float(v))
                    for k, v in row.items()} for row in csv.DictReader(fh)]

    print("fig1 data overview");    fig_data_overview()
    print("fig2 decision regions"); fig_decision_regions()
    print("fig3 decision surface"); fig_decision_3d()
    print("fig4 learning curves");  fig_learning_curves()
    print("fig5 accuracy");         fig_accuracy(summary)
    print("done")


if __name__ == "__main__":
    main()


def fig_capacity():
    """Test and training accuracy against hidden-layer size, both overlap sets.

    Two panels rather than one, because the two datasets have different
    ceilings and plotting them on a shared axis would invite a comparison
    between their absolute levels that is not the point. The point is the
    slope, which is flat in both.
    """
    import csv as _csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "capacity_sweep.csv")
    rows = list(_csv.DictReader(open(path)))

    datasets = ["Moons_2D_Overlap", "Gaussian_2D_Overlap"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=False)

    for ax, name in zip(axes, datasets):
        sub = [r for r in rows if r["dataset"] == name and ", " not in r["hidden"]]
        sub.sort(key=lambda r: int(r["n_params"]))
        widths = [int(r["hidden"].strip("(),")) for r in sub]
        test = np.array([float(r["test_accuracy_mean"]) for r in sub]) * 100
        std = np.array([float(r["test_accuracy_std"]) for r in sub]) * 100
        train = np.array([float(r["train_accuracy_mean"]) for r in sub]) * 100
        ceiling = float(sub[0]["ceiling"]) * 100

        ax.axhline(ceiling, color=INK2, lw=1.2, ls="--", zorder=1)
        ax.annotate(f"estimated ceiling {ceiling:.1f}%", xy=(0.02, ceiling),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="left", fontsize=8, color=INK2)

        ax.fill_between(widths, test - std, test + std, color=C0, alpha=0.13,
                        zorder=2)
        ax.plot(widths, test, "-o", color=C0, lw=2, markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4,
                label="test accuracy")
        ax.plot(widths, train, "-o", color=C1, lw=2, markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                label="training accuracy")

        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel("hidden units (log scale)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Extra capacity buys nothing: both curves are flat, and "
                 "training accuracy is flat too", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_capacity.png"))
    plt.close(fig)
