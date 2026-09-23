import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})

C1 = "#1f6feb"   # class 1
C0 = "#d1242f"   # class 0
LINE = "#1a1a1a"
FILL = "#1f6feb"

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
import os; os.makedirs(OUT, exist_ok=True)


def boundary_xy(w, xlim):
    """Return points on w0 + w1*x1 + w2*x2 = 0 over xlim."""
    w0, w1, w2 = w
    xs = np.linspace(*xlim, 200)
    if abs(w2) > 1e-12:
        return xs, -(w0 + w1*xs)/w2
    return np.full(200, -w0/w1), np.linspace(-10, 10, 200)


def draw_plane(ax, w, xlim, ylim, shade=True, label=None, color=LINE, ls="-", lw=2.0):
    xs, ys = boundary_xy(w, xlim)
    m = (ys > ylim[0]-50) & (ys < ylim[1]+50)
    ax.plot(xs[m], ys[m], ls, color=color, lw=lw, zorder=3, label=label)
    if shade:
        X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
        NET = w[0] + w[1]*X + w[2]*Y
        ax.contourf(X, Y, (NET >= 0).astype(float), levels=[0.5, 1.5],
                    colors=[FILL], alpha=0.10, zorder=0)


def normal_arrow(ax, w, base, scale=0.35):
    n = np.array([w[1], w[2]], dtype=float)
    n = n/np.linalg.norm(n)*scale
    ax.add_patch(FancyArrowPatch(base, base+n, arrowstyle="-|>", mutation_scale=13,
                                 color=C1, lw=1.8, zorder=5))


# ---------------------------------------------------------------- Fig 1: AN schematic
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

inputs = [(1.0, 4.8, r"$x_0 = 1$", r"$w_0 = 0.5$"),
          (1.0, 3.0, r"$x_1$",     r"$w_1 = 1.0$"),
          (1.0, 1.2, r"$x_2$",     r"$w_2 = -0.3$")]
sx, sy = 5.0, 3.0
for x, y, lab, wlab in inputs:
    ax.add_patch(Circle((x, y), 0.34, fc="white", ec=LINE, lw=1.4, zorder=3))
    ax.text(x, y, lab, ha="center", va="center", fontsize=10, zorder=4)
    ax.annotate("", xy=(sx-0.72, sy+(y-sy)*0.16), xytext=(x+0.36, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2))
    mx, my = (x+sx)/2 - 0.1, y + (sy-y)*0.42
    ax.text(mx, my+0.22, wlab, ha="center", fontsize=9.5, color="#444")

ax.add_patch(plt.Rectangle((sx-0.72, sy-0.95), 1.45, 1.9, fc="#f2f5f9", ec=LINE, lw=1.4, zorder=3))
ax.text(sx, sy+0.35, r"$\Sigma$", ha="center", va="center", fontsize=17, zorder=4)
ax.text(sx, sy-0.48, r"$f$", ha="center", va="center", fontsize=14, zorder=4)
ax.plot([sx-0.72, sx+0.73], [sy-0.05, sy-0.05], color=LINE, lw=1.0, zorder=4)
ax.text(sx, sy+1.25, "summation unit (SU)", ha="center", fontsize=9, color="#444")
ax.text(sx, sy-1.35, "step activation", ha="center", fontsize=9, color="#444")

ax.annotate("", xy=(8.15, sy), xytext=(sx+0.75, sy),
            arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4))
ax.text(6.0, sy+0.36, r"net $=w_0+w_1x_1+w_2x_2$", ha="left", fontsize=8.4, color="#444")
ax.text(8.35, sy+0.30, r"$a = f(\mathrm{net})$", ha="left", va="center", fontsize=10.5)
ax.text(8.35, sy-0.20, r"$= 1$ if net $\geq 0$", ha="left", va="center", fontsize=9.5)
ax.text(8.35, sy-0.62, r"$= 0$ if net $< 0$", ha="left", va="center", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_an.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------- Fig 2: decision boundary + 5 points (1.2, 1.3)
w = (0.5, 1.0, -0.3)
pts = [("a", 1.0, 0.2), ("b", 0.0, 0.0), ("c", -0.1, -0.5), ("d", 1.7, 0.1), ("e", 1.8, -1.4)]
xlim, ylim = (-1.5, 2.5), (-2.0, 2.5)

fig, ax = plt.subplots(figsize=(5.6, 5.2))
draw_plane(ax, w, xlim, ylim)
for nm, x, y in pts:
    ax.plot(x, y, "o", ms=8, mfc=C1, mec="white", mew=1.4, zorder=6)
    ax.annotate(f"{nm}  ({x}, {y})", (x, y), textcoords="offset points",
                xytext=(9, 7), fontsize=8.6, color="#222")
normal_arrow(ax, w, np.array([0.5, 1.667+0.5*10/3*0]), 0.55)
ax.text(1.45, 0.95, "Class 1 side\n(“front”, net $\\geq$ 0)", color=C1, fontsize=9.5, ha="center")
ax.text(-0.85, 1.6, "Class 0 side\n(net < 0)", color="#777", fontsize=9.5, ha="center")
ax.text(-1.35, -1.75, r"boundary:  $0.5 + x_1 - 0.3x_2 = 0$" "\n" r"i.e. $x_2 = \frac{5+10x_1}{3}$",
        fontsize=9, color="#333")
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.axhline(0, color="#999", lw=0.7); ax.axvline(0, color="#999", lw=0.7)
ax.set_title("Part 1.2 and 1.3: decision boundary; all five points are Class 1", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_boundary.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------- Fig 3: 1.4 non-separable point
fig, ax = plt.subplots(figsize=(5.6, 5.2))
draw_plane(ax, w, xlim, ylim, shade=False, color="#bbb", ls="--", lw=1.3)
ax.plot([0.0, 1.8], [0.0, -1.4], color="#888", lw=1.3, ls=":", zorder=2)
for nm, x, y in pts:
    ax.plot(x, y, "o", ms=8, mfc=C1, mec="white", mew=1.4, zorder=6)
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9, color="#222")
ax.plot(0.9, -0.7, "s", ms=10, mfc=C0, mec="white", mew=1.4, zorder=7)
ax.annotate("new point $(0.9,\\,-0.7)$\nlabeled Class 0", (0.9, -0.7), textcoords="offset points",
            xytext=(12, -26), fontsize=9, color=C0)
ax.text(0.30, -0.58, "segment b to e", fontsize=8.6, color="#666")
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.axhline(0, color="#999", lw=0.7); ax.axvline(0, color="#999", lw=0.7)
ax.set_title("Part 1.4: a Class-0 point that makes the data non-separable", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_nonsep.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------- Fig 4: step vs sigmoid (1.5, 1.6, 1.7)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
z = np.linspace(-6, 6, 600)
ax = axes[0]
ax.step(z, (z >= 0).astype(float), where="post", color=LINE, lw=2, label=r"step $\{0,1\}$")
ax.plot(z, np.where(z >= 0, 1, -1), color="#8957e5", lw=1.6, ls="--", label=r"step $\{-1,1\}$")
ax.plot(z, 1/(1+np.exp(-z)), color=C1, lw=2, label=r"logistic $\sigma$")
ax.axvline(0, color="#999", lw=0.8); ax.axhline(0.5, color="#bbb", lw=0.8, ls=":")
ax.set_xlabel("net"); ax.set_ylabel(r"$f(\mathrm{net})$"); ax.legend(fontsize=8.5, frameon=False)
ax.set_title("All three switch at net $=0$", fontsize=10)

ax = axes[1]
X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
NET = w[0] + w[1]*X + w[2]*Y
im = ax.contourf(X, Y, 1/(1+np.exp(-NET)), levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.9)
xs, ys = boundary_xy(w, xlim)
m = (ys > ylim[0]) & (ys < ylim[1])
ax.plot(xs[m], ys[m], color="k", lw=2, zorder=3)
for nm, x, y in pts:
    ax.plot(x, y, "o", ms=7, mfc="white", mec="k", mew=1.3, zorder=6)
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5)
fig.colorbar(im, ax=ax, label=r"$\sigma(\mathrm{net})$", fraction=0.046)
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
ax.set_title(r"$\sigma = 0.5$ contour is the same line", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_sigmoid.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------- Fig 5: Part 2 combined boundaries
plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})

fig, ax = plt.subplots(figsize=(8.0, 6.6))
xl2, yl2 = (-1.0, 1.5), (-1.5, 1.5)
xs = np.linspace(xl2[0], xl2[1], 400)

def bline(w, style, color, lw, lab):
    ys = -(w[0] + w[1]*xs)/w[2]
    ax.plot(xs, ys, style, color=color, lw=lw, label=lab, zorder=3)

ax.plot(-0.1, -1.0, "s", ms=11, color="red", zorder=6, label="Class 0: (-0.1, -1.0)")
ax.plot( 0.2, -0.9, "s", ms=11, color="red", zorder=6, label="Class 0: (0.2, -0.9)")
ax.plot( 0.6,  0.8, "^", ms=12, color="darkgreen", zorder=6, label="Class 1: (0.6, 0.8)")
ax.plot( 0.0,  0.0, "^", ms=12, color="darkgreen", zorder=6, label="Class 1: (0.0, 0.0)")

bline(( 0.5, 1.0, -0.3), "--",  "black",     2.0, r"Initial: $0.5 + x_1 - 0.3x_2 = 0$")
bline((-0.5, 1.1,  0.7), "-.",  "magenta",   1.8, r"After Update 1 (a): $-0.5 + 1.1x_1 + 0.7x_2 = 0$")
bline(( 0.5, 1.1,  0.7), ":",   "blue",      2.2, r"After Update 2 (d) / Pass 1: $0.5 + 1.1x_1 + 0.7x_2 = 0$")
bline((-0.5, 0.9,  1.6), "-",   "darkturquoise", 1.8, r"After Update 3 (Pass 2 b): $-0.5 + 0.9x_1 + 1.6x_2 = 0$")
bline(( 0.5, 0.9,  1.6), "-",   "darkgreen", 2.6, r"Final (Pass 2 complete): $0.5 + 0.9x_1 + 1.6x_2 = 0$")

ax.axhline(0, color="gray", lw=1.0, ls="--", zorder=1)
ax.axvline(0, color="gray", lw=1.0, ls="--", zorder=1)
ax.grid(True, ls=":", lw=0.7, alpha=0.6)
ax.set_xlim(*xl2); ax.set_ylim(*yl2)
ax.set_xlabel("$x_1$", fontsize=12); ax.set_ylabel("$x_2$", fontsize=12)
ax.set_title("Part 2: Decision Boundaries After Weight Updates", fontsize=13)
ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_part2.png", bbox_inches="tight"); plt.close(fig)

print("done")
