"""Schematic of the discretization and the consensus split, per variable.

Three panels, drawn from the same index arithmetic as ``archive/cpp_minimal``:

  (a) the staggered grid: u on the N² nodes, (qx, qy, r, δ, θ) on the (N−1)²
      cells, and the forward-difference stencils Kx, Ky anchored at each cell's
      top-left node (``problem.hpp: build_stencils``);
  (b) the k×k tile partition of the cells, and the tile each node joins
      (``partition.hpp``);
  (c) the variables the consensus form copies, one copy per tile whose rows
      read them (``problem.hpp: split_into_consensus_form``). r, δ, θ are
      never shared; α is copied into every tile.

The copy count is checked against what ``tv_learn`` prints (``copies=``) before
anything is drawn, so the picture cannot drift from the solver.

    python plot_discretization.py                       # N=9, 2×2 tiles, PNG
    python plot_discretization.py --size 13 --nsub 3 --format both --style paper
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from plot_style import PALETTE, apply_style, figure_size, savefig

TILE_COLORS = [PALETTE[c] for c in ("blue", "teal", "orange", "purple")]
# `copies=` printed by archive/cpp_minimal/tv_learn for these (N, k).
KNOWN_COPIES = {(9, 2): 67, (10, 3): 149, (13, 3): 197, (17, 4): 391}


# --- the index arithmetic of archive/cpp_minimal -------------------------------------
def partition(N, k):
    """cell_tile (nc×nc) and node_tile (N×N), as in partition.hpp."""
    nc = N - 1
    step = nc / k
    cut = [int(j * step) for j in range(k)] + [nc]
    cell = np.zeros((nc, nc), dtype=int)
    for a in range(k):
        for c in range(k):
            cell[cut[a]:cut[a + 1], cut[c]:cut[c + 1]] = a * k + c
    node = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            node[i, j] = cell[min(i, nc - 1), min(j, nc - 1)]
    return cell, node, cut


def stencils(N):
    """(cell, node) pairs read by Kx and Ky: (Kx u)_{a,b} = u[a,b+1] − u[a,b],
    (Ky u)_{a,b} = u[a+1,b] − u[a,b]."""
    nc = N - 1
    Kx, Ky = [], []
    for a in range(nc):
        for b in range(nc):
            Kx += [((a, b), (a, b)), ((a, b), (a, b + 1))]
            Ky += [((a, b), (a, b)), ((a, b), (a + 1, b))]
    return Kx, Ky


def consensus_tiles(N, k):
    """Set of tiles whose rows read each u node / qx cell / qy cell."""
    cell, node, _ = partition(N, k)
    Kx, Ky = stencils(N)
    tu = {(i, j): {node[i, j]} for i in range(N) for j in range(N)}   # h1 row
    tqx = {(a, b): {cell[a, b]} for a in range(N - 1) for b in range(N - 1)}   # h3x
    tqy = {c: set(s) for c, s in tqx.items()}                                  # h3y
    for c, n in Kx:
        tu[n].add(cell[c])          # h2x reads u
        tqx[c].add(node[n])         # h1 (divergence Kxᵀ) reads qx
    for c, n in Ky:
        tu[n].add(cell[c])
        tqy[c].add(node[n])
    return tu, tqx, tqy


def count_copies(N, k):
    tu, tqx, tqy = consensus_tiles(N, k)
    shared = sum(len(s) for d in (tu, tqx, tqy) for s in d.values() if len(s) > 1)
    return shared + k * k       # + one α copy per tile


# --- drawing -----------------------------------------------------------------
# Node (i, j) sits at (x, y) = (j, −i): row index grows downward, as in the image.
def xy_node(i, j):
    return j, -i


def xy_cell(a, b):
    return b + 0.5, -(a + 0.5)


def grid_lines(ax, N, color, lw=0.5):
    for t in range(N):
        ax.plot([0, N - 1], [-t, -t], color=color, lw=lw, zorder=1)
        ax.plot([t, t], [0, -(N - 1)], color=color, lw=lw, zorder=1)


def tile_borders(ax, cut, N, color):
    for c in cut[1:-1]:
        ax.plot([c, c], [0, -(N - 1)], color=color, lw=1.6, zorder=2)
        ax.plot([0, N - 1], [-c, -c], color=color, lw=1.6, zorder=2)


def setup(ax, N, pad=0.35):
    ax.set_xlim(-pad, N - 1 + pad)
    ax.set_ylim(-(N - 1) - pad, pad)
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, p, q, color, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, shrinkA=5, shrinkB=5,
                                 zorder=5))


def panel_stencil(ax, ms):
    """(a) a 4×4-node patch with the stencil of one highlighted cell."""
    N = 4
    grid_lines(ax, N, PALETTE["fog"], lw=0.8)
    a, b = 1, 1
    ax.add_patch(Rectangle((b, -(a + 1)), 1, 1, fc=PALETTE["fog"], alpha=0.35,
                           ec="none", zorder=0))
    for i in range(N):
        for j in range(N):
            ax.plot(*xy_node(i, j), "o", ms=ms, mfc=PALETTE["ink"],
                    mec=PALETTE["ink"], zorder=4)
    for aa in range(N - 1):
        for bb in range(N - 1):
            ax.plot(*xy_cell(aa, bb), "s", ms=ms * 0.9, mfc="white",
                    mec=PALETTE["charcoal"], mew=0.9, zorder=4)
    # Kx along the top edge, Ky along the left edge, both starting at the anchor.
    arrow(ax, xy_node(a, b), xy_node(a, b + 1), PALETTE["vermillion"])
    arrow(ax, xy_node(a, b), xy_node(a + 1, b), PALETTE["blue"])
    ax.plot(*xy_node(a, b), "o", ms=ms * 1.9, mfc="none",
            mec=PALETTE["ink"], mew=1.0, zorder=4)
    x, _ = xy_cell(a, b)
    ax.text(x, -(a + 1) - 0.08, r"cell $(a,b)$", ha="center", va="top",
            fontsize="x-small")
    ax.text(b + 0.5, -a + 0.12, r"$K_x$", ha="center", va="bottom",
            color=PALETTE["vermillion"], fontsize="small")
    ax.text(b - 0.1, -(a + 0.5), r"$K_y$", ha="right", va="center",
            color=PALETTE["blue"], fontsize="small")
    ax.text(*np.add(xy_node(a, b), (-0.14, 0.14)), r"$u_{a,b}$",
            ha="right", va="bottom", fontsize="x-small")
    ax.text(*np.add(xy_node(a, b + 1), (0.12, 0.12)), r"$u_{a,b+1}$",
            ha="left", va="bottom", fontsize="x-small")
    ax.text(*np.add(xy_node(a + 1, b), (-0.1, -0.14)), r"$u_{a+1,b}$",
            ha="right", va="top", fontsize="x-small")
    setup(ax, N)


def panel_partition(ax, N, k, ms):
    """(b) tiles of the cells, and the tile each node joins."""
    cell, node, cut = partition(N, k)
    nc = N - 1
    for a in range(nc):
        for b in range(nc):
            ax.add_patch(Rectangle((b, -(a + 1)), 1, 1, ec="none", alpha=0.22,
                                   fc=TILE_COLORS[cell[a, b] % 4], zorder=0))
            ax.plot(*xy_cell(a, b), "s", ms=ms * 0.75, mfc="white",
                    mec=TILE_COLORS[cell[a, b] % 4], mew=0.9, zorder=3)
    grid_lines(ax, N, "white", lw=0.6)
    tile_borders(ax, cut, N, PALETTE["ink"])
    for i in range(N):
        for j in range(N):
            ax.plot(*xy_node(i, j), "o", ms=ms, mfc=TILE_COLORS[node[i, j] % 4],
                    mec="white", mew=0.5, zorder=4)
    setup(ax, N)


def panel_consensus(ax, N, k, ms):
    """(c) consensus variables: copied u nodes and qx/qy cells."""
    cell, node, cut = partition(N, k)
    tu, tqx, tqy = consensus_tiles(N, k)
    nc = N - 1
    grid_lines(ax, N, PALETTE["fog"], lw=0.5)
    tile_borders(ax, cut, N, PALETTE["slate"])
    for (a, b), s in tqx.items():
        if len(s) > 1:
            x, y = xy_cell(a, b)
            ax.add_patch(FancyArrowPatch((x - 0.32, y + 0.12), (x + 0.32, y + 0.12),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=PALETTE["vermillion"], lw=1.1,
                                         zorder=5))
    for (a, b), s in tqy.items():
        if len(s) > 1:
            x, y = xy_cell(a, b)
            ax.add_patch(FancyArrowPatch((x - 0.12, y + 0.32), (x - 0.12, y - 0.32),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=PALETTE["blue"], lw=1.1, zorder=5))
    for (i, j), s in tu.items():
        x, y = xy_node(i, j)
        if len(s) > 1:
            ax.plot(x, y, "o", ms=ms * 1.9, mfc="white", mec=PALETTE["ink"],
                    mew=1.0, zorder=6)
            ax.text(x, y, str(len(s)), ha="center", va="center",
                    fontsize=5.5, zorder=7)
        else:
            ax.plot(x, y, "o", ms=ms * 0.6, mfc=PALETTE["fog"], mec="none",
                    zorder=4)
    setup(ax, N)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", type=int, default=9, help="image side N (nodes)")
    p.add_argument("--nsub", type=int, default=2, help="k, for k×k tiles")
    p.add_argument("--out", default="../results/figures/discretization")
    p.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    p.add_argument("--style", choices=("screen", "paper"), default="screen")
    args = p.parse_args()
    N, k = args.size, args.nsub

    copies = count_copies(N, k)
    if (N, k) in KNOWN_COPIES:
        assert copies == KNOWN_COPIES[N, k], (copies, KNOWN_COPIES[N, k])
        print(f"    copies={copies} (matches tv_learn)")
    else:
        print(f"    copies={copies} (no tv_learn reference for N={N}, k={k})")

    apply_style(args.style)
    ms = 5.0 if args.style == "paper" else 6.0
    fig, axes = plt.subplots(1, 3, figsize=figure_size("twocol", aspect=0.40),
                             gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.08))
    panel_stencil(axes[0], ms)
    panel_partition(axes[1], N, k, ms * 0.8)
    panel_consensus(axes[2], N, k, ms * 0.8)
    axes[0].set_title("(a) grid and stencil", fontsize="medium")
    axes[1].set_title(f"(b) {k}×{k} tiles, N = {N}", fontsize="medium")
    axes[2].set_title("(c) copied variables", fontsize="medium")

    handles = [
        Line2D([], [], ls="", marker="o", mfc=PALETTE["ink"], mec=PALETTE["ink"],
               ms=ms, label=r"node: $u_{i,j}$  ($N^2$)"),
        Line2D([], [], ls="", marker="s", mfc="white", mec=PALETTE["charcoal"],
               ms=ms, label=r"cell: $q^x,q^y,r,\delta,\theta$  ($(N-1)^2$)"),
        Line2D([], [], ls="", marker="o", mfc="white", mec=PALETTE["ink"],
               ms=ms * 1.5, label="copied $u$ (number = copies)"),
        Line2D([], [], color=PALETTE["vermillion"], marker=">", ms=4,
               label="copied $q^x$"),
        Line2D([], [], color=PALETTE["blue"], marker="v", ms=4,
               label="copied $q^y$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize="x-small",
               handletextpad=0.4, columnspacing=1.2)
    fig.text(0.5, 0.0,
             r"$r,\delta,\theta$ are read only by their own cell's rows and are "
             r"never copied;  $\alpha$ is copied once per tile "
             f"({k * k} copies).  Total copies = {copies}.",
             ha="center", va="top", fontsize="x-small", color=PALETTE["charcoal"])
    savefig(fig, args.out, args.format)


if __name__ == "__main__":
    main()
