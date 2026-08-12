"""Shared figure style for this package: one API, two targets.

``apply_style("screen")`` reproduces ``plot_slurm.py``'s existing rcParams
verbatim (font 10, white face, ``axes.edgecolor "0.3"``) and writes 200-dpi PNG
— what you want while looking at a run. ``apply_style("paper")`` switches to
two-column manuscript sizing (font 8, no top/right spines, frameless legends,
Type-42 embedded fonts) and writes vector PDF.

WHY THIS FILE IS A COPY AND NOT AN IMPORT. The widths, the Okabe-Ito palette and
the ``figure_size``/``grid_figure_size`` helpers follow the conventions of
``sparse-ho-fb/expes_fb/shared/plotting.py``, deliberately reproduced here
rather than imported. ``../python/README.md`` pins this package to numpy +
matplotlib and nothing else, and the whole checkout ships as an archival Zenodo
record — a cross-repo import would make a figure un-regenerable from the archive
alone. Keep the numbers in sync by hand if the sibling repo's style moves.

``plot_slurm.py`` is deliberately NOT ported onto this module: its figures are
recorded output, and restyling them would silently change every previously
generated figure in ``results/``.

    from plot_style import apply_style, figure_size, savefig
    apply_style("paper")
    fig, ax = plt.subplots(figsize=figure_size("twocol", aspect=0.42))
    savefig(fig, "/tmp/plots/kodak_gain", "both")
"""

import os

# Two-column manuscript widths in inches (sparse-ho-fb convention).
ONE_COLUMN_WIDTH = 3.35
TWO_COLUMN_WIDTH = 6.85

# Okabe-Ito, colourblind-safe. The four greys lead the cycle so a figure reads
# in print before any colour is spent; the accents are for series that must be
# told apart.
PALETTE = {
    "ink": "#1A1A1A",
    "charcoal": "#4D4D4D",
    "slate": "#737373",
    "fog": "#BDBDBD",
    "blue": "#0072B2",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Stable per-solver colours/markers, so the same solver reads the same way in
# every figure of a sweep. Keys match the `solver`/`label` strings the driver
# and the SLURM scripts use.
SOLVER_STYLES = {
    "ma57": dict(color=PALETTE["ink"], marker="o", linestyle="-",
                 label="monolithic MA57"),
    "ma97": dict(color=PALETTE["charcoal"], marker="o", linestyle="-",
                 label="monolithic MA97"),
    "dd": dict(color=PALETTE["blue"], marker="s", linestyle="--",
               label="DD (per pair)"),
    "dd_cg": dict(color=PALETTE["teal"], marker="^", linestyle="-.",
                  label="DD + CG interface"),
    "dd_minres": dict(color=PALETTE["orange"], marker="v", linestyle=":",
                      label="DD + MINRES interface"),
    "dd_tiles": dict(color=PALETTE["purple"], marker="D", linestyle="--",
                     label="DD (per pair $\\times$ tiles)"),
}

# Roles in the train/held-out split. Filled = optimized over, hollow = unseen.
ROLE_STYLES = {
    "train": dict(color=PALETTE["blue"], filled=True, label="training"),
    "val": dict(color=PALETTE["vermillion"], filled=False, label="held-out"),
}

_SCREEN = {
    # verbatim from plot_slurm.py's _style, so the two agree on screen
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": "0.3", "axes.linewidth": 0.8,
    "xtick.color": "0.3", "ytick.color": "0.3",
    "axes.titleweight": "normal", "savefig.facecolor": "white",
    "figure.dpi": 110, "savefig.dpi": 200,
}

_PAPER = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "figure.titlesize": 9, "figure.titleweight": "normal",
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.titlepad": 4.0, "axes.labelpad": 2.5,
    "lines.linewidth": 1.5, "lines.markersize": 4.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2,
    # Type 42 keeps text selectable/editable in the PDF instead of outlining it.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
}

_MODE = "screen"


def apply_style(mode="screen"):
    """Set rcParams for "screen" (PNG inspection) or "paper" (PDF)."""
    global _MODE
    import matplotlib.pyplot as plt
    if mode not in ("screen", "paper"):
        raise ValueError(f"mode must be screen|paper, got {mode!r}")
    _MODE = mode
    plt.rcParams.update(_SCREEN if mode == "screen" else _PAPER)


def figure_size(width="onecol", *, aspect=0.7, scale=1.0):
    """(w, h) in inches. `width` is "onecol"/"twocol" or an explicit number."""
    if width == "onecol":
        w = ONE_COLUMN_WIDTH
    elif width == "twocol":
        w = TWO_COLUMN_WIDTH
    else:
        w = float(width)
    w *= scale
    return (w, w * aspect)


def grid_figure_size(nrows, ncols, *, width="onecol", panel_aspect=0.78,
                     scale=1.0, extra_height=0.0):
    """Size a regular grid: each panel keeps `panel_aspect` at the shared width."""
    w, _ = figure_size(width, aspect=1.0, scale=scale)
    return (w, w / max(ncols, 1) * panel_aspect * nrows + extra_height)


def savefig(fig, out_base, fmt="png"):
    """Write `out_base` as .png, .pdf, or both. Returns the paths written.

    `out_base` carries no extension — the format decides. Directories are
    created as needed, and every write is announced, matching the `saved → …`
    idiom of the sibling experiment repos.
    """
    if fmt not in ("png", "pdf", "both"):
        raise ValueError(f"format must be png|pdf|both, got {fmt!r}")
    base, ext = os.path.splitext(out_base)
    if ext in (".png", ".pdf"):
        out_base = base
    d = os.path.dirname(out_base)
    if d:
        os.makedirs(d, exist_ok=True)
    exts = ("png", "pdf") if fmt == "both" else (fmt,)
    written = []
    for e in exts:
        p = f"{out_base}.{e}"
        fig.savefig(p, bbox_inches="tight")
        print(f"    saved → {p}")
        written.append(p)
    return written


def close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def solver_style(name):
    """Style for a solver label, falling back to charcoal/solid/o for unknowns."""
    return SOLVER_STYLES.get(
        name, dict(color=PALETTE["charcoal"], marker="o", linestyle="-",
                   label=name))
