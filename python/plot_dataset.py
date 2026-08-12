"""Figures and tables for DATASET runs (``dd_solve_dataset``).

A dataset run learns ONE scalar weight α across S training pairs, so its
interesting quantities are ACROSS pairs, not within one. ``plot_slurm.py``
answers "what did this image look like" — seven panels per solution — and is
still the right tool for that. It is the wrong unit of analysis for a run whose
question is *does one α serve every image, and does it generalize*: it produces
7×S files of which S are byte-identical continuation plots (the level history
and μ-trace describe the one shared solve), and each panel is scaled per-file,
so a side-by-side montage invites exactly the wrong comparison.

This script works from the run's ``--save-report`` JSON and produces a small
number of run-level figures instead.

  --run <report.json>       one run
      <tag>_gain            per-pair PSNR, noisy → reconstructed, train vs
                            held-out. The headline: does the shared α help
                            every image?
      <tag>_contact         contact sheet (noisy | reconstruction | error) on
                            SHARED colour scales, so the panels are comparable
      <tag>_continuation    the continuation path, ONCE (it is shared)

  --sweep <results-dir>     many runs: reports/*.json joined to timings.csv
      scaling               wall/CPU, IPOPT iterations and peak RSS vs S
      dd_vs_mono            ms per Newton step, and the α* agreement check
      tab_dataset.tex       booktabs summary

Decoding is delegated to ``plot_slurm.parse_solution`` and the PSNR/index-set
helpers are imported from there too, so every number here matches the solver's
own reports.

    uv run python plot_dataset.py --run  ../results/slurm_1/reports/report_x.json
    uv run python plot_dataset.py --sweep ../results/slurm_1 --format both
    uv run python plot_dataset.py --run  r.json --max-pairs 4 --format pdf
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

import plot_style as ps
from plot_slurm import fig_continuation, parse_solution, psnr


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_report(path):
    with open(path) as fh:
        rep = json.load(fh)
    rep["_path"] = path
    rep["_dir"] = os.path.dirname(os.path.abspath(path))
    return rep


def solution_paths(rep, sols_dir=None):
    """Locate the per-pair solution files belonging to a report.

    The driver writes ``<prefix>_s%03d.txt`` for training pairs and
    ``<prefix>_v%03d.txt`` for held-out ones. The report does not record the
    prefix (it need not know where the caller put them), so match on the tag:
    ``sol_<tag>_s000.txt`` is what the SLURM script produces.
    """
    if sols_dir is None:
        sols_dir = os.path.join(os.path.dirname(rep["_dir"]), "sols")
    out = {"train": [], "val": []}
    if not os.path.isdir(sols_dir):
        return out
    for role, suf in (("train", "s"), ("val", "v")):
        pat = os.path.join(sols_dir, f"*{rep['tag']}_{suf}[0-9][0-9][0-9].txt")
        out[role] = sorted(glob.glob(pat))
    return out


def pairs_of(rep, role):
    return [p for p in rep["pairs"] if p["role"] == role]


def _short(path, width=22):
    name = os.path.basename(path)
    return name if len(name) <= width else name[: width - 1] + "…"


# --------------------------------------------------------------------------
# figure 1 — the headline: per-pair gain, train vs held-out
# --------------------------------------------------------------------------
def fig_gain(rep, out_base, fmt):
    """Dumbbell per pair: noisy PSNR → PSNR at the learned α, sorted by gain.

    One shared α is applied to every image, so the honest question is not the
    mean but the SPREAD: which images it helps a lot, which barely, and whether
    the held-out ones behave like the training ones. A dumbbell shows level and
    gain at once; a bar of "gain" alone would hide that the images start at
    different PSNRs.

    Training pairs are read from the MPCC's own u_s; held-out pairs have no MPCC
    iterate (psnr_mpcc is null) and are read from the lower level solved at
    Q(α*) — which is what the learned weight MEANS for an unseen image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = []
    for role in ("train", "val"):
        for p in pairs_of(rep, role):
            # psnr_mpcc is null for held-out; psnr_rof is defined for both and
            # is the like-for-like column across the split.
            after = p["psnr_rof"]
            rows.append((role, p, p["psnr_noisy"], after, after - p["psnr_noisy"]))
    if not rows:
        raise SystemExit("report has no pairs")
    rows.sort(key=lambda r: r[4])

    n = len(rows)
    fig, ax = plt.subplots(
        figsize=ps.figure_size("twocol", aspect=max(0.30, 0.055 * n + 0.16)))

    for i, (role, p, before, after, gain) in enumerate(rows):
        st = ps.ROLE_STYLES[role]
        ax.plot([before, after], [i, i], "-", color=st["color"], lw=1.6,
                alpha=0.55, zorder=1, solid_capstyle="round")
        ax.plot([before], [i], "o", ms=4.5, color=ps.PALETTE["fog"],
                markeredgecolor=ps.PALETTE["slate"], markeredgewidth=0.8,
                zorder=2)
        ax.plot([after], [i], "o", ms=6.0, zorder=3,
                color=st["color"] if st["filled"] else "white",
                markeredgecolor=st["color"], markeredgewidth=1.4)
        ax.annotate(f"+{gain:.2f}", xy=(after, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=st["color"])

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{_short(p['path'])}" for _, p, _, _, _ in rows])
    for lbl, (role, *_rest) in zip(ax.get_yticklabels(), rows):
        if role == "val":
            lbl.set_style("italic")
            lbl.set_color(ps.ROLE_STYLES["val"]["color"])
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("PSNR (dB)")
    ax.grid(True, axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)

    # the two means, as vertical rules — the numbers the run reports
    for role in ("train", "val"):
        sel = [r for r in rows if r[0] == role]
        if not sel:
            continue
        m = float(np.mean([r[3] for r in sel]))
        st = ps.ROLE_STYLES[role]
        ax.axvline(m, color=st["color"], lw=1.0,
                   ls="-" if st["filled"] else "--", alpha=0.6, zorder=0)
        ax.annotate(f"mean {m:.2f}", xy=(m, n - 0.6), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.5, color=st["color"],
                    rotation=90, va="top")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.5, color=ps.PALETTE["fog"],
               markeredgecolor=ps.PALETTE["slate"], label="noisy $f$"),
        Line2D([], [], marker="o", ls="none", ms=6, color=ps.ROLE_STYLES["train"]["color"],
               label="training (MPCC)"),
        Line2D([], [], marker="o", ls="none", ms=6, color="white",
               markeredgecolor=ps.ROLE_STYLES["val"]["color"], markeredgewidth=1.4,
               label="held-out (ROF at $Q(\\alpha^*)$)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, ncol=1)

    ntr, nva = len(pairs_of(rep, "train")), len(pairs_of(rep, "val"))
    gap = ""
    if ntr and nva:
        mt = np.mean([p["psnr_rof"] for p in pairs_of(rep, "train")])
        mv = np.mean([p["psnr_rof"] for p in pairs_of(rep, "val")])
        gap = f",  gap {mt - mv:+.2f} dB"
    ax.set_title(
        f"one $\\alpha$ across {ntr} training pairs"
        + (f" + {nva} held out" if nva else "")
        + f"\n$Q(\\alpha^*)$={rep['Q_alpha']:.4f},  N={rep['N']},  "
          f"$\\sigma$={rep['sigma']:g},  {rep['solver']}{gap}",
        fontsize=8)
    fig.tight_layout()
    ps.savefig(fig, out_base, fmt)
    ps.close(fig)


# --------------------------------------------------------------------------
# figure 2 — contact sheet on SHARED scales
# --------------------------------------------------------------------------
def fig_contact(rep, sols, out_base, fmt, max_pairs=8, diff="error"):
    """Pairs across COLUMNS, the three views down rows, SHARED colour scales.

    Two decisions here, both learned from the first attempt.

    SHARED SCALES are the whole point of the figure. ``plot_slurm.py`` computes
    its gray range, its error limit and its log floor per FILE, which is right
    for a single-image run and wrong the moment panels sit side by side —
    identical colours would mean different values in different panels. Both
    scales are computed over every displayed pair first.

    LANDSCAPE, not portrait. One row per pair makes S square panels tall by
    three wide: at S=12 that is a 6.85x27.9 inch sliver, unusable in a paper and
    mostly whitespace. Pairs belong on the x axis — that is what a contact sheet
    is — so the figure grows sideways and stays a sensible shape.
    """
    import matplotlib.pyplot as plt

    tr = [("train", p) for p in sols["train"]]
    va = [("val", p) for p in sols["val"]]
    if not tr and not va:
        raise SystemExit("no solution files found for this report — pass --sols")
    # Take from BOTH roles proportionally rather than filling up with training
    # pairs and truncating the rest: at S=8 held-out=6 a plain head() would drop
    # every held-out image, which is the half of the sheet worth looking at.
    dropped = []
    if len(tr) + len(va) > max_pairs:
        if tr and va:
            n_tr = max(1, round(max_pairs * len(tr) / (len(tr) + len(va))))
            n_tr = min(n_tr, max_pairs - 1, len(tr))
            n_va = min(max_pairs - n_tr, len(va))
        else:
            n_tr, n_va = min(max_pairs, len(tr)), min(max_pairs, len(va))
        dropped = [p for _, p in tr[n_tr:] + va[n_va:]]
        tr, va = tr[:n_tr], va[:n_va]
    items = tr + va

    parsed = [(role, parse_solution(p)) for role, p in items]
    N = parsed[0][1]["N"]

    # shared scales, computed over everything displayed
    lo = min(min(s["f"].min(), s["u"].min()) for _, s in parsed)
    hi = max(max(s["f"].max(), s["u"].max()) for _, s in parsed)
    errs = [(s["u"] - (s["f"] if diff == "residual" else s["u_clean"]))
            for _, s in parsed]
    elim = max(float(np.abs(e).max()) for e in errs) or 1.0

    nc = len(parsed)
    panel = ps.TWO_COLUMN_WIDTH / max(nc, 1)
    fig, axes = plt.subplots(3, nc, figsize=(ps.TWO_COLUMN_WIDTH,
                                             3 * panel + 0.85),
                             squeeze=False, constrained_layout=True)
    im_g = im_e = None
    for j, ((role, s), err) in enumerate(zip(parsed, errs)):
        im_g = axes[0][j].imshow(s["f"].reshape(N, N), cmap="gray", vmin=lo,
                                 vmax=hi, interpolation="nearest")
        axes[1][j].imshow(s["u"].reshape(N, N), cmap="gray", vmin=lo, vmax=hi,
                          interpolation="nearest")
        im_e = axes[2][j].imshow(err.reshape(N, N), cmap="coolwarm", vmin=-elim,
                                 vmax=elim, interpolation="nearest")
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
        col = ps.ROLE_STYLES[role]["color"]
        axes[0][j].set_title(
            f"{_short(_col_path(rep, role, j, parsed), 13)}\n"
            f"{'train' if role == 'train' else 'held-out'}",
            fontsize=6, color=col, pad=3)
        axes[1][j].set_xlabel(f"{psnr(s['u_clean'], s['u']):.2f} dB", fontsize=6,
                              color=col, labelpad=2)

    for i, t in enumerate(("noisy $f$", "recon $u$",
                           "$u-f$" if diff == "residual" else "$u-u^\\dagger$")):
        axes[i][0].set_ylabel(t, fontsize=7)

    fig.colorbar(im_g, ax=axes[0:2, :].ravel().tolist(), location="right",
                 fraction=0.02, pad=0.01, label="intensity")
    fig.colorbar(im_e, ax=axes[2, :].ravel().tolist(), location="right",
                 fraction=0.02, pad=0.01, label=f"$\\pm${elim:.3f}")
    fig.suptitle(
        f"{rep['tag']}   $Q(\\alpha^*)$={rep['Q_alpha']:.4f}   "
        f"shared colour scales across all {nc} pairs", fontsize=8)
    ps.savefig(fig, out_base, fmt)
    ps.close(fig)

    # Never truncate silently: a capped sheet that says nothing reads as though
    # it showed the whole set.
    if dropped:
        print(f"    NOTE --max-pairs {max_pairs}: showed {nc} of "
              f"{nc + len(dropped)} pairs; omitted "
              + ", ".join(os.path.basename(p) for p in dropped))


def _col_path(rep, role, j, parsed):
    """Pair name for column j: the report's per-role order matches the files."""
    same = [p for p in rep["pairs"] if p["role"] == role]
    k = sum(1 for r, _ in parsed[:j] if r == role)
    return same[k]["path"] if k < len(same) else f"{role}[{k}]"


# --------------------------------------------------------------------------
# single-run driver
# --------------------------------------------------------------------------
def run_single(rep, out_dir, fmt, max_pairs, diff, sols_dir):
    os.makedirs(out_dir, exist_ok=True)
    tag = rep["tag"]
    print(f"{tag}:  S={rep['S']} N={rep['N']} {rep['solver']} "
          f"nsub={rep['nsub']}  {rep['iters']} its  Q(a*)={rep['Q_alpha']:.4f}")

    fig_gain(rep, os.path.join(out_dir, f"{tag}_gain"), fmt)

    sols = solution_paths(rep, sols_dir)
    if sols["train"] or sols["val"]:
        fig_contact(rep, sols, os.path.join(out_dir, f"{tag}_contact"), fmt,
                    max_pairs=max_pairs, diff=diff)
        # ONE continuation figure. The μ-trace and level history describe the
        # single shared solve, so every per-pair file carries the same copy and
        # rendering S of them would suggest each image had its own path.
        first = (sols["train"] or sols["val"])[0]
        base = os.path.join(out_dir, f"{tag}_continuation")
        fig_continuation(parse_solution(first), base + ".png", 200)
        print(f"    saved → {base}.png   (one copy: the path is shared)")
    else:
        print("    no solution files found — skipping contact sheet and "
              "continuation (pass --sols, or re-run with --save-solution)")


# --------------------------------------------------------------------------
# sweep: reports/*.json joined to timings.csv
# --------------------------------------------------------------------------
def read_timings(run_dir):
    """{(tag, solver): {wall_s, cpu_s, maxrss_MB, exit}} from timings.csv.

    Schema written by `run_timed` in every SLURM script:
        tag,solver,wall_s,cpu_s,maxrss_MB,exit
    Missing file is not an error — a local sweep has no timings.csv, and the
    report's own wall_s covers it.
    """
    path = os.path.join(run_dir, "timings.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            def num(k):
                v = (row.get(k) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return float("nan")       # the scripts write NA without GNU time
            out[(row["tag"], row["solver"])] = dict(
                wall_s=num("wall_s"), cpu_s=num("cpu_s"),
                maxrss_MB=num("maxrss_MB"), exit=num("exit"))
    return out


def load_sweep(run_dir):
    reps = [load_report(p)
            for p in sorted(glob.glob(os.path.join(run_dir, "reports", "*.json")))]
    if not reps:
        raise SystemExit(f"no reports/*.json in {run_dir!r} — a dataset sweep "
                         "writes them with --save-report")
    tim = read_timings(run_dir)
    for r in reps:
        t = tim.get((r["tag"], r["solver"])) or tim.get((r["tag"], _label(r))) or {}
        # timings.csv wins when present (it measures the whole process with GNU
        # time); the in-process wall_s is the fallback for local runs.
        r["wall_s_meas"] = t.get("wall_s", r.get("wall_s", float("nan")))
        r["cpu_s"] = t.get("cpu_s", float("nan"))
        r["maxrss_MB"] = t.get("maxrss_MB", float("nan"))
    return reps


def _label(rep):
    """The series a run belongs to: solver, plus the tiling when it is on."""
    if rep["solver"] != "dd":
        return rep["solver"]
    if rep.get("interface", "direct") == "cg":
        return "dd_cg"
    if rep.get("interface", "direct") == "minres":
        return "dd_minres"
    return "dd" if rep["nsub"] == 1 else "dd_tiles"


def _series(reps):
    """{label: rows sorted by S}."""
    out = {}
    for r in reps:
        out.setdefault(_label(r), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["S"])
    return out


def fig_scaling(reps, out_base, fmt):
    """Wall clock, IPOPT iterations and peak RSS against training-set size.

    Peak RSS earns its panel because it is what decides whether the next S fits
    at all, and the first thing to check when a job dies silently — the same
    reason the SLURM scripts record it. Note it is whole-process RSS, so it
    includes the instance data and IPOPT's own working set, not just the
    factorization: at small N the DD's per-block bookkeeping can exceed what a
    monolithic factorization needs, and the fill argument only pays where that
    factorization is itself the wall. The panel reports the measurement either
    way; it is not evidence for a direction on its own.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    ser = _series(reps)
    fig, axes = plt.subplots(1, 3, figsize=ps.grid_figure_size(
        1, 3, width="twocol", panel_aspect=0.92, extra_height=0.55))

    panels = [("wall_s_meas", "wall clock (s)", True),
              ("iters", "IPOPT iterations", False),
              ("maxrss_MB", "peak RSS (MB)", True)]
    for ax, (key, ylabel, logy) in zip(axes, panels):
        drew = False
        for lbl, rows in sorted(ser.items()):
            x = [r["S"] for r in rows]
            y = [r.get(key, float("nan")) for r in rows]
            if not np.any(np.isfinite(np.asarray(y, dtype=float))):
                continue
            st = ps.solver_style(lbl)
            ax.plot(x, y, marker=st["marker"], ls=st["linestyle"],
                    color=st["color"], label=st["label"], ms=4)
            drew = True
        ax.set_xlabel("training pairs $S$")
        ax.set_ylabel(ylabel)
        if logy and drew:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        if not drew:
            ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes,
                    ha="center", va="center", color=ps.PALETTE["slate"],
                    fontsize=7)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4),
               bbox_to_anchor=(0.5, -0.04))
    n = reps[0]["N"]
    fig.suptitle(f"dataset scaling at N={n}  ($\\sigma$={reps[0]['sigma']:g})",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    ps.savefig(fig, out_base, fmt)
    ps.close(fig)


def fig_dd_vs_mono(reps, out_base, fmt):
    """Per-Newton-step cost, and the α* agreement that doubles as a check.

    Cost per iteration rather than total wall: two runs that differ in
    iteration count (a stalled one against a converged one) are not comparable
    end-to-end, and dividing makes the linear-algebra cost the thing being read.

    The right-hand panel is a correctness check drawn as a figure: every solver
    must land on the same α*, so any visible separation is a bug, not a result.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    ser = _series(reps)
    fig, axes = plt.subplots(1, 2, figsize=ps.grid_figure_size(
        1, 2, width="twocol", panel_aspect=0.68, extra_height=0.5))

    for lbl, rows in sorted(ser.items()):
        st = ps.solver_style(lbl)
        x = [r["S"] for r in rows]
        ms_per_it = [1e3 * r["wall_s_meas"] / r["iters"] if r["iters"] else np.nan
                     for r in rows]
        axes[0].plot(x, ms_per_it, marker=st["marker"], ls=st["linestyle"],
                     color=st["color"], label=st["label"], ms=4)
        axes[1].plot(x, [r["Q_alpha"] for r in rows], marker=st["marker"],
                     ls=st["linestyle"], color=st["color"], label=st["label"],
                     ms=4, alpha=0.85)

    axes[0].set_xlabel("training pairs $S$")
    axes[0].set_ylabel("ms per Newton step")
    axes[1].set_xlabel("training pairs $S$")
    axes[1].set_ylabel("$Q(\\alpha^*)$")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)

    # quantify the agreement instead of leaving it to the eye
    spread = []
    by_s = {}
    for r in reps:
        by_s.setdefault(r["S"], []).append(r["Q_alpha"])
    for s, vals in by_s.items():
        if len(vals) > 1 and max(map(abs, vals)) > 0:
            spread.append((max(vals) - min(vals)) / max(map(abs, vals)))
    if spread:
        axes[1].set_title(f"max relative spread {max(spread):.1e}", fontsize=7)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4),
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    ps.savefig(fig, out_base, fmt)
    ps.close(fig)


# --------------------------------------------------------------------------
# booktabs table (expes_nm idiom: plain string assembly, no siunitx)
# --------------------------------------------------------------------------
def _format_scalar(value):
    value = float(value)
    if not np.isfinite(value):
        return r"\infty"
    a = abs(value)
    if a == 0:
        return "0"
    if 1e-3 <= a < 1e3:
        if a < 0.01:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if a < 0.1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if a < 10:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        if a < 100:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.0f}"
    mant, expo = f"{value:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(expo)}"


def _format_mean_std(mean, std):
    return rf"${_format_scalar(mean)} \pm {_format_scalar(std)}$"


# Per-column formats, in the spirit of expes_fb's METRICS spec. _format_scalar
# above is the expes_nm general-purpose formatter and stays available, but it
# strips trailing zeros — which in a PSNR column prints "1.3" next to "1.72",
# and in the alpha column collapses 0.065594 to "0.066", hiding exactly the
# agreement between solvers this table exists to show.
def _fmt_psnr(mean, std):
    return rf"${mean:.2f} \pm {std:.2f}$"


def _fmt_alpha(q):
    return rf"${q:.5f}$"


def make_table(reps):
    rows = []
    header = (
        r"\begin{table}[t]" "\n"
        r"  \centering" "\n"
        r"  \caption{Dataset runs: one weight $\alpha$ learned across $S$ training "
        r"pairs, decomposed per pair. PSNR is mean$\pm$std over the pairs of a run; "
        r"held-out images are reconstructed by the lower level at $Q(\alpha^*)$.}" "\n"
        r"  \label{tab:dataset}" "\n"
        r"  \scriptsize" "\n"
        r"  \setlength{\tabcolsep}{4pt}" "\n"
        r"  \renewcommand{\arraystretch}{1.08}" "\n"
        r"  \begin{tabular}{rlrrrrrr}" "\n"
        r"    \toprule" "\n"
        r"    $S$ & Solver & Its & $Q(\alpha^*)$ & Train PSNR & Held-out PSNR "
        r"& Wall\,/\,s & Peak RSS\,/\,MB\\" "\n"
        r"    \midrule")
    last_s = None
    for r in sorted(reps, key=lambda r: (r["S"], _label(r))):
        tr = [p["psnr_rof"] for p in pairs_of(r, "train")]
        va = [p["psnr_rof"] for p in pairs_of(r, "val")]
        if last_s is not None and r["S"] != last_s:
            rows.append(r"    \midrule")
        s_cell = f"${r['S']}$" if r["S"] != last_s else ""
        last_s = r["S"]
        lbl = ps.solver_style(_label(r))["label"]
        va_cell = _fmt_psnr(np.mean(va), np.std(va)) if va else "---"
        w = r.get("wall_s_meas", float("nan"))
        m = r.get("maxrss_MB", float("nan"))
        wall = f"{w:.2f}" if np.isfinite(w) else "---"
        rss = f"{m:.0f}" if np.isfinite(m) else "---"
        rows.append(
            f"    {s_cell} & {lbl} & {r['iters']} & "
            f"{_fmt_alpha(r['Q_alpha'])} & "
            f"{_fmt_psnr(np.mean(tr), np.std(tr))} & {va_cell} & "
            f"{wall} & {rss}\\\\")
    footer = (r"    \bottomrule" "\n" r"  \end{tabular}" "\n" r"\end{table}")
    return header + "\n" + "\n".join(rows) + "\n" + footer


def run_sweep(run_dir, out_dir, fmt, paper_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    reps = load_sweep(run_dir)
    print(f"{len(reps)} run(s) from {os.path.join(run_dir, 'reports')}")
    for r in sorted(reps, key=lambda r: (r["S"], _label(r))):
        print(f"    S={r['S']:<3} {_label(r):<10} {r['iters']:>5} its  "
              f"Q(a*)={r['Q_alpha']:.6f}  wall={r['wall_s_meas']:.1f}s")

    fig_scaling(reps, os.path.join(out_dir, "scaling"), fmt)
    fig_dd_vs_mono(reps, os.path.join(out_dir, "dd_vs_mono"), fmt)

    tex = make_table(reps)
    tpath = os.path.join(out_dir, "tab_dataset.tex")
    with open(tpath, "w") as fh:
        fh.write(tex + "\n")
    print(f"    saved → {tpath}")

    # Opt-in only. This repo names no paired manuscript, so a paper path is
    # never guessed — pass --paper-dir when there is one.
    if paper_dir:
        import shutil
        for sub, names in (("figures", ["scaling", "dd_vs_mono"]),
                           ("tables", ["tab_dataset.tex"])):
            dst = os.path.join(paper_dir, sub)
            if not os.path.isdir(dst):
                print(f"    (paper dir not found, skipped: {dst})")
                continue
            for nm in names:
                for cand in ([f"{nm}.pdf", f"{nm}.png"] if "." not in nm else [nm]):
                    src = os.path.join(out_dir, cand)
                    if os.path.isfile(src):
                        shutil.copyfile(src, os.path.join(dst, cand))
                        print(f"    copied → {os.path.join(dst, cand)}")
                        break


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="a --save-report JSON: figures for ONE run")
    g.add_argument("--sweep", help="a results dir holding reports/ (+ timings.csv)")
    ap.add_argument("--out", default=None, help="output dir (default: <dir>/plots)")
    ap.add_argument("--sols", default=None,
                    help="dir holding the sol_*_s000.txt files "
                         "(default: <run-dir>/../sols)")
    ap.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    ap.add_argument("--style", choices=["screen", "paper"], default=None,
                    help="rcParams preset (default: screen for png, paper otherwise)")
    ap.add_argument("--max-pairs", type=int, default=8,
                    help="columns in the contact sheet (default 8); omitted pairs "
                         "are listed, never dropped silently")
    ap.add_argument("--diff", choices=["error", "residual"], default="error")
    ap.add_argument("--paper-dir", default=None,
                    help="also copy figures/tables into <dir>/figures and "
                         "<dir>/tables (opt-in; nothing is guessed)")
    args = ap.parse_args()

    ps.apply_style(args.style or ("screen" if args.format == "png" else "paper"))

    if args.run:
        rep = load_report(args.run)
        out = args.out or os.path.join(os.path.dirname(rep["_dir"]), "plots")
        run_single(rep, out, args.format, args.max_pairs, args.diff, args.sols)
    else:
        out = args.out or os.path.join(args.sweep, "plots")
        run_sweep(args.sweep, out, args.format, args.paper_dir)


if __name__ == "__main__":
    main()
