"""Publication figures from a SLURM result directory (``<repo>/results/slurm_<id>/``).

For every ``sols/sol_<tag>.txt`` a run produced (the self-contained
``dd_solve_2d --save-solution`` file — it carries the instance *and* the μ-trace,
so nothing else is needed), this writes SEVEN stand-alone single-panel PNGs:

  ``<tag>_noisy.png``         the noisy image f
  ``<tag>_recon.png``         the reconstruction u
  ``<tag>_diff.png``          the difference (error u−u† by default, or u−f)
  ``<tag>_delta.png``         the dual radius δ = |q|
  ``<tag>_indexsets.png``     the MPCC index sets (active/inactive/biactive)
  ``<tag>_residual.png``      the complementarity residual r(1−δ)
  ``<tag>_continuation.png``  the continuation path, wide and polished

Unlike ``plot_2d.py`` (which reproduces ``lifted_mpcc_2d``'s full 12-panel probe
figure), this is a lightweight, paper-facing post-processor: it depends only on
``numpy`` + ``matplotlib`` — no ``cyipopt``, no image, no C++ — because a
``--save-solution`` file since 2026-07-21 already embeds ``N``/stencil/σ/
``u_clean``/``f`` and, for a μ-coupled run, the per-iteration ``(μ, t, weight,
comp)`` trace. Decoding is the fixed staggered offset arithmetic; the two tiny
helpers (``psnr``, ``index_sets``) are copied verbatim from ``lifted_mpcc_2d`` so
the numbers match the solver's own reports.

    python plot_slurm.py ../results/slurm_4266
    python plot_slurm.py ../results/slurm_4266 --out /tmp/figs --dpi 200
    python plot_slurm.py ../results/slurm_4266 --only mariposa_N128_k8
    python plot_slurm.py ../results/slurm_4266 --diff residual
"""

import argparse
import glob
import os

import numpy as np


# --- verbatim from lifted_mpcc_2d / _staggered, so numbers match the solver ---
def psnr(clean, recon, data_range=1.0):
    mse = float(np.mean((clean - recon) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)


def index_sets(r, w, eps):
    """0 active (r≈0), 1 inactive (jump: r>0, δ pinned at 1), 2 biactive corner.

    Exhaustive once ``eps ≥ √t`` (``r>eps`` and ``w>eps`` ⇒ ``rw>t``, violating
    the Scholtes row) — the corner is resolved at the O(√t) scale, never at
    machine zero. See ``lifted_mpcc_2d.index_sets``.
    """
    act = (r > eps) & (w <= eps)
    bi = (r <= eps) & (w <= eps)
    out = np.zeros(r.shape, dtype=int)
    out[act] = 1
    out[bi] = 2
    return out


# --- the self-contained --save-solution file (see plot_2d.load_solution) ------
def parse_solution(path):
    """Parse a ``dd_solve_2d --save-solution`` file into a decoded dict.

    Layout: header (n_var, n_lev, t_last, nsub, w_flag), then n_lev per-level
    rows of 8, then x, then the trailing instance block (N, s_flag, σ, u_clean,
    f) and, for a μ-coupled solve, the μ-trace (count, ncols, rows of
    ``iter μ t weight comp``). Raises if the instance block is missing (a file
    from before the self-contained format — this script has no fallback).
    """
    tok = open(path).read().split()
    n_var, n_lev = int(tok[0]), int(tok[1])
    t_last, nsub, w_flag = float(tok[2]), int(tok[3]), int(tok[4])
    i, history = 5, []
    for _ in range(n_lev):
        t, status, iters, comp_res, weight, obj, xi_max, ok = tok[i:i + 8]
        i += 8
        history.append(dict(t=float(t), status=int(status), iters=int(iters),
                            comp_res=float(comp_res), weight=float(weight),
                            obj=float(obj), xi_max=float(xi_max),
                            converged=bool(int(ok))))
    x = np.array(tok[i:i + n_var], dtype=float); i += n_var
    if len(tok) <= i:
        raise SystemExit(f"{path!r} predates the self-contained format "
                         "(no embedded instance) — cannot plot without --data.")
    N, s_flag, sigma = int(tok[i]), int(tok[i + 1]), float(tok[i + 2]); i += 3
    m_u = N * N
    u_clean = np.array(tok[i:i + m_u], dtype=float); i += m_u
    f = np.array(tok[i:i + m_u], dtype=float); i += m_u
    mu_trace = None
    if len(tok) > i:
        n_tr = int(tok[i]); i += 1
        ncols = 3 if len(tok) - i == 3 * n_tr else int(tok[i])
        if ncols != 3:
            i += 1
        if n_tr:
            mu_trace = np.array(tok[i:i + ncols * n_tr],
                                dtype=float).reshape(n_tr, ncols)

    m_q = (N - 1) * (N - 1)
    if n_var != m_u + 5 * m_q + 1:
        raise SystemExit(f"{path!r}: N={N} implies n={m_u + 5 * m_q + 1} but the "
                         f"file has {n_var} variables")
    off = dict(u=0, qx=m_u, qy=m_u + m_q, r=m_u + 2 * m_q, delta=m_u + 3 * m_q,
               theta=m_u + 4 * m_q, alpha=m_u + 5 * m_q)
    a = float(x[off["alpha"]])
    return dict(
        N=N, nc=N - 1, m_u=m_u, m_q=m_q, sigma=sigma,
        stencil="averaged" if s_flag else "onesided",
        weight_mode="exp" if w_flag else "linear",
        alpha=a, weight=(np.exp(a) if w_flag else a),
        u_clean=u_clean, f=f,
        u=x[off["u"]:off["u"] + m_u],
        r=x[off["r"]:off["r"] + m_q].reshape(N - 1, N - 1),
        delta=x[off["delta"]:off["delta"] + m_q].reshape(N - 1, N - 1),
        qx=x[off["qx"]:off["qx"] + m_q].reshape(N - 1, N - 1),
        qy=x[off["qy"]:off["qy"] + m_q].reshape(N - 1, N - 1),
        t_last=t_last, nsub=nsub, history=history, mu_trace=mu_trace,
    )


# --- shared cosmetics ---------------------------------------------------------
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "0.3", "axes.linewidth": 0.8,
        "xtick.color": "0.3", "ytick.color": "0.3",
        "axes.titleweight": "normal", "savefig.facecolor": "white",
    })


def _tag_line(sol):
    q = "e^\\alpha" if sol["weight_mode"] == "exp" else "\\alpha"
    return (f"N={sol['N']}, k={sol['nsub']}, $\\sigma$={sol['sigma']:g}, "
            f"{sol['stencil']}, $Q({q})$={sol['weight']:.4f}, "
            f"$t$={sol['t_last']:.1e}")


# Each panel below is its own single-axis PNG. A shared helper builds a square
# figure, draws the image, adds the colorbar, and saves — so the panels are
# byte-for-byte the same content that used to live in the triptychs, just split.
def _image_panel(draw, path, dpi, *, square=True, cbar=True):
    """``draw(ax) -> mappable|None``; save a clean single-panel figure.

    ``square`` sizes an image panel (with room for a colorbar); ``cbar`` adds one
    from the returned mappable. No global suptitle — the axes title carries the
    label and the tag is in the filename.
    """
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(5.6, 5.2) if square else (6.4, 5.0))
    mappable = draw(ax)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar and mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _gray_range(sol):
    noisy, recon = sol["f"], sol["u"]
    return float(min(noisy.min(), recon.min())), float(max(noisy.max(), recon.max()))


# --- the six image panels (were the two triptychs) ----------------------------
def panel_noisy(sol, path, dpi):
    N, (vmin, vmax) = sol["N"], _gray_range(sol)
    p = psnr(sol["u_clean"], sol["f"])

    def draw(ax):
        im = ax.imshow(sol["f"].reshape(N, N), cmap="gray", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"noisy $f$   {p:.2f} dB")
        return im
    _image_panel(draw, path, dpi)


def panel_recon(sol, path, dpi):
    N, (vmin, vmax) = sol["N"], _gray_range(sol)
    p, p0 = psnr(sol["u_clean"], sol["u"]), psnr(sol["u_clean"], sol["f"])

    def draw(ax):
        im = ax.imshow(sol["u"].reshape(N, N), cmap="gray", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"reconstruction $u$   {p:.2f} dB  ($+{p - p0:.2f}$)")
        return im
    _image_panel(draw, path, dpi)


def panel_diff(sol, path, dpi, diff_mode):
    N = sol["N"]
    recon = sol["u"].reshape(N, N)
    if diff_mode == "residual":            # what the method removed
        err = recon - sol["f"].reshape(N, N)
        ttl = "difference $u - f$   (removed)"
    else:                                  # error vs ground truth (default)
        err = recon - sol["u_clean"].reshape(N, N)
        ttl = "difference $u - u^\\dagger$   (error)"
    lim = float(np.abs(err).max()) or 1.0

    def draw(ax):
        im = ax.imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        ax.set_title(f"{ttl}   max $|\\cdot|$ = {lim:.3f}")
        return im
    _image_panel(draw, path, dpi)


def panel_delta(sol, path, dpi):
    nc, t_last, d = sol["nc"], sol["t_last"], sol["delta"]
    eps_w = min(3.0 * np.sqrt(max(t_last, 0.0)), 0.5)
    n_sat = int((d >= 1.0 - eps_w).sum())

    def draw(ax):
        im = ax.imshow(d, cmap="viridis", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        if 0 < n_sat < d.size:
            ax.contour(d, levels=[1.0 - eps_w], colors="w", linewidths=0.8)
        ax.set_title(f"$\\delta = |q|$   ({nc}$\\times${nc} cells),   "
                     f"saturated {n_sat}/{d.size}\n"
                     f"(white: $1-\\epsilon_w$,  $\\epsilon_w$ = {eps_w:.1e})")
        return im
    _image_panel(draw, path, dpi)


def panel_indexsets(sol, path, dpi):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    t_last, r, d = sol["t_last"], sol["r"], sol["delta"]
    w = 1.0 - d
    eps_w = min(3.0 * np.sqrt(max(t_last, 0.0)), 0.5)
    iset = index_sets(r, w, eps_w)
    iso = ListedColormap(["#2c7fb8", "#f0c419", "#d7191c"])
    n_ac, n_in, n_bi = (int((iset == v).sum()) for v in (0, 1, 2))

    def draw(ax):
        ax.imshow(iset, cmap=iso, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
                  interpolation="nearest")
        handles = [plt.Line2D([], [], marker="s", ls="", ms=9, mfc=c, mec="none",
                              label=l)
                   for c, l in zip(iso.colors,
                                   (f"active  $r\\leq\\epsilon_w<w$   ({n_ac})",
                                    f"inactive  $r>\\epsilon_w\\geq w$   ({n_in})",
                                    f"biactive  both $\\leq\\epsilon_w$   ({n_bi})"))]
        ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.92)
        ax.set_title(f"MPCC index sets   ($\\epsilon_w$ = {eps_w:.1e})\n"
                     f"biactive corner: {n_bi / max(iset.size, 1):.0%}")
        return None
    _image_panel(draw, path, dpi, cbar=False)


def panel_residual(sol, path, dpi):
    from matplotlib.colors import LogNorm
    t_last, r, d = sol["t_last"], sol["r"], sol["delta"]
    comp = r * (1.0 - d)
    pos = comp[comp > 0]
    lo = max(float(pos.min()) if pos.size else t_last * 1e-7, t_last * 1e-7)

    def draw(ax):
        im = ax.imshow(np.maximum(comp, lo), cmap="cividis",
                       norm=LogNorm(vmin=lo, vmax=max(float(comp.max()), t_last)),
                       interpolation="nearest")
        ax.set_title(f"complementarity residual $r(1-\\delta)$\n"
                     f"$\\leq t$ = {t_last:.1e}   (max {float(comp.max()):.1e})")
        return im
    _image_panel(draw, path, dpi)


# --- figure 3: the continuation path, wide and polished -----------------------
def fig_continuation(sol, path, dpi):
    import matplotlib.pyplot as plt
    _style(plt)
    mt, hist = sol["mu_trace"], sol["history"]
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    C_W, C_C = "#1f4e79", "#c1121f"

    if mt is not None and len(mt) and mt.shape[1] >= 5:
        # μ-coupled single solve: the homotopy t = max(t_min, c·μ) runs INSIDE
        # the solve, so the natural x-axis is the IPOPT iteration.
        it, _mu, ts, ws, cr = mt.T[:5]
        ax.plot(it, ws, "-", color=C_W, lw=2.0, label="weight $Q(\\alpha)$")
        ax.fill_between(it, ws, ws.min(), color=C_W, alpha=0.08)
        ax.set_xlabel("IPOPT iteration")
        ax.set_ylabel("weight $Q(\\alpha)$", color=C_W)
        ax.tick_params(axis="y", labelcolor=C_W)
        ax.set_xlim(it.min(), it.max())

        ax2 = ax.twinx()
        with np.errstate(invalid="ignore"):
            crp = np.where(cr > 0, cr, np.nan)
        ax2.plot(it, crp, "-", color=C_C, lw=1.6, label="$\\max\\, r(1-\\delta)$")
        ax2.plot(it, ts, ":", lw=1.4, color="0.45", drawstyle="steps-post",
                 label="Scholtes $t$")
        ax2.set_yscale("log")
        ax2.set_ylabel("$\\max\\, r(1-\\delta)$,   $t$", color=C_C)
        ax2.tick_params(axis="y", labelcolor=C_C)
        sub = ("$\\mu$-coupled single solve   "
               "($t=\\max(t_{\\min},\\,c\\,\\mu)$, tightened per iteration)")
        # annotate the converged endpoint
        ax.annotate(f"$Q(\\alpha^*)$ = {ws[-1]:.4f}",
                    xy=(it[-1], ws[-1]), xytext=(-8, 8),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=C_W)
    elif hist:
        # geometric schedule: one point per Scholtes level, tightening t → 0.
        ts = np.array([h["t"] for h in hist])
        ws = np.array([h["weight"] for h in hist])
        cr = np.array([h["comp_res"] for h in hist])
        okm = np.array([h["converged"] for h in hist])
        ax.plot(ts, ws, "-o", ms=5, color=C_W, label="weight $Q(\\alpha)$")
        if (~okm).any():
            ax.plot(ts[~okm], ws[~okm], "x", ms=11, mew=2.2, color=C_C,
                    label="level failed")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("Scholtes level $t$   (tightening $\\rightarrow$)")
        ax.set_ylabel("weight $Q(\\alpha)$", color=C_W)
        ax.tick_params(axis="y", labelcolor=C_W)
        ax2 = ax.twinx()
        ax2.plot(ts, cr, "-s", ms=5, color=C_C, label="$\\max\\, r(1-\\delta)$")
        ax2.plot(ts, ts, ":", lw=1.2, color="0.45", label="$t$")
        ax2.set_yscale("log")
        ax2.set_ylabel("$\\max\\, r(1-\\delta)$,   $t$", color=C_C)
        ax2.tick_params(axis="y", labelcolor=C_C)
        sub = "geometric Scholtes continuation (one point per level)"
    else:
        raise SystemExit("no continuation data (neither μ-trace nor level history)")

    ax.grid(True, which="major", axis="x", ls="-", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # merged legend across both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center",
              ncol=3, framealpha=0.92, bbox_to_anchor=(0.5, -0.16))
    p_recon = psnr(sol["u_clean"], sol["u"])
    fig.suptitle(f"continuation path  —  {_tag_line(sol)}\n"
                 f"final: $Q(\\alpha^*)$ = {sol['weight']:.4f},   "
                 f"recon {p_recon:.2f} dB",
                 fontsize=10.5, y=1.02)
    ax.set_title(sub, fontsize=9.5, color="0.35", pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slurmdir", help="a SLURM result dir with a sols/ subfolder")
    ap.add_argument("--out", default=None,
                    help="output dir for the PNGs (default: <slurmdir>/plots)")
    ap.add_argument("--only", default=None,
                    help="only process the sol_<TAG>.txt matching this tag")
    ap.add_argument("--diff", choices=["error", "residual"], default="error",
                    help="figure 1 difference panel: error = u-u_clean (default), "
                         "residual = u-f (what the method removed)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    sols_dir = os.path.join(args.slurmdir, "sols")
    if not os.path.isdir(sols_dir):
        raise SystemExit(f"no sols/ subfolder in {args.slurmdir!r}")
    files = sorted(glob.glob(os.path.join(sols_dir, "sol_*.txt")))
    if args.only:
        files = [f for f in files
                 if os.path.basename(f) == f"sol_{args.only}.txt"]
    if not files:
        raise SystemExit(f"no matching sol_*.txt in {sols_dir!r}")

    out_dir = args.out or os.path.join(args.slurmdir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for path in files:
        tag = os.path.basename(path)[len("sol_"):-len(".txt")]
        sol = parse_solution(path)

        def out(name):
            return os.path.join(out_dir, f"{tag}_{name}.png")

        panels = [
            ("noisy", lambda p: panel_noisy(sol, p, args.dpi)),
            ("recon", lambda p: panel_recon(sol, p, args.dpi)),
            ("diff", lambda p: panel_diff(sol, p, args.dpi, args.diff)),
            ("delta", lambda p: panel_delta(sol, p, args.dpi)),
            ("indexsets", lambda p: panel_indexsets(sol, p, args.dpi)),
            ("residual", lambda p: panel_residual(sol, p, args.dpi)),
            ("continuation", lambda p: fig_continuation(sol, p, args.dpi)),
        ]
        print(f"{tag}:  recon {psnr(sol['u_clean'], sol['u']):.2f} dB, "
              f"weight {sol['weight']:.4f}")
        for name, fn in panels:
            p = out(name)
            fn(p)
            print(f"    wrote {p}")


if __name__ == "__main__":
    main()
