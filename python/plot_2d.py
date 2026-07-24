"""Plot a C++ (`dd_solve_2d`) solution with the Python's own figures.

The 2D sibling of ``plot_1d.py`` — same division of labour: the C++ solves, the
Python draws, because ``lifted_mpcc_2d.py``'s three figures are already written and
validated and the C++ solution is numerically the same vector. See ``plot_1d.py``
for the full rationale.

Three figures, all from ``lifted_mpcc_2d``:

* ``--save-plot``     the 3×4 solution panel (node mesh, cell mesh, the MPCC itself)
* ``--save-domains``  the tiling with border nodes/cells and the block sizes
* ``--save-dd-plot``  the arrowhead + interface ``S``, from ``--save-dd`` — the
  matrix IPOPT actually handed the solver, not a reconstruction

``--solution`` is all you need: since 2026-07-21 that file carries the instance
(``u_clean``, ``f``, N, stencil, σ) as well as the iterate, so nothing has to be
passed twice and the image route works with no ``.txt`` in sight. ``--data`` stays
available for older solution files and for the Python dumps.

**Never pass the image to ``--data``.** The figures need the exact ``u_clean``/``f``
that were solved, and re-decoding a PNG here would produce a different noise
realization (NumPy's RNG is not the C++ one) — a silently wrong picture. The script
refuses rather than guess.

    (cd ../cpp && ./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 4 \
        --solver dd --save-solution sol.txt --save-dd dd.txt)
    python plot_2d.py --solution ../cpp/sol.txt --dd-dump ../cpp/dd.txt \
        --save-plot sol.png --save-domains dom.png --save-dd-plot dd.png
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lifted_mpcc_2d import (  # noqa: E402
    Lifted2DMPCC, Partition2D, kkt_owner, plot_arrowhead, plot_domains,
    plot_solution,
)


def load_instance(path):
    """The ``dump_data_2d.py`` file: header, u_clean, f, x0, owner."""
    try:
        tok = open(path).read().split()
    except UnicodeDecodeError:
        raise SystemExit(
            f"--data {path!r} is not a text instance file.\n"
            "  It must be a dump_data_2d.py instance (or a dd_solve_2d --save-data\n"
            "  file), NOT the image itself: the figures need the exact u_clean/f\n"
            "  that were solved, and re-decoding the image here would plot a\n"
            "  different noise realization.\n"
            "  Easiest fix: drop --data entirely — a --save-solution file written by\n"
            "  a current dd_solve_2d already carries the instance.")
    N, n_var = int(tok[0]), int(tok[1])
    w_flag, s_flag, sigma = int(tok[5]), int(tok[6]), float(tok[7])
    i, m_u = 9, N * N
    u_clean = np.array(tok[i:i + m_u], dtype=float); i += m_u
    f = np.array(tok[i:i + m_u], dtype=float); i += m_u
    return (N, u_clean, f, "exp" if w_flag else "linear",
            "averaged" if s_flag else "onesided", sigma)


def load_solution(path):
    """The ``--save-solution`` file: header, one row per level, then x — and, from
    2026-07-21, a trailing instance block (N, stencil, sigma, u_clean, f) that makes
    the file self-contained (``inst`` is None for a file written before that), and,
    from 2026-07-23, a trailing μ-trace block (count and column count, then one
    ``iter μ t weight comp`` row per IPOPT iteration of a μ-coupled single solve;
    weight/comp are NaN on restoration iterations; count 0 for geometric runs; an
    early same-day 3-column ``iter μ t`` variant is also accepted). ``mu_trace``
    is None when absent or empty — the per-level panel is drawn then."""
    tok = open(path).read().split()
    n_var, n_lev = int(tok[0]), int(tok[1])
    t_last, nsub = float(tok[2]), int(tok[3])
    w_flag = int(tok[4])
    i, history = 5, []
    for _ in range(n_lev):
        t, status, iters, comp_res, weight, obj, xi_max, ok = tok[i:i + 8]
        i += 8
        history.append({"t": float(t), "status": int(status), "iters": int(iters),
                        "comp_res": float(comp_res), "weight": float(weight),
                        "obj": float(obj), "xi_max": float(xi_max),
                        "converged": bool(int(ok))})
    x = np.array(tok[i:i + n_var], dtype=float); i += n_var
    inst, mu_trace = None, None
    if len(tok) > i:                       # trailing instance block
        N, s_flag, sigma = int(tok[i]), int(tok[i + 1]), float(tok[i + 2])
        i += 3
        m_u = N * N
        u_clean = np.array(tok[i:i + m_u], dtype=float); i += m_u
        f = np.array(tok[i:i + m_u], dtype=float); i += m_u
        inst = (N, u_clean, f, "exp" if w_flag else "linear",
                "averaged" if s_flag else "onesided", sigma)
    if len(tok) > i:                       # trailing μ-trace block
        n_tr = int(tok[i]); i += 1
        # 5-column current format writes the column count; the early 3-column
        # variant wrote none — disambiguate by the remaining token count.
        ncols = 3
        if len(tok) - i not in (3 * n_tr,):
            ncols = int(tok[i]); i += 1
        if n_tr:
            mu_trace = np.array(tok[i:i + ncols * n_tr],
                                dtype=float).reshape(n_tr, ncols)
    return x, t_last, nsub, history, inst, mu_trace


class DumpPartition:
    """The partition the C++ run ACTUALLY used, read off the dumped owner map.

    "Dump, don't reconstruct", same principle as the arrowhead itself: the
    solution-file header records ``nsub`` but NOT the partition type, so a
    Python-side ``Partition2D(N, nsub)`` would silently draw k×k tiles for a
    ``--partition strip`` run. Everything here is derived from the owner vector:

    * ``cell_owner_2d`` from the ``r`` columns (r/δ/θ are never complicating, so
      their owner is the cell's subdomain, never −1);
    * ``node_owner_2d`` by the anchor rule (partition-type agnostic);
    * ``cut_rows`` / ``cut_cols`` wherever adjacent cells change owner — a strip
      partition simply has no vertical cuts;
    * ``promoted_cells`` — the cells whose (λ_h2x, λ_h2y) / (λ_h3x, λ_h3y) pairs
      the C++ promoted to the border (empty for strips).

    Duck-typed for ``plot_domains`` (node_owner_2d, cut_rows, cut_cols, label).
    """

    def __init__(self, arrow, prob):
        nc, off = prob.N - 1, prob.off
        own = np.asarray(arrow.owner)
        self.cell_owner_2d = own[off["r"]:off["r"] + prob.m_q].reshape(nc, nc)
        idx = np.clip(np.arange(prob.N) - 1, 0, nc - 1)          # anchor rule
        self.node_owner_2d = self.cell_owner_2d[np.ix_(idx, idx)]
        co = self.cell_owner_2d
        self.cut_rows = [int(a) for a in range(1, nc) if (co[a] != co[a - 1]).any()]
        self.cut_cols = [int(b) for b in range(1, nc)
                         if (co[:, b] != co[:, b - 1]).any()]
        self.n_sub = len(arrow.loc)
        self.striped = bool(self.cut_rows) and not self.cut_cols
        self.k = self.n_sub if self.striped else int(round(np.sqrt(self.n_sub)))
        self.label = (f"{self.n_sub} strips" if self.striped
                      else f"{self.k}$\\times${self.k} tiles")
        # promoted dual pairs: border λ_c entries mapped back to their cell
        lam0 = prob.n + prob.n_ineq
        prom = arrow.bord[arrow.bord >= prob.n] - lam0
        cells = set()
        for row in ("h2x", "h2y", "h3x", "h3y"):
            lo = prob.roff[row]
            sel = prom[(prom >= lo) & (prom < lo + prob.m_q)]
            cells.update(int(e) for e in sel - lo)
        self.promoted_cells = np.array(sorted(cells), dtype=int)


class DumpedArrowhead:
    """The C++ arrowhead, exposing exactly the fields ``plot_arrowhead`` reads.

    NOT a reconstruction — see ``plot_1d.DumpedArrowhead`` for why that matters.
    In 2D there is one extra wrinkle worth seeing in the picture: the driver
    promotes the ``(λ_h3x, λ_h3y)`` pair of each cut-corner cell to the border, so
    ``p`` here is ``2(k−1)²`` larger than the Python probe's and the border is not
    purely primal.
    """

    def __init__(self, path):
        tok = open(path).read().split()
        dim, nnz, p, n_sub = (int(v) for v in tok[:4])
        i = 4
        tri = np.array(tok[i:i + 3 * nnz], dtype=float).reshape(nnz, 3)
        i += 3 * nnz
        owner = np.array(tok[i:i + dim], dtype=int)
        i += dim
        S = np.array(tok[i:i + p * p], dtype=float).reshape(p, p)
        L = sp.coo_matrix((tri[:, 2], (tri[:, 0].astype(np.int64),
                                       tri[:, 1].astype(np.int64))),
                          shape=(dim, dim)).tocsr()
        self.A = (L + L.T - sp.diags(L.diagonal())).tocsr()
        self.loc = [np.flatnonzero(owner == k) for k in range(n_sub)]
        self.bord = np.flatnonzero(owner < 0)
        self.p, self.S, self.owner = p, S, owner
        self.S_dense = lambda: self.S      # the C++ dumps S densely already
        if len(self.bord) != p:
            raise SystemExit(f"dump inconsistent: {len(self.bord)} border indices "
                             f"but p={p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None,
                    help="a dump_data_2d.py instance (or dd_solve_2d --save-data) "
                         "file. Optional: a current --save-solution file already "
                         "carries the instance. Never the image itself.")
    ap.add_argument("--solution", required=True,
                    help="the dd_solve_2d --save-solution file")
    ap.add_argument("--dd-dump", default=None, dest="dd_dump",
                    help="the dd_solve_2d --save-dd file (needed for --save-dd-plot)")
    ap.add_argument("--nsub", type=int, default=None)
    ap.add_argument("--save-plot", default=None, dest="save_plot")
    ap.add_argument("--save-domains", default=None, dest="save_domains")
    ap.add_argument("--save-dd-plot", default=None, dest="save_dd_plot")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--label", default=None,
                    help="extra text for the figure titles, e.g. the solver used")
    args = ap.parse_args()
    if not (args.save_plot or args.save_domains or args.save_dd_plot or args.show):
        raise SystemExit("nothing to do: pass --save-plot/--save-domains/"
                         "--save-dd-plot and/or --show")
    if args.save_dd_plot and not args.dd_dump:
        raise SystemExit("--save-dd-plot needs --dd-dump FILE (run dd_solve_2d "
                         "with --solver dd --save-dd FILE)")

    x, t_last, nsub, history, inst, mu_trace = load_solution(args.solution)
    if args.data:
        inst = load_instance(args.data)        # explicit --data wins
    elif inst is None:
        raise SystemExit(
            f"{args.solution!r} predates the self-contained format and no --data "
            "was given.\n  Either re-run dd_solve_2d --save-solution (it now embeds "
            "the instance),\n  or pass --data pointing at the matching "
            "dump_data_2d.py / --save-data file.")
    N, u_clean, f, weight, stencil, sigma = inst
    if args.nsub is not None:
        nsub = args.nsub

    prob = Lifted2DMPCC(f, u_clean, N, weight=weight, stencil=stencil)
    if len(x) != prob.n:
        raise SystemExit(f"solution has {len(x)} entries, this instance needs {prob.n}")

    a = float(x[prob.off["alpha"]])
    title = (f"2D staggered lifted TV-MPCC (C++ dd_solve_2d) — N={N}, k={nsub}, "
             f"σ={sigma}, {stencil}, Q(α)={'α' if weight == 'linear' else 'e^α'}, "
             f"t={t_last:.1e}, weight={prob.weight_of_alpha(a):.4f}")
    if args.label:
        title += f"  [{args.label}]"

    # Load the arrowhead first: it is the source of the partition ACTUALLY used
    # (owner map → DumpPartition; correct for tile AND strip runs), of the border
    # markers, and of the per-subdomain block sizes in plot_domains.
    arrow = DumpedArrowhead(args.dd_dump) if args.dd_dump else None
    if arrow is not None:
        part = DumpPartition(arrow, prob)
        col_owner = np.asarray(arrow.owner[:prob.n])
        cuts = (part.cut_rows, part.cut_cols)
        promoted = part.promoted_cells
    else:
        # No dump → the partition type is unknowable (the solution header only
        # records nsub); assume k×k tiles like the Python probe, and say so.
        part = Partition2D(N, nsub)
        _, col_owner, _ = kkt_owner(prob, part)
        cuts = (part.cut_nodes, part.cut_nodes)
        promoted = None
        if args.save_domains:
            print("note: no --dd-dump given — the domain map ASSUMES a "
                  f"{nsub}x{nsub} tile partition; pass --dd-dump to draw the "
                  "partition the run actually used (e.g. --partition strip)")

    if args.save_plot:
        plot_solution(prob, x, t_last, history, path=args.save_plot,
                      show=args.show, title=title, cuts=cuts, mu_trace=mu_trace)
        print(f"saved {args.save_plot}")
    if args.save_domains:
        plot_domains(prob, part, col_owner, arrow=arrow, path=args.save_domains,
                     show=args.show, title=title, promoted=promoted)
        print(f"saved {args.save_domains}")
    if arrow is not None:
        if len(arrow.owner) != prob.n + 2 * prob.n_ineq + prob.n_eq:
            raise SystemExit(f"the dd dump has KKT dim {len(arrow.owner)}, this "
                             f"instance has {prob.n + 2 * prob.n_ineq + prob.n_eq}")
        dims = [len(i) for i in arrow.loc]
        print(f"arrowhead from C++: KKT {arrow.A.shape[0]}×{arrow.A.shape[0]}, "
              f"nnz={arrow.A.nnz}, {len(dims)} blocks of dim {dims}, p={arrow.p}")
        plot_arrowhead(prob, col_owner, arrow, path=args.save_dd_plot,
                       show=args.show, title=title + "  — last Newton step")
        if args.save_dd_plot:
            print(f"saved {args.save_dd_plot}")


if __name__ == "__main__":
    main()
