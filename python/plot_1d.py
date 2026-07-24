"""Plot a C++ (`dd_solve_1d`) solution with the Python's own 4-panel figure.

The C++ driver does the solving; the picture stays here. ``plot_solution`` in
``../lifted_mpcc_1d.py`` is already written, validated and the thing every other
1D result in this repo is read from, and the C++ solution is numerically the same
vector — so re-implementing it in C++ would only create a second figure to keep in
sync. This script is the adapter: it reads the instance file
(``dump_data_1d.py``) plus the solution file (``dd_solve_1d --save-solution``),
rebuilds the ``Lifted1DMPCC``/``Partition1D`` pair the plotter wants, and calls it.

Because the panels are identical in construction to the probe's, a C++ figure and
a ``lifted_mpcc_1d.py --save-plot`` figure can be compared directly — which is the
point: it is the same solution reached through a different linear solver.

    (cd ../cpp && ./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd \
        --save-solution sol_1d_64.txt)
    python plot_1d.py --data ../cpp/data/data_1d_64.txt \
        --solution ../cpp/sol_1d_64.txt --save-plot sol_dd.png
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lifted_mpcc_1d import (  # noqa: E402
    Lifted1DMPCC, Partition1D, plot_arrowhead, plot_solution,
)


def load_instance(path):
    """The ``dump_data_1d.py`` file: header, u_clean, f, x0, owner."""
    tok = open(path).read().split()
    i = 0

    def take(k, cast=float):
        nonlocal i
        out = [cast(v) for v in tok[i:i + k]]
        i += k
        return np.array(out) if k > 1 else out[0]

    n = int(tok[0]); n_var = int(tok[1]); w_flag = int(tok[5]); sigma = float(tok[6])
    i = 8
    u_clean = take(n)
    f = take(n)
    take(n_var)                      # x0 — the C++ started from it, not needed here
    return n, u_clean, f, ("exp" if w_flag else "linear"), sigma


def load_solution(path):
    """The ``--save-solution`` file: header, one row per attempted level, then x."""
    tok = open(path).read().split()
    n_var, n_lev = int(tok[0]), int(tok[1])
    t_last, nsub = float(tok[2]), int(tok[3])
    i = 5
    history = []
    for _ in range(n_lev):
        t, status, iters, comp_res, weight, obj, xi_max, ok = tok[i:i + 8]
        i += 8
        history.append({"t": float(t), "status": int(status), "iters": int(iters),
                        "comp_res": float(comp_res), "weight": float(weight),
                        "obj": float(obj), "xi_max": float(xi_max),
                        "converged": bool(int(ok))})
    x = np.array([float(v) for v in tok[i:i + n_var]])
    return x, t_last, nsub, history


class DumpedArrowhead:
    """The C++ arrowhead, exposing exactly the fields ``plot_arrowhead`` reads.

    Deliberately NOT a reconstruction. ``lifted_mpcc_1d.dd_probe`` rebuilds IPOPT's
    KKT from the returned iterate and has to *guess* the pieces IPOPT keeps to
    itself — the barrier Σ (it uses the central-path ``z²/μ``), and δ_w/δ_c. What
    ``--save-dd`` writes is the matrix IPOPT actually handed the solver, with its
    own regularization already in it, plus ``S`` exactly as the solver assembled
    it. So this figure shows the arrowhead that produced a step, not a model of one.

    ``A`` is symmetrized from the lower triangle IPOPT passes; ``loc``/``bord`` come
    from the owner map, in ascending index order — the same order the C++ uses for
    ``S``, so the interface axis labels line up.
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

        rows = tri[:, 0].astype(np.int64)
        cols = tri[:, 1].astype(np.int64)
        L = sp.coo_matrix((tri[:, 2], (rows, cols)), shape=(dim, dim)).tocsr()
        self.A = (L + L.T - sp.diags(L.diagonal())).tocsr()
        self.loc = [np.flatnonzero(owner == k) for k in range(n_sub)]
        self.bord = np.flatnonzero(owner < 0)
        self.p = p
        self.S = S
        self.owner = owner
        if len(self.bord) != p:
            raise SystemExit(f"dump inconsistent: {len(self.bord)} border indices "
                             f"but p={p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the dump_data_1d.py instance file")
    ap.add_argument("--solution", required=True,
                    help="the dd_solve_1d --save-solution file")
    ap.add_argument("--dd-dump", default=None, dest="dd_dump",
                    help="the dd_solve_1d --save-dd file (the arrowhead of the "
                         "last Newton step) — needed for --save-dd-plot")
    ap.add_argument("--nsub", type=int, default=None,
                    help="override the partition drawn as subdomain bands")
    ap.add_argument("--save-plot", default=None, dest="save_plot")
    ap.add_argument("--save-dd-plot", default=None, dest="save_dd_plot",
                    help="write the arrowhead/interface structure figure here")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--label", default=None,
                    help="extra text for the figure title, e.g. the solver used")
    args = ap.parse_args()
    if not (args.save_plot or args.save_dd_plot or args.show):
        raise SystemExit("nothing to do: pass --save-plot/--save-dd-plot and/or --show")
    if args.save_dd_plot and not args.dd_dump:
        raise SystemExit("--save-dd-plot needs --dd-dump FILE (run dd_solve_1d "
                         "with --solver dd --save-dd FILE)")

    n, u_clean, f, weight, sigma = load_instance(args.data)
    x, t_last, nsub, history = load_solution(args.solution)
    if args.nsub is not None:
        nsub = args.nsub

    prob = Lifted1DMPCC(f, u_clean, weight=weight)
    if len(x) != prob.n:
        raise SystemExit(f"solution has {len(x)} entries, this instance needs {prob.n}")
    part = Partition1D(n, nsub)

    a = float(x[prob.off["alpha"]])
    title = (f"1D staggered lifted TV-MPCC (C++ dd_solve_1d) — n={n}, k={nsub}, "
             f"σ={sigma}, Q(α)={'α' if weight == 'linear' else 'e^α'}, "
             f"t={t_last:.1e}, weight={prob.weight_of_alpha(a):.4f}")
    if args.label:
        title += f"  [{args.label}]"
    if args.save_plot or (args.show and not args.save_dd_plot):
        plot_solution(prob, part, x, t_last, history, path=args.save_plot,
                      show=args.show, title=title)
        if args.save_plot:
            print(f"saved {args.save_plot}")

    if args.dd_dump:
        arrow = DumpedArrowhead(args.dd_dump)
        if len(arrow.owner) != prob.n + 2 * prob.n_ineq + prob.n_eq:
            raise SystemExit(f"the dd dump has KKT dim {len(arrow.owner)}, this "
                             f"instance has {prob.n + 2 * prob.n_ineq + prob.n_eq}")
        dims = [len(i) for i in arrow.loc]
        print(f"arrowhead from C++: KKT {arrow.A.shape[0]}×{arrow.A.shape[0]}, "
              f"nnz={arrow.A.nnz}, {len(dims)} blocks of dim {dims}, border p={arrow.p}")
        plot_arrowhead(prob, part, arrow, path=args.save_dd_plot, show=args.show,
                       title=title + "  — last Newton step")
        if args.save_dd_plot:
            print(f"saved {args.save_dd_plot}")


if __name__ == "__main__":
    main()
