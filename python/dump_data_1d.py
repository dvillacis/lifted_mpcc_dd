"""Export the staggered-1D instance C++ needs: data, warm start, and the owner map.

``../lifted_mpcc_1d.py`` is the reference implementation; ``dd_solve_1d.cpp`` is the
same formulation solved by real IPOPT with the arrowhead DD as its linear solver.
For the two to be comparable at all, three things must be identical, and none of
them can be recomputed in C++:

* **the data** — NumPy's RNG has no C++ equivalent, so an independently generated
  noise realization makes the two solvers solve different problems (same reason
  ``dump_data.py`` exists for the 2D driver);
* **the warm start** — this repo's own finding is that ``--init cold`` selects the
  spurious no-regularization branch (in 1D it stalls at ``t = 3e-1``, status −3), so
  an A/B from a cold start compares basins, not solvers. The lifted Chambolle–Pock
  point from :func:`initial_point` is dumped whole;
* **the owner map** — ``kkt_owner`` is the validated border rule. The C++ driver
  builds its own copy from the same rule; the dumped array is what certifies the
  two agree (``dd_solve_1d --self-check`` diffs them).

Format (whitespace-separated, so C++ ``operator>>`` reads it directly)::

    n  n_var  m_con  kkt_dim  nsub  weight_flag  sigma  seed     (weight: 0=linear, 1=exp)
    u_clean…   (n)
    f…         (n)
    x0…        (n_var)
    owner…     (kkt_dim)     subdomain of each KKT index, −1 = border

    python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt
    (cd ../cpp && ./dd_solve_1d --data data/data_1d_64.txt --solver dd --nsub 4)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lifted_mpcc_1d import (  # noqa: E402
    Lifted1DMPCC, Partition1D, initial_point, kkt_owner, make_signal,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=64, help="number of NODES (edges = n−1)")
    ap.add_argument("--nsub", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", choices=["linear", "exp"], default="linear")
    ap.add_argument("--alpha0", type=float, default=None,
                    help="initial weight (linear) or log-weight (exp); default 0.7·σ")
    ap.add_argument("--init", choices=["cp", "cp-scan", "cold"], default="cp")
    ap.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    w0 = args.alpha0 if args.alpha0 is not None else 0.7 * args.sigma
    if args.weight == "exp" and args.alpha0 is not None:
        w0 = float(np.exp(args.alpha0))          # --alpha0 is the log-weight
    if w0 <= 0.0:
        raise SystemExit("the initial weight must be > 0")

    u_clean, f = make_signal(args.n, args.sigma, args.seed)
    prob = Lifted1DMPCC(f, u_clean, weight=args.weight, reg_alpha=args.reg_alpha)
    part = Partition1D(args.n, args.nsub)
    x0 = initial_point(prob, w0, init=args.init)
    owner = kkt_owner(prob, part)
    kkt_dim = prob.n + 2 * prob.n_ineq + prob.n_eq
    assert len(owner) == kkt_dim

    with open(args.out, "w") as fh:
        fh.write(f"{args.n} {prob.n} {prob.m_con} {kkt_dim} {args.nsub} "
                 f"{0 if args.weight == 'linear' else 1} {args.sigma:.17g} "
                 f"{args.seed}\n")
        for v in (u_clean, f, x0):
            fh.write(" ".join(f"{x:.17g}" for x in np.asarray(v).ravel()) + "\n")
        fh.write(" ".join(str(int(o)) for o in owner) + "\n")

    p = int((owner < 0).sum())
    print(f"wrote {args.out}")
    print(f"  n={args.n} edges={prob.n_edges} nsub={args.nsub} weight={args.weight} "
          f"init={args.init} w0={w0:.6f}")
    print(f"  n_var={prob.n}  m_con={prob.m_con} ({prob.n_eq} eq + {prob.n_ineq} "
          f"ineq)  KKT dim={kkt_dim}")
    print(f"  border p={p} (= 2(k−1)+1 = {2 * (args.nsub - 1) + 1})   "
          f"block dims={[int((owner == k).sum()) for k in range(args.nsub)]}")


if __name__ == "__main__":
    main()
