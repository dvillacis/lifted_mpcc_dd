"""Export the staggered-2D instance C++ needs: data, warm start, and the owner map.

The 2D sibling of ``dump_data_1d.py`` — same three reasons the C++ cannot recompute
any of it (NumPy's RNG has no C++ equivalent; the cold start selects a different
branch, so an A/B from one compares basins rather than solvers; and ``kkt_owner`` is
the validated border rule). See that file for the full rationale.

One 2D-specific note: the border here is **derived from the Jacobian sparsity** (a
primal column is complicating iff its rows are owned by ≥2 tiles), which is
stencil-agnostic — so the dumped owner map is meaningful for `onesided` and
`averaged` alike, and the C++ reproduces it by running the identical rule over its
own Jacobian structure.

Format (whitespace-separated, read directly by C++ ``operator>>``)::

    N n_var m_con kkt_dim nsub weight_flag stencil_flag sigma seed
    u_clean…   (N²)
    f…         (N²)
    x0…        (n_var)
    owner…     (kkt_dim)     subdomain of each KKT index, −1 = border

    python dump_data_2d.py --N 16 --nsub 2 -o ../cpp/data/data_2d_16.txt
    (cd ../cpp && ./dd_solve_2d --data data/data_2d_16.txt --solver dd --nsub 2)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpcc_utils import (  # noqa: E402
    Lifted2DMPCC, Partition2D, initial_point, kkt_owner,
    load_image, make_phantom,
)

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "images", "cameraman.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DEFAULT_IMAGE)
    ap.add_argument("--phantom", action="store_true")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--nsub", type=int, default=2, help="tiles PER DIRECTION (k²  total)")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", choices=["linear", "exp"], default="linear")
    ap.add_argument("--stencil", choices=["onesided", "averaged"], default="onesided")
    ap.add_argument("--alpha0", type=float, default=None,
                    help="initial weight (linear) or log-weight (exp); default 0.7·σ")
    ap.add_argument("--init", choices=["cp", "cp-scan", "cold"], default="cp")
    ap.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    w0 = args.alpha0 if args.alpha0 is not None else 0.7 * args.sigma
    if args.weight == "exp" and args.alpha0 is not None:
        w0 = float(np.exp(args.alpha0))
    if w0 <= 0.0:
        raise SystemExit("the initial weight must be > 0")

    if args.phantom:
        u_clean, f = make_phantom(args.N, args.sigma, args.seed)
    else:
        u_clean, f = load_image(args.data, args.N, args.sigma, args.seed)

    prob = Lifted2DMPCC(np.asarray(f).ravel(), np.asarray(u_clean).ravel(), args.N,
                        weight=args.weight, stencil=args.stencil,
                        reg_alpha=args.reg_alpha)
    part = Partition2D(args.N, args.nsub)
    x0 = initial_point(prob, w0, init=args.init)
    owner, col_owner, _ = kkt_owner(prob, part)
    kkt_dim = prob.n + 2 * prob.n_ineq + prob.n_eq
    assert len(owner) == kkt_dim

    with open(args.out, "w") as fh:
        fh.write(f"{args.N} {prob.n} {prob.m_con} {kkt_dim} {args.nsub} "
                 f"{0 if args.weight == 'linear' else 1} "
                 f"{0 if args.stencil == 'onesided' else 1} "
                 f"{args.sigma:.17g} {args.seed}\n")
        for v in (np.asarray(u_clean).ravel(), np.asarray(f).ravel(), x0):
            fh.write(" ".join(f"{x:.17g}" for x in np.asarray(v).ravel()) + "\n")
        fh.write(" ".join(str(int(o)) for o in owner) + "\n")

    p = int((owner < 0).sum())
    n_sub = args.nsub * args.nsub
    print(f"wrote {args.out}")
    print(f"  N={args.N} nodes={prob.m_u} cells={prob.m_q} nsub={args.nsub}×"
          f"{args.nsub}={n_sub} stencil={args.stencil} weight={args.weight} "
          f"init={args.init} w0={w0:.6f}")
    print(f"  n_var={prob.n}  m_con={prob.m_con} ({prob.n_eq} eq + {prob.n_ineq} "
          f"ineq)  KKT dim={kkt_dim}")
    print(f"  border p={p}   block dims={[int((owner == k).sum()) for k in range(n_sub)]}")
    names = {"u": 0, "qx": 0, "qy": 0}
    for c in np.flatnonzero(col_owner < 0):
        if c < prob.off["qx"]:
            names["u"] += 1
        elif c < prob.off["qy"]:
            names["qx"] += 1
        elif c < prob.off["r"]:
            names["qy"] += 1
    print(f"  complicating columns: u={names['u']} qx={names['qx']} "
          f"qy={names['qy']} alpha=1")


if __name__ == "__main__":
    main()
