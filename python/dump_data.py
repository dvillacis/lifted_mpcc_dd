"""Export the exact (u_clean, f) pair Python uses, so C++ solves identical data.

NumPy's RNG cannot be reproduced in C++, so an image loaded independently on each
side gets a *different* noise realization and the two solvers are not comparable.
This writes the Python-side data in the ``.txt`` format ``image_io::load_txt``
reads: ``N``, then N² clean values, then N² noisy values.

    python dump_data.py --N 16 --sigma 0.1 -o ../cpp/data/data_16.txt
    (cd ../cpp && ./dd_solve --data data/data_16.txt --solver mumps)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpcc_utils import load_image, make_phantom  # noqa: E402

DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "images", "cameraman.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DEFAULT_IMAGE)
    ap.add_argument("--phantom", action="store_true")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.phantom:
        u_clean, f = make_phantom(args.N, args.sigma, args.seed)
    else:
        u_clean, f = load_image(args.data, args.N, args.sigma, args.seed)

    with open(args.out, "w") as fh:
        fh.write(f"{args.N}\n")
        for v in (u_clean, f):
            fh.write(" ".join(f"{x:.17g}" for x in np.asarray(v).ravel()))
            fh.write("\n")
    print(f"wrote {args.out}  (N={args.N}, sigma={args.sigma}, seed={args.seed})")


if __name__ == "__main__":
    main()
