# lifted-mpcc-dd

A **domain-decomposition linear solver for IPOPT**, applied to bilevel
total-variation image denoising written as a *lifted mathematical program with
complementarity constraints* (MPCC) in the unit-ball dual formulation.

<!-- TODO after the first Zenodo release: paste the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This is the companion code for the paper *(see [Citation](#citation))*. It packages
the C++ solver, the Python reference implementations it is validated against, and
the scripts that reproduce every figure.

## What it does

The lower-level TV-denoising problem is lifted to an MPCC and relaxed with a
Scholtes ε-continuation; each continuation level is a smooth NLP solved by IPOPT.
The contribution here is the **linear solver inside IPOPT**: a custom
`SparseSymLinearSolverInterface` (`DDArrowheadSolver`) that produces every
interior-point Newton step by **domain decomposition** rather than a monolithic
factorization —

- permute the KKT system to **arrowhead** form (a pure permutation — no
  reformulation), one block per image subdomain plus a shared interface border;
- factorize each subdomain block `W_k` locally with **HSL MA57** (macOS) or
  **MA97** (Linux/HPC), form its local Schur complement `S_k`;
- assemble and solve the interface (Schur) system **directly** by default, or
  iteratively (**preconditioned CG**, or **MINRES** in 2D) as a distributed-DD
  prototype;
- answer IPOPT's **inertia query** from the distributed pieces via the
  **Haynsworth identity** `In(A) = Σ_k In(W_k) + In(S)` — *without ever
  factorizing the full KKT*.

Three validated drivers are included: a uniform square-grid 2D solver, and
staggered (cell-centred) 1D and 2D solvers.

> **On performance — an honest result.** On a single node this DD solver is
> **correct but not faster** than monolithic MA57: the interface factorization is
> an Amdahl floor, and a sparse direct solve on these 2D image KKTs is already
> near-linear. Its value is elsewhere — the Haynsworth inertia feasibility result,
> a large **fill / memory** reduction (the full factor is never formed in one
> address space), and the path to **distributed memory**. See
> [`cpp/README.md`](cpp/README.md) for the full measured story, including the
> no-crossover benchmarks.

## Repository layout

```
lifted-mpcc-dd/
├── README.md              ← you are here
├── LICENSE                ← BSD-3-Clause (+ third-party notes)
├── CITATION.cff           ← how to cite
├── .zenodo.json           ← Zenodo deposit metadata
├── docs/RELEASE.md        ← how to cut a Zenodo release
├── cpp/                   ← the C++ domain-decomposition solver
│   ├── dd_solve.cpp          uniform 2D driver
│   ├── dd_solve_1d.cpp       staggered 1D driver
│   ├── dd_solve_2d.cpp       staggered 2D driver
│   ├── dd_solver.hpp         DDArrowheadSolver (the arrowhead solver)
│   ├── mpcc_*.hpp            the IPOPT TNLPs (problem definitions)
│   ├── ma57_block.hpp        HSL MA57 wrapper
│   ├── ma97_block.hpp        HSL MA97 wrapper (HPC backend)
│   ├── image_io.hpp          image / phantom loader (vendored)
│   ├── third_party/          stb_image.h (vendored, public-domain/MIT)
│   ├── build*.sh             build helpers (macOS / Linux / MA97+OpenMP)
│   ├── data/                 sample 1D instances (.txt)
│   └── README.md             detailed technical notes & measurements
├── python/                ← reference implementations + reproduction helpers
│   ├── lifted_mpcc_*.py      the monolithic cyipopt reference solvers
│   ├── dump_data*.py         write byte-identical instances for the C++ solver
│   ├── plot_*.py             render the solution / arrowhead figures
│   ├── requirements.txt
│   └── README.md
├── slurm/                 ← example HPC batch scripts
└── images/                ← the bundled test image (cameraman)
```

## Requirements

**C++ solver**

- A C++17 compiler (clang on macOS, g++ on Linux).
- [IPOPT](https://coin-or.github.io/Ipopt/) 3.14 — Homebrew `ipopt` (macOS) or
  conda-forge `ipopt` (Linux). The custom-solver path links against IPOPT's
  internal symbols, which the standard shared library exports, so **no IPOPT
  rebuild is needed**.
- [Eigen](https://eigen.tuxfamily.org/) 3 (local sparse assembly).
- **HSL MA57** (macOS/default) or **MA97** (HPC). Proprietary, *free for academic
  use* under the [HSL licence](https://licences.stfc.ac.uk/product/coin-hsl); not
  distributed here. Without it, the monolithic reference route still works with
  IPOPT's bundled MUMPS (`--solver mumps`), but the DD route (`--solver dd`)
  requires MA57/MA97.

**Python reference** — Python 3.11 and `pip install -r python/requirements.txt`
(numpy, scipy, cyipopt, pillow, matplotlib). See [`python/README.md`](python/README.md).

## Build

```bash
cd cpp

# macOS (Homebrew ipopt/eigen/libomp, HSL MA57 in $HSLDIR):
./build.sh dd_solve_1d.cpp -o dd_solve_1d
./build.sh dd_solve_2d.cpp -o dd_solve_2d
./build.sh dd_solve.cpp    -o dd_solve

# Linux (conda env active; auto-picks MA57 or MA97):
./build_linux.sh dd_solve_1d.cpp -o dd_solve_1d
# ... or build everything with MA97 + OpenMP on the cluster:
./build_all_ma97_omp.sh
```

The build scripts document their environment assumptions and the `HSLDIR` /
`HSLLIB` / `CONDA_PREFIX` overrides at the top of each file.

## Quick start

From `cpp/`, after building:

```bash
# 1D: solve a bundled instance and check the port against the reference (5 numbers)
./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --self-check

# 1D: use domain decomposition as the actual linear solver
./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --solver dd

# 1D: opt-in preconditioned-CG interface (the distributed-DD prototype)
./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --solver dd --interface cg

# verify every DD Newton step against MA57-on-the-full-matrix
DD_CHECK=1 ./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --solver dd

# 2D: denoise the bundled cameraman with the DD solver (image → 32×32)
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 4 --solver dd
```

To solve the **same instance in Python and C++** (byte-identical), generate the
dump with the Python helper first:

```bash
cd python
python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt
cd ../cpp && ./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd
```

Full flag reference and the measured findings (interface preconditioners, the
dual peel, MA97 threading, the no-crossover benchmarks) are in
[`cpp/README.md`](cpp/README.md).

## Citation

If you use this software, please cite **both** the software (the Zenodo record)
and the accompanying paper. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button
from it. The Zenodo DOI is added to this section after the first release
(see [`docs/RELEASE.md`](docs/RELEASE.md)).

## License

BSD 3-Clause — see [`LICENSE`](LICENSE). The vendored `stb_image.h` is public
domain / MIT. IPOPT and HSL MA57/MA97 are **not** distributed with this software
and carry their own licences.
