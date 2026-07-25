# lifted-mpcc-dd

A **domain-decomposition linear solver for IPOPT**, applied to bilevel
total-variation image denoising written as a *lifted mathematical program with
complementarity constraints* (MPCC) in the unit-ball dual formulation.

<!-- TODO after the first Zenodo release: paste the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This is the companion code for the paper *(see [Citation](#citation))*. It packages
the C++ solver together with Python helpers that generate byte-identical solver
inputs (from the extracted reference formulation) and render the result figures.

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
>
> A 2026-07-25 profiling pass cut wall clock **1.5–2.2×** with new defaults that
> leave every iteration trajectory byte-identical (MA57 BLAS3 multi-RHS, MC64
> scaling off, and a forward-only Schur formation), and `--hessian exact` — the
> analytic `eval_h` the drivers previously never used — is worth another 1.4–3.9×
> up to about N=48. That narrows the gap to monolithic MA57 from 7.0× to 3.1× at
> N=32, but does not change the conclusion above. The same section records the
> negative results — including the OpenMP finding, where the subdomain loops do
> scale **3.8× on 8 threads** but total wall clock still regresses, because the
> serial remainder grows faster on this 4P+4E laptop. That one needs re-measuring
> on the homogeneous Linux/MA97 target.

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
│   ├── build*.sh             build helpers (macOS / Linux / MA57+OMP / MA97+OMP)
│   ├── data/                 sample 1D instances (.txt)
│   └── README.md             detailed technical notes & measurements
├── python/                ← reproduction helpers (data generation + plotting)
│   ├── mpcc_utils.py         data-gen core extracted from the Python reference
│   ├── dump_data*.py         write byte-identical instances for the C++ solver
│   ├── plot_slurm.py         render 2D result figures from --save-solution dumps
│   ├── pyproject.toml        uv project: dependencies + Python floor
│   ├── .python-version       pinned CPython 3.11
│   ├── uv.lock               exact resolved versions (committed)
│   ├── requirements.txt      generated pip fallback
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

**Python helpers** (data generation + plotting only — no IPOPT needed) — managed
with [uv](https://docs.astral.sh/uv/):

```bash
cd python && uv sync
```

`uv` fetches the pinned CPython 3.11 itself, so no system Python is needed; the
dependencies are numpy, scipy, pillow and matplotlib, locked in `python/uv.lock`.
Without uv, `pip install -r python/requirements.txt` into a 3.11 venv installs the
same versions. See [`python/README.md`](python/README.md).

## Build

```bash
cd cpp

# macOS (Homebrew ipopt/eigen/libomp, HSL MA57 in $HSLDIR):
./build.sh dd_solve_1d.cpp -o dd_solve_1d
./build.sh dd_solve_2d.cpp -o dd_solve_2d
./build.sh dd_solve.cpp    -o dd_solve

# Linux (conda env active; auto-picks MA57 or MA97):
./build_linux.sh dd_solve_1d.cpp -o dd_solve_1d
# ... or build everything with OpenMP across the subdomains:
./build_all_ma57_omp.sh          # MA57 (all parallel regions live)
./build_all_ma97_omp.sh          # MA97 on the cluster (serial W_k factorization)
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
uv run python dump_data_1d.py --n 64 --nsub 4 -o ../cpp/data/data_1d_64.txt
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
