# lifted-mpcc-dd

A **domain-decomposition interior point solver** for bilevel total-variation
image denoising, written as a *lifted mathematical program with complementarity
constraints* (MPCC).

<!-- TODO after the first Zenodo release: paste the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

This is the companion code for the paper *(see [Citation](#citation))*. It
packages the C++ solver `tv_dd` together with Python helpers that render the
result figures.

## What it does

`tv_dd` learns the TV-denoising weight α of an image. The lower-level denoising
problem is replaced by its optimality system, lifted to polar coordinates, with one
complementarity condition per cell, relaxed in the sense of Scholtes and driven to
a small level t_min during **one** interior point solve.

- **Consensus form.** Every variable referenced by two image tiles gets a local
  copy per tile and a linking row, so the Newton systems take the arrowhead form
  of Lueg, Bynum, Laird and Biegler (Optim. Eng. 27 (2026) 555–585).
- **Own interior point method.** A primal-dual barrier method with IPOPT's filter
  line search, restoration phase and inertia correction, written around the
  decomposition (why it replaced IPOPT: `docs/cpp_paper_solver`, §7).
- **Schur-complement domain decomposition** for every Newton system: MUMPS
  factorizes each tile block and returns its local Schur complement; PCG with
  additive Schwarz solves the interface system; the inertia is checked tile by
  tile (Haynsworth).
- **MPI over tiles**, each rank holding only its part of the problem; runs are
  bit-identical for any number of ranks.

The full description of the model and the algorithm is
[`cpp/README.md`](cpp/README.md); the measurements, the end-game studies and the
options are there too.

## Repository layout

```
lifted-mpcc-dd/
├── README.md              ← you are here
├── LICENSE                ← BSD-3-Clause (+ third-party notes)
├── CITATION.cff           ← how to cite
├── .zenodo.json           ← Zenodo deposit metadata
├── cpp/                   ← the solver: tv_dd (see cpp/README.md)
│   ├── main.cpp              command line, report, CSV line, .npz output
│   ├── tv_mpcc.hpp           the lifted MPCC in consensus form, continuation, level gate
│   ├── ipm.hpp               the interior point method
│   ├── restoration.hpp       the restoration phase's feasibility problem
│   ├── schur_dd.hpp          the Schur-complement decomposition of each Newton system
│   ├── blocks.hpp, mumps_block.hpp   tile factorizations (MUMPS, sparse, dense)
│   ├── precond.hpp           additive Schwarz and PCG
│   ├── comm.hpp              MPI layer (tile ownership, tile-ordered sums)
│   ├── endgame_literature.md the end game: literature and measurements
│   └── build.sh              build (serial, OMP=1, MPI=1, MUMPS optional)
├── python/                ← plotting helpers (plot_slurm.py reads tv_dd's .npz)
├── images/                ← bundled test images (cameraman 512², mariposa 1184²)
├── docs/                  ← technical notes (LaTeX), incl. cpp_paper_solver
└── archive/               ← earlier solvers, kept for reference (see below)
```

## Requirements

- A C++17 compiler (clang on macOS, g++ on Linux).
- [Eigen](https://eigen.tuxfamily.org/) 3, LAPACK, zlib.
- **MUMPS** (sequential), optional but strongly recommended: COIN-OR
  [ThirdParty-Mumps](https://github.com/coin-or-tools/ThirdParty-Mumps), found
  through pkg-config `coinmumps` (`~/.local/coinmumps` is searched by default).
- **MPI** (Open MPI or MPICH), optional, for distributed runs.
- No IPOPT and no HSL.

Python helpers (plotting only), managed with [uv](https://docs.astral.sh/uv/):

```bash
cd python && uv sync
```

## Build and run

```bash
cd cpp
MPI=1 ./build.sh                 # also: ./build.sh (serial), OMP=1 ./build.sh, MUMPS=0
./mumps_check                    # once per machine: must print PASSED

./tv_dd --size 32 --nsub 4                                          # cameraman 32×32, 4×4 tiles
OMP_NUM_THREADS=1 mpirun -np 8 ./tv_dd --size 128 --nsub 8          # 64 tiles over 8 ranks
./tv_dd --data ../images/mariposa.png --size 48 --nsub 4 --save-solution run.npz
./tv_dd --help
```

Every run ends with one machine-readable `CSV,` line; `--save-solution` writes an
`.npz` that `python/plot_slurm.py` renders. Build details (MUMPS, MPI linking
order) and every option are in [`cpp/README.md`](cpp/README.md).

## The archive

`archive/` holds the earlier solvers, unchanged and still buildable, for reference
and for the measurements quoted in `cpp/README.md` and `docs/`:

- `archive/cpp/` — the first code: a domain-decomposition **linear solver inside
  IPOPT** (`DDArrowheadSolver`, HSL MA57/MA97 tiles), with uniform 2D and
  staggered 1D/2D drivers (`dd_solve*`) and its README of measurements;
- `archive/cpp_minimal/` — the readable minimal MPCC solver under IPOPT
  (`tv_learn`), the problem formulation `cpp/` was ported from;
- `archive/cpp_dd_toy/` — Lueg et al.'s decomposition on a toy problem and on the
  MPCC under two hosts (IPOPT and a paper-style IPM), where `cpp/`'s configuration
  was selected;
- `archive/slurm/`, `archive/python/` — the batch scripts and data dumps of the
  archived drivers.

They need IPOPT 3.14 and, for `archive/cpp`'s DD route, HSL MA57/MA97. The full
history is in git; the state just before the archive was created is commit
`444f5b9`.

## Citation

If you use this software, please cite **both** the software (the Zenodo record)
and the accompanying paper. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button
from it. The Zenodo DOI is added to this section after the first release
(see [`docs/RELEASE.md`](docs/RELEASE.md)).

## License

BSD 3-Clause — see [`LICENSE`](LICENSE). The vendored `stb_image.h` is public
domain / MIT. MUMPS, Eigen and, for the archived code, IPOPT and HSL MA57/MA97 are
**not** distributed with this software and carry their own licences.
