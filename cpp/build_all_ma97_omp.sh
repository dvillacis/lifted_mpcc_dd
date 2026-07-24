#!/bin/bash
# Build ALL binaries with the MA97 backend and OpenMP. With MA97,
# dd_solver.hpp applies the SPLIT threading model automatically (settled on
# the HPC 2026-07-23 via ma97_smoke_par + gdb): the W_k FACTORIZE loop is
# serial — concurrent ma97_factor heap-corrupts (free() in rfact_block,
# module-global state, not curable by env vars) — while the BACKSOLVE loops
# run one thread per core, which is 97.5% of the S_k cost. Run from cpp/ on
# the cluster with the conda env active:
#
#   conda activate mkl_imaging
#   ./build_all_ma97_omp.sh
#
# Runtime: set the thread count and (belt-and-braces) pin MKL's layer:
#
#   export OMP_NUM_THREADS=8
#   export MKL_THREADING_LAYER=SEQUENTIAL
#
# Validate once per machine/library build before real runs:
#
#   ./ma97_smoke                                        # ABI/library gate
#   OMP_NUM_THREADS=8 DD_PAR_SKIP_FACTOR=1 ./ma97_smoke_par   # concurrency stress (phase B must be OK)
#   DD_CHECK=1 ./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --solver dd
#
# Full concurrent factorization needs a different library: a serial-built
# (no-OpenMP) libhsl_ma97, or MA57 — see the SymBlock note in dd_solver.hpp.
set -e
cd "$(dirname "$0")"

export HSL_SOLVER=ma97          # explicit (auto-detect would pick it anyway)

OMP=1 ./build_linux.sh dd_solve.cpp    -o dd_solve
OMP=1 ./build_linux.sh dd_solve_1d.cpp -o dd_solve_1d
OMP=1 ./build_linux.sh dd_solve_2d.cpp -o dd_solve_2d
./build_linux.sh ma97_smoke.cpp -o ma97_smoke        # gates need no OMP /
OMP=1 ./build_linux.sh ma97_smoke_par.cpp -o ma97_smoke_par   # / stress does

echo
echo "built: dd_solve dd_solve_1d dd_solve_2d ma97_smoke ma97_smoke_par"
echo "       (MA97: serial W_k factorization, parallel W_k backsolves)"
echo "before real runs:"
echo "  export OMP_NUM_THREADS=<cores>  MKL_THREADING_LAYER=SEQUENTIAL"
echo "  ./ma97_smoke && OMP_NUM_THREADS=8 DD_PAR_SKIP_FACTOR=1 ./ma97_smoke_par"
echo "  DD_CHECK=1 ./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --solver dd"
