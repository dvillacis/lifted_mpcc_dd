#!/bin/bash
# Build ALL binaries with the MA57 backend and OpenMP — the MA57 twin of
# build_all_ma97_omp.sh.
#
# MA57 vs MA97 threading. Under MA97 dd_solver.hpp runs the SPLIT model: the W_k
# FACTORIZE loop is serial because concurrent ma97_factor heap-corrupts. MA57 has
# no such restriction — it has been validated concurrent for years — so ALL the
# parallel regions are live here:
#   * the W_k factorization loop            (factorize)
#   * the S_k formation back-solves         (factorize)
#   * both subdomain loops in solve_one     (the arrowhead back-solve)
#   * apply_S, in --interface cg/minres
#
# READ THIS BEFORE BENCHMARKING. On this project's macOS dev box OpenMP makes the
# TOTAL wall clock WORSE even though the parallel phases scale properly — at
# N=96 with 8x8 tiles, factor+schur go 0.83s -> 0.22s (3.8x) at 8 threads while
# the run gets slower, because the serial remainder degrades faster than the
# parallel part gains on a 4-performance + 4-efficiency core chip. Rebuilding
# MA57 against Accelerate ruled out the BLAS as the cause, and KMP_BLOCKTIME=0 /
# OMP_WAIT_POLICY=passive change nothing. See "Measured negative results" in
# README.md. The per-phase numbers (DD_TIME=1) are the meaningful signal on such
# a machine; the wall clock is not. On a homogeneous many-core Linux node this
# caveat is expected NOT to apply — that is the configuration worth measuring.
#
#   ./build_all_ma57_omp.sh
#   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 ./dd_solve_2d ... --solver dd
#
# OPENBLAS_NUM_THREADS=1 (or VECLIB_MAXIMUM_THREADS=1 against Accelerate) matters:
# the DD loops already use every core, so a threaded BLAS underneath them is
# nested oversubscription.
#
# HSL MA57 is located exactly as build.sh does it — $HSLDIR, default
# ~/.local/hsl-ma57. Override for a different prefix:
#
#   HSLDIR=/opt/hsl-ma57 ./build_all_ma57_omp.sh
set -e
cd "$(dirname "$0")"

# macOS builds through build.sh (Homebrew toolchain, libomp); Linux through
# build_linux.sh, pinned to MA57 so it cannot auto-pick MA97 underneath us.
if [ "$(uname -s)" = "Darwin" ]; then
  BUILDER=./build.sh
else
  BUILDER=./build_linux.sh
  export HSL_SOLVER=ma57
fi

for src in dd_solve dd_solve_1d dd_solve_2d; do
  echo "building $src (MA57 + OpenMP)"
  OMP=1 $BUILDER $src.cpp -o $src
done

# The MUMPS W_k backend is optional; its canary only builds where COIN
# ThirdParty-Mumps is visible to pkg-config, so skip rather than fail without it.
if pkg-config --exists coinmumps 2>/dev/null; then
  echo "building mumps_smoke (coinmumps found)"
  $BUILDER mumps_smoke.cpp -o mumps_smoke
  MUMPS_NOTE="mumps_smoke"
else
  echo "skipping mumps_smoke (no coinmumps via pkg-config)"
  MUMPS_NOTE="(mumps_smoke skipped)"
fi

echo
echo "built: dd_solve dd_solve_1d dd_solve_2d $MUMPS_NOTE"
echo "       (MA57: every parallel region live, factorization included)"
echo
echo "validate before real runs:"
echo "  ./dd_solve_1d --data data/data_1d_n256_k4.txt --nsub 4 --self-check"
echo "  DD_CHECK=1 OMP_NUM_THREADS=8 ./dd_solve_1d --data data/data_1d_n256_k4.txt \\"
echo "      --nsub 4 --solver dd            # inertia + per-step residual vs full MA57"
echo
echo "then, and read the OpenMP caveat at the top of this script first:"
echo "  export OMP_NUM_THREADS=<cores> OPENBLAS_NUM_THREADS=1"
echo "  DD_TIME=1 ./dd_solve_2d --data ../images/cameraman.png --size 96 --nsub 8 \\"
echo "      --solver dd --print-level 0     # per-phase timings, the honest signal"
