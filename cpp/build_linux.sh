#!/bin/bash
# Linux build helper for the DD solvers — the sibling of build.sh (macOS).
#
# Toolchain and libraries come from a LOCAL ANACONDA/CONDA ENVIRONMENT: activate
# the env first (or pass CONDA_PREFIX=/path/to/envs/<name> explicitly) and the
# script picks up
#
#   * the compiler — conda-forge's `gxx_linux-64` package exports $CXX on
#     activation and that is respected verbatim; otherwise the env's
#     <triplet>-conda-linux-gnu-g++ (or bin/g++) is used. Override with CXX=.
#   * IPOPT via `pkg-config ipopt` with the env's lib/pkgconfig FIRST on
#     PKG_CONFIG_PATH (conda-forge package: `ipopt`). The custom-solver path
#     links against IPOPT's *internal* symbols (SparseSymLinearSolverInterface,
#     TSymLinearSolver, AlgorithmBuilder) — conda-forge's shared libipopt.so
#     exports them by default; a build configured with hidden visibility would
#     need IPOPT rebuilt without it.
#   * Eigen via `pkg-config eigen3` (conda-forge package: `eigen`), falling back
#     to $CONDA_PREFIX/include/eigen3, then /usr/include/eigen3.
#   * an HSL block solver as a SHARED library. HSL is not on conda-forge
#     (license), so this stays a separate install. Two backends:
#       - MA57 (the validated reference): $HSLDIR/lib/libhsl_ma57.so, with
#         $CONDA_PREFIX/lib tried first if HSLDIR is unset.
#       - MA97 (HSL_SOLVER=ma97, or auto-picked when only libhsl_ma97.so is
#         found — the usual HPC case, since IPOPT's ma97/spral builds ship it):
#         compiles with -DDD_USE_MA97 and links libhsl_ma97.so instead. The
#         library must export the C interface (ma97_*_d symbols — checked with
#         nm when available); note libhsl_ma97.so does NOT contain MA57.
#     Force the choice with HSL_SOLVER=ma57|ma97 (default: auto, MA57 first).
#     The library's BLAS/LAPACK/gfortran deps normally ride along as DT_NEEDED
#     entries of the .so; if yours was linked without them, list the extras in
#     HSL_EXTRA_LIBS, e.g. HSL_EXTRA_LIBS="-lopenblas -lgfortran".
#
# The env's lib/ is put on the link line and baked into the rpath, so the
# binaries find conda's libipopt.so (and its MUMPS/BLAS deps) at runtime without
# LD_LIBRARY_PATH games.
#
# OpenMP (parallel W_k/S_k/apply_S loops) is opt-in, as on macOS: OMP=1.
#
#   conda activate <env>
#   ./build_linux.sh dd_solve.cpp    -o dd_solve
#   ./build_linux.sh dd_solve_1d.cpp -o dd_solve_1d
#   OMP=1 ./build_linux.sh dd_solve_2d.cpp -o dd_solve_2d
#   CONDA_PREFIX=$HOME/anaconda3/envs/mpcc ./build_linux.sh dd_solve.cpp -o dd_solve
#
# Runtime note: the drivers' --solver ma57 route (IPOPT's own MA57) reads the
# HSLLIB env var for the library IPOPT should dlopen — on Linux point it at the
# same .so:  HSLLIB=$HSLDIR/lib/libhsl_ma57.so ./dd_solve_2d --solver ma57 ...
# On an MA97 build the monolithic reference is --solver ma97 instead (no HSLLIB
# needed when IPOPT itself was built with MA97, as conda/HPC ipopt builds are).
set -e
# The sources include "image_io.hpp" and third_party/stb_image.h. In this
# standalone package they are vendored next to the sources; the parent-dir case
# (the original development monorepo) is also accepted — both layouts work.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../image_io.hpp" ]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -f "$SCRIPT_DIR/image_io.hpp" ]; then
  ROOT="$SCRIPT_DIR"
else
  echo "error: image_io.hpp not found in $SCRIPT_DIR/.. or $SCRIPT_DIR" >&2
  echo "       copy image_io.hpp and third_party/stb_image.h from the repo root" >&2
  echo "       (keeping the third_party/ subdirectory) next to the sources, or" >&2
  echo "       copy the full repo so they sit one level above this script" >&2
  exit 1
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: CONDA_PREFIX is not set — activate the anaconda environment first" >&2
  echo "       (conda activate <env>), or pass it explicitly:" >&2
  echo "       CONDA_PREFIX=\$HOME/anaconda3/envs/<name> $0 ..." >&2
  exit 1
fi

# Compiler: an activated conda compiler package (gxx_linux-64) exports CXX —
# use it as-is. Otherwise take the env's cross-triplet g++/c++, then bin/g++,
# and only then the system g++.
if [ -z "${CXX:-}" ]; then
  for c in "$CONDA_PREFIX"/bin/*-conda-linux-gnu-g++ \
           "$CONDA_PREFIX"/bin/*-conda-linux-gnu-c++ \
           "$CONDA_PREFIX/bin/g++" "$CONDA_PREFIX/bin/clang++"; do
    if [ -x "$c" ]; then CXX="$c"; break; fi
  done
fi
if [ -z "${CXX:-}" ]; then
  echo "warning: no compiler found in $CONDA_PREFIX/bin (install gxx_linux-64" >&2
  echo "         in the env); falling back to system g++" >&2
  CXX=g++
fi

# Resolve IPOPT/Eigen .pc files from the env first, whatever pkg-config is used.
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

if ! pkg-config --exists ipopt; then
  echo "error: pkg-config cannot find ipopt in $CONDA_PREFIX" >&2
  echo "       install it in the env:  conda install -c conda-forge ipopt" >&2
  exit 1
fi

if pkg-config --exists eigen3; then
  EIGEN_CFLAGS="$(pkg-config --cflags eigen3)"
elif [ -d "$CONDA_PREFIX/include/eigen3" ]; then
  EIGEN_CFLAGS="-I$CONDA_PREFIX/include/eigen3"
elif [ -d /usr/include/eigen3 ]; then
  EIGEN_CFLAGS="-I/usr/include/eigen3"
else
  echo "error: Eigen not found — install it in the env:" >&2
  echo "       conda install -c conda-forge eigen" >&2
  exit 1
fi

OMPFLAGS=()
if [ "${OMP:-0}" = "1" ]; then
  OMPFLAGS=(-fopenmp)
fi

# HSL block-solver backend. Neither MA57 nor MA97 is a conda package; search
# $HSLDIR/lib (if set), then the env's lib/, then the standalone MA57 prefix.
# HSL_SOLVER forces the backend; the default is auto (MA57 first — the
# validated reference — then MA97).
find_hsl() {  # $1 = library basename without lib/.so
  for d in ${HSLDIR:+"$HSLDIR/lib"} "$CONDA_PREFIX/lib" "$HOME/.local/hsl-ma57/lib"; do
    if [ -f "$d/lib$1.so" ]; then echo "$d"; return 0; fi
  done
  return 1
}
HSL_SOLVER="${HSL_SOLVER:-auto}"
HSL_DEFS=()
case "$HSL_SOLVER" in
  auto)
    if HSL_LIBDIR="$(find_hsl hsl_ma57)"; then HSL_SOLVER=ma57
    elif HSL_LIBDIR="$(find_hsl hsl_ma97)"; then
      HSL_SOLVER=ma97
      echo "note: no libhsl_ma57.so found — building the MA97 backend (-DDD_USE_MA97)" >&2
    else
      echo "error: no HSL block solver found (searched \${HSLDIR}/lib," >&2
      echo "       $CONDA_PREFIX/lib, ~/.local/hsl-ma57/lib for" >&2
      echo "       libhsl_ma57.so / libhsl_ma97.so)." >&2
      echo "       Install one and/or set HSLDIR to its prefix (the one holding" >&2
      echo "       lib/), or copy the .so into $CONDA_PREFIX/lib. A static-only" >&2
      echo "       .a also works if you add its Fortran/BLAS dependencies via" >&2
      echo "       HSL_EXTRA_LIBS." >&2
      exit 1
    fi ;;
  ma57|ma97)
    if ! HSL_LIBDIR="$(find_hsl "hsl_$HSL_SOLVER")"; then
      echo "error: libhsl_$HSL_SOLVER.so not found (searched \${HSLDIR}/lib," >&2
      echo "       $CONDA_PREFIX/lib, ~/.local/hsl-ma57/lib)" >&2
      exit 1
    fi ;;
  *) echo "error: HSL_SOLVER must be ma57, ma97 or auto" >&2; exit 1 ;;
esac
HSL_LIB="hsl_$HSL_SOLVER"
if [ "$HSL_SOLVER" = "ma97" ]; then
  HSL_DEFS=(-DDD_USE_MA97)
  # MA97 + OpenMP (OMP=1): dd_solver.hpp automatically applies the SPLIT model
  # — serial factorize loop (concurrent ma97_factor heap-corrupts; measured,
  # gdb'd and not curable by env vars, 2026-07-23), parallel backsolve loops
  # (validated by ma97_smoke_par phase B; they carry 97.5% of the S_k cost).
  # The wrapper calls MA97's C interface; a Fortran-only build lacks it.
  if command -v nm >/dev/null 2>&1 && \
     ! nm -D --defined-only "$HSL_LIBDIR/lib$HSL_LIB.so" 2>/dev/null | grep -qw ma97_analyse_coord_d; then
    echo "error: $HSL_LIBDIR/lib$HSL_LIB.so does not export ma97_analyse_coord_d —" >&2
    echo "       it was built without the C interface (hsl_ma97_ciface); rebuild" >&2
    echo "       hsl_ma97 with it (IPOPT's own ma97 route requires it too)" >&2
    exit 1
  fi
fi

# Link-line notes, both learned on the HPC (2026-07-23):
#   -rpath-link lets ld resolve the TRANSITIVE DT_NEEDED closure of the conda
#     libs (libipopt → libhsl_ma97/libspral/libdmumps → MKL, metis, gfortran,
#     gomp, ...) — without it every one of those surfaces as "libX not found
#     (try using -rpath)" plus undefined-reference spam.
#   --allow-shlib-undefined covers shared libs from OUTSIDE the env (e.g. an
#     OHPC libhwloc pulled in via LD_LIBRARY_PATH) whose system deps the conda
#     toolchain's sysroot cannot see (libxml2); the runtime loader resolves
#     them. Undefined symbols in OUR objects still error, so it masks nothing.
# shellcheck disable=SC2086  # HSL_EXTRA_LIBS is deliberately word-split
exec "$CXX" -std=c++17 -O2 \
  $EIGEN_CFLAGS -I"$ROOT" -I"$ROOT/third_party" $(pkg-config --cflags ipopt) \
  "${HSL_DEFS[@]}" "${OMPFLAGS[@]}" \
  "$@" \
  -L"$HSL_LIBDIR" -l"$HSL_LIB" -Wl,-rpath,"$HSL_LIBDIR" \
  ${HSL_EXTRA_LIBS:-} \
  -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
  -Wl,-rpath-link,"$CONDA_PREFIX/lib" -Wl,--allow-shlib-undefined \
  $(pkg-config --libs ipopt)
