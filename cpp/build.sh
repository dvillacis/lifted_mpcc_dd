#!/bin/bash
# Build helper (macOS) for the domain-decomposition custom-IPOPT-linear-solver.
# Direct interface solve by default; opt-in CG interface via --interface cg.
#
# Handles three environment quirks:
#   * IPOPT 3.14 (Homebrew) via `pkg-config ipopt`. The custom-solver path links
#     against IPOPT's *internal* symbols (SparseSymLinearSolverInterface,
#     TSymLinearSolver, AlgorithmBuilder) — these ARE exported by the Homebrew
#     dylib, so no IPOPT rebuild is needed.
#   * Eigen (`brew --prefix eigen`) for the local sparse assembly.
#   * macOS CLT quirk: the toolchain's usr/include/c++/v1 is empty on this box, so
#     force the SDK's libc++ via -nostdinc++ -isystem ... -isysroot. Without this
#     even #include <iostream> fails.
#
# OpenMP is optional (parallel W_k factorizations): pass OMP=1 to enable it. It is
# OFF by default and the code compiles to a serial loop without it.
#
#   ./build.sh dd_solve.cpp    -o dd_solve
#   ./build.sh dd_solve_1d.cpp -o dd_solve_1d
#   ./build.sh dd_solve_2d.cpp -o dd_solve_2d
#   OMP=1 ./build.sh dd_solve.cpp -o dd_solve
set -e
SDK="$(xcrun --show-sdk-path)"
EIGEN="$(brew --prefix eigen)/include/eigen3"

# image_io.hpp + third_party/stb_image.h. In this standalone package they are
# vendored next to the sources; the parent-dir case is kept so the script also
# works inside the original development monorepo.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/image_io.hpp" ]; then
  ROOT="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../image_io.hpp" ]; then
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "error: image_io.hpp not found next to the sources or one level up" >&2
  exit 1
fi

OMPFLAGS=()
if [ "${OMP:-0}" = "1" ]; then
  LIBOMP="$(brew --prefix libomp)"
  OMPFLAGS=(-Xpreprocessor -fopenmp -I"$LIBOMP/include" -L"$LIBOMP/lib" -lomp)
fi

# COIN ThirdParty-Mumps (optional): enables the --wk-backend mumps W_k backend
# (partial-factorization Schur, mumps_block.hpp). Detected via pkg-config;
# without it the build is byte-identical to before and the flag errors cleanly
# at runtime.
MUMPSFLAGS=()
MUMPSLIBS=()
if pkg-config --exists coinmumps 2>/dev/null; then
  MUMPSFLAGS=($(pkg-config --cflags coinmumps) -DDD_HAVE_MUMPS)
  MUMPSLIBS=($(pkg-config --libs coinmumps) -Wl,-rpath,"$(pkg-config --variable=libdir coinmumps)")
fi

# HSL MA57 — the DD solver factorizes each subdomain block W_k with it. Override
# the prefix with HSLDIR if the library lives elsewhere. The dylib pulls in
# openblas / libfakemetis / libgfortran through its own absolute-path load
# commands, so only MA57 itself needs to be named here.
HSLDIR="${HSLDIR:-$HOME/.local/hsl-ma57}"
if [ ! -f "$HSLDIR/lib/libhsl_ma57.dylib" ]; then
  echo "error: HSL MA57 not found at $HSLDIR/lib/libhsl_ma57.dylib" >&2
  echo "       set HSLDIR to the install prefix (the one holding lib/ and include/)" >&2
  exit 1
fi

exec clang++ -std=c++17 -O2 \
  -nostdinc++ -isystem "$SDK/usr/include/c++/v1" -isysroot "$SDK" \
  -I"$EIGEN" -I"$ROOT" -I"$ROOT/third_party" $(pkg-config --cflags ipopt) \
  "${OMPFLAGS[@]}" "${MUMPSFLAGS[@]}" \
  "$@" \
  -L"$HSLDIR/lib" -lhsl_ma57 -Wl,-rpath,"$HSLDIR/lib" \
  "${MUMPSLIBS[@]}" \
  $(pkg-config --libs ipopt)
