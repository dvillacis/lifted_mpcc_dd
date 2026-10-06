#!/bin/bash
# Build tv_learn.  Needs IPOPT (found through pkg-config), Eigen and zlib.
#
#   ./build.sh            serial
#   OMP=1 ./build.sh      OpenMP: the per-tile loops and the Z columns run in parallel
#
# macOS: Homebrew IPOPT, Eigen and (for OMP=1) libomp.
# Linux: activate a conda env with ipopt and eigen installed first.
set -e
cd "$(dirname "$0")"

FLAGS=(-std=c++17 -O2)
LIBS=()

if [ "$(uname)" = "Darwin" ]; then
  CXX="${CXX:-clang++}"
  SDK="$(xcrun --show-sdk-path)"
  # the Command Line Tools' own libc++ headers can be missing: use the SDK's
  FLAGS+=(-nostdinc++ -isystem "$SDK/usr/include/c++/v1" -isysroot "$SDK")
  FLAGS+=(-I"$(brew --prefix eigen)/include/eigen3")
  if [ "${OMP:-0}" = "1" ]; then
    LIBOMP="$(brew --prefix libomp)"
    FLAGS+=(-Xpreprocessor -fopenmp -I"$LIBOMP/include")
    LIBS+=(-L"$LIBOMP/lib" -lomp)
  fi
else
  CXX="${CXX:-g++}"
  [ -n "${CONDA_PREFIX:-}" ] && export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  if pkg-config --exists eigen3; then
    FLAGS+=($(pkg-config --cflags eigen3))
  else
    FLAGS+=(-I"${CONDA_PREFIX:-/usr}/include/eigen3")
  fi
  [ "${OMP:-0}" = "1" ] && FLAGS+=(-fopenmp)
  LIBS+=(-Wl,-rpath,"$(pkg-config --variable=libdir ipopt)")
fi

# -lz: zlib compresses the .npz solution files (a system library everywhere)
"$CXX" "${FLAGS[@]}" $(pkg-config --cflags ipopt) main.cpp -o tv_learn \
  "${LIBS[@]}" -lz $(pkg-config --libs ipopt)
echo "built ./tv_learn"
