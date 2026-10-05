#!/bin/bash
# Build dd_toy.  Needs IPOPT (through pkg-config), Eigen and LAPACK (dsytrf).
#
#   ./build.sh            serial
#   OMP=1 ./build.sh      OpenMP over the tiles in factorize()
#
# macOS: Homebrew IPOPT and Eigen; LAPACK from the Accelerate framework.
# Linux: activate a conda env with ipopt, eigen and lapack first
#        (or set LAPACK_LIBS, e.g. LAPACK_LIBS="-lopenblas").
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
  LIBS+=(${LAPACK_LIBS:--framework Accelerate})
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
  LIBS+=(${LAPACK_LIBS:--llapack -lblas})
  [ "${OMP:-0}" = "1" ] && FLAGS+=(-fopenmp)
  LIBS+=(-Wl,-rpath,"$(pkg-config --variable=libdir ipopt)")
fi

"$CXX" "${FLAGS[@]}" $(pkg-config --cflags ipopt) main.cpp -o dd_toy \
  "${LIBS[@]}" $(pkg-config --libs ipopt)
echo "built $(pwd)/dd_toy"
