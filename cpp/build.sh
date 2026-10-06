#!/bin/bash
# Build tv_dd.  Needs Eigen, LAPACK (dsytrf) and zlib — no IPOPT, no HSL.
# MUMPS is optional: it enables --block-solver mumps / --fallback mumps.
#
#   ./build.sh            serial
#   OMP=1 ./build.sh      OpenMP over the tiles (then set OMP_NUM_THREADS or --threads)
#   MUMPS=0 ./build.sh    build without MUMPS even if it is found
#   MPI=1 ./build.sh      distribute the tiles over MPI ranks (run with mpirun -np R)
#
# MUMPS is found through pkg-config "coinmumps" (COIN-OR ThirdParty-Mumps;
# ~/.local/coinmumps is searched by default), or given explicitly with
#   MUMPS_CFLAGS="-I/path/include" MUMPS_LIBS="-L/path/lib -ldmumps_seq -lmumps_common_seq ..."
# (e.g. conda-forge's mumps-seq).  With MUMPS, mumps_check is built too: run it
# once on every machine before using MUMPS tiles.
#
# macOS: Homebrew Eigen (and libomp for OMP=1); LAPACK from Accelerate.
# Linux: Eigen through pkg-config eigen3 (or $CONDA_PREFIX/include/eigen3) and
#        -llapack -lblas; override with LAPACK_LIBS="-lopenblas" etc.
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
  if pkg-config --exists eigen3 2>/dev/null; then
    FLAGS+=($(pkg-config --cflags eigen3))
  else
    FLAGS+=(-I"${CONDA_PREFIX:-/usr}/include/eigen3")
  fi
  [ -n "${CONDA_PREFIX:-}" ] && LIBS+=(-L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib")
  LIBS+=(${LAPACK_LIBS:--llapack -lblas})
  [ "${OMP:-0}" = "1" ] && FLAGS+=(-fopenmp)
fi

# ---- optional MPI.  The MPI library must come BEFORE MUMPS on the link line:
# MUMPS's sequential build exports its own stand-ins for MPI_Init, MPI_Comm_rank,
# ... (libseq), and the linker binds each symbol to the first library that
# exports it.  So the mpicxx wrapper (which appends -lmpi last) is not used;
# its flags are, in the right order.
PFLAGS=()
PLIBS=()
if [ "${MPI:-0}" = "1" ]; then
  if mpicxx --showme:compile >/dev/null 2>&1; then          # Open MPI
    PFLAGS=(-DDD_HAVE_MPI $(mpicxx --showme:compile))
    PLIBS=($(mpicxx --showme:link))
  else                                                       # MPICH and derivatives
    PFLAGS=(-DDD_HAVE_MPI $(mpicxx -compile_info | cut -d' ' -f2-))
    PLIBS=($(mpicxx -link_info | cut -d' ' -f2-))
  fi
fi

# ---- optional MUMPS
MFLAGS=()
MLIBS=()
if [ "${MUMPS:-1}" != "0" ]; then
  export PKG_CONFIG_PATH="$HOME/.local/coinmumps/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  if [ -n "${MUMPS_LIBS:-}" ]; then
    MFLAGS=(-DDD_HAVE_MUMPS ${MUMPS_CFLAGS:-})
    MLIBS=(${MUMPS_LIBS})
  elif pkg-config --exists coinmumps 2>/dev/null; then
    MFLAGS=(-DDD_HAVE_MUMPS $(pkg-config --cflags coinmumps))
    MLIBS=($(pkg-config --libs coinmumps) -Wl,-rpath,"$(pkg-config --variable=libdir coinmumps)")
  fi
fi

"$CXX" "${FLAGS[@]}" "${PFLAGS[@]}" "${MFLAGS[@]}" main.cpp -o tv_dd "${LIBS[@]}" "${PLIBS[@]}" "${MLIBS[@]}" -lz
echo "built $(pwd)/tv_dd ($([ ${#MFLAGS[@]} -gt 0 ] && echo "with" || echo "without") MUMPS, $([ ${#PFLAGS[@]} -gt 0 ] && echo "with" || echo "without") MPI)"
if [ ${#MFLAGS[@]} -gt 0 ]; then
  "$CXX" "${FLAGS[@]}" "${MFLAGS[@]}" mumps_check.cpp -o mumps_check "${LIBS[@]}" "${MLIBS[@]}"
  echo "built $(pwd)/mumps_check  (run it once on this machine)"
fi
