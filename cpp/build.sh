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
#        -llapack -lblas; override with LAPACK_LIBS="-lopenblas" etc.  In a
#        conda env, conda-forge's mumps-seq is picked up when nothing else is.
#
# The cluster (OpenHPC, recognized by /opt/ohpc; HPC=1 / HPC=0 forces it on /
# off) always gets Open MPI and MUMPS, built from inside the conda env:
#   conda activate mkl_imaging && ./build.sh
#   * MPI=1 by default; mpicxx comes from the openmpi5 module, loaded here
#     (HPC_MODULES) when it is not on PATH yet.  The job must load the same.
#   * the system g++ (HPC_CXX) and, through -B/usr/bin, the system linker.
#     Conda's compiler and ld link against conda's own sysroot and cannot
#     resolve what libmpi.so needs (libpmix, libibverbs, libpsm2, ...); conda's
#     ld is first on PATH, hence the -B.
#   * no MUMPS is an error, not a quiet fallback to the sparse+dense tiles
#     (~1000x slower at N=128): conda install -c conda-forge mumps-seq
set -e
cd "$(dirname "$0")"

HPC_AUTO=0
[ "$(uname)" = "Linux" ] && [ -d /opt/ohpc ] && HPC_AUTO=1
case "${HPC:-}" in 0|1) ;; *) HPC=$HPC_AUTO ;; esac
if [ "$HPC" = "1" ]; then
  MPI="${MPI:-1}"
  CXX="${HPC_CXX:-/usr/bin/g++}"
  if [ "$MPI" = "1" ] && ! command -v mpicxx >/dev/null 2>&1; then
    type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true
    module load ${HPC_MODULES:-intel/2025.0.4 openmpi5/5.0.10} 2>/dev/null || true
    if ! command -v mpicxx >/dev/null 2>&1; then
      echo "build.sh: no mpicxx; module load openmpi5 (or set HPC_MODULES)" >&2
      exit 1
    fi
  fi
  echo "HPC build: CXX=$CXX  mpicxx=$(command -v mpicxx || echo none)  conda=${CONDA_PREFIX:-none}"
fi

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
  [ "$HPC" = "1" ] && FLAGS+=(-B/usr/bin)
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
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/include/dmumps_c.h" ] \
       && [ -e "$CONDA_PREFIX/lib/libdmumps_seq.so" ]; then   # conda-forge mumps-seq
    MFLAGS=(-DDD_HAVE_MUMPS -isystem "$CONDA_PREFIX/include")
    MLIBS=(-L"$CONDA_PREFIX/lib")
    for l in dmumps_seq mumps_common_seq pord_seq mpiseq_seq; do
      [ -e "$CONDA_PREFIX/lib/lib$l.so" ] && MLIBS+=(-l$l)
    done
  fi
  if [ "$HPC" = "1" ] && [ ${#MFLAGS[@]} -eq 0 ]; then
    echo "build.sh: no MUMPS found.  In the conda env: conda install -c conda-forge mumps-seq" >&2
    echo "          (or give MUMPS_CFLAGS / MUMPS_LIBS; MUMPS=0 builds without it)" >&2
    for f in include/dmumps_c.h lib/libdmumps_seq.so; do
      [ -e "${CONDA_PREFIX:-/nonexistent}/$f" ] || echo "          missing: ${CONDA_PREFIX:-<no conda env>}/$f" >&2
    done
    exit 1
  fi
fi

"$CXX" "${FLAGS[@]}" "${PFLAGS[@]}" "${MFLAGS[@]}" main.cpp -o tv_dd "${LIBS[@]}" "${PLIBS[@]}" "${MLIBS[@]}" -lz
echo "built $(pwd)/tv_dd ($([ ${#MFLAGS[@]} -gt 0 ] && echo "with" || echo "without") MUMPS, $([ ${#PFLAGS[@]} -gt 0 ] && echo "with" || echo "without") MPI)"
if [ ${#MFLAGS[@]} -gt 0 ]; then
  "$CXX" "${FLAGS[@]}" "${MFLAGS[@]}" mumps_check.cpp -o mumps_check "${LIBS[@]}" "${MLIBS[@]}"
  echo "built $(pwd)/mumps_check  (run it once on this machine)"
fi
