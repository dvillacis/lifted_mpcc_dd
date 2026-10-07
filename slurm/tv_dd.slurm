#!/bin/bash
# tv_dd with Open MPI on one node.  Submit from cpp/ or the repo root:
#   sbatch tv_dd.slurm
#   TV_SIZE=128 TV_NSUB=8 TV_IMAGE=../images/mariposa.png sbatch tv_dd.slurm
#   sbatch --ntasks=16 tv_dd.slurm
#
#SBATCH --job-name=tv_dd
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
 
set -eo pipefail

# ---- problem
TV_SIZE=${TV_SIZE:-1024}
TV_NSUB=${TV_NSUB:-16}
TV_IMAGE=${TV_IMAGE:-../images/mariposa.png}
TV_EXTRA=${TV_EXTRA:-}
NP=$SLURM_NTASKS

# ---- environment (conda first, modules after: Open MPI's mpirun wins)
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
module load intel/2025.0.4 openmpi5/5.0.10
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- run inside cpp/ (image paths are relative to it)
cd "$SLURM_SUBMIT_DIR"
if [ ! -x ./tv_dd ] && [ -d cpp ]; then cd cpp; fi
if [ ! -x ./tv_dd ]; then
  echo "no tv_dd in $SLURM_SUBMIT_DIR or its cpp/"
  exit 1
fi

echo "dir:    $PWD"
echo "node:   $(hostname)   ranks: $NP"
echo "mpirun: $(which mpirun)"

if ldd ./tv_dd | grep -q "not found"; then
  echo "missing libraries:"
  ldd ./tv_dd | grep "not found"
  exit 1
fi

# extra Open MPI flags; on ucx/psm2/openib complaints use:
#   TV_MCA="--mca pml ob1 --mca btl self,sm" sbatch tv_dd.slurm
TV_MCA=${TV_MCA:-}

# run_mpi() {
#   local np=$1
#   shift
#   mpirun -np "$np" --bind-to core --map-by core $TV_MCA \
#     -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS "$@"
# }
# ---- Función MPI corregida para Híbrido (MPI + OpenMP/MKL)
run_mpi() {
  local np=$1
  shift
  
  # --map-by slot:PE=N instruye a OpenMPI a agrupar N cores por cada tarea MPI.
  # --bind-to core asegura que los hilos no salten entre diferentes procesadores.
  mpirun -np "$np" \
    --map-by slot:PE=$SLURM_CPUS_PER_TASK \
    --bind-to core \
    $TV_MCA \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS "$@"
}

# ---- smoke test: one CSV line, ranks field (10th) = 2.  Two lines or
# ranks=1 means sequential MUMPS's fake MPI got bound instead of Open MPI.
SMOKE=$(run_mpi 2 ./tv_dd --size 32 --nsub 4 --quiet | grep '^CSV,' || true)
NLINES=$(echo "$SMOKE" | grep -c '^CSV,' || true)
RANKS=$(echo "$SMOKE" | head -1 | cut -d, -f10)
if [ "$NLINES" != 1 ] || [ "$RANKS" != 2 ]; then
  echo "MPI smoke test FAILED:"
  echo "$SMOKE"
  exit 1
fi
echo "MPI smoke test ok"

# ---- the run
OUT="$PWD/../results/tvdd_$SLURM_JOB_ID"
mkdir -p "$OUT"
NAME=$(basename "$TV_IMAGE")
TAG="${NAME%.*}_N${TV_SIZE}_k${TV_NSUB}_np${NP}"

# ARGS=(--data "$TV_IMAGE" --size "$TV_SIZE" --nsub "$TV_NSUB" --t-min 0.01 --t-mu-scale 10
#       --save-solution "$OUT/$TAG.npz" $TV_EXTRA)
ARGS=(--data "$TV_IMAGE" --size "$TV_SIZE" --nsub "$TV_NSUB" --t-min 0.001 --t-mu-scale 10 --max-iter 100
      --save-solution "$OUT/$TAG.npz" $TV_EXTRA)
echo "command: ./tv_dd ${ARGS[*]}"

run_mpi "$NP" ./tv_dd "${ARGS[@]}" | tee "$OUT/$TAG.log"

grep '^CSV,' "$OUT/$TAG.log" > "$OUT/summary.csv" || true
echo "results in $OUT"
