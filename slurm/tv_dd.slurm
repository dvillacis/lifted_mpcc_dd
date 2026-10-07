#!/bin/bash
#SBATCH --job-name=tv_dd
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
 
set -eo pipefail

TV_SIZE=${TV_SIZE:-1024}
TV_NSUB=${TV_NSUB:-16}
TV_IMAGE=${TV_IMAGE:-../images/mariposa.png}
TV_EXTRA=${TV_EXTRA:-}
NP=$SLURM_NTASKS

source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
module load intel/2025.0.4 openmpi5/5.0.10

# Hilos OpenMP por tarea para acelerar MUMPS y operaciones algebraicas locales
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

cd "$SLURM_SUBMIT_DIR"
if [ ! -x ./tv_dd ] && [ -d cpp ]; then cd cpp; fi

# 1. BYPASS DE BEEGFS: Copiar la imagen al disco temporal ultra-rápido del nodo
LOCAL_IMG="/tmp/${SLURM_JOB_ID}_$(basename "$TV_IMAGE")"
cp "$TV_IMAGE" "$LOCAL_IMG"

# 2. BYPASS DE RED MPI: Forzar memoria compartida local y tunear colectivas para PCG
TV_MCA="--mca pml ob1 --mca btl self,vader"
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_allreduce_algorithm=5

run_mpi() {
  local np=$1
  shift
  mpirun -np "$np" \
    --map-by slot:PE=$SLURM_CPUS_PER_TASK \
    --bind-to core \
    $TV_MCA \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x OMP_PROC_BIND -x OMP_PLACES "$@"
}

OUT="$PWD/../results/tvdd_$SLURM_JOB_ID"
mkdir -p "$OUT"
NAME=$(basename "$TV_IMAGE")
TAG="${NAME%.*}_N${TV_SIZE}_k${TV_NSUB}_np${NP}"

ARGS=(--data "$LOCAL_IMG" --size "$TV_SIZE" --nsub "$TV_NSUB" --t-min 0.005 
      --save-solution "$OUT/$TAG.npz" $TV_EXTRA)

echo "command: ./tv_dd ${ARGS[*]}"
run_mpi "$NP" ./tv_dd "${ARGS[@]}" | tee "$OUT/$TAG.log"

grep '^CSV,' "$OUT/$TAG.log" > "$OUT/summary.csv" || true

# Limpieza del archivo temporal en el nodo
rm -f "$LOCAL_IMG"
echo "results in $OUT"