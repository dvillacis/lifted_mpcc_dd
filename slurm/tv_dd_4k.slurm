#!/bin/bash
#SBATCH --job-name=tv_dd
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
 
set -eo pipefail

# Asegurar la existencia de la carpeta de salida para los registros de SLURM
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_out

TV_SIZE=${TV_SIZE:-3840}
TV_NSUB=${TV_NSUB:-64}
TV_IMAGE=${TV_IMAGE:-../images/tucan.jpg}
TV_EXTRA=${TV_EXTRA:-}
NP=$SLURM_NTASKS

source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
module load intel/2025.0.4 openmpi5/5.0.10

# 1 hilo estricto por tarea para evitar contención en las factorizaciones directas de MUMPS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ ! -x ./tv_dd ] && [ -d cpp ]; then cd cpp; fi

# 1. BYPASS DE BEEGFS (MULTINODO): sbcast transmite síncronamente al /tmp de los 3 nodos
LOCAL_IMG="/tmp/${SLURM_JOB_ID}_$(basename "$TV_IMAGE")"
sbcast -f "$TV_IMAGE" "$LOCAL_IMG"

# 2. BYPASS DE RED MPI: Vader para intra-nodo, TCP para la comunicación inter-nodo en los 3 nodos
TV_MCA="--mca pml ob1 --mca btl self,vader,tcp"
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_allreduce_algorithm=5

run_mpi() {
  local np=$1
  shift
  mpirun -np "$np" \
    --bind-to core \
    $TV_MCA \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS "$@"
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

# 3. Limpieza sincronizada en los 3 nodos físicos
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 rm -f "$LOCAL_IMG"
echo "results in $OUT"