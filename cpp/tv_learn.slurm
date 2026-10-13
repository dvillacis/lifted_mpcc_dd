#!/bin/bash
# tv_learn with Open MPI on one node.  Submit from cpp/ or the repo root:
#   sbatch tv_learn.slurm                                       cameraman, one image
#   TL_TRAIN=~/data/kodak/train.txt TL_VAL=~/data/kodak/val.txt TL_SIZE=64 TL_NSUB=2 sbatch tv_learn.slurm
#   TL_EXTRA="--op deblur --blur-sigma 1 --sigma 0.02 --block 16" sbatch tv_learn.slurm
#
# Writes results/tvlearn_<job>/{reports/report_<tag>.json, sols/sol_<tag>_s000.npz ..., <tag>.log,
# summary.csv}: the layout python/plot_dataset.py --run and python/plot_slurm.py read.
#
#SBATCH --job-name=tv_learn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -eo pipefail

# ---- problem
TL_TRAIN=${TL_TRAIN:-../images/cameraman.png}
TL_VAL=${TL_VAL:-}
TL_SIZE=${TL_SIZE:-64}
TL_NSUB=${TL_NSUB:-4}
TL_EXTRA=${TL_EXTRA:-}
NP=$SLURM_NTASKS

# ---- environment (conda first, modules after: Open MPI's mpirun wins)
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
module load intel/2025.0.4 openmpi5/5.0.10
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- run inside cpp/ (relative image paths are relative to it)
cd "$SLURM_SUBMIT_DIR"
if [ ! -x ./tv_learn ] && [ -d cpp ]; then cd cpp; fi
if [ ! -x ./tv_learn ]; then
  echo "no tv_learn in $SLURM_SUBMIT_DIR or its cpp/ (TARGETS=tv_learn ./build.sh)"
  exit 1
fi

echo "dir:    $PWD"
echo "node:   $(hostname)   ranks: $NP"
echo "mpirun: $(which mpirun)"

if ldd ./tv_learn | grep -q "not found"; then
  echo "missing libraries:"
  ldd ./tv_learn | grep "not found"
  exit 1
fi

TL_MCA=${TL_MCA:-}
run_mpi() {
  local np=$1
  shift
  mpirun -np "$np" --bind-to core --map-by core $TL_MCA \
    -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS "$@"
}

# ---- smoke test: one CSV line with ranks (10th field) = 2 (see tv_dd.slurm)
SMOKE=$(run_mpi 2 ./tv_learn --size 32 --nsub 4 --quiet | grep '^CSV,' || true)
NLINES=$(echo "$SMOKE" | grep -c '^CSV,' || true)
RANKS=$(echo "$SMOKE" | head -1 | cut -d, -f10)
if [ "$NLINES" != 1 ] || [ "$RANKS" != 2 ]; then
  echo "MPI smoke test FAILED:"
  echo "$SMOKE"
  exit 1
fi
echo "MPI smoke test ok"

# ---- the run
OUT="$PWD/../results/tvlearn_$SLURM_JOB_ID"
mkdir -p "$OUT/reports" "$OUT/sols"
NAME=$(basename "$TL_TRAIN")
TAG="${NAME%.*}_N${TL_SIZE}_k${TL_NSUB}_np${NP}"

ARGS=(--train "$TL_TRAIN" --size "$TL_SIZE" --nsub "$TL_NSUB" --tag "$TAG"
      --save-report "$OUT/reports/report_$TAG.json" --save-solution "$OUT/sols/sol_$TAG")
[ -n "$TL_VAL" ] && ARGS+=(--val "$TL_VAL")
ARGS+=($TL_EXTRA)
echo "command: ./tv_learn ${ARGS[*]}"

run_mpi "$NP" ./tv_learn "${ARGS[@]}" | tee "$OUT/$TAG.log"

grep '^CSV,' "$OUT/$TAG.log" > "$OUT/summary.csv" || true
echo "results in $OUT"
