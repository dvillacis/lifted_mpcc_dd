#!/bin/bash
# 1D staggered DD sweep on the HPC — RUNS PREBUILT BINARIES ONLY (no build, no
# gates: compile beforehand with ./build_all_ma97_omp.sh and validate once with
# ma97_smoke + a DD_CHECK run — see that script's header). Every bundled
# instance (data/data_1d_n*_k*.txt) is run under three solvers (IPOPT-ma97
# monolithic reference, DD direct interface, DD CG interface); logs,
# solution/arrowhead dumps, a summary, and (best-effort) the plots land in
# <repo>/results/slurm_<jobid>/.
#
#   sbatch run_1d.slurm                      # from cpp/ on the cluster
#   REPO=/path/to/lifted-mpcc-dd sbatch run_1d.slurm   # repo elsewhere
#
# Based on the Single_node_conda.slurm template. Deviations, on purpose:
#   * --mem=16G, no --exclusive: 16 small 1D runs, not a whole-node job
#     (restore `--mem=0` / `--exclusive` if the cluster policy wants full nodes).
#   * no srun: many short runs inside a single-node allocation — direct
#     execution binds fine and avoids per-call scheduler overhead.
#
#SBATCH --job-name=mpcc_dd1d
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --time=04:00:00

set -o pipefail

# --- 0. Configuration --------------------------------------------------------
# Repo root = the directory holding cpp/ and images/. Explicit REPO= always
# wins; otherwise walk UP from the submit directory looking for that pair, so
# `sbatch` works from the repo root, from cpp/, or from slurm/ without anyone
# having to know the checkout's name. (sbatch copies the script to a node spool
# dir, so $BASH_SOURCE is useless here — SLURM_SUBMIT_DIR is the reliable
# anchor.) The old default hardcoded ~/lifted-mpcc-dd, which does not match what
# `git clone` of lifted_mpcc_dd produces.
if [ -z "${REPO:-}" ]; then
  _d="${SLURM_SUBMIT_DIR:-$PWD}"
  while [ "$_d" != "/" ] && [ -n "$_d" ]; do
    if [ -d "$_d/cpp" ] && [ -d "$_d/images" ]; then REPO="$_d"; break; fi
    _d="$(dirname "$_d")"
  done
fi
REPO="${REPO:-$HOME/lifted-mpcc-dd}"           # last-resort default
BIN=./dd_solve_1d                               # prebuilt (build_all_ma97_omp.sh)
NS=(256 512 1024 2048)                          # instances: data_1d_n${n}_k${k}.txt
KS=(4 8 16 32)

# --- 1. Environment ----------------------------------------------------------
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
# THREADS: how many OpenMP threads the DD loops get. Defaults to the whole
# allocation, but note the parallel loops run ONE TASK PER SUBDOMAIN, so the
# thread count wants to divide nsub^2. At nsub=8 that is 64 tasks:
#     T=16 -> 4 waves, 100%      T=32 -> 2 waves, 100%
#     T=44 -> 2 waves,  73%      (the second wave leaves 24 cores idle)
# So 44 threads can be SLOWER than 32 on the same node. Amdahl bites too: the
# measured serial share is ~10-26%, capping speedup near 4-5x however many cores
# are thrown at it, and under MA97 the W_k FACTORIZE loop is serial by necessity
# (concurrent ma97_factor heap-corrupts), which shrinks the parallel part
# further. The full allocation is still worth requesting: the monolithic
# --solver ma97 reference run uses MA97's own internal parallelism across all of
# it, and owning the node keeps the timings clean.
#
# MEASURE BEFORE COMMITTING A SWEEP — one size, a few thread counts:
#   for T in 8 16 32 44; do
#     OMP_NUM_THREADS=$T DD_TIME=1 ./dd_solve_2d --data ../images/cameraman.png \
#         --size 96 --nsub 8 --solver dd --print-level 0 --max-iter 40
#   done
# The per-phase DD_TIME line (factor/schur are the parallel ones) is the signal.
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=$THREADS
export MKL_THREADING_LAYER=SEQUENTIAL          # keep MKL off libiomp5 under our libgomp threads
export OMP_PROC_BIND=close                     # pin threads to cores (no migration)
export OMP_PLACES=cores

cd "$REPO/cpp" || {
  echo "error: no cpp/ under REPO=$REPO"
  echo "       searched upward from ${SLURM_SUBMIT_DIR:-$PWD} for a dir holding cpp/ and images/"
  echo "       fix: REPO=/path/to/checkout sbatch $0"
  exit 1
}
[ -x "$BIN" ] || { echo "error: $BIN missing — build first: ./build_all_ma97_omp.sh"; exit 1; }

# Results go to the REPO ROOT, not into cpp/ — cpp/ is source, and a sweep
# writes hundreds of MB of dumps and figures. Resolved absolutely (after the cd)
# so a relative REPO= still lands somewhere unambiguous, and so the path printed
# below is the one to actually cd into.
RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots"
echo "🚀 Job started at: $(date)   results -> $RUN"

# --- 2. Runs -----------------------------------------------------------------
for n in "${NS[@]}"; do
  for k in "${KS[@]}"; do
    data="data/data_1d_n${n}_k${k}.txt"
    [ -f "$data" ] || { echo "skip: $data missing"; continue; }
    tag="n${n}_k${k}"

    $BIN --data "$data" --nsub "$k" --solver ma97 \
        > "$RUN/logs/${tag}_ma97.log" 2>&1
    $BIN --data "$data" --nsub "$k" --solver dd \
        --save-solution "$RUN/sols/sol_${tag}.txt" --save-dd "$RUN/sols/dd_${tag}.txt" \
        > "$RUN/logs/${tag}_dd.log" 2>&1
    $BIN --data "$data" --nsub "$k" --solver dd --interface cg \
        > "$RUN/logs/${tag}_dd_cg.log" 2>&1
    echo "  done $tag  ($(date +%T))"
  done
done

# --- 3. Summary --------------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "binary: $BIN  OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
  for f in "$RUN"/logs/n*_*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 12 "$f"
  done
} > "$RUN/summary.txt"

# --- 4. Plots — 1D figures are drawn by the reference implementation's plotter,
#         which is not part of this archival package. The self-contained
#         --save-solution / --save-dd dumps in sols/ are kept for external plotting.

# keep the SLURM stdout/err with the rest (still open, cp not mv)
cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 Job finished at: $(date)   see $RUN/summary.txt"
conda deactivate
