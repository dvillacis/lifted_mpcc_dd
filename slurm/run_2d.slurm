#!/bin/bash
# 2D staggered DD sweep on the mariposa image — RUNS PREBUILT BINARIES ONLY
# (compile beforehand with ./build_all_ma97_omp.sh; validate once per machine
# per that script's header). Each size N runs the PNG route (C++ hybrid-CP warm
# start, area-downsample loader) under four solvers: IPOPT-ma97 monolithic
# reference, DD direct interface, DD CG, DD MINRES. Logs, solution/arrowhead
# dumps, a summary, and (best-effort) the plots land in <repo>/results/slurm_<jobid>/.
#
#   sbatch run_2d.slurm                      # from cpp/ on the cluster
#   REPO=/path/to/lifted-mpcc-dd sbatch run_2d.slurm
#
# Notes:
#   * The PNG route's noise draw differs from Python (std::mt19937 vs NumPy
#     PCG64) — self-consistent across solvers here, but not comparable to
#     Python numbers; use dump_data_2d.py dumps for that (see cpp/README.md).
#   * The 2D driver defaults to the μ-coupled t-update (single solve). It has
#     no best-completed-level fallback — if a size misbehaves (mariposa has a
#     known bad-α₀ barrier-stall mode), re-run that size with
#     `--t-update geometric`.
#   * MA97 threading: W_k factorization serial, backsolves parallel (settled
#     2026-07-23); OMP_NUM_THREADS also feeds MA97's internal parallelism,
#     which the big monolithic ma97 runs use fully.
#
#SBATCH --job-name=mpcc_dd2d
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=12:00:00

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
BIN=./dd_solve_2d                               # prebuilt (build_all_ma97_omp.sh)
IMG=../images/cameraman.png                     # bundled; override with any image (e.g. a higher-res one for large N)
NS=(32 48 64 80 96 128)                         # image sizes to sweep
KS=( 8  8  8  8  8   8)                         # nsub per size (edit per-entry)

# --- 1. Environment ----------------------------------------------------------
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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
[ -f "$IMG" ] || { echo "error: $IMG missing"; exit 1; }

# Results go to the REPO ROOT, not into cpp/ — cpp/ is source, and a sweep
# writes hundreds of MB of dumps and figures. Resolved absolutely (after the cd)
# so a relative REPO= still lands somewhere unambiguous, and so the path printed
# below is the one to actually cd into.
RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots"
echo "🚀 Job started at: $(date)   results -> $RUN"

# --- 2. Runs -----------------------------------------------------------------
for i in "${!NS[@]}"; do
  n="${NS[$i]}"; k="${KS[$i]}"
  tag="mariposa_N${n}_k${k}"
  common=(--data "$IMG" --size "$n" --nsub "$k" --init cp)

  $BIN "${common[@]}" --solver ma97 \
      > "$RUN/logs/${tag}_ma97.log" 2>&1
  $BIN "${common[@]}" --solver dd \
      --save-solution "$RUN/sols/sol_${tag}.txt" --save-dd "$RUN/sols/dd_${tag}.txt" \
      > "$RUN/logs/${tag}_dd.log" 2>&1
  $BIN "${common[@]}" --solver dd --interface cg \
      > "$RUN/logs/${tag}_dd_cg.log" 2>&1
  $BIN "${common[@]}" --solver dd --interface minres \
      > "$RUN/logs/${tag}_dd_minres.log" 2>&1
  echo "  done $tag  ($(date +%T))"
done

# --- 3. Summary --------------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "binary: $BIN  image: $IMG  OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
  for f in "$RUN"/logs/mariposa_*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 15 "$f"
  done
} > "$RUN/summary.txt"

# --- 4. Plots (best-effort: needs matplotlib; the self-contained --save-solution
#         dumps in sols/ can also be plotted later with python/plot_slurm.py) ----
export MPLBACKEND=Agg
if ! python ../python/plot_slurm.py "$RUN" --out "$RUN/plots" \
        >> "$RUN/logs/plots.log" 2>&1; then
  echo "note: plotting failed — last errors:"
  tail -n 20 "$RUN/logs/plots.log"
  echo "(the self-contained sols/ dumps can be plotted later with python/plot_slurm.py)"
fi

# keep the SLURM stdout/err with the rest (still open, cp not mv)
cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 Job finished at: $(date)   see $RUN/summary.txt"
conda deactivate
