#!/bin/bash
# 2D staggered DD sweep on the mariposa image — RUNS PREBUILT BINARIES ONLY
# (compile beforehand with ./build_all_ma57_omp.sh; validate once per machine
# per that script's header). Each size N runs the PNG route (C++ hybrid-CP warm
# start, area-downsample loader) under four solvers: IPOPT-ma57 monolithic
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
#   * MA57 threading: all parallel regions are live (W_k factorization, the S_k
#     back-solves, and both subdomain loops in solve_one). OMP_NUM_THREADS also
#     feeds the BLAS under the big monolithic ma57 runs — see THREADS below.
#
#SBATCH --job-name=mpcc_dd2d
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
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
BIN=./dd_solve_2d                               # prebuilt (build_all_ma57_omp.sh)
# (IMG/IMGTAG resolved below, after the repo root is known)
NS=(32 48 64 80 96 128)                         # image sizes to sweep
KS=( 8  8  8  8  8   8)                         # nsub per size (edit per-entry)

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
# are thrown at it. Unlike MA97 — whose W_k factorize loop must run serial,
# because concurrent ma97_factor heap-corrupts — MA57 is validated concurrent, so
# EVERY parallel region is live here and the parallel fraction is as large as this
# solver gets. The full allocation is still worth requesting: the monolithic
# --solver ma57 reference run uses its BLAS across all of it, and owning the node
# keeps the timings clean.
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
# Image. mariposa.png if it is there, else the bundled cameraman. IMG= overrides.
# The TAG is derived from whatever is actually loaded, so logs, solution dumps
# and figures can never again claim one image while the run used another (which
# is what "mariposa_*" tags on cameraman runs used to do).
if [ -z "${IMG:-}" ]; then
  for _c in ../images/mariposa.png ../images/cameraman.png; do
    [ -f "$_c" ] && { IMG="$_c"; break; }
  done
fi
IMG="${IMG:-../images/cameraman.png}"
IMGTAG="$(basename "$IMG")"; IMGTAG="${IMGTAG%.*}"
[ -x "$BIN" ] || { echo "error: $BIN missing — build first: ./build_all_ma57_omp.sh"; exit 1; }
# IPOPT's OWN --solver ma57 reference run dlopen()s MA57 at RUNTIME via its
# hsllib option, unlike the DD runs which link it at build time through an
# rpath. So the reference can fail on a node where the DD runs are perfectly
# fine. driver_common.hpp searches HSLLIB, then $HSLDIR/lib, then
# ~/.local/hsl-ma57/lib, then the bare soname. Warn rather than abort: a sweep
# whose DD numbers are good but whose baseline is missing is still worth having,
# and IPOPT prints its own DYNAMIC_LIBRARY_FAILURE into the ma57 log.
_ma57_found=""
for _c in "$HSLLIB" "$HSLDIR/lib/libhsl_ma57.so" "$HOME/.local/hsl-ma57/lib/libhsl_ma57.so"; do
  [ -n "$_c" ] && [ -f "$_c" ] && { _ma57_found="$_c"; break; }
done
if [ -n "$_ma57_found" ]; then
  echo "MA57 for the monolithic reference: $_ma57_found"
else
  echo "warning: no libhsl_ma57.so found via HSLLIB / \$HSLDIR/lib / ~/.local/hsl-ma57/lib."
  echo "         The DD runs are unaffected (they link it directly); the --solver ma57"
  echo "         baseline will fail. Set HSLLIB=/path/to/libhsl_ma57.so to fix."
fi
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
  tag="${IMGTAG}_N${n}_k${k}"
  common=(--data "$IMG" --size "$n" --nsub "$k" --init cp)

  $BIN "${common[@]}" --solver ma57 \
      > "$RUN/logs/${tag}_ma57.log" 2>&1
  $BIN "${common[@]}" --solver dd \
      --save-solution "$RUN/sols/sol_${tag}.txt" --save-dd "$RUN/sols/dd_${tag}.txt" \
      > "$RUN/logs/${tag}_dd.log" 2>&1
  $BIN "${common[@]}" --solver dd --interface cg \
      > "$RUN/logs/${tag}_dd_cg.log" 2>&1
  $BIN "${common[@]}" --solver dd --interface minres \
      > "$RUN/logs/${tag}_dd_minres.log" 2>&1
  # Plot THIS solution now, not at the end of the job: a sweep at this size can
  # be killed by walltime or the OOM killer partway through, and figures for the
  # cases that did finish are worth having. --only keeps it to the one tag, so
  # nothing is re-rendered on later passes.
  MPLBACKEND=Agg python ../python/plot_slurm.py "$RUN" --out "$RUN/plots" --only "$tag" \
      >> "$RUN/logs/plots.log" 2>&1 \
    && echo "  plot k=$k  -> $RUN/plots" \
    || echo "  plot k=$k  FAILED (see logs/plots.log; sol_${tag}.txt is still on disk)"
  echo "  done $tag  ($(date +%T))"
done

# --- 3. Summary --------------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "binary: $BIN  image: $IMG  OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
  for f in "$RUN"/logs/*_N*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 15 "$f"
  done
} > "$RUN/summary.txt"

# --- 4. Plots (best-effort: needs matplotlib; the self-contained --save-solution
#         dumps in sols/ can also be plotted later with python/plot_slurm.py) ----

# keep the SLURM stdout/err with the rest (still open, cp not mv)
cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 Job finished at: $(date)   see $RUN/summary.txt"
conda deactivate
