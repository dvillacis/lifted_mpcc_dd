#!/bin/bash
# 2D staggered DD at N=512 — RUNS PREBUILT BINARIES ONLY (build first with
# ./build_all_ma57_omp.sh from cpp/, and validate once per machine per that
# script's header).
#
#   sbatch slurm/run_2d_N512.slurm              # from anywhere in the checkout
#   PILOT=1  sbatch slurm/run_2d_N512.slurm     # 12 Newton steps, sizing probe
#   KS="16"  sbatch slurm/run_2d_N512.slurm     # one nsub instead of the bracket
#   NOREF=1  sbatch slurm/run_2d_N512.slurm     # skip the monolithic baseline
#
# WHY N=512 IS THE CHEAP ONE: images/cameraman.png is natively 512x512, so this
# runs at the source resolution with NO resampling. Every smaller size in the
# older sweeps was a downsample of this same image.
#
# SIZING (from the measured N-sweep, extrapolated — see below). dim ~ 4.3M,
# and per factorization at 44 threads the model gives, in seconds:
#
#     nsub    dim_k       p    factor  schur  S-fact  solve   total
#       12    30037   22528      0.13   1.71    0.41   1.63    3.88
#       16    16896   30720      0.08   0.96    0.66   1.63    3.33   <- flat optimum
#       24     7509   47104      0.06   0.55    1.25   1.63    3.48
#
# The optimum is FLAT between 12 and 32: more subdomains shrink the local work
# but grow the interface, and S-fact ~ p^1.5 with p = 4(nsub-1)N takes over. The
# default KS brackets it so the real optimum is MEASURED, not trusted — these
# models extrapolate ~5x beyond anything measured and set the configuration, not
# a prediction. Run PILOT=1 first.
#
# MEMORY. The driver is the interface matrix: nnz(S) ~ 60 N^2, i.e. ~16M entries
# here, INDEPENDENT of nsub (per-tile cliques shrink exactly as their count
# grows). Its factor is a multiple of that. 64G is a guess with headroom —
# CHECK IT against the node spec, and against the monolithic baseline, which
# needs far more than the DD route by construction.
#
#SBATCH --job-name=mpcc_dd2d_N512
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --time=24:00:00

set -o pipefail

# --- 0. Configuration --------------------------------------------------------
N=512
KS="${KS:-12 16 24}"                            # nsub bracket around the modelled optimum
BIN=./dd_solve_2d
# (IMG/IMGTAG resolved below, after the repo root is known)
PILOT="${PILOT:-0}"                             # 1 = short run to validate the sizing
NOREF="${NOREF:-0}"                             # 1 = skip the monolithic ma57 baseline
EXTRA=()
[ "$PILOT" = "1" ] && EXTRA+=(--max-iter 12)

# --- 1. Repo root ------------------------------------------------------------
if [ -z "${REPO:-}" ]; then
  _d="${SLURM_SUBMIT_DIR:-$PWD}"
  while [ "$_d" != "/" ] && [ -n "$_d" ]; do
    if [ -d "$_d/cpp" ] && [ -d "$_d/images" ]; then REPO="$_d"; break; fi
    _d="$(dirname "$_d")"
  done
fi
REPO="${REPO:-$HOME/lifted-mpcc-dd}"

# --- 2. Environment ----------------------------------------------------------
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
# One task per subdomain, so the thread count wants to divide nsub^2; with the
# bracket above (144/256/576 subdomains) 44 is fine — all are >> 44, so the
# tail wave is a small fraction. THREADS= overrides without touching SBATCH.
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=$THREADS
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_PROC_BIND=close
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
[ -f "$IMG" ] || { echo "error: $IMG missing"; exit 1; }

# Source must be at least N x N or the loader upsamples and the "512" is fake.
python - "$IMG" "$N" <<'EOF' || exit 1
import struct, sys
p, n = sys.argv[1], int(sys.argv[2])
w, h = struct.unpack('>II', open(p, 'rb').read(24)[16:24])
if min(w, h) < n:
    print(f"error: {p} is {w}x{h}, smaller than --size {n};")
    print(f"       the loader would UPSAMPLE (bilinear) and invent no detail.")
    print(f"       Supply a >= {n}x{n} source via IMG=/path/to/image.png")
    sys.exit(1)
print(f"source image {p}: {w}x{h}  -> N={n} "
      f"({'native, no resampling' if min(w,h)==n else 'area-downsampled'})")
EOF

# IPOPT's own --solver ma57 dlopen()s MA57 at runtime (the DD binaries link it
# via rpath and are unaffected). Warn, don't abort.
if [ "$NOREF" != "1" ]; then
  _found=""
  for _c in "$HSLLIB" "$HSLDIR/lib/libhsl_ma57.so" "$HOME/.local/hsl-ma57/lib/libhsl_ma57.so"; do
    [ -n "$_c" ] && [ -f "$_c" ] && { _found="$_c"; break; }
  done
  [ -n "$_found" ] && echo "MA57 for the monolithic baseline: $_found" \
    || echo "warning: no libhsl_ma57.so found — the --solver ma57 baseline will fail (DD runs unaffected)"
fi

RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots"
echo "🚀 N=$N  nsub in [$KS]  threads=$OMP_NUM_THREADS  pilot=$PILOT  $(date)"
echo "   results -> $RUN"

# --- 3. Runs -----------------------------------------------------------------
for k in $KS; do
  tag="${IMGTAG}_N${N}_k${k}"
  common=(--data "$IMG" --size "$N" --nsub "$k" --init cp "${EXTRA[@]}")

  t0=$SECONDS
  $BIN "${common[@]}" --solver dd \
      --save-solution "$RUN/sols/sol_${tag}.txt" \
      > "$RUN/logs/${tag}_dd.log" 2>&1
  echo "  dd   k=$k  $((SECONDS-t0))s  ($(date +%T))"
  # Plot THIS solution now, not at the end of the job: a sweep at this size can
  # be killed by walltime or the OOM killer partway through, and figures for the
  # cases that did finish are worth having. --only keeps it to the one tag, so
  # nothing is re-rendered on later passes.
  MPLBACKEND=Agg python ../python/plot_slurm.py "$RUN" --out "$RUN/plots" --only "$tag" \
      >> "$RUN/logs/plots.log" 2>&1 \
    && echo "  plot k=$k  -> $RUN/plots" \
    || echo "  plot k=$k  FAILED (see logs/plots.log; sol_${tag}.txt is still on disk)"

  if [ "$NOREF" != "1" ]; then
    t0=$SECONDS
    $BIN "${common[@]}" --solver ma57 > "$RUN/logs/${tag}_ma57.log" 2>&1
    echo "  ma57 k=$k  $((SECONDS-t0))s"
    break   # the baseline does not depend on nsub — run it once
  fi
done

# --- 4. Summary + plots ------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "N=$N  KS=$KS  threads=$OMP_NUM_THREADS  pilot=$PILOT  image=$IMG"
  for f in "$RUN"/logs/*_N*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 20 "$f"
  done
} > "$RUN/summary.txt"

cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 finished $(date)   see $RUN/summary.txt"
conda deactivate
