#!/bin/bash
# 2D staggered DD at N=1024 — the largest instance in this package, and the one
# where the DD route is supposed to earn its keep. RUNS PREBUILT BINARIES ONLY
# (./build_all_ma57_omp.sh from cpp/ first).
#
#   PILOT=1 sbatch slurm/run_2d_N1024.slurm     # DO THIS FIRST — 12 Newton steps
#   IMG=/path/to/big.png sbatch slurm/run_2d_N1024.slurm
#   KS="32" NOREF=1 sbatch slurm/run_2d_N1024.slurm
#
# READ BEFORE SUBMITTING — three things differ from the smaller sweeps.
#
# 1. YOU NEED A >= 1024x1024 SOURCE IMAGE. images/cameraman.png is 512x512, so
#    the loader would bilinearly UPSAMPLE it: a 1024 grid carrying 512 worth of
#    detail, which is a smoother denoising problem than it looks and not
#    comparable with the N<=512 numbers. The guard below refuses that by
#    default. IMG= a real high-resolution image, or set ALLOW_UPSAMPLE=1 if a
#    synthetic-but-larger instance is genuinely what you want.
#
# 2. THE MONOLITHIC BASELINE IS OFF BY DEFAULT (NOREF=1 is the default here,
#    the opposite of the smaller scripts). dim ~ 17.3M: a monolithic MA57
#    factorization of that is exactly the fill/memory wall this package exists
#    to route around, so it is likely to OOM or to eat the entire allocation
#    before the DD runs get a turn. Set NOREF=0 to attempt it deliberately —
#    that failure IS a result worth recording, but record it in its own job.
#
# 3. SIZING (measured N-sweep, extrapolated ~10x — sets configuration, not
#    prediction). dim ~ 17.3M; per factorization at 44 threads, seconds:
#
#      nsub    dim_k        p   factor  schur  S-fact  solve   total
#        16    67584    61440     0.67  10.84    1.86   6.51   19.87
#        24    30037    94208     0.46   6.25    3.53   6.51   16.75
#        32    16896   126976     0.34   3.96    5.52   6.51   16.32  <- flat optimum
#        40    10813   159744     0.27   2.81    7.79   6.51   17.37
#
#    Flat from 24 to 40, and note S-fact overtaking schur across that range —
#    the serial interface starting to dominate is precisely the large-image wall
#    documented in cpp/README.md. The default KS brackets the optimum so it gets
#    MEASURED.
#
# MEMORY. nnz(S) ~ 60 N^2 ~ 63M entries, INDEPENDENT of nsub, and its factor is a
# multiple of that. 256G is a guess with headroom; CHECK IT against the node.
# If the job dies with no output, suspect the OOM killer first and halve nsub^2
# or raise --mem before suspecting the solver.
#
#SBATCH --job-name=mpcc_dd2d_N1024
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=256G
#SBATCH --partition=normal
#SBATCH --time=48:00:00

set -o pipefail

# --- 0. Configuration --------------------------------------------------------
N=1024
KS="${KS:-24 32 40}"
BIN=./dd_solve_2d
# (IMG/IMGTAG resolved below, after the repo root is known)
PILOT="${PILOT:-0}"
NOREF="${NOREF:-1}"                             # default OFF at this size — see note 2
ALLOW_UPSAMPLE="${ALLOW_UPSAMPLE:-0}"
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

python - "$IMG" "$N" "$ALLOW_UPSAMPLE" <<'EOF' || exit 1
import struct, sys
p, n, allow = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
w, h = struct.unpack('>II', open(p, 'rb').read(24)[16:24])
if min(w, h) < n and not allow:
    print(f"error: {p} is {w}x{h}, smaller than --size {n}.")
    print(f"       The loader would bilinearly UPSAMPLE — a {n} grid carrying {min(w,h)}")
    print(f"       worth of detail, not comparable with the N<={min(w,h)} results.")
    print(f"       fix: IMG=/path/to/a/>={n}px/image.png sbatch $0")
    print(f"       or:  ALLOW_UPSAMPLE=1 to proceed deliberately")
    sys.exit(1)
kind = ("native, no resampling" if min(w, h) == n
        else "area-downsampled" if min(w, h) > n else "UPSAMPLED (allowed explicitly)")
print(f"source image {p}: {w}x{h}  -> N={n}  ({kind})")
EOF

if [ "$NOREF" != "1" ]; then
  echo "NOTE: monolithic ma57 baseline ENABLED at N=$N (dim ~17M) — expect heavy"
  echo "      memory use; this is the case the DD route exists to avoid."
  _found=""
  for _c in "$HSLLIB" "$HSLDIR/lib/libhsl_ma57.so" "$HOME/.local/hsl-ma57/lib/libhsl_ma57.so"; do
    [ -n "$_c" ] && [ -f "$_c" ] && { _found="$_c"; break; }
  done
  [ -n "$_found" ] && echo "      MA57: $_found" \
    || echo "      warning: no libhsl_ma57.so found — the baseline will fail"
fi

RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots"
echo "🚀 N=$N  nsub in [$KS]  threads=$OMP_NUM_THREADS  pilot=$PILOT  noref=$NOREF  $(date)"
echo "   results -> $RUN"

# --- 3. Runs -----------------------------------------------------------------
for k in $KS; do
  tag="${IMGTAG}_N${N}_k${k}"
  common=(--data "$IMG" --size "$N" --nsub "$k" --init cp "${EXTRA[@]}")

  t0=$SECONDS
  # DD_TIME prints the phase split every 25 factorizations: at this size that is
  # the number that matters (is S-fact overtaking schur, as the model predicts?).
  DD_TIME=1 $BIN "${common[@]}" --solver dd \
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
done

if [ "$NOREF" != "1" ]; then
  k=$(echo $KS | awk '{print $1}')
  t0=$SECONDS
  $BIN --data "$IMG" --size "$N" --nsub "$k" --init cp "${EXTRA[@]}" --solver ma57 \
      > "$RUN/logs/${IMGTAG}_N${N}_ma57.log" 2>&1
  echo "  ma57 $((SECONDS-t0))s (exit $?)"
fi

# --- 4. Summary + plots ------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "N=$N  KS=$KS  threads=$OMP_NUM_THREADS  pilot=$PILOT  noref=$NOREF  image=$IMG"
  echo
  echo "--- phase split (DD_TIME), last line per run ---"
  for f in "$RUN"/logs/*_dd.log; do
    echo "$(basename "$f"): $(grep 'dd-time' "$f" | tail -1)"
  done
  for f in "$RUN"/logs/*_N*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 20 "$f"
  done
} > "$RUN/summary.txt"

cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 finished $(date)   see $RUN/summary.txt"
conda deactivate
