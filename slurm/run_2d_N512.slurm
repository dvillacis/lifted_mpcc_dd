#!/bin/bash
# 2D staggered DD at N=512 — RUNS PREBUILT BINARIES ONLY (build first with
# ./build_all_ma57_omp.sh from cpp/, and validate once per machine per that
# script's header).
#
#   sbatch slurm/run_2d_N512.slurm              # from anywhere in the checkout
#   PILOT=1  sbatch slurm/run_2d_N512.slurm     # 12 Newton steps, sizing probe
#   KS="16"  sbatch slurm/run_2d_N512.slurm     # one nsub instead of the bracket
#   BARRIER_TOL=2000     sbatch ...             # raise the mu-advance gate (DD_BARRIER_TOL)
#   MU_STRATEGY=adaptive sbatch ...             # gate-free mu oracle (DD_MU_STRATEGY)
#
# DD ONLY (2026-08-20): the monolithic ma57 baseline was removed from this
# script — at this size it is a separate, memory-heavy experiment that deserves
# its own job if ever wanted. This script measures the DD route.
#
# BARRIER ADVANCE. Monotone mu only cuts once E_mu <= f*mu, and the gate is a
# max-norm over an O(N^2) degenerate set, so the stall scales with N (measured:
# N=1024 plateaued 62x above the default f=10 gate for 200+ its; the N=128
# A/B/C in cpp/README.md: 610 its at f=10 vs 131 at f=1000, same solution).
# Default here is the measured fix f=1000. If a plateau still appears (inf_du
# flat at fixed lg(mu) with inf_pr tiny), the log names the needed value:
# f ~ 3x max(inf_du/mu) over the plateau. MU_STRATEGY=adaptive removes the
# gate entirely (measured: same answer, rougher path).
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
# CHECK IT against the node spec.
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
BARRIER_TOL="${BARRIER_TOL:-1000}"              # mu-advance gate f (DD_BARRIER_TOL); 10 = IPOPT default
MU_STRATEGY="${MU_STRATEGY:-}"                  # empty = monotone (default); "adaptive" = gate-free
export DD_BARRIER_TOL="$BARRIER_TOL"
[ -n "$MU_STRATEGY" ] && export DD_MU_STRATEGY="$MU_STRATEGY"
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
# Numeric output must be locale-independent. On a machine with a comma decimal
# separator (es_*, de_*, fr_* ...) awk/printf emit "1,32" — which silently turns
# every timings.csv row into one with an extra field, and makes the solver's own
# tables unparseable. C locale for the whole job, so every number logged here is
# machine-readable wherever it runs.
export LC_ALL=C
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

# --- timing ------------------------------------------------------------------
# $SECONDS is integer and wall-only. At this size three numbers matter: WALL
# (what you wait), CPU (with 44 threads this is far larger — their ratio is the
# achieved parallelism), and PEAK RSS (what decides whether the next size fits
# at all, and the first thing to check when a job dies silently). GNU time -v
# supplies CPU and RSS; wall is taken here so it stays exact even without it.
_now() {   # seconds with sub-second resolution where date supports %N
  local t; t=$(date +%s.%N 2>/dev/null)
  case "$t" in *N*|"") date +%s;; *) printf '%s' "$t";; esac
}
HAVE_GNUTIME=0
/usr/bin/time -v true >/dev/null 2>&1 && HAVE_GNUTIME=1
[ "$HAVE_GNUTIME" = "1" ] || echo "note: GNU time -v unavailable — wall clock only, no CPU/RSS"

run_timed() {   # run_timed <tag> <label> <logfile> -- cmd...
  local tag="$1" lbl="$2" log="$3"; shift 3; [ "$1" = "--" ] && shift
  local t0 t1 wall cpu rss rc tf us sy
  tf="$(mktemp)"; t0=$(_now)
  if [ "$HAVE_GNUTIME" = "1" ]; then
    /usr/bin/time -v -o "$tf" "$@" >> "$log" 2>&1; rc=$?
  else
    "$@" >> "$log" 2>&1; rc=$?
  fi
  t1=$(_now)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  cpu=NA; rss=NA
  if [ -s "$tf" ]; then
    us=$(awk -F': ' '/User time/{print $2}' "$tf")
    sy=$(awk -F': ' '/System time/{print $2}' "$tf")
    cpu=$(awk -v u="$us" -v s="$sy" 'BEGIN{printf "%.2f", u+s}')
    rss=$(awk -F': ' '/Maximum resident set size/{printf "%.0f", $2/1024}' "$tf")
    { echo; echo "--- GNU time -v ---"; cat "$tf"; } >> "$log"
  fi
  rm -f "$tf"
  printf '%s,%s,%s,%s,%s,%s\n' "$tag" "$lbl" "$wall" "$cpu" "$rss" "$rc" >> "$TIMINGS"
  echo "[timing] $tag $lbl  wall ${wall}s  cpu ${cpu}s  maxRSS ${rss}MB  exit $rc" >> "$log"
  echo "  $lbl k=${k:-?}  wall ${wall}s  cpu ${cpu}s  maxRSS ${rss}MB  exit $rc"
  return $rc
}

RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots"
TIMINGS="$RUN/timings.csv"
echo "tag,solver,wall_s,cpu_s,maxrss_MB,exit" > "$TIMINGS"
echo "🚀 N=$N  nsub in [$KS]  threads=$OMP_NUM_THREADS  pilot=$PILOT  barrier_tol=$DD_BARRIER_TOL  mu=${DD_MU_STRATEGY:-monotone}  $(date)"
echo "   results -> $RUN"

# --- 3. Runs -----------------------------------------------------------------
for k in $KS; do
  tag="${IMGTAG}_N${N}_k${k}"
  common=(--data "$IMG" --size "$N" --nsub "$k" --init cp "${EXTRA[@]}")

  run_timed "$tag" dd "$RUN/logs/${tag}_dd.log" -- \
      $BIN "${common[@]}" --solver dd --save-solution "$RUN/sols/sol_${tag}.txt"
  # Plot THIS solution now, not at the end of the job: a sweep at this size can
  # be killed by walltime or the OOM killer partway through, and figures for the
  # cases that did finish are worth having. --only keeps it to the one tag, so
  # nothing is re-rendered on later passes.
  MPLBACKEND=Agg python ../python/plot_slurm.py "$RUN" --out "$RUN/plots" --only "$tag" \
      >> "$RUN/logs/plots.log" 2>&1 \
    && echo "  plot k=$k  -> $RUN/plots" \
    || echo "  plot k=$k  FAILED (see logs/plots.log; sol_${tag}.txt is still on disk)"
done

# --- 4. Summary + plots ------------------------------------------------------
{
  echo "job $SLURM_JOB_ID  node $(hostname)  $(date)"
  echo "N=$N  KS=$KS  threads=$OMP_NUM_THREADS  pilot=$PILOT  image=$IMG"
  echo "barrier_tol_factor=$DD_BARRIER_TOL  mu_strategy=${DD_MU_STRATEGY:-monotone}"
  for f in "$RUN"/logs/*_N*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 20 "$f"
  done
  echo
  echo "--- timings (wall/cpu seconds, peak RSS MB) ---"
  column -t -s, "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
} > "$RUN/summary.txt"

cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null

echo "🏁 finished $(date)   see $RUN/summary.txt"
conda deactivate
