#!/bin/bash
# DATASET sweep — one weight alpha learned across S training pairs, decomposed
# PER PAIR. RUNS PREBUILT BINARIES ONLY (compile beforehand with
# ./build_all_ma57_omp.sh; validate once per machine per that script's header).
#
# For each training-set size S the sweep runs the monolithic IPOPT-ma57
# reference and the DD arrowhead. At --nsub 1 the DD has ONE subdomain per
# training pair and its interface is exactly {alpha}, p=1: no cut cells, no
# corner promotion, no dual peel. --nsub k>1 additionally tiles each pair.
#
#   DATA=/path/to/kodak sbatch run_dataset.slurm         # from cpp/ on the cluster
#   DATA=... SS="4 8 16" sbatch run_dataset.slurm
#   DATA=... REPO=/path/to/lifted-mpcc-dd sbatch run_dataset.slurm
#
# TWO THINGS THIS SCRIPT DOES THAT run_2d.slurm DOES NOT:
#
#   1. DATA= IS REQUIRED. The package bundles two images, which is not a
#      dataset. Without DATA= this falls back to ../images and runs a 2-pair
#      SMOKE TEST — enough to prove the pipeline, useless as a result — and says
#      so loudly rather than quietly producing a "sweep" over two pictures.
#
#   2. DD_BARRIER_TOL=1000 BY DEFAULT. Measured 2026-08-05 on Kodak N=32 S=8:
#      under IPOPT's default monotone barrier gate BOTH solvers hit max_iter
#      3000 without converging — the monolithic one with alpha pinned at its
#      starting value w0 = 0.7*sigma for all 3000 iterations. Raising the gate
#      turns that into 73 converged iterations, a 41x reduction, and both
#      solvers then agree on alpha* to 6 significant figures. A sweep run
#      WITHOUT this knob measures the stall, not the solver. Set
#      DD_BARRIER_TOL= (empty) to deliberately reproduce the stall.
#
#SBATCH --job-name=mpcc_dataset
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
# Repo root = the directory holding cpp/ and images/. Explicit REPO= always
# wins; otherwise walk UP from the submit directory looking for that pair, so
# `sbatch` works from the repo root, from cpp/, or from slurm/.
if [ -z "${REPO:-}" ]; then
  _d="${SLURM_SUBMIT_DIR:-$PWD}"
  while [ "$_d" != "/" ] && [ -n "$_d" ]; do
    if [ -d "$_d/cpp" ] && [ -d "$_d/images" ]; then REPO="$_d"; break; fi
    _d="$(dirname "$_d")"
  done
fi
REPO="${REPO:-$HOME/lifted-mpcc-dd}"           # last-resort default
BIN=./dd_solve_dataset                          # prebuilt (build_all_ma57_omp.sh)

N="${N:-32}"                                    # image side length
SIGMA="${SIGMA:-0.1}"
SEED="${SEED:-0}"
SS=(${SS:-2 4 6 8 12})                          # training-set sizes to sweep
KS=(${KS:-1})                                   # nsub per pair; 1 => p=1
VAL="${VAL:-6}"                                 # held-out pairs per run
SOLVERS=(${SOLVERS:-ma57 dd})

# --- 1. Environment ----------------------------------------------------------
source /opt/ohpc/pub/apps/conda/etc/profile.d/conda.sh
conda activate mkl_imaging
# Numeric output must be locale-independent: on a comma-decimal machine awk and
# printf emit "1,32", which silently gives every timings.csv row an extra field.
export LC_ALL=C
THREADS="${THREADS:-$SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=$THREADS
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# See note 2 in the header. Exported (not passed per-run) so it applies to the
# monolithic reference too — it is the solver that stalls worst without it.
if [ -z "${DD_BARRIER_TOL+x}" ]; then export DD_BARRIER_TOL=1000; fi
[ -n "${DD_BARRIER_TOL:-}" ] \
  && echo "barrier gate: DD_BARRIER_TOL=$DD_BARRIER_TOL" \
  || echo "barrier gate: DEFAULT (expect stalls at S>=8 — this is deliberate)"

cd "$REPO/cpp" || {
  echo "error: no cpp/ under REPO=$REPO"
  echo "       searched upward from ${SLURM_SUBMIT_DIR:-$PWD} for a dir holding cpp/ and images/"
  echo "       fix: REPO=/path/to/checkout sbatch $0"
  exit 1
}
[ -x "$BIN" ] || { echo "error: $BIN missing — build first: ./build_all_ma57_omp.sh"; exit 1; }

# The dataset. There is no bundled one; see note 1 in the header.
SMOKE=0
if [ -z "${DATA:-}" ]; then
  DATA=../images
  SMOKE=1
  echo "############################################################"
  echo "#  WARNING: DATA= not set.                                 #"
  echo "#  Falling back to the 2 bundled images. This is a SMOKE    #"
  echo "#  TEST of the pipeline, NOT a dataset result — do not      #"
  echo "#  report numbers from it.                                  #"
  echo "#      DATA=/path/to/kodak sbatch run_dataset.slurm         #"
  echo "############################################################"
  SS=(2); VAL=0
fi
[ -e "$DATA" ] || { echo "error: DATA=$DATA does not exist"; exit 1; }
DATATAG="$(basename "$(cd "$DATA" 2>/dev/null && pwd || echo "$DATA")")"
DATATAG="${DATATAG%.*}"

# IPOPT's OWN --solver ma57 reference dlopen()s MA57 at RUNTIME via its hsllib
# option, unlike the DD runs which link it at build time through an rpath — so
# the reference can fail on a node where the DD runs are fine. Warn rather than
# abort: a sweep whose DD numbers are good but whose baseline is missing is
# still worth having.
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

# --- timing ------------------------------------------------------------------
# WALL (what you wait), CPU (with 44 threads far larger — their ratio is the
# achieved parallelism), and PEAK RSS (what decides whether the next S fits, and
# the first thing to check when a job dies silently).
_now() {
  local t; t=$(date +%s.%N 2>/dev/null)
  case "$t" in *N*|"") date +%s;; *) printf '%s' "$t";; esac
}
HAVE_GNUTIME=0
/usr/bin/time -v true >/dev/null 2>&1 && HAVE_GNUTIME=1
[ "$HAVE_GNUTIME" = "1" ] || echo "note: GNU time -v unavailable — wall clock only, no CPU/RSS"

# Interpreter for the plotting/aggregation steps. Inside the conda env `python`
# is the right name; outside it (a bare login shell, a dry run on a laptop) only
# `python3` may exist, and hardcoding `python` turns every figure step into a
# silent "command not found" in plots.log. Resolve once, and treat its absence
# as a warning — the reports and solution dumps are the results, the figures are
# regenerable from them at any time.
PY="${PY:-}"
if [ -z "$PY" ]; then
  for _p in python python3; do
    command -v "$_p" >/dev/null 2>&1 && { PY="$_p"; break; }
  done
fi
[ -n "$PY" ] && echo "python for figures: $(command -v "$PY")" \
  || echo "warning: no python on PATH — figures/aggregation will be SKIPPED (the
         reports/ and sols/ dumps are still written; run
         python/plot_dataset.py --sweep <resultdir> afterwards)"

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
  echo "  $lbl S=${S:-?}  wall ${wall}s  cpu ${cpu}s  maxRSS ${rss}MB  exit $rc"
  return $rc
}

# Results go to the REPO ROOT, not into cpp/ — cpp/ is source, and a sweep
# writes hundreds of MB of dumps and figures.
RUN="$(cd .. && pwd)/results/slurm_${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/logs" "$RUN/sols" "$RUN/plots" "$RUN/reports"
TIMINGS="$RUN/timings.csv"
echo "tag,solver,wall_s,cpu_s,maxrss_MB,exit" > "$TIMINGS"
echo "🚀 Job started at: $(date)   results -> $RUN"
echo "   data=$DATA  N=$N  sigma=$SIGMA  S in [${SS[*]}]  nsub in [${KS[*]}]  val=$VAL"

# --- 2. Runs -----------------------------------------------------------------
for S in "${SS[@]}"; do
  for k in "${KS[@]}"; do
    for sv in "${SOLVERS[@]}"; do
      # The label distinguishes the DD series by tiling, since --nsub 1 (p=1)
      # and --nsub k are different decompositions of the same problem and must
      # not share a line in the scaling figures.
      lbl="$sv"
      [ "$sv" = "dd" ] && [ "$k" != "1" ] && lbl="dd_tiles"
      tag="${DATATAG}_N${N}_S${S}_${lbl}"
      common=(--data "$DATA" --size "$N" --limit "$S" --val-limit "$VAL"
              --sigma "$SIGMA" --seed "$SEED" --nsub "$k")

      run_timed "$tag" "$lbl" "$RUN/logs/${tag}.log" -- \
        $BIN "${common[@]}" --solver "$sv" \
          --save-report   "$RUN/reports/report_${tag}.json" \
          --save-manifest "$RUN/reports/manifest_${tag}.csv" \
          --save-solution "$RUN/sols/sol_${tag}" \
          --save-val-solution "$RUN/sols/sol_${tag}"

      # Plot THIS run now, not at the end of the job: a sweep can be killed by
      # walltime or the OOM killer partway through, and figures for the runs
      # that did finish are worth having.
      if [ -n "$PY" ] && [ -f "$RUN/reports/report_${tag}.json" ]; then
        MPLBACKEND=Agg $PY ../python/plot_dataset.py \
            --run "$RUN/reports/report_${tag}.json" --out "$RUN/plots" \
            --sols "$RUN/sols" --format both >> "$RUN/logs/plots.log" 2>&1 \
          && echo "  plot $tag -> $RUN/plots" \
          || echo "  plot $tag FAILED (see logs/plots.log; the report and sols are still on disk)"
      fi
      echo "  done $tag  ($(date +%T))"
    done
  done
done

# --- 3. Cross-run figures + table --------------------------------------------
if [ -n "$PY" ]; then
  MPLBACKEND=Agg $PY ../python/plot_dataset.py --sweep "$RUN" \
      --out "$RUN/plots" --format both >> "$RUN/logs/plots.log" 2>&1 \
    && echo "sweep figures -> $RUN/plots (scaling, dd_vs_mono, tab_dataset.tex)" \
    || echo "sweep figures FAILED (see logs/plots.log)"
  $PY ../python/aggregate_runs.py "$RUN" -o "$RUN/runs.csv" \
      >> "$RUN/logs/plots.log" 2>&1 \
    && echo "aggregate -> $RUN/runs.csv" \
    || echo "aggregate FAILED (see logs/plots.log)"
else
  echo "figures/aggregation skipped (no python); regenerate later with:"
  echo "  python ../python/plot_dataset.py --sweep $RUN --format both"
fi

# --- 4. Summary --------------------------------------------------------------
{
  echo "job ${SLURM_JOB_ID:-local}  node $(hostname)  $(date)"
  echo "binary: $BIN  data: $DATA  N=$N sigma=$SIGMA seed=$SEED val=$VAL"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_THREADING_LAYER=$MKL_THREADING_LAYER"
  echo "DD_BARRIER_TOL=${DD_BARRIER_TOL:-<default>}"
  [ "$SMOKE" = "1" ] && echo "*** SMOKE TEST (no DATA=): 2 bundled images, not a dataset result ***"
  for f in "$RUN"/logs/*_S*.log; do
    echo; echo "===== $(basename "$f") ====="; tail -n 20 "$f"
  done
  echo
  echo "--- timings (wall/cpu seconds, peak RSS MB) ---"
  column -t -s, "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
} > "$RUN/summary.txt"

# keep the SLURM stdout/err with the rest (still open, cp not mv)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cp "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN/logs/" 2>/dev/null
fi

echo "🏁 Job finished at: $(date)   see $RUN/summary.txt"
conda deactivate
