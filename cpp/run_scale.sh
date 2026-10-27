#!/usr/bin/env bash
# Run the large-instance experiments requested for the paper's short results
# section. Submit with sbatch on the cluster, or run directly from cpp/.
#
#   TV_RANKS=6 ./run_scale.sh                 # N=256 only (default)
#   TV_RANKS=6 TV_RUN_512=1 ./run_scale.sh    # opt in to N=512
#   TV_RANKS=32 TV_RUN_1024=1 sbatch run_scale.sh
#   TV_TRAIN_SET=bsds64 TV_TRAIN_LIMIT=0 ./run_scale.sh
#
# tv_dd runs mariposa at N=256, 512 and 1024 from the two-stage start, using
# t_min=1e-3, k=16 at N=256 and k=32 at N=512 by default. The
# 1024 solution is saved as an NPZ for the reconstruction figure. If
# TV_TRAIN_SET names a dataset with train.txt (e.g. bsds64), tv_learn also runs
# the full training set at TV_TRAIN_SIZE. Outputs are kept in one timestamped
# directory under results/scale_*/; no existing results are overwritten.
set -eo pipefail

CPP_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$CPP_DIR/.." && pwd)
cd "$CPP_DIR"

TV_RANKS=${TV_RANKS:-${SLURM_NTASKS:-6}}
TV_NSUB_256=${TV_NSUB_256:-16}
TV_NSUB_512=${TV_NSUB_512:-32}
TV_NSUB_1024=${TV_NSUB_1024:-32}
TV_TMIN_256=${TV_TMIN_256:-1e-3}
TV_TMIN_512=${TV_TMIN_512:-1e-3}
TV_TMIN_1024=${TV_TMIN_1024:-1e-3}
TV_RUN_512=${TV_RUN_512:-0}
TV_RUN_1024=${TV_RUN_1024:-0}
TV_TRAIN_SET=${TV_TRAIN_SET:-}
TV_TRAIN_SIZE=${TV_TRAIN_SIZE:-64}
TV_TRAIN_NSUB=${TV_TRAIN_NSUB:-1}
TV_TRAIN_LIMIT=${TV_TRAIN_LIMIT:-0}
TV_RESULTS=${TV_RESULTS:-$REPO_DIR/results}
TV_MCA=${TV_MCA:-}
STAMP=$(date '+%Y%m%d_%H%M%S')
OUT="$TV_RESULTS/scale_$STAMP"
mkdir -p "$OUT"

die() { echo "run_scale.sh: $*" >&2; exit 1; }
[[ "$TV_RANKS" =~ ^[1-9][0-9]*$ ]] || die "TV_RANKS must be positive"
for v in TV_NSUB_256 TV_NSUB_512 TV_NSUB_1024; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v must be positive"
done
[ -x ./tv_dd ] || die "missing tv_dd; build with MPI=1 ./build.sh"
command -v mpirun >/dev/null || die "mpirun not found"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPIRUN_EXTRA=()
[[ -n "$TV_MCA" ]] && read -r -a MPIRUN_EXTRA <<< "$TV_MCA"
run_dd() {
  local n=$1 nsub tmin tag
  case "$n" in
    256) nsub=$TV_NSUB_256; tmin=$TV_TMIN_256 ;;
    512) nsub=$TV_NSUB_512; tmin=$TV_TMIN_512 ;;
    1024) nsub=$TV_NSUB_1024; tmin=$TV_TMIN_1024 ;;
  esac
  tag="mariposa_N${n}_k${nsub}_tmin${tmin}"
  local args=(--data ../images/mariposa.png --size "$n" --nsub "$nsub"
    --t-min "$tmin" --start smoothed --save-solution "$OUT/$tag.npz")
  echo "[$(date '+%F %T')] tv_dd N=$n ranks=$TV_RANKS nsub=$nsub t_min=$tmin"
  mpirun -np "$TV_RANKS" "${MPIRUN_EXTRA[@]}" ./tv_dd "${args[@]}" 2>&1 | tee "$OUT/$tag.log"
  grep '^CSV,' "$OUT/$tag.log" >> "$OUT/tv_dd_summary.csv" || true
}

echo "output: $OUT"
echo "This run uses the two-stage start; N=256 is always included."
run_dd 256
if [[ "$TV_RUN_512" == 1 ]]; then run_dd 512; fi
if [[ "$TV_RUN_1024" == 1 ]]; then run_dd 1024; fi

LOGS=("$OUT"/mariposa_N*_k*_tmin*.log)
python3 ./logs_to_latex.py "${LOGS[@]}" --output "$OUT/scaling_table.tex"

if [[ -n "$TV_TRAIN_SET" ]]; then
  [ -x ./tv_learn ] || die "TV_TRAIN_SET requested but tv_learn is missing; build with TARGETS=tv_learn MPI=1 ./build.sh"
  TRAIN_FILE="$REPO_DIR/images/$TV_TRAIN_SET/train.txt"
  [ -f "$TRAIN_FILE" ] || die "no training manifest: $TRAIN_FILE"
  TAG="${TV_TRAIN_SET}_N${TV_TRAIN_SIZE}_k${TV_TRAIN_NSUB}_full"
  TRAIN_OUT="$OUT/tvlearn_$TAG"
  mkdir -p "$TRAIN_OUT/reports" "$TRAIN_OUT/sols"
  ARGS=(--train "$TRAIN_FILE" --size "$TV_TRAIN_SIZE" --nsub "$TV_TRAIN_NSUB"
    --limit "$TV_TRAIN_LIMIT" --start smoothed --tag "$TAG"
    --save-report "$TRAIN_OUT/reports/report_$TAG.json"
    --save-solution "$TRAIN_OUT/sols/sol_$TAG")
  VAL_FILE="$REPO_DIR/images/$TV_TRAIN_SET/val.txt"
  [ ! -f "$VAL_FILE" ] || ARGS+=(--val "$VAL_FILE")
  echo "[$(date '+%F %T')] tv_learn set=$TV_TRAIN_SET size=$TV_TRAIN_SIZE ranks=$TV_RANKS limit=$TV_TRAIN_LIMIT"
  mpirun -np "$TV_RANKS" "${MPIRUN_EXTRA[@]}" ./tv_learn "${ARGS[@]}" 2>&1 | tee "$TRAIN_OUT/$TAG.log"
  grep '^CSV,' "$TRAIN_OUT/$TAG.log" > "$TRAIN_OUT/summary.csv" || true
fi

echo "finished: $OUT"
