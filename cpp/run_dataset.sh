#!/bin/bash
# Learn the TV weight on a prepared set images/<SET>/ (train.txt, val.txt) with
# tv_learn, then draw the dataset figures.  Sets: kodak (train kodim01-16, held
# out 17-24), and those of python/make_datasets.py (bsds64, cartoon64, ...).
#
#   ./run_dataset.sh SET SIZE [extra tv_learn options]
#   ./run_dataset.sh kodak 64
#   ./run_dataset.sh bsds64 64 --limit 64 --val-limit 32
#   ./run_dataset.sh cartoon64 64 --limit 128 --block 8
#   NP=4 NSUB=4 ./run_dataset.sh kodak 96 --op deblur --blur-sigma 1 --sigma 0.02
#
# NP: MPI ranks (default 8), NSUB: tiles per side per image (default 1: the images
# couple only through alpha; more for large images), TAG:
# run name (default <SET>_N<SIZE>_k<NSUB> plus the extra options, e.g.
# kodak_N128_k1_block_16).
# Output: ../results/tvlearn_<TAG>/ with the log, reports/report_<TAG>.json,
# sols/sol_<TAG>_s000.npz ... and fig/.  An existing run is not overwritten
# unless FORCE=1.
set -eo pipefail

if [ $# -lt 2 ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
SET=$1
SIZE=$2
shift 2
NP=${NP:-8}
NSUB=${NSUB:-1}
if [ -z "${TAG:-}" ]; then   # kodak_N<SIZE>_k<NSUB>, then the extra options: --block 16 → _block_16
  TAG=${SET}_N${SIZE}_k$NSUB
  for a in "$@"; do TAG="${TAG}_$(echo "${a#--}" | tr -c 'A-Za-z0-9.\n-' '-')"; done
fi

cd "$(dirname "$0")"
[ -x ./tv_learn ] || { echo "no ./tv_learn: TARGETS=tv_learn MPI=1 ./build.sh" >&2; exit 1; }
[ -f "../images/$SET/train.txt" ] || { echo "no ../images/$SET/train.txt (python/make_datasets.py)" >&2; exit 1; }
R=../results/tvlearn_$TAG
if [ -e "$R" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "$R exists: set TAG=<name> for a new run, or FORCE=1 to overwrite it" >&2
  exit 1
fi
mkdir -p "$R/reports" "$R/sols"

# every MPI rank needs a tile: at most S·NSUB² ranks (S = training images, after --limit)
S=$(grep -v -c -E '^[[:space:]]*(#|$)' "../images/$SET/train.txt" || true)
prev=
for a in "$@"; do [ "$prev" = --limit ] && [ "$a" -gt 0 ] && [ "$a" -lt "$S" ] && S=$a; prev=$a; done
if [ "$NP" -gt $((S * NSUB * NSUB)) ]; then
  echo "NP=$NP > $((S * NSUB * NSUB)) tiles ($S images x ${NSUB}x$NSUB): using NP=$((S * NSUB * NSUB))"
  NP=$((S * NSUB * NSUB))
fi
VAL=()
[ -f "../images/$SET/val.txt" ] && VAL=(--val "../images/$SET/val.txt")
ARGS=(--train "../images/$SET/train.txt" "${VAL[@]}"
      --size "$SIZE" --nsub "$NSUB" --tag "$TAG"
      --save-report "$R/reports/report_$TAG.json" --save-solution "$R/sols/sol_$TAG" "$@")
echo "mpirun -np $NP ./tv_learn ${ARGS[*]}"
echo "log: $R/$TAG.log"

OMP_NUM_THREADS=1 mpirun -np "$NP" ./tv_learn "${ARGS[@]}" 2>&1 | tee "$R/$TAG.log"

(cd ../python && uv run python plot_dataset.py --run "../results/tvlearn_$TAG/reports/report_$TAG.json" \
    --out "../results/tvlearn_$TAG/fig" --max-pairs 8)
echo "results in $(cd "$R" && pwd)"
