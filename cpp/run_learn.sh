#!/bin/bash
# One tv_learn experiment: learn the TV weight on an image (or a set of images),
# over NSUB x NSUB subdomains per image, on NP MPI ranks.
#
#   ./run_learn.sh [options] [-- extra tv_learn options]
#
#   ./run_learn.sh                                         cameraman, defaults below
#   ./run_learn.sh -i mariposa.png -n 128 -k 4 -p 8
#   ./run_learn.sh -i kodak -n 64 -k 2 -p 8 --block 16     a set in ../images/
#   ./run_learn.sh -i reno.jpg --op deblur --blur-sigma 1 --sigma 0.02
#   ./run_learn.sh -i phantom64.png --op inpaint --mask-frac 0.5 --reg dtv
#   ./run_learn.sh -c exp/deblur.conf -p 4                 a config file, then overrides
#   ./run_learn.sh -i cameraman.png -- --start uniform --verbose 2
#
# data
#   -i, --image X       an image, a directory, a LIST.txt, or a set name in ../images/
#                       (kodak, bsds64, ...: its train.txt, and val.txt if there is one)
#       --val X         held-out images (default: the set's val.txt, else none)
#       --limit L       training images used (0: all)       --val-limit L
#   -n, --size N        images resized to N x N
#       --sigma S       noise level
# subdomains and cores
#   -k, --nsub K        K x K tiles per image
#   -p, --np P          MPI ranks (cores); capped at the number of tiles
# problem
#       --op OP         denoise | inpaint | deblur
#       --blur-sigma S  (deblur)      --mask-frac F  (inpaint)
#       --reg R         tv | dtv
#       --block B       0: one scalar alpha; B: alpha constant on B x B blocks
#       --start S       smoothed | scalar | uniform | interior | ipm
#       --t-min T       --max-iter M   (empty: tv_learn's defaults)
# run
#       --tag NAME      run name (default <image>_<op>_N<size>_k<nsub>[_b<block>])
#       --out DIR       results go to DIR/tvlearn_<tag>/ (default the repo's results/)
#       --plot          figures with python/plot_dataset.py after the run
#   -c, --config FILE   a bash file setting the variables below (IMAGE=..., NP=...);
#                       flags after it override it
#       --dry-run       print the command and stop
#       --force         overwrite an existing run with the same tag
#   -h, --help
#
# Output: <out>/tvlearn_<tag>/ with <tag>.log, reports/report_<tag>.json,
# sols/sol_<tag>*.npz, summary.csv (the CSV line), params.txt and rerun.sh.
# Environment: MPIRUN_ARGS (e.g. "--oversubscribe", "--bind-to core").
set -eo pipefail

# ---- defaults: edit here, or override with a config file or flags
IMAGE=cameraman.png
VAL=
LIMIT=0
VAL_LIMIT=0
SIZE=64
SIGMA=0.1
NSUB=4
NP=4
OP=denoise
BLUR_SIGMA=1
MASK_FRAC=0.5
REG=tv
BLOCK=0
START=smoothed
T_MIN=
MAX_ITER=
TAG=
OUT=                            # empty: <repo>/results
PLOT=0
DRY=0
FORCE=0
EXTRA=()

usage() { sed -n '2,/^set -eo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "run_learn.sh: $*" >&2; exit 1; }
need() { [ -n "${2:-}" ] || die "$1 needs a value"; }

CALLER_DIR=$PWD
CPP_DIR=$(cd "$(dirname "$0")" && pwd)
IMAGES_DIR=$(cd "$CPP_DIR/../images" 2>/dev/null && pwd || true)

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    -c|--config) need "$1" "${2:-}"; [ -f "$2" ] || die "no config file $2"
                 # shellcheck disable=SC1090
                 source "$2"; shift ;;
    -i|--image|--data|--train) need "$1" "${2:-}"; IMAGE=$2; shift ;;
    --val) need "$1" "${2:-}"; VAL=$2; shift ;;
    --limit) need "$1" "${2:-}"; LIMIT=$2; shift ;;
    --val-limit) need "$1" "${2:-}"; VAL_LIMIT=$2; shift ;;
    -n|--size) need "$1" "${2:-}"; SIZE=$2; shift ;;
    --sigma) need "$1" "${2:-}"; SIGMA=$2; shift ;;
    -k|--nsub) need "$1" "${2:-}"; NSUB=$2; shift ;;
    -p|--np|--cores) need "$1" "${2:-}"; NP=$2; shift ;;
    --op) need "$1" "${2:-}"; OP=$2; shift ;;
    --blur-sigma) need "$1" "${2:-}"; BLUR_SIGMA=$2; shift ;;
    --mask-frac) need "$1" "${2:-}"; MASK_FRAC=$2; shift ;;
    --reg) need "$1" "${2:-}"; REG=$2; shift ;;
    --block) need "$1" "${2:-}"; BLOCK=$2; shift ;;
    --start) need "$1" "${2:-}"; START=$2; shift ;;
    --t-min) need "$1" "${2:-}"; T_MIN=$2; shift ;;
    --max-iter) need "$1" "${2:-}"; MAX_ITER=$2; shift ;;
    --tag) need "$1" "${2:-}"; TAG=$2; shift ;;
    --out) need "$1" "${2:-}"; OUT=$2; shift ;;
    --plot) PLOT=1 ;;
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) die "unknown option $1 (tv_learn's own options go after --; -h for help)" ;;
  esac
  shift
done

for v in SIZE NSUB NP BLOCK LIMIT VAL_LIMIT; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a non-negative integer, got '${!v}'"
done
[ "$NP" -ge 1 ] || die "NP must be >= 1"
case "$OP" in denoise|inpaint|deblur) ;; *) die "--op must be denoise, inpaint or deblur" ;; esac
case "$REG" in tv|dtv) ;; *) die "--reg must be tv or dtv" ;; esac

# ---- resolve the data: a path (from where you are), a file in ../images/, or a set name
resolve() {   # prints an absolute path, or nothing
  local p=$1
  case "$p" in /*) ;; *) [ -e "$CALLER_DIR/$p" ] && p=$CALLER_DIR/$p || p=$IMAGES_DIR/$p ;; esac
  [ -e "$p" ] || return 0
  if [ -d "$p" ]; then (cd "$p" && pwd); else echo "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; fi
}
SET_NAME=
if [ -d "$IMAGES_DIR/$IMAGE" ] && [ -f "$IMAGES_DIR/$IMAGE/train.txt" ] && [ ! -e "$CALLER_DIR/$IMAGE" ]; then
  SET_NAME=$IMAGE
  TRAIN=$IMAGES_DIR/$IMAGE/train.txt
  [ -z "$VAL" ] && [ -f "$IMAGES_DIR/$IMAGE/val.txt" ] && VAL=$IMAGES_DIR/$IMAGE/val.txt
else
  TRAIN=$(resolve "$IMAGE")
  [ -n "$TRAIN" ] || die "no image or set '$IMAGE' (looked in $CALLER_DIR and $IMAGES_DIR)"
fi
if [ -n "$VAL" ]; then
  V=$(resolve "$VAL"); [ -n "$V" ] || die "no held-out data '$VAL'"; VAL=$V
fi

# number of training images, for the rank cap (every rank needs a tile)
count_images() {
  local p=$1
  if [ -d "$p" ]; then
    find "$p" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
      -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.bmp' -o -iname '*.pgm' \) | wc -l | tr -d ' '
  elif [[ "$p" == *.txt ]]; then
    grep -v -c -E '^[[:space:]]*(#|$)' "$p" || true
  else
    echo 1
  fi
}
S=$(count_images "$TRAIN")
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$S" ] && S=$LIMIT
TILES=$((S * NSUB * NSUB))
if [ "$NP" -gt "$TILES" ]; then
  echo "note: NP=$NP > $TILES tiles ($S image(s) x ${NSUB}x$NSUB): using NP=$TILES"
  NP=$TILES
fi
CORES=$( (nproc || sysctl -n hw.physicalcpu) 2>/dev/null || echo 0)
if [ "$CORES" -gt 0 ] && [ "$NP" -gt "$CORES" ]; then
  echo "warning: NP=$NP > $CORES cores; Open MPI may need MPIRUN_ARGS=--oversubscribe" >&2
fi

# ---- tag and output directory
if [ -z "$TAG" ]; then
  if [ -n "$SET_NAME" ]; then NAME=$SET_NAME; else NAME=$(basename "$TRAIN"); NAME=${NAME%.*}; fi
  TAG=${NAME}_${OP}_N${SIZE}_k${NSUB}
  [ "$BLOCK" -gt 0 ] && TAG=${TAG}_b$BLOCK
  [ "$REG" != tv ] && TAG=${TAG}_$REG
fi
if [ -z "$OUT" ]; then OUT=$(cd "$CPP_DIR/.." && pwd)/results
else case "$OUT" in /*) ;; *) OUT=$CALLER_DIR/$OUT ;; esac
fi
cd "$CPP_DIR"
R=$OUT/tvlearn_$TAG

# ---- the tv_learn command
ARGS=(--train "$TRAIN" --size "$SIZE" --nsub "$NSUB" --sigma "$SIGMA"
      --op "$OP" --reg "$REG" --block "$BLOCK" --start "$START")
[ -n "$VAL" ] && ARGS+=(--val "$VAL")
[ "$LIMIT" -gt 0 ] && ARGS+=(--limit "$LIMIT")
[ "$VAL_LIMIT" -gt 0 ] && ARGS+=(--val-limit "$VAL_LIMIT")
[ "$OP" = deblur ] && ARGS+=(--blur-sigma "$BLUR_SIGMA")
[ "$OP" = inpaint ] && ARGS+=(--mask-frac "$MASK_FRAC")
[ -n "$T_MIN" ] && ARGS+=(--t-min "$T_MIN")
[ -n "$MAX_ITER" ] && ARGS+=(--max-iter "$MAX_ITER")
ARGS+=(--tag "$TAG" --save-report "$R/reports/report_$TAG.json" --save-solution "$R/sols/sol_$TAG")
[ ${#EXTRA[@]} -gt 0 ] && ARGS+=("${EXTRA[@]}")

# shellcheck disable=SC2206
MPIRUN_EXTRA=(${MPIRUN_ARGS:-})
if [ "$NP" -gt 1 ]; then
  RUN=(mpirun -np "$NP" "${MPIRUN_EXTRA[@]}" ./tv_learn)
else
  RUN=(./tv_learn)
fi

printf '%-10s %s\n' data "$TRAIN" held-out "${VAL:-none}" images "$S" \
  size "${SIZE}x$SIZE" tiles "${NSUB}x$NSUB per image ($TILES in all)" ranks "$NP" \
  problem "$OP, $REG, sigma $SIGMA, block $BLOCK, start $START" output "$R"
echo "command:   ${RUN[*]} ${ARGS[*]}"
[ "$DRY" = 1 ] && exit 0

# ---- checks
[ -x ./tv_learn ] || die "no $CPP_DIR/tv_learn: build it with  TARGETS=tv_learn MPI=1 ./build.sh"
if [ "$NP" -gt 1 ]; then
  command -v mpirun >/dev/null 2>&1 || die "NP=$NP needs mpirun (conda install -c conda-forge openmpi), or use -p 1"
  if command -v otool >/dev/null 2>&1; then LINKS=$(otool -L ./tv_learn); else LINKS=$(ldd ./tv_learn); fi
  echo "$LINKS" | grep -q 'libmpi[._]' \
    || die "tv_learn was built without MPI: MPI=1 TARGETS=tv_learn ./build.sh, or use -p 1"
fi
if [ -e "$R" ]; then
  [ "$FORCE" = 1 ] || die "$R exists: --tag NAME for a new run, or --force to overwrite it"
  rm -rf "$R"
fi
mkdir -p "$R/reports" "$R/sols"

# record what was run
{
  echo "date      $(date '+%Y-%m-%d %H:%M:%S')"
  echo "host      $(hostname)"
  echo "git       $(git -C "$CPP_DIR" rev-parse --short HEAD 2>/dev/null || echo none)$(git -C "$CPP_DIR" diff --quiet 2>/dev/null || echo ' (modified)')"
  for v in IMAGE TRAIN VAL LIMIT VAL_LIMIT SIZE SIGMA NSUB NP OP BLUR_SIGMA MASK_FRAC REG BLOCK START T_MIN MAX_ITER TAG; do
    printf '%-10s %s\n' "$v" "${!v}"
  done
  echo "extra     ${EXTRA[*]}"
} > "$R/params.txt"
{
  echo "#!/bin/bash"
  echo "# rerun of $TAG ($(date '+%Y-%m-%d'))"
  echo "cd $(printf '%q' "$CPP_DIR")"
  printf 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
  printf ' %q' "${RUN[@]}" "${ARGS[@]}"
  echo
} > "$R/rerun.sh"
chmod +x "$R/rerun.sh"

# ---- run (BLAS on one thread per rank)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SECONDS=0
set +e
"${RUN[@]}" "${ARGS[@]}" 2>&1 | tee "$R/$TAG.log"
STATUS=${PIPESTATUS[0]}
set -e
WALL=$SECONDS

grep '^CSV,' "$R/$TAG.log" > "$R/summary.csv" || true
echo
if [ "$STATUS" != 0 ]; then
  echo "tv_learn exited with status $STATUS after ${WALL}s; log: $R/$TAG.log" >&2
  exit "$STATUS"
fi

if [ "$PLOT" = 1 ]; then
  if command -v uv >/dev/null 2>&1; then
    (cd ../python && uv run python plot_dataset.py --run "$R/reports/report_$TAG.json" \
        --out "$R/fig" --max-pairs 8) || echo "plotting failed (the run itself is fine)" >&2
  else
    echo "--plot needs uv (python/pyproject.toml); skipped" >&2
  fi
fi
echo "done in ${WALL}s; results in $R"
