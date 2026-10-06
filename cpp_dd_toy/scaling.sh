#!/bin/bash
# Scaling study for dd_toy.  Build first with  OMP=1 ./build.sh
#
#   ./scaling.sh tiles    N=256, tiles 4..32, one-level AS vs each coarse space
#   ./scaling.sh weak     32×32-node tiles, N = 64 … 1024, AS + linear coarse
#   ./scaling.sh strong   N=512, 16×16 tiles, 1/2/4/8 threads, AS + linear coarse
#   ./scaling.sh all
#
# Every run appends one CSV line (see the header) to $OUT (default scaling.csv).
# THREADS sets the thread count for "tiles" and "weak" (default 8).
set -e
cd "$(dirname "$0")"
OUT="${OUT:-scaling.csv}"
THREADS="${THREADS:-8}"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[ -f "$OUT" ] || echo "N,P,solver,blocks,interface,precond,coarse,inertia,threads,ipopt_iters,objective,wall,t_factor,t_blocks,t_precond,t_coarse,t_tile_solves,t_interface,cg_mean,cg_max,rss_mb" > "$OUT"

run() {   # run <threads> <args...>
  local th="$1"; shift
  echo "+ OMP_NUM_THREADS=$th ./dd_toy $*" >&2
  OMP_NUM_THREADS="$th" ./dd_toy "$@" --print-level 0 | grep '^CSV,' | cut -d, -f2- >> "$OUT"
  tail -1 "$OUT" >&2
}

PCG=(--schur pcg --precond as --obj-scaling auto)

tiles() {
  for P in 4 8 16 32; do
    run "$THREADS" --N 256 --tiles $P "${PCG[@]}"
    for c in faces sides linear; do run "$THREADS" --N 256 --tiles $P "${PCG[@]}" --coarse $c; done
  done
}

weak() {
  for N in 64 128 256 512 1024; do
    run "$THREADS" --N $N --tiles $((N / 32)) "${PCG[@]}" --coarse linear
  done
}

strong() {
  for th in 1 2 4 8; do run "$th" --N 512 --tiles 16 "${PCG[@]}" --coarse linear; done
}

case "${1:-all}" in
  tiles) tiles ;;
  weak) weak ;;
  strong) strong ;;
  all) tiles; weak; strong ;;
  *) echo "usage: $0 tiles|weak|strong|all" >&2; exit 1 ;;
esac
echo "results in $OUT" >&2
