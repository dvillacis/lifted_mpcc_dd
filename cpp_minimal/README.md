# cpp_minimal

A small, readable version of `../cpp` with the algorithmic choices fixed:

| choice | value |
|---|---|
| formulation | consensus |
| linear solver | arrowhead domain decomposition, CG on the interface (`ddsimple`); `--sff-solver` swaps CG for a sparse Eigen LDLᵀ (below) |
| Hessian | exact |
| warm start | Chambolle–Pock |
| continuation | μ-coupled Scholtes, one IPOPT solve |
| CG iteration cap | 100000 |
| complementarity row | Scholtes, r(1−δ) ≤ t (`--comp fb` writes the same set through the smoothed Fischer–Burmeister function; measured in `docs/cpp_minimal_profile`, not the default) |

It is equivalent to

```
../cpp/dd_solve_2d --solver ddsimple --hessian exact --formulation consensus \
                   --cg-max-iter 100000 --init cp --data IMG --size N --nsub K \
                   [--save-solution FILE]
```

and reproduces that run **exactly** (with the default `--sff-solver cg`): same per-iteration trace (μ, t, inf_pr,
inf_du, objective at every IPOPT iteration), same CG counts, same α\*.

The table below was measured before the 2026-10-02 stencil switch (forward
differences at each cell's top-left node; previously the mirror image, see
`../cpp/README.md`). The equivalence with `dd_solve_2d` still holds after it —
cameraman N=32, 4×4 now gives 34 its, α\* = 0.070402, 26.40 dB in both.

| instance | IPOPT its | CG iterations | α\* | PSNR | wall (serial, M5 Pro) |
|---|---|---|---|---|---|
| cameraman N=32, 4×4 | 39 | 99 781 | 0.070779 | 26.34 dB | 2.2 s |
| cameraman N=32, 8×8 | 57 | 2 820 232 | 0.070782 | 26.33 dB | 94 s |
| mariposa N=32, 4×4 | 132 | 244 387 | 0.064481 | 23.39 dB | 5.5 s |

(checked with and without OpenMP; results do not depend on the thread count.)
The `--save-solution` files are byte-identical to `../cpp`'s too, in both
formats, including when IPOPT fails and the banked fallback level is written.
The α\* values agree with IPOPT's own MUMPS on the same problems (0.070845,
0.070780, 0.064477).

These numbers are from after one fix made to both solvers at once: a
factorization whose Z-column CG meets non-positive curvature (S_ff not SPD) is
now refused, so IPOPT regularizes instead of stepping on an inertia that was
one to three short of the truth (section 3 of `arrowhead.hpp`).  Before the
fix the same three runs took 41 / 126 / 191 IPOPT iterations and 102 568 /
9 739 417 / 446 241 CG iterations, and the mariposa run ended at a different
point (α\* = 0.052432).

**One deliberate difference:** here t only follows μ in IPOPT's regular mode.
`../cpp` also follows the restoration phase's own barrier, which can drive t to
t_min before the original problem gets there (mariposa N=256, 4×4). The runs
above never trigger that, so they are unchanged; runs that do will diverge from
`../cpp` from the first restoration phase whose μ falls below t/10.

Each log line also shows an `r` marker in restoration, the step sizes, the
line-search trials, IPOPT's inertia regularization, and the linear-solver work
since the previous line (factorize calls, wrong-inertia rejections, interface
solves, rejected solves, CG iterations).

## Build and run

```
./build.sh                 # serial
OMP=1 ./build.sh           # OpenMP
./tv_learn --size 32 --nsub 4
./tv_learn --data ../images/mariposa.png --size 48 --nsub 4 --sigma 0.1 --seed 0
./tv_learn --size 32 --nsub 4 --save-solution run.npz   # or run.txt
./tv_learn --data ../images/mariposa.png --size 32 --nsub 4 --comp fb
./tv_learn --size 32 --nsub 4 --sff-solver direct-all  # no CG (see below)
```

`--save-solution` writes what `../cpp` writes, so `python/plot_2d.py` and the
other readers work unchanged. A `.txt` extension gives the old text format; any
other name gives a NumPy `.npz` with named arrays: the instance (`u_clean`, `f`,
`N`, `sigma`, ...), the solution (`u`, `qx`, `qy`, `r`, `delta`, `theta`,
`alpha`, `x`), the continuation summary (`levels`) and the per-iteration
`mu_trace` (iter, μ, t, weight, max r(1−δ)).

Needs IPOPT (via `pkg-config`), Eigen and zlib. On macOS those come from Homebrew
(plus `libomp` for `OMP=1`); on Linux, activate a conda env with `ipopt` and
`eigen`. Thread count: `OMP_NUM_THREADS`.

## `--sff-solver`: CG or a direct solve on S_ff

Every system with the kept interface block S_ff goes through CG by default. The
flag replaces CG with a sparse Eigen `SimplicialLDLT` of S_ff, assembled from C
and the dense tile blocks S_k (`Arrowhead::Options::sff_direct`):

| value | Z = S_ff⁻¹ S_fP (the peel columns) | interface solve S_ff⁻¹ r_f |
|---|---|---|
| `cg` (default) | one warm-started CG solve per column | CG |
| `direct` | one multi-RHS back-solve through the LDLᵀ | CG |
| `direct-all` | one multi-RHS back-solve through the LDLᵀ | back-solve through the LDLᵀ |

The inertia argument is unchanged. S_ff must still be positive definite for
#neg(S) = #neg(T), and a non-positive LDLᵀ pivot now plays the role of CG's
pᵀAp ≤ 0: the factorization is refused and IPOPT regularizes. Z is judged by
the same 1e-2 residual bar as the CG columns.

The trade-off is structural. Assembling and factorizing S_ff is a global serial
step, which the decomposition is designed to avoid, so the direct modes are a
single-node shortcut rather than the distributed method. On one node they are
much faster, because the peel columns dominate CG mode's cost.

Measured, cameraman N=32, 4×4 tiles, serial build, M5 Pro:

| `--sff-solver` | wall | factorize (peel) | solve | CG iterations | IPOPT its | α\* |
|---|---|---|---|---|---|---|
| `cg` | 1.96 s | 0.64 s (0.55 s) | 1.29 s | 99 781 | 39 | 0.070779 |
| `direct` | 0.92 s | 0.16 s (0.07 s) | 0.74 s | 39 242 | 39 | 0.070779 |
| `direct-all` | 0.31 s | 0.16 s (0.07 s) | 0.13 s | 0 | 39 | 0.070779 |

All three take the same IPOPT path: identical iterations, refused
factorizations and wrong-inertia rejections. The OpenMP build gives the same
results. `direct` also cuts the interface CG work, because an exact Z removes
the per-column error that CG otherwise has to fight in the full-interface
residual (143 rejected interface answers under `cg`, none under `direct`).

The final report prints the split as
`linear solver wall: factorize … (peel …)  solve …`, for any mode.

The same switch exists in the full driver as
`../cpp/dd_solve_2d --solver ddsimple --sff-solver cg|direct|direct-all`. There,
with the permutation formulation, it measured 48.8 → 11.2 → 6.1 s at cameraman
N=32 4×4 and 119 → 31 → 17 s at N=64 4×4, with the same PSNR in every run.

## Reading order

| file | lines | what it is |
|---|---|---|
| `main.cpp` | 182 | the whole run, top to bottom: load, build, configure IPOPT, solve, report, save |
| `problem.hpp` | 764 | the NLP: the bilevel problem, the Chambolle–Pock start, the consensus split, IPOPT callbacks, the continuation |
| `partition.hpp` | 38 | k×k tiles |
| `image.hpp` | 70 | PNG → clean and noisy image |
| `arrowhead.hpp` | 973 | the KKT solver: LDLᵀ per tile, Schur complements, peel, CG (or direct S_ff) |
| `ipopt_bridge.hpp` | 104 | hands `arrowhead.hpp` to IPOPT as its linear solver |
| `save.hpp` | 106 | `--save-solution`: the `.npz` and `.txt` writers |
| `npz_writer.hpp` | 224 | NumPy archive writer, copied unchanged from `../cpp` |

Each file opens with a comment explaining its piece of the method; the headers
of `problem.hpp` and `arrowhead.hpp` are the two to read first.

## How one run flows

```
image.hpp      clean, noisy
partition.hpp  cell → tile, node → tile
problem.hpp    Chambolle–Pock on the unsplit problem  → x_start, θ_ref
               consensus split: copy shared variables, add linking rows
               kkt_owner(): every KKT unknown → its tile, or −1 (border)
main.cpp       IPOPT options, μ-coupled t, OptimizeNLP
  └─ every IPOPT iteration
       problem.hpp  eval_f / eval_g / eval_jac_g / eval_h
                    intermediate_callback: level gate, t ← max(t_min, 10μ)
       ipopt_bridge.hpp → arrowhead.hpp
         factorize(): LDLᵀ(W_k) → S_k → C, diag(S) → preconditioner → Z, T, #neg(T)
         solve():     tiles → interface (CG or LDLᵀ on S_ff, + T) → tiles, refined
```

## What was left out, and why that is safe

Everything removed is either unreachable with the choices above or pure
telemetry; the identical traces are the check.

- **Other formulations and options**: the permutation formulation (and with it
  corner-dual promotion, the dual peel and `--border-reg`), limited-memory
  Hessian, exponential weight, averaged stencil, strip partitions, geometric
  continuation, `.txt` instances, the MA57/MUMPS solvers.
- **Experiments that default off**: the `DDS_COARSE` coarse space, every
  `DD_*`/`DDS_*` environment knob.
- **Telemetry**: the warnings system, most counters, debug and self-check output,
  the `--save-dd` and `--save-data` dumps.

One behaviour differs from `../cpp`: **downsampling only.** `--size` must be
smaller than the image side (the full loader also upsamples bilinearly).
