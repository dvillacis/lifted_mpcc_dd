# cpp_paper — the solver for the paper experiments

`tv_dd` learns the TV-denoising weight α of an image by solving the lifted MPCC with a primal-dual interior point method. Every Newton system is solved by Schur-complement domain decomposition in the sense of

> L. R. Lueg, M. L. Bynum, C. D. Laird, L. T. Biegler, *Domain decomposition preconditioners for Schur complement systems arising in structured nonlinear optimization problems*, Optim. Eng. 27 (2026) 555–585.

The folder is self-contained: nothing is included from the rest of the repository. It needs only Eigen, LAPACK and zlib, with no IPOPT and no HSL. MUMPS (sequential) is optional and strongly recommended for the tile factorizations. It contains exactly one configuration, the one that came out best in the comparisons made in `../cpp_dd_toy` (see "Provenance").

## The method in one page

**Problem** (`tv_mpcc.hpp`). This is the bilevel TV-weight learning problem, with the lower level replaced by its optimality system:
- The gradient and the dual are lifted to polar form (r, θ) and (δ, θ). This leaves smooth equations plus one complementarity r(1 − δ) = 0, relaxed to r(1 − δ) ≤ t (Scholtes).
- **Consensus form.** Every variable referenced by two tiles (border u, border qx/qy, and α) gets a local copy per tile, plus a linking row `copy − consensus = 0`. Every row then belongs to one tile, and the tiles couple only through the consensus variables. This is the structure (1) of Lueg et al.
- **μ-coupled continuation.** t = max(t_min, 10μ), at most halving per iteration. A level gate stops the solve once inf_pr, inf_du and μ are all ≤ √t at t = t_min.
- **Start.** Chambolle–Pock for plain TV denoising, lifted to the MPCC variables.

**IPM** (`ipm.hpp`). Primal-dual barrier method.
- **Steps:** a filter line search on (θ = ‖constraints‖₁, barrier objective φ_μ) in the style of Wächter & Biegler (IPOPT's constants), backtracking from the fraction-to-the-boundary step. `--line-search none` gives the paper's plain fraction-to-the-boundary method.
- **Restoration phase** (`restoration.hpp`), when no step down to the minimum length is acceptable. As in IPOPT, it minimizes the ℓ₁ infeasibility ρ‖p + n‖₁ + ζ/2‖D(x − x̄)‖² subject to g(x) − p + n within the row bounds, with p, n ≥ 0 and ρ = 1000, ζ = √μ̄. The phase is an `NLP` like any other, solved by this same IPM: p_r and n_r join the tile of row r, so the tiles and the border are unchanged.
  - **Its Newton systems have the original pattern.** p and n are *elastic* variables: each enters one row, linearly, with no Hessian entry. The IPM eliminates them, as IPOPT does: e enters row r as c_e·x_e with diagonal a_e = Σ_e + δ, so dx_e = (rhs_e − c_e dλ_r)/a_e, and row r gets −c_e²/a_e on its diagonal and −c_e·rhs_e/a_e on its right-hand side. The proximity term ζD² goes on the primal diagonal.
  - **The phase reuses the original problem's decomposition**, MUMPS instances included (`SchurDD::same_structure` checks the pattern; otherwise the phase would build its own). The inertia test is unchanged, since a_e > 0. It stops at the first point with θ ≤ 0.9·θ (start) that the filter, augmented with the start, accepts. Then the bound multipliers come from the phase (reset to 1 if any exceeds 10³) and λ from least squares. If the phase fails, the full step is taken and the filter cleared (counted). `--restoration off` disables it.
- **Restoration on slow progress** (opt-in, `--stall-iter K`). The same phase also starts, before the step, when the IPM is nearly feasible (θ ≤ θ_min, the filter's switching threshold) but the barrier error E_μ has not halved in K iterations of the same barrier problem (same μ, same t, μ above its floor). If the phase fails there, the IPM simply goes on.
- Inequality rows get slacks.
- μ follows IPOPT's monotone rule with κ_ε = 1000, κ_μ = 0.7, θ_μ = 1.1, starting from μ₀ = 0.1.
  - **Floor t_min/10** (`--mu-min`; with `--t-mu-scale s`, t_min/s), not IPOPT's tol/10, so that μ/t stays ≥ 0.1 at t = t_min, as the continuation t = 10μ intends. With tol/10, mariposa N=640 (t_min = 5·10⁻⁴) sat at μ/t = 0.01 for 216 of its 251 iterations.
  - **At most one decrease right after a restoration phase** (`--resto-mu-steps`). The phase ends with least-squares multipliers, which can make E_μ look small for that one iteration: on mariposa N=640 (inf_du 7.5·10⁻³ there), μ fell 13× in that iteration (6.3·10⁻⁵ → 5·10⁻⁶, the old floor). t followed to t_min within three iterations, and 123k complementarity rows were suddenly violated. With the new floor, that run's next decrease reaches the floor anyway, so the cap did not act there; it guards the case of a restoration well above the floor.
- The constraint multipliers start from the least-squares estimate.
- **Regularization:** one δ on all tile primals (IPOPT's δ_w schedule), raised when some In(W_k) is wrong (eq. 24) or S is not positive definite (eq. 25). A dual δ_c = 10⁻⁶ is in every Newton system (`--dual-reg`; stabilization for the near-singular rows at flattened single-pixel features, see "The end game"); with `--dual-reg 0` it is added only when a tile has too *few* negative eigenvalues (its Jacobian is rank-deficient, and no δ fixes that), as IPOPT does. Nothing is added on the consensus variables.
  - Each iteration first tries δ = 0, as IPOPT does. With `--delta-start last`, an iteration that follows one with δ > 0 starts from δ_last/3 instead (δ > 0 is needed in most iterations here). This changes the path, so it is not the default.
  - With δ_c > 0, every wrong In(W_k) leads to the same decision (raise δ). So each rank stops factorizing its tiles at the first tile with a wrong inertia; the rest would be refused anyway. The run is unchanged. Only the counters can differ: which of "singular" and "wrong In(W_k)" is reported, and the count of tiles with too few negatives.

**Linear algebra** (`schur_dd.hpp`, `blocks.hpp`, `precond.hpp`):
- **Tile blocks, default when built with MUMPS.** MUMPS factorizes the augmented tile matrix [W_k B_kᵀ; B_k 0] with its Schur-complement feature. That returns S_k directly and gives In(W_k) exactly through its pivoted (1×1/2×2) LDLᵀ (`mumps_block.hpp`).
  - **Pivot threshold** CNTL(1) = 10⁻⁵ (MUMPS's default is 0.01). At 0.01 about 15% of the pivots of a tile were delayed, which grows the fronts; 10⁻⁵ cuts the tile flops by about 40%.
    - IPOPT's 10⁻⁶ gains little more and fails `mumps_check`'s accuracy test (1.3·10⁻⁹; IPOPT relies on iterative refinement). The measurements are in `mumps_block.hpp`.
  - **Tile solves** go through MUMPS's condensation and expansion (ICNTL(26) = 1, then 2): one forward sweep gives B_k W_k⁻¹ r_k, one backward sweep W_k⁻¹(r_k − B_kᵀ u_y). This is half the work of two full solves. The self-test checks both.
- **Tile blocks without MUMPS.** A sparse unpivoted LDLᵀ in a KKT-aware level order. A tile falls back to dense Bunch–Kaufman for one factorization when a pivot is zero or tiny relative to its row (a rank-deficient −JH⁻¹Jᵀ block, which needs 2×2 pivots). The inertia stays exact here too.
- **Interface.** The local Schur complements S_k = −B_k W_k⁻¹ B_kᵀ (from MUMPS, or by forward solves) give S = C + Σ N_k S_k N_kᵀ (eqs. 14–15).
- **Interface solve, default `--schur pcg`.** PCG on S, applied through the stored S_k, with the additive Schwarz preconditioner on the local assembled complements S̃_k = N_kᵀ S N_k (eqs. 18–19), at tolerance 1e-10. A direction with pᵀSp ≤ 0 is the paper's Sec. 3.5 test for S not positive definite.
  - **Indefinite S̃_k.** S̃_k is a principal submatrix of S, so if its Cholesky factorization fails, S is not positive definite. The solve is then refused before any tile solve or PCG iteration. In every run measured, such an S̃_k was followed by a PCG breakdown anyway. The eigenvalue clipping is kept only for `--precond asd`, whose blocks are not submatrices of S.
  - **Lazy build.** The preconditioner (the S̃_k exchange and the block factorizations) is built by the first solve after a factorization. A factorization refused for In(W_k) therefore costs none of it. Its time still counts as "factorize".
- **`--schur direct`.** The assembled S is factorized: exact, and the reference for checking PCG.
- **Parallelism: MPI over tiles** (`MPI=1`, `comm.hpp`). Each rank owns a contiguous range of tiles and **holds only their part of the problem**: their variables, rows and KKT unknowns, plus the consensus variables (the border), which every rank holds and updates identically. A rank evaluates the functions and derivatives of its own rows, runs the IPM's elementwise work on its own entries, factorizes its W_k with its own MUMPS, forms its S_k and does its tile solves. The PCG vectors live on the border and are the same on every rank.
  - **Exchange.** Never MPI sums of floating-point values:
    - every global sum (θ, φ_μ, ∇φ_μᵀd, the scaling sums, the objective) is formed per tile, the partials are all-gathered and added in tile order (`TileSum`), the consensus part last;
    - the consensus rows of Jᵀλ and K·d, and every border contribution in each solve and each PCG product, are all-gathered per tile and added in tile order (`SchurDD::border_product`);
    - maxima and minima (inf_pr, inf_du, the fraction to the boundary) are exact reductions;
    - for the Schwarz blocks S̃_k, each tile j sends only the part of S_j on the border unknowns it shares with tile k: an edge block for neighbours, and for distant tiles the single entry of α, which every tile shares (`MPI_Alltoallv`). With `--schur direct` every rank assembles all of S, so every S_k is all-gathered.
  - **Consequence.** A run is bit-identical for any number of ranks and threads (verified: 1–4 ranks and 4 threads on mariposa N=48; 1–3 ranks on cameraman N=32 with PCG, direct, and the sparse+dense tiles of a build without MUMPS; 4 and 8 ranks on cameraman N=256 through a restoration phase). No rank can drift from the others.
  - **Memory.** Per rank, it is the rank's share of the problem plus a few per-pixel arrays (image, warm start, the variable → tile map). See "Memory" below.
  - **`--diag`, `--classify`, the solution and the `.npz`** are gathered on rank 0 in the global layout (for `--diag`/`--classify` at every iteration where they print).
- **OpenMP within a rank** (`OMP=1`) runs the per-tile loops in parallel. MUMPS's C interface is not thread-safe, even across separate instances: concurrent calls crashed or aborted in `mumps_check`. So every MUMPS call takes one process-wide lock: within a rank, MUMPS tiles go one at a time. MPI ranks are separate processes, so they don't share that lock. **Use MPI ranks, not threads, for the tile factorizations.**

## Files

| file | content |
|---|---|
| `main.cpp` | `tv_dd`: CLI, the fixed settings, report, CSV line, `.npz` output |
| `tv_mpcc.hpp` | the MPCC in consensus form, continuation and level gate |
| `nlp.hpp` | the problem interface the IPM uses (replaces `Ipopt::TNLP`) |
| `ipm.hpp` | the interior point method |
| `restoration.hpp` | the restoration phase's feasibility problem, as an `NLP` |
| `schur_dd.hpp` | the Schur-complement decomposition of each Newton system |
| `comm.hpp` | the MPI layer (tile ownership, all-gathers, tile-ordered sums `TileSum`, gathers on rank 0), or a single-rank stand-in without MPI |
| `blocks.hpp` | factorizations of W_k (MUMPS, or sparse level-ordered LDLᵀ + fallback, or dense) and of S |
| `mumps_block.hpp` | one tile through MUMPS: augmented matrix, Schur complement, inertia, solves; self-test |
| `mumps_check.cpp` | `mumps_check`: Schur layout, accuracy vs dense, and concurrency, to run once per machine |
| `precond.hpp` | additive Schwarz and PCG |
| `image.hpp`, `third_party/stb_image.h` | image → clean/noisy instance |
| `partition.hpp` | k×k tiles |
| `save.hpp`, `npz_writer.hpp` | `.npz` output, readable by `../python/plot_slurm.py` |

## Build and run

```bash
./build.sh                       # serial
OMP=1 ./build.sh                 # OpenMP (macOS: brew install libomp)
MPI=1 ./build.sh                 # MPI over tiles (macOS: brew install open-mpi)
./mumps_check                    # once per machine, if built with MUMPS: must print PASSED

OMP_NUM_THREADS=1 mpirun -np 8 ./tv_dd --size 128 --nsub 8    # 64 tiles over 8 ranks

./tv_dd --size 32 --nsub 4
./tv_dd --data ../images/mariposa.png --size 48 --nsub 4 --save-solution run.npz
./tv_dd --size 32 --nsub 4 --schur direct          # exact interface solve
./tv_dd --help
```

On Linux, build inside a conda env with `eigen` and `lapack`, or set `LAPACK_LIBS` (e.g. `"-lopenblas"`). Threads: `--threads n` or `OMP_NUM_THREADS`. BLAS is pinned to one thread inside each tile.

**MUMPS.** `build.sh` finds MUMPS through pkg-config `coinmumps`, i.e. COIN-OR ThirdParty-Mumps; `~/.local/coinmumps` is searched by default. On this laptop it was installed with:

```bash
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && cd ThirdParty-Mumps && ./get.Mumps
./configure --prefix=$HOME/.local/coinmumps --with-lapack-lflags="-L$(brew --prefix openblas)/lib -lopenblas"
make -j8 && make install          # MUMPS 5.9.1 (ThirdParty-Mumps 3.0.14)
```

Elsewhere, give the flags directly: `MUMPS_CFLAGS="-I…/include" MUMPS_LIBS="-L…/lib -ldmumps_seq -lmumps_common_seq …"` (e.g. conda-forge's `mumps-seq`). `MUMPS=0 ./build.sh` builds without it. `tv_dd` runs MUMPS's self-test at startup and refuses to use MUMPS if it fails.

**MPI and sequential MUMPS in one binary.** Sequential MUMPS ships its own stand-ins for some MPI functions (`MPI_Init`, `MPI_Comm_rank`, `MPI_Finalize`, … in its `libseq`), under the same names as the real MPI library. `build.sh` therefore does not use the `mpicxx` wrapper, which puts `-lmpi` last. It takes the wrapper's flags and links the real MPI library *before* MUMPS. On macOS each symbol binds to the first library that exports it; check with `nm -m tv_dd | grep _MPI_Init`, which must say `(from libmpi…)`. On Linux, symbols are resolved globally at run time instead. **Before trusting MPI runs on a new machine:** run `mpirun -np 2 ./tv_dd --size 32 --nsub 4` and compare its iteration log with a run of a build without MPI (`MPI=0`); they must be identical.

| option | default | |
|---|---|---|
| `--data` | `../images/cameraman.png` | image file |
| `--size` | 32 | N (image is N×N) |
| `--nsub` | 4 | tiles per side (k×k tiles) |
| `--sigma`, `--seed` | 0.1, 0 | noise level and seed |
| `--t-min` | 1e-4 | final relaxation level; IPM tol = max(1e-8, 0.1·t_min), acceptable tol = max(1e-8, t_min) |
| `--schur` | `pcg` | `pcg` or `direct` |
| `--cg-tol`, `--cg-maxit` | 1e-10, 2000 | PCG |
| `--block-solver` | `mumps` (with MUMPS), else `sparse` | `dense` = Bunch–Kaufman on every W_k (reference) |
| `--fallback` | `mumps` (with MUMPS), else `dense` | where a failed `sparse` factorization goes |
| `--max-iter` | 3000 | IPM iterations |
| `--line-search` | `filter` | `none` = fraction-to-the-boundary steps only (the paper's setting) |
| `--restoration` | `on` | `off` = full step when the line search fails |
| `--stall-iter` | 0 (off) | K: restoration also after K iterations without progress (see the method) |
| `--delta-start` | `zero` | `last`: after an iteration that needed δ > 0, start from δ_last/3 instead of 0 (see the method) |
| `--alpha-y` | `primal` | step of λ: `primal` (α_pr), `bound-mult` (α_du), `full` (1), `min-dual-infeas` (IPOPT's `alpha_for_y`) |
| `--kappa-sigma` | 1e10 | after each step μ/(κ_Σ s) ≤ z ≤ κ_Σ μ/s |
| `--kappa-eps`, `--mu-steps` | 1000, 0 | μ decreases while E_μ ≤ κ_ε·μ, at most S times per iteration (0: no limit); experiment, see "The end game" |
| `--mu-min` | t_min / t-mu-scale | floor of μ; 0 = IPOPT's tol/10 (the behaviour before 2026-10-06) |
| `--resto-mu-steps` | 1 | at most this many μ decreases in the iteration after a restoration phase; 0 = no cap |
| `--t-comp-ratio` | 0 (off) | K > 0: t is never lowered below max r(1−δ)/K, so it cannot run ahead of an iterate whose products lag (experiment, see "The end game") |
| `--t-mu-scale` | 10 | t = max(t_min, scale·μ); 1 is Raghunathan–Biegler's μ/t = 1 (fewer iterations on N ≥ 96, see "The end game") |
| `--precond` | `as` | interface preconditioner: `as` additive Schwarz on S̃_k = N_kᵀ S N_k (eq. 19); `asd` S_k with its diagonal replaced by diag(S) (eq. 21; weaker, see "Known limitations") |
| `--vw-mu` | 0 (off) | M > 0: Raghunathan–Biegler's modified step (eq. 3.7, choice (ii) of 3.8) once μ ≤ M, as in IPOPT-C's source: every bound, η = 0.1·μ_prev/(1 + max(‖c‖∞, ‖z‖∞)), off in restoration, matching line-search gradient (IPOPT-C: M = 5·10⁻⁶); `--vw-bounds comp` restricts it to r ≥ 0, 1−δ ≥ 0, `--vw-linesearch plain` keeps the plain barrier gradient |
| `--t-rate` | 0.5 | t falls by at most this factor per iteration; 0: no limit (with `--t-mu-scale 1`: t = μ, IPOPT-C) |
| `--vw-linesearch` | `modified` | with `--vw-mu`: the line search's ∇φ_μᵀd uses the barrier term of the modified step, ∓(z + (μ − sz)/(s + ηz)) instead of ∓μ/s, as IPOPT-C's `filter.F` does; `plain` = the barrier gradient unchanged (the behaviour before) |
| `--penalty` | 0 (off) | π₀ > 0: exact-penalty formulation f + π Σ r(1−δ), no product rows (experiment, fails from N=128 on; see "The end game"); `--penalty-max 1e8` (cap; = π₀ keeps π fixed), `--penalty-hessian on\|off` |
| `--cleanup` | `off` | `gate`/`reach`: active-set clean-up after the level gate / as soon as t = t_min (experiment, does not converge; see "The end game"); `--cleanup-biactive smaller\|both\|none`, `--cleanup-eps`, `--cleanup-dual-reg`, `--cleanup-mu` |
| `--t-update` | `every` | `solved`: t is lowered only when the barrier problem is solved (E_μ ≤ κ_ε·μ), not at every iteration; experiment |
| `--bounds` | `rows` | `vars`: r, δ, α ≥ 0 as variable bounds instead of slacked rows (experiment; the rows stay, free) |
| `--diag` | off | per iteration: where the dual residual sits and what blocked the step (see "The end game") |
| `--dual-reg C`, `--dual-reg-exp K` | 1e-6, 0 | δ_c = C·μ^K in every Newton system; C = 0: only on demand (the behaviour before) |
| `--classify FILE\|-` | off | per end-game iteration: MPCC multipliers γ = a − ξ(1−δ), ν = b − ξr and their sign classes on the near-biactive cells; FILE: the last point per cell (CSV) |
| `--relax-cells THR` | 0 (off) | sign-driven per-cell relaxation (experiment, see `endgame_literature.md`); `--relax-shift 1e-4`, `--relax-persist 5`, `--relax-exp 0.7` |
| `mpirun -np R` | 1 | MPI ranks (build with `MPI=1`); tiles are split into R contiguous ranges |
| `--save-solution` | — | `.npz` with the instance, solution, run summary and μ-trace |
| `--quiet`, `--verbose 0/1/2` | | iteration table off; linear-algebra detail |

Every run ends with one machine-readable line, starting `CSV,`, with these fields:

`image,N,nsub,sigma,seed,t_min,interface,blocks,ranks,threads,status,ok,iters,t,alpha,comp,psnr_noisy,psnr_recon,wall,t_factor,t_solve,factorizations,fallbacks,cg_mean,cg_max,rss_mb`

Times and memory are the maximum over the ranks; memory is per rank.

Status codes: 0 converged, 1 iteration limit, 2 stopped by the level gate (the normal finish), 3 acceptable, −1 failure. `ok` is 1 for 0, 3, or a level-gate stop.

## Validated results

> **The tables below predate the performance changes of 2026-10-06** (MUMPS pivot threshold 10⁻⁵, condensation/expansion tile solves, stopping at a wrong In(W_k), lazy preconditioner, the S̃_k test). In single-rank trials on a loaded machine (with a 10⁻⁶ threshold), the changes ran cam96, mar32 and mar48 identically. Cameraman N=128 8×8 split in its end game by rounding: 90 instead of 104 iterations, PSNR 27.45 dB either way. On that instance, tile factorization flops fell from 1.2·10¹¹ to 2.7·10¹⁰ and wall time from 77 s to 35 s. The tables have not been re-run since.

**Default configuration** (MUMPS tiles, filter line search, PCG; 4 MPI ranks × 1 thread):

| instance | IPM its | α\* | PSNR noisy → recon | wall |
|---|---|---|---|---|
| cameraman N=32, 4×4 | 29 | 0.070432 | 20.34 → 26.40 dB | 0.14 s |
| cameraman N=32, 2×2 | 28 | 0.070395 | 20.34 → 26.40 dB | 0.10 s |
| cameraman N=32, 8×8 | 29 | 0.070450 | 20.34 → 26.40 dB | 0.35 s |
| cameraman N=48, 4×4 | 30 | 0.070171 | → 26.67 dB | 0.30 s |
| mariposa N=32, 4×4 | 51 | 0.065930 | → 23.43 dB | 0.23 s |
| mariposa N=48, 4×4 | 135 | 0.055690 | → 24.26 dB | 1.28 s |
| cameraman N=128, 8×8 (8 ranks) | 91 | 0.069459 | → 27.45 dB | 7.6 s |

All runs stop at the level gate (t = 10⁻⁴).
- **Cameraman:** the line search changes almost nothing (0–3 step halvings per run).
- **Mariposa:** it reaches better stationary points than the plain method. At N=32, IPOPT's point (23.43 dB) instead of 23.38 dB; at N=48, 24.26 dB, where the plain method reached 24.12 and IPOPT 24.25.
- No run had an iteration without an acceptable step, so none used the restoration phase; adding the phase left these runs unchanged, value for value. Only N=256 needs it (below).
- **Determinism:** with the line search active, the iteration logs on 1 and 3 ranks are bit-identical (cameraman and mariposa N=32).

**Tile backends, without the line search** (`--line-search none`; 4 threads, 1 process). Columns compare `mumps`, `sparse+mumps` (sparse with MUMPS fallback) and `sparse+dense` (the default without MUMPS, the configuration selected in `../cpp_dd_toy`):

| instance | mumps: its, α\*, PSNR | wall / peak RSS: mumps | sparse+mumps | sparse+dense |
|---|---|---|---|---|
| cameraman N=32, 4×4 | 29, 0.070432, 26.40 dB | 0.44 s / 30 MB | 0.51 s / 47 MB | 3.8 s / 235 MB |
| cameraman N=32, 2×2 | 28, 0.070395, 26.40 dB | 0.32 s / 28 MB | 2.1 s / 62 MB | 28.8 s / 1.1 GB |
| cameraman N=32, 8×8 | 28, 0.070450, 26.40 dB | 1.00 s / 32 MB | 0.93 s / 41 MB | 1.6 s / 104 MB |
| cameraman N=48, 4×4 | 31, 0.070171, 26.67 dB | 0.98 s / 50 MB | 2.8 s / 105 MB | 29.9 s / 1.1 GB |
| mariposa N=32, 4×4 | 28, 0.067168, 23.38 dB | 0.35 s / 40 MB | 0.40 s / 58 MB | 6.1 s / 238 MB |
| mariposa N=48, 4×4 | 48, 0.066556, 24.12 dB | 1.41 s / 62 MB | 3.9 s / 116 MB | 48.3 s / 1.2 GB |

All runs stop at the level gate (t = 10⁻⁴).
- **The three backends take the same path** (same iterations, α\* and PSNR) on every instance except mariposa N=32.
- **There, the paths split at the start.** The least-squares system for the initial multipliers is rank-deficient on that instance. MUMPS detects the null pivot, so the IPM starts from λ₀ = 0 (also IPOPT's rule when its estimate fails). Dense Bunch–Kaufman returns an estimate that passes the size check.
- **From the two starts this nonconvex instance reaches two stationary points:** 28 iterations, α\* = 0.067168, 23.38 dB (MUMPS) against 48 iterations, α\* = 0.065930, 23.43 dB (dense).
- With `--block-solver sparse --fallback dense`, the iteration logs of cameraman N=32, 4×4 (PCG and direct interface) are identical, value for value, to the configuration as tested in `../cpp_dd_toy` (`dd_mpcc --host simple --reg global`).

**Larger instances** (cameraman, MUMPS tiles, 8 threads):

| N | tiles | KKT unknowns | IPM its | PCG mean / max | wall | peak RSS | α\*, PSNR |
|---|---|---|---|---|---|---|---|
| 96 | 8×8 | 164k | 57 | 203 / 386 | 9.8 s | 191 MB | 0.072219, 27.54 dB |
| 128 | 8×8 | 289k | 89 | 144 / 248 | 27.8 s | 355 MB | 0.069421, 27.45 dB |
| 128 | 16×16 | 305k | 100 | 376 / 953 | 37.0 s | 361 MB | 0.069422, 27.45 dB |

**Threads with MUMPS.** Cameraman N=48 4×4 takes 0.95 s on 1 thread and 1.07 s on 8: within one process the MUMPS calls are serialized, so threads don't help the factorization. MPI ranks do:

**MPI strong scaling** (cameraman N=128, 8×8 tiles, 289k KKT unknowns, MUMPS tiles, 1 thread per rank; this laptop: 5 performance + 10 efficiency cores):

| ranks | wall | speed-up | factorize | solve | rest (IPM, functions) | memory/rank |
|---|---|---|---|---|---|---|
| 1 | 33.9 s | 1.0× | 26.7 s | 5.9 s | 1.3 s | 374 MB |
| 2 | 17.4 s | 1.9× | 13.6 s | 3.1 s | 0.7 s | 224 MB |
| 4 | 10.1 s | 3.3× | 7.7 s | 2.0 s | 0.4 s | 124 MB |
| 8 | 6.6 s | 5.1× | 4.6 s | 1.7 s | 0.3 s | 113 MB |

All runs give the same 104 iterations and α\* = 0.069462, 27.45 dB.
- The factorization phase scales 5.8× on 8 ranks. Beyond 5 ranks this laptop uses its efficiency cores.
- The solve phase scales less (3.5×): its tile solves are distributed, but each PCG iteration exchanges the border contributions, and the vector work on the border is the same on every rank.
- The rest (IPM, function evaluations) is distributed too and falls from 1.3 to 0.3 s. With the replicated IPM of the earlier version it stayed at 0.6–0.7 s on any number of ranks (then: 89 iterations, α\* = 0.069421, 369/244/183/196 MB per rank on 1/2/4/8 ranks).

**N=256** (cameraman, 1.1M KKT unknowns, 8 ranks, MUMPS tiles) converges with the restoration phase; without it, it reached the 300-iteration limit with both tilings.

| tiles | IPM its (+ restoration) | δ > 0 in | PCG mean / max | wall | memory/rank | α\*, PSNR |
|---|---|---|---|---|---|---|
| 16×16 | 151 (+14) | 146 its | 354 / 906 | 94 s | 793 MB | 0.072284, 28.55 dB |
| 8×8 | 301 (+56) | 296 its | 192 / 369 | 199 s | 787 MB | 0.072250, 28.55 dB |

Both stop at the level gate (t = 10⁻⁴); the 16×16 log is bit-identical on 4 and 8 ranks, restoration phase included. This table and the ones below were measured with δ_c on demand (`--dual-reg 0`). With the current default δ_c = 10⁻⁶: 16×16 161 (+10) iterations, 75 s, 28.55 dB; 8×8 99 (+15) iterations, 54 s, α\* 0.072177, 28.55 dB.
- **One restoration phase decides each run.** Before it, the IPM stalls at μ = 10⁻³·² with t near 7·10⁻³, inf_du 10²–10³ and δ ≈ 10¹–10²·⁷. Tiles are indefinite there (every inertia refusal is a tile with too *many* negative eigenvalues), and the line search is not the limit: mostly full steps.
- **16×16:** the line search fails at iteration 39 and the phase takes θ 3.4·10⁻⁴ → 1.2·10⁻⁴ in 14 iterations. Afterwards t halves at once (7.0·10⁻³ → 3.5·10⁻³), and δ drops below 1 within 7 iterations (from 10¹–10²·⁵ before).
- **8×8:** the stall lasts until iteration 176, when the phase runs (56 iterations, θ 1.7·10⁻⁴ → 7.6·10⁻⁵); δ drops more slowly there (around 10⁰ for 60 iterations), and the gate follows 125 iterations later. The two tilings are different NLPs (the consensus copies depend on the tiles), so their paths differ.
- **Restoration on slow progress** (`--stall-iter K`, 8 ranks). The iteration where μ first leaves 10⁻³·², where t first reaches t_min, and the last iteration:

  | tiles | K | leaves μ = 10⁻³·² | t = t_min from | IPM its (+ restoration) | wall | α\*, PSNR |
  |---|---|---|---|---|---|---|
  | 16×16 | off | 40 | 95 | 151 (+14) | 94 s | 0.072284, 28.55 dB |
  | 16×16 | 10 | 34 | 124 | 179 (+16) | 119 s | 0.072311, 28.56 dB |
  | 16×16 | 15, 20 | 40 | 95 | 151 (+14): does not fire before the line search fails | 101 s, 95 s | 0.072284, 28.55 dB |
  | 8×8 | off | 177 | 275 | 301 (+56) | 199 s | 0.072250, 28.55 dB |
  | 8×8 | 10 | 34 | 93 | 105 (+25) | 78 s | 0.072242, 28.55 dB |
  | 8×8 | 15 | 63 | 69 | 326 (+42) | 229 s | 0.071344, 28.55 dB |
  | 8×8 | 20 | 72 | 78 | 216 (+39) | 152 s | 0.069829, 28.53 dB |

  - **The trigger does what it is meant to:** on 8×8 the stall at μ = 10⁻³·² ends at iteration 34–72 instead of 177.
  - **The total does not follow.** After t reaches t_min, the IPM needs from 11 to 256 more iterations to pass the gate (inf_du ≤ √t). This end game differs from run to run because each restoration lands the iterate somewhere else in a nonconvex landscape, and it dominates the count. IPOPT also spends about 250 iterations at its final μ.
  - **No effect elsewhere:** with K = 10, 15 or 20, the six small instances and cameraman N=128 do not trigger it and are unchanged, value for value. A first version without the θ ≤ θ_min condition fired early at large θ, changed mariposa's stationary points and had failed phases; the condition removed that.
  - **So it stays opt-in.** K = 10 gave the best total on these two runs, but two nonconvex runs are too few to choose a default.
- **IPOPT + MUMPS on the same NLP** (serial, `../cpp_dd_toy/dd_mpcc --host mumps --size 256 --nsub 16 --max-iter 300`) also enters its restoration phase: 22 iterations, from iteration 3 on. It uses δ_w up to about 90, and reaches the gate at iteration 298: α\* = 0.0730, 28.55 dB, in 286 s and 1.2 GB.

**The end game: why inf_du stays high once t = t_min** (cameraman N=256, `--diag`). The level gate needs inf_du ≤ √t = 10⁻², and the last 50–250 iterations are spent getting there.
- **Where the dual residual sits.** Almost entirely in two places; u, q, θ, α and the copies are at 10⁻³ or below throughout.
  - *Cells where r → 0 (r ≈ 10⁻⁸) with δ < 1*: λ_h2 grows to 10²; the residual is in the stationarity of r's bound (the slack of hr, or r itself with `--bounds vars`), up to 4·10¹. The same cells recur with both formulations of the bounds, e.g. cell (112, 28). This is consistent with the rank condition on the active set 𝒜 in the manuscript's degeneracy analysis, whose conditioning degrades where flat regions meet.
  - *Active complementarity rows* (r(1−δ) = t): λ_comp grows from about 7 to 10² within 15 iterations, and the slack's stationarity −λ_comp + z = 0 lags behind. 20–50k slacks exceed the gate.
- **Why it closes slowly.** For a slack, stationarity is linear and one Newton step closes it, but λ moves with α_pr (≈ 1) and z with α_du = 0.01–0.5. α_du is set by bound pairs far off the central path: z ≈ 10⁻⁷ at r ≈ 10⁻⁴, so s·z ≈ 10⁻¹¹ ≪ μ, and the step dz ≈ −z·ds/s cuts α_du for the whole system. Of 132 limited dual steps, 92 were r-slacks like this, 20 comp slacks, 20 δ bounds.
- **What was tried** (IPM iterations (+ restoration); default: N=128 8×8 91, N=256 16×16 151 (+14), N=256 8×8 301 (+56)):

  | variant | small six | N=128 8×8 | N=256 16×16 | N=256 8×8 |
  |---|---|---|---|---|
  | `--alpha-y bound-mult` | 21–107 its, fewer on all six | 109 (+33) | 533 | limit (600) |
  | `--kappa-sigma 1e4` | fewer on 4 of 6, mariposa lower | 92 | 208 (+23) | 252 (+53) |
  | `--kappa-sigma 1e2` | about the same | 84 | 249 (+73) | 379 |
  | `--bounds vars` | fewer on 5 of 6, mariposa lower | **63** (4.7 s) | 176 (95 s) | 229 (+56) (179 s) |

  `full` and `min-dual-infeas` were mixed on the small set (min-dual-infeas: 417 iterations and 20 restoration phases on mariposa N=48). "Mariposa lower" means at least one mariposa run ends at a lower stationary point (23.38 dB at N=32; 24.12 or 24.25 dB at N=48) instead of 23.43/24.26 dB. All N=128/256 runs end at the same PSNR (27.45 / 28.53–28.56 dB).
- **Reading.** Making λ follow z (`bound-mult`) or recentring z (κ_Σ) treats the symptom and slows the earlier phase. Variable bounds help most (N=128: 91 → 63 iterations) but leave the same cells in the end game. The residual comes from multipliers growing at degenerate cells, which the manuscript handles by gauge fixing (freezing the degenerate angles) and active-set cleanup; this solver has neither. None of the variants is the default.
- **Re-measured with the μ floor t_min/10 and the post-restoration cap (2026-10-06; 12 ranks; IPM its (+ restoration its), wall, α\*, PSNR).** The matrix was stopped after 7 of 15 runs; cameraman N=256 was not reached.

  | variant | mariposa N=512 16×16, t_min 5·10⁻⁴ | mariposa N=640 16×16, t_min 5·10⁻⁴ |
  |---|---|---|
  | defaults | 136 (+11), 82 s, 0.07210, 29.947 dB | 80 (+12), 96 s, 0.07154, 30.472 dB |
  | `--bounds vars` | 66 (+10), 49 s, 0.07192, 29.939 dB | 100 (+11), 122 s, 0.07257, 30.526 dB |
  | `--t-mu-scale 1` | 345, 192 s, 0.07150, 29.913 dB | — |
  | `--dual-reg 1e-5` | **51, 27 s**, 0.07117, 29.912 dB | — |
  | `--t-comp-ratio 2` | 118 (+11), 83 s, 0.07219, 29.950 dB | — |

  - Before the floor (tol/10, no cap): mariposa N=512 took 209 iterations, N=640 251 (+12).
  - None of the variants is the default yet. `--dual-reg 1e-5` is 3× faster on N=512 but ends at a different point (α\* −1.3%, −0.035 dB) and is untested elsewhere. `--bounds vars` is faster on N=512 and slower on N=640. `--t-comp-ratio 2` was measured on N=512 only, where t reaches t_min late (iteration 97), so the catch-up it targets is short there anyway; N=640 (catch-up of 28 iterations after t = t_min at iteration 36) is the instance to test it on.
- **Follow-up** (`endgame_literature.md` has the literature and the numbers). The MPCC multipliers of the final point (`--classify`) are S-stationary to solve accuracy on all but a few clusters of cells: (1) L-shaped triples of flat cells around one node, a single-pixel feature flattened by TV, where the local rows are near-singular and λ_h2 reaches 3·10²; (2) about ten near-biactive cells with both MPCC multipliers negative (C-stationary), where ξ grows like 1/√t. A dual regularization δ_c in every Newton system brings (1) down to 2–4·10¹ and cuts N=256 8×8 from 301 (+56) to 99 (+15) iterations; it is now the default (10⁻⁶). A sign-driven per-cell relaxation for (2) (DeMiguel et al. 2005) made N=256 slower (16×16: 263 iterations, 8×8: 140) and is off.
- **Where the end game comes from, and the pace of the continuation.** The long stretch at t = t_min is not tied to t_min. It follows the last large drop of μ:
  - **Large drops.** With κ_ε = 1000 a single iterate with a small error lets μ fall to about E_μ/1000 in one iteration: 6× at N=128 (iteration 35), 40× right after the restoration phase at N=256 (iteration 29).
  - **t follows.** t (= 10μ, at most halving per iteration) comes down within a few iterations, from an iterate that was stationary only for the old (t, μ).
  - **Re-convergence is slow.** The IPM then re-converges a fixed, nonconvex barrier problem at small μ: δ ≈ 0.1–1, α_du 0.05–0.3, inf_du 5–30, for 50–120 iterations.
  - **N=128 shows it above t_min.** The stall is at t = 2.4·10⁻⁴ (iterations 38–91); the last step to 10⁻⁴ then costs 10 iterations.
  - **A larger t_min shortens the stall but keeps it** (cameraman, 8 ranks, IPM its, PSNR):

    | t_min | N=128 8×8 | N=256 16×16 |
    |---|---|---|
    | 1e-4 | 104, 27.45 dB | 161 (+10), 28.55 dB |
    | 2e-4 | 96, 27.46 dB | 127 (+10), 28.56 dB |
    | 5e-4 | 79, 27.46 dB | 87 (+10), 28.58 dB |
  - **How small t needs to be: t_min = 10⁻⁴ suffices for the results.** Cameraman solved at t_min = 10⁻⁴ and at 10⁻⁶ (8 ranks; gray levels on the 0–255 scale of the 8-bit image):

    | N | max \|u(10⁻⁴) − u(10⁻⁶)\| | rms difference | pixels differing by > 1 gray level | α\* difference | rms error vs clean, 10⁻⁴ / 10⁻⁶ |
    |---|---|---|---|---|---|
    | 32 | 0.50 gray levels | 0.08 | 0 of 1024 | 0.23% | 0.04787 / 0.04793 |
    | 64 | 0.57 | 0.09 | 0 of 4096 | 0.02% | 0.04465 / 0.04469 |
    | 128 | 0.73 | 0.10 | 0 of 16384 | 0.47% | 0.04240 / 0.04246 |

    - **Below what can be seen.** Every pixel agrees to within one gray level, below the image's quantization; the noise (σ = 0.1) is about 25 gray levels. The reconstruction error is the same to the fourth digit, and α\* moves less than it does between tilings.
    - **Why.** A cell holding both a gradient and |q| < 1 has r ≈ 1 − |q| ≈ √t: 2.5 gray levels at t = 10⁻⁴, a quarter of one at 10⁻⁶.
    - **What a smaller t would sharpen is the stationarity certificate, not the image.** The theory is a limit t → 0 with tolerances o(t) (Kanzow–Schwartz; `endgame_literature.md` §1), which no finite t, and not the √t gate, satisfies exactly. At 10⁻⁴ the certificate is approximate: the residuals and the multiplier classification (`--classify`; S-stationary to solve accuracy except a few clusters, §5).
    - **The runs below at t ≈ 10⁻⁶ therefore probe the method's limits, not the operating point.**

  - **Slowing μ, or lowering t only on solved barrier problems, is worse** (IPM its (+ restoration its), 600 = iteration limit):

    | variant | N=128 8×8 | N=256 16×16 |
    |---|---|---|
    | default (κ_ε = 1000, t every iteration) | 104 | 161 (+10) |
    | `--kappa-eps 10` | 244 (+42), 2 phases | 600, t never reaches t_min |
    | `--mu-steps 1` | 118 (+37) | 316 (+30), 3 phases |
    | `--t-update solved` | 600 (+119), t never reaches t_min | 600 (+130), 9 phases |
    | `--kappa-eps 10 --t-update solved` | 600 (+148) | — |
    | `--mu-steps 1 --t-update solved` | 232 | — |

  - **Reading.** Every halving of t costs a re-convergence: 15–100 iterations each with `--t-update solved`. Lowering t on consecutive iterations, as the default does, pays it once at the end. Decoupling t from μ also lets μ run ahead: μ = 10⁻⁶ at t = 2·10⁻³, so μ/t ≈ 5·10⁻⁴. The iterate then jams (α_pr ≈ 10⁻¹¹, one restoration phase after another), as Raghunathan & Biegler (2005) warn: μ/t must stay bounded away from zero. The defaults stay.
  - **The active-set clean-up does not converge** (`--cleanup`, the manuscript's "active-set cleanup"; implemented in `TvMpcc::begin_cleanup`). The setup:
    - From the last point of the continuation, every cell gets the smaller of r and w = 1 − δ pinned to zero (rows hr or hd become equalities, the bound δ ≤ 1 is dropped), and the product rows are freed.
    - The NLP is re-solved warm (`NLPPoint`), with each product multiplier ξ moved into the pinned row, so that γ = a − ξw and ν = b − ξr start consistent.
    - **Result.** On cameraman and mariposa N=32 and mariposa N=48 the re-solve does not converge: 3000 iterations, repeated failed restoration phases, the pinned rows still violated by about 5·10⁻³.
    - **What does not help.** Pinning both or neither on near-biactive cells (`--cleanup-biactive`), larger δ_c (10⁻⁴, 10⁻³), a larger starting μ (10⁻⁴), starting as soon as t = t_min (`reach`).
    - **Why.** At t = 10⁻⁴ the active set is not identified. 406 of the 961 cells of cameraman N=32 end the continuation with both r and w above 10⁻³ (gradient magnitudes ~√t). The cells whose pin fails are exactly such cells: pinned at r = 0 (r ≈ 6·10⁻³, w ≈ 1.6·10⁻²), the re-solve drives w → 0 while r stays near 4·10⁻³.
    - **The multipliers cannot choose the branch either.** On these cells the bound multipliers are ≈ 0, so γ ≈ −ξw and ν ≈ −ξr have the same sign (both negative on 209 of the 406).
    - **Flipping does not converge either.** Pins whose member stays > 10⁻³ while the other goes to zero were flipped over up to six re-solves of 150 iterations (scratch code, not kept). Violations fell to ≈ 10⁻³ and then stayed there with nothing left to flip: groups of pins are jointly inconsistent. A flattened region must also satisfy TV's condition on q along its boundary.
    - **The code stays as an opt-in experiment;** the defaults are unchanged.
  - **The exact-penalty formulation works on the small instances and fails from N=128 on** (`--penalty π₀`; Leyffer, López-Calva & Nocedal 2006).
    - **The formulation.** The product rows are dropped and π Σ r(1−δ) is added to the objective, with r ≥ 0, 1−δ ≥ 0 kept. π grows by LLN's dynamic rule: ×10 when max min(r, w) > μ^0.4 and it has not decreased over 3 iterations. The stopping rule is the same gate as the continuation, plus max r·w ≤ t_min.
    - **Small instances (π₀ = 10):** cameraman N=32: 40 its, 26.39 dB; mariposa N=32: 28 its, 23.42 dB; mariposa N=48: 73 its, 24.10 dB. Complementarity r·w ≤ 2·10⁻⁶ instead of 10⁻⁴, iterations comparable to the continuation (34/34/46). π₀ = 1 fails on mariposa N=48 (π runs to 10⁸); π₀ = 100 needs 15 restoration phases on cameraman N=32.
    - **N=128 and N=256 (8 ranks): no run converges in 400–600 iterations.**

      | variant (cameraman N=128 8×8) | its | max r·w | PSNR |
      |---|---|---|---|
      | adaptive π₀ = 10 | 600, π → 10⁸ | 5·10⁻⁶ | 16.56 dB |
      | adaptive π₀ = 100 | 600, π → 10⁸ | 3·10⁻⁶ | 8.76 dB |
      | π = 10 fixed | 400 (+144 in 6 restoration phases) | 2·10⁻² | 27.42 dB |
      | π = 100 fixed | 400 (+107 in 11 phases) | 0.11 | 27.35 dB |
      | π = 10 fixed, without π's Hessian term | 400 (+3 phases) | 10⁻² | 27.44 dB |
      | adaptive π₀ = 10, without π's Hessian term | 400, π → 10⁵ | 8·10⁻⁴ | 27.37 dB |

      cameraman N=256 16×16, adaptive π₀ = 10: 600 its, π → 10⁸, 16.13 dB.
    - **Why: the penalty's curvature.** The term π·r·(1−δ) puts the indefinite block π[[0, −1], [−1, 0]] into the Hessian of every cell. The IPM's single δ for all tile primals must then exceed π: δ ≈ 5–150 on most iterations, against δ ≲ 1 for the continuation. The steps become damped gradient steps: mean α_pr 0.11–0.16 against 0.66. With π fixed, 82% of the cells still have r·w > 10⁻⁴ after 400 iterations.
    - **Why π blows up.** The adaptive rule raises π every 4 iterations up to the cap, with max min(r, w) frozen at 4.8·10⁻². Leaving π's Hessian term out removes the large δ but not the slow progress (mean α_pr 0.21).
    - **The code stays as an opt-in experiment;** the defaults are unchanged.
  - **Raghunathan & Biegler's coupling μ/t = 1 (`--t-mu-scale 1`) shortens the end game** (`endgame_literature.md` §1 and §9).
    - **The idea.** RB keep μ/t bounded away from zero, because their Lemma 2.7 bounds the bound multipliers from below by (μ/2t)·(the other member). Their Table 5.1 found μ/t = 1 best and 0.1 worse; 0.1 is our default (t = 10μ).
    - **Results** (8 ranks; IPM its (+ restoration its), PSNR):

      | instance | t = 10μ (default) | t = μ |
      |---|---|---|
      | cameraman N=256 16×16 | 161 (+10), 28.547 dB, 57 s | **88 (+10), 28.554 dB, 36 s** |
      | cameraman N=256 8×8 | 99 (+15), 28.552 dB | **53 (+11)**, 28.539 dB, α\* 0.0706 (instead of 0.0722) |
      | cameraman N=128 8×8 | 104, 27.453 dB | 99, 27.448 dB |
      | mariposa N=128 8×8 | 97, 26.478 dB | 70 (+20), 26.478 dB |
      | mariposa N=96 8×8 | 75, 25.784 dB | 68, 25.779 dB |
      | cameraman N=512 16×16 | **76 (+13), 28.620 dB, 157 s** | 292 (+91, 10 restoration phases), 28.616 dB, 681 s |
      | small six, 4 ranks (its) | 34/31/33/28/34/46 | 22/29/26/28/85/79 |

    - **Small instances.** On mariposa N=32 and N=48, μ/t = 1 is slower but ends at better stationary points (23.83 vs 23.43 dB, 24.23 vs 24.12 dB).
    - **Without a floor** (`--t-min 1e-6`, t never frozen, as in IPOPT-C): 93–504 iterations on the small six. That solves to a much tighter complementarity, so the comparison is not like for like.
    - **RB's modified complementarity block** (`--vw-mu`, eqs. 3.7–3.8, after Vicente & Wright). At their threshold 5·10⁻⁶ it never switches on with t_min = 10⁻⁴: μ stays above it. Switched on at 5·10⁻⁵ together with μ/t = 1: N=256 16×16 81 (+10) instead of 88 (+10), N=128 unchanged.
    - **Not the default:** at N=512 it needs four times the iterations (292 against 76).
  - **IPOPT-C, aligned with its source code** (`endgame_literature.md` §10).
    - **What `tv_dd` reproduces.** The source (last Fortran IPOPT, `--enable-mpcc`) applies the modified step of RB's eq. 3.7 on every bound, with η = 0.1·μ_prev/(1 + max(‖c‖∞, ‖z‖∞)). It is off in restoration, with the matching barrier gradient in the line search. The relaxation is t = μ with no floor. `--vw-mu` now does the same, and `--t-mu-scale 1 --t-rate 0` gives t = μ.
    - **At the operating point t_min = 10⁻⁴, neither ingredient is a uniform gain.**
      - t = μ: cameraman N=256 161 → 88 iterations, but N=512 76 → 292.
      - Eq. 3.7 from μ ≤ 5·10⁻⁵: N=256 161 → 76, but mariposa N=96 75 → 205 and N=512 76 → 92. These runs used the version before the alignment (η from ‖λ‖∞, plain line-search gradient).
      - The aligned version at IPOPT-C's threshold 5·10⁻⁶ never switches on with t_min = 10⁻⁴, because μ stays above it.
    - **IPOPT-C's own regime, solving to r·w ≈ 10⁻⁶.** This tests the method's limits, not the paper's operating point (see "How small t needs to be" above). We ran it to reproduce IPOPT-C's configuration, which has no floor on t.
      - IPOPT-C converges where our coupling does not on mariposa (N=48: 83 vs 196 its; N=96: 246 vs cap).
      - Our coupling is faster on cameraman up to N=128 (169 vs cap).
      - No configuration converges on cameraman N=256 within 600 iterations.
      - Under t = μ the interface PCG hits its 2000-iteration limit in most iterations (527 of 600 with IPOPT-C), so the one-level Schwarz preconditioner is not enough at t ≈ 10⁻⁶.
    - **The defaults stay:** t = max(t_min, 10μ), t_min = 10⁻⁴ (or 5·10⁻⁴), no modified step.
  - **The IPM stops at its tolerance only once the problem is in its final form** (`NLP::final_form()`, here t = t_min). This changes nothing for the default runs. Without it, `--t-update solved` ended as "acceptable" at a looser t.

**Reference.** IPOPT with MUMPS on the same NLP (`../cpp_dd_toy/dd_mpcc --host mumps`) takes 45–131 iterations on the six small instances. It reaches the same PSNR to 0.01 dB and α\* within 0.06% on cameraman (all four) and on mariposa N=32. On mariposa N=48 it stops at α\* = 0.0586 (24.25 dB). This IPM, with its line search, reaches α\* = 0.0557 (24.26 dB) there; without it, 0.0666 (24.12 dB).

**Memory** (MPI, 1 thread per rank; peak RSS of the largest rank). Before the problem was distributed, every rank held the whole IPM, all function evaluations and the KKT pattern of every tile, and the restoration phase built a second full copy. Then per-rank memory hardly fell with more ranks.

| instance | ranks | before | now | |
|---|---|---|---|---|
| cameraman N=128, 8×8 | 8 | 206 MB | 118 MB | 97 → 104 its (end game, rounding), 27.45 dB both |
| cameraman N=256, 16×16 | 8 | 1304 MB | 438 MB | same 161 (+10) its, α\* 0.072384, 28.55 dB; wall 66 → 57 s |
| cameraman N=256, 8×8 | 8 | — | 359 MB | 99 (+15) its, α\* 0.072169, 28.55 dB; 46 s |
| cameraman N=512, 16×16, first 5 its | 8 | — | 985 MB | 4.56M KKT unknowns |

- **What is left, on rank 0** (N=256, 16×16, 8 ranks, live heap from `malloc_zone_statistics`):
  - the problem and the setup: ~80 MB;
  - after the first factorization: 160 MB, steady over the iterations;
  - during the restoration phase: 228 MB. The phase adds only its IPM vectors, since it reuses the outer decomposition. Before p and n were eliminated, the phase built its own KKT system (×1.9 the size) and MUMPS instances: 390 MB, and 613 MB peak RSS for the whole N=256 run.
- **RSS is higher than the live heap.** The macOS allocator keeps freed memory: RSS goes 219 → 327 MB while the live heap stays at 165 MB.
- **Results against the previous replicated version.** Global sums are now added per tile, so values differ from it by rounding. The paths agree on the small instances (iterations and α\* unchanged on the six, with `--schur direct`, `--bounds vars`, `--alpha-y min-dual-infeas`, `--relax-cells`), but can split in a long end game (N=128 above).
- **Eliminating p and n** changes the restoration phase only by rounding: on N=256 16×16 its 10 iterations agree with the uncondensed phase in every printed digit. Its `lin.res` column now measures the condensed system that is solved (≈10⁻¹⁰, PCG's tolerance), not the full one.

## Known limitations

- **Nonconvex instances:** the stationary point reached can depend on details: the initial multipliers, the line search, the host (mariposa).
- **The restoration phase is simpler than IPOPT's.** Its proximity term ζ is fixed for the whole phase (IPOPT ties it to the phase's μ), its slacks have no proximity term, and it does not recurse: a failed line search inside the phase takes the full step.
- **MUMPS is serialized** within one process (its C interface is not thread-safe). Parallel tile factorizations come from MPI ranks.
- **N=256 converges, but only after a restoration phase gets the IPM out of a stall** at μ = 10⁻³·² (see "Validated results"). By default the stall ends only when the line search happens to fail: at iteration 39 with 16×16 tiles, at 176 with 8×8. `--stall-iter` ends it earlier, but the end game at t = t_min (11–256 iterations to the gate) then dominates and varies from run to run. A coarse-to-fine warm start from N=128 is not implemented.
- **Without MUMPS, large tiles are slow and memory-hungry.** About 38% of the sparse tile factorizations fall back to dense Bunch–Kaufman, which costs t_k² doubles per tile.
- **One-level preconditioner.** PCG iteration counts grow with the number of tiles (16×16 vs 8×8 at N=128: 376 vs 144 mean). A coarse space was effective on a PDE toy problem but has not been tested on this MPCC, so it is not included.
  - **The diagonally assembled variant, eq. 21 (`--precond asd`), is weaker here.** It uses tile k's own S_k with diag(S), so ranks exchange only the diagonals of the S_k, not the neighbours' blocks. It needs 3–4× more PCG iterations than AS everywhere (8 ranks for N ≥ 96, 4 ranks for the small ones; IPM its, PCG mean per solve):

    | instance | AS | ASd |
    |---|---|---|
    | cam32 4×4 / 2×2 / 8×8, cam48 | 34/31/33/28 its; PCG 55/19/155/51 | same its; PCG 186/77/559/204 (8×8: 3 solves unconverged) |
    | mar32 / mar48 | 34/46; PCG 40/36 | same; PCG 174/149 |
    | cam128 8×8 | 104 its, 6.4 s; PCG 111 | 89 its, 7.6 s; PCG 337 |
    | mar96 8×8 | 75, 2.8 s; PCG 84 | 75, 3.9 s; PCG 265 |
    | cam256 16×16 | 161, **54 s**; PCG 129, build 4.5 s | 112, 60 s; PCG 477 (1 solve unconverged), build 1.8 s |
    | cam256 16×16, IPOPT-C regime (t ≈ 10⁻⁶; a limits test, not the operating point) | 600 (cap), PCG at its limit in 527 its, 5 restoration phases, 820 s | 600 (cap), PCG mean 1624, 529 solves unconverged, 18 phases, 1089 s |

    - The fewer IPM iterations at N=128/256 come from the inexact solves changing the end-game path, not from a better preconditioner. PSNR is the same to 0.01 dB.
    - The remedy the PCG failures point to is a coarse space (Lueg et al. eq. 22; implemented in `../cpp_dd_toy`, where `linear` modes removed the growth with the tile count, but not the late-barrier growth).

## Provenance

The configuration was selected in `../cpp_dd_toy` (see its README, "The MPCC under both hosts"):
- the paper-style IPM against IPOPT;
- one global δ against per-tile δ_k: the paper's Sec. 3.5 option failed on 4 of 6 instances;
- PCG with additive Schwarz against the direct interface solve.

The problem formulation is that of `../cpp_minimal/problem.hpp`, ported to `nlp.hpp`. The Fischer–Burmeister variant, the IPOPT plumbing and the failed-solve banking were left out.
