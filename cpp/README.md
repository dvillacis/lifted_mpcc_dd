# cpp — the solver for the paper experiments (`tv_dd`)

`tv_dd` learns the TV-denoising weight α of an image by solving the lifted MPCC with a primal-dual interior point method. Every Newton system is solved by Schur-complement domain decomposition in the sense of

> L. R. Lueg, M. L. Bynum, C. D. Laird, L. T. Biegler, *Domain decomposition preconditioners for Schur complement systems arising in structured nonlinear optimization problems*, Optim. Eng. 27 (2026) 555–585.

The folder is self-contained: nothing is included from the rest of the repository. It needs only Eigen, LAPACK and zlib, with no IPOPT and no HSL. MUMPS (sequential) is optional and strongly recommended for the tile factorizations. It contains exactly one configuration, the one that came out best in the comparisons made in `../archive/cpp_dd_toy` (see "Provenance").

A second binary, `tv_learn` (`tv_learn.hpp`), uses the same IPM and decomposition to learn a weight *map* on B×B blocks over a *set* of images, for denoising, inpainting and deblurring, with TV or directional TV. With its defaults it solves `tv_dd`'s problem bit for bit (see "tv_learn").

## The method in one page

**Problem** (`tv_mpcc.hpp`). This is the bilevel TV-weight learning problem, with the lower level replaced by its optimality system:
- The gradient and the dual are lifted to polar form (r, θ) and (δ, θ). This leaves smooth equations plus one complementarity r(1 − δ) = 0, relaxed to r(1 − δ) ≤ t (Scholtes).
- **Consensus form.** Every variable referenced by two tiles (border u, border qx/qy, and α) gets a local copy per tile, plus a linking row `copy − consensus = 0`. Every row then belongs to one tile, and the tiles couple only through the consensus variables. This is the structure (1) of Lueg et al.
- **μ-coupled continuation.** t = max(t_min, 10μ), at most halving per iteration. A level gate stops the solve once inf_pr and μ are ≤ √t at t = t_min; the dual infeasibility is not tested (`--gate kkt` adds inf_du ≤ √t, the rule before 2026-10-07 and in the measurements below).
- **Start: two stages** (the default since 2026-10-09; see "The two-stage method"). Stage 1, the smoothed bilevel method (`smoothed.hpp`), finds the basin; stage 2, the MPCC, starts from stage 1's weight and its Huber lower-level solution, lifted to the MPCC variables, and refines it on the exact problem. `--start uniform`: Chambolle–Pock for plain TV denoising at α₀ = 0.7σ, lifted (the start before 2026-10-09).

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
  - With δ_c > 0, every wrong In(W_k) leads to the same decision (raise δ). So **every rank stops factorizing its tiles as soon as any rank meets a tile with a wrong inertia or a singular one**; the rest would be refused anyway. The rank that meets it sends a small nonblocking message to the others, which look for it between tiles (`StopSignal`, `comm.hpp`); after the exchange of the results every rank receives exactly the messages of the ranks that sent one, so nothing is left in flight.
    - **The iterates are unchanged.** Which bad tile is met first depends on timing, so a singular W_k and a wrong In(W_k) are counted together ("wrong In(W_k) or singular W_k" in the summary), and the count of sparse tile factorizations that fell back can vary by a few between runs. The count of tiles with too few negatives can differ too.
    - Before 2026-10-07 each rank stopped only at its own first bad tile, and the others factorized all their tiles while the IPM waited for them. At N=128 8×8 (8 ranks) refused factorizations were 43% of the factorization time.

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
    - the consensus rows of Jᵀλ and K·d, and every border contribution in each solve and each PCG product, are all-gathered per tile and added in tile order (`SchurDD::border_product`). The per-tile border outputs live in one flat array with fixed offsets (`flat_`): each product writes into it, one all-gather spreads it, and nothing is allocated or copied per product;
    - maxima and minima (inf_pr, inf_du, the fraction to the boundary) are exact reductions;
    - for the Schwarz blocks S̃_k, each tile j sends only the part of S_j on the border unknowns it shares with tile k: an edge block for neighbours, and for distant tiles the single entry of α, which every tile shares (`MPI_Alltoallv`). With `--schur direct` every rank assembles all of S, so every S_k is all-gathered.
  - **Consequence.** A run is bit-identical for any number of ranks and threads (verified: 1–4 ranks and 4 threads on mariposa N=48; 1–3 ranks on cameraman N=32 with PCG, direct, and the sparse+dense tiles of a build without MUMPS; 4 and 8 ranks on cameraman N=256 through a restoration phase). No rank can drift from the others.
  - **Memory.** Per rank, it is the rank's share of the problem. The setup builds it from the grid geometry for the rank's own tiles: which tiles reference a variable, the copies' numbering and the stencils follow from the stencil itself, so no global table is built. The only per-pixel arrays are the warm start, during the setup (every rank computes it for the whole image), and the image on rank 0. See "Memory" below.
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
| `main_learn.cpp` | `tv_learn`: CLI, per-image report, CSV line, `.npz` files and JSON report |
| `tv_learn.hpp` | the MPCC of `tv_learn`: S images, operators, block-wise α, sharing tables |
| `learn_partition.hpp` | k×k tiles per image (image-major ids), α blocks, cuts snapped to the blocks |
| `operators.hpp` | the data terms (denoise, inpaint, deblur), the Gaussian blur, DTV weights, K_ξ |
| `pdhg.hpp` | the lower level solved directly (warm start, held-out evaluation) |
| `dataset.hpp` | folder / manifest listing and selection (from `../archive/cpp/dataset.hpp`) |
| `save_learn.hpp`, `learn_util.hpp` | `tv_learn`'s outputs; PSNR, memory, `--check-derivatives` |

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

**On the cluster** (OpenHPC, recognized by `/opt/ohpc`; `HPC=0` turns it off), `conda activate mkl_imaging && ./build.sh` always builds with Open MPI and MUMPS: it loads the `openmpi5` module if `mpicxx` is missing, compiles with the system `g++` and links with the system linker (`-B/usr/bin`; conda's linker cannot resolve the system libraries `libmpi.so` needs), and takes MUMPS from conda-forge's `mumps-seq`. Without MUMPS it stops instead of building the sparse+dense tiles, which are about 1000× slower at N=128. `tv_dd.slurm` runs it with the same modules.

**MUMPS.** `build.sh` finds MUMPS through pkg-config `coinmumps`, i.e. COIN-OR ThirdParty-Mumps; `~/.local/coinmumps` is searched by default. On this laptop it was installed with:

```bash
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && cd ThirdParty-Mumps && ./get.Mumps
./configure --prefix=$HOME/.local/coinmumps --with-lapack-lflags="-L$(brew --prefix openblas)/lib -lopenblas"
make -j8 && make install          # MUMPS 5.9.1 (ThirdParty-Mumps 3.0.14)
```

Elsewhere, give the flags directly: `MUMPS_CFLAGS="-I…/include" MUMPS_LIBS="-L…/lib -ldmumps_seq -lmumps_common_seq …"` (e.g. conda-forge's `mumps-seq`). `MUMPS=0 ./build.sh` builds without it. `tv_dd` runs MUMPS's self-test at startup and refuses to use MUMPS if it fails.

**MPI and sequential MUMPS in one binary.** Sequential MUMPS ships its own stand-ins for some MPI functions (`MPI_Init`, `MPI_Comm_rank`, `MPI_Finalize`, … in its `libseq`), under the same names as the real MPI library. `build.sh` therefore does not use the `mpicxx` wrapper, which puts `-lmpi` last. It takes the wrapper's flags and links the real MPI library *before* MUMPS. On macOS each symbol binds to the first library that exports it; check with `nm -m tv_dd | grep _MPI_Init` (`tv_dd` calls `MPI_Init_thread`), which must say `(from libmpi…)`. On Linux, symbols are resolved globally at run time instead. **Before trusting MPI runs on a new machine:** run `mpirun -np 2 ./tv_dd --size 32 --nsub 4` and compare its iteration log with a run of a build without MPI (`MPI=0`); they must be identical.

| option | default | |
|---|---|---|
| `--data` | `../images/cameraman.png` | image file |
| `--size` | 32 | N (image is N×N) |
| `--nsub` | 4 | tiles per side (k×k tiles) |
| `--sigma`, `--seed` | 0.1, 0 | noise level and seed |
| `--t-min` | 1e-5 | final relaxation level (1e-4 before 2026-10-11; the measurements below say which they used); IPM tol = max(1e-8, 0.1·t_min), acceptable tol = max(1e-8, t_min) |
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
| `--gate` | `primal` | the level gate tests inf_pr, μ ≤ √t; `kkt` also inf_du ≤ C·√t (the rule before 2026-10-07) |
| `--gate-scale` | 1 | with `--gate kkt`, C: the dual test is inf_du ≤ C·√t; inf_pr and μ stay at √t (experiment) |
| `--t-rate` | 0.5 | t falls by at most this factor per iteration; 0: no limit (with `--t-mu-scale 1`: t = μ, IPOPT-C) |
| `--vw-linesearch` | `modified` | with `--vw-mu`: the line search's ∇φ_μᵀd uses the barrier term of the modified step, ∓(z + (μ − sz)/(s + ηz)) instead of ∓μ/s, as IPOPT-C's `filter.F` does; `plain` = the barrier gradient unchanged (the behaviour before) |
| `--penalty` | 0 (off) | π₀ > 0: exact-penalty formulation f + π Σ r(1−δ), no product rows (experiment, fails from N=128 on; see "The end game"); `--penalty-max 1e8` (cap; = π₀ keeps π fixed), `--penalty-hessian on\|off` |
| `--cleanup` | `off` | `gate`/`reach`: active-set clean-up after the level gate / as soon as t = t_min (experiment, does not converge; see "The end game"); `--cleanup-biactive smaller\|both\|none`, `--cleanup-eps`, `--cleanup-dual-reg`, `--cleanup-mu` |
| `--t-update` | `every` | `solved`: t is lowered only when the barrier problem is solved (E_μ ≤ κ_ε·μ), not at every iteration; experiment |
| `--bounds` | `rows` | `vars`: r, δ, α ≥ 0 as variable bounds instead of slacked rows (experiment; the rows stay, free) |
| `--diag` | off | per iteration: where the dual residual sits and what blocked the step (see "The end game") |
| `--dual-reg C`, `--dual-reg-exp K` | 1e-6, 0 | δ_c = C·μ^K in every Newton system; C = 0: only on demand (the behaviour before) |
| `--classify FILE\|-` | off | per end-game iteration: MPCC multipliers γ = a − ξ(1−δ), ν = b − ξr and their sign classes on the near-biactive cells; FILE: the last point per cell (CSV) |
| `--follow-comp K`, `--follow-stall M` | 3, 20 | once μ and t have not moved for M iterations, a cell whose product row stays violated while ξ rises, K iterations in a row, gets its own level t_e = its product, lowered as the product falls; 0: off (see "End-game tools" and `endgame_literature.md` §11) |
| `--relax-cells THR` | 0 (off) | sign-driven per-cell relaxation (experiment, see `endgame_literature.md`); `--relax-shift 1e-4`, `--relax-persist 5`, `--relax-exp 0.7` |
| `mpirun -np R` | 1 | MPI ranks (build with `MPI=1`); tiles are split into R contiguous ranges |
| `--save-solution` | — | `.npz` with the instance, solution, run summary and μ-trace |
| `--quiet`, `--verbose 0/1/2` | | iteration table off; linear-algebra detail |
| `--start` | `smoothed` | two stages (stage 1 on rank 0, its weight and fields broadcast); `uniform`: Chambolle–Pock at `--alpha0` |
| `--start-gamma`, `--start-iters`, `--start-ftol` | 1e-3, 1000, 1e-9 | stage 1: Huber parameter; L-BFGS iterations (a safeguard: stage 1 stops on its convergence tests; 100 before 2026-10-11); stop once J falls by less than this (relative) over 5 iterations |
| `--start-solver` | `auto` | stage 1's Newton systems: `eigen` (sparse LDLᵀ), `mumps`; `auto` = MUMPS for N ≥ 128 |
| `--start-newton` | `primal` | stage 1's lower-level Newton: `primal` (Newton on the Huber energy) or `pd` (the primal-dual semismooth Newton of De los Reyes–Schönlieb–Valkonen, (SSN-1)–(SSN-3): the dual its own unknown, projected onto the unit ball where it enters H, H symmetrized) |
| `--start-huber` | `c2` | `--start ipm` only: the C² Huber (3.2) of De los Reyes–Schönlieb–Valkonen (`c2`), Definition 1's (`c1`, as stage 1), that Huber as rows max(γ, \|Ku\|) q − Ku = 0 with q its own unknown (`dual`), and those rows with q projected onto \|q\| ≤ 1 in their derivatives (`dualproj`) |
| `--start-ipm-init` | `pdhg` | `--start ipm` only: its start, the exact TV solution at `--alpha0` lifted (`pdhg`), or the smoothed lower level at `--alpha0` with q = h(Ku), feasible for `c1` and the dual rows (`smoothed`) |
| `--alpha0` | 0.7σ | the uniform start's weight, and stage 1's start |
| `--color` | `auto` | `always`, `never`; `auto` is plain under mpirun/srun or when not a terminal; `NO_COLOR` and `CLICOLOR_FORCE` are honoured |

**Output.** Homebrew-style sections (`ui.hpp`): `==>` headers per stage, `✔` / `Warning:` / `Error:` outcome lines, aligned key–value rows. Only the `CSV,` line is meant for machines (the slurm scripts read it).

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

**Tile backends, without the line search** (`--line-search none`; 4 threads, 1 process). Columns compare `mumps`, `sparse+mumps` (sparse with MUMPS fallback) and `sparse+dense` (the default without MUMPS, the configuration selected in `../archive/cpp_dd_toy`):

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
- With `--block-solver sparse --fallback dense`, the iteration logs of cameraman N=32, 4×4 (PCG and direct interface) are identical, value for value, to the configuration as tested in `../archive/cpp_dd_toy` (`dd_mpcc --host simple --reg global`).

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
- **IPOPT + MUMPS on the same NLP** (serial, `../archive/cpp_dd_toy/dd_mpcc --host mumps --size 256 --nsub 16 --max-iter 300`) also enters its restoration phase: 22 iterations, from iteration 3 on. It uses δ_w up to about 90, and reaches the gate at iteration 298: α\* = 0.0730, 28.55 dB, in 286 s and 1.2 GB.

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
    - **What a smaller t would sharpen is the stationarity certificate, not the image.** The theory is a limit t → 0 with tolerances O(t) (Kanzow–Schwartz 2015, Thm 2 of the published version, under MPEC-MFCQ; `endgame_literature.md` §1), which no finite t, and not the √t gate, satisfies exactly. At 10⁻⁴ the certificate is approximate: the residuals and the multiplier classification (`--classify`; S-stationary to solve accuracy except a few clusters, §5).
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

**Reference.** IPOPT with MUMPS on the same NLP (`../archive/cpp_dd_toy/dd_mpcc --host mumps`) takes 45–131 iterations on the six small instances. It reaches the same PSNR to 0.01 dB and α\* within 0.06% on cameraman (all four) and on mariposa N=32. On mariposa N=48 it stops at α\* = 0.0586 (24.25 dB). This IPM, with its line search, reaches α\* = 0.0557 (24.26 dB) there; without it, 0.0666 (24.12 dB).

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

**Performance and memory changes of 2026-10-07.** None of them changes a number: on the small six (4 ranks), cameraman N=32 and mariposa N=48 on 1 and 3 ranks, cameraman N=128 8×8 and N=256 16×16 (8 ranks) and mariposa N=512 16×16 (8 ranks, t_min = 5·10⁻⁴, with a restoration phase), the iteration logs are identical to those before. So are the `--diag` output and the `.npz` (byte for byte), the `--schur direct`, `--block-solver dense|sparse` and `--cleanup` runs, an OpenMP build on 1 and 3 threads, and a build without MUMPS.
- **Every rank stops at the first bad tile of any rank** (see the method).
- **PCG exchange without allocations:** the flat border array (see "Exchange").
- **The warm start** applies the gradient stencils directly instead of through triplet lists. The terms are added in the same order, so the result is the same to the last bit. N=512 on 8 ranks: 26 s → 5.7 s per rank (not in the reported wall, which times only the IPM).
- **The setup builds only the rank's part** (`TvMpcc::build_local`, `Partition` keeps one band per grid line). Before, every rank built global tables of the whole problem (the tiles of every variable, the copy maps, the stencils, the row owners) and then cut them down; that transient grew with N², not with 1/ranks.
- **Copies dropped:** MUMPS writes S_k straight into the tile's array, and takes the values straight into its own input array (no `mval_`); zero diagonals go only to the unknowns without a diagonal entry (24% fewer entries); the routed W_k triplets are freed after MUMPS's analysis and B_k is not built for MUMPS tiles; the restoration phase moves its start and end points instead of copying them; the global λ is no longer gathered (max |ξ| is an exact max over the ranks), the solution is gathered without indices, and ranks other than 0 drop the image after the setup.
- **Measured** (8 ranks, 1 thread each, the machine loaded by other jobs, so times ±10%; memory: peak physical footprint of the largest rank, which unlike RSS counts compressed pages, and the setup's peak RSS from a run stopped after the setup):

  | instance | setup peak | peak footprint | IPM wall | factorize / solve |
  |---|---|---|---|---|
  | cameraman N=128 8×8 (best of 3) | | | 3.78 → 3.07 s | 2.21 → 1.72 s / 1.22 → 1.09 s |
  | cameraman N=256 16×16, full run | 75 → 32 MB | 338 → 257 MB | 19.3 → 16.2 s | 8.46 → 7.09 s / 9.33 → 7.42 s |
  | mariposa N=512 16×16, full run | 251 → 88 MB | 976 → 756 MB | 96 → 87 s (job 110 → 90 s) | 57.2 → 49.1 s / 32.3 → 30.8 s |
  | mariposa N=1024 16×16, 5 its | 856 → 239 MB | 2507 → 2002 MB | 44 → 23 s (job 107 → 39 s) | 28.4 → 15.8 s / 9.6 → 3.5 s |

  - **What is left at N=1024 is MUMPS:** the live heap after 5 iterations is 185 MB of the 2 GB footprint; the rest is MUMPS's factors and its workspace during factorization (64×64-pixel tiles, t_k ≈ 70k). The ordering is what sets the factor size; this MUMPS build uses AMD whatever ICNTL(7) asks for, so a MUMPS with METIS is the next thing to try.
  - **Peak RSS is not a reliable measure here.** Under memory pressure macOS compresses pages, which RSS does not count: at N=1024 the peak RSS of the same binary varied between 1.3 and 1.7 GB. The footprint above does count them.

## tv_learn: weight maps, other operators, datasets

`tv_learn` learns the weight of the lower level

  u_s(α) = argmin_u ½uᵀA_s u − F_sᵀu + Σ_e α_e |(K_ξ u)_e|

over training images s = 1..S, minimizing the mean (`--loss mean`, the default) or the sum of ½‖u_s − u_clean,s‖². The upper level is that of `tv_dd`, generalized in three ways.

- **α on blocks** (`--block B`): α_e is constant on B×B blocks of cells and shared by all images; B = 0 (default) is one scalar.
  - With B ≥ 2 the tile cuts move to the nearest multiple of B, so no tile cuts a block.
  - B = 1 on one image is ill-posed (one weight per cell), so expect over-fitting.
  - The ridge is ½·10⁻⁴‖α‖², as for the scalar. There is no smoothness penalty between blocks yet: it would make every block on a tile edge a consensus variable.
- **Operators** (`--op`, `operators.hpp`):
  - `denoise`: A = I, F = f.
  - `inpaint`: A = diag(m + ε(1−m)), F = m∘f + ε(1−m)·f̄. Here m is the random mask of observed pixels (`--mask-frac` missing), f̄ the mean of the observed values and ε = `--mask-eps` (10⁻³). The ε term keeps the lower level strongly convex on the holes.
  - `deblur`: A = CᵀC, F = Cᵀf, with C a Gaussian blur (`--blur-sigma`) truncated at w = ⌈2σ_b⌉ and reflected at the border.
- **Directional TV** (`--reg dtv`): K_ξ = M_e·K with M_e = I − γξξᵀ and ξ = ∇v/√(|∇v|² + η²), where v is a smoothed guide.
  - The guide comes from `--guide clean|data` (default `clean`, an oracle). Held-out images always use their data.

### The formulation

The weight stays where `tv_mpcc.hpp` has it: h1 = A u − F + K_ξᵀ(α∘q), with δ ∈ [0, 1] and the product r(1 − δ) ≤ t. So with one image, A = I, K and one block, the NLP **is** `TvMpcc`'s, in the same order (variables, rows, nonzeros, and the order of every floating-point sum).

Scaling the dual instead (p = αq, which takes α out of h1) was considered and left out:
- it would make the h3 residuals O(α), so the gate inf_pr ≤ √t would be about 14× looser;
- it would need the continuation re-validated.

### Sharing

Which tiles read what follows from the stencils (`tiles_of_u`, `tiles_of_q`, `LearnPartition::alpha_tiles`):
- **u:** the h2 cells, as before. With a blur, also every h1 row within R = 2w nodes, so the border becomes a band 4w wide and tiles must be at least 4w cells wide.
- **q:** for DTV, the three nodes of its cell for both components.
- **α_b:** the nodes next to the block, in every image. A block inside one tile is a tile variable; any other block is a consensus variable with a copy per tile.

Tile ids are image-major (s·k² + tile). The construction checks that no row reads a variable of another tile.

### Checks (laptop, 2026-10-08)

**Against `tv_dd`.** `tv_learn` with `--start uniform` (and the other defaults) is bit-identical to `tv_dd` in the whole iteration log, status, iterations, α\* and PSNR:
- cameraman N=32 2×2, 4×4, 8×8; N=48 4×4; mariposa N=32, 48 4×4 (1 and 3 ranks);
- cameraman N=128 8×8 (8 ranks).

**Identity cases.** These are also bit-identical to plain denoising:
- an inpainting mask with no holes and ε = 0;
- a blur of radius 0;
- one block covering the image (`--nsub 1 --block 31`).

**Derivatives.** `--check-derivatives` (central differences on every column, plus Hessian directions) gives relative errors ≤ 3·10⁻⁹ for blocks (B = 1, 3, 4, 8), inpainting, deblurring, DTV and their combinations.

**Rank independence.** Runs are bit-identical for any number of ranks:
- a block map (cameraman N=33, 4×4 tiles, B = 8: 1, 2, 4 ranks);
- a dataset (S = 3, 3×3 tiles each, B = 16: 1, 2, 5 ranks, where ranks hold tiles of two images).

**Lower level.** Every run reports max|u_s − u_PDHG(α\*)|, the MPCC's u against the lower level solved at the learned map. It is ≤ 8·10⁻³ (√t = 10⁻²) in the runs below, except inpainting: 1.6·10⁻², where the holes are conditioned by ε = 10⁻³.

| instance (4 ranks, 4×4 tiles) | its | α\* | PSNR f → u | note |
|---|---|---|---|---|
| cameraman N=33, scalar / B=8 / B=4 | 23 / 49 / 49 | 0.0687 / mean 0.0793 / mean 0.0841 | 20.16 → 25.53 / 25.95 / 26.52 dB | training PSNR grows with the blocks |
| phantom N=64, B=8 (σ=0.1) | 98 | blocks 0.069–0.116 | 20.37 → 30.86 dB | flat blocks 0.082, blocks with an edge 0.075 on average |
| cameraman N=64, TV | 45 | 0.0700 | 20.39 → 27.00 dB | p = 748 |
| cameraman N=64, DTV, guide data / clean | 31 / 153 | 0.0765 / 0.0749 | 27.07 / 28.05 dB | `clean` is an oracle |
| cameraman N=64, inpaint 30%, σ=0.05 | 27 | 0.0343 | 9.84 → 27.37 dB | f = 0 on the holes |
| cameraman N=64, deblur σ_b=1, σ=0.02 | 74 | 0.0106 | 25.29 → 27.54 dB | p = 2875: the band of CᵀC |
| cameraman, mariposa, phantom N=49, B=16; held out: mariposa, phantom | 24 | 0.069–0.072 | train 26.80, held-out 26.60 dB | `plot_dataset.py` reads the report |

**Two copies of one image.** With the same noise and the mean loss, two copies give α\* = 0.070133, against 0.070181 for the image alone. This is not bitwise, nor should it be: the barrier terms of the image variables double while that of α does not, so the central path differs.

**DTV with γ = 0** gives α\* = 0.070165 against TV's 0.070181 (N=48). Its stencil keeps structural zeros and its warm start is PDHG, so it is not bit-identical either.

### Running it

```bash
TARGETS=tv_learn MPI=1 ./build.sh          # leaves tv_dd as it is
mpirun -np 4 ./tv_learn --size 64 --nsub 4 --block 8
mpirun -np 4 ./tv_learn --size 64 --nsub 4 --op deblur --blur-sigma 1 --sigma 0.02
mpirun -np 4 ./tv_learn --train kodak/train.txt --val kodak/val.txt --size 64 --nsub 2 \
    --save-report run/reports/report_k.json --save-solution run/sols/sol_k --tag k
(cd ../python && uv run python plot_dataset.py --run ../cpp/run/reports/report_k.json)
```

**Data.** `--train` / `--val` take a folder (sorted), a `.txt` manifest (one path per line) or one image; `--skip/--stride/--limit` select from the training list.
- Prepared sets live in `../images/<set>/` with `train.txt` and `val.txt`. The PNGs stay out of git; the lists are tracked.
  - `kodak`: train kodim01–16, held out 17–24.
  - `bsds64`: BSDS500 crops from `../python/make_datasets.py bsds`; 400 train (BSD400), 100 held out (the val folder, which contains BSD68).
  - `cartoon64`: the Google Cartoon Set 10k from `make_datasets.py cartoon` (CC BY 4.0); 1000 train, 500 held out, composited on white.
- The raw archives sit in `../images/` and are unpacked into `../images/raw/`, both ignored by git.
- `./run_dataset.sh SET SIZE [options]` runs a set and draws its figures (`./run_kodak.sh SIZE` = `run_dataset.sh kodak`). Each run goes to `../results/tvlearn_<SET>_N<SIZE>_<options>/`, never over an existing one unless `FORCE=1`.
- Training image i has noise seed `--seed` + i.
- Held-out image i has seed + 10⁶ + i.
- The mask seed is `--mask-seed` + the image's seed.

**The start (two stages by default; see "The two-stage method" below).** `--start smoothed`: stage 1 runs the smoothed bilevel method (`smoothed.hpp`) on the same block map, and stage 2 the MPCC from its result. `--start uniform` is `tv_dd --start uniform` (the uniform weight `--alpha0`, Chambolle–Pock with A = I and TV, else `pdhg.hpp` to `--warm-tol`), and the only start that reproduces `tv_dd` bit for bit with `tv_learn`'s other defaults; `--alpha-init FILE` starts from a given map (lower levels by PDHG).

**Held-out images** are evaluated by `pdhg.hpp` at the learned map; so are the training images, as the lower-level check.

**Outputs.**
- `--save-solution`: one `.npz` per image (`_s000` training, `_v000` held-out; one image: the file itself) with `plot_slurm.py`'s keys plus `alpha_map`, `mask`, `block`, `op`.
- `--save-report`: the JSON `plot_dataset.py` reads.
- The CSV line has `tv_dd`'s fields (α = the map's mean, PSNR = training means) plus `S, op, dtv, block, n_alpha, psnr_val`.
- `tv_learn.slurm` runs it on the cluster.

**End-game tools.** `--follow-comp` is on by default in both `tv_dd` and `tv_learn`; `--endgame-diag` and `--freeze-xi` are `tv_learn` only.
- `--endgame-diag FILE` (read-only; the run is unchanged). Each iteration prints one `[du]` line: where the dual residual sits, on the cells or off them (u, α, copies), with the cell and entry that hold its maximum, the number of cells above √t, and the cell with the largest ξ. At the end, FILE gets the last iteration's per-cell record as CSV: r, w = 1 − δ, ξ, a, b, γ, ν, the cell's residual and its entry, t_e.
- `--freeze-xi X` (opt-in; off by default; `--freeze-persist 3`). A cell whose ξ has stayed ≥ X for 3 iterations while t > t_min keeps that iteration's t in its product row, r(1 − δ) ≤ t_e, while t falls elsewhere. At the moment of freezing t_e = t, so the problem does not jump. The report prints the number of frozen cells and the range of their t_e. See "The end game at t_min = 10⁻⁵" in `endgame_literature.md` §11.
- `--follow-comp K` (default 3; 0: off; `--freeze-xi` turns it off, and the two cannot be asked for together). A cell is locked when, K iterations in a row, its product row is violated, ξ rises, the product does not fall and t has not moved. Its t_e is then set to r(1 − δ) − s, so its row holds at the current point, and afterwards only lowered, following the product; the cell is released once t_e ≤ t. The report prints the raises and the cells still above t at the end, with their largest t_e. On cameraman N=128 scalar at t_min = 10⁻⁵ it passes the gate for K = 2, 3 and 5 (199–544 iterations instead of the 3000-iteration limit), leaving 25–31 cells at t_e ≤ 2.5·10⁻⁴. The test is armed only after μ and t have not moved for `--follow-stall 20` iterations; unarmed, it fired on runs that converge anyway (mariposa 8×8: 212 cells left at up to 2.2·10⁻³). Armed (K = 3), it passes cameraman N=128 scalar at t_min = 10⁻⁵ in 179 iterations instead of the 3000-iteration limit, with 36 cells left at t_e ≤ 2.1·10⁻⁴, and leaves the six instances that converge anyway unchanged or nearly so. K = 5 is too slow. On the other stalling runs (cameraman N=64 at t_min = 10⁻⁶, mariposa N=128, cameraman N=256; all fail at the defaults) it passes the gate in 244–559 iterations with an equal or lower exact loss, leaving 49–399 cells at t_e ≤ 2.3·10⁻⁴; runs that do not stall are unchanged. The default since 2026-10-11 (`endgame_literature.md` §11, steps 3–5). A followed cell's certificate holds at its own t_e, not at t_min: the report's `followed cells` line gives how many are left and their largest t_e, to be reported with the S/M/C classes.

**Baselines** (one rank; no MPCC solve). Both print the result's upper-level loss with the *exact* TV lower level (`pdhg.hpp`), training and held-out PSNR and SSIM, and a `CSV-GRID` / `CSV-SMOOTH` line; the MPCC run prints the same exact loss (`upper-level loss: ... with the exact lower level at alpha*`), so the three are compared on one objective.
- `--grid A0:A1:N`: N log-spaced scalar weights, each image's lower level solved by PDHG; the best weight by the training loss.
- `--smoothed G` (`smoothed.hpp`): the standard smoothed bilevel method. Huber-TV lower level with parameter G by Newton (sparse LDLᵀ of A + K_ξᵀDK_ξ), gradient by the adjoint equation with the same factorization (checked against central differences: relative error ≤ 2·10⁻⁵), projected L-BFGS on α ≥ 0. Same operators, blocks and ridge as the MPCC. `--smoothed-tol 1e-6`, `--smoothed-iters 100`.
- First comparison, cameraman N=64: scalar weight, the three agree (α\* 0.07001 MPCC, 0.06984 grid, 0.07036 / 0.06991 / 0.07001 smoothed with G = 10⁻² / 10⁻³ / 10⁻⁴; exact loss 4.09057–4.09068, the MPCC lowest). 8×8 blocks: the smoothed method (G = 10⁻³) reaches an exact loss of 3.4948, the MPCC 3.5715 (default gate), 3.5266 (dual gate at 10⁻²·√t) and 3.5257 (t_min = 10⁻⁶, iteration limit): a different, worse stationary point of the nonconvex problem.
- The table that set the default (2026-10-09; smoothed ≤ 400 iterations, "from smoothed": `--alpha-init` with the map and a PDHG start; "tight": `--gate kkt --gate-scale 0.01`), cameraman 8×8 / 4×4 / 16×16, mariposa 8×8, phantom 8×8, BSDS 16 images 8×8: MPCC from the uniform start 3.5715 / 3.1323 / 3.8611 / 5.2607 / 1.6685 / 4.1745, smoothed 3.4948 / 2.9070 / 3.7816 / 4.9057 / 1.4117 / 4.1458, MPCC from smoothed 3.4950 / 2.9065 / 3.7822 / 4.9058 / 1.4113 / 4.1463, + tight gate 3.4938 / 2.9048 / 3.7819 / 4.9053 / 1.4108 / 4.1460. Superseded by the two-stage table below.

### The two-stage method

The upper level is nonconvex, and the MPCC is a local method: from a uniform weight it reaches a strongly stationary point of that weight's basin, which on block weights is a worse one. The method is therefore a hybrid (decided 2026-10-10, and so presented in the paper):

- **Stage 1 explores** (`smoothed.hpp`): Huber-TV lower level (`--start-gamma 1e-3`) by Newton, the gradient by the adjoint, projected L-BFGS on α ≥ 0 from the uniform `--alpha0`. It stops on its convergence tests: the projected gradient below 10⁻⁶ of its first value, J falling by less than `--start-ftol 1e-9` (relative) over 5 iterations, or a line search at the precision of J. `--start-iters 1000` is only a safeguard. Before 2026-10-11 the cap was 100, which bound on every block instance and once left stage 1 in a worse basin (mariposa 8×8: final exact loss 4.9055 against 4.9042 from the converged stage 1).
  - Its last lower-level solutions are the MPCC's start, lifted directly: u, the Huber dual q = K_ξu / max(γ, |K_ξu|) (so h1 holds to Newton's tolerance and |q| ≤ 1), r = |K_ξu|, δ = |q|, θ from q, α the stage-1 map. No PDHG or Chambolle–Pock solve.
  - In `tv_learn` stage 1 runs on every rank with the images split over the ranks (contiguous blocks, per-image parts summed in image order: the same bits for any number of ranks). 16 BSDS images 8×8: 6.56 / 4.20 / 2.36 / 1.53 s on 1 / 2 / 4 / 8 ranks (4.3×). `tv_dd` runs it on rank 0 and broadcasts the weight and the fields.
  - The Hessian pattern is fixed (explicit zeros keep 4 entries per cell), so it is analysed once per image.
  - Its Huber function is Definition 1 of De los Reyes–Valkonen (*The structure of optimal parameters*, and Def. 2.1 of the TGV paper) with the reciprocal parameter: `--start-gamma` γ̂ is their 1/γ (γ̂ = 10⁻³ is their γ = 1000).
  - The Hessian H = A + K_ξᵀDK_ξ is refilled in place (2026-10-12): its pattern, A's and a 3×3 block per cell, is fixed once, then each Newton iteration writes the values (H equals the old A + KᵀDK by products to 1.7·10⁻¹⁶ relative, for denoise, DTV and deblur; the stage-1 path changes at roundoff, e.g. cameraman 8×8: 252 instead of 256 iterations, final exact loss 3.493626 against 3.493737). Stage 1 alone, one rank, 30 iterations, 8×8 blocks, peak memory before → after: N = 128 59 → 51 MB (MUMPS), 41 → 33 (CG); 256 174 → 156, 89 → 66; 512 654 → 590, 270 → 204. The assembly time (in "factorizations") with CG: N = 512 40 → 3.4 s, 256 6.2 → 0.4 s.
  - `--start-newton pd` (2026-10-12): the same solutions (to 10⁻¹¹–10⁻¹⁶) and the same gradient (to 10⁻¹³; at convergence the dual is reset to ψ(u), so the adjoint uses the exact Hessian). Cold solves at N = 64, α = 0.07, primal → pd Newton iterations: denoise 22/54/143 → 9/14/21 for γ̂ = 10⁻²/10⁻³/10⁻⁴, DTV 21/61/116 → 9/14/19, deblur 76/209/300 → 15/20/23. Stage 1 at N = 256, 30 iterations (the same α): 594 → 233 Newton iterations with MUMPS, 631 → 332 with CG. It does not change the L-BFGS tail (the cost of cameraman 4×4), and the stage-1 path differs (cameraman 8×8: 336 against 252 iterations; final exact loss 3.493633 against 3.493626).
  - Its Newton systems (`--start-solver`): `eigen` (sparse LDLᵀ), `mumps` (positive definite; `auto` for N ≥ 128), or, in `tv_learn`, `pcg-jacobi` / `pcg-ic` (Newton–CG with inexact directions, forcing η = min(0.1, √‖∇E‖)). Stage 1 alone (`--stage1-only`), one denoised image, MUMPS against Newton–CG with Jacobi: N = 128 1.6 / 1.4 s, 256 9.9 / 3.4 s, 512 67 / 15 s (peak memory 654 / 273 MB). Deblurring is slower (N = 128: 244 / 54 s; N = 256 with CG failed in its first evaluation), and inpainting from u = f has lower-level failures (8–32 per run at N = 256).
- **Stage 2 refines and certifies**: the MPCC on the exact TV lower level. It removes the smoothing bias, lowers the exact loss in every instance below, and gives the multipliers and the S/M/C classes of the biactive cells (`--classify`). Use `--gate kkt` for that certificate.

Measured 2026-10-10 (N = 64, σ = 0.1, 4×4 tiles per image, BSDS one tile per image; 4 ranks; `--gate kkt --classify`; exact upper-level loss; times in s):

| instance | MPCC alone (`--start uniform`): loss, its, time | stage 1 alone (`--smoothed 1e-3 --smoothed-iters 100`, 1 rank): loss, time | two stages: loss, MPCC its, stage 1 + 2 | biactive S / M / C (two stages; MPCC alone) |
|---|---|---|---|---|
| cameraman scalar | 4.090574, 45, 0.63 | 4.090591, 0.35 | **4.090573**, 37, 0.37 + 0.37 | 1301 / 148 / 66; 1260 / 169 / 81 |
| cameraman 4×4 | 3.09280, 94, 1.16 | 2.91336, 1.80 | **2.90511**, 42, 1.87 + 0.51 | 1044 / 145 / 77; 1401 / 1 / 0 |
| cameraman 8×8 | 3.57184, 92, 1.30 | 3.49474, 1.48 | **3.49397**, 33, 1.58 + 0.37 | 1020 / 181 / 71; 1359 / 57 / 6 |
| cameraman 16×16 | 3.83105, 94, 1.17 | 3.78160, 1.12 | **3.78154**, 36, 1.20 + 0.38 | 1336 / 14 / 1; 1459 / 11 / 0 |
| mariposa 8×8 | 5.03412, 178, 2.19 | 4.90617, 1.83 | **4.90548**, 31, 1.91 + 0.35 | 762 / 169 / 84; 1090 / 3 / 1 |
| phantom 8×8 | 1.64198, 107, 1.26 | 1.41166, 1.29 | **1.41072**, 123, 1.32 + 1.47 | 650 / 4 / 4; 1370 / 0 / 0 |
| BSDS 16 images 8×8 | 4.17362, 88, 21.80 | 4.1457788, 6.81 | **4.1457770**, 39, 1.84 + 6.18 | 23538 / 283 / 3; 23719 / 313 / 8 |

- The two stages give the lowest exact loss everywhere. Over stage 1 alone the gain is small (0.0004–0.28%; PSNR the same to 0.01 dB): what stage 2 adds is exactness and the certificate. Over the MPCC alone it is 0.7–16% on block weights.
- MPCC iterations fall from 88–178 to 31–42 (phantom: 123 against 107).
- On one 64×64 image stage 1 is the larger part (1.2–1.9 s against about 0.4 s); on 16 images the MPCC is (1.8 against 6.2 s), and the two stages take less time in total than the MPCC alone (8.0 against 21.8 s).

**At the defaults since 2026-10-11** (stage 1 to convergence, t_min = 10⁻⁵; `--gate kkt --classify`, 4 ranks, one run at a time; exact loss with `--rof-iters 200000`, which equals 2M. The default 20k under-converges PDHG on cameraman block maps by up to 2·10⁻⁴ in the loss.) Columns: smoothed γ = 10⁻³ / 10⁻⁴ (converged, 1 rank) / two stages, then MPCC iterations, stage 1 + 2 in s, biactive S/M/C at ε = √t:

| instance | sm. 1e-3 | sm. 1e-4 | two stages | its | s | S/M/C |
|---|---|---|---|---|---|---|
| cameraman scalar | 4.0905911 | 4.0905744 | **4.0905736** | 57 | 0.5 + 0.8 | 214/0/0 |
| cameraman 4×4 | 2.906797 | 2.906294 | **2.902742** | 58 | 24.6 (cap) + 1.4 | 188/5/5 |
| cameraman 8×8 | 3.494887 | 3.493984 | **3.493737** | 86 | 5.6 + 1.8 | 196/0/0 |
| mariposa 8×8 | 4.905732 | **4.904316** | 4.905113 | 46 | 9.9 + 1.0 | 120/0/0 |
| BSDS 16 crops 8×8 | 4.1457788 | **4.1456925** | 4.1462153 | 97 (1 restoration) | 3.4 + 36.1 | 2525/0/0 |
| cameraman N=128 scalar | | | 14.77083 | **3000, gate not passed** | 3.6 + 1693 | 833/0/0 |
| cameraman N=128 8×8 | | | 11.93479 | 200 | 146 (cap) + 36 | 935/0/0 |

- The certificate holds at 10⁻⁵ everywhere except cameraman 4×4. The exact loss is within ±0.02% of the best smoothed point (lower on 3 of 5); PSNR is equal to 0.01 dB.
- **t_min = 10⁻⁵ breaks the end game on cameraman N=128 scalar**: δ > 0 in 2952 of 3000 iterations, 210 failed line searches each followed by a restoration phase, inf_du never ≤ √t. At 10⁻⁴ the same run takes 39 iterations. See "Known limitations". With `--follow-comp 3` (now the default) it passes the gate in 179 iterations, 36 cells left at t_e ≤ 2.1·10⁻⁴ (see "End-game tools").

**Alternatives for the start, tried and not kept** (2026-10-09/10; the options still exist):
- `--start scalar`: stage 1 on one weight, which then starts every block. Cheaper stage 1 (0.3–1.6 s), but the MPCC ends between the uniform and the two-stage results (cameraman 8×8: 3.5706; mariposa 8×8: 5.0490).
- `--start ipm`: the smoothed problem solved by this IPM in the lifted variables, q = h(K u) with h the C¹ Huber clipping (`--start-gamma`, `--start-tol`); also the polar and the Cartesian √ smoothings before it. With γ = 10⁻³ it reached the 3000-iteration limit; it converges for γ ≥ 0.03, but the MPCC then ends worse (cameraman 8×8 / 4×4 / 16×16, mariposa: 3.649 / 2.992 / 4.739 / 4.996).
- `--start ipm` with other Huber forms (`--start-huber`, 2026-10-12; cameraman 64×64 8×8, exact loss after stage 2, against 3.4936 for the two stages). Every one reached the 3000-iteration limit in stage 1 (35–468 restoration phases entered): c2 6.724 (γ̂ = 10⁻²) / 3.802 (10⁻³); c1 3.505 / 3.682; dual 3.509 / 4.753; dualproj 3.501 / **3.49366** (stage 2 in 103 iterations, the basin of the two stages). So the C² Huber's second derivative (−c³ = −10⁹ at γ̂ = 10⁻³ in the band of width γ̂²) is not what stops it, and the README's earlier "converges for γ ≥ 0.03" does not reach 10⁻². The projected Jacobian is the only form that finds the basin; it still does not converge. An explicit ball |q|² ≤ 1 with the dual rows was not tried: it is implied by the rows, active wherever |Ku| > γ, and fails LICQ and MFCQ there.
- `--start ipm --start-ipm-init smoothed` (2026-10-12, same instance): stage 1 from the smoothed lower level at α₀, inf_pr 2·10⁻¹⁴. The first step, accepted by the filter (θ_max = 10⁴ from a feasible point), puts inf_pr back to 2 (Huber rows) or 12–100 (dual rows); every run again reached 3000 iterations. Loss after stage 2, γ̂ = 10⁻² / 10⁻³: c1 4.099 / 83 821 (α diverged), c2 stopped by hand (stalled at steps of 10⁻⁷) / 3.528, dual 3.533 / 3.695, dualproj 3.515 / 3.49389. The infeasible start is not the cause; the full steps far outside the region of width about γ where the linearization holds are. Not tried: `--max-soc`, a smaller θ_max.
- `--start interior` (`--t0`): the start of the earlier MPCC codes (`bimpcc`): u = f, q = 0, δ = 0, r = |Ku|, θ the angle of Ku, a large first t. With `--t0` 1 (the default t) or 1000 it reached the iteration limit in 10 of 12 runs, α unbounded (up to 4·10⁵) or collapsing to 0; the two runs that reached the gate are useless (cameraman 4×4: 4.091, the scalar loss; BSDS: 18.7, α ≈ 0.001).
- A PDHG warm start of stage 1's Newton solves: rejected (a warm start for the warm start); stage 1 is made faster by its own solvers instead.
- Lower first μ (< 0.1), a slower t continuation, a tighter first t, a larger Huber γ in stage 1: no better or worse.
- Continuation in stage 1's γ (`--start-gamma0 G`, `--start-gamma-rate 0.1`, `--start-gamma-sigma 0.1`; Hintermüller–Wu 2015, Alg. 6.2), 2026-10-12. Cameraman 4×4, stage 1 alone, 1000 iterations each: from 10⁻¹ with σ = 0.1 it never left γ = 10⁻² (J and the projected gradient crawl there as at 10⁻³); with σ = 1, from 10⁻² / 10⁻¹, J = 2.91027 / 2.91111 in 30.4 / 26.7 s, against 2.91057 in 33.3 s at a fixed 10⁻³. The cost is not the small γ: at every γ, J is within 0.05% of its final value by iteration 200, and then L-BFGS crawls with the projected gradient at 5–10% of its first value. The large-γ weights are also larger (mean 0.28, max 1.23 at 10⁻², against 0.105 / 0.40), so a coarse level does not start the next one close.

**Limits.**
- The warm start is computed for every image on every rank, because consensus variables are held everywhere.
- A scalar α shared by more than about 1000 tiles makes the S̃_k exchange O(K²) (`plan_exchange_S`).

## Known limitations

- **The end game at t_min = 10⁻⁵ (the default since 2026-10-11) can fail to pass the dual gate.** Cameraman N=128 scalar: 3000 iterations, 210 restoration phases, 1693 s (39 iterations at 10⁻⁴); cameraman N=64 scalar at 10⁻⁶: 1685 iterations, 183 phases. The weight is right in both; the iterate cannot bring inf_du below √t. The cause is a few branch-locked cells (product row violated, ξ rising at a fixed t); `--follow-comp 3`, now the default, gives them their own t_e and passes the gate on all four stalling runs measured (179–559 iterations), at the price of 36–399 cells whose certificate holds only at t_e ≤ 2.3·10⁻⁴ (see "End-game tools").
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
    - The remedy the PCG failures point to is a coarse space (Lueg et al. eq. 22; implemented in `../archive/cpp_dd_toy`, where `linear` modes removed the growth with the tile count, but not the late-barrier growth).

## Provenance

The configuration was selected in `../archive/cpp_dd_toy` (see its README, "The MPCC under both hosts"):
- the paper-style IPM against IPOPT;
- one global δ against per-tile δ_k: the paper's Sec. 3.5 option failed on 4 of 6 instances;
- PCG with additive Schwarz against the direct interface solve.

The problem formulation is that of `../archive/cpp_minimal/problem.hpp`, ported to `nlp.hpp`. The Fischer–Burmeister variant, the IPOPT plumbing and the failed-solve banking were left out.
