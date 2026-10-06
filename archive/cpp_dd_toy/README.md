# cpp_dd_toy: Lueg et al.'s Schur-complement decomposition inside IPOPT, on a toy problem

This is a small, self-contained C++/IPOPT implementation of the linear algebra in

> L. R. Lueg, M. L. Bynum, C. D. Laird, L. T. Biegler, *Domain decomposition
> preconditioners for Schur complement systems arising in structured nonlinear
> optimization problems*, Optim. Eng. 27 (2026) 555–585.

It is applied to a 2D semilinear optimal control problem instead of the MPCC. The
point is to see what the decomposition does inside IPOPT without the MPCC
machinery: complementarity, μ-coupled relaxation, the α and cross-point peels of
`../cpp_minimal`. The tile layout is the same as in the image problems.

Read the files in this order:

| file | what it is |
|---|---|
| `problem.hpp` | The NLP as an `Ipopt::TNLP`, written in the paper's form (1): local variables, local copies, linking rows. `kkt_owner()` says which tile owns each KKT unknown; `coarse_hints()` gives the coarse space the side and position of each y. |
| `schur_dd.hpp` | The paper's Sec. 2–3: routing, S_k, S, the interface solve, inertia, the coarse basis. Each step is commented with its equation number. It knows nothing about IPOPT. |
| `blocks.hpp` | Factorizations of W_k (sparse LDLᵀ in a KKT-aware order, or dense Bunch–Kaufman) and of S. |
| `precond.hpp` | One-level Schwarz blocks, the coarse term (eq. 22) and PCG. |
| `simple_ipm.hpp` | A second host: a minimal primal-dual IPM in the paper's setting (no line search, per-tile δ_k, C = 0). See "Two hosts". |
| `ipopt_bridge.hpp` | Makes `SchurDD` IPOPT's linear solver (`SparseSymLinearSolverInterface` + `AlgorithmBuilder`). Same pattern as `../cpp_minimal/ipopt_bridge.hpp`. |
| `main.cpp` | CLI, IPOPT options, summary (one machine-readable `CSV,` line per run). |
| `scaling.sh` | The scaling study (see "Scaling to large problems"). |
| `mpcc_main.cpp` | `dd_mpcc`: the MPCC of `../cpp_minimal` (included unchanged) solved with `SchurDD` under IPOPT, IPOPT+MUMPS or the simple host. See "The MPCC under both hosts". |

## The problem

```
min  ½h² Σ (u − u_d)²  +  ½αh² Σ f²
s.t. 4u_g − Σ_nb u_nb + h²(κ u_g³ − f_g) = 0      every node g of an N×N grid (u = 0 on ∂Ω)
     −fmax ≤ f ≤ fmax
```

- u_d = amp·sin(πx)·sin(2πy).
- With the defaults (κ=1, α=1e-4, fmax=40, amp=2) about half of the controls end on a bound.
- **κ > 0:** in every case I tried, IPOPT never needed an inertia correction.
- **κ < 0:** gives a focusing nonlinearity, and IPOPT regularizes often. For example, `--kappa -5 --amp 5` needs 54 corrections in 53 iterations.

## The decomposition (paper eq. 1)

```
   tile A          tile B          A row next to the cut references u across it.
 · · · ·  ┆  · · · ·              That u becomes a complicating variable y_g.
 · · · a ─┼─ b · · ·              Every tile that uses it gets a local copy,
 · · · ·  ┆  · · · ·              and each copy gets a linking row
                                       copy − y_g = 0                 (1d)
```

- The grid is cut into `--tiles P` × P tiles.
- y is u at every node on either side of a cut. Nodes on a tile edge have 2 copies; nodes next to a tile corner have 3.
- The objective term of u_g sits on the home tile's copy, so y appears only in the linking rows. This is exactly the paper's structure: R_k = [0, 0, 0, −N_k, 0].
- `--tiles 1` has no copies. That is the monolithic problem.

IPOPT's KKT matrix has the unknowns `[x | λ]` in TNLP order. All rows are equalities, and f has only variable bounds, so there are no slack rows. Grouped by tile it is the arrowhead (5):

```
⎡ W_1            B_1ᵀ ⎤      W_k: the tile's own KKT block (6), including the copies
⎢      ⋱          ⋮   ⎥           and their linking rows (the "I" in block (2,4))
⎢          W_K   B_Kᵀ ⎥      B_k: the −1 of each linking row on y  (paper: R̄_k = N_kᵀR_k)
⎣ B_1  ⋯   B_K    C   ⎦      C:   y–y block = δ_w·I  (paper: 0; see "Differences")
```

| paper | code (`schur_dd.hpp`) |
|---|---|
| (10) Schur system, (11) back-substitution | `SchurDD::solve` |
| (14) S_k = −R̄_k W_k⁻¹ R̄_kᵀ, (15) S = Σ N_k S_k N_kᵀ | `SchurDD::factorize`, `assemble_S` |
| (12) S·v by one back-solve per tile | `apply_S` with `--matvec backsolve` |
| xSC: form S and factorize it | `--schur direct` |
| PCG with (17) block Jacobi, (19) additive Schwarz, (21) diagonally assembled | `--schur pcg --precond bj\|as\|asd`; `Schwarz` class |
| (24) In(W_k) | Bunch–Kaufman (LAPACK `dsytrf`) on each W_k |
| (25) In(S) | direct: Bunch–Kaufman on S; pcg: curvature test pᵀSp ≤ 0 in CG |
| (26) In(S_k) = (p_k,0,0) ⇒ In(S) = (p,0,0) | `--local-inertia` |

## Build and run

```bash
./build.sh                    # Homebrew IPOPT + Eigen; LAPACK from Accelerate (macOS)
OMP=1 ./build.sh              # with OpenMP over the tiles (then set OMP_NUM_THREADS)
                              # (also builds dd_mpcc when ../cpp_minimal is present)
export OMP_NUM_THREADS=1

./dd_toy --N 32 --tiles 1 --solver mumps        # monolithic reference
./dd_toy --N 32 --tiles 4 --solver mumps        # same NLP with copies, IPOPT's MUMPS
./dd_toy --N 32 --tiles 4                       # Schur DD, direct
./dd_toy --N 32 --tiles 4 --schur pcg --precond as
./dd_toy --N 12 --tiles 3 --check --verbose 3   # see everything (below)
./dd_toy --help
```

On Linux, activate a conda env with ipopt, eigen and lapack, or set `LAPACK_LIBS`.

**Reading `--verbose`:**
```
[dd] fact   7  C_yy=0.00e+00  Σ#neg(W_k)=320  #neg(S)=0  | dense #neg(A)=320
[dd]      S_4   p_k=28  eig ∈ [-1.688e-12, 4.826e+04]  #neg=0  #(|λ|≤1e-10·max)=4     (level 3)
[dd]      solve: pcg its=  41  rel.res=8.7e-11                                          (level 2)
[dd]      check: ‖u_dd − u_dense‖/‖u_dense‖ = 1.1e-13   #neg: dd 320, dense 320       (--check)
```

- One `fact` line per matrix IPOPT hands over.
- `C_yy` is IPOPT's δ_w. On the very first factorization it is 1: that is IPOPT's least-squares multiplier estimate, which also goes through this solver.
- IPOPT wants #neg = number of constraints. If the count is wrong it raises δ_w and refactorizes, so the next `fact` line shows the new δ_w.
- `--check` builds the full KKT matrix densely on every factorization. It compares the DD solve and inertia with an eigendecomposition and an LU. Use it only at small sizes; the default limit is dim ≤ 4000.

## What it shows (measured, macOS arm64, IPOPT 3.14.20)

**1. The decomposition is exact.**
- `--tiles 1 --solver mumps`, `--tiles 4 --solver mumps` and `--tiles 4` (direct) reach the same objective, 1.306357919523e-01 at N=32. With 4×4 tiles: p=348 complicating variables, max p_k=60, max t_k=284.
- The IPOPT iteration log of DD-direct matches MUMPS's to every printed digit. This holds for the convex default and for the nonconvex `--kappa -20 --amp 5` (35 iterations, 16 inertia corrections) and `--kappa -5 --amp 5` (53 iterations, 54 corrections).
- At N=12 with 3×3 tiles, `--check` gives a solve error of 1e-13 to 4e-13 against the dense LU and the same inertia as the dense eigendecomposition on every factorization.

**2. In every direct run, wrong inertia came from S, never from a W_k.**
- Σ#neg(W_k) always equalled the constraint count. Every inertia correction IPOPT made was because S was not positive definite (`S not PD` = `wrong inertia` in the summary).
- So the nonconvexity lives in the interface. This is the part PCG has to detect.

**3. S_k is only positive *semi*definite.**
- S_k is the Hessian of tile k's value function in its copies.
- At an interior tile corner, one row of tile A references two foreign copies (from the right tile and the tile below). Neither copy appears anywhere else in A. Moving them in opposite directions costs A nothing, so S_k has an exact null vector there.
- `--verbose 3` shows it: the number of zero eigenvalues is 1, 2 or 4, which is the number of interior corners of the tile (corner, edge and centre tiles). The "negative" eigenvalues are round-off at about 1e-13·‖S_k‖.
- This is the same structure that `../cpp_minimal` peels off as "cross points".
- Consequences:
  - **block Jacobi (17)** uses the singular S_k and does badly: at N=32, 4×4, it is worse than no preconditioner (see the table). Raising the clipping floor for its non-SPD blocks from 1e-8 to 1e-3 did not help, so the clipping alone is not the cause; S_k also carries only its own tile's share of each y.
  - **the local check (26)** fails at every factorization. With `--local-inertia`, IPOPT regularizes once per iteration on a problem that needs no regularization: 14 corrections in 14 iterations. The paper calls (26) conservative; here it is always wrong.
- AS (19) and ASd (21) don't have this problem: S̃_k and the diagonal swap bring in the neighbours' contributions.

**4. Preconditioners**, default problem (κ=1). (These two tables were measured with the first version: dense blocks, default objective scaling. The scaling section below repeats the comparison at N=256 with everything converged.) PCG relative tolerance is 1e-10 with a 2000-iteration cap. N=32 with 4×4 tiles:

| solve | IPOPT its | PCG its mean / max | unconverged solves | wall |
|---|---|---|---|---|
| direct (xSC) | 12 | — | — | 0.18 s |
| pcg none | 17 | 1714 / 2000 | 76 | 2.7 s |
| pcg bj | 55 | 1996 / 2000 | 431 | 19 s |
| pcg as | 12 | 90 / 431 | 0 | 0.23 s |
| pcg asd | 12 | 316 / 1708 | 0 | 0.38 s |

N=64 (all 11 IPOPT iterations, objective 1.310496495350e-01):

| tiles | p | max p_k | max t_k | direct | pcg as: its mean / max | pcg asd: its mean / max |
|---|---|---|---|---|---|---|
| 4×4 | 732 | 124 | 956 | 2.8 s | 90 / 403, 3.0 s | 458 / 2000 (1 unconverged), 3.7 s |
| 8×8 | 1596 | 60 | 284 | 0.9 s | 263 / 1513, 2.8 s | 704 / 2000 (2 unconverged), 6.2 s |

- Going from 16 to 64 tiles makes the blocks cheaper (direct: 2.8 s → 0.9 s), but PCG needs about 3× more iterations.
- This is the one-level scaling problem the paper raises before eq. (22). A coarse-space term (two-level AS) is what fixes it, and it is not implemented here.

- AS < ASd as expected. Here AS ≠ ASd because the pairs of y along one cut are shared by both tiles at that cut (the paper's Fig. 3a situation).
- The PCG iteration counts grow sharply in the late IPM iterations, as Σ for the active bounds blows up.
- Unconverged interface solves change the IPOPT path. At N=64 with no preconditioner, IPOPT needs 49 iterations instead of 11.

**5. The PCG inertia check is not rigorous, and on nonconvex problems that changes the IPM** (`--kappa -5/-20 --amp 5`, N=32, 4×4).
- The paper's check only refuses a matrix when CG meets pᵀSp ≤ 0. CG often solves an indefinite S without ever meeting such a direction.
- `--kappa -5`:
  - Direct makes 54 inertia corrections in 53 iterations.
  - PCG-AS makes none, finishes in 14 iterations and reaches the same solution (|Δu| ≈ 1e-5).
- `--kappa -20`:
  - Direct converges in 35 iterations.
  - PCG-AS and PCG-ASd do not converge in 150 iterations (each made 9 corrections).
- `--pcg-inertia exact` is a diagnostic that factorizes S only to count its negative eigenvalues. With it, PCG-AS reproduces the direct trace line for line in both cases. So the difference comes entirely from the missing inertia information, not from the inexact interface solves.
- `--check` in PCG mode audits this. At N=12, 3×3, `--kappa -20 --amp 5`, 4 factorizations that PCG accepted had the wrong inertia according to the dense eigendecomposition. The worst PCG solve in that run was off by 77% against the dense LU (`--check` doesn't report which solve it was).
- A breakdown can also happen in IPOPT's iterative refinement, after the matrix was already accepted. IPOPT aborts if the solver refuses there, so the bridge keeps the CG iterate and counts it ("breakdowns AFTER the matrix was accepted" in the summary).

## Scaling to large problems

Three things limited the first version to N ≲ 64:
- dense W_k;
- a dense p×p border block C and a dense assembled S;
- one-level preconditioners whose PCG iterations grow with the tile count.

Now:
- **Sparse W_k** (`blocks.hpp`): unpivoted LDLᵀ in the KKT-aware level order. It is exact: the DD-direct IPOPT log is identical to MUMPS's.
- **No dense p×p objects:**
  - only the small S_k are stored;
  - S·v and the AS blocks S̃_k are assembled from them through a border → (tile, position) map;
  - S itself is assembled (sparse) only for the direct solve and the exact-inertia diagnostic.
- **OpenMP over tiles** (`OMP=1 ./build.sh`, `--threads n`).
- **A coarse space** (eq. 22) built from the interface faces, i.e. the y's shared by the same set of tiles. `--coarse`:
  - `faces`: one constant per face;
  - `sides`: faces split by the side of the cut, which adds the jump across the cut (a normal derivative);
  - `linear`: sides plus linear functions along each face.

  `sides` and `linear` need the side and position of each y; the problem supplies them as hints. `--coarse-mode balanced` (default) applies the coarse term in balancing Neumann–Neumann form; `additive` is eq. 22 literally.
- **`--obj-scaling auto`.** The objective carries h², so on fine grids IPOPT's absolute tolerances are met far from the optimum. At N=1024 with the default scaling, IPOPT stopped after 9 iterations at objective 0.13349 instead of 0.13119; even N=256 stopped at 0.131234 instead of 0.131176. `auto` scales the objective by 1/h² inside IPOPT. All numbers below use it.

Measured with `./scaling.sh` (macOS arm64, 8 threads unless noted, PCG on AS, tol 1e-8, PCG tolerance 1e-10 with a 2000-iteration cap). Every run converged to the same objective for its N.

**Tile count at fixed N=256** (mean / max PCG iterations per solve, and wall time):

| tiles | one-level AS | + faces | + sides | + linear |
|---|---|---|---|---|
| 4×4 | 397 / 1126, 15 s | 292 / 840, 17 s | 617 / 2000, 29 s | 442 / 2000, 23 s |
| 8×8 | 1430 / 2000, 34 s | 1112 / 2000, 43 s | 1115 / 2000, 45 s | **258 / 865, 12 s** |
| 16×16 | 1824 / 2000, 75 s | 1306 / 2000, 71 s | 1431 / 2000, 100 s | **297 / 1022, 18 s** |
| 32×32 | 1995 / 2000, 578 s (56 IPOPT its) | 1571 / 2000, 215 s | 1496 / 2000, 247 s | **167 / 721, 27 s** |

- One-level AS degrades with the tile count until every solve hits the cap. At 32×32 the unconverged interface solves even push IPOPT from 16 to 56 iterations.
- With `linear`, the PCG count no longer grows with the tile count. Constants per face (`faces`, `sides`) are not enough.
  - The reason: eliminating the control through the PDE leaves an interface operator of fourth order (α|Δu|² in the reduced objective). Its "rigid" modes are not just constants: they include linears, and the slope across the cut.
- At 4×4 the coarse space does not pay. The tiles are big, the one-level method is already good, and the extra work per iteration costs more than it saves.

**Weak scaling** (32×32-node tiles, AS + linear coarse):

| N | KKT unknowns (p) | tiles | IPOPT its | PCG mean / max | factorize | interface PCG | total | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 64 | 13k (252) | 2×2 | 14 | 16 / 25 | 0.08 s | 0.10 s | 0.2 s | 36 MB |
| 128 | 55k (1.5k) | 4×4 | 16 | 850 / 2000 | 0.4 s | 9.1 s | 9.7 s | 152 MB |
| 256 | 225k (7k) | 8×8 | 16 | 258 / 865 | 2.4 s | 7.7 s | 10.4 s | 472 MB |
| 512 | 0.91M (30k) | 16×16 | 16 | 381 / 1348 | 11.4 s | 52.7 s | 65 s | 1.6 GB |
| 1024 | 3.6M (123k) | 32×32 | 16 | 407 / 1570 | 54.0 s | 304.0 s | 365 s | 5.5 GB |

- N=1024 (3.6M KKT unknowns, 123k complicating variables, 1024 tiles) solves in 6 minutes on a laptop. IPOPT's own work is negligible (under 1%).
- The IPOPT iteration count is flat in N, and the PCG count stays in a few hundred. N=128 is an outlier, as is 4×4 above: with few tiles the coarse space hurts more than it helps.
- **The remaining cost is the barrier, not the decomposition.** PCG needs about 20 iterations per solve early in the IPM. Mid-path (μ ≈ 1e-6), when many controls approach their bounds, it needs 500–900. The barrier term Σ then varies over many orders of magnitude between active and inactive regions: a high-contrast problem that fixed geometric coarse modes don't capture. Coefficient-adaptive coarse spaces (GenEO-type) are the known remedy. A naive version (eigenvectors of S_k v = λ S̃_k v) made things worse here, and was dropped. Loosening the PCG tolerance (1e-4 … 1e-8) did not help either: IPOPT's iterative refinement adds solves.

**Strong scaling** (N=512, 16×16 tiles, AS + linear coarse):

| threads | total | factorize | interface PCG | speedup (total / factorize) |
|---|---|---|---|---|
| 1 | 266 s | 67.2 s | 194.5 s | 1.0 / 1.0 |
| 2 | 148 s | 35.1 s | 110.7 s | 1.8 / 1.9 |
| 4 | 94 s | 20.8 s | 71.1 s | 2.8 / 3.2 |
| 8 | 81 s | 15.1 s | 63.9 s | 3.3 / 4.4 |

- The factorization phase (W_k, S_k) scales well.
- PCG scales less well. Each iteration is a few small per-tile products plus the gathers and the serial coarse solve, so synchronization dominates.
- The laptop has 4 performance cores, which caps the gain beyond 4 threads.

## Two hosts: IPOPT and a paper-style IPM

The paper does not use IPOPT. It uses its own IPM in parapint, with MA27 blocks, *"a simple fraction-to-the-boundary rule"* for the steps, and per-partition regularization δ_k as an option (Sec. 3.5). `--host simple` (`simple_ipm.hpp`, about 350 lines) is that kind of IPM, driving the same `SchurDD`:

- primal-dual barrier with the bound multipliers eliminated; IPOPT's monotone μ rule and scaled error measures;
- fraction-to-the-boundary steps; no line search, no restoration;
- regularization only on the tile primals:
  - `--reg local`: when In(W_k) is wrong (eq. 24), raise only δ_k;
  - `--reg global`: one δ for all tiles;
  - in both modes, when S is not positive definite, every tile is raised;
- nothing on the complicating variables: C = 0, as in eq. (5);
- In(S) only as far as the interface solve can tell: exactly in direct mode, by the PCG curvature test (or `--pcg-inertia exact`) in PCG mode.

Its log prints the relative residual of each step against the true KKT matrix (`lin.res`). There is no iterative refinement, so this column shows how accurate the linear algebra really is.

**Same problem, same linear algebra, two hosts** (N=32, 4×4, `--amp 5 --obj-scaling auto`, iteration limit 300; the simple host starts from IPOPT's least-squares multipliers):

| κ | interface | IPOPT | simple, local δ_k | simple, global δ |
|---|---|---|---|---|
| 1 (amp 2) | direct | 14 its | 15 its | 15 its |
| −5 | direct | 41 its, 42 corrections | 43 its | 43 its |
| −5 | pcg as | 14 its, 0 corrections | 18 its | 18 its |
| −5 | pcg as + linear coarse | 14 its | 18 its | 18 its |
| −20 | direct | 92 its, 91 corrections | 80 its | 144 its |
| −20 | pcg as | **fails** (300 its) | **fails** (300 its) | **fails** (83 its) |
| −20 | pcg as + linear coarse | **fails** (300 its) | **fails** (300 its) | **fails** (300 its) |
| −20 | pcg as, `--pcg-inertia exact` | 87 its, 88 corrections | 93 its | 93 its |

All converged runs reach the same objective to about 1e-8 relative. What this shows:

- **At N=32 the PCG failure at κ = −20 is the method, not IPOPT.** Both hosts fail with the paper's curvature check. Both work once S's inertia is known exactly.
- **The κ = −5 shortcut is not IPOPT either.** With PCG, both hosts see no S defect and finish in 14–18 iterations instead of about 42.
- **On the toy, local δ_k helps a little.** W_k rarely has the wrong inertia; almost every refusal is S not positive definite, which raises every tile in both modes. Where the modes differ (κ = −20, direct), local takes 80 iterations against 144 for global. On the MPCC it is the other way round (next section).
- **At N=256 the picture changes** (8×8 tiles, AS + linear coarse, iteration limit 300; IPOPT rows measured before the per-tile dense fallback existed, when a zero pivot still went to IPOPT as "singular"). For κ = −20, every mode converges to the same objective, 0.9659958. Unlike at N=32, the curvature check now catches the indefinite S (89 negative-curvature refusals). So the N=32 failure is a property of that instance, not a general verdict on the check:

  | κ = −20, N=256 | IPOPT its | refused factorizations | wall |
  |---|---|---|---|
  | direct (exact In(S)) | 154 | 97 | 114 s |
  | pcg, curvature (paper) | 146 | 95 | 84 s |
  | pcg, `--pcg-inertia exact` | 230 | 227 | 463 s (164 s factorizing S) |
  | pcg, `--pcg-inertia free` (IPOPT's inertia-free test) | 152 | 190 | 90 s |
  | pcg, curvature, simple host (local δ_k; current code) | 186 | 58 | 87 s, 2.9 GB peak (62 dense tile fallbacks) |

  For κ = −5 (amp 5) at N=256, *nothing* converges within 300 iterations, not even IPOPT with the exact direct solve (267 singular S factorizations along the way). That is the nonconvex problem itself at this resolution, not the decomposition.
- **The simple host is fragile.** Without a line search or restoration phase, a large δ (up to about 1e10) gives steps of length about 1e-6, and the iteration can run away (dual infeasibility 1e10; the global PCG run at κ = −20 failed this way). It is also sensitive to its start: before it used least-squares multipliers, the κ = −20 runs took 84 (direct) and 66 (exact) iterations, and AS + linear coarse converged in 293. IPOPT's globalization is what keeps its direct runs safe at κ = −20.

## The MPCC under both hosts

`dd_mpcc` takes the TV-weight-learning MPCC from `../cpp_minimal/problem.hpp` (consensus form, μ-coupled Scholtes continuation, level gate) and solves it with this folder's linear algebra. The problem, image loader and partition are included unchanged, as are the continuation and IPOPT settings of `../cpp_minimal/main.cpp`. Three ways to run it:

```bash
./dd_mpcc --size 32 --nsub 4 --host mumps            # IPOPT + its own MUMPS: the reference
./dd_mpcc --size 32 --nsub 4                          # IPOPT + SchurDD (direct)
./dd_mpcc --size 32 --nsub 4 --host simple --reg global [--schur pcg --precond as]
```

To run the MPCC, `SimpleIPM` gained:
- general inequality rows, via slacks in IPOPT's `[x | s | λ_c | λ_d]` layout, so the MPCC's owner map works unchanged;
- IPOPT's μ-schedule parameters (`cpp_minimal` uses barrier_tol_factor 1000, decrease 0.7, power 1.1), plus acceptable-level stopping;
- IPOPT's least-squares initial multipliers;
- a call to the problem's `intermediate_callback` every iteration, so t follows μ and the level gate can stop the solve, exactly as under IPOPT;
- dual regularization δ_c when a tile has too *few* negative eigenvalues. Below 4×4-cell tiles a tile's own Jacobian can be rank-deficient, and then no δ_w fixes the count.

**Exactness first.** With dense tile blocks, IPOPT + SchurDD reproduces IPOPT + MUMPS line for line (cameraman N=32, 4×4: 47 iterations, α\* = 0.070394, 48 identical log lines). The sparse tile blocks needed two additions on the MPCC (`blocks.hpp`):
- **A dense fallback per tile.** The static level order meets exact zero pivots: two rows leaning on the same L1 variable give a rank-deficient −J H⁻¹ Jᵀ block, which only 2×2 pivots handle. Such a tile is refactorized with dense Bunch–Kaufman for that one matrix.
- **A pivot test that also triggers the fallback:** |d_i| < 1e-14 × (largest entry of row i of W_k). Without it (exact zeros only), the IPOPT log drifted from MUMPS's (48 vs 47 iterations).

With both, the IPOPT path has the same 47 iterations and the same α\* as MUMPS. The logs agree to the printed digits except round-off in the 7th–8th digit late in the solve. About 38% of the MPCC's tile factorizations take the fallback; the toy's convex runs take none, and its nonconvex κ = −20 runs about 1%.

A first version compared each pivot with the largest pivot of the whole block. It made the MPCC log bit-identical to MUMPS's, but late in the barrier W_k legitimately spans many orders of magnitude (Σ on active bounds), so on the toy it sent healthy blocks to the dense path. At N=128, κ = −20 it refused 745 of 3600 tile factorizations, and each dense tile (t_k ≈ 3500, about 100 MB) was built on every thread at once: 2.4 GB peak and 100 s, against 1.1 GB and 24 s now. Dense fallbacks are also built one at a time (an OpenMP critical section), and a tile's dense factor is freed as soon as its sparse factorization succeeds again. Convex runs never fall back and keep their normal footprint (N=128: 154 MB).

`../cpp_minimal/tv_learn` itself takes a different path on the same instance: 34 iterations, α\* = 0.070402. Its linear algebra is not exact in IPOPT's sense: unpivoted blocks, the peel, and CG accepted at a residual of 1e-2.

**IPOPT vs the paper-style host** (level gate at t = 10⁻⁴, iteration limit 500, 4 threads; peak memory in brackets):

| instance | IPOPT + MUMPS | simple, global δ, direct | simple, global δ, PCG (AS) | simple, local δ_k (Sec. 3.5) |
|---|---|---|---|---|
| cameraman N=32, 4×4 | 47 its, α\* 0.070394, 26.40 dB [32 MB] | 29 its, 0.070432, 26.40 dB [245 MB] | 29 its, 0.070432 | **fails** (60 its) |
| cameraman N=32, 2×2 | 52 its, 0.070431 [31 MB] | 28 its, 0.070395 [1.0 GB] | 28 its, 0.070395 | 31 its, 0.070395 |
| cameraman N=32, 8×8 | 56 its, 0.070448 [30 MB] | 38 its, 0.070446 [97 MB] | 28 its, 0.070450 | **fails** (142 its) |
| cameraman N=48, 4×4 | 45 its, 0.070214, 26.68 dB [48 MB] | 31 its, 0.070171, 26.67 dB [1.1 GB] | 31 its, 0.070171 | 26 its, 0.070171 |
| mariposa N=32, 4×4 | 72 its, 0.065937, 23.43 dB [43 MB] | 45 its, 0.065937, 23.43 dB [239 MB] | 48 its, 0.065930 | **fails** (370 its) |
| mariposa N=48, 4×4 | 131 its, **0.058578, 24.25 dB** [59 MB] | 48 its, 0.066556, 24.12 dB [1.1 GB] | 48 its, 0.066556 | **fails** (25 its) |

What this shows:

- **The paper-style host works on the MPCC, with one shared δ.** With global regularization it reaches the level gate on every instance, with the direct and with the PCG interface. It needs fewer IPM iterations than IPOPT (28–48 against 45–131), because IPOPT spends iterations in its restoration phase and in line-search trials early in the continuation: on cameraman N=32 it enters restoration at iteration 2 (iterations 3r–11r) and takes up to 33 line-search trials per step until about iteration 15.
- **Per-tile regularization (Sec. 3.5) fails on the MPCC** in 4 of 6 instances. Unlike the toy, the MPCC's tile blocks themselves often have the wrong inertia (71 refusals in the first cameraman run): the nonconvexity lives inside the tiles. Regularizing only those tiles leaves their neighbours' steps unchanged, the iterate oscillates, and δ_k climbs until the step collapses (δ ≈ 10³⁹, steps ≈ 10⁻⁶).
- **Same answers, except one.** On five instances both hosts reach the same α\* and PSNR to the reported digits. On mariposa N=48 they stop at different points of this nonconvex problem: IPOPT at α\* = 0.0586 (24.25 dB), the simple host at 0.0666 (24.12 dB). Without globalization the simple host has no mechanism that prefers the better one.
- **The PCG curvature check suffices here.** PCG takes the same number of iterations as direct (or close to it), and its negative-curvature detections match the direct solve's S defects: 4 and 4 on cameraman 4×4, 17 and 15 on mariposa N=32, 10 and 10 on mariposa N=48.
- **Time and memory are not the point at these sizes, and they show the weak spot.** IPOPT + MUMPS takes 0.5–3 s and 30–60 MB. The DD runs take 1.5–48 s and up to 1.1 GB, almost all of it from the dense tile fallbacks: about 38% of the MPCC's tile factorizations need one, and a dense tile costs t_k² doubles (2×2 tiles at N=32, or 4×4 at N=48, have t_k of several thousand). The proper fix is a *pivoted* sparse factorization per tile, i.e. MA27/MA57/MUMPS as the paper uses; `../cpp/mumps_block.hpp` has such a backend but needs COIN's MUMPS headers, which this machine lacks. Until then: keep tiles small (memory grows like t_k²).

## Differences from the paper

- **Host algorithm.** The paper uses its own IPM (parapint, fraction-to-boundary steps). The default host here is IPOPT (filter line search, its own inertia correction); `--host simple` is the paper-style alternative (see above).
  - IPOPT adds δ_w to *every* primal diagonal, including y, so C = δ_w·I instead of 0.
  - The paper's per-tile δ_H^k, δ_C^k (Sec. 3.5) can't be expressed through IPOPT's interface. Correction is global, as in the paper's experiments.
- **Block factorization.** The paper uses MA27 (pivoted). Here the default is an *unpivoted* sparse LDLᵀ (Eigen) in a KKT-aware order (`blocks.hpp`), with a per-tile dense fallback when a pivot is zero or tiny relative to its row (see "The MPCC under both hosts"); `--block-solver dense` is the pivoted dense Bunch–Kaufman reference.
  - With AMD alone, an unpivoted LDLᵀ breaks down on the zero diagonals of W_k: every constraint row, and every foreign copy. IPOPT would then add δ_c/δ_w at every step (this is what `../cpp_minimal` does), and the trace would no longer match MUMPS.
  - The order eliminates (L1) primals with a nonzero diagonal, (L2) the rows coupled to them, then (L3) the remaining primals, each followed by (L4) its linking row. Each level keeps AMD's relative order.
  - Interleaving L3 with L4 is necessary. At an interior tile corner, two foreign copies sit in one row only, so after L1/L2 their block is rank one: the corner null mode of S_k again. Eliminating the second copy would hit an exact zero pivot. Taking it right after its partner copy and linking row avoids that, because the pair [m 1; 1 0] has determinant −1.
  - With this order, the DD-direct IPOPT log is identical to MUMPS's for κ = 1, −5 and −20 (N=32, 4×4), with no singular factorization. PCG iteration counts differ slightly from the dense blocks (AS mean 99.8 vs 89.3 at N=32), because S_k carries different round-off and S is very ill-conditioned late in the IPM. The IPOPT path is the same.
- **Block Jacobi** splits C's diagonal (δ_w) evenly over the tiles that touch each y, so that Σ_k blocks = S exactly.
- **Shared memory.** `OMP=1 ./build.sh` runs every per-tile loop with OpenMP: factorization, S_k, tile solves, S·v, the Schwarz blocks and the coarse assembly. Tiles write to private buffers that a parallel loop over the border then gathers, so there are no atomics. There is no MPI; the communication in Sec. 3.3 is just indexing here.

## Command-line options

```
problem    --N 32 --tiles 4 --kappa 1 --alpha 1e-4 --fmax 40 --amp 2
solver     --solver dd|mumps
blocks     --block-solver sparse|dense
interface  --schur direct|pcg  --precond none|bj|as|asd
           --coarse none|faces|sides|linear  --coarse-mode balanced|additive
           --cg-tol 1e-10  --cg-maxit 2000  --matvec sk|backsolve
inertia    --pcg-inertia curvature|exact|free  --neg-curv-tol 1e-12  --local-inertia
debug      --check [maxdim]  --verbose 0..3
ipopt      --print-level 5  --tol 1e-8  --max-iter 500  --obj-scaling 1|auto|x
host       --host ipopt|simple  --reg local|global
run        --threads n
output     --save sol.csv     (i, j, x, y, tile, u, f, u_d per node)
```
