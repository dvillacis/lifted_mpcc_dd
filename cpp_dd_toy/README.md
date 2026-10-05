# cpp_dd_toy: Lueg et al.'s Schur-complement decomposition inside IPOPT, on a toy problem

This is a small, self-contained C++/IPOPT implementation of the linear algebra in

> L. R. Lueg, M. L. Bynum, C. D. Laird, L. T. Biegler, *Domain decomposition
> preconditioners for Schur complement systems arising in structured nonlinear
> optimization problems*, Optim. Eng. 27 (2026) 555–585.

It is applied to a 2D semilinear optimal control problem instead of the MPCC. The
point is to see what the decomposition does inside IPOPT without the MPCC
machinery: complementarity, μ-coupled relaxation, the α and cross-point peels of
`../cpp_minimal`. The tile layout is the same as in the image problems.

The code is about 1250 lines. Read the files in this order:

| file | what it is |
|---|---|
| `problem.hpp` | The NLP as an `Ipopt::TNLP`, written in the paper's form (1): local variables, local copies, linking rows. `kkt_owner()` says which tile owns each KKT unknown. |
| `schur_dd.hpp` | The paper's Sec. 2–3, in Eigen + LAPACK. Each step is commented with its equation number. It knows nothing about IPOPT. |
| `ipopt_bridge.hpp` | Makes `SchurDD` IPOPT's linear solver (`SparseSymLinearSolverInterface` + `AlgorithmBuilder`). Same pattern as `../cpp_minimal/ipopt_bridge.hpp`. |
| `main.cpp` | CLI, IPOPT options, summary. |

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

**4. Preconditioners**, default problem (κ=1). PCG relative tolerance is 1e-10 with a 2000-iteration cap. N=32 with 4×4 tiles:

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

## Differences from the paper

- **Host algorithm.** The paper uses its own IPM (parapint, fraction-to-boundary steps). Here it is IPOPT (filter line search, its own inertia correction).
  - IPOPT adds δ_w to *every* primal diagonal, including y, so C = δ_w·I instead of 0.
  - The paper's per-tile δ_H^k, δ_C^k (Sec. 3.5) can't be expressed through IPOPT's interface. Correction is global, as in the paper's experiments.
- **Block factorization.** Each W_k uses dense Bunch–Kaufman (LAPACK) instead of MA27. Tiles are small, and it gives exact inertia with pivoting.
  - An unpivoted sparse LDLᵀ (as in `../cpp_minimal`) breaks down on the copy/linking pairs `[0 1; 1 0]` in W_k. IPOPT would then have to add δ_c/δ_w at every step, and the trace would no longer match MUMPS.
  - The cost is O(t_k³) per tile. Keep t_k ≲ 1000, i.e. use more tiles for bigger N.
- **Block Jacobi** splits C's diagonal (δ_w) evenly over the tiles that touch each y, so that Σ_k blocks = S exactly.
- **Serial.** `OMP=1 ./build.sh` parallelizes the per-tile factorization loop with OpenMP. There is no MPI; the communication in Sec. 3.3 is just indexing here.

## Command-line options

```
problem  --N 32 --tiles 4 --kappa 1 --alpha 1e-4 --fmax 40 --amp 2
solver   --solver dd|mumps
dd       --schur direct|pcg  --precond none|bj|as|asd  --cg-tol 1e-10  --cg-maxit 2000
         --matvec sk|backsolve  --pcg-inertia curvature|exact  --local-inertia
         --check [maxdim]  --verbose 0..3
ipopt    --print-level 5  --tol 1e-8  --max-iter 500
output   --save sol.csv     (i, j, x, y, tile, u, f, u_d per node)
```
