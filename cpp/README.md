# The DD-arrowhead IPOPT solver — technical notes

> **Provenance note.** This is the design-and-findings document for the C++
> solver, carried over verbatim from its original development monorepo so the
> measured record stays intact. References below to sibling paths — `../cpp`
> (the wider experimental archive), `../dd_kkt.py`, `../lifted_mpcc_*.py`,
> `../dd_structure.py`, `../CLAUDE.md`, "the Python reference / lab" — point to
> that development repository. In **this** standalone package the Python
> reference implementations live in [`../python`](../python); the pruned
> experimental history (`../cpp`) is not included. Treat those paths as
> historical citations, not files you will find here. In particular, prose below
> mentioning the multi-panel plotters `plot_1d.py` / `plot_2d.py` /
> `lifted_mpcc_*.plot_solution` refers to the reference repository — this package
> ships only the lightweight `../python/plot_slurm.py` (2D result figures) and the
> `../python/dump_data*.py` instance generators (backed by `../python/mpcc_utils.py`).
> For a clean orientation start from the [top-level README](../README.md).

A cleaned-up rewrite of `../cpp`: the same three validated solvers (uniform 2D,
staggered 1D, staggered 2D), the same arrowhead domain decomposition as IPOPT's
actual linear solver, with the shared machinery factored out and the
experimental branches pruned. The interface system is solved directly by
default, with an opt-in **CG interface solve carrying the Lueg BJ/ASd
preconditioners** (see below). **`../cpp` remains the archive of the wider
experimental history** (Uzawa/Richardson/Chebyshev, matrix-free `S_ff`
variants, the dense plumbing baseline, one-off diagnostic dumpers) — see
`../cpp/README.md` and `../CLAUDE.md` for those measured findings.

## What was removed relative to `../cpp` (deliberately)

- **The Uzawa/Richardson/Chebyshev interface machinery** (`--interface uzawa`,
  `--uzawa-*`, `DD_PEEL_MATFREE`, its telemetry). What remains iterative is a
  clean **CG interface solve** (below) rebuilt on the Python reference — with
  the dual peel re-ported on top of it (2026-07-22, see below). The direct MA57
  solve of the assembled sparse `S` stays the default — the measured winner at
  every size.
- `EigenDenseSolver` + `hs071_test.cpp` (the plumbing baseline — long validated).
- `--anchor forward` (measured strictly worse; settled dead end).
- `DD_DUMP` / `DD_DUMP_BAD` singular-block dumpers and their offline analysis
  scripts `dd_diagnose.py` / `dd_nullspace.py` (one-off root-cause tooling; the
  findings they produced — border promotion — are baked in).

## The CG interface solve (`--interface cg`, 2026-07-22)

Opt-in preconditioned conjugate gradient on the interface system, the
distributed-DD prototype, mirroring `../dd_kkt.py`'s `solve_interface`:

- **Preconditioners** (`--precond`, the Lueg paper's, as implemented by the
  Python reference `make_preconditioner`): `bj` = each local Schur block
  `S_k = −B̄_k W_k⁻¹ B̄_kᵀ` inverted densely and applied additively (Lueg eq. 17);
  `asd` (default) = the same blocks with the diagonal replaced by the
  **assembled** `diag(S)` (Lueg eq. 20–21); `jacobi` = `diag(|S|)`, the trivial
  baseline. A singular local block is skipped, as in the Python.
- **α peel**: the dense α row/column is eliminated as a scalar Schur step before
  the Krylov solve (α is in every `N_k` and breaks the adjacency-banded
  structure the distributed preconditioners rely on); the peel cache
  `Z = S_ff⁻¹ S_fP` comes from a sparse MA57 factorization of `S_ff` (the
  Z-cache fix below, final form). `--no-alpha-peel` iterates on the full `S`.
- **Dual peel (2026-07-22, ported from `../cpp` (vii))**: on TILE partitions the
  promoted corner-dual pairs are exactly S's negative eigenvalues, so plain CG
  can never run there. `config_peel(n)` (drivers: on by default in CG mode,
  `--no-dual-peel` to A/B) eliminates every border entry with KKT index ≥ n
  together with α as ONE dense Schur block: `Z = S_ff⁻¹ S_fP` (one multi-RHS
  MA57 backsolve — the Z-cache fix below),
  `T = S_PP − S_fPᵀZ` (LU), `Δy_P = T⁻¹(r_P − S_fPᵀ w)`, `Δy_f = w − Z Δy_P`.
  The admissibility gate becomes exact: CG runs iff `In(S)_neg` equals the
  peeled-dual count (from MA57's pivots — no eigensolve). Measured, N=16 k=2
  tile: **0% → 43.6% iterative** (815/1870, 974 fallbacks = the S_ff
  interlacing-leak/conditioning residue, 81 skipped at extra-negative iterates),
  exact solution — matching `../cpp`'s dense-S_ff reference (45%). Uniform tile:
  603/846 (71%) iterative. 1D and strips: structural no-op, numbers reproduce
  byte-identically (the 1×1 peel block is the old scalar α peel).
- **`S` is still assembled and MA57-factorized** — its pivot signs answer the
  Haynsworth inertia query, give an **exact admissibility gate** (CG runs only
  when every negative of `S` is a peeled dual direction, i.e. `In(S)_neg` equals
  the peeled count; no eigensolve needed, unlike `../cpp`'s spectrum test) and a
  free direct fallback whenever CG stalls (acceptance: true residual of the full
  system on the assembled `S`, threshold 1e-2). So CG is a solve-strategy swap,
  not a factorization-avoidance scheme; `apply_S` itself is matrix-free through
  the subdomain backsolves, which is the distributed character being prototyped.
- `--cg-tol` (1e-10), `--cg-max-iter` (500); telemetry (solves / avg its /
  skips / fallbacks / `In(S)_neg`) prints at the end of the run.
- **`--cg-apply assembled|matfree` (2026-07-22, default `assembled`)**: how CG
  applies the operator. `assembled` reuses the S already formed for the inertia
  — one sparse matvec per CG iteration; the Krylov STATISTICS (its/solve,
  convergence, preconditioner quality — the distributed-viability metrics)
  measure the same operator and are unchanged up to FP ordering. `matfree`
  re-derives S·y through K subdomain backsolves per iteration — the faithful
  simulation of the distributed COST profile (each apply redoes work the schur
  phase already paid for once; profiled at ~10× per factorization). Measured at
  `--factor 0.3`: 1D n=64 interface 0.058 → 0.002 s; N=16 tile interface
  11.0 → **0.285 s (39×)**, total 18.5 → 8.2 s (≈ direct's 6.5 s); N=32 k=4
  tile interface 117 → **11.1 s (10×)**, total 183 → 97 s — CG mode is now at
  parity with direct on one node, identical solutions everywhere. Two further
  refinements (same day): with the peel active the CG operator is the
  precomputed kept×kept sparse `S_ff` block (no embed → full-S → extract, no
  wasted peel rows/columns) and the CG loop is allocation-free (hoisted
  `Sp`/`z`/`pvec` buffers, `applyPinv_into`) — validated bit-identical stats on
  1D/N=16/N=32; wall effect below machine-load noise at measurement time, but
  strictly less work per iteration. Note cpp2's CG has NO double S-apply per
  iteration (recursive residual, one apply) — that cost belonged to `../cpp`'s
  Chebyshev loop, which recomputed the true residual every step.

Static-audit fixes (2026-07-22, both modes, trajectory-neutral — the full
regression battery reproduces bit-identically incl. DD_CHECK 234/234; the #6/#7
pass below was re-validated the same way: 1D direct/CG(asd)/CG(jacobi,no-peel),
2D direct/CG tile+dual-peel/CG strip/CG matfree, uniform direct/CG, DD_CHECK,
solution + arrowhead dumps — all byte-identical, OMP included):

- **structure-once routing**: the border-coupling structure is frozen in
  `route_triplets` (B_k compressed patterns + triplet→valuePtr slot maps,
  border-border triplets → sparse-S slots, see the next bullet); `factorize()`
  is now a pure value refresh — no triplet re-routing, no per-factorization
  `setFromTriplets` (sort + alloc ×K), no dense rebuild;
- **sparse-S storage (#6)**: the assembled interface matrix lives ONLY in the
  sparse `Ssp_` (full symmetric pattern = `spat_` ∪ mirror, frozen once; values
  accumulated per factorization through slot maps, like the B_k blocks) — the
  dense p×p `C_`/`S_` pair and their O(p²) zero+copy per factorization are
  gone. Every consumer reads the sparse values: the MA57 refresh, the
  Jacobi/ASd diagonals, the SfP/S_PP peel gathers (one walk over the nP peel
  columns), the sff/`Sffsp_` refresh, the CG apply AND acceptance test (now
  `Ssp_·dy` in both apply modes), and the arrowhead dump (densified
  transiently, small-p only). Accumulation order is preserved, so it is
  trajectory-neutral BY CONSTRUCTION, not just by measurement. Measured N=64
  k=8 (p=1716): maxrss 189 → 136 MB (−52 MB ≈ the 2·p² doubles), identical
  trajectory; at N=128 k=16 (p≈7680, nnz(S)≈1.5%) the dense pair alone was
  2×470 MB plus a ~470 MB memcpy per factorization;
- **CG-loop leftovers (#7)**: `applyPinv_into`'s per-block gather/apply
  temporaries are preallocated members (the last allocation inside the CG
  iteration), and `MultiSolve` caches the DD_CHECK env lookup;
- **solve-path scratch**: `solve_refined`'s five dim-sized buffers and
  `solve_one`'s per-subdomain vectors are members — removes ~10 dim-scale
  allocations per solve × up to 6 solves per RHS × thousands of RHS;
- **cross-level structure reuse**: `CustomSolverBuilder` hands every
  continuation level the SAME solver instance (fresh `TSymLinearSolver` wrapper
  each time), and `InitializeStructure` skips the rebuild when the full
  structure guard matches (dims, ia/ja pattern, N/K, injected owner vector) —
  the partition, routing tables, B/C slot maps and all K+1 MA57 symbolic
  analyses are built once per run instead of once per level (verified: the
  route_triplets debug line fires exactly once across an 8-level run). NB
  `DD_TIME`'s phase timers now accumulate across the RUN, not per level.

Wall-clock effect pending an idle machine (a 300%+ CPU job was resident during
this session; iteration counts are the load-independent validation signal).

**The θ-gauge ridge is the big run-level lever (measured 2026-07-22, N=32 k=4
tile, direct, `--factor 0.3`, full schedule).** `--c-theta 1.0` (ε_θ = c_θ·t, so
the bias decays to nothing along the continuation) attacks the δ_w retry storm
that multiplies factorizations — and it is NOT mariposa-specific: flat cells
make the θ-gauge deficiency generic, and a null θ column sits inside ONE
subdomain, so it hurts the DD blocks before it hurts the full KKT:

| instance | c_θ | IPOPT its | factorizations | facts/it | weight | PSNR |
|---|---|---|---|---|---|---|
| mariposa | 0 | 730 | ~1550 | 2.12 | 0.048549 | 23.73 |
| mariposa | 1.0 | **546** | **~1000** | 1.83 | 0.049038 | 23.73 |
| cameraman | 0 | 1067 | ~2450 | 2.30 | 0.074095 | 26.98 |
| cameraman | 1.0 | **439** | **~800** | 1.82 | 0.073568 | 26.97 |

Factorizations −35% / −67%, iterations −25% / −59%, at ≤0.01 dB and ≤1% weight
shift — and since factor+schur ≈ 80%+ of direct-mode wall scales with the
factorization count, the wall gain tracks these ratios. Left OPT-IN (default
c_θ = 0) so every recorded table stays reproducible; for new experiments
`--c-theta 1.0` is the recommended starting point.
NOT pursued, per measurement: TNLP eval temporaries (<2% of profile), Bt
prealloc (0.16%), S_k symmetry/scatter (~1%), MC64 off (6.5× blowup),
sparse-RHS backsolves (bounded by e-tree reachability; the one real lever on
the dominant S_k phase, but a solver-replacement project).

**Where CG time goes (profiled on mariposa N=32, 4 strips, 2026-07-22).** The
CG interface is a distributed-memory prototype, NOT a single-node speed play —
on one node the direct back-solve of the already-factorized `S` is free, and CG
re-derives it through ~p/3 matrix-free `apply_S` calls, each costing K subdomain
backsolves. Measured on the 4 loose levels: serial direct 56 s wall vs serial CG
125 s (interface 67 s of it); `OMP=1` cuts the interface to 32 s (2.1× — capped
by K=4 blocks per apply_S plus the serial CG recurrence), total 61 s ≈ direct.
Two costs are structural: (i) each Newton step triggers ~4 interface solves —
`solve_refined`'s sweeps each rerun a full CG solve, and they genuinely gain ≥2×
per sweep (the arrowhead's accuracy floor at degenerate iterates), so they can't
be skipped; (ii) mariposa's degeneracy causes ~1.7 factorizations per iteration
(δ_w retries), each paying the `S_k` phase. Four fixes are baked in (2026-07-22), all leaving the DIRECT path untouched
(validated trajectories reproduce bit-identically):

1. persistent MA57CD scratch instead of a zeroed per-call allocation (~20% off
   the interface; ~10⁶ backsolves/run);
2. CG-mode refinement stops on stagnation (defensive; measured a no-op — the
   sweeps genuinely gain ≥2× each);
3. **Z-cache off CG** — the peel's `Z = S_ff⁻¹S_fP` was costing **64% of ALL CG
   iterations** when built by CG (~150 its × nP columns per factorization, and
   nP = 4(k−1)²+1 grows quadratically with k). First replaced by a dense LU of
   the assembled `S_ff` (the choice `../cpp`'s validated peel default made),
   then (#5) by a **sparse MA57 factorization of `S_ff`** on the
   spat_-restricted adjacency pattern — structure/symbolic analysis frozen per
   kept set, Z as one multi-RHS backsolve, and `In(S_ff)` read off the pivot
   signs as an exact gate that rejects interlacing-leak iterates before CG
   wastes a doomed attempt on an indefinite operator. Exact Z also *improved*
   iterative conversion (N=16 tile: 44% → 62%). The "peel Z-cache CG its"
   telemetry line went with it (it necessarily read 0 CG its); the stats now
   report the rebuild count only;
4. **dead-CG skip** — once CG fails on a factorization's operator, the
   refinement chain's re-attempts on the SAME matrix skip straight to direct
   (they were ~40% of CG work at N=32 k=4; reported as "skipped after a
   failure"). N=16 tile: interface 12.2 → 7.4 s; N=32: a wash on wall (Newton-
   path noise) but fallback stats become honest (1231 → 395 real failures).

Net effect, N=16 k=2 tile CG (full schedule): **20.4 s → 12.3 s** (interface
15.1 → 7.4 s) vs 6.5 s direct. N=32 k=4 tile (5 levels): serial CG 183 s,
OMP 95 s (interface 119 → 63 s, 1.9× over 16 blocks — capped by the serial ASd
apply, the dense peel setup and the CG recurrence) vs ~55 s serial direct. The
REMAINING CG cost is genuine: the `S_ff` conditioning tail as t ↓ 0 (the
"hard-but-standard" residue — a coarse-space preconditioner is the research
lever), plus the structural ~4 interface solves per Newton step. In DIRECT mode
the bottleneck is the `S_k` phase (the p_k backsolves), as documented in
`../cpp` — the sparse-RHS lever remains unattempted with bounded expected gain.

Measured (2026-07-22, identical dumps, every run at the exact direct solution):

| instance | precond | iterative / solves | avg its | fallbacks |
|---|---|---|---|---|
| 1D n=64 k=4 | jacobi | 235/235 | 9.1 | 0 |
| 1D n=64 k=4 | asd | 237/237 | 9.0 | 0 |
| 1D n=64 k=4 | asd, `--no-alpha-peel` | 234/234 | **7.0** | 0 |
| 1D n=64 k=4 | bj | 212/242 | 8.1 | 30 |
| 2D N=16 strip k=2 | asd | **2056/2077 (99%)** | 34.4 | 21 |
| 2D N=16 strip k=2 | jacobi | 2056/2076 | 55.5 | 20 |
| 2D N=16 strip k=2 | bj | 497/1590 (31%) | 25.2 | 1093 |
| 2D N=16 tile k=2 | any, `--no-dual-peel` | 0/1628 (all skipped) | — | 0 |
| 2D N=16 tile k=2 | asd + dual peel | 815/1870 (44%) | 60.0 | 974 |

Readings, all consistent with the repo's standing findings: **ASd is the one
distributed preconditioner that works** (in 2D it beats Jacobi 1.6× on
iterations and BJ outright); **BJ is weak** — kept strictly as the Lueg A/B
lever, its fallbacks are the measure of that; on the **tile** partition the
promoted corner duals make `S` structurally indefinite (`In(S)_neg = 4` at every
solve) so without the dual peel the gate skips CG entirely — the peel (default)
cures that, and `--partition strip`, whose `S` is SPD everywhere, avoids it
altogether; and in 1D the α peel does not pay (`p ≤ 15`; full-S CG's 7.0 avg its
reproduces `../cpp`'s measured number exactly) — it is the 2D/distributed
device. The 1D `DD_CHECK` under CG: 234/234 inertia MATCH, max rel-err 1e-15.

## The signed-MA57 MINRES interface (`--interface minres`, 2026-07-23)

The third interface mode, and the first that is **indefiniteness-proof**: no SPD
gate, no dual peel — preconditioned MINRES on the full assembled `S`, tile
partitions included. The preconditioner is `M = L|D|Lᵀ` built from a SNAPSHOT
MA57 factorization of `S` (MC64 off so the `JOB=2/3/4` partial solves compose
exactly): with `S = LDLᵀ`, the preconditioned operator is similar to `|D|⁻¹D`,
whose spectrum is EXACTLY `{−1,+1}` (Gill–Murray–Ponceleón–Saunders 1992), so
MINRES converges in ~2 iterations no matter what `In(S)_neg` is. The offline
probe measured 2 its at rel-res ~5e-8 on dumped tile matrices with 4 and 36
negative eigenvalues and cond up to 6e11 — where CG (gated or peeled) diverges.

Implementation notes, all in `dd_solver.hpp` / `ma57_block.hpp`:

- **`|D⁻¹|` is recovered FORMAT-FREE by bit-pattern probing** of the `JOB=3`
  D-solve: the all-ones vector plus every bit pattern and its complement
  (`1 + 2⌈log₂p⌉` RHS, one batched MA57CD call). For each (row, bit) exactly one
  of the pattern/complement pair has `b(i)=0`, and there the response is
  `e_off·b(j)` — so the 2×2 partner index and value read off directly, with no
  dependence on MA57's factor layout. `|E|` per 2×2 block is the closed form
  `(E² + |det E|·I)/√(tr E² + 2|det E|)`.
- **Everything is self-checked per snapshot**: the JOB-composition test
  (`L∘D∘Lᵀ` solves ≡ the full solve), the recovered structure re-validated on an
  independent random vector, and the pairing checked to be a symmetric perfect
  matching. Any failure disables the preconditioner (those solves go direct and
  are counted) rather than risking a wrong step. Measured: 0 failures across
  every run so far.
- **True-residual refinement is REQUIRED, not cosmetic.** MINRES's stopping
  test is an `M⁻¹`-norm estimate; on cond(S) ~ 1e10 the true residual stalls
  orders above the direct solve, and without refinement the Newton path drifts
  enough to derail whole levels (measured: N=16 tile died at `t=1.5e-2`,
  status 2). With up to 3 cheap refinement passes (each ~2 its) the N=16 run
  reproduces the direct trajectory EXACTLY — 1861 IPOPT its, best level
  1.115e-4, weight 0.070756, 24.97 dB — with 6426/6426 interface solves
  accepted iteratively at 2.8 avg its, 0 fallbacks, `In(S)_neg = 4` throughout.
  `DD_CHECK` over that whole run: **4285/4285 inertia MATCH vs
  MA57-on-the-full-matrix, max step rel-err 4.1e-9.**
- **N=32 k=4 tile (`In(S)_neg = 36`, where CG diverges outright): full schedule,
  16373/16379 solves iterative at 3.3 avg its, 0 fallbacks.** The 2 snapshot
  self-check failures over the run were caught and those 6 solves went direct —
  the guard doing its job. At this size the trajectory is NOT bit-identical to
  direct: 1e-8-level step differences compound and the run lands on a nearby
  stationary point (weight 0.0709 / 26.34 dB vs direct's 0.0576 / 26.16 dB —
  both healthy; same "reaches the same-quality solution" contract as the CG
  mode, per the standing 2D finding).
- **The inertia contract is untouched**: `s_ma57_` is still factorized fresh
  every step (Haynsworth needs exact `In(S)`); the snapshot is a SECOND
  factorization (only every `--minres-lag` M-th step). So single-node this mode
  cannot be faster than direct — it exists to measure the distributed question.

**The lag experiment (`--minres-lag M`) and its answer: staleness does NOT
pay.** If the signed factorization stayed a good preconditioner while `S`
drifts, the serial S-factorization (the measured Amdahl floor of the DD solver)
could be amortized over M Newton steps in a distributed setting. Measured at
lag 8 (cameraman tile; both runs complete the full schedule and land on the
good solution — N=16: 0.070756 / 24.97 dB; N=32: 0.058247 / 26.18 dB):

| precond age | N=16 k=2 avg its | N=32 k=4 avg its (fallbacks) |
|---|---|---|
| 0 | 2.8 | 3.4 (0) |
| 1 | 66.9 | 483.1 (53) |
| 3 | 119.6 | 620.1 (105) |
| 7 | 189.3 | 847.0 (159) |

**One Newton step of drift already destroys the ±1 clustering** — 24× at N=16,
140× at N=32, and the degradation GROWS with size. The mid-continuation `S`
(Σ = z²/μ terms swinging along the barrier path) changes far too fast for
factorization reuse at this granularity; the amortization idea is measured
DOWN. What survives: `--interface minres` at lag 1 is the first
fully-iterative-capable interface on tile partitions (~3 avg its, ~100%
iterative acceptance where the CG gate skips or the peel leaks), with
machine-checked steps and honest fallback accounting.

**|ASd|-MINRES and two-level (coarse-space) |ASd| — measured DOWN offline
(2026-07-23, scratch probes on the dumped tail matrices; don't wire, don't
re-litigate).** The distributed-friendly SPD preconditioner (per-block
absolute-value ASd — eigendecompose each local Schur clique, flip negative
eigenvalues) DOES solve the indefiniteness (negatives cluster in [−5.8,−2.1]
at N=16 where Jacobi spreads them to −0.002) but only halves the iterations
(177→86 at N=16, 3459→1730 at N=32) — the obstacle moves to the classic
one-level conditioning tail. The coarse-space remedy was then measured to its
CEILING with oracle coarse spaces (exact worst preconditioned modes): N=32
tail matrix, 1772 → 309 its at nc=32, 245 at nc=64, and even deflating BOTH
spectral tails (all 60 modes outside |λ| ∈ [0.3,10]) leaves 330 its — the
one-level block approximation error is spread across the WHOLE spectrum, not
concentrated in any deflatable subspace. Buildable coarse spaces don't even
approach the oracle (GenEO-lite block modes: 1091 its at nc=256 of p=400;
partition-of-unity constants: no gain) because the bad modes are global, not
in local block spans. Root cause consistent with the standing findings: at the
Scholtes tail the relaxation-active set is O(cells), not O(corners) (the
Python probe measured 105/225 cells ON `rw=t` at `t=2.2e-4`), so `Σ = z²/μ`
degenerates a HIGH-dimensional interface subspace — no low-dimensional device
fixes that. Direct `S` stays necessary at the deep tail; the iterative
interface is a mid-continuation/easy-factorization tool. The last open lever —
maybe the tail is better conditioned under the μ-coupled schedule — was then
ported and tested the same day and is ALSO refuted: see the next section
(cond(S) improves 70×, the preconditioned spectrum does not move). The
tail-interface pathology is schedule-independent and this line is closed.

## The μ-coupled t-update (`--t-update mu`, 2D driver, 2026-07-23)

Port of the validated `../lifted_mpcc_unitball_v2.py` mechanism to the
staggered 2D C++ driver: ONE IPOPT solve, with the TNLP's
`intermediate_callback` slaving the Scholtes level to the barrier,
`t = max(t_min, c·μ)` (`c = --t-mu-scale`, default 10), tightening only, the
θ-gauge ridge kept ∝ t. Legal mid-solve because t enters only `eval_g`'s comp
rows additively (Jacobian/Hessian t-free); requires monotone μ (`init_app`
sets it). Implementation: the callback lives in `mpcc_base.hpp` (armed by
`t_mu_scale_ > 0`, with the Python port's live `[mu-coupled]` progress rows and
the pinned-t stall warning), the single-solve driver is
`driver_common.hpp::run_mu_coupled` (RunResult-shaped, so summaries and dumps
work unchanged).

**Measured — a large solver win, bigger than Python's.** Cameraman PNG route,
`--solver dd`, identical instances and depth, geometric vs μ-coupled:

| instance | geometric | μ-coupled |
|---|---|---|
| cameraman N=16 k=2 | 1861 its → 0.070756 / 24.97 dB | **48 its** → 0.070831 / 24.97 dB |
| cameraman N=32 k=4 | ~4500 its → 0.0576 / 26.16 dB | **72 its** → 0.070885 / 26.34 dB |
| mariposa N=128 k=8 (`--interface cg`, `--factor 0.3` baseline) | 3039 its → 0.067214 / 26.52 dB | **438 its** → 0.067328 / 26.52 dB |

Full matched depth, status 0, every instance — 7–60× fewer iterations (the
C++ geometric driver pays its 8–50+ cold levels; Python's measured 2–4× had
the CP warm start amortized differently). At N=32 the μ-coupled run also lands
on the good branch (0.0709, matching the exact-TV optimum family) where the
geometric direct run wandered to 0.0576. The mariposa run is the robustness
data point: its barrier path is rough (inf_du spikes to 2.7e6, μ bounces up in
restoration episodes mid-solve) and the monotone tightening guard rides it out
to the exact geometric solution with a 12× cheaper interface (134 vs 1660 s
wall). **μ-coupling is the 2D driver's DEFAULT since 2026-07-23** (user
decision, on the three-for-three coverage above). Two standing caveats travel
with that: (1) every table recorded BEFORE the flip was produced under the
geometric schedule — reproduce them with `--t-update geometric`; (2) the
single solve has NO best-completed-level fallback on failure, and the
forced-bad-α₀ barrier-stall mode (t hostage to a frozen μ; the mariposa
`--alpha0 -2` repro) remains real — on a suspect instance, or whenever a run
ends status ≠ 0 with nothing reported, re-run with `--t-update geometric`
before concluding anything about the instance.

**What it does NOT change: the tail interface pathology (the redirect
hypothesis is refuted).** Dumping the last factorization at matched depth
(`--t-min 1.115e-4`, `--save-dd`) and re-running the offline interface probes,
geometric → μ-coupled:

| metric (N=32 k=4 tail S) | geometric | μ-coupled |
|---|---|---|
| cond(S) | 6.0e11 | **8.6e9 (70× better)** |
| In(S)_neg | 36 | 36 |
| MINRES + Jacobi | 3459 its | 3815 its |
| one-level \|ASd\| | 1772 its | 1938 its |
| oracle coarse nc=32 / band | 309 / 330 its | 309 / 315 its |
| preconditioned spectrum | [−389,−1] ∪ [1.6e-3,244] | [−220,−1] ∪ [6.2e-4,270] |
| modes with \|λ\| < 0.3 / > 10 | 34 / 26 | 43 / 26 |
| In(S_ff)_neg after peel | 8 | 10 |

The raw conditioning improves 70× (the Raghunathan–Biegler ξ-control is real),
but the PRECONDITIONED spectrum — what governs every iterative interface — is
unchanged, because the high-dimensional degeneracy (relaxation-active set is
O(cells) at the tail) is intrinsic to where the continuation lands, not to the
path. So: use `--t-update mu` for the 39–60× solver win; do NOT expect it to
revive the distributed-interface line. Direct `S` at the tail stands,
schedule-independent.

**The continuation-path figure follows the mode (2026-07-23).** A μ-coupled run
has no outer levels to tabulate — the continuation happens INSIDE the solve —
so the callback records an `(iter, μ, t, weight, max r(1−δ))` trace every
iteration (weight/comp read off the CURRENT accepted iterate via the
documented `TNLPAdapter::ResortX` recipe; NaN on restoration iterations, drawn
as gaps), `--save-solution` appends it as a trailing count-prefixed block (0
for geometric; readers of any vintage stay compatible), and
`plot_2d.py`/`lifted_mpcc_2d.plot_solution` draw the panel accordingly. The
μ-coupled panel keeps the geometric panel's story — the weight converging
(left axis) and the complementarity tightening under the dotted stepping `t`
(right axis, log) — but on the ITERATION axis, since `t` plateaus at the floor
while the solution still moves; the loose-relaxation weight overshoot the
per-level tables always showed is visible in-solve. Only the final iterate is
a converged point — intermediate `t` values are passed through, never
certified. Geometric runs keep the classic per-level weight/comp panel.

```bash
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 4 --solver dd
                                    # μ-coupled DEFAULT: single solve, 72 its (vs ~4500)
./dd_solve_2d ... --t-update geometric   # the per-level continuation (pre-flip tables,
                                         # suspect instances — keeps a best-level fallback)
./dd_solve_2d ... --t-mu-scale 1         # the c=1 A/B (Python: c≈1 ≈ c=10)
```

## The MUMPS W_k backend (`--wk-backend mumps|hybrid`, 2026-07-25)

An opt-in second backend for the subdomain blocks, attacking the measured
direct-mode bottleneck head-on: the S_k phase forms `S_k = −B̄_k W_k⁻¹ B̄_kᵀ`
by p_k dense MA57 backsolves per block (97.5% of the phase; the sparse-RHS
lever this README flags as "a solver-replacement project"). COIN
ThirdParty-Mumps computes exactly that Schur complement natively during a
**partial factorization** of the augmented local matrix
`[[W_k, B̄_kᵀ],[B̄_k, 0]]` (ICNTL(19)=1, Schur list = the appended border
indices), with BLAS-3 fronts instead of naive dense-RHS backsolves, and
reports the negative pivots of the factored interior (INFOG(12)), so the
Haynsworth inertia contract is untouched. Three runtime modes in one binary:

- `--wk-backend ma57` (default) — the validated reference, byte-identical to
  before this change;
- `--wk-backend mumps` — factors, S_k, solves and inertia all from MUMPS;
- `--wk-backend hybrid` — MUMPS forms S_k, MA57 factors the SAME W_k for the
  solve path and the inertia (two factorizations per block per step).

Implementation: `mumps_block.hpp` (`MumpsSchurBlock`, duck-typed to the
`SymBlock` call sites), guarded by `-DDD_HAVE_MUMPS` (build.sh auto-detects
`pkg-config coinmumps`; without it the build is unchanged and the flag errors
cleanly). **`mumps_smoke.cpp` is the gate — run it on any new machine/library
build before trusting the backend.** It validated every piece of API fine
print against Eigen dense references (MUMPS 5.5.0 arm64): Schur values exact
(1.8e-13), sign convention matches `Sk_` directly, INFOG(12) == In(W)_neg,
`JOB=3 + ICNTL(26)=0` == the interior solve `W⁻¹b` (single and multi RHS),
duplicates summed, honest singular reporting (null-pivot detection,
ICNTL(24)=1). One empirical finding is baked in: for SYM=2 the centralized
Schur comes back as the **lower triangle of the full p×p array with the upper
triangle zeroed** (established by a NaN-sentinel probe — the manual is vaguer
than the library), so `factorize()` mirrors it.

Correctness gates (this machine, 2026-07-25): 1D n=256 k=4 `DD_CHECK` —
579/579 inertia MATCH for both backends, **iteration-identical per-level
trajectories** (max step rel-err 3.2e-13 ma57 / 2.9e-10 mumps); 2D cameraman
N=16 k=2 — 674/674 MATCH, 0 mismatches, all three backends land on the
recorded reference (0.070831 / 24.97 dB); hybrid max step rel-err 9.7e-10.

**Measured (serial, cameraman, this arm64 laptop; DD_TIME phase sums in s —
in mumps/hybrid modes JOB=2 does factor+Schur at once, so compare the
factor+schur SUM, not the split):**

| run | backend | n_fact | factor+schur | per fact | solve | wall |
|---|---|---|---|---|---|---|
| N=32 k=4 μ-coupled | ma57 | 150 | 3.69 | 24.6 ms | 1.84 | 5.9 |
| | mumps | 150 | 1.72 | 11.4 ms | 10.35 | 13.9 |
| | hybrid | 150 | 2.81 | 18.7 ms | 2.12 | **5.7** |
| N=64 k=8 μ-coupled | ma57 | 150 | 17.15 | 114.3 ms | 8.39 | 31.4 |
| | mumps | 200 | 9.50 | 47.5 ms | 49.10 | 81.9 |
| | hybrid | 200 | 16.17 | **80.8 ms** | 9.33 | 35.2 |
| N=32 k=4 geometric | ma57 | 6725 | 177.2 | 26.3 ms | 54.5 | 245.8 |
| | hybrid | 7100 | 146.0 | 20.6 ms | 71.3 | **231.6** |

Readings, in the order they were learned:

1. **The partial-factorization Schur genuinely kills the S_k phase**: 12.27 s
   → 0.05 s at N=64, 119.6 s → 0.32 s on the geometric schedule. Isolating
   the Schur-formation cost at N=64 (mumps factor 47.5 ms/fact minus an
   MA57-like factorization 32.5 ms/fact ≈ 15 ms vs 81.8 ms of backsolves):
   **~5.4× cheaper Schur formation.**
2. **Pure `mumps` mode is measured DOWN for this architecture**: dmumps
   JOB=3 costs ~6× MA57CD per single-RHS backsolve (per-call workspace setup
   + scaling application vs MA57's persistent scratch), and the DD solve path
   lives on ~10⁵–10⁶ such backsolves (solve_one, refinement, matfree
   applies). Net wall loses at every tested size (5.9→13.9, 31.4→81.9). Its
   geometric-schedule leg was skipped as strictly dominated.
3. **`hybrid` wins per factorization everywhere** — 1.31× (N=32 μ), 1.41×
   (N=64 μ), 1.28× (N=32 geometric) — and the gap GROWS with N as the schur
   share grows. Wall wins where trajectories match (5.9→5.7, 245.8→231.6).
   At N=64 μ-coupled the wall regressed (31.4→35.2) for a benign reason: the
   MUMPS-computed S differs at roundoff, the barrier path diverges (the
   standing MINRES-mode precedent), and the ma57 leg stopped at
   status 1/155 its where both MUMPS-Schur legs ran to **status 0**/218 its —
   same solution either way (0.070977–8 / 27.02 dB). Load-independent
   per-factorization cost is the honest comparator, and there hybrid wins.
4. 1D is not the target regime (p_k ≤ 2 border columns per block — nothing
   for the Schur feature to amortize); it is correctness-gated but slower
   under mumps, as expected.

Standing recommendation: `ma57` stays the default (every recorded table
reproduces bit-identically). For large-N 2D direct-mode runs, `--wk-backend
hybrid` is the measured lever — its advantage scales with exactly the phase
that dominates at scale. The open follow-ups: the MUMPS factorize loop is
SERIAL by design (no mumps_smoke_par-style concurrency audit yet — the MA97
lesson), so an OMP A/B needs that audit first; and the pure-mumps solve tax
would need a persistent-workspace solve path that dmumps_c does not expose.

## Layout

| file | role |
|---|---|
| `ma57_block.hpp` | RAII wrapper around HSL MA57 for one symmetric-indefinite block |
| `mumps_block.hpp` | RAII wrapper around COIN Mumps: partial-factorization Schur for one W_k (`--wk-backend mumps\|hybrid`) |
| `mumps_smoke.cpp` | MUMPS API validation gate vs Eigen dense — run on every new machine/library build |
| `dd_solver.hpp` | `DDArrowheadSolver` (the custom `SparseSymLinearSolverInterface`, direct + CG interface) + `CustomSolverBuilder` |
| `mpcc_base.hpp` | shared TNLP base: objective, bounds, Q(α), warm start, `finalize_solution` |
| `mpcc_tnlp.hpp` | uniform-grid formulation (port of `../lifted_mpcc_unitball_v2.py`) |
| `mpcc_1d_tnlp.hpp` | staggered 1D formulation (port of `../lifted_mpcc_1d.py`) |
| `mpcc_2d_tnlp.hpp` | staggered 2D formulation (port of `../lifted_mpcc_2d.py`), incl. the C++ Chambolle–Pock warm start for the image route |
| `driver_common.hpp` | PSNR, t-schedule, IPOPT options, the Scholtes continuation loop, `--self-check` checksums |
| `dd_solve.cpp` | uniform driver (`--solver mumps\|ma57\|dd`) |
| `dd_solve_1d.cpp` | staggered 1D driver (partition + injected owner map) |
| `dd_solve_2d.cpp` | staggered 2D driver (tile/strip partition, corner promotion, image route) |
| `../python/dump_data*.py` | export the exact instance + CP warm start + owner map (via `mpcc_utils`) |
| `../python/plot_slurm.py` | render 2D result figures from a directory of `--save-solution` dumps |

Two 2D-figure improvements (2026-07-22): the solution figure overlays the DD
cuts on every image panel (the 2D analogue of the 1D subdomain bands), and the
domain map draws the partition **actually used**, derived from the dumped owner
map (`DumpPartition` in `plot_2d.py` — "dump, don't reconstruct"): a
`--partition strip` run now shows its strips and horizontal-only cuts (the
Python-side reconstruction silently assumed k×k tiles), and the promoted
corner-dual cells are starred on tile runs with `p` taken from the arrowhead.
Without `--dd-dump` the partition type is unknowable from the solution file
alone; the script then assumes tiles and says so — pass the dump for strip runs.

The TNLP derivative cores (`eval_g` / `eval_jac_g` / `eval_h`, the structure
arrays, the operators) are **verbatim** from `../cpp` — value arrays are matched
to the structure arrays positionally, and none of that was touched. Only the
boilerplate around them moved into `mpcc_base.hpp` / `driver_common.hpp`.

## Build

Prerequisites: IPOPT 3.14 (Homebrew, `pkg-config ipopt`), Eigen
(`brew --prefix eigen`), HSL MA57 at `$HSLDIR` (default `~/.local/hsl-ma57`).
`image_io.hpp` and `third_party/stb_image.h` are vendored next to the sources.
Optional: COIN ThirdParty-Mumps visible as `pkg-config coinmumps` — build.sh
then adds `-DDD_HAVE_MUMPS` and the `--wk-backend mumps|hybrid` modes light
up (run `./mumps_smoke` once first; without coinmumps the build is unchanged).

```bash
cd cpp
./build.sh dd_solve.cpp    -o dd_solve            # macOS (Homebrew IPOPT/Eigen)
./build.sh dd_solve_1d.cpp -o dd_solve_1d
./build.sh dd_solve_2d.cpp -o dd_solve_2d
OMP=1 ./build.sh dd_solve_2d.cpp -o dd_solve_2d   # optional: parallel W_k loops
```

On **Linux** use `build_linux.sh` (same interface, `OMP=1`/`CXX=` respected):
a conda env supplies the toolchain, IPOPT (`pkg-config ipopt`) and Eigen; the
HSL block solver is separate. Two backends, `HSL_SOLVER=ma57|ma97` (default
auto — MA57 if `libhsl_ma57.so` is found, else MA97):

- **MA57** — the validated reference: `$HSLDIR/lib/libhsl_ma57.so` (extra link
  deps via `HSL_EXTRA_LIBS`, e.g. `"-lopenblas -lgfortran"` for a static or
  thinly-linked MA57). For the `--solver ma57` route set
  `HSLLIB=$HSLDIR/lib/libhsl_ma57.so` at runtime.
- **MA97** (`ma97_block.hpp`, compiled via `-DDD_USE_MA97`; added 2026-07-23
  for the HPC, whose HSL install has ma97/spral but no MA57) — an
  API-identical wrapper over MA97's C interface (`ma97_*_d` must be exported;
  the script checks with `nm`). `solve_job` keeps MA57's JOB numbering
  (1 full / 2 L / 3 D / 4 Lᵀ → MA97 jobs 0/1/2/3), so the CG/MINRES partial-
  solve machinery carries over; the MINRES compose is self-checked at runtime
  as always. MC64 scaling defaults ON to mirror MA57 (`DD_MA97_NO_SCALING=1`
  to disable). The monolithic reference on such a box is `--solver ma97`
  (IPOPT's own MA97 — no `HSLLIB` needed when IPOPT links it directly).
  **MA97 threading — SETTLED 2026-07-23 with `ma97_smoke_par` (the standalone
  concurrency stress test) + gdb: concurrent `ma97_factor` heap-corrupts
  (`free()` inside `rfact_block` on a worker thread — module-global state in
  the library's front memory management; OMP_STACKSIZE, `ulimit -s`,
  `MKL_THREADING_LAYER=SEQUENTIAL/GNU` and MC64-off were ALL measured to still
  segfault, don't re-try), while concurrent `ma97_solve` is safe (phase B
  clean at 8 threads).** So `OMP=1` MA97 builds get the SPLIT model
  automatically: serial `W_k` factorize loop, parallel backsolve loops — the
  backsolves are 97.5% of the S_k cost, so most of the across-block win
  survives. Validate on any new machine with
  `OMP_NUM_THREADS=8 DD_PAR_SKIP_FACTOR=1 ./ma97_smoke_par` (phase B must be
  OK) plus one `DD_CHECK=1` run. Full concurrent factorization needs a
  serial-built (no-OpenMP) `libhsl_ma97` or MA57 — the split is the ceiling
  for this library build. (The removed `MA97_CONCURRENT=1` lever re-enabled
  the factor loop; it is a measured segfault, don't resurrect it.)
  **Status: validated on the cluster 2026-07-23 — `ma97_smoke.cpp` (ABI
  canary, inertia, value refresh, JOB 2→3→4 compose) all OK, and the 1D
  n=256 k=4 `DD_CHECK=1 --solver dd` run is clean serially. Run `DD_CHECK=1`
  once on any new instance class anyway. MA57-vs-MA97 trajectories differ in
  path (different pivot orders), not in solution.** `ma97_smoke.cpp` stays in
  the tree as the first thing to run on any new machine/library build:
  `HSL_SOLVER=ma97 ./build_linux.sh ma97_smoke.cpp -o ma97_smoke`.
  Hardened 2026-07-24: the struct layouts are compile-gated
  (`static_assert` in `ma97_block.hpp`, which also carries a 96-byte
  defensive margin after the documented ABI on both structs), and the smoke
  adds a rank-deficient 3×3 exercising the `action=1` singular path the DD
  solver's SINGULAR routing relies on.

The link line bakes in `-rpath`/`-rpath-link` to `$CONDA_PREFIX/lib` (so ld can
resolve the transitive MKL/metis/gfortran closure of the conda libs) and
`--allow-shlib-undefined` (module-provided libs outside the env, e.g. an OHPC
hwloc, may reference system libs the conda sysroot cannot see — the runtime
loader resolves them). Both were learned on the HPC (2026-07-23).

```bash
./build_linux.sh dd_solve_2d.cpp -o dd_solve_2d              # auto backend
HSL_SOLVER=ma97 ./build_linux.sh dd_solve_2d.cpp -o dd_solve_2d
OMP=1 ./build_linux.sh dd_solve_2d.cpp -o dd_solve_2d_omp
```

## Run

Instances must come from the dump scripts whenever the run has to be comparable
with Python (NumPy's RNG cannot be reproduced in C++, and the cold start selects
a different basin — see `../CLAUDE.md`):

`data/` holds a pre-generated 1D grid for scaling studies —
`data_1d_n{256,512,1024,2048}_k{4,8,16,32}.txt` (defaults: linear weight,
σ=0.1, seed 0, CP warm start; all 16 owner maps verified identical to Python's
and the corner cases solved: n=256 k=4 → 0.072061 / +5.05 dB in 131 its;
n=2048 k=32 with `--interface cg` → 0.070532 / +6.09 dB in 165 its, CG 540/540
iterative at 19.2 avg its, p=63). `results/` holds one CG+ASd run per instance
(logs + solution dumps + last-Newton-step arrowhead dumps + `summary_cg_asd.csv`,
2026-07-22; `plots/` holds the matching 4-panel solution figures `sol_1d_*.png`
and arrowhead/interface figures `dd_1d_*.png` via `plot_1d.py` — the dump guard
was widened to dim ≤ 50000 so the n=2048 KKTs dump too): all 16 reach the
full schedule with 100% iterative interface convergence — 7262 CG solves total,
0 skipped / 0 fallbacks / 0 not-at-tol — and the avg CG iteration count tracks
the border size only (≈9 at p=7 → ≈19 at p=63, flat in n from 256 to 2048).
Regenerate or extend with:

```bash
python ../python/dump_data_1d.py --n 64 --nsub 4 -o data/data_1d_64.txt
./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --self-check     # derivative + owner gate
./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd      # 0.071568, +4.02 dB, 113 it
DD_CHECK=1 ./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd   # vs MA57-full, every solve

python ../python/dump_data_2d.py --N 16 --nsub 2 -o data/data_2d_16.txt
./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd      # 0.074386, 24.79 dB, 485 it
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 4 --solver dd  # PNG route (C++ CP warm start)
./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd --partition strip   # no cross corners ⇒ no promotion, S SPD
./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd --interface cg      # PCG interface, ASd (Lueg) default
./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd --interface cg     # tile + dual peel (both default)
./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd --partition strip \
    --interface cg --precond bj                                              # the Lueg BJ A/B on the SPD strip S
./dd_solve_2d --data ../images/cameraman.png --size 16 --nsub 2 --solver dd \
    --interface minres                                                       # signed-MA57 MINRES: indefinite-proof, ~2.8 avg its
./dd_solve_2d --data ../images/cameraman.png --size 16 --nsub 2 --solver dd \
    --interface minres --minres-lag 8                                        # the staleness experiment (prints its-vs-age table)

python ../python/dump_data.py --N 16 -o data/data_16.txt
./dd_solve --data data/data_16.txt --solver dd --nsub 2            # uniform grid, cold start
```

Performance flags (see [Performance defaults](#performance-defaults-2026-07-25-profiling-pass)):

```bash
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 6 --solver dd \
    --hessian exact                          # analytic eval_h — 1.9x here, but see the N=48 crossover
./dd_solve_1d --data data/data_1d_n1024_k16.txt --solver dd --hessian exact   # 2.9x in 1D

# restoring the pre-2026-07-25 behaviour, knob by knob
./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 6 --solver dd \
    --schur backsolve --ma57-scaling on      # old S_k route + MC64 scaling
DD_MA57_ICNTL13=10 ./dd_solve_2d ...         # old BLAS2 multi-RHS threshold

DD_SCHUR_CHECK=1 ./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 6 \
    --solver dd                              # forward S_k vs the JOB=1 route, per block
DD_STATS=1 ./dd_solve_2d --data ../images/cameraman.png --size 32 --nsub 6 \
    --solver dd --max-iter 1                 # the B_k sparsity budget
```

`--schur forward` always forces MC64 scaling off on the `W_k` blocks regardless of
`--ma57-scaling`, since its JOB=2/3 partial solves only compose without it.

The drivers write reusable data dumps: `--save-solution` (the reported iterate +
per-level history + the embedded instance) and `--save-dd` (the arrowhead). The
2D result figures are rendered by the self-contained `../python/plot_slurm.py`,
which walks a directory of `--save-solution` files (`<dir>/sols/sol_*.txt`):

```bash
mkdir -p results/run/sols
./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd \
    --save-solution results/run/sols/sol_16.txt
python ../python/plot_slurm.py results/run     # one PNG set per sol_*.txt
```

The full multi-panel probe / arrowhead / domain figures (and single-file 1D
plotting) are produced by the reference implementation's plotters, which are not
part of this archival package.

## Environment switches

| var | effect |
|---|---|
| `DD_CHECK=1` | verify every solve + inertia against MA57 on the full matrix (small N only) |
| `DD_SCHUR_CHECK=1` | verify each `S_k` from `--schur forward` against the JOB=1 route, per block |
| `DD_TIME=1` | print the phase timers (factor / schur / S-fact / solve) every 25 factorizations |
| `DD_STATS=1` | one-shot `B̄_k` sparsity budget (dim, `p_k`, nnz, interface-touched rows) |
| `DD_DEBUG=1` | partition summary, singular-block reports, refinement warnings |
| `DD_HESSIAN=` | overrides `--hessian` (`exact` / `limited-memory`) |
| `DD_MA57_SCALING=1` | re-enable MC64 scaling (same as `--ma57-scaling on`) |
| `DD_MA57_ICNTL13=n` | override MA57's BLAS2/BLAS3 multi-RHS threshold |
| `DD_MA57_ICNTL6=n` | MA57 pivot ordering (2=AMD, 4=METIS, default 5=auto) — see the METIS note |
| `HSLLIB` | path of the MA57 dylib IPOPT's own `--solver ma57` should load |
| `HSLDIR` | (build) install prefix of HSL MA57, default `~/.local/hsl-ma57` |

## Performance defaults (2026-07-25 profiling pass)

A sampling profile of a small instance (N=32, 6×6 tiles, KKT dim 16466) found
most of the run outside the DD algebra. Three findings became **new defaults**;
all three are *trajectory-preserving* — iteration count, exit status and PSNR are
byte-for-byte what the previous defaults produced on every instance below, so
these are pure wall-clock wins, not a change of algorithm.

1. **MA57 was back-solving the Schur RHS block one column at a time.** The `S_k`
   formation solves `W_k X = B̄_kᵀ` with `p_k ≈ 20–40` columns at once, but at
   HSL's default `ICNTL(13)` the profile caught MA57CD on its BLAS2 kernels
   (`ma57qd/rd` 14% self, against `ma57xd/yd` 3%). Raising the threshold past
   `p_k` moves the block onto the BLAS3 path. → **`ICNTL(13)` now set high.**
2. **MC64 scaling was recomputed at every factorization** (`mc64wd/dd/ed` ≈ 10%
   of the run) — MA57 re-derives it per call, and the DD solver refactorizes every
   `W_k` at every Newton step. Turning it off is also *more* accurate here: over
   an N=32 6×6 run `DD_CHECK`'s max per-step rel-err is 3.5e-9 unscaled vs 4.4e0
   scaled. → **scaling off; `--ma57-scaling on` restores it.**
3. **The backward substitution in the `S_k` formation is redundant.** With
   `W_k = P L D Lᵀ Pᵀ`, `S_k = −B̄_k W_k⁻¹ B̄_kᵀ = −Yᵀ D⁻¹ Y` for
   `Y = L⁻¹Pᵀ B̄_kᵀ`: the `Lᵀ` half cancels against the `B̄_k` on the left. Exact
   (`DD_SCHUR_CHECK` max rel-err ~1e-13), worth a consistent 7–17%.
   → **`--schur forward` is the default; `--schur backsolve` restores JOB=1.**

| instance | old defaults | new defaults | speedup |
|---|---|---|---|
| 2D N=32, 6×6 tiles | 9.19 s (147 it) | **4.22 s** (147 it) | 2.18× |
| 2D N=32, 4×4 tiles | 9.47 s (147 it) | **4.52 s** (147 it) | 2.10× |
| 2D N=48, 4×4 tiles | 30.6 s (132 it) | **17.7 s** (132 it) | 1.73× |
| 2D N=64, 4×4 tiles | 73.9 s (155 it) | **48.6 s** (155 it) | 1.52× |
| 1D n=256 k=4 | 0.109 s (122 it) | **0.072 s** (122 it) | 1.51× |
| 1D n=512 k=8 | 0.254 s (177 it) | **0.163 s** (177 it) | 1.55× |
| 1D n=1024 k=16 | 0.490 s (165 it) | **0.301 s** (165 it) | 1.63× |

### `--hessian exact` — a big win, but only up to about N=48

All three TNLPs implement a full analytic `eval_h`, which the drivers never used:
`hessian_approximation` was pinned to `limited-memory`. That is expensive in a way
the profile made obvious — IPOPT's `LowRankAugSystemSolver::UpdateFactorization`
was **44%** of the N=32 run, because under L-BFGS every Newton step re-solves the
augmented system with the correction vectors as extra right-hand sides.

Exact roughly halves the iteration count, but its Hessian block is denser so each
remaining iteration costs more, and the two effects cross over:

| instance | limited-memory | `--hessian exact` | |
|---|---|---|---|
| 1D n=1024 k=16 | 0.301 s (165 it) | **0.103 s** (90 it) | 2.9× |
| 1D n=2048 k=32 | 0.919 s (181 it) | **0.233 s** (83 it) | 3.9× |
| 2D N=32, 6×6 | 4.22 s (147 it) | **2.18 s** (72 it) | 1.94× |
| 2D N=48, 4×4 | 17.7 s (132 it) | **12.3 s** (78 it) | 1.44× |
| 2D N=64, 4×4 | **48.6 s** (155 it) | 77.2 s (135 it) | 0.63× ✗ |

So it stays **opt-in**, defaulting to `limited-memory` to keep every previously
recorded trajectory reproducible. Use `--hessian exact` in 1D and in 2D up to
about N=48.

Together with the new defaults, DD's gap to monolithic MA57 on the same instance
narrows from 7.0× to 3.1× at N=32 (2.8× when both use the exact Hessian) and from
5.1× to 2.9× at N=48. The Amdahl story of `cpp/README.md` is unchanged — this
pass removed overhead, not the interface floor.

### Measured negative results

Kept here so they are not re-attempted:

- **OpenMP across subdomains: the decomposition parallelizes, the run does not.**
  This one was diagnosed twice, and the first diagnosis was wrong — recorded here
  because the correction is the useful part.

  *First pass:* at 8 threads the run took 14.5 s against 5.0 s serial, and the
  profile blamed OpenBLAS — `blas_memory_alloc` 21%, `__psynch_cvwait` 32%, its
  global buffer allocator serializing MA57's many tiny BLAS calls under a lock.

  *Second pass, after the BLAS3 + scaling defaults landed:* most of that
  disappeared (2.62 s vs 2.19 s), so the allocator was mostly a symptom of the
  BLAS2 call storm, not the cause. To settle it, MA57 was **rebuilt against Apple
  Accelerate** (no OpenBLAS in the link at all — see the recipe below). Serial
  performance is *identical*, 1.372 s vs 1.350 s, and the OpenBLAS symbols vanish
  from the profile — but the threaded regression is unchanged. The BLAS was not
  the cause.

  What the phase timers actually show (N=32, 6×6 tiles, 36 subdomains, Accelerate,
  interleaved repeats):

  | threads | total | `factor`+`schur` (the OMP loops) | everything else |
  |---|---|---|---|
  | 1 | 1.368 s | 0.830 s | 0.538 s |
  | 4 | 1.509 s | **0.263 s** (3.2×) | 1.246 s |
  | 8 | 1.938 s | **0.221 s** (3.8×) | 1.717 s |

  **The DD-parallel work scales 3.8× on 8 threads, exactly as the design claims.**
  What kills the run is the *serial remainder*, which grows by more than the
  parallel phases save — and it grows already at 4 threads, where all workers fit
  on this laptop's 4 performance cores (it has 4 P + 4 E). `KMP_BLOCKTIME=0` and
  `OMP_WAIT_POLICY=passive` change nothing, so it is not idle-spin. The remaining
  suspect is macOS scheduling on a hybrid-core consumer chip — thread migration
  and QoS demotion of the main thread once a worker pool is live — i.e. an
  artefact of the measurement platform, not of the algorithm. **The parallel-DD
  scaling claim should be re-measured on the homogeneous Linux/MA97 HPC target
  before anything is concluded from it**; on this box the per-phase numbers above
  are the meaningful result, not the wall clock.
- **Lagging the Schur complement does not work** (`--schur-lag L`, kept as an
  opt-in experiment). Reusing the cached `S_k` for `L−1` factorizations does cut
  the phase cost (19 ms → 9.4 ms per factorization), but IPOPT then needs **525
  factorizations instead of 125** and lands on *acceptable* rather than *optimal*,
  and the result is insensitive to `L`. μ falls geometrically, so `W_k`'s barrier
  diagonal — and hence `S_k` — changes by orders of magnitude between steps; it is
  not a small perturbation. The guards are in place (a stale `S` may never decide
  an inertia rejection, and a refinement residual that stops converging latches a
  rebuild), so the failure is the algorithm's, not the plumbing's.
- **A second Haynsworth level on the interface (`--nested`) is a pessimization.**
  The identity composes exactly — that part worked. Partition the border into 4
  quadrant groups plus a separator and

      In(S) = Σ_j In(S_j) + In(S'),    S' = C' − Σ_j E_j S_j⁻¹ E_jᵀ

  reproduces `In(S)` with **0 inertia mismatches** against monolithic MA57 over a
  whole run, same iteration count, same α, same PSNR. The geometric separator is
  small, as predicted: 17.8% of p at 6×6 tiles, 12.8–13.2% at 8×8.

  It is still **8× slower**, and the gap widens with size:

  | N | k | p | flat `S-fact` | `--nested` |
  |---|---|---|---|---|
  | 32 | 6 | 696 | 0.00186 s | 0.00725 s |
  | 48 | 8 | 1464 | 0.00551 s | 0.03921 s |
  | 64 | 8 | 1912 | 0.01095 s | 0.09078 s |

  Fitted exponents in p: flat **2.57**, nested **3.14**. Parallelism does not
  rescue it — over the 4 groups at T=4 it goes 0.091 → 0.048 s, against flat's
  0.013 s.

  The reason is worth stating, because the estimate that motivated the work was
  wrong in an instructive way. That estimate priced the level-2 serial core at
  `sep^1.5`. But `S'` is a **Schur complement, so it is dense** — factorizing it
  costs `O(sep³)`, and with `sep ≈ 0.13p` that is `O(p³)`, asymptotically worse
  than the `O(p^1.5)` MA57 already achieves on the sparse `S`. Forming it is not
  free either: `n_s` back-solves per group is another `~O(p^2.25)`.

  The deeper point: **MA57 already does nested dissection when it factorizes S.**
  An explicit second level replaces an implicit, fill-optimizing elimination with
  a manual, dense one. On a single node that can only lose — the same character as
  this package's headline result, now confirmed one level down. An explicit level
  earns its keep for *distributed memory*, where no rank can hold `S` at all, not
  for serial time. Kept flag-gated and default-off as the evidence.
- **MUMPS partial-factorization Schur is a wash** here (4.95 s vs 5.01 s).
- **CG / MINRES interface**: neutral to worse at these sizes; `--cg-apply matfree`
  is ~3× worse, as its comment already says.

### Rebuilding MA57 against Accelerate (and the METIS trap)

The Accelerate build above is reproducible in a couple of minutes and installs
beside the stock one — nothing existing is overwritten, and the in-tree
`src/.libs` that IPOPT's own `--solver ma57` loads via `HSLLIB` is untouched:

```bash
cp -R ~/src/hsl/hsl_ma57-5.3.2 /tmp/ma57acc && cd /tmp/ma57acc
make distclean
# cp's timestamp skew otherwise triggers a maintainer-mode autotools rebuild
touch configure.ac aclocal.m4
sleep 1 && touch configure $(find . -name "Makefile.in") $(find . -name "*.h.in")
./configure --prefix=$HOME/.local/hsl-ma57-accelerate FC=gfortran CC=clang \
    --with-blas="-framework Accelerate"
sleep 1 && touch config.status $(find . -name Makefile)
make -j8 && make install

cd <repo>/cpp
HSLDIR=$HOME/.local/hsl-ma57-accelerate ./build.sh dd_solve_2d.cpp -o dd_solve_2d
```

MA57 needs only BLAS — `dgemm/dgemv/dtpmv/dtpsv/idamax/isamax` and their single
precision twins, no LAPACK — and none of the single-precision `*dot*` functions
whose f2c return-value convention is the usual Accelerate hazard, so the swap is
clean. **Verdict: correct (`DD_CHECK` clean, 0 inertia mismatches) and
performance-neutral.** Worth having as a diagnostic — it removes OpenBLAS from
the profile entirely — but it is not a speedup.

**Real METIS does not work with MA57 5.3.2 here.** Both METIS libraries on this
machine (Homebrew and `/usr/local`) are **5.1.0**, and MA57 calls
`metis_nodend_` with the **METIS 4** argument list. METIS 5 exports a symbol of
that name, so `configure` accepts it and the link succeeds — but the call passes
a mismatched argument list, so the ordering is garbage: forcing it with
`DD_MA57_ICNTL6=4` produces a factorization that exhausts its workspace
(`W_0 ... status=-3` at N=48). With the default `ICNTL(6)=5` (automatic) MA57
never selects METIS at these block sizes, which is why a real-METIS build is
bit-for-bit identical to the stock `fakemetis` one. Using METIS for real needs
either METIS **4.0.3** (what COIN-OR `ThirdParty-Metis` ships) or a shim
translating the 4.x call to `METIS_NodeND`. `DD_MA57_ICNTL6` is provided to test
this; `4` is a no-op error (`-18`) against the `fakemetis` stub.

### Where the remaining time goes

After all of the above, the `S_k` formation is still the largest single phase.
`DD_STATS=1` prints its sparsity budget: at N=32 6×6 each `B̄_kᵀ` column carries
only **4–6 nonzeros in dim_k ≈ 440** (~1.2% dense), which is what makes
sparse-RHS pruning attractive — but the *union* of interior rows the interface
touches is **24% of dim_k**, and that union bounds what the solve can skip. So the
reachable gain is a few×, not orders of magnitude. MA57 exposes no sparse-RHS
entry point; MUMPS `ICNTL(20)`/`ICNTL(30)` (sparse RHS / selected entries of the
inverse) is the route if this is pursued.

## Validation record (2026-07-22, this refactor vs `../cpp` on identical data)

- `--self-check` five checksums, partition, border and owner-vs-Python: **identical**
  in 1D (`p=7`, dims `[256,270,270,273]`, 0 differing owner entries) and 2D
  (`p=64`, `+4` promoted corner duals — exactly the expected Python delta).
- 1D `--solver dd` (n=64, k=4): **identical per-level iteration counts**, 113 its,
  weight 0.071568, 21.32 → 25.34 dB. `DD_CHECK`: 234/234 inertia MATCH, max
  rel-err 1.0e-14.
- 2D `--solver dd` (N=16, k=2): **identical per-level iteration counts**, 485 its,
  weight 0.074386, 24.79 dB. `DD_CHECK` (t ≥ 0.3): 377 solves, 0 MISMATCH, max
  rel-err 1.0e-13.
- Uniform `--solver dd` (N=16, k=2, cold start): identical trajectory
  (116/124/129/89/402/84 its per level) and identical stop behaviour.
- Full round trip `dump_data_1d.py` → solve → `plot_1d.py`/`plot_2d.py`: data file
  byte-identical to `../cpp`'s, all five figures render.

## The solver in one paragraph

IPOPT hands over the regularized augmented KKT (triplets, lower triangle) and
asks for solves plus the inertia. The KKT indices are labelled by subdomain
(built-in tiling for the uniform grid; an owner map injected by the staggered
drivers, where `u` is node-length but the lift blocks are edge/cell-length);
the complicating columns — first `u` past each cut, last `q` before it, the
global `α`, plus the promoted corner dual pairs in 2D — are permuted to the
border, giving a bordered block-diagonal (arrowhead) matrix. Each `W_k` is
factorized by MA57 (symbolic analysis reused across Newton steps), the interface
Schur complement `S = C − Σ_k B_k W_k⁻¹ B_kᵀ` is assembled on its sparse
adjacency pattern and also factorized by MA57, and the inertia is answered by
Haynsworth additivity `In(A) = Σ_k In(W_k) + In(S)` — the full KKT is never
factorized. Solves run through best-effort iterative refinement against the true
triplets. A rank-deficient block reports `SYMSOLVER_SINGULAR` honestly (no
artificial shifts) and IPOPT's δ_w loop cures it. Failure classification is
explicit (2026-07-24): out-of-memory/workspace exhaustion in a block or the
interface factorization reports `SYMSOLVER_FATAL_ERROR` with a message instead
of masquerading as SINGULAR (a δ_w bump cannot fix OOM and would loop); a
failed backsolve during S_k formation is latched per factorization epoch and
discards that factorization as SINGULAR (the assembled S would be garbage);
and both HSL wrappers refuse to factorize after `analyze` reports out-of-range
triplets (MA57 INFO(3) / MA97 `matrix_outrange` — a warning the libraries
otherwise absorb by silently solving the wrong matrix).
