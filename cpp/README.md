# The DD-arrowhead IPOPT solver — technical notes

> **Provenance note.** This is the design-and-findings document for the C++
> solver, carried over verbatim from its original development monorepo so the
> measured record stays intact. References below to sibling paths — `../cpp`
> (the wider experimental archive), `../dd_kkt.py`, `../lifted_mpcc_*.py`,
> `../dd_structure.py`, `../CLAUDE.md`, "the Python reference / lab" — point to
> that development repository. In **this** standalone package the Python
> reference implementations live in [`../python`](../python); the pruned
> experimental history (`../cpp`) is not included. Treat those paths as
> historical citations, not files you will find here. For a clean orientation
> start from the [top-level README](../README.md).

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

## Layout

| file | role |
|---|---|
| `ma57_block.hpp` | RAII wrapper around HSL MA57 for one symmetric-indefinite block |
| `dd_solver.hpp` | `DDArrowheadSolver` (the custom `SparseSymLinearSolverInterface`, direct + CG interface) + `CustomSolverBuilder` |
| `mpcc_base.hpp` | shared TNLP base: objective, bounds, Q(α), warm start, `finalize_solution` |
| `mpcc_tnlp.hpp` | uniform-grid formulation (port of `../lifted_mpcc_unitball_v2.py`) |
| `mpcc_1d_tnlp.hpp` | staggered 1D formulation (port of `../lifted_mpcc_1d.py`) |
| `mpcc_2d_tnlp.hpp` | staggered 2D formulation (port of `../lifted_mpcc_2d.py`), incl. the C++ Chambolle–Pock warm start for the image route |
| `driver_common.hpp` | PSNR, t-schedule, IPOPT options, the Scholtes continuation loop, `--self-check` checksums |
| `dd_solve.cpp` | uniform driver (`--solver mumps\|ma57\|dd`) |
| `dd_solve_1d.cpp` | staggered 1D driver (partition + injected owner map) |
| `dd_solve_2d.cpp` | staggered 2D driver (tile/strip partition, corner promotion, image route) |
| `dump_data*.py` | export Python's exact instance + CP warm start + owner map |
| `plot_1d.py`, `plot_2d.py` | draw C++ results with the Python's own validated figures |

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

Figures are drawn by the **Python** plotters (the C++ solution is numerically the
same vector; re-implementing the panels would just create a second thing to keep
in sync):

```bash
./dd_solve_1d --data data/data_1d_64.txt --nsub 4 --solver dd \
    --save-solution sol_1d.txt --save-dd dd_1d.txt
python ../python/plot_1d.py --data data/data_1d_64.txt --solution sol_1d.txt \
    --dd-dump dd_1d.txt --save-plot sol.png --save-dd-plot dd.png

./dd_solve_2d --data data/data_2d_16.txt --nsub 2 --solver dd \
    --save-solution sol_2d.txt --save-dd dd_2d.txt
python ../python/plot_2d.py --solution sol_2d.txt --dd-dump dd_2d.txt \
    --save-plot s.png --save-domains d.png --save-dd-plot a.png   # solution file is self-contained
```

## Environment switches

| var | effect |
|---|---|
| `DD_CHECK=1` | verify every solve + inertia against MA57 on the full matrix (small N only) |
| `DD_TIME=1` | print the phase timers (factor / schur / S-fact / solve) every 25 factorizations |
| `DD_DEBUG=1` | partition summary, singular-block reports, refinement warnings |
| `HSLLIB` | path of the MA57 dylib IPOPT's own `--solver ma57` should load |
| `HSLDIR` | (build) install prefix of HSL MA57, default `~/.local/hsl-ma57` |

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
artificial shifts) and IPOPT's δ_w loop cures it.
