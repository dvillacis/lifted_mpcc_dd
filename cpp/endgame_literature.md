# The end game at t = t_min: what the MPCC literature says

Notes from reading the IPM/relaxation literature (Biegler, Raghunathan, Kanzow,
Schwartz and related work) against the end-game symptoms of `tv_dd` documented in
README.md ("The end game: why inf_du stays high once t = t_min").  Full text of the
papers marked (read) was read; (abstract) means only the abstract was checked.

## Our symptoms (N=256, t = t_min = 1e-4)

- dual residual concentrated at cells with r → 0 (≈1e-8), δ < 1 (the active set 𝒜),
  where the lifting multipliers λ_h2 grow to ~1e2;
- the relaxed-product multipliers ξ = λ_comp grow from ~7 to ~1e2 (≈ 1/√t);
- α_du small (0.01–0.5), limited by bound pairs far off the central path
  (z ≈ 1e-7 at slack ≈ 1e-4, s·z ≪ μ);
- t is frozen at t_min while μ keeps falling; the level gate stops at
  inf_pr, inf_du, μ ≤ √t = 1e-2.

## 1. Where our setup departs from the theory

**Kanzow & Schwartz, "The price of inexactness: convergence properties of
relaxation methods for MPECs revisited", Math. Oper. Res. 40:253–275, 2015** (read,
Würzburg preprint 315, 2013).
- Def. 2.1: x is ε-stationary if ‖∇L‖∞ ≤ ε, g ≤ ε, λ ≥ −ε, |gᵢλᵢ| ≤ ε, |h| ≤ ε.
  For Scholtes this includes approximate complementarity of the product-row
  multiplier, |ξᵢ(GᵢHᵢ − t)| ≤ ε.
- Thm 3.2 (preprint numbering): t_k ↓ 0, **ε_k = o(t_k)**, MPEC-MFCQ at x* ⇒ x* is
  **C-stationary**.  The proof uses o(t) explicitly (GᵢHᵢ/t_k → 1).  Example 5.5
  shows O(t) is not enough for Steffensen–Ulbrich; no Scholtes counterexample.
  The CQ (MPEC-MFCQ) is used only to bound the modified multipliers.
- Kadrani, Steffensen–Ulbrich and Kanzow–Schwartz relaxations drop to *weak*
  stationarity under inexact solves unless the iterates avoid thin zones near the
  kink; Scholtes and Lin–Fukushima "do not lose anything".
- **Manuscript fix:** the manuscript cites "[kanzow2015price, Thm 2]" with
  ε_k = O(t_k).  The condition is o(t_k), under MPEC-MFCQ, which the lifted problem
  does not satisfy (ker 𝕂* ≠ {0}).  Check the theorem number against the MOR
  version (the preprint calls it 3.2).
  - **Checked 2026-10-11 against the MOR version** (p. 258, page image): it is
    Theorem 2 and states **ε_k = O(t_k)** under MPEC-MFCQ.  The manuscript's
    citation is right; only the MPEC-MFCQ caveat remains.
- **Code:** the level gate stops at inf_du ≤ √t = 100·t, which violates even O(t).

**Raghunathan & Biegler, "An interior point method for MPCCs", SIAM J. Optim.
15(3):720–750, 2005 (IPOPT-C)** (read, published version).
- Only the product rows are relaxed: Wy + s_cc = t·e, s_cc in the barrier (2.16);
  w, y ≥ 0 stay exact.  In practice t_l = χ_t μ_l, updated per barrier problem,
  **never frozen at a floor** (§5.1).  μ/t ∈ {10, 1, 0.1, μ^0.1} tested; μ/t = 1
  best, χ_t ≥ 1 recommended (Table 5.1).
- Lemma 2.7, (2.20): λ_cc > μ/t, λ_w ≥ (μ/2t)y*, λ_y ≥ (μ/2t)w*.  With μ/t
  bounded away from zero this keeps iterates away from the non-strictly-
  complementary multipliers of the polyhedral multiplier set S_λ (2.8).
  **With t frozen and μ → 0 this protection vanishes** — consistent with our
  collapsed z ≈ 1e-7.
- No upper bound on multipliers: "we need to bound μ/t away from zero but we
  cannot guarantee bounded multipliers" (§5); "it remains to be seen if a
  mechanism for ensuring bounded multipliers can be developed" (§6).  Without
  strict complementarity / MPCC-LICQ: "often results in unbounded multipliers and
  breakdown" (§5.4).
- Modified step (3.7)–(3.8): only the complementarity block of the bound rows is
  changed, choice (ii) z̃ = z + ηλ, λ̃ = λ (Vicente–Wright type), which caps the
  condensed diagonal λ/(z+ηλ) at 1/η.  IPOPT-C applies it only for μ ≤ 5e-6, with
  η = 0.1 μ_{l−1}/(1 + ‖λ‖∞).  No δ_c on equality rows, nothing on product rows.

## 2. ξ ~ 1/√t is the expected Scholtes behaviour

- **Ralph & Wright, "Some properties of regularization and penalization schemes
  for MPECs", Optim. Methods Softw. 19(5):527–556, 2004** (read).  Eq. 33c–d: the
  *recombined* multipliers τ − ξH → τ*, ν − ξG → ν* converge; ξ itself need not.
  At a biactive pair on rw = t with r ~ w ~ √t and τ* ≠ 0, ξ ~ |τ*|/√t — exactly
  our growth 7 → 1e2.  Prop. 3.6: ξ bounded only under MPEC-LICQ, S-stationarity
  and an O(t) approach.
- **Andreani, Haeser, Secchin, Silva, "New sequential optimality conditions for
  MPCCs and algorithmic consequences", SIAM J. Optim. 29(4):3201–3230, 2019**
  (read).  AW/AC/AM-stationarity allow unbounded multiplier approximations; a raw
  multiplier blow-up is not a failure certificate.  See also A. Ramos,
  "MPEC-AKKT" (preprint 2016/17).
- **What to monitor:** γ = a − ξ(1−δ), ν = b − ξr (manuscript eq.
  recovered_impl) and their signs on the near-biactive cells.  If γ* < 0 there,
  the limit is C/M- but not S-stationary and no Scholtes schedule (and no finite
  penalty) keeps ξ bounded.

## 3. Remedies in the literature

| remedy | sources | targets | notes for us |
|---|---|---|---|
| exact penalty / elastic mode: π Σ r(1−δ) in the objective, adaptive π | Leyffer, López-Calva, Nocedal, SIAM J. Optim. 17(1):52–77, 2006 (read); Ralph–Wright Thm 5.1; Anitescu, SIAM J. Optim. 15:1203–1236, 2005 (abstract); Anitescu, Tseng, Wright, Math. Program. 110:337–371, 2007 (abstract); CCOpt alg. 2 | ξ blow-up: the complementarity multiplier is π, bounded at S-stationary points; strictly feasible interior kept | LLN Lemma 4.3: π > π_min ⇒ LICQ, SC, SOSC for the penalty NLP.  Dynamic update (Fig. 4): π ← 10π when ‖min(r,w)‖ > μ^0.4 and r·w not decreasing over 3 iterations.  π diverges if no S-stationary point exists. |
| two-sided relaxation, per pair, driven by multiplier signs | DeMiguel, Friedlander, Nogales, Scholtes, SIAM J. Optim. 16:587–609, 2005 (read); CCOpt "RelaxLB" end game | lost interior, small α_du, active-set identification | x ≥ −δ₁, −δ₂ and X₁x₂ ≤ δ_c per pair; shrink δ_i when the MPEC multiplier z_i > r̄*, shrink δ_c when one is < −r̄*.  Thm 3.1: LICQ, SC, SOSC of the relaxed NLP; product multiplier 0 at biactive pairs.  Globalized with a merit function, not a filter. |
| dual regularization δ_c always on; RB modified bound block | Raghunathan–Biegler (Vicente & Wright, COAP 2002); Wright, COAP 11:253–276, 1998 (abstract); Gill & Robinson, SIAM J. Optim. 23(4), 2013 (abstract) | rank-deficient equality Jacobian, non-unique λ, attraction to critical multipliers (Izmailov & Solodov, Math. Program. 117:271–304, 2009; TOP 23:1–26, 2015) | CCOpt on bilevel instances with degenerate non-complementarity Jacobians: "unbounded steps in the dual variables without regularization … easily solved if a small fixed dual regularization δc is chosen".  Matches our ker 𝕂*. |
| gauge fixing / variable elimination; proximal (not fixed-reference) θ term | Biegler, *Nonlinear Programming* (SIAM 2010) ch. 11 guideline "if possible, do some variable eliminations in the KKT conditions" (via lecture slides, unverified); Friedlander & Orban, Math. Program. Comput. 4:71–107, 2012 (abstract) | θ on 𝒯 | our stuck cells are in 𝒜 (r → 0, δ < 1), not 𝒯: gauge fixing at max(r,δ) ≤ √t would not catch them. |
| crossover / cleanup re-solve: fix r = 0 or δ = 1 per cell, solve one branch NLP | Pozharskiy, Pacaud, Diehl, Nurkanović, "CCOpt", arXiv 2604.18726, 2026 (read); Hoheisel, Kanzow, Schwartz, Math. Program. 137:257–288, 2013, Alg. 2 (read); manuscript "active-set cleanup" | the whole t → 0 end game | removes the product rows and the bound degeneracy; ker 𝕂* remains, so still needs δ_c. |

Further findings:
- **Hoheisel–Kanzow–Schwartz 2013**: Scholtes is still the most robust relaxation
  in practice (90.5% of MacMPEC with SNOPT); Kadrani and Kanzow–Schwartz converge
  slowly when solutions are not strongly stationary.  Alg. 2 stops early once the
  iterate is MPCC-feasible.
- **CCOpt (2026)** names our symptom: "a rapid increase in the magnitude of the
  Lagrange multipliers for the complementarity lower bounds and Scholtes
  relaxation, as well as numerical issues due to the near LICQ violation."  Its
  defaults: τ(μ) roll-off rule (no frozen floor), critical-multiplier clipping of
  the 2×2 product blocks, lower-bound-relaxation end game, δ_c, crossover to an
  active-set MPCC solver.  Note: CCOpt §3.7 attributes M-stationarity of inexact
  Scholtes limits to Kanzow–Schwartz; KS prove C-stationarity.
- Unverified: Baumrucker, Renfro, Biegler, Comput. Chem. Eng. 32:2903–2913, 2008
  found the penalty formulation most robust (second-hand summary only).

## 4. Plan

1. Diagnose: recombined multipliers γ, ν and an S/M/C/W classification on the
   near-biactive cells at the final iterate; δ_c always on (c·μ^κ).
2. Structural change, chosen by step 1: LLN penalty formulation if the limit is
   S-stationary; otherwise Scholtes with t ∝ μ down to ~1e-3 followed by a
   branch-NLP cleanup with δ_c.
3. Manuscript: cite KS as ε_k = o(t_k) under MPEC-MFCQ and flag that the lifted
   problem does not satisfy it; check the theorem number; restate the gate.

## 5. Step 1 results: MPCC multipliers and δ_c (2026-10-05)

Tools: `--classify FILE` prints, per end-game iteration, the sets B (r, 1−δ ≤ ε),
𝒜, ℐ at ε = 3√t and √t and the sign classes of γ = a − ξ(1−δ), ν = b − ξr on B;
FILE gets the final point per cell (r, w = 1−δ, ξ, a, b, γ, ν, ‖λ_h2‖).
`--dual-reg C --dual-reg-exp K` puts δ_c = C·μ^K in every Newton system.

**Default run, final point (t = 1e-4).**
- The biactive set is large: at N=256, 8.5k cells at ε = √t (13%), 27k at 3√t (42%).
- On B, γ and ν are ≥ −1e-3 for 99% of cells (the 99th-percentile multiplier
  scale is 8e-2, the gate 1e-2): S-stationary to the accuracy of the solve.
- The large multipliers are confined to a few clusters of adjacent cells:
  - **cluster type 1** (cameraman 256: (111,28), (112,28), (111,29); mariposa 48:
    (4,2), (5,2), (4,3); mariposa 32: (29,19), (29,20), (30,19)): an L-shaped
    triple of cells around one node, all with r ≈ 1e-8 and δ anywhere in
    [0, 0.6], γ ≈ +1e2, ‖λ_h2‖ = 2–3e2.  A single-pixel feature flattened by TV;
    the three r = 0 rows around one node make the local system near-singular — the
    D2 rank condition in its smallest instance.  Only some of these cells are in
    𝒯 (δ ≈ 0), so gauge fixing at max(r, δ) ≤ √t would not catch all of them.
  - **cluster type 2** (cameraman 256: cells (16–18, 200–202)): near-biactive,
    ξ up to 1.2e2, with γ and ν **both negative** (γ = −2.6, ν = −0.57):
    C-stationary, not S.  This is where ξ ~ 1/√t is unavoidable (Ralph–Wright).
- N=128 has neither: max ‖λ_h2‖ = 1.2, max ξ = 2.9.

**δ_c always on** (iterations; N=256 with restoration iterations; α, PSNR where
they changed; last column: max ‖λ_h2‖ / max ξ at the final point of N=256):

| δ_c | cam32 4×4/2×2/8×8 | cam48 | mar32 | mar48 | cam128 | cam256 16×16 | λ_h2 / ξ |
|---|---|---|---|---|---|---|---|
| default (on demand) | 29/28/29 | 30 | 51 (23.43 dB) | 135 (24.27 dB) | 91 | 151 (+14), 100 s | 305 / 121 |
| 1e-8·μ^¼ | 30/32/26 | 30 | 26 (23.38 dB) | 108 (24.23 dB) | 91 | 137 (+64), 114 s | 36 / 135 |
| 1e-6 | 34/31/33 | 28 | 34 (23.43 dB) | 46 (24.12 dB) | 97 | 161 (+10), 75 s | 25 / 91 |
| 1e-4 | 65/81/41 | 63 | 47 (23.39 dB) | 117 (24.14 dB) | 60 | 54 (+14), 26 s; α 0.0707, 28.540 dB | 7 / 2.9 |

Reading:
- δ_c does what the stabilization literature predicts on cluster type 1: the
  multipliers of the near-singular rows drop from 3e2 to 2–4e1 (1e-6, 1e-8·μ^¼)
  and to 7 (1e-4).
- It does not remove cluster type 2 unless δ_c = 1e-4, and then the run stops at
  a different point (α 0.0707 instead of 0.0723) and the small cases slow down.
- The iteration count at N=256 barely moves for δ_c ≤ 1e-6: the end game is
  held by cluster type 2 (C-type pairs, ξ ~ 1/√t), not by cluster type 1.
- Mariposa N=48 drops from 135 to 46 iterations with δ_c = 1e-6, at a lower
  PSNR (24.12 vs 24.27 dB): a different local solution.

## 6. Step 2: δ_c default and the sign-driven per-cell relaxation

**δ_c = 1e-6 is now the default** (`--dual-reg 0` restores the on-demand δ_c).
N=256 8×8: 99 (+15) iterations, 54 s, α 0.072177, 28.552 dB (on demand: 301 (+56),
199 s).  Small set: cam32 4×4/2×2/8×8 34/31/33, cam48 28, mar32 34, mar48 46
(24.12 dB instead of 24.27 dB — a different local solution).

**Per-cell relaxation** (`--relax-cells THR`; `tv_mpcc.hpp` flag_cells(), and
`NLP::bounds_changed()` so the IPM re-reads moved bounds).  Once t = t_min, a cell
whose γ or ν stays below −threshold for `relax_persist` = 5 consecutive iterations
is flagged for the rest of the solve: r ≥ −1e-4, δ ≤ 1 + 1e-4, r(1−δ) ≤ 0.

| variant | small six | cam128 | cam256 16×16 | cam256 8×8 |
|---|---|---|---|---|
| default (no relaxation) | 34/31/33/28/34/46 | 97 | 161 (+10), 75 s | 99 (+15), 54 s, max ξ 21 |
| fixed threshold 0.1, flag at once | 34/32/34/61/37/55, 3–27 cells | 97, 41 cells | 410, 31 666 cells, 28.525 dB | stopped at 520, 6 749 cells |
| fixed 0.1, persist 5 | 34/31/33/28/34/66 (26 cells on mar48) | — | — | — |
| max(0.1, inf_du^0.7), persist 5 | 34/31/33/28/34/60 (2 cells on mar48) | 97, 0 cells | 263 (+10), 1 583 cells, 104 s, 28.534 dB, max ξ 344 | 140 (+15), 31 cells, 70 s, max ξ 75 |

Reading:
- With a fixed threshold the flagging cascades: each flagged cell moves its
  bounds, the dual residual jumps, and more multiplier estimates go negative.
  The residual-scaled threshold (DeMiguel et al.'s ‖res‖^(1−τ)) stops the
  cascade at 8×8 but not fully at 16×16.
- Even where few cells are flagged, it does not help: the flagged cells sit at
  the corner, and with the product right-hand side at 0 the product row is
  degenerate there again, so ξ grows (max ξ 75 vs 21 at 8×8; 344 at 16×16).
  DeMiguel et al.'s theory sets δ_c = 0 only for pairs that are on a branch
  (not biactive) at a strongly stationary point — not our case.
- Kept as an opt-in experiment, off by default.

## 7. Step 3: the active-set clean-up (2026-10-06)

Implemented as `--cleanup gate|reach` (`TvMpcc::begin_cleanup`, `NLPPoint` warm
start in `ipm.hpp`), as the manuscript describes it.
- **Setup.** Per cell the smaller of r, w = 1 − δ is pinned to 0, the product rows
  are dropped, and the NLP is re-solved warm, with ξ moved into the pinned rows
  (γ = a − ξw, ν = b − ξr).
- **Result.** It does not converge on cameraman/mariposa N=32 or mariposa N=48:
  3000 iterations, repeated failed restoration phases, pinned rows violated by
  ~5e-3.
- **Not helped by:**
  - pinning both (TNLP) or neither (RNLP) on cells with r, w ≤ √t;
  - δ_c of 1e-4 or 1e-3;
  - a starting μ of 1e-4;
  - starting as soon as t = t_min.

Why: at t = 1e-4 the active set is not identified.
- **Ambiguous cells.** 42% of the cells of cameraman N=32 have r, w > 1e-3, with
  r·w ≈ t. The pins that fail are on such cells: the re-solve drives the
  unpinned member to 0 and leaves the pinned one at ~4e-3.
- **No sign information.** On these cells the bound multipliers are ~0, so
  γ ≈ −ξw and ν ≈ −ξr always share a sign.
- **Flipping stalls.** A flipping iteration (flip pins whose member stays > 1e-3
  while the other goes to 0) brings the violation to ~1e-3 and stalls: groups of
  pins are jointly inconsistent. A flat region also needs the TV boundary
  condition on q, |Σ(f − c)| ≤ α·perimeter.
- **Consequence.** A clean-up would need either a much smaller t (where the end
  game is the problem) or a combinatorial choice over a large fraction of the
  cells.
- **Manuscript.** Its "pin the smaller, re-solve" description should not be
  presented as working here without this caveat. The remaining structural
  option from §3 is the exact-penalty (elastic) formulation.

## 8. Step 4: the exact-penalty formulation (2026-10-06)

`--penalty π₀` (LLN 2006): product rows dropped, π Σ r(1−δ) in the objective,
r ≥ 0 and 1−δ ≥ 0 kept, π ×10 when max min(r, w) > μ^0.4 without a 10% decrease
over 3 iterations.  Gate as in the continuation, plus max r·w ≤ t_min.
- **Small instances (π₀ = 10).** cam32, mar32, mar48 converge in 40/28/73
  iterations, at r·w ≤ 2e-6 and the same PSNR to 0.02 dB.
- **N ≥ 128.** No variant converges within 400–600 iterations: adaptive π (it
  runs to the cap 1e8, PSNR 16.6 dB at N=128), fixed π = 10 or 100, with or
  without π's Hessian term.
- **Cause.** π r w adds π[[0,−1],[−1,0]] to every cell's Hessian.  With one δ
  for all tile primals the inertia correction needs δ ≳ π on most iterations
  (5–150 vs ≲ 1 in the continuation), and the steps shrink (α_pr 0.11–0.16 vs
  0.66).  LLN's examples have few complementarity pairs; here there are 16k
  (N=128) to 65k (N=256).
- **What might still work.**
  - Per-cell convexification: the inertia correction applied only to the
    (r, δ) blocks of the cells, which needs a per-variable δ in the IPM.
  - A penalty on min(r, w) with a smoothing.
  - Not tried.

## 9. Step 5: Raghunathan–Biegler's coupling and modified block (2026-10-06)

From §1, the two RB items not yet tried:
- **μ/t bounded away from zero** (Lemma 2.7; Table 5.1: μ/t = 1 best, 0.1 worse;
  we used 0.1).  `--t-mu-scale 1` sets t = max(t_min, μ).
- **The Vicente–Wright modified complementarity block** (eqs. 3.7–3.8), on the
  bounds of the complementarity variables only (r ≥ 0, 1 − δ ≥ 0, not the
  product rows): s + ηz in place of s in Z s = μ, η = 0.1 μ_prev/(1 + max z),
  from μ ≤ 5e-6 as in IPOPT-C.  `--vw-mu M`.

Results (8 ranks, t_min = 1e-4):

| instance | t = 10μ | t = μ | t = μ + VW from 5e-5 |
|---|---|---|---|
| cam256 16×16 | 161 (+10), 28.547 dB | 88 (+10), 28.554 dB | 81 (+10), 28.553 dB |
| cam256 8×8 | 99 (+15), 28.552 dB | 53 (+11), 28.539 dB | — |
| cam128 8×8 | 104, 27.453 dB | 99, 27.448 dB | 99 |
| mar128 8×8 | 97, 26.478 dB | 70 (+20), 26.478 dB | — |
| mar96 8×8 | 75, 25.784 dB | 68, 25.779 dB | — |

- **Small six (4 ranks).** t = μ: 22/29/26/28/85/79 its, against
  34/31/33/28/34/46.  Mariposa is slower but ends at better points (23.83,
  24.23 dB).
- **VW at RB's threshold 5e-6** never switches on with t_min = 1e-4: the end
  game sits at μ ≈ 1e-5.
- **Reading.**
  - The coupling, the item the notes linked to our collapsed bound multipliers,
    is the first change that shortens the N=256 end game at the same t_min and
    PSNR (−45% iterations, −37% wall on 16×16).
  - The modified block adds a little on top.
  - Open: whether α\* moving on cam256 8×8 (0.0706 vs 0.0722, −0.013 dB) is
    acceptable, and N=512.

## 10. Step 6: IPOPT-C, aligned with its source code (2026-10-06)

**The source.** IPOPT-C is not in the C++ Ipopt. It is in the last Fortran IPOPT
(COIN-OR tarball `Ipopt-Fortran_2006Oct21.tgz`, `configure --enable-mpcc`,
AMPL only).  The MPCC parts are `update_mpec_eta.f`, `get_sigma.f`,
`get_step_full.F`, `filter.F` and `AMPL_interface/ipoptAMPL.c`.

What the code does (we reimplemented it; the code is CPL-licensed, so nothing is
copied):
- **Modified step, eq. 3.7 with choice (ii) of 3.8, on every bound.**
  Σ = v/(s + ηv), and the dual step is recovered with the same denominator.
  The right-hand side is unchanged.
- **η = 0.1·μ_{l−1}/(1 + max(‖c(x)‖∞, ‖z_L‖∞, ‖z_U‖∞)).** The paper writes
  ‖λ‖∞; the code uses the constraint values, and the multipliers it is passed are
  unused.
- **Threshold and defaults.** It switches on at μ ≤ 5e-6, is off in the
  restoration phase, and is off by default (`impec_trigger = 0`).
- **Line search.** `filter.F` replaces the barrier term ∓μ/s of gᵀd by
  ∓(z + (μ − sz)/(s + ηz)).
- **Relaxation.** t = ccfact·μ^ccpow with defaults 1, 1, i.e. t = μ, recomputed
  at every evaluation, with no floor.
- **Barrier rule (not aligned here).** IPOPT-C sets μ ← μ^1.3 and solves each
  barrier problem to 5·μ^ν. Ours decreases μ while E_μ ≤ 1000μ, with κ_μ = 0.7,
  θ_μ = 1.1.

In `tv_dd`:
- `--vw-mu M` follows the source: all bounds, the η above, off in restoration,
  line search included (`--vw-linesearch plain` drops the last).
- `--t-mu-scale 1 --t-rate 0 --t-min 1e-6` gives t = μ with no floor and no
  halving limit.

### Results at the paper's target t_min = 1e-4 (8 ranks)

These runs predate the alignment: η used ‖λ‖∞, and the line search the plain
μ/s.

| instance | default (t = 10μ) | t = μ | t = 10μ + eq. 3.7 from μ ≤ 5e-5 |
|---|---|---|---|
| cam128 8×8 | 104 | 99 | 88 |
| cam256 16×16 | 161 (+10), 57 s | 88 (+10), 36 s | 76 (+16), 31 s |
| cam256 8×8 | 99 (+15) | 53 (+11), α* 0.0706 | 98 (+43) |
| mar128 8×8 | 97 | 70 (+20) | 97 |
| mar96 8×8 | 75 | 68 | 205 (+16), 1 failed phase |
| cam512 16×16 | **76 (+13), 157 s** | 292 (+91, 10 phases), 681 s | 92 (+17), 213 s |

PSNR is the same to 0.015 dB everywhere.  Neither change is a uniform gain:
- **t = μ** halves the N=256 end game but quadruples N=512.
- **Eq. 3.7** helps N=128 and N=256 16×16 but costs 2.7× on mariposa N=96 and
  20% on N=512.
- At IPOPT-C's own threshold (5e-6) eq. 3.7 never switches on, because our end
  game sits at μ ≈ 1e-5.

### Results in IPOPT-C's regime: t → 1e-6, aligned code

All runs are solved to r·w ≈ 1e-6.  Its = IPM iterations, 600 = cap without
convergence; (+n) = iterations in restoration phases.

| instance | B6: t = 10μ, t_min 1e-6 | C0: t = μ, no floor | C: IPOPT-C (C0 + eq. 3.7 from 5e-6) |
|---|---|---|---|
| cam32 4×4 / 2×2 / 8×8 (4 ranks) | **71 / 69 / 70** | 351 / 407 / 91 | 129 / 65 / 94 |
| cam48 4×4 | **50** | 600 | 389 |
| mar32 4×4 | 600 | 163, 23.83 dB | 163, 23.83 dB |
| mar48 4×4 | 196, 24.10 dB | 88, 24.26 dB | **83, 24.26 dB** |
| cam128 8×8 | **169**, 12 s | 533 (+223), 54 s | 600 (+18) |
| mar96 8×8 | 600 (+95) | 600 (+211) | **246 (+55)**, 25.81 dB, α* 0.0628 (others 0.065–0.066) |
| cam256 16×16 | 600 (+197, 16 phases), r·w 3e-5 | 600 (+197), r·w 1e-5 | 600 (+111), r·w 1e-5 |
| cam512 16×16 | stopped at it 348: 10 restoration phases, α_pr ~1e-7 | not run | not run |

**At N=256 the interface solve fails under t = μ.**  PCG hits its 2000-iteration
limit in 527 of C's 600 iterations (from it 66 on) and in 76 of C0's; B6 hits it
once.  The Newton steps are then inexact, so the Schur complement on the border
is too ill-conditioned for the one-level additive Schwarz preconditioner.  This
is consistent with κ(∇c) = O(1/√t) in the manuscript.

### Reading

- **Eq. 3.7 helps where IPOPT-C's coupling is used.**  It cuts C0 → C on
  cameraman (351 → 129, 407 → 65, 600 → 389).  On mariposa N=96 it is the only
  configuration that converges, at the better stationary points.  With our
  coupling at t_min = 1e-4 it is mixed.
- **The coupling is image dependent.**  t = 10μ is fastest on cameraman up to
  N=128; t = μ is better on mariposa.
- **No configuration solves cameraman N ≥ 256 to r·w ≈ 1e-6 within 600
  iterations.**  At t ≈ 1e-6 the degeneracy shows up in the linear algebra
  too: PCG fails, then restoration phases follow one after another.
- **The defaults stay:** t = max(t_min, 10μ), t_min = 1e-4 (or 5e-4), no
  modified step.  They converge on every instance tested, cameraman N=512 in 76
  iterations.
- **Open.**
  - IPOPT-C's barrier rule, not aligned here.
  - A coarse space for the interface preconditioner, which the PCG failures at
    t ≈ 1e-6 point to.

**Note (2026-10-06): t ≈ 1e-6 is not the operating point.**
- **The comparison.** Cameraman N=32/64/128 solved at t_min = 1e-4 and at 1e-6
  differ by at most 0.73 gray levels (rms 0.1), with α* within 0.5% and the same
  reconstruction error (README, "How small t needs to be").
- **What follows.** The runs of §10 in IPOPT-C's regime (t → 1e-6) measure where
  the method breaks down (PCG, restoration), not what the paper needs.  They were
  run to reproduce IPOPT-C's configuration, which has no floor on t.
- **What 1e-4 means for the certificate.** At t_min = 1e-4 it is approximate:
  the residuals plus the §5 multiplier classification.

## 11. t_min = 1e-5 from the two-stage start (2026-10-11)

Instance: cameraman N=128, one weight, 8×8 tiles, 4 ranks, `--gate kkt`, stage 1
to convergence.  At t_min = 1e-4 it passes the gate in 39 iterations.  At 1e-5
it reaches the 3000-iteration limit (210 restoration phases, max ξ 376, 6 M
cells at ε = 3√t).

**Where it breaks** (`--verbose 1`, first 300 iterations).
- Identical to the 1e-4 run down to t = 8.3e-5 (iteration 41: inf_pr 1.2e-4,
  inf_du 1.9e-2).
- Iteration 42: μ falls 10^-5.1 → 10^-5.6 in one iteration (κ_ε = 1000), and t
  follows to 2.6e-5 within two iterations.
- From then on:
  - inf_du swings between 1e-2 and 80 from one iteration to the next;
  - the line search backtracks to α_pr ≈ 1e-3 … 1e-9, with ‖dp‖ ≈ 1e-2;
  - restoration phases come one after another (16 in 300 iterations), and t
    never reaches 1e-5.
  - δ_w stays at 1e-4 … 1e-5 until the first restoration phase and is ≈ 1 after
    it.  So the inertia correction is not what starts the failure.

**Two hypotheses tested** (400 iterations each; none converges):

| variant | t at the end | inf_du at the end | restoration phases |
|---|---|---|---|
| default | 2.6e-5 (at 300) | ~1e2 | 16 |
| `--dual-reg 0` (no fixed δ_c) | 1e-5, μ = 1e-6 | 1.5e4; max ξ 5.5e3 | 16 |
| `--max-soc 4` (IPOPT's SOC) | 8.3e-5, stuck: the same iterate repeats | 8.4 | 0 |
| both | 2.6e-5 | 4.7e4 | 7 |

- **δ_c.** The idea was that δ_c·dλ breaks θ-descent once ξ is large.  Without
  δ_c, t does reach t_min, but ξ then grows to 5.5e3 on the same handful of M
  cells and inf_du to 1e4.  δ_c only caps the multipliers.
- **SOC.** Second-order corrections do not unblock the line search.  They stall
  earlier, at 8.3e-5.
- **First reading (corrected by step 1 below).** The end game is held by a few
  cells (6 M-stationary cells out of 16k) where ξ grows without bound as t → 0,
  as Ralph–Wright and Kanzow–Schwartz predict at non-S points.  The unscaled
  max-norm inf_du in the gate is dominated by them.
  - Step 1 shows the cells that hold it are not biactive; see below.

**Step 1: where the residual sits** (`--endgame-diag`, the same run, 400 iterations).
- **From t = 8.3e-5 on, one cell holds the maximum** in almost every iteration:
  cell (74, 0) on the left image edge, on its product-row slack (−ξ + z_s).
  - Its ξ grows steadily: 18 → 540 (iteration 59) → 1.1e3 (iteration 150).
  - Off the cells (u, α, copies) the residual stays ≤ 1e-3.
- **The high-ξ cells are not biactive** (`fz30.csv`, the per-cell record).
  - They are on the r → 0 branch, with r stuck at 1e-4…2.5e-3 and w = 1 − δ at
    0.01…0.6.  So r·w is above the cell's t: the product row is violated, and ξ
    grows trying to flatten a gradient that does not vanish.
  - γ = a − ξw is very negative there (−20 … −350).  For a cell in I_{0+} this is
    not a stationarity defect: its multiplier is free.
  - The classes at ε = √t do not see them: all biactive cells are S in the
    frozen run.
- **They form small clusters:**
  - left edge, rows 73–77, columns 0–4: a flat dark region of the clean image
    (the coat) meeting the image boundary;
  - rows 118–121, columns 18–20;
  - (60–63, 7–8) and (16–19, 84–85).
- **ξ keeps growing at fixed t.**  A frozen cell's ξ still rises (1.2e3 at
  t_e = 8.3e-5).  So it is not ξ ~ 1/√t: at those cells the barrier problem at a
  fixed t is itself not being solved.
- **The stall is above t_min.**  μ stops at 10^-5.6, because E_μ never gets
  below κ_ε·μ, and t stays at 10μ = 2.56e-5 for hundreds of iterations.  A cell
  frozen during that stall (t_e = 2.56e-5) changes nothing: `--freeze-xi 100`
  ran identically to the default.

**Step 2: `--freeze-xi X`** (opt-in; cells with ξ ≥ X for 3 iterations keep their
t; 4 ranks, three jobs at a time, so times are indicative; loss = exact loss with
200k PDHG iterations).

| run | its | t at the end | restoration phases | frozen (t_e) | exact loss | S/M/C at √t |
|---|---|---|---|---|---|---|
| cam128 scalar, default (v2) | 3000, limit | 1e-5 | 210 | — | 14.7708297 | 833/0/0 |
| X = 10 | **494, gate** | 1e-5 | 17 | 23 (2.6e-5–8.3e-5) | 14.7705853 | 993/0/0 |
| X = 20 | 600, limit | 1e-5 | 27 | 10 | 14.7706263 | 936/0/0 |
| X = 30 | **198, gate**, 39 s | 1e-5 | 2 | 12 (2.6e-5–8.3e-5) | 14.7706810 | 932/0/0 |
| X = 50 | 600, limit | 1e-5 | 23 | 22 | 14.7705446 | 936/0/0 |
| X = 100 | = default (all freezes at 2.56e-5) | | | 16 | | |
| X = 30, cam N=64 scalar / 4×4 / 8×8, mariposa 8×8 | 57 / 58 / 86 / 46 | 1e-5 | 0 | 0 | identical to v2 | identical |
| X = 30, BSDS 8×8 | 91 (v2: 97) | 1e-5 | 1 | 1 | 4.1461344 (v2: 4.1462153) | 2498/0/0 |
| X = 30, cam128 8×8 | 200 | 1e-5 | 0 | 0 | identical to v2 | identical |

- **Harmless where nothing goes wrong.**  On the six instances that converge
  anyway, X = 30 freezes nothing or one cell, and the results are identical or
  slightly better.
- **Not a reliable fix.**  On the failing instance, X = 30 and 10 pass the gate
  but X = 20 and 50 do not: the outcome is not monotone in X.
  - The run that works froze (74, 0) early, at t = 8.3e-5, before the stall.
  - Once the stall at t = 2.56e-5 has begun, freezing does not end it.
- **Reading.**  The failure mode is a few cells locked between the branches,
  with r·w > t and ξ diverging at fixed t, not ξ ~ 1/√t at non-S biactive cells.
  Freezing early lets the rest of the problem go on and can get past it, but it
  does not remove the lock.  What would target the lock directly (not tried):
  - detect r·w > t_e with ξ rising, and let such a cell keep t_e = r·w (an
    adaptive, per-cell t that follows the iterate's own product); or
  - pin the branch locally, r = 0, the manuscript's clean-up restricted to these
    few cells (the global clean-up of §7 failed on ~40% ambiguous cells; here
    there are about 10–25).

**Step 3: `--follow-comp K`, a per-cell t that follows the product** (opt-in,
2026-10-11; 4 ranks, three jobs at a time; exact loss with 200k PDHG iterations).
- **The rule.**
  - Trigger: a cell is locked when, for K iterations in a row, its product row is
    violated (r·w > t_e), ξ rises, the product does not fall, and t has not
    moved.
  - Action: t_e = r·w − s, so the row holds at the current point.
  - Afterwards t_e is only lowered, to r·w − s each iteration, and the cell is
    released once t_e ≤ t.
- **A first version without "the product does not fall, t has not moved" was a
  false pass.**  It raised t_e on 2–15k cells, mistaking the continuation's
  normal catch-up (every active row is briefly violated after t halves) for a
  lock.  Those runs pass the gate fast with t_e up to 1e-2, worse loss
  (14.7718–14.7780) and far fewer biactive cells.  Discarded.

| run | its | exact loss | S/M/C at √t | raises; cells left with t_e > t (max t_e) | v2 default |
|---|---|---|---|---|---|
| cam128 scalar, K = 2 | **199, gate** | **14.7705414** | 1037/0/0 | 177; 27 (2.5e-4) | 3000 limit, 14.7708297 |
| cam128 scalar, K = 3 | **221, gate** | 14.7709798 | 1020/0/0 | 121; 25 (2.0e-4) | |
| cam128 scalar, K = 5 | **544, gate** | 14.7709840 | 1042/0/0 | 122; 31 (2.0e-4) | |
| cam128 8×8, K = 3 | 239 | 11.9283885 (−0.05%) | 921/0/0 | 20; 15 (2.4e-4) | 200, 11.9347870 |
| cam64 scalar | 86 | 4.0908399 (+0.007%) | 212/0/0 | 13; 13 (4.9e-4) | 57, 4.0905736 |
| cam64 4×4 | 58 | identical | 188/5/5 | 0 | 58 |
| cam64 8×8 | 80 | 3.4937227 | 196/0/0 | 15; 8 (3.5e-5) | 86, 3.4937371 |
| mariposa 8×8 | 66 | 4.9059660 (+0.017%) | 118/0/0 | 212; 212 (2.2e-3) | 46, 4.9051133 |
| BSDS 8×8 | 97 | identical | 2525/0/0 | 0 | 97 |

- **It fixes the failing instance for every K tried** (2, 3, 5).  This is unlike
  `--freeze-xi`, whose outcome was not monotone in X.
- **The price is cells left relaxed:** 25–31 cells end with t_e ≤ 2.5e-4 instead
  of 1e-5.  The certificate there is that of a larger t.
- **It also fires on instances that converge without it,** because the lock test
  holds in an ordinary end phase at t = t_min.
  - Mariposa: 212 cells left at up to 2.2e-3, 20 more iterations, loss +0.017%.
  - Cameraman N=64 scalar: 13 cells, 29 more iterations, loss +0.007%.
  - Cameraman N=128 8×8: 15 cells, a lower loss (−0.05%), a different
    stationary point.
- **Not a default as it stands.**  It would need a trigger that fires only in a
  real stall (e.g. after a restoration phase, or when μ has not moved for M
  iterations), and the report must list the cells with t_e > t_min next to the
  S/M/C classes.

**Step 4: the lock test armed only in a stall** (`--follow-stall M`, default 20:
μ and t unchanged for M iterations in a row; same instances and settings).

| run | its | exact loss | S/M/C at √t | raises; left with t_e > t (max t_e) | v2 default |
|---|---|---|---|---|---|
| cam128 scalar, K = 2 | **271, gate** | 14.7709777 | 1060/0/0 | 437; 37 (1.8e-4) | 3000 limit, 14.7708297 |
| cam128 scalar, K = 3 | **179, gate** | 14.7705871 | 1030/0/0 | 126; 36 (2.1e-4) | |
| cam128 scalar, K = 3, M = 10 | **165, gate** | 14.7705666 | 1048/0/0 | 145; 37 (1.7e-4) | |
| cam128 scalar, K = 5 | 600, limit, t = 2.6e-5 | 14.7720683 | — | 213; 11 | |
| cam128 8×8, K = 3 | 197 | 11.9345098 | 926/0/0 | 1; 0 | 200, 11.9347870 |
| cam64 scalar / 4×4, mariposa 8×8 (M = 20 and 10), BSDS 8×8 | identical to v2 | identical | identical | 0 | |
| cam64 8×8 | 86 | 3.4937373 (v2 3.4937371) | 196/0/0 | 4; 2 (1.1e-5) | 86 |

- **No collateral effect.**  On the six instances that converge without it, it
  does not fire at all (four instances), or it raises 1–4 cells and leaves none,
  or two at t_e ≈ t_min.
- **On the failing instance it passes for K = 2 and 3** (165–271 iterations
  instead of the 3000 limit), with 36–37 cells ending at t_e ≤ 2.1e-4.  K = 5
  reacts too slowly: the stall at t = 2.6e-5 persists.
- **The certificate.**  Every biactive cell at ε = √t is S.  The 36–37 followed
  cells are certified only at their own t_e (≤ 2.1e-4, about 20 t_min).  The
  report lists their number and largest t_e.
- **Candidate default `--follow-comp 3 --follow-stall 20`,** but tested on one
  failing instance only.  Before switching: other runs that stall
  - cameraman N=64 scalar at t_min = 1e-6 (1685 its before);
  - mariposa N=96/128 scalar;
  - cameraman N=256.

**Step 5: other runs that stall** (scalar weight, two-stage start, `--gate kkt
--classify`, exact loss with 200k PDHG iterations; 4 ranks, three jobs at a
time, so times are indicative; F = `--follow-comp 3 --follow-stall 20`).

| instance | default: its, restoration phases, time, exact loss | F: its, time, exact loss | F: raises; left with t_e > t_min (max t_e) | F: S/M/C at √t |
|---|---|---|---|---|
| cam N=64, t_min = 1e-6 | 2000 limit, 214, 1556 s, 4.0905769 | **300, gate**, 15 s, 4.0905760 | 287; 125 (5.5e-5) | 118/0/0 |
| mariposa N=96, 1e-5 | 51, gate, 0, 7 s, 12.1049150 | identical (no raise) | 0 | 353/5/12 |
| mariposa N=128, 1e-5 | 1000 limit, 47, 1141 s, 18.4801669 | **244, gate**, 68 s, 18.4635336 (−0.09%) | 275; 49 (1.1e-4) | 801/0/0 |
| cam N=256 16×16 tiles, 1e-5 | 800 limit at t = 2.6e-5, 56, 2972 s, 45.9069812 | **559, gate**, 496 s, 45.8878595 (−0.04%) | 11 205; 399 (2.3e-4) | 4899/0/0 |

- **Every stalling run passes the gate with F; none passes without it.**  The
  defaults stall the way cam N=128 did: at a fixed (μ, t), with t = 2.6e-5 at
  N=256, and restoration phases one after another.
- **The exact loss is equal or lower** (−0.04% to −0.09% on the two large runs).
- **The cost is the cells left relaxed:** 49 (mariposa 128), 125 (cam64 at
  1e-6, 55·t_min) and 399 (cam256; 0.6% of 65k cells) end with t_e ≤ 2.3e-4.
  N=256 needed 11k raises, so its stall is wider than one cluster.
- **Instances that do not stall are unchanged** (mariposa N=96, and the six of
  step 4).
- **What a default would need, for the paper:**
  - the report of the followed cells next to the S/M/C classes (already
    printed);
  - a statement that the certificate on those cells holds at their t_e;
  - optionally, the classes of the followed cells at ε = √t_e.

**Step 6: the default, in both binaries (2026-10-11).**
- `--follow-comp 3 --follow-stall 20` is now the default in `tv_learn` and `tv_dd`.
  - `TvMpcc` has the same `follow_cells()`.
  - It is off under `--penalty`, and it skips cells flagged by `--relax-cells`.
  - `--follow-comp 0` turns it off.
  - In `tv_learn`, `--freeze-xi` turns it off.
- **The bit-identity gate still passes:** cameraman 32, 4×4 tiles, `--start uniform`: 40 identical iterations, the same as before the change, with no raise.
- **`tv_dd` at its defaults (`--gate kkt --classify -`) gives the same numbers as `tv_learn` with F:**

  | instance | its | raises | cells left | max t_e | biactive at √t |
  |---|---|---|---|---|---|
  | cameraman N=128 scalar | 179 | 126 | 36 | 2.05e-4 | S 1030 |
  | mariposa N=128 | 244 | 275 | 49 | 1.14e-4 | S 801 |
