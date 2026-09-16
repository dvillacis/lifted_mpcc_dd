// =============================================================================
//  dd_solver_simple.hpp — a READABLE domain-decomposition (arrowhead) linear
//  solver for IPOPT.  Eigen only: no HSL/MA57, no MA97, no MUMPS.
//
//  This is the teaching twin of dd_solver.hpp, reduced to the one configuration
//  that survived measurement:
//
//      W_k     Eigen::SimplicialLDLT — sparse unpivoted LDLᵀ, one per subdomain
//      S       never assembled, never factorized: ASd-preconditioned CONJUGATE
//              GRADIENTS on the peeled interface, applied matrix-free through
//              the local Schur blocks S_k
//      In(S)   PREDICTED from the tiny dense peel complement T (§8)
//
//  Full derivations, references and the measurement record live in the
//  companion paper, docs/dd_solver_simple/dd_solver_simple.tex.  This header
//  is self-contained at the "why is this line here" level; the paper is the
//  "prove it" level.
//
//  VALIDATION GATES
//      dd_simple_smoke.cpp   standalone (no IPOPT, no HSL): the arrowhead
//                            solve vs a dense LU, the predicted inertia vs a
//                            dense symmetric eigendecomposition.  Run it after
//                            ANY edit to this file.
//      DDS_SCHUR_CHECK=1     every S_k re-derived by the naive two-sided route
//                            and compared, per block per factorization
//      DDS_DEBUG=1           partition / CG / refusal diagnostics (implies
//                            DDS_WARN=all)
//      DDS_WARN=all|off      §10 warnings: every occurrence, or none at all.
//                            The default prints the first of each kind and
//                            counts the rest; Warnings::get().report(os)
//                            prints the tally (dd_solve_2d does, per run)
//
//  CONTENTS                                                     (code map)
//      §1  the IPOPT contract                                    ─ adapter
//      §2  the arrowhead permutation and the Schur solve         ─ Arrowhead
//      §3  inertia by Haynsworth additivity                      ─ factorize
//      §4  the subdomain blocks: unpivoted LDLᵀ                  ─ Ldlt
//      §5  forming S_k: forward-only, static reach, contraction  ─ factorize
//      §6  the peel                                              ─ peel cache
//      §6a why the cross points must be peeled too               ─ peel sets
//      §7  ASd-preconditioned conjugate gradients                ─ Precond, cg
//      §8  the predicted inertia                                 ─ peel cache
//      §9  refinement and failure semantics                      ─ solve
//      §10 what a run warns about, and why it is only a warning  ─ Warnings
//
// -----------------------------------------------------------------------------
//  §1  WHAT IPOPT ASKS OF A LINEAR SOLVER
// -----------------------------------------------------------------------------
//  At every Newton step IPOPT hands over the symmetric augmented KKT matrix A —
//  lower triangle only, in triplet (i, j, value) form, 1-based, possibly with
//  duplicate entries that must be SUMMED — with its own δ_w/δ_c regularization
//  ALREADY applied.  It wants two things back:
//
//     (a) a solve            A · Δz = r
//     (b) the INERTIA        n_neg = #negative eigenvalues of A
//
//  (b) is not optional.  IPOPT's inertia-correction loop reads n_neg to decide
//  whether the current δ_w makes the reduced Hessian positive definite on the
//  null space of the constraints; a wrong answer sends the filter line search
//  chasing a curvature defect that is not there.
//
//  Failure semantics — and the principle behind every failure path here:
//
//      SYMSOLVER_SINGULAR        IPOPT raises δ_w/δ_c and retries
//      SYMSOLVER_WRONG_INERTIA   IPOPT raises δ_w and retries
//
//  Both retries are REGULARIZATION.  So this file reports SINGULAR only for
//  conditions a larger δ can actually cure (a broken factorization, an inertia
//  it cannot stand behind) and never for conditions it cannot (an iterative
//  solve that merely failed to converge — §9 explains what happens instead).
//
//  IPOPT owns everything else: the filter line search, the μ-homotopy, the
//  restoration phase.  This file owns only the linear algebra.
//
// -----------------------------------------------------------------------------
//  §2  THE ARROWHEAD IDEA
// -----------------------------------------------------------------------------
//  The caller provides an OWNER MAP: for every KKT index, the subdomain that
//  owns it, or −1 meaning "border" (a complicating unknown, shared by ≥ 2
//  subdomains).  Permuting the border indices last turns A into a bordered
//  block-diagonal — "arrowhead" — matrix:
//
//        ⎡ W_1              B_1ᵀ ⎤            W_k : interior block of subdomain k
//        ⎢       W_2        B_2ᵀ ⎥            B_k : its coupling to the border
//    A = ⎢            ⋱      ⋮   ⎥            C   : border–border block
//        ⎢                W_K B_Kᵀ⎥
//        ⎣ B_1   B_2  ⋯   B_K  C ⎦
//
//  This is a pure PERMUTATION — no unknowns are duplicated, no linking
//  constraints are introduced — so C is genuinely nonzero and the B_k carry
//  real Jacobian/Hessian entries.  It requires that no triplet couple two
//  DIFFERENT subdomains directly; route_triplets() verifies that and rejects
//  the owner map as a "partition leak" otherwise.
//
//  Block elimination of the W_k gives the INTERFACE (Schur complement) system
//
//        S = C − Σ_k B_k W_k⁻¹ B_kᵀ                                       (2.1)
//
//  and the three-line solve, for a right-hand side split as r = (r_1…r_K, r_y):
//
//        r_S  = r_y − Σ_k B_k W_k⁻¹ r_k                                   (2.2)
//        Δy   = S⁻¹ r_S                                                   (2.3)
//        Δx_k = W_k⁻¹ (r_k − B_kᵀ Δy)                                     (2.4)
//
//  Everything with a subscript k is INDEPENDENT across k — that is the whole
//  point.  Only (2.3) is global, and it is the only place a global object
//  could appear.  It never does: S is applied matrix-free,
//
//        S·y = C·y + Σ_k N_k S_k N_kᵀ y,      S_k = −B_k W_k⁻¹ B_kᵀ,      (2.5)
//
//  through the LOCAL dense Schur blocks S_k, which each subdomain forms on its
//  own (§5).  N_k selects the border positions subdomain k touches.  This is
//  the distributed algorithm, not a simulation of it: the only communication a
//  real implementation needs per CG iteration is one reduction over the border
//  vector.
//
// -----------------------------------------------------------------------------
//  §3  INERTIA BY HAYNSWORTH ADDITIVITY
// -----------------------------------------------------------------------------
//  For symmetric M = [E Fᵀ; F G] with E nonsingular, Haynsworth's identity
//  states In(M) = In(E) + In(G − F E⁻¹ Fᵀ).  Applied to the arrowhead:
//
//        In(A) = Σ_k In(W_k) + In(S)                                      (3.1)
//
//  Σ_k In(W_k) is FREE — the pivot signs of factorizations that happen anyway,
//  locally, in parallel.  (Eigenvalues would not do: the barrier terms Σ ~
//  z²/μ drive ‖A‖ to ~1e18, where the signs of small eigenvalues are rounding
//  noise.  Pivot signs of an LDLᵀ are exact regardless of scaling — Sylvester's
//  law.  Every inertia statement in this file reduces to pivot signs or to an
//  exact eigendecomposition of a small, deliberately equilibrated matrix.)
//
//  In(S) is the term that would classically force assembling and factorizing
//  S — the one serial step in an otherwise embarrassingly parallel scheme.
//  §8 derives it instead from a second application of the same identity.
//
// -----------------------------------------------------------------------------
//  §4  THE SUBDOMAIN BLOCKS: UNPIVOTED LDLᵀ
// -----------------------------------------------------------------------------
//  Each W_k is factorized by Eigen::SimplicialLDLT, which does NOT pivot for
//  stability — the one real weakness of the Eigen route.  The saving grace is
//  structural: with δ_w > 0, δ_c > 0 the KKT matrix is symmetric QUASI-DEFINITE
//  (positive definite (1,1) block, negative definite (2,2) block), and an
//  unpivoted LDLᵀ of a quasi-definite matrix always exists (Vanderbei).  But
//  IPOPT usually offers δ_w = δ_c = 0 first, where a zero pivot is possible.
//  The breakdown is detected exactly (a zero or non-finite entry of D) and
//  reported SINGULAR; IPOPT responds with δ_w, δ_c > 0 — precisely the
//  regularization that makes the unpivoted factorization safe.  Self-
//  correcting, at the price of a few extra factorizations per run relative to
//  a pivoting Bunch–Kaufman code such as MA57.  That price is the honest cost
//  of dropping HSL, and it is why dd_solver.hpp uses MA57.
//
//  No artificial shift is ever added inside the solver: masking a local rank
//  deficiency would corrupt the inertia signal (3.1) that drives IPOPT's δ_w
//  loop.
//
// -----------------------------------------------------------------------------
//  §5  FORMING S_k: FORWARD-ONLY, STATIC REACH, ROW CONTRACTION
// -----------------------------------------------------------------------------
//  S_k is mathematically DENSE — entry (a,b) is −b_aᵀ W_k⁻¹ b_b with W_k⁻¹
//  full — so the design question is not how to store it but how cheaply it can
//  be FORMED, and (per §7) it is worth forming: CG applies it as a dense
//  p_k × p_k GEMV, cheaper per iteration than any factored or matrix-free
//  alternative, and the ASd preconditioner needs its entries anyway.
//
//  Naively the formation is p_k full back-solves through W_k with dense
//  right-hand sides — the dominant cost of the whole factorization.  Three
//  exact observations remove nearly all of it:
//
//  (a) FORWARD ONLY.  With W = P⁻¹ L D Lᵀ P (Eigen applies no scaling, so the
//      halves compose exactly) and Pᵀ = P⁻¹,
//
//          B W⁻¹ Bᵀ = (L⁻¹ P Bᵀ)ᵀ D⁻¹ (L⁻¹ P Bᵀ) = Yᵀ D⁻¹ Y.              (5.1)
//
//      The Lᵀ half of the solve cancels against the B multiplying it back on
//      the left: the BACKWARD substitution is never performed, and B itself
//      disappears — Y already carries it.
//
//  (b) STATIC REACH.  The columns of Bᵀ hold a handful of nonzeros each, and a
//      forward substitution propagates column j of L only through a nonzero
//      multiplier — so each column of Y lives on the REACH of its right-hand
//      side: the union of elimination-tree paths from the RHS pattern to the
//      root (Gilbert's theorem).  The pattern is fixed for the whole run, so
//      the reach is computed ONCE per column (route_triplets) and both the
//      substitution and everything downstream iterate only those lists.
//      Nothing of size n_k is ever swept: profiled before this structure
//      existed, the pattern-blind O(n_k) sweeps and scans over the mostly-
//      empty dense Y buffer were ~90% of the formation time — the arithmetic
//      was never the cost.
//
//  (c) ROW CONTRACTION.  The product (5.1) is contracted by rows,
//
//          S_k = − Σ_r (1/d_r) · y_rᵀ y_r,     y_r = the r-th row of Y,   (5.2)
//
//      at cost Σ_r nnz(y_r)² ≤ p_k·nnz(Y), instead of a dense GEMM's
//      2·n_k·p_k² — which multiplied almost nothing but zeros.
//
//  Measured together (cameraman N=64, exact Hessian, whole-run wall clock,
//  identical iterations): 2×2 tiles 42.8 s → 16.0 s, 4 strips 25.7 s → 8.1 s
//  against the dense-GEMM formation.  DDS_SCHUR_CHECK=1 referees every block
//  (rel-err ~1e-11 typical).
//
// -----------------------------------------------------------------------------
//  §6  THE PEEL
// -----------------------------------------------------------------------------
//  CG needs a symmetric POSITIVE DEFINITE operator, and S is generally not.
//  THREE kinds of border index are to blame, and the peel removes all three:
//
//    (i)   DUAL border indices.  On a k×k tile partition the driver promotes
//          the cut-corner dual pairs to the border (otherwise the local blocks
//          are structurally rank-deficient).  Those promoted duals are EXACTLY
//          the negative eigenvalues of S — measured: In(S)_neg equals the
//          promoted count at every solve — so on tiles S is indefinite by
//          construction and plain CG can never run on it.
//    (ii)  The scalar α.  It appears in every subdomain, so its row/column of
//          S is dense, wrecking both sparsity and conditioning.
//    (iii) The CROSS POINTS — border positions touched by ≥ 3 subdomains.
//          These do not make S indefinite; they make it BADLY CONDITIONED as
//          the subdomain count grows, which is a different and subtler defect
//          (§6a below).
//
//  Split the border into the peeled set P (all three of the above) and the
//  kept set f:
//
//        ⎡ S_ff  S_fP ⎤ ⎡Δy_f⎤   ⎡r_f⎤
//        ⎣ S_Pfᵀ S_PP ⎦ ⎣Δy_P⎦ = ⎣r_P⎦
//
//  With  Z = S_ff⁻¹ S_fP  and  T = S_PP − S_fPᵀ Z  (dense, |P| × |P|):
//
//        Δy_P = T⁻¹ (r_P − S_fPᵀ S_ff⁻¹ r_f)                              (6.1)
//        Δy_f = S_ff⁻¹ r_f − Z Δy_P                                       (6.2)
//
//  The indefinite, dense and cross-point directions now live entirely inside
//  T — which is solved EXACTLY — and S_ff, the operator CG actually iterates
//  on, is SPD whenever the design premise holds.
//
// -----------------------------------------------------------------------------
//  §6a  WHY THE CROSS POINTS MUST BE PEELED TOO
// -----------------------------------------------------------------------------
//  This is the FETI-DP corner rule, and it is not an analogy: Farhat, Lesoinne,
//  LeTallec, Pierson & Rixen (IJNME 50:1523–1544, 2001) define a corner as
//  "its cross points — that is, the points belonging to more than two
//  subdomains" (their definition D1) and make exactly those unknowns primal.
//  T is then precisely their coarse problem, and the reason it works in 2D is
//  a theorem: Mandel & Tezaur (Numer. Math. 88:543–558, 2001) prove that in
//  two dimensions CORNER CONSTRAINTS ALONE give a condition number bounded by
//  C(1+log(H/h))², INDEPENDENT of the number of subdomains.  (The edge and
//  face averages that dominate the BDDC literature are a 3D requirement — in
//  2D they are not needed.)
//
//  Without them the interface preconditioner is ONE-LEVEL, using only local
//  information, and one-level preconditioners are known not to scale in the
//  partition count.  Lueg, Bynum, Laird & Biegler (Optim. Eng. 27:555–585,
//  2026) — the source of this file's ASd preconditioner — say so of their own
//  method, and propose a two-level coarse correction (their eq. 22) as the
//  remedy, leaving its application to optimization Schur complements as future
//  work.  Note their PDE test problem cuts along time breakpoints, where "any
//  set of two complicating variables are at most shared by one partition" —
//  i.e. NO cross points at all, so ASd was never exercised in this regime.
//  Peeling the cross points is the cheapest possible coarse correction: they
//  go into T, which is already built and already solved exactly.
//
//  MEASURED (cameraman, --hessian exact), cross-point peel off → on:
//
//        N=64  4 strips    129 it, 7.99 s  →  bit-identical (no cross points)
//        N=64  2×2 tiles   178 it, 11.4 s  →  170 it, 10.6 s
//        N=32  3×3 tiles   195 it, 6.31 s  →  113 it, 3.24 s
//        N=32  4×4 tiles  2251 it,  160 s  →  509 it, 36.8 s
//        N=64  4×4 tiles  did not converge →  782 it,  173 s
//
//  Same PSNR in every row.  The gain scales with the cross-point count, and
//  STRIPS ARE BIT-IDENTICAL because they have none — which is the mechanism
//  confirming itself.
//
//  SIZES.  On a k×k tile partition, measured: (k−1)² cross points, each of
//  degree 3, and 4(k−1)² promoted duals (four per cut-corner cell — the driver
//  promotes two rank-1 dual pairs there).  So
//
//        |P| = 4(k−1)² + (k−1)² + 1 = 5(k−1)² + 1,
//
//  measured 6, 21, 46 at k = 2, 3, 4 against interfaces of p = 128, 261, 400.
//  On strips and in 1D there are no cross points and no promoted duals, and
//  |P| = 1 (α alone).  |P| still grows LINEARLY with the subdomain count, so
//  T is only cheap at modest k; that growth is what the DD literature answers
//  with multilevel or inexact coarse solves (Klawonn & Rheinbach, "Inexact
//  FETI-DP methods", IJNME 69:284–307, 2007; Tu, "Three-level BDDC", SISC
//  29:1759–1780, 2007), and build_peel_sets() carries a guard that declines
//  the cross-point peel outright when it would not stay small.
//
//  Z costs one CG solve per column of S_fP, once per factorization — the
//  expensive part of the scheme, and the honest part: EVERY system involving
//  S_ff in this file goes through CG.  Two hard-won details (see the peel
//  cache code for the measurements):
//    · each column is WARM-STARTED from its last accepted value — the systems
//      change slowly along the barrier path, and the dominant cold-start
//      failure mode was CG stalling with zero progress on a column it could
//      solve fine one build later;
//    · a column that still fails means the prediction of §8 cannot be issued,
//      and the whole factorization is refused (SINGULAR) — see §8.
//
// -----------------------------------------------------------------------------
//  §7  ASd-PRECONDITIONED CONJUGATE GRADIENTS
// -----------------------------------------------------------------------------
//  A Krylov interface solve stands or falls on its preconditioner.  Measured
//  (cameraman N=16, 2×2 tiles): with ASd, CG carried the interface solves;
//  with plain Jacobi it stalled on EVERY one.  ASd (Lueg eq. 20–21) is
//  therefore the ONLY preconditioner in this file — see Precond below for the
//  construction and why its diagonal swap is the whole trick.
//
//  The CG routine itself (cg_solve) is textbook PCG plus three provisions
//  that turn "CG assumes SPD" into a runtime check: an rz ≤ 0 guard (the
//  preconditioner is not SPD), a pAp ≤ 0 guard (the operator is not SPD along
//  this direction), and best-iterate memory so a stalled solve still returns
//  something usable.  Interface answers are ACCEPTED only if both CG's own
//  residual and the residual of the full interface system pass 1e-2; §9 says
//  what happens to rejected ones.
//
//  HOW THIS MAPS ONTO LUEG §3.1.  Their strategy is PCG on the Schur system,
//  admissible because THEIR S is SPD outright once inertia correction
//  succeeds — their complicating variables are primal-only and appear only in
//  linear linking constraints, so no dual ever reaches the border.  Ours do
//  (the promoted corner duals), which is why the peel (§6) must first carve
//  out the non-SPD directions: S_ff plays the role their S plays, and CG runs
//  there.  Their §3.5 offers two In(S) checks for the iterative mode — the
//  pAp ≥ 0 monitoring inside PCG ("not a rigorous check", their words) and a
//  conservative per-block In(S_k) test.  The first is exactly our pAp guard,
//  kept as guard and telemetry; the second is inapplicable here, since our
//  S_k routinely carry dual contributions and are indefinite by design.  The
//  §8 prediction replaces both with an exact count.  Finally, their
//  implementation notes name tight PCG tolerances as forced by the absence of
//  a filter line search and of iterative refinement, and list both as
//  "sensible extensions" — this solver has both (IPOPT's filter, and §9),
//  which is what lets the acceptance threshold sit at 1e-2 rather than their
//  1e-9.
//
// -----------------------------------------------------------------------------
//  §8  THE PREDICTED INERTIA — In(S) without S
// -----------------------------------------------------------------------------
//  Haynsworth applies to the peel split of S exactly as it applied to the
//  arrowhead split of A:
//
//        In(S) = In(S_ff) + In(T)                                         (8.1)
//
//  and T is ALREADY BUILT — it is the peel cache of §6, |P| × |P|, so In(T)
//  costs a dense symmetric eigendecomposition of a tiny matrix: nothing.  The
//  entire prediction is one assumption:
//
//        In(S_ff) = 0,  i.e. the kept block is positive definite,         (8.2)
//
//  and what makes that the right assumption to bet on is that it is ALSO
//  exactly what CG requires: the peel exists precisely to make S_ff definite.
//  The scheme is self-consistent — the operator is usable exactly when the
//  inertia is right — and the answer reported to IPOPT is
//
//        n_neg(A) = Σ_k n_neg(W_k) + n_neg(T).                            (8.3)
//
//  Scored against a referee LDLᵀ of the assembled S (temporarily reinstated
//  for the experiment), the prediction is right 199/200 factorizations at
//  N=32 3×3 tiles, every miss off by exactly one, with identical final
//  solutions — the record is in the paper.
//
//  REFUSING TO PREDICT.  If the peel cache cannot be built — a Z column fails
//  to converge, or an eigenvalue of T is too close to zero for its sign to be
//  meaningful — no prediction is issued and the factorization is reported
//  SINGULAR.  A prediction the solver cannot stand behind would corrupt
//  IPOPT's δ_w loop silently; SINGULAR merely costs a re-factorization at a
//  larger δ_w, a failure IPOPT knows how to cure.  (The zero test is made on
//  the diagonally EQUILIBRATED T — a congruence, so the inertia is unchanged —
//  because T mixes barrier-scaled α directions ~1e20 with corner-dual
//  directions ~1e-9, and a test against the global eigenvalue scale mistakes
//  honest tiny eigenvalues for noise: it refused every factorization of the
//  N=64 4×4 run before the equilibration existed.)
//
// -----------------------------------------------------------------------------
//  §9  REFINEMENT AND WHAT HAPPENS WHEN CG'S ANSWER IS POOR
// -----------------------------------------------------------------------------
//  Every solve runs the arrowhead recursion (2.2)–(2.4) inside a few sweeps of
//  ITERATIVE REFINEMENT measured against the ORIGINAL input triplets.  This is
//  not a luxury: near-singular pivots (IPOPT's own δ_c = 1e-8·μ^¼ on the dual
//  directions, for instance) lose ~10 digits through the Schur recursion —
//  measured rel-res ~0.1 unrefined against ~1e-12 refined.  Because the
//  residual is evaluated straight from the triplets, refinement checks the
//  DECOMPOSITION, not just the arithmetic inside it.
//
//  Refinement is also the answer to "what if CG's interface answer is bad?".
//  S was never factorized, so a rejected answer has no direct solve to fall
//  back on.  Reporting SINGULAR would be truthful but useless — δ_w cannot fix
//  a Krylov convergence failure, and the run dies (measured).  So the best
//  iterate is handed back anyway, refinement judges it against the true
//  matrix, and the best refined step seen is returned — the same bargain a
//  monolithic sparse solver makes at a nasty iterate: return your answer and
//  let IPOPT's globalization cope.
//
// -----------------------------------------------------------------------------
//  WHAT IS NOT HERE (deliberately — it lives in dd_solver.hpp)
// -----------------------------------------------------------------------------
//    · MA57 / MA97 / MUMPS block backends and the MUMPS partial-Schur route
//    · a direct or MINRES interface solve; an assembled or factorized S
//    · the second (nested) arrowhead level, the lagged Schur cache
//    · frozen value-slot maps (here the sparse blocks are rebuilt from
//      triplets every Newton step — a few percent slower, far clearer)
//    · the singular-block census, the arrowhead dump, the built-in geometry
//      (here the owner map is always injected)
// =============================================================================
#ifndef DD_SOLVER_SIMPLE_HPP
#define DD_SOLVER_SIMPLE_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace ddsimple {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// =============================================================================
//  Warnings (§10) — the one place a run says out loud that something went wrong.
//
//  Everything reported here is RECOVERABLE BY CONSTRUCTION, which is why these
//  events used to be visible only under DDS_DEBUG: an unpivoted LDLᵀ breakdown
//  becomes SYMSOLVER_SINGULAR and IPOPT answers it by raising δ_w (§4), and a CG
//  answer that fails the acceptance test is still handed back for §9's
//  refinement to judge.  Recoverable is not the same as harmless, though: a run
//  that converges only because IPOPT kept regularizing around a saturating
//  interface solve looks, from the outside, exactly like one that never
//  struggled.  These warnings are the difference.
//
//  They fire PER NEWTON STEP, so printing every one would bury the output.
//  Hence: the FIRST of each kind is printed in full, all of them are counted,
//  and report() prints the tally once at the end of the run.
//      DDS_WARN=all   print every occurrence (DDS_DEBUG implies it)
//      DDS_WARN=off   print nothing; the counters and report() still work
//
//  Not thread-safe, and does not need to be: every call site is serial code —
//  the two OpenMP loops (§4 factorization, §2.2/2.4 solves) report their
//  failures from the serial pass that follows.
// =============================================================================
enum class Warn {
   LdltSymbolic,   // Eigen's analyzePattern failed: the structure is unusable
   LdltNumeric,    // W_k breakdown: Eigen info() != Success, or a zero/NaN pivot
   CoarseNotSpd,   // the DDS_COARSE Galerkin matrix is not SPD (term left off)
   PrecondNotSpd,  // an ASd block is indefinite (used anyway, via U|Λ|⁻¹Uᵀ)
   CgPeelColumn,   // a Z column of the peel cache did not converge → SINGULAR
   CgRejected,     // the interface answer failed the §7 acceptance test
   CgUnusable,     // ... and there was no iterate at all to fall back on
   StepResidual,   // the refined step is still poor against the true triplets
   N_KINDS
};

inline const char* warn_kind(Warn w) {
   switch (w) {
      case Warn::LdltSymbolic: return "symbolic analysis failed";
      case Warn::LdltNumeric:  return "W_k factorization broke down";
      case Warn::CoarseNotSpd: return "coarse Galerkin matrix not SPD";
      case Warn::PrecondNotSpd: return "ASd preconditioner block not SPD";
      case Warn::CgPeelColumn: return "peel column did not converge";
      case Warn::CgRejected:   return "interface CG answer rejected";
      case Warn::CgUnusable:   return "interface solve produced nothing usable";
      case Warn::StepResidual: return "step residual above tolerance";
      default:                 return "unknown";
   }
}

class Warnings {
public:
   static Warnings& get() { static Warnings w; return w; }

   void add(Warn w, const std::string& detail) {
      const int i = (int)w;
      const bool first = (count_[i]++ == 0);
      if (mode_ == Off || (mode_ == First && !first)) return;
      std::cerr << "[dds] WARNING: " << warn_kind(w) << " — " << detail;
      if (mode_ == First)
         std::cerr << "\n              (further occurrences of this warning are"
                      " counted, not printed; DDS_WARN=all shows them all)";
      std::cerr << "\n";
   }

   long count(Warn w) const { return count_[(int)w]; }
   long total() const {
      long t = 0;
      for (int i = 0; i < (int)Warn::N_KINDS; ++i) t += count_[i];
      return t;
   }
   void reset() {
      for (int i = 0; i < (int)Warn::N_KINDS; ++i) count_[i] = 0;
   }

   // End-of-run tally, one entry per kind that fired.  Prints NOTHING when the
   // run produced no warnings, so a clean run's output stays clean.
   void report(std::ostream& os, const char* prefix = "  ") const {
      if (total() == 0) return;
      os << prefix << "warnings:";
      for (int i = 0; i < (int)Warn::N_KINDS; ++i)
         if (count_[i]) os << "  " << warn_kind((Warn)i) << "=" << count_[i];
      os << "\n";
   }

private:
   Warnings() {
      const char* w = std::getenv("DDS_WARN");
      const std::string v = w ? w : "";
      if (v == "off" || v == "0") mode_ = Off;
      else if (v == "all" || std::getenv("DDS_DEBUG")) mode_ = All;
   }
   enum Mode { Off, First, All };
   Mode mode_ = First;
   long count_[(int)Warn::N_KINDS] = {0};
};

inline void warn(Warn w, const std::string& detail) { Warnings::get().add(w, detail); }

// =============================================================================
//  Ldlt — sparse symmetric LDLᵀ (§4), plus the raw pieces §5 needs.
//
//  Wraps Eigen::SimplicialLDLT so the "did it break down?" test lives in
//  exactly one place.  Only the LOWER triangle of the matrix handed to it is
//  read, so lower-only and fully symmetric storage both work.
// =============================================================================
class Ldlt {
public:
   // Symbolic analysis: fill-reducing ordering + elimination tree.  Depends on
   // the sparsity pattern ONLY, and the KKT pattern is constant for a whole
   // run, so this is done once and every Newton step reuses it.
   bool analyze(const SpMat& A) {
      f_.analyzePattern(A);
      analyzed_ = (f_.info() == Eigen::Success);
      if (!analyzed_) why_ = "Eigen analyzePattern() did not return Success";
      return analyzed_;
   }

   // Numeric factorization.  False = this matrix is unusable (breakdown or a
   // zero pivot, §4); the caller must report SINGULAR rather than solve.
   bool factorize(const SpMat& A) {
      ok_ = false;
      n_neg_ = 0;
      why_.clear();
      if (!analyzed_ && !analyze(A)) return false;
      f_.factorize(A);
      // Eigen's own complaint, verbatim in spirit: NumericalIssue is what
      // SimplicialLDLT reports when the unpivoted factorization cannot be
      // completed.  Kept as a string so the caller can say WHICH block and why.
      if (f_.info() != Eigen::Success) {
         why_ = (f_.info() == Eigen::NumericalIssue)
                   ? "Eigen SimplicialLDLT reported NumericalIssue"
                   : "Eigen SimplicialLDLT reported info() != Success";
         return false;
      }
      // Inertia = signs of D (Sylvester, §3).  A zero or non-finite pivot
      // means the unpivoted factorization has broken down: the solve would
      // return garbage and the inertia would be meaningless, so both are
      // refused.
      const Vec& d = f_.vectorD();
      for (int i = 0; i < d.size(); ++i) {
         if (!std::isfinite(d[i]) || d[i] == 0.0) {
            std::ostringstream m;
            m << "pivot D[" << i << "] = " << d[i] << " (zero or non-finite)";
            why_ = m.str();
            return false;
         }
         if (d[i] < 0.0) ++n_neg_;
      }
      ok_ = true;
      return true;
   }

   // In-place solve for nrhs right-hand sides stored column-major (dim × nrhs).
   // The explicit temporary is not optional: Eigen's solve() writes its
   // destination as it goes and must not alias its own right-hand side.
   void solve(double* b, int dim, int nrhs) const {
      Eigen::Map<Mat> B(b, dim, nrhs);
      const Mat X = f_.solve(B);
      B = X;
   }

   int negative_eigenvalues() const { return n_neg_; }
   bool ok() const { return ok_; }
   // Why the last factorize()/analyze() said no — empty when it said yes.
   const std::string& why() const { return why_; }

   // ---- the pieces of the factorization the S_k formation (§5) needs ------
   //
   //  Eigen factorizes  W = P⁻¹ L D Lᵀ P  (its solve applies, in order, P,
   //  L⁻¹, D⁻¹, L⁻ᵀ, P⁻¹).  L is UNIT lower triangular and Eigen stores only
   //  its STRICTLY lower entries; D lives in a separate vector.
   const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>&
   permutationP() const { return f_.permutationP(); }
   const Vec& vectorD() const { return f_.rawD(); }

   //  The elimination tree (parent[j] = etree parent of column j, −1 at a
   //  root), fixed by the symbolic analysis.  It is what turns the sparse-RHS
   //  pruning of §5(b) from a runtime test into a STATIC structure: the
   //  nonzeros of L⁻¹b lie inside the union of etree paths from the nonzero
   //  positions of b to the root (Gilbert's reach theorem), so a right-hand
   //  side with a fixed PATTERN has a fixed reach, computable once.
   const int* etree_parent() const { return f_.rawParent().data(); }

   //  y ← L⁻¹ y, in place, for ONE right-hand side whose pattern is covered
   //  by `reach` — the sparse-RHS forward substitution of §5(b).
   //
   //  A forward substitution propagates column j of L only through the
   //  multiplier y[j]; a zero multiplier contributes nothing.  Iterating the
   //  precomputed reach instead of all n columns deletes the O(n) outer sweep
   //  a pattern-blind version needs — profiled on this code, that sweep
   //  visited ~98% empty columns and cost more than the arithmetic it
   //  guarded.  The numeric test stays as a second, finer pruning within the
   //  reach.
   //
   //  Ascending order is correct because an etree parent always has the
   //  larger index: by the time the loop reaches column j, every column that
   //  could have written into y[j] has already been processed.
   //
   //  (Eigen's own sparse-RHS overload does neither pruning: it converts the
   //  right-hand side to dense panels and runs the ordinary dense solve.)
   void forward_solve_reach(double* y, const std::vector<int>& reach) const {
      const auto& L = f_.rawL();
      for (int j : reach) {
         const double yj = y[j];
         if (yj == 0.0) continue;
         for (SpMat::InnerIterator it(L, j); it; ++it)
            if ((int)it.row() > j) y[it.row()] -= it.value() * yj;
      }
   }

private:
   // Eigen keeps the raw factor, D and the etree protected — reasonably,
   // since they are useless without knowing the exact convention above.  This
   // is the only way to reach them, and it adds no behaviour of its own.
   struct Impl : Eigen::SimplicialLDLT<SpMat, Eigen::Lower> {
      const SpMat& rawL() const { return this->m_matrix; }
      const Vec& rawD() const { return this->m_diag; }
      const Eigen::VectorXi& rawParent() const { return this->m_parent; }
   };
   Impl f_;
   bool analyzed_ = false, ok_ = false;
   int n_neg_ = 0;
   std::string why_;     // Eigen's complaint about the last refusal (§10)
};

// =============================================================================
//  Precond — the ASd interface preconditioner (§7).
//
//  "Additive Schur, assembled diagonal" (Lueg eq. 20–21, as implemented by the
//  Python reference dd_kkt.py::make_preconditioner).  For each subdomain: take
//  its LOCAL Schur block S_k, restrict it to the kept border positions that
//  subdomain touches, REPLACE ITS DIAGONAL by the diagonal of the ASSEMBLED S,
//  invert the small dense result, and apply the inverses additively:
//
//      M⁻¹ = Σ_k R_kᵀ M_k⁻¹ R_k,   M_k = R_k S_k R_kᵀ with diag ← diag(S)|R_k
//
//  The diagonal swap is the whole trick.  S_k on its own describes what
//  subdomain k alone does to the shared border unknowns, so it badly
//  underestimates their true self-coupling — every neighbouring subdomain
//  contributes to that diagonal as well.  diag(S) is the assembled truth, and
//  it is also the one quantity a distributed implementation can share cheaply
//  (a single all-reduce over a p-vector), which is exactly why Lueg picks it —
//  and why factorize() assembles diag(S) without assembling S.
//
//  EVERY M_k IS FACTORIZED BY A SYMMETRIC FACTORIZATION, NEVER INVERTED.  This
//  is the single most consequential line in the file, so here is the evidence.
//
//  CG does not merely need M⁻¹ to be positive definite; it needs it to be
//  SYMMETRIC.  The three-term recurrence derives its conjugacy from
//  <r_i, M⁻¹ r_j> being an inner product, and an M⁻¹ that is symmetric only to
//  1e-9 is not one.  The search directions stop being conjugate, the residual
//  stops descending, and CG grinds to its stall guard having achieved nothing.
//
//  An explicit M.inverse() — and PartialPivLU, which is what Eigen's inverse()
//  actually runs for a general matrix — computes the action through an
//  UNSYMMETRIC route, so round-off lands asymmetrically.  Measured on symmetric
//  indefinite blocks at cond(M) ~ 3e7:
//
//      ‖X − Xᵀ‖/‖X‖    inverse()   PartialPivLU     LDLT
//         mb =  20       5.1e-10       5.1e-10     2.3e-15
//         mb = 120       1.6e-09       1.6e-09     1.8e-13
//
//  LDLT keeps it because P L D Lᵀ Pᵀ is applied as a MIRRORED sequence, so the
//  operator is symmetric by construction rather than by luck.  Four to five
//  orders of magnitude, and the real blocks are worse conditioned than 3e7.
//
//  What that asymmetry cost, N=32/4x4, versus this arrangement:
//
//                            IPOPT its   CG its   rejected   wall
//      inverse(), all blocks       865    8.85M      63     102 s   permutation
//      LLT + LU on indefinite      940    9.49M       0     172 s
//      LLT + LDLT  (this code)     243    2.03M       0      36 s
//
//  Same solution to six figures in all three (α* 0.06448, obj 2.345, PSNR
//  23.39); 2.8x faster than the inverse, with 4.4x fewer CG iterations.  Under
//  the consensus formulation the same swap ran 74 → 59 IPOPT iterations and
//  dropped rejections 529 → 375.
//
//  Cholesky is tried FIRST because attempting it is also the SPD test — the
//  only cheap evidence this file has that §8's premise still holds — and it is
//  half the flops of LDLT when it succeeds.  A block that fails it is NOT
//  refused: measured, dropping an indefinite M_k outright cost 74 → 105 IPOPT
//  iterations, and patching the gap with a Jacobi diagonal cost 74 → 200 and
//  ended in Error_In_Step_Computation.  An indefinite M_k is wrong, but it is
//  wrong densely — it still approximates S_ff⁻¹ where it matters, whereas a
//  dropped block leaves its positions with NO action at all.  So it is recorded
//  (Warn::PrecondNotSpd) and used, and if the AGGREGATE ever goes indefinite,
//  that is what CG's rz guard is for.
//
//  A block LDLT cannot take either is genuinely singular and IS skipped — there
//  is no unsymmetric fallback, because reintroducing one would reintroduce the
//  table above.  That counter read 0 across every run measured.
//
//  NOTE, because it looks like a contradiction: build_peel_cache() refuses to
//  use Eigen's dense LDLT and reaches for a symmetric eigendecomposition
//  instead, on the grounds that Eigen's LDLT is a pivoted Cholesky rather than
//  Bunch–Kaufman and its PIVOT SIGNS are unreliable on indefinite input.  That
//  objection is real and it does not apply here.  It is an objection to reading
//  INERTIA off D — counting how many pivots are negative — which this code
//  never does.  Here LDLT is only ever asked to APPLY M⁻¹, and a preconditioner
//  is not required to be accurate; it is required to be symmetric and cheap.
//  Sign-unreliable pivots cost some preconditioner quality, which CG absorbs as
//  iterations.  An unsymmetric operator costs CG its conjugacy, which it does
//  not absorb at all.  Never route an inertia count through this factorization.
// =============================================================================
class Precond {
public:
   // nf = number of kept border positions (the dimension CG works in).
   // Sk / Nk / keptpos describe the local Schur blocks and how their rows map
   // onto kept positions; diagS_kept is diag(S) gathered on the kept positions.
   // What one build found, for §10's telemetry.  Neither number is a failure:
   // an indefinite block is still used (see the header comment), it just says
   // the SPD premise is not holding everywhere on this interface.
   struct Info {
      long blocks_refused = 0;   // local M_k that failed the Cholesky test
      long positions_indef = 0;  // DISTINCT kept positions such a block covers
                                 // (they overlap between subdomains, so this
                                 // is a set size, never a sum of block sizes)
      long blocks_singular = 0;  // ... and that LDLT could not take either
   };

   Info build(int nf, const std::vector<Mat>& Sk,
              const std::vector<std::vector<int>>& Nk,
              const std::vector<int>& keptpos, const Vec& diagS_kept) {
      nf_ = nf;
      llt_.assign(Nk.size(), Eigen::LLT<Mat>());
      absinv_.assign(Nk.size(), Mat());
      spd_.assign(Nk.size(), 0);
      idx_.assign(Nk.size(), {});
      Info info;
      std::vector<char> hit(nf, 0);     // distinct positions under a bad block
      for (size_t k = 0; k < Nk.size(); ++k) {
         std::vector<int> loc, gl;          // loc: row in S_k;  gl: kept position
         for (int a = 0; a < (int)Nk[k].size(); ++a) {
            const int kp = keptpos[Nk[k][a]];
            if (kp >= 0) { loc.push_back(a); gl.push_back(kp); }
         }
         const int mb = (int)loc.size();
         if (mb == 0) continue;
         Mat M(mb, mb);
         for (int i = 0; i < mb; ++i)
            for (int j = 0; j < mb; ++j) M(i, j) = Sk[k](loc[i], loc[j]);
         for (int i = 0; i < mb; ++i) M(i, i) = diagS_kept[gl[i]];   // the swap
         if (!M.allFinite()) continue;   // nothing to factorize; skip it
         // The SPD test and the SPD factorization are one computation.
         Eigen::LLT<Mat> f(M);
         if (f.info() == Eigen::Success) {
            llt_[k] = std::move(f);
            spd_[k] = 1;
         } else {
            // Indefinite.  Use the matrix ABSOLUTE VALUE, M_k⁻¹ → U|Λ|⁻¹Uᵀ.
            //
            // Why the absolute value and not just a stable factorization: the
            // blocks that fail Cholesky are only BARELY indefinite — measured,
            // their most negative eigenvalue is 1e-6 to 3e-2 of the largest.
            // But inversion amplifies it brutally: λ = −1e-6·λmax becomes
            // −1e6/λmax, which then dominates the additive sum, and the
            // aggregate M⁻¹ is wildly indefinite with a huge norm.  CG's
            // conjugacy needs M⁻¹ SPD, so it collapses.  Measured κ(M⁻¹S_ff)
            // on the factorizations where this fires, versus flipping to |Λ|:
            //
            //     N=32  b12  3/16 bad   1.62e16  →  2.16e3
            //     N=32  b28  2/16 bad   3.05e15  →      950
            //     N=64  b20  2/16 bad   5.20e16  →  2.55e3
            //     N=96  b16  2/16 bad   5.39e16  →  4.31e3
            //     N=96  b36  2/16 bad   4.80e16  →  1.26e3
            //
            // Thirteen orders of magnitude, landing on the same ~1e3 that the
            // all-SPD factorizations achieve.  And the frequency GROWS with the
            // problem: ~2/10 builds at N=32, 5/10 at N=96, >50% at N=256 — so
            // at the sizes that matter this was most of the run.
            //
            // Flipping the sign of a tiny eigenvalue is legitimate here in a
            // way it never would be for a solve: a preconditioner has to be SPD
            // and cheap, not correct.  Note this DOES form an explicit inverse
            // for these few blocks, but U|Λ|⁻¹Uᵀ is symmetric BY CONSTRUCTION
            // (and symmetrised below), so it does not reintroduce the asymmetry
            // that the Cholesky path exists to avoid.  Eigen's LDLT |D| is NOT
            // a substitute — flipping PIVOTS is not flipping EIGENVALUES, and
            // measured it changed nothing at N=256.
            ++info.blocks_refused;
            for (int i = 0; i < mb; ++i) hit[gl[i]] = 1;
            Eigen::SelfAdjointEigenSolver<Mat> es(M);
            if (es.info() != Eigen::Success) { ++info.blocks_singular; continue; }
            Vec d = es.eigenvalues();
            const double sc = d.cwiseAbs().maxCoeff();
            if (!(sc > 0.0)) { ++info.blocks_singular; continue; }
            static const double clip = [] {
               const char* e = std::getenv("DDS_PC_CLIP");
               return e ? std::atof(e) : 1e-6;
            }();
            for (int i = 0; i < mb; ++i)
               d[i] = 1.0 / std::max(d[i], clip * sc);   // CLIP, not |Λ|
            Mat Bi = es.eigenvectors() * d.asDiagonal()
                   * es.eigenvectors().transpose();
            if (!Bi.allFinite()) { ++info.blocks_singular; continue; }
            absinv_[k] = Mat(0.5 * (Bi + Bi.transpose()));   // exactly symmetric
            spd_[k] = 0;
         }
         idx_[k] = gl;
      }
      for (int i = 0; i < nf; ++i) info.positions_indef += hit[i];
      return info;
   }

   // z ← M⁻¹ r
   void apply(const Vec& r, Vec& z) const {
      z.setZero(nf_);
      for (size_t k = 0; k < idx_.size(); ++k) {
         const std::vector<int>& idx = idx_[k];
         const int mb = (int)idx.size();
         if (mb == 0) continue;
         Vec rk(mb);
         for (int i = 0; i < mb; ++i) rk[i] = r[idx[i]];
         const Vec zk = spd_[k] ? Vec(llt_[k].solve(rk))    // L Lᵀ zk = rk
                                : Vec(absinv_[k] * rk);     // U |Λ|⁻¹ Uᵀ
         for (int i = 0; i < mb; ++i) z[idx[i]] += zk[i];   // ADDITIVE
      }
   }

private:
   int nf_ = 0;
   std::vector<Eigen::LLT<Mat>> llt_;       // SPD blocks: Cholesky
   std::vector<Mat> absinv_;                // indefinite blocks: U |Λ|⁻¹ Uᵀ
   std::vector<char> spd_;                  // which of the two holds block k
   std::vector<std::vector<int>> idx_;      // their kept-position lists
};

// =============================================================================
//  cg_solve — preconditioned conjugate gradients (§7).
//
//  It takes applyA, a CALLABLE, not a matrix: the only thing CG ever needs of
//  an operator is its action on a vector, which is what lets the interface
//  operator stay matrix-free (2.5).  Parameter-free (the step lengths come out
//  of the Krylov space itself); the guards and the best-iterate memory are
//  described in §7.  Returns the best relative residual actually achieved;
//  the CALLER decides whether that is good enough.
//
//  What it reports back: `indefinite` records a pAp ≤ 0 event — in exact
//  arithmetic a one-directional PROOF that A is not SPD (not seeing it proves
//  nothing).  In floating point at ‖A‖ ~ 1e18 a merely tiny pAp can round
//  non-positive, so the flag OVER-REPORTS; the referee experiments recorded
//  runs where it fired repeatedly while the §8 prediction stayed correct.
//  Treat it as evidence, not as a certificate — the Stats counters do.
// =============================================================================
struct CgResult {
   double rel = 1.0;          // best relative residual reached
   long iters = 0;
   bool indefinite = false;   // pAp <= 0 was observed
};

template <class ApplyA, class Prec>
inline CgResult cg_solve(ApplyA&& applyA, const Prec& P, const Vec& b, Vec& x,
                         double tol, int maxit, const Vec* x0 = nullptr) {
   const int n = (int)b.size();
   const double bnorm = std::max(b.norm(), 1e-300);
   Vec r(n), z(n);
   // Optional warm start: CG converges from any x₀, so a caller solving a
   // slowly-varying sequence of systems (the peel columns of §6, across
   // Newton steps and δ-retries) can seed with its previous answer.  Costs
   // one extra operator application; a useless x₀ merely reproduces the cold
   // start via the best-iterate memory below.
   if (x0 != nullptr && x0->size() == n && x0->allFinite()) {
      x = *x0;
      applyA(x, r);
      r = b - r;
   } else {
      x.setZero(n);
      r = b;
   }
   P.apply(r, z);
   Vec p = z, Ap(n);
   double rz = r.dot(z);
   CgResult res;
   double best_rel = 1.0;
   Vec best = Vec::Zero(n);
   int best_it = 0, it = 0;
   // Give up after this many iterations with NO improvement in the best
   // residual seen.  It is not a convergence tolerance and it must not be set
   // near sqrt(kappa): CG's residual 2-norm is not monotone, and on an
   // operator this preconditioner leaves at kappa ~ 2e3 the first genuine
   // descent does not appear until roughly sqrt(2e3) ~ 45 iterations.  A guard
   // at 40 therefore fired EXACTLY where CG was about to start converging, and
   // handed back the zero iterate.  Measured (N=32 4x4 consensus, dense
   // eigendecomposition of the assembled S_ff): cond(S_ff) 1e7-1e10,
   // cond(M^-1 S_ff) 1.9e3-2.8e3, so sqrt ~ 43-52 against a guard of 40.
   //
   // Raising it, same alpha*/PSNR throughout (40 -> 100 -> 300):
   //     consensus N=32   rejected 316 -> 24 -> 24,  solves 837 -> 607 -> 607
   //     consensus N=64   rejected 2137 -> 934 -> 87, solves 2480 -> 1325 -> 511
   //                      status -3 -> 1 -> 1,        its 149 -> 104 -> 82
   //     permutation N=32 0 rejected at every setting (it never hits the guard)
   // N=32 saturates by 100 (100/200/400 bit-identical), but N=64 does NOT —
   // it keeps improving to 300, which is the point: the guard has to clear
   // sqrt(kappa), and kappa grows with the problem.  Hence 300 rather than the
   // N=32 plateau.  cg_maxit stays the real ceiling.
   const int stall = 300;
   for (; it < maxit; ++it) {
      const double rel = r.norm() / bnorm;
      if (rel < best_rel) { best_rel = rel; best = x; best_it = it; }
      if (rel <= tol) break;
      if (it - best_it > stall) break;
      if (!(rz > 0.0)) break;                   // preconditioner not SPD
      applyA(p, Ap);                            // the ONLY thing CG needs of A
      const double pAp = p.dot(Ap);
      if (!(pAp > 0.0)) { res.indefinite = true; break; }   // A not SPD here
      const double alpha = rz / pAp;
      x += alpha * p;
      r -= alpha * Ap;
      P.apply(r, z);
      const double rz_new = r.dot(z);
      p = z + (rz_new / rz) * p;
      rz = rz_new;
   }
   x = best;
   res.rel = best_rel;
   res.iters = it;
   return res;
}

// =============================================================================
//  Arrowhead — the solver itself (§2–§9).  Pure Eigen: this class knows
//  nothing about IPOPT, so it can be unit-tested standalone (that is what
//  dd_simple_smoke.cpp does).  The IPOPT adapter is at the bottom of the file.
//
//  Life cycle:
//      set_structure(...)   once per sparsity pattern  (partition + symbolic)
//      values()             refresh the matrix entries  (every Newton step)
//      factorize()          W_k, S_k, peel cache, predicted inertia
//      solve(rhs)           the refined arrowhead solve, in place
// =============================================================================
class Arrowhead {
public:
   struct Options {
      // KKT indices >= n_primal are DUAL unknowns.  Border positions with such
      // an index are peeled (§6 (i)).  Leave it huge to disable the dual peel.
      int n_primal = 1 << 30;
      // KKT index of the scalar α, or −1 if the formulation has none (§6 (ii)).
      int alpha_index = -1;
      // Make the CROSS POINTS primal too — border positions touched by >= 3
      // subdomains (§6 (iii)).  On by default: it is the FETI-DP corner rule,
      // and measured it is what makes tile partitions scale.  Set false to
      // A/B against the α + dual peel alone.
      bool peel_cross_points = true;
      // Two DIFFERENT jobs, so two tolerances.  cg_tol is the target for the
      // INTERFACE SOLVE (§9).  The peel-cache columns of §6 do not need it:
      // asking all 246 of them for 1e-10 is where this solver spends its time
      // — measured at N=32/nsub=8/30 Newton steps, 2.04M CG iterations and
      // 92% of total runtime inside build_peel_cache, against 0.2% in the W_k
      // factorizations the decomposition exists for.
      //
      // But peel_cg_tol canNOT be relaxed to the 1e-2 acceptance bar of
      // build_peel_cache: Z = S_ff⁻¹ S_fP is not scratch for the §8 prediction,
      // it is part of the SOLVE operator, so per-column error lands in the
      // true residual that solve_interface judges at 1e-2 — and it lands there
      // summed over all |P| columns.  Measured (N=32, 30 Newton steps, wall /
      // CG answers rejected, baseline 1e-10 = 74.7s / 0):
      //     1e-8  63.5s /   0      1e-6  49.7s /  57
      //     1e-7  57.1s /   0      1e-5  42.6s / 251
      //     1e-3  52.6s / 967 (and 247 step residuals over kStepResidualWarn)
      // 1e-7 is the knee: 1.31x off the baseline wall with rejected still 0,
      // and identical alpha*, PSNR, refused predictions and warning counts.
      // Past it the interface solve starts handing back steps it knows are
      // bad, which is a worse trade than the time it buys.
      double cg_tol = 1e-10;      // interface solve (§9)
      double peel_cg_tol = 1e-7;  // peel-cache columns (§6); see above
      int cg_maxit = 500;
   };

   // Above this, a refined step is reported as poor (§9/§10).  Not a rejection
   // threshold — the step is still returned, because IPOPT has nothing better
   // to do with a SINGULAR here than raise δ_w and ask for the same thing again.
   //
   // That is not a guess, it is the architecture, and withholding the step was
   // tried: SYMSOLVER_SINGULAR is legal only at FACTORIZATION time.  IPOPT's
   // δ_w loop guards the first SolveOnce for a new matrix, but once the
   // factorization is accepted IPOPT runs its own iterative refinement against
   // it, and a solve that fails there aborts the algorithm outright —
   //     INTERNAL_ABORT, IpPDFullSpaceSolver.cpp:263,
   //     "SolveOnce returns false during iterative refinement"
   // — measured at consensus N=32 for every threshold from 1e-2 to 1e1, where
   // withholding a SINGLE step was enough to end the run (-199, or -2 without
   // leaving the seeded level).  Refusing a bad step has to happen in
   // factorize(), not here.
   static constexpr double kStepResidualWarn = 1e-8;

   struct Stats {
      long solves = 0;        // interface solves attempted
      long iters = 0;         // CG iterations summed over all of them
      long rejected = 0;      // CG answers that failed the acceptance test
                              // (still handed back — there is no fallback —
                              // and judged again by solve()'s refinement, §9)
      long cache_builds = 0;  // peel caches built (one per factorization)
      // §8 falsification telemetry.  The two counters are NOT the same thing:
      //   before  CG saw non-positive curvature while BUILDING the prediction,
      //           so no prediction was issued — the safe outcome (SINGULAR).
      //   after   it happened during a later solve, i.e. an inertia already
      //           handed to IPOPT rests on a premise this run has evidence
      //           against.  The flag over-reports (see CgResult), so treat it
      //           as a hint, not a verdict.
      long indef_before = 0, indef_after = 0;
      long pred_refused = 0;  // factorizations where we declined to predict
      // ASd preconditioner health (§7).  An indefinite block is USED anyway,
      // via LDLT (dropping it measured worse — see the Precond header), so the
      // first two are telemetry on the §8 SPD premise, not a failure count.
      // pc_blocks_singular IS a failure: that block was skipped entirely.
      long pc_blocks_indef = 0, pc_positions_indef = 0, pc_blocks_singular = 0;
   };

   // --------------------------------------------------------------------
   //  STRUCTURE.  irow/jcol are 0-based lower-triangle coordinates, owner is
   //  one label per KKT index (subdomain id, or −1 for border).  Everything
   //  that depends only on the pattern is computed here, once: the numbering,
   //  the triplet routing, the symbolic analyses, the reaches, the peel sets.
   // --------------------------------------------------------------------
   bool set_structure(int dim, std::vector<int> irow, std::vector<int> jcol,
                      std::vector<int> owner, int nsub, const Options& opt) {
      dim_ = dim;
      irow_ = std::move(irow);
      jcol_ = std::move(jcol);
      owner_ = std::move(owner);
      nsub_ = nsub;
      opt_ = opt;
      nnz_ = (int)irow_.size();
      vals_.assign(nnz_, 0.0);
      if ((int)owner_.size() != dim_) {
         std::cerr << "[dds] owner map has " << owner_.size()
                   << " entries, KKT dim is " << dim_ << "\n";
         return false;
      }
      number_unknowns();
      if (!route_triplets()) return false;
      build_peel_sets();
      if (std::getenv("DDS_DEBUG"))
         std::cerr << "[dds] dim=" << dim_ << " nnz=" << nnz_
                   << " subdomains=" << nsub_ << " border p=" << p_
                   << " peeled=" << peel_.size() << " (" << n_peel_dual_
                   << " dual, " << n_peel_cross_ << " cross)  max dim W_k="
                   << max_dimk_ << "\n";
      return true;
   }

   // The nnz-long value array IPOPT writes into before every factorization.
   double* values() { return vals_.data(); }
   int dim() const { return dim_; }

   enum Status { OK, SINGULAR };

   // --------------------------------------------------------------------
   //  FACTORIZATION.  Four steps, in order:
   //     1. factorize every W_k                    (§4, independent per k)
   //     2. form the local Schur blocks S_k        (§5, independent per k)
   //     3. assemble C and diag(S)                 (S itself is never built)
   //     4. build the peel cache, predict In(S)    (§6 + §8)
   // --------------------------------------------------------------------
   Status factorize() {
      n_neg_ = 0;
      peel_valid_ = false;   // the operator changed ⇒ the peel cache is void

      // ---- 1. the subdomain blocks (§4) --------------------------------
      // Independent per k, and Eigen keeps no global state, so this loop is
      // safe to run in parallel (unlike MA97, whose concurrent factorization
      // corrupts its own heap — see dd_solver.hpp).
      std::vector<char> ok(nsub_, 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         fill_from_triplets(W_[k], wtrip_[k]);
         fill_from_triplets(B_[k], btrip_[k]);
         ok[k] = ldlt_[k]->factorize(W_[k]) ? 1 : 0;
      }
      for (int k = 0; k < nsub_; ++k) {
         if (!ok[k]) {
            // Expected on occasion and self-correcting (§4) — but silence here
            // is how a run that regularized its way out of trouble on every
            // second Newton step passes for a clean one.
            std::ostringstream m;
            m << "W_" << k << " (dim " << dimk_[k] << "): " << ldlt_[k]->why()
              << "; reporting SINGULAR, IPOPT answers by raising δ_w";
            warn(Warn::LdltNumeric, m.str());
            return SINGULAR;
         }
         n_neg_ += ldlt_[k]->negative_eigenvalues();   // the Σ_k In(W_k) of (3.1)
      }

      // ---- 2. the local Schur blocks (§5) ------------------------------
      // S_k = −B_k W_k⁻¹ B_kᵀ via (5.1)/(5.2): pruned forward solves on the
      // static reaches, then the sparse row contraction.  Exact throughout —
      // every skipped term is exactly zero.  DDS_SCHUR_CHECK=1 referees each
      // block against the naive two-sided route.
      static const bool schur_check = std::getenv("DDS_SCHUR_CHECK") != nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk == 0) { Sk_[k].resize(0, 0); continue; }
         const int nk = dimk_[k];
         // P B_kᵀ, still sparse.  Eigen's own permutation product, so there is
         // no way to get P and P⁻¹ the wrong way round.
         const SpMat PBt = ldlt_[k]->permutationP() * SpMat(B_[k].transpose());
         const std::vector<std::vector<int>>& RK = reach_[k];

         // Y = L⁻¹ P B_kᵀ, held COMPRESSED on the static reach: column a is
         // stored as the |reach(a)| values aligned with reach_[k][a].  One
         // dense scatter workspace serves every column; the restore step below
         // keeps it zero outside the current column's reach, so nothing of
         // size n_k is ever swept or memset (§5(b)).
         std::vector<std::vector<double>> Yc(pk);
         Vec w = Vec::Zero(nk);
         for (int a = 0; a < pk; ++a) {
            const std::vector<int>& R = RK[a];
            for (SpMat::InnerIterator it(PBt, a); it; ++it) w[it.row()] = it.value();
            ldlt_[k]->forward_solve_reach(w.data(), R);
            std::vector<double>& v = Yc[a];
            v.resize(R.size());
            for (size_t i = 0; i < R.size(); ++i) {
               v[i] = w[R[i]];
               w[R[i]] = 0.0;                          // restore the invariant
            }
         }

         // The row contraction (5.2), driven by the same reaches: group Y's
         // nonzeros by row (counting sort — columns are appended in ascending
         // order, which the upper-triangle accumulation needs), then one small
         // outer product per nonempty row.  Exactly symmetric by construction
         // (upper triangle accumulated, then mirrored), which In(T) and the
         // preconditioner both rely on.
         {
            std::vector<int> rowptr(nk + 1, 0);
            for (int a = 0; a < pk; ++a)
               for (size_t i = 0; i < RK[a].size(); ++i)
                  if (Yc[a][i] != 0.0) ++rowptr[RK[a][i] + 1];
            for (int r = 0; r < nk; ++r) rowptr[r + 1] += rowptr[r];
            std::vector<int> rcol(rowptr[nk]);
            std::vector<double> rval(rowptr[nk]);
            std::vector<int> fill(rowptr.begin(), rowptr.end() - 1);
            for (int a = 0; a < pk; ++a)
               for (size_t i = 0; i < RK[a].size(); ++i)
                  if (Yc[a][i] != 0.0) {
                     const int r = RK[a][i];
                     rcol[fill[r]] = a;
                     rval[fill[r]] = Yc[a][i];
                     ++fill[r];
                  }
            const Vec& d = ldlt_[k]->vectorD();
            Mat& S = Sk_[k];
            S.setZero(pk, pk);
            for (int r = 0; r < nk; ++r) {
               const int b0 = rowptr[r], b1 = rowptr[r + 1];
               if (b0 == b1) continue;
               const double dinv = 1.0 / d[r];
               for (int ib = b0; ib < b1; ++ib) {
                  const int b = rcol[ib];
                  const double wv = dinv * rval[ib];
                  for (int ia = b0; ia <= ib; ++ia)
                     S(rcol[ia], b) -= rval[ia] * wv;
               }
            }
            for (int j = 0; j < pk; ++j)
               for (int i = 0; i < j; ++i) S(j, i) = S(i, j);
         }
      }
      if (schur_check) {                      // serial: it prints
         for (int k = 0; k < nsub_; ++k) {
            const int pk = (int)Nk_[k].size();
            if (pk == 0) continue;
            Mat Ref = Mat(B_[k].transpose());
            ldlt_[k]->solve(Ref.data(), dimk_[k], pk);   // the naive two-sided route
            Ref = -(B_[k] * Ref);
            const double den = Ref.cwiseAbs().maxCoeff();
            const double num = (Sk_[k] - Ref).cwiseAbs().maxCoeff();
            std::cerr << "[dds-schur] k=" << k << " p_k=" << pk
                      << " |S_k|max=" << den
                      << " rel-err=" << (den > 0 ? num / den : num) << "\n";
         }
      }

      // ---- 3. the corner block C and diag(S) ---------------------------
      // C is small (only the border–border KKT couplings) and the matrix-free
      // application (2.5) needs it.  Fully symmetric, since it is applied to
      // vectors rather than factorized.
      std::vector<Trip> t;
      t.reserve(ctrip_.size() * 2);
      for (const auto& e : ctrip_) {
         const double v = vals_[e.t];
         t.emplace_back(e.r, e.c, v);
         if (e.r != e.c) t.emplace_back(e.c, e.r, v);
      }
      C_.resize(p_, p_);
      C_.setFromTriplets(t.begin(), t.end());     // duplicates are summed
      C_.makeCompressed();

      // diag(S), assembled WITHOUT assembling S: the corner diagonal plus each
      // subdomain's own contribution to the border unknowns it touches.  In a
      // distributed code this is one all-reduce over a p-vector — exactly why
      // the ASd preconditioner is built around it (see Precond).
      diagS_.setZero(p_);
      for (const auto& e : ctrip_)
         if (e.r == e.c) diagS_[e.r] += vals_[e.t];
      for (int k = 0; k < nsub_; ++k)
         for (int a = 0; a < (int)Nk_[k].size(); ++a) diagS_[Nk_[k][a]] += Sk_[k](a, a);

      // ---- 4. peel cache + predicted In(S) (§6, §8) --------------------
      // Building the cache HERE rather than lazily at the first solve is what
      // makes the prediction available in time to answer IPOPT.
      if (!build_peel_cache()) {
         // Refusing is the point (§8): a prediction we cannot stand behind
         // would silently corrupt IPOPT's δ_w loop, whereas SINGULAR merely
         // costs a re-factorization at a larger δ_w.
         ++stats_.pred_refused;
         if (std::getenv("DDS_DEBUG"))
            std::cerr << "[dds] peel cache failed, so In(S) cannot be predicted "
                         "→ SINGULAR\n";
         return SINGULAR;
      }
      n_neg_ += t_neg_;                        // Haynsworth twice: (3.1) + (8.1)
      return OK;
   }

   int negative_eigenvalues() const { return n_neg_; }
   const Stats& stats() const { return stats_; }
   void reset_stats() { stats_ = Stats(); }

   // --------------------------------------------------------------------
   //  SOLVE, in place: the arrowhead recursion wrapped in iterative
   //  refinement against the ORIGINAL triplets (§9).  rhs is overwritten by
   //  the best refined step seen.  Returns false only if the residual never
   //  once evaluated finite — then rhs holds nothing usable and the caller
   //  must report SINGULAR.
   // --------------------------------------------------------------------
   bool solve(double* rhs) {
      const std::vector<double> b0(rhs, rhs + dim_);
      std::vector<double> x(b0), r(dim_), Ax(dim_), best;
      // The very first interface solve can genuinely fail (there is no direct
      // fallback); then there is no step to refine and rhs is left alone.
      if (!solve_arrowhead(x.data())) return false;

      double bnorm = 0.0;
      for (int i = 0; i < dim_; ++i) bnorm += b0[i] * b0[i];
      bnorm = std::sqrt(std::max(bnorm, 1e-300));

      double best_res = std::numeric_limits<double>::infinity();
      for (int sweep = 0; sweep < 3; ++sweep) {
         matvec(x.data(), Ax.data());
         double rn = 0.0;
         for (int i = 0; i < dim_; ++i) {
            r[i] = b0[i] - Ax[i];
            rn += r[i] * r[i];
         }
         const double rel = std::sqrt(rn) / bnorm;
         // Seed on the first sweep so `best` can never stay unset (NaN < inf
         // is false, which would otherwise hand back an uninitialised vector).
         if (sweep == 0 || rel < best_res) { best_res = rel; best = x; }
         if (!std::isfinite(rel)) break;
         if (rel <= 1e-11) break;
         if (!solve_arrowhead(r.data())) break;  // r ← A⁻¹r, the correction
         for (int i = 0; i < dim_; ++i) x[i] += r[i];
      }
      if (!std::isfinite(best_res)) {              // rhs left as the RHS
         warn(Warn::StepResidual,
              "refinement residual is not finite; no step returned (SINGULAR)");
         return false;
      }
      // The last honest check in the file: this is the residual of the step
      // actually handed to IPOPT, measured against the ORIGINAL triplets (§9).
      // Refinement targets 1e-11 and normally lands far below the threshold
      // below, so a warning here means the decomposition — not the arithmetic
      // around it — is what IPOPT is being asked to trust.
      if (best_res > kStepResidualWarn) {
         std::ostringstream m;
         m << "step returned with ‖b − Ax‖/‖b‖ = " << best_res << " after 3 "
              "refinement sweeps (warn above " << kStepResidualWarn << ")";
         warn(Warn::StepResidual, m.str());
      }
      std::copy(best.begin(), best.end(), rhs);
      return true;
   }

private:
   // =====================================================================
   //  Structure: numbering, routing, reaches, peel sets  (pattern only)
   // =====================================================================

   // Give every unknown its coordinates in the permuted picture:
   //   border index  i  →  ypos_[i] ∈ [0, p)      its position on the border
   //   interior      i  →  lpos_[i] ∈ [0, dim_k)  its row/column of W_k
   // Both numberings follow ASCENDING KKT index, which matters: it makes the
   // local index order agree with the global one, so a lower-triangle triplet
   // stays lower-triangle after the permutation and we never have to sort or
   // mirror anything.
   void number_unknowns() {
      ypos_.assign(dim_, -1);
      lpos_.assign(dim_, -1);
      dimk_.assign(nsub_, 0);
      p_ = 0;
      for (int i = 0; i < dim_; ++i) {
         if (owner_[i] < 0) ypos_[i] = p_++;
         else lpos_[i] = dimk_[owner_[i]]++;
      }
      max_dimk_ = 0;
      for (int d : dimk_) max_dimk_ = std::max(max_dimk_, d);
   }

   // Sort every input triplet into the arrowhead block it belongs to, discover
   // which border unknowns each subdomain sees (the index list N_k — Lueg's
   // selection matrix), build the sparsity patterns of W_k and B_k, run the
   // symbolic analysis of every W_k, and precompute the §5(b) reaches.  All of
   // this depends on the PATTERN only.
   bool route_triplets() {
      // -- pass 1: which border positions does each subdomain touch? --
      std::vector<std::vector<char>> seen(nsub_, std::vector<char>(p_, 0));
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const int oi = owner_[i], oj = owner_[j];
         if (oi >= 0 && oj >= 0 && oi != oj) {
            // The owner map is supposed to be a partition with no direct
            // subdomain-to-subdomain coupling (§2); if it is not, the
            // arrowhead shape does not hold and nothing below is valid.
            std::cerr << "[dds] partition leak: entry (" << i << "," << j
                      << ") couples subdomains " << oi << " and " << oj << "\n";
            return false;
         }
         if (oi < 0 && oj >= 0) seen[oj][ypos_[i]] = 1;
         if (oj < 0 && oi >= 0) seen[oi][ypos_[j]] = 1;
      }
      Nk_.assign(nsub_, {});
      ykrow_.assign(nsub_, std::vector<int>(p_, -1));
      for (int k = 0; k < nsub_; ++k)
         for (int j = 0; j < p_; ++j)
            if (seen[k][j]) {
               ykrow_[k][j] = (int)Nk_[k].size();   // row of B_k for this border unknown
               Nk_[k].push_back(j);
            }

      // -- pass 2: route each triplet to W_k, B_k or C --
      wtrip_.assign(nsub_, {});
      btrip_.assign(nsub_, {});
      ctrip_.clear();
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const int oi = owner_[i], oj = owner_[j];
         if (oi >= 0 && oj >= 0) {                       // interior–interior → W_k
            wtrip_[oi].push_back({lpos_[i], lpos_[j], t});
         } else if (oi < 0 && oj < 0) {                  // border–border → C
            ctrip_.push_back({ypos_[i], ypos_[j], t});
         } else if (oi < 0) {                            // border row, interior col
            btrip_[oj].push_back({ykrow_[oj][ypos_[i]], lpos_[j], t});
         } else {                                        // interior row, border col
            btrip_[oi].push_back({ykrow_[oi][ypos_[j]], lpos_[i], t});
         }
      }

      // -- allocate the blocks, analyze the W_k, precompute the reaches --
      W_.assign(nsub_, SpMat());
      B_.assign(nsub_, SpMat());
      Sk_.assign(nsub_, Mat());
      // held by pointer: Eigen's sparse solvers are neither copyable nor
      // movable, so they cannot live in a vector directly
      ldlt_.clear();
      for (int k = 0; k < nsub_; ++k) ldlt_.emplace_back(new Ldlt());
      reach_.assign(nsub_, {});
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         W_[k].resize(dimk_[k], dimk_[k]);           // lower triangle only
         B_[k].resize(pk, dimk_[k]);
         fill_from_triplets(W_[k], wtrip_[k]);       // zeros: pattern only
         fill_from_triplets(B_[k], btrip_[k]);       // zeros: pattern for the reach
         if (!ldlt_[k]->analyze(W_[k])) {
            std::ostringstream m;
            m << "W_" << k << " (dim " << dimk_[k] << "): " << ldlt_[k]->why();
            warn(Warn::LdltSymbolic, m.str());
            return false;
         }
         // Symbolic reach of every column of P_k B_kᵀ (§5(b)): walk each
         // nonzero row index up the elimination tree until the root or an
         // already-marked column.  Ascending sort makes the list a valid
         // substitution order (an etree parent always has the larger index).
         const SpMat PBt = ldlt_[k]->permutationP() * SpMat(B_[k].transpose());
         const int* parent = ldlt_[k]->etree_parent();
         std::vector<int> mark(dimk_[k], -1);
         reach_[k].resize(pk);
         for (int a = 0; a < pk; ++a) {
            std::vector<int>& R = reach_[k][a];
            for (SpMat::InnerIterator it(PBt, a); it; ++it)
               for (int j = (int)it.row(); j >= 0 && mark[j] != a; j = parent[j]) {
                  mark[j] = a;
                  R.push_back(j);
               }
            std::sort(R.begin(), R.end());
         }
      }
      return true;
   }

   // Decide which border positions are PEELED (§6): every dual border index,
   // plus α.  The rest are KEPT — those are the ones CG iterates on.
   void build_peel_sets() {
      peel_.clear();
      n_peel_dual_ = 0;
      for (int i = 0; i < dim_; ++i)
         if (owner_[i] < 0 && i >= opt_.n_primal) {
            peel_.push_back(ypos_[i]);
            ++n_peel_dual_;
         }
      const int a = opt_.alpha_index;
      if (a >= 0 && a < dim_ && owner_[a] < 0) peel_.push_back(ypos_[a]);
      std::sort(peel_.begin(), peel_.end());
      peel_.erase(std::unique(peel_.begin(), peel_.end()), peel_.end());

      // ---- the cross points (§6 (iii)) --------------------------------
      //
      // Also make PRIMAL every border position touched by >= 3 subdomains.
      // That is the FETI-DP definition of a corner (Farhat et al. 2001, "D1:
      // its cross points — the points belonging to more than two subdomains"),
      // and in 2D corner constraints ALONE are proved to give a condition
      // number C(1+log(H/h))^2 independent of the subdomain count
      // (Mandel & Tezaur 2001).  Measured here: exactly (k-1)^2 such positions
      // on a k x k tile partition, each of degree 3 — 4 at k=3, 9 at k=4 — so
      // the peel grows by very little.  Strips have none, so this is a no-op
      // there (verified: bit-identical runs).
      //
      // Measured effect (cameraman, --hessian exact), off -> on:
      //   N=32 3x3   195 it, 6.31 s  ->  113 it, 3.24 s
      //   N=32 4x4  2251 it, 160  s  ->  509 it, 36.8 s
      //   N=64 4x4   did not converge (barrier stall)  ->  782 it, 173 s
      // The interpretation is the one Lueg et al. (2025) give for their own
      // ASd: a ONE-LEVEL preconditioner does not scale in the partition count,
      // and the fix is a coarse correction.  Peeling the cross points is the
      // cheapest such correction — it puts them in T, which is solved exactly.
      n_peel_cross_ = 0;
      if (opt_.peel_cross_points && p_ > 0) {
         std::vector<char> already(p_, 0);
         for (int j : peel_) already[j] = 1;
         std::vector<int> deg(p_, 0);
         for (int k = 0; k < nsub_; ++k)
            for (int j : Nk_[k]) ++deg[j];
         std::vector<int> cross;
         for (int j = 0; j < p_; ++j)
            if (!already[j] && deg[j] >= 3) cross.push_back(j);
         // SAFETY VALVE.  The whole bet is that cross points are FEW — on a
         // k x k tile partition there are (k-1)^2 of them against an interface
         // of size O(kN), so |P| stays a small fraction of p and T stays cheap
         // (it is dense, LU-factorized, and costs |P| CG solves to build).  A
         // partition where that is false has no small coarse space of this
         // kind, and peeling anyway would just move the whole interface into a
         // dense direct solve.  Refuse, and keep the α + dual peel.
         const size_t limit = (size_t)std::max(1, p_ / 4);
         if (!cross.empty() && peel_.size() + cross.size() <= limit) {
            peel_.insert(peel_.end(), cross.begin(), cross.end());
            n_peel_cross_ = (int)cross.size();
            std::sort(peel_.begin(), peel_.end());
         } else if (!cross.empty() && std::getenv("DDS_DEBUG")) {
            std::cerr << "[dds] cross-point peel declined: " << cross.size()
                      << " cross points would put |P| at "
                      << (peel_.size() + cross.size()) << " of p=" << p_
                      << " (cap " << limit << ")\n";
         }
      }
      peelpos_.assign(p_, -1);
      keptpos_.assign(p_, -1);
      kept_.clear();
      for (size_t j = 0; j < peel_.size(); ++j) peelpos_[peel_[j]] = (int)j;
      for (int j = 0; j < p_; ++j)
         if (peelpos_[j] < 0) { keptpos_[j] = (int)kept_.size(); kept_.push_back(j); }

      // Edge groups for the DDS_COARSE experiment: kept positions bucketed by
      // the set of tiles touching them (pattern-only, computed once).
      coarse_grp_.assign(kept_.size(), -1);
      n_coarse_ = 0;
      if (const char* cs = std::getenv("DDS_COARSE")) {
         const int nseg = std::max(1, std::atoi(cs));   // segments per edge
         std::vector<std::vector<int>> touch(p_);
         for (int k = 0; k < nsub_; ++k)
            for (int j : Nk_[k]) touch[j].push_back(k);
         // bucket kept positions by touching-tile set = one bucket per edge,
         // then split each bucket into nseg contiguous segments (border
         // positions ascend with KKT index, which tracks the grid, so
         // contiguous-in-bucket is contiguous-along-the-edge)
         std::map<std::vector<int>, std::vector<int>> edges;
         for (size_t a = 0; a < kept_.size(); ++a) {
            std::vector<int>& t = touch[kept_[a]];
            std::sort(t.begin(), t.end());
            edges[t].push_back((int)a);
         }
         for (auto& kv : edges) {
            const std::vector<int>& mem = kv.second;
            const int m = std::min<int>(nseg, (int)mem.size());
            for (size_t i = 0; i < mem.size(); ++i)
               coarse_grp_[mem[i]] = n_coarse_ + (int)(i * m / mem.size());
            n_coarse_ += m;
         }
         if (std::getenv("DDS_DEBUG"))
            std::cerr << "[dds] coarse space: " << n_coarse_ << " groups ("
                      << edges.size() << " edges x " << nseg << " segments)\n";
      }
   }

   // A triplet routed into one of the blocks: (row, col) in that block's own
   // numbering, plus the index of the input entry that supplies its value.
   struct Entry { int r, c, t; };

   // Rebuild a block from its routed triplets.  Duplicate (r,c) pairs are
   // summed by setFromTriplets, exactly as IPOPT's triplet contract requires.
   // Rebuilding rather than refreshing values through a frozen slot map costs
   // a sort per block per Newton step — the price of not having 200 lines of
   // slot bookkeeping here (dd_solver.hpp pays those 200 lines instead).
   void fill_from_triplets(SpMat& M, const std::vector<Entry>& es) {
      std::vector<Trip> t;
      t.reserve(es.size());
      for (const auto& e : es) t.emplace_back(e.r, e.c, vals_[e.t]);
      M.setFromTriplets(t.begin(), t.end());
      M.makeCompressed();
   }

   // =====================================================================
   //  The solve  (§2 recursion, §6 peel, §7 CG, §9 refinement)
   // =====================================================================

   // y = A·x straight from the input triplets (lower triangle + its mirror).
   // This is the ONLY place the true matrix is applied, and it is what the
   // iterative refinement measures its residual against — so refinement checks
   // the decomposition, not just the arithmetic inside it (§9).
   void matvec(const double* x, double* y) const {
      std::fill(y, y + dim_, 0.0);
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         y[i] += vals_[t] * x[j];
         if (i != j) y[j] += vals_[t] * x[i];
      }
   }

   // One arrowhead solve — equations (2.2)–(2.4), in place.  Returns false
   // only when the interface solve produced nothing usable at all.
   bool solve_arrowhead(double* rhs) {
      // -- split the RHS into interior pieces r_k and the border piece r_y --
      std::vector<Vec> rk(nsub_), wk(nsub_);
      for (int k = 0; k < nsub_; ++k) rk[k].resize(dimk_[k]);
      Vec ry(p_);
      for (int i = 0; i < dim_; ++i) {
         if (owner_[i] >= 0) rk[owner_[i]][lpos_[i]] = rhs[i];
         else ry[ypos_[i]] = rhs[i];
      }

      // -- (2.2)  r_S = r_y − Σ_k B_k W_k⁻¹ r_k --
      // The solves are independent; the accumulation into ry is not (adjacent
      // subdomains share border positions), so it is done serially afterwards.
      std::vector<Vec> contrib(nsub_);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         wk[k] = rk[k];
         ldlt_[k]->solve(wk[k].data(), dimk_[k], 1);
         if (!Nk_[k].empty()) contrib[k].noalias() = B_[k] * wk[k];
      }
      for (int k = 0; k < nsub_; ++k)
         for (size_t a = 0; a < Nk_[k].size(); ++a) ry[Nk_[k][a]] -= contrib[k][(int)a];

      // -- (2.3)  Δy = S⁻¹ r_S --
      Vec dy;
      if (!solve_interface(ry, dy)) return false;

      // -- (2.4)  Δx_k = W_k⁻¹ (r_k − B_kᵀ Δy) --
      // Fully independent per k: dy is read-only and every write lands in
      // block-k storage.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk) {
            Vec dyk(pk);
            for (int a = 0; a < pk; ++a) dyk[a] = dy[Nk_[k][a]];
            rk[k].noalias() -= B_[k].transpose() * dyk;
         }
         wk[k] = rk[k];
         ldlt_[k]->solve(wk[k].data(), dimk_[k], 1);
      }

      // -- gather the pieces back into the KKT ordering --
      for (int i = 0; i < dim_; ++i)
         rhs[i] = (owner_[i] >= 0) ? wk[owner_[i]][lpos_[i]] : dy[ypos_[i]];
      return true;
   }

   // The interface operator, matrix-free — equation (2.5).  Every term is
   // LOCAL to one subdomain; the only communication a distributed run needs
   // per application is one reduction over the border vector.
   //
   // This is the iterative strategy of Lueg et al. §3.1 with ONE deliberate
   // change.  Their eq. (12) applies S·u by one W_k BACK-SOLVE per partition
   // per CG iteration, so that not even the local Schur blocks are formed.
   // Here the S_k are formed once per factorization (§5) and each application
   // is a p_k × p_k GEMV — measured ~10× cheaper per iteration, and at the
   // observed 10²–10³ CG iterations per factorization the formation cost is
   // repaid ~20× over (their preconditioners of eqs. 17/20–21 need the local
   // blocks anyway, and §5's reach route makes forming them far cheaper than
   // the p_k full back-solves their cost model assumes).  Same operator, same
   // one-reduction communication pattern; only the local cost model differs.
   // Their per-iteration-back-solve regime is what dd_solver.hpp's
   // APPLY_MATFREE measures.
   void apply_S(const Vec& y, Vec& out) const {
      out.noalias() = C_ * y;
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk == 0) continue;
         Vec yk(pk);
         for (int a = 0; a < pk; ++a) yk[a] = y[Nk_[k][a]];
         const Vec c = Sk_[k] * yk;
         for (int a = 0; a < pk; ++a) out[Nk_[k][a]] += c[a];
      }
   }

   // S_ff v — the operator CG actually iterates on (§6).  Embedding v with
   // zeros on the peeled positions, applying the full S and reading back the
   // kept ones IS S_ff v; there is no need to hold the sub-block separately.
   void apply_Sff_into(const Vec& v, Vec& out) const {
      if (peel_.empty()) { apply_S(v, out); return; }
      Vec full = Vec::Zero(p_);
      for (size_t a = 0; a < kept_.size(); ++a) full[kept_[a]] = v[(int)a];
      Vec Sf;
      apply_S(full, Sf);
      out.resize((int)kept_.size());
      for (size_t a = 0; a < kept_.size(); ++a) out[(int)a] = Sf[kept_[a]];
   }

   // The interface solve (2.3): CG on the peeled S_ff, and CG is all there is.
   // A rejected answer has nowhere to fall back to, so its best iterate is
   // taken anyway and §9's refinement judges it against the true triplets;
   // only a solve that produced no iterate at all is reported as failed.
   bool solve_interface(const Vec& ry, Vec& dy) {
      if (interface_cg(ry, dy)) return true;
      if (dy.size() == p_ && dy.allFinite()) return true;
      warn(Warn::CgUnusable,
           "not even a best iterate survived; this arrowhead solve fails and "
           "IPOPT is told SINGULAR");
      return false;
   }

   // Build, once per factorization, everything §6 and §8 need:
   //   S_fP, S_PP    the peel's blocks of S, read off by applying (2.5) to
   //                 the |P| unit vectors of the peeled positions
   //   Z = S_ff⁻¹ S_fP     one warm-started CG solve per column
   //   T = S_PP − S_fPᵀ Z  dense |P| × |P|, LU-factorized for (6.1)
   //   In(T)               the §8 prediction, from the equilibrated T
   // Returns false — and factorize() reports SINGULAR — whenever any of that
   // cannot be stood behind.
   bool build_peel_cache() {
      if (peel_valid_) return peel_ok_;
      peel_valid_ = true;
      peel_ok_ = false;
      ++stats_.cache_builds;
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      // The ASd preconditioner (§7).  It needs diag(S) on the kept positions —
      // assembled in factorize() without assembling S.
      Vec dkept(nf);
      for (int a = 0; a < nf; ++a) dkept[a] = diagS_[kept_[a]];
      const Precond::Info pci = pc_.build(nf, Sk_, Nk_, keptpos_, dkept);
      stats_.pc_blocks_indef += pci.blocks_refused;
      stats_.pc_positions_indef += pci.positions_indef;
      stats_.pc_blocks_singular += pci.blocks_singular;
      if (pci.blocks_refused) {
         std::ostringstream m;
         m << pci.blocks_refused << " of " << nsub_ << " local blocks failed "
              "Cholesky (" << pci.positions_indef << " of " << nf
           << " kept positions); sign-flipped to U|Λ|⁻¹Uᵀ and used anyway"
           << (pci.blocks_singular
                  ? ", except " + std::to_string(pci.blocks_singular) +
                        " singular even for LDLT and skipped"
                  : "");
         warn(Warn::PrecondNotSpd, m.str());
      }

      // Coarse Galerkin matrix S0 = B0ᵀ S_ff B0 — n_coarse_ APPLICATIONS of
      // S_ff, no solves.  Refused (coarse term simply off) if not SPD.
      coarse_ok_ = false;
      if (n_coarse_ > 0) {
         Mat S0 = Mat::Zero(n_coarse_, n_coarse_);
         Vec e(nf), w;
         for (int g = 0; g < n_coarse_; ++g) {
            e.setZero();
            for (int i = 0; i < nf; ++i) if (coarse_grp_[i] == g) e[i] = 1.0;
            apply_Sff_into(e, w);
            for (int i = 0; i < nf; ++i)
               if (coarse_grp_[i] >= 0) S0(coarse_grp_[i], g) += w[i];
         }
         S0 = 0.5 * (S0 + S0.transpose());
         coarse_llt_.compute(S0);
         coarse_ok_ = (coarse_llt_.info() == Eigen::Success);
         if (!coarse_ok_)
            warn(Warn::CoarseNotSpd,
                 "Eigen LLT of S0 did not return Success; the coarse term is "
                 "off for this factorization (CG falls back to one-level ASd)");
      }

      t_neg_ = 0;
      if (nP == 0) { peel_ok_ = true; return true; }   // In(T) of a 0×0 block

      // S_fP and S_PP, one column at a time: S·e_j for each peeled position j,
      // split into its kept and peeled halves.
      SfP_.setZero(nf, nP);
      Mat SPP = Mat::Zero(nP, nP);
      {
         Vec e = Vec::Zero(p_), col;
         for (int j = 0; j < nP; ++j) {
            e[peel_[j]] = 1.0;
            apply_S(e, col);
            e[peel_[j]] = 0.0;
            for (int a = 0; a < nf; ++a) SfP_(a, j) = col[kept_[a]];
            for (int i = 0; i < nP; ++i) SPP(i, j) = col[peel_[i]];
         }
      }

      // Z = S_ff⁻¹ S_fP, column by column, each solve WARM-STARTED from the
      // column's last accepted value (§6): across a δ-retry only the
      // regularization moved, and across Newton steps the barrier path is
      // continuous, so the previous Z is usually a few CG iterations from the
      // new one.  The warm start also breaks the measured cold-start failure
      // mode — CG stalling at rel=1 with no progress at all on a column it
      // solved fine one build later — by handing it a different Krylov space.
      auto applyA = [this](const Vec& v, Vec& out) { apply_Sff_into(v, out); };
      const TwoLevel P2{&pc_, this};
      Z_.resize(nf, nP);
      if (Zwarm_.rows() != nf || Zwarm_.cols() != nP) Zwarm_ = Mat::Zero(nf, nP);
      for (int j = 0; j < nP; ++j) {
         Vec z;
         const Vec w0 = Zwarm_.col(j);
         const CgResult r = cg_solve(applyA, P2, Vec(SfP_.col(j)), z,
                                     opt_.peel_cg_tol, opt_.cg_maxit, &w0);
         stats_.iters += r.iters;
         if (r.indefinite) ++stats_.indef_before;
         if (!(r.rel < 1e-2) || !z.allFinite()) {
            std::ostringstream m;
            m << "column " << j << " of " << nP << " stopped at rel=" << r.rel
              << " after " << r.iters << " iterations"
              << (r.indefinite ? " (pAp <= 0: S_ff is PROVED indefinite here)" : "")
              << (z.allFinite() ? "" : " (non-finite iterate)")
              << "; no peel cache ⇒ no predicted inertia ⇒ SINGULAR";
            warn(Warn::CgPeelColumn, m.str());
            return false;                       // peel_ok_ stays false
         }
         Z_.col(j) = z;
         Zwarm_.col(j) = z;                    // the next build's starting point
      }
      const Mat T = SPP - SfP_.transpose() * Z_;
      if (!T.allFinite()) return false;
      Tlu_.compute(T);

      // In(T), the §8 prediction.  A dense SYMMETRIC EIGENDECOMPOSITION,
      // deliberately: T is |P| × |P| with |P| ≤ a few dozen, so O(|P|³) is
      // nothing, and it sidesteps the reason this file does not trust Eigen's
      // dense LDLT for inertia (a pivoted Cholesky for semi-definite matrices,
      // not Bunch–Kaufman; on indefinite input its pivot signs are unreliable —
      // dd_solver.hpp measured 485–491 negatives where the truth was 512).
      //
      // EQUILIBRATE FIRST (§8).  T mixes the α direction (barrier-scaled, up
      // to ~1e20) with corner-dual directions (down to ~1e-9), so a zero test
      // against T's global eigenvalue scale mistakes honest tiny eigenvalues
      // for noise — measured at N=64 4×4 it refused EVERY factorization that
      // way (|λ|=4e-9 against scale 1e20) and the run could not start.
      // Inertia is invariant under the congruence T → D T D (Sylvester), so
      // the test is made on the diagonally equilibrated matrix, where "small"
      // is meaningful per direction.
      Mat Teq = T;
      {
         double dmax = 0.0;
         for (int i = 0; i < nP; ++i) dmax = std::max(dmax, std::abs(T(i, i)));
         if (dmax > 0.0) {
            Vec dsc(nP);
            for (int i = 0; i < nP; ++i)
               dsc[i] = 1.0 / std::sqrt(std::max(std::abs(T(i, i)), 1e-16 * dmax));
            Teq = dsc.asDiagonal() * T * dsc.asDiagonal();
         }
      }
      // An eigenvalue still too close to zero means In(T) is not well
      // determined, and a prediction we cannot stand behind is worse than
      // none: refuse (§8).
      Eigen::SelfAdjointEigenSolver<Mat> es(Teq);
      if (es.info() != Eigen::Success) return false;
      const Vec ev = es.eigenvalues();
      const double scale = std::max(ev.cwiseAbs().maxCoeff(), 1e-300);
      t_neg_ = 0;
      for (int i = 0; i < nP; ++i) {
         if (std::abs(ev[i]) < 1e-12 * scale) {
            if (std::getenv("DDS_DEBUG"))
               std::cerr << "[dds] In(T) is undetermined (equilibrated |λ|="
                         << std::abs(ev[i]) << " vs scale " << scale << ")\n";
            return false;
         }
         if (ev[i] < 0.0) ++t_neg_;
      }
      peel_ok_ = true;
      return true;
   }

   // The CG interface solve (§7).  Returns false whenever its answer failed
   // the acceptance test — dy still holds the best iterate, and
   // solve_interface decides what to do with it.  factorize() has already
   // built the peel cache (or reported SINGULAR), so by the time a solve
   // arrives the cache is guaranteed valid.
   bool interface_cg(const Vec& ry, Vec& dy) {
      ++stats_.solves;
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      Vec rf(nf);
      for (int a = 0; a < nf; ++a) rf[a] = ry[kept_[a]];

      Vec g;                                    // g = S_ff⁻¹ r_f
      auto applyA = [this](const Vec& v, Vec& out) { apply_Sff_into(v, out); };
      const TwoLevel P2{&pc_, this};
      const CgResult r = cg_solve(applyA, P2, rf, g, opt_.cg_tol, opt_.cg_maxit);
      const double rel = r.rel;
      stats_.iters += r.iters;
      // A pAp <= 0 event is evidence against the §8 premise that S_ff is SPD.
      // Counted rather than acted on: see the CgResult comment for why it
      // over-reports at these condition numbers.
      if (r.indefinite) ++stats_.indef_after;

      // Assemble the full Δy from CG's best iterate WHATEVER its quality, and
      // judge afterwards.  A rejected step is still the best thing available
      // to a caller with no fallback (see solve_interface).
      dy.resize(p_);
      if (nP == 0) {
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = g[a];
      } else {
         Vec rP(nP);
         for (int j = 0; j < nP; ++j) rP[j] = ry[peel_[j]];
         const Vec dP = Tlu_.solve(rP - SfP_.transpose() * g);   // (6.1)
         const Vec df = g - Z_ * dP;                             // (6.2)
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = df[a];
         for (int j = 0; j < nP; ++j) dy[peel_[j]] = dP[j];
      }

      // Honest acceptance test (§7): the residual of the FULL interface
      // system, not of the S_ff subproblem CG actually saw.  The peel algebra
      // sits between the two, so only this test certifies the answer we are
      // about to return.
      Vec Sdy;
      apply_S(dy, Sdy);
      const double true_rel = (ry - Sdy).norm() / std::max(ry.norm(), 1e-300);
      if (!(rel < 1e-2) || !(true_rel < 1e-2) || !dy.allFinite()) {
         ++stats_.rejected;
         std::ostringstream m;
         m << "S_ff rel=" << rel << ", full-interface rel=" << true_rel
           << " after " << r.iters << " CG iterations (accept < 1e-2)"
           << (r.indefinite ? ", pAp <= 0 seen" : "")
           << (dy.allFinite() ? "" : ", non-finite iterate")
           << "; handed back anyway for §9 refinement to judge";
         warn(Warn::CgRejected, m.str());
         return false;
      }
      return true;
   }

   // ---- problem -------------------------------------------------------
   int dim_ = 0, nnz_ = 0, nsub_ = 0, p_ = 0, max_dimk_ = 0;
   Options opt_;
   std::vector<int> irow_, jcol_, owner_;
   std::vector<double> vals_;

   // ---- numbering (pattern only) ---------------------------------------
   std::vector<int> ypos_;                 // KKT index → border position (−1 if interior)
   std::vector<int> lpos_;                 // KKT index → position inside its W_k
   std::vector<int> dimk_;                 // size of each W_k
   std::vector<std::vector<int>> Nk_;      // border positions each subdomain sees
   std::vector<std::vector<int>> ykrow_;   // border position → its row in B_k (−1 if unseen)

   // ---- static symbolic reaches (§5(b), pattern only) -------------------
   //  reach_[k][a]: ascending list of the L_k-columns reachable from column a
   //  of P_k B_kᵀ — the union of elimination-tree paths from its nonzero
   //  positions.  Computed once in route_triplets, valid for every
   //  factorization; the forward solves and the row contraction both iterate
   //  ONLY these lists.
   std::vector<std::vector<std::vector<int>>> reach_;

   // ---- routed triplets (pattern only) ----------------------------------
   std::vector<std::vector<Entry>> wtrip_, btrip_;
   std::vector<Entry> ctrip_;

   // ---- blocks and factorizations (per Newton step) ---------------------
   std::vector<SpMat> W_, B_;
   std::vector<Mat> Sk_;                   // local Schur complements (§5)
   std::vector<std::unique_ptr<Ldlt>> ldlt_;   // one per subdomain
   SpMat C_;                               // the corner block
   Vec diagS_;                             // diag(S), assembled without S
   int n_neg_ = 0;                         // the (8.3) total
   int t_neg_ = 0;                         // In(T)_neg — the §8 prediction

   // ---- peel + CG (per factorization, except the warm starts) -----------
   std::vector<int> peel_, peelpos_;       // peeled border positions, and the inverse
   std::vector<int> kept_, keptpos_;       // the SPD complement CG iterates on
   int n_peel_dual_ = 0;                   // how many peeled entries are duals
   int n_peel_cross_ = 0;                  // ... and how many are cross points
   Precond pc_;                            // the ASd preconditioner
   // ---- EXPERIMENT (DDS_COARSE=1): two-level additive coarse correction ---
   //  One indicator vector per interface EDGE (kept border positions grouped
   //  by the set of tiles that touch them), plus the ASd term:
   //      M2⁻¹ r = B0 (B0ᵀ S_ff B0)⁻¹ B0ᵀ r  +  M_ASd⁻¹ r
   //  — Lueg's eq. (22) with the partition-of-unity basis.  Unlike the peel,
   //  the coarse GALERKIN matrix needs only APPLICATIONS of S_ff (cheap
   //  GEMVs), never S_ff⁻¹ solves — which is why this can work where the
   //  stride-peel experiment failed (its Z columns each cost a CG solve on
   //  the very operator that is saturating).
   std::vector<int> coarse_grp_;           // kept position -> edge group (-1 none)
   int n_coarse_ = 0;
   Eigen::LLT<Mat> coarse_llt_;
   bool coarse_ok_ = false;
   struct TwoLevel {
      const Precond* base;
      const Arrowhead* A;
      void apply(const Vec& r, Vec& z) const {
         base->apply(r, z);
         if (!A->coarse_ok_) return;
         Vec rc = Vec::Zero(A->n_coarse_);
         for (int i = 0; i < (int)A->coarse_grp_.size(); ++i)
            if (A->coarse_grp_[i] >= 0) rc[A->coarse_grp_[i]] += r[i];
         const Vec zc = A->coarse_llt_.solve(rc);
         for (int i = 0; i < (int)A->coarse_grp_.size(); ++i)
            if (A->coarse_grp_[i] >= 0) z[i] += zc[A->coarse_grp_[i]];
      }
   };
   Mat SfP_, Z_;                           // S_fP and Z = S_ff⁻¹ S_fP
   Mat Zwarm_;                             // last accepted Z columns (warm starts)
   Eigen::PartialPivLU<Mat> Tlu_;          // the dense peel complement T, factorized
   bool peel_valid_ = false, peel_ok_ = false;
   Stats stats_;
};

}  // namespace ddsimple

// =============================================================================
//  IPOPT ADAPTER (§1)
//
//  Everything above is plain Eigen and can be tested on its own.  What follows
//  is the thin layer that makes it look like an IPOPT linear solver, plus the
//  AlgorithmBuilder override that injects it.
//
//  Define DD_SIMPLE_NO_IPOPT to compile the header without IPOPT at all (that
//  is how dd_simple_smoke.cpp builds).
// =============================================================================
#ifndef DD_SIMPLE_NO_IPOPT

#include "IpAlgBuilder.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTSymLinearSolver.hpp"

namespace Ipopt {

class DDSimpleSolver : public SparseSymLinearSolverInterface {
public:
   // The driver must set the geometry BEFORE OptimizeNLP, because IPOPT's
   // AlgorithmBuilder constructs the solver itself and no IPOPT option can
   // carry a vector.  Hence the statics.
   static void config(std::vector<int> owner, int nsub) {
      cfg_owner() = std::move(owner);
      cfg_nsub() = nsub;
   }
   static void config_options(const ddsimple::Arrowhead::Options& o) { cfg_opt() = o; }
   static std::vector<int>& cfg_owner() { static std::vector<int> v; return v; }
   static int& cfg_nsub() { static int v = 1; return v; }
   static ddsimple::Arrowhead::Options& cfg_opt() {
      static ddsimple::Arrowhead::Options v;
      return v;
   }
   // Interface telemetry for the whole run.  Static because one solver object
   // serves every continuation level (see SimpleSolverBuilder).
   static ddsimple::Arrowhead::Stats& stats() {
      static ddsimple::Arrowhead::Stats v;
      return v;
   }

   bool InitializeImpl(const OptionsList&, const std::string&) override { return true; }
   EMatrixFormat MatrixFormat() const override { return Triplet_Format; }
   // The §8 prediction answers the inertia question like a factorization
   // would, so IPOPT's δ_w loop runs completely normally.
   bool ProvidesInertia() const override { return true; }
   bool IncreaseQuality() override { return false; }
   Index NumberOfNegEVals() const override { return a_.negative_eigenvalues(); }

   // Called once per continuation level with the (identical) KKT pattern.  The
   // production solver caches across levels; here we simply rebuild — it costs
   // one symbolic analysis per level, a handful of times per run.
   ESymSolverStatus InitializeStructure(Index dim, Index nnz, const Index* ia,
                                        const Index* ja) override {
      std::vector<int> irow(nnz), jcol(nnz);
      for (Index t = 0; t < nnz; ++t) {          // IPOPT is 1-based, we are 0-based
         irow[t] = ia[t] - 1;
         jcol[t] = ja[t] - 1;
      }
      if (!a_.set_structure((int)dim, std::move(irow), std::move(jcol),
                            cfg_owner(), cfg_nsub(), cfg_opt()))
         return SYMSOLVER_FATAL_ERROR;
      return SYMSOLVER_SUCCESS;
   }

   Number* GetValuesArrayPtr() override { return a_.values(); }

   ESymSolverStatus MultiSolve(bool new_matrix, const Index*, const Index*,
                               Index nrhs, Number* rhs_vals, bool check_NegEVals,
                               Index numberOfNegEVals) override {
      if (new_matrix && a_.factorize() != ddsimple::Arrowhead::OK) {
         stats() = a_.stats();
         return SYMSOLVER_SINGULAR;              // IPOPT answers by raising δ_w (§1)
      }
      // The inertia is not the one IPOPT wants: everything factorized fine,
      // the curvature is simply wrong, and δ_w is the correct cure.  This is
      // IPOPT working as designed, not a failure of the decomposition.
      if (check_NegEVals && a_.negative_eigenvalues() != numberOfNegEVals) {
         stats() = a_.stats();
         return SYMSOLVER_WRONG_INERTIA;
      }
      for (Index c = 0; c < nrhs; ++c)
         if (!a_.solve(rhs_vals + (size_t)c * a_.dim())) {
            stats() = a_.stats();
            return SYMSOLVER_SINGULAR;           // no usable step in this RHS
         }
      stats() = a_.stats();
      return SYMSOLVER_SUCCESS;
   }

private:
   ddsimple::Arrowhead a_;
};

// Injection point.  Overriding SymLinearSolverFactory is what makes IPOPT
// route every Newton system through this solver instead of its own:
//
//    SmartPtr<AlgorithmBuilder> b = new SimpleSolverBuilder();
//    app->OptimizeNLP(new TNLPAdapter(GetRawPtr(tnlp)), b);
//
// ONE solver instance serves the whole process, so the structure, the symbolic
// analyses and the warm starts survive across continuation levels.
class SimpleSolverBuilder : public AlgorithmBuilder {
public:
   SmartPtr<SymLinearSolver> SymLinearSolverFactory(const Journalist&,
                                                    const OptionsList&,
                                                    const std::string&) override {
      static SmartPtr<SparseSymLinearSolverInterface> shared = new DDSimpleSolver();
      return new TSymLinearSolver(shared, NULL);
   }
};

}  // namespace Ipopt

#endif  // DD_SIMPLE_NO_IPOPT
#endif  // DD_SOLVER_SIMPLE_HPP
