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
#include <chrono>
#include <cmath>
#include <cstdio>
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

#include "ipopt_phase.hpp"

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
//  DenseBK — a dense symmetric-indefinite factorization with EXACT inertia
//  (Options::dense_blocks, --block-solver dense; needs LAPACK).
//
//  LAPACK's Bunch–Kaufman dsytrf: P Â Pᵀ = L D Lᵀ with 1×1 AND 2×2 pivots, so a
//  KKT block with zero diagonals (λ, ρ rows) factorizes whatever the order,
//  and Sylvester gives the inertia of Â from D: a 1×1 pivot by its sign, a 2×2
//  pivot by its two eigenvalues. Â = S A S is A after symmetric Ruiz
//  equilibration (a congruence, so In(Â) = In(A)), which puts the barrier Σ
//  (up to 1e10) and the δ-sized entries on one scale before pivots are judged.
//  A pivot eigenvalue with |d| <= tol (DDS_BK_TOL, default 1e-13, on the
//  equilibrated scale) counts as ZERO: its sign is below the backward error,
//  and the block is reported numerically singular.
//  This is the reference route, not a production one: O(n³) per block.
// =============================================================================
#ifdef DD_HAVE_LAPACK
extern "C" {
void dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv,
             double* work, const int* lwork, int* info);
void dsytrs_(const char* uplo, const int* n, const int* nrhs, const double* a,
             const int* lda, const int* ipiv, double* b, const int* ldb, int* info);
}
#endif

class DenseBK {
public:
   // A: symmetric, only its LOWER triangle is read.
   bool factorize(const Mat& A) {
      n_ = (int)A.rows();
      neg_ = pos_ = zero_ = 0;
      ok_ = false;
      min_piv_ = std::numeric_limits<double>::infinity();
      if (n_ == 0) { ok_ = true; return true; }
#ifdef DD_HAVE_LAPACK
      a_ = A.triangularView<Eigen::Lower>();
      a_ += A.triangularView<Eigen::StrictlyLower>().transpose();
      // symmetric Ruiz: s_i ← s_i / sqrt(max_j |Â_ij|), a few sweeps
      sc_.setOnes(n_);
      for (int sweep = 0; sweep < 8; ++sweep) {
         Vec r = a_.cwiseAbs().rowwise().maxCoeff();
         bool done = true;
         for (int i = 0; i < n_; ++i) {
            const double f = (r[i] > 0.0) ? 1.0 / std::sqrt(r[i]) : 1.0;
            if (std::abs(f - 1.0) > 1e-3) done = false;
            r[i] = f;
         }
         a_ = r.asDiagonal() * a_ * r.asDiagonal();
         sc_.array() *= r.array();
         if (done) break;
      }
      ipiv_.resize(n_);
      int lwork = 64 * n_, info = 0;
      std::vector<double> work(lwork);
      dsytrf_("L", &n_, a_.data(), &n_, ipiv_.data(), work.data(), &lwork, &info);
      if (info < 0) return false;
      static const double tol = std::getenv("DDS_BK_TOL") ? std::atof(std::getenv("DDS_BK_TOL")) : 1e-13;
      auto classify = [&](double v) {
         min_piv_ = std::min(min_piv_, std::abs(v));
         if (std::abs(v) <= tol) ++zero_;
         else if (v < 0.0) ++neg_;
         else ++pos_;
      };
      for (int k = 0; k < n_;) {
         if (ipiv_[k] > 0) { classify(a_(k, k)); ++k; continue; }
         const double a = a_(k, k), b = a_(k + 1, k), c = a_(k + 1, k + 1);
         const double h = 0.5 * (a + c), r = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
         classify(h + r);
         classify(h - r);
         k += 2;
      }
      ok_ = (info == 0);
      return true;
#else
      return false;
#endif
   }
   // In place, nrhs columns (n × nrhs, column-major): x = A⁻¹ b = S Â⁻¹ S b.
   void solve(double* b, int nrhs) const {
      if (n_ == 0) return;
#ifdef DD_HAVE_LAPACK
      Eigen::Map<Mat> B(b, n_, nrhs);
      B = sc_.asDiagonal() * B;
      int info = 0;
      dsytrs_("L", &n_, &nrhs, a_.data(), &n_, ipiv_.data(), b, &n_, &info);
      B = sc_.asDiagonal() * B;
#else
      (void)b; (void)nrhs;
#endif
   }
   int negative() const { return neg_; }
   int zero() const { return zero_; }
   bool usable() const { return ok_ && zero_ == 0; }   // a solve means something
   double min_pivot() const { return min_piv_; }        // equilibrated scale

private:
   int n_ = 0, neg_ = 0, pos_ = 0, zero_ = 0;
   bool ok_ = false;
   double min_piv_ = 0.0;
   Mat a_;
   Vec sc_;
   std::vector<int> ipiv_;
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
//  nothing).  In floating point at ‖A‖ ~ 1e18 a merely tiny pAp could round
//  non-positive, and the referee experiments recorded runs where it fired
//  while the §8 prediction stayed correct — but a later check against a dense
//  eigendecomposition of S (cameraman N=32, 4×4 and 8×8 tiles, 78 and 126
//  factorizations) found S_ff genuinely indefinite in EXACTLY the
//  factorizations where a Z column broke down, and the uncorrected inertia
//  one to three short of the truth there.  So during the Z build the flag is
//  ACTED ON: the column counts as failed and the factorization is refused
//  (build_peel_cache).  During a later solve it is still only counted.
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
      // EXPERIMENT (A/B against CG): solve with S_ff through a sparse Eigen
      // LDLᵀ of the ASSEMBLED kept×kept block instead of CG.
      //     0   CG everywhere (the design of §6–§7)
      //     1   the peel columns Z = S_ff⁻¹ S_fP by the direct factorization
      //         (one multi-RHS back-solve); the interface solve stays CG
      //     2   the interface solve g = S_ff⁻¹ r_f by it too — no CG at all
      // Assembling S_ff is one walk over the dense S_k blocks and C, so it
      // is cheap; but a factorization of S_ff is exactly the serial global
      // step §2 exists to avoid, so this is a benchmark switch, not the
      // design.  The §8 prediction keeps its refusal semantics: a non-
      // positive pivot of D is the direct analogue of CG's pAp <= 0, and the
      // factorization is refused (SINGULAR) on it.
      int sff_direct = 0;
      // BORDER REGULARIZATION (0 = off, the historical behaviour).
      //
      // An alternative to promoting the multipliers of all-border constraint
      // rows.  Such a row owns no interior column, so its multiplier's row of
      // W_k is structurally EMPTY and W_k is singular for every value of the
      // entries — which is why the driver promotes those multipliers to the
      // border, and why S is then indefinite and the dual peel of §6 is
      // forced (In(S) has exactly one negative eigenvalue per border dual).
      //
      // Leaving them in W_k and regularizing the empty diagonal instead moves
      // that negative eigenvalue from S into W_k, where it is read off a pivot
      // sign for free.  The reported inertia is UNCHANGED —
      //     n_neg = Σ_k n_neg(W_k) + n_neg(S)
      // simply counts the same directions on the other side of the split —
      // but S becomes SPD, so CG is valid on the WHOLE interface and no peel,
      // no Z and no T are needed for definiteness.
      //
      // The price is conditioning, not sign: with δ on the empty diagonal the
      // Schur complement gains +(1/δ)·bbᵀ per regularized row, b being that
      // row's border Jacobian — a POSITIVE semidefinite rank-one spike.  The
      // exactly enforced constraint has become a penalty of parameter 1/δ, so
      // δ trades ‖Ã − A‖ against κ(S): too small and CG cannot converge, too
      // large and §9's refinement has to work harder to recover the step.
      // Use with --no-promote-corners; with promotion on there is nothing
      // structurally empty left to regularize.
      double border_reg = 0.0;
      // DROP THE CORNER REGULARIZATION (--drop-corner-reg, false = historical).
      //
      // In Lueg's form — the consensus formulation with the objective on the
      // copies — the complicating variables appear only in linear linking rows,
      // so the corner block C is zero and needs no regularization (Lueg et al.
      // 2026, §2). Inside IPOPT it is not: the global inertia correction adds
      // δ_w·I to every primal diagonal, the consensus variables included, so a
      // regular iteration hands us C = δ_w·I. With this switch C is replaced by
      // its Lueg value, 0, at every factorization where that is exactly what it
      // is — C a scalar multiple of I with no off-diagonal entry, outside the
      // restoration phase (ipopt_phase.hpp; there C also carries the
      // restoration objective's proximity term, a real part of that problem,
      // and is kept). The δ_w on the interior blocks W_k is untouched. The step
      // then solves a matrix that differs from IPOPT's by δ_w on the border
      // diagonal, which IPOPT's iterative refinement sees.
      bool drop_corner = false;
      // PER-BLOCK DUAL REGULARIZATION (--block-dual-reg; dual_start < 0 = off).
      //
      // A W_k can be singular although the full KKT matrix is not: in the
      // consensus form every copied variable is pinned by its linking row, so
      // W_k is the tile's problem with that data fixed, and the multipliers of
      // gradient rows reading only copied nodes lose their pinning where r ≈ 0
      // (measured: monolithic MUMPS never needs δ_c, the tile blocks do).
      // Reporting SINGULAR then makes IPOPT switch on its GLOBAL δ_c, for every
      // block and for the rest of the run, after which min|λ(W_k)| = δ_c.
      //
      // Instead, a W_k whose LDLᵀ breaks down is refactorized with −ε_k on its
      // DUAL diagonal only (Lueg et al. 2026 §3.5: δ_C^k per partition), ε_k on
      // the ladder kRegMin·10^j up to kRegMax, and the next factorization
      // starts one rung below the last ε_k that worked. The other blocks are
      // untouched. The regularized blocks only precondition: solve() refines
      // against the ORIGINAL triplets, so the step returned is IPOPT's own
      // (the full matrix is nonsingular, so the refinement converges), and
      // IPOPT never sees a singular factorization from a W_k.
      //
      // dual_start = first KKT index of a multiplier (n + n_ineq in the
      // drivers' ordering primal | slacks | λ_c | λ_d).
      int dual_start = -1;
      // W_k-ONLY REGULARIZATION (--wk-reg-h / --wk-reg-c; 0 = off).
      //
      // Not IPOPT's δ_w/δ_c: those come from IPOPT's inertia-correction retries
      // and sit on the WHOLE KKT matrix (corner C and linking rows included).
      // These are two fixed constants put on the subdomain blocks only, for
      // the conditioning of W_k, as in Lueg et al. 2026 (§3.5):
      //
      //          x_k                 y_k                 λ_k              ρ_k
      //   x_k [ ∇²_xx L + δ_H^W      ∇²_xy L             (∇_x h)ᵀ          0 ]
      //   y_k [ ∇²_yx L              ∇²_yy L + δ_H^W     (∇_y h)ᵀ          I ]
      //   λ_k [ ∇_x h                ∇_y h               −δ_C^W I          0 ]
      //   ρ_k [ 0                    I                   0                 0 ]
      //
      // (on top of whatever IPOPT's own δ_w/δ_c already put there). +wk_reg_h
      // goes on every primal diagonal of W_k (x_k, y_k and the inequality
      // slacks), −wk_reg_c on every multiplier diagonal EXCEPT the linking
      // multipliers ρ_k = KKT indices [link_begin, link_end), which the identity
      // (ρ_k, y_k) already pins. C and B_k are untouched. Like dual_start this
      // only changes the factorization: S is formed from the regularized W_k,
      // solve() refines against the ORIGINAL triplets, and the inertia reported
      // to IPOPT is that of the regularized matrix (Haynsworth on W̃_k and S̃),
      // which equals IPOPT's whenever no eigenvalue of the true matrix is
      // smaller in magnitude than the shifts.
      // wk_dual_start = first KKT index of a multiplier (n + n_ineq).
      double wk_reg_h = 0.0, wk_reg_c = 0.0;
      int wk_dual_start = -1, link_begin = -1, link_end = -1;
      // PER-BLOCK INERTIA CORRECTION (--block-inertia; Lueg et al. 2026 §3.5).
      //
      // Each W_k gets its own δ_H^k (primal diagonal: x_k, y_k, slacks) and
      // δ_C^k (multiplier diagonal, linking ρ_k excluded), chosen like IPOPT
      // chooses its global δ_w/δ_c but block by block, until
      //     In(W̃_k) = (#primal unknowns of tile k, #multipliers of tile k, 0).
      // Per factorization, block k first tries δ_H^k = δ_C^k = 0; a breakdown
      // or too FEW negative pivots (a rank-deficient block Jacobian) switches
      // on δ_C^k = 1e-8·μ^{1/4} (IPOPT's jacobian_regularization formula); too
      // MANY negative pivots climb the δ_H^k ladder of IPOPT's defaults —
      // first 1e-4 (or last/3 when the block needed one before), then ×100
      // (first escalation) / ×8, give up above 1e20. With every W̃_k right, the
      // full matrix has IPOPT's inertia iff S̃ = C − Σ B_k W̃_k⁻¹ B_kᵀ has its
      // own (positive definite in the consensus form); when it has not, the
      // reported count is off and IPOPT raises its GLOBAL δ_w — Lueg's global
      // fallback, which also reaches C. Unlike dual_start / wk_reg_*, the
      // shifts are part of the matrix solved: solve() refines against
      // IPOPT's triplets PLUS the block shifts (as IPOPT's SolveOnce solves its
      // own perturbed system), and IPOPT's iterative refinement then measures
      // the step against its unperturbed Newton system, exactly as it does for
      // its own δ_w. Uses wk_dual_start / link_begin / link_end.
      bool block_inertia = false;
      // DENSE BLOCKS (--block-solver dense; needs LAPACK). Every W_k by
      // DenseBK (Bunch–Kaufman, exact inertia, no breakdown on ρρ = 0), S_k by
      // dense solves, and S̃ assembled dense and factorized the same way: its
      // EXACT inertia goes into Haynsworth (no peel, no prediction) and the
      // interface solve is direct. The reference for "correct inertia".
      bool dense_blocks = false;
      // INTERFACE INERTIA (--interface-inertia; needs dense_blocks and
      // block_inertia). When every W̃_k is right but S̃ is not PD, shift the
      // corner block C by δ_S·I (the consensus variables' diagonal) on the
      // same ladder until it is, instead of handing IPOPT a wrong inertia and
      // letting its GLOBAL δ_w hit every block again. Only S̃ is refactorized.
      bool interface_inertia = false;
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
      //   before  CG saw non-positive curvature while BUILDING the prediction;
      //           the factorization was refused (SINGULAR) and IPOPT
      //           regularized — the safe outcome, enforced in build_peel_cache.
      //   after   it happened during a later solve, i.e. an inertia already
      //           handed to IPOPT rests on a premise this run has evidence
      //           against.  Counted only (see CgResult).
      long indef_before = 0, indef_after = 0;
      long pred_refused = 0;  // factorizations where we declined to predict
      // ASd preconditioner health (§7).  An indefinite block is USED anyway,
      // via LDLT (dropping it measured worse — see the Precond header), so the
      // first two are telemetry on the §8 SPD premise, not a failure count.
      // pc_blocks_singular IS a failure: that block was skipped entirely.
      long pc_blocks_indef = 0, pc_positions_indef = 0, pc_blocks_singular = 0;
      // Options::drop_corner telemetry, per factorization: C = δ_w·I with
      // δ_w > 0 zeroed; C already 0 (IPOPT did not regularize); kept because
      // IPOPT was in restoration; kept because C was not a multiple of I.
      long corner_dropped = 0, corner_zero = 0, corner_kept_resto = 0,
           corner_kept_nonscalar = 0;
      // Options::dual_start telemetry: block factorizations that needed a
      // local ε_k > 0, the largest ε_k used, and blocks still broken at kRegMax.
      long blockreg_used = 0, blockreg_failed = 0;
      double blockreg_max = 0.0;
      // Options::block_inertia telemetry, over all block factorizations:
      // needed δ_H^k > 0 / needed δ_C^k > 0 / no shift found (→ SINGULAR),
      // extra LDLᵀ factorizations spent on the ladders, largest δ_H^k, and
      // factorizations IPOPT still found with the wrong inertia (S̃ not PD).
      long binert_blocks = 0, binert_dh = 0, binert_dc = 0, binert_failed = 0;
      long binert_retries = 0, binert_wrong = 0, binert_dc_link = 0;
      long binert_ipopt_dc = 0;   // factorizations IPOPT's own δ_c was on
      // Options::dense_blocks: W_k factorization attempts with a numerically
      // zero pivot (ladder attempts included); factorizations with S̃
      // indefinite / numerically singular.
      long dense_w_zero = 0, dense_s_indef = 0, dense_s_zero = 0;
      // Options::interface_inertia: factorizations that needed δ_S > 0, the
      // largest δ_S, extra S̃ factorizations
      long sfix_used = 0, sfix_retries = 0;
      double sfix_max = 0.0;
      double binert_dh_max = 0.0;
      // Wall-clock seconds, for the CG-vs-direct A/B (Options::sff_direct).
      double t_factor = 0.0;  // factorize() total (W_k, S_k, C, peel cache)
      double t_peel = 0.0;    // ... of which build_peel_cache()
      double t_solve = 0.0;   // solve() total (arrowhead + refinement)
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
      // Eigen has its own OpenMP threading for dense products.  Under the
      // threaded peel-column loop of build_peel_cache that is NESTED
      // parallelism: it oversubscribes the machine, and — because a threaded
      // GEMM reduces in a different order — it makes the CG iteration counts
      // depend on OMP_NUM_THREADS (measured: 891628 at T=1,2,12 against 892986
      // at T=8, same answer but not the same arithmetic).  One thread here, and
      // the parallelism that matters stays ours.
#ifdef _OPENMP
      Eigen::setNbThreads(1);
#endif
      number_unknowns();
      if (!route_triplets()) return false;
      build_peel_sets();
      sff_analyzed_ = false;   // a new pattern ⇒ a new symbolic analysis
      if (std::getenv("DDS_DEBUG"))
         std::cerr << "[dds] dim=" << dim_ << " nnz=" << nnz_
                   << " subdomains=" << nsub_ << " border p=" << p_
                   << " peeled=" << peel_.size() << " (" << n_peel_dual_
                   << " dual, " << n_peel_cross_ << " cross)  max dim W_k="
                   << max_dimk_
                   << (opt_.border_reg > 0.0
                           ? "  border-reg rows=" + std::to_string(n_reg_)
                           : std::string())
                   << "\n";
      return true;
   }

   // The nnz-long value array IPOPT writes into before every factorization.
   double* values() { return vals_.data(); }
   int dim() const { return dim_; }

   enum Status { OK, SINGULAR, WRONG_INERTIA };

   // --------------------------------------------------------------------
   //  FACTORIZATION.  Four steps, in order:
   //     1. factorize every W_k                    (§4, independent per k)
   //     2. form the local Schur blocks S_k        (§5, independent per k)
   //     3. assemble C and diag(S)                 (S itself is never built)
   //     4. build the peel cache, predict In(S)    (§6 + §8)
   // --------------------------------------------------------------------
   Status factorize() {
      const auto t0 = std::chrono::steady_clock::now();
      struct Tally {
         double& acc; std::chrono::steady_clock::time_point t0;
         ~Tally() {
            acc += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
         }
      } tally{stats_.t_factor, t0};
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
         zhits_[k] = 0;
         // Options::dual_start: start one rung below the ε_k that last worked
         // (0 once the ladder has walked down past kRegMin), climb on breakdown.
         if (opt_.dual_start >= 0) {
            const double e = last_eps_[k] / 10.0;
            eps_[k] = (e >= kRegMin) ? e : 0.0;
         }
         fill_from_triplets(B_[k], btrip_[k]);
         if (opt_.block_inertia) {
            ok[k] = factorize_block_inertia(k) ? 1 : 0;
            continue;
         }
         fill_W(k);
         ok[k] = factor_block(k) ? 1 : 0;
         while (!ok[k] && opt_.dual_start >= 0 && eps_[k] < kRegMax) {
            eps_[k] = (eps_[k] == 0.0) ? kRegMin : 10.0 * eps_[k];
            fill_W(k);
            ok[k] = ldlt_[k]->factorize(W_[k]) ? 1 : 0;
         }
         if (opt_.dual_start >= 0) last_eps_[k] = ok[k] ? eps_[k] : kRegMax;
      }
      if (opt_.dense_blocks)
         for (int k = 0; k < nsub_; ++k) stats_.dense_w_zero += zhits_[k];
      if (opt_.block_inertia) {
         // IPOPT's global δ_c, read off its own dual diagonal
         for (int t = 0; t < nnz_; ++t)
            if (irow_[t] == jcol_[t] && irow_[t] >= opt_.wk_dual_start && vals_[t] != 0.0) {
               ++stats_.binert_ipopt_dc;
               break;
            }
         for (int k = 0; k < nsub_; ++k) {
            ++stats_.binert_blocks;
            if (dh_[k] > 0.0) ++stats_.binert_dh;
            if (dc_[k] > 0.0) ++stats_.binert_dc;
            if (dc_link_[k]) ++stats_.binert_dc_link;
            if (!ok[k]) ++stats_.binert_failed;
            stats_.binert_retries += tries_[k] - 1;
            stats_.binert_dh_max = std::max(stats_.binert_dh_max, dh_[k]);
         }
         update_shift();
      }
      // DDS_VERIFY_WK: hand every W_k, exactly as it was factorized (block shifts
      // of dual_start / wk_reg_* / block_inertia included)
      // (lower triangle, global KKT indices), to the driver's checker, built at
      // the iterate the callback recorded.
      // Every factorization of a regular iteration is checked (the iterate is
      // fixed within an iteration; only IPOPT's δ_w/δ_c change between tries).
      if (ipopt_phase::wk_hook() && ipopt_phase::iterate().fresh &&
          !ipopt_phase::restoration()) {
         std::vector<std::vector<ipopt_phase::WkEntry>> blocks(nsub_);
         std::vector<std::vector<int>> glob(nsub_);
         for (int k = 0; k < nsub_; ++k) glob[k].assign(dimk_[k], -1);
         for (int i = 0; i < dim_; ++i)
            if (owner_[i] >= 0) glob[owner_[i]][lpos_[i]] = i;
         for (int k = 0; k < nsub_; ++k) {
            fill_W(k);
            for (int j = 0; j < W_[k].outerSize(); ++j)
               for (SpMat::InnerIterator it(W_[k], j); it; ++it)
                  blocks[k].push_back({glob[k][it.row()], glob[k][it.col()], it.value()});
         }
         ipopt_phase::wk_hook()(blocks);
      }
      if (opt_.dual_start >= 0)
         for (int k = 0; k < nsub_; ++k) {
            if (eps_[k] > 0.0) ++stats_.blockreg_used;
            if (!ok[k]) ++stats_.blockreg_failed;
            stats_.blockreg_max = std::max(stats_.blockreg_max, eps_[k]);
         }
      // DDS_COND=dense0: the exact spectrum of every W_k (dim <= 3000) at each
      // factorization ATTEMPT, before the breakdown test — to tell a singular
      // W_k from an unpivoted-LDLᵀ breakdown on a nonsingular one.
      if (const char* dc = std::getenv("DDS_COND"); dc && std::string(dc) == "dense0") {
         static int n_try = 0;
         ++n_try;
         for (int k = 0; k < nsub_; ++k) {
            if (dimk_[k] > 3000) continue;
            const SpMat full = W_[k].selfadjointView<Eigen::Lower>();
            const Vec ev = Eigen::SelfAdjointEigenSolver<Mat>(Mat(full), Eigen::EigenvaluesOnly)
                               .eigenvalues();
            const Vec a = ev.cwiseAbs();
            int neg = 0;
            for (int i = 0; i < ev.size(); ++i) neg += ev[i] < 0.0;
            std::printf("[dds-cond0] try=%d W_%d dim=%d ldlt=%s  min|lam|=%.3e max|lam|=%.3e "
                        "kappa=%.3e  negative=%d\n", n_try, k, dimk_[k],
                        ok[k] ? "ok    " : "BROKE ", a.minCoeff(), a.maxCoeff(),
                        a.maxCoeff() / a.minCoeff(), neg);
         }
         std::fflush(stdout);
      }
      for (int k = 0; k < nsub_; ++k) {
         if (!ok[k]) {
            // Expected on occasion and self-correcting (§4) — but silence here
            // is how a run that regularized its way out of trouble on every
            // second Newton step passes for a clean one.
            std::ostringstream m;
            m << "W_" << k << " (dim " << dimk_[k] << "): "
              << (opt_.dense_blocks ? std::string("numerically zero Bunch–Kaufman pivot")
                                    : ldlt_[k]->why())
              << "; reporting SINGULAR, IPOPT answers by raising δ_w";
            warn(Warn::LdltNumeric, m.str());
            return SINGULAR;
         }
         n_neg_ += block_neg(k);                       // the Σ_k In(W_k) of (3.1)
      }

      // DDS_COND=1: estimate κ₂(W_k) = max|λ| / min|λ| for every subdomain at
      // every factorization, of the matrix exactly as factorized (barrier Σ and
      // IPOPT's δ_w included). W_k is symmetric indefinite, so max|λ| comes from
      // power iteration and min|λ| from inverse iteration through the LDLᵀ just
      // computed. One line per factorization on stdout, so it interleaves in
      // order with the [mu-coupled] iteration lines.
      if (std::getenv("DDS_COND")) report_condition();

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
         if (opt_.dense_blocks) {                // S_k = −B_k W̃_k⁻¹ B_kᵀ, dense
            Mat X = Mat(SpMat(B_[k].transpose()));
            dbk_[k].solve(X.data(), pk);
            Mat S = -(B_[k] * X);
            Sk_[k] = 0.5 * (S + S.transpose());
            continue;
         }
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

      // DDS_CHECK_C=1: report the corner block at every factorization — its
      // numerical nonzeros, largest entry, largest OFF-diagonal entry and the
      // diagonal's range.  In the consensus formulation the border is the
      // consensus variables only, so C should be diagonal: the objective's
      // curvature on them (--objective consensus) plus IPOPT's δ_w, or δ_w·I
      // alone (--objective copies, strict Lueg form).
      if (std::getenv("DDS_CHECK_C")) {
         static int n_fact = 0;
         int nnz = 0;
         double cmax = 0.0, offmax = 0.0;
         double dmin = std::numeric_limits<double>::infinity(), dmax = -dmin;
         std::vector<double> d(p_, 0.0);
         for (int j = 0; j < C_.outerSize(); ++j)
            for (SpMat::InnerIterator it(C_, j); it; ++it) {
               if (it.value() != 0.0) ++nnz;
               cmax = std::max(cmax, std::abs(it.value()));
               if (it.row() == it.col()) d[it.row()] = it.value();
               else offmax = std::max(offmax, std::abs(it.value()));
            }
         for (double v : d) { dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
         std::cerr << "[dds-C] fact=" << ++n_fact << " p=" << p_ << " nnz=" << nnz
                   << " max|C|=" << cmax << " max|offdiag|=" << offmax
                   << " diag in [" << dmin << ", " << dmax << "]"
                   << (opt_.drop_corner ? (ipopt_phase::restoration() ? " resto" : " regular")
                                        : "")
                   << "\n";
      }

      // Options::drop_corner: replace C by its Lueg value 0 where IPOPT's C is
      // exactly δ_w·I (regular iteration, scalar diagonal, nothing off it).
      cscale_ = 1.0;
      if (opt_.drop_corner && p_ > 0) {
         if (ipopt_phase::restoration()) {
            ++stats_.corner_kept_resto;
         } else {
            bool scalar = true;
            const double c0 = C_.coeff(0, 0);
            for (int j = 0; j < C_.outerSize() && scalar; ++j)
               for (SpMat::InnerIterator it(C_, j); it; ++it)
                  if ((it.row() != it.col() && it.value() != 0.0) ||
                      (it.row() == it.col() && it.value() != c0)) { scalar = false; break; }
            // an implicit (structurally absent) diagonal entry is 0
            if (scalar && c0 != 0.0)
               for (int i = 0; i < p_ && scalar; ++i)
                  if (C_.coeff(i, i) != c0) scalar = false;
            if (scalar) {
               cscale_ = 0.0;
               C_.setZero();
               ++(c0 != 0.0 ? stats_.corner_dropped : stats_.corner_zero);
            } else {
               ++stats_.corner_kept_nonscalar;
            }
         }
      }

      // diag(S), assembled WITHOUT assembling S: the corner diagonal plus each
      // subdomain's own contribution to the border unknowns it touches.  In a
      // distributed code this is one all-reduce over a p-vector — exactly why
      // the ASd preconditioner is built around it (see Precond).
      diagS_.setZero(p_);
      for (const auto& e : ctrip_)
         if (e.r == e.c) diagS_[e.r] += cscale_ * vals_[e.t];
      for (int k = 0; k < nsub_; ++k)
         for (int a = 0; a < (int)Nk_[k].size(); ++a) diagS_[Nk_[k][a]] += Sk_[k](a, a);

      // Options::dense_blocks: S̃ assembled and factorized densely; its exact
      // inertia replaces the §8 prediction and the interface solve is direct.
      if (opt_.dense_blocks) {
         Mat S = Mat(C_) * cscale_;
         for (int k = 0; k < nsub_; ++k)
            for (int a = 0; a < (int)Nk_[k].size(); ++a)
               for (int b = 0; b < (int)Nk_[k].size(); ++b)
                  S(Nk_[k][a], Nk_[k][b]) += Sk_[k](a, b);
         if (!dS_.factorize(S)) return SINGULAR;
         if (dS_.negative() > 0) ++stats_.dense_s_indef;
         if (!dS_.usable()) ++stats_.dense_s_zero;
         // Options::interface_inertia: δ_S on C until S̃ is PD
         if (opt_.interface_inertia && (!dS_.usable() || dS_.negative() > 0)) {
            const Ladder& L = ladder();
            double ds = (last_ds_ > 0.0) ? std::max(1e-20, L.dec * last_ds_) : L.first;
            for (;;) {
               Mat Sd = S;
               Sd.diagonal().array() += ds;
               ++stats_.sfix_retries;
               if (!dS_.factorize(Sd)) return SINGULAR;
               if (dS_.usable() && dS_.negative() == 0) break;
               ds *= (last_ds_ == 0.0 || 1e5 * last_ds_ < ds) ? L.inc_first : L.inc;
               if (ds > 1e20) return WRONG_INERTIA;
            }
            last_ds_ = ds;
            ++stats_.sfix_used;
            stats_.sfix_max = std::max(stats_.sfix_max, ds);
            for (int i = 0; i < dim_; ++i)
               if (owner_[i] < 0) shift_[i] = ds;
         }
         if (!dS_.usable()) return SINGULAR;
         n_neg_ += dS_.negative();
         return OK;
      }

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
         // Options::dual_start: the refusal means CG met non-positive
         // curvature on S_ff, i.e. S is not SPD — a curvature (δ_w) problem,
         // not a rank one. SINGULAR would make IPOPT reach for its GLOBAL δ_c
         // first (and keep it), which is exactly what the per-block dual
         // regularization is there to avoid; report the inertia as wrong.
         return (opt_.dual_start >= 0 || opt_.block_inertia) ? WRONG_INERTIA : SINGULAR;
      }
      n_neg_ += t_neg_;                        // Haynsworth twice: (3.1) + (8.1)
      return OK;
   }

   int negative_eigenvalues() const { return n_neg_; }
   // Options::block_inertia: IPOPT rejected the reported inertia (S̃ not PD)
   void note_wrong_inertia() { if (opt_.block_inertia) ++stats_.binert_wrong; }
   // How many structurally uncoupled unknowns border_reg filled in.  Zero
   // when the option is off, and zero with corner promotion on (those
   // rows left W_k for the border instead).
   int border_reg_rows() const { return n_reg_; }
   // The KKT indices border_reg fills in.  Must be set BEFORE
   // set_structure; ignored unless Options::border_reg > 0.
   void set_reg_rows(std::vector<int> r) { reg_rows_ = std::move(r); }
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
      const auto t0 = std::chrono::steady_clock::now();
      struct Tally {
         double& acc; std::chrono::steady_clock::time_point t0;
         ~Tally() {
            acc += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
         }
      } tally{stats_.t_solve, t0};
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

      // -- pass 2b: the border-regularization diagonal --------------------
      // reg_rows_ holds the KKT indices of the rank-1 pair multipliers the
      // driver would otherwise have promoted to the border (see
      // Options::border_reg).  They are NOT structurally empty in W_k — the
      // (h3x,h3y) pair keeps its (δ,θ) columns with determinant δ, and
      // collapses only as δ → 0 — so no pattern test can find them and the
      // caller has to say which they are.
      //
      // Sign = the inertia the unknown is supposed to carry, so that n_neg is
      // unchanged with respect to the promoting route: these are constraint
      // multipliers, hence negative directions.  An index already on the
      // border, or out of range, is skipped: with promotion ON there is
      // nothing here to do.
      wreg_.assign(nsub_, {});
      n_reg_ = 0;
      if (opt_.border_reg > 0.0) {
         for (int idx : reg_rows_) {
            if (idx < 0 || idx >= dim_) continue;
            const int k = owner_[idx];
            if (k < 0) continue;                 // promoted after all
            wreg_[k].push_back({lpos_[idx],
                                (idx >= opt_.n_primal) ? -1.0 : 1.0});
            ++n_reg_;
         }
      }

      // Options::dual_start: the local positions of each block's multipliers,
      // the only diagonal entries the per-block dual regularization touches.
      wdual_.assign(nsub_, {});
      eps_.assign(nsub_, 0.0);
      last_eps_.assign(nsub_, 0.0);
      if (opt_.dual_start >= 0)
         for (int i = opt_.dual_start; i < dim_; ++i)
            if (owner_[i] >= 0) wdual_[owner_[i]].push_back(lpos_[i]);
      // Options::wk_reg_h / wk_reg_c: the primal and the non-linking multiplier
      // positions of each block.
      // Options::block_inertia uses the same positions for δ_H^k / δ_C^k and
      // needs each block's multiplier count (its target negative inertia).
      wkh_.assign(nsub_, {});
      wkc_.assign(nsub_, {});
      ndual_.assign(nsub_, 0);
      wkr_.assign(nsub_, {});
      dc_link_.assign(nsub_, 0);
      dh_.assign(nsub_, 0.0);
      dc_.assign(nsub_, 0.0);
      last_dh_.assign(nsub_, 0.0);
      tries_.assign(nsub_, 1);
      shift_.clear();
      const bool shift_h = opt_.wk_reg_h > 0.0 || opt_.block_inertia;
      const bool shift_c = opt_.wk_reg_c > 0.0 || opt_.block_inertia;
      for (int i = 0; i < dim_; ++i) {
         if (owner_[i] < 0) continue;
         if (i < opt_.wk_dual_start) {
            if (shift_h) wkh_[owner_[i]].push_back(lpos_[i]);
         } else {
            ++ndual_[owner_[i]];
            if (!(i >= opt_.link_begin && i < opt_.link_end)) {
               if (shift_c) wkc_[owner_[i]].push_back(lpos_[i]);
            } else if (opt_.block_inertia) {
               wkr_[owner_[i]].push_back(lpos_[i]);
            }
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
      dbk_.assign(nsub_, DenseBK());
      zhits_.assign(nsub_, 0);
      reach_.assign(nsub_, {});
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         W_[k].resize(dimk_[k], dimk_[k]);           // lower triangle only
         B_[k].resize(pk, dimk_[k]);
         fill_W(k);                                  // zeros: pattern only
         fill_from_triplets(B_[k], btrip_[k]);       // zeros: pattern for the reach
         const bool analyzed = ldlt_[k]->analyze(W_[k]);
         if (!analyzed) {
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

   // W_k from its triplets PLUS the border-regularization diagonal.  Kept
   // separate from fill_from_triplets because the added entries have no
   // triplet index: their value is ±border_reg, not vals_[t].  setFromTriplets
   // SUMS duplicates, so a diagonal IPOPT already supplied is added to rather
   // than overwritten.  With border_reg == 0 the wreg_ lists are empty and
   // this is byte-identical to the historical path.
   // Largest and smallest |λ| of W_k (DDS_COND). Rayleigh quotients of the
   // power / inverse iterations, stopped at 1e-8 relative change or 500 steps;
   // a deterministic start vector so runs are comparable.
   void extreme_eigs(int k, double& lmax, double& lmin, Vec* vmin = nullptr) const {
      const int n = dimk_[k];
      const auto Wsym = W_[k].selfadjointView<Eigen::Lower>();
      Vec x(n), y(n);
      for (int i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * std::sin(1.0 + i);
      x.normalize();
      lmax = 0.0;
      for (int it = 0; it < 500; ++it) {
         y = Wsym * x;
         const double rq = std::abs(x.dot(y));
         const double ny = y.norm();
         if (ny == 0.0) break;
         x = y / ny;
         if (it > 5 && std::abs(ny - lmax) <= 1e-8 * ny) { lmax = ny; break; }
         lmax = ny;
         (void)rq;
      }
      for (int i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * std::cos(1.0 + i);
      x.normalize();
      double inv = 0.0;                          // estimate of 1/min|λ|
      for (int it = 0; it < 500; ++it) {
         y = x;
         block_solve(k, y.data(), 1);           // y = W_k⁻¹ x
         const double ny = y.norm();
         if (!std::isfinite(ny) || ny == 0.0) { inv = std::numeric_limits<double>::infinity(); break; }
         x = y / ny;
         if (it > 5 && std::abs(ny - inv) <= 1e-8 * ny) { inv = ny; break; }
         inv = ny;
      }
      lmin = (inv > 0.0) ? 1.0 / inv : 0.0;
      if (vmin) *vmin = x;
   }

   void report_condition() {
      static int n_fact = 0;
      ++n_fact;
      std::vector<double> kap(nsub_), lmx(nsub_), lmn(nsub_);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         extreme_eigs(k, lmx[k], lmn[k]);
         kap[k] = (lmn[k] > 0.0) ? lmx[k] / lmn[k] : std::numeric_limits<double>::infinity();
      }
      // DDS_COND=dense: referee the iterative estimates with a dense
      // eigendecomposition of every W_k up to dimension 3000.
      if (std::string(std::getenv("DDS_COND")) == "dense") {
         double worst = 0.0;
         for (int k = 0; k < nsub_; ++k) {
            if (dimk_[k] > 3000) continue;
            const SpMat full = W_[k].selfadjointView<Eigen::Lower>();
            const Mat D = Mat(full);
            const Vec ev = Eigen::SelfAdjointEigenSolver<Mat>(D, Eigen::EigenvaluesOnly)
                               .eigenvalues().cwiseAbs();
            const double kd = ev.maxCoeff() / ev.minCoeff();
            worst = std::max(worst, std::abs(kap[k] - kd) / kd);
         }
         std::printf("[dds-cond] dense check: max relative error of the kappa "
                     "estimates = %.2e\n", worst);
      }
      if (std::string(std::getenv("DDS_COND")) == "blocks") {
         std::printf("[dds-condk] fact=%d", n_fact);
         for (int k = 0; k < nsub_; ++k)
            std::printf(" %.3e/%.1e", kap[k], (opt_.dual_start >= 0) ? eps_[k] : 0.0);
         std::printf("\n");
      }
      std::vector<double> s = kap;
      std::sort(s.begin(), s.end());
      const int kworst = (int)(std::max_element(kap.begin(), kap.end()) - kap.begin());
      // DDS_COND=vec: where the near-null direction of the worst block lives —
      // the 12 largest components of its min-|λ| eigenvector, as GLOBAL KKT
      // indices (decode them with the driver's layout).
      if (std::string(std::getenv("DDS_COND")) == "vec" && n_fact <= 3) {
         double a, b;
         Vec v;
         extreme_eigs(kworst, a, b, &v);
         std::vector<int> glob(dimk_[kworst], -1);
         for (int i = 0; i < dim_; ++i)
            if (owner_[i] == kworst) glob[lpos_[i]] = i;
         std::vector<int> idx(v.size());
         for (int i = 0; i < (int)v.size(); ++i) idx[i] = i;
         std::partial_sort(idx.begin(), idx.begin() + std::min<int>(12, (int)idx.size()),
                           idx.end(), [&](int p, int q) { return std::abs(v[p]) > std::abs(v[q]); });
         std::printf("[dds-cond] fact=%d W_%d min-|lam| eigenvector, largest components "
                     "(global KKT index:value):", n_fact, kworst);
         for (int j = 0; j < std::min<int>(12, (int)idx.size()); ++j)
            std::printf(" %d:%.3f", glob[idx[j]], v[idx[j]]);
         std::printf("\n");
      }
      std::printf("[dds-cond] fact=%d %s nsub=%d dim(W_k) in [%d,%d]  kappa max=%.3e "
                  "median=%.3e min=%.3e  worst W_%d: |lam|max=%.3e |lam|min=%.3e\n",
                  n_fact, ipopt_phase::restoration() ? "resto  " : "regular", nsub_,
                  *std::min_element(dimk_.begin(), dimk_.end()),
                  *std::max_element(dimk_.begin(), dimk_.end()),
                  s.back(), s[s.size() / 2], s.front(), kworst, lmx[kworst], lmn[kworst]);
      std::fflush(stdout);
   }

   void fill_W(int k) {
      std::vector<Trip> t;
      t.reserve(wtrip_[k].size() + wreg_[k].size() + wdual_[k].size() +
                wkh_[k].size() + wkc_[k].size() + wkr_[k].size());
      for (const auto& e : wtrip_[k]) t.emplace_back(e.r, e.c, vals_[e.t]);
      // Options::wk_reg_h / wk_reg_c: the fixed W_k-only shifts (empty = off)
      // (+ Options::block_inertia: this factorization's δ_H^k / δ_C^k)
      for (int pos : wkh_[k]) t.emplace_back(pos, pos, opt_.wk_reg_h + dh_[k]);
      for (int pos : wkc_[k]) t.emplace_back(pos, pos, -(opt_.wk_reg_c + dc_[k]));
      for (int pos : wkr_[k]) t.emplace_back(pos, pos, dc_link_[k] ? -dc_[k] : 0.0);
      // Options::dual_start: −ε_k on the dual diagonal. Always present, as an
      // explicit zero when ε_k = 0, so the pattern the symbolic analysis saw
      // never changes.
      for (int pos : wdual_[k]) t.emplace_back(pos, pos, -eps_[k]);
      for (const auto& g : wreg_[k])
         t.emplace_back(g.first, g.first, g.second * opt_.border_reg);
      W_[k].setFromTriplets(t.begin(), t.end());
      W_[k].makeCompressed();
   }

   // =====================================================================
   //  The solve  (§2 recursion, §6 peel, §7 CG, §9 refinement)
   // =====================================================================

   // y = A·x straight from the input triplets (lower triangle + its mirror).
   // This is the ONLY place the true matrix is applied, and it is what the
   // iterative refinement measures its residual against — so refinement checks
   // the decomposition, not just the arithmetic inside it (§9).
   // The matrix the refinement of solve() measures against. With
   // Options::drop_corner active (cscale_ = 0) the border–border entries are
   // left out, so the refinement converges to the same modified system the
   // arrowhead solves instead of pulling the step back to IPOPT's δ_w·I corner.
   void matvec(const double* x, double* y) const {
      std::fill(y, y + dim_, 0.0);
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const double v = (owner_[i] < 0 && owner_[j] < 0) ? cscale_ * vals_[t]
                                                           : vals_[t];
         y[i] += v * x[j];
         if (i != j) y[j] += v * x[i];
      }
      // Options::block_inertia: the block shifts are part of the matrix solved
      if (!shift_.empty())
         for (int i = 0; i < dim_; ++i) y[i] += shift_[i] * x[i];
   }

   // One W_k factorization through whichever block solver is configured.
   // False = unusable (sparse: LDLᵀ breakdown; dense: a numerically zero
   // pivot — the matrix is singular to working precision).
   bool factor_block(int k) {
      if (!opt_.dense_blocks) return ldlt_[k]->factorize(W_[k]);
      const bool good = dbk_[k].factorize(Mat(W_[k])) && dbk_[k].usable();
      if (!good) ++zhits_[k];
      return good;
   }
   int block_neg(int k) const {
      return opt_.dense_blocks ? dbk_[k].negative() : ldlt_[k]->negative_eigenvalues();
   }
   void block_solve(int k, double* b, int nrhs) const {
      if (opt_.dense_blocks) dbk_[k].solve(b, nrhs);
      else ldlt_[k]->solve(b, dimk_[k], nrhs);
   }

   // Options::block_inertia: factorize W_k with its own inertia correction
   // (see the option). Leaves dh_[k], dc_[k], tries_[k] and the factorization
   // of the accepted W̃_k; false = no shift on the ladder gave the inertia.
   bool factorize_block_inertia(int k) {
      const Ladder& L = ladder();
      const double kDhFirst = L.first, kDhMin = 1e-20, kDhMax = 1e20;
      const double kIncFirst = L.inc_first, kInc = L.inc, kDec = L.dec;
      const double mu = ipopt_phase::mu() > 0.0 ? ipopt_phase::mu() : 0.1;
      const double dc_val = 1e-8 * std::pow(mu, 0.25);
      dh_[k] = 0.0;
      dc_[k] = 0.0;
      dc_link_[k] = 0;
      tries_[k] = 0;
      for (;;) {
         fill_W(k);
         const bool good = factor_block(k);
         ++tries_[k];
         const int neg = good ? block_neg(k) : -1;
         if (good && neg == ndual_[k]) break;
         if (!good || neg < ndual_[k]) {
            // a zero pivot or too few negative directions: rank, not
            // curvature — δ_C^k first, once; δ_H^k cannot add negatives.
            // Lueg's ρρ = 0 is an exact zero pivot whenever the (unpivoted)
            // ordering eliminates a ρ row before its copy y, which no primal
            // shift repairs: if δ_C^k on λ_k alone is not enough, put it on ρ_k
            // too (dc_link_) — the only departure from Lueg's block form.
            // (Forcing y right before ρ in the ordering was tried, 2026-10-04:
            // worse — d_y is often 0 or tiny there; the pair needs the 2×2
            // pivot [d_y 1; 1 0], which an unpivoted LDLᵀ cannot take.)
            if (dc_[k] == 0.0) { dc_[k] = dc_val; continue; }
            if (!dc_link_[k] && !wkr_[k].empty()) { dc_link_[k] = 1; continue; }
            if (good) return false;
         }
         if (dh_[k] == 0.0)
            dh_[k] = (last_dh_[k] == 0.0) ? kDhFirst : std::max(kDhMin, kDec * last_dh_[k]);
         else
            dh_[k] *= (last_dh_[k] == 0.0 || 1e5 * last_dh_[k] < dh_[k]) ? kIncFirst : kInc;
         if (dh_[k] > kDhMax) return false;
      }
      if (dh_[k] > 0.0) last_dh_[k] = dh_[k];
      return true;
   }

   // The δ_H^k / δ_S ladder: IPOPT's defaults (first_hessian_perturbation,
   // perturb_inc_fact_first, perturb_inc_fact, perturb_dec_fact), overridable
   // for exploration by DDS_BI_FIRST / DDS_BI_INCFIRST / DDS_BI_INC / DDS_BI_DEC.
   struct Ladder { double first, inc_first, inc, dec; };
   static const Ladder& ladder() {
      static const Ladder L = [] {
         auto env = [](const char* n, double d) {
            const char* v = std::getenv(n);
            return v ? std::atof(v) : d;
         };
         return Ladder{env("DDS_BI_FIRST", 1e-4), env("DDS_BI_INCFIRST", 100.0),
                       env("DDS_BI_INC", 8.0), env("DDS_BI_DEC", 1.0 / 3.0)};
      }();
      return L;
   }

   // Options::block_inertia: the per-unknown shift of the matrix solved
   void update_shift() {
      shift_.assign(dim_, 0.0);
      for (int i = 0; i < dim_; ++i) {
         const int k = owner_[i];
         if (k < 0) continue;
         if (i < opt_.wk_dual_start) shift_[i] = opt_.wk_reg_h + dh_[k];
         else if (!(i >= opt_.link_begin && i < opt_.link_end))
            shift_[i] = -(opt_.wk_reg_c + dc_[k]);
         else if (dc_link_[k])
            shift_[i] = -dc_[k];
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
         block_solve(k, wk[k].data(), 1);
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
         block_solve(k, wk[k].data(), 1);
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
   // EXPERIMENT (Options::sff_direct): S_ff assembled as a sparse matrix —
   // C restricted to the kept positions plus every S_k block scattered onto
   // them — and factorized by Eigen's SimplicialLDLT.  Same unpivoted LDLᵀ
   // caveat as §4; S_ff is meant to be SPD (that is what the peel is for),
   // and when it is, the factorization is safe and D > 0 confirms it.  A
   // non-positive pivot is the direct-solve analogue of CG's pAp <= 0 and is
   // refused the same way, so the §8 prediction keeps its guarantees.
   bool factorize_Sff_direct() {
      const int nf = (int)kept_.size();
      std::vector<Trip> t;
      {
         size_t est = ctrip_.size() * 2;
         for (int k = 0; k < nsub_; ++k) est += Nk_[k].size() * Nk_[k].size();
         t.reserve(est);
      }
      for (const auto& e : ctrip_) {
         const int i = keptpos_[e.r], j = keptpos_[e.c];
         if (i < 0 || j < 0) continue;
         const double v = cscale_ * vals_[e.t];      // Options::drop_corner
         t.emplace_back(i, j, v);
         if (i != j) t.emplace_back(j, i, v);
      }
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk == 0) continue;
         const Mat& S = Sk_[k];
         for (int a = 0; a < pk; ++a) {
            const int i = keptpos_[Nk_[k][a]];
            if (i < 0) continue;
            for (int b = 0; b < pk; ++b) {
               const int j = keptpos_[Nk_[k][b]];
               if (j >= 0) t.emplace_back(i, j, S(a, b));
            }
         }
      }
      Sff_.resize(nf, nf);
      Sff_.setFromTriplets(t.begin(), t.end());    // duplicates are summed
      Sff_.makeCompressed();
      if (!sff_analyzed_) { sff_ldlt_.analyzePattern(Sff_); sff_analyzed_ = true; }
      sff_ldlt_.factorize(Sff_);
      if (sff_ldlt_.info() != Eigen::Success) {
         warn(Warn::CgPeelColumn,
              "sparse LDLT of the assembled S_ff did not succeed; no peel "
              "cache ⇒ no predicted inertia ⇒ SINGULAR");
         return false;
      }
      const Vec D = sff_ldlt_.vectorD();
      int nonpos = 0;
      for (int i = 0; i < nf; ++i) if (!(D[i] > 0.0)) ++nonpos;
      if (nonpos) {
         ++stats_.indef_before;
         std::ostringstream m;
         m << nonpos << " of " << nf << " pivots of the S_ff LDLT are not "
              "positive: S_ff is not SPD here (the direct analogue of CG's "
              "pAp <= 0); no peel cache ⇒ no predicted inertia ⇒ SINGULAR";
         warn(Warn::CgPeelColumn, m.str());
         return false;
      }
      return true;
   }

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
      if (opt_.dense_blocks) {
         dy = ry;
         dS_.solve(dy.data(), 1);
         return dy.allFinite();
      }
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
      const auto t0 = std::chrono::steady_clock::now();
      struct Tally {
         double& acc; std::chrono::steady_clock::time_point t0;
         ~Tally() {
            acc += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
         }
      } tally{stats_.t_peel, t0};
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

      // EXPERIMENT (Options::sff_direct): assemble and factorize S_ff.  Done
      // before the nP == 0 early return because mode 2 needs the
      // factorization for the interface solve even with nothing peeled.
      if (opt_.sff_direct > 0 && !factorize_Sff_direct()) return false;

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
      // THE loop worth threading.  The columns are independent — one operator,
      // one preconditioner, a different right-hand side each — and this loop is
      // ~90% of a factorization against 0.2% in the W_k loop of factorize(),
      // which is why OMP=1 alone buys nothing measurable (1.00 cores at every
      // OMP_NUM_THREADS before this was threaded).
      //
      // Thread safety: apply_Sff_into, apply_S and Precond::apply are const,
      // write only their output argument and keep every scratch vector on the
      // stack; distinct j touch distinct columns of a column-major Z_.  What is
      // NOT safe is the shared state the serial body used freely — stats_, the
      // Warnings singleton behind warn(), and the early return — so all three
      // move to the serial tail below.
      //
      // DETERMINISM.  The serial loop returned at the FIRST bad column, so
      // columns after it were never solved and never banked a warm start.  An
      // OpenMP loop cannot break, and a "skip once someone failed" flag would
      // make which columns bank a warm start depend on thread timing — i.e.
      // make the whole run nondeterministic.  So every column is solved, and
      // the tail then reproduces the serial semantics exactly: warm starts are
      // banked only up to the first failure, and the counters see only the
      // columns the serial loop would have seen.  Identical state out, at the
      // cost of work that is discarded on a failing build.
      std::vector<long> it_j(nP, 0);
      std::vector<char> indef_j(nP, 0), bad_j(nP, 0), fin_j(nP, 1);
      std::vector<double> rel_j(nP, 1.0);
      if (opt_.sff_direct > 0) {
         // One multi-RHS back-solve through the sparse LDLᵀ instead of nP
         // CG solves.  Judged by the same per-column residual bar as CG.
         Z_ = sff_ldlt_.solve(SfP_);
         const Mat R = Sff_ * Z_ - SfP_;
         for (int j = 0; j < nP; ++j) {
            rel_j[j] = R.col(j).norm() / std::max(SfP_.col(j).norm(), 1e-300);
            fin_j[j] = Z_.col(j).allFinite() ? 1 : 0;
            bad_j[j] = (!(rel_j[j] < 1e-2) || !fin_j[j]) ? 1 : 0;
         }
      } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int j = 0; j < nP; ++j) {
         Vec z;
         const Vec w0 = Zwarm_.col(j);
         const CgResult r = cg_solve(applyA, P2, Vec(SfP_.col(j)), z,
                                     opt_.peel_cg_tol, opt_.cg_maxit, &w0);
         it_j[j] = r.iters;
         indef_j[j] = r.indefinite ? 1 : 0;
         rel_j[j] = r.rel;
         fin_j[j] = z.allFinite() ? 1 : 0;
         // A breakdown (pAp <= 0) fails the column too: S_ff is then not SPD,
         // so the §8 count #neg(T) would not be #neg(S).  Refuse, let IPOPT
         // raise δ_w, and predict on the next attempt instead.
         bad_j[j] = (!(r.rel < 1e-2) || !z.allFinite() || indef_j[j]) ? 1 : 0;
         if (!bad_j[j]) Z_.col(j) = z;
      }
      }   // sff_direct == 0

      // -- serial tail: the shared state, in the serial loop's order ---------
      int first_bad = nP;
      for (int j = 0; j < nP; ++j) if (bad_j[j]) { first_bad = j; break; }
      const int seen = (first_bad < nP) ? first_bad + 1 : nP;
      for (int j = 0; j < seen; ++j) {
         stats_.iters += it_j[j];
         if (indef_j[j]) ++stats_.indef_before;
      }
      // Bank the warm starts the serial loop would have banked BEFORE it
      // returned: every column strictly before the first bad one.  Missing
      // this throws away the warm start on exactly the builds that most need
      // it next time — the ones IPOPT is about to retry with a larger δ_w.
      for (int j = 0; j < first_bad; ++j) Zwarm_.col(j) = Z_.col(j);
      if (first_bad < nP) {
         const int j = first_bad;
         std::ostringstream m;
         m << "column " << j << " of " << nP << " stopped at rel=" << rel_j[j]
           << " after " << it_j[j] << " iterations"
           << (indef_j[j] ? " (pAp <= 0: S_ff is PROVED indefinite here)" : "")
           << (fin_j[j] ? "" : " (non-finite iterate)")
           << "; no peel cache ⇒ no predicted inertia ⇒ SINGULAR";
         warn(Warn::CgPeelColumn, m.str());
         return false;                          // peel_ok_ stays false
      }
      // (the success path banked every column just above: first_bad == nP)
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
      CgResult r;
      if (opt_.sff_direct >= 2) {
         // EXPERIMENT: the sparse LDLᵀ built in build_peel_cache.  Its
         // residual is reported through the same `rel` CG would report.
         g = sff_ldlt_.solve(rf);
         r.rel = (Sff_ * g - rf).norm() / std::max(rf.norm(), 1e-300);
         r.iters = 0;
      } else {
         auto applyA = [this](const Vec& v, Vec& out) { apply_Sff_into(v, out); };
         const TwoLevel P2{&pc_, this};
         r = cg_solve(applyA, P2, rf, g, opt_.cg_tol, opt_.cg_maxit);
      }
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
   // (local index, sign) per block: the structurally uncoupled unknowns
   // whose diagonal border_reg fills in.  Empty unless border_reg > 0.
   std::vector<std::vector<std::pair<int, double>>> wreg_;
   std::vector<int> reg_rows_;             // KKT indices to regularize
   int n_reg_ = 0;                         // how many landed in a block
   std::vector<Entry> ctrip_;
   double cscale_ = 1.0;                   // 0 when Options::drop_corner zeroed C
   // Options::dual_start: per-block dual positions, current and last good ε_k
   std::vector<std::vector<int>> wdual_;
   std::vector<double> eps_, last_eps_;
   // Options::wk_reg_h / wk_reg_c: local positions carrying +δ_H^W / −δ_C^W
   std::vector<std::vector<int>> wkh_, wkc_;
   // Options::block_inertia: per block target #negatives, current δ_H^k /
   // δ_C^k, last accepted δ_H^k > 0, LDLᵀ attempts; shift_ = matrix solved − A
   // wkr_ = linking-multiplier positions, shifted only when dc_link_[k]
   std::vector<int> ndual_, tries_;
   std::vector<std::vector<int>> wkr_;
   std::vector<char> dc_link_;
   std::vector<double> dh_, dc_, last_dh_, shift_;
   static constexpr double kRegMin = 1e-10, kRegMax = 1e-4;

   // ---- blocks and factorizations (per Newton step) ---------------------
   std::vector<SpMat> W_, B_;
   std::vector<Mat> Sk_;                   // local Schur complements (§5)
   std::vector<std::unique_ptr<Ldlt>> ldlt_;   // one per subdomain
   std::vector<DenseBK> dbk_;                  // Options::dense_blocks: one per subdomain
   DenseBK dS_;                                // Options::dense_blocks: S̃
   std::vector<int> zhits_;                    // per block, this factorization
   double last_ds_ = 0.0;                      // Options::interface_inertia
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
   // EXPERIMENT (Options::sff_direct): the assembled S_ff and its sparse LDLᵀ.
   SpMat Sff_;
   Eigen::SimplicialLDLT<SpMat> sff_ldlt_;
   bool sff_analyzed_ = false;
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
   // The rank-1 pair multipliers --border-reg regularizes in place.
   static void config_reg_rows(std::vector<int> r) { cfg_reg_rows() = std::move(r); }
   static std::vector<int>& cfg_owner() { static std::vector<int> v; return v; }
   static int& cfg_nsub() { static int v = 1; return v; }
   static ddsimple::Arrowhead::Options& cfg_opt() {
      static ddsimple::Arrowhead::Options v;
      return v;
   }
   static std::vector<int>& cfg_reg_rows() {
      static std::vector<int> v;
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
      a_.set_reg_rows(cfg_reg_rows());
      if (!a_.set_structure((int)dim, std::move(irow), std::move(jcol),
                            cfg_owner(), cfg_nsub(), cfg_opt()))
         return SYMSOLVER_FATAL_ERROR;
      return SYMSOLVER_SUCCESS;
   }

   Number* GetValuesArrayPtr() override { return a_.values(); }

   ESymSolverStatus MultiSolve(bool new_matrix, const Index*, const Index*,
                               Index nrhs, Number* rhs_vals, bool check_NegEVals,
                               Index numberOfNegEVals) override {
      if (new_matrix) {
         const auto st = a_.factorize();
         if (st != ddsimple::Arrowhead::OK) {
            stats() = a_.stats();
            // --block-dual-reg: a refused inertia prediction is a curvature
            // problem; let IPOPT answer it with δ_w alone (wrong inertia).
            if (st == ddsimple::Arrowhead::WRONG_INERTIA && check_NegEVals) {
               a_.note_wrong_inertia();
               stats() = a_.stats();
               return SYMSOLVER_WRONG_INERTIA;
            }
            return SYMSOLVER_SINGULAR;           // IPOPT answers by raising δ_w (§1)
         }
      }
      // The inertia is not the one IPOPT wants: everything factorized fine,
      // the curvature is simply wrong, and δ_w is the correct cure.  This is
      // IPOPT working as designed, not a failure of the decomposition.
      if (check_NegEVals && a_.negative_eigenvalues() != numberOfNegEVals) {
         a_.note_wrong_inertia();
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
