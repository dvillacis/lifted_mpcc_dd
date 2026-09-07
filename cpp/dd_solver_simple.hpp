// =============================================================================
//  dd_solver_simple.hpp — a READABLE domain-decomposition (arrowhead) linear
//  solver for IPOPT.  Eigen only: no HSL/MA57, no MA97, no MUMPS.
//
//  This is the teaching twin of dd_solver.hpp.  Same mathematics, same IPOPT
//  contract, ~1/5 of the code, because every research lever of the production
//  file is gone (see "WHAT IS NOT HERE" at the bottom of this comment).
//
// -----------------------------------------------------------------------------
//  1.  WHAT IPOPT ASKS OF A LINEAR SOLVER
// -----------------------------------------------------------------------------
//  At every Newton step IPOPT hands us the symmetric augmented KKT matrix A —
//  lower triangle only, in triplet (i, j, value) form, 1-based, possibly with
//  duplicate entries that must be SUMMED — with its own δ_w/δ_c regularization
//  ALREADY applied.  It then wants two things back:
//
//     (a) a solve            A · Δz = r
//     (b) the INERTIA        n_neg = #negative eigenvalues of A
//
//  (b) is not optional.  IPOPT's inertia-correction loop reads n_neg to decide
//  whether the current δ_w makes the reduced Hessian positive definite on the
//  null space of the constraints; if we lie about it the filter line search
//  chases a curvature defect that is not there.  So a solver that only does
//  (a) — a pure Krylov method, say — cannot be plugged in here without also
//  switching IPOPT to its inertia-free mode.
//
//  IPOPT owns everything else: the filter line search, the μ-homotopy, the
//  restoration phase.  We own only the linear algebra.
//
// -----------------------------------------------------------------------------
//  2.  THE ARROWHEAD IDEA
// -----------------------------------------------------------------------------
//  The caller gives us an OWNER MAP: for every KKT index, the subdomain that
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
//  NOTE this is a pure PERMUTATION — no unknowns are duplicated, no linking
//  constraints are introduced — so C is genuinely nonzero and the B_k carry
//  real Jacobian/Hessian entries.
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
//  point of the decomposition.  Only (2.3) is global.
//
//  The inertia comes for free, by HAYNSWORTH ADDITIVITY:
//
//        In(A) = Σ_k In(W_k) + In(S)                                      (2.5)
//
//  so we answer IPOPT's inertia query WITHOUT ever factorizing the full KKT.
//  (Eigenvalues would not do: with the barrier terms Σ ~ z²/μ these matrices
//  reach ‖A‖ ~ 1e18 and eigenvalue signs become rounding noise.  Pivot signs
//  of an LDLᵀ are exact regardless of scaling.)
//
// -----------------------------------------------------------------------------
//  3.  WHAT SOLVES WHAT, IN THIS FILE
// -----------------------------------------------------------------------------
//    W_k   Eigen::SimplicialLDLT  (sparse LDLᵀ, no pivoting).  Gives both the
//          local solves and In(W_k) from the signs of D.  The S_k formation
//          does NOT go through its solve(): it uses the raw L and D directly,
//          for a forward-only, sparse-right-hand-side route — see the long
//          comment at step 2 of factorize().
//    S     assembled sparse, then
//            · Eigen::SimplicialLDLT — for In(S) only (eq. 2.5), plus a free
//              direct fallback since the factorization exists anyway;
//            · preconditioned CONJUGATE GRADIENTS — the actual interface solve,
//              with the additive-Schur (ASd) preconditioner by default. Which
//              preconditioner is used is not a detail: see §6.
//
//  CG needs a symmetric POSITIVE DEFINITE operator, and S is generally not:
//  see §4.  The factorization of S therefore does double duty — it is the exact
//  admissibility gate for CG (we know In(S) before we try) and the safety net
//  when CG fails.
//
//  Which raises the obvious objection, and §7 is about it: if S is assembled and
//  factorized anyway, CG cannot possibly SAVE anything.  Options::inertia_free
//  is the answer — it deletes that factorization, at a price worth measuring.
//
//  ⚠ THE ONE REAL WEAKNESS of the Eigen route.  SimplicialLDLT does NOT pivot.
//  On a symmetric QUASI-DEFINITE matrix (positive definite (1,1) block, negative
//  definite (2,2) block) an unpivoted LDLᵀ always exists, and IPOPT's δ_w > 0,
//  δ_c > 0 make the KKT matrix exactly that.  But IPOPT usually starts a step
//  with δ_w = δ_c = 0, and then a zero pivot is possible.  We detect it
//  (zero / non-finite D entry) and return SYMSOLVER_SINGULAR.  IPOPT responds
//  by raising δ_w and δ_c and handing us the matrix again — which is precisely
//  the regularization that makes the unpivoted factorization safe.  So the
//  failure is self-correcting; the price is extra factorizations per iteration
//  compared with a pivoting Bunch–Kaufman code such as MA57.  That price is
//  the honest cost of dropping HSL, and it is why dd_solver.hpp uses MA57.
//
// -----------------------------------------------------------------------------
//  4.  WHY THERE IS A "PEEL", AND WHY IT IS THE ONLY EXTRA IDEA HERE
// -----------------------------------------------------------------------------
//  Two kinds of border index make S unfit for CG:
//
//    (i)  DUAL border indices.  On a k×k tile partition the driver promotes the
//         cut-corner dual pairs to the border (otherwise the local blocks are
//         structurally rank-deficient).  Those promoted duals are EXACTLY the
//         negative eigenvalues of S — measured: In(S)_neg equals the promoted
//         count at every single solve.  So S is indefinite by construction and
//         plain CG can never run on a tile partition.
//    (ii) The scalar α.  It appears in every subdomain, so its row/column of S
//         is dense, which wrecks both sparsity and the conditioning.
//
//  The PEEL removes both before the Krylov solve.  Split the border into the
//  peeled set P (the promoted duals + α) and the kept set f:
//
//        ⎡ S_ff  S_fP ⎤ ⎡Δy_f⎤   ⎡r_f⎤
//        ⎣ S_Pfᵀ S_PP ⎦ ⎣Δy_P⎦ = ⎣r_P⎦
//
//  With  Z = S_ff⁻¹ S_fP  and  T = S_PP − S_fPᵀ Z  (dense, |P| × |P|, tiny):
//
//        Δy_P = T⁻¹ (r_P − S_fPᵀ S_ff⁻¹ r_f)                              (4.1)
//        Δy_f = S_ff⁻¹ r_f − Z Δy_P                                       (4.2)
//
//  The indefinite and dense directions now live entirely inside T, and S_ff —
//  the operator CG actually iterates on — is SPD.  Z is built once per
//  factorization, one CG solve per column of S_fP.  |P| is small: 2(k−1)² + 1
//  on a k×k tile partition (3 at k=2, 19 at k=4), and just 1 (α alone) on
//  strips or in 1D.
//
//  This is the FETI-DP/BDDC corner treatment, and it is the same construction
//  dd_solver.hpp calls the α peel + dual peel.
//
// -----------------------------------------------------------------------------
//  5.  WHAT IS NOT HERE (deliberately) — look in dd_solver.hpp for these
// -----------------------------------------------------------------------------
//    · MA57 / MA97 / MUMPS block backends and the MUMPS partial-Schur route
//    · the second (nested) arrowhead level on S
//    · signed-LDLᵀ-preconditioned MINRES on the full indefinite S
//    · the LAGGED Schur (caching S_k across Newton steps). The FORWARD-ONLY
//      Schur is kept — see the S_k comment in factorize(); it is exact, it is
//      four lines, and it is what lets the sparse-RHS pruning pay
//    · matrix-free application of S (the distributed cost simulation)
//    · the BJ (block-Jacobi-on-S_k) preconditioner — here you get Jacobi and
//      ASd, which are the two that matter (see the Precond comment)
//    · frozen value-slot maps (here: the sparse matrices are simply rebuilt
//      from triplets every Newton step — a few percent slower, far clearer)
//    · the singular-block census, the arrowhead dump, DD_CHECK
//    · dd_solver.hpp's APPLY_MATFREE (S·y re-derived through K subdomain
//      back-solves EVERY iteration, modelling the regime where the local Schur
//      blocks are not kept).  The inertia-free mode here keeps the S_k, which is
//      what a real distributed code does — see apply_S()
//    · the built-in 2D image geometry (here: the owner map is always injected)
//
//  Environment switches:  DDS_DEBUG=1        partition / CG diagnostics
//                         DDS_SCHUR_CHECK=1  every S_k vs the naive route
//
//  (Sections 6 and 7 below carry the measurements.)
//
// -----------------------------------------------------------------------------
//  6.  MEASURED (cameraman, --hessian exact, macOS, 2026-09-07)
// -----------------------------------------------------------------------------
//  Against IPOPT's own monolithic MUMPS on the SAME instance — the point of the
//  comparison is that the decomposition must not change the answer:
//
//    N=16   mumps                            48 it   α*=0.070831  PSNR 24.97 dB
//           ddsimple 2×2 tiles, direct       50 it   α*=0.070831  PSNR 24.97 dB
//           ddsimple 2×2 tiles, cg           50 it   α*=0.070831  PSNR 24.97 dB
//           ddsimple 2 strips,  cg           50 it   α*=0.070831  PSNR 24.97 dB
//    N=32   mumps                            65 it   α*=0.070859  PSNR 26.34 dB
//           ddsimple 3 strips,  cg           75 it   α*=0.070859  PSNR 26.34 dB
//           ddsimple 3×3 tiles, cg           75 it   α*=0.070859  PSNR 26.34 dB
//
//  Identical solutions everywhere; the ~15% extra iterations are the unpivoted
//  LDLᵀ asking IPOPT for regularization it would not otherwise have needed (§3).
//
//  How much work CG actually carried (solves it completed / solves attempted):
//
//    N=16  2 strips    246/246   45 its/solve   ← no peel needed, no fallbacks
//    N=16  2×2 tiles   105/129   90 its/solve
//    N=16  4×4 tiles    64/ 90 1219 its/solve   ← the peel-by-CG cost, see §4
//    N=32  3 strips    346/346   72 its/solve
//    N=32  3×3 tiles   176/208  334 its/solve
//
//  Cost of the S_k formation — the forward-only + sparse-RHS route of step 2 in
//  factorize(), against the naive "p_k full back-solves with a dense RHS".
//  Whole-run wall clock, --interface direct so the interface solve does not
//  blur the comparison, iteration counts and PSNR identical in every row:
//
//    N=32  3×3 tiles    1.63 s → 0.90 s   1.8×
//    N=48  3×3 tiles    9.06 s → 5.25 s   1.7×
//    N=64  4×4 tiles   23.48 s → 14.19 s  1.65×
//
//  Why only ~1.7× when the triangular-solve work drops far more than that: of
//  L, the pruned forward substitution visits
//
//    N=32  4.4% of the columns, 13.8% of the nonzeros
//    N=48  3.1% of the columns, 14.1% of the nonzeros
//    N=64  3.0% of the columns, 13.7% of the nonzeros
//
//  — about 1/7 of the work — and the backward half is not performed at all, so
//  the triangular solves themselves get roughly 14× cheaper.  What is left is
//  everything else: the W_k factorizations, the Yᵀ D⁻¹ Y products, the assembly
//  and factorization of S.  Those now dominate, which is the point at which
//  optimizing this phase further stops paying.
//
//  (Note how much lower the COLUMN percentage is than the NONZERO percentage:
//  the columns a sparse right-hand side reaches are the fat ones near the root
//  of the elimination tree, where the fill lives.  Pruning by column count
//  would look ~30× better than the work actually saved — worth remembering
//  before quoting a reach ratio as a speedup.)
//
//  DDS_SCHUR_CHECK=1 compares every S_k against the naive two-sided route:
//  rel-err ~1e-12 typical, worst 3e-8 at the most ill-conditioned iterates,
//  where the two roundings diverge and the iterative refinement is what keeps
//  the step honest.
//
//  Two things to read off this.  First, on STRIP partitions there is no peel and
//  no fallback at all: CG carries every interface solve, which is the cleanest
//  demonstration that the scheme works.  Second, the preconditioner matters
//  enormously — with JACOBI instead of ASD the same N=16 2×2 run fell back on
//  EVERY solve (0/50 carried).  A Krylov interface solve stands or falls on its
//  preconditioner, and that is the honest headline of this file.
//
// -----------------------------------------------------------------------------
//  7.  INERTIA-FREE MODE — and the contradiction it resolves
// -----------------------------------------------------------------------------
//  THE OBJECTION.  The point of a Krylov interface solve is to never form or
//  factorize the global object.  But IPOPT wants In(A), Haynsworth turns that
//  into In(S), and In(S) comes only from a factorization of S — so in the
//  DEFAULT mode S is assembled and factorized every Newton step regardless, the
//  direct back-solve is then nearly free, and CG is strictly overhead.  It
//  cannot win.  The objection is correct.
//
//  Note where it bites: NOT on Σ_k In(W_k), which is free (those factorizations
//  happen anyway, locally, in parallel).  Only In(S) forces the one global
//  object and the one serial factorization.
//
//  Options::inertia_free removes exactly that.  S is never assembled and never
//  factorized; the interface operator is applied through the local Schur blocks
//  (apply_S); IPOPT is told ProvidesInertia() = false and falls back on its
//  inertia-free curvature test (Chiang & Zavala — it will refuse to run unless
//  neg_curv_test_tol > 0, so the driver sets both together).
//
//  WHAT IT COSTS.  Three things go at once, and only the first is obvious:
//    · the serial factorization of S               ← the point
//    · the exact CG admissibility gate.  In(S) told us BEFORE a solve whether
//      S_ff was SPD.  Without it CG just runs, and its pSp ≤ 0 guard finds out
//      during the solve, along whichever directions the Krylov space happens to
//      explore.  Strictly weaker.
//    · the free direct fallback.  With no safety net, a CG failure has nowhere
//      to go.  Reporting SINGULAR is truthful but useless — IPOPT answers with
//      δ_w, which cannot fix a Krylov convergence failure, and the run dies
//      (measured: IPOPT Internal_Error on 3 of 4 configurations).  So the mode
//      hands back CG's best iterate instead and lets solve()'s refinement judge
//      it against the true triplets.  That is what makes it survive at all.
//
//  MEASURED (cameraman, --hessian exact, ASd, same solutions to 2 dp of PSNR):
//
//    run                    inertia            inertia-free          its   wall
//    N=16  2 strips      50 it   0.10 s      71 it   0.11 s         1.4×   1.1×
//    N=16  2×2 tiles     50 it   0.10 s     108 it   0.24 s         2.2×   2.4×
//    N=32  3 strips      75 it   1.48 s     115 it   1.81 s         1.5×   1.2×
//    N=32  3×3 tiles     75 it   1.71 s     555 it  20.23 s         7.4×  11.8×
//
//  CG failures (fallbacks / attempted solves) tell the story:
//    N=16 2 strips     0/245  →    5/362
//    N=16 2×2 tiles   24/129  →   96/564
//    N=32 3 strips     0/349  →  232/790
//    N=32 3×3 tiles   32/208  → 1007/3071   (1.67M CG iterations in total)
//
//  READ IT LIKE THIS.  On STRIPS — no promoted duals, S genuinely SPD — the mode
//  works: ~1.4× the iterations and essentially the same wall clock, with the
//  serial factorization gone.  That is a real result: the Amdahl floor CAN be
//  removed here, and what you pay is IPOPT iterations, not linear algebra.
//
//  On TILES it degrades badly, and the reason is precisely the second and third
//  costs above.  Tile partitions are where S is indefinite by construction, so
//  they are exactly where knowing In(S) in advance was doing the most work — and
//  losing the gate turns "skip CG, back-solve instead" into "run CG for 500
//  iterations, fail, take the bad step anyway, make IPOPT clean it up".
//
//  So the inertia is not only a tax.  It also buys the certificate that CG may
//  run and the insurance for when it may not, and dropping it costs both.  §8 is
//  the attempt to keep them without the factorization.
//
// -----------------------------------------------------------------------------
//  8.  PREDICTED INERTIA  (Options::PREDICTED) — the middle road
// -----------------------------------------------------------------------------
//  IDEA.  Do not compute In(S); DERIVE it from something already at hand.
//  Haynsworth applies to the peel split of S exactly as it applied to the
//  arrowhead split of A:
//
//        In(S) = In(S_ff) + In(T),        T = S_PP − S_fPᵀ S_ff⁻¹ S_fP     (8.1)
//
//  and T is ALREADY BUILT — it is the peel cache of §4, |P| × |P| with |P|
//  between 1 and a few dozen.  So In(T) costs a dense symmetric
//  eigendecomposition of a tiny matrix: nothing.
//
//  The entire prediction is therefore one assumption:
//
//        In(S_ff) = 0,  i.e. the field block is positive definite.           (8.2)
//
//  What makes that the right assumption to bet on is that IT IS ALSO WHAT CG
//  REQUIRES.  The peel exists precisely to make S_ff definite (§4); if (8.2)
//  fails, CG could not have run either.  The mode is self-consistent: the
//  operator is usable exactly when the inertia is right.
//
//  WHY In(S)_neg IS PREDICTABLE AT ALL.  Write In(A)_neg = m + ν, where m is the
//  number of dual unknowns and ν the negative-curvature count of the reduced
//  Hessian — IPOPT drives δ_w until ν = 0.  Each W_k is itself a saddle-point
//  matrix, so In(W_k)_neg = m_k + ν_k with m_k its interior duals.  Every dual
//  row is owned by one subdomain or promoted to the border, so Σ_k m_k =
//  m − p_dual.  Substituting into In(A) = Σ_k In(W_k) + In(S):
//
//        In(S)_neg = p_dual + ν − Σ_k ν_k                                   (8.3)
//
//  which is exactly the measured fact of §4 — In(S)_neg equals the promoted
//  corner-dual count — whenever the local and global curvature counts agree.
//  (8.1) is the better route in practice because it needs no such assumption
//  about ν: T sees whatever the corner duals actually contribute.
//
//  REFUSING TO PREDICT.  If the peel cache cannot be built — a CG solve for a
//  column of Z fails to converge — no prediction is issued and the factorization
//  is reported SINGULAR.  That is deliberate and it is the mode's safety
//  property: a prediction we cannot stand behind would corrupt IPOPT's δ_w loop
//  silently, whereas SINGULAR merely costs a re-factorization at a larger δ_w.
//  The same applies when an eigenvalue of T is too close to zero for its sign to
//  be meaningful.
//
//  MEASURED (cameraman, --hessian exact, ASd, DDS_INERTIA_CHECK=1 scoring every
//  prediction against a factorization of S computed purely as a referee):
//
//    run              exact          predicted                  none
//    N=16 2 strips   50 it 0.11 s   50 it 0.09 s  78/79 ✓    71 it 0.11 s
//    N=16 2×2 tiles  50 it 0.10 s   80 it 0.21 s  99/99 ✓   108 it 0.24 s
//    N=32 3 strips   75 it 1.49 s   80 it 1.48 s 111/117 ✓  115 it 1.83 s
//    N=32 3×3 tiles  75 it 1.72 s  125 it 4.51 s 132/133 ✓ 1051 it 63.31 s
//
//  Every mismatch was off by exactly ONE.  All three modes reach the same
//  solution (PSNR identical to 2 dp).
//
//  READ IT LIKE THIS.  The prediction is right 99–100% of the time, and it
//  recovers most of what §7 gave up: at N=32 3×3 it turns 1051 iterations and
//  63 s back into 125 iterations and 4.5 s.  On STRIPS it is free — same
//  iteration count, same wall clock as EXACT, with the serial factorization of S
//  gone.  That is the result this whole line of questioning was after.
//
//  What still costs on TILES is not the prediction being wrong; it is the
//  REFUSALS (90 and 205 of them above), where CG could not build the peel cache
//  and the factorization had to be thrown away.  So the remaining obstacle has
//  moved: it is no longer "we must factorize S to know its inertia", it is "the
//  peel cache needs a Krylov solve that sometimes does not converge".  That is a
//  preconditioning problem, and preconditioning problems are the kind a
//  distributed implementation can attack.
// =============================================================================
#ifndef DD_SOLVER_SIMPLE_HPP
#define DD_SOLVER_SIMPLE_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace ddsimple {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// -----------------------------------------------------------------------------
//  A sparse symmetric LDLᵀ with the two things we need from it: a solve, and
//  the inertia read off the signs of D.  Wraps Eigen::SimplicialLDLT so the
//  "did it break down?" test lives in exactly one place (see §3 of the header
//  comment for why breakdown is possible at all).
//
//  Only the LOWER triangle of the matrix handed to compute() is read, so a
//  matrix stored lower-only and a matrix stored fully symmetric both work.
// -----------------------------------------------------------------------------
class Ldlt {
public:
   // Symbolic analysis: fill-reducing ordering + elimination tree.  Depends on
   // the sparsity pattern ONLY, and the KKT pattern is constant for a whole
   // run, so this is done once and every Newton step reuses it.
   bool analyze(const SpMat& A) {
      f_.analyzePattern(A);
      analyzed_ = (f_.info() == Eigen::Success);
      return analyzed_;
   }

   // Numeric factorization.  False = this matrix is unusable (breakdown or a
   // zero pivot); the caller must report SYMSOLVER_SINGULAR rather than solve.
   bool factorize(const SpMat& A) {
      ok_ = false;
      n_neg_ = 0;
      if (!analyzed_ && !analyze(A)) return false;
      f_.factorize(A);
      if (f_.info() != Eigen::Success) return false;
      // Inertia = signs of D.  A zero or non-finite pivot means the unpivoted
      // factorization has broken down: the solve would return garbage and the
      // inertia would be meaningless, so both are refused.
      const Vec& d = f_.vectorD();
      for (int i = 0; i < d.size(); ++i) {
         if (!std::isfinite(d[i]) || d[i] == 0.0) return false;
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

   // ---- the pieces of the factorization, for the sparse-RHS Schur route ----
   //
   //  Eigen factorizes  W = P⁻¹ L D Lᵀ P  (its solve applies, in order, P, L⁻¹,
   //  D⁻¹, L⁻ᵀ, P⁻¹).  L is UNIT lower triangular and Eigen stores only its
   //  STRICTLY lower entries; D lives in a separate vector.  Both are needed by
   //  the S_k formation in Arrowhead::factorize — see the long comment there.
   const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>&
   permutationP() const { return f_.permutationP(); }
   const Vec& vectorD() const { return f_.rawD(); }

   //  y ← L⁻¹ y, in place, for ONE right-hand side — the SPARSE-RHS forward
   //  substitution, and the whole point of it is the `continue`.
   //
   //  A forward substitution propagates column j of L only through the
   //  multiplier y[j].  When y[j] is zero, column j contributes nothing and can
   //  be skipped entirely — so a right-hand side with a handful of nonzeros
   //  touches only the part of L those nonzeros can reach, instead of all of it.
   //  Correct for ANY y (a zero multiplier really does contribute nothing), so
   //  no structural analysis, reach computation or elimination tree is needed:
   //  the test IS the pruning.
   //
   //  Ascending j is the right order because L is lower triangular — by the time
   //  the loop reaches column j, every column that could have written into y[j]
   //  has already been processed, so y[j] is final.
   //
   //  Eigen's own sparse-RHS overload does NOT do this: it converts the
   //  right-hand side to dense panels (solve_sparse_through_dense_panels) and
   //  runs the ordinary dense solve on them.
   //
   //  The one inefficiency left is the outer sweep itself, which is O(n) even
   //  when almost nothing is reached; an elimination-tree reach would visit only
   //  the reached columns, at the cost of a second data structure.  Measured at
   //  ~3% of the work here, so it is not worth the machinery.
   void forward_solve(double* y) const {
      const auto& L = f_.rawL();
      const int n = (int)L.cols();
      for (int j = 0; j < n; ++j) {
         const double yj = y[j];
         if (yj == 0.0) continue;              // <- the pruning
         for (SpMat::InnerIterator it(L, j); it; ++it)
            if ((int)it.row() > j) y[it.row()] -= it.value() * yj;
      }
   }

private:
   // Eigen keeps the raw factor and D protected — reasonably, since they are
   // useless without knowing the exact convention above.  This is the only way
   // to reach them, and it adds no behaviour of its own.
   struct Impl : Eigen::SimplicialLDLT<SpMat, Eigen::Lower> {
      const SpMat& rawL() const { return this->m_matrix; }
      const Vec& rawD() const { return this->m_diag; }
   };
   Impl f_;
   bool analyzed_ = false, ok_ = false;
   int n_neg_ = 0;
};

// -----------------------------------------------------------------------------
//  The interface preconditioner.  Two choices, both taken from Lueg's paper as
//  implemented by the Python reference (dd_kkt.py's make_preconditioner):
//
//    JACOBI   P = diag(|S_ff|).  Trivial, and the right answer in 1D, where the
//             interface is so small that CG terminates in <= p steps anyway.
//             It does NOT work on a 2D interface: measured on cameraman N=16,
//             2x2 tiles, every CG solve stalled and fell back to the direct
//             route.  Kept because it is the honest baseline to compare with.
//
//    ASD      (Lueg eq. 20-21) "additive Schur, assembled diagonal".  For each
//             subdomain take its LOCAL Schur block S_k, restrict it to the kept
//             border positions that subdomain touches, REPLACE ITS DIAGONAL by
//             the diagonal of the ASSEMBLED S, invert the small dense result,
//             and apply the inverses additively.
//
//             The diagonal swap is the whole trick.  S_k on its own describes
//             what subdomain k alone does to the shared border unknowns, so it
//             badly underestimates their true self-coupling — every other
//             subdomain touching the same unknown contributes to that diagonal
//             as well.  diag(S) is the assembled truth, and it is also the one
//             quantity a distributed implementation can share cheaply (a single
//             all-reduce over a vector), which is exactly why Lueg picks it.
//
//  A singular local block is simply skipped: a preconditioner is allowed to be
//  incomplete, and if what remains is useless CG's rz-breakdown guard catches
//  it and the solve falls back to the direct route.
// -----------------------------------------------------------------------------
class Precond {
public:
   enum Kind { JACOBI, ASD };

   // nf = number of kept border positions (the dimension CG works in).
   // Sk / Nk / keptpos describe the local Schur blocks and how their rows map
   // onto kept positions; diagS_kept is diag(S) gathered on the kept positions.
   void build(Kind kind, int nf, const std::vector<Mat>& Sk,
              const std::vector<std::vector<int>>& Nk,
              const std::vector<int>& keptpos, const Vec& diagS_kept) {
      kind_ = kind;
      nf_ = nf;
      if (kind_ == JACOBI) {
         // |.| rather than the raw diagonal: an entry that has drifted negative
         // off the central path would make P indefinite, and CG's rz > 0 guard
         // would then abort a solve that is otherwise perfectly fine.
         d_.resize(nf);
         for (int c = 0; c < nf; ++c)
            d_[c] = 1.0 / std::max(std::abs(diagS_kept[c]), 1e-300);
         return;
      }
      Minv_.assign(Nk.size(), Mat());
      idx_.assign(Nk.size(), {});
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
         const Mat Mi = M.inverse();
         if (!Mi.allFinite()) continue;     // singular local block -> skip it
         Minv_[k] = Mi;
         idx_[k] = gl;
      }
   }

   // z <- P^-1 r
   void apply(const Vec& r, Vec& z) const {
      if (kind_ == JACOBI) { z = r.cwiseProduct(d_); return; }
      z.setZero(nf_);
      for (size_t k = 0; k < idx_.size(); ++k) {
         const std::vector<int>& idx = idx_[k];
         const int mb = (int)idx.size();
         if (mb == 0) continue;
         Vec rk(mb);
         for (int i = 0; i < mb; ++i) rk[i] = r[idx[i]];
         const Vec zk = Minv_[k] * rk;
         for (int i = 0; i < mb; ++i) z[idx[i]] += zk[i];   // ADDITIVE
      }
   }

private:
   Kind kind_ = ASD;
   int nf_ = 0;
   Vec d_;                                  // JACOBI: the already-inverted diagonal
   std::vector<Mat> Minv_;                  // ASD: the small dense block inverses
   std::vector<std::vector<int>> idx_;      // ASD: their kept-position lists
};

// -----------------------------------------------------------------------------
//  Preconditioned conjugate gradients.
//
//  Note the signature: it takes applyA, a CALLABLE, not a matrix.  That is not
//  decoration — the only thing CG ever needs of an operator is its action on a
//  vector, and keeping that in the type is what lets the same routine serve
//  both the assembled interface matrix and the matrix-free one (see §7).
//
//  Parameter-free (the step lengths come out of the Krylov space itself), and
//  it keeps the BEST iterate rather than the last one so a stalled run still
//  returns something usable.  The two breakdown guards are what turn "CG
//  assumes SPD" into a runtime check:
//        rz  ≤ 0  →  the preconditioner is not SPD
//        pSp ≤ 0  →  A is not SPD along this search direction
//  Either one means the caller must fall back to the direct solve.
//
//  Returns the relative residual actually achieved; the caller decides whether
//  that is good enough (interface_cg() applies its own acceptance test).
// -----------------------------------------------------------------------------
//  What CG reports back.  `indefinite` is the interesting one: pAp <= 0 exhibits
//  a search direction along which A is not positive definite.  In exact
//  arithmetic that is a one-directional PROOF of indefiniteness (not seeing it
//  proves nothing).  In floating point on these matrices it is weaker than it
//  sounds: with ‖A‖ ~ 1e18 a pAp that is merely tiny can come out non-positive
//  from rounding alone, so the flag over-reports.  §8 measured exactly that —
//  runs where it fired a dozen times still predicted the inertia correctly on
//  78 of 79 factorizations.  Treat it as evidence, not as a certificate.
struct CgResult {
   double rel = 1.0;      // best relative residual reached
   long iters = 0;
   bool indefinite = false;   // pAp <= 0 was observed
};

template <class ApplyA>
inline CgResult cg_solve(ApplyA&& applyA, const Precond& P, const Vec& b, Vec& x,
                         double tol, int maxit) {
   const int n = (int)b.size();
   const double bnorm = std::max(b.norm(), 1e-300);
   x.setZero(n);
   Vec r = b, z(n);
   P.apply(r, z);
   Vec p = z, Ap(n);
   double rz = r.dot(z);
   CgResult res;
   double best_rel = 1.0;
   Vec best = Vec::Zero(n);
   int best_it = 0, it = 0;
   const int stall = 40;   // give up after this many iterations with no gain
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
//  The solver itself.  Pure Eigen — this class knows nothing about IPOPT, so it
//  can be unit-tested standalone.  The IPOPT adapter is at the bottom of the
//  file.
//
//  Life cycle:
//        set_structure(...)   once per sparsity pattern  (partition + symbolic)
//        values()             refresh the matrix entries  (every Newton step)
//        factorize()          W_k, S_k, S, inertia
//        solve(rhs)           the arrowhead solve, in place
// =============================================================================
class Arrowhead {
public:
   struct Options {
      // KKT indices >= n_primal are DUAL unknowns.  Border positions with such
      // an index are peeled (§4 (i)).  Leave it huge to disable the dual peel.
      int n_primal = 1 << 30;
      // KKT index of the scalar α, or −1 if the formulation has none (§4 (ii)).
      int alpha_index = -1;
      bool use_cg = true;      // false = always use the direct LDLᵀ solve of S
      // How In(S) is obtained — the axis §7 and §8 are about.
      //   EXACT      assemble S, factorize it, read the pivot signs.  The
      //              default, and the only mode with a direct fallback.
      //   PREDICTED  never assemble or factorize S; In(S) = In(T) from the tiny
      //              dense peel complement, assuming S_ff is SPD (§8).  Still
      //              reports an inertia to IPOPT, so δ_w works normally.
      //   NONE       never assemble or factorize S and report NO inertia; IPOPT
      //              must use its inertia-free curvature test (§7).
      // PREDICTED and NONE both force use_cg — there is nothing to solve
      // directly with.
      enum InertiaMode { EXACT, PREDICTED, NONE };
      InertiaMode inertia = EXACT;
      // ASd is the default; Jacobi is measurably useless on a 2D interface.
      Precond::Kind precond = Precond::ASD;
      double cg_tol = 1e-10;
      int cg_maxit = 500;
   };

   struct Stats {
      long solves = 0;        // interface solves attempted through CG
      long iters = 0;         // CG iterations summed over all of them
      long fallbacks = 0;     // CG ran but its answer was rejected
      long skipped = 0;       // CG never ran (S inadmissible for it)
      long cache_builds = 0;  // peel caches built (one per factorization)
      // §8 telemetry.  The two falsification counters are NOT the same thing:
      //   before  CG saw non-positive curvature while BUILDING the prediction,
      //           so no prediction was issued — the safe outcome (SINGULAR).
      //   after   it happened during a later solve, i.e. an inertia already
      //           handed to IPOPT rests on a premise this run has evidence
      //           against.  Compare it with pred_wrong before concluding
      //           anything: the flag over-reports (see CgResult).
      long indef_before = 0, indef_after = 0;
      long pred_refused = 0;  // factorizations where we declined to predict
      long pred_checked = 0;  // DDS_INERTIA_CHECK: predictions compared
      long pred_wrong = 0;    //   ... of which disagreed with the true In(S)
      long pred_maxerr = 0;   //   ... the largest |predicted − true| seen
   };

   // --------------------------------------------------------------------
   //  STRUCTURE.  irow/jcol are 0-based lower-triangle coordinates, owner is
   //  one label per KKT index (subdomain id, or −1 for border).  Everything
   //  that depends only on the pattern is computed here, once.
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
                   << " dual)  max dim W_k=" << max_dimk_ << "\n";
      return true;
   }

   // The nnz-long value array IPOPT writes into before every factorization.
   double* values() { return vals_.data(); }
   int dim() const { return dim_; }

   enum Status { OK, SINGULAR };

   // --------------------------------------------------------------------
   //  FACTORIZATION.  Four steps, in order:
   //     1. factorize every W_k                       (independent per k)
   //     2. form the local Schur blocks S_k = −B_k W_k⁻¹ B_kᵀ  (independent)
   //     3. assemble S = C + Σ_k scatter(S_k)         (eq. 2.1)
   //     4. factorize S — for In(S) (eq. 2.5) and as CG's safety net
   // --------------------------------------------------------------------
   Status factorize() {
      n_neg_ = 0;
      peel_valid_ = false;   // S changed ⇒ the peel cache is void
      cg_dead_ = false;      // …and CG gets a fresh chance on the new operator

      // ---- 1. the subdomain blocks ------------------------------------
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
            // No artificial shift here.  A masked rank deficiency would
            // corrupt the very inertia signal that drives IPOPT's δ_w loop, so
            // we report the truth and let IPOPT regularize (§3).
            if (std::getenv("DDS_DEBUG"))
               std::cerr << "[dds] W_" << k << " (dim " << dimk_[k]
                         << ") factorization failed → SINGULAR\n";
            return SINGULAR;
         }
         n_neg_ += ldlt_[k]->negative_eigenvalues();   // the Σ_k In(W_k) of (2.5)
      }

      // ---- 2. the local Schur blocks ----------------------------------
      //  S_k = −B_k W_k⁻¹ B_kᵀ, a dense p_k × p_k block, where p_k = |N_k| is
      //  the number of border unknowns subdomain k actually touches.
      //
      //  Written naively this is p_k FULL back-solves (forward, diagonal,
      //  backward) through W_k with a dense right-hand side, and it is the
      //  dominant cost of the entire factorization.  Two observations remove
      //  most of it.  Both are EXACT — nothing here is an approximation:
      //
      //  (a) FORWARD ONLY.  With W_k = P⁻¹ L D Lᵀ P (Eigen does no scaling, so
      //      the halves compose exactly),
      //
      //          B_k W_k⁻¹ B_kᵀ = (L⁻¹ P B_kᵀ)ᵀ D⁻¹ (L⁻¹ P B_kᵀ) = Yᵀ D⁻¹ Y,
      //
      //      using Pᵀ = P⁻¹.  The Lᵀ half of the solve cancels against the B_k
      //      that multiplies it back on the left, so the BACKWARD substitution
      //      is never performed at all.  As a bonus, B_k itself disappears from
      //      the product: Y already carries it.
      //
      //  (b) SPARSE RIGHT-HAND SIDE.  The columns of B_kᵀ carry only a handful
      //      of nonzeros each (each border unknown couples to a few interior
      //      ones), so most of L is unreachable from them and the pruned
      //      forward substitution in Ldlt::forward_solve never touches it.
      //
      //  The two compose especially well, and not by accident: (b) can ONLY
      //  help the forward half — a back-substituted vector is dense however
      //  sparse the right-hand side was — and (a) is exactly what deletes the
      //  backward half.  Separately each is worth a little; together they turn
      //  "solve with every column of B_kᵀ" into "touch the reachable part of L
      //  once per column".
      //
      //  What is paid instead is the Yᵀ D⁻¹ Y product, a dense p_k × p_k × dim_k
      //  GEMM.  It replaces the backward triangular solve, so the trade is only
      //  a win while p_k stays small against dim_k — i.e. large subdomains with
      //  small interfaces, which is the regime the decomposition is for anyway.
      //
      //  DDS_SCHUR_CHECK=1 verifies every block against the naive two-sided
      //  route, per block per factorization.
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
         Mat Y = Mat::Zero(nk, pk);
         for (int a = 0; a < pk; ++a) {
            double* col = Y.data() + (size_t)a * nk;    // Y is column-major
            for (SpMat::InnerIterator it(PBt, a); it; ++it) col[it.row()] = it.value();
            ldlt_[k]->forward_solve(col);               // Y(:,a) ← L⁻¹ P B_kᵀ(:,a)
         }
         const Mat Z = ldlt_[k]->vectorD().cwiseInverse().asDiagonal() * Y;
         Sk_[k].noalias() = -(Y.transpose() * Z);
         // Yᵀ D⁻¹ Y is symmetric in exact arithmetic; the computed product
         // differs in the last bits.  In(S) and the scatter both assume exact
         // symmetry, so enforce it in place.
         for (int j = 0; j < pk; ++j)
            for (int i = 0; i < j; ++i) {
               const double m = 0.5 * (Sk_[k](i, j) + Sk_[k](j, i));
               Sk_[k](i, j) = m;
               Sk_[k](j, i) = m;
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

      // ---- 3. the corner block C --------------------------------------
      // Always built: it is small (only the border–border KKT couplings) and
      // both interface routes need it.  Fully symmetric, since it is applied to
      // vectors rather than factorized.
      std::vector<Trip> t;
      t.reserve(ctrip_.size() * 2 + s_entries_);
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
      // distributed code this is one all-reduce over a p-vector — which is
      // exactly why Lueg's ASd preconditioner is built around it (see Precond).
      diagS_.setZero(p_);
      for (const auto& e : ctrip_)
         if (e.r == e.c) diagS_[e.r] += vals_[e.t];
      for (int k = 0; k < nsub_; ++k)
         for (int a = 0; a < (int)Nk_[k].size(); ++a) diagS_[Nk_[k][a]] += Sk_[k](a, a);

      // ---- 4. In(S) ----------------------------------------------------
      // This is the step §7 and §8 are about: the one global object and the one
      // serial factorization in an otherwise embarrassingly parallel scheme.  It
      // exists ONLY because IPOPT asks for In(A) and In(A) = Σ_k In(W_k) + In(S)
      // needs In(S).
      //
      // DDS_INERTIA_CHECK=1 assembles and factorizes S even in the modes that do
      // not need it, purely to print predicted-vs-true.  It changes nothing the
      // algorithm uses — apply_S() keys off the MODE, not off whether S_ happens
      // to exist — so the checked run takes exactly the same steps.
      static const bool inertia_check = std::getenv("DDS_INERTIA_CHECK") != nullptr;
      const bool need_S = (opt_.inertia == Options::EXACT) || inertia_check;
      int s_true = -1;
      if (need_S) {
         for (int k = 0; k < nsub_; ++k) {        // scatter every S_k
            const int pk = (int)Nk_[k].size();
            for (int b = 0; b < pk; ++b)
               for (int a = 0; a < pk; ++a)
                  t.emplace_back(Nk_[k][a], Nk_[k][b], Sk_[k](a, b));
         }
         S_.resize(p_, p_);
         S_.setFromTriplets(t.begin(), t.end());
         S_.makeCompressed();
         if (!s_analyzed_) {
            // The pattern of S is fixed by the partition, so this happens once.
            if (!ldltS_.analyze(S_)) {
               std::cerr << "[dds] symbolic analysis of S failed\n";
               return SINGULAR;
            }
            s_analyzed_ = true;
         }
         if (!ldltS_.factorize(S_)) {
            if (opt_.inertia != Options::EXACT) {
               s_true = -1;                       // check only; not fatal here
            } else {
               if (std::getenv("DDS_DEBUG"))
                  std::cerr << "[dds] interface matrix S (p=" << p_
                            << ") factorization failed → SINGULAR\n";
               return SINGULAR;
            }
         } else {
            s_true = ldltS_.negative_eigenvalues();
         }
      }
      if (opt_.inertia == Options::EXACT) {
         s_neg_ = s_true;
         n_neg_ += s_neg_;                        // Haynsworth, eq. (2.5)
         return OK;
      }
      if (opt_.inertia == Options::NONE) {
         s_neg_ = -1;                             // unknown, and nobody may ask
         n_neg_ = 0;
         return OK;
      }
      // ---- PREDICTED (§8) ----------------------------------------------
      // In(S) = In(S_ff) + In(T) by Haynsworth on the peel split, and the whole
      // prediction is the single assumption In(S_ff) = 0 — which is ALSO exactly
      // what CG requires of S_ff, so the mode is self-consistent: the operator
      // is usable iff the inertia is right.  T is |P| × |P| — three to a few
      // dozen — so In(T) is computed exactly, densely, and for nothing.
      //
      // Building the peel cache HERE rather than lazily at the first solve is
      // what makes the prediction available in time to answer IPOPT.
      if (!build_peel_cache()) {
         // Refusing is the point: a prediction we cannot stand behind would
         // silently corrupt IPOPT's δ_w loop, whereas SINGULAR merely costs a
         // re-factorization at a larger δ_w.
         ++stats_.pred_refused;
         if (std::getenv("DDS_DEBUG"))
            std::cerr << "[dds] peel cache failed, so In(S) cannot be predicted "
                         "→ SINGULAR\n";
         return SINGULAR;
      }
      s_neg_ = t_neg_;
      n_neg_ += s_neg_;
      if (inertia_check) {
         ++stats_.pred_checked;
         if (s_true >= 0 && s_true != s_neg_) {
            ++stats_.pred_wrong;
            stats_.pred_maxerr =
               std::max(stats_.pred_maxerr, (long)std::abs(s_true - s_neg_));
         }
         std::cerr << "[dds-inertia] predicted In(S)_neg=" << s_neg_
                   << "  true=" << (s_true < 0 ? std::string("n/a")
                                               : std::to_string(s_true))
                   << (s_true == s_neg_ ? "  MATCH" : "  MISMATCH")
                   << "   (peeled duals=" << n_peel_dual_ << ", |P|="
                   << peel_.size() << ")\n";
      }
      return OK;
   }

   int negative_eigenvalues() const { return n_neg_; }
   const Stats& stats() const { return stats_; }
   void reset_stats() { stats_ = Stats(); }

   // --------------------------------------------------------------------
   //  SOLVE, in place.  rhs is overwritten by the solution.
   //
   //  Wrapped in a few sweeps of ITERATIVE REFINEMENT against the ORIGINAL
   //  triplets.  This is not a luxury: near-singular pivots (IPOPT's own
   //  δ_c = 1e-8·μ^¼ on the dual directions, for instance) lose ~10 digits
   //  through the Schur assembly — measured rel-res ~0.1 unrefined against
   //  ~1e-12 refined.  We keep the best iterate seen and always hand it back,
   //  which is exactly what a monolithic sparse solver does at a nasty
   //  iterate: return its answer and let IPOPT's globalization cope.
   //
   //  Returns false only if the residual never even evaluated finite — then
   //  rhs holds nothing usable and the caller must report SINGULAR.
   // --------------------------------------------------------------------
   bool solve(double* rhs) {
      const std::vector<double> b0(rhs, rhs + dim_);
      std::vector<double> x(b0), r(dim_), Ax(dim_), best;
      // Without a direct fallback the very first solve can genuinely fail; then
      // there is no step to refine and rhs is left alone.
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
      if (!std::isfinite(best_res)) return false;   // rhs left as the RHS
      std::copy(best.begin(), best.end(), rhs);
      return true;
   }

private:
   // =====================================================================
   //  Structure: numbering, routing, peel sets
   // =====================================================================

   // Give every unknown its coordinates in the permuted picture:
   //   border index  i  →  ypos_[i] ∈ [0, p)      its column of S
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
   // selection matrix), build the sparsity patterns of W_k and B_k, and run the
   // symbolic analysis of every W_k.  All of this depends on the PATTERN only.
   bool route_triplets() {
      // -- pass 1: which border positions does each subdomain touch? --
      std::vector<std::vector<char>> seen(nsub_, std::vector<char>(p_, 0));
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const int oi = owner_[i], oj = owner_[j];
         if (oi >= 0 && oj >= 0 && oi != oj) {
            // The owner map is supposed to be a partition with no direct
            // subdomain-to-subdomain coupling; if it is not, the arrowhead
            // shape does not hold and nothing below is valid.
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
      s_entries_ = 0;
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

      // -- allocate the blocks and analyze the W_k --
      W_.assign(nsub_, SpMat());
      B_.assign(nsub_, SpMat());
      Sk_.assign(nsub_, Mat());
      // held by pointer: Eigen's sparse solvers are neither copyable nor
      // movable, so they cannot live in a vector directly
      ldlt_.clear();
      for (int k = 0; k < nsub_; ++k) ldlt_.emplace_back(new Ldlt());
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         s_entries_ += (size_t)pk * pk;              // for the S assembly reserve
         W_[k].resize(dimk_[k], dimk_[k]);           // lower triangle only
         B_[k].resize(pk, dimk_[k]);
         fill_from_triplets(W_[k], wtrip_[k]);       // zeros: pattern only
         if (!ldlt_[k]->analyze(W_[k])) {
            std::cerr << "[dds] symbolic analysis of W_" << k << " failed\n";
            return false;
         }
      }
      return true;
   }

   // Decide which border positions are PEELED (§4): every dual border index,
   // plus α.  The rest are KEPT — those are the ones CG iterates on.
   void build_peel_sets() {
      peel_.clear();
      n_peel_dual_ = 0;
      if (opt_.use_cg) {
         for (int i = 0; i < dim_; ++i)
            if (owner_[i] < 0 && i >= opt_.n_primal) {
               peel_.push_back(ypos_[i]);
               ++n_peel_dual_;
            }
         const int a = opt_.alpha_index;
         if (a >= 0 && a < dim_ && owner_[a] < 0) peel_.push_back(ypos_[a]);
         std::sort(peel_.begin(), peel_.end());
         peel_.erase(std::unique(peel_.begin(), peel_.end()), peel_.end());
      }
      peelpos_.assign(p_, -1);
      keptpos_.assign(p_, -1);
      kept_.clear();
      for (size_t j = 0; j < peel_.size(); ++j) peelpos_[peel_[j]] = (int)j;
      for (int j = 0; j < p_; ++j)
         if (peelpos_[j] < 0) { keptpos_[j] = (int)kept_.size(); kept_.push_back(j); }
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
   //  The solve
   // =====================================================================

   // y = A·x straight from the input triplets (lower triangle + its mirror).
   // This is the ONLY place the true matrix is applied, and it is what the
   // iterative refinement measures its residual against — so refinement checks
   // the decomposition, not just the arithmetic inside it.
   void matvec(const double* x, double* y) const {
      std::fill(y, y + dim_, 0.0);
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         y[i] += vals_[t] * x[j];
         if (i != j) y[j] += vals_[t] * x[i];
      }
   }

   // One arrowhead solve — equations (2.2)–(2.4), in place.  Returns false only
   // when the interface solve failed outright, which can happen only when there
   // is no factorization of S to fall back on (inertia-free mode).
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

   // ---------------------------------------------------------------------
   //  APPLYING THE INTERFACE OPERATOR.  Two routes, same operator:
   //
   //    assembled     out = S·y, one sparse matrix–vector product.
   //    matrix-free   out = C·y + Σ_k N_k S_k (N_kᵀ y) — the SAME sum that the
   //                  assembly would have carried out, evaluated on the fly.
   //                  S is never formed, and every term is LOCAL to one
   //                  subdomain: each rank owns its dense S_k, and the only
   //                  communication is one reduction over the border vector.
   //
   //  This is the distributed algorithm, not a simulation of it.  Note what it
   //  does NOT do: it does not re-derive S·y through K subdomain back-solves
   //  every iteration (dd_solver.hpp's APPLY_MATFREE does, to model the stricter
   //  regime where the local Schur blocks are not kept either).  Keeping S_k is
   //  the normal choice — it is formed once per Newton step, locally, and turns
   //  every CG iteration into a small dense product.
   // ---------------------------------------------------------------------
   void apply_S(const Vec& y, Vec& out) const {
      if (opt_.inertia == Options::EXACT) { out.noalias() = S_ * y; return; }
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

   // S_ff v — the operator CG actually iterates on.  Embedding v with zeros on
   // the peeled positions, applying the full S and reading back the kept ones
   // IS S_ff v; there is no need to hold the sub-block separately.
   void apply_Sff_into(const Vec& v, Vec& out) const {
      if (peel_.empty()) { apply_S(v, out); return; }
      Vec full = Vec::Zero(p_);
      for (size_t a = 0; a < kept_.size(); ++a) full[kept_[a]] = v[(int)a];
      Vec Sf;
      apply_S(full, Sf);
      out.resize((int)kept_.size());
      for (size_t a = 0; a < kept_.size(); ++a) out[(int)a] = Sf[kept_[a]];
   }

   // ---------------------------------------------------------------------
   //  The interface solve (2.3).
   //
   //  DEFAULT MODE: preconditioned CG on S_ff with the peel, and the direct
   //  LDLᵀ back-solve as the safety net.  The fallback is FREE — S is already
   //  factorized, because the inertia needed it — which is what makes it safe
   //  to be aggressive about rejecting a CG answer we do not fully trust.  It
   //  is also why CG cannot WIN in this mode: see §7.
   //
   //  INERTIA-FREE MODE: CG is all there is.  A failure is reported honestly
   //  and IPOPT regularizes, exactly as for a singular block.
   // ---------------------------------------------------------------------
   bool solve_interface(const Vec& ry, Vec& dy) {
      if (opt_.use_cg && cg_admissible() && interface_cg(ry, dy)) return true;
      if (opt_.inertia == Options::EXACT) {     // the free direct safety net
         dy = ry;
         ldltS_.solve(dy.data(), p_, 1);
         return true;
      }
      // PREDICTED / NONE: there is no safety net, so take CG's best iterate even
      // though we do not trust it.  That is the same bargain the rest of this
      // file makes — solve() measures the TRUE residual against the original
      // triplets and keeps the best step it has seen — and it matters here:
      // reporting SINGULAR instead sends IPOPT into a δ_w loop that cannot fix
      // a Krylov convergence failure, and the run dies.  Only a solve that
      // produced no iterate at all (a failed peel cache) is reported as failed.
      if (dy.size() == p_ && dy.allFinite()) return true;
      if (std::getenv("DDS_DEBUG"))
         std::cerr << "[dds-cg] interface solve produced nothing and there is "
                      "no factorization of S to fall back on\n";
      return false;
   }

   // Is CG allowed to run on this factorization's S?
   //
   // In the default mode the gate is EXACT, not a heuristic: In(S) comes from
   // the LDLᵀ pivots, and the peeled duals are the only negative directions we
   // have arranged to remove.  If S has any other negative eigenvalue, S_ff is
   // not SPD and CG has no business being there.
   //
   // Inertia-free mode cannot ask that question — not knowing In(S) is the whole
   // point — so CG simply runs, and its own pSp ≤ 0 breakdown guard is what
   // catches an indefinite operator.  That is strictly weaker: the guard finds
   // out during the solve rather than before it, and only along the directions
   // the Krylov space happens to explore.
   bool cg_admissible() {
      // EXACT is the only mode with something to skip TO.  In PREDICTED the gate
      // would be circular (we predicted S_ff SPD; asking whether S_ff is SPD is
      // the same statement), and in NONE there is nothing to ask with.  So there
      // CG always runs and its own pAp ≤ 0 guard is the only protection — which
      // is weaker, but is also the falsification test of §8.
      if (opt_.inertia != Options::EXACT) return true;
      if (cg_dead_) { ++stats_.skipped; return false; }
      if (s_neg_ != n_peel_dual_) { ++stats_.skipped; return false; }
      return true;
   }

   // Build, once per factorization, everything the peel needs (§4):
   //   S_fP, S_PP    the peel's blocks of S, by applying S to the |P| unit
   //                 vectors of the peeled positions — so this works unchanged
   //                 whether S is assembled or not
   //   Z = S_ff⁻¹ S_fP     one CG solve per column of S_fP
   //   T = S_PP − S_fPᵀ Z  dense, |P| × |P|, factorized by LU
   //
   // Note the honesty of the Z construction: EVERY system involving S_ff in
   // this file, including these, is solved by CG.  It is also the expensive
   // part — |P| CG solves per factorization.  dd_solver.hpp instead factorizes
   // S_ff sparsely once and gets Z with a single multi-RHS back-solve; that is
   // the optimization to reach for in the DEFAULT mode, where S is factorized
   // anyway.  It is not available inertia-free, which is a real cost of that
   // mode rather than an oversight.
   bool build_peel_cache() {
      if (peel_valid_) return peel_ok_;
      peel_valid_ = true;
      peel_ok_ = false;
      ++stats_.cache_builds;
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      // The preconditioner (see the Precond comment above).  It needs diag(S)
      // on the kept positions — assembled in factorize() without assembling S.
      Vec dkept(nf);
      for (int a = 0; a < nf; ++a) dkept[a] = diagS_[kept_[a]];
      pc_.build(opt_.precond, nf, Sk_, Nk_, keptpos_, dkept);

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

      // Z = S_ff⁻¹ S_fP, column by column.
      auto applyA = [this](const Vec& v, Vec& out) { apply_Sff_into(v, out); };
      Z_.resize(nf, nP);
      for (int j = 0; j < nP; ++j) {
         Vec z;
         const CgResult r = cg_solve(applyA, pc_, Vec(SfP_.col(j)), z,
                                     opt_.cg_tol, opt_.cg_maxit);
         stats_.iters += r.iters;
         if (r.indefinite) ++stats_.indef_before;
         if (!(r.rel < 1e-2) || !z.allFinite()) {
            if (std::getenv("DDS_DEBUG"))
               std::cerr << "[dds-cg] peel column " << j << " did not converge"
                            " (rel=" << r.rel << (r.indefinite
                                ? ", S_ff PROVED indefinite)" : ")") << "\n";
            return false;                       // peel_ok_ stays false
         }
         Z_.col(j) = z;
      }
      const Mat T = SPP - SfP_.transpose() * Z_;
      if (!T.allFinite()) return false;
      Tlu_.compute(T);

      // In(T), for the predicted-inertia mode (§8).  A dense SYMMETRIC
      // EIGENDECOMPOSITION, deliberately: T is |P| × |P| with |P| ≤ a few dozen,
      // so O(|P|³) is nothing, and it sidesteps the reason this file does not
      // trust Eigen's dense LDLT for inertia (it is a pivoted Cholesky for
      // semi-definite matrices, not Bunch–Kaufman, and on indefinite input its
      // pivot signs are unreliable — dd_solver.hpp measured 485–491 negatives
      // where the truth was 512).
      //
      // An eigenvalue too close to zero means In(T) is not well determined, and
      // a prediction we cannot stand behind is worse than none: refuse.
      Eigen::SelfAdjointEigenSolver<Mat> es(T);
      if (es.info() != Eigen::Success) return false;
      const Vec ev = es.eigenvalues();
      const double scale = std::max(ev.cwiseAbs().maxCoeff(), 1e-300);
      t_neg_ = 0;
      for (int i = 0; i < nP; ++i) {
         if (std::abs(ev[i]) < 1e-12 * scale) {
            if (std::getenv("DDS_DEBUG"))
               std::cerr << "[dds] In(T) is undetermined (|λ|=" << std::abs(ev[i])
                         << " vs scale " << scale << ")\n";
            return false;
         }
         if (ev[i] < 0.0) ++t_neg_;
      }
      peel_ok_ = true;
      return true;
   }

   // The CG interface solve.  Returns false whenever its answer must not be
   // used — the caller then falls back to the direct back-solve, or reports
   // failure if there is none.
   bool interface_cg(const Vec& ry, Vec& dy) {
      ++stats_.solves;
      if (!build_peel_cache()) { ++stats_.fallbacks; cg_dead_ = true; return false; }
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      Vec rf(nf);
      for (int a = 0; a < nf; ++a) rf[a] = ry[kept_[a]];

      Vec g;                                    // g = S_ff⁻¹ r_f
      auto applyA = [this](const Vec& v, Vec& out) { apply_Sff_into(v, out); };
      const CgResult r = cg_solve(applyA, pc_, rf, g, opt_.cg_tol, opt_.cg_maxit);
      const double rel = r.rel;
      stats_.iters += r.iters;
      // A pAp <= 0 event is evidence against the §8 premise that S_ff is SPD.
      // Counted rather than acted on: see the CgResult comment for why it
      // over-reports at these condition numbers.
      if (r.indefinite) ++stats_.indef_after;

      // Assemble the full Δy from CG's best iterate WHATEVER its quality, and
      // judge afterwards.  A rejected step is still the best thing available to
      // a caller that has no fallback (see solve_interface).
      dy.resize(p_);
      if (nP == 0) {
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = g[a];
      } else {
         Vec rP(nP);
         for (int j = 0; j < nP; ++j) rP[j] = ry[peel_[j]];
         const Vec dP = Tlu_.solve(rP - SfP_.transpose() * g);   // (4.1)
         const Vec df = g - Z_ * dP;                             // (4.2)
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = df[a];
         for (int j = 0; j < nP; ++j) dy[peel_[j]] = dP[j];
      }

      // Honest acceptance test: the residual of the FULL interface system, not
      // of the S_ff subproblem CG actually saw.  The peel algebra sits between
      // the two, so only this test certifies the answer we are about to return.
      Vec Sdy;
      apply_S(dy, Sdy);
      const double true_rel = (ry - Sdy).norm() / std::max(ry.norm(), 1e-300);
      if (!(rel < 1e-2) || !(true_rel < 1e-2) || !dy.allFinite()) {
         ++stats_.fallbacks;
         cg_dead_ = true;
         return false;
      }
      return true;
   }

   // ---- problem -------------------------------------------------------
   int dim_ = 0, nnz_ = 0, nsub_ = 0, p_ = 0, max_dimk_ = 0;
   Options opt_;
   std::vector<int> irow_, jcol_, owner_;
   std::vector<double> vals_;

   // ---- numbering -----------------------------------------------------
   std::vector<int> ypos_;                 // KKT index → border position (−1 if interior)
   std::vector<int> lpos_;                 // KKT index → position inside its W_k
   std::vector<int> dimk_;                 // size of each W_k
   std::vector<std::vector<int>> Nk_;      // border positions each subdomain sees
   std::vector<std::vector<int>> ykrow_;   // border position → its row in B_k (−1 if unseen)

   // ---- routed triplets ------------------------------------------------
   std::vector<std::vector<Entry>> wtrip_, btrip_;
   std::vector<Entry> ctrip_;
   size_t s_entries_ = 0;                  // Σ_k p_k², the S-assembly reserve

   // ---- blocks and factorizations --------------------------------------
   std::vector<SpMat> W_, B_;
   std::vector<Mat> Sk_;                   // local Schur complements
   std::vector<std::unique_ptr<Ldlt>> ldlt_;   // one per subdomain
   SpMat C_;                               // the corner block, always built
   Vec diagS_;                             // diag(S), assembled without S
   SpMat S_;                               // the assembled interface matrix
   Ldlt ldltS_;                            // its factorization: inertia + fallback
   bool s_analyzed_ = false;               // (S_/ldltS_ unused unless EXACT)
   int n_neg_ = 0;
   int s_neg_ = 0;                         // In(S)_neg, or −1 when not computed
   int t_neg_ = 0;                         // In(T)_neg — the §8 prediction

   // ---- peel + CG ------------------------------------------------------
   std::vector<int> peel_, peelpos_;       // peeled border positions, and the inverse
   std::vector<int> kept_, keptpos_;       // the SPD complement CG iterates on
   int n_peel_dual_ = 0;                   // how many peeled entries are duals
   Precond pc_;                            // the interface preconditioner
   Mat SfP_, Z_;                           // S_fP and Z = S_ff⁻¹ S_fP
   Eigen::PartialPivLU<Mat> Tlu_;          // the dense peel Schur complement T
   bool peel_valid_ = false, peel_ok_ = false;
   bool cg_dead_ = false;                  // CG failed on THIS factorization's S
   Stats stats_;
};

}  // namespace ddsimple

// =============================================================================
//  IPOPT ADAPTER
//
//  Everything above is plain Eigen and can be tested on its own.  What follows
//  is the thin layer that makes it look like an IPOPT linear solver, plus the
//  AlgorithmBuilder override that injects it.
//
//  Define DD_SIMPLE_NO_IPOPT to compile the header without IPOPT at all (that
//  is how the standalone unit test of Arrowhead builds).
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
   // The one place the inertia experiment is visible to IPOPT.  EXACT and
   // PREDICTED both answer the inertia question — one from a factorization of S,
   // one from In(T) — so IPOPT's δ_w loop runs normally.  Only NONE returns
   // false, which makes IPOPT fall back on its inertia-free curvature test
   // (Chiang & Zavala) — and it REFUSES to do that unless neg_curv_test_tol has
   // been set positive, so the driver must set both together.
   bool ProvidesInertia() const override {
      using O = ddsimple::Arrowhead::Options;
      return cfg_opt().inertia != O::NONE;
   }
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
         return SYMSOLVER_SINGULAR;              // IPOPT answers by raising δ_w
      }
      // The inertia is not the one IPOPT wants: everything factorized fine, the
      // curvature is simply wrong, and δ_w is the correct cure.  This is IPOPT
      // working as designed, not a failure of the decomposition.  (IPOPT does
      // not ask when ProvidesInertia() is false; the guard makes that explicit.)
      if (check_NegEVals && ProvidesInertia() &&
          a_.negative_eigenvalues() != numberOfNegEVals) {
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

// Injection point.  Overriding SymLinearSolverFactory is what makes IPOPT route
// every Newton system through our solver instead of its own:
//
//    SmartPtr<AlgorithmBuilder> b = new SimpleSolverBuilder();
//    app->OptimizeNLP(new TNLPAdapter(GetRawPtr(tnlp)), b);
//
// ONE solver instance serves the whole process, so the structure and the
// symbolic analyses survive across continuation levels.
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
