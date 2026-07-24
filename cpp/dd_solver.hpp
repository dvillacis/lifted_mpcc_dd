// The domain-decomposition arrowhead linear solver for the lifted TV-MPCC,
// injected into IPOPT via a custom AlgorithmBuilder.
//
// IPOPT hands a linear solver the symmetric augmented KKT matrix (lower triangle,
// triplet, 1-based, duplicate entries to be summed) with its own δ_w/δ_c
// regularization ALREADY APPLIED, asks for a solve, and queries the inertia
// (#negative eigenvalues) so its inertia-correction loop can pick δ_w/δ_c. So
// IPOPT owns globalization — filter line search, μ-homotopy, restoration — and we
// own only the linear algebra.
//
// This mirrors the Python lab in ../dd_kkt.py, the reference implementation: same
// permutation, same local Schur complements, same Haynsworth inertia.
//
// IPOPT's augmented KKT for the uniform 2D formulation is dim = 17m+1, ordered
// [x | s | λ_c | λ_d] with
//   x   : u,qx,qy,r,δ,θ,α          (6m+1)
//   s   : slacks for hr,hd,comp     (3m)
//   λ_c : h1,h2x,h2y,h3x,h3y        (5m)
//   λ_d : hr,hd,comp                (3m)
// Cut the image into nsub×nsub tiles. Slacks and multipliers inherit their row's
// pixel and rows are never duplicated, so ONLY primal columns can be complicating:
// the first u past each cut, the last q before it, plus the global α (dense — it
// is in every h1). Permuting those p columns to the border makes the matrix
// bordered block-diagonal (arrowhead):
//
//     [ W_1          B_1ᵀ ]
//     [     ⋱        ⋮    ]        S = C − Σ_k B_k W_k⁻¹ B_kᵀ
//     [        W_K   B_Kᵀ ]        Δx_k = W_k⁻¹(r_k − B_kᵀ Δy)
//     [ B_1 ⋯  B_K   C    ]
//
// Note this is the PERMUTATION route, not Lueg's duplicate-and-link reformulation:
// no copies are introduced, so the border carries real Jacobian/Hessian entries
// and the corner C is nonzero. The Schur elimination and the Haynsworth inertia
// identity are unaffected.
//
// The interface system is solved DIRECTLY by default (MA57 on the assembled
// sparse S) — the measured winner at every size. An OPT-IN preconditioned
// conjugate-gradient interface solve (--interface cg) is provided as the
// distributed-DD prototype, with the two preconditioners of Lueg's paper as
// implemented by the Python reference (../dd_kkt.py's make_preconditioner):
//
//   BJ  (Lueg eq. 17)    per subdomain invert the LOCAL Schur block
//                        S_k = −B̄_k W_k⁻¹ B̄_kᵀ on its border indices, apply
//                        additively. Standing measured expectation: it does NOT
//                        converge here (the Python lab never got it below tol) —
//                        it is an A/B lever, not a recommendation.
//   ASd (Lueg eq. 20–21) the same blocks but with the diagonal replaced by the
//                        ASSEMBLED diag(S) — the one quantity shared across
//                        ranks. The one distributed preconditioner the lab
//                        measured to actually converge.
//   jacobi               P = diag(|S|), the trivial baseline — and the measured
//                        1D winner (CG terminates in ≤ p steps there anyway).
//
// As in ../dd_kkt.solve_interface, the dense α row/column is peeled off as a
// scalar Schur step BEFORE the Krylov solve (α is in every N_k, so it breaks the
// adjacency-banded structure the distributed preconditioners rely on); CG then
// runs on the field block S_ff. --no-alpha-peel iterates on the full S instead.
//
// On TILE partitions the promoted corner-dual pairs are exactly the negative
// eigenvalues of S (structural, measured), so plain CG can never run there.
// The DUAL PEEL (config_peel) eliminates those entries together with α as one
// dense Schur block T before the Krylov solve, leaving the SPD field block
// S_ff — the FETI-DP/BDDC-style corner treatment ported from ../cpp (vii).
//
// Either way S is STILL assembled and MA57-factorized: its pivot signs answer
// IPOPT's Haynsworth inertia query, which also gives an exact admissibility
// gate (CG runs only when every negative of S is a peeled dual direction) and
// a free direct fallback whenever CG fails. So CG is a solve-strategy swap,
// not a factorization-avoidance scheme. The remaining history
// (Chebyshev/Richardson, matrix-free S_ff variants) lives in ../cpp/README.md.
//
// Inertia comes from LDLᵀ pivot signs via Haynsworth additivity,
// In(A) = Σ_k In(W_k) + In(S), so IPOPT's inertia query is answered WITHOUT ever
// factorizing the full KKT. Eigenvalues would not do: with Σ ~ z²/μ these matrices
// reach ‖A‖ ~ 1e18 and eigenvalue signs become noise.
//
// Environment switches:
//   DD_CHECK=1   verify every solve + the inertia against MA57 on the full matrix
//   DD_TIME=1    print the phase timers every 25 factorizations
//   DD_DEBUG=1   partition summary, singular-block reports, refinement warnings
#ifndef DD_SOLVER_HPP
#define DD_SOLVER_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef _OPENMP
#include <omp.h>
#endif

// Block LDLᵀ backend: MA57 (the validated reference) by default; -DDD_USE_MA97
// swaps in the API-identical MA97 wrapper for machines whose HSL install lacks
// MA57 (build_linux.sh selects it). Member names below keep their historical
// ma57_ prefixes — they name the ROLE (the block factorization), not the code.
//
// THREADING under MA97 — SETTLED on the HPC 2026-07-23 with ma97_smoke_par
// (the standalone concurrency stress test; run it on any new machine):
//   * concurrent ma97_factor CRASHES — free() inside
//     __hsl_ma97_double_MOD_rfact_block on a worker thread (gdb), i.e. heap
//     corruption in MA97's internal front memory management, module-global
//     state. NOT curable from outside: OMP_STACKSIZE, ulimit -s,
//     MKL_THREADING_LAYER=SEQUENTIAL/GNU and MC64-off were all measured to
//     still segfault. Don't re-try env fixes; a rebuilt serial (no-OpenMP)
//     libhsl_ma97 or MA57 are the only routes to concurrent factorization.
//   * concurrent ma97_solve is SAFE — phase B of the stress test passes at
//     8 threads with exact residuals/inertia.
// Hence the SPLIT below, the built-in MA97 threading model: the FACTORIZE
// loop is serial under DD_USE_MA97, the two BACKSOLVE loops stay parallel.
// The backsolves are 97.5% of the S_k phase (settled ../cpp measurement), so
// this keeps most of the across-block win. MA57 (serial inside, validated
// concurrent for years) keeps all three loops parallel.
#ifdef DD_USE_MA97
#include "ma97_block.hpp"
using SymBlock = Ma97Block;
#else
#include "ma57_block.hpp"
using SymBlock = Ma57Block;
#endif

#include "IpAlgBuilder.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTSymLinearSolver.hpp"

namespace Ipopt {

class DDArrowheadSolver : public SparseSymLinearSolverInterface {
public:
   using SpMat = Eigen::SparseMatrix<double>;
   using Trip = Eigen::Triplet<double>;

   // The driver sets the geometry before OptimizeNLP (IPOPT options cannot carry
   // it). Static because AlgorithmBuilder constructs the solver itself.
   static void config(int N, int nsub) { cfgN() = N; cfgK() = nsub; cfgOwner().clear(); }
   static int& cfgN() { static int v = 0; return v; }
   static int& cfgK() { static int v = 2; return v; }

   // Formulation-agnostic route: the driver hands in the subdomain label of every
   // KKT index (−1 = border) and the number of subdomains, and NOTHING else here
   // knows what the problem is. Everything past the partition — triplet routing,
   // W_k/S factorization, the Schur assembly, the Haynsworth inertia — works off
   // owner_/ypos_/lpos_ alone. Used by the staggered 1D/2D drivers, where u and
   // the edge/cell blocks have DIFFERENT lengths and the built-in uniform layout
   // cannot apply. Leave it empty (the default) to keep the built-in 2D geometry.
   static void config_owner(std::vector<int> owner, int nsub) {
      cfgOwner() = std::move(owner);
      cfgK() = nsub;
   }
   static std::vector<int>& cfgOwner() { static std::vector<int> v; return v; }

   // Dump the arrowhead (matrix + owner map + assembled S) after every successful
   // factorization, overwriting, so the file ends up holding the LAST Newton step
   // actually taken. Read by plot_1d.py / plot_2d.py, which draw it with the
   // Python's own plot_arrowhead. Dumping rather than reconstructing in Python
   // matters: this is the matrix IPOPT handed us, with ITS δ_w/δ_c already applied
   // and our S as actually assembled — a reconstruction would have to guess both.
   static void config_dump_arrow(const std::string& path) { cfgDumpArrow() = path; }
   static std::string& cfgDumpArrow() { static std::string v; return v; }

   // ---- interface-solve configuration ------------------------------------
   // DIRECT (default) = the MA57 back-solve of the assembled S. CG = the
   // preconditioned conjugate-gradient prototype described in the header.
   // MINRES = signed-MA57-preconditioned MINRES on the FULL (indefinite) S —
   // no SPD gate, no peel; see config_minres below.
   enum { IFACE_DIRECT = 0, IFACE_CG = 1, IFACE_MINRES = 2 };
   enum { PRECOND_JACOBI = 0, PRECOND_BJ = 1, PRECOND_ASD = 2 };
   // How CG applies the interface operator. ASSEMBLED (default) reuses the S
   // already formed for the inertia — one sparse matvec per CG iteration; the
   // Krylov statistics (its/solve, convergence, preconditioner quality) measure
   // the SAME operator, so the distributed-viability numbers are unchanged.
   // MATFREE re-derives S·y through K subdomain backsolves per iteration — the
   // faithful simulation of the distributed cost profile (profiled ~10× the
   // schur phase's work per factorization, redone every apply).
   enum { APPLY_ASSEMBLED = 0, APPLY_MATFREE = 1 };
   static void config_cg_apply(int mode) { cfgCgApply() = mode; }
   static int& cfgCgApply() { static int v = APPLY_ASSEMBLED; return v; }
   static void config_interface(int mode, int precond, bool alpha_peel,
                                double tol, int maxit) {
      cfgIface() = mode; cfgPrecond() = precond; cfgAlphaPeel() = alpha_peel;
      cfgCgTol() = tol; cfgCgMaxit() = maxit;
   }
   // The KKT index of the scalar α (the drivers' oa), needed by the α peel. The
   // solver is otherwise formulation-agnostic and cannot find α on its own.
   static void config_alpha(int kkt_index) { cfgAlpha() = kkt_index; }
   // DUAL PEEL: peel every border position with KKT index ≥ n_primal — the
   // promoted corner-dual pairs of a TILE partition. They are exactly the
   // negative eigenvalues of S (Haynsworth, measured: In(S)_neg equals the
   // promoted count at EVERY solve), so peeling them as a small dense block
   // leaves the SPD field complement S_ff for CG — this is what makes
   // --interface cg usable on tile partitions at all (without it the SPD gate
   // skips every solve there). n_primal ≥ dim (the default) disables it; a
   // border with no dual indices (1D, strips) makes it a structural no-op.
   static void config_peel(int n_primal) { cfgPeelPrimal() = n_primal; }
   static int& cfgPeelPrimal() { static int v = 1 << 30; return v; }
   // SIGNED-MA57 MINRES (--interface minres): preconditioned MINRES on the FULL
   // indefinite S — no SPD gate, no peel. The preconditioner is M = L|D|Lᵀ from
   // a SNAPSHOT MA57 factorization of S (MC64 off, so the JOB=2/3/4 partial
   // solves compose exactly), rebuilt every `lag` factorizations. At lag=1 the
   // preconditioned spectrum is exactly {−1,+1} (Gill–Murray–Ponceleón–Saunders
   // 1992: M⁻¹S ~ |D|⁻¹D), so MINRES needs ~2 iterations REGARDLESS of
   // In(S)_neg — indefiniteness stops being an obstacle. lag>1 is THE staleness
   // experiment: its/solve vs preconditioner age measures whether the serial
   // S factorization (the measured Amdahl floor of the DD solver) could be
   // amortized across Newton steps in a distributed setting. The inertia
   // contract is untouched either way: s_ma57_ is still factorized fresh every
   // step and In(S) stays exact.
   static void config_minres(int lag) { cfgMinresLag() = lag < 1 ? 1 : lag; }
   static int& cfgMinresLag() { static int v = 1; return v; }
   static int& cfgIface() { static int v = IFACE_DIRECT; return v; }
   static int& cfgPrecond() { static int v = PRECOND_ASD; return v; }
   static bool& cfgAlphaPeel() { static bool v = true; return v; }
   static double& cfgCgTol() { static double v = 1e-10; return v; }
   static int& cfgCgMaxit() { static int v = 500; return v; }
   static int& cfgAlpha() { static int v = -1; return v; }

   // Interface telemetry, summed across every interface solve of the whole run
   // (static because AlgorithmBuilder may build/destroy the solver per level).
   struct InterfaceStats {
      long solves = 0, iters = 0, nonconv = 0, fallbacks = 0, skipped_indef = 0;
      long sneg_sum = 0, sneg_max = 0, s_spd = 0;   // In(S)_neg distribution
      long peeled = 0;                              // dual entries peeled (per level)
      long cache_builds = 0;   // peel Z/T cache rebuilds (one per CG-mode fact.)
      long skipped_dead = 0;   // solves skipped after a failure on the SAME matrix
      // ---- signed-MA57 MINRES telemetry (--interface minres) ----
      long sgn_builds = 0;     // preconditioner snapshot factorizations
      long sgn_fail = 0;       // snapshot self-check failures (precond disabled)
      long sgn_unavail = 0;    // MINRES solves skipped: no valid preconditioner
      // The lag experiment's raw data: MINRES cost by preconditioner age
      // (= factorizations since the snapshot; age 0 is a fresh factorization).
      static constexpr int MAXAGE = 64;
      long age_cnt[MAXAGE] = {};   // attempted MINRES solves at this age
      long age_its[MAXAGE] = {};   // their summed iterations
      long age_fb[MAXAGE] = {};    // their direct fallbacks
      double wall = 0.0;
   };
   static void reset_interface_stats() { s_if() = InterfaceStats(); }
   static InterfaceStats interface_stats() { return s_if(); }
   static InterfaceStats& s_if() { static InterfaceStats v; return v; }

   // OpenMP threads available to the subdomain loops (1 when built without OMP).
   static int omp_threads() {
#ifdef _OPENMP
      return omp_get_max_threads();
#else
      return 1;
#endif
   }

   bool InitializeImpl(const OptionsList&, const std::string&) override { return true; }
   EMatrixFormat MatrixFormat() const override { return Triplet_Format; }
   bool ProvidesInertia() const override { return true; }
   bool IncreaseQuality() override { return false; }
   Index NumberOfNegEVals() const override { return nneg_; }

   ESymSolverStatus InitializeStructure(Index dim, Index nnz,
                                        const Index* ia, const Index* ja) override {
      // #2 (perf audit): the SAME solver object is reused across continuation
      // levels (see CustomSolverBuilder), and every level hands over the SAME
      // KKT structure. If the incoming pattern and the partition config match
      // what was frozen, skip the rebuild — partition, triplet routing, the
      // B/C slot maps and all K+1 MA57 symbolic analyses carry over, and only
      // the per-level snapshot below runs. The guard is a FULL comparison
      // (dims, ia/ja, N/K, the injected owner vector), so a genuinely new
      // structure can never silently reuse a stale layout.
      const bool same_structure =
         frozen_ && dim == dim_ && nnz == nnz_ &&
         cfgN() == frzN_ && cfgK() == frzK_ && cfgOwner() == frzOwnerCfg_ &&
         [&] {
            for (Index k = 0; k < nnz; ++k)
               if (ia[k] - 1 != irow_[k] || ja[k] - 1 != jcol_[k]) return false;
            return true;
         }();
      if (!same_structure) {
      dim_ = dim; nnz_ = nnz;
      N_ = cfgN(); K_ = cfgK();
      m_ = N_ * N_;
      frozen_ = false;
      const bool injected = !cfgOwner().empty();
      if (injected) {
         if ((Index)cfgOwner().size() != dim_) {
            std::cerr << "[dd] injected owner has " << cfgOwner().size()
                      << " entries, KKT dim is " << dim_ << "\n";
            return SYMSOLVER_FATAL_ERROR;
         }
      } else if (17 * m_ + 1 != dim_) {
         std::cerr << "[dd] dim " << dim_ << " != 17m+1 for N=" << N_ << "\n";
         return SYMSOLVER_FATAL_ERROR;
      }
      irow_.resize(nnz); jcol_.resize(nnz);
      for (Index k = 0; k < nnz; ++k) { irow_[k] = ia[k] - 1; jcol_[k] = ja[k] - 1; }
      if (injected) {
         owner_ = cfgOwner();
         nsub_ = K_;
         std::vector<int> ycols;
         for (Index i = 0; i < dim_; ++i) if (owner_[i] < 0) ycols.push_back((int)i);
         finalize_partition(ycols);
      } else {
         build_partition();
      }
      if (!route_triplets()) return SYMSOLVER_FATAL_ERROR;
      frzN_ = cfgN(); frzK_ = cfgK(); frzOwnerCfg_ = cfgOwner();
      frozen_ = true;
      }                        // end !same_structure — per-level snapshot below
      vals_.assign(nnz_, 0.0);
      // Interface-solve mode, snapshotted from the statics. The peel set holds
      // the border positions eliminated by a dense Schur step before CG: the
      // dense α column (α peel) and the promoted corner-dual entries (dual
      // peel, KKT index ≥ config_peel's n_primal). kept_ lists the positions
      // CG actually iterates on — the field block S_ff.
      iface_ = cfgIface(); precond_ = cfgPrecond(); cg_apply_ = cfgCgApply();
      cg_tol_ = cfgCgTol(); cg_maxit_ = cfgCgMaxit();
      minres_lag_ = cfgMinresLag();
      sgn_valid_ = false;      // new level ⇒ force a fresh snapshot (age 0)
      sgn_age_ = 0;
      peel_.clear();
      n_peel_dual_ = 0;
      if (iface_ == IFACE_CG) {
         const int nprim = cfgPeelPrimal();
         for (Index i = 0; i < dim_; ++i)
            if (owner_[i] == -1 && (int)i >= nprim) {
               peel_.push_back(ypos_[i]);
               ++n_peel_dual_;
            }
         const int akkt = cfgAlpha();
         if (cfgAlphaPeel() && akkt >= 0 && akkt < dim_ && owner_[akkt] == -1)
            peel_.push_back(ypos_[akkt]);
         std::sort(peel_.begin(), peel_.end());
      }
      kept_.clear();
      keptpos_.assign(p_, -1);
      peelpos_.assign(p_, -1);
      for (size_t j = 0; j < peel_.size(); ++j) peelpos_[peel_[j]] = (int)j;
      for (int j = 0; j < p_; ++j)
         if (peelpos_[j] < 0) { keptpos_[j] = (int)kept_.size(); kept_.push_back(j); }
      s_if().peeled = n_peel_dual_;
      if (std::getenv("DD_DEBUG"))
         std::cerr << "[dd] " << (!cfgOwner().empty() ? "injected owner" : "2D image layout")
                   << "  dim=" << dim_ << " N=" << N_ << " k=" << K_
                   << " n_sub=" << nsub_ << " p=" << p_
                   << " max dim W_k=" << max_dimk_ << "\n";
      return SYMSOLVER_SUCCESS;
   }

   Number* GetValuesArrayPtr() override { return vals_.data(); }

   ESymSolverStatus MultiSolve(bool new_matrix, const Index*, const Index*,
                               Index nrhs, Number* rhs_vals,
                               bool check_NegEVals, Index numberOfNegEVals) override {
      if (new_matrix) {
         ESymSolverStatus st = factorize();
         if (st != SYMSOLVER_SUCCESS) return st;
      }
      if (check_NegEVals && nneg_ != numberOfNegEVals) return SYMSOLVER_WRONG_INERTIA;
      // #7 (perf audit): one env lookup per process, not two per RHS.
      static const bool dd_check = std::getenv("DD_CHECK") != nullptr;
      for (Index c = 0; c < nrhs; ++c) {
         Number* b = rhs_vals + (size_t)c * dim_;
         std::vector<double> saved;
         if (dd_check) saved.assign(b, b + dim_);
         // ALWAYS solve through best-effort refinement against the TRUE
         // triplets. Even when no block was flagged singular, near-singular
         // pivots (e.g. IPOPT's own δ_c = 1e-8·μ^¼ on the corner-dual
         // directions) lose ~10 digits through the local Schur assembly —
         // measured rel-res ~0.1 on unrefined solves, ~1e-12 refined.
         solve_refined(b);
         if (dd_check) check_solve(saved, b);
      }
      return SYMSOLVER_SUCCESS;
   }

private:
   // y = A_true · x from the triplets IPOPT gave us (its δ_w/δ_c included).
   void matvec(const double* x, double* y) const {
      std::fill(y, y + dim_, 0.0);
      for (Index t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         y[i] += vals_[t] * x[j];
         if (i != j) y[j] += vals_[t] * x[i];
      }
   }

   // Best-effort iterative refinement against the true triplets. Keeps the
   // iterate with the smallest residual and ALWAYS returns it — mirroring what
   // a monolithic MA57/MUMPS solve does at a nasty iterate (return the
   // factorization's answer, let IPOPT's globalization cope). Refinement here
   // only ever improves on the raw arrowhead solve; at well-conditioned
   // iterates it reaches ~1e-12 in one or two sweeps.
   void solve_refined(Number* b) {
      const auto tic = std::chrono::steady_clock::now();
      // Member scratch (#4, perf audit): five dim-sized buffers per RHS, times
      // up to 6 solve_one calls, times thousands of RHS per run — allocator
      // traffic with fixed sizes. assign()/resize() reuse capacity after the
      // first call. All of this is serial (MultiSolve is), so members are safe.
      sr_b0_.assign(b, b + dim_);
      sr_x_.assign(b, b + dim_);
      sr_r_.resize(dim_);
      sr_Ax_.resize(dim_);
      std::vector<double>&b0 = sr_b0_, &x = sr_x_, &r = sr_r_, &Ax = sr_Ax_;
      std::vector<double>& best = sr_best_;
      solve_one(x.data());
      double bn = 0.0;
      for (Index i = 0; i < dim_; ++i) bn += b0[i] * b0[i];
      bn = std::sqrt(std::max(bn, 1e-300));
      double best_res = std::numeric_limits<double>::infinity();
      double prev_res = std::numeric_limits<double>::infinity();
      for (int it = 0; it < 5; ++it) {
         matvec(x.data(), Ax.data());
         double rn = 0.0;
         for (Index i = 0; i < dim_; ++i) { r[i] = b0[i] - Ax[i]; rn += r[i] * r[i]; }
         const double relres = std::sqrt(rn) / bn;
         if (relres < best_res) { best_res = relres; best = x; }
         if (relres <= 1e-11 || relres > 1e3 * best_res) break;
         // CG mode only: a refinement sweep costs a FULL interface CG solve
         // (~p/3 apply_S matvecs), and with cg_tol ~1e-10 the 1e-11 target above
         // is often unreachable — without this break the loop burns all 5 sweeps
         // for no digits (measured: 4.1 interface solves per Newton step on
         // mariposa N=32). Stop once a sweep stops buying at least 2×. The
         // DIRECT path is deliberately untouched: its sweeps cost a cheap
         // back-solve and every validated iteration trajectory stays identical.
         if (iface_ == IFACE_CG && it > 0 && relres > 0.5 * prev_res) break;
         prev_res = relres;
         solve_one(r.data());
         for (Index i = 0; i < dim_; ++i) x[i] += r[i];
      }
      if (best_res > 1e-6 && std::getenv("DD_DEBUG"))
         std::cerr << "[dd] refinement best-effort rel-res " << best_res << "\n";
      std::copy(best.begin(), best.end(), b);
      t_solve_ += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - tic).count();
   }

   // DD_CHECK=1: verify the arrowhead solve against a reference solve of the SAME
   // matrix, and the distributed inertia against the reference's. Small N only —
   // this is the "does the decomposition reproduce the monolithic Newton step"
   // gate, the C++ twin of ../dd_probe.py's rel-err column.
   void check_solve(const std::vector<double>& rhs, const Number* got) {
      // Reference = MA57 on the WHOLE matrix. Deliberately not an Eigen dense
      // factorization: Eigen::LDLT is a pivoted Cholesky for semi-definite
      // matrices, not Bunch–Kaufman, and on these indefinite KKTs it reported
      // 485–491 negative pivots where the true count was 512. A wrong reference
      // is worse than no reference.
      if (!ref_ma57_) {
         std::vector<int> irn(nnz_), jcn(nnz_);
         for (Index t = 0; t < nnz_; ++t) { irn[t] = irow_[t] + 1; jcn[t] = jcol_[t] + 1; }
         ref_ma57_.reset(new SymBlock());
         if (!ref_ma57_->analyze(dim_, irn, jcn)) {
            std::cerr << "[dd-check] reference full-matrix analysis failed\n";
            ref_ma57_.reset();
            return;
         }
      }
      std::copy(vals_.begin(), vals_.end(), ref_ma57_->values());
      if (!ref_ma57_->factorize()) {
         std::cerr << "[dd-check] reference full-matrix factorization failed\n";
         return;
      }
      std::vector<double> ref = rhs;
      ref_ma57_->solve(ref.data(), 1);

      // residual ‖A·got − b‖ / ‖b‖, computed straight from the triplets
      std::vector<double> Ax(dim_, 0.0);
      matvec(got, Ax.data());
      double dn = 0, rn = 0, resn = 0, bn = 0;
      for (Index i = 0; i < dim_; ++i) {
         dn += (got[i] - ref[i]) * (got[i] - ref[i]);
         rn += ref[i] * ref[i];
         resn += (Ax[i] - rhs[i]) * (Ax[i] - rhs[i]);
         bn += rhs[i] * rhs[i];
      }
      // the reference's own residual — the fair yardstick at nasty iterates
      std::vector<double> Ar(dim_, 0.0);
      matvec(ref.data(), Ar.data());
      double refres = 0.0;
      for (Index i = 0; i < dim_; ++i)
         refres += (Ar[i] - rhs[i]) * (Ar[i] - rhs[i]);
      const int ref_neg = ref_ma57_->negative_eigenvalues();
      std::cerr << "[dd-check] rel-err " << std::sqrt(dn / std::max(rn, 1e-300))
                << "  rel-res " << std::sqrt(resn / std::max(bn, 1e-300))
                << "  (ref " << std::sqrt(refres / std::max(bn, 1e-300)) << ")"
                << "  inertia dd=" << nneg_ << " ref(full)=" << ref_neg
                << (nneg_ == ref_neg ? "  MATCH" : "  MISMATCH") << "\n";
   }

   // ---- geometry ---------------------------------------------------------
   static std::vector<std::pair<int, int>> tile_bounds(int n, int k) {
      std::vector<std::pair<int, int>> out;
      int base = n / k, rem = n % k, lo = 0;
      for (int c = 0; c < k; ++c) {
         int hi = lo + base + (c < rem ? 1 : 0);
         out.push_back({lo, hi});
         lo = hi;
      }
      return out;
   }

   // Label every KKT index with its subdomain (−1 = border), mirroring
   // ../dd_structure.py. Complicating rule: u_{ij} if its left/up neighbour is in
   // another tile; qx_{ij} (j ≤ N−2) if its right neighbour is; qy_{ij} (i ≤ N−2)
   // if its down neighbour is; α always.
   void build_partition() {
      const int m = m_, N = N_;
      nsub_ = K_ * K_;
      auto rb = tile_bounds(N, K_), cb = tile_bounds(N, K_);
      std::vector<int> trow(N), tcol(N);
      for (int b = 0; b < K_; ++b) {
         for (int i = rb[b].first; i < rb[b].second; ++i) trow[i] = b;
         for (int j = cb[b].first; j < cb[b].second; ++j) tcol[j] = b;
      }
      sub_of_pixel_.assign(m, 0);
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) sub_of_pixel_[i * N + j] = trow[i] * K_ + tcol[j];

      auto sub = [&](int i, int j) { return sub_of_pixel_[i * N + j]; };
      owner_.assign(dim_, 0);
      // primal blocks u,qx,qy,r,δ,θ then α; then 11m slacks/multipliers
      for (int b = 0; b < 6; ++b)
         for (int q = 0; q < m; ++q) owner_[b * m + q] = sub_of_pixel_[q];
      for (int q = 0; q < 11 * m; ++q) owner_[6 * m + 1 + q] = sub_of_pixel_[q % m];

      std::vector<int> ycols;
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            const int p = i * N + j, s = sub(i, j);
            bool ucomp = (j >= 1 && sub(i, j - 1) != s) || (i >= 1 && sub(i - 1, j) != s);
            if (ucomp) ycols.push_back(0 * m + p);
            const bool qxc = (j <= N - 2 && sub(i, j + 1) != s);
            const bool qyc = (i <= N - 2 && sub(i + 1, j) != s);
            if (qxc) ycols.push_back(1 * m + p);
            if (qyc) ycols.push_back(2 * m + p);
            // Cut-corner pixel: BOTH qx and qy left for the border, so inside
            // the block the dual pair (λ_h3x, λ_h3y) keeps only its δ/θ
            // couplings — rank-1 whenever δ_pixel ≈ 0 (measured by SVD of a
            // dumped block: the null vector was exactly this pair). Promote the
            // pair to the border as well: the block stays full rank with NO
            // artificial shift, and the Haynsworth inertia stays exact. Costs 2
            // border entries per cross corner ⇒ p grows by 2(k−1)².
            if (qxc && qyc) {
               ycols.push_back(9 * m + 1 + 3 * m + p);   // λ_h3x
               ycols.push_back(9 * m + 1 + 4 * m + p);   // λ_h3y
            }
         }
      std::sort(ycols.begin(), ycols.end());
      ycols.push_back(6 * m);            // α last (see finalize_partition)
      finalize_partition(ycols);
   }

   // Common tail of both partition routes: fix the border ordering, number the
   // local positions inside each W_k, and size the blocks.
   //
   // ``ycols`` is the border list IN THE ORDER IT WILL INDEX S. That order is
   // load-bearing for reproducibility, not just cosmetics: it decides S's sparsity
   // pattern and hence MA57's pivoting. The 2D builder deliberately appends α
   // AFTER sorting (the promoted corner-dual indices 9m+1+3m+p are larger than
   // 6m), so do NOT "simplify" it into one global sort — that would silently
   // perturb every validated 2D number. The injected route sorts everything,
   // which makes its border order coincide with Python's
   // ``np.flatnonzero(owner == -1)``.
   void finalize_partition(const std::vector<int>& ycols) {
      p_ = (int)ycols.size();

      ypos_.assign(dim_, -1);
      for (int j = 0; j < p_; ++j) { ypos_[ycols[j]] = j; owner_[ycols[j]] = -1; }

      // position of each local index inside its own W_k
      dimk_.assign(nsub_, 0);
      lpos_.assign(dim_, -1);
      for (Index i = 0; i < dim_; ++i)
         if (owner_[i] >= 0) lpos_[i] = dimk_[owner_[i]]++;
      max_dimk_ = 0;
      for (int d : dimk_) max_dimk_ = std::max(max_dimk_, d);
   }

   // Precompute where every triplet lands, which y-entries each subdomain sees
   // (the index list that is Lueg's selection matrix N_k), and — because the
   // sparsity of every W_k is fixed for the whole solve — run MA57's SYMBOLIC
   // analysis once here. Only values change between Newton steps.
   bool route_triplets() {
      seen_.assign(nsub_, std::vector<char>(p_, 0));
      wtrip_.assign(nsub_, {});
      for (Index t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const int oi = owner_[i], oj = owner_[j];
         if (oi >= 0 && oj >= 0 && oi != oj) {
            // Invariant violation (an owner map coupling two subdomains).
            // Return false → SYMSOLVER_FATAL_ERROR, not a throw: IPOPT catches
            // only IpoptException, so a std:: exception aborts the process.
            std::cerr << "[dd] partition leak at (" << i << "," << j << ")\n";
            return false;
         }
         if (oi >= 0 && oj >= 0) wtrip_[oi].push_back((int)t);
         if (oi < 0 && oj >= 0) seen_[oj][ypos_[i]] = 1;
         if (oj < 0 && oi >= 0) seen_[oi][ypos_[j]] = 1;
      }
      Nk_.assign(nsub_, {});
      ykrow_.assign(nsub_, std::vector<int>(p_, -1));
      for (int k = 0; k < nsub_; ++k)
         for (int j = 0; j < p_; ++j)
            if (seen_[k][j]) { ykrow_[k][j] = (int)Nk_[k].size(); Nk_[k].push_back(j); }

      // MA57 wants ONE triangle in 1-based coordinate form. IPOPT already hands
      // us the lower triangle, so the routed triplets go through unchanged.
      ma57_.clear();
      ma57_.resize(nsub_);
      for (int k = 0; k < nsub_; ++k) {
         std::vector<int> irn, jcn;
         irn.reserve(wtrip_[k].size());
         jcn.reserve(wtrip_[k].size());
         for (int t : wtrip_[k]) {
            irn.push_back(lpos_[irow_[t]] + 1);
            jcn.push_back(lpos_[jcol_[t]] + 1);
         }
         ma57_[k].reset(new SymBlock());
         if (!ma57_[k]->analyze(dimk_[k], irn, jcn)) {
            std::cerr << "[dd] block analysis failed for W_" << k
                      << " (status=" << ma57_[k]->status() << ")\n";
            return false;
         }
      }

      // #1 (2026-07-22 perf audit): FREEZE the border-coupling structure. The
      // KKT sparsity is constant for the whole solve, so the per-factorization
      // work of re-routing every triplet, rebuilding tB triplet vectors and
      // re-running setFromTriplets on all K B_k (sort + allocate, identical
      // result every time) is pure waste — measured as Eigen assignment/alloc
      // frames in the profile. Here, ONCE: build each B_k's compressed pattern
      // and map every border-coupling triplet to its valuePtr slot (duplicates
      // accumulate in triplet order, matching setFromTriplets' summation), and
      // record every border-border triplet's (row, col) border positions for
      // the #6 sparse-S slot maps below. factorize() then only refreshes values.
      B_.assign(nsub_, SpMat());
      Sk_.assign(nsub_, Eigen::MatrixXd());
      btrip_t_.assign(nsub_, {});
      btrip_slot_.assign(nsub_, {});
      ctrip_t_.clear();
      // (row, col) border positions per ctrip entry — consumed by #6, transient
      std::vector<std::pair<int, int>> cpairs;
      {
         std::vector<std::vector<Trip>> tB(nsub_);
         // (row, col, triplet) per subdomain, in encounter order
         std::vector<std::vector<std::array<int, 3>>> ents(nsub_);
         for (Index t = 0; t < nnz_; ++t) {
            const int i = irow_[t], j = jcol_[t];
            const int oi = owner_[i], oj = owner_[j];
            if (oi >= 0 && oj >= 0) continue;
            if (oi < 0 && oj < 0) {
               // corner-block coupling S(ypos_[i], ypos_[j]) + its mirror
               ctrip_t_.push_back((int)t);
               cpairs.emplace_back(ypos_[i], ypos_[j]);
            } else if (oi < 0) {                    // border row, local col
               tB[oj].push_back(Trip(ykrow_[oj][ypos_[i]], lpos_[j], 0.0));
               ents[oj].push_back({ykrow_[oj][ypos_[i]], lpos_[j], (int)t});
            } else {                                // local row, border col
               tB[oi].push_back(Trip(ykrow_[oi][ypos_[j]], lpos_[i], 0.0));
               ents[oi].push_back({ykrow_[oi][ypos_[j]], lpos_[i], (int)t});
            }
         }
         for (int k = 0; k < nsub_; ++k) {
            B_[k].resize((int)Nk_[k].size(), dimk_[k]);
            B_[k].setFromTriplets(tB[k].begin(), tB[k].end());
            B_[k].makeCompressed();
            const int* outer = B_[k].outerIndexPtr();
            const int* inner = B_[k].innerIndexPtr();
            for (const auto& e : ents[k]) {
               const int r = e[0], c = e[1];
               const int* lo = inner + outer[c];
               const int* hi = inner + outer[c + 1];
               const int slot = (int)(std::lower_bound(lo, hi, r) - inner);
               btrip_slot_[k].push_back(slot);
               btrip_t_[k].push_back(e[2]);
            }
         }
      }

      // The interface matrix S is SPARSE by subdomain adjacency (the vault's
      // §3.6/§7.5 point): S(i,j) ≠ 0 only if some subdomain sees both y_i and
      // y_j, so its pattern is the union of the N_k×N_k cliques — and every
      // entry of the corner block C lies inside some clique too (both endpoints
      // of a KKT coupling are seen by the row's subdomain). Handing MA57 the
      // dense triangle instead was measured to dominate the whole solve at
      // N=64 (11 s per 75 factorizations at k=8).
      {
         std::vector<std::pair<int, int>> pat;
         for (int k = 0; k < nsub_; ++k)
            for (int a : Nk_[k])
               for (int b : Nk_[k])
                  if (a >= b) pat.emplace_back(a, b);
         for (int i = 0; i < p_; ++i) pat.emplace_back(i, i);   // full diagonal
         std::sort(pat.begin(), pat.end());
         pat.erase(std::unique(pat.begin(), pat.end()), pat.end());
         spat_.assign(pat.begin(), pat.end());
         std::vector<int> irn, jcn;
         irn.reserve(spat_.size());
         jcn.reserve(spat_.size());
         for (const auto& e : spat_) { irn.push_back(e.first + 1); jcn.push_back(e.second + 1); }
         s_ma57_.reset(new SymBlock());
         if (!s_ma57_->analyze(p_, irn, jcn)) {
            std::cerr << "[dd] block analysis failed for S (status="
                      << s_ma57_->status() << ")\n";
            return false;
         }
         if (std::getenv("DD_DEBUG"))
            std::cerr << "[dd] S pattern: " << spat_.size() << " nnz of "
                      << (size_t)p_ * (p_ + 1) / 2 << " dense ("
                      << 100.0 * spat_.size() / ((size_t)p_ * (p_ + 1) / 2)
                      << "%)\n";
      }

      // #6 (2026-07-22 perf audit): the assembled S lives in the SPARSE Ssp_
      // (full symmetric pattern = spat_ ∪ its mirror), never in a dense p×p
      // array. The dense C_/S_ pair cost O(p²) memory plus an O(p²) copy and
      // zero per factorization — ~470 MB EACH at N=128 k=16, where nnz(S) is
      // ~1.5% — to carry structural zeros no consumer reads. Frozen here, once:
      // the pattern and a value slot for every writer/reader — the C triplets
      // (cslot1_/cslot2_, entry + mirror, exactly the old cflat pair), the S_k
      // scatter (sk_slot_, col-major per block; within one block every (a,b)
      // hits a DISTINCT slot since Nk is injective, so only the serial k order
      // matters and the accumulation is bit-identical to the dense path), the
      // s_ma57_ lower-triangle refresh (ma57_src_) and the diagonal
      // (sdiag_slot_, always present — spat_ carries the full diagonal). Csp_
      // (the matfree corner apply) gets the same pattern-once treatment.
      {
         std::vector<Trip> tS;
         tS.reserve(2 * spat_.size());
         for (const auto& e : spat_) {
            tS.push_back(Trip(e.first, e.second, 0.0));
            if (e.first != e.second) tS.push_back(Trip(e.second, e.first, 0.0));
         }
         Ssp_.resize(p_, p_);
         Ssp_.setFromTriplets(tS.begin(), tS.end());
         Ssp_.makeCompressed();
         const int* outer = Ssp_.outerIndexPtr();
         const int* inner = Ssp_.innerIndexPtr();
         auto slot_of = [&](int r, int c) -> int {
            const int* lo = inner + outer[c];
            const int* hi = inner + outer[c + 1];
            const int* it = std::lower_bound(lo, hi, r);
            return (it != hi && *it == r) ? (int)(it - inner) : -1;
         };
         bool missing = false;
         ma57_src_.resize(spat_.size());
         for (size_t e = 0; e < spat_.size(); ++e) {
            ma57_src_[e] = slot_of(spat_[e].first, spat_[e].second);
            if (ma57_src_[e] < 0) missing = true;
         }
         sdiag_slot_.resize(p_);
         for (int j = 0; j < p_; ++j) {
            sdiag_slot_[j] = slot_of(j, j);
            if (sdiag_slot_[j] < 0) missing = true;
         }
         cslot1_.resize(cpairs.size());
         cslot2_.resize(cpairs.size());
         for (size_t e = 0; e < cpairs.size(); ++e) {
            cslot1_[e] = slot_of(cpairs[e].first, cpairs[e].second);
            cslot2_[e] = cpairs[e].first != cpairs[e].second
                            ? slot_of(cpairs[e].second, cpairs[e].first) : -1;
            if (cslot1_[e] < 0) missing = true;
         }
         sk_slot_.assign(nsub_, {});
         for (int k = 0; k < nsub_; ++k) {
            const int pk = (int)Nk_[k].size();
            sk_slot_[k].resize((size_t)pk * pk);
            for (int b = 0; b < pk; ++b)
               for (int a = 0; a < pk; ++a) {
                  const int s = slot_of(Nk_[k][a], Nk_[k][b]);
                  sk_slot_[k][(size_t)b * pk + a] = s;
                  if (s < 0) missing = true;
               }
         }
         if (missing) {
            // cannot happen while the documented clique-coverage invariant
            // holds (every assembled entry lies inside some N_k×N_k clique)
            std::cerr << "[dd] S pattern does not cover an assembled entry\n";
            return false;
         }
      }
      {
         std::vector<Trip> tC;
         tC.reserve(2 * cpairs.size());
         for (const auto& pr : cpairs) {
            tC.push_back(Trip(pr.first, pr.second, 0.0));
            if (pr.first != pr.second) tC.push_back(Trip(pr.second, pr.first, 0.0));
         }
         Csp_.resize(p_, p_);
         Csp_.setFromTriplets(tC.begin(), tC.end());
         Csp_.makeCompressed();
         const int* outer = Csp_.outerIndexPtr();
         const int* inner = Csp_.innerIndexPtr();
         auto cslot = [&](int r, int c) -> int {
            const int* lo = inner + outer[c];
            const int* hi = inner + outer[c + 1];
            return (int)(std::lower_bound(lo, hi, r) - inner);
         };
         csp1_.resize(cpairs.size());
         csp2_.resize(cpairs.size());
         for (size_t e = 0; e < cpairs.size(); ++e) {
            csp1_[e] = cslot(cpairs[e].first, cpairs[e].second);
            csp2_[e] = cpairs[e].first != cpairs[e].second
                          ? cslot(cpairs[e].second, cpairs[e].first) : -1;
         }
      }
      return true;
   }

   // ---- assemble + factorize --------------------------------------------
   ESymSolverStatus factorize() {
      const int K = nsub_;
      pc_valid_ = false;      // S changes ⇒ rebuild the CG preconditioner/peel cache
      cg_dead_ = false;       // …and give CG a fresh chance on the new operator
      const auto tic0 = std::chrono::steady_clock::now();

      // VALUE refresh only — every structure below was frozen in
      // route_triplets() (#1/#6, perf audit): W_k coordinate arrays, the B_k
      // compressed patterns via their slot maps, and the sparse S — its C
      // couplings and (below) the S_k blocks accumulate straight into
      // Ssp_.valuePtr() in the same order as the dense path did, so every
      // downstream value is bit-identical while the O(p²) zero+copy is gone.
      for (int k = 0; k < K; ++k) {
         double* a = ma57_[k]->values();
         const auto& idx = wtrip_[k];
         for (size_t e = 0; e < idx.size(); ++e) a[e] = vals_[idx[e]];
      }
      {
         double* sv = Ssp_.valuePtr();
         std::fill(sv, sv + Ssp_.nonZeros(), 0.0);
         for (size_t e = 0; e < ctrip_t_.size(); ++e) {
            const double v = vals_[ctrip_t_[e]];
            sv[cslot1_[e]] += v;
            if (cslot2_[e] >= 0) sv[cslot2_[e]] += v;
         }
      }
      nneg_ = 0;
      if ((int)okbuf_.size() != K) okbuf_.assign(K, 1);
      std::vector<char>& ok = okbuf_;

      // Local factorizations: independent per k — the embarrassingly-parallel
      // phase the whole scheme is built around. NO artificial shifts here: a
      // rank-deficient block reports SINGULAR to IPOPT exactly like a monolithic
      // MA57 would (a masked deficiency would corrupt the inertia signal that
      // drives IPOPT's δ_w loop — measured, it did). The structurally-forced
      // deficiency (corner dual pairs) is prevented above by border promotion;
      // what remains (θ-gauge columns at r=δ=0) is singular in the FULL matrix
      // too, so SINGULAR is the truthful answer and IPOPT's δ_w cures it.
// FACTORIZATION loop: serial under MA97 — concurrent ma97_factor heap-corrupts
// (gdb 2026-07-23: free() in rfact_block; see the SymBlock threading note).
#if defined(_OPENMP) && !defined(DD_USE_MA97)
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < K; ++k) {
         double* bv = B_[k].valuePtr();
         std::fill(bv, bv + B_[k].nonZeros(), 0.0);
         const auto& tt = btrip_t_[k];
         const auto& ss = btrip_slot_[k];
         for (size_t e = 0; e < tt.size(); ++e) bv[ss[e]] += vals_[tt[e]];
         ok[k] = (ma57_[k]->factorize() && !ma57_[k]->singular()) ? 1 : 0;
      }
      const auto tic1 = std::chrono::steady_clock::now();
      t_factor_ += std::chrono::duration<double>(tic1 - tic0).count();

      for (int k = 0; k < K; ++k) {
         if (!ok[k]) {
            if (std::getenv("DD_DEBUG"))
               std::cerr << "[dd] W_" << k << " block status=" << ma57_[k]->status()
                         << " rank=" << ma57_[k]->rank() << "/" << dimk_[k]
                         << " → SINGULAR\n";
            return SYMSOLVER_SINGULAR;       // IPOPT responds by bumping δ_w
         }
         nneg_ += ma57_[k]->negative_eigenvalues();
      }

      // S = C − Σ_k B̄_k W_k⁻¹ B̄_kᵀ, scattered by the selection lists N_k.
      // MA57CD solves in place, so each B̄_kᵀ block is copied first.
      // MEASURED SPLIT of this loop (N=64 k=8, 8 threads, 75 factorizations,
      // thread-summed CPU): backsolves 97.5%, the B_k·Bt product 2.3%, the Bt
      // densify/alloc 0.16%; the scatter below is 0.2%. So the p_k backsolves
      // ARE the S_k phase; the only lever that could move it is a sparse-RHS
      // block solver (the RHS columns are 0.3–1% dense; MA57CD is dense-RHS).
// BACKSOLVE loop: parallel under BOTH backends — concurrent ma97_solve is
// validated (ma97_smoke_par phase B, 2026-07-23; only concurrent FACTOR crashes).
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < K; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk == 0) { Sk_[k].resize(0, 0); continue; }
         Eigen::MatrixXd Bt = Eigen::MatrixXd(B_[k].transpose());   // dim_k × p_k
         ma57_[k]->solve(Bt.data(), pk);                            // p_k backsolves
         Sk_[k] = -(B_[k] * Bt);
      }
      const auto tic1b = std::chrono::steady_clock::now();
      {
         double* sv = Ssp_.valuePtr();
         for (int k = 0; k < K; ++k) {
            const double* skd = Sk_[k].data();           // col-major pk×pk
            const std::vector<int>& sl = sk_slot_[k];
            for (size_t e = 0; e < sl.size(); ++e) sv[sl[e]] += skd[e];
         }
      }
      const auto tic2 = std::chrono::steady_clock::now();
      t_schur_ += std::chrono::duration<double>(tic2 - tic1).count();
      t_scatter_ += std::chrono::duration<double>(tic2 - tic1b).count();

      // Interface: MA57 as well, not Eigen's dense LDLT. S *should* be positive
      // definite once IPOPT's δ_w has done its job, but during the correction loop
      // it is not — and Eigen::LDLT is a pivoted Cholesky for semi-definite
      // matrices, not Bunch–Kaufman, so its D signs are unreliable exactly then
      // (measured: it reported 485–491 negatives where the true count was 512).
      // Getting In(S) wrong would break the correction loop that is supposed to
      // make S definite in the first place.
      {
         double* a = s_ma57_->values();
         const double* sv = Ssp_.valuePtr();
         for (size_t e = 0; e < spat_.size(); ++e) a[e] = sv[ma57_src_[e]];
      }
      const bool okS = s_ma57_->factorize() && !s_ma57_->singular();
      t_sfact_ += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - tic2).count();
      ++n_fact_;
      if (std::getenv("DD_TIME") && n_fact_ % 25 == 0)
         std::cerr << "[dd-time] n_fact=" << n_fact_
                   << "  factor " << t_factor_ << "s  schur " << t_schur_
                   << "s (scatter " << t_scatter_ << "s)  S-fact " << t_sfact_
                   << "s  solve " << t_solve_ << "s\n";
      if (!okS) {
         if (std::getenv("DD_DEBUG"))
            std::cerr << "[dd] S block status=" << s_ma57_->status()
                      << " rank=" << s_ma57_->rank() << "/" << p_
                      << " → SINGULAR\n";
         return SYMSOLVER_SINGULAR;
      }
      if (iface_ == IFACE_CG && cg_apply_ == APPLY_MATFREE) {
         // The matfree simulation applies the corner block sparsely; refresh
         // its values from the border-border triplets. (Assembled mode needs
         // nothing here — Ssp_ IS the assembled S, always current.)
         double* cv = Csp_.valuePtr();
         std::fill(cv, cv + Csp_.nonZeros(), 0.0);
         for (size_t e = 0; e < ctrip_t_.size(); ++e) {
            const double v = vals_[ctrip_t_[e]];
            cv[csp1_[e]] += v;
            if (csp2_[e] >= 0) cv[csp2_[e]] += v;
         }
      }
      s_neg_ = s_ma57_->negative_eigenvalues();   // In(S)_neg — also the CG SPD gate
      nneg_ += s_neg_;                            // Haynsworth: In(A) = ΣIn(W_k)+In(S)
      // Signed-MA57 MINRES: age the snapshot preconditioner; rebuild every
      // minres_lag_-th factorization (lag=1 ⇒ every one). s_ma57_ above stays
      // fresh regardless — the inertia signal is never lagged.
      if (iface_ == IFACE_MINRES) {
         if (sgn_valid_) ++sgn_age_;
         if (!sgn_valid_ || sgn_age_ >= minres_lag_) build_signed_precond();
      }
      // Overwritten every factorization ⇒ the file holds the last Newton step.
      if (!cfgDumpArrow().empty()) dump_arrowhead(cfgDumpArrow());
      return SYMSOLVER_SUCCESS;
   }

   // Everything plot_1d.py / plot_2d.py need to rebuild the arrowhead view:
   //   dim nnz p nsub
   //   irow jcol val        × nnz   (lower triangle, exactly as IPOPT gave it)
   //   owner                × dim   (subdomain of each index, −1 = border)
   //   S                    × p·p   (row-major, the assembled interface matrix)
   // The border list and the per-subdomain index lists are derivable from owner,
   // so they are not written; ascending index order makes Python's
   // np.flatnonzero(owner == -1) agree with our ypos_ by construction.
   void dump_arrowhead(const std::string& fn) {
      // Guard: S alone is p² doubles, and p ≈ 4N(k−1) on the 2D grid — at N=128
      // k=16 that is a 470 MB write per factorization. This is a small-problem
      // diagnostic (it is what the drivers expose as --save-dd). The dim limit
      // admits the 1D scaling grid (n=2048 ⇒ dim 34804 at ~2 MB/write, p ≤ 63);
      // the p limit is what keeps the 2D S write bounded.
      if (dim_ > 50000 || p_ > 4000) {
         static bool warned = false;
         if (!warned) {
            std::cerr << "[dd] arrowhead dump skipped: dim=" << dim_ << " p=" << p_
                      << " too large (limits 50000 / 4000)\n";
            warned = true;
         }
         return;
      }
      std::ofstream w(fn);
      if (!w) return;
      w << dim_ << " " << nnz_ << " " << p_ << " " << nsub_ << "\n";
      w << std::setprecision(17);
      for (Index t = 0; t < nnz_; ++t)
         w << irow_[t] << " " << jcol_[t] << " " << vals_[t] << "\n";
      for (Index i = 0; i < dim_; ++i) w << owner_[i] << (i + 1 < dim_ ? ' ' : '\n');
      // densify S transiently for the row-major write — small-p only (guarded
      // above), and the byte layout of the file is unchanged
      const Eigen::MatrixXd D = Eigen::MatrixXd(Ssp_);
      for (int i = 0; i < p_; ++i)
         for (int j = 0; j < p_; ++j) w << D(i, j) << (j + 1 < p_ ? ' ' : '\n');
   }

   // ---- CG interface solve (opt-in, IFACE_CG) ----------------------------
   // Matrix-free apply of the interface operator: S y = C y − Σ_k B̄_k W_k⁻¹ B̄_kᵀ y,
   // using the SAME local machinery as solve_one (the block MA57 backsolves, the
   // selection lists N_k). This is the distributed-prototype character of the CG
   // path: a rank would apply S without ever assembling it. (The assembled Ssp_
   // is used only for the inertia/fallback factorization, the preconditioner
   // diagonals and the cheap true-residual acceptance test.)
   //
   // The K subdomain backsolves are independent; each thread writes ONLY
   // contrib_[k] and the scatter is a serial fixed-order loop, so the result is
   // bit-identical at any thread count.
   Eigen::VectorXd apply_S(const Eigen::VectorXd& y) {
      // ASSEMBLED mode: S was already paid for by the inertia factorization —
      // one sparse matvec, no backsolves. Same operator, same Krylov behavior.
      if (cg_apply_ == APPLY_ASSEMBLED) return Ssp_ * y;
      // MATFREE mode (the distributed cost simulation) below.
      // C is applied SPARSELY (Csp_, pattern frozen, values refreshed per
      // factorization): the corner block
      // has only the border-border KKT couplings, and the dense p×p matvec was
      // ~10% of every apply_S in the profile. Per-block scratch (yk_/tk_) is
      // persistent — two Eigen allocations per block per CG iteration showed up
      // as allocator traffic; each thread touches only its own k, so the
      // buffers are race-free under the OMP loop.
      Eigen::VectorXd Sy = Csp_ * y;
      if ((int)contrib_.size() != nsub_) {
         contrib_.resize(nsub_); yk_.resize(nsub_); tk_.resize(nsub_);
      }
// BACKSOLVE loop: parallel under BOTH backends (see the S_k loop note).
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         if (pk == 0) { contrib_[k].resize(0); continue; }
         Eigen::VectorXd& yk = yk_[k];
         yk.resize(pk);
         for (int a = 0; a < pk; ++a) yk[a] = y[Nk_[k][a]];
         Eigen::VectorXd& t = tk_[k];
         t.resize(dimk_[k]);
         t.noalias() = B_[k].transpose() * yk;                          // dim_k
         ma57_[k]->solve(t.data(), 1);                                  // W_k⁻¹ t
         contrib_[k].resize(pk);
         contrib_[k].noalias() = B_[k] * t;                             // pk
      }
      for (int k = 0; k < nsub_; ++k) {                  // deterministic serial scatter
         const int pk = (int)Nk_[k].size();
         for (int a = 0; a < pk; ++a) Sy[Nk_[k][a]] -= contrib_[k][a];
      }
      return Sy;
   }

   // Apply the field block S_ff to a kept-length vector, into a caller-owned
   // buffer. ASSEMBLED mode uses the precomputed sparse operator directly
   // (Ssp_ when nothing is peeled, the kept×kept Sffsp_ when the peel is
   // active) — no temporaries, no wasted peel rows. MATFREE keeps the faithful
   // distributed simulation: embed on the kept positions (0 at the peeled
   // entries), apply the full S through the subdomain backsolves, extract —
   // S·[v;0] restricted to kept is exactly S_ff v.
   void apply_Sff_into(const Eigen::VectorXd& v, Eigen::VectorXd& out) {
      if (cg_apply_ == APPLY_ASSEMBLED) {
         if (peel_.empty()) out.noalias() = Ssp_ * v;
         else out.noalias() = Sffsp_ * v;
         return;
      }
      if (peel_.empty()) { out = apply_S(v); return; }
      Eigen::VectorXd full = Eigen::VectorXd::Zero(p_);
      for (size_t a = 0; a < kept_.size(); ++a) full[kept_[a]] = v[(int)a];
      Eigen::VectorXd Sf = apply_S(full);
      out.resize((int)kept_.size());
      for (size_t a = 0; a < kept_.size(); ++a) out[(int)a] = Sf[kept_[a]];
   }

   // Build (once per factorization) the interface preconditioner on the kept
   // positions, and the α-peel cache. The preconditioners are the Python
   // reference's (../dd_kkt.py make_preconditioner, Lueg eq. 17 / 20–21):
   //   JACOBI  P = diag(|S_ff|)              (|·| ⇒ SPD even off the central path)
   //   BJ      M_k = S_k|kept, inverted densely, applied additively
   //   ASd     the same M_k but with the ASSEMBLED diag(S) on its diagonal
   // A singular local block is SKIPPED, as in the Python; if that leaves the
   // preconditioner effectively dead, CG's rz-breakdown guard catches it and the
   // solve falls back to direct.
   //
   // Peel cache (α + promoted duals as ONE dense Schur block):
   //   Z = S_ff⁻¹ S_fP  (each column by CG — the matrix-free spirit is preserved)
   //   T = S_PP − S_fPᵀ Z  (nP×nP dense, LU),
   // reused for every solve of this factorization:
   //   Δy_P = T⁻¹ (r_P − S_fPᵀ S_ff⁻¹ r_f),   Δy_f = S_ff⁻¹ r_f − Z Δy_P.
   // With peel = {α} alone this is exactly the scalar α peel; with the promoted
   // duals included, the indefinite directions of S live entirely in T and the
   // CG operator S_ff is SPD up to the (rare, measured) interlacing leak — which
   // the pSp/rz breakdown guards and the true-residual acceptance catch.
   void ensure_precond() {
      if (pc_valid_) return;
      pc_valid_ = true;
      const int nk = (int)kept_.size();
      const double* sv = Ssp_.valuePtr();     // #6: all S reads go through here
      if (precond_ == PRECOND_JACOBI) {
         pc_d_.resize(nk);
         for (int a = 0; a < nk; ++a)
            pc_d_[a] = std::max(std::abs(sv[sdiag_slot_[kept_[a]]]), 1e-300);
      } else {
         pc_Minv_.assign(nsub_, Eigen::MatrixXd());
         pc_idx_.assign(nsub_, {});
         pc_maxmb_ = 0;
         for (int k = 0; k < nsub_; ++k) {
            std::vector<int> loc, gl;          // loc: index in Sk_[k]; gl: kept-local
            for (int a = 0; a < (int)Nk_[k].size(); ++a) {
               const int kp = keptpos_[Nk_[k][a]];
               if (kp >= 0) { loc.push_back(a); gl.push_back(kp); }
            }
            const int mb = (int)loc.size();
            if (mb == 0) continue;
            pc_maxmb_ = std::max(pc_maxmb_, mb);
            Eigen::MatrixXd M(mb, mb);
            for (int i = 0; i < mb; ++i)
               for (int j = 0; j < mb; ++j) M(i, j) = Sk_[k](loc[i], loc[j]);
            if (precond_ == PRECOND_ASD)
               for (int i = 0; i < mb; ++i)
                  M(i, i) = sv[sdiag_slot_[kept_[gl[i]]]];
            Eigen::MatrixXd Mi = M.inverse();
            if (!Mi.allFinite()) continue;     // singular local block → skip (as Python)
            pc_Minv_[k] = Mi;
            pc_idx_[k] = gl;
         }
      }
      peel_ok_ = peel_.empty();
      if (!peel_.empty()) {
         const int nP = (int)peel_.size();
         // #6: gather S_fP and S_PP in ONE walk over the nP peel columns of the
         // sparse S. Every border position is kept XOR peeled, and the dense
         // gathers' structural zeros are supplied by the setZero.
         SfP_.setZero(nk, nP);
         Eigen::MatrixXd SPP = Eigen::MatrixXd::Zero(nP, nP);
         {
            const int* outer = Ssp_.outerIndexPtr();
            const int* inner = Ssp_.innerIndexPtr();
            for (int j = 0; j < nP; ++j) {
               const int c = peel_[j];
               for (int idx = outer[c]; idx < outer[c + 1]; ++idx) {
                  const int r = inner[idx];
                  const int kr = keptpos_[r];
                  if (kr >= 0) SfP_(kr, j) = sv[idx];
                  else SPP(peelpos_[r], j) = sv[idx];
               }
            }
         }
         // Z = S_ff⁻¹ S_fP through a SPARSE MA57 factorization of S_ff on the
         // spat_-restricted adjacency pattern (#5 — replaces the dense O(nf³)
         // LU; the peel-cache-by-CG variant before that cost 64% of all CG
         // iterations). Structure and symbolic analysis are frozen per kept
         // set; here only values are refreshed and factorized, and Z is one
         // multi-RHS backsolve. MA57's pivot signs give In(S_ff) exactly: an
         // indefinite S_ff (the Cauchy-interlacing leak) is rejected HERE,
         // before CG wastes a doomed attempt — the S_ff twin of the In(S) gate.
         ++s_if().cache_builds;
         bool ok = ensure_sff_structure();
         if (ok) {
            double* av = sff_ma57_->values();
            for (size_t e = 0; e < sffsrc_.size(); ++e) av[e] = sv[sffsrc_[e]];
            ok = sff_ma57_->factorize() && !sff_ma57_->singular() &&
                 sff_ma57_->negative_eigenvalues() == 0;
         }
         if (ok && cg_apply_ == APPLY_ASSEMBLED) {
            double* fv = Sffsp_.valuePtr();
            for (size_t e = 0; e < sffsp_slot_.size(); ++e)
               fv[sffsp_slot_[e]] = sv[sffsp_src_[e]];
         }
         if (ok) {
            Z_ = SfP_;
            sff_ma57_->solve(Z_.data(), nP);
            ok = Z_.allFinite();
         }
         if (ok) {
            Eigen::MatrixXd T = SPP - SfP_.transpose() * Z_;
            ok = T.allFinite();
            if (ok) Tlu_.compute(T);
         }
         peel_ok_ = ok;
         if (!peel_ok_ && std::getenv("DD_DEBUG"))
            std::cerr << "[dd-cg] peel cache failed (nP=" << nP
                      << ") → direct fallback\n";
      }
   }

   // Freeze the S_ff structure for the CURRENT kept set (#5, perf audit):
   // the spat_-restricted lower-triangle pattern with its MA57 symbolic
   // analysis, the Ssp_ value slots feeding its values, and (assembled mode) the
   // sparse CG operator Sffsp_'s pattern + value-slot maps. Rebuilt only when
   // kept_ changes (in practice: once per run). This replaces the dense
   // O(nf³) PartialPivLU of S_ff, which at N=64/k=8 (nf ≈ 1900) would cost
   // seconds PER FACTORIZATION; MA57 on the adjacency-sparse pattern also
   // returns In(S_ff) from its pivot signs — the exact S_ff twin of the
   // In(S) admissibility gate (it catches the Cauchy-interlacing leak before
   // CG wastes a doomed attempt on an indefinite operator).
   bool ensure_sff_structure() {
      if (sff_ma57_ && sff_kept_ == kept_) return true;
      sff_kept_ = kept_;
      sff_ma57_.reset();
      const int nk = (int)kept_.size();
      // spat_ restricted to kept×kept; kept_ ascending ⇒ order is preserved,
      // so global lower-triangle pairs stay lower-triangle locally.
      std::vector<int> irn, jcn;
      sffsrc_.clear();
      for (size_t e = 0; e < spat_.size(); ++e) {
         const int ka = keptpos_[spat_[e].first], kb = keptpos_[spat_[e].second];
         if (ka < 0 || kb < 0) continue;
         irn.push_back(ka + 1);
         jcn.push_back(kb + 1);
         sffsrc_.push_back(ma57_src_[e]);   // #6: Ssp_ slot of the lower element
      }
      sff_ma57_.reset(new SymBlock());
      if (!sff_ma57_->analyze(nk, irn, jcn)) {
         std::cerr << "[dd-cg] block analysis failed for S_ff (status="
                   << sff_ma57_->status() << ") — peel disabled\n";
         sff_ma57_.reset();
         return false;
      }
      if (cg_apply_ == APPLY_ASSEMBLED) {
         // Sffsp_ pattern (both triangles) + slot maps for per-factorization
         // value refresh — the same structure-once trick as the B_k blocks.
         std::vector<Trip> tS;
         std::vector<std::array<int, 3>> ents;   // (row, col, Ssp_ value slot)
         for (size_t e = 0; e < sffsrc_.size(); ++e) {
            const int ka = irn[e] - 1, kb = jcn[e] - 1;
            tS.push_back(Trip(ka, kb, 0.0));
            ents.push_back({ka, kb, sffsrc_[e]});
            if (ka != kb) {
               tS.push_back(Trip(kb, ka, 0.0));
               ents.push_back({kb, ka, sffsrc_[e]});
            }
         }
         Sffsp_.resize(nk, nk);
         Sffsp_.setFromTriplets(tS.begin(), tS.end());
         Sffsp_.makeCompressed();
         sffsp_slot_.clear();
         sffsp_src_.clear();
         const int* outer = Sffsp_.outerIndexPtr();
         const int* inner = Sffsp_.innerIndexPtr();
         for (const auto& e : ents) {
            const int r = (int)e[0], c = (int)e[1];
            const int* lo = inner + outer[c];
            const int* hi = inner + outer[c + 1];
            sffsp_slot_.push_back((int)(std::lower_bound(lo, hi, r) - inner));
            sffsp_src_.push_back(e[2]);
         }
      }
      return true;
   }

   // P⁻¹r on the kept positions, into a caller-owned buffer: Jacobi scaling, or
   // the additive BJ/ASd block sum (small per-block temporaries only).
   void applyPinv_into(const Eigen::VectorXd& r, Eigen::VectorXd& z) {
      if (precond_ == PRECOND_JACOBI) { z = r.cwiseQuotient(pc_d_); return; }
      z.setZero();
      // #7: the per-block gather/apply pair are members sized once per
      // preconditioner build — this runs every CG iteration, and a fresh pair
      // of Eigen vectors per block per iteration was the one allocation left
      // inside the CG loop.
      if (pp_rk_.size() < pc_maxmb_) { pp_rk_.resize(pc_maxmb_); pp_zk_.resize(pc_maxmb_); }
      for (int k = 0; k < nsub_; ++k) {
         const std::vector<int>& idx = pc_idx_[k];
         const int mb = (int)idx.size();
         if (mb == 0) continue;
         for (int i = 0; i < mb; ++i) pp_rk_[i] = r[idx[i]];
         pp_zk_.head(mb).noalias() = pc_Minv_[k] * pp_rk_.head(mb);
         for (int i = 0; i < mb; ++i) z[idx[i]] += pp_zk_[i];
      }
   }

   // Preconditioned CG on S_ff (one matrix-free apply_S per step). Parameter-free:
   // the step sizes come from the Krylov space, no spectrum estimate needed. The
   // breakdown guards (rz ≤ 0 ⇒ P not SPD, pSp ≤ 0 ⇒ S not SPD in this direction)
   // catch what the In(S) gate let through and hand the caller its fallback;
   // best-iterate tracking + a stagnation guard bound the work on stalls.
   Eigen::VectorXd cg_kept(const Eigen::VectorXd& b, double& best_rel, long& iters) {
      const int n = (int)b.size();
      const double bn = std::max(b.norm(), 1e-300);
      const int stall = 40;
      long it = 0, best_it = 0;
      best_rel = 1.0;
      Eigen::VectorXd best = Eigen::VectorXd::Zero(n);
      Eigen::VectorXd x = Eigen::VectorXd::Zero(n), r = b;
      // All loop vectors preallocated once: the assembled-mode matvec is
      // microseconds, so per-iteration temporaries would be a visible tax.
      Eigen::VectorXd z(n), pvec(n), Sp(n);
      applyPinv_into(r, z);
      pvec = z;
      double rz = r.dot(z);
      for (; it < cg_maxit_; ++it) {
         const double rel = r.norm() / bn;
         if (rel < best_rel) { best_rel = rel; best = x; best_it = it; }
         if (rel <= cg_tol_) break;
         if (it - best_it > stall) break;
         if (!(rz > 0.0)) break;                       // P⁻¹ not SPD ⇒ breakdown
         apply_Sff_into(pvec, Sp);
         const double pSp = pvec.dot(Sp);
         if (!(pSp > 0.0)) break;                      // S_ff not SPD in this dir
         const double alpha = rz / pSp;
         x += alpha * pvec;
         r -= alpha * Sp;
         applyPinv_into(r, z);
         const double rz_new = r.dot(z);
         pvec = z + (rz_new / rz) * pvec;              // cwise ⇒ alias-safe
         rz = rz_new;
      }
      iters = it;
      return best;
   }

   // The full CG interface solve: SPD gate → preconditioner/peel cache → CG on
   // S_ff → α recovery → true-residual acceptance on the assembled S. Returns
   // false whenever the caller should use the direct MA57 back-solve instead
   // (which is free — S is already factorized for the inertia).
   bool solve_interface_cg(const Eigen::VectorXd& ry, Eigen::VectorXd& dy) {
      const auto tic = std::chrono::steady_clock::now();
      ++s_if().solves;
      s_if().sneg_sum += s_neg_;
      s_if().sneg_max = std::max(s_if().sneg_max, (long)s_neg_);
      if (s_neg_ == 0) ++s_if().s_spd;
      bool ok = false;
      do {
         // CG needs an SPD operator on the KEPT block; In(S) is exact from the
         // MA57 pivots, and the promoted corner duals of a tile partition are
         // exactly S's negative eigenvalues (measured: In(S)_neg equals the
         // promoted count at EVERY solve). So the gate is: every negative must
         // be one of the peeled dual directions — with no dual peel that is the
         // plain SPD test, with it a tile partition's S_ff is admissible and CG
         // runs on the complement.
         if (s_neg_ != n_peel_dual_) { ++s_if().skipped_indef; break; }
         // Once CG has failed on THIS factorization's operator, every further
         // attempt on it (the refinement chain re-solves the same matrix) is a
         // doomed ~p/3-iteration prelude to the direct fallback — measured 40%
         // of all CG work at N=32 k=4 tile. Skip straight to direct until the
         // next factorization.
         if (cg_dead_) { ++s_if().skipped_dead; break; }
         ensure_precond();
         if (!peel_.empty() && !peel_ok_) {
            ++s_if().fallbacks; cg_dead_ = true; break;
         }
         const int nk = (int)kept_.size();
         Eigen::VectorXd rf(nk);
         for (int a = 0; a < nk; ++a) rf[a] = ry[kept_[a]];
         double rel = 1.0;
         long its = 0;
         Eigen::VectorXd w = cg_kept(rf, rel, its);
         s_if().iters += its;
         if (rel > 1e-2) { ++s_if().fallbacks; cg_dead_ = true; break; }
         if (rel > cg_tol_) ++s_if().nonconv;
         dy.resize(p_);
         if (!peel_.empty()) {
            const int nP = (int)peel_.size();
            Eigen::VectorXd rP(nP);
            for (int j = 0; j < nP; ++j) rP[j] = ry[peel_[j]];
            const Eigen::VectorXd dP = Tlu_.solve(rP - SfP_.transpose() * w);
            const Eigen::VectorXd wf = w - Z_ * dP;
            for (int a = 0; a < nk; ++a) dy[kept_[a]] = wf[a];
            for (int j = 0; j < nP; ++j) dy[peel_[j]] = dP[j];
         } else {
            for (int a = 0; a < nk; ++a) dy[kept_[a]] = w[a];
         }
         // Honest acceptance: the residual of the FULL system on the assembled S
         // (the peel algebra and the cached z carry the CG error).
         const double true_rel =
            (ry - Ssp_ * dy).norm() / std::max(ry.norm(), 1e-300);
         if (true_rel > 1e-2) { ++s_if().fallbacks; cg_dead_ = true; break; }
         ok = true;
      } while (false);
      s_if().wall += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - tic).count();
      return ok;
   }

   // ---- signed-MA57 MINRES interface (--interface minres) -----------------
   // Snapshot S = L D Lᵀ with MC64 OFF (so MA57's JOB=2/3/4 partial solves
   // compose exactly to JOB=1 — self-checked below), recover D⁻¹'s 1×1/2×2
   // block structure FORMAT-FREE by bit-pattern probing of the JOB=3 solve,
   // and store |D⁻¹|. Then M⁻¹ = L⁻ᵀ|D|⁻¹L⁻¹ is SPD, and M⁻¹S has spectrum
   // exactly {−1,+1} while the snapshot is fresh, so MINRES needs ~2 its no
   // matter how indefinite S is. Every step self-validates; any failure
   // disables the preconditioner (those solves go direct) instead of risking
   // a wrong step.
   void build_signed_precond() {
      const auto tic = std::chrono::steady_clock::now();
      sgn_valid_ = false;
      sgn_age_ = 0;
      ++s_if().sgn_builds;
      bool ok = true;
      if (!sgn_ma57_ || sgn_p_ != p_ || sgn_ne_ != spat_.size()) {
         sgn_ma57_.reset(new SymBlock());
         sgn_ma57_->scaling_off();       // partial solves need unscaled factors
         std::vector<int> irn, jcn;
         irn.reserve(spat_.size());
         jcn.reserve(spat_.size());
         for (const auto& e : spat_) {
            irn.push_back(e.first + 1);
            jcn.push_back(e.second + 1);
         }
         ok = sgn_ma57_->analyze(p_, irn, jcn);
         if (ok) { sgn_p_ = p_; sgn_ne_ = spat_.size(); }
         else sgn_ma57_.reset();
      }
      if (ok) {                          // same value routing as s_ma57_
         double* a = sgn_ma57_->values();
         const double* sv = Ssp_.valuePtr();
         for (size_t e = 0; e < spat_.size(); ++e) a[e] = sv[ma57_src_[e]];
         ok = sgn_ma57_->factorize() && !sgn_ma57_->singular();
      }
      // deterministic pseudo-random probe values (reproducible, no <random>)
      unsigned lcg_state = 0x9d2c5680u;
      auto lcg = [&lcg_state]() {
         lcg_state = 1103515245u * lcg_state + 12345u;
         return (double)(lcg_state >> 8) / (double)(1u << 23) - 1.0;
      };
      // (1) composition self-check: JOB4∘JOB3∘JOB2 must reproduce JOB1 on THIS
      // factorization (catches scaled factors or unexpected JOB semantics).
      if (ok) {
         Eigen::VectorXd v(p_);
         for (int i = 0; i < p_; ++i) v[i] = lcg();
         Eigen::VectorXd z1 = v, z2 = v;
         sgn_ma57_->solve_job(1, z1.data());
         sgn_ma57_->solve_job(2, z2.data());
         sgn_ma57_->solve_job(3, z2.data());
         sgn_ma57_->solve_job(4, z2.data());
         ok = (z1 - z2).norm() <= 1e-6 * std::max(z1.norm(), 1e-300);
      }
      // (2) recover E = D⁻¹ as JOB=3 applies it (original ordering): symmetric,
      // ≤1 off-diagonal partner per row (a permuted 1×1/2×2 block diagonal).
      // Probes: all-ones + every bit pattern AND its complement. For each
      // (row i, bit k) exactly one of the pattern/complement pair has b(i)=0,
      // and there c = e_off·b(j) — so e_off and every bit of the partner j
      // read off directly. 1 + 2·⌈log₂p⌉ D-solves, one batched MA57CD call.
      std::vector<double> e_ii, e_off;
      if (ok) {
         int nb = 1;
         while ((1 << nb) < p_) ++nb;
         const int nprob = 1 + 2 * nb;
         Eigen::MatrixXd P(p_, nprob);
         for (int i = 0; i < p_; ++i) {
            P(i, 0) = 1.0;
            for (int k = 0; k < nb; ++k) {
               const double bit = (double)((i >> k) & 1);
               P(i, 1 + k) = bit;
               P(i, 1 + nb + k) = 1.0 - bit;
            }
         }
         sgn_ma57_->solve_job(3, P.data(), nprob);
         sgn_partner_.assign(p_, -1);
         e_ii.assign(p_, 0.0);
         e_off.assign(p_, 0.0);
         for (int i = 0; i < p_ && ok; ++i) {
            const double s = P(i, 0);        // = e_ii + e_off (row sum of E)
            double scale = std::abs(s);
            for (int c = 1; c < nprob; ++c)
               scale = std::max(scale, std::abs(P(i, c)));
            const double tau = 1e-8 * scale + 1e-300;
            double off = 0.0;                // the largest b(i)=0 response
            for (int k = 0; k < nb; ++k) {
               const int zc = ((i >> k) & 1) ? (1 + nb + k) : (1 + k);
               if (std::abs(P(i, zc)) > std::abs(off)) off = P(i, zc);
            }
            if (std::abs(off) <= tau) { e_ii[i] = s; continue; }   // 1×1 block
            int j = 0;
            for (int k = 0; k < nb && ok; ++k) {
               const int zc = ((i >> k) & 1) ? (1 + nb + k) : (1 + k);
               const double c = P(i, zc);
               int bprobe;                   // the partner's bit in that probe
               if (std::abs(c) <= tau) bprobe = 0;
               else if (std::abs(c - off) <= 1e-6 * std::abs(off) + tau) bprobe = 1;
               else { ok = false; break; }   // three distinct values: not 2-nnz
               const int bitj = ((i >> k) & 1) ? (1 - bprobe) : bprobe;
               j |= bitj << k;
            }
            if (!ok) break;
            if (j == i || j < 0 || j >= p_) { ok = false; break; }
            e_ii[i] = s - off;
            e_off[i] = off;
            sgn_partner_[i] = j;
         }
         // pairing must be a perfect matching with symmetric values
         if (ok)
            for (int i = 0; i < p_; ++i) {
               const int j = sgn_partner_[i];
               if (j < 0) continue;
               if (sgn_partner_[j] != i ||
                   std::abs(e_off[i] - e_off[j]) >
                      1e-6 * std::abs(e_off[i]) + 1e-300) { ok = false; break; }
            }
      }
      // (3) independent validation of the recovery on a fresh random vector
      if (ok) {
         Eigen::VectorXd wv(p_);
         for (int i = 0; i < p_; ++i) wv[i] = lcg();
         Eigen::VectorXd Dw = wv;
         sgn_ma57_->solve_job(3, Dw.data());
         Eigen::VectorXd Ew(p_);
         for (int i = 0; i < p_; ++i) {
            double r = e_ii[i] * wv[i];
            if (sgn_partner_[i] >= 0) r += e_off[i] * wv[sgn_partner_[i]];
            Ew[i] = r;
         }
         ok = (Ew - Dw).norm() <= 1e-6 * std::max(Dw.norm(), 1e-300);
      }
      if (!ok) {
         ++s_if().sgn_fail;
         if (std::getenv("DD_DEBUG"))
            std::cerr << "[dd-minres] signed-precond build failed (p=" << p_
                      << ") — solves on this snapshot go direct\n";
         s_if().wall += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tic).count();
         return;
      }
      // (4) |E| blockwise. 1×1: |e|. 2×2 [[a,b],[b,c]]: |E| = √(E²) =
      // (E² + |det E|·I) / √(tr E² + 2|det E|) — closed form, SPD by
      // construction (eigenvalues |λ±|).
      sgn_diag_.assign(p_, 0.0);
      sgn_offv_.assign(p_, 0.0);
      for (int i = 0; i < p_; ++i) {
         const int j = sgn_partner_[i];
         if (j < 0) { sgn_diag_[i] = std::abs(e_ii[i]); continue; }
         if (j < i) continue;                // handled from the lower index
         const double a = e_ii[i], c = e_ii[j], b = e_off[i];
         const double dt = std::abs(a * c - b * b);
         const double denom = std::sqrt(a * a + c * c + 2 * b * b + 2 * dt);
         if (!(denom > 0)) {                 // ~zero block: degrade to diagonal
            sgn_diag_[i] = std::abs(a);
            sgn_diag_[j] = std::abs(c);
            sgn_partner_[i] = sgn_partner_[j] = -1;
            continue;
         }
         sgn_diag_[i] = (a * a + b * b + dt) / denom;
         sgn_diag_[j] = (c * c + b * b + dt) / denom;
         sgn_offv_[i] = sgn_offv_[j] = b * (a + c) / denom;
      }
      sgn_valid_ = true;
      s_if().wall += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - tic).count();
   }

   // M⁻¹v = L⁻ᵀ|D|⁻¹L⁻¹v, in place. SPD: a congruence of the SPD |D⁻¹|.
   void sgn_apply_Minv(Eigen::VectorXd& v) {
      sgn_ma57_->solve_job(2, v.data());
      mr_t_ = v;
      for (int i = 0; i < p_; ++i) {
         double r = sgn_diag_[i] * mr_t_[i];
         const int j = sgn_partner_[i];
         if (j >= 0) r += sgn_offv_[i] * mr_t_[j];
         v[i] = r;
      }
      sgn_ma57_->solve_job(4, v.data());
   }

   // Preconditioned MINRES (Paige–Saunders; recurrence as in scipy's minres)
   // on the full interface operator via apply_S. Returns true iff the
   // preconditioned-residual estimate reached cg_tol_; the caller applies the
   // honest true-residual acceptance either way.
   bool minres_kernel(const Eigen::VectorXd& b, Eigen::VectorXd& x, long& its) {
      x = Eigen::VectorXd::Zero(p_);
      Eigen::VectorXd r1 = b, r2 = b, y = b;
      sgn_apply_Minv(y);
      double beta1 = b.dot(y);
      its = 0;
      if (!(beta1 > 0) || !std::isfinite(beta1)) return false;  // M broken / b=0
      beta1 = std::sqrt(beta1);
      double oldb = 0, beta = beta1, dbar = 0, epsln = 0, phibar = beta1,
             cs = -1, sn = 0;
      Eigen::VectorXd w = Eigen::VectorXd::Zero(p_),
                      w2 = Eigen::VectorXd::Zero(p_), v(p_);
      bool conv = false;
      while (its < cg_maxit_) {
         ++its;
         v = y / beta;
         y = apply_S(v);
         if (its >= 2) y -= (beta / oldb) * r1;
         const double alfa = v.dot(y);
         y -= (alfa / beta) * r2;
         r1.swap(r2);
         r2 = y;
         y = r2;
         sgn_apply_Minv(y);
         oldb = beta;
         const double bet2 = r2.dot(y);
         if (!(bet2 >= 0) || !std::isfinite(bet2)) break;  // M not SPD / NaN
         beta = std::sqrt(bet2);
         const double oldeps = epsln;
         const double delta = cs * dbar + sn * alfa;
         const double gbar = sn * dbar - cs * alfa;
         epsln = sn * beta;
         dbar = -cs * beta;
         double gamma = std::sqrt(gbar * gbar + beta * beta);
         gamma = std::max(gamma, 1e-300);
         cs = gbar / gamma;
         sn = beta / gamma;
         const double phi = cs * phibar;
         phibar = sn * phibar;
         mr_wn_ = (v - oldeps * w2 - delta * w) / gamma;
         w2.swap(w);
         w = mr_wn_;
         x += phi * w;
         if (phibar <= cg_tol_ * beta1) { conv = true; break; }
         if (beta <= 1e-300 * beta1) break;   // Krylov space exhausted
      }
      return conv;
   }

   // The full MINRES interface solve: no SPD gate, no peel — the signed
   // preconditioner absorbs the indefiniteness. Same honest true-residual
   // acceptance and dead-matrix short-circuit as the CG path; returns false
   // whenever the caller should use the direct back-solve (which is free —
   // S is already factorized for the inertia).
   bool solve_interface_minres(const Eigen::VectorXd& ry, Eigen::VectorXd& dy) {
      const auto tic = std::chrono::steady_clock::now();
      ++s_if().solves;
      s_if().sneg_sum += s_neg_;
      s_if().sneg_max = std::max(s_if().sneg_max, (long)s_neg_);
      if (s_neg_ == 0) ++s_if().s_spd;
      const int age = std::min(sgn_age_, InterfaceStats::MAXAGE - 1);
      bool ok = false;
      do {
         if (!sgn_valid_) { ++s_if().sgn_unavail; break; }
         if (cg_dead_) { ++s_if().skipped_dead; break; }
         long its = 0;
         minres_kernel(ry, dy, its);
         // TRUE-residual refinement. MINRES's stopping test is an M⁻¹-norm
         // ESTIMATE whose attainable accuracy on cond(S)~1e10 stalls orders
         // above the direct back-solve (measured in the Python probe: 1e-5
         // true rel-res at estimate-tol, → 1e-11 after one refinement pass) —
         // and without it the Newton path drifts enough to derail levels.
         // Each pass is ~2 more its; the direct-quality step is what keeps
         // --interface minres trajectory-comparable to direct.
         const double bnorm = std::max(ry.norm(), 1e-300);
         Eigen::VectorXd resid = ry - Ssp_ * dy;
         double true_rel = resid.norm() / bnorm;
         for (int pass = 0; pass < 3 && true_rel > 10 * cg_tol_; ++pass) {
            long rits = 0;
            Eigen::VectorXd corr;
            minres_kernel(resid, corr, rits);
            if (rits == 0) break;               // breakdown on the correction
            its += rits;
            dy += corr;
            resid = ry - Ssp_ * dy;
            const double rel2 = resid.norm() / bnorm;
            if (!(rel2 < true_rel)) { dy -= corr; break; }   // no progress
            true_rel = rel2;
         }
         s_if().iters += its;
         ++s_if().age_cnt[age];
         s_if().age_its[age] += its;
         if (true_rel > cg_tol_) ++s_if().nonconv;
         if (!(true_rel <= 1e-2)) {
            ++s_if().fallbacks;
            ++s_if().age_fb[age];
            cg_dead_ = true;
            break;
         }
         ok = true;
      } while (false);
      s_if().wall += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - tic).count();
      return ok;
   }

   // ---- one solve (vault §6 pseudocode) ---------------------------------
   void solve_one(Number* rhs) {
      const int K = nsub_;
      // Member scratch (#4): rk/wk/ry keep their storage across the ~6 calls
      // per RHS and the thousands of RHS per run.
      if ((int)so_rk_.size() != K) {
         so_rk_.resize(K);
         so_wk_.resize(K);
         for (int k = 0; k < K; ++k) {
            so_rk_[k].resize(dimk_[k]);
            so_wk_[k].resize(dimk_[k]);
         }
         so_ry_.resize(p_);
      }
      std::vector<Eigen::VectorXd>&rk = so_rk_, &wk = so_wk_;
      Eigen::VectorXd& ry = so_ry_;
      for (Index i = 0; i < dim_; ++i) {
         if (owner_[i] >= 0) rk[owner_[i]][lpos_[i]] = rhs[i];
         else ry[ypos_[i]] = rhs[i];
      }
      // r_S = r_y − Σ_k B̄_k W_k⁻¹ r_k   (MA57CD solves in place)
      for (int k = 0; k < K; ++k) {
         wk[k] = rk[k];
         ma57_[k]->solve(wk[k].data(), 1);
         if (Nk_[k].empty()) continue;
         Eigen::VectorXd contrib = B_[k] * wk[k];
         for (size_t a = 0; a < Nk_[k].size(); ++a) ry[Nk_[k][a]] -= contrib[(int)a];
      }
      // Interface: opt-in preconditioned CG (with the direct factorization as
      // its safety net), or the direct MA57 back-solve of the assembled S.
      Eigen::VectorXd dy;
      bool cg_ok = false;
      if (iface_ == IFACE_CG) cg_ok = solve_interface_cg(ry, dy);
      else if (iface_ == IFACE_MINRES) cg_ok = solve_interface_minres(ry, dy);
      if (!cg_ok) {
         dy = ry;
         s_ma57_->solve(dy.data(), 1);
      }
      // Δx_k = W_k⁻¹(r_k − B̄_kᵀ N_kᵀ Δy)
      for (int k = 0; k < K; ++k) {
         if (!Nk_[k].empty()) {
            so_dyk_.resize((int)Nk_[k].size());
            for (size_t a = 0; a < Nk_[k].size(); ++a) so_dyk_[(int)a] = dy[Nk_[k][a]];
            so_bd_.resize(dimk_[k]);
            so_bd_.noalias() = B_[k].transpose() * so_dyk_;
            rk[k] -= so_bd_;
         }
         wk[k] = rk[k];
         ma57_[k]->solve(wk[k].data(), 1);
      }
      for (Index i = 0; i < dim_; ++i)
         rhs[i] = (owner_[i] >= 0) ? wk[owner_[i]][lpos_[i]] : dy[ypos_[i]];
   }

   // ---- state ------------------------------------------------------------
   Index dim_ = 0, nnz_ = 0, nneg_ = 0;
   int N_ = 0, K_ = 2, m_ = 0, nsub_ = 0, p_ = 0, max_dimk_ = 0;
   // Interface-solve mode, snapshotted from the statics in InitializeStructure.
   int iface_ = IFACE_DIRECT, precond_ = PRECOND_ASD, cg_maxit_ = 500;
   int cg_apply_ = APPLY_ASSEMBLED;
   double cg_tol_ = 1e-10;
   int s_neg_ = 0;              // In(S)_neg of the current factorization (CG gate)
   int n_peel_dual_ = 0;        // peeled DUAL entries (the admissible In(S)_neg)
   // ---- signed-MA57 MINRES state (--interface minres) ----
   int minres_lag_ = 1;         // rebuild the snapshot every this-many facts
   int sgn_age_ = 0;            // factorizations since the snapshot (0 = fresh)
   bool sgn_valid_ = false;     // snapshot factorized + |D⁻¹| recovered + checked
   int sgn_p_ = -1;             // pattern the snapshot was analyzed for
   size_t sgn_ne_ = 0;
   std::unique_ptr<SymBlock> sgn_ma57_;   // the snapshot factorization (MC64 off)
   std::vector<int> sgn_partner_;          // |D⁻¹| 2×2 partner (−1 = 1×1 block)
   std::vector<double> sgn_diag_, sgn_offv_;  // |D⁻¹| entries (symmetric)
   Eigen::VectorXd mr_t_, mr_wn_;          // M⁻¹-apply / MINRES scratch
   std::vector<int> peel_;      // peeled border positions (promoted duals + α)
   std::vector<int> kept_, keptpos_;   // border positions CG iterates on
   std::vector<int> peelpos_;   // border position → index in peel_ (−1 = kept)
   // Per-subdomain apply_S scratch (contribution, gathered border values, local
   // solve vector), reused across CG iterations so the parallel loop does not
   // reallocate every matvec.
   std::vector<Eigen::VectorXd> contrib_, yk_, tk_;
   SpMat Csp_;                  // sparse corner block C (matfree CG apply)
   // #6: THE assembled S — full symmetric pattern frozen in route_triplets,
   // values accumulated per factorization through the slot maps below. Sole
   // storage of S (the dense p×p C_/S_ are gone): feeds the s_ma57_ refresh,
   // the preconditioner diagonals, the SfP/S_PP peel gathers, the sff/Sffsp_
   // refresh, the CG apply and acceptance test, and the arrowhead dump.
   SpMat Ssp_;
   SpMat Sffsp_;                // sparse kept×kept S_ff (assembled apply, peel on)
   // Preconditioner + peel cache, rebuilt once per factorization.
   bool pc_valid_ = false, peel_ok_ = false;
   bool cg_dead_ = false;   // CG failed on the current factorization's operator
   Eigen::VectorXd pc_d_;                    // Jacobi diagonal on kept
   std::vector<Eigen::MatrixXd> pc_Minv_;    // BJ/ASd block inverses
   std::vector<std::vector<int>> pc_idx_;    // their kept-local index lists
   Eigen::VectorXd pp_rk_, pp_zk_;           // #7: applyPinv block buffers
   int pc_maxmb_ = 0;                        // largest BJ/ASd block (sizes them)
   Eigen::MatrixXd SfP_, Z_;                 // S_fP and Z = S_ff⁻¹ S_fP (nf×nP)
   Eigen::PartialPivLU<Eigen::MatrixXd> Tlu_;   // dual Schur T, factorized
   // #5: frozen sparse S_ff (peel cache + assembled CG operator slots)
   std::unique_ptr<SymBlock> sff_ma57_;
   std::vector<int> sff_kept_;               // kept set the structure was built for
   std::vector<int> sffsrc_;                 // Ssp_ value slot per S_ff lower-tri entry
   std::vector<int> sffsp_slot_;             // Sffsp_ valuePtr slots (both triangles)
   std::vector<int> sffsp_src_;              // their Ssp_ value sources
   std::vector<Index> irow_, jcol_;
   std::vector<Number> vals_;
   std::vector<int> sub_of_pixel_, owner_, ypos_, lpos_, dimk_;
   std::vector<std::vector<int>> Nk_, ykrow_;
   // #1: frozen border-coupling routing (built once in route_triplets)
   std::vector<std::vector<int>> btrip_t_, btrip_slot_;   // B_k value-slot maps
   std::vector<int> ctrip_t_;                             // border-border triplets
   // #6: frozen slot maps into Ssp_'s (and Csp_'s) value arrays
   std::vector<int> cslot1_, cslot2_;        // C triplet → Ssp_ slots (entry, mirror)
   std::vector<int> csp1_, csp2_;            // C triplet → Csp_ slots (matfree apply)
   std::vector<int> ma57_src_;               // spat_ (lower-tri) entry → Ssp_ slot
   std::vector<int> sdiag_slot_;             // diagonal (j,j) → Ssp_ slot
   std::vector<std::vector<int>> sk_slot_;   // per k: Sk_ entry (col-major) → Ssp_ slot
   std::vector<char> okbuf_;
   // #2: frozen-structure fast path across levels
   bool frozen_ = false;
   int frzN_ = -1, frzK_ = -1;
   std::vector<int> frzOwnerCfg_;
   // #4: solve-path scratch (serial use only)
   std::vector<double> sr_b0_, sr_x_, sr_r_, sr_Ax_, sr_best_;
   std::vector<Eigen::VectorXd> so_rk_, so_wk_;
   Eigen::VectorXd so_ry_, so_dyk_, so_bd_;
   std::vector<std::vector<char>> seen_;
   std::vector<SpMat> B_;
   // Which input triplets belong to each W_k, in the order MA57 analyzed them.
   std::vector<std::vector<int>> wtrip_;
   // One MA57 factorization per subdomain (held by pointer: the workspace is
   // large and the object is not cheaply copyable). Symbolic analysis is done
   // once in route_triplets(); only factorize() runs per Newton step.
   std::vector<std::unique_ptr<SymBlock>> ma57_;
   std::vector<Eigen::MatrixXd> Sk_;   // local Schur blocks (feed the BJ/ASd blocks)
   std::unique_ptr<SymBlock> s_ma57_;
   std::vector<std::pair<int, int>> spat_;   // sparse lower-tri pattern of S
   std::unique_ptr<SymBlock> ref_ma57_;     // DD_CHECK reference only
   double t_factor_ = 0, t_schur_ = 0, t_scatter_ = 0, t_sfact_ = 0, t_solve_ = 0;
   long n_fact_ = 0;
};

// ---------------------------------------------------------------------------
// Injection: override AlgorithmBuilder::SymLinearSolverFactory so every Newton
// step goes through SolverT. Used as
//   SmartPtr<AlgorithmBuilder> b = new CustomSolverBuilder<DDArrowheadSolver>();
//   app->OptimizeNLP(new TNLPAdapter(GetRawPtr(tnlp)), b);
// ---------------------------------------------------------------------------
template <class SolverT>
class CustomSolverBuilder : public AlgorithmBuilder {
public:
   SmartPtr<SymLinearSolver> SymLinearSolverFactory(
      const Journalist&, const OptionsList&, const std::string&) override {
      // ONE solver instance for the whole process (#2, perf audit): each
      // continuation level gets a fresh TSymLinearSolver wrapper, but the
      // underlying DD solver — with its partition, routing tables and MA57
      // symbolic analyses — is carried over; InitializeStructure's
      // same-structure guard decides what may be reused. DD_TIME's phase
      // timers therefore accumulate across the RUN now, not per level.
      static SmartPtr<SparseSymLinearSolverInterface> shared = new SolverT();
      return new TSymLinearSolver(shared, NULL);
   }
};

}  // namespace Ipopt

#endif  // DD_SOLVER_HPP
