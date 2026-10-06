// arrowhead.hpp — a domain-decomposition solver for IPOPT's KKT system.
//
// Pure Eigen: nothing here knows about IPOPT.  ipopt_bridge.hpp plugs it in.
//
// ============================================================================
//  1. THE ARROWHEAD
// ============================================================================
// Number the KKT unknowns tile by tile, with the border (the consensus variables)
// last.  Tiles never couple directly, so the matrix is an ARROWHEAD:
//
//        ⎡ W_1            B_1ᵀ ⎤ ⎡x_1⎤   ⎡r_1⎤
//        ⎢      ⋱          ⋮   ⎥ ⎢ ⋮ ⎥ = ⎢ ⋮ ⎥
//        ⎢          W_K   B_Kᵀ ⎥ ⎢x_K⎥   ⎢r_K⎥
//        ⎣ B_1  ⋯   B_K    C   ⎦ ⎣ y ⎦   ⎣r_y⎦
//
// Eliminating the tiles leaves the INTERFACE (Schur complement) system
//
//        S y = r_y − Σ_k B_k W_k⁻¹ r_k,     S = C + Σ_k S_k,   S_k = −B_k W_k⁻¹ B_kᵀ
//
// and then each x_k = W_k⁻¹ (r_k − B_kᵀ y) independently.
//
// ============================================================================
//  2. INERTIA
// ============================================================================
// IPOPT also needs the number of negative eigenvalues of the KKT matrix.
// Haynsworth's formula splits it along the same lines:
//
//        #neg(A) = Σ_k #neg(W_k) + #neg(S)
//
// #neg(W_k) is free: count negative pivots of W_k's LDLᵀ.  S is never formed.
//
// ============================================================================
//  3. THE INTERFACE SOLVE: CG, AFTER A PEEL
// ============================================================================
// S is solved by preconditioned conjugate gradients, which needs S symmetric
// positive definite.  Two kinds of border unknown spoil that, and are PEELED off
// into a small set P, leaving the rest f ("kept") for CG:
//
//   · α — shared by every tile, so its row of S is dense and badly scaled;
//   · the CROSS POINTS — border unknowns touched by ≥ 3 tiles.  They do not break
//     definiteness, but without them CG's convergence degrades as tiles are added.
//
// With the split S = [S_ff S_fP; S_Pf S_PP], block elimination gives
//
//        Z = S_ff⁻¹ S_fP              (one CG solve per peeled column)
//        T = S_PP − S_fPᵀ Z           (small, dense, factorized directly)
//        y_P = T⁻¹ (r_P − S_fPᵀ S_ff⁻¹ r_f)
//        y_f = S_ff⁻¹ r_f − Z y_P
//
// and, again by Haynsworth, #neg(S) = #neg(S_ff) + #neg(T) = #neg(T), using the
// premise that S_ff is positive definite (the premise CG needs anyway).  So the
// inertia costs one eigendecomposition of the small matrix T.
//
// The premise is CHECKED, not assumed: if CG meets a direction of non-positive
// curvature (pᵀ S_ff p ≤ 0) while computing Z, the factorization is refused and
// IPOPT regularizes.  Without that check a factorization with an indefinite
// S_ff would report an inertia one to three short of the truth, IPOPT would
// accept it, and the run would drift to another point (measured at N=32: a
// dense eigendecomposition of S found S_ff indefinite in exactly the
// factorizations where CG broke down, at 4×4 and 8×8 tiles alike).
//
// ============================================================================
//  4. ONE FACTORIZATION, ONE SOLVE
// ============================================================================
//   factorize():  LDLᵀ of every W_k  →  every S_k  →  C and diag(S)
//                 →  preconditioner  →  Z, T, #neg(T)
//   solve():      the block elimination above, wrapped in iterative refinement
//                 against the TRUE matrix.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace dd {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// =============================================================================
//  Ldlt — sparse LDLᵀ of one W_k, plus the internals needed to form S_k fast.
// =============================================================================
class Ldlt {
public:
   // Fill-reducing ordering + elimination tree.  Depends on the pattern only.
   bool analyze(const SpMat& A) {
      f_.analyzePattern(A);
      analyzed_ = (f_.info() == Eigen::Success);
      return analyzed_;
   }

   // Numeric factorization W = P⁻¹ L D Lᵀ P.  Returns false on breakdown
   // (Eigen failed, or a zero / non-finite pivot).  Counts negative pivots.
   bool factorize(const SpMat& A) {
      n_neg_ = 0;
      if (!analyzed_ && !analyze(A)) return false;
      f_.factorize(A);
      if (f_.info() != Eigen::Success) return false;
      const Vec& d = f_.vectorD();
      for (int i = 0; i < d.size(); ++i) {
         if (!std::isfinite(d[i]) || d[i] == 0.0) return false;
         if (d[i] < 0.0) ++n_neg_;
      }
      return true;
   }

   // In-place solve for nrhs right-hand sides stored column by column.
   void solve(double* b, int dim, int nrhs) const {
      Eigen::Map<Mat> B(b, dim, nrhs);
      const Mat X = f_.solve(B);   // not in place: solve() must not alias its input
      B = X;
   }

   int negative_eigenvalues() const { return n_neg_; }

   const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>& P() const {
      return f_.permutationP();
   }
   const Vec& D() const { return f_.rawD(); }
   const int* etree_parent() const { return f_.rawParent().data(); }

   // y ← L⁻¹ y for a right-hand side whose nonzeros lie inside `reach` (an
   // ascending list of columns).  Visiting only the reach is what makes forming
   // S_k cheap: B_kᵀ is very sparse, so each column of L⁻¹PB_kᵀ touches few columns.
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
   // Eigen keeps L, D and the elimination tree protected; this exposes them.
   struct Impl : Eigen::SimplicialLDLT<SpMat, Eigen::Lower> {
      const SpMat& rawL() const { return this->m_matrix; }
      const Vec& rawD() const { return this->m_diag; }
      const Eigen::VectorXi& rawParent() const { return this->m_parent; }
   };
   Impl f_;
   bool analyzed_ = false;
   int n_neg_ = 0;
};

// =============================================================================
//  Precond — the additive "ASd" preconditioner for CG on S_ff.
//
//  For each tile k, take its block S_k restricted to the kept border unknowns it
//  touches, REPLACE ITS DIAGONAL by the diagonal of the full S, and apply the
//  inverses of these small dense blocks additively:
//
//        M⁻¹ = Σ_k R_kᵀ M_k⁻¹ R_k
//
//  The diagonal swap matters: S_k alone sees only one tile's share of a border
//  unknown's self-coupling, while diag(S) sums every tile's share.
//
//  Each M_k is applied through a SYMMETRIC factorization (never an explicit
//  inverse): CG needs M⁻¹ symmetric to round-off.  A block that is not SPD is
//  still used, with its eigenvalues clipped to be positive.
// =============================================================================
class Precond {
public:
   void build(int nf, const std::vector<Mat>& Sk, const std::vector<std::vector<int>>& Nk,
              const std::vector<int>& keptpos, const Vec& diagS_kept) {
      nf_ = nf;
      llt_.assign(Nk.size(), Eigen::LLT<Mat>());
      clipped_inv_.assign(Nk.size(), Mat());
      spd_.assign(Nk.size(), 0);
      idx_.assign(Nk.size(), {});
      for (size_t k = 0; k < Nk.size(); ++k) {
         // local row of S_k  and  its position among the kept unknowns
         std::vector<int> loc, pos;
         for (int a = 0; a < (int)Nk[k].size(); ++a) {
            const int kp = keptpos[Nk[k][a]];
            if (kp >= 0) { loc.push_back(a); pos.push_back(kp); }
         }
         const int mb = (int)loc.size();
         if (mb == 0) continue;
         Mat M(mb, mb);
         for (int i = 0; i < mb; ++i)
            for (int j = 0; j < mb; ++j) M(i, j) = Sk[k](loc[i], loc[j]);
         for (int i = 0; i < mb; ++i) M(i, i) = diagS_kept[pos[i]];   // the swap
         if (!M.allFinite()) continue;

         Eigen::LLT<Mat> chol(M);
         if (chol.info() == Eigen::Success) {           // SPD: use Cholesky
            llt_[k] = std::move(chol);
            spd_[k] = 1;
         } else {                                       // indefinite: clip eigenvalues
            Eigen::SelfAdjointEigenSolver<Mat> es(M);
            if (es.info() != Eigen::Success) continue;
            Vec d = es.eigenvalues();
            const double scale = d.cwiseAbs().maxCoeff();
            if (!(scale > 0.0)) continue;
            for (int i = 0; i < mb; ++i) d[i] = 1.0 / std::max(d[i], 1e-6 * scale);
            Mat inv = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
            if (!inv.allFinite()) continue;
            clipped_inv_[k] = Mat(0.5 * (inv + inv.transpose()));   // exactly symmetric
            spd_[k] = 0;
         }
         idx_[k] = pos;
      }
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
         const Vec zk = spd_[k] ? Vec(llt_[k].solve(rk)) : Vec(clipped_inv_[k] * rk);
         for (int i = 0; i < mb; ++i) z[idx[i]] += zk[i];
      }
   }

private:
   int nf_ = 0;
   std::vector<Eigen::LLT<Mat>> llt_;   // SPD blocks
   std::vector<Mat> clipped_inv_;       // indefinite blocks: U·max(Λ, ε)⁻¹·Uᵀ
   std::vector<char> spd_;
   std::vector<std::vector<int>> idx_;  // kept positions each block acts on
};

// =============================================================================
//  cg_solve — preconditioned conjugate gradients on an operator given as a
//  function applyA(v, out).  Optionally warm-started from x0.
//
//  It returns the BEST iterate seen (CG's residual is not monotone) and stops on:
//    · relative residual ≤ tol,        · maxit iterations,
//    · 300 iterations without a new best residual,
//    · rᵀz ≤ 0 (preconditioner not SPD),  · pᵀAp ≤ 0 (operator not SPD).
//  The last one is reported back as `breakdown`: it is a one-directional proof
//  that the operator is not SPD, which the caller may act on.
// =============================================================================
struct CgResult {
   double rel = 1.0;         // best relative residual reached
   long iters = 0;
   bool breakdown = false;   // pᵀAp ≤ 0 was met
};

template <class ApplyA>
inline CgResult cg_solve(ApplyA&& applyA, const Precond& M, const Vec& b, Vec& x,
                         double tol, int maxit, const Vec* x0 = nullptr) {
   const int n = (int)b.size();
   const double bnorm = std::max(b.norm(), 1e-300);
   Vec r(n), z(n);
   if (x0 != nullptr && x0->size() == n && x0->allFinite()) {
      x = *x0;
      applyA(x, r);
      r = b - r;
   } else {
      x.setZero(n);
      r = b;
   }
   M.apply(r, z);
   Vec p = z, Ap(n);
   double rz = r.dot(z);

   CgResult res;
   double best_rel = 1.0;
   Vec best = Vec::Zero(n);
   int best_it = 0, it = 0;
   const int stall = 5000;
   for (; it < maxit; ++it) {
      const double rel = r.norm() / bnorm;
      if (rel < best_rel) { best_rel = rel; best = x; best_it = it; }
      if (rel <= tol) break;
      if (it - best_it > stall) break;
      if (!(rz > 0.0)) break;
      applyA(p, Ap);
      const double pAp = p.dot(Ap);
      if (!(pAp > 0.0)) { res.breakdown = true; break; }
      const double alpha = rz / pAp;
      x += alpha * p;
      r -= alpha * Ap;
      M.apply(r, z);
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
//  Arrowhead — the solver.
//
//    set_structure(...)   once: numbering, routing, symbolic analysis, peel set
//    values()             IPOPT writes the matrix entries here
//    factorize()          every Newton step
//    solve(rhs)           every right-hand side, in place
// =============================================================================
class Arrowhead {
public:
   struct Options {
      int alpha_index = -1;     // KKT index of α (peeled)
      double cg_tol = 1e-10;    // interface solve
      double peel_cg_tol = 1e-7;// the Z columns: looser is fine up to here
      int cg_maxit = 100000;
      // How the systems with S_ff are solved:
      //   0  CG everywhere (section 3 of the header)
      //   1  Z by a sparse LDLᵀ of the ASSEMBLED S_ff; interface solve still CG
      //   2  interface solve by that LDLᵀ too: no CG at all
      // Assembling and factorizing S_ff is a global serial step, so 1 and 2 are
      // single-node shortcuts, not the distributed method.  A non-positive
      // pivot plays the role of CG's pᵀAp ≤ 0 and refuses the factorization.
      int sff_direct = 0;
   };

   struct Stats {
      long solves = 0;          // interface CG solves
      long iters = 0;           // CG iterations, all solves and Z columns
      long rejected = 0;        // interface answers above 1e-2 (still used)
      long factorizations = 0;  // peel caches built
      long attempts = 0;        // factorize() calls, refused ones included
      long wrong_inertia = 0;   // factorizations IPOPT rejected for their inertia
                                // (counted by ipopt_bridge.hpp)
      double t_factorize = 0.0; // wall seconds in factorize()
      double t_peel = 0.0;      //   ... of which the preconditioner, Z and T
      double t_solve = 0.0;     // wall seconds in solve()
   };

   enum Status { OK, SINGULAR };

   // --------------------------------------------------------------------------
   //  Structure.  irow/jcol are 0-based lower-triangle coordinates; owner gives
   //  each KKT unknown its tile, or −1 for the border.
   // --------------------------------------------------------------------------
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
      if ((int)owner_.size() != dim_) return false;
#ifdef _OPENMP
      Eigen::setNbThreads(1);   // our loops are threaded; keep Eigen's own off
#endif
      number_unknowns();
      if (!route_triplets()) return false;
      choose_peel();
      return true;
   }

   double* values() { return vals_.data(); }
   int dim() const { return dim_; }
   int negative_eigenvalues() const { return n_neg_; }
   const Stats& stats() const { return stats_; }

   // --------------------------------------------------------------------------
   //  Factorization.
   // --------------------------------------------------------------------------
   Status factorize() {
      const auto t_start = std::chrono::steady_clock::now();
      struct Timer {
         double& acc;
         std::chrono::steady_clock::time_point t0;
         ~Timer() {
            acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         }
      } timer{stats_.t_factorize, t_start};
      ++stats_.attempts;
      n_neg_ = 0;
      peel_valid_ = false;

      // (a) LDLᵀ of every W_k — independent per tile.
      std::vector<char> ok(nsub_, 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) {
         fill(W_[k], wtrip_[k]);
         fill(B_[k], btrip_[k]);
         ok[k] = ldlt_[k]->factorize(W_[k]) ? 1 : 0;
      }
      for (int k = 0; k < nsub_; ++k) {
         if (!ok[k]) return SINGULAR;   // IPOPT answers by regularizing more
         n_neg_ += ldlt_[k]->negative_eigenvalues();
      }

      // (b) S_k = −B_k W_k⁻¹ B_kᵀ — independent per tile.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < nsub_; ++k) form_local_schur(k);

      // (c) The corner block C (symmetric), and diag(S) without forming S.
      std::vector<Trip> t;
      t.reserve(ctrip_.size() * 2);
      for (const auto& e : ctrip_) {
         const double v = vals_[e.t];
         t.emplace_back(e.r, e.c, v);
         if (e.r != e.c) t.emplace_back(e.c, e.r, v);
      }
      C_.resize(p_, p_);
      C_.setFromTriplets(t.begin(), t.end());
      C_.makeCompressed();

      diagS_.setZero(p_);
      for (const auto& e : ctrip_)
         if (e.r == e.c) diagS_[e.r] += vals_[e.t];
      for (int k = 0; k < nsub_; ++k)
         for (int a = 0; a < (int)Nk_[k].size(); ++a) diagS_[Nk_[k][a]] += Sk_[k](a, a);

      // (d) Preconditioner, Z, T and #neg(T).  If that fails there is no
      //     inertia to report, so the factorization is refused.
      if (!build_peel()) return SINGULAR;
      n_neg_ += t_neg_;
      return OK;
   }

   // --------------------------------------------------------------------------
   //  Solve, in place.  One arrowhead solve, then up to 3 refinement sweeps
   //  against the TRUE matrix; the best iterate is returned.  False only if no
   //  finite residual was ever reached.
   // --------------------------------------------------------------------------
   bool solve(double* rhs) {
      const auto t_start = std::chrono::steady_clock::now();
      struct Timer {
         double& acc;
         std::chrono::steady_clock::time_point t0;
         ~Timer() {
            acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         }
      } timer{stats_.t_solve, t_start};
      const std::vector<double> b(rhs, rhs + dim_);
      std::vector<double> x(b), r(dim_), Ax(dim_), best;
      if (!solve_arrowhead(x.data())) return false;

      double bnorm = 0.0;
      for (int i = 0; i < dim_; ++i) bnorm += b[i] * b[i];
      bnorm = std::sqrt(std::max(bnorm, 1e-300));

      double best_res = std::numeric_limits<double>::infinity();
      for (int sweep = 0; sweep < 3; ++sweep) {
         matvec(x.data(), Ax.data());
         double rn = 0.0;
         for (int i = 0; i < dim_; ++i) {
            r[i] = b[i] - Ax[i];
            rn += r[i] * r[i];
         }
         const double rel = std::sqrt(rn) / bnorm;
         if (sweep == 0 || rel < best_res) { best_res = rel; best = x; }
         if (!std::isfinite(rel)) break;
         if (rel <= 1e-11) break;
         if (!solve_arrowhead(r.data())) break;   // r ← A⁻¹ r, the correction
         for (int i = 0; i < dim_; ++i) x[i] += r[i];
      }
      if (!std::isfinite(best_res)) return false;
      std::copy(best.begin(), best.end(), rhs);
      return true;
   }

private:
   // ==========================================================================
   //  Structure
   // ==========================================================================

   // Border unknown i → ypos_[i] ∈ [0, p).   Tile unknown i → lpos_[i] ∈ [0, dim W_k).
   // Both follow ascending KKT index, so lower-triangle entries stay lower-triangle.
   void number_unknowns() {
      ypos_.assign(dim_, -1);
      lpos_.assign(dim_, -1);
      dimk_.assign(nsub_, 0);
      p_ = 0;
      for (int i = 0; i < dim_; ++i) {
         if (owner_[i] < 0) ypos_[i] = p_++;
         else lpos_[i] = dimk_[owner_[i]]++;
      }
   }

   // Route every matrix entry to W_k, B_k or C; find which border unknowns each
   // tile touches (N_k); analyze every W_k; precompute the reaches for S_k.
   bool route_triplets() {
      std::vector<std::vector<char>> seen(nsub_, std::vector<char>(p_, 0));
      for (int t = 0; t < nnz_; ++t) {
         const int oi = owner_[irow_[t]], oj = owner_[jcol_[t]];
         if (oi >= 0 && oj >= 0 && oi != oj) return false;   // tiles must not couple
         if (oi < 0 && oj >= 0) seen[oj][ypos_[irow_[t]]] = 1;
         if (oj < 0 && oi >= 0) seen[oi][ypos_[jcol_[t]]] = 1;
      }
      Nk_.assign(nsub_, {});
      std::vector<std::vector<int>> row_in_Bk(nsub_, std::vector<int>(p_, -1));
      for (int k = 0; k < nsub_; ++k)
         for (int j = 0; j < p_; ++j)
            if (seen[k][j]) {
               row_in_Bk[k][j] = (int)Nk_[k].size();
               Nk_[k].push_back(j);
            }

      wtrip_.assign(nsub_, {});
      btrip_.assign(nsub_, {});
      ctrip_.clear();
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         const int oi = owner_[i], oj = owner_[j];
         if (oi >= 0 && oj >= 0)
            wtrip_[oi].push_back({lpos_[i], lpos_[j], t});                   // → W_k
         else if (oi < 0 && oj < 0)
            ctrip_.push_back({ypos_[i], ypos_[j], t});                       // → C
         else if (oi < 0)
            btrip_[oj].push_back({row_in_Bk[oj][ypos_[i]], lpos_[j], t});    // → B_k
         else
            btrip_[oi].push_back({row_in_Bk[oi][ypos_[j]], lpos_[i], t});    // → B_k
      }

      W_.assign(nsub_, SpMat());
      B_.assign(nsub_, SpMat());
      Sk_.assign(nsub_, Mat());
      ldlt_.clear();
      for (int k = 0; k < nsub_; ++k) ldlt_.emplace_back(new Ldlt());
      reach_.assign(nsub_, {});
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         W_[k].resize(dimk_[k], dimk_[k]);
         B_[k].resize(pk, dimk_[k]);
         fill(W_[k], wtrip_[k]);   // values are zero here: only the pattern matters
         fill(B_[k], btrip_[k]);
         if (!ldlt_[k]->analyze(W_[k])) return false;

         // reach_[k][a]: columns of L_k that column a of P·B_kᵀ can touch during
         // forward substitution = the elimination-tree paths from its nonzeros.
         const SpMat PBt = ldlt_[k]->P() * SpMat(B_[k].transpose());
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

   // The peeled set P: α plus every border unknown touched by ≥ 3 tiles.
   // (If that would exceed a quarter of the border, the cross points are skipped.)
   void choose_peel() {
      peel_.clear();
      const int a = opt_.alpha_index;
      if (a >= 0 && a < dim_ && owner_[a] < 0) peel_.push_back(ypos_[a]);

      std::vector<int> degree(p_, 0);
      for (int k = 0; k < nsub_; ++k)
         for (int j : Nk_[k]) ++degree[j];
      std::vector<char> already(p_, 0);
      for (int j : peel_) already[j] = 1;
      std::vector<int> cross;
      for (int j = 0; j < p_; ++j)
         if (!already[j] && degree[j] >= 3) cross.push_back(j);
      if (!cross.empty() && peel_.size() + cross.size() <= (size_t)std::max(1, p_ / 4)) {
         peel_.insert(peel_.end(), cross.begin(), cross.end());
         std::sort(peel_.begin(), peel_.end());
      }

      std::vector<char> peeled(p_, 0);
      for (int j : peel_) peeled[j] = 1;
      keptpos_.assign(p_, -1);
      kept_.clear();
      for (int j = 0; j < p_; ++j)
         if (!peeled[j]) { keptpos_[j] = (int)kept_.size(); kept_.push_back(j); }
   }

   // A routed matrix entry: (row, col) inside its block, and its index in vals_.
   struct Entry { int r, c, t; };

   void fill(SpMat& M, const std::vector<Entry>& entries) {
      std::vector<Trip> t;
      t.reserve(entries.size());
      for (const auto& e : entries) t.emplace_back(e.r, e.c, vals_[e.t]);
      M.setFromTriplets(t.begin(), t.end());   // duplicate entries are summed
      M.makeCompressed();
   }

   // ==========================================================================
   //  Factorization pieces
   // ==========================================================================

   // S_k = −B_k W_k⁻¹ B_kᵀ.  With W_k = P⁻¹ L D Lᵀ P and Y = L⁻¹ P B_kᵀ,
   //
   //        S_k = −Yᵀ D⁻¹ Y = −Σ_rows r  (1/D_r) · y_r y_rᵀ.
   //
   // Each column of Y is computed by a forward substitution restricted to its
   // reach; then rows of Y are visited once each to accumulate the outer products.
   void form_local_schur(int k) {
      const int pk = (int)Nk_[k].size();
      if (pk == 0) { Sk_[k].resize(0, 0); return; }
      const int nk = dimk_[k];
      const SpMat PBt = ldlt_[k]->P() * SpMat(B_[k].transpose());
      const std::vector<std::vector<int>>& reach = reach_[k];

      // Y column by column, stored only on its reach.
      std::vector<std::vector<double>> Y(pk);
      Vec w = Vec::Zero(nk);
      for (int a = 0; a < pk; ++a) {
         const std::vector<int>& R = reach[a];
         for (SpMat::InnerIterator it(PBt, a); it; ++it) w[it.row()] = it.value();
         ldlt_[k]->forward_solve_reach(w.data(), R);
         Y[a].resize(R.size());
         for (size_t i = 0; i < R.size(); ++i) {
            Y[a][i] = w[R[i]];
            w[R[i]] = 0.0;   // leave w zero for the next column
         }
      }

      // Regroup Y's nonzeros by row (columns stay in ascending order).
      std::vector<int> rowptr(nk + 1, 0);
      for (int a = 0; a < pk; ++a)
         for (size_t i = 0; i < reach[a].size(); ++i)
            if (Y[a][i] != 0.0) ++rowptr[reach[a][i] + 1];
      for (int r = 0; r < nk; ++r) rowptr[r + 1] += rowptr[r];
      std::vector<int> rcol(rowptr[nk]);
      std::vector<double> rval(rowptr[nk]);
      std::vector<int> next(rowptr.begin(), rowptr.end() - 1);
      for (int a = 0; a < pk; ++a)
         for (size_t i = 0; i < reach[a].size(); ++i)
            if (Y[a][i] != 0.0) {
               const int r = reach[a][i];
               rcol[next[r]] = a;
               rval[next[r]] = Y[a][i];
               ++next[r];
            }

      // Accumulate the upper triangle row by row, then mirror.
      const Vec& d = ldlt_[k]->D();
      Mat& S = Sk_[k];
      S.setZero(pk, pk);
      for (int r = 0; r < nk; ++r) {
         const int b0 = rowptr[r], b1 = rowptr[r + 1];
         if (b0 == b1) continue;
         const double dinv = 1.0 / d[r];
         for (int ib = b0; ib < b1; ++ib) {
            const int b = rcol[ib];
            const double wv = dinv * rval[ib];
            for (int ia = b0; ia <= ib; ++ia) S(rcol[ia], b) -= rval[ia] * wv;
         }
      }
      for (int j = 0; j < pk; ++j)
         for (int i = 0; i < j; ++i) S(j, i) = S(i, j);
   }

   // Build the preconditioner, S_fP, S_PP, Z, T and #neg(T).
   bool build_peel() {
      if (peel_valid_) return peel_ok_;
      peel_valid_ = true;
      peel_ok_ = false;
      ++stats_.factorizations;
      const auto t_start = std::chrono::steady_clock::now();
      struct Timer {
         double& acc;
         std::chrono::steady_clock::time_point t0;
         ~Timer() {
            acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         }
      } timer{stats_.t_peel, t_start};
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      Vec diag_kept(nf);
      for (int a = 0; a < nf; ++a) diag_kept[a] = diagS_[kept_[a]];
      pc_.build(nf, Sk_, Nk_, keptpos_, diag_kept);

      if (opt_.sff_direct > 0 && !factorize_Sff()) return false;

      t_neg_ = 0;
      if (nP == 0) { peel_ok_ = true; return true; }

      // S_fP and S_PP: apply S to the unit vector of each peeled unknown.
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

      // Z = S_ff⁻¹ S_fP, one CG solve per column, warm-started from the previous
      // factorization's Z (the matrices change slowly along the IPOPT path).
      //
      // A column fails if CG did not reach 1e-2, or produced a non-finite
      // iterate, or BROKE DOWN on non-positive curvature: the last case means
      // S_ff is not SPD, so #neg(T) would not be #neg(S) (section 3 of the
      // header) and the factorization must be refused, not used.
      //
      // The columns are independent, so this loop is threaded — it is by far the
      // most expensive part of a factorization.  To keep results identical for
      // any thread count, every column is solved, and the bookkeeping below then
      // behaves exactly as a serial loop that stops at the first failed column.
      auto applyA = [this](const Vec& v, Vec& out) { apply_Sff(v, out); };
      Z_.resize(nf, nP);
      if (Zwarm_.rows() != nf || Zwarm_.cols() != nP) Zwarm_ = Mat::Zero(nf, nP);
      std::vector<long> iters(nP, 0);
      std::vector<char> failed(nP, 0);
      if (opt_.sff_direct > 0) {
         // All columns in one multi-RHS back-solve, judged by the same 1e-2 bar.
         Z_ = sff_ldlt_.solve(SfP_);
         const Mat R = Sff_ * Z_ - SfP_;
         for (int j = 0; j < nP; ++j) {
            const double rel = R.col(j).norm() / std::max(SfP_.col(j).norm(), 1e-300);
            failed[j] = (!(rel < 1e-2) || !Z_.col(j).allFinite()) ? 1 : 0;
         }
      } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int j = 0; j < nP; ++j) {
         Vec z;
         const Vec z0 = Zwarm_.col(j);
         const CgResult r =
            cg_solve(applyA, pc_, Vec(SfP_.col(j)), z, opt_.peel_cg_tol, opt_.cg_maxit, &z0);
         iters[j] = r.iters;
         failed[j] = (!(r.rel < 1e-2) || !z.allFinite() || r.breakdown) ? 1 : 0;
         if (!failed[j]) Z_.col(j) = z;
      }
      }   // CG columns
      int first_failed = nP;
      for (int j = 0; j < nP; ++j)
         if (failed[j]) { first_failed = j; break; }
      const int counted = (first_failed < nP) ? first_failed + 1 : nP;
      for (int j = 0; j < counted; ++j) stats_.iters += iters[j];
      for (int j = 0; j < first_failed; ++j) Zwarm_.col(j) = Z_.col(j);
      if (first_failed < nP) return false;   // no Z ⇒ no T ⇒ no inertia

      // T and its inertia.  Equilibrate first (T mixes wildly different scales)
      // — a congruence D·T·D, which keeps the inertia — then count negative
      // eigenvalues.  An eigenvalue too close to zero means #neg(T) is not
      // well determined: refuse rather than guess.
      const Mat T = SPP - SfP_.transpose() * Z_;
      if (!T.allFinite()) return false;
      Tlu_.compute(T);

      Mat Teq = T;
      double dmax = 0.0;
      for (int i = 0; i < nP; ++i) dmax = std::max(dmax, std::abs(T(i, i)));
      if (dmax > 0.0) {
         Vec s(nP);
         for (int i = 0; i < nP; ++i)
            s[i] = 1.0 / std::sqrt(std::max(std::abs(T(i, i)), 1e-16 * dmax));
         Teq = s.asDiagonal() * T * s.asDiagonal();
      }
      Eigen::SelfAdjointEigenSolver<Mat> es(Teq);
      if (es.info() != Eigen::Success) return false;
      const Vec ev = es.eigenvalues();
      const double scale = std::max(ev.cwiseAbs().maxCoeff(), 1e-300);
      for (int i = 0; i < nP; ++i) {
         if (std::abs(ev[i]) < 1e-12 * scale) return false;
         if (ev[i] < 0.0) ++t_neg_;
      }
      peel_ok_ = true;
      return true;
   }

   // ==========================================================================
   //  Solve pieces
   // ==========================================================================

   // y = A·x using the input triplets (lower triangle + mirror): the true matrix.
   void matvec(const double* x, double* y) const {
      std::fill(y, y + dim_, 0.0);
      for (int t = 0; t < nnz_; ++t) {
         const int i = irow_[t], j = jcol_[t];
         y[i] += vals_[t] * x[j];
         if (i != j) y[j] += vals_[t] * x[i];
      }
   }

   // One arrowhead solve, in place (section 1 of the header).
   bool solve_arrowhead(double* rhs) {
      std::vector<Vec> rk(nsub_), wk(nsub_);
      for (int k = 0; k < nsub_; ++k) rk[k].resize(dimk_[k]);
      Vec ry(p_);
      for (int i = 0; i < dim_; ++i) {
         if (owner_[i] >= 0) rk[owner_[i]][lpos_[i]] = rhs[i];
         else ry[ypos_[i]] = rhs[i];
      }

      // Interface right-hand side:  r_y − Σ_k B_k W_k⁻¹ r_k.
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

      // Interface solve.
      Vec dy;
      if (!interface_solve(ry, dy)) return false;

      // Back-substitute into every tile:  x_k = W_k⁻¹ (r_k − B_kᵀ y).
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

      for (int i = 0; i < dim_; ++i)
         rhs[i] = (owner_[i] >= 0) ? wk[owner_[i]][lpos_[i]] : dy[ypos_[i]];
      return true;
   }

   // Options::sff_direct: assemble S_ff = C_ff + Σ_k (S_k)_ff as a sparse matrix
   // and factorize it.  S_ff should be SPD (that is what the peel is for); a
   // non-positive pivot says it is not, and the factorization is refused.
   bool factorize_Sff() {
      const int nf = (int)kept_.size();
      std::vector<Trip> t;
      for (const auto& e : ctrip_) {
         const int i = keptpos_[e.r], j = keptpos_[e.c];
         if (i < 0 || j < 0) continue;
         t.emplace_back(i, j, vals_[e.t]);
         if (i != j) t.emplace_back(j, i, vals_[e.t]);
      }
      for (int k = 0; k < nsub_; ++k) {
         const int pk = (int)Nk_[k].size();
         for (int a = 0; a < pk; ++a) {
            const int i = keptpos_[Nk_[k][a]];
            if (i < 0) continue;
            for (int b = 0; b < pk; ++b) {
               const int j = keptpos_[Nk_[k][b]];
               if (j >= 0) t.emplace_back(i, j, Sk_[k](a, b));
            }
         }
      }
      Sff_.resize(nf, nf);
      Sff_.setFromTriplets(t.begin(), t.end());
      Sff_.makeCompressed();
      if (!sff_analyzed_) { sff_ldlt_.analyzePattern(Sff_); sff_analyzed_ = true; }
      sff_ldlt_.factorize(Sff_);
      if (sff_ldlt_.info() != Eigen::Success) return false;
      const Vec d = sff_ldlt_.vectorD();
      for (int i = 0; i < nf; ++i)
         if (!(d[i] > 0.0)) return false;
      return true;
   }

   // S·y = C·y + Σ_k (scatter) S_k (gather) y — never forms S.
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

   // S_ff·v: embed v on the kept unknowns (zeros on the peeled ones), apply S,
   // read the kept unknowns back.
   void apply_Sff(const Vec& v, Vec& out) const {
      if (peel_.empty()) { apply_S(v, out); return; }
      Vec full = Vec::Zero(p_);
      for (size_t a = 0; a < kept_.size(); ++a) full[kept_[a]] = v[(int)a];
      Vec Sfull;
      apply_S(full, Sfull);
      out.resize((int)kept_.size());
      for (size_t a = 0; a < kept_.size(); ++a) out[(int)a] = Sfull[kept_[a]];
   }

   // S·dy = ry via the peel (section 3 of the header).  An answer whose residual
   // is above 1e-2 is counted as rejected but still returned: the refinement in
   // solve() judges it against the true matrix.
   bool interface_solve(const Vec& ry, Vec& dy) {
      ++stats_.solves;
      const int nf = (int)kept_.size(), nP = (int)peel_.size();

      Vec rf(nf);
      for (int a = 0; a < nf; ++a) rf[a] = ry[kept_[a]];
      Vec g;                                                   // g = S_ff⁻¹ r_f
      CgResult r;
      if (opt_.sff_direct >= 2) {
         g = sff_ldlt_.solve(rf);
         r.rel = (Sff_ * g - rf).norm() / std::max(rf.norm(), 1e-300);
      } else {
         auto applyA = [this](const Vec& v, Vec& out) { apply_Sff(v, out); };
         r = cg_solve(applyA, pc_, rf, g, opt_.cg_tol, opt_.cg_maxit);
      }
      stats_.iters += r.iters;

      dy.resize(p_);
      if (nP == 0) {
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = g[a];
      } else {
         Vec rP(nP);
         for (int j = 0; j < nP; ++j) rP[j] = ry[peel_[j]];
         const Vec dP = Tlu_.solve(rP - SfP_.transpose() * g);   // y_P
         const Vec df = g - Z_ * dP;                             // y_f
         for (int a = 0; a < nf; ++a) dy[kept_[a]] = df[a];
         for (int j = 0; j < nP; ++j) dy[peel_[j]] = dP[j];
      }

      Vec Sdy;
      apply_S(dy, Sdy);
      const double true_rel = (ry - Sdy).norm() / std::max(ry.norm(), 1e-300);
      if (!(r.rel < 1e-2) || !(true_rel < 1e-2) || !dy.allFinite()) ++stats_.rejected;
      return dy.allFinite();
   }

   // ---- problem -------------------------------------------------------------
   int dim_ = 0, nnz_ = 0, nsub_ = 0, p_ = 0;
   Options opt_;
   std::vector<int> irow_, jcol_, owner_;
   std::vector<double> vals_;

   // ---- numbering and routing (pattern only) ---------------------------------
   std::vector<int> ypos_, lpos_, dimk_;
   std::vector<std::vector<int>> Nk_;                 // border unknowns tile k touches
   std::vector<std::vector<std::vector<int>>> reach_; // forward-solve reaches
   std::vector<std::vector<Entry>> wtrip_, btrip_;
   std::vector<Entry> ctrip_;

   // ---- per factorization ----------------------------------------------------
   std::vector<SpMat> W_, B_;
   std::vector<Mat> Sk_;
   std::vector<std::unique_ptr<Ldlt>> ldlt_;
   SpMat C_;
   Vec diagS_;
   int n_neg_ = 0, t_neg_ = 0;

   // ---- peel -----------------------------------------------------------------
   std::vector<int> peel_, kept_, keptpos_;
   Precond pc_;
   Mat SfP_, Z_, Zwarm_;
   Eigen::PartialPivLU<Mat> Tlu_;
   SpMat Sff_;                                  // Options::sff_direct only
   Eigen::SimplicialLDLT<SpMat> sff_ldlt_;
   bool sff_analyzed_ = false;
   bool peel_valid_ = false, peel_ok_ = false;
   Stats stats_;
};

}  // namespace dd
