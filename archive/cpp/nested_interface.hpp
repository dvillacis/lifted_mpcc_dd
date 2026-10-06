// A SECOND arrowhead level, applied to the DD interface matrix S.
//
// Why this exists. The outer solver answers IPOPT's inertia query by Haynsworth,
// In(A) = Σ_k In(W_k) + In(S), and never factorizes the full KKT. But it does
// factorize S directly, once per Newton step, on one thread. Measured, that is
// the serial floor: fitting the S-fact timer over constant-subdomain-size pairs
// gives S-fact ∝ p^1.5 (exponents 1.52 / 1.42 / 1.49), and in the regime large
// images force — hold the subdomain size fixed, grow the tile count with N — the
// interface grows as p ~ 4(k−1)N ~ N²/H while the parallel work only grows as N².
// So the serial term scales like N³ against N² and eventually owns the run.
//
// The fix is the identity itself, applied again. Partition the border into g
// groups plus a level-2 separator; then
//
//     In(S) = Σ_j In(S_j) + In(S'),     S' = C' − Σ_j E_j S_j⁻¹ E_jᵀ
//
// with the S_j independent (parallel) and S' the only serial piece. Measured on
// dumped interfaces, the geometric separator is 12.8% of p at k=8 and 25.2% at
// k=4 — it scales like O(N) while p scales like O(kN), so the deeper the tiling
// the more this pays, which is exactly the large-image regime. The inertia stays
// EXACT: no inertia-free curvature test, no iteration-count penalty.
//
// PROTOTYPE SCOPE (--nested, default off): one extra level, not recursion, and
// S' is factorized directly. X_j = S_j⁻¹E_jᵀ is formed densely (n_j × n_s), which
// is what bounds the size this is useful at; recursing instead of widening is the
// way out, and is the reason the algebra here is written against a generic group
// map rather than the tile geometry.
#ifndef NESTED_INTERFACE_HPP
#define NESTED_INTERFACE_HPP

#include <algorithm>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

class NestedInterface {
public:
   using SpMat = Eigen::SparseMatrix<double>;
   enum { OK = 0, SINGULAR = 1, RESOURCE = 2 };

   int groups() const { return ng_; }
   int separator() const { return ns_; }
   bool active() const { return ng_ > 0; }
   int negative_eigenvalues() const { return nneg_; }

   // Freeze the structure. S is the FULL symmetric interface matrix (both
   // triangles stored, as Ssp_ is); grp[b] is the level-1 group of border
   // position b, or −1 for the level-2 separator. Returns false if the map is
   // unusable, in which case the caller must fall back to the flat path.
   bool analyze(const SpMat& S, const std::vector<int>& grp, int ng) {
      const int p = (int)S.rows();
      if (ng <= 0 || (int)grp.size() != p) return false;
      ng_ = ng;
      grp_ = grp;
      loc_.assign(p, -1);
      idx_.assign(ng, {});
      sep_.clear();
      for (int b = 0; b < p; ++b) {
         const int g = grp_[b];
         if (g < 0) { loc_[b] = (int)sep_.size(); sep_.push_back(b); }
         else if (g < ng) { loc_[b] = (int)idx_[g].size(); idx_[g].push_back(b); }
         else return false;
      }
      ns_ = (int)sep_.size();
      // A group that never appears, or an empty separator, means the caller's
      // geometry did not actually split anything — the flat path is then both
      // simpler and faster, so refuse rather than silently degrade.
      if (ns_ == 0) return false;
      for (int j = 0; j < ng; ++j) if (idx_[j].empty()) return false;

      blk_.clear();
      blk_.resize(ng);
      src_.assign(ng, {});
      Esrc_.assign(ng, {});
      E_.assign(ng, SpMat());
      Csrc_.clear();
      Cij_.clear();

      // One walk over S classifies every stored entry: inside a group (its own
      // block), group×separator (the coupling E_j), or separator×separator (C').
      // Everything is recorded as a SLOT into S.valuePtr() so the per-Newton-step
      // refresh is a gather, never a re-scan.
      std::vector<std::vector<Eigen::Triplet<double>>> tE(ng);
      std::vector<std::vector<int>> irn(ng), jcn(ng);
      for (int c = 0; c < p; ++c) {
         for (SpMat::InnerIterator it(S, c); it; ++it) {
            const int r = (int)it.row();
            const int slot = (int)(&it.value() - S.valuePtr());
            const int gr = grp_[r], gc = grp_[c];
            if (gr >= 0 && gc >= 0 && gr != gc) {
               // THE correctness condition, checked rather than assumed: removing
               // the separator must disconnect the groups. If any S entry couples
               // two of them, S is not block-diagonal on the groups, the Schur
               // complement below is not S', and the inertia would be quietly
               // wrong — the worst possible failure here. A caller whose geometry
               // does not line up (e.g. a coarse split that does not fall on a
               // fine tile boundary, so a subdomain straddles two groups) gets a
               // refusal and the validated flat path, not a wrong answer.
               std::cerr << "[nested] groups " << gr << " and " << gc
                         << " are coupled in S — the level-2 separator does not "
                            "disconnect them; refusing\n";
               return false;
            }
            if (gr >= 0 && gr == gc) {
               if (loc_[r] >= loc_[c]) {          // one triangle, local indices
                  irn[gr].push_back(loc_[r] + 1);
                  jcn[gr].push_back(loc_[c] + 1);
                  src_[gr].push_back(slot);
               }
            } else if (gr >= 0 && gc < 0) {       // E_j row=sep col=group
               tE[gr].push_back(Eigen::Triplet<double>(loc_[c], loc_[r], 0.0));
               Esrc_[gr].push_back(slot);
            } else if (gr < 0 && gc < 0) {        // C' (dense, small)
               Cij_.push_back({loc_[r], loc_[c]});
               Csrc_.push_back(slot);
            }
            // gr<0 && gc>=0 is the mirror of the E case above; skipping it is
            // what makes E_j appear exactly once.
         }
      }
      for (int j = 0; j < ng; ++j) {
         const int nj = (int)idx_[j].size();
         E_[j].resize(ns_, nj);
         E_[j].setFromTriplets(tE[j].begin(), tE[j].end());
         E_[j].makeCompressed();
         // setFromTriplets sorts, so recover where each recorded slot landed.
         const int* outer = E_[j].outerIndexPtr();
         const int* inner = E_[j].innerIndexPtr();
         std::vector<int> reordered(Esrc_[j].size());
         for (size_t e = 0; e < tE[j].size(); ++e) {
            const int r = tE[j][e].row(), c = tE[j][e].col();
            const int* lo = inner + outer[c];
            const int* hi = inner + outer[c + 1];
            const int* p2 = std::lower_bound(lo, hi, r);
            if (p2 == hi || *p2 != r) return false;
            reordered[(size_t)(p2 - inner)] = Esrc_[j][e];
         }
         Esrc_[j].swap(reordered);

         blk_[j].reset(new SymBlock());
         blk_[j]->scaling_off();   // partial solves are not used, but keep the
                                   // blocks on the same footing as the W_k
         if (!blk_[j]->analyze(nj, irn[j], jcn[j])) {
            std::cerr << "[nested] analysis failed for S_" << j << "\n";
            return false;
         }
      }
      // S' is dense and small; give MA57 the full lower triangle once.
      {
         std::vector<int> irs, jcs;
         irs.reserve((size_t)ns_ * (ns_ + 1) / 2);
         jcs.reserve((size_t)ns_ * (ns_ + 1) / 2);
         for (int c = 0; c < ns_; ++c)
            for (int r = c; r < ns_; ++r) { irs.push_back(r + 1); jcs.push_back(c + 1); }
         sp_.reset(new SymBlock());
         sp_->scaling_off();
         if (!sp_->analyze(ns_, irs, jcs)) {
            std::cerr << "[nested] analysis failed for S'\n";
            return false;
         }
      }
      Cp_.setZero(ns_, ns_);
      X_.assign(ng, Eigen::MatrixXd());
      rj_.assign(ng, Eigen::VectorXd());
      wj_.assign(ng, Eigen::VectorXd());
      cj_.assign(ng, Eigen::VectorXd());
      for (int j = 0; j < ng; ++j) {
         const int nj = (int)idx_[j].size();
         rj_[j].resize(nj); wj_[j].resize(nj); cj_[j].resize(ns_);
      }
      rs_.resize(ns_);
      return true;
   }

   // Refresh from S, factorize the blocks, form and factorize S'.
   int factorize(const SpMat& S) {
      const double* sv = S.valuePtr();
      for (int j = 0; j < ng_; ++j) {
         double* a = blk_[j]->values();
         const std::vector<int>& sr = src_[j];
         for (size_t e = 0; e < sr.size(); ++e) a[e] = sv[sr[e]];
         double* ev = E_[j].valuePtr();
         const std::vector<int>& er = Esrc_[j];
         for (size_t e = 0; e < er.size(); ++e) ev[e] = sv[er[e]];
      }
      Cp_.setZero();
      for (size_t e = 0; e < Csrc_.size(); ++e)
         Cp_(Cij_[e][0], Cij_[e][1]) = sv[Csrc_[e]];

      // Independent per group — the whole point of the level.
      std::vector<char> ok(ng_, 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int j = 0; j < ng_; ++j) {
         ok[j] = (blk_[j]->factorize() && !blk_[j]->singular()) ? 1 : 0;
         if (ok[j]) {
            X_[j] = Eigen::MatrixXd(E_[j].transpose());   // n_j × n_s
            blk_[j]->solve(X_[j].data(), ns_);            // S_j⁻¹E_jᵀ
         }
      }
      nneg_ = 0;
      for (int j = 0; j < ng_; ++j) {
         if (!ok[j]) return blk_[j]->resource_failure() ? RESOURCE : SINGULAR;
         if (blk_[j]->solve_failed()) return SINGULAR;
         nneg_ += blk_[j]->negative_eigenvalues();
      }
      // S' = C' − Σ_j E_j S_j⁻¹ E_jᵀ   (serial: every j writes all of S')
      for (int j = 0; j < ng_; ++j) Cp_.noalias() -= E_[j] * X_[j];

      {
         double* a = sp_->values();
         size_t e = 0;
         for (int c = 0; c < ns_; ++c)
            for (int r = c; r < ns_; ++r) a[e++] = Cp_(r, c);
      }
      if (!sp_->factorize() || sp_->singular())
         return sp_->resource_failure() ? RESOURCE : SINGULAR;
      nneg_ += sp_->negative_eigenvalues();
      return OK;
   }

   // Arrowhead solve on S, in place, y of length p.
   bool solve(double* y) {
      for (int j = 0; j < ng_; ++j) {
         const std::vector<int>& ix = idx_[j];
         for (size_t a = 0; a < ix.size(); ++a) rj_[j][(int)a] = y[ix[a]];
      }
      for (int a = 0; a < ns_; ++a) rs_[a] = y[sep_[a]];
      // r_s = y_s − Σ_j E_j S_j⁻¹ y_j : solves parallel, accumulation serial
      // (every j contributes to all of r_s), exactly as at the outer level.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int j = 0; j < ng_; ++j) {
         wj_[j] = rj_[j];
         if (!blk_[j]->solve(wj_[j].data(), 1)) wj_[j].setZero();
         cj_[j].noalias() = E_[j] * wj_[j];
      }
      for (int j = 0; j < ng_; ++j) rs_ -= cj_[j];

      if (!sp_->solve(rs_.data(), 1)) return false;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int j = 0; j < ng_; ++j) {
         wj_[j].noalias() = rj_[j] - E_[j].transpose() * rs_;
         if (!blk_[j]->solve(wj_[j].data(), 1)) wj_[j].setZero();
      }
      for (int j = 0; j < ng_; ++j) {
         const std::vector<int>& ix = idx_[j];
         for (size_t a = 0; a < ix.size(); ++a) y[ix[a]] = wj_[j][(int)a];
      }
      for (int a = 0; a < ns_; ++a) y[sep_[a]] = rs_[a];
      return true;
   }

private:
   int ng_ = 0, ns_ = 0, nneg_ = 0;
   std::vector<int> grp_, loc_, sep_;
   std::vector<std::vector<int>> idx_;
   std::vector<std::unique_ptr<SymBlock>> blk_;   // S_j
   std::unique_ptr<SymBlock> sp_;                 // S'
   std::vector<std::vector<int>> src_, Esrc_;     // value slots into S
   std::vector<SpMat> E_;
   std::vector<std::array<int, 2>> Cij_;
   std::vector<int> Csrc_;
   Eigen::MatrixXd Cp_;
   std::vector<Eigen::MatrixXd> X_;
   std::vector<Eigen::VectorXd> rj_, wj_, cj_;
   Eigen::VectorXd rs_;
};

#endif  // NESTED_INTERFACE_HPP
