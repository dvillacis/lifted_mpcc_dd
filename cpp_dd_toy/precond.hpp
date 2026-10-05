// precond.hpp — preconditioners for the interface system S u_y = r_S, and PCG.
//
//   Schwarz  one level:  M⁻¹ r = Σ_k N_k M_k⁻¹ N_kᵀ r           (paper 17, 19, 21)
//   Coarse   the coarse term of the two-level method              (paper 22)
//               Q r = Z S₀⁻¹ Zᵀ r,   S₀ = Zᵀ S Z
//            where the columns of Z span a coarse space (here: one column per
//            interface "face", optionally enriched with local eigenvectors).
//
//   Combined:
//     additive   M⁻¹ + Q                                           (paper 22)
//     balanced   Pᵀ M⁻¹ P + Q,  P = I − S Q                         (balancing
//                Neumann–Neumann form: symmetric, removes the coarse components
//                exactly; costs two more products with S per iteration)
#pragma once

#include <algorithm>
#include <vector>

#include "blocks.hpp"

namespace dd {

// =============================================================================
//  Schwarz — one dense block per tile, each on the border indices idx_k.
//  SPD blocks use Cholesky; others have their eigenvalues clipped to be
//  positive (CG needs an SPD preconditioner).  set() may be called for
//  different k in parallel; apply() is parallel over blocks, then gathers.
// =============================================================================
class Schwarz {
public:
   void reset(int p, int nblocks) {
      p_ = p;
      blocks_.assign(nblocks, Block());
   }
   bool set(int k, const std::vector<int>& idx, const Mat& M) {
      Block& b = blocks_[k];
      b.idx = idx;
      if (idx.empty()) return true;
      Eigen::LLT<Mat> chol(M);
      if (chol.info() == Eigen::Success) {
         b.llt = std::move(chol);
         b.spd = true;
      } else {
         Eigen::SelfAdjointEigenSolver<Mat> es(M);
         Vec d = es.eigenvalues();
         const double scale = std::max(d.cwiseAbs().maxCoeff(), 1e-300);
         for (int i = 0; i < d.size(); ++i) d[i] = 1.0 / std::max(d[i], 1e-8 * scale);
         b.inv = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
         b.spd = false;
      }
      return b.spd;
   }
   // After all set(): the inverse map border index → (block, position).
   void finalize() {
      start_.assign(p_ + 1, 0);
      for (const Block& b : blocks_)
         for (int i : b.idx) ++start_[i + 1];
      for (int i = 0; i < p_; ++i) start_[i + 1] += start_[i];
      at_.assign(start_[p_], {0, 0});
      std::vector<int> fill(start_.begin(), start_.end() - 1);
      for (int k = 0; k < (int)blocks_.size(); ++k)
         for (int a = 0; a < (int)blocks_[k].idx.size(); ++a)
            at_[fill[blocks_[k].idx[a]]++] = {k, a};
      used_ = start_[p_] > 0;
   }
   bool empty() const { return !used_; }

   void apply(const Vec& r, Vec& z) const {
      if (!used_) { z = r; return; }
      const int K = (int)blocks_.size();
#pragma omp parallel for schedule(dynamic, 4)
      for (int k = 0; k < K; ++k) {
         const Block& b = blocks_[k];
         const int m = (int)b.idx.size();
         if (m == 0) continue;
         Vec rk(m);
         for (int i = 0; i < m; ++i) rk[i] = r[b.idx[i]];
         b.out = b.spd ? Vec(b.llt.solve(rk)) : Vec(b.inv * rk);
      }
      z.resize(p_);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p_; ++i) {
         double s = 0.0;
         for (int q = start_[i]; q < start_[i + 1]; ++q) s += blocks_[at_[q][0]].out[at_[q][1]];
         z[i] = start_[i] == start_[i + 1] ? r[i] : s;   // uncovered index: identity
      }
   }

private:
   struct Block {
      std::vector<int> idx;
      Eigen::LLT<Mat> llt;
      Mat inv;
      bool spd = true;
      mutable Vec out;
   };
   int p_ = 0;
   bool used_ = false;
   std::vector<Block> blocks_;
   std::vector<int> start_;
   std::vector<std::array<int, 2>> at_;
};

// =============================================================================
//  Coarse — Q r = Z S₀⁻¹ Zᵀ r.  S₀ is small: dense Cholesky up to a few
//  thousand coarse unknowns, sparse Cholesky beyond.
// =============================================================================
class Coarse {
public:
   // false if S₀ is not positive definite (then the coarse term is skipped).
   bool build(const SpMat& Z, const SpMat& S0) {
      Z_ = Z;
      n0_ = (int)S0.rows();
      ok_ = false;
      if (n0_ == 0) return false;
      dense_ = n0_ <= 4000;
      if (dense_) {
         dl_.compute(Mat(S0));
         ok_ = dl_.info() == Eigen::Success;
      } else {
         sl_.compute(S0);
         ok_ = sl_.info() == Eigen::Success;
      }
      return ok_;
   }
   bool ok() const { return ok_; }
   int dim() const { return n0_; }
   void apply(const Vec& r, Vec& out) const {   // out = Q r
      const Vec r0 = Z_.transpose() * r;
      const Vec x0 = dense_ ? Vec(dl_.solve(r0)) : Vec(sl_.solve(r0));
      out = Z_ * x0;
   }

private:
   SpMat Z_;
   int n0_ = 0;
   bool ok_ = false, dense_ = true;
   Eigen::LLT<Mat> dl_;
   Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> sl_;
};

// =============================================================================
//  pcg — preconditioned conjugate gradients from x = 0.
//  Stops at ‖r‖ ≤ tol·‖b‖ or maxit; pᵀSp ≤ 0 stops it with breakdown = true
//  (S is not positive definite: the paper's Sec. 3.5 check).
// =============================================================================
struct CgResult {
   int iters = 0;
   double rel = 0.0;
   bool breakdown = false;
};

template <class ApplyS, class ApplyM>
CgResult pcg(ApplyS&& applyS, ApplyM&& applyM, const Vec& b, Vec& x, double tol, int maxit) {
   const int n = (int)b.size();
   CgResult res;
   x.setZero(n);
   const double bnorm = b.norm();
   if (bnorm == 0.0) return res;
   Vec r = b, z(n), Ap(n);
   applyM(r, z);
   Vec p = z;
   double rz = r.dot(z);
   for (; res.iters < maxit; ++res.iters) {
      if (r.norm() <= tol * bnorm) break;
      applyS(p, Ap);
      const double pAp = p.dot(Ap);
      if (!(pAp > 0.0)) { res.breakdown = true; break; }
      const double a = rz / pAp;
      x += a * p;
      r -= a * Ap;
      applyM(r, z);
      const double rz_new = r.dot(z);
      p = z + (rz_new / rz) * p;
      rz = rz_new;
   }
   res.rel = r.norm() / bnorm;
   return res;
}

}  // namespace dd
