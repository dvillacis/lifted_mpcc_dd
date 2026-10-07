// precond.hpp — the additive Schwarz preconditioner and PCG for the interface
// system S u_y = r_S (Lueg et al., eqs. 18–19):
//
//     M⁻¹ r = Σ_k N_k S̃_k⁻¹ N_kᵀ r,    S̃_k = N_kᵀ S N_k
//
// S̃_k is the "local assembled Schur complement": S restricted to the border
// unknowns tile k touches, including the contributions of every other tile.
#pragma once

#include <algorithm>
#include <vector>

#include "blocks.hpp"

namespace dd {

// =============================================================================
//  Schwarz — one dense block per tile, each on the border indices idx_k.
//  SPD blocks use Cholesky.  A block that is not SPD makes set() return
//  false; with clip, its eigenvalues are clipped to be positive (CG needs an
//  SPD preconditioner), without, the block is left unusable.  The AS blocks
//  S̃_k = N_kᵀ S N_k are principal submatrices of S, so one that is not SPD
//  proves S is not positive definite: SchurDD then refuses the factorization
//  without running PCG, and needs no clipping.  set() may be called for
//  different k in parallel; apply_blocks() is parallel over blocks.
// =============================================================================
class Schwarz {
public:
   void reset(int nblocks) { blocks_.assign(nblocks, Block()); }
   // Block k on the border indices idx (only the blocks this rank owns are set).
   bool set(int k, const std::vector<int>& idx, const Mat& M, bool clip) {
      Block& b = blocks_[k];
      b.idx = idx;
      if (idx.empty()) return true;
      Eigen::LLT<Mat> chol(M);
      if (chol.info() == Eigen::Success) {
         b.llt = std::move(chol);
         b.spd = true;
      } else if (!clip) {
         b.spd = false;
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
   // out + off[k] ← M_k⁻¹ N_kᵀ r for the blocks in [k0, k1) (block k has
   // off[k+1] − off[k] unknowns, or none).  The caller adds the outputs up
   // over the border (SchurDD::apply_M), so that tiles owned by other ranks
   // can contribute too.
   void apply_blocks(const Vec& r, int k0, int k1, double* out, const int* off) {
#pragma omp parallel for schedule(dynamic, 4)
      for (int k = k0; k < k1; ++k) {
         Block& b = blocks_[k];
         const int m = (int)b.idx.size();
         if (m == 0) continue;
         Eigen::Map<Vec> rk(out + off[k], m);
         for (int i = 0; i < m; ++i) rk[i] = r[b.idx[i]];
         if (b.spd) {
            b.llt.solveInPlace(rk);
         } else {
            const Vec t = b.inv * rk;
            rk = t;
         }
      }
   }

private:
   struct Block {
      std::vector<int> idx;
      Eigen::LLT<Mat> llt;
      Mat inv;
      bool spd = true;
   };
   std::vector<Block> blocks_;
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
