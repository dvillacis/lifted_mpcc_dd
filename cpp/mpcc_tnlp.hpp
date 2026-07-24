// The uniform-grid unit-ball lifted TV-MPCC as an Ipopt::TNLP — a C++ port of
// ../lifted_mpcc_unitball_v2.py's LiftedTVMPCC.
//
//   x = [u | qx | qy | r | δ | θ | α],           n     = 6m+1
//   rows h1,h2x,h2y,h3x,h3y,hr,hd,comp,          m_con = 8m        (m = N²)
//
//   h1  : u − f + e^α·(Kxᵀqx + Kyᵀqy) = 0     ← Q(α)=e^α sits in the STATE row
//   h2x : (Kx u) − r·cosθ = 0                    h2y : (Ky u) − r·sinθ = 0
//   h3x : qx − δ·cosθ = 0                        h3y : qy − δ·sinθ = 0
//   hr  : r ≥ 0        hd : δ ≥ 0             ← explicit inequality ROWS, not boxes
//   comp: r·(1−δ) − t ≤ 0                     ← Scholtes, written directly on δ
//
// There is no slack w and no h4: because the ball is the plain box δ ≤ 1, the
// matching slack would be the α-free affine w = 1−δ, which is redundant. Only
// δ ≤ 1 and the α box remain as variable bounds.
//
// The Scholtes level t enters the comp row additively (fixed upper bound 0), so it
// changes no derivative and the continuation driver can just write t_ between
// levels. Exact Lagrangian Hessian is provided — the DD solver needs it.
//
// 20 Jacobian pieces and 8 Hessian blocks, assembled in the SAME order as their
// structure arrays (as in the Python, they are matched positionally, not by key).
// Objective, bounds and warm-start bookkeeping live in MpccTNLPBase.
#ifndef MPCC_TNLP_HPP
#define MPCC_TNLP_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "mpcc_base.hpp"

namespace Ipopt {

struct Tri { int r, c; double v; };

class MpccTNLP : public MpccTNLPBase {
public:
   int N = 0, m = 0;
   double alpha0_ = -2.66;

   // row offsets (rhr/rcomp live in the base — the shared bounds need them)
   int rh1, rh2x, rh2y, rh3x, rh3y, rhd;

   std::vector<Tri> Kx_, Ky_, KxT_, KyT_;

   explicit MpccTNLP(const std::string& datafile, const image_io::Opts& opt = {}) {
      image_io::load_data(datafile, opt, N, uclean_, f_);
      m = N * N; n = 6 * m + 1; mcon = 8 * m;
      n_state = m; n_lift = m;
      weight_exp = true; has_ha = false;         // Q(α) = e^α, α boxed [−15, 15]
      ou = 0; oqx = m; oqy = 2 * m; oR = 3 * m; oD = 4 * m; oTh = 5 * m; oa = 6 * m;
      rh1 = 0; rh2x = m; rh2y = 2 * m; rh3x = 3 * m; rh3y = 4 * m;
      rhr = 5 * m; rhd = 6 * m; rcomp = 7 * m;
      n_eq = rhr;
      n_ineq = mcon - n_eq;
      kkt_dim = n + 2 * n_ineq + n_eq;           // = 17m+1
      build_grad_ops();
      build_structures();
   }

   // ---- forward differences with Neumann (no-flux) boundary ---------------
   // C-order p = i*N + j. Kx differences along columns, Ky along rows; the last
   // column/row of D is zeroed, so qx on column N−1 and qy on row N−1 never enter
   // h1 at all (they stay strictly local — this is what the DD partition relies on).
   void build_grad_ops() {
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            int p = i * N + j;
            if (j < N - 1) { Kx_.push_back({p, p, -1.0}); Kx_.push_back({p, p + 1, 1.0}); }
            if (i < N - 1) { Ky_.push_back({p, p, -1.0}); Ky_.push_back({p, p + N, 1.0}); }
         }
      for (const auto& t : Kx_) KxT_.push_back({t.c, t.r, t.v});
      for (const auto& t : Ky_) KyT_.push_back({t.c, t.r, t.v});
      auto bycol = [](const Tri& a, const Tri& b) {
         return a.r != b.r ? a.r < b.r : a.c < b.c;
      };
      std::sort(KxT_.begin(), KxT_.end(), bycol);
      std::sort(KyT_.begin(), KyT_.end(), bycol);
   }

   // ---- helpers ----------------------------------------------------------
   void divq(const double* x, std::vector<double>& out) const {
      out.assign(m, 0.0);
      for (const auto& t : KxT_) out[t.r] += t.v * x[oqx + t.c];
      for (const auto& t : KyT_) out[t.r] += t.v * x[oqy + t.c];
   }
   void applyK(const std::vector<Tri>& K, const double* v, std::vector<double>& out) const {
      out.assign(m, 0.0);
      for (const auto& t : K) out[t.r] += t.v * v[t.c];
   }

   // ---- structures (20 Jacobian pieces / 8 Hessian blocks) ---------------
   void build_structures() {
      auto J = [&](int r, int c) { jr_.push_back(r); jc_.push_back(c); };
      for (int i = 0; i < m; ++i) J(rh1 + i, ou + i);                       // 1
      for (const auto& t : KxT_) J(rh1 + t.r, oqx + t.c);                   // 2
      for (const auto& t : KyT_) J(rh1 + t.r, oqy + t.c);                   // 3
      for (int i = 0; i < m; ++i) J(rh1 + i, oa);                           // 4 (dense col)
      for (const auto& t : Kx_) J(rh2x + t.r, ou + t.c);                    // 5
      for (int i = 0; i < m; ++i) J(rh2x + i, oR + i);                      // 6
      for (int i = 0; i < m; ++i) J(rh2x + i, oTh + i);                     // 7
      for (const auto& t : Ky_) J(rh2y + t.r, ou + t.c);                    // 8
      for (int i = 0; i < m; ++i) J(rh2y + i, oR + i);                      // 9
      for (int i = 0; i < m; ++i) J(rh2y + i, oTh + i);                     // 10
      for (int i = 0; i < m; ++i) J(rh3x + i, oqx + i);                     // 11
      for (int i = 0; i < m; ++i) J(rh3x + i, oD + i);                      // 12
      for (int i = 0; i < m; ++i) J(rh3x + i, oTh + i);                     // 13
      for (int i = 0; i < m; ++i) J(rh3y + i, oqy + i);                     // 14
      for (int i = 0; i < m; ++i) J(rh3y + i, oD + i);                      // 15
      for (int i = 0; i < m; ++i) J(rh3y + i, oTh + i);                     // 16
      for (int i = 0; i < m; ++i) J(rcomp + i, oR + i);                     // 17
      for (int i = 0; i < m; ++i) J(rcomp + i, oD + i);                     // 18
      for (int i = 0; i < m; ++i) J(rhr + i, oR + i);                       // 19
      for (int i = 0; i < m; ++i) J(rhd + i, oD + i);                       // 20

      auto H = [&](int r, int c) { hr_.push_back(r); hc_.push_back(c); };
      for (int i = 0; i < m; ++i) H(ou + i, ou + i);                        // (u,u)
      for (int i = 0; i < m; ++i) H(oTh + i, oR + i);                       // (θ,r)
      for (int i = 0; i < m; ++i) H(oTh + i, oD + i);                       // (θ,δ)
      for (int i = 0; i < m; ++i) H(oTh + i, oTh + i);                      // (θ,θ)
      for (int i = 0; i < m; ++i) H(oD + i, oR + i);                        // (δ,r) comp cross
      for (int i = 0; i < m; ++i) H(oa, oqx + i);                           // (α,qx)
      for (int i = 0; i < m; ++i) H(oa, oqy + i);                           // (α,qy)
      H(oa, oa);                                                            // (α,α)
   }

   // u = f, q = 0 ⇒ h1 holds and δ = 0; only complementarity is violated. This is
   // the spurious-branch-prone cold start of the Python (`--init cold`); this
   // driver's job is solver benchmarking, not basin selection, so the reference
   // α/PSNR comparisons are run from a dumped Python CP point.
   void cold_start(Number* x) const {
      std::vector<double> gx, gy;
      applyK(Kx_, f_.data(), gx);
      applyK(Ky_, f_.data(), gy);
      for (int i = 0; i < m; ++i) {
         x[ou + i] = f_[i];
         x[oqx + i] = 0.0; x[oqy + i] = 0.0;
         x[oR + i] = std::hypot(gx[i], gy[i]);
         x[oD + i] = 0.0;
         x[oTh + i] = std::atan2(gy[i], gx[i]);
      }
      x[oa] = alpha0_;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      const double ea = std::exp(x[oa]);
      std::vector<double> dq, kxu, kyu;
      divq(x, dq);
      applyK(Kx_, x + ou, kxu);
      applyK(Ky_, x + ou, kyu);
      for (int i = 0; i < m; ++i) {
         const double c = std::cos(x[oTh + i]), s = std::sin(x[oTh + i]);
         const double r = x[oR + i], d = x[oD + i];
         g[rh1 + i]   = x[ou + i] - f_[i] + ea * dq[i];
         g[rh2x + i]  = kxu[i] - r * c;
         g[rh2y + i]  = kyu[i] - r * s;
         g[rh3x + i]  = x[oqx + i] - d * c;
         g[rh3y + i]  = x[oqy + i] - d * s;
         g[rhr + i]   = r;
         g[rhd + i]   = d;
         g[rcomp + i] = r * (1.0 - d) - t_;
      }
      return true;
   }

   bool eval_jac_g(Index, const Number* x, bool, Index, Index nele, Index* iRow,
                   Index* jCol, Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = jr_[k]; jCol[k] = jc_[k]; }
         return true;
      }
      const double ea = std::exp(x[oa]);
      std::vector<double> dq;
      divq(x, dq);
      Index k = 0;
      for (int i = 0; i < m; ++i) values[k++] = 1.0;                        // 1
      for (const auto& t : KxT_) values[k++] = ea * t.v;                    // 2
      for (const auto& t : KyT_) values[k++] = ea * t.v;                    // 3
      for (int i = 0; i < m; ++i) values[k++] = ea * dq[i];                 // 4
      for (const auto& t : Kx_) values[k++] = t.v;                          // 5
      for (int i = 0; i < m; ++i) values[k++] = -std::cos(x[oTh + i]);      // 6
      for (int i = 0; i < m; ++i) values[k++] = x[oR + i] * std::sin(x[oTh + i]);  // 7
      for (const auto& t : Ky_) values[k++] = t.v;                          // 8
      for (int i = 0; i < m; ++i) values[k++] = -std::sin(x[oTh + i]);      // 9
      for (int i = 0; i < m; ++i) values[k++] = -x[oR + i] * std::cos(x[oTh + i]); // 10
      for (int i = 0; i < m; ++i) values[k++] = 1.0;                        // 11
      for (int i = 0; i < m; ++i) values[k++] = -std::cos(x[oTh + i]);      // 12
      for (int i = 0; i < m; ++i) values[k++] = x[oD + i] * std::sin(x[oTh + i]);  // 13
      for (int i = 0; i < m; ++i) values[k++] = 1.0;                        // 14
      for (int i = 0; i < m; ++i) values[k++] = -std::sin(x[oTh + i]);      // 15
      for (int i = 0; i < m; ++i) values[k++] = -x[oD + i] * std::cos(x[oTh + i]); // 16
      for (int i = 0; i < m; ++i) values[k++] = 1.0 - x[oD + i];            // 17
      for (int i = 0; i < m; ++i) values[k++] = -x[oR + i];                 // 18
      for (int i = 0; i < m; ++i) values[k++] = 1.0;                        // 19
      for (int i = 0; i < m; ++i) values[k++] = 1.0;                        // 20
      return true;
   }

   // H = σ_f∇²J + Σ λ_k ∇²c_k. h1 is bilinear in (α,q) → (α,qx)=e^α·Kx·λ_h1,
   // (α,qy)=e^α·Ky·λ_h1, (α,α)=e^α·⟨λ_h1,div q⟩. comp r(1−δ) gives the indefinite
   // cross (δ,r) = −ξ. hr/hd are linear ⇒ no contribution.
   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hr_[k]; jCol[k] = hc_[k]; }
         return true;
      }
      const double ea = std::exp(x[oa]);
      std::vector<double> dq, kxl, kyl;
      divq(x, dq);
      applyK(Kx_, lam + rh1, kxl);
      applyK(Ky_, lam + rh1, kyl);
      Index k = 0;
      for (int i = 0; i < m; ++i) values[k++] = obj_factor;                 // (u,u)
      for (int i = 0; i < m; ++i)                                           // (θ,r)
         values[k++] = lam[rh2x + i] * std::sin(x[oTh + i])
                     - lam[rh2y + i] * std::cos(x[oTh + i]);
      for (int i = 0; i < m; ++i)                                           // (θ,δ)
         values[k++] = lam[rh3x + i] * std::sin(x[oTh + i])
                     - lam[rh3y + i] * std::cos(x[oTh + i]);
      for (int i = 0; i < m; ++i) {                                         // (θ,θ)
         const double c = std::cos(x[oTh + i]), s = std::sin(x[oTh + i]);
         values[k++] = x[oR + i] * (lam[rh2x + i] * c + lam[rh2y + i] * s)
                     + x[oD + i] * (lam[rh3x + i] * c + lam[rh3y + i] * s)
                     + obj_factor * eps_theta_;               // TR gauge ridge
      }
      for (int i = 0; i < m; ++i) values[k++] = -lam[rcomp + i];            // (δ,r)
      for (int i = 0; i < m; ++i) values[k++] = ea * kxl[i];                // (α,qx)
      for (int i = 0; i < m; ++i) values[k++] = ea * kyl[i];                // (α,qy)
      double aa = 0.0;
      for (int i = 0; i < m; ++i) aa += lam[rh1 + i] * dq[i];
      values[k++] = ea * aa + obj_factor * reg_alpha_;                      // (α,α)
      return true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_TNLP_HPP
