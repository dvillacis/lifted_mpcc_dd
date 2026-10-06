// The staggered 1D lifted TV-MPCC as an Ipopt::TNLP — a C++ port of
// ../lifted_mpcc_1d.py's Lifted1DMPCC (the whiteboard example).
//
// The primal u lives on the n NODES, the dual/lift blocks on the n−1 EDGES, and
// the difference operator is genuinely rectangular — no Neumann padding row, so
// Kᵀ has no dead column:
//
//     K ∈ R^{(n−1)×n},  (K u)_e = u_{e+1} − u_e
//     Kᵀ ∈ R^{n×(n−1)}, (Kᵀq)_i = q_{i−1} − q_i      (q_{−1} = q_{n−1} = 0)
//
//   x = [u (n) | qx | qy | r | δ | θ | α],        n_var = n + 5·mE + 1,  mE = n−1
//   rows h1 (n) | h2x | h2y | h3x | h3y | hr | hd | [ha] | comp
//                                                 m_con = n + 7·mE + has_ha
//
//   h1  : u − f + Q(α)·Kᵀqx = 0      ← the state row carries the weight
//   h2x : K u − r·cosθ = 0            h2y : − r·sinθ = 0   (the 1D y-gradient is 0)
//   h3x : qx − δ·cosθ = 0             h3y : qy − δ·sinθ = 0
//   hr  : r ≥ 0    hd : δ ≥ 0    [ha : α ≥ 0]  ← explicit inequality ROWS
//   comp: r·(1−δ) − t ≤ 0             ← Scholtes, written directly on δ, LAST
//
// plus the box δ ≤ 1. ONLY h1 is node-length; every other block is edge-length —
// which is exactly why dd_solver.hpp's built-in uniform geometry (dim = 17m+1)
// cannot describe this problem, and why the driver injects an owner map instead.
//
// Q(α) has two modes behind the base's Q/dQ/d2Q, as in the Python: `linear` (the
// board's bare α, carrying the explicit row ha : α ≥ 0 and no α box) and `exp`
// (Q = e^α with the [−15,15] box and no ha row). They are solution-equivalent —
// measured, weight 0.071568 vs e^{−2.637107} at n=64.
//
// 19 Jacobian pieces (18 without ha) and 7 Hessian blocks, assembled in the SAME
// order as their structure arrays: they are matched POSITIONALLY, not by key, in
// both languages. Objective, bounds and warm-start bookkeeping live in
// MpccTNLPBase.
#ifndef MPCC_1D_TNLP_HPP
#define MPCC_1D_TNLP_HPP

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpcc_base.hpp"

namespace Ipopt {

class Mpcc1DTNLP : public MpccTNLPBase {
public:
   int nN = 0, mE = 0;
   int file_nsub = 0;                     // nsub the dump was written for
   double sigma_ = 0.1;                   // noise level the dump was generated at
   std::vector<int> file_owner_;          // Python's kkt_owner, for --self-check

   // row offsets (rhr/rcomp live in the base)
   int rh1, rh2x, rh2y, rh3x, rh3y, rhd, rha;

   explicit Mpcc1DTNLP(const std::string& datafile) {
      load(datafile);
      mE = nN - 1;
      n_state = nN; n_lift = mE;
      ou = 0; oqx = nN; oqy = nN + mE; oR = nN + 2 * mE; oD = nN + 3 * mE;
      oTh = nN + 4 * mE; oa = nN + 5 * mE;
      n = nN + 5 * mE + 1;
      rh1 = 0; rh2x = nN; rh2y = nN + mE; rh3x = nN + 2 * mE; rh3y = nN + 3 * mE;
      rhr = nN + 4 * mE; rhd = nN + 5 * mE; rha = nN + 6 * mE;
      rcomp = rha + (has_ha ? 1 : 0);
      mcon = rcomp + mE;
      n_eq = rhr;                      // h1|h2x|h2y|h3x|h3y
      n_ineq = mcon - n_eq;            // hr|hd|[ha]|comp
      kkt_dim = n + 2 * n_ineq + n_eq;
      build_structures();
   }

   // ---- the file written by dump_data_1d.py ------------------------------
   void load(const std::string& fn) {
      std::ifstream in(fn);
      if (!in) throw std::runtime_error("cannot open " + fn);
      int nvar_f, mcon_f, kkt_f, wflag, seed;
      in >> nN >> nvar_f >> mcon_f >> kkt_f >> file_nsub >> wflag >> sigma_ >> seed;
      if (!in) throw std::runtime_error("bad header in " + fn);
      weight_exp = (wflag != 0);
      has_ha = !weight_exp;
      uclean_.resize(nN); f_.resize(nN); x_start_.resize(nvar_f);
      file_owner_.resize(kkt_f);
      for (double& v : uclean_) in >> v;
      for (double& v : f_) in >> v;
      for (double& v : x_start_) in >> v;
      for (int& v : file_owner_) in >> v;
      if (!in) throw std::runtime_error("truncated data file " + fn);
   }

   // ---- operators (applied directly; K has 2 nonzeros per row) ------------
   // (K u)_e = u_{e+1} − u_e
   void applyK(const double* u, std::vector<double>& out) const {
      out.assign(mE, 0.0);
      for (int e = 0; e < mE; ++e) out[e] = u[e + 1] - u[e];
   }
   // (Kᵀq)_i = q_{i−1} − q_i, with q_{−1} = q_{mE} = 0
   void applyKT(const double* q, std::vector<double>& out) const {
      out.assign(nN, 0.0);
      for (int i = 0; i < nN; ++i) {
         if (i >= 1) out[i] += q[i - 1];
         if (i < mE) out[i] -= q[i];
      }
   }

   // ---- structures (19 Jacobian pieces / 7 Hessian blocks) ---------------
   // Kᵀ is walked in ROW-MAJOR order to match the Python's KT.tocoo(), whose data
   // array _KTv the values loop reuses: row i holds (col i−1, +1) then (col i, −1).
   void build_structures() {
      auto J = [&](int r, int c) { jr_.push_back(r); jc_.push_back(c); };
      for (int i = 0; i < nN; ++i) J(rh1 + i, ou + i);                    // 1  h1/∂u
      for (int i = 0; i < nN; ++i) {                                      // 2  h1/∂qx
         if (i >= 1) J(rh1 + i, oqx + i - 1);
         if (i < mE) J(rh1 + i, oqx + i);
      }
      for (int i = 0; i < nN; ++i) J(rh1 + i, oa);                        // 3  h1/∂α
      for (int e = 0; e < mE; ++e) {                                      // 4  h2x/∂u
         J(rh2x + e, ou + e); J(rh2x + e, ou + e + 1);
      }
      for (int e = 0; e < mE; ++e) J(rh2x + e, oR + e);                   // 5
      for (int e = 0; e < mE; ++e) J(rh2x + e, oTh + e);                  // 6
      for (int e = 0; e < mE; ++e) J(rh2y + e, oR + e);                   // 7
      for (int e = 0; e < mE; ++e) J(rh2y + e, oTh + e);                  // 8
      for (int e = 0; e < mE; ++e) J(rh3x + e, oqx + e);                  // 9
      for (int e = 0; e < mE; ++e) J(rh3x + e, oD + e);                   // 10
      for (int e = 0; e < mE; ++e) J(rh3x + e, oTh + e);                  // 11
      for (int e = 0; e < mE; ++e) J(rh3y + e, oqy + e);                  // 12
      for (int e = 0; e < mE; ++e) J(rh3y + e, oD + e);                   // 13
      for (int e = 0; e < mE; ++e) J(rh3y + e, oTh + e);                  // 14
      for (int e = 0; e < mE; ++e) J(rhr + e, oR + e);                    // 15
      for (int e = 0; e < mE; ++e) J(rhd + e, oD + e);                    // 16
      if (has_ha) J(rha, oa);                                             // 17 ha/∂α
      for (int e = 0; e < mE; ++e) J(rcomp + e, oR + e);                  // 18
      for (int e = 0; e < mE; ++e) J(rcomp + e, oD + e);                  // 19

      auto H = [&](int r, int c) { hr_.push_back(r); hc_.push_back(c); };
      for (int i = 0; i < nN; ++i) H(ou + i, ou + i);                     // (u,u)
      for (int e = 0; e < mE; ++e) H(oTh + e, oR + e);                    // (θ,r)
      for (int e = 0; e < mE; ++e) H(oTh + e, oD + e);                    // (θ,δ)
      for (int e = 0; e < mE; ++e) H(oTh + e, oTh + e);                   // (θ,θ)
      for (int e = 0; e < mE; ++e) H(oD + e, oR + e);                     // (δ,r) comp
      for (int e = 0; e < mE; ++e) H(oa, oqx + e);                        // (α,qx)
      H(oa, oa);                                                          // (α,α)
   }

   // u = f, q = 0 — the no-regularization manifold, this repo's known route to the
   // spurious near-noisy branch (in 1D it stalls at t = 3e-1). Kept only as an A/B
   // lever; the default start is the dumped Chambolle–Pock point.
   void cold_start(double* x, double w0) const {
      std::vector<double> Ku;
      applyK(f_.data(), Ku);
      for (int i = 0; i < nN; ++i) x[ou + i] = f_[i];
      for (int e = 0; e < mE; ++e) {
         x[oqx + e] = 0.0; x[oqy + e] = 0.0;
         x[oR + e] = std::abs(Ku[e]);
         x[oD + e] = 0.0;
         x[oTh + e] = (Ku[e] >= 0.0) ? 0.0 : M_PI;
      }
      x[oa] = weight_exp ? std::log(w0) : w0;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      const double Qa = Q(x[oa]);
      std::vector<double> KTq, Ku;
      applyKT(x + oqx, KTq);
      applyK(x + ou, Ku);
      for (int i = 0; i < nN; ++i) g[rh1 + i] = x[ou + i] - f_[i] + Qa * KTq[i];
      for (int e = 0; e < mE; ++e) {
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         const double r = x[oR + e], d = x[oD + e];
         g[rh2x + e]  = Ku[e] - r * c;
         g[rh2y + e]  = -r * s;
         g[rh3x + e]  = x[oqx + e] - d * c;
         g[rh3y + e]  = x[oqy + e] - d * s;
         g[rhr + e]   = r;
         g[rhd + e]   = d;
         g[rcomp + e] = r * (1.0 - d) - t_;
      }
      if (has_ha) g[rha] = x[oa];
      return true;
   }

   bool eval_jac_g(Index, const Number* x, bool, Index, Index nele, Index* iRow,
                   Index* jCol, Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = jr_[k]; jCol[k] = jc_[k]; }
         return true;
      }
      const double Qa = Q(x[oa]), dQa = dQ(x[oa]);
      std::vector<double> KTq;
      applyKT(x + oqx, KTq);
      Index k = 0;
      for (int i = 0; i < nN; ++i) values[k++] = 1.0;                     // 1
      for (int i = 0; i < nN; ++i) {                                      // 2  Q·Kᵀ
         if (i >= 1) values[k++] = Qa * 1.0;
         if (i < mE) values[k++] = Qa * -1.0;
      }
      for (int i = 0; i < nN; ++i) values[k++] = dQa * KTq[i];            // 3
      for (int e = 0; e < mE; ++e) { values[k++] = -1.0; values[k++] = 1.0; }  // 4
      for (int e = 0; e < mE; ++e) values[k++] = -std::cos(x[oTh + e]);   // 5
      for (int e = 0; e < mE; ++e)                                        // 6
         values[k++] = x[oR + e] * std::sin(x[oTh + e]);
      for (int e = 0; e < mE; ++e) values[k++] = -std::sin(x[oTh + e]);   // 7
      for (int e = 0; e < mE; ++e)                                        // 8
         values[k++] = -x[oR + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < mE; ++e) values[k++] = 1.0;                     // 9
      for (int e = 0; e < mE; ++e) values[k++] = -std::cos(x[oTh + e]);   // 10
      for (int e = 0; e < mE; ++e)                                        // 11
         values[k++] = x[oD + e] * std::sin(x[oTh + e]);
      for (int e = 0; e < mE; ++e) values[k++] = 1.0;                     // 12
      for (int e = 0; e < mE; ++e) values[k++] = -std::sin(x[oTh + e]);   // 13
      for (int e = 0; e < mE; ++e)                                        // 14
         values[k++] = -x[oD + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < mE; ++e) values[k++] = 1.0;                     // 15
      for (int e = 0; e < mE; ++e) values[k++] = 1.0;                     // 16
      if (has_ha) values[k++] = 1.0;                                      // 17
      for (int e = 0; e < mE; ++e) values[k++] = 1.0 - x[oD + e];         // 18
      for (int e = 0; e < mE; ++e) values[k++] = -x[oR + e];              // 19
      return true;
   }

   // H = σ_f∇²J + Σ λ_k ∇²c_k. h1 is bilinear in (α,qx) → (α,qx) = Q'(α)·K λ_h1 and
   // (α,α) = Q''(α)·⟨λ_h1, Kᵀqx⟩ — identically 0 for the LINEAR weight, which
   // therefore takes its only (α,α) curvature from the reg-α ridge. comp r(1−δ)
   // gives the indefinite cross (δ,r) = −ξ. hr/hd/ha are linear ⇒ no contribution.
   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hr_[k]; jCol[k] = hc_[k]; }
         return true;
      }
      std::vector<double> KTq, Kl;
      applyKT(x + oqx, KTq);
      applyK(lam + rh1, Kl);
      Index k = 0;
      for (int i = 0; i < nN; ++i) values[k++] = obj_factor;              // (u,u)
      for (int e = 0; e < mE; ++e)                                        // (θ,r)
         values[k++] = lam[rh2x + e] * std::sin(x[oTh + e])
                     - lam[rh2y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < mE; ++e)                                        // (θ,δ)
         values[k++] = lam[rh3x + e] * std::sin(x[oTh + e])
                     - lam[rh3y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < mE; ++e) {                                      // (θ,θ)
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         values[k++] = x[oR + e] * (lam[rh2x + e] * c + lam[rh2y + e] * s)
                     + x[oD + e] * (lam[rh3x + e] * c + lam[rh3y + e] * s)
                     + obj_factor * eps_theta_;
      }
      for (int e = 0; e < mE; ++e) values[k++] = -lam[rcomp + e];         // (δ,r)
      for (int e = 0; e < mE; ++e) values[k++] = dQ(x[oa]) * Kl[e];       // (α,qx)
      double aa = 0.0;
      for (int i = 0; i < nN; ++i) aa += lam[rh1 + i] * KTq[i];
      values[k++] = d2Q(x[oa]) * aa + obj_factor * reg_alpha_;            // (α,α)
      return true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_1D_TNLP_HPP
