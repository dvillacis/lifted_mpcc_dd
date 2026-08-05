// The MULTI-SAMPLE staggered 2D lifted TV-MPCC: one scalar weight α learned
// across a whole training set, as an Ipopt::TNLP.
//
//   min_α  (1/S)·Σ_s ½‖u_s − u_clean,s‖²  +  ½·reg_α·α²
//   s.t.   every sample s satisfies its own lifted lower-level system
//
// The multi-sample sibling of mpcc_2d_tnlp.hpp's Mpcc2DTNLP — same staggered
// mesh, same one-sided/averaged stencils, same 21 Jacobian pieces and 8 Hessian
// blocks, replicated once per training pair. The samples share NOTHING except
// α, which is the entire point: α is the only complicating column, so one
// subdomain per training pair is an arrowhead with an interface of p = 1.
//
// LAYOUT — FIELD-MAJOR, and this is the decision the whole file turns on:
//
//   x    = [ u_0 … u_{S−1} | qx_all | qy_all | r_all | δ_all | θ_all | α ]
//   rows = [ h1_all | h2x_all | h2y_all | h3x_all | h3y_all
//                   | hr_all | hd_all | (ha) | comp_all ]
//
// i.e. EXACTLY Mpcc2DTNLP's layout with m_u → S·m_u and m_q → S·m_q. Sample s
// owns [s·m_u, (s+1)·m_u) of the u field and [s·m_q, (s+1)·m_q) of each lift
// field. Because the fields stay contiguous, MpccTNLPBase needs no change at
// all: eval_f/eval_grad_f sum over n_state = S·m_u, get_bounds_info puts δ ≤ 1
// over n_lift = S·m_q, intermediate_callback's max r(1−δ) scans n_lift, and
// set_theta_ref slices [oTh, oTh+n_lift). Stacking sample-major — one full
// [u|q|r|δ|θ] block per sample — would break every one of those.
//
// The piece order below is Mpcc2DTNLP's with an outer `for s` wrapped around
// each piece, so at S = 1 this class emits a BIT-IDENTICAL structure and value
// sequence. That is not incidental: `--limit 1` against dd_solve_2d on the same
// image is the primary correctness gate for this file, and it only works
// because the orders coincide.
//
// MEAN vs SUM. loss_scale_ (MpccTNLPBase) carries 1/S for --loss mean, applied
// to the data term AND the θ-gauge ridge so their ratio is S-independent;
// reg_alpha_ is untouched, α being one scalar for the whole dataset. --loss sum
// sets it to 1.
#ifndef MPCC_DATASET_TNLP_HPP
#define MPCC_DATASET_TNLP_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.hpp"
// Mpcc2DTNLP itself is not used here; the include gives `struct Tri2`, so the
// two formulations can never disagree about the triplet type.
#include "mpcc_2d_tnlp.hpp"
#include "mpcc_base.hpp"

namespace Ipopt {

class MpccDatasetTNLP : public MpccTNLPBase {
public:
   int S_ = 0;                            // training pairs
   int N = 0, m_u = 0, m_q = 0, nc = 0;   // PER-SAMPLE sizes
   bool averaged = false;                 // stencil
   double sigma_ = 0.1;
   std::vector<std::string> paths_;       // provenance, for the reports/dumps

   // row offsets (rhr/rcomp live in the base)
   int rh1, rh2x, rh2y, rh3x, rh3y, rhd, rha;

   // ONE mesh, shared by every sample — the operators are identical, only the
   // data differs, so building them per sample would be pure waste.
   std::vector<Tri2> Kx_, Ky_, KxT_, KyT_;

   // `pairs` come from the dataset manager already decoded; they are consumed
   // into the field-major uclean_/f_ and only the paths are kept.
   MpccDatasetTNLP(const std::vector<dataset::Pair>& pairs, int size, double sigma,
                   bool exp_weight, bool averaged_stencil, bool mean_loss) {
      if (pairs.empty()) throw std::runtime_error("no training pairs");
      S_ = (int)pairs.size();
      N = size;
      weight_exp = exp_weight;
      has_ha = !weight_exp;
      averaged = averaged_stencil;
      sigma_ = sigma;

      nc = N - 1;
      m_u = N * N;
      m_q = nc * nc;
      const int SU = S_ * m_u, SQ = S_ * m_q;

      // field-major gather of the decoded pairs
      uclean_.resize((size_t)SU);
      f_.resize((size_t)SU);
      paths_.reserve(S_);
      for (int s = 0; s < S_; ++s) {
         const dataset::Pair& p = pairs[s];
         if ((int)p.uclean.size() != m_u || (int)p.f.size() != m_u)
            throw std::runtime_error("pair " + p.path + " has the wrong size");
         std::copy(p.uclean.begin(), p.uclean.end(), uclean_.begin() + (size_t)s * m_u);
         std::copy(p.f.begin(), p.f.end(), f_.begin() + (size_t)s * m_u);
         paths_.push_back(p.path);
      }

      n_state = SU;
      n_lift = SQ;
      ou = 0; oqx = SU; oqy = SU + SQ; oR = SU + 2 * SQ; oD = SU + 3 * SQ;
      oTh = SU + 4 * SQ; oa = SU + 5 * SQ;
      n = SU + 5 * SQ + 1;
      rh1 = 0; rh2x = SU; rh2y = SU + SQ; rh3x = SU + 2 * SQ;
      rh3y = SU + 3 * SQ; rhr = SU + 4 * SQ; rhd = SU + 5 * SQ;
      rha = SU + 6 * SQ;
      rcomp = rha + (has_ha ? 1 : 0);
      mcon = rcomp + SQ;
      n_eq = rhr;
      n_ineq = mcon - n_eq;
      kkt_dim = n + 2 * n_ineq + n_eq;

      loss_scale_ = mean_loss ? 1.0 / (double)S_ : 1.0;

      build_grad_ops();
      build_structures();
   }

   // ---- operators (identical to Mpcc2DTNLP's; one mesh for all samples) ----
   void build_grad_ops() {
      auto node = [&](int i, int j) { return i * N + j; };
      for (int a = 0; a < nc; ++a)
         for (int b = 0; b < nc; ++b) {
            const int cell = a * nc + b;
            if (!averaged) {
               Kx_.push_back({cell, node(a + 1, b), -1.0});
               Kx_.push_back({cell, node(a + 1, b + 1), 1.0});
               Ky_.push_back({cell, node(a, b + 1), -1.0});
               Ky_.push_back({cell, node(a + 1, b + 1), 1.0});
            } else {
               Kx_.push_back({cell, node(a, b), -0.5});
               Kx_.push_back({cell, node(a, b + 1), 0.5});
               Kx_.push_back({cell, node(a + 1, b), -0.5});
               Kx_.push_back({cell, node(a + 1, b + 1), 0.5});
               Ky_.push_back({cell, node(a, b), -0.5});
               Ky_.push_back({cell, node(a, b + 1), -0.5});
               Ky_.push_back({cell, node(a + 1, b), 0.5});
               Ky_.push_back({cell, node(a + 1, b + 1), 0.5});
            }
         }
      auto bycol = [](const Tri2& x, const Tri2& y) {
         return x.r != y.r ? x.r < y.r : x.c < y.c;
      };
      std::sort(Kx_.begin(), Kx_.end(), bycol);
      std::sort(Ky_.begin(), Ky_.end(), bycol);
      for (const auto& t : Kx_) KxT_.push_back({t.c, t.r, t.v});
      for (const auto& t : Ky_) KyT_.push_back({t.c, t.r, t.v});
      std::sort(KxT_.begin(), KxT_.end(), bycol);
      std::sort(KyT_.begin(), KyT_.end(), bycol);
   }

   // One sample: v is that sample's field base pointer, out has length `len`.
   void applyK(const std::vector<Tri2>& K, const double* v, int len,
               std::vector<double>& out) const {
      out.assign(len, 0.0);
      for (const auto& t : K) out[t.r] += t.v * v[t.c];
   }

   // Kxᵀqx + Kyᵀqy for EVERY sample at once, into a field-major S·m_u vector.
   // Needed whole by Jacobian piece 4 (the dense α column) and by the (α,α)
   // Hessian entry, both of which run over all samples.
   void divq_all(const double* x, std::vector<double>& out) const {
      out.assign((size_t)S_ * m_u, 0.0);
      for (int s = 0; s < S_; ++s) {
         double* o = out.data() + (size_t)s * m_u;
         const double* qx = x + oqx + (size_t)s * m_q;
         const double* qy = x + oqy + (size_t)s * m_q;
         for (const auto& t : KxT_) o[t.r] += t.v * qx[t.c];
         for (const auto& t : KyT_) o[t.r] += t.v * qy[t.c];
      }
   }

   // ---- structures: Mpcc2DTNLP's 21 J pieces / 8 H blocks, each per sample --
   // The `for s` is INSIDE each piece, so at S=1 the emitted sequence is
   // identical to the single-sample class's. eval_jac_g/eval_h fill values in
   // exactly this order — matched POSITIONALLY, not by key.
   void build_structures() {
      auto J = [&](int r, int c) { jr_.push_back(r); jc_.push_back(c); };
      for (int s = 0; s < S_; ++s) {                                        // 1
         const int uo = s * m_u;
         for (int i = 0; i < m_u; ++i) J(rh1 + uo + i, ou + uo + i);
      }
      for (int s = 0; s < S_; ++s) {                                        // 2
         const int uo = s * m_u, co = s * m_q;
         for (const auto& t : KxT_) J(rh1 + uo + t.r, oqx + co + t.c);
      }
      for (int s = 0; s < S_; ++s) {                                        // 3
         const int uo = s * m_u, co = s * m_q;
         for (const auto& t : KyT_) J(rh1 + uo + t.r, oqy + co + t.c);
      }
      for (int s = 0; s < S_; ++s) {                       // 4 dense alpha col
         const int uo = s * m_u;
         for (int i = 0; i < m_u; ++i) J(rh1 + uo + i, oa);
      }
      for (int s = 0; s < S_; ++s) {                                        // 5
         const int uo = s * m_u, co = s * m_q;
         for (const auto& t : Kx_) J(rh2x + co + t.r, ou + uo + t.c);
      }
      for (int s = 0; s < S_; ++s) {                                        // 6
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh2x + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 7
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh2x + co + e, oTh + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 8
         const int uo = s * m_u, co = s * m_q;
         for (const auto& t : Ky_) J(rh2y + co + t.r, ou + uo + t.c);
      }
      for (int s = 0; s < S_; ++s) {                                        // 9
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh2y + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 10
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh2y + co + e, oTh + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 11
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3x + co + e, oqx + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 12
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3x + co + e, oD + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 13
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3x + co + e, oTh + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 14
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3y + co + e, oqy + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 15
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3y + co + e, oD + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 16
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rh3y + co + e, oTh + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 17
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rhr + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 18
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rhd + co + e, oD + co + e);
      }
      if (has_ha) J(rha, oa);                       // 19 — ONE global row
      for (int s = 0; s < S_; ++s) {                                        // 20
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rcomp + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                        // 21
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) J(rcomp + co + e, oD + co + e);
      }

      auto H = [&](int r, int c) { hr_.push_back(r); hc_.push_back(c); };
      for (int s = 0; s < S_; ++s) {                                  // (u,u)
         const int uo = s * m_u;
         for (int i = 0; i < m_u; ++i) H(ou + uo + i, ou + uo + i);
      }
      for (int s = 0; s < S_; ++s) {                                  // (θ,r)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oTh + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                  // (θ,δ)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oTh + co + e, oD + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                  // (θ,θ)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oTh + co + e, oTh + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                  // (δ,r)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oD + co + e, oR + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                  // (α,qx)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oa, oqx + co + e);
      }
      for (int s = 0; s < S_; ++s) {                                  // (α,qy)
         const int co = s * m_q;
         for (int e = 0; e < m_q; ++e) H(oa, oqy + co + e);
      }
      H(oa, oa);                                                      // (α,α)
   }

   // ---- starting points ---------------------------------------------------
   // u = f, q = 0 — the no-regularization manifold. An A/B lever only; see
   // mpcc_2d_tnlp.hpp, this is the documented route to the spurious near-noisy
   // branch.
   void cold_start(double* x, double w0) const {
      for (int s = 0; s < S_; ++s) {
         const int uo = s * m_u, co = s * m_q;
         const double* fs = f_.data() + uo;
         std::vector<double> gx, gy;
         applyK(Kx_, fs, m_q, gx);
         applyK(Ky_, fs, m_q, gy);
         for (int i = 0; i < m_u; ++i) x[ou + uo + i] = fs[i];
         for (int e = 0; e < m_q; ++e) {
            x[oqx + co + e] = 0.0;
            x[oqy + co + e] = 0.0;
            x[oR + co + e] = std::hypot(gx[e], gy[e]);
            x[oD + co + e] = 0.0;
            x[oTh + co + e] = std::atan2(gy[e], gx[e]);
         }
      }
      x[oa] = weight_exp ? std::log(w0) : w0;
   }

   // Hybrid Chambolle–Pock for ROF  min_u ½‖u−f‖² + lam·‖∇u‖_{2,1} on ONE
   // sample. Ported from mpcc_2d_tnlp.hpp's chambolle_pock with `f` as a
   // parameter instead of the member, which is what lets the very same routine
   // serve two roles: building the warm start for a training pair, and
   // APPLYING the learned weight to a held-out image (rof_reconstruct below) —
   // this ROF problem IS the MPCC's lower level.
   //
   // ‖K‖² is stencil-dependent: 8 one-sided, 4 averaged.
   void chambolle_pock(double lam, const double* fs, std::vector<double>& u,
                       std::vector<double>& qx, std::vector<double>& qy,
                       int n_iter = 3000, int n_accel = 300,
                       double tol = 1e-9) const {
      const double kn = std::sqrt(averaged ? 4.0 : 8.0);
      const double tau0 = 0.99 / (kn * lam), sig0 = 0.99 / kn;
      double tau = tau0, sig_lam = sig0;
      u.assign(fs, fs + m_u);
      std::vector<double> ubar = u, u_new(m_u), kxu, kyu, dq;
      qx.assign(m_q, 0.0);
      qy.assign(m_q, 0.0);
      for (int k = 0; k < n_iter; ++k) {
         const bool accel = k < n_accel;
         if (!accel && tau != tau0) { tau = tau0; sig_lam = sig0; }
         applyK(Kx_, ubar.data(), m_q, kxu);
         applyK(Ky_, ubar.data(), m_q, kyu);
         for (int e = 0; e < m_q; ++e) {
            qx[e] += sig_lam * kxu[e];
            qy[e] += sig_lam * kyu[e];
            const double nrm = std::max(1.0, std::hypot(qx[e], qy[e]));
            qx[e] /= nrm; qy[e] /= nrm;             // project onto the unit ball
         }
         dq.assign(m_u, 0.0);
         for (const auto& t : KxT_) dq[t.r] += t.v * qx[t.c];
         for (const auto& t : KyT_) dq[t.r] += t.v * qy[t.c];
         double h1 = 0.0;
         for (int i = 0; i < m_u; ++i) {
            u_new[i] = (tau * fs[i] + u[i] - tau * lam * dq[i]) / (tau + 1.0);
            h1 = std::max(h1, std::abs(u_new[i] - u[i]));
         }
         h1 /= tau;
         if (accel) {
            const double th = 1.0 / std::sqrt(1.0 + 2.0 * tau);
            for (int i = 0; i < m_u; ++i) ubar[i] = u_new[i] + th * (u_new[i] - u[i]);
            tau *= th; sig_lam /= th;
         } else {
            for (int i = 0; i < m_u; ++i) ubar[i] = 2.0 * u_new[i] - u[i];
         }
         u = u_new;
         if (h1 <= tol) break;
      }
   }

   // Lift a per-sample CP solution into the MPCC variables. θ comes from the
   // DUAL (θ = ∠q), not from ∇u — see mpcc_2d_tnlp.hpp for why the other way
   // round puts O(1) error on h3 at every flat cell and nothing converges.
   //
   // The S CP solves are fully independent, and at S·3000 iterations they are
   // the one genuinely expensive part of setup, so this is the one loop here
   // worth threading.
   void cp_start(double* x, double w0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int s = 0; s < S_; ++s) {
         const int uo = s * m_u, co = s * m_q;
         std::vector<double> u, qx, qy, gx, gy;
         chambolle_pock(w0, f_.data() + uo, u, qx, qy);
         applyK(Kx_, u.data(), m_q, gx);
         applyK(Ky_, u.data(), m_q, gy);
         for (int i = 0; i < m_u; ++i) x[ou + uo + i] = u[i];
         for (int e = 0; e < m_q; ++e) {
            x[oqx + co + e] = qx[e];
            x[oqy + co + e] = qy[e];
            x[oR + co + e] = std::hypot(gx[e], gy[e]);
            x[oD + co + e] = std::hypot(qx[e], qy[e]);
            x[oTh + co + e] = std::atan2(qy[e], qx[e]);
         }
      }
      x[oa] = weight_exp ? std::log(w0) : w0;
   }

   // Apply a learned weight to arbitrary images: solve the lower level alone at
   // lam = Q(α*) for every pair and hand back the reconstructions. No new
   // machinery — held-out evaluation is exactly the ROF solve above, which is
   // why the validation split costs no extra MPCC.
   void rof_reconstruct(double lam, const std::vector<dataset::Pair>& pairs,
                        std::vector<std::vector<double>>& recon) const {
      recon.assign(pairs.size(), {});
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int s = 0; s < (int)pairs.size(); ++s) {
         std::vector<double> qx, qy;
         chambolle_pock(lam, pairs[s].f.data(), recon[s], qx, qy);
      }
   }

   // ---- evaluation --------------------------------------------------------
   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      const double Qa = Q(x[oa]);
      std::vector<double> dq, kxu, kyu;
      for (int s = 0; s < S_; ++s) {
         const int uo = s * m_u, co = s * m_q;
         const double* qx = x + oqx + co;
         const double* qy = x + oqy + co;
         dq.assign(m_u, 0.0);
         for (const auto& t : KxT_) dq[t.r] += t.v * qx[t.c];
         for (const auto& t : KyT_) dq[t.r] += t.v * qy[t.c];
         applyK(Kx_, x + ou + uo, m_q, kxu);
         applyK(Ky_, x + ou + uo, m_q, kyu);
         for (int i = 0; i < m_u; ++i)
            g[rh1 + uo + i] = x[ou + uo + i] - f_[uo + i] + Qa * dq[i];
         for (int e = 0; e < m_q; ++e) {
            const double c = std::cos(x[oTh + co + e]), sn = std::sin(x[oTh + co + e]);
            const double r = x[oR + co + e], d = x[oD + co + e];
            g[rh2x + co + e]  = kxu[e] - r * c;
            g[rh2y + co + e]  = kyu[e] - r * sn;
            g[rh3x + co + e]  = x[oqx + co + e] - d * c;
            g[rh3y + co + e]  = x[oqy + co + e] - d * sn;
            g[rhr + co + e]   = r;
            g[rhd + co + e]   = d;
            g[rcomp + co + e] = r * (1.0 - d) - t_;
         }
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
      std::vector<double> dq;
      divq_all(x, dq);
      Index k = 0;
      for (int s = 0; s < S_; ++s)                                          // 1
         for (int i = 0; i < m_u; ++i) values[k++] = 1.0;
      for (int s = 0; s < S_; ++s)                                          // 2
         for (const auto& t : KxT_) values[k++] = Qa * t.v;
      for (int s = 0; s < S_; ++s)                                          // 3
         for (const auto& t : KyT_) values[k++] = Qa * t.v;
      for (int s = 0; s < S_; ++s)                                          // 4
         for (int i = 0; i < m_u; ++i) values[k++] = dQa * dq[s * m_u + i];
      for (int s = 0; s < S_; ++s)                                          // 5
         for (const auto& t : Kx_) values[k++] = t.v;
      for (int s = 0; s < S_; ++s)                                          // 6
         for (int e = 0; e < m_q; ++e)
            values[k++] = -std::cos(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 7
         for (int e = 0; e < m_q; ++e)
            values[k++] = x[oR + s * m_q + e] * std::sin(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 8
         for (const auto& t : Ky_) values[k++] = t.v;
      for (int s = 0; s < S_; ++s)                                          // 9
         for (int e = 0; e < m_q; ++e)
            values[k++] = -std::sin(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 10
         for (int e = 0; e < m_q; ++e)
            values[k++] = -x[oR + s * m_q + e] * std::cos(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 11
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0;
      for (int s = 0; s < S_; ++s)                                          // 12
         for (int e = 0; e < m_q; ++e)
            values[k++] = -std::cos(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 13
         for (int e = 0; e < m_q; ++e)
            values[k++] = x[oD + s * m_q + e] * std::sin(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 14
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0;
      for (int s = 0; s < S_; ++s)                                          // 15
         for (int e = 0; e < m_q; ++e)
            values[k++] = -std::sin(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 16
         for (int e = 0; e < m_q; ++e)
            values[k++] = -x[oD + s * m_q + e] * std::cos(x[oTh + s * m_q + e]);
      for (int s = 0; s < S_; ++s)                                          // 17
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0;
      for (int s = 0; s < S_; ++s)                                          // 18
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0;
      if (has_ha) values[k++] = 1.0;                                        // 19
      for (int s = 0; s < S_; ++s)                                          // 20
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0 - x[oD + s * m_q + e];
      for (int s = 0; s < S_; ++s)                                          // 21
         for (int e = 0; e < m_q; ++e) values[k++] = -x[oR + s * m_q + e];
      return true;
   }

   // (α,qx) = Q'(α)·Kx λ_h1 per sample; (α,α) = Q''(α)·Σ_s⟨λ_h1,s, div q_s⟩
   // accumulates across the WHOLE dataset — α is one variable in every sample's
   // h1 row, which is exactly why it is the only complicating column. comp
   // gives the indefinite (δ,r). loss_scale_ rides on the two objective terms
   // ((u,u) and the θ ridge) to match eval_f.
   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hr_[k]; jCol[k] = hc_[k]; }
         return true;
      }
      std::vector<double> dq;
      divq_all(x, dq);
      // Kx λ_h1 and Ky λ_h1, per sample, contiguous like the lift fields
      std::vector<double> kxl((size_t)S_ * m_q, 0.0), kyl((size_t)S_ * m_q, 0.0);
      for (int s = 0; s < S_; ++s) {
         const double* ls = lam + rh1 + (size_t)s * m_u;
         double* kx = kxl.data() + (size_t)s * m_q;
         double* ky = kyl.data() + (size_t)s * m_q;
         for (const auto& t : Kx_) kx[t.r] += t.v * ls[t.c];
         for (const auto& t : Ky_) ky[t.r] += t.v * ls[t.c];
      }
      Index k = 0;
      for (int s = 0; s < S_; ++s)                                     // (u,u)
         for (int i = 0; i < m_u; ++i) values[k++] = obj_factor * loss_scale_;
      for (int s = 0; s < S_; ++s)                                     // (θ,r)
         for (int e = 0; e < m_q; ++e) {
            const int c = s * m_q + e;
            values[k++] = lam[rh2x + c] * std::sin(x[oTh + c])
                        - lam[rh2y + c] * std::cos(x[oTh + c]);
         }
      for (int s = 0; s < S_; ++s)                                     // (θ,δ)
         for (int e = 0; e < m_q; ++e) {
            const int c = s * m_q + e;
            values[k++] = lam[rh3x + c] * std::sin(x[oTh + c])
                        - lam[rh3y + c] * std::cos(x[oTh + c]);
         }
      for (int s = 0; s < S_; ++s)                                     // (θ,θ)
         for (int e = 0; e < m_q; ++e) {
            const int c = s * m_q + e;
            const double co = std::cos(x[oTh + c]), sn = std::sin(x[oTh + c]);
            values[k++] = x[oR + c] * (lam[rh2x + c] * co + lam[rh2y + c] * sn)
                        + x[oD + c] * (lam[rh3x + c] * co + lam[rh3y + c] * sn)
                        + obj_factor * loss_scale_ * eps_theta_;
         }
      for (int s = 0; s < S_; ++s)                                     // (δ,r)
         for (int e = 0; e < m_q; ++e) values[k++] = -lam[rcomp + s * m_q + e];
      const double dQa = dQ(x[oa]);
      for (int s = 0; s < S_; ++s)                                     // (α,qx)
         for (int e = 0; e < m_q; ++e) values[k++] = dQa * kxl[s * m_q + e];
      for (int s = 0; s < S_; ++s)                                     // (α,qy)
         for (int e = 0; e < m_q; ++e) values[k++] = dQa * kyl[s * m_q + e];
      double aa = 0.0;
      for (int i = 0; i < S_ * m_u; ++i) aa += lam[rh1 + i] * dq[i];
      values[k++] = d2Q(x[oa]) * aa + obj_factor * reg_alpha_;         // (α,α)
      return true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_DATASET_TNLP_HPP
