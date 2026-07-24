// The staggered 2D lifted TV-MPCC as an Ipopt::TNLP — a C++ port of
// ../lifted_mpcc_2d.py's Lifted2DMPCC (the whiteboard's cell-centred mesh).
//
// `u` lives on the N² NODES, `qx, qy, r, δ, θ` on the (N−1)² CELL CENTRES, and the
// gradient is rectangular (m_q × m_u) with no Neumann zero-row, so −Kᵀ is the exact
// discrete divergence. Unit mesh spacing, as on the board.
//
//   x = [u (m_u) | qx | qy | r | δ | θ | α],   n = m_u + 5·m_q + 1
//   rows h1 (m_u) | h2x | h2y | h3x | h3y | hr | hd | [ha] | comp
//                                             m_con = m_u + 7·m_q + has_ha
//
//   h1  : u − f + Q(α)·(Kxᵀqx + Kyᵀqy) = 0        ← the state row carries Q(α)
//   h2x : Kx u − r·cosθ = 0      h2y : Ky u − r·sinθ = 0
//   h3x : qx − δ·cosθ = 0        h3y : qy − δ·sinθ = 0
//   hr  : r ≥ 0   hd : δ ≥ 0   [ha : α ≥ 0]       ← explicit inequality ROWS
//   comp: r·(1−δ) − t ≤ 0                          ← Scholtes, on δ, LAST
//
// plus the box δ ≤ 1. Two meshes in one layout — h1/ha are node/scalar-length while
// everything else is cell-length — is exactly what dd_solver.hpp's built-in uniform
// geometry cannot express, hence the injected owner map from the driver.
//
// STENCILS (both reproduce the board; `onesided` is the default and the measured
// winner — see ../CLAUDE.md, the averaged one annihilates the Nyquist mode so
// TV(checkerboard) = 0 and it denoises visibly worse):
//   onesided : Kx = Sh⊗D, Ky = D⊗Sh   → (Kx u)_{a,b} = u[a+1,b+1] − u[a+1,b]
//                                        (Ky u)_{a,b} = u[a+1,b+1] − u[a,b+1]
//              both anchored at the COMMON node (a+1,b+1) — which is what the
//              partition's anchor rule keys off, and why only one dual component
//              crosses each cut.
//   averaged : Kx = A⊗D, Ky = D⊗A     → the ½ bilinear stencil, 4 nodes per cell.
//
// 21 Jacobian pieces (20 without ha) and 8 Hessian blocks, assembled in the SAME
// order as their structure arrays — matched positionally, not by key, in both
// languages. Objective, bounds and warm-start bookkeeping live in MpccTNLPBase.
#ifndef MPCC_2D_TNLP_HPP
#define MPCC_2D_TNLP_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "mpcc_base.hpp"

namespace Ipopt {

struct Tri2 { int r, c; double v; };

class Mpcc2DTNLP : public MpccTNLPBase {
public:
   int N = 0, m_u = 0, m_q = 0, nc = 0;
   bool averaged = false;                 // stencil
   int file_nsub = 0;
   double sigma_ = 0.1;
   std::vector<int> file_owner_;          // Python's kkt_owner, for --self-check

   // row offsets (rhr/rcomp live in the base)
   int rh1, rh2x, rh2y, rh3x, rh3y, rhd, rha;

   std::vector<Tri2> Kx_, Ky_, KxT_, KyT_;

   // Two input routes:
   //   *.txt  → the dump_data_2d.py instance: Python's exact data AND its lifted CP
   //            warm start AND its owner map. The only reproducible one.
   //   image  → decode a PNG/JPEG here (stb_image, via ../image_io.hpp) and build
   //            the warm start with the C++ Chambolle–Pock below. Convenient, but
   //            NOT the same instance as Python's — see load_image_data().
   Mpcc2DTNLP(const std::string& datafile, const image_io::Opts& opt = {},
              bool exp_weight = false, bool averaged_stencil = false) {
      if (image_io::ends_with(datafile, ".txt")) {
         load(datafile);
      } else {
         weight_exp = exp_weight;
         has_ha = !weight_exp;
         averaged = averaged_stencil;
         load_image_data(datafile, opt);
      }
      nc = N - 1; m_u = N * N; m_q = nc * nc;
      n_state = m_u; n_lift = m_q;
      ou = 0; oqx = m_u; oqy = m_u + m_q; oR = m_u + 2 * m_q; oD = m_u + 3 * m_q;
      oTh = m_u + 4 * m_q; oa = m_u + 5 * m_q;
      n = m_u + 5 * m_q + 1;
      rh1 = 0; rh2x = m_u; rh2y = m_u + m_q; rh3x = m_u + 2 * m_q;
      rh3y = m_u + 3 * m_q; rhr = m_u + 4 * m_q; rhd = m_u + 5 * m_q;
      rha = m_u + 6 * m_q;
      rcomp = rha + (has_ha ? 1 : 0);
      mcon = rcomp + m_q;
      n_eq = rhr;
      n_ineq = mcon - n_eq;
      kkt_dim = n + 2 * n_ineq + n_eq;
      build_grad_ops();
      build_structures();
   }

   // ---- the file written by dump_data_2d.py ------------------------------
   void load(const std::string& fn) {
      std::ifstream in(fn);
      if (!in) throw std::runtime_error("cannot open " + fn);
      int nvar_f, mcon_f, kkt_f, wflag, sflag, seed;
      in >> N >> nvar_f >> mcon_f >> kkt_f >> file_nsub >> wflag >> sflag
         >> sigma_ >> seed;
      if (!in) throw std::runtime_error("bad header in " + fn);
      weight_exp = (wflag != 0);
      has_ha = !weight_exp;
      averaged = (sflag != 0);
      const int mu = N * N;
      uclean_.resize(mu); f_.resize(mu); x_start_.resize(nvar_f);
      file_owner_.resize(kkt_f);
      for (double& v : uclean_) in >> v;
      for (double& v : f_) in >> v;
      for (double& v : x_start_) in >> v;
      for (int& v : file_owner_) in >> v;
      if (!in) throw std::runtime_error("truncated data file " + fn);
   }

   // Decode an image into (u_clean, f). Convenient, and deliberately kept honest:
   // **this is NOT the instance ../lifted_mpcc_2d.py builds from the same file.**
   // Three independent reasons, none of them fixable here:
   //   * NORMALIZATION — Python divides by 255; image_io min–max stretches to
   //     [0,1] by default. That changes the contrast and therefore the optimal
   //     weight, so `opt.minmax = false` is set by the driver to match Python.
   //   * RESIZE — PIL's BILINEAR scales its filter support when downsampling (it
   //     averages over the whole reduction footprint); image_io's bilerp samples
   //     4 neighbours. At 512→32 those are visibly different images.
   //   * NOISE — NumPy's PCG64 cannot be reproduced by std::mt19937, full stop.
   // Use dump_data_2d.py whenever the run has to be comparable with Python; use
   // this when you just want to point the solver at a picture.
   void load_image_data(const std::string& fn, const image_io::Opts& opt) {
      image_io::load_image(fn, opt, N, uclean_, f_);
      sigma_ = opt.sigma;
      file_nsub = 0;                 // no dumped partition to cross-check against
      file_owner_.clear();
      x_start_.clear();              // filled by cp_start() once sizes are known
   }

   // ---- operators --------------------------------------------------------
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

   void applyK(const std::vector<Tri2>& K, const double* v, int len,
               std::vector<double>& out) const {
      out.assign(len, 0.0);
      for (const auto& t : K) out[t.r] += t.v * v[t.c];
   }
   void divq(const double* x, std::vector<double>& out) const {
      out.assign(m_u, 0.0);
      for (const auto& t : KxT_) out[t.r] += t.v * x[oqx + t.c];
      for (const auto& t : KyT_) out[t.r] += t.v * x[oqy + t.c];
   }

   // ---- structures (21 Jacobian pieces / 8 Hessian blocks) ---------------
   void build_structures() {
      auto J = [&](int r, int c) { jr_.push_back(r); jc_.push_back(c); };
      for (int i = 0; i < m_u; ++i) J(rh1 + i, ou + i);                   // 1
      for (const auto& t : KxT_) J(rh1 + t.r, oqx + t.c);                 // 2
      for (const auto& t : KyT_) J(rh1 + t.r, oqy + t.c);                 // 3
      for (int i = 0; i < m_u; ++i) J(rh1 + i, oa);                       // 4 dense col
      for (const auto& t : Kx_) J(rh2x + t.r, ou + t.c);                  // 5
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oR + e);                  // 6
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oTh + e);                 // 7
      for (const auto& t : Ky_) J(rh2y + t.r, ou + t.c);                  // 8
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oR + e);                  // 9
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oTh + e);                 // 10
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oqx + e);                 // 11
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oD + e);                  // 12
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oTh + e);                 // 13
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oqy + e);                 // 14
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oD + e);                  // 15
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oTh + e);                 // 16
      for (int e = 0; e < m_q; ++e) J(rhr + e, oR + e);                   // 17
      for (int e = 0; e < m_q; ++e) J(rhd + e, oD + e);                   // 18
      if (has_ha) J(rha, oa);                                             // 19
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oR + e);                 // 20
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oD + e);                 // 21

      auto H = [&](int r, int c) { hr_.push_back(r); hc_.push_back(c); };
      for (int i = 0; i < m_u; ++i) H(ou + i, ou + i);                    // (u,u)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oR + e);                   // (θ,r)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oD + e);                   // (θ,δ)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oTh + e);                  // (θ,θ)
      for (int e = 0; e < m_q; ++e) H(oD + e, oR + e);                    // (δ,r)
      for (int e = 0; e < m_q; ++e) H(oa, oqx + e);                       // (α,qx)
      for (int e = 0; e < m_q; ++e) H(oa, oqy + e);                       // (α,qy)
      H(oa, oa);                                                          // (α,α)
   }

   // u = f, q = 0 — the no-regularization manifold. Kept as an A/B lever only; the
   // default start is the dumped Chambolle–Pock point (../CLAUDE.md: the cold start
   // is this repo's known route to the spurious near-noisy branch, though at N=16
   // the 2D script happens to converge to the same solution).
   void cold_start(double* x, double w0) const {
      std::vector<double> gx, gy;
      applyK(Kx_, f_.data(), m_q, gx);
      applyK(Ky_, f_.data(), m_q, gy);
      for (int i = 0; i < m_u; ++i) x[ou + i] = f_[i];
      for (int e = 0; e < m_q; ++e) {
         x[oqx + e] = 0.0; x[oqy + e] = 0.0;
         x[oR + e] = std::hypot(gx[e], gy[e]);
         x[oD + e] = 0.0;
         x[oTh + e] = std::atan2(gy[e], gx[e]);
      }
      x[oa] = weight_exp ? std::log(w0) : w0;
   }

   // ---- Chambolle–Pock warm start (the image route) ----------------------
   // Hybrid CP for ROF  min_u ½‖u−f‖² + lam·‖∇u‖_{2,1}, ported from
   // ../lifted_mpcc_2d.py's chambolle_pock_2d: an accelerated phase (γ = 1, the
   // primal is 1-strongly convex) then fixed steps that polish the h1 fixed point,
   // stopping on the exact residual identity ‖u₊−u‖∞/τ = ‖h1‖∞.
   //
   // Why bother instead of cold-starting: at the CP solution
   // u = f − lam·(Kxᵀqx + Kyᵀqy) with the dual on the unit ball, which is exactly
   // this MPCC's lower-level system at Q(α) = lam — so the lift is almost feasible
   // (h1 ≈ 0, h3 exact). `u = f, q = 0` instead sits on the no-regularization
   // manifold, this repo's documented route to the spurious near-noisy branch.
   //
   // ‖K‖² is stencil-dependent: 8 one-sided, 4 averaged (averaging multiplies the
   // symbol by cos(ξ⊥/2)).
   void chambolle_pock(double lam, std::vector<double>& u,
                       std::vector<double>& qx, std::vector<double>& qy,
                       int n_iter = 3000, int n_accel = 300,
                       double tol = 1e-9) const {
      const double kn = std::sqrt(averaged ? 4.0 : 8.0);
      const double tau0 = 0.99 / (kn * lam), sig0 = 0.99 / kn;
      double tau = tau0, sig_lam = sig0;
      u = f_;
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
            u_new[i] = (tau * f_[i] + u[i] - tau * lam * dq[i]) / (tau + 1.0);
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

   // Lift a CP solution to the MPCC variables. θ comes from the DUAL (θ = ∠q), not
   // from ∇u: that makes h3 exact and leaves the error on h2 = ∇u − |∇u|·dir, which
   // is at noise level. The other way round puts O(1) error on h3 at every flat
   // cell (a flat cell still carries a unit-size dual) and nothing converges — the
   // 1D sibling measured exactly that.
   void cp_start(double* x, double w0) {
      std::vector<double> u, qx, qy, gx, gy;
      chambolle_pock(w0, u, qx, qy);
      applyK(Kx_, u.data(), m_q, gx);
      applyK(Ky_, u.data(), m_q, gy);
      for (int i = 0; i < m_u; ++i) x[ou + i] = u[i];
      for (int e = 0; e < m_q; ++e) {
         x[oqx + e] = qx[e];
         x[oqy + e] = qy[e];
         x[oR + e] = std::hypot(gx[e], gy[e]);
         x[oD + e] = std::hypot(qx[e], qy[e]);
         x[oTh + e] = std::atan2(qy[e], qx[e]);
      }
      x[oa] = weight_exp ? std::log(w0) : w0;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      const double Qa = Q(x[oa]);
      std::vector<double> dq, kxu, kyu;
      divq(x, dq);
      applyK(Kx_, x + ou, m_q, kxu);
      applyK(Ky_, x + ou, m_q, kyu);
      for (int i = 0; i < m_u; ++i) g[rh1 + i] = x[ou + i] - f_[i] + Qa * dq[i];
      for (int e = 0; e < m_q; ++e) {
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         const double r = x[oR + e], d = x[oD + e];
         g[rh2x + e]  = kxu[e] - r * c;
         g[rh2y + e]  = kyu[e] - r * s;
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
      std::vector<double> dq;
      divq(x, dq);
      Index k = 0;
      for (int i = 0; i < m_u; ++i) values[k++] = 1.0;                    // 1
      for (const auto& t : KxT_) values[k++] = Qa * t.v;                  // 2
      for (const auto& t : KyT_) values[k++] = Qa * t.v;                  // 3
      for (int i = 0; i < m_u; ++i) values[k++] = dQa * dq[i];            // 4
      for (const auto& t : Kx_) values[k++] = t.v;                        // 5
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);  // 6
      for (int e = 0; e < m_q; ++e)                                       // 7
         values[k++] = x[oR + e] * std::sin(x[oTh + e]);
      for (const auto& t : Ky_) values[k++] = t.v;                        // 8
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);  // 9
      for (int e = 0; e < m_q; ++e)                                       // 10
         values[k++] = -x[oR + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 11
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);  // 12
      for (int e = 0; e < m_q; ++e)                                       // 13
         values[k++] = x[oD + e] * std::sin(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 14
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);  // 15
      for (int e = 0; e < m_q; ++e)                                       // 16
         values[k++] = -x[oD + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 17
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 18
      if (has_ha) values[k++] = 1.0;                                      // 19
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0 - x[oD + e];        // 20
      for (int e = 0; e < m_q; ++e) values[k++] = -x[oR + e];             // 21
      return true;
   }

   // (α,qx) = Q'(α)·Kx λ_h1 has length m_q (Kx is m_q×m_u, λ_h1 is m_u);
   // (α,α) = Q''(α)·⟨λ_h1, div q⟩ vanishes identically for the linear weight, whose
   // only (α,α) curvature is then the reg-α ridge. comp gives the indefinite (δ,r).
   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hr_[k]; jCol[k] = hc_[k]; }
         return true;
      }
      std::vector<double> dq, kxl, kyl;
      divq(x, dq);
      applyK(Kx_, lam + rh1, m_q, kxl);
      applyK(Ky_, lam + rh1, m_q, kyl);
      Index k = 0;
      for (int i = 0; i < m_u; ++i) values[k++] = obj_factor;             // (u,u)
      for (int e = 0; e < m_q; ++e)                                       // (θ,r)
         values[k++] = lam[rh2x + e] * std::sin(x[oTh + e])
                     - lam[rh2y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e)                                       // (θ,δ)
         values[k++] = lam[rh3x + e] * std::sin(x[oTh + e])
                     - lam[rh3y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) {                                     // (θ,θ)
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         values[k++] = x[oR + e] * (lam[rh2x + e] * c + lam[rh2y + e] * s)
                     + x[oD + e] * (lam[rh3x + e] * c + lam[rh3y + e] * s)
                     + obj_factor * eps_theta_;
      }
      for (int e = 0; e < m_q; ++e) values[k++] = -lam[rcomp + e];        // (δ,r)
      const double dQa = dQ(x[oa]);
      for (int e = 0; e < m_q; ++e) values[k++] = dQa * kxl[e];           // (α,qx)
      for (int e = 0; e < m_q; ++e) values[k++] = dQa * kyl[e];           // (α,qy)
      double aa = 0.0;
      for (int i = 0; i < m_u; ++i) aa += lam[rh1 + i] * dq[i];
      values[k++] = d2Q(x[oa]) * aa + obj_factor * reg_alpha_;            // (α,α)
      return true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_2D_TNLP_HPP
