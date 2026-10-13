// pdhg.hpp — the lower level of operators.hpp, solved directly:
//
//   min_u  ½uᵀAu − Fᵀu + Σ_e w_e |(K_ξ u)_e|,   w_e ≥ 0 the weights per cell.
//
// With q_e in the unit ball the optimality condition is A u − F + K_ξᵀ(w∘q) = 0,
// the row h1 of TvLearn.  Used for the warm start of tv_learn (w = α₀ everywhere)
// and to evaluate a learned weight map on held-out images.
//
//   A diagonal (denoise, inpaint): Chambolle–Pock with the exact prox of the data
//     term; accelerated (strong convexity min A_ii) for the first 300 iterations,
//     then fixed steps, as tv_mpcc.hpp's warm start.
//   A = CᵀC (deblur): Condat–Vũ, an explicit gradient step on the data term
//     (‖A‖ ≤ 1 for a normalized kernel), τ(½‖A‖ + σ‖L‖²) ≤ 0.99 with L = w∘K_ξ.
// Both stop when max |u⁺ − u| / τ ≤ tol (or after max_iter).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "operators.hpp"

struct PdhgResult {
   std::vector<double> u, qx, qy;
   int iters = 0;
   double residual = 0.0;
};

inline PdhgResult pdhg(const Instance& in, const Blur* blur, const std::vector<double>& w,
                       double tol = 1e-9, int max_iter = 20000) {
   const int N = in.N, nc = N - 1, m_u = N * N, m_q = nc * nc;
   PdhgResult R;
   double wmax = 0.0;
   for (double v : w) wmax = std::max(wmax, v);
   R.qx.assign(m_q, 0.0);
   R.qy.assign(m_q, 0.0);
   const bool diag = !in.diag.empty();
   if (wmax <= 0.0 && diag) {   // no regularization: A u = F
      R.u.resize(m_u);
      for (int i = 0; i < m_u; ++i) R.u[i] = in.rhs[i] / in.diag[i];
      return R;
   }
   const double L = std::max(wmax, 1e-12) * std::sqrt(8.0);   // ‖w∘K_ξ‖ ≤ max w · √8
   double mu_s = 1.0;
   if (diag) for (double a : in.diag) mu_s = std::min(mu_s, a);
   const double tau0 = diag ? 0.99 / L : 0.99 / (0.5 + L), sig0 = diag ? 0.99 / L : 1.0 / L;
   double tau = tau0, sig = sig0;
   std::vector<double> u = diag ? in.f : in.rhs, ubar = u, u_new(m_u), kt, au, gx, gy;
   std::vector<double> px(m_q), py(m_q);
   for (int it = 0; it < max_iter; ++it) {
      const bool accelerated = diag && it < 300;
      if (!accelerated && tau != tau0) { tau = tau0; sig = sig0; }
      apply_K(in, ubar, gx, gy);   // dual step and projection on |q_e| ≤ 1
      for (int e = 0; e < m_q; ++e) {
         R.qx[e] += sig * w[e] * gx[e];
         R.qy[e] += sig * w[e] * gy[e];
         const double nrm = std::max(1.0, std::hypot(R.qx[e], R.qy[e]));
         R.qx[e] /= nrm;
         R.qy[e] /= nrm;
         px[e] = w[e] * R.qx[e];
         py[e] = w[e] * R.qy[e];
      }
      apply_KT(in, px, py, kt);
      double residual = 0.0;
      if (diag) {
         for (int i = 0; i < m_u; ++i) {
            u_new[i] = (u[i] - tau * kt[i] + tau * in.rhs[i]) / (1.0 + tau * in.diag[i]);
            residual = std::max(residual, std::abs(u_new[i] - u[i]));
         }
      } else {
         apply_A(in, blur, u, au);
         for (int i = 0; i < m_u; ++i) {
            u_new[i] = u[i] - tau * (au[i] - in.rhs[i] + kt[i]);
            residual = std::max(residual, std::abs(u_new[i] - u[i]));
         }
      }
      residual /= tau;
      if (accelerated) {
         const double th = 1.0 / std::sqrt(1.0 + 2.0 * mu_s * tau);
         for (int i = 0; i < m_u; ++i) ubar[i] = u_new[i] + th * (u_new[i] - u[i]);
         tau *= th;
         sig /= th;
      } else {
         for (int i = 0; i < m_u; ++i) ubar[i] = 2.0 * u_new[i] - u[i];
      }
      u.swap(u_new);
      R.iters = it + 1;
      R.residual = residual;
      if (residual <= tol) break;
   }
   R.u = u;
   return R;
}

// The residual of the optimality condition, max |A u − F + K_ξᵀ(w∘q)|.
inline double pdhg_kkt(const Instance& in, const Blur* blur, const std::vector<double>& w,
                       const PdhgResult& R) {
   const int m_q = (int)R.qx.size();
   std::vector<double> px(m_q), py(m_q), kt, au;
   for (int e = 0; e < m_q; ++e) { px[e] = w[e] * R.qx[e]; py[e] = w[e] * R.qy[e]; }
   apply_KT(in, px, py, kt);
   apply_A(in, blur, R.u, au);
   double r = 0.0;
   for (size_t i = 0; i < au.size(); ++i) r = std::max(r, std::abs(au[i] - in.rhs[i] + kt[i]));
   return r;
}
