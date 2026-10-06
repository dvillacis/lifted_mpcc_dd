// DDS_VERIFY_WK=1 — rebuild every subdomain block W_k from the TNLP's own
// derivatives at the current IPOPT iterate and compare it, entry by entry, with
// what dd_solver_simple.hpp assembled from IPOPT's KKT triplets.
//
// In the consensus formulation W_k should be (unknowns of tile k only; x_k the
// tile's own primal variables, y_k its local copies, λ_k the multipliers of its
// constraint rows, ρ_k those of its linking rows y_k − N_kᵀ y = 0):
//
//          x_k              y_k              λ_k          ρ_k
//   x_k [ ∇²_xx L + Σ + δ_H   ∇²_xy L          (∇_x h)ᵀ      0  ]
//   y_k [ ∇²_yx L         ∇²_yy L + Σ + δ_H    (∇_y h)ᵀ      I  ]
//   λ_k [ ∇_x h            ∇_y h              −δ_C I        0  ]
//   ρ_k [ 0                I                  0           −δ_C ]
//
// plus, because IPOPT keeps them explicit, the inequality slacks s (diagonal
// Σ_s + δ_H, coupled to their own multiplier by −1) and their multipliers
// (diagonal −δ_D). The checker reports, per verified factorization:
//   * entries / missing / unexpected — every derivative entry whose row and
//     column both belong to tile k must be in W_k, and nothing else may be;
//   * the largest deviation of the Hessian off-diagonals and of the Jacobian
//     entries from the TNLP's eval_h / eval_jac_g at the iterate;
//   * the linking block: each ρ row has exactly one entry, +1 at its own copy;
//   * the diagonals: W_ii − ∇²L_ii − Σ_i over all primal unknowns (must be one
//     constant, IPOPT's δ_H = δ_w) and the multiplier diagonals (−δ_C).
#ifndef VERIFY_WK_HPP
#define VERIFY_WK_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "ipopt_phase.hpp"
#include "mpcc_2d_consensus_tnlp.hpp"

namespace verify_wk {

struct Range {
   double lo = std::numeric_limits<double>::infinity(), hi = -lo;
   long n = 0;
   void add(double v) { lo = std::min(lo, v); hi = std::max(hi, v); ++n; }
};

inline void install(Ipopt::Mpcc2DTNLP* p, const std::vector<int>& owner) {
   using Ipopt::Index;
   using Ipopt::Number;
   Index n, m, nnzj, nnzh;
   Ipopt::TNLP::IndexStyleEnum style;
   p->get_nlp_info(n, m, nnzj, nnzh, style);
   const int n_eq = p->n_eq, n_ineq = p->n_ineq;
   const int dC = n + n_ineq, dD = n + n_ineq + n_eq;   // λ_c and λ_d offsets
   auto* cons = dynamic_cast<Ipopt::Mpcc2DConsensusTNLP*>(p);

   std::vector<Index> jr(nnzj), jc(nnzj), hr(nnzh), hc(nnzh);
   p->eval_jac_g(n, nullptr, false, m, nnzj, jr.data(), jc.data(), nullptr);
   p->eval_h(n, nullptr, false, 1.0, m, nullptr, false, nnzh, hr.data(), hc.data(), nullptr);
   // IPOPT relaxes every finite bound by bound_relax_factor = 1e-8 (default)
   std::vector<Number> xl(n), xu(n), gl(m), gu(m);
   p->get_bounds_info(n, xl.data(), xu.data(), m, gl.data(), gu.data());
   for (int i = 0; i < n; ++i) {
      if (xl[i] > -1e19) xl[i] -= 1e-8 * std::max(1.0, std::abs(xl[i]));
      if (xu[i] < 1e19) xu[i] += 1e-8 * std::max(1.0, std::abs(xu[i]));
   }

   ipopt_phase::wk_hook() = [=](const std::vector<std::vector<ipopt_phase::WkEntry>>& blocks) {
      const ipopt_phase::Iterate& it = ipopt_phase::iterate();
      const Number* x = it.x.data();
      std::vector<Number> jv(nnzj), hv(nnzh);
      p->eval_jac_g(n, x, true, m, nnzj, nullptr, nullptr, jv.data());
      p->eval_h(n, x, true, 1.0, m, it.lambda.data(), true, nnzh, nullptr, nullptr, hv.data());
      std::map<std::pair<int, int>, double> J, H;           // H keyed (max, min)
      for (Index t = 0; t < nnzj; ++t) J[{jr[t], jc[t]}] += jv[t];
      for (Index t = 0; t < nnzh; ++t)
         H[{std::max(hr[t], hc[t]), std::min(hr[t], hc[t])}] += hv[t];
      auto sigma = [&](int i) {
         double s = 0.0;
         if (xl[i] > -1e19) s += it.z_L[i] / (x[i] - xl[i]);
         if (xu[i] < 1e19) s += it.z_U[i] / (xu[i] - x[i]);
         return s;
      };
      auto row_of = [&](int g) {            // TNLP constraint row of a multiplier
         return g < dD ? g - dC : n_eq + (g - dD);
      };
      auto is_link = [&](int row) {
         return cons && row >= cons->rlink && row < cons->rlink + cons->n_link;
      };

      long entries = 0, missing = 0, unexpected = 0, link_bad = 0;
      double errH = 0.0, errJ = 0.0, errS = 0.0;
      Range dH, dC_, dLink, dD_, dSlack;
      long nx = 0, ny = 0, nl = 0, nr = 0, ns = 0;
      for (size_t k = 0; k < blocks.size(); ++k) {
         std::map<std::pair<int, int>, double> W;
         for (const auto& e : blocks[k])
            W[{std::max(e.gi, e.gj), std::min(e.gi, e.gj)}] += e.v;
         entries += (long)W.size();
         std::map<int, int> link_hits;      // ρ row → off-diagonal entries seen
         for (const auto& [key, v] : W) {
            const int a = key.first, b = key.second;
            if (a < n) {                                            // primal–primal
               const auto h = H.find({a, b});
               const double hval = (h == H.end()) ? 0.0 : h->second;
               if (a != b) errH = std::max(errH, std::abs(v - hval));
               else dH.add(v - hval - sigma(a));
            } else if (a < dC) {                                    // slack row
               if (a == b) dSlack.add(v);
               else if (v != 0.0) ++unexpected;
            } else {                                                // multiplier row
               const int row = row_of(a);
               if (a == b) {
                  if (a >= dD) dD_.add(v);
                  else if (is_link(row)) dLink.add(v);
                  else dC_.add(v);
               } else if (b < n) {                                  // ∇h entry
                  const auto j = J.find({row, b});
                  const double jval = (j == J.end()) ? 0.0 : j->second;
                  errJ = std::max(errJ, std::abs(v - jval));
                  if (j == J.end() && v != 0.0) ++unexpected;
                  if (is_link(row)) {
                     ++link_hits[row];
                     const int l = row - cons->rlink;
                     if (b != cons->link_copy_[l] || v != 1.0) ++link_bad;
                  }
               } else if (b < dC) {                                 // multiplier × slack
                  const bool own = (a >= dD) && (b - n == a - dD);
                  if (own) errS = std::max(errS, std::abs(v + 1.0));
                  else if (v != 0.0) ++unexpected;
               } else if (v != 0.0) {
                  ++unexpected;                                     // multiplier × multiplier
               }
            }
         }
         // completeness: every derivative entry inside tile k must be in W_k
         for (const auto& [rc, val] : J) {
            const int g = (rc.first < n_eq) ? dC + rc.first : dD + (rc.first - n_eq);
            if (owner[g] == (int)k && owner[rc.second] == (int)k &&
                !W.count({std::max(g, rc.second), std::min(g, rc.second)}))
               ++missing;
         }
         for (const auto& [ij, val] : H)
            if (owner[ij.first] == (int)k && owner[ij.second] == (int)k && !W.count(ij))
               ++missing;
         // each linking row of the tile: exactly one off-diagonal entry
         for (int g = 0; g < (int)owner.size(); ++g) {
            if (owner[g] != (int)k) continue;
            if (g < n) {
               (cons && g >= cons->n_orig) ? ++ny : ++nx;
            } else if (g < dC) {
               ++ns;
            } else {
               const int row = row_of(g);
               if (is_link(row)) {
                  ++nr;
                  if (link_hits[row] != 1) ++link_bad;
               } else {
                  ++nl;
               }
            }
         }
      }
      auto rg = [](const Range& r) {
         static char buf[8][64];
         static int slot = 0;
         char* b = buf[slot++ % 8];
         if (r.n == 0) std::snprintf(b, 64, "none");
         else std::snprintf(b, 64, "[%.3e, %.3e]", r.lo, r.hi);
         return b;
      };
      std::printf("[verify-Wk] iter=%d blocks=%zu | unknowns: x_k=%ld y_k=%ld slacks=%ld "
                  "lambda_k=%ld rho_k=%ld | entries=%ld missing=%ld unexpected=%ld\n",
                  it.iter, blocks.size(), nx, ny, ns, nl, nr, entries, missing, unexpected);
      std::printf("[verify-Wk]   max |W - d2L| off-diag=%.2e   max |W - dh|=%.2e   "
                  "max |W + 1| (lambda_d, own slack)=%.2e   (rho,y)=I violations=%ld\n",
                  errH, errJ, errS, link_bad);
      std::printf("[verify-Wk]   W_ii - d2L_ii - Sigma_i (primal) in %s   lambda_c diag in %s   "
                  "rho diag in %s   lambda_d diag in %s   slack diag in %s\n",
                  rg(dH), rg(dC_), rg(dLink), rg(dD_), rg(dSlack));
      std::fflush(stdout);
   };
}

}  // namespace verify_wk

#endif  // VERIFY_WK_HPP
