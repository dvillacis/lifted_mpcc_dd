// simple_ipm.hpp — a minimal primal-dual interior point method, as a second
// host for SchurDD next to IPOPT.  It follows the setting of Lueg et al.
// (Sec. 2 and 3.5) rather than IPOPT's, so that the two can be compared on the
// same problem with the same linear algebra:
//
//   · fraction-to-the-boundary steps, NO line search, no restoration phase
//     ("we choose step sizes by a simple fraction-to-the-boundary rule");
//   · per-tile regularization δ_k (Sec. 3.5): if In(W_k) is wrong, only tile k
//     is regularized (--reg local), or every tile (--reg global);
//   · no regularization on the complicating variables (C is the objective's
//     curvature there, nothing added), as in eq. (5);
//   · In(S) only as far as the interface solve can tell: exactly in direct
//     mode, through the PCG curvature test (or a factorization, --pcg-inertia
//     exact) in PCG mode.  If S is not positive definite every tile is raised.
//
// Problem class: any TNLP with bounds and general constraints g_l ≤ g(x) ≤ g_u.
// As in IPOPT, rows with g_l = g_u are equalities c(x) = 0 (written c(x) − g_l),
// the others get a slack: d(x) − s = 0 with g_l ≤ s ≤ g_u.  The primal vector is
// p = [x | s], the rows are [c | d] in their order of appearance — IPOPT's KKT
// layout [x | s | λ_c | λ_d], so a TNLP's owner map for IPOPT works unchanged.
//
//   barrier problem   min s_f·f(x) − μ Σ ln(p − l) − μ Σ ln(u − p)   s.t. A(p) = 0
//   Lagrangian        s_f·f + λᵀA   (IPOPT's sign convention, so eval_h fits)
//   Newton system     ⎡ W + Σ + δ_k   Aᵀ  ⎤ ⎡dp⎤ = − ⎡ ∇_p L − μ/(p−l) + μ/(u−p) ⎤
//                     ⎣ A            −δ_c ⎦ ⎣dλ⎦     ⎣ A(p)                      ⎦
//   with Σ = z_L/(p−l) + z_U/(u−p) and W the Hessian of L (x block only); the
//   bound multipliers follow from dz_L = μ/(p−l) − z_L − Σ_L dp,
//   dz_U = μ/(u−p) − z_U + Σ_U dp.
//
// μ follows IPOPT's monotone rule: once the barrier problem is solved to κ_ε·μ,
// μ ← max(tol/10, min(κ_μ·μ, μ^θ)).  Error measures use IPOPT's s_d, s_c.  The
// constraint multipliers start from IPOPT's least-squares estimate.  The TNLP's
// intermediate_callback is called once per iteration (without IPOPT's internal
// objects), so problems that steer themselves by μ — the MPCC's t-continuation —
// behave as under IPOPT; returning false stops the solve (status 2).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "IpTNLP.hpp"
#include "schur_dd.hpp"

class SimpleIPM {
public:
   enum class Reg { Local, Global };
   struct Options {
      double tol = 1e-8;
      int max_iter = 500;
      double obj_scaling = 1.0;     // s_f
      double mu0 = 0.1;
      double kappa_eps = 10.0;      // IPOPT barrier_tol_factor
      double kappa_mu = 0.2;        // IPOPT mu_linear_decrease_factor
      double theta_mu = 1.5;        // IPOPT mu_superlinear_decrease_power
      double bound_push = 1e-2;     // IPOPT bound_push = bound_frac
      double acceptable_tol = 0.0;  // > 0: also stop at E₀ ≤ this ...
      double acceptable_dual = 1e10;//     ... with inf_du ≤ this, for acceptable_iter its
      int acceptable_iter = 15;
      bool ls_mult_init = true;     // least-squares λ₀ (IPOPT's default)
      Reg reg = Reg::Local;
      int print_level = 1;          // 0: quiet, 1: one line per iteration
   };
   struct Result {
      int status = -1;            // 0 converged, 1 iteration limit, 2 stopped by the
                                  // problem's callback, 3 acceptable, −1 failure
      int iters = 0;
      double obj = 0.0, inf_pr = 0.0, inf_du = 0.0, wall = 0.0;
      long tile_corrections = 0;  // factorizations refused for a wrong In(W_k)
      long s_corrections = 0;     // ... for S not positive definite
      long singular = 0;          // ... for a zero pivot
      long tiles_regularized = 0; // Σ over iterations of #tiles with δ_k > 0
   };

   Result solve(Ipopt::TNLP& nlp, const std::vector<int>& owner, int n_tiles,
                const dd::SchurDD::Options& dopt, const dd::SchurDD::Hints* hints,
                dd::SchurDD::Stats* stats) {
      using Index = Ipopt::Index;
      const auto t0 = std::chrono::steady_clock::now();
      Result res;
      const Options& o = opt;

      // ---- problem data, rows split into [c | d]
      Index n, m, nnz_j, nnz_h;
      Ipopt::TNLP::IndexStyleEnum style;
      nlp.get_nlp_info(n, m, nnz_j, nnz_h, style);
      const int base = style == Ipopt::TNLP::FORTRAN_STYLE ? 1 : 0;
      std::vector<double> xl(n), xu(n), gl(m), gu(m);
      nlp.get_bounds_info(n, xl.data(), xu.data(), m, gl.data(), gu.data());
      std::vector<int> rowpos(m), drows;
      int mc = 0;
      for (Index r = 0; r < m; ++r)
         if (gl[r] == gu[r]) rowpos[r] = mc++;
         else drows.push_back(r);
      const int md = (int)drows.size();
      for (int i = 0; i < md; ++i) rowpos[drows[i]] = mc + i;
      const int np = n + md;          // primals: x and slacks
      const int N = np + m;           // KKT dimension

      std::vector<double> pl(np), pu(np);
      for (Index i = 0; i < n; ++i) { pl[i] = xl[i]; pu[i] = xu[i]; }
      for (int i = 0; i < md; ++i) { pl[n + i] = gl[drows[i]]; pu[n + i] = gu[drows[i]]; }
      std::vector<double> p(np, 0.0), lam(m, 0.0), zl(np, 0.0), zu(np, 0.0);
      nlp.get_starting_point(n, true, p.data(), false, nullptr, nullptr, m, false, nullptr);
      std::vector<double> g(m);
      nlp.eval_g(n, p.data(), true, m, g.data());
      for (int i = 0; i < md; ++i) p[n + i] = g[drows[i]];
      std::vector<char> hl(np), hu(np);
      int nbounds = 0;
      for (int i = 0; i < np; ++i) {
         hl[i] = pl[i] > -1e19;
         hu[i] = pu[i] < 1e19;
         nbounds += hl[i] + hu[i];
         // push the start strictly inside (IPOPT's bound_push / bound_frac)
         const double k = o.bound_push;
         const double ql = hl[i] ? std::min(k * std::max(1.0, std::abs(pl[i])),
                                            hu[i] ? k * (pu[i] - pl[i]) : 1e300) : 0.0;
         const double qu = hu[i] ? std::min(k * std::max(1.0, std::abs(pu[i])),
                                            hl[i] ? k * (pu[i] - pl[i]) : 1e300) : 0.0;
         if (hl[i]) p[i] = std::max(p[i], pl[i] + ql);
         if (hu[i]) p[i] = std::min(p[i], pu[i] - qu);
         if (hl[i]) zl[i] = 1.0;
         if (hu[i]) zu[i] = 1.0;
      }

      // ---- KKT pattern: Hessian | p diagonal | Jacobian | −I of the slacks | row diagonal
      std::vector<Index> jr(nnz_j), jc(nnz_j), hr(nnz_h), hc(nnz_h);
      std::vector<double> lam_orig(m, 0.0);
      nlp.eval_jac_g(n, p.data(), true, m, nnz_j, jr.data(), jc.data(), nullptr);
      nlp.eval_h(n, p.data(), true, 1.0, m, lam_orig.data(), true, nnz_h, hr.data(), hc.data(),
                 nullptr);
      for (Index t = 0; t < nnz_j; ++t) { jr[t] -= base; jc[t] -= base; }
      for (Index t = 0; t < nnz_h; ++t) { hr[t] -= base; hc[t] -= base; }
      const int off_pd = nnz_h, off_j = off_pd + np, off_s = off_j + nnz_j, off_rd = off_s + md;
      const int nnz = off_rd + m;
      std::vector<int> ir(nnz), ic(nnz);
      for (Index t = 0; t < nnz_h; ++t) {
         ir[t] = std::max(hr[t], hc[t]);
         ic[t] = std::min(hr[t], hc[t]);
      }
      for (int i = 0; i < np; ++i) ir[off_pd + i] = ic[off_pd + i] = i;
      for (Index t = 0; t < nnz_j; ++t) {
         ir[off_j + t] = np + rowpos[jr[t]];
         ic[off_j + t] = jc[t];
      }
      for (int i = 0; i < md; ++i) {
         ir[off_s + i] = np + mc + i;
         ic[off_s + i] = n + i;
      }
      for (Index r = 0; r < m; ++r) ir[off_rd + r] = ic[off_rd + r] = np + r;

      dd::SchurDD dd;
      dd.set_stats_sink(stats);
      if (!dd.set_structure(N, ir, ic, owner, np, n_tiles, dopt, hints)) return res;
      double* val = dd.values();

      // ---- evaluation at the current point: f, A(p), Jacobian, ∇_p L
      const double s_f = o.obj_scaling;
      double f = 0.0;
      std::vector<double> grad(n), A(m), jv(nnz_j), gL(np), hv(nnz_h);
      auto evaluate = [&](bool new_x) {
         nlp.eval_f(n, p.data(), new_x, f);
         nlp.eval_grad_f(n, p.data(), false, grad.data());
         nlp.eval_g(n, p.data(), false, m, g.data());
         nlp.eval_jac_g(n, p.data(), false, m, nnz_j, nullptr, nullptr, jv.data());
         for (Index r = 0; r < m; ++r)
            A[rowpos[r]] = gl[r] == gu[r] ? g[r] - gl[r] : g[r];
         for (int i = 0; i < md; ++i) A[mc + i] -= p[n + i];
         for (Index i = 0; i < n; ++i) gL[i] = s_f * grad[i];
         for (Index t = 0; t < nnz_j; ++t) gL[jc[t]] += jv[t] * lam[rowpos[jr[t]]];
         for (int i = 0; i < md; ++i) gL[n + i] = -lam[mc + i];
      };
      auto write_jacobian = [&]() {
         for (Index t = 0; t < nnz_j; ++t) val[off_j + t] = jv[t];
         for (int i = 0; i < md; ++i) val[off_s + i] = -1.0;
      };

      // ---- least-squares multipliers: [I Aᵀ; A 0][w; λ] = [−(∇_p f − z_L + z_U); 0]
      evaluate(true);
      if (o.ls_mult_init && m > 0) {
         for (Index t = 0; t < nnz_h; ++t) val[t] = 0.0;
         for (int i = 0; i < np; ++i) val[off_pd + i] = 1.0;
         write_jacobian();
         for (Index r = 0; r < m; ++r) val[off_rd + r] = 0.0;
         std::vector<double> b(N, 0.0);
         for (Index i = 0; i < n; ++i) b[i] = -(s_f * grad[i] - zl[i] + zu[i]);
         for (int i = n; i < np; ++i) b[i] = -(-zl[i] + zu[i]);
         bool ok = dd.factorize() && dd.solve(b.data());
         double big = 0.0;
         for (Index r = 0; r < m; ++r) big = std::max(big, std::abs(b[np + r]));
         if (ok && big <= 1e3)
            for (Index r = 0; r < m; ++r) lam[r] = b[np + r];
         evaluate(false);
      }

      std::vector<double> sig(np), rhs(N), d(N), delta(n_tiles, 0.0), last(n_tiles, 0.0);
      double mu = o.mu0, delta_c = 0.0, ap = 0.0, ad = 0.0, dmax = 0.0;
      int acc_count = 0;
      bool fail = false;

      if (o.print_level > 0)
         std::printf("iter    objective    inf_pr   inf_du lg(mu)  ||dp||  lg(rg) #reg  alpha_pr "
                     "alpha_du   pcg  lin.res\n");
      for (res.iters = 0;; ++res.iters) {
         // ---- optimality errors (IPOPT's scaled E_μ)
         auto errors = [&](double mu_, double& e_pr, double& e_du, double& e_co) {
            e_pr = 0.0;
            for (double v : A) e_pr = std::max(e_pr, std::abs(v));
            e_du = 0.0;
            e_co = 0.0;
            for (int i = 0; i < np; ++i) {
               e_du = std::max(e_du, std::abs(gL[i] - zl[i] + zu[i]));
               if (hl[i]) e_co = std::max(e_co, std::abs((p[i] - pl[i]) * zl[i] - mu_));
               if (hu[i]) e_co = std::max(e_co, std::abs((pu[i] - p[i]) * zu[i] - mu_));
            }
         };
         auto scaled = [&](double mu_) {
            double y1 = 0.0, z1 = 0.0;
            for (double v : lam) y1 += std::abs(v);
            for (int i = 0; i < np; ++i) z1 += zl[i] + zu[i];
            const double s_d = std::max(100.0, (y1 + z1) / std::max(1, m + nbounds)) / 100.0;
            const double s_c = std::max(100.0, z1 / std::max(1, nbounds)) / 100.0;
            double e_pr, e_du, e_co;
            errors(mu_, e_pr, e_du, e_co);
            return std::max({e_du / s_d, e_pr, e_co / s_c});
         };
         double e_pr, e_du, e_co;
         errors(0.0, e_pr, e_du, e_co);
         res.obj = f;
         res.inf_pr = e_pr;
         res.inf_du = e_du / s_f;
         const double E0 = scaled(0.0);
         if (!std::isfinite(E0)) { fail = true; break; }
         if (E0 <= o.tol) { res.status = 0; break; }
         if (o.acceptable_tol > 0.0 && E0 <= o.acceptable_tol && res.inf_du <= o.acceptable_dual) {
            if (++acc_count >= o.acceptable_iter) { res.status = 3; break; }
         } else {
            acc_count = 0;
         }
         if (res.iters >= o.max_iter) { res.status = 1; break; }
         // the problem's own per-iteration hook (may change the problem: re-evaluate)
         if (!nlp.intermediate_callback(Ipopt::RegularMode, res.iters, f, e_pr, res.inf_du, mu,
                                        0.0, dmax, ad, ap, 0, nullptr, nullptr)) {
            res.status = 2;
            break;
         }
         evaluate(false);
         // monotone μ update
         const double mu_min = o.tol / 10.0;
         while (mu > mu_min && scaled(mu) <= o.kappa_eps * mu)
            mu = std::max(mu_min, std::min(o.kappa_mu * mu, std::pow(mu, o.theta_mu)));

         // ---- Newton system
         for (Index r = 0; r < m; ++r) lam_orig[r] = lam[rowpos[r]];
         nlp.eval_h(n, p.data(), false, s_f, m, lam_orig.data(), true, nnz_h, nullptr, nullptr,
                    hv.data());
         for (int i = 0; i < np; ++i) {
            sig[i] = (hl[i] ? zl[i] / (p[i] - pl[i]) : 0.0) + (hu[i] ? zu[i] / (pu[i] - p[i]) : 0.0);
            rhs[i] = -gL[i] + (hl[i] ? mu / (p[i] - pl[i]) : 0.0) - (hu[i] ? mu / (pu[i] - p[i]) : 0.0);
         }
         for (Index r = 0; r < m; ++r) rhs[np + r] = -A[r];

         const long cg0 = stats->cg_iters;
         std::fill(delta.begin(), delta.end(), 0.0);
         bool solved = false;
         for (int attempt = 0; attempt < 60 && !solved; ++attempt) {
            for (Index t = 0; t < nnz_h; ++t) val[t] = hv[t];
            for (int i = 0; i < np; ++i)
               val[off_pd + i] = sig[i] + (owner[i] >= 0 ? delta[owner[i]] : 0.0);
            write_jacobian();
            for (Index r = 0; r < m; ++r) val[off_rd + r] = -delta_c;

            if (!dd.factorize()) {
               ++res.singular;
               if (delta_c == 0.0) delta_c = 1e-8 * std::pow(mu, 0.25);
               else if (!raise_all(delta, last, o.reg)) break;
               continue;
            }
            std::vector<int> bad;
            bool too_few = false;
            for (int k = 0; k < n_tiles; ++k)
               if (dd.tile_negative(k) != dd.tile_duals(k)) {
                  bad.push_back(k);
                  too_few = too_few || dd.tile_negative(k) < dd.tile_duals(k);
               }
            // Too FEW negative eigenvalues: no δ_w can fix that (the tile's own
            // Jacobian is rank-deficient), so add the dual regularization δ_c,
            // as IPOPT does when its solver reports the matrix singular.
            if (too_few && delta_c == 0.0) {
               ++res.tile_corrections;
               delta_c = 1e-8 * std::pow(mu, 0.25);
               continue;
            }
            if (!bad.empty()) {   // eq. (24) violated: regularize those tiles
               ++res.tile_corrections;
               bool ok = true;
               if (o.reg == Reg::Local)
                  for (int k : bad) ok = ok && raise(delta[k], last[k]);
               else
                  ok = raise_all(delta, last, o.reg);
               if (!ok) break;
               continue;
            }
            if (dd.s_negative() > 0) {   // eq. (25) violated, seen exactly
               ++res.s_corrections;
               if (!raise_all(delta, last, o.reg)) break;
               continue;
            }
            d = rhs;
            if (!dd.solve(d.data())) {   // PCG met pᵀSp ≤ 0
               ++res.s_corrections;
               if (!raise_all(delta, last, o.reg)) break;
               continue;
            }
            solved = true;
         }
         if (!solved) { fail = true; break; }
         int nreg = 0;
         dmax = 0.0;
         for (int k = 0; k < n_tiles; ++k)
            if (delta[k] > 0.0) {
               last[k] = delta[k];
               ++nreg;
               dmax = std::max(dmax, delta[k]);
            }
         res.tiles_regularized += nreg;

         // ---- accuracy of the step: ‖K d − rhs‖ / ‖rhs‖ against the true matrix
         std::vector<double> Kd(N, 0.0);
         for (int t = 0; t < nnz; ++t) {
            Kd[ir[t]] += val[t] * d[ic[t]];
            if (ir[t] != ic[t]) Kd[ic[t]] += val[t] * d[ir[t]];
         }
         double rn = 0.0, bn = 0.0;
         for (int i = 0; i < N; ++i) {
            rn += (Kd[i] - rhs[i]) * (Kd[i] - rhs[i]);
            bn += rhs[i] * rhs[i];
         }
         const double linres = std::sqrt(rn / std::max(bn, 1e-300));

         // ---- bound multipliers and fraction to the boundary
         const double tau = std::max(0.99, 1.0 - mu);
         ap = 1.0;
         ad = 1.0;
         double dpn = 0.0;
         std::vector<double> dzl(np, 0.0), dzu(np, 0.0);
         for (int i = 0; i < np; ++i) {
            const double dp = d[i];
            dpn = std::max(dpn, std::abs(dp));
            if (hl[i]) {
               const double s = p[i] - pl[i];
               dzl[i] = mu / s - zl[i] - zl[i] / s * dp;
               if (dp < 0.0) ap = std::min(ap, -tau * s / dp);
               if (dzl[i] < 0.0) ad = std::min(ad, -tau * zl[i] / dzl[i]);
            }
            if (hu[i]) {
               const double s = pu[i] - p[i];
               dzu[i] = mu / s - zu[i] + zu[i] / s * dp;
               if (dp > 0.0) ap = std::min(ap, tau * s / dp);
               if (dzu[i] < 0.0) ad = std::min(ad, -tau * zu[i] / dzu[i]);
            }
         }
         for (int i = 0; i < np; ++i) {
            p[i] += ap * d[i];
            zl[i] += ad * dzl[i];
            zu[i] += ad * dzu[i];
            // keep z within a factor 1e10 of μ/s (IPOPT's κ_Σ safeguard)
            if (hl[i]) {
               const double s = p[i] - pl[i];
               zl[i] = std::max(std::min(zl[i], 1e10 * mu / s), mu / (1e10 * s));
            }
            if (hu[i]) {
               const double s = pu[i] - p[i];
               zu[i] = std::max(std::min(zu[i], 1e10 * mu / s), mu / (1e10 * s));
            }
         }
         for (Index r = 0; r < m; ++r) lam[r] += ap * d[np + r];

         if (o.print_level > 0)
            std::printf("%4d %14.7e %8.2e %8.2e %6.1f %7.2e %6s %4d %8.2e %8.2e %5ld %8.1e\n",
                        res.iters, f, e_pr, e_du / s_f, std::log10(mu), dpn,
                        dmax > 0 ? fmt_lg(dmax).c_str() : "-", nreg, ap, ad,
                        stats->cg_iters - cg0, linres);
         evaluate(true);
      }
      if (fail) res.status = -1;
      for (Index r = 0; r < m; ++r) lam_orig[r] = lam[rowpos[r]];
      nlp.finalize_solution(res.status == 0 || res.status == 3 ? Ipopt::SUCCESS
                            : res.status == 2                  ? Ipopt::USER_REQUESTED_STOP
                                                               : Ipopt::MAXITER_EXCEEDED,
                            n, p.data(), zl.data(), zu.data(), m, g.data(), lam_orig.data(),
                            res.obj, nullptr, nullptr);
      res.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return res;
   }

   Options opt;

private:
   static std::string fmt_lg(double v) {
      char b[16];
      std::snprintf(b, sizeof b, "%.1f", std::log10(v));
      return b;
   }
   // IPOPT's δ_w schedule, per tile: first try 1e-4 (or a third of the last
   // value that worked), then ×100 the first time, ×8 after.
   static bool raise(double& delta, double last) {
      if (delta == 0.0) delta = last == 0.0 ? 1e-4 : std::max(1e-20, last / 3.0);
      else delta *= last == 0.0 ? 100.0 : 8.0;
      return delta < 1e40;
   }
   static bool raise_all(std::vector<double>& delta, const std::vector<double>& last, Reg reg) {
      if (reg == Reg::Global) {   // one δ shared by all tiles
         double d = delta.empty() ? 0.0 : delta[0];
         double l = 0.0;
         for (double v : last) l = std::max(l, v);
         if (!raise(d, l)) return false;
         std::fill(delta.begin(), delta.end(), d);
         return true;
      }
      for (size_t k = 0; k < delta.size(); ++k)
         if (!raise(delta[k], last[k])) return false;
      return true;
   }
};
