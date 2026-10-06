// ipm.hpp — a primal-dual interior point method in the setting of Lueg et al.
// (Sec. 2), with every Newton system solved by the Schur-complement domain
// decomposition of schur_dd.hpp:
//
//   · a filter line search on (θ, φ_μ) (Wächter & Biegler 2006, IPOPT's
//     constants), backtracking from the fraction-to-the-boundary step.  If
//     nothing down to α_min is acceptable, a restoration phase minimizes the
//     ℓ₁ infeasibility (restoration.hpp) with this same method and the same
//     decomposition, until the original problem accepts a point; if that
//     fails too, the full step is taken and the filter cleared.
//     --line-search none gives the plain fraction-to-the-boundary method of
//     the paper.  Optionally (stall_iter) the restoration phase also starts
//     when a barrier problem makes no progress while nearly feasible: E_μ not
//     halved in stall_iter iterations with θ ≤ θ_min;
//   · one regularization δ for all tile primals (IPOPT's δ_w schedule), raised
//     when some In(W_k) is wrong (eq. 24) or S is not positive definite (eq. 25,
//     seen through the PCG curvature test, an indefinite Schwarz block S̃_k, or
//     the direct factorization); nothing on the consensus variables.  With
//     δ_c > 0 the factorization stops at the first tile with a wrong
//     inertia.  Each iteration tries δ = 0 first (IPOPT), or, with
//     delta_from_last, δ_last/3 after an iteration that needed δ > 0;
//   · a dual regularization δ_c when a tile has too FEW negative eigenvalues
//     (its own Jacobian is rank-deficient: no δ can fix that) or a pivot is zero;
//   · elastic variables (nlp.hpp; the restoration phase's p and n) are
//     eliminated from the Newton systems: e enters row r as c_e·x_e and has the
//     diagonal a_e = Σ_e + δ, so  dx_e = (rhs_e − c_e dλ_r)/a_e, and row r gets
//     −c_e²/a_e on its diagonal and −c_e·rhs_e/a_e on its right-hand side.  The
//     KKT pattern is then that of the problem without them, and the restoration
//     phase reuses the original problem's decomposition.
//
// Problem class: NLP (nlp.hpp), bounds and general constraints g_l ≤ g(x) ≤ g_u.
// Rows with g_l = g_u are equalities c(x) = 0 (written c(x) − g_l), the others
// get a slack: d(x) − s = 0 with g_l ≤ s ≤ g_u.  The primal vector is
// p = [x | s], the rows are [c | d] in their order of appearance.
//
//   barrier problem   min f(x) − μ Σ ln(p − l) − μ Σ ln(u − p)   s.t. A(p) = 0
//   Lagrangian        f + λᵀA
//   Newton system     ⎡ W + Σ + δ   Aᵀ  ⎤ ⎡dp⎤ = − ⎡ ∇_p L − μ/(p−l) + μ/(u−p) ⎤
//                     ⎣ A          −δ_c ⎦ ⎣dλ⎦     ⎣ A(p)                      ⎦
//   with Σ = z_L/(p−l) + z_U/(u−p) and W the Hessian of L (x block only); the
//   bound multipliers follow from dz_L = μ/(p−l) − z_L − Σ_L dp,
//   dz_U = μ/(u−p) − z_U + Σ_U dp.
//
// μ follows IPOPT's monotone rule: once the barrier problem is solved to κ_ε·μ,
// μ ← max(μ_min, min(κ_μ·μ, μ^θ)), μ_min = tol/10 unless set (tv_dd: t_min/10),
// with at most resto_mu_steps decreases right after a restoration phase.  Error measures use IPOPT's s_d, s_c.  The
// constraint multipliers start from IPOPT's least-squares estimate.  The
// problem's iteration() hook runs once per iteration; returning false stops
// the solve (status 2).
//
// MPI: every rank runs this method on its LOCAL problem (nlp.hpp): the
// unknowns of its tiles and the consensus variables.  Elementwise work is
// local; maxima and minima are reduced over the ranks; sums go through
// dd::TileSum, one partial per tile added in tile order, and the consensus
// rows of Jᵀλ and K·d through SchurDD::border_product, likewise in tile order.
// So every rank takes the same decisions, and a run is bit-identical for any
// number of ranks.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "nlp.hpp"
#include "restoration.hpp"
#include "schur_dd.hpp"

class IPM {
public:
   struct Options {
      double tol = 1e-8;
      int max_iter = 500;
      double mu0 = 0.1;
      double kappa_eps = 10.0;      // IPOPT barrier_tol_factor
      double kappa_mu = 0.2;        // IPOPT mu_linear_decrease_factor
      double theta_mu = 1.5;        // IPOPT mu_superlinear_decrease_power
      double vw_mu = 0.0;           // > 0: once μ ≤ vw_mu, the step of Raghunathan & Biegler
                                    // (2005), eq. 3.7 with choice (ii) of 3.8: in the block
                                    // Λ dz + Z dλ_z = −(ZΛe − μe) of every bound, Z is
                                    // replaced by Z + ηΛ (the right-hand side is kept).
                                    // As in IPOPT-C's source (Fortran IPOPT, update_mpec_eta.f):
                                    // η = 0.1 μ_prev/(1 + max(‖c(x)‖∞, ‖z_L‖∞, ‖z_U‖∞)) (the
                                    // paper writes ‖λ‖∞), off in the restoration phase,
                                    // μ_thresh = 5e-6 there
      bool vw_comp_only = false;    // ... only on the NLP's complementarity_bounds()
      bool vw_ls_grad = true;       // ... and in the line search's ∇φ_μ·d, the barrier term
                                    // ∓μ/s becomes ∓(z + (μ − s z)/(s + ηz)), as in IPOPT-C's
                                    // filter.F, so the Armijo and switching tests see the
                                    // modified step (false: plain ∓μ/s)
      int mu_max_steps = 0;         // > 0: at most this many μ decreases per iteration
                                    // (0: as many as E_μ ≤ κ_ε·μ allows, IPOPT)
      double mu_min = 0.0;          // > 0: the floor of μ (0: tol/10, IPOPT)
      int resto_mu_steps = 1;       // > 0: at most this many μ decreases in the iteration
                                    // right after a successful restoration phase (0: no cap).
                                    // The phase ends with least-squares λ, which can make E_μ
                                    // look small for that one iteration and let μ fall to its
                                    // floor at once (mariposa N=640: 3.2e-4 → 5e-6)
      double bound_push = 1e-2;     // IPOPT bound_push = bound_frac
      double acceptable_tol = 0.0;  // > 0: also stop at E₀ ≤ this ...
      double acceptable_dual = 1e10;//     ... with inf_du ≤ this, for acceptable_iter its
      int acceptable_iter = 15;
      bool ls_mult_init = true;     // least-squares λ₀ (IPOPT's default)
      bool line_search = true;      // filter line search (else: fraction to the boundary only)
      bool restoration = true;      // restoration phase when the line search fails
      int resto_max_iter = 200;     // iterations per restoration phase
      // Step for the row multipliers λ (IPOPT's alpha_for_y): Primal = α_pr,
      // BoundMult = α_du (the step of z), Full = 1, MinDualInfeas = the α in
      // [0, 1] that minimizes ‖∇L − z_L + z_U‖₂ at the new point and new z.
      enum class AlphaY { Primal, BoundMult, Full, MinDualInfeas };
      AlphaY alpha_y = AlphaY::Primal;
      double kappa_sigma = 1e10;    // after each step μ/(κ_Σ s) ≤ z ≤ κ_Σ μ/s (IPOPT: 1e10)
      double dual_reg = 0.0;        // > 0: δ_c = dual_reg·μ^dual_reg_exp in EVERY Newton
      double dual_reg_exp = 0.25;   //   system (stabilization for rank-deficient rows);
                                    //   0: δ_c only when a tile asks for it (IPOPT)
      bool delta_from_last = false; // after an iteration that needed δ > 0, start from
                                    // δ_last/3 instead of trying δ = 0 first (IPOPT tries 0;
                                    // here δ > 0 is needed in most iterations)
      int stall_iter = 0;           // > 0: restoration also when the barrier error E_μ
                                    // has not halved in this many iterations (θ ≤ θ_min,
                                    // μ > its floor)
      int print_level = 1;          // 0: quiet, 1: one line per iteration
      std::string label;            // prefix of the log lines (the restoration phase: "r")
   };
   struct Result {
      int status = -1;            // 0 converged, 1 iteration limit, 2 stopped by the
                                  // problem's callback, 3 acceptable, −1 failure
      int iters = 0;
      double obj = 0.0, inf_pr = 0.0, inf_du = 0.0, wall = 0.0;
      long tile_corrections = 0;  // factorizations refused for a wrong In(W_k)
      long tile_too_few = 0;      // ... of which: some tile had too FEW negative eigenvalues
      long s_corrections = 0;     // ... for S not positive definite
      long singular = 0;          // ... for a zero pivot
      long regularized = 0;       // iterations that needed δ > 0
      long ls_backtracks = 0;     // step halvings in the line search
      long ls_failures = 0;       // iterations where no step was acceptable
      long restorations = 0;      // restoration phases started (line search failed or stall) ...
      long stall_restorations = 0;// ... of which for a stall
      long resto_failures = 0;    // ... that found no acceptable point (step taken anyway)
      long resto_iters = 0;       // iterations spent in them (not in iters)
   };
   // A point in the IPM's own layout: p = [x | s], λ in row order [c | d],
   // bound multipliers for p.  Optional start and end of solve().
   struct Point {
      std::vector<double> p, lam, zl, zu;
   };

   // shared: a decomposition to reuse if its structure is this problem's KKT
   // pattern (the restoration phase passes the original problem's).
   Result solve(NLP& nlp, const dd::SchurDD::Options& dopt, dd::SchurDD::Stats* stats,
                const Point* start = nullptr, Point* end = nullptr,
                dd::SchurDD* shared = nullptr) {
      using Index = int;
      const auto t0 = std::chrono::steady_clock::now();
      Result res;
      const Options& o = opt;

      // ---- problem data, rows split into [c | d]
      const Index n = nlp.num_vars(), m = nlp.num_rows();
      const int n_tiles = nlp.n_tiles();
      std::vector<double> xl(n), xu(n), gl(m), gu(m);
      nlp.bounds(xl.data(), xu.data(), gl.data(), gu.data());
      std::vector<int> rowpos(m), drows;
      int mc = 0;
      for (Index r = 0; r < m; ++r)
         if (gl[r] == gu[r]) rowpos[r] = mc++;
         else drows.push_back(r);
      const int md = (int)drows.size();
      for (int i = 0; i < md; ++i) rowpos[drows[i]] = mc + i;
      const int np = n + md;          // primals: x and slacks
      const int N = np + m;           // the full layout [x | s | λ] of the IPM's vectors
      // elastic variables: the last n_el of x, eliminated from the KKT systems,
      // whose unknowns are [x' | s | λ] (x' = the first n0 variables)
      const int n_el = nlp.n_elastic(), n0 = n - n_el;
      const int npk = np - n_el;      // KKT primals
      const int NK = npk + m;         // KKT dimension

      std::vector<double> pl(np), pu(np);
      for (Index i = 0; i < n; ++i) { pl[i] = xl[i]; pu[i] = xu[i]; }
      for (int i = 0; i < md; ++i) { pl[n + i] = gl[drows[i]]; pu[n + i] = gu[drows[i]]; }
      std::vector<double> p(np, 0.0), lam(m, 0.0), zl(np, 0.0), zu(np, 0.0);
      std::vector<double> g(m);
      std::vector<char> hl(np), hu(np);
      for (int i = 0; i < np; ++i) {
         hl[i] = pl[i] > -1e19;
         hu[i] = pu[i] < 1e19;
      }
      if (start) {   // given (strictly interior) point and multipliers
         p = start->p;
         lam = start->lam;
         zl = start->zl;
         zu = start->zu;
      } else if (warm) {   // a point of this or a related problem, in the NLP's layout
         for (Index i = 0; i < n; ++i) {
            p[i] = warm->x[i];
            zl[i] = warm->zl[i];
            zu[i] = warm->zu[i];
         }
         for (int i = 0; i < md; ++i) {
            const int r = drows[i];
            p[n + i] = warm->s[r];
            zl[n + i] = warm->szl[r];
            zu[n + i] = warm->szu[r];
         }
         for (Index r = 0; r < m; ++r) lam[rowpos[r]] = warm->lam[r];
         const double mu_w = warm->mu > 0.0 ? warm->mu : o.mu0;
         for (int i = 0; i < np; ++i) {
            const double k = o.bound_push;
            if (hl[i] && !(p[i] > pl[i])) p[i] = pl[i] + k * std::max(1.0, std::abs(pl[i]));
            if (hu[i] && !(p[i] < pu[i])) p[i] = pu[i] - k * std::max(1.0, std::abs(pu[i]));
            zl[i] = hl[i] ? (zl[i] > 0.0 ? zl[i] : mu_w / (p[i] - pl[i])) : 0.0;
            zu[i] = hu[i] ? (zu[i] > 0.0 ? zu[i] : mu_w / (pu[i] - p[i])) : 0.0;
         }
      } else {
         nlp.start(p.data());
         nlp.g(p.data(), g.data());
         for (int i = 0; i < md; ++i) p[n + i] = g[drows[i]];
      }
      for (int i = 0; i < np && !start && !warm; ++i) {
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
      std::vector<Index> jr, jc, hr, hc;
      std::vector<double> lam_orig(m, 0.0);
      nlp.jac_structure(jr, jc);
      nlp.hess_structure(hr, hc);
      const Index nnz_j = (Index)jr.size(), nnz_h = (Index)hr.size();
      // Jacobian entries in the KKT (jtrip: KKT entry → triplet) and the
      // elastic ones (el_t: elastic e → its triplet, el_row: → its row)
      std::vector<int> jtrip, el_t(n_el, -1), el_row(n_el, -1);
      jtrip.reserve(nnz_j - n_el);
      bool elastic_ok = true;
      for (Index t = 0; t < nnz_j; ++t) {
         if (jc[t] < n0) { jtrip.push_back(t); continue; }
         const int e = jc[t] - n0;
         elastic_ok = elastic_ok && el_t[e] < 0;
         el_t[e] = t;
         el_row[e] = rowpos[jr[t]];
      }
      for (Index t = 0; t < nnz_h; ++t) elastic_ok = elastic_ok && hr[t] < n0 && hc[t] < n0;
      for (int e = 0; e < n_el; ++e) elastic_ok = elastic_ok && el_t[e] >= 0;
      if (dd::Comm::any(!elastic_ok)) {
         if (o.print_level > 0) std::printf("%s an elastic variable is not in exactly one row\n", o.label.c_str());
         return res;
      }
      const int nnz_jk = (int)jtrip.size();
      const int off_pd = nnz_h, off_j = off_pd + npk, off_s = off_j + nnz_jk, off_rd = off_s + md;
      const int nnz = off_rd + m;
      std::vector<int> ir(nnz), ic(nnz);
      for (Index t = 0; t < nnz_h; ++t) {
         ir[t] = std::max(hr[t], hc[t]);
         ic[t] = std::min(hr[t], hc[t]);
      }
      for (int i = 0; i < npk; ++i) ir[off_pd + i] = ic[off_pd + i] = i;
      for (int k = 0; k < nnz_jk; ++k) {
         ir[off_j + k] = npk + rowpos[jr[jtrip[k]]];
         ic[off_j + k] = jc[jtrip[k]];
      }
      for (int i = 0; i < md; ++i) {
         ir[off_s + i] = npk + mc + i;
         ic[off_s + i] = n0 + i;
      }
      for (Index r = 0; r < m; ++r) ir[off_rd + r] = ic[off_rd + r] = npk + r;

      // owner of each unknown, full layout [x | s | λ_c | λ_d] and KKT layout
      std::vector<int> owner(N), kowner(NK);
      for (Index i = 0; i < n; ++i) owner[i] = nlp.var_tile()[i];
      for (int i = 0; i < md; ++i) owner[n + i] = nlp.row_tile()[drows[i]];
      for (Index r = 0; r < m; ++r) owner[np + rowpos[r]] = nlp.row_tile()[r];
      for (int i = 0; i < n0; ++i) kowner[i] = owner[i];
      for (int i = n; i < N; ++i) kowner[i - n_el] = owner[i];

      // the decomposition: the shared one if it has this structure
      std::unique_ptr<dd::SchurDD> own;
      dd::SchurDD* ddp = shared;
      if (ddp && !ddp->same_structure(NK, ir, ic, kowner, npk, n_tiles)) {
         if (o.print_level > 0)
            std::printf("%s     (KKT pattern differs from the shared one: own decomposition)\n",
                        o.label.c_str());
         ddp = nullptr;
      }
      if (!ddp) {
         own = std::make_unique<dd::SchurDD>();
         own->set_stats_sink(stats);
         if (!own->set_structure(NK, ir, ic, kowner, npk, n_tiles, dopt)) return res;
         ddp = own.get();
      }
      dd::SchurDD& dd = *ddp;
      double* val = dd.values();

      // global counts: rows (never consensus) and bounds (consensus ones once)
      long nb_tile = 0, nb_cons = 0;
      for (int i = 0; i < np; ++i) (owner[i] < 0 ? nb_cons : nb_tile) += hl[i] + hu[i];
      const long m_all = dd::Comm::sum((long)m), nbounds = dd::Comm::sum(nb_tile) + nb_cons;
      dd::TileSum sum1(n_tiles), sum2(n_tiles, 2);
      // the consensus part of Aᵀy (y by row, in [c | d] order): out[i] for
      // border unknown i, summed over the tiles' linking entries
      dd::Vec at_border;
      auto add_at_border = [&](const std::vector<double>& jvals, auto y, std::vector<double>& out) {
         dd.border_product([&](int t) { return t >= off_j && t < off_s ? jvals[jtrip[t - off_j]] : 0.0; },
                           [&](int j) { return j >= npk ? y(j - npk) : 0.0; }, at_border);
         for (int i = 0; i < (int)at_border.size(); ++i) out[dd.border_index(i)] += at_border[i];
      };

      // ---- evaluation at the current point: f, A(p), Jacobian, ∇_p L
      double f = 0.0;
      std::vector<double> grad(n), A(m), jv(nnz_j), gL(np), hv(nnz_h);
      auto evaluate = [&](bool) {
         f = nlp.f(p.data());
         nlp.grad_f(p.data(), grad.data());
         nlp.g(p.data(), g.data());
         nlp.jac_values(p.data(), jv.data());
         for (Index r = 0; r < m; ++r)
            A[rowpos[r]] = gl[r] == gu[r] ? g[r] - gl[r] : g[r];
         for (int i = 0; i < md; ++i) A[mc + i] -= p[n + i];
         for (Index i = 0; i < n; ++i) gL[i] = grad[i];
         for (Index t = 0; t < nnz_j; ++t)
            if (owner[jc[t]] >= 0) gL[jc[t]] += jv[t] * lam[rowpos[jr[t]]];
         add_at_border(jv, [&](int j) { return lam[j]; }, gL);
         for (int i = 0; i < md; ++i) gL[n + i] = -lam[mc + i];
      };
      auto write_jacobian = [&]() {
         for (int k = 0; k < nnz_jk; ++k) val[off_j + k] = jv[jtrip[k]];
         for (int i = 0; i < md; ++i) val[off_s + i] = -1.0;
      };
      // the row diagonal v (−δ_c), minus c_e²/a_e for the elastic variables
      std::vector<double> el_a(n_el);   // a_e, set before each factorization
      auto write_rowdiag = [&](double v) {
         for (Index r = 0; r < m; ++r) val[off_rd + r] = v;
         for (int e = 0; e < n_el; ++e) {
            const double c = jv[el_t[e]];
            val[off_rd + el_row[e]] -= c * c / el_a[e];
         }
      };
      // Solve with the factorized KKT system for a right-hand side rf in the
      // full layout; df in the full layout (the elastic variables recovered).
      // rk, dk keep the condensed right-hand side and solution.
      std::vector<double> rk(NK), dk(NK);
      auto kkt_solve = [&](const std::vector<double>& rf, std::vector<double>& df) {
         for (int i = 0; i < n0; ++i) rk[i] = rf[i];
         for (int i = n; i < N; ++i) rk[i - n_el] = rf[i];
         for (int e = 0; e < n_el; ++e) rk[npk + el_row[e]] -= jv[el_t[e]] * rf[n0 + e] / el_a[e];
         dk = rk;
         const bool ok = dd.solve(dk.data());
         df.resize(N);
         for (int i = 0; i < n0; ++i) df[i] = dk[i];
         for (int i = n; i < N; ++i) df[i] = dk[i - n_el];
         for (int e = 0; e < n_el; ++e)
            df[n0 + e] = (rf[n0 + e] - jv[el_t[e]] * df[np + el_row[e]]) / el_a[e];
         return ok;
      };

      // ---- least-squares multipliers: [I Aᵀ; A 0][w; λ] = [−(∇_p f − z_L + z_U); 0],
      //      kept if no larger than 1e3 (else λ = 0)
      auto ls_multipliers = [&]() {
         for (Index t = 0; t < nnz_h; ++t) val[t] = 0.0;
         for (int i = 0; i < npk; ++i) val[off_pd + i] = 1.0;
         std::fill(el_a.begin(), el_a.end(), 1.0);
         write_jacobian();
         write_rowdiag(0.0);
         std::vector<double> b(N, 0.0);
         for (Index i = 0; i < n; ++i) b[i] = -(grad[i] - zl[i] + zu[i]);
         for (int i = n; i < np; ++i) b[i] = -(-zl[i] + zu[i]);
         std::vector<double> w(N, 0.0);
         bool ok = dd.factorize() && kkt_solve(b, w);
         double big = 0.0;
         for (Index r = 0; r < m; ++r) big = std::max(big, std::abs(w[np + r]));
         big = dd::Comm::max(big);
         for (Index r = 0; r < m; ++r) lam[r] = ok && big <= 1e3 ? w[np + r] : 0.0;
         evaluate(false);
      };
      evaluate(true);
      if (o.ls_mult_init && m > 0 && !start && !warm) ls_multipliers();

      std::vector<double> sig(np), rhs(N), d(N), delta(n_tiles, 0.0), last(n_tiles, 0.0);
      std::vector<double> hdiag(n, 0.0);   // the NLP's diagonal Hessian part, if any
      // Raghunathan–Biegler's modified complementarity block on these bounds
      std::vector<char> vwl(np, 1), vwu(np, 1);
      if (o.vw_mu > 0.0 && o.vw_comp_only) {
         std::fill(vwl.begin(), vwl.end(), 0);
         std::fill(vwu.begin(), vwu.end(), 0);
         nlp.complementarity_bounds(vwl.data(), vwu.data());
      }
      double mu_prev = 0.0, eta_vw = 0.0;  // μ before its last decrease (0: none yet); η (0: off)
      bool has_hdiag = false;
      double mu = warm && warm->mu > 0.0 ? warm->mu : o.mu0;
      double delta_c = 0.0, ap = 0.0, ad = 0.0, dmax = 0.0;
      int acc_count = 0;
      bool fail = false;
      int pr_block = -1, du_block = -1;   // which bound limited the last step
      bool just_restored = false;     // the last iteration ended with a restoration phase
      double stall_best = HUGE_VAL;   // lowest E_μ of the current barrier problem ...
      int stall_count = 0;            // ... and iterations since it last halved

      // ---- filter line search state (Wächter & Biegler 2006, IPOPT's constants)
      std::vector<std::array<double, 2>> filter;   // (θ, φ) pairs
      double theta_max = -1.0, theta_min = 0.0, mu_filter = -1.0;
      std::vector<double> gt(m), At(m), pt(np);
      auto theta_of = [&](const std::vector<double>& a) {   // a by row, [c | d]
         for (Index r = 0; r < m; ++r) sum1.add(owner[np + r], std::abs(a[r]));
         return sum1.total();
      };
      auto phi_of = [&](const std::vector<double>& pp, double fval) {   // barrier objective
         for (int i = 0; i < np; ++i) {
            if (hl[i]) sum1.add(owner[i], -mu * std::log(pp[i] - pl[i]));
            if (hu[i]) sum1.add(owner[i], -mu * std::log(pu[i] - pp[i]));
         }
         return fval + sum1.total();
      };
      auto residual_at = [&](const std::vector<double>& pp, std::vector<double>& out) {
         nlp.g(pp.data(), gt.data());
         for (Index r = 0; r < m; ++r) out[rowpos[r]] = gl[r] == gu[r] ? gt[r] - gl[r] : gt[r];
         for (int i = 0; i < md; ++i) out[mc + i] -= pp[n + i];
      };
      auto dominated = [&](double th, double ph) {
         for (const auto& e : filter)
            if (th >= e[0] && ph >= e[1]) return true;
         return false;
      };
      auto augment_filter = [&](double th, double ph) {   // drop what the new entry dominates
         const std::array<double, 2> e{(1.0 - 1e-5) * th, ph - 1e-8 * th};
         filter.erase(std::remove_if(filter.begin(), filter.end(),
                                     [&](const std::array<double, 2>& q) {
                                        return q[0] >= e[0] && q[1] >= e[1];
                                     }),
                      filter.end());
         filter.push_back(e);
      };

      // ---- restoration phase (Wächter & Biegler 2006, Sec. 3.3): from the
      //      current iterate (θ0, φ0), solve the ℓ₁ feasibility problem of
      //      restoration.hpp until it reaches a point with θ ≤ κ_resto·θ0 that
      //      the filter, augmented with the current iterate, accepts.  Then the
      //      bound multipliers come from the phase (reset to 1 if above 1e3)
      //      and λ from least squares.  true: p, z, λ replaced.
      auto restore = [&](double th0, double ph0) {
         ++res.restorations;
         augment_filter(th0, ph0);
         double a_inf = 0.0;
         for (double v : A) a_inf = std::max(a_inf, std::abs(v));
         a_inf = dd::Comm::max(a_inf);
         const double mu_r = std::max(mu, a_inf), rho = 1000.0, kappa_resto = 0.9;
         auto accept = [&](const double* q) {   // q = [x | p | n | s] of the phase
            for (Index i = 0; i < n; ++i) pt[i] = q[i];
            for (int i = 0; i < md; ++i) pt[n + i] = q[n + 2 * m + i];
            residual_at(pt, At);
            const double tht = theta_of(At);
            if (tht > kappa_resto * th0 || tht > theta_max) return false;
            const double pht = phi_of(pt, nlp.f(pt.data()));
            return std::isfinite(pht) && !dominated(tht, pht);
         };
         RestorationNLP rnlp(nlp, rho, std::sqrt(mu_r), std::vector<double>(p.begin(), p.begin() + n),
                             accept);
         // start: x and s as they are, p_r and n_r solving A_r − p_r + n_r = 0
         // on the central path for μ̄ (IPOPT's closed form), z = μ̄/(p, n) for
         // them and min(ρ, z) for the rest, λ = 0
         const int nr = n + 2 * m;
         Point s0;
         s0.p.assign(nr + md, 0.0);
         s0.zl.assign(nr + md, 0.0);
         s0.zu.assign(nr + md, 0.0);
         s0.lam.assign(m, 0.0);
         for (Index i = 0; i < n; ++i) {
            s0.p[i] = p[i];
            s0.zl[i] = std::min(rho, zl[i]);
            s0.zu[i] = std::min(rho, zu[i]);
         }
         for (int i = 0; i < md; ++i) {
            s0.p[nr + i] = p[n + i];
            s0.zl[nr + i] = std::min(rho, zl[n + i]);
            s0.zu[nr + i] = std::min(rho, zu[n + i]);
         }
         for (Index r = 0; r < m; ++r) {
            const double c = A[rowpos[r]], a = (mu_r - rho * c) / (2.0 * rho);
            const double nn = a + std::sqrt(a * a + mu_r * c / (2.0 * rho)), pp = c + nn;
            s0.p[n + r] = pp;
            s0.p[n + m + r] = nn;
            s0.zl[n + r] = mu_r / pp;
            s0.zl[n + m + r] = mu_r / nn;
         }
         IPM phase;
         phase.opt = o;
         phase.opt.mu0 = mu_r;
         phase.opt.mu_min = 0.0;   // the phase keeps IPOPT's floor, tol/10
         phase.opt.bound_push = 0.0;
         phase.opt.ls_mult_init = false;
         phase.opt.restoration = false;
         phase.opt.vw_mu = 0.0;   // IPOPT-C: no modified step in the restoration phase
         phase.opt.acceptable_tol = 0.0;
         phase.opt.max_iter = o.resto_max_iter;
         phase.opt.label = "r";
         const dd::SchurDD::Stats keep = *stats;   // (if the phase built its own decomposition)
         Point e;
         const Result rr = phase.solve(rnlp, dopt, stats, &s0, &e, &dd);   // reuses dd
         stats->n_tiles = keep.n_tiles;
         stats->dim = keep.dim;
         stats->p = keep.p;
         stats->max_pk = keep.max_pk;
         stats->max_tk = keep.max_tk;
         res.resto_iters += rr.iters;
         res.tile_corrections += rr.tile_corrections;
         res.tile_too_few += rr.tile_too_few;
         res.s_corrections += rr.s_corrections;
         res.singular += rr.singular;
         res.ls_backtracks += rr.ls_backtracks;
         if (!rnlp.accepted()) {
            ++res.resto_failures;
            if (o.print_level > 0)
               std::printf("%s     restoration failed after %d iterations (status %d)\n",
                           o.label.c_str(), rr.iters, rr.status);
            return false;
         }
         double zmax = 0.0;
         for (Index i = 0; i < n; ++i) {
            p[i] = e.p[i];
            zl[i] = e.zl[i];
            zu[i] = e.zu[i];
         }
         for (int i = 0; i < md; ++i) {
            p[n + i] = e.p[nr + i];
            zl[n + i] = e.zl[nr + i];
            zu[n + i] = e.zu[nr + i];
         }
         for (int i = 0; i < np; ++i) zmax = std::max({zmax, zl[i], zu[i]});
         if (dd::Comm::max(zmax) > 1e3)
            for (int i = 0; i < np; ++i) {
               zl[i] = hl[i] ? 1.0 : 0.0;
               zu[i] = hu[i] ? 1.0 : 0.0;
            }
         evaluate(false);
         ls_multipliers();
         residual_at(p, At);
         const double th1 = theta_of(At);   // collective: on every rank, printed or not
         if (o.print_level > 0)
            std::printf("%s     restoration: %d iterations, theta %.2e -> %.2e\n", o.label.c_str(),
                        rr.iters, th0, th1);
         return true;
      };

      if (o.print_level > 0 && o.label.empty())
         std::printf("iter    objective    inf_pr   inf_du lg(mu)  ||dp||  lg(rg) alpha_pr  "
                     "alpha_du  ls   pcg  lin.res\n");
      for (res.iters = 0;; ++res.iters) {
         // ---- optimality errors (IPOPT's scaled E_μ)
         auto errors = [&](double mu_, double& e_pr, double& e_du, double& e_co) {
            double e[3] = {0.0, 0.0, 0.0};
            for (double v : A) e[0] = std::max(e[0], std::abs(v));
            for (int i = 0; i < np; ++i) {
               e[1] = std::max(e[1], std::abs(gL[i] - zl[i] + zu[i]));
               if (hl[i]) e[2] = std::max(e[2], std::abs((p[i] - pl[i]) * zl[i] - mu_));
               if (hu[i]) e[2] = std::max(e[2], std::abs((pu[i] - p[i]) * zu[i] - mu_));
            }
            dd::Comm::max(e, 3);
            e_pr = e[0];
            e_du = e[1];
            e_co = e[2];
         };
         auto scaled = [&](double mu_) {
            for (Index r = 0; r < m; ++r) sum2.add(owner[np + r], std::abs(lam[r]), 0);
            for (int i = 0; i < np; ++i) sum2.add(owner[i], zl[i] + zu[i], 1);
            double yz[2];
            sum2.total(yz);
            const double y1 = yz[0], z1 = yz[1];
            const double s_d = std::max(100.0, (y1 + z1) / std::max(1L, m_all + nbounds)) / 100.0;
            const double s_c = std::max(100.0, z1 / std::max(1L, nbounds)) / 100.0;
            double e_pr, e_du, e_co;
            errors(mu_, e_pr, e_du, e_co);
            return std::max({e_du / s_d, e_pr, e_co / s_c});
         };
         double e_pr, e_du, e_co;
         errors(0.0, e_pr, e_du, e_co);
         res.obj = f;
         res.inf_pr = e_pr;
         res.inf_du = e_du;
         const double E0 = scaled(0.0);
         if (!std::isfinite(E0)) { fail = true; break; }
         const bool final_form = nlp.final_form();
         if (E0 <= o.tol && final_form) { res.status = 0; break; }
         if (o.acceptable_tol > 0.0 && E0 <= o.acceptable_tol && res.inf_du <= o.acceptable_dual &&
             final_form) {
            if (++acc_count >= o.acceptable_iter) { res.status = 3; break; }
         } else {
            acc_count = 0;
         }
         if (res.iters >= o.max_iter) { res.status = 1; break; }
         // the problem's own per-iteration hook (may change the problem: re-evaluate)
         const double f_before = f;
         const std::vector<double> A_before = A;
         std::vector<double> dres(np);
         for (int i = 0; i < np; ++i) dres[i] = gL[i] - zl[i] + zu[i];
         for (Index r = 0; r < m; ++r) lam_orig[r] = lam[rowpos[r]];
         const bool barrier_solved = scaled(mu) <= o.kappa_eps * mu;
         if (!nlp.iteration({res.iters, f, e_pr, res.inf_du, mu, dmax, p.data(), dres.data(),
                             lam_orig.data(), zl.data(), zu.data(), barrier_solved, pr_block,
                             du_block})) {
            res.status = 2;
            break;
         }
         bool moved = false;
         if (nlp.bounds_changed()) {   // re-read the bounds (the barrier problem changed)
            nlp.bounds(xl.data(), xu.data(), gl.data(), gu.data());
            for (Index i = 0; i < n; ++i) { pl[i] = xl[i]; pu[i] = xu[i]; }
            for (int i = 0; i < md; ++i) { pl[n + i] = gl[drows[i]]; pu[n + i] = gu[drows[i]]; }
            for (int i = 0; i < np; ++i) {
               const bool l = pl[i] > -1e19, u = pu[i] < 1e19;
               if (l && !(p[i] > pl[i])) p[i] = pl[i] + o.bound_push * std::max(1.0, std::abs(pl[i]));
               if (u && !(p[i] < pu[i])) p[i] = pu[i] - o.bound_push * std::max(1.0, std::abs(pu[i]));
               if (l && !hl[i]) zl[i] = mu / (p[i] - pl[i]);
               if (u && !hu[i]) zu[i] = mu / (pu[i] - p[i]);
               if (!l) zl[i] = 0.0;
               if (!u) zu[i] = 0.0;
               hl[i] = l;
               hu[i] = u;
            }
            moved = true;
         }
         evaluate(false);
         bool new_problem = moved || f != f_before || dd::Comm::any(A != A_before);   // the problem changed itself
         if (new_problem) filter.clear();
         // monotone μ update (capped right after a restoration phase)
         const double mu_min = o.mu_min > 0.0 ? o.mu_min : o.tol / 10.0;
         int max_steps = o.mu_max_steps;
         if (just_restored && o.resto_mu_steps > 0)
            max_steps = max_steps > 0 ? std::min(max_steps, o.resto_mu_steps) : o.resto_mu_steps;
         just_restored = false;
         for (int steps = 0; mu > mu_min && scaled(mu) <= o.kappa_eps * mu &&
                             (max_steps <= 0 || steps < max_steps); ++steps)
         {
            const double mu_old = mu;
            mu = std::max(mu_min, std::min(o.kappa_mu * mu, std::pow(mu, o.theta_mu)));
            mu_prev = mu_old;
         }
         if (mu != mu_filter) {   // a new barrier problem: a new filter
            filter.clear();
            mu_filter = mu;
            new_problem = true;
         }
         // ---- slow progress: the same barrier problem for stall_iter iterations
         //      without E_μ halving, while nearly feasible (θ ≤ θ_min, the filter's
         //      switching threshold) → restoration phase (if it fails: carry on)
         if (new_problem) {
            stall_best = HUGE_VAL;
            stall_count = 0;
         }
         if (o.line_search && o.restoration && o.stall_iter > 0 && mu > mu_min &&
             theta_max > 0.0 && theta_of(A) <= theta_min) {
            const double E = scaled(mu);
            if (E < 0.5 * stall_best) {
               stall_best = E;
               stall_count = 0;
            } else if (++stall_count >= o.stall_iter) {
               stall_best = HUGE_VAL;
               stall_count = 0;
               ++res.stall_restorations;
               if (o.print_level > 0)
                  std::printf("%s%4d  slow progress: restoration phase\n", o.label.c_str(),
                              res.iters);
               if (restore(theta_of(A), phi_of(p, f))) {
                  just_restored = true;
                  continue;
               }
            }
         }

         // ---- Newton system
         for (Index r = 0; r < m; ++r) lam_orig[r] = lam[rowpos[r]];
         nlp.hess_values(p.data(), 1.0, lam_orig.data(), hv.data());
         has_hdiag = nlp.hess_diag(p.data(), 1.0, hdiag.data());
         eta_vw = 0.0;
         if (o.vw_mu > 0.0 && mu <= o.vw_mu) {
            double scale = 0.0;   // max(‖c(x)‖∞, ‖z_L‖∞, ‖z_U‖∞), as in update_mpec_eta.f
            for (Index r = 0; r < m; ++r) scale = std::max(scale, std::abs(A[r]));
            for (int i = 0; i < np; ++i) scale = std::max({scale, zl[i], zu[i]});
            eta_vw = 0.1 * (mu_prev > 0.0 ? mu_prev : mu) / (1.0 + dd::Comm::max(scale));
         }
         for (int i = 0; i < np; ++i) {
            sig[i] = (hl[i] ? zl[i] / (p[i] - pl[i]) : 0.0) + (hu[i] ? zu[i] / (pu[i] - p[i]) : 0.0);
            rhs[i] = -gL[i] + (hl[i] ? mu / (p[i] - pl[i]) : 0.0) - (hu[i] ? mu / (pu[i] - p[i]) : 0.0);
            if (eta_vw > 0.0 && ((vwl[i] && hl[i]) || (vwu[i] && hu[i]))) {
               // z·(s + η z)-form: Σ = z/(s + ηz), rhs ± (z + (μ − s z)/(s + ηz))
               sig[i] = 0.0;
               rhs[i] = -gL[i];
               if (hl[i]) {
                  const double s = p[i] - pl[i], den = s + (vwl[i] ? eta_vw * zl[i] : 0.0);
                  sig[i] += zl[i] / den;
                  rhs[i] += zl[i] + (mu - s * zl[i]) / den;
               }
               if (hu[i]) {
                  const double s = pu[i] - p[i], den = s + (vwu[i] ? eta_vw * zu[i] : 0.0);
                  sig[i] += zu[i] / den;
                  rhs[i] -= zu[i] + (mu - s * zu[i]) / den;
               }
            }
         }
         for (Index r = 0; r < m; ++r) rhs[np + r] = -A[r];

         const long cg0 = stats->cg_iters;
         std::fill(delta.begin(), delta.end(), 0.0);
         if (o.delta_from_last && dmax > 0.0) raise_all(delta, last);   // dmax: the last iteration's δ
         if (o.dual_reg > 0.0) delta_c = o.dual_reg * std::pow(mu, o.dual_reg_exp);
         bool solved = false;
         for (int attempt = 0; attempt < 60 && !solved; ++attempt) {
            for (Index t = 0; t < nnz_h; ++t) val[t] = hv[t];
            auto diag = [&](int i) { return sig[i] + (owner[i] >= 0 ? delta[owner[i]] : 0.0); };
            for (int i = 0; i < n0; ++i)
               val[off_pd + i] = has_hdiag ? diag(i) + hdiag[i] : diag(i);
            for (int i = n; i < np; ++i) val[off_pd + i - n_el] = diag(i);
            for (int e = 0; e < n_el; ++e) el_a[e] = diag(n0 + e);
            write_jacobian();
            write_rowdiag(-delta_c);

            // With δ_c > 0, every wrong In(W_k) leads to the same decision
            // (raise δ), so the factorization may stop at the first such tile.
            // With δ_c = 0 the tiles with too FEW negatives matter: all are
            // factorized.
            if (!dd.factorize(delta_c > 0.0)) {
               ++res.singular;
               if (delta_c == 0.0) delta_c = 1e-8 * std::pow(mu, 0.25);
               else if (!raise_all(delta, last)) break;
               continue;
            }
            if (dd.inertia_stopped()) {   // eq. (24) violated (stopped at that tile)
               ++res.tile_corrections;
               if (!raise_all(delta, last)) break;
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
            if (too_few) ++res.tile_too_few;
            if (too_few && delta_c == 0.0) {
               ++res.tile_corrections;
               delta_c = 1e-8 * std::pow(mu, 0.25);
               continue;
            }
            if (!bad.empty()) {   // eq. (24) violated
               ++res.tile_corrections;
               if (!raise_all(delta, last)) break;
               continue;
            }
            if (dd.s_negative() > 0) {   // eq. (25) violated, seen exactly
               ++res.s_corrections;
               if (!raise_all(delta, last)) break;
               continue;
            }
            if (!kkt_solve(rhs, d)) {   // PCG met pᵀSp ≤ 0
               ++res.s_corrections;
               if (!raise_all(delta, last)) break;
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
         if (nreg > 0) ++res.regularized;

         // ---- accuracy of the step: ‖K d − rhs‖ / ‖rhs‖ against the true matrix
         //      (a border row's coupling entries come from all tiles), in the
         //      KKT layout: the condensed system that was solved
         std::vector<double> Kd(NK, 0.0);
         for (int t = 0; t < nnz; ++t) {
            const bool br = kowner[ir[t]] < 0, bc = kowner[ic[t]] < 0;
            if (!br || bc) Kd[ir[t]] += val[t] * dk[ic[t]];
            if (ir[t] != ic[t] && (!bc || br)) Kd[ic[t]] += val[t] * dk[ir[t]];
         }
         {
            dd::Vec kb;
            dd.border_product([&](int t) { return val[t]; }, [&](int j) { return dk[j]; }, kb);
            for (int i = 0; i < (int)kb.size(); ++i) Kd[dd.border_index(i)] += kb[i];
         }
         for (int i = 0; i < NK; ++i) {
            sum2.add(kowner[i], (Kd[i] - rk[i]) * (Kd[i] - rk[i]), 0);
            sum2.add(kowner[i], rk[i] * rk[i], 1);
         }
         double rb[2];
         sum2.total(rb);
         const double linres = std::sqrt(rb[0] / std::max(rb[1], 1e-300));

         // ---- bound multipliers and fraction to the boundary
         const double tau = std::max(0.99, 1.0 - mu);
         ap = 1.0;
         ad = 1.0;
         pr_block = du_block = -1;
         double dpn = 0.0;
         std::vector<double> dzl(np, 0.0), dzu(np, 0.0);
         for (int i = 0; i < np; ++i) {
            const double dp = d[i];
            dpn = std::max(dpn, std::abs(dp));
            if (hl[i]) {
               const double s = p[i] - pl[i];
               dzl[i] = eta_vw > 0.0 && vwl[i] ? (mu - s * zl[i] - zl[i] * dp) / (s + eta_vw * zl[i])
                                               : mu / s - zl[i] - zl[i] / s * dp;
               if (dp < 0.0 && -tau * s / dp < ap) { ap = -tau * s / dp; pr_block = i; }
               if (dzl[i] < 0.0 && -tau * zl[i] / dzl[i] < ad) { ad = -tau * zl[i] / dzl[i]; du_block = i; }
            }
            if (hu[i]) {
               const double s = pu[i] - p[i];
               dzu[i] = eta_vw > 0.0 && vwu[i] ? (mu - s * zu[i] + zu[i] * dp) / (s + eta_vw * zu[i])
                                               : mu / s - zu[i] + zu[i] / s * dp;
               if (dp > 0.0 && tau * s / dp < ap) { ap = tau * s / dp; pr_block = i; }
               if (dzu[i] < 0.0 && -tau * zu[i] / dzu[i] < ad) { ad = -tau * zu[i] / dzu[i]; du_block = i; }
            }
         }
         {
            double a[2] = {ap, ad};
            dd::Comm::min(a, 2);
            if (a[0] != ap) pr_block = -1;   // set on another rank
            if (a[1] != ad) du_block = -1;
            ap = a[0];
            ad = a[1];
            dpn = dd::Comm::max(dpn);
         }
         // ---- filter line search on (θ, φ_μ), backtracking from the
         //      fraction-to-the-boundary step.  If no step down to α_min is
         //      acceptable: the restoration phase, and if that fails as well,
         //      the full step with the filter cleared.
         int nback = 0;
         char ltype = ' ';
         bool restored = false;
         if (o.line_search) {
            const double th0 = theta_of(A), ph0 = phi_of(p, f);
            if (theta_max < 0.0) {
               theta_max = 1e4 * std::max(1.0, th0);
               theta_min = 1e-4 * std::max(1.0, th0);
            }
            // ∇φ_μ · dp
            for (Index i = 0; i < n; ++i) sum1.add(owner[i], grad[i] * d[i]);
            const bool vw_ls = eta_vw > 0.0 && o.vw_ls_grad;
            for (int i = 0; i < np; ++i) {
               if (hl[i]) {
                  const double s = p[i] - pl[i];
                  const double gb = vw_ls && vwl[i] ? zl[i] + (mu - s * zl[i]) / (s + eta_vw * zl[i])
                                                    : mu / s;
                  sum1.add(owner[i], -gb * d[i]);
               }
               if (hu[i]) {
                  const double s = pu[i] - p[i];
                  const double gb = vw_ls && vwu[i] ? zu[i] + (mu - s * zu[i]) / (s + eta_vw * zu[i])
                                                    : mu / s;
                  sum1.add(owner[i], gb * d[i]);
               }
            }
            const double gd = sum1.total();
            const double g_th = 1e-5, g_ph = 1e-8, delta_sw = 1.0, s_th = 1.1, s_ph = 2.3;
            const double eta = 1e-4;
            double amin = g_th;
            if (gd < 0.0) {
               amin = std::min(amin, g_ph * th0 / -gd);
               if (th0 <= theta_min)
                  amin = std::min(amin, delta_sw * std::pow(th0, s_th) / std::pow(-gd, s_ph));
            }
            amin *= 0.05;
            double alpha = ap;
            bool accepted = false, ftype = false;
            for (;;) {
               for (int i = 0; i < np; ++i) pt[i] = p[i] + alpha * d[i];
               const double ft = nlp.f(pt.data());
               residual_at(pt, At);
               const double tht = theta_of(At), pht = phi_of(pt, ft);
               if (std::isfinite(pht) && tht <= theta_max && !dominated(tht, pht)) {
                  const bool sw = gd < 0.0 && alpha * std::pow(-gd, s_ph) > delta_sw * std::pow(th0, s_th);
                  if (sw && th0 <= theta_min) {        // f-type: Armijo on φ
                     accepted = pht <= ph0 + eta * alpha * gd;
                     ftype = true;
                  } else {                              // h-type: θ or φ decreases enough
                     accepted = tht <= (1.0 - g_th) * th0 || pht <= ph0 - g_ph * th0;
                     ftype = false;
                  }
               }
               if (accepted) break;
               ++nback;
               alpha *= 0.5;
               if (alpha < amin || nback > 60) break;
            }
            if (accepted) {
               ap = alpha;
               ltype = ftype ? 'f' : 'h';
               if (!ftype) augment_filter(th0, ph0);
            } else {
               ++res.ls_failures;
               res.ls_backtracks += nback;
               nback = 0;
               if (o.print_level > 0)
                  std::printf("%s%4d  no acceptable step: restoration phase\n", o.label.c_str(),
                              res.iters);
               restored = o.restoration && restore(th0, ph0);
               if (!restored) {
                  filter.clear();
                  ltype = '!';
               }
            }
            res.ls_backtracks += nback;
         }
         if (restored) {
            just_restored = true;
            continue;
         }

         for (int i = 0; i < np; ++i) {
            p[i] += ap * d[i];
            zl[i] += ad * dzl[i];
            zu[i] += ad * dzu[i];
            // keep z within a factor κ_Σ of μ/s (IPOPT's safeguard)
            const double ks = o.kappa_sigma;
            if (hl[i]) {
               const double s = p[i] - pl[i];
               zl[i] = std::max(std::min(zl[i], ks * mu / s), mu / (ks * s));
            }
            if (hu[i]) {
               const double s = pu[i] - p[i];
               zu[i] = std::max(std::min(zu[i], ks * mu / s), mu / (ks * s));
            }
         }
         double ay = o.alpha_y == Options::AlphaY::BoundMult ? ad
                   : o.alpha_y == Options::AlphaY::Full    ? 1.0
                                                            : ap;
         if (o.alpha_y == Options::AlphaY::MinDualInfeas) {
            // at the new p and z: r(α) = r0 + α·v, r0 = ∇L(λ) − z_L + z_U, v = Aᵀdλ
            evaluate(true);
            std::vector<double> v(np, 0.0);
            for (Index t = 0; t < nnz_j; ++t)
               if (owner[jc[t]] >= 0) v[jc[t]] += jv[t] * d[np + rowpos[jr[t]]];
            add_at_border(jv, [&](int j) { return d[np + j]; }, v);
            for (int i = 0; i < md; ++i) v[n + i] = -d[np + mc + i];
            for (int i = 0; i < np; ++i) {
               sum2.add(owner[i], (gL[i] - zl[i] + zu[i]) * v[i], 0);
               sum2.add(owner[i], v[i] * v[i], 1);
            }
            double rvv[2];
            sum2.total(rvv);
            const double rv = rvv[0], vv = rvv[1];
            ay = vv > 0.0 ? std::min(1.0, std::max(0.0, -rv / vv)) : ap;
         }
         for (Index r = 0; r < m; ++r) lam[r] += ay * d[np + r];

         if (o.print_level > 0)
            std::printf("%s%4d %14.7e %8.2e %8.2e %6.1f %7.2e %6s %8.2e%c %8.2e %3d %5ld %8.1e%s\n",
                        o.label.c_str(), res.iters, f, e_pr, e_du, std::log10(mu), dpn,
                        dmax > 0 ? fmt_lg(dmax).c_str() : "-", ap, ltype, ad, nback,
                        stats->cg_iters - cg0, linres, nlp.log_extra().c_str());
         evaluate(true);
      }
      if (fail) res.status = -1;
      if (end) *end = Point{p, lam, zl, zu};
      if (export_point) {
         NLPPoint& e = *export_point;
         e.x.assign(p.begin(), p.begin() + n);
         e.zl.assign(zl.begin(), zl.begin() + n);
         e.zu.assign(zu.begin(), zu.begin() + n);
         e.s.assign(m, 0.0);
         e.szl.assign(m, 0.0);
         e.szu.assign(m, 0.0);
         e.lam.assign(m, 0.0);
         for (int i = 0; i < md; ++i) {
            e.s[drows[i]] = p[n + i];
            e.szl[drows[i]] = zl[n + i];
            e.szu[drows[i]] = zu[n + i];
         }
         for (Index r = 0; r < m; ++r) e.lam[r] = lam[rowpos[r]];
         e.mu = mu;
      }
      for (Index r = 0; r < m; ++r) lam_orig[r] = lam[rowpos[r]];
      nlp.finalize(res.status, p.data(), lam_orig.data(), res.obj);
      res.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return res;
   }

   Options opt;
   // Optional, in the NLP's layout: a warm start (x, slacks, all multipliers
   // and μ; bound multipliers of bounds that no longer exist are dropped, the
   // others kept positive, and the point pushed inside the bounds if needed),
   // and where to put the final point.
   const NLPPoint* warm = nullptr;
   NLPPoint* export_point = nullptr;

private:
   static std::string fmt_lg(double v) {
      char b[16];
      std::snprintf(b, sizeof b, "%.1f", std::log10(v));
      return b;
   }
   // IPOPT's δ_w schedule: first try 1e-4 (or a third of the last value that
   // worked), then ×100 the first time, ×8 after.
   static bool raise(double& delta, double last) {
      if (delta == 0.0) delta = last == 0.0 ? 1e-4 : std::max(1e-20, last / 3.0);
      else delta *= last == 0.0 ? 100.0 : 8.0;
      return delta < 1e40;
   }
   // One δ shared by all tiles (a per-tile δ_k, the paper's Sec. 3.5 option,
   // failed on 4 of 6 MPCC instances in our tests).
   static bool raise_all(std::vector<double>& delta, const std::vector<double>& last) {
      double d = delta.empty() ? 0.0 : delta[0];
      double l = 0.0;
      for (double v : last) l = std::max(l, v);
      if (!raise(d, l)) return false;
      std::fill(delta.begin(), delta.end(), d);
      return true;
   }
};
