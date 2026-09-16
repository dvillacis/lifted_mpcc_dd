// Shared Ipopt::TNLP boilerplate for the three lifted TV-MPCC formulations
// (uniform 2D, staggered 1D, staggered 2D).
//
// Every formulation in this repo has the same shape: a state block u, the lifted
// dual/polar blocks (qx, qy, r, δ, θ) of one common length, and the scalar weight
// parameter α — with
//
//   objective   ½‖u − u_clean‖² + ½·reg_α·α² [+ ½·ε_θ·‖θ − θ_ref‖²]
//   bounds      δ ≤ 1 (the unit ball), the α box (exp weight) or α ≤ w_max
//               (linear weight, whose α ≥ 0 is the explicit row ha)
//   rows        n_eq equalities, then [rhr, rcomp) as g ≥ 0, then comp ≤ 0 LAST
//
// so the objective callbacks, the bounds, the warm-start bookkeeping and the
// Q(α) machinery are identical and live here. What stays in each derived class is
// exactly the validated formulation core: the difference operators, the Jacobian /
// Hessian STRUCTURE arrays, and eval_g / eval_jac_g / eval_h — whose value arrays
// are matched to the structure POSITIONALLY, not by key, and are deliberately not
// touched by this refactor.
#ifndef MPCC_BASE_HPP
#define MPCC_BASE_HPP

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "IpTNLP.hpp"
// For reading the CURRENT iterate inside intermediate_callback (the documented
// IPOPT recipe: ip_cq → OrigIpoptNLP → TNLPAdapter → ResortX). Internal
// headers, same dependency class as dd_solver.hpp's AlgorithmBuilder use.
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpTNLPAdapter.hpp"

namespace Ipopt {

class MpccTNLPBase : public TNLP {
public:
   // ---- layout, filled by the derived constructor -------------------------
   int n = 0, mcon = 0;                   // variables / constraint rows
   int n_eq = 0, n_ineq = 0, kkt_dim = 0; // kkt_dim = n + 2·n_ineq + n_eq
   int n_state = 0;                       // length of u (= length of h1)
   int n_lift = 0;                        // length of qx/qy/r/δ/θ (edges or cells)
   int ou = 0, oqx = 0, oqy = 0, oR = 0, oD = 0, oTh = 0, oa = 0;   // column offsets
   int rhr = 0, rcomp = 0;                // row offsets the shared bounds need
   bool has_ha = false;                   // linear weight: explicit row α ≥ 0
   bool weight_exp = false;               // Q(α) = e^α (else Q(α) = α)

   std::vector<double> f_, uclean_, x_start_;
   std::vector<int> jr_, jc_;             // Jacobian structure
   std::vector<int> hr_, hc_;             // Hessian structure (lower triangle)

   double alpha_lo_ = -15.0, alpha_hi_ = 15.0;   // α box (exp weight only)
   double w_max_ = 2e19;                         // optional cap on the linear weight
   double reg_alpha_ = 1e-4;
   // Scale on the DATA term (and, with it, the θ ridge below). 1.0 for the
   // single-instance drivers — they must stay bit-identical, and the
   // --self-check checksums are the gate on that. The dataset driver sets 1/S
   // so the upper level is the MEAN over training pairs rather than the sum:
   // then the data term stays O(1) as the training set grows and --reg-alpha
   // keeps the meaning it has in the single-image tables. The ridge is scaled
   // ALONGSIDE the data term, not independently — what matters is their ratio,
   // and holding it fixed is what makes S=1 reproduce dd_solve_2d exactly.
   // reg_alpha_ is deliberately NOT scaled: α is one scalar for the whole
   // dataset, so its penalty is not a per-sample quantity.
   double loss_scale_ = 1.0;
   double t_ = 1.0;                              // current Scholtes level
   // TR (Tikhonov) gauge ridge ½·ε_θ·‖θ − θ_ref‖², the D1 fix for the angle gauge:
   // where r = δ = 0 the angle θ is undetermined, the (θ,θ) Hessian entry vanishes
   // and the KKT picks up a null direction. This matters far more for the DD
   // solver than for a monolithic one — a null θ column sits INSIDE one subdomain,
   // making that W_k singular on its own even when the full KKT is not. The driver
   // sets eps_theta_ = c_theta·t per level, so the bias → 0 along the continuation.
   double eps_theta_ = 0.0;
   std::vector<double> theta_ref_;

   // μ-coupled Scholtes update (--t-update mu; port of the validated
   // ../lifted_mpcc_unitball_v2.py mechanism, 2026-07-23): with t_mu_scale_ > 0
   // the intermediate callback slaves the relaxation level to the barrier,
   // t = max(t_floor_, t_mu_scale_·μ), TIGHTENING ONLY — the monotone guard
   // keeps restoration-phase μ (the restoration problem's own barrier) and any
   // oscillation from loosening the comp row mid-solve. Legal because t enters
   // ONLY eval_g's comp rows additively — the Jacobian/Hessian are t-free;
   // IPOPT is not told the NLP changed, and the one-iteration lag of its cached
   // constraint values is a mild one-sided perturbation the next evaluation
   // absorbs (measured in Python: equal solutions, 2–4× fewer iterations,
   // deeper stable tails — t ∝ μ keeps ξ ~ μ/slack controlled through the
   // corner, the Raghunathan–Biegler coupling applied externally). REQUIRES
   // monotone μ (driver_common::init_app sets it).
   double t_mu_scale_ = 0.0;      // 0 = off (the geometric drivers)
   double t_floor_ = 0.0;
   double c_theta_live_ = 0.0;    // keeps eps_theta_ = c_θ·t in sync
   // RATE LIMIT on the t-ratchet: the smallest ratio t⁺/t allowed in one
   // iteration (0 = no limit, the historical behaviour).  t is slaved to μ, so
   // a single monotone μ cut drags t down by the SAME factor at once — and
   // IPOPT's first cut is large: measured at N=256 consensus, μ goes 0.1 →
   // 6.98e-4 before the first Newton step, tightening complementarity 143x in
   // one step.  The run enters restoration at it=2 with inf_du = 1.0e+03 and
   // never recovers; by it=20 it is still 1.8e+02 against a μ gate of 0.698,
   // so μ cannot advance and the solve stalls.  Limiting how fast t may fall
   // decouples the complementarity schedule from μ's jumps without changing
   // where t ends up: t still converges to max(t_floor, c·μ) once μ settles.
   double t_rate_ = 0.0;
   int mu_progress_every_ = 0;    // live rows under print_level 0 (0 = silent)
   int mu_stall_iters_ = 0;
   bool mu_stall_warned_ = false;
   // The in-solve continuation trace, (iter, μ, t, weight, max r(1−δ)) flat
   // 5-tuples appended at every callback — the μ-coupled run's analogue of the
   // per-level history (there are no outer levels to tabulate). weight/comp
   // come from the CURRENT accepted iterate via the documented TNLPAdapter
   // recipe; during restoration phases (where the NLP is the restoration
   // problem's) they are recorded as NaN and plot as honest gaps. Dumped by
   // --save-solution and drawn by plot_2d.py as the continuation-path panel.
   std::vector<double> mu_hist_;
   std::vector<double> cb_x_;     // ResortX scratch (n primal entries)

   // Banked deepest-converged-level fallback (2026-07-24). The μ-coupled run is
   // ONE IPOPT solve, so on a tail failure finalize_solution hands back only the
   // failed final iterate and every converged intermediate level is lost. In
   // monotone mode a μ DECREASE means the μ_prev barrier subproblem met its
   // tolerance — i.e. a converged Scholtes level — and the iterate that
   // satisfied it is the one recorded at the callback BEFORE the drop. So keep
   // a one-callback lag (prev_*) and, at each regular-mode μ drop, bank the
   // lagged point, keeping the smallest max r(1−δ) — the same tightest-
   // converged-level rule run_scholtes reports by. Restoration callbacks are
   // excluded twice over (mode check + comp_now = NaN there), so restoration's
   // own μ never fakes a level. run_mu_coupled resets this state and falls back
   // to banked_x_ when IPOPT's final status is a failure.
   bool have_banked_ = false;
   double banked_comp_ = 1e300, banked_t_ = 0.0;
   std::vector<double> banked_x_;
   double mu_prev_ = 1e300;
   bool prev_valid_ = false;
   double prev_comp_ = 1e300, prev_t_ = 0.0;
   std::vector<double> prev_x_;

   // THE SCHOLTES LEVEL GATE (docs/inf_du_stall).  How accurately is one
   // relaxed level worth solving?
   //
   // A relaxed NLP at parameter t approximates the MPCC only to within the
   // relaxation's own error, so a termination gate that shrinks faster than
   // that error spends iterations on the relaxation rather than on the
   // problem.  The scale the geometry hands back is sqrt(t): on the strongly
   // active set the relaxed row sits at r(1−δ) = t, where its own gradient is
   //
   //     ‖∇_(r,δ) [ r(1−δ) ]‖ = ‖(1−δ, −r)‖ ≥ sqrt(2t),
   //
   // with equality at the knee r = 1−δ = sqrt(t) — and measured at N=64,
   // t=1e-4, the active cells sit ON that floor (min 1.408e-2, 5th pct
   // 1.436e-2, against sqrt(2t) = 1.41e-2), a factor ~70 below the O(1) rows
   // of the state system.  The multiplier block those rows carry is
   // conditioned like 1/sqrt(t), which is where the forty-iteration dual
   // plateau lives.  The gate is therefore
   //
   //     inf_pr ≤ κ·sqrt(t),  inf_du ≤ κ·sqrt(t),  μ ≤ κ·sqrt(t)
   //
   // tested only once t has descended to level_gate_t_ (the level being
   // reported: t_min for the μ-coupled single solve, this level's t for the
   // geometric continuation), only in RegularMode, and satisfied for
   // level_gate_hits_ consecutive callbacks.  Firing returns false, which
   // IPOPT turns into User_Requested_Stop — finalize_solution still runs, so
   // the iterate is kept.
   //
   // WHY HERE AND NOT IN IPOPT'S OPTIONS.  The obvious implementation is
   // acceptable_tol = κ·sqrt(t), and it is wrong: acceptable_tol also governs
   // the RESTORATION phase's convergence test, so loosening it lets
   // restoration "converge" at a point that is still infeasible for the
   // original problem and IPOPT reports that as a local-infeasibility
   // certificate.  Measured, N=64 4×4 consensus, monolithic mumps:
   //
   //     acceptable_tol = t     = 1e-4   status 1, 141 it, t=1e-4,    α*=0.067484
   //     acceptable_tol = √t    = 1e-2   status 2, 92 it, EXIT "Converged to a
   //                                     point of local infeasibility", falling
   //                                     back to the banked t=2.39e-4, α*=0.067261
   //
   // The two runs are bit-identical for their first 94 callbacks — the option
   // changes nothing about the path, only the verdict — and the last row of
   // the second is `92r`, i.e. a restoration iteration.  Raising `tol` instead
   // is worse still: it additionally sets the barrier floor
   // μ_min = tol/(barrier_tol_factor+1).  A gate in the callback touches
   // neither, so IPOPT's own tolerances stay at their tested values and the
   // filter, the restoration phase and the μ schedule are untouched.
   double level_tol_scale_ = 0.0;   // κ; 0 disables the gate entirely
   double level_gate_floor_ = 0.0;  // --tol, as a floor on how tight κ√t may get
   double level_gate_t_ = 0.0;      // only test once t_ has reached this level
   int level_gate_hits_ = 1;        // consecutive qualifying callbacks required
   int level_gate_seen_ = 0;
   bool level_gate_fired_ = false;

   bool intermediate_callback(AlgorithmMode mode, Index iter, Number obj_value,
                              Number inf_pr, Number inf_du, Number mu, Number,
                              Number, Number, Number, Index,
                              const IpoptData* ip_data,
                              IpoptCalculatedQuantities* ip_cq) override {
      // The Scholtes level gate (see the member comment).  Ahead of the
      // μ-coupled bookkeeping below, because it serves both continuations.
      if (level_tol_scale_ > 0.0 && mode == RegularMode && iter > 0 &&
          t_ <= level_gate_t_ * (1.0 + 1e-9)) {
         const double gate =
            std::max(level_gate_floor_, level_tol_scale_ * std::sqrt(t_));
         if (inf_pr <= gate && inf_du <= gate && mu <= gate) {
            if (++level_gate_seen_ >= level_gate_hits_) {
               level_gate_fired_ = true;
               if (mu_progress_every_ > 0)
                  std::printf("    [level gate] t=%.2e  inf_pr=%.1e  inf_du=%.1e "
                              " mu=%.1e  all <= %.1e = %.3g*sqrt(t) — stopping\n",
                              t_, inf_pr, inf_du, mu, gate, level_tol_scale_);
               return false;            // → Ipopt::User_Requested_Stop
            }
         } else {
            level_gate_seen_ = 0;
         }
      }
      if (t_mu_scale_ <= 0.0) return true;
      double t_new = std::max(t_floor_, t_mu_scale_ * mu);
      if (t_rate_ > 0.0 && t_new < t_ * t_rate_)      // no faster than this
         t_new = std::max(t_floor_, t_ * t_rate_);
      const bool tightened = t_new < t_;
      if (tightened) {
         t_ = t_new;
         if (c_theta_live_ > 0.0) eps_theta_ = c_theta_live_ * t_new;
      }
      // weight and max r(1−δ) at the CURRENT accepted iterate (the IPOPT
      // recipe: ip_cq → OrigIpoptNLP → TNLPAdapter → ResortX). The casts fail
      // during restoration (the NLP there is RestoIpoptNLP with extra
      // variables) — those iterations record NaN and plot as gaps.
      double w_now = std::numeric_limits<double>::quiet_NaN();
      double comp_now = std::numeric_limits<double>::quiet_NaN();
      if (ip_cq && ip_data) {
         OrigIpoptNLP* onlp =
            dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
         TNLPAdapter* ad =
            onlp ? dynamic_cast<TNLPAdapter*>(GetRawPtr(onlp->nlp())) : nullptr;
         if (ad && IsValid(ip_data->curr()) && IsValid(ip_data->curr()->x())) {
            if ((int)cb_x_.size() != n) cb_x_.assign(n, 0.0);
            ad->ResortX(*ip_data->curr()->x(), cb_x_.data());
            w_now = Q(cb_x_[oa]);
            double c = 0.0;
            for (int e = 0; e < n_lift; ++e)
               c = std::max(c, cb_x_[oR + e] * (1.0 - cb_x_[oD + e]));
            comp_now = c;
         }
      }
      // Bank the lagged iterate at each regular-mode μ drop (see the member
      // comment). std::isfinite(comp_now) is false during restoration (the
      // dynamic_casts above fail → NaN), so both prev_* and mu_prev_ freeze
      // across restoration phases and a restoration μ can neither bank nor
      // register as a drop.
      if (mode == RegularMode && std::isfinite(comp_now)) {
         if (prev_valid_ && mu < mu_prev_ * (1.0 - 1e-12) &&
             prev_comp_ < banked_comp_) {
            banked_x_ = prev_x_;
            banked_comp_ = prev_comp_;
            banked_t_ = prev_t_;
            have_banked_ = true;
         }
         prev_x_ = cb_x_;
         prev_comp_ = comp_now;
         prev_t_ = t_;
         prev_valid_ = true;
         mu_prev_ = mu;
      }
      mu_hist_.push_back((double)iter);
      mu_hist_.push_back(mu);
      mu_hist_.push_back(t_);
      mu_hist_.push_back(w_now);
      mu_hist_.push_back(comp_now);
      if (mu_progress_every_ > 0) {
         mu_stall_iters_ = tightened ? 0 : mu_stall_iters_ + 1;
         if (iter > 0 && iter % mu_progress_every_ == 0) {
            std::printf("    [mu-coupled] it=%4d  mu=%.2e  t=%.2e  inf_pr=%.1e  "
                        "inf_du=%.1e  obj=%.4e\n",
                        (int)iter, mu, t_, inf_pr, inf_du, obj_value);
            std::fflush(stdout);
         }
         // A run whose t never tightens is stuck at the incoming relaxation
         // with μ frozen — the near-singular-first-level failure documented for
         // the Python port; flag it so the long single solve is not mistaken
         // for a hang.
         if (!mu_stall_warned_ && mu_stall_iters_ >= 100 &&
             t_ > t_floor_ * (1 + 1e-9)) {
            mu_stall_warned_ = true;
            std::printf("    [mu-coupled] warning: t pinned at %.2e for %d "
                        "iters (mu not decreasing, inf_du=%.1e) — the barrier "
                        "is stuck so t cannot tighten (near-singular level).\n",
                        t_, mu_stall_iters_, inf_du);
            std::fflush(stdout);
         }
      }
      return true;
   }

   // warm start between continuation levels + last solution
   bool have_warm_ = false;
   std::vector<double> wx_, wzL_, wzU_, wlam_;
   std::vector<double> sol_x_, sol_lam_, sol_zL_, sol_zU_;
   double sol_obj_ = 0.0;

   // ---- the weight Q(α) ---------------------------------------------------
   double Q(double a) const { return weight_exp ? std::exp(a) : a; }
   double dQ(double a) const { return weight_exp ? std::exp(a) : 1.0; }
   double d2Q(double a) const { return weight_exp ? std::exp(a) : 0.0; }
   double weight_of_alpha(double a) const { return Q(a); }

   // The gauge ridge is measured against the STARTING angles; record them once the
   // initial point is known.
   void set_theta_ref(const double* x) {
      theta_ref_.assign(x + oTh, x + oTh + n_lift);
   }

   // ---- shared TNLP pieces ------------------------------------------------
   bool get_nlp_info(Index& n_, Index& m_, Index& nnz_jac, Index& nnz_h,
                     IndexStyleEnum& style) override {
      n_ = n; m_ = mcon; nnz_jac = (Index)jr_.size(); nnz_h = (Index)hr_.size();
      style = C_STYLE;
      return true;
   }

   // Only δ ≤ 1 (and, under the exp weight, the α box) is a variable bound:
   // r ≥ 0, δ ≥ 0 (and α ≥ 0 for the linear weight) are explicit ROWS, so IPOPT
   // slacks them instead of enforcing them by barrier.
   bool get_bounds_info(Index, Number* xl, Number* xu, Index, Number* gl,
                        Number* gu) override {
      for (int i = 0; i < n; ++i) { xl[i] = -2e19; xu[i] = 2e19; }
      for (int e = 0; e < n_lift; ++e) xu[oD + e] = 1.0;   // δ ≤ 1 (the unit ball)
      if (has_ha) { xu[oa] = w_max_; }
      else { xl[oa] = alpha_lo_; xu[oa] = alpha_hi_; }
      for (int i = 0; i < n_eq; ++i) { gl[i] = 0.0; gu[i] = 0.0; }
      for (int i = rhr; i < rcomp; ++i) { gl[i] = 0.0; gu[i] = 2e19; }
      for (int i = rcomp; i < mcon; ++i) { gl[i] = -2e19; gu[i] = 0.0; }
      return true;
   }

   bool get_starting_point(Index, bool init_x, Number* x, bool init_z, Number* zL,
                           Number* zU, Index, bool init_lam, Number* lam) override {
      if (init_x) {
         const std::vector<double>& src = have_warm_ ? wx_ : x_start_;
         for (int i = 0; i < n; ++i) x[i] = src[i];
      }
      // Defensive: unreachable through run_scholtes (warm_start_init_point is
      // only enabled once warm data exists), but never hand IPOPT
      // uninitialized memory if it ever asks for duals we don't have.
      if (init_z)
         for (int i = 0; i < n; ++i) {
            zL[i] = have_warm_ ? wzL_[i] : 0.0;
            zU[i] = have_warm_ ? wzU_[i] : 0.0;
         }
      if (init_lam)
         for (int i = 0; i < mcon; ++i) lam[i] = have_warm_ ? wlam_[i] : 0.0;
      return true;
   }

   bool eval_f(Index, const Number* x, bool, Number& obj) override {
      double s = 0.0;
      for (int i = 0; i < n_state; ++i) {
         const double d = x[ou + i] - uclean_[i];
         s += d * d;
      }
      obj = 0.5 * loss_scale_ * s + 0.5 * reg_alpha_ * x[oa] * x[oa];
      if (eps_theta_ != 0.0) {
         double g = 0.0;
         for (int e = 0; e < n_lift; ++e) {
            const double d = x[oTh + e] - theta_ref_[e];
            g += d * d;
         }
         obj += 0.5 * loss_scale_ * eps_theta_ * g;
      }
      return true;
   }

   bool eval_grad_f(Index, const Number* x, bool, Number* g) override {
      for (int i = 0; i < n; ++i) g[i] = 0.0;
      for (int i = 0; i < n_state; ++i)
         g[ou + i] = loss_scale_ * (x[ou + i] - uclean_[i]);
      g[oa] = reg_alpha_ * x[oa];
      if (eps_theta_ != 0.0)
         for (int e = 0; e < n_lift; ++e)
            g[oTh + e] = loss_scale_ * eps_theta_ * (x[oTh + e] - theta_ref_[e]);
      return true;
   }

   void finalize_solution(SolverReturn, Index, const Number* x, const Number* zL,
                          const Number* zU, Index, const Number*, const Number* lam,
                          Number obj, const IpoptData*,
                          IpoptCalculatedQuantities*) override {
      sol_x_.assign(x, x + n);
      sol_zL_.assign(zL, zL + n);
      sol_zU_.assign(zU, zU + n);
      sol_lam_.assign(lam, lam + mcon);
      sol_obj_ = obj;
      wx_ = sol_x_; wzL_ = sol_zL_; wzU_ = sol_zU_; wlam_ = sol_lam_;
      have_warm_ = true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_BASE_HPP
