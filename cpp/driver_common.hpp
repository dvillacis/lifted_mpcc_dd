// Shared plumbing for the three Scholtes-continuation drivers (dd_solve,
// dd_solve_1d, dd_solve_2d): PSNR, the t-schedule, the common IPOPT options,
// the continuation loop itself, and the end-of-run summary. Each driver keeps
// only what is formulation-specific — CLI flags, the partition/owner map, its
// self-check and its save formats.
#ifndef DRIVER_COMMON_HPP
#define DRIVER_COMMON_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "dd_solver.hpp"
#include "mpcc_base.hpp"

namespace driver {

inline double psnr(const std::vector<double> &clean, const double *u, int m) {
  double mse = 0;
  for (int i = 0; i < m; ++i) {
    double d = clean[i] - u[i];
    mse += d * d;
  }
  mse /= m;
  return mse == 0 ? 1e9 : 10.0 * std::log10(1.0 / mse);
}

// Reject schedule parameters the loop below cannot terminate on: factor >= 1
// never decreases t (the vector grows until OOM), and tmin <= 0 lets t
// underflow to exactly 0.0, after which 0 >= 0 pushes 0.0 forever. Drivers
// call this right after parsing so the error is loud; the cap inside
// scholtes_schedule is the backstop for any caller that forgets.
inline bool valid_schedule(double t0, double tmin, double factor) {
  if (!(tmin > 0.0) || !(t0 >= tmin) || !(factor > 0.0) || !(factor < 1.0)) {
    std::cerr << "invalid continuation schedule: need t-min > 0, t0 >= t-min, "
                 "0 < factor < 1 (got t0=" << t0 << " t-min=" << tmin
              << " factor=" << factor << ")\n";
    return false;
  }
  return true;
}

inline std::vector<double> scholtes_schedule(double t0, double tmin,
                                             double factor) {
  // Backstop cap: with valid parameters the deepest real schedule is tens of
  // levels (t0=1 → 1e-6 at factor 0.85 is ~92); 10000 is unreachable except
  // through the non-terminating parameter cases valid_schedule rejects.
  static constexpr int MAX_LEVELS = 10000;
  std::vector<double> sched;
  for (double t = t0; t >= tmin * (1 - 1e-12) && (int)sched.size() < MAX_LEVELS;
       t *= factor)
    sched.push_back(t);
  return sched;
}

// Common IPOPT options; `solver` selects IPOPT's own linear solver
// (mumps|ma57|ma97 — anything else, i.e. "dd", leaves the choice to the custom
// AlgorithmBuilder). ma97 needs an IPOPT built with MA97 support (e.g. the HPC
// conda env, whose libipopt.so links libhsl_ma97.so directly); ma57 loads the
// library at runtime via hsllib. Returns false if IPOPT initialization fails.
inline bool init_app(Ipopt::SmartPtr<Ipopt::IpoptApplication> &app,
                     int print_level, int max_iter, const std::string &solver,
                     const std::string &hessian = "limited-memory") {
  app->Options()->SetStringValue("sb", "yes");
  app->Options()->SetIntegerValue("print_level", print_level);
  app->Options()->SetIntegerValue("max_iter", max_iter);
  app->Options()->SetIntegerValue("acceptable_iter", 1);
  app->Options()->SetStringValue("mu_strategy", "monotone");
  // HESSIAN (--hessian exact|limited-memory, default limited-memory).
  //
  // All three TNLPs implement a full analytic eval_h, so "exact" is available
  // everywhere — but which one WINS is size-dependent, so this is deliberately
  // NOT flipped by default. Profiling (2026-07-25) put IPOPT's
  // LowRankAugSystemSolver::UpdateFactorization at 44% of an N=32 dd run: under
  // limited-memory every Newton step re-solves the augmented system with the
  // L-BFGS correction vectors as extra RHS. Exact removes that entirely and
  // roughly halves the iteration count — but its Hessian block is denser, so
  // each remaining iteration costs more. Measured (dd, wall clock):
  //     1D n256/n512/n1024   122→76, 177→92, 165→90 iterations   exact wins
  //     2D N=32  6×6 tiles   147→67 its,  9.4s → 5.3s            exact wins
  //     2D N=48  6×6 tiles   132→78 its, 26.6s → 18.7s           exact wins
  //     2D N=64  4×4 tiles   155→135 its, 72.9s → 117.8s         exact LOSES
  // So: use exact up to roughly N=48, limited-memory beyond it. The crossover is
  // where the denser KKT outruns the halved iteration count.
  //   DD_LM_HISTORY=n     limited_memory_max_history (IPOPT default 6) — the
  //                       multiplier on those extra RHS.
  //   DD_LM_AUG=extended  put the low-rank terms in the augmented matrix
  //                       instead of Sherman–Morrison multi-solves.
  {
    const char *env = std::getenv("DD_HESSIAN");
    app->Options()->SetStringValue("hessian_approximation",
                                   env ? env : hessian.c_str());
    if (const char *h = std::getenv("DD_LM_HISTORY"))
      app->Options()->SetIntegerValue("limited_memory_max_history", std::atoi(h));
    if (const char *a = std::getenv("DD_LM_AUG"))
      app->Options()->SetStringValue("limited_memory_aug_solver", a);
  }
  app->Options()->SetNumericValue("acceptable_tol", 1e-2);
  app->Options()->SetNumericValue("acceptable_dual_inf_tol", 1e2);
  if (solver == "mumps" || solver == "ma57" || solver == "ma97")
    app->Options()->SetStringValue("linear_solver", solver);
  if (solver == "ma57") {
    // Same dylib the Python drivers use (overridable via the HSLLIB env var).
    const char *hsl = std::getenv("HSLLIB");
    app->Options()->SetStringValue(
        "hsllib", hsl ? hsl
                      : "/Users/davidvillacis/src/hsl/hsl_ma57-5.3.2/src/.libs/"
                        "libhsl_ma57.dylib");
  }
  if (app->Initialize() != Ipopt::Solve_Succeeded) {
    std::cerr << "IPOPT initialization failed\n";
    return false;
  }
  return true;
}

// One attempted Scholtes level, for the per-level table and the solution dumps.
struct Level {
  double t;
  int status, iters;
  double comp_res, weight, obj, xi_max;
  bool converged;
};

struct RunResult {
  std::vector<double> best_x;
  double best_comp = 1e300;
  double best_t = 0.0;
  int total_iter = 0;
  std::vector<Level> hist;
};

// The Scholtes continuation. Per level: write t (and the θ-gauge ridge ε_θ =
// c_θ·t) into the TNLP, tighten IPOPT's tolerances with t, optimize through the
// caller's closure, log the table row, and keep the TIGHTEST converged level
// (smallest max r·(1−δ)) — NOT the smallest loss: a loose level overfits
// u_clean and is not a real lower-level solution. Stop at the first failed
// level.
//
// `value_is_alpha`: the per-level value column is α itself (the uniform driver,
// whose table stays diffable against ../lifted_mpcc_unitball_v2.py) or Q(α)
// (the staggered drivers, matching their Python tables).
//
// `warm_start`: pass the warm-start point (primal + duals) to IPOPT between
// levels. The staggered drivers keep this OPT-IN (--dual-warmstart, measured
// mixed elsewhere in the repo); the uniform driver has always carried it.
template <class OptimizeFn>
RunResult run_scholtes(Ipopt::SmartPtr<Ipopt::IpoptApplication> app,
                       Ipopt::MpccTNLPBase &p, const std::vector<double> &sched,
                       double c_theta, double tol, bool warm_start,
                       bool value_is_alpha, OptimizeFn &&optimize) {
  std::cout << std::scientific << std::setprecision(3);
  std::cout << "        t     status  iters      comp_res"
            << (value_is_alpha ? "       alpha" : "      weight")
            << "         obj     sec\n";

  RunResult res;
  res.best_t = sched.empty() ? p.t_ : sched.front();

  for (double t : sched) {
    p.t_ = t;
    p.eps_theta_ = c_theta * t; // TR gauge ridge, weight ∝ t (0 = off)
    app->Options()->SetNumericValue("tol", std::max(tol, 0.1 * t));
    app->Options()->SetNumericValue("acceptable_tol", std::max(tol, 1.0 * t));
    if (warm_start && p.have_warm_)
      app->Options()->SetStringValue("warm_start_init_point", "yes");

    // If IPOPT fails without reaching finalize_solution (exception paths,
    // invalid-problem returns), sol_x_ would still hold the PREVIOUS level's
    // solution and the failed level's table row would silently reprint its
    // stats. Clear first — such a row now reads 0/empty instead. (The warm-
    // start copies wx_/wlam_ and have_warm_ are untouched.)
    p.sol_x_.clear();
    p.sol_lam_.clear();
    p.sol_obj_ = 0.0;

    // clock() sums CPU across threads — fine as a within-solver progress
    // column, NEVER as a cross-solver comparison (it penalizes a parallel
    // solver by ~the thread count; see ../CLAUDE.md).
    const double cpu0 = (double)clock() / CLOCKS_PER_SEC;
    const Ipopt::ApplicationReturnStatus st = optimize();
    const double cpu = (double)clock() / CLOCKS_PER_SEC - cpu0;

    const int iters =
        IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
    res.total_iter += (iters > 0 ? iters : 0);

    // complementarity residual max r·(1−δ) and ξ = max|λ_comp| at the solution
    double comp = 0.0, xi_max = 0.0;
    const std::vector<double> &x = p.sol_x_;
    if (!x.empty())
      for (int e = 0; e < p.n_lift; ++e)
        comp = std::max(comp, x[p.oR + e] * (1.0 - x[p.oD + e]));
    for (int e = 0; e < p.n_lift && !p.sol_lam_.empty(); ++e)
      xi_max = std::max(xi_max, std::abs(p.sol_lam_[p.rcomp + e]));

    const double value =
        x.empty() ? 0.0
                  : (value_is_alpha ? x[p.oa] : p.weight_of_alpha(x[p.oa]));
    const bool ok = (st == Ipopt::Solve_Succeeded ||
                     st == Ipopt::Solved_To_Acceptable_Level);
    res.hist.push_back({t, (int)st, iters, comp,
                        x.empty() ? 0.0 : p.weight_of_alpha(x[p.oa]),
                        p.sol_obj_, xi_max, ok});

    std::cout << "  " << std::setw(9) << t << std::setw(8) << (int)st
              << std::setw(8) << iters << std::setw(14) << comp << std::setw(12)
              << value << std::setw(12) << p.sol_obj_ << std::setw(9) << cpu
              << "\n";

    if (ok && comp < res.best_comp) {
      res.best_comp = comp;
      res.best_x = x;
      res.best_t = t;
    } else if (!ok) {
      std::cout << "  [stop] IPOPT status " << (int)st << "\n";
      break;
    }
  }
  return res;
}

// The μ-coupled single-solve continuation (--t-update mu), port of the Python
// v2 driver: ONE IPOPT solve, the TNLP's intermediate callback slaving
// t = max(t_min, c·μ) (see mpcc_base.hpp). Seeded at t = max(t_min, c·μ₀) with
// IPOPT's monotone default μ₀ = 0.1; the θ-gauge ridge stays ∝ t through
// c_theta_live_. RunResult is shaped like run_scholtes's (one hist row), so
// print_summary and the solution dumps work unchanged.
template <class OptimizeFn>
RunResult run_mu_coupled(Ipopt::SmartPtr<Ipopt::IpoptApplication> app,
                         Ipopt::MpccTNLPBase &p, double t_min,
                         double t_mu_scale, double c_theta, double tol,
                         int print_level, OptimizeFn &&optimize) {
  const double mu0 = 0.1;
  p.t_mu_scale_ = t_mu_scale;
  p.t_floor_ = t_min;
  p.c_theta_live_ = c_theta;
  p.t_ = std::max(t_min, t_mu_scale * mu0);
  p.eps_theta_ = c_theta * p.t_;
  p.mu_progress_every_ = (print_level == 0) ? 25 : 0;
  p.mu_stall_iters_ = 0;
  p.mu_stall_warned_ = false;
  p.mu_hist_.clear();
  p.have_banked_ = false;
  p.banked_comp_ = 1e300;
  p.banked_t_ = 0.0;
  p.banked_x_.clear();
  p.mu_prev_ = 1e300;
  p.prev_valid_ = false;
  app->Options()->SetNumericValue("tol", std::max(tol, 0.1 * t_min));
  app->Options()->SetNumericValue("acceptable_tol", std::max(tol, 1.0 * t_min));

  std::cout << std::scientific << std::setprecision(3);
  std::cout << "  [mu-coupled] single solve: t seeded " << p.t_ << ", floor "
            << t_min << ", c=" << t_mu_scale << "\n";
  RunResult res;
  p.sol_x_.clear();
  p.sol_lam_.clear();
  p.sol_obj_ = 0.0;
  const double cpu0 = (double)clock() / CLOCKS_PER_SEC;
  const Ipopt::ApplicationReturnStatus st = optimize();
  const double cpu = (double)clock() / CLOCKS_PER_SEC - cpu0;
  const int iters =
      IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
  res.total_iter = (iters > 0 ? iters : 0);
  double comp = 0.0, xi_max = 0.0;
  const std::vector<double> &x = p.sol_x_;
  if (!x.empty())
    for (int e = 0; e < p.n_lift; ++e)
      comp = std::max(comp, x[p.oR + e] * (1.0 - x[p.oD + e]));
  for (int e = 0; e < p.n_lift && !p.sol_lam_.empty(); ++e)
    xi_max = std::max(xi_max, std::abs(p.sol_lam_[p.rcomp + e]));
  const bool ok =
      (st == Ipopt::Solve_Succeeded || st == Ipopt::Solved_To_Acceptable_Level);
  const double w = x.empty() ? 0.0 : p.weight_of_alpha(x[p.oa]);
  res.hist.push_back({p.t_, (int)st, iters, comp, w, p.sol_obj_, xi_max, ok});
  std::cout << "        t     status  iters      comp_res      weight"
               "         obj     sec\n";
  std::cout << "  " << std::setw(9) << p.t_ << std::setw(8) << (int)st
            << std::setw(8) << iters << std::setw(14) << comp << std::setw(12)
            << w << std::setw(12) << p.sol_obj_ << std::setw(9) << cpu << "\n";
  if (ok) {
    res.best_comp = comp;
    res.best_x = x;
    res.best_t = p.t_;
  } else if (p.have_banked_) {
    // The single solve failed, but intermediate barrier subproblems DID
    // converge (each monotone μ drop banked one) — report the tightest of
    // those instead of discarding the whole run, mirroring run_scholtes's
    // best-completed-level rule. The table row above still records the
    // failed final attempt honestly.
    std::cout << "  [stop] IPOPT status " << (int)st
              << " — falling back to the deepest banked level t=" << p.banked_t_
              << " (max r(1-d) " << p.banked_comp_ << ")\n";
    res.best_comp = p.banked_comp_;
    res.best_x = p.banked_x_;
    res.best_t = p.banked_t_;
  } else {
    std::cout << "  [stop] IPOPT status " << (int)st
              << " (no converged level to bank — nothing to report)\n";
  }
  return res;
}

inline void print_summary(const Ipopt::MpccTNLPBase &p, const RunResult &r) {
  if (r.best_x.empty())
    return;
  std::cout
      << "----------------------------------------------------------------\n";
  std::cout << "  best level t  : " << std::scientific << r.best_t << "\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  total IPOPT it: " << r.total_iter << "\n";
  std::cout << "  alpha*        : " << r.best_x[p.oa] << "\n";
  std::cout << "  weight Q(a*)  : " << p.weight_of_alpha(r.best_x[p.oa])
            << "\n";
  std::cout << std::setprecision(2);
  std::cout << "  PSNR noisy    : " << psnr(p.uclean_, p.f_.data(), p.n_state)
            << " dB\n";
  std::cout << "  PSNR recon    : "
            << psnr(p.uclean_, r.best_x.data(), p.n_state) << " dB\n";
}

// End-of-run telemetry for the CG interface solve (--interface cg). Printed
// even when the run does not converge — the skip/fallback counts are exactly
// what diagnoses an interface the iteration cannot handle.
inline void print_interface_stats(const std::string &precond, bool alpha_peel,
                                  double cg_tol) {
  using Ipopt::DDArrowheadSolver;
  const DDArrowheadSolver::InterfaceStats st =
      DDArrowheadSolver::interface_stats();
  if (!st.solves)
    return;
  // "attempted" = solves where CG actually iterated: indefinite-S skips never
  // enter CG, and dead-skips (a failure already seen on this factorization)
  // run 0 its — including those diluted the avg (audit fix 2026-07-22; rows
  // with 0 dead-skips, i.e. all recorded 1D and 2D-strip tables, are
  // unchanged; tile rows read slightly higher under this definition).
  const long attempted = st.solves - st.skipped_indef - st.skipped_dead;
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "  CG interface (" << precond
            << (alpha_peel ? ", alpha-peel" : ", no-alpha-peel");
  if (st.peeled > 0)
    std::cout << ", dual-peel " << st.peeled << " duals";
  std::cout << "): " << st.solves << " solves, " << st.iters << " total its ("
            << (attempted ? (double)st.iters / attempted : 0.0)
            << " avg over attempted), " << st.nonconv
            << " did not reach tol=" << std::scientific << std::setprecision(1)
            << cg_tol << std::defaultfloat << "\n";
  std::cout << "    " << st.skipped_indef
            << " skipped (S indefinite — CG needs SPD), " << st.fallbacks
            << " fell back to the direct MA57 solve, "
            << (attempted - st.fallbacks) << " converged iteratively\n";
  if (st.skipped_dead)
    std::cout << "    " << st.skipped_dead
              << " skipped after a failure on the "
                 "same factorization (doomed re-attempts avoided)\n";
  if (st.cache_builds)
    std::cout << "    peel Z/T cache rebuilt on " << st.cache_builds
              << " factorization(s) (sparse-MA57 backsolve, #5)\n";
  std::cout << std::fixed << std::setprecision(2) << "    In(S)_neg: mean "
            << (st.solves ? (double)st.sneg_sum / st.solves : 0.0) << ", max "
            << st.sneg_max << ", SPD on " << st.s_spd << " / " << st.solves
            << " solves\n";
  std::cout << std::setprecision(3) << "    interface wall " << st.wall
            << " s over " << DDArrowheadSolver::omp_threads()
            << " OpenMP thread(s)\n";
}

// End-of-run telemetry for the signed-MA57 MINRES interface (--interface
// minres). The its-vs-age table IS the lag experiment: age = factorizations
// since the preconditioner snapshot. At age 0 the preconditioned spectrum is
// exactly {−1,+1} (signed-LDLᵀ, GMPS 1992) so MINRES needs ~2 its; how fast
// its/solve and the fallback rate grow with age is the measured answer to
// "can the serial S factorization be amortized across Newton steps?".
inline void print_minres_stats(double tol, int lag) {
  using Ipopt::DDArrowheadSolver;
  const DDArrowheadSolver::InterfaceStats st =
      DDArrowheadSolver::interface_stats();
  if (!st.solves)
    return;
  const long attempted = st.solves - st.sgn_unavail - st.skipped_dead;
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "  signed-MA57 MINRES interface (lag=" << lag
            << "): " << st.solves << " solves, " << st.iters << " total its ("
            << (attempted ? (double)st.iters / attempted : 0.0)
            << " avg over attempted), " << st.nonconv
            << " did not reach tol=" << std::scientific << std::setprecision(1)
            << tol << std::defaultfloat << "\n";
  std::cout << "    " << (attempted - st.fallbacks) << " accepted iteratively, "
            << st.fallbacks << " fell back to the direct MA57 solve";
  if (st.skipped_dead)
    std::cout << ", " << st.skipped_dead << " dead-skipped";
  if (st.sgn_unavail)
    std::cout << ", " << st.sgn_unavail << " without preconditioner";
  std::cout << "\n";
  std::cout << "    preconditioner: " << st.sgn_builds << " snapshot build(s), "
            << st.sgn_fail << " self-check failure(s)\n";
  if (lag > 1) {
    std::cout << "    its vs preconditioner age (the lag experiment):\n";
    for (int a = 0; a < DDArrowheadSolver::InterfaceStats::MAXAGE; ++a) {
      if (!st.age_cnt[a])
        continue;
      std::cout << "      age " << std::setw(2) << a << ": " << std::setw(6)
                << st.age_cnt[a] << " solves, avg " << std::fixed
                << std::setprecision(1) << (double)st.age_its[a] / st.age_cnt[a]
                << " its, " << st.age_fb[a] << " fallback(s)\n";
    }
  }
  std::cout << std::fixed << std::setprecision(2) << "    In(S)_neg: mean "
            << (st.solves ? (double)st.sneg_sum / st.solves : 0.0) << ", max "
            << st.sneg_max << ", SPD on " << st.s_spd << " / " << st.solves
            << " solves\n";
  std::cout << std::setprecision(3) << "    interface wall " << st.wall
            << " s over " << DDArrowheadSolver::omp_threads()
            << " OpenMP thread(s)\n";
}

// The five evaluation checksums of --self-check, reproducible from Python in a
// few lines. Catches a mis-ported derivative BEFORE IPOPT ever runs — the
// failure mode that would otherwise look like "DD does not converge". The
// partition part of the self-check stays in each driver.
inline void print_checksums(Ipopt::MpccTNLPBase &p) {
  const int n = p.n, mcon = p.mcon;
  std::vector<double> x = p.x_start_, g(mcon), gf(n);
  double obj = 0.0;
  p.eval_f(n, x.data(), true, obj);
  p.eval_grad_f(n, x.data(), true, gf.data());
  p.eval_g(n, x.data(), true, mcon, g.data());
  std::vector<double> jv(p.jr_.size()), hv(p.hr_.size()), lam(mcon, 1.0);
  p.eval_jac_g(n, x.data(), true, mcon, (Ipopt::Index)jv.size(), NULL, NULL,
               jv.data());
  p.eval_h(n, x.data(), true, 1.0, mcon, lam.data(), true,
           (Ipopt::Index)hv.size(), NULL, NULL, hv.data());

  double gfmax = 0, gmax = 0, sj = 0, sh = 0;
  for (double v : gf)
    gfmax = std::max(gfmax, std::abs(v));
  for (double v : g)
    gmax = std::max(gmax, std::abs(v));
  for (double v : jv)
    sj += std::abs(v);
  for (double v : hv)
    sh += std::abs(v);

  std::cout << std::scientific << std::setprecision(12);
  std::cout << "  self-check at the starting point, t=" << p.t_ << ":\n";
  std::cout << "    f(x0)        = " << obj << "\n";
  std::cout << "    max|grad f|  = " << gfmax << "\n";
  std::cout << "    max|g(x0)|   = " << gmax << "\n";
  std::cout << "    sum|J|       = " << sj << "   (" << p.jr_.size()
            << " nnz)\n";
  std::cout << "    sum|H|       = " << sh << "   (" << p.hr_.size()
            << " nnz, lam=1, obj_factor=1)\n";
  std::cout << std::defaultfloat;
}

} // namespace driver

#endif // DRIVER_COMMON_HPP
