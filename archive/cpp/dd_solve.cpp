// Scholtes-continuation driver for the uniform-grid unit-ball lifted TV-MPCC.
//
//   --solver mumps|ma57|ma97   IPOPT's own linear solver (validates the TNLP port
//                         against the Python cyipopt reference at equal N)
//   --solver dd  --nsub k the domain-decomposition arrowhead solver
//
// Output columns mirror ../lifted_mpcc_unitball_v2.py's per-level table so the two
// can be diffed directly.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
#include "dd_solver.hpp"
#include "driver_common.hpp"
#include "mpcc_tnlp.hpp"

using namespace Ipopt;

int main(int argc, char** argv) {
   std::string data, solver = "ma57";
   std::string interface_solver = "direct", precond = "asd", cg_apply = "assembled";
   std::string wk_backend = "ma57";
   double t0 = 1.0, tmin = 1e-4, factor = 0.3, tol = 1e-6, sigma = 0.1;
   double alpha0 = 0.0, c_theta = 0.0, cg_tol = 1e-10;
   bool alpha0_set = false, alpha_peel = true, dual_peel = true;
   int size = 0, seed = 0, nsub = 2, maxiter = 1500, printlevel = 0, cg_maxit = 500;

   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
         if (i + 1 >= argc) {           // argv[argc] is NULL — not a string
            std::cerr << a << " needs a value\n";
            std::exit(2);
         }
         return argv[++i];
      };
      if      (a == "--data")     data = next();
      else if (a == "--solver")   solver = next();
      else if (a == "--nsub")     nsub = std::stoi(next());
      else if (a == "--t0")       t0 = std::stod(next());
      else if (a == "--t-min")    tmin = std::stod(next());
      else if (a == "--factor")   factor = std::stod(next());
      else if (a == "--tol")      tol = std::stod(next());
      else if (a == "--size")     size = std::stoi(next());
      else if (a == "--sigma")    sigma = std::stod(next());
      else if (a == "--seed")     seed = std::stoi(next());
      else if (a == "--alpha0")   { alpha0 = std::stod(next()); alpha0_set = true; }
      else if (a == "--c-theta")  c_theta = std::stod(next());
      else if (a == "--max-iter") maxiter = std::stoi(next());
      else if (a == "--print-level") printlevel = std::stoi(next());
      else if (a == "--interface") interface_solver = next();
      else if (a == "--precond")  precond = next();
      else if (a == "--wk-backend") wk_backend = next();
      else if (a == "--cg-tol")   cg_tol = std::stod(next());
      else if (a == "--cg-max-iter") cg_maxit = std::stoi(next());
      else if (a == "--cg-apply") cg_apply = next();
      else if (a == "--no-alpha-peel") alpha_peel = false;
      else if (a == "--no-dual-peel") dual_peel = false;
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (data.empty()) {
      std::cerr << "usage: dd_solve --data <image|phantom.txt> [--size N] "
                   "[--solver mumps|ma57|ma97|dd] [--nsub k] ...\n";
      return 2;
   }
   if (solver != "mumps" && solver != "ma57" && solver != "ma97" && solver != "dd") {
      std::cerr << "--solver must be mumps|ma57|ma97|dd\n";
      return 2;
   }
   if (interface_solver != "direct" && interface_solver != "cg") {
      std::cerr << "--interface must be direct|cg\n";
      return 2;
   }
   if (interface_solver == "cg" && solver != "dd") {
      std::cerr << "--interface cg needs --solver dd (it replaces the arrowhead's "
                   "interface solve)\n";
      return 2;
   }
   if (precond != "jacobi" && precond != "bj" && precond != "asd") {
      std::cerr << "--precond must be jacobi|bj|asd\n";
      return 2;
   }
   if (cg_apply != "assembled" && cg_apply != "matfree") {
      std::cerr << "--cg-apply must be assembled|matfree\n";
      return 2;
   }
   if (wk_backend != "ma57" && wk_backend != "mumps" &&
       wk_backend != "hybrid") {
      std::cerr << "--wk-backend must be ma57|mumps|hybrid\n";
      return 2;
   }
   if (wk_backend != "ma57" && solver != "dd") {
      std::cerr << "--wk-backend " << wk_backend
                << " needs --solver dd (it selects the "
                   "arrowhead's W_k block backend)\n";
      return 2;
   }
#ifndef DD_HAVE_MUMPS
   if (wk_backend != "ma57") {
      std::cerr << "--wk-backend " << wk_backend
                << ": this binary was built without MUMPS "
                   "support (build with COIN ThirdParty-Mumps visible to "
                   "pkg-config as coinmumps)\n";
      return 2;
   }
#endif
   if (!driver::valid_schedule(t0, tmin, factor)) return 2;
   if (sigma <= 0) {
      std::cerr << "--sigma must be > 0 (the default alpha0 is log(0.7*sigma))\n";
      return 2;
   }
   if (!image_io::ends_with(data, ".txt") && size <= 0) {
      std::cerr << "image input requires --size N (target side length)\n";
      return 2;
   }

   SmartPtr<MpccTNLP> mpcc =
      new MpccTNLP(data, image_io::Opts{size, sigma, (unsigned)seed});
   if (solver == "dd" && (nsub < 1 || nsub > mpcc->N)) {
      std::cerr << "--nsub must be in [1, N] (N=" << mpcc->N << ")\n";
      return 2;
   }
   // Noise-aware default α₀ = log(0.7σ), calibrated to the cameraman TV optimum.
   mpcc->alpha0_ = alpha0_set ? alpha0 : std::log(0.7 * sigma);
   mpcc->x_start_.resize(mpcc->n);
   mpcc->cold_start(mpcc->x_start_.data());
   // The gauge ridge is measured against the starting angles; capture them once.
   mpcc->set_theta_ref(mpcc->x_start_.data());

   std::cout << "Lifted TV-MPCC (unit-ball, C++)  N=" << mpcc->N << "  m=" << mpcc->m
             << "  n=" << mpcc->n << "  m_con=" << mpcc->mcon
             << "  KKT dim=" << mpcc->kkt_dim
             << "  solver=" << solver;
   if (solver == "dd") std::cout << "  nsub=" << nsub << "×" << nsub;
   if (solver == "dd" && wk_backend != "ma57")
      std::cout << "  wk-backend=" << wk_backend
                << (wk_backend == "hybrid" ? "(ma57-solves+mumps-schur)"
                                           : "(schur)");
   if (c_theta > 0) std::cout << "  c_theta=" << c_theta;
   std::cout << "  α₀=" << mpcc->alpha0_ << "\n";

   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   if (!driver::init_app(app, printlevel, maxiter, solver)) return 1;

   if (solver == "dd") {
      DDArrowheadSolver::config(mpcc->N, nsub);
      DDArrowheadSolver::config_interface(
         interface_solver == "cg" ? DDArrowheadSolver::IFACE_CG
                                  : DDArrowheadSolver::IFACE_DIRECT,
         precond == "jacobi" ? DDArrowheadSolver::PRECOND_JACOBI
         : precond == "bj"   ? DDArrowheadSolver::PRECOND_BJ
                             : DDArrowheadSolver::PRECOND_ASD,
         alpha_peel, cg_tol, cg_maxit);
      DDArrowheadSolver::config_cg_apply(
         cg_apply == "matfree" ? DDArrowheadSolver::APPLY_MATFREE
                               : DDArrowheadSolver::APPLY_ASSEMBLED);
      DDArrowheadSolver::config_wk_backend(
         wk_backend == "mumps"    ? DDArrowheadSolver::WK_MUMPS
         : wk_backend == "hybrid" ? DDArrowheadSolver::WK_HYBRID
                                  : DDArrowheadSolver::WK_MA57);
      DDArrowheadSolver::config_alpha(mpcc->oa);
      // The built-in tile partition promotes the cut-corner dual pairs too, so
      // the dual peel applies here exactly as in the staggered 2D driver.
      DDArrowheadSolver::config_peel(dual_peel ? mpcc->n : (1 << 30));
      DDArrowheadSolver::reset_interface_stats();
   }
   auto optimize = [&]() -> ApplicationReturnStatus {
      if (solver == "dd") {
         SmartPtr<AlgorithmBuilder> b = new CustomSolverBuilder<DDArrowheadSolver>();
         return app->OptimizeNLP(new TNLPAdapter(GetRawPtr(mpcc)), b);
      }
      return app->OptimizeTNLP(GetRawPtr(mpcc));
   };

   driver::RunResult res = driver::run_scholtes(
      app, *mpcc, driver::scholtes_schedule(t0, tmin, factor), c_theta, tol,
      /*warm_start=*/true, /*value_is_alpha=*/true, optimize);
   driver::print_summary(*mpcc, res);
   if (solver == "dd" && interface_solver == "cg")
      driver::print_interface_stats(precond, alpha_peel, cg_tol);
   return 0;
}
