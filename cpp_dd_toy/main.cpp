// dd_toy — Lueg et al.'s Schur-complement decomposition inside IPOPT, on a
// 2D semilinear optimal control problem.  See README.md.
//
//   ./dd_toy --N 32 --tiles 4                          DD, direct Schur solve
//   ./dd_toy --N 32 --tiles 4 --schur pcg --precond asd
//   ./dd_toy --N 32 --tiles 4 --solver mumps           same NLP, IPOPT's MUMPS
//   ./dd_toy --N 12 --tiles 3 --check --verbose 2      compare every solve to dense
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLPAdapter.hpp"
#include "ipopt_bridge.hpp"
#include "problem.hpp"

using namespace Ipopt;

static void usage() {
   std::printf(
      "usage: dd_toy [options]\n"
      "  problem    --N 32  --tiles 4  --kappa 1  --alpha 1e-4  --fmax 40  --amp 2\n"
      "  solver     --solver dd|mumps         (dd: the Schur decomposition; mumps: reference)\n"
      "  dd         --schur direct|pcg  --precond none|bj|as|asd\n"
      "             --cg-tol 1e-10  --cg-maxit 2000\n"
      "             --matvec sk|backsolve     (S·v by eq. 15 or eq. 12)\n"
      "             --pcg-inertia curvature|exact  (pcg: In(S) from the CG curvature test (paper)\n"
      "                                       or from a dense factorization of S (diagnostic))\n"
      "             --local-inertia           (pcg: also require every S_k SPD, eq. 26)\n"
      "             --check [maxdim]          (dense reference per factorization; default 4000)\n"
      "             --verbose 0|1|2\n"
      "  ipopt      --print-level 5  --tol 1e-8  --max-iter 500\n"
      "  output     --save sol.csv\n");
}

int main(int argc, char** argv) {
   SemilinearDD::Params par;
   dd::SchurDD::Options dopt;
   std::string solver = "dd", save;
   int print_level = 5, max_iter = 500;
   double tol = 1e-8;

   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> const char* {
         if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
         return argv[++i];
      };
      if (a == "--N") par.N = std::atoi(next());
      else if (a == "--tiles") par.P = std::atoi(next());
      else if (a == "--kappa") par.kappa = std::atof(next());
      else if (a == "--alpha") par.alpha = std::atof(next());
      else if (a == "--fmax") par.fmax = std::atof(next());
      else if (a == "--amp") par.amp = std::atof(next());
      else if (a == "--solver") solver = next();
      else if (a == "--schur") {
         const std::string v = next();
         dopt.interface = v == "pcg" ? dd::SchurDD::Interface::Pcg : dd::SchurDD::Interface::Direct;
      } else if (a == "--precond") {
         const std::string v = next();
         dopt.precond = v == "none" ? dd::SchurDD::Precond::None
                      : v == "bj"   ? dd::SchurDD::Precond::BJ
                      : v == "as"   ? dd::SchurDD::Precond::AS
                                    : dd::SchurDD::Precond::ASd;
      } else if (a == "--cg-tol") dopt.cg_tol = std::atof(next());
      else if (a == "--cg-maxit") dopt.cg_maxit = std::atoi(next());
      else if (a == "--matvec") dopt.matvec_backsolve = std::string(next()) == "backsolve";
      else if (a == "--local-inertia") dopt.local_inertia = true;
      else if (a == "--pcg-inertia") dopt.pcg_exact_inertia = std::string(next()) == "exact";
      else if (a == "--check") {
         dopt.check_max_dim = 4000;
         if (i + 1 < argc && argv[i + 1][0] != '-') dopt.check_max_dim = std::atoi(next());
      } else if (a == "--verbose") dopt.verbose = std::atoi(next());
      else if (a == "--print-level") print_level = std::atoi(next());
      else if (a == "--tol") tol = std::atof(next());
      else if (a == "--max-iter") max_iter = std::atoi(next());
      else if (a == "--save") save = next();
      else { usage(); return a == "--help" || a == "-h" ? 0 : 1; }
   }
   if (par.N < 2 || par.P < 1 || par.P > par.N) { std::fprintf(stderr, "need 1 ≤ tiles ≤ N\n"); return 1; }

   SmartPtr<SemilinearDD> problem = new SemilinearDD(par);
   std::printf("problem   N=%d (h=%.4g)  κ=%g  α=%g  |f|≤%g  amp=%g\n", par.N, 1.0 / (par.N + 1),
               par.kappa, par.alpha, par.fmax, par.amp);
   std::printf("partition %d×%d tiles  vars=%d  rows=%d  complicating y: p=%d  copies=%d\n",
               par.P, par.P, problem->n_vars(), problem->n_rows(), problem->n_complicating(),
               problem->n_copies());

   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   OptionsList& opt = *app->Options();
   opt.SetStringValue("sb", "yes");
   opt.SetIntegerValue("print_level", print_level);
   opt.SetIntegerValue("max_iter", max_iter);
   opt.SetNumericValue("tol", tol);
   opt.SetStringValue("hessian_approximation", "exact");
   if (solver == "mumps") opt.SetStringValue("linear_solver", "mumps");
   if (app->Initialize() != Solve_Succeeded) { std::fprintf(stderr, "IPOPT init failed\n"); return 1; }

   const auto t0 = std::chrono::steady_clock::now();
   ApplicationReturnStatus status;
   if (solver == "mumps") {
      status = app->OptimizeTNLP(GetRawPtr(problem));
   } else {
      DDLinearSolver::configure(problem->kkt_owner(), problem->n_tiles(), dopt);
      SmartPtr<AlgorithmBuilder> builder = new DDSolverBuilder();
      status = app->OptimizeNLP(new TNLPAdapter(GetRawPtr(problem)), builder);
   }
   const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

   const int iters = IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
   std::printf("\n==== result ===========================================================\n");
   std::printf("status %d   IPOPT iterations %d   objective %.12e   wall %.3fs\n", (int)status,
               iters, problem->objective, wall);
   std::printf("max |copy − y| = %.2e   controls at a bound: %d of %d\n", problem->max_link_gap,
               problem->n_at_bound, par.N * par.N);

   if (solver != "mumps") {
      const auto& s = DDLinearSolver::stats();
      const char* pc[] = {"none", "bj", "as", "asd"};
      std::printf("---- linear solver: Schur DD, %s", dopt.interface == dd::SchurDD::Interface::Direct
                                                      ? "direct (xSC)" : "pcg");
      if (dopt.interface == dd::SchurDD::Interface::Pcg)
         std::printf(", precond %s, matvec %s, inertia %s", pc[(int)dopt.precond],
                     dopt.matvec_backsolve ? "back-solve (12)" : "S_k (15)",
                     dopt.pcg_exact_inertia ? "exact (dense S)" : "CG curvature");
      std::printf("\n");
      std::printf("  nP=%d  KKT dim=%d  p=%d  max p_k=%d  max t_k=%d\n", s.n_tiles, s.dim, s.p,
                  s.max_pk, s.max_tk);
      std::printf("  factorizations %ld  (singular %ld, wrong inertia %ld, S not PD %ld)\n",
                  s.factorizations, s.singular, DDLinearSolver::wrong_inertia(), s.s_not_pd);
      if (dopt.interface == dd::SchurDD::Interface::Pcg)
         std::printf("  pcg: solves %ld, iterations mean %.1f max %ld, negative curvature %ld, "
                     "unconverged %ld, non-SPD local blocks %ld\n",
                     s.solves, s.solves ? (double)s.cg_iters / s.solves : 0.0, s.cg_max,
                     s.cg_breakdowns, s.cg_unconverged, s.local_not_pd);
      if (DDLinearSolver::late_breakdowns())
         std::printf("  pcg: %ld breakdowns AFTER the matrix was accepted (CG iterate used)\n",
                     DDLinearSolver::late_breakdowns());
      std::printf("  time: factorize %.3fs  solve %.3fs\n", s.t_factor, s.t_solve);
      if (s.checks)
         std::printf("  dense check: %ld solves, max rel. error %.1e, inertia mismatches %ld\n",
                     s.checks, s.check_max_err, s.check_inertia_mismatch);
   }

   if (!save.empty()) {
      if (problem->save_csv(save)) std::printf("saved %s\n", save.c_str());
      else std::fprintf(stderr, "could not write %s\n", save.c_str());
   }
   return status == Solve_Succeeded || status == Solved_To_Acceptable_Level ? 0 : 2;
}
