// dd_toy — Lueg et al.'s Schur-complement decomposition inside IPOPT, on a
// 2D semilinear optimal control problem.  See README.md.
//
//   ./dd_toy --N 32 --tiles 4                          DD, direct Schur solve
//   ./dd_toy --N 32 --tiles 4 --schur pcg --precond as
//   ./dd_toy --N 512 --tiles 16 --schur pcg --precond as --coarse faces --threads 8
//   ./dd_toy --N 32 --tiles 4 --solver mumps           same NLP, IPOPT's MUMPS
//   ./dd_toy --N 12 --tiles 3 --check --verbose 2      compare every solve to dense
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLPAdapter.hpp"
#include "ipopt_bridge.hpp"
#include "problem.hpp"
#include "simple_ipm.hpp"

using namespace Ipopt;

static void usage() {
   std::printf(
      "usage: dd_toy [options]\n"
      "  problem    --N 32  --tiles 4  --kappa 1  --alpha 1e-4  --fmax 40  --amp 2\n"
      "  solver     --solver dd|mumps         (dd: the Schur decomposition; mumps: reference)\n"
      "             --host ipopt|simple       (simple: the paper-style IPM of simple_ipm.hpp)\n"
      "             --reg local|global        (simple host: regularize bad tiles only, or all)\n"
      "  blocks     --block-solver sparse|dense   (W_k: sparse LDLᵀ, or dense Bunch–Kaufman)\n"
      "  interface  --schur direct|pcg  --precond none|bj|as|asd\n"
      "             --coarse none|faces|sides|linear  --coarse-mode balanced|additive\n"
      "             --cg-tol 1e-10  --cg-maxit 2000\n"
      "             --matvec sk|backsolve     (S·v by eq. 15 or eq. 12)\n"
      "  inertia    --pcg-inertia curvature|exact|free   (pcg: paper's CG test, a factorization\n"
      "                                       of S, or IPOPT's inertia-free test)\n"
      "             --neg-curv-tol 1e-12      (for --pcg-inertia free)\n"
      "             --local-inertia           (pcg: also require every S_k SPD, eq. 26)\n"
      "  debug      --check [maxdim]          (dense reference per factorization; default 4000)\n"
      "             --verbose 0|1|2|3\n"
      "  ipopt      --print-level 5  --tol 1e-8  --max-iter 500\n"
      "             --obj-scaling 1|auto|x    (auto = 1/h²: needed for N ≳ 256)\n"
      "  run        --threads n               (OpenMP threads; build with OMP=1)\n"
      "  output     --save sol.csv\n");
}

static double peak_rss_mb() {
   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
   return ru.ru_maxrss / 1048576.0;   // bytes
#else
   return ru.ru_maxrss / 1024.0;      // kilobytes
#endif
}

int main(int argc, char** argv) {
   // one BLAS thread inside each tile; the parallelism is over tiles
   setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
   setenv("OPENBLAS_NUM_THREADS", "1", 0);
   Eigen::setNbThreads(1);

   SemilinearDD::Params par;
   dd::SchurDD::Options dopt;
   std::string solver = "dd", save, host = "ipopt";
   SimpleIPM::Reg reg = SimpleIPM::Reg::Local;
   int print_level = 5, max_iter = 500, threads = 0;
   double tol = 1e-8, neg_curv_tol = 1e-12, obj_scaling = 1.0;

   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
         if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
         return argv[++i];
      };
      if (a == "--N") par.N = std::stoi(next());
      else if (a == "--tiles") par.P = std::stoi(next());
      else if (a == "--kappa") par.kappa = std::stod(next());
      else if (a == "--alpha") par.alpha = std::stod(next());
      else if (a == "--fmax") par.fmax = std::stod(next());
      else if (a == "--amp") par.amp = std::stod(next());
      else if (a == "--solver") solver = next();
      else if (a == "--host") host = next();
      else if (a == "--reg") reg = next() == "global" ? SimpleIPM::Reg::Global : SimpleIPM::Reg::Local;
      else if (a == "--block-solver") dopt.sparse_blocks = next() != "dense";
      else if (a == "--schur")
         dopt.interface = next() == "pcg" ? dd::SchurDD::Interface::Pcg : dd::SchurDD::Interface::Direct;
      else if (a == "--precond") {
         const std::string v = next();
         dopt.precond = v == "none" ? dd::SchurDD::Precond::None
                      : v == "bj"   ? dd::SchurDD::Precond::BJ
                      : v == "as"   ? dd::SchurDD::Precond::AS
                                    : dd::SchurDD::Precond::ASd;
      } else if (a == "--coarse") {
         const std::string v = next();
         dopt.coarse = v == "faces"  ? dd::SchurDD::CoarseSpace::Faces
                     : v == "sides"  ? dd::SchurDD::CoarseSpace::Sides
                     : v == "linear" ? dd::SchurDD::CoarseSpace::Linear
                                     : dd::SchurDD::CoarseSpace::None;
      }
      else if (a == "--coarse-mode")
         dopt.coarse_mode = next() == "additive" ? dd::SchurDD::CoarseMode::Additive
                                                 : dd::SchurDD::CoarseMode::Balanced;
      else if (a == "--cg-tol") dopt.cg_tol = std::stod(next());
      else if (a == "--cg-maxit") dopt.cg_maxit = std::stoi(next());
      else if (a == "--matvec") dopt.matvec_backsolve = next() == "backsolve";
      else if (a == "--local-inertia") dopt.local_inertia = true;
      else if (a == "--pcg-inertia") {
         const std::string v = next();
         dopt.inertia = v == "exact" ? dd::SchurDD::Inertia::Exact
                      : v == "free"  ? dd::SchurDD::Inertia::Free
                                     : dd::SchurDD::Inertia::Curvature;
      } else if (a == "--neg-curv-tol") neg_curv_tol = std::stod(next());
      else if (a == "--check") {
         dopt.check_max_dim = 4000;
         if (i + 1 < argc && argv[i + 1][0] != '-') dopt.check_max_dim = std::stoi(next());
      } else if (a == "--verbose") dopt.verbose = std::stoi(next());
      else if (a == "--print-level") print_level = std::stoi(next());
      else if (a == "--tol") tol = std::stod(next());
      else if (a == "--obj-scaling") {
         const std::string v = next();
         obj_scaling = v == "auto" ? -1.0 : std::stod(v);
      }
      else if (a == "--max-iter") max_iter = std::stoi(next());
      else if (a == "--threads") threads = std::stoi(next());
      else if (a == "--save") save = next();
      else { usage(); return a == "--help" || a == "-h" ? 0 : 1; }
   }
   if (par.N < 2 || par.P < 1 || par.P > par.N) { std::fprintf(stderr, "need 1 ≤ tiles ≤ N\n"); return 1; }
   const bool pcg = dopt.interface == dd::SchurDD::Interface::Pcg;
   if (!pcg) dopt.inertia = dd::SchurDD::Inertia::Exact;   // direct mode factorizes S anyway

   int nthreads = 1;
#ifdef _OPENMP
   if (threads > 0) omp_set_num_threads(threads);
   nthreads = omp_get_max_threads();
#else
   if (threads > 1) std::fprintf(stderr, "note: built without OpenMP (OMP=1 ./build.sh); --threads ignored\n");
#endif

   const auto t_setup = std::chrono::steady_clock::now();
   SmartPtr<SemilinearDD> problem = new SemilinearDD(par);
   std::printf("problem   N=%d (h=%.4g)  κ=%g  α=%g  |f|≤%g  amp=%g\n", par.N, 1.0 / (par.N + 1),
               par.kappa, par.alpha, par.fmax, par.amp);
   std::printf("partition %d×%d tiles  vars=%d  rows=%d  complicating y: p=%d  copies=%d  "
               "(setup %.2fs, %d threads)\n",
               par.P, par.P, problem->n_vars(), problem->n_rows(), problem->n_complicating(),
               problem->n_copies(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t_setup).count(),
               nthreads);

   if (host == "simple") {
      // ---- the paper-style IPM: same problem, same SchurDD, no IPOPT
      if (dopt.inertia == dd::SchurDD::Inertia::Free) {
         std::fprintf(stderr, "note: --pcg-inertia free is an IPOPT option; using curvature\n");
         dopt.inertia = dd::SchurDD::Inertia::Curvature;
      }
      SimpleIPM ipm;
      ipm.opt.tol = tol;
      ipm.opt.max_iter = max_iter;
      ipm.opt.obj_scaling = obj_scaling < 0.0 ? (par.N + 1.0) * (par.N + 1.0) : obj_scaling;
      ipm.opt.reg = reg;
      ipm.opt.print_level = print_level > 0 ? 1 : 0;
      dd::SchurDD::Hints hints;
      problem->coarse_hints(hints.side, hints.xy);
      dd::SchurDD::Stats s;
      const auto r = ipm.solve(*problem, problem->kkt_owner(), problem->n_tiles(), dopt, &hints, &s);
      const char* st = r.status == 0 ? "converged" : r.status == 1 ? "iteration limit" : "FAILED";
      const double cg_mean = s.solves ? (double)s.cg_iters / s.solves : 0.0;
      std::printf("\n==== result (simple host, reg %s) ===================================\n",
                  reg == SimpleIPM::Reg::Local ? "local" : "global");
      std::printf("status %d (%s)   iterations %d   objective %.12e   wall %.3fs   peak RSS %.0f MB\n",
                  r.status, st, r.iters, problem->objective, r.wall, peak_rss_mb());
      std::printf("inf_pr %.1e  inf_du %.1e   max |copy − y| = %.2e   controls within 1%% of a bound: %d of %d\n",
                  r.inf_pr, r.inf_du, problem->max_link_gap, problem->n_at_bound, par.N * par.N);
      std::printf("  refused factorizations: tile inertia %ld, S not PD %ld, singular %ld;"
                  "  tile-iterations with δ_k > 0: %ld\n",
                  r.tile_corrections, r.s_corrections, r.singular, r.tiles_regularized);
      std::printf("  nP=%d  KKT dim=%d  p=%d   factorizations %ld (tile blocks redone densely: %ld)",
                  s.n_tiles, s.dim, s.p, s.factorizations, s.dense_fallbacks);
      if (pcg)
         std::printf("   pcg: solves %ld, iterations mean %.1f max %ld, unconverged %ld", s.solves,
                     cg_mean, s.cg_max, s.cg_unconverged);
      std::printf("\n  time: factorize %.2fs  solve %.2fs  rest %.2fs\n", s.t_factor, s.t_solve,
                  r.wall - s.t_factor - s.t_solve);
      if (!save.empty() && problem->save_csv(save)) std::printf("saved %s\n", save.c_str());
      return r.status == 0 ? 0 : 2;
   }

   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   OptionsList& opt = *app->Options();
   opt.SetStringValue("sb", "yes");
   opt.SetIntegerValue("print_level", print_level);
   opt.SetIntegerValue("max_iter", max_iter);
   opt.SetNumericValue("tol", tol);
   opt.SetStringValue("hessian_approximation", "exact");
   // The objective carries a factor h², so its gradient and the multipliers
   // shrink like h² and IPOPT's absolute tolerances are met too early on fine
   // grids.  "auto" scales the objective by 1/h² inside IPOPT (the reported
   // objective is unchanged).
   if (obj_scaling < 0.0) obj_scaling = (par.N + 1.0) * (par.N + 1.0);
   if (obj_scaling != 1.0) opt.SetNumericValue("obj_scaling_factor", obj_scaling);
   if (solver == "mumps") opt.SetStringValue("linear_solver", "mumps");
   if (solver != "mumps" && dopt.inertia == dd::SchurDD::Inertia::Free)
      opt.SetNumericValue("neg_curv_test_tol", neg_curv_tol);   // inertia-free curvature test
   if (app->Initialize() != Solve_Succeeded) { std::fprintf(stderr, "IPOPT init failed\n"); return 1; }

   const auto t0 = std::chrono::steady_clock::now();
   ApplicationReturnStatus status;
   if (solver == "mumps") {
      status = app->OptimizeTNLP(GetRawPtr(problem));
   } else {
      dd::SchurDD::Hints hints;
      problem->coarse_hints(hints.side, hints.xy);
      DDLinearSolver::configure(problem->kkt_owner(), problem->n_vars(), problem->n_tiles(), dopt,
                                std::move(hints));
      SmartPtr<AlgorithmBuilder> builder = new DDSolverBuilder();
      status = app->OptimizeNLP(new TNLPAdapter(GetRawPtr(problem)), builder);
   }
   const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

   const int iters = IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
   std::printf("\n==== result ===========================================================\n");
   std::printf("status %d   IPOPT iterations %d   objective %.12e   wall %.3fs   peak RSS %.0f MB\n",
               (int)status, iters, problem->objective, wall, peak_rss_mb());
   std::printf("max |copy − y| = %.2e   controls within 1%% of a bound: %d of %d\n", problem->max_link_gap,
               problem->n_at_bound, par.N * par.N);

   if (solver == "mumps") {
      std::printf("CSV,%d,%d,mumps,,,,,%d,%d,%.12e,%.3f,,,,,,,,,%.0f\n", par.N, par.P, nthreads,
                  iters, problem->objective, wall, peak_rss_mb());
   } else {
      const auto& s = DDLinearSolver::stats();
      const char* pc[] = {"none", "bj", "as", "asd"};
      const char* in[] = {"CG curvature", "exact", "inertia-free (IPOPT)"};
      const char* cs[] = {"none", "faces", "sides", "linear"};
      const std::string coarse = dopt.coarse == dd::SchurDD::CoarseSpace::None ? "none"
                               : std::string(cs[(int)dopt.coarse]) +
                                    (dopt.coarse_mode == dd::SchurDD::CoarseMode::Additive ? "-additive"
                                                                                          : "-balanced");
      std::printf("---- linear solver: Schur DD, blocks %s, %s", dopt.sparse_blocks ? "sparse" : "dense",
                  pcg ? "pcg" : "direct (xSC)");
      if (pcg)
         std::printf(", precond %s, coarse %s, matvec %s, inertia %s", pc[(int)dopt.precond],
                     coarse.c_str(), dopt.matvec_backsolve ? "back-solve (12)" : "S_k (15)",
                     in[(int)dopt.inertia]);
      std::printf("\n");
      std::printf("  nP=%d  KKT dim=%d  p=%d  max p_k=%d  max t_k=%d  faces=%d\n", s.n_tiles, s.dim,
                  s.p, s.max_pk, s.max_tk, s.n_faces);
      std::printf("  factorizations %ld  (singular %ld, wrong inertia %ld, S not PD %ld;"
                  " tile blocks redone densely %ld)\n",
                  s.factorizations, s.singular, DDLinearSolver::wrong_inertia(), s.s_not_pd,
                  s.dense_fallbacks);
      const double cg_mean = s.solves ? (double)s.cg_iters / s.solves : 0.0;
      if (pcg) {
         std::printf("  pcg: solves %ld, iterations mean %.1f max %ld, negative curvature %ld, "
                     "unconverged %ld, non-SPD local blocks %ld\n",
                     s.solves, cg_mean, s.cg_max, s.cg_breakdowns, s.cg_unconverged, s.local_not_pd);
         if (dopt.coarse != dd::SchurDD::CoarseSpace::None)
            std::printf("  coarse: dim %ld, failures %ld\n", s.coarse_dim, s.coarse_failures);
      }
      if (DDLinearSolver::late_breakdowns())
         std::printf("  pcg: %ld breakdowns AFTER the matrix was accepted (CG iterate used)\n",
                     DDLinearSolver::late_breakdowns());
      std::printf("  time: factorize %.2fs [W_k + S_k %.2f, S %.2f, precond %.2f, coarse %.2f]\n",
                  s.t_factor, s.t_blocks, s.t_sdirect, s.t_precond, s.t_coarse);
      std::printf("        solve     %.2fs [tile solves %.2f, interface %.2f]   rest of IPOPT %.2fs\n",
                  s.t_solve, s.t_tile_solves, s.t_interface, wall - s.t_factor - s.t_solve);
      if (s.checks)
         std::printf("  dense check: %ld solves, max rel. error %.1e, inertia mismatches %ld\n",
                     s.checks, s.check_max_err, s.check_inertia_mismatch);
      // N,P,solver,blocks,interface,precond,coarse,inertia,threads,iters,obj,wall,
      // fact,blocks,precond,coarse,tile_solves,interface,cg_mean,cg_max,rss
      std::printf("CSV,%d,%d,dd,%s,%s,%s,%s,%s,%d,%d,%.12e,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%ld,%.0f\n",
                  par.N, par.P, dopt.sparse_blocks ? "sparse" : "dense", pcg ? "pcg" : "direct",
                  pcg ? pc[(int)dopt.precond] : "", pcg ? coarse.c_str() : "",
                  pcg ? in[(int)dopt.inertia] : "exact", nthreads, iters, problem->objective, wall,
                  s.t_factor, s.t_blocks, s.t_precond, s.t_coarse, s.t_tile_solves, s.t_interface,
                  cg_mean, s.cg_max, peak_rss_mb());
   }

   if (!save.empty()) {
      if (problem->save_csv(save)) std::printf("saved %s\n", save.c_str());
      else std::fprintf(stderr, "could not write %s\n", save.c_str());
   }
   return status == Solve_Succeeded || status == Solved_To_Acceptable_Level ? 0 : 2;
}
