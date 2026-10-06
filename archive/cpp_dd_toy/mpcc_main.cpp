// dd_mpcc — the MPCC of ../cpp_minimal (TV-weight learning, consensus form,
// μ-coupled Scholtes continuation) solved with this folder's SchurDD, under
// either host: IPOPT (as ../cpp_minimal/tv_learn does) or the paper-style
// SimpleIPM.  The problem class, the image loader and the partition are
// included from ../cpp_minimal unchanged.
//
//   ./dd_mpcc --size 32 --nsub 4                       IPOPT host, direct Schur
//   ./dd_mpcc --size 32 --nsub 4 --host simple         paper-style host
//   ./dd_mpcc --size 32 --nsub 4 --host simple --schur pcg --precond as
//
// The continuation and IPOPT settings are those of ../cpp_minimal/main.cpp;
// SimpleIPM gets the same μ schedule (barrier_tol_factor 1000, linear
// decrease 0.7, superlinear power 1.1), tolerances and starting point.
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLPAdapter.hpp"
#include "ipopt_bridge.hpp"
#include "simple_ipm.hpp"
#include "../cpp_minimal/problem.hpp"

using namespace Ipopt;

static double psnr(const std::vector<double>& clean, const double* u) {
   double mse = 0.0;
   for (size_t i = 0; i < clean.size(); ++i) mse += (clean[i] - u[i]) * (clean[i] - u[i]);
   mse /= (double)clean.size();
   return mse == 0.0 ? 1e9 : 10.0 * std::log10(1.0 / mse);
}

static void usage() {
   std::printf(
      "usage: dd_mpcc [options]\n"
      "  instance   --data ../images/cameraman.png  --size 32  --nsub 4  --sigma 0.1  --seed 0\n"
      "             --comp scholtes|fb\n"
      "  host       --host ipopt|simple|mumps  --reg local|global  --max-iter 3000\n"
      "             (mumps: IPOPT with its own MUMPS, no decomposition — the reference)\n"
      "  linear     --block-solver sparse|dense  --schur direct|pcg  --precond none|bj|as|asd\n"
      "             --coarse none|faces  --coarse-mode balanced|additive\n"
      "             --cg-tol 1e-10  --cg-maxit 2000  --pcg-inertia curvature|exact|free\n"
      "             --verbose 0..3  --threads n\n"
      "  output     --print-level 0|5 (IPOPT host)  --quiet (simple host: no iteration table)\n");
}

int main(int argc, char** argv) {
   setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
   setenv("OPENBLAS_NUM_THREADS", "1", 0);
   Eigen::setNbThreads(1);

   std::string data = "../images/cameraman.png", comp = "scholtes", host = "ipopt";
   int N = 32, nsub = 4, max_iter = 3000, threads = 0, print_level = 0;
   double sigma = 0.1;
   unsigned seed = 0;
   bool quiet = false;
   SimpleIPM::Reg reg = SimpleIPM::Reg::Local;
   dd::SchurDD::Options dopt;
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--quiet") { quiet = true; continue; }
      if (a == "--help" || a == "-h" || i + 1 >= argc) { usage(); return a == "--help" ? 0 : 1; }
      const std::string v = argv[++i];
      if (a == "--data") data = v;
      else if (a == "--size") N = std::stoi(v);
      else if (a == "--nsub") nsub = std::stoi(v);
      else if (a == "--sigma") sigma = std::stod(v);
      else if (a == "--seed") seed = (unsigned)std::stoul(v);
      else if (a == "--comp") comp = v;
      else if (a == "--host") host = v;
      else if (a == "--reg") reg = v == "global" ? SimpleIPM::Reg::Global : SimpleIPM::Reg::Local;
      else if (a == "--max-iter") max_iter = std::stoi(v);
      else if (a == "--block-solver") dopt.sparse_blocks = v != "dense";
      else if (a == "--schur")
         dopt.interface = v == "pcg" ? dd::SchurDD::Interface::Pcg : dd::SchurDD::Interface::Direct;
      else if (a == "--precond")
         dopt.precond = v == "none" ? dd::SchurDD::Precond::None
                      : v == "bj"   ? dd::SchurDD::Precond::BJ
                      : v == "as"   ? dd::SchurDD::Precond::AS
                                    : dd::SchurDD::Precond::ASd;
      else if (a == "--coarse")
         dopt.coarse = v == "faces" ? dd::SchurDD::CoarseSpace::Faces : dd::SchurDD::CoarseSpace::None;
      else if (a == "--coarse-mode")
         dopt.coarse_mode = v == "additive" ? dd::SchurDD::CoarseMode::Additive
                                            : dd::SchurDD::CoarseMode::Balanced;
      else if (a == "--cg-tol") dopt.cg_tol = std::stod(v);
      else if (a == "--cg-maxit") dopt.cg_maxit = std::stoi(v);
      else if (a == "--pcg-inertia")
         dopt.inertia = v == "exact" ? dd::SchurDD::Inertia::Exact
                      : v == "free"  ? dd::SchurDD::Inertia::Free
                                     : dd::SchurDD::Inertia::Curvature;
      else if (a == "--verbose") dopt.verbose = std::stoi(v);
      else if (a == "--threads") threads = std::stoi(v);
      else if (a == "--print-level") print_level = std::stoi(v);
      else { usage(); return 1; }
   }
   const bool pcg = dopt.interface == dd::SchurDD::Interface::Pcg;
   if (!pcg) dopt.inertia = dd::SchurDD::Inertia::Exact;
#ifdef _OPENMP
   if (threads > 0) omp_set_num_threads(threads);
#endif

   // ---- the instance (as ../cpp_minimal/main.cpp)
   const Image img = load_image(data, N, sigma, seed);
   const Partition part(N, nsub);
   SmartPtr<ConsensusTV> problem = new ConsensusTV(
      img, part, sigma,
      comp == "fb" ? ConsensusTV::Comp::FischerBurmeister : ConsensusTV::Comp::Scholtes);
   const int n_x = problem->n + problem->n_ineq;   // primals in the KKT: x and slacks
   std::printf("MPCC  N=%d  %dx%d tiles  variables=%d  constraints=%d (%d inequalities)  copies=%d"
               "  host=%s\n",
               N, nsub, nsub, problem->n, problem->mcon, problem->n_ineq, problem->n_link,
               host.c_str());

   // ---- μ-coupled continuation (the callback in problem.hpp drives it)
   const double t_min = 1e-4, tol_target = 1e-8, mu0 = 0.1;
   const double tol = std::max(tol_target, 0.1 * t_min);
   problem->t_min = t_min;
   problem->t = std::max(t_min, problem->t_mu_scale * mu0);
   problem->eps_theta = problem->c_theta * problem->t;
   problem->gate_floor = tol_target;

   const auto t0 = std::chrono::steady_clock::now();
   int status_code = 0, iters = 0;
   bool ok = false;
   std::string status_text;
   long wrong_inertia = 0;
   dd::SchurDD::Stats s;
   SimpleIPM::Result r;

   if (host == "simple") {
      if (dopt.inertia == dd::SchurDD::Inertia::Free) dopt.inertia = dd::SchurDD::Inertia::Curvature;
      SimpleIPM ipm;
      ipm.opt.tol = tol;
      ipm.opt.max_iter = max_iter;
      ipm.opt.mu0 = mu0;
      ipm.opt.kappa_eps = 1000.0;
      ipm.opt.kappa_mu = 0.7;
      ipm.opt.theta_mu = 1.1;
      ipm.opt.acceptable_tol = std::max(tol_target, t_min);
      ipm.opt.acceptable_dual = 1e2;
      ipm.opt.acceptable_iter = 1;
      ipm.opt.reg = reg;
      ipm.opt.print_level = quiet ? 0 : 1;
      r = ipm.solve(*problem, problem->kkt_owner(), part.n_tiles, dopt, nullptr, &s);
      status_code = r.status;
      iters = r.iters;
      ok = r.status == 0 || r.status == 3 || (r.status == 2 && problem->gate_fired);
      status_text = r.status == 0 ? "converged" : r.status == 3 ? "acceptable"
                  : r.status == 2 ? (problem->gate_fired ? "level gate" : "stopped")
                  : r.status == 1 ? "iteration limit" : "FAILED";
   } else {
      SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
      OptionsList& opt = *app->Options();
      opt.SetStringValue("sb", "yes");
      opt.SetIntegerValue("print_level", print_level);
      opt.SetIntegerValue("max_iter", max_iter);
      opt.SetIntegerValue("acceptable_iter", 1);
      opt.SetStringValue("mu_strategy", "monotone");
      opt.SetStringValue("hessian_approximation", "exact");
      opt.SetNumericValue("acceptable_dual_inf_tol", 1e2);
      opt.SetNumericValue("barrier_tol_factor", 1000.0);
      opt.SetNumericValue("mu_linear_decrease_factor", 0.7);
      opt.SetNumericValue("mu_superlinear_decrease_power", 1.1);
      opt.SetNumericValue("tol", tol);
      opt.SetNumericValue("acceptable_tol", std::max(tol_target, t_min));
      if (dopt.inertia == dd::SchurDD::Inertia::Free) opt.SetNumericValue("neg_curv_test_tol", 1e-12);
      if (app->Initialize() != Solve_Succeeded) { std::fprintf(stderr, "IPOPT init failed\n"); return 1; }
      ApplicationReturnStatus st;
      if (host == "mumps") {
         opt.SetStringValue("linear_solver", "mumps");
         st = app->OptimizeTNLP(GetRawPtr(problem));
      } else {
         DDLinearSolver::configure(problem->kkt_owner(), n_x, part.n_tiles, dopt);
         SmartPtr<AlgorithmBuilder> builder = new DDSolverBuilder();
         st = app->OptimizeNLP(new TNLPAdapter(GetRawPtr(problem)), builder);
      }
      status_code = (int)st;
      iters = IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
      ok = st == Solve_Succeeded || st == Solved_To_Acceptable_Level ||
           (st == User_Requested_Stop && problem->gate_fired);
      status_text = st == User_Requested_Stop && problem->gate_fired ? "level gate"
                  : ok ? "converged" : "FAILED";
      s = DDLinearSolver::stats();
      wrong_inertia = DDLinearSolver::wrong_inertia();
   }
   const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
   const double rss = ru.ru_maxrss / 1048576.0;
#else
   const double rss = ru.ru_maxrss / 1024.0;
#endif
   std::printf("\n==== result (%s host) ===================================================\n",
               host.c_str());
   std::printf("status %d (%s)   iterations %d   t=%.2e   wall %.2fs   peak RSS %.0f MB\n",
               status_code, status_text.c_str(), iters, problem->t, wall, rss);
   if (host == "simple")
      std::printf("refused factorizations: tile inertia %ld, S not PD %ld, singular %ld;"
                  "  tile-iterations with δ_k > 0: %ld\n",
                  r.tile_corrections, r.s_corrections, r.singular, r.tiles_regularized);
   else
      std::printf("refused factorizations (wrong inertia): %ld, S not PD %ld, singular %ld\n",
                  wrong_inertia, s.s_not_pd, s.singular);
   std::printf("linear: blocks %s, %s;  nP=%d  KKT dim=%d  p=%d  factorizations %ld"
               " (tile blocks redone densely: %ld)",
               dopt.sparse_blocks ? "sparse" : "dense", pcg ? "pcg" : "direct", s.n_tiles, s.dim,
               s.p, s.factorizations, s.dense_fallbacks);
   if (pcg)
      std::printf("  pcg mean %.1f max %ld, breakdowns %ld, unconverged %ld",
                  s.solves ? (double)s.cg_iters / s.solves : 0.0, s.cg_max, s.cg_breakdowns,
                  s.cg_unconverged);
   std::printf("\n");
   const std::vector<double>& x = problem->solution;
   if (!x.empty()) {
      std::printf("alpha*      %.6f\n", x[problem->oa]);
      std::printf("max r(1-d)  %.2e\n", problem->max_complementarity(x.data()));
      std::printf("PSNR noisy  %.2f dB\n", psnr(img.clean, img.noisy.data()));
      std::printf("PSNR recon  %.2f dB\n", psnr(img.clean, x.data() + problem->ou));
   }
   return ok ? 0 : 2;
}
