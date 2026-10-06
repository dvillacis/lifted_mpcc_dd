// main.cpp — learn the TV-denoising weight α of one image with IPOPT, solving
// every Newton system by domain decomposition.
//
// The algorithmic choices are FIXED here (the full cpp/ driver makes them flags):
//   formulation  consensus                 (problem.hpp)
//   solver       arrowhead DD, CG interface (arrowhead.hpp, ipopt_bridge.hpp)
//   Hessian      exact
//   start        Chambolle–Pock             (problem.hpp)
//   continuation μ-coupled Scholtes, one IPOPT solve
//   CG           up to 100000 iterations
//
// Only the instance is configurable, plus how the complementarity row is
// written (problem.hpp: --comp scholtes, the reference, or --comp fb):
//   ./tv_learn [--data IMG] [--size N] [--nsub K] [--sigma S] [--seed SEED]
//              [--comp scholtes|fb] [--save-solution FILE.npz | FILE.txt]
//              [--sff-solver cg|direct|direct-all] [--bound-push EPS]
// --sff-solver picks how the systems with S_ff are solved (arrowhead.hpp,
// Options::sff_direct): cg (the default), direct = Z by a sparse LDLT of the
// assembled S_ff, direct-all = the interface solve by it too.
// --bound-push sets IPOPT's bound_push and bound_frac (default 1e-2 each).  The
// Chambolle–Pock start puts δ = 1 exactly on every active cell; the default push
// moves those to δ = 0.99 before the first iteration and breaks h3 by 1% there.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLPAdapter.hpp"
#include "image.hpp"
#include "ipopt_bridge.hpp"
#include "partition.hpp"
#include "problem.hpp"
#include "save.hpp"

using namespace Ipopt;

static double psnr(const std::vector<double>& clean, const double* u) {
   double mse = 0.0;
   for (size_t i = 0; i < clean.size(); ++i) mse += (clean[i] - u[i]) * (clean[i] - u[i]);
   mse /= (double)clean.size();
   return mse == 0.0 ? 1e9 : 10.0 * std::log10(1.0 / mse);
}

int main(int argc, char** argv) {
   // ---- 1. command line -------------------------------------------------------
   std::string data = "../images/cameraman.png", save_file, comp = "scholtes";
   std::string sff_solver = "cg";
   double bound_push = -1.0;   // < 0: IPOPT's default
   int N = 32, nsub = 4;
   double sigma = 0.1;
   unsigned seed = 0;
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; return 2; }
      const std::string v = argv[++i];
      if (a == "--data") data = v;
      else if (a == "--size") N = std::stoi(v);
      else if (a == "--nsub") nsub = std::stoi(v);
      else if (a == "--sigma") sigma = std::stod(v);
      else if (a == "--seed") seed = (unsigned)std::stoul(v);
      else if (a == "--comp") comp = v;
      else if (a == "--save-solution") save_file = v;
      else if (a == "--sff-solver") sff_solver = v;
      else if (a == "--bound-push") bound_push = std::stod(v);
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (nsub < 1 || nsub > N - 1) { std::cerr << "need 1 <= nsub <= size-1\n"; return 2; }
   if (comp != "scholtes" && comp != "fb") { std::cerr << "--comp must be scholtes|fb\n"; return 2; }
   const int sff_direct = sff_solver == "cg" ? 0 : sff_solver == "direct" ? 1
                        : sff_solver == "direct-all" ? 2 : -1;
   if (sff_direct < 0) { std::cerr << "--sff-solver must be cg|direct|direct-all\n"; return 2; }

   // ---- 2. the instance -------------------------------------------------------
   const Image img = load_image(data, N, sigma, seed);
   const Partition part(N, nsub);
   SmartPtr<ConsensusTV> problem = new ConsensusTV(
      img, part, sigma,
      comp == "fb" ? ConsensusTV::Comp::FischerBurmeister : ConsensusTV::Comp::Scholtes);
   std::printf("N=%d  %dx%d tiles  variables=%d  constraints=%d  copies=%d  comp=%s"
               "  sff-solver=%s\n",
               N, nsub, nsub, problem->n, problem->mcon, problem->n_link, comp.c_str(),
               sff_solver.c_str());

   // ---- 3. IPOPT --------------------------------------------------------------
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   OptionsList& opt = *app->Options();
   opt.SetStringValue("sb", "yes");
   opt.SetIntegerValue("print_level", 0);
   opt.SetIntegerValue("max_iter", 3000);
   opt.SetIntegerValue("acceptable_iter", 1);
   opt.SetStringValue("mu_strategy", "monotone");   // the μ-coupling needs monotone μ
   opt.SetStringValue("hessian_approximation", "exact");
   opt.SetNumericValue("acceptable_tol", 1e-2);
   opt.SetNumericValue("acceptable_dual_inf_tol", 1e2);
   if (app->Initialize() != Solve_Succeeded) { std::cerr << "IPOPT init failed\n"; return 1; }
   // Barrier schedule tuned for the consensus form: advance μ earlier (gate 1000)
   // and in gentler steps, since t — and so complementarity — follows μ.
   opt.SetNumericValue("barrier_tol_factor", 1000.0);
   opt.SetNumericValue("mu_linear_decrease_factor", 0.7);
   opt.SetNumericValue("mu_superlinear_decrease_power", 1.1);
   if (bound_push > 0.0) {
      opt.SetNumericValue("bound_push", bound_push);
      opt.SetNumericValue("bound_frac", bound_push);
   }

   // ---- 4. μ-coupled continuation (the callback in problem.hpp drives it) -----
   const double t_min = 1e-4, tol = 1e-8, mu0 = 0.1;
   problem->t_min = t_min;
   problem->t = std::max(t_min, problem->t_mu_scale * mu0);   // = 1: IPOPT's first μ
   problem->eps_theta = problem->c_theta * problem->t;
   problem->gate_floor = tol;
   opt.SetNumericValue("tol", std::max(tol, 0.1 * t_min));
   opt.SetNumericValue("acceptable_tol", std::max(tol, t_min));

   // ---- 5. plug in the domain-decomposition linear solver ---------------------
   dd::Arrowhead::Options dd_opt;
   dd_opt.alpha_index = problem->oa;
   dd_opt.cg_tol = 1e-10;
   dd_opt.peel_cg_tol = 1e-7;
   dd_opt.cg_maxit = 100000;
   dd_opt.sff_direct = sff_direct;
   DDLinearSolver::configure(problem->kkt_owner(), part.n_tiles, dd_opt);
   problem->linear_stats = [] {
      const auto& s = DDLinearSolver::stats();
      return ConsensusTV::LinearStats{s.attempts, s.wrong_inertia, s.solves, s.rejected,
                                      s.iters};
   };

   // ---- 6. solve --------------------------------------------------------------
   const auto t0 = std::chrono::steady_clock::now();
   SmartPtr<AlgorithmBuilder> builder = new DDSolverBuilder();
   const ApplicationReturnStatus status =
      app->OptimizeNLP(new TNLPAdapter(GetRawPtr(problem)), builder);
   const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

   // ---- 7. report -------------------------------------------------------------
   // Status 5 (User_Requested_Stop) is the level gate firing: a normal finish.
   const bool ok = status == Solve_Succeeded || status == Solved_To_Acceptable_Level ||
                   (status == User_Requested_Stop && problem->gate_fired);
   const int iters = IsValid(app->Statistics()) ? app->Statistics()->IterationCount() : -1;
   const auto& st = DDLinearSolver::stats();
   std::printf("\nIPOPT status %d (%s)  iterations %d  t=%.2e  wall %.2fs\n", (int)status,
               ok ? "ok" : "FAILED", iters, problem->t, wall);
   std::printf("interface CG: solves=%ld  iterations=%ld  rejected=%ld  factorizations=%ld"
               "  (attempts=%ld  wrong inertia=%ld)\n",
               st.solves, st.iters, st.rejected, st.factorizations, st.attempts,
               st.wrong_inertia);
   std::printf("linear solver wall: factorize %.2fs (peel %.2fs)  solve %.2fs\n",
               st.t_factorize, st.t_peel, st.t_solve);

   // The summary row describes the FINAL iterate, whatever happened.
   const std::vector<double>& xf = problem->solution;
   Level level{problem->t, (int)status, iters, 0.0, 0.0, problem->objective, 0.0, ok};
   if (!xf.empty()) {
      level.comp_res = problem->max_complementarity(xf.data());
      level.weight = xf[problem->oa];
   }
   for (int e = 0; e < problem->m_q && !problem->multipliers.empty(); ++e)
      level.xi_max = std::max(level.xi_max, std::abs(problem->multipliers[problem->rcomp + e]));

   // The reported answer: the final iterate if the solve succeeded, otherwise
   // the tightest level banked on the way, otherwise nothing.
   std::vector<double> x;
   double t_best = 0.0;
   if (ok) {
      x = xf;
      t_best = problem->t;
   } else if (problem->have_banked) {
      std::printf("IPOPT failed: reporting the deepest banked level t=%.3e (max r(1-d) %.3e)\n",
                  problem->banked_t, problem->banked_comp);
      x = problem->banked_x;
      t_best = problem->banked_t;
   } else {
      std::printf("IPOPT failed and no level was banked: nothing to report\n");
      return 1;
   }

   std::printf("alpha*      %.6f\n", x[problem->oa]);
   std::printf("PSNR noisy  %.2f dB\n", psnr(img.clean, img.noisy.data()));
   std::printf("PSNR recon  %.2f dB\n", psnr(img.clean, x.data() + problem->ou));
   if (!save_file.empty()) {
      save_solution(save_file, *problem, img, level, x, t_best, nsub, sigma);
      std::printf("wrote %s\n", save_file.c_str());
   }
   return 0;
}
