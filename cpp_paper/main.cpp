// tv_dd — learn the TV-denoising weight α of one image by solving the lifted
// MPCC (tv_mpcc.hpp) with a primal-dual interior point method (ipm.hpp) whose
// Newton systems are solved by Schur-complement domain decomposition
// (schur_dd.hpp).  No IPOPT, no HSL: Eigen, LAPACK and zlib only.
//
//   ./tv_dd --size 32 --nsub 4
//   ./tv_dd --data ../images/mariposa.png --size 48 --nsub 4 --save-solution run.npz
//   ./tv_dd --size 32 --nsub 4 --schur direct        exact interface solve (check)
//
// Fixed choices (see README.md for the measurements behind them):
//   formulation   consensus form, Scholtes relaxation r(1−δ) ≤ t
//   continuation  t = max(t_min, 10·μ), at most halving per iteration; level gate
//   IPM           fraction-to-the-boundary steps, one global δ, δ_c on rank loss,
//                 μ schedule κ_ε = 1000, κ_μ = 0.7, θ_μ = 1.1, least-squares λ₀
//   tiles         MUMPS on [W_k B_kᵀ; B_k 0] when built with it, else the sparse
//                 level-ordered LDLᵀ with a dense Bunch–Kaufman fallback
//   interface     PCG with additive Schwarz (S̃_k = N_kᵀ S N_k), tol 1e-10
#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "comm.hpp"
#include "image.hpp"
#include "ipm.hpp"
#include "partition.hpp"
#include "save.hpp"
#include "tv_mpcc.hpp"

static double psnr(const std::vector<double>& clean, const double* u) {
   double mse = 0.0;
   for (size_t i = 0; i < clean.size(); ++i) mse += (clean[i] - u[i]) * (clean[i] - u[i]);
   mse /= (double)clean.size();
   return mse == 0.0 ? 1e9 : 10.0 * std::log10(1.0 / mse);
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

static void usage() {
   std::printf(
      "usage: tv_dd [options]\n"
      "  instance   --data ../images/cameraman.png  --size 32  --nsub 4  --sigma 0.1  --seed 0\n"
      "  relaxation --t-min 1e-4\n"
      "  solver     --schur pcg|direct  --cg-tol 1e-10  --cg-maxit 2000\n"
      "             --precond as|asd (additive Schwarz on S~_k = N_k' S N_k, eq. 19, or on\n"
      "               S_k with its diagonal replaced by diag(S), eq. 21)\n"
      "             --block-solver mumps|sparse|dense  --fallback mumps|dense\n"
      "             (defaults: mumps/mumps if built with MUMPS, else sparse/dense)\n"
      "             --max-iter 3000  --threads n  --line-search filter|none  --restoration on|off\n"
      "             --stall-iter K (0 = off)  --alpha-y primal|bound-mult|full|min-dual-infeas\n"
      "             --delta-start zero|last (each iteration tries delta = 0 first, IPOPT; or,\n"
      "               after an iteration that needed delta > 0, delta_last/3)\n"
      "             --kappa-sigma 1e10  --bounds rows|vars\n"
      "             --kappa-eps 1000 (μ decreases while E_μ <= κ_ε·μ)  --mu-steps S (at most\n"
      "               S decreases per iteration; 0 = no limit)\n"
      "             --t-mu-scale 10 (t = max(t_min, scale*mu); Raghunathan-Biegler: 1)\n"
      "             --mu-min M (floor of mu; default t_min/t-mu-scale, so mu/t >= 1/scale\n"
      "               at t_min; 0: the IPM's tol/10)  --resto-mu-steps S (at most S decreases\n"
      "               of mu right after a restoration phase; default 1, 0: no cap)\n"
      "             --t-rate 0.5 (t falls by at most this factor per iteration; 0: no limit)\n"
      "             --t-comp-ratio K (> 0: t never below max r(1-d)/K; 0: off)\n"
      "             --vw-mu M (> 0: their modified step, eq. 3.7 with 3.8(ii), once mu <= M;\n"
      "               IPOPT-C: 5e-6)  --vw-bounds all|comp (every bound, as in the paper,\n"
      "               or only r >= 0, 1-delta >= 0)  --vw-linesearch modified|plain (the\n"
      "               line search's barrier gradient follows the modified step, IPOPT-C)\n"
      "             --t-update every|solved (t follows 10·μ at every iteration, or only\n"
      "               when the barrier problem is solved, E_mu <= kappa_eps·mu)\n"
      "             --dual-reg 1e-6 (δ_c = C·μ^K every iteration; 0: only on demand)\n"
      "             --dual-reg-exp 0  --relax-cells THR (0 = off)  --relax-shift 1e-4\n"
      "             --relax-persist 5  --relax-exp 0.7 (threshold max(THR, inf_du^exp))\n"
      "             --classify FILE|- (MPCC multipliers per end-game iteration; FILE: the\n"
      "               last point per cell as CSV r,w,xi,a,b,gamma,nu,lam_h2, w = 1-delta)\n"
      "  penalty    --penalty PI0 (> 0: exact-penalty formulation pi*sum r(1-delta) in the\n"
      "               objective instead of the relaxed rows; pi grows x10 when needed)\n"
      "             --penalty-max 1e8 (cap on pi; = PI0: pi fixed)  --penalty-hessian on|off\n"
      "               (off: leave the penalty's indefinite r-delta term out of the Hessian)\n"
      "  clean-up   --cleanup off|gate|reach (re-solve with the active set pinned, the\n"
      "               products dropped: after the level gate, or as soon as t = t_min)\n"
      "             --cleanup-dual-reg C (δ_c there; default: as --dual-reg)  --cleanup-mu M\n"
      "               (its starting μ; default: the continuation's last μ)\n"
      "             --cleanup-biactive smaller|both|none (cells with r, 1-delta <= eps: pin\n"
      "               the smaller / both / neither)  --cleanup-eps E (default sqrt(t_min))\n"
      "  output     --save-solution FILE.npz  --quiet  --verbose 0|1|2  --diag\n");
}

static int run(int argc, char** argv);

int main(int argc, char** argv) {
#ifdef DD_HAVE_MPI
   MPI_Init(&argc, &argv);
#endif
   const int code = run(argc, argv);
#ifdef DD_HAVE_MPI
   MPI_Finalize();
#endif
   return code;
}

static int run(int argc, char** argv) {
   const bool root = dd::Comm::root();
   const int ranks = dd::Comm::size();
   // one BLAS thread inside each tile; the parallelism is over tiles
   setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
   setenv("OPENBLAS_NUM_THREADS", "1", 0);
   Eigen::setNbThreads(1);

   std::string data = "../images/cameraman.png", save_file;
   int N = 32, nsub = 4, max_iter = 3000, threads = 0, stall_iter = 0;
   double kappa_sigma = 1e10, dual_reg = 1e-6, dual_reg_exp = 0.0;
   double kappa_eps = 1000.0;
   int mu_steps = 0, resto_mu_steps = 1;
   double mu_min = -1.0;   // < 0: t_min / t_mu_scale
   bool t_on_solved = false;
   std::string cleanup = "off";
   double cleanup_dual_reg = -1.0, cleanup_mu = 0.0;   // < 0 / 0: as in the continuation
   int cleanup_biactive = 0;
   double penalty = 0.0, penalty_max = 1e8;
   double vw_mu = 0.0, t_mu_scale = 10.0, t_rate = 0.5, t_comp_ratio = 0.0;
   bool vw_comp_only = false, vw_ls_grad = true;
   bool penalty_hessian = true;
   double cleanup_eps = 0.0;
   double relax_thr = 0.0, relax_shift = 1e-4;
   int relax_persist = 5;
   double relax_exp = 0.7;
   double sigma = 0.1, t_min = 1e-4;
   unsigned seed = 0;
   bool quiet = false, line_search = true, restoration = true, diag = false, bounds_rows = true;
   bool delta_from_last = false;
   bool classify = false;
   std::string classify_file;
   dd::SchurDD::Options dopt;
   IPM::Options::AlphaY alpha_y = IPM::Options::AlphaY::Primal;
#ifdef DD_HAVE_MUMPS
   dopt.blocks = dd::Backend::Mumps;     // pivoted sparse tiles: fastest and smallest
   dopt.fallback = dd::Backend::Mumps;   // (used by --block-solver sparse)
#endif
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--quiet") { quiet = true; continue; }
      if (a == "--diag") { diag = true; continue; }
      if (a == "--help" || a == "-h") { if (root) usage(); return 0; }
      if (i + 1 >= argc) { if (root) usage(); return 1; }
      const std::string v = argv[++i];
      if (a == "--data") data = v;
      else if (a == "--size") N = std::stoi(v);
      else if (a == "--nsub") nsub = std::stoi(v);
      else if (a == "--sigma") sigma = std::stod(v);
      else if (a == "--seed") seed = (unsigned)std::stoul(v);
      else if (a == "--t-min") t_min = std::stod(v);
      else if (a == "--schur")
         dopt.interface = v == "direct" ? dd::SchurDD::Interface::Direct : dd::SchurDD::Interface::Pcg;
      else if (a == "--cg-tol") dopt.cg_tol = std::stod(v);
      else if (a == "--precond")
         dopt.precond = v == "asd" ? dd::SchurDD::Precond::ASd : dd::SchurDD::Precond::AS;
      else if (a == "--cg-maxit") dopt.cg_maxit = std::stoi(v);
      else if (a == "--block-solver" || a == "--fallback") {
         const dd::Backend b = v == "dense" ? dd::Backend::Dense
                             : v == "mumps" ? dd::Backend::Mumps
                                            : dd::Backend::Sparse;
         (a == "--block-solver" ? dopt.blocks : dopt.fallback) = b;
      }
      else if (a == "--max-iter") max_iter = std::stoi(v);
      else if (a == "--line-search") line_search = v != "none";
      else if (a == "--restoration") restoration = v != "off";
      else if (a == "--stall-iter") stall_iter = std::stoi(v);
      else if (a == "--delta-start") delta_from_last = v == "last";
      else if (a == "--kappa-sigma") kappa_sigma = std::stod(v);
      else if (a == "--kappa-eps") kappa_eps = std::stod(v);
      else if (a == "--mu-steps") mu_steps = std::stoi(v);
      else if (a == "--mu-min") mu_min = std::stod(v);
      else if (a == "--resto-mu-steps") resto_mu_steps = std::stoi(v);
      else if (a == "--t-update") t_on_solved = v == "solved";
      else if (a == "--cleanup") cleanup = v;
      else if (a == "--penalty") penalty = std::stod(v);
      else if (a == "--vw-mu") vw_mu = std::stod(v);
      else if (a == "--vw-bounds") vw_comp_only = v == "comp";
      else if (a == "--vw-linesearch") vw_ls_grad = v != "plain";
      else if (a == "--t-mu-scale") t_mu_scale = std::stod(v);
      else if (a == "--t-rate") t_rate = std::stod(v);
      else if (a == "--t-comp-ratio") t_comp_ratio = std::stod(v);
      else if (a == "--penalty-max") penalty_max = std::stod(v);
      else if (a == "--penalty-hessian") penalty_hessian = v != "off";
      else if (a == "--cleanup-dual-reg") cleanup_dual_reg = std::stod(v);
      else if (a == "--cleanup-mu") cleanup_mu = std::stod(v);
      else if (a == "--cleanup-biactive") cleanup_biactive = v == "both" ? 1 : v == "none" ? 2 : 0;
      else if (a == "--cleanup-eps") cleanup_eps = std::stod(v);
      else if (a == "--classify") { classify = true; classify_file = v == "-" ? "" : v; }
      else if (a == "--dual-reg") dual_reg = std::stod(v);
      else if (a == "--dual-reg-exp") dual_reg_exp = std::stod(v);
      else if (a == "--relax-cells") relax_thr = std::stod(v);
      else if (a == "--relax-shift") relax_shift = std::stod(v);
      else if (a == "--relax-persist") relax_persist = std::stoi(v);
      else if (a == "--relax-exp") relax_exp = std::stod(v);
      else if (a == "--bounds") bounds_rows = v != "vars";
      else if (a == "--alpha-y")
         alpha_y = v == "bound-mult" ? IPM::Options::AlphaY::BoundMult
                 : v == "full"       ? IPM::Options::AlphaY::Full
                 : v == "min-dual-infeas" ? IPM::Options::AlphaY::MinDualInfeas
                                          : IPM::Options::AlphaY::Primal;
      else if (a == "--threads") threads = std::stoi(v);
      else if (a == "--verbose") dopt.verbose = std::stoi(v);
      else if (a == "--save-solution") save_file = v;
      else { if (root) usage(); return 1; }
   }
   if (nsub < 1 || nsub > N - 1) { std::fprintf(stderr, "need 1 <= nsub <= size-1\n"); return 1; }
   if (dopt.fallback == dd::Backend::Sparse) dopt.fallback = dd::Backend::Dense;
   const bool use_mumps = dopt.blocks == dd::Backend::Mumps ||
                          (dopt.blocks == dd::Backend::Sparse && dopt.fallback == dd::Backend::Mumps);
#ifdef DD_HAVE_MUMPS
   if (use_mumps && !dd::MumpsBlock::self_test()) {
      std::fprintf(stderr, "MUMPS self test failed (Schur layout or inertia): run ./mumps_check\n");
      return 1;
   }
#else
   if (use_mumps) {
      std::fprintf(stderr, "this tv_dd was built without MUMPS (see build.sh)\n");
      return 1;
   }
#endif
#ifdef _OPENMP
   if (threads > 0) omp_set_num_threads(threads);
   const int nthreads = omp_get_max_threads();
#else
   const int nthreads = 1;
#endif

   // ---- the instance
   const Image img = load_image(data, N, sigma, seed);
   const Partition part(N, nsub);
   TvMpcc problem(img, part, sigma);
   problem.print = root;
   problem.diag = diag;
   problem.classify = classify;
   problem.relax_cells = relax_thr > 0.0;
   problem.relax_thr = relax_thr;
   problem.relax_shift = relax_shift;
   problem.relax_persist = relax_persist;
   problem.relax_exp = relax_exp;
   problem.bounds_as_rows = bounds_rows;
   problem.penalty = penalty;
   problem.penalty_max = penalty_max;
   problem.penalty_hessian = penalty_hessian;
   problem.t_on_solved = t_on_solved;
   const char* bname[] = {"sparse", "dense", "mumps"};
   const std::string blocks = dopt.blocks == dd::Backend::Sparse
                                 ? std::string("sparse+") + bname[(int)dopt.fallback]
                                 : bname[(int)dopt.blocks];
   if (root)
      std::printf("tv_dd  N=%d  %dx%d tiles  variables=%d  constraints=%d (%d inequalities)"
                  "  copies=%d  interface=%s  blocks=%s  ranks=%d  threads/rank=%d\n",
                  N, nsub, nsub, problem.n, problem.mcon, problem.n_ineq, problem.n_link,
                  dopt.interface == dd::SchurDD::Interface::Pcg ? "pcg" : "direct",
                  blocks.c_str(), ranks, nthreads);

   // ---- μ-coupled continuation
   const double tol_target = 1e-8, mu0 = 0.1;
   problem.t_min = t_min;
   problem.t_mu_scale = t_mu_scale;
   problem.t_rate = t_rate;
   problem.t_comp_ratio = t_comp_ratio;
   problem.t = std::max(t_min, problem.t_mu_scale * mu0);
   problem.eps_theta = problem.c_theta * problem.t;
   problem.gate_floor = tol_target;

   // ---- the IPM
   IPM ipm;
   ipm.opt.tol = std::max(tol_target, 0.1 * t_min);
   ipm.opt.max_iter = max_iter;
   ipm.opt.mu0 = mu0;
   ipm.opt.kappa_eps = kappa_eps;
   ipm.opt.mu_max_steps = mu_steps;
   // μ's floor keeps the continuation's coupling t = scale·μ at t_min (μ/t ≥ 1/scale);
   // the IPM's own default, tol/10, let μ/t fall to 0.01 (mariposa N=640)
   ipm.opt.mu_min = mu_min >= 0.0 ? mu_min : t_min / t_mu_scale;
   ipm.opt.resto_mu_steps = resto_mu_steps;
   ipm.opt.vw_mu = vw_mu;
   ipm.opt.vw_comp_only = vw_comp_only;
   ipm.opt.vw_ls_grad = vw_ls_grad;
   ipm.opt.kappa_mu = 0.7;
   ipm.opt.theta_mu = 1.1;
   ipm.opt.acceptable_tol = std::max(tol_target, t_min);
   ipm.opt.acceptable_dual = 1e2;
   ipm.opt.acceptable_iter = 1;
   ipm.opt.print_level = quiet || !root ? 0 : 1;
   ipm.opt.line_search = line_search;
   ipm.opt.restoration = restoration;
   ipm.opt.stall_iter = stall_iter;
   ipm.opt.delta_from_last = delta_from_last;
   ipm.opt.alpha_y = alpha_y;
   ipm.opt.kappa_sigma = kappa_sigma;
   ipm.opt.dual_reg = dual_reg;
   ipm.opt.dual_reg_exp = dual_reg_exp;
   dd::SchurDD::Stats s;
   NLPPoint final_pt;
   if (cleanup != "off") ipm.export_point = &final_pt;
   problem.stop_at_tmin = cleanup == "reach";
   problem.cleanup_biactive = cleanup_biactive;
   problem.cleanup_eps = cleanup_eps;
   IPM::Result r = ipm.solve(problem, dopt, &s);

   // ---- the active-set clean-up: a re-solve with the active set pinned
   IPM::Result rc;
   bool cleaned = false;
   const int main_iters = r.iters;
   if (cleanup != "off" && r.status == 2 && (problem.gate_fired || problem.reached_tmin)) {
      const std::array<long, 4> pins = problem.begin_cleanup(final_pt);
      if (root)
         std::printf("\nclean-up: cells pinned at r = 0: %ld, at delta = 1: %ld, both: %ld,"
                     " neither: %ld; products dropped\n", pins[0], pins[1], pins[2], pins[3]);
      const std::vector<double> sol = problem.solution, mult = problem.multipliers;
      const double obj = problem.objective;
      IPM ipm2;
      ipm2.opt = ipm.opt;
      ipm2.opt.ls_mult_init = false;
      ipm2.opt.bound_push = 1e-8;
      if (cleanup_dual_reg >= 0.0) ipm2.opt.dual_reg = cleanup_dual_reg;
      if (cleanup_mu > 0.0) final_pt.mu = cleanup_mu;
      ipm2.warm = &final_pt;
      NLPPoint clean_pt;
      ipm2.export_point = &clean_pt;
      rc = ipm2.solve(problem, dopt, &s);
      cleaned = rc.status == 0 || rc.status == 3;
      const TvMpcc::CleanupReport rep = problem.cleanup_report(clean_pt, 1e-4, 1e-6);
      if (root)
         std::printf("clean-up: status %d, %d iterations;  max pinned-row violation %.1e,"
                     " max r*w %.1e;  biactive cells (r, w <= 1e-4): %ld, of which with an MPCC"
                     " multiplier < -1e-6: %ld (min %.2e)\n",
                     rc.status, rc.iters, rep.max_pin, rep.max_rw, rep.biactive, rep.wrong,
                     rep.min_mult);
      if (!cleaned) {   // keep the continuation's answer
         problem.solution = sol;
         problem.multipliers = mult;
         problem.objective = obj;
         if (root) std::printf("clean-up failed: reporting the continuation's point\n");
      }
      r.iters += rc.iters;
      r.wall += rc.wall;
      if (cleaned) r.status = rc.status;
   }

   const bool ok = cleaned || r.status == 0 || r.status == 3 || (r.status == 2 && problem.gate_fired);
   const char* st = r.status == 0 ? "converged" : r.status == 3 ? "acceptable"
                  : r.status == 2 ? (problem.gate_fired ? "level gate" : "stopped")
                  : r.status == 1 ? "iteration limit" : "FAILED";
   const std::vector<double>& x = problem.solution;
   const double alpha = x.empty() ? NAN : x[problem.oa];
   const double comp = x.empty() ? NAN : problem.max_complementarity(x.data());
   const double p_noisy = psnr(img.clean, img.noisy.data());
   const double p_recon = x.empty() ? NAN : psnr(img.clean, x.data() + problem.ou);
   double xi_max = 0.0;
   for (int e = 0; e < problem.m_q && !problem.multipliers.empty(); ++e)
      xi_max = std::max(xi_max, std::abs(problem.multipliers[problem.rcomp + e]));
   const double cg_mean = s.solves ? (double)s.cg_iters / s.solves : 0.0;

   // timings and memory: the slowest rank (the others wait for it)
   const double wall = dd::Comm::max(r.wall), rss = dd::Comm::max(peak_rss_mb());
   const double tf = dd::Comm::max(s.t_factor), tb = dd::Comm::max(s.t_blocks);
   const double ts = dd::Comm::max(s.t_sdirect), tp = dd::Comm::max(s.t_precond);
   const double tsol = dd::Comm::max(s.t_solve), tts = dd::Comm::max(s.t_tile_solves);
   const double ti = dd::Comm::max(s.t_interface);
   if (!root) return ok ? 0 : 2;

   std::printf("\nstatus %d (%s)   iterations %d   t=%.2e   wall %.2fs   peak RSS %.0f MB/rank\n",
               r.status, st, r.iters, problem.t, wall, rss);
   if (penalty > 0.0)
      std::printf("penalty: pi %.0e -> %.0e (%ld raises)\n", penalty, problem.penalty,
                  problem.penalty_raises);
   if (cleanup != "off")
      std::printf("clean-up (%s): %s, %d + %d iterations (continuation + clean-up)\n",
                  cleanup.c_str(), rc.iters == 0 ? "not run" : cleaned ? "converged" : "FAILED",
                  main_iters, rc.iters);
   std::printf("refused factorizations: wrong In(W_k) %ld (too few negatives: %ld), S not PD %ld,"
               " singular %ld;  iterations with δ > 0: %ld\n",
               r.tile_corrections, r.tile_too_few, r.s_corrections, r.singular, r.regularized);
   if (line_search)
      std::printf("line search: %ld step halvings, %ld iterations without an acceptable step;"
                  "  restoration: %ld phases (%ld for a stall, %ld failed), %ld iterations\n",
                  r.ls_backtracks, r.ls_failures, r.restorations, r.stall_restorations,
                  r.resto_failures, r.resto_iters);
   if (problem.relax_cells)
      std::printf("relaxed cells: %ld (r >= -%.0e, delta <= 1+%.0e, r(1-delta) <= 0)\n",
                  problem.relaxed_total, relax_shift, relax_shift);
   std::printf("linear: nP=%d  KKT dim=%d  p=%d  max p_k=%d  max t_k=%d  factorizations %ld"
               " (sparse tile factorizations that fell back: %ld)\n",
               s.n_tiles, s.dim, s.p, s.max_pk, s.max_tk, s.factorizations, s.fallbacks);
   if (dopt.interface == dd::SchurDD::Interface::Pcg)
      std::printf("        pcg: solves %ld, iterations mean %.1f max %ld, negative curvature %ld,"
                  " unconverged %ld;  refused before PCG (some S~_k not SPD) %ld\n",
                  s.solves, cg_mean, s.cg_max, s.cg_breakdowns, s.cg_unconverged,
                  s.s_tilde_indefinite);
   std::printf("time:   factorize %.2fs [blocks %.2f, S %.2f, preconditioner %.2f]  solve %.2fs"
               " [tiles %.2f, interface %.2f]  rest %.2fs   (max over ranks)\n",
               tf, tb, ts, tp, tsol, tts, ti, wall - tf - tsol);
   std::printf("alpha*      %.6f\n", alpha);
   std::printf("max r(1-d)  %.2e\n", comp);
   std::printf("PSNR noisy  %.2f dB\n", p_noisy);
   std::printf("PSNR recon  %.2f dB\n", p_recon);
   // image,N,nsub,sigma,seed,t_min,interface,blocks,ranks,threads,status,ok,iters,t,alpha,comp,
   // psnr_noisy,psnr_recon,wall,t_factor,t_solve,factorizations,fallbacks,cg_mean,cg_max,rss_mb
   std::printf("CSV,%s,%d,%d,%g,%u,%g,%s,%s,%d,%d,%d,%d,%d,%.3e,%.6f,%.3e,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%ld,%ld,%.1f,%ld,%.0f\n",
               data.c_str(), N, nsub, sigma, seed, t_min,
               dopt.interface == dd::SchurDD::Interface::Pcg ? "pcg" : "direct", blocks.c_str(),
               ranks, nthreads,
               r.status, ok ? 1 : 0, r.iters, problem.t, alpha, comp, p_noisy, p_recon, wall,
               tf, tsol, s.factorizations, s.fallbacks, cg_mean, s.cg_max, rss);

   if (!save_file.empty() && !x.empty()) {
      const RunSummary run{problem.t, r.status, r.iters, comp, alpha, problem.objective,
                           xi_max, ok};
      if (save_npz(save_file, problem, img, run, nsub, sigma))
         std::printf("wrote %s\n", save_file.c_str());
   }
   if (root && !classify_file.empty() && !problem.mpcc_cells.empty()) {
      if (FILE* fp = std::fopen(classify_file.c_str(), "w")) {
         std::fprintf(fp, "cell,r,w,xi,a,b,gamma,nu,lam_h2\n");
         const double* c = problem.mpcc_cells.data();
         for (int e = 0; e < problem.m_q; ++e, c += 8)
            std::fprintf(fp, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n", e, c[0], c[1], c[2],
                         c[3], c[4], c[5], c[6], c[7]);
         std::fclose(fp);
         std::printf("wrote %s\n", classify_file.c_str());
      }
   }
   return ok ? 0 : 2;
}
