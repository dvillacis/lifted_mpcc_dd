// tv_learn — learn a TV weight (a scalar, or a map on B×B blocks) over a set
// of training images, for denoising, inpainting or deblurring, with TV or
// directional TV (tv_learn.hpp, operators.hpp), by the same IPM and
// Schur-complement domain decomposition as tv_dd.
//
//   ./tv_learn --size 32 --nsub 4                              tv_dd's problem, bit for bit
//   ./tv_learn --data ../images/mariposa.png --size 48 --block 8
//   ./tv_learn --size 64 --nsub 4 --op deblur --blur-sigma 1
//   ./tv_learn --size 64 --nsub 4 --op inpaint --mask-frac 0.5 --reg dtv
//   ./tv_learn --train kodak/train.txt --val kodak/val.txt --size 64 --nsub 2 \
//              --save-report run.json --save-solution sols/sol_run
//
// Held-out images (--val) are evaluated by solving their lower level at the
// learned map (pdhg.hpp); so are the training images, as a check that the
// MPCC's u is the lower-level solution at its own weight.
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
#include "dataset.hpp"
#include "ipm.hpp"
#include "learn_partition.hpp"
#include "learn_util.hpp"
#include "operators.hpp"
#include "pdhg.hpp"
#include "save_learn.hpp"
#include "smoothed.hpp"
#include "tv_learn.hpp"
#include "ui.hpp"

// held-out image i is seeded seed + kValSeed + i, so that it never shares the
// noise (or the mask) of training image i
static constexpr unsigned kValSeed = 1000000;

static void usage() {
   std::printf(
      "usage: tv_learn [options]\n"
      "  data       --data IMAGE (one training image; default ../images/cameraman.png)\n"
      "             --train DIR|LIST.txt|IMAGE  --val DIR|LIST.txt|IMAGE (held out)\n"
      "             --skip 0 --stride 1 --limit 0 (training selection; 0 = all)  --val-limit 0\n"
      "             --size 32  --nsub 4 (tiles per side, per image)  --sigma 0.1\n"
      "             --seed 0 (training image i: seed + i, held-out: seed + 1e6 + i;\n"
      "               --same-seed: seed for all)\n"
      "             --loss mean|sum\n"
      "  weight     --block B (0: one scalar alpha; B: alpha constant on BxB blocks of cells,\n"
      "               shared by all images; tile cuts snap to multiples of B)\n"
      "             --start smoothed|scalar|uniform (smoothed, the default: two stages, the smoothed\n"
      "               bilevel method first (--start-gamma 1e-3, --start-iters 1000 (a safeguard), --start-ftol\n"
      "               1e-9: stop once J falls by less than this, relative, over 5 iterations;\n"
      "               --start-solver auto|eigen|mumps|pcg-jacobi|pcg-ic: its Newton systems\n"
      "               (pcg: inexact Newton-CG); auto: MUMPS\n"
      "               for N >= 128, Eigen below;\n"
      "               --start-gamma0 G (0: off): continuation, gamma from G down to\n"
      "               --start-gamma, times --start-gamma-rate 0.1 once |proj grad| <=\n"
      "               --start-gamma-sigma 0.1 (gamma/G) x the first;\n"
      "               --start-newton primal|pd: its lower-level Newton, on the primal Huber\n"
      "               energy or primal-dual semismooth (De los Reyes-Schoenlieb-Valkonen))\n"
      "               then the MPCC from its weight map and its lower-level solutions, lifted;\n"
      "               scalar: stage 1 on one weight, which then starts every block;\n"
      "               interior: no stage 1; the MPCC from u = f, q = 0, delta = 0, r = |Ku|,\n"
      "               theta = the angle of Ku and the weight --alpha0 (no solve before it);\n"
      "               ipm: stage 1 is the smoothed problem q = h(Ku), h the C1 Huber clipping of\n"
      "               the dual (gamma: --start-gamma; --start-huber c2: the C2 Huber (3.2) of\n"
      "               De los Reyes et al., c1: Definition 1's, g/max(gamma, |g|), dual: that\n"
      "               Huber as rows max(gamma, |Ku|) q - Ku = 0, q its own unknown; dualproj:\n"
      "               those rows, q projected onto |q| <= 1 in their derivatives), solved by\n"
      "               this IPM from the uniform\n"
      "               start (--start-ipm-init pdhg: the exact TV solution at --alpha0, lifted;\n"
      "               smoothed: the smoothed lower level at --alpha0, q = h(Ku), feasible),\n"
      "               to --start-tol 1e-6, then the MPCC from its solution;\n"
      "               uniform: the MPCC from the weight --alpha0 everywhere, tv_dd's start)\n"
      "             --alpha0 A (the uniform start's weight, and stage 1's; default 0.7*sigma)\n"
      "             --mu0 M (the first barrier parameter, and t = 10 M; default 0.1)\n"
      "             --t0 T (the first relaxation t, instead of 10 mu0)\n"
      "             --bound-push K (IPOPT's bound_push = bound_frac; default 1e-2)\n"
      "             --alpha-init FILE (start from this map: one value per block, whitespace\n"
      "               separated)  --save-alpha FILE (write the learned block values)\n"
      "             --stage1-only (stop after stage 1, with its peak memory)\n"
      "             --color auto|always|never (auto: a terminal, unless NO_COLOR; plain under\n"
      "               mpirun, where --color always or CLICOLOR_FORCE=1 asks for colour)\n"
      "             --classify (the MPCC multipliers and the class S/M/C/W of the biactive cells\n"
      "               at the last iterate with t = t_min)\n"
      "             --warm-tol 1e-6  --warm-iters 20000 (PDHG warm start, when A != I or DTV)\n"
      "  operator   --op denoise|inpaint|deblur\n"
      "             --blur-sigma 1  --blur-radius w (default ceil(2 sigma_b); tiles >= 4w wide)\n"
      "             --mask-frac 0.5 (missing pixels)  --mask-eps 1e-3  --mask-seed 1000\n"
      "             --reg tv|dtv  --dtv-gamma 0.9  --dtv-eta 0.05  --dtv-smooth 1\n"
      "             --guide clean|data (DTV direction field of the training images;\n"
      "               held-out images always use their data)\n"
      "  relaxation --t-min 1e-5\n"
      "             --follow-comp 3 (a cell whose product row stays violated while its\n"
      "               multiplier rises, K iterations in a row, gets t_e = its product, then\n"
      "               lowered as the product falls; armed only after mu and t have not\n"
      "               moved for --follow-stall 20 iterations; 0: off)\n"
      "             --freeze-xi X (opt-in, replaces --follow-comp: a cell whose product\n"
      "               multiplier stays >= X for --freeze-persist 3 iterations keeps the t\n"
      "               of that moment)\n"
      "             --endgame-diag FILE (per iteration: where the dual residual sits; the\n"
      "               last iteration's per-cell record to FILE as CSV)\n"
      "  solver     as tv_dd: --schur pcg|direct  --cg-tol  --cg-maxit  --precond as|asd\n"
      "             --block-solver mumps|sparse|dense  --fallback mumps|dense  --max-iter 3000\n"
      "             --threads n  --line-search filter|none  --restoration on|off  --stall-iter K\n"
      "             --alpha-y ...  --delta-start zero|last  --kappa-sigma  --bounds rows|vars\n"
      "             --kappa-eps  --mu-steps  --t-mu-scale  --gate primal|kkt  --gate-scale\n"
      "             --mu-min  --resto-mu-steps  --t-rate  --t-comp-ratio  --vw-mu  --vw-bounds\n"
      "             --vw-linesearch  --t-update every|solved  --dual-reg  --dual-reg-exp\n"
      "             --max-soc K (second-order corrections per iteration; IPOPT: 4; 0 = off)\n"
      "  baselines  (one rank; no MPCC solve)\n"
      "             --grid A0:A1:N (N log-spaced scalar weights; each image's TV lower level solved\n"
      "               by PDHG; the best weight by the training loss, then the held-out images)\n"
      "             --smoothed G (the smoothed bilevel method: Huber-TV lower level with parameter\n"
      "               G by Newton, adjoint gradients, projected L-BFGS on alpha >= 0; --block as\n"
      "               for the MPCC)  --smoothed-tol 1e-6 (projected gradient, relative)\n"
      "               --smoothed-iters 100;  both report the exact (TV) lower level at the result\n"
      "  output     --save-solution FILE.npz|PREFIX (one image: FILE.npz; else PREFIX_s000.npz\n"
      "               ... for training, PREFIX_v000.npz ... for held-out images)\n"
      "             --save-report FILE.json  --tag NAME  --rof-iters 20000\n"
      "             --check-derivatives (finite differences at the start, then exit; 1 rank)\n"
      "             --quiet  --verbose 0|1|2\n");
}

static int run(int argc, char** argv);

static std::vector<double> read_alpha(const std::string& file) {
   std::vector<double> a;
   FILE* fp = std::fopen(file.c_str(), "r");
   if (!fp) throw std::runtime_error("cannot read " + file);
   double v;
   while (std::fscanf(fp, "%lf", &v) == 1) a.push_back(v);
   std::fclose(fp);
   return a;
}
static void write_alpha(const std::string& file, const std::vector<double>& a) {
   FILE* fp = std::fopen(file.c_str(), "w");
   if (!fp) { ui::error("cannot write %s", file.c_str()); return; }
   for (double v : a) std::fprintf(fp, "%.17g\n", v);
   std::fclose(fp);
   ui::ok("Wrote %s (%zu block values)", file.c_str(), a.size());
}

int main(int argc, char** argv) {
#ifdef DD_HAVE_MPI
   int provided = 0;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
   int code = 1;
   try {
      code = run(argc, argv);
   } catch (const std::exception& e) {
      ui::error("tv_learn: %s", e.what());
#ifdef DD_HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
   }
#ifdef DD_HAVE_MPI
   MPI_Finalize();
#endif
   return code;
}

static int run(int argc, char** argv) {
   const bool root = dd::Comm::root();
   const int ranks = dd::Comm::size();
   setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
   setenv("OPENBLAS_NUM_THREADS", "1", 0);
   Eigen::setNbThreads(1);

   std::string train_spec, val_spec, save_file, report_file, tag, loss = "mean";
   dataset::Select sel, vsel;
   int N = 32, nsub = 4, block = 0, max_iter = 3000, threads = 0, stall_iter = 0, rof_iters = 20000;
   int warm_max_iter = 20000;
   double warm_tol = 1e-6;
   double alpha0 = -1.0;
   OpConfig cfg;
   double kappa_sigma = 1e10, dual_reg = 1e-6, dual_reg_exp = 0.0, kappa_eps = 1000.0;
   int max_soc = 0;
   std::string grid_spec, alpha_init_file, save_alpha_file, start = "smoothed";
   double start_gamma = 1e-3, mu0 = 0.1, bound_push = 1e-2, start_ftol = 1e-9, start_tol = 1e-6;
   double start_gamma0 = 0.0, start_gamma_rate = 0.1, start_gamma_sigma = 0.1;   // stage 1's γ continuation
   std::string start_newton = "primal";   // stage 1's lower-level Newton: primal | pd
   std::string start_huber = "c2";        // --start ipm: the Huber of (3.2) (c2) or Definition 1 (c1)
   std::string start_ipm_init = "pdhg";   // --start ipm: its start, exact TV (pdhg) or smoothed
   double t0 = 0.0;   // > 0: the first relaxation t (else 10·μ0, as tv_dd)
   std::string start_solver = "auto", color_mode = "auto";
   int start_iters = 1000;
   bool classify = false, stage1_only = false;
   std::string diag_file;
   double freeze_xi = 0.0;
   int freeze_persist = 3;
   int follow_persist = -1, follow_stall = 20;   // -1: the default, 3 (0 with --freeze-xi)
   double smooth_gamma = 0.0, smooth_tol = 1e-6;
   int smooth_iters = 100;
   int mu_steps = 0, resto_mu_steps = 1;
   double mu_min = -1.0;
   bool t_on_solved = false;
   double vw_mu = 0.0, gate_du_scale = 1.0, t_mu_scale = 10.0, t_rate = 0.5, t_comp_ratio = 0.0;
   bool vw_comp_only = false, vw_ls_grad = true, gate_dual = false;
   double sigma = 0.1, t_min = 1e-5;
   unsigned seed = 0;
   bool quiet = false, line_search = true, restoration = true, bounds_rows = true;
   bool delta_from_last = false, check = false, same_seed = false;
   dd::SchurDD::Options dopt;
   IPM::Options::AlphaY alpha_y = IPM::Options::AlphaY::Primal;
#ifdef DD_HAVE_MUMPS
   dopt.blocks = dd::Backend::Mumps;
   dopt.fallback = dd::Backend::Mumps;
#endif
   for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--quiet") { quiet = true; continue; }
      if (a == "--check-derivatives") { check = true; continue; }
      if (a == "--same-seed") { same_seed = true; continue; }
      if (a == "--classify") { classify = true; continue; }
      if (a == "--stage1-only") { stage1_only = true; continue; }
      if (a == "--help" || a == "-h") { if (root) usage(); return 0; }
      if (i + 1 >= argc) { if (root) usage(); return 1; }
      const std::string v = argv[++i];
      if (a == "--data" || a == "--train") train_spec = v;
      else if (a == "--val") val_spec = v;
      else if (a == "--skip") sel.skip = std::stoi(v);
      else if (a == "--stride") sel.stride = std::stoi(v);
      else if (a == "--limit") sel.limit = std::stoi(v);
      else if (a == "--val-limit") vsel.limit = std::stoi(v);
      else if (a == "--size") N = std::stoi(v);
      else if (a == "--nsub") nsub = std::stoi(v);
      else if (a == "--sigma") sigma = std::stod(v);
      else if (a == "--seed") seed = (unsigned)std::stoul(v);
      else if (a == "--loss") loss = v;
      else if (a == "--block") block = std::stoi(v);
      else if (a == "--alpha0") alpha0 = std::stod(v);
      else if (a == "--op") cfg.op = v;
      else if (a == "--blur-sigma") cfg.blur_sigma = std::stod(v);
      else if (a == "--blur-radius") cfg.blur_radius = std::stoi(v);
      else if (a == "--mask-frac") cfg.mask_frac = std::stod(v);
      else if (a == "--mask-eps") cfg.mask_eps = std::stod(v);
      else if (a == "--mask-seed") cfg.mask_seed = (unsigned)std::stoul(v);
      else if (a == "--reg") cfg.dtv = v == "dtv";
      else if (a == "--dtv-gamma") cfg.dtv_gamma = std::stod(v);
      else if (a == "--dtv-eta") cfg.dtv_eta = std::stod(v);
      else if (a == "--dtv-smooth") cfg.dtv_smooth = std::stod(v);
      else if (a == "--guide") cfg.guide = v;
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
      else if (a == "--vw-mu") vw_mu = std::stod(v);
      else if (a == "--vw-bounds") vw_comp_only = v == "comp";
      else if (a == "--vw-linesearch") vw_ls_grad = v != "plain";
      else if (a == "--t-mu-scale") t_mu_scale = std::stod(v);
      else if (a == "--gate") gate_dual = v == "kkt";
      else if (a == "--gate-scale") gate_du_scale = std::stod(v);
      else if (a == "--t-rate") t_rate = std::stod(v);
      else if (a == "--t-comp-ratio") t_comp_ratio = std::stod(v);
      else if (a == "--endgame-diag") diag_file = v;
      else if (a == "--freeze-xi") freeze_xi = std::stod(v);
      else if (a == "--freeze-persist") freeze_persist = std::stoi(v);
      else if (a == "--follow-comp") follow_persist = std::stoi(v);
      else if (a == "--follow-stall") follow_stall = std::stoi(v);
      else if (a == "--dual-reg") dual_reg = std::stod(v);
      else if (a == "--dual-reg-exp") dual_reg_exp = std::stod(v);
      else if (a == "--bounds") bounds_rows = v != "vars";
      else if (a == "--max-soc") max_soc = std::stoi(v);
      else if (a == "--grid") grid_spec = v;
      else if (a == "--start") start = v;
      else if (a == "--mu0") mu0 = std::stod(v);
      else if (a == "--t0") t0 = std::stod(v);
      else if (a == "--bound-push") bound_push = std::stod(v);
      else if (a == "--start-gamma") start_gamma = std::stod(v);
      else if (a == "--start-ftol") start_ftol = std::stod(v);
      else if (a == "--start-gamma0") start_gamma0 = std::stod(v);
      else if (a == "--start-gamma-rate") start_gamma_rate = std::stod(v);
      else if (a == "--start-gamma-sigma") start_gamma_sigma = std::stod(v);
      else if (a == "--start-newton") start_newton = v;
      else if (a == "--start-huber") start_huber = v;
      else if (a == "--start-ipm-init") start_ipm_init = v;
      else if (a == "--start-tol") start_tol = std::stod(v);
      else if (a == "--start-solver") start_solver = v;
      else if (a == "--color") color_mode = v;
      else if (a == "--start-iters") start_iters = std::stoi(v);
      else if (a == "--alpha-init") alpha_init_file = v;
      else if (a == "--save-alpha") save_alpha_file = v;
      else if (a == "--smoothed") smooth_gamma = std::stod(v);
      else if (a == "--smoothed-tol") smooth_tol = std::stod(v);
      else if (a == "--smoothed-iters") smooth_iters = std::stoi(v);
      else if (a == "--alpha-y")
         alpha_y = v == "bound-mult" ? IPM::Options::AlphaY::BoundMult
                 : v == "full"       ? IPM::Options::AlphaY::Full
                 : v == "min-dual-infeas" ? IPM::Options::AlphaY::MinDualInfeas
                                          : IPM::Options::AlphaY::Primal;
      else if (a == "--threads") threads = std::stoi(v);
      else if (a == "--verbose") dopt.verbose = std::stoi(v);
      else if (a == "--save-solution") save_file = v;
      else if (a == "--save-report") report_file = v;
      else if (a == "--tag") tag = v;
      else if (a == "--rof-iters") rof_iters = std::stoi(v);
      else if (a == "--warm-tol") warm_tol = std::stod(v);
      else if (a == "--warm-iters") warm_max_iter = std::stoi(v);
      else { if (root) usage(); return 1; }
   }
   if (train_spec.empty()) train_spec = "../images/cameraman.png";
   ui::init(color_mode);
   if (nsub < 1 || nsub > N - 1) { ui::error("need 1 <= nsub <= size-1"); return 1; }
   if (cfg.op != "denoise" && cfg.op != "inpaint" && cfg.op != "deblur") {
      ui::error("--op must be denoise, inpaint or deblur");
      return 1;
   }
   if (loss != "mean" && loss != "sum") { ui::error("--loss must be mean or sum"); return 1; }
   if (start != "smoothed" && start != "scalar" && start != "uniform" && start != "ipm" &&
       start != "interior") {
      ui::error("--start must be smoothed, scalar, uniform, interior or ipm");
      return 1;
   }
   if (start_huber != "c1" && start_huber != "c2" && start_huber != "dual" && start_huber != "dualproj") {
      ui::error("--start-huber must be c1, c2, dual or dualproj");
      return 1;
   }
   if (start_ipm_init != "pdhg" && start_ipm_init != "smoothed") {
      ui::error("--start-ipm-init must be pdhg or smoothed");
      return 1;
   }
   if (start_newton != "primal" && start_newton != "pd") {
      ui::error("--start-newton must be primal or pd");
      return 1;
   }
   if (alpha0 < 0.0) alpha0 = 0.7 * sigma;
   if (dopt.fallback == dd::Backend::Sparse) dopt.fallback = dd::Backend::Dense;
   const bool use_mumps = dopt.blocks == dd::Backend::Mumps ||
                          (dopt.blocks == dd::Backend::Sparse && dopt.fallback == dd::Backend::Mumps);
#ifdef DD_HAVE_MUMPS
   if (use_mumps && !dd::MumpsBlock::self_test()) {
      ui::error("MUMPS self test failed (Schur layout or inertia): run ./mumps_check");
      return 1;
   }
#else
   if (use_mumps) {
      ui::error("this tv_learn was built without MUMPS (see build.sh)");
      return 1;
   }
#endif
#ifdef _OPENMP
   if (threads > 0) omp_set_num_threads(threads);
   const int nthreads = omp_get_max_threads();
#else
   const int nthreads = 1;
#endif

   // ---- the instances
   Blur blur;
   if (cfg.deblur()) blur = Blur(N, cfg.blur_sigma, cfg.blur_radius);
   const Blur* bp = cfg.deblur() ? &blur : nullptr;
   const std::vector<std::string> files = dataset::list_images(train_spec);
   std::vector<Instance> train, val;
   auto seed_of = [&](int i) { return same_seed ? seed : seed + (unsigned)i; };
   for (int i : dataset::pick(files, sel))
      train.push_back(make_instance(files[i], same_seed ? 0 : i, N, sigma, seed_of(i), cfg, bp, cfg.guide));
   if (!val_spec.empty()) {
      const std::vector<std::string> vf = dataset::list_images(val_spec);
      for (int i : dataset::pick(vf, vsel))
         val.push_back(make_instance(vf[i], same_seed ? 0 : i, N, sigma,
                                     seed_of(i) + (same_seed ? 0u : kValSeed), cfg, bp, "data"));
   }
   const int S = (int)train.size();
   if (tag.empty()) tag = S == 1 ? dataset::stem(train[0].path) : dataset::stem(train_spec);
   const double loss_scale = loss == "mean" ? 1.0 / S : 1.0;
   if (root) {
      const std::string wdesc =
          block >= 1 ? std::to_string(block) + "x" + std::to_string(block) + "-cell blocks" : "one weight";
      ui::header("tv_learn: %d image%s %dx%d (%s), %s%s, %s, sigma %g; %d rank%s x %d thread%s", S,
                 S > 1 ? "s" : "", N, N, S == 1 ? dataset::stem(train[0].path).c_str() : train_spec.c_str(),
                 cfg.op.c_str(), cfg.dtv ? " + DTV" : "", wdesc.c_str(), sigma, ranks, ranks > 1 ? "s" : "",
                 nthreads, nthreads > 1 ? "s" : "");
      if (!val.empty()) ui::kv("held-out", "%zu images (%s)", val.size(), val_spec.c_str());
      if (bp) ui::kv("blur", "sigma %.2f, radius %d (A = C'C reaches %d nodes)", cfg.blur_sigma, blur.w,
                     2 * blur.w);
   }

   // The exact (TV) lower level of one image at a weight per cell: loss ½‖u − u_clean‖²,
   // PSNR, SSIM (pdhg.hpp to 1e-9, at most --rof-iters iterations).
   struct Eval { double loss, psnr, ssim; int iters; };
   auto exact_eval = [&](const Instance& in, const std::vector<double>& acell, double tol) {
      const PdhgResult P = pdhg(in, bp, acell, tol, rof_iters);
      double l = 0.0;
      for (size_t i = 0; i < P.u.size(); ++i) l += 0.5 * (P.u[i] - in.clean[i]) * (P.u[i] - in.clean[i]);
      return Eval{l, psnr(in.clean, P.u.data()), ssim(in.clean, P.u.data(), N), P.iters};
   };

   // ---- the baselines: a grid search over a scalar weight, the smoothed bilevel method
   if (!grid_spec.empty() || smooth_gamma > 0.0) {
      if (ranks > 1) {
         if (root) ui::error("--grid and --smoothed run on one rank");
         return 1;
      }
      const int m_q = (N - 1) * (N - 1);
      auto report_exact = [&](const char* what, const std::vector<double>& acell, double ridge) {
         double Jtr = ridge, ptr = 0, str = 0, pva = 0, sva = 0;
         for (const Instance& in : train) {
            const Eval e = exact_eval(in, acell, 1e-9);
            Jtr += loss_scale * e.loss;
            ptr += e.psnr / S;
            str += e.ssim / S;
         }
         for (const Instance& in : val) {
            const Eval e = exact_eval(in, acell, 1e-9);
            pva += e.psnr / val.size();
            sva += e.ssim / val.size();
         }
         (void)what;
         ui::kv("upper-level loss", "%.8e (the exact lower level)", Jtr);
         ui::kv("PSNR, SSIM", "train %.3f dB, %.4f", ptr, str);
         if (!val.empty()) ui::kv("held-out", "%.3f dB, %.4f", pva, sva);
         return std::array<double, 5>{Jtr, ptr, str, val.empty() ? NAN : pva, val.empty() ? NAN : sva};
      };
      if (!grid_spec.empty()) {
         double a0 = 0, a1 = 0;
         int na = 0;
         if (std::sscanf(grid_spec.c_str(), "%lf:%lf:%d", &a0, &a1, &na) != 3 || a0 <= 0 || a1 <= a0 || na < 2) {
            ui::error("--grid needs A0:A1:N with 0 < A0 < A1, N >= 2");
            return 1;
         }
         const auto t0 = std::chrono::steady_clock::now();
         ui::header("Grid search: %d weights in [%g, %g] (TV lower level, PDHG to 1e-8)", na, a0, a1);
         std::printf("        alpha     train loss   PSNR (dB)   SSIM\n");
         double best_a = a0, best_J = HUGE_VAL;
         for (int k = 0; k < na; ++k) {
            const double a = a0 * std::pow(a1 / a0, (double)k / (na - 1));
            const std::vector<double> acell(m_q, a);
            double J = 0.5 * 1e-4 * a * a, pm = 0, sm = 0;
            for (const Instance& in : train) {
               const Eval e = exact_eval(in, acell, 1e-8);
               J += loss_scale * e.loss;
               pm += e.psnr / S;
               sm += e.ssim / S;
            }
            std::printf("  %12.6f  %.8e  %8.3f  %.4f\n", a, J, pm, sm);
            if (J < best_J) { best_J = J; best_a = a; }
         }
         const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         ui::ok("Grid search: best alpha %.6f (training loss %.8e), %d weights x %d images in %.2f s",
                best_a, best_J, na, S, wall);
         const auto r = report_exact("grid best", std::vector<double>(m_q, best_a), 0.5 * 1e-4 * best_a * best_a);
         std::printf("CSV-GRID,%s,%d,%d,%g,%s,%d,%.6f,%.8e,%.3f,%.4f,%.3f,%.4f,%.2f\n",
                     train_spec.c_str(), N, S, sigma, cfg.op.c_str(), na, best_a, r[0], r[1], r[2],
                     r[3], r[4], wall);
      }
      if (smooth_gamma > 0.0) {
         const LearnPartition spart(N, 1, S, block);
         ui::header("The smoothed bilevel method (Huber, gamma %.0e), L-BFGS", smooth_gamma);
         const smoothed::Result R = smoothed::solve(train, bp, spart, smooth_gamma, alpha0, loss_scale,
                                                    true, smooth_tol, smooth_iters);
         double am = 0, amin = 1e300, amax = -1e300, ridge = 0;
         for (double v : R.alpha) {
            am += v / R.alpha.size();
            amin = std::min(amin, v);
            amax = std::max(amax, v);
            ridge += 0.5 * 1e-4 * v * v;
         }
         ui::outcome(R.converged, "The smoothed method %s (%s): %d iterations, %.2f s; smoothed loss %.8e",
                     R.converged ? "converged" : "stopped", R.stop, R.iters, R.wall, R.J);
         ui::header("Result");
         if (R.alpha.size() == 1) ui::kv("alpha", "%.6f", R.alpha[0]);
         else ui::kv("alpha", "%zu blocks: mean %.6f, min %.6f, max %.6f", R.alpha.size(), am, amin, amax);
         if (!save_alpha_file.empty()) write_alpha(save_alpha_file, R.alpha);
         std::vector<double> acell(m_q);
         for (int e = 0; e < m_q; ++e) acell[e] = R.alpha[spart.block(e)];
         const auto r = report_exact("smoothed", acell, ridge);
         std::printf("CSV-SMOOTH,%s,%d,%d,%g,%s,%g,%d,%d,%d,%d,%ld,%d,%.6f,%.8e,%.8e,%.3f,%.4f,%.3f,%.4f,%.2f\n",
                     train_spec.c_str(), N, S, sigma, cfg.op.c_str(), smooth_gamma, block,
                     R.converged ? 1 : 0, R.iters, R.lower_solves, R.newton_iters, R.lower_failures,
                     am, R.J, r[0], r[1], r[2], r[3], r[4], R.wall);
      }
      return 0;
   }

   const LearnPartition part(N, nsub, S, block);
   if (ranks > part.n_tiles) {   // a rank without a tile hangs the solver (tv_dd crashes)
      if (root)
         ui::error("%d ranks but only %d tiles (%d images x %dx%d): every rank needs a tile; use"
                   " mpirun -np %d or fewer, or a larger --nsub", ranks, part.n_tiles, S, nsub, nsub,
                   part.n_tiles);
      return 1;
   }
   // ---- the IPM's options (the MPCC's; stage 1 of --start ipm changes a few)
   const double tol_target = 1e-8;
   auto configure_ipm = [&](IPM& ipm) {
      ipm.opt.tol = std::max(tol_target, 0.1 * t_min);
      ipm.opt.max_iter = max_iter;
      ipm.opt.mu0 = mu0;
      ipm.opt.bound_push = bound_push;
      ipm.opt.kappa_eps = kappa_eps;
      ipm.opt.mu_max_steps = mu_steps;
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
      ipm.opt.max_soc = max_soc;
   };

   // ---- the start: a given map, stage 1 (the smoothed bilevel method), or uniform
   std::vector<double> ainit;
   StartFields fields;   // stage 1's lower-level solutions, the MPCC's start
   double stage1_wall = 0.0;
   std::string start_used = start;
   if (!alpha_init_file.empty()) {
      ainit = read_alpha(alpha_init_file);
      start_used = "file";
   } else if (start == "interior") {
      fields.theta_from_u = true;
      fields.u.resize(S);
      fields.qx.assign(S, std::vector<double>((size_t)(N - 1) * (N - 1), 0.0));
      fields.qy.assign(S, std::vector<double>((size_t)(N - 1) * (N - 1), 0.0));
      for (int s = 0; s < S; ++s) fields.u[s] = train[s].f;
   } else if (start == "ipm") {
      // stage 1: the smoothed problem q = h(Ku) in Cartesian form (tv_learn.hpp, cartesian_gamma)
      // by the same IPM and decomposition, from the uniform start; its solution
      // starts the MPCC
      const auto t1 = std::chrono::steady_clock::now();
      // smoothed: the lower level of the smoothed problem at alpha0 (Definition 1's Huber, Newton
      // to 1e-10), q = h(Ku); feasible for c1 and the dual rows, off by O(gamma) in c2's band
      StartFields f1;
      if (start_ipm_init == "smoothed") {
         const LearnPartition spart(N, 1, S, block);
         const smoothed::Result R0 = smoothed::solve(train, bp, spart, start_gamma, alpha0, loss_scale,
                                                     false, 1e-6, 0, 1e-10, start_ftol,
                                                     start_solver == "mumps" ||
                                                             (start_solver == "auto" && N >= 128)
                                                         ? 1
                                                         : 0,
                                                     true, 0.0, 0.1, 0.1, start_newton == "pd");
         f1.u = R0.u;
         f1.qx = R0.qx;
         f1.qy = R0.qy;
      }
      TvLearn p1(train, part, bp, alpha0, loss_scale, warm_tol, warm_max_iter, {},
                 f1.u.empty() ? nullptr : &f1, start_gamma);
      p1.print = root;
      p1.bounds_as_rows = bounds_rows;
      p1.cart_c1 = start_huber == "c1";
      p1.cart_dual = start_huber == "dual" || start_huber == "dualproj";
      p1.cart_dual_proj = start_huber == "dualproj";
      if (check) {   // the Cartesian rows' derivatives, then exit
         if (ranks > 1) { if (root) ui::error("--check-derivatives: one rank only"); return 1; }
         const double err = check_derivatives(p1, 1, 1 << 30, 10);
         return err < 1e-5 ? 0 : 2;
      }
      IPM ipm1;
      configure_ipm(ipm1);
      ipm1.opt.tol = start_tol;
      ipm1.opt.acceptable_tol = 0.0;
      ipm1.opt.mu_min = 0.0;
      if (root) {
         ui::header("Stage 1: the smoothed problem q = h(Ku) (%s, gamma %.0e), by the IPM",
                    p1.cart_dual_proj ? "Definition 1's Huber, primal-dual rows, q projected in the derivatives"
                    : p1.cart_dual ? "Definition 1's Huber, primal-dual rows max(gamma, |Ku|) q - Ku = 0"
                    : p1.cart_c1 ? "Huber of Definition 1, h = g/max(gamma, |g|)" : "C1 Huber clipping",
                    start_gamma);
         ui::kv("problem", "%d variables, %d constraints (Cartesian: u, q, alpha), to %.0e", p1.n,
                p1.mcon, start_tol);
         ui::kv("start", "%s at alpha %g", f1.u.empty() ? "the exact TV solution (lifted)"
                : "the smoothed lower level, q = h(Ku)", alpha0);
      }
      dd::SchurDD::Stats s1;
      const IPM::Result r1 = ipm1.solve(p1, dopt, &s1);
      std::vector<double> sol = p1.solution;   // rank 0's; to every rank
      sol.resize(p1.n);
      dd::Comm::bcast(sol.data(), (int)sol.size());
      fields = p1.fields_of(sol, ainit);
      stage1_wall = dd::Comm::max(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
      const double ipm1_wall = dd::Comm::max(r1.wall);   // every rank (a collective)
      if (root) {
         double am = 0, amin = 1e300, amax = -1e300;
         for (double v : ainit) { am += v / ainit.size(); amin = std::min(amin, v); amax = std::max(amax, v); }
         const char* st1 = r1.status == 0 ? "converged" : r1.status == 1 ? "iteration limit"
                         : r1.status == 3 ? "acceptable" : "FAILED";
         (void)ipm1_wall;
         ui::outcome(r1.status == 0 || r1.status == 3, "Stage 1 %s after %d iterations, %.2f s;"
                     " objective %.8e; alpha mean %.6f [%.6f, %.6f]", st1, r1.iters, stage1_wall,
                     r1.obj, am, amin, amax);
      }
      if (stage1_only) {
         const double rss = dd::Comm::max(peak_rss_mb());
         if (root) ui::kv("peak memory", "%.0f MB/rank", rss);
         return 0;
      }
   } else if (start == "smoothed" || start == "scalar") {
      // every rank: the images split over the ranks, the same result on each.
      // smoothed: stage 1 on the same block map; scalar: on one weight, which then
      // starts every block
      const LearnPartition spart(N, 1, S, start == "scalar" ? 0 : block);
      if (root)
         ui::header("Stage 1: the smoothed bilevel method (Huber, gamma %.0e)%s, L-BFGS", start_gamma,
                    start == "scalar" ? " on one weight" : "");
      const smoothed::Result R1 = smoothed::solve(train, bp, spart, start_gamma, alpha0, loss_scale,
                                                  !quiet && root, 1e-6, start_iters, 1e-10,
                                                  start_ftol,
                                                  start_solver == "pcg-ic"       ? 3
                                                  : start_solver == "pcg-jacobi" ? 2
                                                  : start_solver == "mumps" ||
                                                          (start_solver == "auto" && N >= 128)
                                                      ? 1
                                                      : 0,
                                                  true, start_gamma0, start_gamma_rate,
                                                  start_gamma_sigma, start_newton == "pd");
      ainit = start == "scalar" ? std::vector<double>(part.n_blocks, R1.alpha[0]) : R1.alpha;
      stage1_wall = R1.wall;
      fields.u = R1.u;
      fields.qx = R1.qx;
      fields.qy = R1.qy;
      if (root) {
         double am = 0, amin = 1e300, amax = -1e300;
         for (double v : ainit) { am += v / ainit.size(); amin = std::min(amin, v); amax = std::max(amax, v); }
         ui::outcome(R1.converged, "Stage 1 %s (%s): %d iterations, %d lower-level solves, %.2f s;"
                     " alpha mean %.6f [%.6f, %.6f]", R1.converged ? "converged" : "stopped", R1.stop,
                     R1.iters, R1.lower_solves, R1.wall, am, amin, amax);
      }
      if (stage1_only) {   // profiling: stop here
         const double rss = dd::Comm::max(peak_rss_mb());
         if (root) ui::kv("peak memory", "%.0f MB/rank", rss);
         return 0;
      }
   }
   TvLearn problem(train, part, bp, alpha0, loss_scale, warm_tol, warm_max_iter, ainit,
                   fields.u.empty() ? nullptr : &fields);
   problem.classify = classify;
   problem.diag = !diag_file.empty();
   problem.freeze_xi = freeze_xi;
   problem.freeze_persist = freeze_persist;
   if (freeze_xi > 0.0 && follow_persist > 0) {
      if (root) std::fprintf(stderr, "--freeze-xi and --follow-comp exclude each other\n");
      return 1;
   }
   if (follow_persist < 0) follow_persist = freeze_xi > 0.0 ? 0 : 3;
   problem.follow = follow_persist > 0;
   problem.follow_persist = follow_persist;
   problem.follow_stall = follow_stall;
   problem.print = root;
   problem.bounds_as_rows = bounds_rows;
   problem.t_on_solved = t_on_solved;
   const char* bname[] = {"sparse", "dense", "mumps"};
   const std::string blocks = dopt.blocks == dd::Backend::Sparse
                                 ? std::string("sparse+") + bname[(int)dopt.fallback]
                                 : bname[(int)dopt.blocks];
   const bool two_stage = start_used == "smoothed" || start_used == "scalar" || start_used == "ipm";
   if (root) {
      if (two_stage) ui::header("Stage 2: the MPCC by the IPM, from stage 1's point");
      else if (start_used == "file")
         ui::header("The MPCC by the IPM, from the weights in %s", alpha_init_file.c_str());
      else if (start_used == "interior")
         ui::header("The MPCC by the IPM, from the interior point u = f, q = 0 and the weight %.4f",
                    alpha0);
      else ui::header("The MPCC by the IPM, from the uniform weight %.4f", alpha0);
      ui::kv("problem", "%d variables, %d constraints (%d inequalities), %d copies; %d weight%s",
             problem.n, problem.mcon, problem.n_ineq, problem.n_link, part.n_blocks,
             part.n_blocks > 1 ? "s" : "");
      std::string cuts;
      for (int c : part.cut) cuts += (cuts.empty() ? "" : " ") + std::to_string(c);
      ui::kv("tiles", "%dx%d per image (%d in all); cuts at cells %s%s", nsub, nsub, part.n_tiles,
             cuts.c_str(), block >= 2 ? " (snapped to the blocks)" : "");
      ui::kv("linear algebra", "%s interface, %s tile blocks",
             dopt.interface == dd::SchurDD::Interface::Pcg ? "pcg" : "direct", blocks.c_str());
      if (problem.warm_residual > 0.0)
         ui::kv("warm start", "PDHG %d iterations%s, residual %.1e, h1 residual %.1e",
                problem.warm_iters, problem.warm_iters >= warm_max_iter ? " (the limit)" : "",
                problem.warm_residual, problem.warm_kkt);
   }

   if (check) {
      if (ranks > 1) { if (root) ui::error("--check-derivatives: one rank only"); return 1; }
      problem.eps_theta = 1.0;   // exercise the θ ridge
      const double err = check_derivatives(problem, 1, 1 << 30, 10);
      return err < 1e-5 ? 0 : 2;
   }

   // ---- μ-coupled continuation (as tv_dd)
   problem.t_min = t_min;
   problem.t_mu_scale = t_mu_scale;
   problem.t_rate = t_rate;
   problem.t_comp_ratio = t_comp_ratio;
   problem.t = t0 > 0.0 ? t0 : std::max(t_min, problem.t_mu_scale * mu0);
   problem.eps_theta = problem.c_theta * problem.t;
   problem.gate_floor = tol_target;
   problem.gate_dual = gate_dual;
   problem.gate_du_scale = gate_du_scale;

   IPM ipm;
   configure_ipm(ipm);
   dd::SchurDD::Stats s;
   IPM::Result r = ipm.solve(problem, dopt, &s);
   if (!diag_file.empty()) problem.write_diag(diag_file);   // collective

   const bool ok = r.status == 0 || r.status == 3 || (r.status == 2 && problem.gate_fired);
   const char* st = r.status == 0 ? "converged" : r.status == 3 ? "acceptable"
                  : r.status == 2 ? (problem.gate_fired ? "level gate" : "stopped")
                  : r.status == 1 ? "iteration limit" : "FAILED";
   const std::vector<double>& x = problem.solution;
   const double cg_mean = s.solves ? (double)s.cg_iters / s.solves : 0.0;
   const double wall = dd::Comm::max(r.wall), rss = dd::Comm::max(peak_rss_mb());
   const double tf = dd::Comm::max(s.t_factor), tb = dd::Comm::max(s.t_blocks);
   const double ts = dd::Comm::max(s.t_sdirect), tp = dd::Comm::max(s.t_precond);
   const double tsol = dd::Comm::max(s.t_solve), tts = dd::Comm::max(s.t_tile_solves);
   const double ti = dd::Comm::max(s.t_interface);
   if (!root) return ok ? 0 : 2;

   ui::outcome(ok, "%s %s (status %d) after %d iterations, t = %.1e, %.2f s",
               two_stage ? "Stage 2" : "The MPCC", st, r.status, r.iters, problem.t, wall);
   auto print_solver = [&]() {
      ui::header("Solver");
      ui::kv("refused steps", "wrong In(W_k) or singular W_k %ld (too few negatives %ld), S not PD %ld,"
             " singular %ld; iterations with delta > 0: %ld", r.tile_corrections, r.tile_too_few,
             r.s_corrections, r.singular, r.regularized);
      if (line_search) {
         ui::kv("line search", "%ld step halvings, %ld iterations without an acceptable step",
                r.ls_backtracks, r.ls_failures);
         ui::kv("restoration", "%ld phases (%ld for a stall, %ld failed), %ld iterations",
                r.restorations, r.stall_restorations, r.resto_failures, r.resto_iters);
      }
      if (max_soc > 0)
         ui::kv("second order", "%ld corrections computed, %ld taken", r.soc_tried, r.soc_accepted);
      ui::kv("linear systems", "%d tiles, KKT dimension %d, border %d (max per tile %d), max tile %d;"
             " %ld factorizations (%ld sparse fallbacks)", s.n_tiles, s.dim, s.p, s.max_pk, s.max_tk,
             s.factorizations, s.fallbacks);
      if (dopt.interface == dd::SchurDD::Interface::Pcg)
         ui::kv("pcg", "%ld solves, %.1f iterations on average (max %ld); negative curvature %ld,"
                " unconverged %ld, refused before PCG %ld", s.solves, cg_mean, s.cg_max,
                s.cg_breakdowns, s.cg_unconverged, s.s_tilde_indefinite);
      ui::kv("time split", "factorize %.2f s [blocks %.2f, S %.2f, preconditioner %.2f], solve %.2f s"
             " [tiles %.2f, interface %.2f], rest %.2f s (slowest rank)", tf, tb, ts, tp, tsol, tts,
             ti, wall - tf - tsol);
   };
   if (x.empty()) { print_solver(); return 2; }


   // ---- the learned weight, and what it does on every image
   const int m_u = N * N, m_q = (N - 1) * (N - 1);
   std::vector<double> ablk(part.n_blocks);
   double amean = 0.0, amin = 1e300, amax = -1e300;
   for (int b = 0; b < part.n_blocks; ++b) {
      ablk[b] = x[problem.oa + b];
      amean += ablk[b] / part.n_blocks;
      amin = std::min(amin, ablk[b]);
      amax = std::max(amax, ablk[b]);
   }
   const std::vector<double> acell = problem.alpha_cells(x.data());
   const double comp = problem.max_complementarity(x.data());
   ui::header("Weights");
   if (part.n_blocks == 1) ui::kv("alpha*", "%.6f", ablk[0]);
   else {
      ui::kv("alpha*", "%d blocks: mean %.6f, min %.6f, max %.6f", part.n_blocks, amean, amin, amax);
      if (part.nb <= 12) {   // the map itself, block rows top to bottom (as in the image)
         ui::kv("map", "%dx%d cells per block, rows top to bottom:", block, block);
         for (int i = 0; i < part.nb; ++i) {
            std::string row;
            char b[16];
            for (int j = 0; j < part.nb; ++j) {
               std::snprintf(b, sizeof b, " %.4f", ablk[i * part.nb + j]);
               row += b;
            }
            ui::line("                  %s", row.c_str());
         }
      } else {
         ui::kv("map", "%dx%d blocks: alpha_map in the .npz files, alpha_blocks in the report",
                part.nb, part.nb);
      }
   }
   ui::kv("max r(1-delta)", "%.2e", comp);
   if (!save_alpha_file.empty()) write_alpha(save_alpha_file, ablk);
   if (classify && problem.classes.iter >= 0) {
      const auto& C = problem.classes;
      for (int k = 0; k < 2; ++k)
         ui::kv(k == 0 ? "classes" : "", "eps %.1e (iteration %d, tau %.1e): biactive %ld [S %ld, M %ld,"
                " C %ld, W %ld; min gamma %.2e, min nu %.2e, max xi %.2e], active %ld, inactive %ld",
                C.eps[k], C.iter, C.tau, C.nB[k], C.nS[k], C.nM[k], C.nC[k], C.nW[k], C.gmin[k],
                C.nmin[k], C.xiB[k], C.nA[k], C.nI[k]);
   }
   if (freeze_xi > 0.0)
      ui::kv("frozen cells", "%ld (xi >= %.1e for %d iterations), keeping t in [%.2e, %.2e]",
             problem.frozen, freeze_xi, freeze_persist, problem.frozen_tmin, problem.frozen_tmax);
   if (follow_persist > 0)
      ui::kv("followed cells", "%ld raises (locked %d iterations, armed after a %d-iteration stall);"
             " at the end %ld cells with t_e > t, max t_e %.2e", problem.follow_events,
             follow_persist, follow_stall, problem.follow_now, problem.follow_tmax);

   std::vector<ReportRow> rows;
   double pn_tr = 0.0, pr_tr = 0.0, pr_val = 0.0, ss_tr = 0.0, ss_val = 0.0, J_exact = 0.0;
   for (double v : ablk) J_exact += 0.5 * 1e-4 * v * v;   // the ridge, as in the objective
   const LearnSummary run{problem.t, r.status, r.iters, comp, amean, problem.objective,
                          problem.xi_max, ok};
   const bool one_file = S == 1 && val.empty() && save_file.size() > 4 &&
                         save_file.compare(save_file.size() - 4, 4, ".npz") == 0;
   auto file_of = [&](const char* role, int j) {
      if (one_file) return save_file;
      char b[16];
      std::snprintf(b, sizeof b, "_%s%03d.npz", role, j);
      return save_file + b;
   };
   ui::header("Images (the lower level solved again at alpha*)");
   for (int j = 0; j < S + (int)val.size(); ++j) {
      const bool is_train = j < S;
      const Instance& in = is_train ? train[j] : val[j - S];
      ReportRow row;
      row.role = is_train ? "train" : "val";
      row.path = in.path;
      row.index = in.index;
      row.seed = in.seed;
      row.psnr_noisy = psnr(in.clean, in.f.data());
      PdhgResult P = pdhg(in, bp, acell, 1e-9, rof_iters);
      row.psnr_rof = psnr(in.clean, P.u.data());
      row.lower_kkt = pdhg_kkt(in, bp, acell, P);
      const double ss = ssim(in.clean, P.u.data(), N);
      PairFields pf;
      if (is_train) {
         const int s = j;
         const double* xs = x.data();
         row.psnr_mpcc = psnr(in.clean, xs + problem.ou + s * m_u);
         double du = 0.0;
         for (int i = 0; i < m_u; ++i) du = std::max(du, std::abs(xs[problem.ou + s * m_u + i] - P.u[i]));
         pn_tr += row.psnr_noisy / S;
         pr_tr += row.psnr_mpcc / S;
         ss_tr += ss / S;
         for (int i = 0; i < m_u; ++i)
            J_exact += loss_scale * 0.5 * (P.u[i] - in.clean[i]) * (P.u[i] - in.clean[i]);
         ui::line("train %3d  %-32s PSNR %.2f -> %.2f dB (exact: %.2f dB, max|u - u_exact| %.1e)", s,
                  dataset::stem(in.path).c_str(), row.psnr_noisy, row.psnr_mpcc, row.psnr_rof, du);
         if (!save_file.empty()) {
            auto seg = [&](int off, int len) {
               return std::vector<double>(xs + off + s * len, xs + off + (s + 1) * len);
            };
            pf.u = seg(problem.ou, m_u);
            pf.qx = seg(problem.oqx, m_q);
            pf.qy = seg(problem.oqy, m_q);
            pf.r = seg(problem.oR, m_q);
            pf.delta = seg(problem.oD, m_q);
            pf.theta = seg(problem.oTh, m_q);
         }
      } else {
         pr_val += row.psnr_rof / val.size();
         ss_val += ss / val.size();
         ui::line("val   %3d  %-32s PSNR %.2f -> %.2f dB (%d PDHG iterations)", j - S,
                  dataset::stem(in.path).c_str(), row.psnr_noisy, row.psnr_rof, P.iters);
         if (!save_file.empty()) pf = lift_fields(in, P.u, P.qx, P.qy);
      }
      if (!save_file.empty()) {
         const std::string fn = file_of(is_train ? "s" : "v", is_train ? j : j - S);
         if (save_pair_npz(fn, in, pf, acell, run, problem.mu_trace, nsub, block, sigma, cfg,
                           problem.n_orig))
            ui::line("           wrote %s", fn.c_str());
      }
      rows.push_back(row);
   }
   ui::header("Result");
   if (S == 1) ui::kv("PSNR", "%.2f dB noisy -> %.2f dB", pn_tr, pr_tr);
   else ui::kv("PSNR", "%.2f dB noisy -> %.2f dB (mean over %d images)", pn_tr, pr_tr, S);
   if (!val.empty()) ui::kv("PSNR held-out", "%.2f dB (mean over %zu images)", pr_val, val.size());
   ui::kv("upper-level loss", "%.8e with the exact lower level at alpha* (MPCC objective %.8e)",
          J_exact, problem.objective);
   if (val.empty()) ui::kv("SSIM", "%.4f", ss_tr);
   else ui::kv("SSIM", "train %.4f, held-out %.4f", ss_tr, ss_val);
   if (two_stage)
      ui::kv("time", "stage 1 %.2f s + stage 2 %.2f s = %.2f s", stage1_wall, wall, stage1_wall + wall);
   else
      ui::kv("time", "%.2f s", wall);
   ui::kv("peak memory", "%.0f MB/rank", rss);
   print_solver();

   // tv_dd's CSV fields (alpha: the scalar or the map's mean; PSNR: training means;
   // wall: the MPCC stage), then S, op, dtv, block, n_alpha, psnr_val, start, stage1_wall
   std::printf("CSV,%s,%d,%d,%g,%u,%g,%s,%s,%d,%d,%d,%d,%d,%.3e,%.6f,%.3e,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%ld,%ld,%.1f,%ld,%.0f,%d,%s,%d,%d,%d,%.3f,%s,%.3f\n",
               train_spec.c_str(), N, nsub, sigma, seed, t_min,
               dopt.interface == dd::SchurDD::Interface::Pcg ? "pcg" : "direct", blocks.c_str(),
               ranks, nthreads, r.status, ok ? 1 : 0, r.iters, problem.t, amean, comp, pn_tr, pr_tr,
               wall, tf, tsol, s.factorizations, s.fallbacks, cg_mean, s.cg_max, rss, S,
               cfg.op.c_str(), cfg.dtv ? 1 : 0, block, part.n_blocks, val.empty() ? NAN : pr_val,
               start_used.c_str(), stage1_wall);
   if (!report_file.empty() &&
       save_report(report_file, tag, S, N, nsub, block, sigma, cfg, loss, r.iters, r.status, wall,
                   amean, ablk, rows, start_used, stage1_wall))
      ui::ok("Wrote %s", report_file.c_str());
   return ok ? 0 : 2;
}
