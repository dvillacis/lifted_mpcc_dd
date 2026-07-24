// Scholtes-continuation driver for the STAGGERED 1D lifted TV-MPCC, with the
// arrowhead domain decomposition as IPOPT's actual linear solver.
//
//   --solver mumps|ma57|ma97   IPOPT's own linear solver (the monolithic reference,
//                         validates the TNLP port against ../lifted_mpcc_1d.py)
//   --solver dd --nsub k  the domain-decomposition arrowhead solver — every Newton
//                         system of every level is factorized and solved by
//                         Σ_k W_k + the interface S, and every inertia query is
//                         answered by Haynsworth, without ever factorizing the
//                         full KKT
//
// This is the counterpart of ../lifted_mpcc_1d.py's DD *probe*: there the
// arrowhead is reconstructed after the fact and checked on a random RHS; here it
// produces the steps. IPOPT keeps everything else — filter line search,
// μ-homotopy, δ_w/δ_c inertia correction, restoration.
//
// The one piece dd_solver.hpp cannot supply is the geometry: its built-in
// partition assumes the uniform 2D layout (dim = 17m+1, one pixel length), and on
// the staggered grid u is node-length while every other block is edge-length. So
// this driver builds the owner map itself — the same rule as
// ../lifted_mpcc_1d.py's kkt_owner — and injects it with config_owner().
//
// Data, warm start and a reference owner map come from dump_data_1d.py; run it
// first. Output columns mirror the Python's per-level table so the two diff directly.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
#include "dd_solver.hpp"
#include "driver_common.hpp"
#include "mpcc_1d_tnlp.hpp"

using namespace Ipopt;

// ---------------------------------------------------------------------------
// The partition, mirroring ../lifted_mpcc_1d.py's Partition1D + kkt_owner.
//
// Edges are cut into k contiguous runs and every node follows its OUTGOING edge
// (the last node follows the last edge), so the local state row of Ω_k is the
// board's N_k u − N_k f + Q(α)·K_kᵀ q_k = 0 and the shared interface node appears
// in both neighbours. Hence the complicating columns are, per cut, the first u
// past it and the last qx before it, plus the global α:  p = 2(k−1)+1.
//
// Only PRIMAL columns are ever border. In particular the ha row (α ≥ 0), whose
// only column is the border α, keeps its slack and multiplier in subdomain 0:
// those are dual directions, and putting them on the border would make S
// indefinite by construction, In(S) = (p−1,1,0), defeating the δ_w loop that runs
// on exactly that signal.
// ---------------------------------------------------------------------------
struct Partition1D {
   int nN, mE, k;
   std::vector<int> edge_owner, node_owner, cut_edges, cut_nodes;

   Partition1D(int nN_, int k_) : nN(nN_), mE(nN_ - 1), k(k_) {
      // np.linspace(0, mE, k+1).astype(int): truncated evenly spaced bounds, with
      // the last one exact. Reproduced literally so the C++ and Python partitions
      // agree edge for edge (the dumped owner map is diffed against this).
      std::vector<int> bnd(k + 1);
      const double step = (double)mE / (double)k;
      for (int j = 0; j < k; ++j) bnd[j] = (int)(j * step);
      bnd[k] = mE;
      edge_owner.assign(mE, 0);
      for (int j = 0; j < k; ++j)
         for (int e = bnd[j]; e < bnd[j + 1]; ++e) edge_owner[e] = j;
      node_owner.assign(nN, 0);
      for (int i = 0; i < mE; ++i) node_owner[i] = edge_owner[i];
      node_owner[mE] = edge_owner[mE - 1];
      for (int j = 0; j < k - 1; ++j) {
         cut_edges.push_back(bnd[j + 1] - 1);
         cut_nodes.push_back(bnd[j + 1]);
      }
   }
};

static std::vector<int> kkt_owner(const Mpcc1DTNLP& p, const Partition1D& part) {
   const int nN = p.nN, mE = p.mE;
   std::vector<int> owner(p.kkt_dim, 0);

   for (int i = 0; i < nN; ++i) owner[p.ou + i] = part.node_owner[i];
   for (int e = 0; e < mE; ++e) {
      owner[p.oqx + e] = owner[p.oqy + e] = owner[p.oR + e] =
         owner[p.oD + e] = owner[p.oTh + e] = part.edge_owner[e];
   }
   owner[p.oa] = -1;                                 // α is global

   const int s0 = p.n;                               // slacks (inequality rows)
   const int c0 = s0 + p.n_ineq;                     // λ_c (equality rows)
   const int d0 = c0 + p.n_eq;                       // λ_d (inequality rows)
   const int ineq0 = p.rhr;
   const int bases[2] = {s0, d0};
   for (int b = 0; b < 2; ++b) {
      const int base = bases[b];
      const int blk[3] = {p.rhr, p.rhd, p.rcomp};
      for (int t = 0; t < 3; ++t)
         for (int e = 0; e < mE; ++e)
            owner[base + blk[t] - ineq0 + e] = part.edge_owner[e];
      if (p.has_ha) owner[base + p.rha - ineq0] = 0;   // dual dirs stay local
   }
   for (int i = 0; i < nN; ++i) owner[c0 + p.rh1 + i] = part.node_owner[i];
   const int eqb[4] = {p.rh2x, p.rh2y, p.rh3x, p.rh3y};
   for (int t = 0; t < 4; ++t)
      for (int e = 0; e < mE; ++e) owner[c0 + eqb[t] + e] = part.edge_owner[e];

   for (size_t j = 0; j < part.cut_edges.size(); ++j) {
      owner[p.ou + part.cut_nodes[j]] = -1;      // first u past the cut
      owner[p.oqx + part.cut_edges[j]] = -1;     // last qx before the cut
   }
   return owner;
}

// Write what plot_1d.py needs to drive lifted_mpcc_1d.plot_solution: the reported
// iterate plus the continuation history. Plotting stays in Python on purpose —
// the 4-panel figure is already written and validated there, and the C++ solution
// is numerically the same vector, so re-implementing it here would only create a
// second thing to keep in sync.
//
//   n_var  n_levels  t_last  nsub  weight_flag
//   t  status  iters  comp_res  weight  obj  xi_max  converged     × n_levels
//   x…  (n_var)
static void save_solution(const std::string& fn, const Mpcc1DTNLP& p,
                          const std::vector<driver::Level>& hist,
                          const std::vector<double>& x, double t_last, int nsub) {
   std::ofstream out(fn);
   if (!out) { std::cerr << "cannot write " << fn << "\n"; return; }
   out << std::setprecision(17);
   out << p.n << " " << hist.size() << " " << t_last << " " << nsub << " "
       << (p.weight_exp ? 1 : 0) << "\n";
   for (const driver::Level& l : hist)
      out << l.t << " " << l.status << " " << l.iters << " " << l.comp_res << " "
          << l.weight << " " << l.obj << " " << l.xi_max << " "
          << (l.converged ? 1 : 0) << "\n";
   for (int i = 0; i < p.n; ++i) out << x[i] << (i + 1 < p.n ? ' ' : '\n');
}

// ---------------------------------------------------------------------------
// --self-check: five numbers plus the partition, all reproducible from Python in
// a few lines. It catches a mis-ported derivative BEFORE IPOPT ever runs, which
// is the failure mode that would otherwise look like "DD does not converge".
// ---------------------------------------------------------------------------
static void self_check(Mpcc1DTNLP& p, const std::vector<int>& owner,
                       const Partition1D& part, int nsub) {
   driver::print_checksums(p);

   int p_border = 0, mismatch = 0;
   std::vector<int> dims(nsub, 0), border;
   for (int i = 0; i < p.kkt_dim; ++i) {
      if (owner[i] < 0) { ++p_border; border.push_back(i); }
      else dims[owner[i]]++;
      if (!p.file_owner_.empty() && p.file_owner_[i] != owner[i]) ++mismatch;
   }
   std::cout << "    partition    : p=" << p_border << " (= 2(k−1)+1 = "
             << 2 * (nsub - 1) + 1 << ")   block dims=[";
   for (int j = 0; j < nsub; ++j) std::cout << dims[j] << (j + 1 < nsub ? ", " : "");
   std::cout << "]\n";
   std::cout << "    border idx   : [";
   for (size_t j = 0; j < border.size(); ++j)
      std::cout << border[j] << (j + 1 < border.size() ? ", " : "");
   std::cout << "]\n";
   std::cout << "    cut nodes/edges: [";
   for (size_t j = 0; j < part.cut_nodes.size(); ++j)
      std::cout << "u" << part.cut_nodes[j] + 1 << " ";
   for (size_t j = 0; j < part.cut_edges.size(); ++j)
      std::cout << "q" << part.cut_edges[j] + 1 << " ";
   std::cout << "alpha]\n";
   if (p.file_owner_.empty())
      std::cout << "    owner vs Python: no reference in the data file\n";
   else if (nsub != p.file_nsub)
      std::cout << "    owner vs Python: SKIPPED (file was dumped for nsub="
                << p.file_nsub << ")\n";
   else
      std::cout << "    owner vs Python: " << (mismatch ? "MISMATCH" : "identical")
                << " (" << mismatch << " differing entries)\n";
}

int main(int argc, char** argv) {
   std::string data, solver = "ma57", init = "file", save_sol, save_dd;
   std::string interface_solver = "direct", precond = "asd";
   double t0 = 1.0, tmin = 1e-4, factor = 0.3, tol = 1e-8, c_theta = 0.0;
   std::string cg_apply = "assembled";
   double wmax = 2e19, reg_alpha = 1e-4, cg_tol = 1e-10;
   int nsub = 2, maxiter = 1000, printlevel = 0, cg_maxit = 500;
   bool check = false, nsub_set = false, dual_warm = false, alpha_peel = true;

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
      else if (a == "--nsub")     { nsub = std::stoi(next()); nsub_set = true; }
      else if (a == "--t0")       t0 = std::stod(next());
      else if (a == "--t-min")    tmin = std::stod(next());
      else if (a == "--factor")   factor = std::stod(next());
      else if (a == "--tol")      tol = std::stod(next());
      else if (a == "--c-theta")  c_theta = std::stod(next());
      else if (a == "--w-max")    wmax = std::stod(next());
      else if (a == "--reg-alpha") reg_alpha = std::stod(next());
      else if (a == "--init")     init = next();
      else if (a == "--max-iter") maxiter = std::stoi(next());
      else if (a == "--print-level") printlevel = std::stoi(next());
      else if (a == "--save-solution") save_sol = next();
      else if (a == "--save-dd")  save_dd = next();
      else if (a == "--self-check")  check = true;
      else if (a == "--dual-warmstart") dual_warm = true;
      else if (a == "--interface") interface_solver = next();
      else if (a == "--precond")  precond = next();
      else if (a == "--cg-tol")   cg_tol = std::stod(next());
      else if (a == "--cg-max-iter") cg_maxit = std::stoi(next());
      else if (a == "--cg-apply") cg_apply = next();
      else if (a == "--no-alpha-peel") alpha_peel = false;
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (data.empty()) {
      std::cerr << "usage: dd_solve_1d --data <data/data_1d_N.txt> "
                   "[--solver mumps|ma57|ma97|dd] [--nsub k] [--self-check]\n"
                   "                   [--save-solution FILE] [--save-dd FILE]\n"
                   "                   [--interface direct|cg] "
                   "[--precond jacobi|bj|asd] [--no-alpha-peel]\n"
                   "  --interface cg (needs --solver dd): preconditioned CG on the\n"
                   "    arrowhead's interface instead of the direct MA57 back-solve,\n"
                   "    with the Lueg preconditioners (bj = local Schur blocks,\n"
                   "    asd = diagonally-assembled, jacobi = diag(|S|)) and the\n"
                   "    alpha row peeled as a scalar Schur step (dd_kkt.py's route).\n"
                   "  generate the data file first:\n"
                   "    python ../python/dump_data_1d.py --n 64 --nsub 4 "
                   "-o data/data_1d_64.txt\n"
                   "  and plot the result with:\n"
                   "    python ../python/plot_1d.py --data data/data_1d_64.txt "
                   "--solution sol.txt --save-plot sol.png\n";
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

   SmartPtr<Mpcc1DTNLP> mpcc = new Mpcc1DTNLP(data);
   mpcc->w_max_ = wmax;
   mpcc->reg_alpha_ = reg_alpha;
   if (!nsub_set && mpcc->file_nsub > 0) nsub = mpcc->file_nsub;
   if (nsub < 1 || nsub > mpcc->mE) {
      std::cerr << "need 1 <= nsub <= n-1\n";
      return 2;
   }
   if (init == "cold") mpcc->cold_start(mpcc->x_start_.data(), 0.7 * mpcc->sigma_);
   else if (init != "file") { std::cerr << "--init must be file|cold\n"; return 2; }
   if (!save_dd.empty() && solver != "dd") {
      std::cerr << "--save-dd needs --solver dd (there is no arrowhead otherwise)\n";
      return 2;
   }
   mpcc->set_theta_ref(mpcc->x_start_.data());
   mpcc->t_ = t0;

   Partition1D part(mpcc->nN, nsub);
   std::vector<int> owner = kkt_owner(*mpcc, part);

   std::cout << "1D lifted TV-MPCC (staggered, C++)  nodes=" << mpcc->nN
             << "  edges=" << mpcc->mE << "  n=" << mpcc->n
             << "  m_con=" << mpcc->mcon << " (" << mpcc->n_eq << " eq + "
             << mpcc->n_ineq << " ineq)  KKT dim=" << mpcc->kkt_dim << "\n";
   std::cout << "  Q(a) = " << (mpcc->weight_exp ? "e^a" : "a")
             << (mpcc->has_ha ? "  (+ explicit row ha: a >= 0)" : "  (a boxed)")
             << "   init=" << init << "   solver=" << solver;
   if (solver == "dd") std::cout << "  nsub=" << nsub;
   if (solver == "dd" && interface_solver == "cg")
      std::cout << "  interface=cg(" << precond
                << (alpha_peel ? ",alpha-peel" : ",no-peel")
                << ",tol=" << cg_tol << ",maxit=" << cg_maxit << ")";
   if (c_theta > 0) std::cout << "  c_theta=" << c_theta;
   std::cout << "\n";

   if (check) {
      self_check(*mpcc, owner, part, nsub);
      return 0;
   }

   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   if (!driver::init_app(app, printlevel, maxiter, solver)) return 1;

   if (solver == "dd") {
      DDArrowheadSolver::config_owner(owner, nsub);
      DDArrowheadSolver::config_dump_arrow(save_dd);
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
      DDArrowheadSolver::config_alpha(mpcc->oa);
      DDArrowheadSolver::reset_interface_stats();
   }
   auto optimize = [&]() -> ApplicationReturnStatus {
      if (solver == "dd") {
         SmartPtr<AlgorithmBuilder> b = new CustomSolverBuilder<DDArrowheadSolver>();
         return app->OptimizeNLP(new TNLPAdapter(GetRawPtr(mpcc)), b);
      }
      return app->OptimizeTNLP(GetRawPtr(mpcc));
   };

   // Primal-only level-to-level warm start by DEFAULT, matching
   // ../lifted_mpcc_1d.py (whose --dual-warmstart is likewise opt-in, measured
   // mixed elsewhere in this repo). Carrying the duals changes the path, and a
   // path difference would muddy the A/B against the Python reference.
   driver::RunResult res = driver::run_scholtes(
      app, *mpcc, driver::scholtes_schedule(t0, tmin, factor), c_theta, tol,
      /*warm_start=*/dual_warm, /*value_is_alpha=*/false, optimize);
   driver::print_summary(*mpcc, res);
   if (solver == "dd" && interface_solver == "cg")
      driver::print_interface_stats(precond, alpha_peel, cg_tol);

   if (!res.best_x.empty()) {
      if (!save_sol.empty()) {
         save_solution(save_sol, *mpcc, res.hist, res.best_x, res.best_t, nsub);
         std::cout << "  wrote " << save_sol << "  (plot it with "
                      "`python ../python/plot_1d.py --data " << data
                   << " --solution " << save_sol << " --save-plot sol.png`)\n";
      }
      if (!save_dd.empty())
         std::cout << "  wrote " << save_dd << "  (the arrowhead of the LAST Newton "
                      "step; plot it with `--dd-dump " << save_dd
                   << " --save-dd-plot dd.png`)\n";
   }
   return 0;
}
