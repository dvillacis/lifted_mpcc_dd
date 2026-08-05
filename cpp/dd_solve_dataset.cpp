// Scholtes-continuation driver for the MULTI-SAMPLE lifted TV-MPCC: learn ONE
// scalar weight α across a training SET of images, with the arrowhead domain
// decomposition applied PER TRAINING PAIR.
//
//   --data <dir|manifest.txt|image> --limit S    S training pairs
//   --solver dd --nsub 1                         one subdomain per pair
//   --solver dd --nsub k                         k×k tiles inside each pair too
//
// WHY THIS DECOMPOSITION. The samples of a bilevel learning problem are coupled
// by NOTHING except the shared hyperparameter: sample s's lifted lower-level
// system involves only (u_s, qx_s, qy_s, r_s, δ_s, θ_s) and α. So the "column
// is complicating iff its rows span ≥2 subdomains" rule — the same rule
// dd_solve_2d.cpp reads off the Jacobian sparsity — selects exactly {α}, and
// one block per training pair gives an arrowhead with an interface of **p = 1**:
//
//     [ W_1          b_1 ]         W_s = the whole single-image KKT of pair s
//     [     ⋱        ⋮   ]         S   = c − Σ_s b_sᵀ W_s⁻¹ b_s   (1×1)
//     [        W_S   b_S ]
//     [ b_1ᵀ ⋯ b_Sᵀ  c   ]
//
// No cut cells, no cut-corner rank deficiency, no promoted duals, no dual peel —
// every W_s is just a well-posed single-image KKT, and the Haynsworth inertia
// In(A) = Σ_s In(W_s) + In(S) costs one extra sign. This is the cleanest
// decomposition this package can express, and it is the one whose blocks live
// on separate ranks without any halo exchange at all.
//
// --nsub k > 1 composes the two decompositions: each pair is additionally cut
// into k×k tiles by the SAME Partition2D geometry and the SAME anchor rule as
// dd_solve_2d, giving S·k² subdomains, a wider border, and the cut-corner dual
// promotion back again. k = 1 is not a special case in the code — it falls out
// of the general path with an empty spatial cut.
//
// The dataset itself (folder listing, selection, per-pair noise seeding, the
// upsample guard) lives in dataset.hpp; the formulation in mpcc_dataset_tnlp.hpp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
#include "dd_solver.hpp"
#include "driver_common.hpp"
#include "mpcc_dataset_tnlp.hpp"
#include "partition_2d.hpp"

using namespace Ipopt;

// ---------------------------------------------------------------------------
// Label every KKT index with its subdomain (−1 = border).
//
// Same three rules as dd_solve_2d.cpp's kkt_owner, with one extra index level:
//   subdomain(sample s, cell e) = s·part.n_sub + part.cell_owner[e]
//   subdomain(sample s, node i) = s·part.n_sub + part.node_owner[i]
// so at part.n_sub == 1 (--nsub 1) the subdomain IS the sample.
//
// Rows are never duplicated; the single scalar ha row goes to subdomain 0 (its
// slack and multiplier are dual directions, and bordering them would make S
// indefinite by construction and defeat the δ_w loop). Only primal columns can
// be complicating, and which ones is read off the Jacobian sparsity — stencil-
// agnostic, and here also sample-agnostic: with no spatial cut the rule finds α
// and nothing else, entirely on its own.
// ---------------------------------------------------------------------------
static std::vector<int> kkt_owner_dataset(const MpccDatasetTNLP& p,
                                          const Partition2D& part,
                                          bool promote_corners = true,
                                          std::vector<int>* col_owner_out = nullptr,
                                          int* n_promoted = nullptr) {
   const int S = p.S_, m_u = p.m_u, m_q = p.m_q, K = part.n_sub;
   const int SQ = S * m_q;
   std::vector<int> row_owner(p.mcon, 0), col_owner(p.n, 0);

   for (int s = 0; s < S; ++s) {
      const int uo = s * m_u, co = s * m_q, base = s * K;
      for (int i = 0; i < m_u; ++i) row_owner[p.rh1 + uo + i] = base + part.node_owner[i];
      const int rblk[7] = {p.rh2x, p.rh2y, p.rh3x, p.rh3y, p.rhr, p.rhd, p.rcomp};
      for (int b = 0; b < 7; ++b)
         for (int e = 0; e < m_q; ++e)
            row_owner[rblk[b] + co + e] = base + part.cell_owner[e];

      for (int i = 0; i < m_u; ++i) col_owner[p.ou + uo + i] = base + part.node_owner[i];
      const int cblk[5] = {p.oqx, p.oqy, p.oR, p.oD, p.oTh};
      for (int b = 0; b < 5; ++b)
         for (int e = 0; e < m_q; ++e)
            col_owner[cblk[b] + co + e] = base + part.cell_owner[e];
   }
   if (p.has_ha) row_owner[p.rha] = 0;
   col_owner[p.oa] = -1;                      // α is global (dense h1 column,
                                              // in EVERY sample's state row)

   // A column is complicating iff its rows span ≥2 subdomains. −2 = not seen.
   std::vector<int> seen(p.n, -2);
   for (size_t t = 0; t < p.jr_.size(); ++t) {
      const int c = p.jc_[t], o = row_owner[p.jr_[t]];
      if (seen[c] == -2) seen[c] = o;
      else if (seen[c] >= 0 && seen[c] != o) seen[c] = -1;
   }
   for (int c = 0; c < p.n; ++c) if (seen[c] == -1) col_owner[c] = -1;

   // KKT ordering: primal | slacks (ineq rows) | λ_c (eq rows) | λ_d (ineq rows)
   const auto eq_beg = row_owner.begin(), eq_end = row_owner.begin() + p.n_eq;
   const auto ineq_beg = eq_end, ineq_end = row_owner.end();
   std::vector<int> owner;
   owner.reserve(p.kkt_dim);
   owner.insert(owner.end(), col_owner.begin(), col_owner.end());
   owner.insert(owner.end(), ineq_beg, ineq_end);      // slacks
   owner.insert(owner.end(), eq_beg, eq_end);          // λ_c
   owner.insert(owner.end(), ineq_beg, ineq_end);      // λ_d

   // CUT-CORNER DUAL PROMOTION, per sample — see dd_solve_2d.cpp for the full
   // derivation. Two independent rank-1 pairs per cross corner: (λ_h3x, λ_h3y)
   // where qx AND qy are both border (rank-1 at δ ≈ 0), and (λ_h2x, λ_h2y)
   // where the cell's ENTIRE u-stencil is border (rank-1 at r = 0, i.e. on any
   // flat cell). With --nsub 1 there are no cross corners, no border qx/qy and
   // no fully-cut stencil, so this loop promotes nothing and the border stays
   // exactly {α} — the p = 1 case, reached without a special path.
   int promoted = 0;
   if (promote_corners) {
      const int lam_c0 = p.n + p.n_ineq;         // start of the λ_c block
      // which u columns each cell's h2x/h2y rows touch — read off the Jacobian
      // structure, so stencil-agnostic like the border rule itself
      std::vector<int> u_cols(SQ, 0), u_bord(SQ, 0);
      for (size_t t = 0; t < p.jr_.size(); ++t) {
         const int r = p.jr_[t], c = p.jc_[t];
         if (c < p.oqx) {                        // a u column
            int e = -1;
            if (r >= p.rh2x && r < p.rh2y) e = r - p.rh2x;
            else if (r >= p.rh2y && r < p.rh3x) e = r - p.rh2y;
            if (e >= 0) { ++u_cols[e]; if (col_owner[c] < 0) ++u_bord[e]; }
         }
      }
      for (int e = 0; e < SQ; ++e) {
         if (col_owner[p.oqx + e] < 0 && col_owner[p.oqy + e] < 0) {
            owner[lam_c0 + p.rh3x + e] = -1;     // δ≈0 rank-1 pair
            owner[lam_c0 + p.rh3y + e] = -1;
            promoted += 2;
         }
         if (u_cols[e] > 0 && u_bord[e] == u_cols[e]) {
            owner[lam_c0 + p.rh2x + e] = -1;     // r≈0 rank-1 pair
            owner[lam_c0 + p.rh2y + e] = -1;
            promoted += 2;
         }
      }
   }
   if (col_owner_out) *col_owner_out = col_owner;
   if (n_promoted) *n_promoted = promoted;
   return owner;
}

// Gather sample s's slice of the field-major iterate back into the SINGLE-SAMPLE
// layout [u|qx|qy|r|δ|θ|α], so each pair's solution can be written in exactly
// dd_solve_2d's --save-solution format and read by python/plot_slurm.py with no
// changes at all. That reuse is the reason for one file per pair rather than a
// new multi-sample format nothing can plot yet.
static std::vector<double> slice_sample(const MpccDatasetTNLP& p,
                                        const std::vector<double>& x, int s) {
   const int m_u = p.m_u, m_q = p.m_q;
   std::vector<double> xs((size_t)m_u + 5 * m_q + 1);
   const int u1 = m_u, q1 = m_q;
   for (int i = 0; i < m_u; ++i) xs[i] = x[p.ou + s * m_u + i];
   const int src[5] = {p.oqx, p.oqy, p.oR, p.oD, p.oTh};
   for (int b = 0; b < 5; ++b)
      for (int e = 0; e < m_q; ++e)
         xs[u1 + b * q1 + e] = x[src[b] + s * m_q + e];
   xs[u1 + 5 * q1] = x[p.oa];
   return xs;
}

// One pair's solution, in dd_solve_2d.cpp's save_solution format (header,
// per-level history, the iterate, the instance, the μ-trace). The level history
// and the μ-trace are shared by the whole run — they describe the one solve —
// so every per-pair file carries the same copy; only the iterate and the
// instance block are that pair's own.
static void save_solution_sample(const std::string& fn, const MpccDatasetTNLP& p,
                                 const std::vector<driver::Level>& hist,
                                 const std::vector<double>& x, double t_last,
                                 int nsub, int s) {
   std::ofstream out(fn);
   if (!out) { std::cerr << "cannot write " << fn << "\n"; return; }
   const int m_u = p.m_u, m_q = p.m_q;
   const int n1 = m_u + 5 * m_q + 1;
   const std::vector<double> xs = slice_sample(p, x, s);
   out << std::setprecision(17);
   out << n1 << " " << hist.size() << " " << t_last << " " << nsub << " "
       << (p.weight_exp ? 1 : 0) << "\n";
   for (const driver::Level& l : hist)
      out << l.t << " " << l.status << " " << l.iters << " " << l.comp_res << " "
          << l.weight << " " << l.obj << " " << l.xi_max << " "
          << (l.converged ? 1 : 0) << "\n";
   for (int i = 0; i < n1; ++i) out << xs[i] << (i + 1 < n1 ? ' ' : '\n');
   out << p.N << " " << (p.averaged ? 1 : 0) << " " << p.sigma_ << "\n";
   for (int blk = 0; blk < 2; ++blk) {
      const std::vector<double>& v = blk ? p.f_ : p.uclean_;
      for (int i = 0; i < m_u; ++i)
         out << v[(size_t)s * m_u + i] << (i + 1 < m_u ? ' ' : '\n');
   }
   const size_t NC = 5;
   out << p.mu_hist_.size() / NC << " " << NC << "\n";
   for (size_t i = 0; i + NC - 1 < p.mu_hist_.size(); i += NC) {
      out << (long)p.mu_hist_[i];
      for (size_t j = 1; j < NC; ++j) out << " " << p.mu_hist_[i + j];
      out << "\n";
   }
}

static void self_check(MpccDatasetTNLP& p, const std::vector<int>& owner,
                       const std::vector<int>& col_owner, int n_sub,
                       int n_promoted) {
   driver::print_checksums(p);

   int p_border = 0;
   std::vector<int> dims(n_sub, 0);
   for (int i = 0; i < p.kkt_dim; ++i) {
      if (owner[i] < 0) ++p_border;
      else dims[owner[i]]++;
   }
   int nu = 0, nqx = 0, nqy = 0;
   for (int c = 0; c < p.n; ++c)
      if (col_owner[c] < 0) {
         if (c < p.oqx) ++nu;
         else if (c < p.oqy) ++nqx;
         else if (c < p.oR) ++nqy;
      }
   // n_sub is S·k² and can be large; print the distinct block sizes rather than
   // every one of them (with --nsub 1 they are identical by construction, and
   // that they ARE identical is the thing worth seeing).
   int dmin = dims.empty() ? 0 : dims[0], dmax = dmin;
   for (int d : dims) { dmin = std::min(dmin, d); dmax = std::max(dmax, d); }
   std::cout << "    partition    : p=" << p_border << "   n_sub=" << n_sub
             << "   block dims " << (dmin == dmax ? "all " : "min ") << dmin;
   if (dmin != dmax) std::cout << ", max " << dmax;
   std::cout << "\n";
   std::cout << "    complicating : u=" << nu << " qx=" << nqx << " qy=" << nqy
             << " alpha=1" << (n_promoted ? "  + " : "")
             << (n_promoted ? std::to_string(n_promoted) : "")
             << (n_promoted ? " promoted corner duals" : "") << "\n";
   if (p_border == 1)
      std::cout << "    the border is exactly {alpha}: the training pairs are "
                   "coupled by nothing else\n";
}

int main(int argc, char** argv) {
   std::string data, val_data, solver = "ma57", init = "cp";
   std::string save_sol, save_dd, save_manifest;
   std::string interface_solver = "direct", precond = "asd";
   std::string wk_backend = "ma57";
   double t0 = 1.0, tmin = 1e-4, factor = 0.85, tol = 1e-8, c_theta = 1.0;
   std::string cg_apply = "assembled";
   double wmax = 2e19, reg_alpha = 1e-4, sigma = 0.1, cg_tol = 1e-10;
   int nsub = 1, maxiter = 3000, printlevel = 0, size = 0, seed = 0, cg_maxit = 500;
   int limit = 4, skip = 0, stride = 1, val_limit = 0;
   int minres_lag = 1, schur_lag = 1;
   std::string schur_mode = "forward";
   std::string hessian = "limited-memory";
   std::string ma57_scaling = "off";
   std::string t_update = "mu";
   double t_mu_scale = 10.0;
   std::string weight = "linear", stencil = "onesided", normalize = "255";
   std::string partition = "tile", loss = "mean";
   bool check = false, dual_warm = false, promote = true;
   bool alpha_peel = true, dual_peel = true, allow_upsample = false;

   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
         if (i + 1 >= argc) {
            std::cerr << a << " needs a value\n";
            std::exit(2);
         }
         return argv[++i];
      };
      // ---- dataset ----
      if      (a == "--data")     data = next();
      else if (a == "--limit")    limit = std::stoi(next());
      else if (a == "--skip")     skip = std::stoi(next());
      else if (a == "--stride")   stride = std::stoi(next());
      else if (a == "--val-data") val_data = next();
      else if (a == "--val-limit") val_limit = std::stoi(next());
      else if (a == "--allow-upsample") allow_upsample = true;
      else if (a == "--save-manifest") save_manifest = next();
      else if (a == "--size")     size = std::stoi(next());
      else if (a == "--sigma")    sigma = std::stod(next());
      else if (a == "--seed")     seed = std::stoi(next());
      else if (a == "--normalize") normalize = next();
      // ---- model ----
      else if (a == "--loss")     loss = next();
      else if (a == "--weight")   weight = next();
      else if (a == "--stencil")  stencil = next();
      else if (a == "--w-max")    wmax = std::stod(next());
      else if (a == "--reg-alpha") reg_alpha = std::stod(next());
      else if (a == "--init")     init = next();
      // ---- solver ----
      else if (a == "--solver")   solver = next();
      else if (a == "--nsub")     nsub = std::stoi(next());
      else if (a == "--partition") partition = next();
      else if (a == "--no-promote-corners") promote = false;
      else if (a == "--interface") interface_solver = next();
      else if (a == "--precond")  precond = next();
      else if (a == "--wk-backend") wk_backend = next();
      else if (a == "--cg-tol")   cg_tol = std::stod(next());
      else if (a == "--cg-max-iter") cg_maxit = std::stoi(next());
      else if (a == "--cg-apply") cg_apply = next();
      else if (a == "--minres-lag") minres_lag = std::stoi(next());
      else if (a == "--schur-lag")  schur_lag = std::stoi(next());
      else if (a == "--schur")      schur_mode = next();
      else if (a == "--no-alpha-peel") alpha_peel = false;
      else if (a == "--no-dual-peel") dual_peel = false;
      else if (a == "--hessian")   hessian = next();
      else if (a == "--ma57-scaling") ma57_scaling = next();
      // ---- continuation ----
      else if (a == "--t0")       t0 = std::stod(next());
      else if (a == "--t-min")    tmin = std::stod(next());
      else if (a == "--factor")   factor = std::stod(next());
      else if (a == "--tol")      tol = std::stod(next());
      else if (a == "--c-theta")  c_theta = std::stod(next());
      else if (a == "--t-update")   t_update = next();
      else if (a == "--t-mu-scale") t_mu_scale = std::stod(next());
      else if (a == "--max-iter") maxiter = std::stoi(next());
      else if (a == "--print-level") printlevel = std::stoi(next());
      else if (a == "--dual-warmstart") dual_warm = true;
      // ---- output ----
      else if (a == "--self-check")  check = true;
      else if (a == "--save-solution") save_sol = next();
      else if (a == "--save-dd")  save_dd = next();
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (data.empty() || size <= 0) {
      std::cerr <<
         "usage: dd_solve_dataset --data <dir|manifest.txt|image> --size N [options]\n"
         "\n"
         "  Learn ONE scalar weight alpha across a training SET, with the\n"
         "  arrowhead DD applied per training pair.\n"
         "\n"
         "  dataset:\n"
         "    --data <spec>       a FOLDER of images (listed and sorted), a .txt\n"
         "                        MANIFEST (one image path per line, # comments,\n"
         "                        authored order kept), or a single image file.\n"
         "                        NOTE the dumped-instance .txt of dump_data_2d.py\n"
         "                        is not a dataset and is refused.\n"
         "    --size N            crop/resize every image to N x N   [required]\n"
         "    --limit S           training pairs (default 4)\n"
         "    --skip M --stride D select from the listing before --limit\n"
         "    --sigma s --seed k  noise; pair at listing index i gets seed k+i, so\n"
         "                        --limit changes never re-randomize kept pairs\n"
         "    --allow-upsample    permit a source smaller than N (refused by default)\n"
         "    --val-data <spec>   held-out source (default: the same listing, the\n"
         "                        V entries after the training slice)\n"
         "    --val-limit V       evaluate alpha* on V held-out pairs (default 0)\n"
         "    --save-manifest F   CSV of exactly what was loaded\n"
         "\n"
         "  decomposition:\n"
         "    --solver dd --nsub 1   ONE subdomain per training pair; the border is\n"
         "                           exactly {alpha}, so p=1 and S is 1x1 (default)\n"
         "    --solver dd --nsub k   additionally cut each pair into k x k tiles\n"
         "                           (S*k^2 subdomains, the dd_solve_2d geometry)\n"
         "    --partition tile|strip, --interface, --precond, --wk-backend,\n"
         "    --schur, --schur-lag, --hessian, --t-update ... as in dd_solve_2d\n"
         "\n"
         "  model:\n"
         "    --loss mean|sum     upper level averages (default) or sums over pairs\n"
         "    --weight linear|exp --stencil onesided|averaged --reg-alpha\n"
         "\n"
         "  NOTE on larger training sets: the default monotone barrier gate\n"
         "  stalls as S grows (measured at S=8, N=32: 3000 iterations with no\n"
         "  progress, for the MONOLITHIC solver too). DD_BARRIER_TOL=1000 clears\n"
         "  it outright -- 3000 stalled iterations became 73 converged ones.\n"
         "  Treat it as effectively required past a handful of pairs.\n"
         "\n"
         "  examples:\n"
         "    DD_BARRIER_TOL=1000 \\\n"
         "    ./dd_solve_dataset --data ~/data/kodak --size 32 --limit 8 \\\n"
         "        --val-limit 4 --solver dd --nsub 1 --save-solution sol\n"
         "    ./dd_solve_dataset --data ../images/cameraman.png --size 32 \\\n"
         "        --limit 1 --self-check      # the S=1 gate vs dd_solve_2d\n";
      return 2;
   }
   if (solver != "mumps" && solver != "ma57" && solver != "ma97" && solver != "dd") {
      std::cerr << "--solver must be mumps|ma57|ma97|dd\n";
      return 2;
   }
   if (partition != "tile" && partition != "strip") {
      std::cerr << "--partition must be tile|strip\n";
      return 2;
   }
   if (loss != "mean" && loss != "sum") {
      std::cerr << "--loss must be mean|sum\n";
      return 2;
   }
   if (interface_solver != "direct" && interface_solver != "cg" &&
       interface_solver != "minres") {
      std::cerr << "--interface must be direct|cg|minres\n";
      return 2;
   }
   if (interface_solver != "direct" && solver != "dd") {
      std::cerr << "--interface " << interface_solver
                << " needs --solver dd (it replaces the arrowhead's "
                   "interface solve)\n";
      return 2;
   }
   if (hessian != "exact" && hessian != "limited-memory") {
      std::cerr << "--hessian must be exact|limited-memory\n";
      return 2;
   }
   if (ma57_scaling != "on" && ma57_scaling != "off") {
      std::cerr << "--ma57-scaling must be on|off\n";
      return 2;
   }
   hsl_block_scaling_default() = (ma57_scaling == "on");
   if (schur_mode != "backsolve" && schur_mode != "forward") {
      std::cerr << "--schur must be backsolve|forward\n";
      return 2;
   }
   if (schur_lag < 1) { std::cerr << "--schur-lag must be >= 1\n"; return 2; }
   if (schur_lag > 1 && solver != "dd") {
      std::cerr << "--schur-lag needs --solver dd\n";
      return 2;
   }
   if (minres_lag < 1) { std::cerr << "--minres-lag must be >= 1\n"; return 2; }
   if (wk_backend != "ma57" && wk_backend != "mumps" && wk_backend != "hybrid") {
      std::cerr << "--wk-backend must be ma57|mumps|hybrid\n";
      return 2;
   }
   if (wk_backend != "ma57" && solver != "dd") {
      std::cerr << "--wk-backend " << wk_backend << " needs --solver dd\n";
      return 2;
   }
#ifndef DD_HAVE_MUMPS
   if (wk_backend != "ma57") {
      std::cerr << "--wk-backend " << wk_backend
                << ": this binary was built without MUMPS support\n";
      return 2;
   }
#endif
   if (t_update != "geometric" && t_update != "mu") {
      std::cerr << "--t-update must be geometric|mu\n";
      return 2;
   }
   if (t_mu_scale <= 0) { std::cerr << "--t-mu-scale must be > 0\n"; return 2; }
   if (t_update == "geometric") {
      if (!driver::valid_schedule(t0, tmin, factor)) return 2;
   } else if (!(tmin > 0) || !(t0 >= tmin)) {
      std::cerr << "--t-min must be > 0 and --t0 >= --t-min\n";
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
   if (weight != "linear" && weight != "exp") {
      std::cerr << "--weight must be linear|exp\n";
      return 2;
   }
   if (stencil != "onesided" && stencil != "averaged") {
      std::cerr << "--stencil must be onesided|averaged\n";
      return 2;
   }
   if (normalize != "255" && normalize != "minmax") {
      std::cerr << "--normalize must be 255|minmax\n";
      return 2;
   }
   if (init != "cp" && init != "cold") {
      std::cerr << "--init must be cp|cold (there is no dumped multi-sample "
                   "instance to read a `file` start from)\n";
      return 2;
   }
   if (limit < 1) { std::cerr << "--limit must be >= 1\n"; return 2; }
   if (val_limit < 0) { std::cerr << "--val-limit must be >= 0\n"; return 2; }
   if (nsub < 1 || nsub > size - 1) {
      std::cerr << "need 1 <= --nsub <= N-1\n";
      return 2;
   }
   if (!save_dd.empty() && solver != "dd") {
      std::cerr << "--save-dd needs --solver dd (there is no arrowhead otherwise)\n";
      return 2;
   }

   // ---- load the dataset -------------------------------------------------
   image_io::Opts iopt;
   iopt.size = size;
   iopt.sigma = sigma;
   iopt.minmax = (normalize == "minmax");
   iopt.area_downsample = true;      // as in dd_solve_2d: point-sampling aliases

   std::vector<dataset::Pair> train, val;
   std::vector<std::string> files;
   try {
      files = dataset::list_images(data);
      dataset::Select sel;
      sel.skip = skip; sel.stride = stride; sel.limit = limit;
      const std::vector<int> tidx = dataset::pick(files, sel);
      if ((int)tidx.size() < limit) {
         std::cerr << "--limit " << limit << " but only " << tidx.size()
                   << " image(s) available in " << data << " after --skip "
                   << skip << " --stride " << stride << "\n";
         return 2;
      }
      train = dataset::load_pairs(files, tidx, iopt, (unsigned)seed, allow_upsample);

      if (val_limit > 0) {
         if (!val_data.empty()) {
            const std::vector<std::string> vfiles = dataset::list_images(val_data);
            dataset::Select vs;
            vs.limit = val_limit;
            const std::vector<int> vidx = dataset::pick(vfiles, vs);
            if ((int)vidx.size() < val_limit) {
               std::cerr << "--val-limit " << val_limit << " but only "
                         << vidx.size() << " image(s) in " << val_data << "\n";
               return 2;
            }
            val = dataset::load_pairs(vfiles, vidx, iopt, (unsigned)seed,
                                      allow_upsample);
         } else {
            // Same listing, the entries AFTER the training slice — disjoint by
            // construction, and still seeded by listing index so a pair keeps
            // its noise whichever side of the split it lands on.
            dataset::Select vs;
            vs.skip = tidx.empty() ? 0 : tidx.back() + 1;
            vs.stride = stride;
            vs.limit = val_limit;
            const std::vector<int> vidx = dataset::pick(files, vs);
            if ((int)vidx.size() < val_limit) {
               std::cerr << "--val-limit " << val_limit << " but only "
                         << vidx.size() << " image(s) left in " << data
                         << " after the " << limit
                         << " training pair(s). Use --val-data to point the "
                            "held-out slice at another folder.\n";
               return 2;
            }
            val = dataset::load_pairs(files, vidx, iopt, (unsigned)seed,
                                      allow_upsample);
         }
      }
   } catch (const std::exception& e) {
      std::cerr << "dataset error: " << e.what() << "\n";
      return 2;
   }

   SmartPtr<MpccDatasetTNLP> mpcc =
      new MpccDatasetTNLP(train, size, sigma, weight == "exp",
                          stencil == "averaged", loss == "mean");
   mpcc->w_max_ = wmax;
   mpcc->reg_alpha_ = reg_alpha;

   for (dataset::Pair& p : train)
      p.psnr_noisy = driver::psnr(p.uclean, p.f.data(), mpcc->m_u);
   for (dataset::Pair& p : val)
      p.psnr_noisy = driver::psnr(p.uclean, p.f.data(), mpcc->m_u);

   const double w0 = 0.7 * sigma;       // the noise-aware default weight
   mpcc->x_start_.assign(mpcc->n, 0.0);
   if (init == "cold") mpcc->cold_start(mpcc->x_start_.data(), w0);
   else                mpcc->cp_start(mpcc->x_start_.data(), w0);
   mpcc->set_theta_ref(mpcc->x_start_.data());
   mpcc->t_ = t0;

   Partition2D part(mpcc->N, nsub, partition == "strip");
   const int n_sub = mpcc->S_ * part.n_sub;
   std::vector<int> col_owner;
   int n_promoted = 0;
   std::vector<int> owner =
      kkt_owner_dataset(*mpcc, part, promote, &col_owner, &n_promoted);

   if (solver == "dd" && interface_solver != "direct" && nsub == 1) {
      std::cerr << "--interface " << interface_solver
                << " with --nsub 1: the interface is 1x1 (alpha alone), so "
                   "there is nothing for a Krylov method to iterate on. Use "
                   "--interface direct, or --nsub k > 1 for a real interface.\n";
      return 2;
   }

   // ---- report the instance ----------------------------------------------
   std::cout << "dataset lifted TV-MPCC (staggered 2D, C++)  S=" << mpcc->S_
             << " pairs  N=" << mpcc->N << "  nodes/pair=" << mpcc->m_u
             << "  cells/pair=" << mpcc->m_q << "\n";
   std::cout << "  n=" << mpcc->n << "  m_con=" << mpcc->mcon << " ("
             << mpcc->n_eq << " eq + " << mpcc->n_ineq << " ineq)"
             << "  KKT dim=" << mpcc->kkt_dim << "\n";
   std::cout << "  stencil=" << (mpcc->averaged ? "averaged" : "onesided")
             << "  Q(a) = " << (mpcc->weight_exp ? "e^a" : "a")
             << (mpcc->has_ha ? "  (+ row ha: a >= 0)" : "  (a boxed)")
             << "  loss=" << loss << "(scale=" << mpcc->loss_scale_ << ")"
             << "  init=" << init << "  solver=" << solver;
   if (solver == "dd") {
      if (nsub == 1) std::cout << "  1 subdomain per pair (" << n_sub << " blocks)";
      else if (part.striped)
         std::cout << "  " << nsub << " strips/pair (" << n_sub << " blocks)";
      else
         std::cout << "  " << nsub << "x" << nsub << " tiles/pair (" << n_sub
                   << " blocks)";
      if (n_promoted) std::cout << "  +" << n_promoted << " promoted corner duals";
      else if (!promote) std::cout << "  (corner promotion OFF)";
      if (wk_backend == "mumps") std::cout << "  wk-backend=mumps(schur)";
   }
   if (solver == "dd" && interface_solver == "cg")
      std::cout << "  interface=cg(" << precond
                << (alpha_peel ? ",alpha-peel" : ",no-alpha-peel")
                << (dual_peel ? ",dual-peel" : ",no-dual-peel")
                << ",tol=" << cg_tol << ",maxit=" << cg_maxit << ")";
   if (solver == "dd" && interface_solver == "minres")
      std::cout << "  interface=minres(signed-ma57,lag=" << minres_lag
                << ",tol=" << cg_tol << ",maxit=" << cg_maxit << ")";
   if (t_update == "mu") std::cout << "  t-update=mu(c=" << t_mu_scale << ")";
   if (c_theta > 0) std::cout << "  c_theta=" << c_theta;
   std::cout << "\n";

   std::cout << "  training pairs (sigma=" << sigma << ", seed=listing index + "
             << seed << "):\n";
   std::cout << std::fixed << std::setprecision(2);
   for (int s = 0; s < mpcc->S_; ++s) {
      const dataset::Pair& p = train[s];
      std::cout << "    [" << std::setw(3) << s << "] " << p.path << "  "
                << p.src_w << "x" << p.src_h << "->" << size
                << "  seed=" << p.seed << "  noisy " << p.psnr_noisy << " dB\n";
   }
   if (!val.empty()) {
      std::cout << "  held-out pairs:\n";
      for (size_t s = 0; s < val.size(); ++s)
         std::cout << "    [" << std::setw(3) << s << "] " << val[s].path
                   << "  " << val[s].src_w << "x" << val[s].src_h << "->" << size
                   << "  seed=" << val[s].seed << "  noisy "
                   << val[s].psnr_noisy << " dB\n";
   }
   std::cout << std::defaultfloat;

   if (!save_manifest.empty()) {
      try {
         dataset::write_manifest(save_manifest, train, size, sigma, "train");
         if (!val.empty())
            dataset::write_manifest(save_manifest, val, size, sigma, "val", true);
         std::cout << "  wrote " << save_manifest << "  (the dataset as loaded)\n";
      } catch (const std::exception& e) {
         std::cerr << "manifest error: " << e.what() << "\n";
         return 1;
      }
   }

   if (check) {
      self_check(*mpcc, owner, col_owner, n_sub, n_promoted);
      return 0;
   }

   // ---- solve --------------------------------------------------------------
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   if (!driver::init_app(app, printlevel, maxiter, solver, hessian)) return 1;

   if (solver == "dd") {
      DDArrowheadSolver::config_owner(owner, n_sub);
      DDArrowheadSolver::config_dump_arrow(save_dd);
      DDArrowheadSolver::config_interface(
         interface_solver == "cg"       ? DDArrowheadSolver::IFACE_CG
         : interface_solver == "minres" ? DDArrowheadSolver::IFACE_MINRES
                                        : DDArrowheadSolver::IFACE_DIRECT,
         precond == "jacobi" ? DDArrowheadSolver::PRECOND_JACOBI
         : precond == "bj"   ? DDArrowheadSolver::PRECOND_BJ
                             : DDArrowheadSolver::PRECOND_ASD,
         alpha_peel, cg_tol, cg_maxit);
      DDArrowheadSolver::config_minres(minres_lag);
      DDArrowheadSolver::config_schur_lag(schur_lag);
      DDArrowheadSolver::config_schur_forward(schur_mode == "forward");
      DDArrowheadSolver::config_cg_apply(
         cg_apply == "matfree" ? DDArrowheadSolver::APPLY_MATFREE
                               : DDArrowheadSolver::APPLY_ASSEMBLED);
      DDArrowheadSolver::config_alpha(mpcc->oa);
      DDArrowheadSolver::config_peel(dual_peel ? mpcc->n : (1 << 30));
      DDArrowheadSolver::config_wk_backend(
         wk_backend == "mumps"    ? DDArrowheadSolver::WK_MUMPS
         : wk_backend == "hybrid" ? DDArrowheadSolver::WK_HYBRID
                                  : DDArrowheadSolver::WK_MA57);
      DDArrowheadSolver::reset_interface_stats();
   }
   auto optimize = [&]() -> ApplicationReturnStatus {
      if (solver == "dd") {
         SmartPtr<AlgorithmBuilder> b = new CustomSolverBuilder<DDArrowheadSolver>();
         return app->OptimizeNLP(new TNLPAdapter(GetRawPtr(mpcc)), b);
      }
      return app->OptimizeTNLP(GetRawPtr(mpcc));
   };

   driver::RunResult res =
      (t_update == "mu")
         ? driver::run_mu_coupled(app, *mpcc, tmin, t_mu_scale, c_theta, tol,
                                  printlevel, optimize)
         : driver::run_scholtes(app, *mpcc,
                                driver::scholtes_schedule(t0, tmin, factor),
                                c_theta, tol, /*warm_start=*/dual_warm,
                                /*value_is_alpha=*/false, optimize);

   if (res.best_x.empty()) {
      if (solver == "dd" && interface_solver == "cg")
         driver::print_interface_stats(precond, alpha_peel, cg_tol);
      if (solver == "dd" && interface_solver == "minres")
         driver::print_minres_stats(cg_tol, minres_lag);
      return 0;
   }

   // ---- report -------------------------------------------------------------
   const double alpha = res.best_x[mpcc->oa];
   const double lam = mpcc->weight_of_alpha(alpha);
   std::cout << "----------------------------------------------------------------\n";
   std::cout << "  best level t  : " << std::scientific << res.best_t << "\n";
   std::cout << std::fixed << std::setprecision(6);
   std::cout << "  total IPOPT it: " << res.total_iter << "\n";
   std::cout << "  alpha*        : " << alpha << "\n";
   std::cout << "  weight Q(a*)  : " << lam << "\n";

   // Per-pair PSNR from the MPCC's own u_s, and — from the SAME learned weight —
   // the reconstruction the lower level alone produces. They must agree to
   // lower-level tolerance: a gap means the complementarity is not tight, i.e.
   // u_s is not actually a minimizer of the inner problem at Q(α*), which is
   // the failure mode a single objective value hides completely.
   std::vector<std::vector<double>> rec_tr, rec_val;
   mpcc->rof_reconstruct(lam, train, rec_tr);
   if (!val.empty()) mpcc->rof_reconstruct(lam, val, rec_val);

   std::cout << std::fixed << std::setprecision(2);
   std::cout << "\n  per-pair PSNR (dB):\n";
   std::cout << "                                        noisy    MPCC u_s   "
                "ROF at Q(a*)\n";
   double sn = 0, sm = 0, sr = 0;
   for (int s = 0; s < mpcc->S_; ++s) {
      const double pm = driver::psnr(train[s].uclean,
                                     res.best_x.data() + mpcc->ou + s * mpcc->m_u,
                                     mpcc->m_u);
      const double pr = driver::psnr(train[s].uclean, rec_tr[s].data(), mpcc->m_u);
      sn += train[s].psnr_noisy; sm += pm; sr += pr;
      std::string nm = train[s].path;
      const size_t k = nm.find_last_of('/');
      if (k != std::string::npos) nm = nm.substr(k + 1);
      if (nm.size() > 30) nm = nm.substr(0, 30);
      std::cout << "    train [" << std::setw(3) << s << "] " << std::left
                << std::setw(31) << nm << std::right << std::setw(8)
                << train[s].psnr_noisy << std::setw(12) << pm << std::setw(14)
                << pr << "\n";
   }
   const double S = (double)mpcc->S_;
   std::cout << "    " << std::left << std::setw(41) << "  mean (train)"
             << std::right << std::setw(8) << sn / S << std::setw(12) << sm / S
             << std::setw(14) << sr / S << "\n";

   if (!val.empty()) {
      double vn = 0, vr = 0;
      for (size_t s = 0; s < val.size(); ++s) {
         const double pr = driver::psnr(val[s].uclean, rec_val[s].data(), mpcc->m_u);
         vn += val[s].psnr_noisy; vr += pr;
         std::string nm = val[s].path;
         const size_t k = nm.find_last_of('/');
         if (k != std::string::npos) nm = nm.substr(k + 1);
         if (nm.size() > 30) nm = nm.substr(0, 30);
         std::cout << "    val   [" << std::setw(3) << s << "] " << std::left
                   << std::setw(31) << nm << std::right << std::setw(8)
                   << val[s].psnr_noisy << std::setw(12) << "-" << std::setw(14)
                   << pr << "\n";
      }
      const double V = (double)val.size();
      std::cout << "    " << std::left << std::setw(41) << "  mean (held-out)"
                << std::right << std::setw(8) << vn / V << std::setw(12) << "-"
                << std::setw(14) << vr / V << "\n";
      std::cout << "    generalization gap (train - held-out, ROF column): "
                << (sr / S - vr / V) << " dB\n";
   }
   std::cout << std::defaultfloat;

   if (solver == "dd" && interface_solver == "cg")
      driver::print_interface_stats(precond, alpha_peel, cg_tol);
   if (solver == "dd" && interface_solver == "minres")
      driver::print_minres_stats(cg_tol, minres_lag);

   if (!save_sol.empty()) {
      // One file per pair, each in dd_solve_2d's format — python/plot_slurm.py
      // reads them unchanged.
      std::string pre = save_sol;
      if (image_io::ends_with(pre, ".txt")) pre = pre.substr(0, pre.size() - 4);
      for (int s = 0; s < mpcc->S_; ++s) {
         char suf[32];
         std::snprintf(suf, sizeof suf, "_s%03d.txt", s);
         save_solution_sample(pre + suf, *mpcc, res.hist, res.best_x, res.best_t,
                              nsub, s);
      }
      std::cout << "  wrote " << pre << "_s000.txt .. " << pre << "_s"
                << std::setfill('0') << std::setw(3) << (mpcc->S_ - 1)
                << std::setfill(' ') << ".txt  (" << mpcc->S_
                << " per-pair solutions, plot_slurm.py format)\n";
   }
   if (!save_dd.empty())
      std::cout << "  wrote " << save_dd
                << "  (arrowhead of the LAST Newton step)\n";
   return 0;
}
