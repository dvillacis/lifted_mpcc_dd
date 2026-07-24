// Scholtes-continuation driver for the STAGGERED 2D lifted TV-MPCC, with the
// arrowhead domain decomposition as IPOPT's actual linear solver.
//
//   --solver mumps|ma57|ma97   IPOPT's own linear solver (the monolithic reference,
//                         validates the TNLP port against ../lifted_mpcc_2d.py)
//   --solver dd --nsub k  the DD arrowhead solver — k×k tiles, every Newton system
//                         factorized and solved by Σ_k W_k + the interface S, every
//                         inertia query answered by Haynsworth
//
// The 2D sibling of dd_solve_1d.cpp; see that file for the design. Two things are
// 2D-specific:
//
//   * **the anchor rule** — node (i,j) belongs to the tile of cell (i−1,j−1),
//     clamped: the cell the node anchors under the one-sided stencil. Measured in
//     Python, it is what keeps only ONE dual component crossing each cut
//     (qx at vertical cuts, qy at horizontal), and it cuts the interface from
//     p=90 to p=60 at N=16 k=2 (538 → 364 at N=32 k=4).
//   * **the border comes from the Jacobian sparsity**, not from hand-derived
//     geometry: a primal column is complicating iff the rows it appears in are
//     owned by ≥2 tiles. That is stencil-agnostic (it gives the right, larger
//     border for `averaged`, whose stencil genuinely touches all four nodes of a
//     cell) and it is the same rule ../dd_structure.py --self-test validates.
//
// Data, warm start and a reference owner map come from dump_data_2d.py.
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
#include "mpcc_2d_tnlp.hpp"

using namespace Ipopt;

// ---------------------------------------------------------------------------
// Partition of the (N−1)² cells. TILE = k×k (mirrors ../lifted_mpcc_2d.py). STRIP =
// k horizontal row-strips (cut ONE direction only): then no cell sits at a 4-way
// cross corner, so no cell has both qx AND qy on the border, so the cut-corner dual
// rank-deficiency never arises — the 2D generalization of the working 1D DD, at the
// price of a wider interface and only k subdomains.
// ---------------------------------------------------------------------------
struct Partition2D {
   int N, nc, k, n_sub;
   bool striped;
   std::vector<int> bounds, cell_owner, node_owner;

   Partition2D(int N_, int k_, bool striped_ = false)
       : N(N_), nc(N_ - 1), k(k_), n_sub(striped_ ? k_ : k_ * k_), striped(striped_) {
      // np.linspace(0, nc, k+1).astype(int) — truncated, last one exact.
      bounds.assign(k + 1, 0);
      const double step = (double)nc / (double)k;
      for (int j = 0; j < k; ++j) bounds[j] = (int)(j * step);
      bounds[k] = nc;

      cell_owner.assign(nc * nc, 0);
      if (striped) {
         // Only the ROW index chooses the strip; columns are never cut.
         for (int a = 0; a < k; ++a)
            for (int i = bounds[a]; i < bounds[a + 1]; ++i)
               for (int j = 0; j < nc; ++j) cell_owner[i * nc + j] = a;
      } else {
         for (int a = 0; a < k; ++a)
            for (int c = 0; c < k; ++c)
               for (int i = bounds[a]; i < bounds[a + 1]; ++i)
                  for (int j = bounds[c]; j < bounds[c + 1]; ++j)
                     cell_owner[i * nc + j] = a * k + c;
      }

      // anchor rule: node (i,j) → cell (i−1,j−1), clamped — the cell the node
      // anchors under the one-sided stencil
      node_owner.assign(N * N, 0);
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            const int ci = std::min(std::max(i - 1, 0), nc - 1);
            const int cj = std::min(std::max(j - 1, 0), nc - 1);
            node_owner[i * N + j] = cell_owner[ci * nc + cj];
         }
   }
};

// Label every KKT index with its subdomain (−1 = border), mirroring
// ../lifted_mpcc_2d.py's kkt_owner. Rows are never duplicated (the scalar ha row
// goes to tile 0 — its slack and multiplier are dual directions, and bordering them
// would make S indefinite by construction and defeat the δ_w loop). Only primal
// columns can be complicating, and which ones is read off the Jacobian sparsity.
static std::vector<int> kkt_owner(const Mpcc2DTNLP& p, const Partition2D& part,
                                  bool promote_corners = true,
                                  std::vector<int>* col_owner_out = nullptr,
                                  int* n_promoted = nullptr) {
   const int m_u = p.m_u, m_q = p.m_q;
   std::vector<int> row_owner(p.mcon, 0), col_owner(p.n, 0);

   for (int i = 0; i < m_u; ++i) row_owner[p.rh1 + i] = part.node_owner[i];
   const int rblk[7] = {p.rh2x, p.rh2y, p.rh3x, p.rh3y, p.rhr, p.rhd, p.rcomp};
   for (int b = 0; b < 7; ++b)
      for (int e = 0; e < m_q; ++e) row_owner[rblk[b] + e] = part.cell_owner[e];
   if (p.has_ha) row_owner[p.rha] = 0;

   for (int i = 0; i < m_u; ++i) col_owner[p.ou + i] = part.node_owner[i];
   const int cblk[5] = {p.oqx, p.oqy, p.oR, p.oD, p.oTh};
   for (int b = 0; b < 5; ++b)
      for (int e = 0; e < m_q; ++e) col_owner[cblk[b] + e] = part.cell_owner[e];
   col_owner[p.oa] = -1;                      // α is global (dense h1 column)

   // A column is complicating iff its rows span ≥2 tiles. −2 = not seen yet.
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

   // CUT-CORNER DUAL PROMOTION. At each of the (k−1)² cross corners of the tiling
   // there is exactly one cell whose qx AND qy are both complicating (the vertical
   // cut's qx column meets the horizontal cut's qy row). Inside its block that
   // cell's dual pair (λ_h3x, λ_h3y) then keeps only its δ/θ couplings, which are
   // rank-1 whenever δ ≈ 0 — the deficiency SVD root-caused for the uniform 2D
   // solver. It is DUAL-side, so IPOPT's δ_w (primal-only) can never repair it,
   // and it degrades the solve silently rather than reporting SINGULAR.
   //
   // THERE IS A SECOND, INDEPENDENT DEFICIENCY OF THE SAME SHAPE ON THE PRIMAL
   // SIDE, and it is the one that actually bites (root-caused 2026-07-21 by SVD
   // of a dumped block). A cell whose ENTIRE u-stencil is on the border loses
   // every u coupling from its h2 rows, so (λ_h2x, λ_h2y) keeps only the 2×2
   //
   //     [ ∂h2x/∂r  ∂h2x/∂θ ]   [ −cos θ    r sin θ ]
   //     [ ∂h2y/∂r  ∂h2y/∂θ ] = [ −sin θ   −r cos θ ],   det = r,
   //
   // which is **rank-1 exactly when r = 0** — i.e. on any flat cell, which is most
   // of them, at every level. Measured at N=32 k=4, four W_k came back numerically
   // singular (σ_min ~ 1e-16 at ‖W‖ = 1e2) while the FULL matrix was fine, and
   // the null vectors sat on exactly these pairs.
   //
   // Both sets have (k−1)² members — one per cross corner — but they are DIFFERENT
   // cells (the dual set is where qx and qy cross; this one is where the node
   // stencil is entirely cut away), so both promotions are needed. Promotion costs
   // 2 border entries per corner each; blocks keep full rank with NO artificial
   // shift and the Haynsworth inertia stays exact.
   int promoted = 0;
   if (promote_corners) {
      const int lam_c0 = p.n + p.n_ineq;         // start of the λ_c block
      // which u columns each cell's h2x/h2y rows touch — read off the Jacobian
      // structure, so it is stencil-agnostic like the border rule itself
      std::vector<int> u_cols(m_q, 0), u_bord(m_q, 0);
      for (size_t t = 0; t < p.jr_.size(); ++t) {
         const int r = p.jr_[t], c = p.jc_[t];
         if (c < p.oqx) {                        // a u column
            int e = -1;
            if (r >= p.rh2x && r < p.rh2y) e = r - p.rh2x;
            else if (r >= p.rh2y && r < p.rh3x) e = r - p.rh2y;
            if (e >= 0) { ++u_cols[e]; if (col_owner[c] < 0) ++u_bord[e]; }
         }
      }
      for (int e = 0; e < m_q; ++e) {
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

// The solution file is SELF-CONTAINED: header, per-level history, the reported
// iterate, and then the instance itself (N, stencil, sigma, u_clean, f). Carrying
// the data costs a few KB and means plot_2d.py needs nothing else — which matters
// most on the image route, where there is no .txt instance to point at and
// re-decoding the PNG in Python would silently plot a DIFFERENT noise realization.
static void save_solution(const std::string& fn, const Mpcc2DTNLP& p,
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
   // trailing instance block
   out << p.N << " " << (p.averaged ? 1 : 0) << " " << p.sigma_ << "\n";
   for (const std::vector<double>* v : {&p.uclean_, &p.f_})
      for (size_t i = 0; i < v->size(); ++i)
         out << (*v)[i] << (i + 1 < v->size() ? ' ' : '\n');
   // trailing μ-trace block (2026-07-23): count and column count, then one
   // (iter, μ, t, weight, max r(1−δ)) row per IPOPT iteration of the μ-coupled
   // single solve — the in-solve continuation path plot_2d.py draws in place
   // of the per-level one (weight/comp are NaN on restoration iterations).
   // count 0 for geometric runs; readers that predate the block ignore
   // trailing tokens.
   const size_t NC = 5;
   out << p.mu_hist_.size() / NC << " " << NC << "\n";
   for (size_t i = 0; i + NC - 1 < p.mu_hist_.size(); i += NC) {
      out << (long)p.mu_hist_[i];
      for (size_t j = 1; j < NC; ++j) out << " " << p.mu_hist_[i + j];
      out << "\n";
   }
}

static void self_check(Mpcc2DTNLP& p, const std::vector<int>& owner,
                       const std::vector<int>& col_owner, int nsub, int n_sub,
                       int n_promoted) {
   driver::print_checksums(p);

   int p_border = 0, mismatch = 0;
   std::vector<int> dims(n_sub, 0);
   for (int i = 0; i < p.kkt_dim; ++i) {
      if (owner[i] < 0) ++p_border;
      else dims[owner[i]]++;
      if (!p.file_owner_.empty() && p.file_owner_[i] != owner[i]) ++mismatch;
   }
   int nu = 0, nqx = 0, nqy = 0;
   for (int c = 0; c < p.n; ++c)
      if (col_owner[c] < 0) {
         if (c < p.oqx) ++nu;
         else if (c < p.oqy) ++nqx;
         else if (c < p.oR) ++nqy;
      }
   std::cout << "    partition    : p=" << p_border << "   block dims=[";
   for (int j = 0; j < n_sub; ++j) std::cout << dims[j] << (j + 1 < n_sub ? ", " : "");
   std::cout << "]\n";
   std::cout << "    complicating : u=" << nu << " qx=" << nqx << " qy=" << nqy
             << " alpha=1" << (n_promoted ? "  + " : "")
             << (n_promoted ? std::to_string(n_promoted) : "")
             << (n_promoted ? " promoted corner duals" : "") << "\n";
   if (p.file_owner_.empty())
      std::cout << "    owner vs Python: no reference in the data file\n";
   else if (nsub != p.file_nsub)
      std::cout << "    owner vs Python: SKIPPED (file was dumped for nsub="
                << p.file_nsub << ")\n";
   else if (n_promoted)
      std::cout << "    owner vs Python: " << mismatch << " differing entries — "
                << "expected exactly " << n_promoted
                << " (the promoted corner duals; Python does not promote)\n";
   else
      std::cout << "    owner vs Python: " << (mismatch ? "MISMATCH" : "identical")
                << " (" << mismatch << " differing entries)\n";
}

int main(int argc, char** argv) {
   std::string data, solver = "ma57", init = "file", save_sol, save_dd, save_data;
   std::string interface_solver = "direct", precond = "asd";
   double t0 = 1.0, tmin = 1e-4, factor = 0.85, tol = 1e-8, c_theta = 1.0;
   std::string cg_apply = "assembled";
   double wmax = 2e19, reg_alpha = 1e-4, sigma = 0.1, cg_tol = 1e-10;
   int nsub = 2, maxiter = 3000, printlevel = 0, size = 0, seed = 0, cg_maxit = 500;
   int minres_lag = 1;
   // μ-coupled is the DEFAULT since 2026-07-23 (measured 7–60× fewer iterations
   // at identical solutions on cameraman N=16/32 and mariposa N=128 k=8;
   // see README). `--t-update geometric` restores the per-level continuation —
   // needed to reproduce every table recorded before the flip, and the safer
   // mode on suspect instances: the single μ-coupled solve has NO
   // best-completed-level fallback on failure, and a bad α₀ can stall the
   // barrier with t pinned (the documented mariposa --alpha0 -2 mode).
   std::string t_update = "mu";
   double t_mu_scale = 10.0;
   std::string weight = "linear", stencil = "onesided", normalize = "255";
   // TILE is the default (k×k subdomains, the Python probe's geometry); its
   // cut-corner rank deficiency is handled by border promotion, and its
   // interface indefiniteness under --interface cg by the dual peel (also on by
   // default). STRIP (--partition strip) remains the no-cross-corner
   // alternative: SPD S with no promotion/peel, at the price of only k
   // subdomains and a wider border.
   std::string partition = "tile";
   bool check = false, nsub_set = false, dual_warm = false, promote = true;
   bool alpha_peel = true, dual_peel = true;

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
      else if (a == "--save-data") save_data = next();
      else if (a == "--self-check")  check = true;
      else if (a == "--dual-warmstart") dual_warm = true;
      else if (a == "--no-promote-corners") promote = false;
      else if (a == "--size")     size = std::stoi(next());
      else if (a == "--sigma")    sigma = std::stod(next());
      else if (a == "--seed")     seed = std::stoi(next());
      else if (a == "--weight")   weight = next();
      else if (a == "--stencil")  stencil = next();
      else if (a == "--normalize") normalize = next();
      else if (a == "--partition") partition = next();
      else if (a == "--interface") interface_solver = next();
      else if (a == "--precond")  precond = next();
      else if (a == "--cg-tol")   cg_tol = std::stod(next());
      else if (a == "--cg-max-iter") cg_maxit = std::stoi(next());
      else if (a == "--cg-apply") cg_apply = next();
      else if (a == "--minres-lag") minres_lag = std::stoi(next());
      else if (a == "--t-update")   t_update = next();
      else if (a == "--t-mu-scale") t_mu_scale = std::stod(next());
      else if (a == "--no-alpha-peel") alpha_peel = false;
      else if (a == "--no-dual-peel") dual_peel = false;
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (data.empty()) {
      std::cerr << "usage: dd_solve_2d --data <data/data_2d_N.txt|image.png> "
                   "[--size N] [--solver mumps|ma57|ma97|dd] [--nsub k]\n"
                   "                   [--self-check] [--save-solution FILE] "
                   "[--save-dd FILE] [--partition tile|strip]\n"
                   "                   [--interface direct|cg|minres] "
                   "[--precond jacobi|bj|asd] [--no-alpha-peel] ...\n"
                   "  --interface cg (needs --solver dd): preconditioned CG on the\n"
                   "    interface. On tile partitions the promoted corner duals make\n"
                   "    S indefinite; the DUAL PEEL (on by default) eliminates them as\n"
                   "    a dense Schur block so CG runs on the SPD complement\n"
                   "    (--no-dual-peel to A/B; strips need no peel).\n"
                   "  --interface minres (needs --solver dd): signed-MA57 MINRES on\n"
                   "    the FULL indefinite S — no SPD gate, no peel. Preconditioner\n"
                   "    L|D|L^T from a snapshot factorization of S, rebuilt every\n"
                   "    --minres-lag M factorizations (default 1 = every step, ~2\n"
                   "    its/solve); lag>1 prints the its-vs-age staleness table.\n"
                   "  --t-update mu|geometric (default mu): one solve with\n"
                   "    t = max(t_min, c*mu) slaved to the barrier (c = --t-mu-scale,\n"
                   "    default 10) vs the per-level geometric continuation\n"
                   "    (--t0/--factor; use it to reproduce pre-2026-07-23 tables and\n"
                   "    on suspect instances — it keeps a best-converged-level\n"
                   "    fallback that the single mu solve does not have).\n"
                   "  generate the data file first:\n"
                   "    python ../python/dump_data_2d.py --N 16 --nsub 2 "
                   "-o data/data_2d_16.txt\n"
                   "  and plot the result with:\n"
                   "    python ../python/plot_2d.py --solution sol.txt "
                   "--save-plot sol.png\n";
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
   if (minres_lag < 1) {
      std::cerr << "--minres-lag must be >= 1\n";
      return 2;
   }
   if (t_update != "geometric" && t_update != "mu") {
      std::cerr << "--t-update must be geometric|mu\n";
      return 2;
   }
   if (t_mu_scale <= 0) {
      std::cerr << "--t-mu-scale must be > 0\n";
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
   // These three used to be bare string compares — a typo silently selected
   // the default arm of a settled A/B (e.g. `--stencil averagd` ran onesided
   // while the run was recorded as averaged).
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

   // An image path needs --size (and takes --sigma/--seed/--weight/--stencil);
   // a .txt dump carries all of that already and ignores them.
   image_io::Opts iopt;
   iopt.size = size; iopt.sigma = sigma; iopt.seed = (unsigned)seed;
   iopt.minmax = (normalize == "minmax");
   // area-average when downsampling, matching PIL's reducing BILINEAR — without it
   // a 512→16 crop aliases badly and the instance is materially different from
   // Python's (measured: 0.0457 / +2.55 dB vs 0.0744 / +4.68 dB)
   iopt.area_downsample = true;
   // same detector the TNLP constructor uses — the two must never disagree
   const bool from_image = !image_io::ends_with(data, ".txt");
   if (from_image && size <= 0) {
      std::cerr << "an image input needs --size N (the target side length)\n";
      return 2;
   }
   SmartPtr<Mpcc2DTNLP> mpcc =
      new Mpcc2DTNLP(data, iopt, weight == "exp", stencil == "averaged");
   mpcc->w_max_ = wmax;
   mpcc->reg_alpha_ = reg_alpha;
   if (!nsub_set && mpcc->file_nsub > 0) nsub = mpcc->file_nsub;
   if (nsub < 1 || nsub > mpcc->nc) { std::cerr << "need 1 <= nsub <= N-1\n"; return 2; }
   const double w0 = 0.7 * mpcc->sigma_;      // the noise-aware default weight
   if (from_image && init == "file") init = "cp";   // no dumped warm start to use
   if (mpcc->x_start_.empty()) mpcc->x_start_.assign(mpcc->n, 0.0);
   if (init == "cold") mpcc->cold_start(mpcc->x_start_.data(), w0);
   else if (init == "cp") mpcc->cp_start(mpcc->x_start_.data(), w0);
   else if (init != "file") {
      std::cerr << "--init must be file|cp|cold\n";
      return 2;
   }
   if (!save_dd.empty() && solver != "dd") {
      std::cerr << "--save-dd needs --solver dd (there is no arrowhead otherwise)\n";
      return 2;
   }
   mpcc->set_theta_ref(mpcc->x_start_.data());
   mpcc->t_ = t0;

   Partition2D part(mpcc->N, nsub, partition == "strip");
   std::vector<int> col_owner;
   int n_promoted = 0;
   std::vector<int> owner = kkt_owner(*mpcc, part, promote, &col_owner, &n_promoted);

   std::cout << "2D lifted TV-MPCC (staggered, C++)  N=" << mpcc->N
             << "  nodes=" << mpcc->m_u << "  cells=" << mpcc->m_q
             << "  n=" << mpcc->n << "  m_con=" << mpcc->mcon
             << " (" << mpcc->n_eq << " eq + " << mpcc->n_ineq << " ineq)"
             << "  KKT dim=" << mpcc->kkt_dim << "\n";
   std::cout << "  stencil=" << (mpcc->averaged ? "averaged" : "onesided")
             << "  Q(a) = " << (mpcc->weight_exp ? "e^a" : "a")
             << (mpcc->has_ha ? "  (+ row ha: a >= 0)" : "  (a boxed)")
             << "   init=" << init << "   solver=" << solver;
   if (solver == "dd") {
      if (part.striped) std::cout << "  partition=" << nsub << " strips";
      else std::cout << "  nsub=" << nsub << "x" << nsub << " tiles";
      if (n_promoted) std::cout << "  +" << n_promoted << " promoted corner duals";
      else if (!promote) std::cout << "  (corner promotion OFF)";
   }
   if (solver == "dd" && interface_solver == "cg")
      std::cout << "  interface=cg(" << precond
                << (alpha_peel ? ",alpha-peel" : ",no-alpha-peel")
                << (dual_peel ? ",dual-peel" : ",no-dual-peel")
                << ",tol=" << cg_tol << ",maxit=" << cg_maxit << ")";
   if (solver == "dd" && interface_solver == "minres")
      std::cout << "  interface=minres(signed-ma57,lag=" << minres_lag
                << ",tol=" << cg_tol << ",maxit=" << cg_maxit << ")";
   if (t_update == "mu")
      std::cout << "  t-update=mu(c=" << t_mu_scale << ")";
   if (c_theta > 0) std::cout << "  c_theta=" << c_theta;
   std::cout << "\n";

   // --save-data: write the instance THIS RUN actually loaded, in
   // dump_data_2d.py's format, so (a) plot_2d.py works for an image run too and
   // (b) the C++-decoded image is inspectable/diffable against Python's.
   if (!save_data.empty()) {
      std::ofstream out(save_data);
      if (!out) { std::cerr << "cannot write " << save_data << "\n"; return 1; }
      out << std::setprecision(17);
      out << mpcc->N << " " << mpcc->n << " " << mpcc->mcon << " " << mpcc->kkt_dim
          << " " << nsub << " " << (mpcc->weight_exp ? 1 : 0) << " "
          << (mpcc->averaged ? 1 : 0) << " " << mpcc->sigma_ << " 0\n";
      const std::vector<double>* blocks[3] = {&mpcc->uclean_, &mpcc->f_,
                                              &mpcc->x_start_};
      for (const auto* v : blocks) {
         for (size_t i = 0; i < v->size(); ++i)
            out << (*v)[i] << (i + 1 < v->size() ? ' ' : '\n');
      }
      // the owner map WITHOUT the promotions, matching what Python's kkt_owner
      // produces, so the file stays a drop-in for the Python-side tools
      std::vector<int> plain = kkt_owner(*mpcc, part, false);
      for (size_t i = 0; i < plain.size(); ++i)
         out << plain[i] << (i + 1 < plain.size() ? ' ' : '\n');
      std::cout << "  wrote " << save_data << "  (the instance as loaded here)\n";
   }

   if (check) {
      self_check(*mpcc, owner, col_owner, nsub, part.n_sub, n_promoted);
      return 0;
   }

   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
   if (!driver::init_app(app, printlevel, maxiter, solver)) return 1;

   if (solver == "dd") {
      DDArrowheadSolver::config_owner(owner, part.n_sub);
      DDArrowheadSolver::config_dump_arrow(save_dd);
      DDArrowheadSolver::config_interface(
         interface_solver == "cg"     ? DDArrowheadSolver::IFACE_CG
         : interface_solver == "minres" ? DDArrowheadSolver::IFACE_MINRES
                                        : DDArrowheadSolver::IFACE_DIRECT,
         precond == "jacobi" ? DDArrowheadSolver::PRECOND_JACOBI
         : precond == "bj"   ? DDArrowheadSolver::PRECOND_BJ
                             : DDArrowheadSolver::PRECOND_ASD,
         alpha_peel, cg_tol, cg_maxit);
      DDArrowheadSolver::config_minres(minres_lag);
      DDArrowheadSolver::config_cg_apply(
         cg_apply == "matfree" ? DDArrowheadSolver::APPLY_MATFREE
                               : DDArrowheadSolver::APPLY_ASSEMBLED);
      DDArrowheadSolver::config_alpha(mpcc->oa);
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

   driver::RunResult res =
      (t_update == "mu")
         ? driver::run_mu_coupled(app, *mpcc, tmin, t_mu_scale, c_theta, tol,
                                  printlevel, optimize)
         : driver::run_scholtes(app, *mpcc,
                                driver::scholtes_schedule(t0, tmin, factor),
                                c_theta, tol, /*warm_start=*/dual_warm,
                                /*value_is_alpha=*/false, optimize);
   driver::print_summary(*mpcc, res);
   if (solver == "dd" && interface_solver == "cg")
      driver::print_interface_stats(precond, alpha_peel, cg_tol);
   if (solver == "dd" && interface_solver == "minres")
      driver::print_minres_stats(cg_tol, minres_lag);

   if (!res.best_x.empty()) {
      if (!save_sol.empty()) {
         save_solution(save_sol, *mpcc, res.hist, res.best_x, res.best_t, nsub);
         std::cout << "  wrote " << save_sol << "\n";
      }
      if (!save_dd.empty())
         std::cout << "  wrote " << save_dd << "  (arrowhead of the LAST Newton step)\n";
   }
   return 0;
}
