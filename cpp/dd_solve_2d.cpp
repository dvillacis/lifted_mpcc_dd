// Scholtes-continuation driver for the STAGGERED 2D lifted TV-MPCC, with the
// arrowhead domain decomposition as IPOPT's actual linear solver.
//
//   --solver mumps|ma57|ma97   IPOPT's own linear solver (the monolithic reference,
//                         validates the TNLP port against ../lifted_mpcc_2d.py)
//   --solver dd --nsub k  the DD arrowhead solver — k×k tiles, every Newton system
//                         factorized and solved by Σ_k W_k + the interface S, every
//                         inertia query answered by Haynsworth
//   --solver ddsimple     the SAME decomposition in dd_solver_simple.hpp: Eigen
//                         only (no HSL, no MUMPS), interface solved by CG, and
//                         written to be read. Use it to understand the method or
//                         on a machine without MA57; --solver dd stays the
//                         production route.
//
// The 2D sibling of dd_solve_1d.cpp; see that file for the design. Two things are
// 2D-specific:
//
//   * **the anchor rule** — node (i,j) belongs to the tile of cell (i,j),
//     clamped: the cell the node anchors under the one-sided (forward-difference)
//     stencil (cell (i−1,j−1) before the 2026-10-02 stencil switch). Measured in
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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpTNLPAdapter.hpp"
// The MA57 production solver is optional at build time: without HSL the
// build defines nothing and --solver dd is rejected at runtime, while
// --solver ddsimple (and IPOPT's own mumps) still work. build.sh sets
// DD_HAVE_MA57 iff it found the library.
#ifdef DD_HAVE_MA57
#include "dd_solver.hpp"
#endif
// The readable, HSL-free twin of dd_solver.hpp (--solver ddsimple): Eigen
// LDLᵀ blocks + conjugate gradients on the interface. See its header comment.
#include "dd_solver_simple.hpp"
#include "driver_common.hpp"
#include "mpcc_2d_tnlp.hpp"
// The consensus (duplicate-and-link) reformulation — Lueg's structure imposed
// on the same MPCC, selected by --formulation consensus. Same solutions; a
// different KKT sparsity in which the border is purely primal.
#include "mpcc_2d_consensus_tnlp.hpp"
#include "verify_wk.hpp"
// struct Partition2D — the tile/strip cell ownership and the anchor rule, now
// shared with dd_solve_dataset.cpp (which partitions each training pair with it).
#include "partition_2d.hpp"
// Dependency-free NumPy .npz writer — the structured solution format.
#include "npz_writer.hpp"

using namespace Ipopt;

// Label every KKT index with its subdomain (−1 = border), mirroring
// ../lifted_mpcc_2d.py's kkt_owner. Rows are never duplicated (the scalar ha row
// goes to tile 0 — its slack and multiplier are dual directions, and bordering them
// would make S indefinite by construction and defeat the δ_w loop). Only primal
// columns can be complicating, and which ones is read off the Jacobian sparsity.
// rank1_rows_out, when non-null, receives the KKT indices of the rank-1 pair
// multipliers this routine WOULD promote, whether or not promote_corners is on.
// That is what --border-reg regularizes in place instead of bordering (see
// Arrowhead::Options::border_reg): same detection, same rows, other cure.
static std::vector<int> kkt_owner(const Mpcc2DTNLP& p, const Partition2D& part,
                                  bool promote_corners = true,
                                  std::vector<int>* col_owner_out = nullptr,
                                  int* n_promoted = nullptr,
                                  std::vector<int>* rank1_rows_out = nullptr) {
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
   if (promote_corners || rank1_rows_out) {
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
      // The pair (h3x,h3y) restricted to W_k keeps only its (δ,θ) columns,
      //     [ −cosθ   δ sinθ ]
      //     [ −sinθ  −δ cosθ ],   det = δ,
      // so it collapses to rank 1 exactly as δ → 0; (h2x,h2y) does the same in
      // (r,θ) with det = r.  The deficiency is NUMERICAL, not structural — the
      // rows are fully populated — which is why a pattern test cannot find it
      // and the detection has to be this geometric one.
      auto mark = [&](int idx) {
         if (promote_corners) owner[idx] = -1;
         if (rank1_rows_out) rank1_rows_out->push_back(idx);
      };
      for (int e = 0; e < m_q; ++e) {
         if (col_owner[p.oqx + e] < 0 && col_owner[p.oqy + e] < 0) {
            mark(lam_c0 + p.rh3x + e);           // δ≈0 rank-1 pair
            mark(lam_c0 + p.rh3y + e);
            promoted += 2;
         }
         if (u_cols[e] > 0 && u_bord[e] == u_cols[e]) {
            mark(lam_c0 + p.rh2x + e);           // r≈0 rank-1 pair
            mark(lam_c0 + p.rh2y + e);
            promoted += 2;
         }
      }
      if (!promote_corners) promoted = 0;
   }
   if (col_owner_out) *col_owner_out = col_owner;
   if (n_promoted) *n_promoted = promoted;
   return owner;
}

// ---- level-2 (nested) owner map: quadrants of the tile grid -----------------
// The nested interface needs the border split into groups whose removal of a
// small separator DISCONNECTS them in S. S's graph is a union of cliques, one
// per subdomain's border set N_k, so the condition is simply: every fine tile
// must lie wholly inside one group. Grouping fine tiles into the four quadrants
// of the k x k grid satisfies that, and the entities on the two middle cut lines
// — the ones that span quadrants — fall out as the separator automatically.
//
// Built by reusing kkt_owner itself against a coarsened partition, so the
// "spans >= 2 owners => complicating" rule is the SAME validated code that
// produces the outer map (alpha included: it is global, hence separator).
// NestedInterface::analyze re-checks the disconnection property and refuses if
// this geometry ever fails to deliver it.
static std::vector<int> kkt_owner2(const Mpcc2DTNLP &p, const Partition2D &part,
                                   int *ngroups) {
  const int k = part.k;
  if (k < 2) { *ngroups = 0; return {}; }
  Partition2D q = part;                     // copy: same bounds, coarser owners
  const int half = k / 2;
  std::vector<int> quad(part.n_sub, 0);
  for (int a = 0; a < k; ++a)
    for (int c = 0; c < k; ++c)
      quad[a * k + c] = (a >= half ? 2 : 0) + (c >= half ? 1 : 0);
  for (auto &o : q.cell_owner) o = quad[o];
  for (auto &o : q.node_owner) o = quad[o];
  q.n_sub = 4;
  *ngroups = 4;
  return kkt_owner(p, q, /*promote_corners=*/false);
}

// The solution file is SELF-CONTAINED: header, per-level history, the reported
// iterate, and then the instance itself (N, stencil, sigma, u_clean, f). Carrying
// the data costs a few KB and means plot_2d.py needs nothing else — which matters
// most on the image route, where there is no .txt instance to point at and
// re-decoding the PNG in Python would silently plot a DIFFERENT noise realization.
// ---------------------------------------------------------------------------
//  SOLUTION OUTPUT.  Two formats, chosen by the extension of --save-solution:
//
//    .npz  (default, recommended)  a NumPy archive of NAMED, SHAPED arrays —
//          self-describing, exact (raw IEEE-754), ~3x smaller than the text
//          form, and readable by np.load / MATLAB / Julia / R.
//    .txt  the original positional token stream, kept so existing result
//          directories and any external scripts keep working.
//
//  Both carry the same content: the solution, the instance it was solved on,
//  the per-level continuation history and the μ-trace — self-contained, so a
//  plot needs nothing but this one file.
// ---------------------------------------------------------------------------
static void save_solution_npz(const std::string& fn, const Mpcc2DTNLP& p,
                              const std::vector<driver::Level>& hist,
                              const std::vector<double>& x, double t_last,
                              int nsub) {
   // The consensus formulation carries local copies past the original layout;
   // the leading n_orig entries are the consensus values, which at a feasible
   // point equal every copy — i.e. exactly the original-formulation solution.
   const Mpcc2DConsensusTNLP* cons = dynamic_cast<const Mpcc2DConsensusTNLP*>(&p);
   const int nw = cons ? cons->n_orig : p.n;
   const size_t mu = (size_t)p.m_u, mq = (size_t)p.m_q, nc = (size_t)p.nc;
   const size_t Nn = (size_t)p.N;

   npz::Writer w(fn);
   if (!w.ok()) { std::cerr << "cannot write " << fn << "\n"; return; }

   // -- what this run was ------------------------------------------------
   w.scalar_i("N", p.N);
   w.scalar_i("nsub", nsub);
   w.scalar_i("n_var", nw);
   w.scalar_d("sigma", p.sigma_);
   w.scalar_d("t_last", t_last);
   w.scalar_i("weight_exp", p.weight_exp ? 1 : 0);
   w.scalar_i("averaged", p.averaged ? 1 : 0);
   w.scalar_i("collocated", p.collocated ? 1 : 0);   // cell arrays are N×N then
   w.scalar_i("consensus", cons ? 1 : 0);

   // -- the instance (so the file is self-contained) ---------------------
   w.array_d("u_clean", p.uclean_.data(), {Nn, Nn});
   w.array_d("f", p.f_.data(), {Nn, Nn});

   // -- the solution, as SHAPED, NAMED fields rather than one flat vector --
   w.array_d("u", x.data() + p.ou, {Nn, Nn});
   w.array_d("qx", x.data() + p.oqx, {nc, nc});
   w.array_d("qy", x.data() + p.oqy, {nc, nc});
   w.array_d("r", x.data() + p.oR, {nc, nc});
   w.array_d("delta", x.data() + p.oD, {nc, nc});
   w.array_d("theta", x.data() + p.oTh, {nc, nc});
   w.scalar_d("alpha", x[p.oa]);
   w.scalar_d("weight", p.Q(x[p.oa]));
   // the raw primal vector too: authoritative, and what a warm start needs
   w.array_d("x", x.data(), {(size_t)nw});

   // -- the continuation history, one row per level ----------------------
   std::vector<double> lv;
   lv.reserve(hist.size() * 8);
   for (const driver::Level& l : hist) {
      lv.push_back(l.t); lv.push_back((double)l.status); lv.push_back((double)l.iters);
      lv.push_back(l.comp_res); lv.push_back(l.weight); lv.push_back(l.obj);
      lv.push_back(l.xi_max); lv.push_back(l.converged ? 1.0 : 0.0);
   }
   w.array_d("levels", lv.data(), {hist.size(), 8});
   // and the in-solve μ-trace of the μ-coupled route
   w.array_d("mu_trace", p.mu_hist_.data(), {p.mu_hist_.size() / 5, 5});

   (void)mu; (void)mq;
   if (!w.close()) std::cerr << "failed writing " << fn << "\n";
}

static void save_solution_txt(const std::string& fn, const Mpcc2DTNLP& p,
                          const std::vector<driver::Level>& hist,
                          const std::vector<double>& x, double t_last, int nsub) {
   std::ofstream out(fn);
   if (!out) { std::cerr << "cannot write " << fn << "\n"; return; }
   // The consensus formulation carries local copies past the original layout
   // (see mpcc_2d_consensus_tnlp.hpp). Write only the CONSENSUS variables —
   // at a feasible point every copy equals the consensus value it links to, so
   // the leading n_orig entries ARE the original-formulation solution vector.
   // Truncating here keeps the file format identical for both formulations, so
   // python/plot_slurm.py and everything downstream read them unchanged (that
   // reader validates n against N and would otherwise reject a consensus run).
   const Mpcc2DConsensusTNLP* cons =
      dynamic_cast<const Mpcc2DConsensusTNLP*>(&p);
   const int nw = cons ? cons->n_orig : p.n;
   out << std::setprecision(17);
   out << nw << " " << hist.size() << " " << t_last << " " << nsub << " "
       << (p.weight_exp ? 1 : 0) << "\n";
   for (const driver::Level& l : hist)
      out << l.t << " " << l.status << " " << l.iters << " " << l.comp_res << " "
          << l.weight << " " << l.obj << " " << l.xi_max << " "
          << (l.converged ? 1 : 0) << "\n";
   for (int i = 0; i < nw; ++i) out << x[i] << (i + 1 < nw ? ' ' : '\n');
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
   // --solver defaults to ddsimple: the Eigen-only twin is the one route that
   // needs no HSL and no MUMPS, so the default works on every machine the
   // binary builds on (HSL is optional — see build_linux.sh). --solver dd is
   // the HSL-backed production path and must be asked for by name.
   std::string data, solver = "ddsimple", init = "file", save_sol, save_dd, save_data;
   std::string interface_solver = "direct", precond = "asd";
   std::string wk_backend = "ma57";
   double t0 = 1.0, tmin = 1e-4, factor = 0.85, tol = 1e-8, c_theta = 1.0;
   std::string cg_apply = "assembled";
   double wmax = 2e19, reg_alpha = 1e-4, sigma = 0.1, cg_tol = 1e-10;
   // Peel-cache columns do not need the interface solve's 1e-10; 1e-7 is
   // the measured knee (see Arrowhead::Options::peel_cg_tol).
   double peel_cg_tol = 1e-7;
   // Regularize the structurally empty W_k diagonals instead of promoting the
   // corner duals to the border (Arrowhead::Options::border_reg).  Pair it
   // with --no-promote-corners; 0 = off.
   double border_reg = 0.0;
   std::vector<int> rank1_rows;   // filled by kkt_owner when --border-reg is on
   // --sff-solver cg|direct|direct-all (ddsimple only): the CG-vs-Eigen A/B
   // for the systems with S_ff (Arrowhead::Options::sff_direct).  `direct`
   // takes the peel columns Z through a sparse LDLT of the assembled S_ff;
   // `direct-all` the interface solve too.
   std::string sff_solver = "cg";
   int nsub = 2, maxiter = 3000, printlevel = 0, size = 0, seed = 0, cg_maxit = 500;
   int minres_lag = 1;
   int schur_lag = 1;
   std::string schur_mode = "forward";
   bool nested = false;
   std::string hessian = "limited-memory";
   std::string ma57_scaling = "off";
   // μ-coupled is the DEFAULT since 2026-07-23 (measured 7–60× fewer iterations
   // at identical solutions on cameraman N=16/32 and mariposa N=128 k=8;
   // see README). `--t-update geometric` restores the per-level continuation —
   // needed to reproduce every table recorded before the flip, and still the
   // safer mode on suspect instances: a bad α₀ can stall the barrier with t
   // pinned (the documented mariposa --alpha0 -2 mode). Since 2026-07-24 a
   // failed μ-coupled solve falls back to the deepest BANKED level (one banked
   // per monotone μ drop, smallest max r(1−δ) kept — see mpcc_base.hpp), so a
   // tail failure reports/saves the tightest converged level instead of
   // nothing; a stall before the FIRST μ drop still reports nothing.
   std::string t_update = "mu";
   double t_mu_scale = 10.0;
   std::string weight = "linear", stencil = "onesided", normalize = "255";
   // --grid staggered|collocated (2026-10-02): where q, r, δ, θ live. staggered
   // (default, every validated run) puts them on the (N−1)² cells between the
   // pixels; collocated puts them on the N² pixels with u, using the
   // Neumann forward difference (zero K rows on the last column/row) — see the
   // GRIDS note in mpcc_2d_tnlp.hpp. Image route only.
   std::string grid = "staggered";
   // TILE is the default (k×k subdomains, the Python probe's geometry); its
   // cut-corner rank deficiency is handled by border promotion, and its
   // interface indefiniteness under --interface cg by the dual peel (also on by
   // default). STRIP (--partition strip) remains the no-cross-corner
   // alternative: SPD S with no promotion/peel, at the price of only k
   // subdomains and a wider border.
   std::string partition = "tile";
   bool check = false, nsub_set = false, dual_warm = false, promote = true;
   bool alpha_peel = true, dual_peel = true;
   // --no-cross-peel (ddsimple only): stop making the cross points — border
   // unknowns touched by >=3 subdomains — primal. They are the FETI-DP corner
   // set; peeling them is on by default because it is what makes tile
   // partitions scale (see dd_solver_simple.hpp §6a). Use this to A/B.
   bool cross_peel = true;
   // --formulation permutation|consensus: decompose the monolithic KKT by pure
   // permutation (default, the validated route), or rebuild the NLP in Lueg's
   // duplicate-and-link form so every complicating variable is primal and
   // appears only in linear linking rows (see mpcc_2d_consensus_tnlp.hpp).
   std::string formulation = "permutation";
   // --objective consensus|copies (consensus formulation only): where the
   // upper-level objective is evaluated.  consensus (default) leaves it on the
   // original indices, so the corner block C is a PSD diagonal; copies
   // distributes it over the local copies (1/|T(u_i)| per border node,
   // 1/n_tiles for the alpha ridge) and moves the alpha box with it, giving
   // strict Lueg form, C = delta_w*I. Same problem either way -- the weights
   // make the two objectives agree wherever the linking rows hold.
   std::string objective = "consensus";
   // --drop-corner-reg (needs --solver ddsimple, --formulation consensus,
   // --objective copies, --hessian exact): in that Lueg form the corner block C
   // that IPOPT hands over is exactly delta_w*I in every regular iteration;
   // replace it by Lueg's C = 0 there (see Options::drop_corner in
   // dd_solver_simple.hpp). Restoration iterations are left alone.
   bool drop_corner_reg = false;
   // --block-dual-reg (needs --solver ddsimple): regularize a W_k that fails to
   // factorize with -eps_k on ITS dual diagonal only, instead of reporting
   // SINGULAR and letting IPOPT switch on a global delta_c for every block
   // (Lueg et al. 2026 §3.5; see Options::dual_start in dd_solver_simple.hpp).
   bool block_dual_reg = false;
   // --wk-reg-h <d> / --wk-reg-c <d> (need --solver ddsimple): fixed W_k-only
   // regularization, Lueg's delta_H on the primal and delta_C on the
   // non-linking multiplier diagonals of every subdomain block, independent of
   // IPOPT's delta_w/delta_c (see Options::wk_reg_h in dd_solver_simple.hpp).
   double wk_reg_h = 0.0, wk_reg_c = 0.0;
   // --block-inertia (needs --solver ddsimple): Lueg's per-block inertia
   // correction — each W_k gets its own delta_H^k / delta_C^k until it has the
   // right inertia; IPOPT's global delta_w only when S is not PD (see
   // Options::block_inertia in dd_solver_simple.hpp).
   bool block_inertia = false;
   // --block-solver sparse|dense (ddsimple): dense = Bunch–Kaufman (LAPACK)
   // W_k and S with EXACT inertia, direct interface solve — the reference
   // route (see Options::dense_blocks in dd_solver_simple.hpp).
   std::string block_solver = "sparse";
   // --interface-inertia (needs --block-solver dense --block-inertia): shift
   // C by delta_S until S is PD instead of reporting a wrong inertia to IPOPT.
   bool interface_inertia = false;
   // --solver ddsimple has no --interface/--inertia knobs left: it is CG on
   // the peeled interface with the PREDICTED inertia, full stop (see the
   // dd_solver_simple.hpp header).  We only need to know whether --interface
   // was actually typed, to reject a contradictory request.
   bool interface_set = false;

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
      else if (a == "--grid")     grid = next();
      else if (a == "--normalize") normalize = next();
      else if (a == "--partition") partition = next();
      else if (a == "--interface") { interface_solver = next(); interface_set = true; }
      else if (a == "--precond")  precond = next();
      else if (a == "--wk-backend") wk_backend = next();
      else if (a == "--cg-tol")   cg_tol = std::stod(next());
      else if (a == "--peel-cg-tol") peel_cg_tol = std::stod(next());
      else if (a == "--border-reg") border_reg = std::stod(next());
      else if (a == "--sff-solver") sff_solver = next();
      else if (a == "--cg-max-iter") cg_maxit = std::stoi(next());
      else if (a == "--cg-apply") cg_apply = next();
      else if (a == "--minres-lag") minres_lag = std::stoi(next());
      else if (a == "--schur-lag")  schur_lag = std::stoi(next());
      else if (a == "--schur")      schur_mode = next();
      else if (a == "--nested")     nested = true;
      else if (a == "--hessian")   hessian = next();
      else if (a == "--ma57-scaling") ma57_scaling = next();
      else if (a == "--t-update")   t_update = next();
      else if (a == "--t-mu-scale") t_mu_scale = std::stod(next());
      else if (a == "--no-alpha-peel") alpha_peel = false;
      else if (a == "--no-dual-peel") dual_peel = false;
      else if (a == "--no-cross-peel") cross_peel = false;
      else if (a == "--formulation") formulation = next();
      else if (a == "--objective") objective = next();
      else if (a == "--drop-corner-reg") drop_corner_reg = true;
      else if (a == "--block-dual-reg") block_dual_reg = true;
      else if (a == "--wk-reg-h") wk_reg_h = std::stod(next());
      else if (a == "--wk-reg-c") wk_reg_c = std::stod(next());
      else if (a == "--block-inertia") block_inertia = true;
      else if (a == "--block-solver") block_solver = next();
      else if (a == "--interface-inertia") interface_inertia = true;
      else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
   }
   if (data.empty()) {
      std::cerr << "usage: dd_solve_2d --data <cpp2/data_2d_N.txt|image.png> "
                   "[--size N] [--solver mumps|ma57|ma97|dd|ddsimple "
                   "(default ddsimple)] "
                   "[--nsub k]\n"
                   "                   [--self-check] [--save-solution FILE] "
                   "[--save-dd FILE] [--partition tile|strip]\n"
                   "                   [--interface direct|cg|minres] "
                   "[--precond jacobi|bj|asd] [--no-alpha-peel]\n"
                   "                   [--wk-backend ma57|mumps] ...\n"
                   "  --wk-backend mumps (needs --solver dd, a MUMPS-enabled "
                   "build): each W_k is\n"
                   "    factorized by MUMPS with the partial-factorization "
                   "Schur — S_k comes out\n"
                   "    of the factorization instead of p_k backsolves. "
                   "Validate with ./mumps_smoke\n"
                   "    and DD_CHECK=1 on any new machine.\n"
                   "  --solver ddsimple: the same decomposition implemented in\n"
                   "    dd_solver_simple.hpp — Eigen only (no HSL/MUMPS), written to be\n"
                   "    READ. One configuration: the interface matrix S is never\n"
                   "    assembled or factorized — ASd-preconditioned CG on the peeled\n"
                   "    interface, In(S) PREDICTED as In(T) from the tiny dense peel\n"
                   "    complement. Honours --nsub, --partition, --cg-tol,\n"
                   "    --cg-max-iter, --no-alpha-peel, --no-dual-peel and\n"
                   "    --no-cross-peel.\n"
                   "  --block-dual-reg (ddsimple): a W_k that fails to factorize is\n"
                   "    regularized on its own dual diagonal (-eps_k), instead of\n"
                   "    IPOPT's global delta_c on every block.\n"
                   "  --wk-reg-h d, --wk-reg-c d (ddsimple): fixed W_k-only\n"
                   "    regularization (Lueg): +d on the primal, -d on the non-linking\n"
                   "    multiplier diagonals of every W_k; IPOPT's matrix is unchanged.\n"
                   "  --block-inertia (ddsimple): Lueg's per-block inertia correction,\n"
                   "    delta_H^k / delta_C^k per W_k until In(W_k) is right; IPOPT's\n"
                   "    global delta_w only when the Schur complement is not PD.\n"
                   "  --block-solver sparse|dense (ddsimple, default sparse): dense =\n"
                   "    Bunch-Kaufman W_k and S (LAPACK), exact inertia, direct\n"
                   "    interface solve; the reference route, O(n^3) per block.\n"
                   "  --drop-corner-reg (ddsimple, consensus, --objective copies,\n"
                   "    --hessian exact): use Lueg's C = 0 for the corner block where\n"
                   "    IPOPT's is exactly delta_w*I (regular iterations only).\n"
                   "  --formulation permutation|consensus (default permutation):\n"
                   "    permutation decomposes the monolithic KKT in place; consensus\n"
                   "    rebuilds the NLP in Lueg's duplicate-and-link form (one local\n"
                   "    copy per shared variable per tile + linear linking rows), so\n"
                   "    the border is purely primal: no promoted duals, S SPD after\n"
                   "    inertia correction. Same solutions (checked vs mumps); needs\n"
                   "    --solver ddsimple or mumps.\n"
                   "  --no-cross-peel (needs --solver ddsimple): stop making the CROSS\n"
                   "    POINTS primal. A cross point is a border unknown touched by >=3\n"
                   "    subdomains -- (k-1)^2 of them on a k x k tile partition. Making\n"
                   "    them primal is the FETI-DP corner rule and is ON by default: it\n"
                   "    is a no-op on strips (which have none) and worth 1.7-4.3x on\n"
                   "    tiles, and N=64 4x4 does not converge without it.\n"
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
                   "  --grid staggered|collocated (default staggered): where\n"
                   "    q, r, delta, theta live. collocated puts them on the N^2\n"
                   "    pixels with u (Neumann forward differences, zero K rows on\n"
                   "    the last column/row); image input only, --stencil onesided,\n"
                   "    no --save-data, --save-solution must be .npz.\n"
                   "  generate the data file first:\n"
                   "    uv run python cpp2/dump_data_2d.py --N 16 --nsub 2 "
                   "-o cpp2/data_2d_16.txt\n"
                   "  --save-solution FILE: .npz (recommended) writes a NumPy\n"
                   "    archive of named, shaped arrays; .txt writes the legacy\n"
                   "    positional text. Both are self-contained (solution +\n"
                   "    instance + continuation history + mu-trace).\n"
                   "  and plot the result with (seven panels per solution):\n"
                   "    ./dd_solve_2d ... --save-solution runs/sols/sol_TAG.npz\n"
                   "    (cd ../python && uv sync && uv run python plot_slurm.py "
                   "../cpp/runs)\n";
      return 2;
   }
   if (solver != "mumps" && solver != "ma57" && solver != "ma97" &&
       solver != "dd" && solver != "ddsimple") {
      std::cerr << "--solver must be mumps|ma57|ma97|dd|ddsimple\n";
      return 2;
   }
#ifndef DD_HAVE_MA57
   if (solver == "dd") {
      std::cerr << "--solver dd needs MA57, and this binary was built without "
                   "HSL; use --solver ddsimple (Eigen only) or rebuild with "
                   "HSLDIR set\n";
      return 2;
   }
#endif
   if (partition != "tile" && partition != "strip") {
      std::cerr << "--partition must be tile|strip\n";
      return 2;
   }
   if (interface_solver != "direct" && interface_solver != "cg" &&
       interface_solver != "minres") {
      std::cerr << "--interface must be direct|cg|minres\n";
      return 2;
   }
   if (solver == "ddsimple" && interface_set && interface_solver != "cg") {
      std::cerr << "--solver ddsimple only implements --interface cg (its "
                   "direct route was removed with the assembled S)\n";
      return 2;
   }
   if (solver != "ddsimple" && interface_solver != "direct" && solver != "dd") {
      std::cerr << "--interface " << interface_solver
                << " needs --solver dd (it replaces the arrowhead's "
                   "interface solve)\n";
      return 2;
   }
   if (formulation != "permutation" && formulation != "consensus") {
      std::cerr << "--formulation must be permutation|consensus\n";
      return 2;
   }
   if (formulation == "consensus") {
      if (solver != "ddsimple" && solver != "mumps") {
         std::cerr << "--formulation consensus supports --solver ddsimple (the "
                      "decomposition) and mumps (the monolithic equivalence "
                      "reference)\n";
         return 2;
      }
      if (check || !save_data.empty() || !save_dd.empty()) {
         std::cerr << "--formulation consensus does not support --self-check/"
                      "--save-data/--save-dd (they assume the permutation "
                      "layout)\n";
         return 2;
      }
   }
   if (objective != "consensus" && objective != "copies") {
      std::cerr << "--objective must be consensus|copies\n";
      return 2;
   }
   if (block_dual_reg && solver != "ddsimple") {
      std::cerr << "--block-dual-reg needs --solver ddsimple (it changes how the "
                   "W_k blocks are factorized)\n";
      return 2;
   }
   if ((wk_reg_h != 0.0 || wk_reg_c != 0.0) &&
       (solver != "ddsimple" || wk_reg_h < 0.0 || wk_reg_c < 0.0)) {
      std::cerr << "--wk-reg-h/--wk-reg-c need --solver ddsimple and values >= 0\n";
      return 2;
   }
   if (block_solver != "sparse" && block_solver != "dense") {
      std::cerr << "--block-solver must be sparse|dense\n";
      return 2;
   }
   if (block_solver == "dense") {
#ifndef DD_HAVE_LAPACK
      std::cerr << "--block-solver dense needs LAPACK (rebuild: build.sh links OpenBLAS "
                   "when Homebrew has it; build_linux.sh links the env's liblapack)\n";
      return 2;
#endif
      if (solver != "ddsimple" || block_dual_reg || border_reg > 0.0) {
         std::cerr << "--block-solver dense needs --solver ddsimple and excludes "
                      "--block-dual-reg / --border-reg\n";
         return 2;
      }
   }
   if (interface_inertia && (block_solver != "dense" || !block_inertia)) {
      std::cerr << "--interface-inertia needs --block-solver dense --block-inertia\n";
      return 2;
   }
   if (block_inertia && (solver != "ddsimple" || block_dual_reg)) {
      std::cerr << "--block-inertia needs --solver ddsimple and excludes --block-dual-reg\n";
      return 2;
   }
   if (drop_corner_reg &&
       (solver != "ddsimple" || formulation != "consensus" || objective != "copies" ||
        hessian != "exact")) {
      std::cerr << "--drop-corner-reg needs --solver ddsimple --formulation consensus "
                   "--objective copies --hessian exact (only there is the corner "
                   "block exactly delta_w*I, and restoration is detected through "
                   "the exact Hessian's obj_factor)\n";
      return 2;
   }
   if (objective != "consensus" && formulation != "consensus") {
      std::cerr << "--objective copies needs --formulation consensus (there "
                   "are no copies to put the objective on otherwise)\n";
      return 2;
   }
   if (solver == "ddsimple" && precond != "asd") {
      std::cerr << "--precond " << precond << " is not implemented in "
                   "dd_solver_simple.hpp; ASd is its only preconditioner\n";
      return 2;
   }
   if (nested && solver != "dd") {
      std::cerr << "--nested needs --solver dd (it adds a second level to the "
                   "arrowhead's interface)\n";
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
   if (schur_lag < 1) {
      std::cerr << "--schur-lag must be >= 1\n";
      return 2;
   }
   if (schur_lag > 1 && solver != "dd") {
      std::cerr << "--schur-lag needs --solver dd (it caches the arrowhead's "
                   "local Schur blocks)\n";
      return 2;
   }
   if (minres_lag < 1) {
      std::cerr << "--minres-lag must be >= 1\n";
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
   if (t_update != "geometric" && t_update != "mu") {
      std::cerr << "--t-update must be geometric|mu\n";
      return 2;
   }
   if (t_mu_scale <= 0) {
      std::cerr << "--t-mu-scale must be > 0\n";
      return 2;
   }
   // The geometric schedule must terminate (factor >= 1 or t-min <= 0 would
   // grow the level vector until OOM); the mu-coupled single solve only uses
   // t0 as the seed and t-min as the floor, so factor is unconstrained there.
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
   if (grid != "staggered" && grid != "collocated") {
      std::cerr << "--grid must be staggered|collocated\n";
      return 2;
   }
   const bool collocated = (grid == "collocated");
   if (collocated) {
      // Python's dump_data_2d.py, the .txt solution format and its readers all
      // assume the staggered (N−1)² cell layout.
      if (image_io::ends_with(data, ".txt")) {
         std::cerr << "--grid collocated needs an image input (the .txt instances "
                      "are staggered)\n";
         return 2;
      }
      if (stencil != "onesided") {
         std::cerr << "--grid collocated uses its own (Neumann forward-difference) "
                      "stencil; --stencil averaged is staggered-only\n";
         return 2;
      }
      if (!save_data.empty()) {
         std::cerr << "--save-data writes dump_data_2d.py's staggered format; not "
                      "available with --grid collocated\n";
         return 2;
      }
      if (!save_sol.empty() &&
          !(save_sol.size() >= 4 && save_sol.compare(save_sol.size() - 4, 4, ".npz") == 0)) {
         std::cerr << "--grid collocated saves .npz only (the legacy .txt solution "
                      "format is staggered)\n";
         return 2;
      }
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
   const bool consensus = (formulation == "consensus");
   SmartPtr<Mpcc2DTNLP> mpcc = consensus
      ? new Mpcc2DConsensusTNLP(data, iopt, weight == "exp", stencil == "averaged",
                                collocated)
      : new Mpcc2DTNLP(data, iopt, weight == "exp", stencil == "averaged", collocated);
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

   Partition2D part(mpcc->N, nsub, partition == "strip", mpcc->collocated);
   std::vector<int> col_owner;
   int n_promoted = 0;
   std::vector<int> owner;
   if (consensus) {
      // Duplicate-and-link: rebuild the NLP so every complicating variable is
      // primal and linear-only, then read the owner map off the construction.
      // No corner promotion is needed — rows are never cut, so the rank
      // deficiencies that forced it cannot arise.
      Mpcc2DConsensusTNLP* c = static_cast<Mpcc2DConsensusTNLP*>(GetRawPtr(mpcc));
      c->split_objective_ = (objective == "copies");   // must precede init
      c->init_consensus(part);
      owner = c->kkt_owner_consensus();
   } else {
      owner = kkt_owner(*mpcc, part, promote, &col_owner, &n_promoted,
                        border_reg > 0.0 ? &rank1_rows : nullptr);
   }

   // DDS_VERIFY_WK=1: rebuild each W_k from the TNLP's derivatives at every
   // iteration and compare with what ddsimple assembled (verify_wk.hpp).
   if (std::getenv("DDS_VERIFY_WK") && solver == "ddsimple")
      verify_wk::install(GetRawPtr(mpcc), owner);

   std::cout << "2D lifted TV-MPCC (" << (mpcc->collocated ? "collocated" : "staggered")
             << ", C++)  N=" << mpcc->N
             << "  nodes=" << mpcc->m_u << "  cells=" << mpcc->m_q
             << "  n=" << mpcc->n << "  m_con=" << mpcc->mcon
             << " (" << mpcc->n_eq << " eq + " << mpcc->n_ineq << " ineq)"
             << "  KKT dim=" << mpcc->kkt_dim << "\n";
   std::cout << "  stencil=" << (mpcc->averaged ? "averaged" : "onesided")
             << "  Q(a) = " << (mpcc->weight_exp ? "e^a" : "a")
             << (mpcc->has_ha ? "  (+ row ha: a >= 0)" : "  (a boxed)")
             << "   init=" << init << "   solver=" << solver;
   if (solver == "dd" || solver == "ddsimple") {
      if (part.striped) std::cout << "  partition=" << nsub << " strips";
      else std::cout << "  nsub=" << nsub << "x" << nsub << " tiles";
      if (consensus) {
         const Mpcc2DConsensusTNLP* c =
            static_cast<const Mpcc2DConsensusTNLP*>(GetRawPtr(mpcc));
         std::cout << "  formulation=consensus (+" << c->n_link
                   << " copies/links, no promoted duals)"
                   << "  objective=" << objective
                   << (c->split_objective_ ? " (C=0)" : " (C=PSD diag)");
      }
      else if (n_promoted) std::cout << "  +" << n_promoted << " promoted corner duals";
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
   if (!driver::init_app(app, printlevel, maxiter, solver, hessian)) return 1;
   // Barrier-advance gate, formulation-scoped default (2026-09-08).  For
   // CONSENSUS runs the monotone gate defaults to 1000: measured 80→29 its at
   // N=32 3×3, 332→138 at N=128 4×4, and it is what lets N=256 advance past
   // the level the default gate stalled on — same α*/PSNR each time.  NOT
   // flipped for the permutation form: the same gate 14×-regressed it at
   // N=32 (113→1603 its).  DD_BARRIER_TOL always wins when set, and
   // DD_BARRIER_TOL=10 reproduces the pre-flip consensus tables.
   if (consensus && !std::getenv("DD_BARRIER_TOL"))
      app->Options()->SetNumericValue("barrier_tol_factor", 1000.0);
   // Gentler monotone cuts, formulation-scoped for the same reason as the gate
   // above.  μ⁺ = min(κ_μ·μ, μ^θ_μ) at IPOPT's (0.2, 1.5) can cut μ by two
   // orders in ONE step, and because t = c·μ is slaved to it the
   // complementarity constraint tightens by the same factor at once.  Measured
   // at N=32 4×4 consensus that put μ at 1.5e-4 before the first Newton step
   // and threw the run into restoration for 17 iterations (inf_pr 1.4e-3 → 1.0,
   // inf_du → 1.0e+03).  At (0.7, 1.1) the first cut lands at 7.0e-4 and
   // inf_du peaks at 4.5 instead.
   //
   // What that buys, measured, same α*/PSNR at each size:
   //     consensus N=32   status 1 → 0 (Succeeded), 59 → 123 its
   //     consensus N=64   360 → 149 its, 1.18M → 423k CG its, 9122 → 2480 solves
   // and in BOTH the §8 falsification counter "non-positive curvature ... after"
   // goes to 0 — no inertia is handed to IPOPT while this run holds evidence
   // that S_ff is indefinite there.  N=32 pays 2× the iterations for that.
   //
   // NOT flipped for the permutation form, which it destroys: 243 → 3000 its
   // (max-iter, status -1) at N=32.  The two env vars always win when set, and
   // DD_MU_LINEAR_DECREASE=0.2 DD_MU_SUPERLINEAR_POWER=1.5 reproduces the
   // pre-flip consensus tables.  Other points measured at N=32 consensus and
   // rejected: (0.5,1.2) → -2, (0.9,1.05) → the worse local solution
   // (obj 2.346 vs 2.200), (0.7,1.5) → -3 at 1670 its, (0.2,1.1) → obj 2.344.
   if (consensus && !std::getenv("DD_MU_LINEAR_DECREASE"))
      app->Options()->SetNumericValue("mu_linear_decrease_factor", 0.7);
   if (consensus && !std::getenv("DD_MU_SUPERLINEAR_POWER"))
      app->Options()->SetNumericValue("mu_superlinear_decrease_power", 1.1);
   // DD_DERIV_TEST=first|second: run IPOPT's derivative checker against the
   // TNLP callbacks — the validation gate for a new formulation's eval code.
   if (const char* dt = std::getenv("DD_DERIV_TEST")) {
      app->Options()->SetStringValue(
         "derivative_test", std::string(dt) == "first" ? "first-order"
                                                       : "second-order");
      app->Options()->SetNumericValue("derivative_test_perturbation", 1e-7);
      app->Options()->SetIntegerValue("print_level", 4);
   }

#ifdef DD_HAVE_MA57
   if (solver == "dd") {
      DDArrowheadSolver::config_owner(owner, part.n_sub);
      if (nested) {
        int ng2 = 0;
        std::vector<int> owner2 = kkt_owner2(*mpcc, part, &ng2);
        DDArrowheadSolver::config_owner2(std::move(owner2), ng2);
      }
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
#endif
   // The readable Eigen-only twin (dd_solver_simple.hpp). It takes the SAME
   // owner map as --solver dd, so the two are directly comparable: same
   // partition, same interface, same Haynsworth inertia — only the numerical
   // kernels differ (Eigen LDLᵀ instead of MA57, CG instead of the direct
   // interface back-solve).
   if (solver == "ddsimple") {
      ddsimple::Arrowhead::Options o;
      // Border positions with a KKT index >= n are DUAL — the promoted corner
      // duals. They are S's negative eigenvalues, so CG needs them peeled.
      o.n_primal = dual_peel ? mpcc->n : (1 << 30);   // consensus: updated n
      o.alpha_index = alpha_peel ? mpcc->oa : -1;
      o.peel_cross_points = cross_peel;
      o.cg_tol = cg_tol;
      o.peel_cg_tol = peel_cg_tol;
      o.border_reg = border_reg;
      o.drop_corner = drop_corner_reg;
      // multipliers start after primal + slacks (primal | slacks | λ_c | λ_d)
      o.dual_start = block_dual_reg ? mpcc->n + mpcc->n_ineq : -1;
      // W_k-only regularization; the linking multipliers (consensus only) are
      // KKT rows n + n_ineq + [rlink, rlink + n_link) and stay unregularized
      o.wk_reg_h = wk_reg_h;
      o.wk_reg_c = wk_reg_c;
      o.wk_dual_start = mpcc->n + mpcc->n_ineq;
      o.block_inertia = block_inertia;
      o.dense_blocks = (block_solver == "dense");
      o.interface_inertia = interface_inertia;
      if (auto* c = dynamic_cast<Mpcc2DConsensusTNLP*>(GetRawPtr(mpcc))) {
         o.link_begin = o.wk_dual_start + c->rlink;
         o.link_end = o.link_begin + c->n_link;
      }
      o.cg_maxit = cg_maxit;
      if (sff_solver == "cg") o.sff_direct = 0;
      else if (sff_solver == "direct") o.sff_direct = 1;
      else if (sff_solver == "direct-all") o.sff_direct = 2;
      else {
         std::cerr << "unknown --sff-solver " << sff_solver
                   << " (cg|direct|direct-all)\n";
         return 2;
      }
      DDSimpleSolver::config(owner, part.n_sub);
      DDSimpleSolver::config_options(o);
      DDSimpleSolver::config_reg_rows(rank1_rows);
      std::cout << "  interface=cg(asd"
                << (alpha_peel ? ",alpha-peel" : ",no-alpha-peel")
                << (dual_peel ? ",dual-peel" : ",no-dual-peel")
                << (cross_peel ? ",cross-peel" : ",no-cross-peel")
                << ",tol=" << cg_tol << ",peel-tol=" << peel_cg_tol
                << ",maxit=" << cg_maxit << ")"
                << (o.sff_direct == 1 ? "  sff-solver=direct (Z by sparse LDLT of S_ff)"
                  : o.sff_direct == 2 ? "  sff-solver=direct-all (Z and interface by sparse LDLT of S_ff)"
                  : "") << "\n"
                << "  inertia=PREDICTED: S is never assembled or factorized; "
                   "In(S) = In(T) from the |P|x|P| peel complement\n";
      if (block_solver == "dense")
         std::cout << "  block solver: DENSE Bunch-Kaufman (LAPACK) W_k and S, exact inertia, "
                      "direct interface solve (overrides the CG/peel route above)\n";
      if (block_inertia)
         std::cout << "  block inertia: per-W_k delta_H^k/delta_C^k (Lueg), "
                      "IPOPT delta_w only when S is not PD\n";
      if (border_reg > 0.0) {
         std::cout << "  border-reg=" << border_reg << " on "
                   << rank1_rows.size() << " rank-1 pair multipliers";
         if (promote)
            std::cout << "  (WARNING: corner promotion still ON — those rows "
                         "went to the border, so there is nothing to "
                         "regularize; add --no-promote-corners)";
         std::cout << "\n";
      }
   }
   auto optimize = [&]() -> ApplicationReturnStatus {
#ifdef DD_HAVE_MA57
      if (solver == "dd") {
         SmartPtr<AlgorithmBuilder> b = new CustomSolverBuilder<DDArrowheadSolver>();
         return app->OptimizeNLP(new TNLPAdapter(GetRawPtr(mpcc)), b);
      }
#endif
      if (solver == "ddsimple") {
         SmartPtr<AlgorithmBuilder> b = new SimpleSolverBuilder();
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
#ifdef DD_HAVE_MA57
   if (solver == "dd" && interface_solver == "cg")
      driver::print_interface_stats(precond, alpha_peel, cg_tol);
   if (solver == "dd" && interface_solver == "minres")
      driver::print_minres_stats(cg_tol, minres_lag);
#endif
   if (solver == "ddsimple") {
      const auto& st = DDSimpleSolver::stats();
      if (st.pred_refused || st.indef_before || st.indef_after)
         std::cout << "  prediction refused=" << st.pred_refused
                   << "  non-positive curvature seen: before=" << st.indef_before
                   << " (no prediction issued)  after=" << st.indef_after
                   << " (prediction already given)\n";
      std::cout << "  interface CG: solves=" << st.solves
                << "  iterations=" << st.iters;
      if (st.solves) std::cout << " (" << (double)st.iters / (double)st.solves << "/solve)";
      std::cout << "  rejected=" << st.rejected
                << "  peel caches=" << st.cache_builds << "\n";
      std::cout << "  ddsimple wall: factorize=" << st.t_factor << " s"
                << " (peel cache " << st.t_peel << " s)"
                << "  solve=" << st.t_solve << " s\n";
      if (st.pc_blocks_indef)
         std::cout << "  ASd preconditioner: indefinite blocks (LDLT, not LLT)="
                   << st.pc_blocks_indef << "  positions affected="
                   << st.pc_positions_indef
                   << "  singular blocks skipped=" << st.pc_blocks_singular
                   << "\n";
      if (block_dual_reg)
         std::cout << "  per-block dual reg (--block-dual-reg): blocks regularized="
                   << st.blockreg_used << "  largest eps="
                   << [&] { std::ostringstream e; e << std::scientific << std::setprecision(1)
                                                    << st.blockreg_max; return e.str(); }()
                   << "  still broken at the cap=" << st.blockreg_failed << "\n";
      if (block_solver == "dense")
         std::cout << "  dense blocks: W_k attempts with a zero pivot=" << st.dense_w_zero
                   << "  S indefinite=" << st.dense_s_indef << "  S singular="
                   << st.dense_s_zero << "\n";
      if (interface_inertia)
         std::cout << "  interface inertia (--interface-inertia): delta_S>0 in " << st.sfix_used
                   << " factorizations  largest delta_S="
                   << [&] { std::ostringstream e; e << std::scientific << std::setprecision(1)
                                                    << st.sfix_max; return e.str(); }()
                   << "  S refactorizations=" << st.sfix_retries << "\n";
      if (block_inertia) {
         const double fr = st.binert_blocks ? 100.0 / (double)st.binert_blocks : 0.0;
         std::cout << "  per-block inertia (--block-inertia): block factorizations="
                   << st.binert_blocks << "  delta_H^k>0 in " << st.binert_dh << " ("
                   << [&] { std::ostringstream e; e << std::fixed << std::setprecision(1)
                                                    << fr * st.binert_dh << "%)  delta_C^k>0 in "
                                                    << st.binert_dc << " (" << fr * st.binert_dc
                                                    << "%, on rho_k too in " << st.binert_dc_link
                                                    << ")  largest delta_H^k=" << std::scientific
                                                    << st.binert_dh_max; return e.str(); }()
                   << "\n    extra LDLT=" << st.binert_retries << "  no shift found="
                   << st.binert_failed << "  IPOPT wrong inertia (S not PD)="
                   << st.binert_wrong << "  factorizations with IPOPT's own delta_c on="
                   << st.binert_ipopt_dc << "\n";
      }
      if (wk_reg_h > 0.0 || wk_reg_c > 0.0)
         std::cout << "  W_k-only regularization: "
                   << [&] { std::ostringstream e; e << std::scientific << std::setprecision(1)
                                                    << "delta_H^W=" << wk_reg_h
                                                    << " (primal)  delta_C^W=" << wk_reg_c;
                            return e.str(); }()
                   << " (multipliers, linking rows excluded)\n";
      if (drop_corner_reg)
         std::cout << "  corner block (--drop-corner-reg): delta_w dropped="
                   << st.corner_dropped << "  already zero=" << st.corner_zero
                   << "  kept in restoration=" << st.corner_kept_resto
                   << "  kept (not c*I)=" << st.corner_kept_nonscalar << "\n";
      // The §10 tally: silent unless something actually warned during the run.
      ddsimple::Warnings::get().report(std::cout);
   }

   if (!res.best_x.empty()) {
      if (!save_sol.empty()) {
         if (save_sol.size() > 4 &&
             save_sol.compare(save_sol.size() - 4, 4, ".txt") == 0)
            save_solution_txt(save_sol, *mpcc, res.hist, res.best_x, res.best_t, nsub);
         else
            save_solution_npz(save_sol, *mpcc, res.hist, res.best_x, res.best_t, nsub);
         std::cout << "  wrote " << save_sol << "\n";
      }
      if (!save_dd.empty())
         std::cout << "  wrote " << save_dd << "  (arrowhead of the LAST Newton step)\n";
   }
   return 0;
}
