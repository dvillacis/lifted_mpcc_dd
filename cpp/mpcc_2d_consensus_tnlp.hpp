// The CONSENSUS (duplicate-and-link) reformulation of the staggered 2D lifted
// TV-MPCC — the Lueg-form sibling of mpcc_2d_tnlp.hpp (--formulation consensus).
//
// WHY THIS FILE EXISTS.  The permutation route decomposes the MONOLITHIC KKT
// system: the border is discovered algebraically, the shared unknowns appear
// NONLINEARLY in the constraints of several tiles, the corner block C carries
// real Hessian/Jacobian entries, and cut constraint rows force dual variables
// onto the border (the promoted corner pairs) — which is what made S indefinite
// and required the dual peel and the predicted inertia's refusal machinery.
// Lueg et al. (Optim. Eng. 27:555–585, 2026) never face any of that because
// their formulation IMPOSES, before any linear algebra, that complicating
// variables appear only in LINEAR linking constraints (their eq. 2d).  This
// class imposes the same structure on our problem:
//
//   · every constraint ROW is assigned wholly to one tile (h1 by the node's
//     anchor tile, cell rows by the cell's tile, ha to tile 0) — rows are
//     never cut, so the rank deficiencies that forced corner-dual promotion
//     cannot arise and NO duals reach the border;
//   · every variable referenced by rows of ≥ 2 tiles (border u nodes, border
//     qx/qy cells, and α — never r/δ/θ, which are strictly cell-local) gets
//     one LOCAL COPY per referencing tile; the rows are rewritten against the
//     copies, and the original index becomes the CONSENSUS variable;
//   · one linear LINKING ROW per copy, x_copy − x_consensus = 0, inserted at
//     the end of the equality block so the base class's bounds logic is
//     untouched.  Its multiplier lives in the copy's tile.
//
// The consensus variables then appear in linking rows and (for u) in the
// quadratic objective / (for α) in the ridge and box.  Keeping the objective
// on the ORIGINAL indices is the DEFAULT (--objective consensus) and is
// deliberate: its only structural effect is a PSD DIAGONAL corner block
//
//     C = δ_w·I + diag(σ on each border u node, 0 on every border qx/qy,
//                      σ·reg_α + Σ_α on the consensus α),
//
// which can only push S further into positive definiteness.  That is the one
// (benign) departure from strict Lueg form, where the complicating variables
// carry no objective terms at all.
//
// --objective copies RECOVERS strict Lueg form (split_objective_ below).  The
// fidelity of a border node is distributed over its copies with weight
// 1/|T(u_i)|, the α ridge over the α copies with weight 1/n_tiles, and the α
// box moves from the consensus α to the copies.  The weights are what keep it
// the SAME problem: at any point feasible for the linking rows all copies of a
// node agree, so Σ_k (1/|T|)·½(ũ⁽ᵏ⁾−f)² = ½(u−f)².  No consensus column is
// then left with any curvature or any bound, so C = δ_w·I exactly and C = 0 at
// δ_w = 0.  MEASURED: it costs nothing — with --solver mumps the entire IPOPT
// trace is IDENTICAL to the default's, because the linking rows are linear and
// the start is feasible for them, so the iterates never leave the manifold
// where the two objectives agree, and the Newton systems differ only by a
// shift w_k·∇Φ in the linking multipliers ν_k.  See docs/consensus_formulation
// (§6.2) for the proof and the ddsimple numbers.
//
// EQUIVALENCE.  This is an exact reformulation — the linking constraints force
// copies equal to consensus values at any feasible point, so the solution set
// is the original one (checked by the driver: --solver mumps on both
// formulations must agree on α* and PSNR).  What changes is the SPARSITY
// PATTERN handed to the decomposition: the KKT border consists of consensus
// variables only, all primal, giving the ddsimple solver the structure its
// theory wants — S SPD after inertia correction, no promoted duals, T SPD.
//
// SIZES.  n grows by one copy per (shared variable, tile) pair and m by the
// same count of linking rows.  On a k×k tile partition that is O(kN) extra
// unknowns — the ghost layers plus k copies of α.
//
// IMPLEMENTATION.  Subclasses Mpcc2DTNLP and rebuilds the structure arrays and
// the three eval callbacks against precomputed EFFECTIVE INDEX tables; the
// evaluation formulas are the parent's, entry for entry, just aimed at the
// row-tile's copies.  IPOPT sums duplicate triplet entries, which the split
// (α,·) Hessian blocks rely on.  Everything else (objective, bounds, warm
// starts, the μ-coupled callback, banking) is inherited from MpccTNLPBase
// unchanged — the member offsets it reads (n, n_eq, rhr, rcomp, oD, oa, oTh)
// are updated by init_consensus() to the consensus layout.
#ifndef MPCC_2D_CONSENSUS_TNLP_HPP
#define MPCC_2D_CONSENSUS_TNLP_HPP

#include <algorithm>
#include <vector>

#include "ipopt_phase.hpp"
#include "mpcc_2d_tnlp.hpp"
#include "partition_2d.hpp"

namespace Ipopt {

class Mpcc2DConsensusTNLP : public Mpcc2DTNLP {
public:
   using Mpcc2DTNLP::Mpcc2DTNLP;

   // ---- consensus layout, filled by init_consensus ----------------------
   int n_orig = 0, m_orig = 0, n_eq_orig = 0;
   int n_link = 0, rlink = 0;          // linking rows: [rlink, rlink + n_link)
   int n_tiles = 0;
   std::vector<int> node_tile_, cell_tile_;
   std::vector<int> row_tile_;         // consensus row  → tile
   std::vector<int> col_tile_;         // consensus var  → tile (−1 = consensus)
   std::vector<int> link_copy_, link_orig_;   // linking row ℓ: copy − orig = 0

   // ---- effective-index tables (what makes eval cheap) ------------------
   struct Ent { int r, c; double v; }; // r: node or cell id; c: consensus column
   std::vector<int> ueff1_;            // h1 row i → u index its tile uses
   std::vector<int> a_of_h1_;          // h1 row i → α index its tile uses
   std::vector<int> aeff_;             // tile k   → α index (copy, or oa if unshared)
   std::vector<Ent> DXT_, DYT_;        // h1 divergence, columns remapped per row tile
   std::vector<Ent> KX2_, KY2_;        // h2 u-stencils, columns remapped per cell tile
   std::vector<int> qxeff3_, qyeff3_;  // h3 row e → qx/qy index its tile uses
   std::vector<Ent> fid_;              // objective fidelity: {node, column, weight}

   // --objective copies: distribute the objective over the COPIES instead of
   // leaving it on the consensus indices, so that no consensus column carries
   // objective curvature and the corner block C empties to IPOPT's δ_w·I —
   // strict Lueg form.  Must be set BEFORE init_consensus(), which bakes it
   // into fid_ and the Hessian structure.  Default false: see the header.
   bool split_objective_ = false;

   // Build the whole consensus structure from a partition.  Must be called
   // once, after the parent constructor (and after x_start_ is filled, which
   // it extends copy-wise).
   void init_consensus(const Partition2D& part) {
      n_orig = n; m_orig = mcon; n_eq_orig = n_eq;
      n_tiles = part.n_sub;
      node_tile_ = part.node_owner;
      cell_tile_ = part.cell_owner;

      // -- which tiles reference each potentially-shared variable? --------
      // u node i: its own h1 row's tile plus the tiles of cells whose h2
      // stencil touches it.  qx/qy cell e: its own h3 row's tile plus the
      // tiles of nodes whose h1 divergence touches it.  α: every tile (h1 is
      // everywhere).  r/δ/θ appear only in their own cell's rows — never
      // shared, never copied.
      std::vector<std::vector<int>> tiles(n_orig);
      auto touch = [&](int v, int k) {
         auto& t = tiles[v];
         if (std::find(t.begin(), t.end(), k) == t.end()) t.push_back(k);
      };
      for (int i = 0; i < m_u; ++i) touch(ou + i, node_tile_[i]);
      for (const auto& t : Kx_) touch(ou + t.c, cell_tile_[t.r]);
      for (const auto& t : Ky_) touch(ou + t.c, cell_tile_[t.r]);
      for (int e = 0; e < m_q; ++e) {
         touch(oqx + e, cell_tile_[e]);
         touch(oqy + e, cell_tile_[e]);
         touch(oR + e, cell_tile_[e]);      // r/δ/θ: strictly cell-local —
         touch(oD + e, cell_tile_[e]);      // referenced only here, so they
         touch(oTh + e, cell_tile_[e]);     // are never copied
      }
      for (const auto& t : KxT_) touch(oqx + t.c, node_tile_[t.r]);
      for (const auto& t : KyT_) touch(oqy + t.c, node_tile_[t.r]);
      for (int k = 0; k < n_tiles; ++k) touch(oa, k);
      for (auto& t : tiles) std::sort(t.begin(), t.end());

      // -- allocate copies.  Order matters twice over: α copies come LAST so
      // every (α, ·) Hessian entry stays in the lower triangle, and each
      // shared variable's copies are contiguous so eff() is one small scan.
      std::vector<int> base(n_orig, -1);      // shared var → first copy index
      col_tile_.assign(n_orig, 0);
      link_copy_.clear(); link_orig_.clear();
      std::vector<int> copy_tile;
      int next = n_orig;
      auto alloc = [&](int v) {
         if ((int)tiles[v].size() <= 1) {
            col_tile_[v] = tiles[v].empty() ? 0 : tiles[v][0];
            return;
         }
         col_tile_[v] = -1;                   // the consensus variable
         base[v] = next;
         for (int k : tiles[v]) {
            link_copy_.push_back(next);
            link_orig_.push_back(v);
            copy_tile.push_back(k);
            ++next;
         }
      };
      for (int i = 0; i < m_u; ++i) alloc(ou + i);
      for (int e = 0; e < m_q; ++e) alloc(oqx + e);
      for (int e = 0; e < m_q; ++e) alloc(oqy + e);
      for (int e = 0; e < m_q; ++e) { alloc(oR + e); alloc(oD + e); alloc(oTh + e); }
      alloc(oa);                              // α last (see above)
      n_link = (int)link_copy_.size();
      col_tile_.resize(next);
      for (int c = 0; c < n_link; ++c) col_tile_[n_orig + c] = copy_tile[c];

      auto eff = [&](int v, int k) {          // variable v as seen by tile k
         if (base[v] < 0) return v;
         const auto& t = tiles[v];
         for (size_t j = 0; j < t.size(); ++j)
            if (t[j] == k) return base[v] + (int)j;
         return v;                            // unreachable for a valid partition
      };

      // -- new sizes and row offsets: linking rows extend the equality block,
      // so the inequality blocks (hr, hd, ha, comp) shift by n_link and the
      // base class's bounds logic keeps working verbatim.
      n = next;
      rlink = n_eq_orig;
      n_eq = n_eq_orig + n_link;
      rhr += n_link; rhd += n_link; rha += n_link; rcomp += n_link;
      mcon = m_orig + n_link;
      kkt_dim = n + 2 * n_ineq + n_eq;

      // -- row → tile (consensus row numbering) ---------------------------
      row_tile_.assign(mcon, 0);
      for (int i = 0; i < m_u; ++i) row_tile_[rh1 + i] = node_tile_[i];
      const int cblk[7] = {rh2x, rh2y, rh3x, rh3y, rhr, rhd, rcomp};
      for (int b = 0; b < 7; ++b)
         for (int e = 0; e < m_q; ++e) row_tile_[cblk[b] + e] = cell_tile_[e];
      if (has_ha) row_tile_[rha] = 0;
      for (int l = 0; l < n_link; ++l) row_tile_[rlink + l] = copy_tile[l];

      // -- effective-index tables -----------------------------------------
      ueff1_.resize(m_u); a_of_h1_.resize(m_u);
      for (int i = 0; i < m_u; ++i) {
         ueff1_[i] = eff(ou + i, node_tile_[i]);
         a_of_h1_[i] = eff(oa, node_tile_[i]);
      }
      aeff_.resize(n_tiles);
      for (int k = 0; k < n_tiles; ++k) aeff_[k] = eff(oa, k);
      DXT_.clear(); DYT_.clear(); KX2_.clear(); KY2_.clear();
      for (const auto& t : KxT_) DXT_.push_back({t.r, eff(oqx + t.c, node_tile_[t.r]), t.v});
      for (const auto& t : KyT_) DYT_.push_back({t.r, eff(oqy + t.c, node_tile_[t.r]), t.v});
      for (const auto& t : Kx_)  KX2_.push_back({t.r, eff(ou + t.c, cell_tile_[t.r]), t.v});
      for (const auto& t : Ky_)  KY2_.push_back({t.r, eff(ou + t.c, cell_tile_[t.r]), t.v});
      qxeff3_.resize(m_q); qyeff3_.resize(m_q);
      for (int e = 0; e < m_q; ++e) {
         qxeff3_[e] = eff(oqx + e, cell_tile_[e]);
         qyeff3_[e] = eff(oqy + e, cell_tile_[e]);
      }

      // -- the objective's fidelity terms, {node i, column, weight}.  By
      // default one term per node on the ORIGINAL column — which for a border
      // node IS the consensus column, and is what puts a PSD diagonal into the
      // corner block C.  Under split_objective_ a border node's term is
      // DISTRIBUTED over its copies with weight 1/|T(u_i)|: equal at any
      // feasible point (all copies agree), so the same problem, but nothing
      // lands on a consensus column.  The α ridge is split the same way, by
      // 1/n_tiles, inside the (α,α) Hessian block and eval_f/eval_grad_f.
      fid_.clear();
      for (int i = 0; i < m_u; ++i) {
         const int v = ou + i;
         if (!split_objective_ || base[v] < 0) { fid_.push_back({i, v, 1.0}); continue; }
         const double w = 1.0 / (double)tiles[v].size();
         for (size_t j = 0; j < tiles[v].size(); ++j)
            fid_.push_back({i, base[v] + (int)j, w});
      }

      build_consensus_structures();

      // -- the starting point: copies start equal to their consensus value,
      // which makes the initial point exactly feasible for the linking rows.
      if ((int)x_start_.size() == n_orig) {
         x_start_.resize(n);
         for (int l = 0; l < n_link; ++l)
            x_start_[link_copy_[l]] = x_start_[link_orig_[l]];
      }
   }

   // The owner map for the consensus KKT system, in the driver's standard
   // ordering (primal | slacks | λ_c | λ_d).  Every dual is interior — the
   // point of the reformulation — and the border is exactly the consensus set.
   std::vector<int> kkt_owner_consensus() const {
      std::vector<int> owner;
      owner.reserve(kkt_dim);
      owner.insert(owner.end(), col_tile_.begin(), col_tile_.end());
      for (int r = n_eq; r < mcon; ++r) owner.push_back(row_tile_[r]);   // slacks
      for (int r = 0; r < n_eq; ++r) owner.push_back(row_tile_[r]);      // λ_c
      for (int r = n_eq; r < mcon; ++r) owner.push_back(row_tile_[r]);   // λ_d
      return owner;
   }

   // ---- structure: the parent's 21 Jacobian pieces + 2 linking pieces, and
   // the parent's 8 Hessian blocks with the (α,·) blocks split per h1-row
   // tile.  Values below match this order POSITIONALLY, as everywhere in this
   // repo.  Duplicate (row,col) pairs are legal — IPOPT sums them — and the
   // split (α,·) pieces produce them by design.
   void build_consensus_structures() {
      jr_.clear(); jc_.clear(); hr_.clear(); hc_.clear();
      auto J = [&](int r, int c) { jr_.push_back(r); jc_.push_back(c); };
      for (int i = 0; i < m_u; ++i) J(rh1 + i, ueff1_[i]);               // 1
      for (const auto& t : DXT_) J(rh1 + t.r, t.c);                      // 2
      for (const auto& t : DYT_) J(rh1 + t.r, t.c);                      // 3
      for (int i = 0; i < m_u; ++i) J(rh1 + i, a_of_h1_[i]);             // 4
      for (const auto& t : KX2_) J(rh2x + t.r, t.c);                     // 5
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oR + e);                 // 6
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oTh + e);                // 7
      for (const auto& t : KY2_) J(rh2y + t.r, t.c);                     // 8
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oR + e);                 // 9
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oTh + e);                // 10
      for (int e = 0; e < m_q; ++e) J(rh3x + e, qxeff3_[e]);             // 11
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oD + e);                 // 12
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oTh + e);                // 13
      for (int e = 0; e < m_q; ++e) J(rh3y + e, qyeff3_[e]);             // 14
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oD + e);                 // 15
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oTh + e);                // 16
      for (int e = 0; e < m_q; ++e) J(rhr + e, oR + e);                  // 17
      for (int e = 0; e < m_q; ++e) J(rhd + e, oD + e);                  // 18
      if (has_ha) J(rha, aeff_[0]);                                      // 19
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oR + e);                // 20
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oD + e);                // 21
      for (int l = 0; l < n_link; ++l) J(rlink + l, link_copy_[l]);      // 22
      for (int l = 0; l < n_link; ++l) J(rlink + l, link_orig_[l]);      // 23

      auto H = [&](int r, int c) { hr_.push_back(r); hc_.push_back(c); };
      for (const auto& e : fid_) H(e.c, e.c);                            // (u,u)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oR + e);                  // (θ,r)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oD + e);                  // (θ,δ)
      for (int e = 0; e < m_q; ++e) H(oTh + e, oTh + e);                 // (θ,θ)
      for (int e = 0; e < m_q; ++e) H(oD + e, oR + e);                   // (δ,r)
      for (const auto& t : DXT_) H(a_of_h1_[t.r], t.c);                  // (α,qx)
      for (const auto& t : DYT_) H(a_of_h1_[t.r], t.c);                  // (α,qy)
      for (int k = 0; k < n_tiles; ++k) H(aeff_[k], aeff_[k]);           // (α,α) d²Q
      if (!split_objective_) H(oa, oa);                                  // (α,α) reg
   }

   // ---- the objective.  Arithmetic identical to the parent's in the default
   // mode (fid_ is then one unit-weight term per node on ou+i); under
   // split_objective_ the fidelity rides on fid_'s weighted copy columns and
   // the ridge on the α copies.  Both agree at any point feasible for the
   // linking rows, which is all exactness requires.
   bool eval_f(Index, const Number* x, bool, Number& obj) override {
      double s = 0.0;
      for (const auto& e : fid_) {
         const double d = x[e.c] - uclean_[e.r];
         s += e.v * d * d;
      }
      double a2 = 0.0;
      if (split_objective_)
         for (int k = 0; k < n_tiles; ++k)
            a2 += x[aeff_[k]] * x[aeff_[k]] / (double)n_tiles;
      else
         a2 = x[oa] * x[oa];
      obj = 0.5 * loss_scale_ * s + 0.5 * reg_alpha_ * a2;
      if (eps_theta_ != 0.0) {
         double g = 0.0;
         for (int e = 0; e < n_lift; ++e) {
            const double d = x[oTh + e] - theta_ref_[e];
            g += d * d;
         }
         obj += 0.5 * loss_scale_ * eps_theta_ * g;
      }
      return true;
   }

   bool eval_grad_f(Index, const Number* x, bool, Number* g) override {
      for (int i = 0; i < n; ++i) g[i] = 0.0;
      for (const auto& e : fid_)
         g[e.c] += loss_scale_ * e.v * (x[e.c] - uclean_[e.r]);
      if (split_objective_)
         for (int k = 0; k < n_tiles; ++k)
            g[aeff_[k]] += reg_alpha_ * x[aeff_[k]] / (double)n_tiles;
      else
         g[oa] += reg_alpha_ * x[oa];
      if (eps_theta_ != 0.0)
         for (int e = 0; e < n_lift; ++e)
            g[oTh + e] = loss_scale_ * eps_theta_ * (x[oTh + e] - theta_ref_[e]);
      return true;
   }

   // The α box is the only variable bound a consensus column can carry (δ ≤ 1
   // is cell-local), and its barrier term Σ_α sits in C.  Under
   // split_objective_ it moves to the copies — leaving it on the consensus α
   // would put Σ_α straight back into the corner block this mode exists to
   // empty.  Everything else is the parent's.
   bool get_bounds_info(Index nn, Number* xl, Number* xu, Index mm, Number* gl,
                        Number* gu) override {
      if (!Mpcc2DTNLP::get_bounds_info(nn, xl, xu, mm, gl, gu)) return false;
      if (!split_objective_) return true;
      xl[oa] = -2e19; xu[oa] = 2e19;            // first: aeff_[k] == oa if n_tiles == 1
      for (int k = 0; k < n_tiles; ++k) {
         if (has_ha) { xu[aeff_[k]] = w_max_; }
         else { xl[aeff_[k]] = alpha_lo_; xu[aeff_[k]] = alpha_hi_; }
      }
      return true;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      std::vector<double> dq(m_u, 0.0), kxu(m_q, 0.0), kyu(m_q, 0.0);
      for (const auto& t : DXT_) dq[t.r] += t.v * x[t.c];
      for (const auto& t : DYT_) dq[t.r] += t.v * x[t.c];
      for (const auto& t : KX2_) kxu[t.r] += t.v * x[t.c];
      for (const auto& t : KY2_) kyu[t.r] += t.v * x[t.c];
      for (int i = 0; i < m_u; ++i)
         g[rh1 + i] = x[ueff1_[i]] - f_[i] + Q(x[a_of_h1_[i]]) * dq[i];
      for (int e = 0; e < m_q; ++e) {
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         const double r = x[oR + e], d = x[oD + e];
         g[rh2x + e]  = kxu[e] - r * c;
         g[rh2y + e]  = kyu[e] - r * s;
         g[rh3x + e]  = x[qxeff3_[e]] - d * c;
         g[rh3y + e]  = x[qyeff3_[e]] - d * s;
         g[rhr + e]   = r;
         g[rhd + e]   = d;
         g[rcomp + e] = r * (1.0 - d) - t_;
      }
      if (has_ha) g[rha] = x[aeff_[0]];
      for (int l = 0; l < n_link; ++l)
         g[rlink + l] = x[link_copy_[l]] - x[link_orig_[l]];
      return true;
   }

   bool eval_jac_g(Index, const Number* x, bool, Index, Index nele, Index* iRow,
                   Index* jCol, Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = jr_[k]; jCol[k] = jc_[k]; }
         return true;
      }
      std::vector<double> dq(m_u, 0.0);
      for (const auto& t : DXT_) dq[t.r] += t.v * x[t.c];
      for (const auto& t : DYT_) dq[t.r] += t.v * x[t.c];
      Index k = 0;
      for (int i = 0; i < m_u; ++i) values[k++] = 1.0;                    // 1
      for (const auto& t : DXT_) values[k++] = Q(x[a_of_h1_[t.r]]) * t.v; // 2
      for (const auto& t : DYT_) values[k++] = Q(x[a_of_h1_[t.r]]) * t.v; // 3
      for (int i = 0; i < m_u; ++i)                                       // 4
         values[k++] = dQ(x[a_of_h1_[i]]) * dq[i];
      for (const auto& t : KX2_) values[k++] = t.v;                       // 5
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);  // 6
      for (int e = 0; e < m_q; ++e)                                       // 7
         values[k++] = x[oR + e] * std::sin(x[oTh + e]);
      for (const auto& t : KY2_) values[k++] = t.v;                       // 8
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);  // 9
      for (int e = 0; e < m_q; ++e)                                       // 10
         values[k++] = -x[oR + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 11
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);  // 12
      for (int e = 0; e < m_q; ++e)                                       // 13
         values[k++] = x[oD + e] * std::sin(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 14
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);  // 15
      for (int e = 0; e < m_q; ++e)                                       // 16
         values[k++] = -x[oD + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 17
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                    // 18
      if (has_ha) values[k++] = 1.0;                                      // 19
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0 - x[oD + e];        // 20
      for (int e = 0; e < m_q; ++e) values[k++] = -x[oR + e];             // 21
      for (int l = 0; l < n_link; ++l) values[k++] = 1.0;                 // 22
      for (int l = 0; l < n_link; ++l) values[k++] = -1.0;                // 23
      return true;
   }

   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == NULL) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hr_[k]; jCol[k] = hc_[k]; }
         return true;
      }
      // restoration evaluates this Hessian with obj_factor = 0 (ipopt_phase.hpp)
      ipopt_phase::restoration() = (obj_factor == 0.0);
      std::vector<double> dq(m_u, 0.0);
      for (const auto& t : DXT_) dq[t.r] += t.v * x[t.c];
      for (const auto& t : DYT_) dq[t.r] += t.v * x[t.c];
      Index k = 0;
      for (const auto& e : fid_) values[k++] = obj_factor * e.v;          // (u,u)
      for (int e = 0; e < m_q; ++e)                                       // (θ,r)
         values[k++] = lam[rh2x + e] * std::sin(x[oTh + e])
                     - lam[rh2y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e)                                       // (θ,δ)
         values[k++] = lam[rh3x + e] * std::sin(x[oTh + e])
                     - lam[rh3y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) {                                     // (θ,θ)
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         values[k++] = x[oR + e] * (lam[rh2x + e] * c + lam[rh2y + e] * s)
                     + x[oD + e] * (lam[rh3x + e] * c + lam[rh3y + e] * s)
                     + obj_factor * eps_theta_;
      }
      for (int e = 0; e < m_q; ++e) values[k++] = -lam[rcomp + e];        // (δ,r)
      // (α,qx)/(α,qy): the parent's aggregated entries split per h1 row, so
      // each contribution lands between the ROW TILE's α copy and its q copy.
      for (const auto& t : DXT_)
         values[k++] = dQ(x[a_of_h1_[t.r]]) * t.v * lam[rh1 + t.r];
      for (const auto& t : DYT_)
         values[k++] = dQ(x[a_of_h1_[t.r]]) * t.v * lam[rh1 + t.r];
      // (α,α): the d²Q part per tile (zero for the linear weight), then the
      // reg-α ridge — on the CONSENSUS α by default (the objective lives on the
      // original indices, see the header comment), or split 1/n_tiles onto the
      // copies under split_objective_, where it must NOT reach the consensus
      // column.
      std::vector<double> aa(n_tiles, 0.0);
      for (int i = 0; i < m_u; ++i) aa[node_tile_[i]] += lam[rh1 + i] * dq[i];
      const double rw = split_objective_ ? reg_alpha_ / (double)n_tiles : 0.0;
      for (int t = 0; t < n_tiles; ++t)
         values[k++] = d2Q(x[aeff_[t]]) * aa[t] + obj_factor * rw;
      if (!split_objective_) values[k++] = obj_factor * reg_alpha_;
      return true;
   }
};

}  // namespace Ipopt

#endif  // MPCC_2D_CONSENSUS_TNLP_HPP
