// tv_mpcc.hpp — learning a TV-denoising weight, written as one smooth NLP in
// consensus form for the domain decomposition.
//
// ============================================================================
//  THE BILEVEL PROBLEM
// ============================================================================
// Find the weight α whose TV-denoised image u is closest to the clean one:
//
//     min_α  ½‖u(α) − u_clean‖²    where  u(α) = argmin_u ½‖u − f‖² + α·TV(u)
//
// The lower level is replaced by its optimality system.  TV's kink is removed by
// LIFTING both the gradient Ku and the dual variable q to polar coordinates,
// Ku = r·(cosθ, sinθ) and q = δ·(cosθ, sinθ), which turns the lower level into
// smooth equations plus one complementarity r·(1 − δ) = 0.  That complementarity
// is relaxed to r·(1 − δ) ≤ t (Scholtes), and t is driven to zero during the solve:
//
//   min   ½‖u − u_clean‖² + ½·reg_α·α² + ½·ε_θ·‖θ − θ_ref‖²
//   s.t.  h1  : u − f + α·(Kxᵀqx + Kyᵀqy) = 0       optimality of u
//         h2x : Kx u − r·cosθ = 0                     ∇u in polar form
//         h2y : Ky u − r·sinθ = 0
//         h3x : qx − δ·cosθ = 0                       q in polar form
//         h3y : qy − δ·sinθ = 0
//         hr  : r ≥ 0      hd : δ ≥ 0      ha : α ≥ 0
//         comp: r·(1 − δ) − t ≤ 0                     relaxed complementarity
//         and the bound δ ≤ 1.
//
// u lives on the N² NODES; qx, qy, r, δ, θ live on the (N−1)² CELLS; α is a scalar.
// Kx, Ky are forward differences anchored at each cell's top-left node.
// The ε_θ ridge pins θ where r = δ = 0 (there the angle is undetermined).
//
// ============================================================================
//  THE CONSENSUS FORM
// ============================================================================
// To decompose, every variable referenced by rows of ≥ 2 tiles (border u nodes,
// border qx/qy cells, and α) gets one LOCAL COPY per tile.  Each tile's rows are
// rewritten to use its own copies, and one linear LINKING ROW per copy,
//
//     x_copy − x_consensus = 0,
//
// ties the copies to the original variable, which becomes the CONSENSUS variable.
// Now no row is shared between tiles, every multiplier belongs to one tile, and
// the only unknowns coupling tiles are the consensus variables — all primal.
// That is exactly the structure of Lueg et al.'s eq. (1).  The objective stays on
// the original indices, so it is untouched.
//
// ============================================================================
//  THE μ-COUPLED CONTINUATION (iteration())
// ============================================================================
// One IPM solve, with t following the barrier: t = max(t_min, 10·μ), never
// loosening and at most halving per iteration.  This is legal mid-solve because
// t enters only g.  Once t = t_min, the LEVEL GATE stops the solve as soon as
// the primal and dual infeasibility and μ are all below √t: a relaxation at
// level t is only accurate to about √t, so solving it more tightly is wasted.
//
// ============================================================================
//  THE EXACT-PENALTY FORMULATION (penalty > 0, opt-in)
// ============================================================================
// Leyffer, López-Calva & Nocedal (SIAM J. Optim. 17, 2006): instead of the
// relaxed rows r(1 − δ) ≤ t, the complementarity enters the objective,
//     f + π Σ_e r_e (1 − δ_e),   r ≥ 0,  1 − δ ≥ 0,
// and the product rows are free.  At an S-stationary point with π large
// enough, the penalty problem's KKT points are the MPCC's, and the
// complementarity multiplier is π (bounded).  π grows like LLN's dynamic
// rule: ×10 when max_e min(r_e, 1 − δ_e) > μ^0.4 and has not decreased by 10%
// over the last 3 iterations.  t still follows μ, only for the θ ridge.  The
// gate: once t = t_min, stop when inf_pr, inf_du, μ ≤ √t_min and
// max r(1 − δ) ≤ t_min (the accuracy class of the relaxation).
//
// ============================================================================
//  THE ACTIVE-SET CLEAN-UP (begin_cleanup())
// ============================================================================
// After the continuation (or as soon as t reaches t_min), the inferred active
// set is pinned and the relaxed products dropped, and the NLP is re-solved
// (the manuscript's "active-set cleanup"): per cell the smaller of r and
// w = 1 − δ is fixed, r = 0 (row hr becomes an equality) or δ = 1 (row hd
// becomes δ = 1, the bound δ ≤ 1 is dropped); the product rows become free.
// There is no relaxation and no continuation left.  At the end, on the cells
// that are biactive (the free member also ≈ 0), the pinned row's multiplier is
// the MPCC multiplier of the pinned constraint, γ of r ≥ 0 or ν of w ≥ 0; it
// must be ≥ 0 for S-stationarity (cleanup_report()).
//
// ============================================================================
//  THE LOCAL VIEW (MPI)
// ============================================================================
// The layout above is GLOBAL: every rank numbers the variables and rows the
// same way (offsets, copies, linking rows), but holds only the variables and
// rows of its own tiles plus the consensus variables (nlp.hpp).  The NLP
// interface works on the local vectors, in ascending global order; the local
// tables (nodes_, cells_, the stencils, the patterns) cover the rank's own
// nodes and cells.  The objective is separable per variable and goes through
// a TileSum.  Global tables kept on every rank are O(#variables) integers
// (col_tile_) or per pixel; nothing of the size of the Jacobian is global.
// finalize() gathers the solution and multipliers on rank 0 in the global
// layout, and so do --diag and --classify for their printouts.
//
// Everything below keeps a fixed ORDER for variables, rows and nonzeros: the
// IPM builds the KKT matrix from these arrays, and the order is the pattern.
// A tile's entries come in the same relative order on any number of ranks.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "comm.hpp"
#include "image.hpp"
#include "nlp.hpp"
#include "partition.hpp"

class TvMpcc : public NLP {
public:
   // ---- continuation state ------------------------------------------------
   double t = 1.0;             // current relaxation level
   double t_min = 1e-4;        // the level the answer is reported at
   double t_mu_scale = 10.0;   // t follows the barrier: t = max(t_min, 10·μ)
   double t_rate = 0.5;        // ... but may at most halve per iteration
   double t_comp_ratio = 0.0;  // κ_t > 0: ... and never below max r(1−δ)/κ_t, so that t does
                               // not run ahead of an iterate whose products lag behind
                               // (mariposa N=640: t fell 7× in 3 iterations while
                               // r(1−δ) stayed at 6.5·t_min; 40–77 iterations to catch up)
   double penalty = 0.0;       // π > 0: the exact-penalty formulation (see the header)
   double penalty_max = 1e8;
   bool penalty_hessian = true; // false: leave π's (indefinite) r–δ term out of the Hessian
   bool stop_at_tmin = false;  // stop the solve as soon as t = t_min (for a clean-up)
   // Clean-up on cells where both r and w are ≤ cleanup_eps (near-biactive):
   // 0 pin the smaller (the manuscript), 1 pin both (tightened NLP), 2 pin
   // neither (relaxed NLP: r ≥ 0, w ≥ 0, no product).  cleanup_eps ≤ 0: √t_min.
   int cleanup_biactive = 0;
   double cleanup_eps = 0.0;
   bool reached_tmin = false;  // ... and it did
   bool t_on_solved = false;   // lower t only when the barrier problem at the current
                               // (μ, t) is solved (E_μ ≤ κ_ε·μ), not at every iteration
   double c_theta = 1.0;       // ε_θ = c_θ · t
   double eps_theta = 0.0;
   double gate_floor = 1e-8;   // floor on the level gate
   bool gate_fired = false;
   bool print = true;          // the gate message (off on all MPI ranks but one)
   bool diag = false;          // per iteration: where the dual residual sits (diagnose())
   bool classify = false;      // per iteration once t = t_min: MPCC multipliers (classify_point())
   // ... and the last point's per-cell table, 8 columns per cell:
   //     r, 1−δ, ξ, a, b, γ, ν, ‖λ_h2‖     (see classify_point())
   std::vector<double> mpcc_cells;

   // ---- sign-driven per-cell relaxation (end game, opt-in) -------------------
   // DeMiguel, Friedlander, Nogales & Scholtes (2005), CCOpt's end game: once
   // t = t_min, a cell whose MPCC multiplier γ = a − ξ(1−δ) or ν = b − ξr is
   // below −max(relax_thr, inf_du^relax_exp) in relax_persist consecutive
   // iterations is FLAGGED for the rest of the solve (the multiplier estimates
   // are only as good as the dual residual: DeMiguel et al. use ‖res‖^(1−τ),
   // τ = 0.3, as the sign threshold): its bounds are
   // relaxed to r ≥ −relax_shift and δ ≤ 1 + relax_shift, and its product row
   // tightened to r(1 − δ) ≤ 0.  The relaxed bounds keep an interior around the
   // corner (r > 0 > 1−δ or the reverse) while the product forces one branch, so
   // a pair held at the corner by the relaxation (both multipliers negative:
   // C-stationary) can leave it along a descent branch, and ξ stays bounded.
   bool relax_cells = false;
   double relax_thr = 0.1;
   double relax_shift = 1e-4;
   int relax_persist = 5;
   double relax_exp = 0.7;
   long relaxed_total = 0;
   bool bounds_as_rows = true; // r, δ, α ≥ 0 as rows hr, hd, ha (slacked); false: as
                               // variable bounds, the rows kept but free (experiment)

   // ---- sizes and offsets -------------------------------------------------
   int N, nc, m_u, m_q;        // image side, cells per side, #nodes, #cells
   int n, mcon, n_eq, n_ineq;  // #variables, #rows, #equality rows, #inequality rows
   int n_link = 0;             // #copies = #linking rows
   int n_orig = 0;             // #variables before the copies were added

   // Column offsets of the ORIGINAL variables.  Copies are appended after them.
   //   x = [ u | qx | qy | r | δ | θ | α | copies... ]
   int ou, oqx, oqy, oR, oD, oTh, oa;
   // Row offsets.  Linking rows are appended to the equality block.
   int rh1, rh2x, rh2y, rh3x, rh3y, rlink, rhr, rhd, rha, rcomp;

   std::vector<double> x_start;     // Chambolle–Pock warm start (local)
   std::vector<double> solution;    // written by finalize, on rank 0, global: x ...
   std::vector<double> multipliers; //   ... the constraint multipliers λ
   double objective = 0.0;          //   ... and f(x)

   // One row (iter, μ, t, weight, max r(1−δ)) per IPM iteration.
   std::vector<double> mu_trace;

   // Build the whole NLP.  Order matters: the warm start and θ_ref are computed
   // on the unsplit problem, THEN the consensus copies are appended.
   TvMpcc(const Image& img, const Partition& part, double sigma)
       : N(img.N), clean_(img.clean), f_(img.noisy), sum_(part.n_tiles) {
      nc = N - 1;
      m_u = N * N;
      m_q = nc * nc;
      layout_unsplit();
      build_stencils();
      warm_start_chambolle_pock(0.7 * sigma);
      theta_ref_.assign(x_start.begin() + oTh, x_start.begin() + oTh + m_q);
      split_into_consensus_form(part);
      build_local(part);
   }

   // max over cells of r·(1 − δ): how far from exact complementarity x is
   // (x global, e.g. the solution).
   double max_complementarity(const double* x) const {
      double c = 0.0;
      for (int e = 0; e < m_q; ++e) c = std::max(c, x[oR + e] * (1.0 - x[oD + e]));
      return c;
   }

   // =========================================================================
   //  NLP interface (local, see the header)
   // =========================================================================
   int num_vars() const override { return (int)gvar_.size(); }
   int num_rows() const override { return (int)grow_.size(); }
   int n_tiles() const override { return n_tiles_; }
   const std::vector<int>& var_tile() const override { return vtile_; }
   const std::vector<int>& row_tile() const override { return rtile_; }

   // Only δ ≤ 1 is a variable bound.  r ≥ 0, δ ≥ 0, α ≥ 0 are ROWS (hr, hd, ha).
   void bounds(double* xl, double* xu, double* gl, double* gu) const override {
      const int nl = num_vars(), ml = num_rows();
      for (int i = 0; i < nl; ++i) { xl[i] = -2e19; xu[i] = 2e19; }
      for (const Cell& c : cells_) xu[c.d] = 1.0;
      xu[la_] = kAlphaMax;
      for (int i = 0; i < ml; ++i) {
         const int r = grow_[i];
         if (r < n_eq) { gl[i] = 0.0; gu[i] = 0.0; }                          // = 0
         else if (r < rcomp) { gl[i] = 0.0; gu[i] = 2e19; }                   // ≥ 0
         else { gl[i] = -2e19; gu[i] = 0.0; }                                  // ≤ 0
         if (!bounds_as_rows && r >= rhr && r < rcomp) { gl[i] = -2e19; gu[i] = 2e19; }
      }
      if (!bounds_as_rows) {
         for (const Cell& c : cells_) xl[c.r] = xl[c.d] = 0.0;
         xl[la_] = 0.0;
      }
      if (penalty > 0.0)
         for (const Cell& q : cells_) { gl[q.comp] = -2e19; gu[q.comp] = 2e19; }
      for (size_t b = 0; b < pinned_.size(); ++b) {   // the clean-up
         const Cell& q = cells_[b];
         gl[q.comp] = -2e19;
         gu[q.comp] = 2e19;
         if (pinned_[b] & kPinR) {
            gl[q.hr] = gu[q.hr] = 0.0;
            xl[q.r] = -2e19;
         }
         if (pinned_[b] & kPinD) {
            gl[q.hd] = gu[q.hd] = 1.0;
            xu[q.d] = 2e19;
         }
      }
      for (size_t b = 0; b < relaxed_.size(); ++b)
         if (relaxed_[b]) {
            (bounds_as_rows ? gl[cells_[b].hr] : xl[cells_[b].r]) = -relax_shift;
            xu[cells_[b].d] = 1.0 + relax_shift;
         }
   }

   void start(double* x) const override { std::copy(x_start.begin(), x_start.end(), x); }

   // ½‖u − u_clean‖² + ½·reg_α·α² + ½·ε_θ·‖θ − θ_ref‖², one term per variable
   double f(const double* x) override {
      for (int l = 0; l < num_vars(); ++l) {
         const double term = objective_term(l, x[l]);
         if (term != 0.0) sum_.add(vtile_[l], term);
      }
      if (penalty > 0.0)
         for (const Cell& q : cells_) sum_.add(vtile_[q.r], penalty * x[q.r] * (1.0 - x[q.d]));
      return sum_.total();
   }

   void grad_f(const double* x, double* g) override {
      for (int l = 0; l < num_vars(); ++l) {
         const int v = gvar_[l];
         g[l] = v < ou + m_u               ? x[l] - clean_[v - ou]
              : v == oa                    ? kRegAlpha * x[l]
              : v >= oTh && v < oTh + m_q  ? eps_theta * (x[l] - theta_ref_[v - oTh])
                                           : 0.0;
      }
      if (penalty > 0.0)
         for (const Cell& q : cells_) {
            g[q.r] += penalty * (1.0 - x[q.d]);
            g[q.d] -= penalty * x[q.r];
         }
   }

   void g(const double* x, double* g) override {
      std::vector<double> div(nodes_.size(), 0.0), kxu(cells_.size(), 0.0), kyu(cells_.size(), 0.0);
      for (const auto& e : div_x_) div[e.r] += e.v * x[e.c];
      for (const auto& e : div_y_) div[e.r] += e.v * x[e.c];
      for (const auto& e : grad_x_) kxu[e.r] += e.v * x[e.c];
      for (const auto& e : grad_y_) kyu[e.r] += e.v * x[e.c];
      for (size_t a = 0; a < nodes_.size(); ++a) {
         const Node& q = nodes_[a];
         g[q.h1] = x[q.u] - f_[q.i] + x[q.a] * div[a];
      }
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double c = std::cos(x[q.th]), s = std::sin(x[q.th]);
         const double r = x[q.r], d = x[q.d];
         g[q.h2x] = kxu[b] - r * c;
         g[q.h2y] = kyu[b] - r * s;
         g[q.h3x] = x[q.qx] - d * c;
         g[q.h3y] = x[q.qy] - d * s;
         g[q.hr] = r;
         g[q.hd] = d;
         g[q.comp] = r * (1.0 - d) - (relaxed_.empty() || !relaxed_[b] ? t : 0.0);
      }
      if (ha_ >= 0) g[ha_] = x[ha_alpha_];
      for (const Link& l : links_) g[l.row] = x[l.copy] - x[l.orig];
   }

   void jac_structure(std::vector<int>& row, std::vector<int>& col) const override {
      row = jac_row_;
      col = jac_col_;
   }

   // Values in exactly the order of jac_row_/jac_col_ (see build_local).
   void jac_values(const double* x, double* values) override {
      std::vector<double> div(nodes_.size(), 0.0);
      for (const auto& e : div_x_) div[e.r] += e.v * x[e.c];
      for (const auto& e : div_y_) div[e.r] += e.v * x[e.c];
      int k = 0;
      for (size_t a = 0; a < nodes_.size(); ++a) values[k++] = 1.0;                   // h1/u
      for (const auto& e : div_x_) values[k++] = x[nodes_[e.r].a] * e.v;              // h1/qx
      for (const auto& e : div_y_) values[k++] = x[nodes_[e.r].a] * e.v;              // h1/qy
      for (size_t a = 0; a < nodes_.size(); ++a) values[k++] = div[a];                // h1/α
      for (const auto& e : grad_x_) values[k++] = e.v;                                // h2x/u
      for (const Cell& q : cells_) values[k++] = -std::cos(x[q.th]);                  // h2x/r
      for (const Cell& q : cells_) values[k++] = x[q.r] * std::sin(x[q.th]);          // h2x/θ
      for (const auto& e : grad_y_) values[k++] = e.v;                                // h2y/u
      for (const Cell& q : cells_) values[k++] = -std::sin(x[q.th]);                  // h2y/r
      for (const Cell& q : cells_) values[k++] = -x[q.r] * std::cos(x[q.th]);         // h2y/θ
      for (size_t b = 0; b < cells_.size(); ++b) values[k++] = 1.0;                   // h3x/qx
      for (const Cell& q : cells_) values[k++] = -std::cos(x[q.th]);                  // h3x/δ
      for (const Cell& q : cells_) values[k++] = x[q.d] * std::sin(x[q.th]);          // h3x/θ
      for (size_t b = 0; b < cells_.size(); ++b) values[k++] = 1.0;                   // h3y/qy
      for (const Cell& q : cells_) values[k++] = -std::sin(x[q.th]);                  // h3y/δ
      for (const Cell& q : cells_) values[k++] = -x[q.d] * std::cos(x[q.th]);         // h3y/θ
      for (size_t b = 0; b < cells_.size(); ++b) values[k++] = 1.0;                   // hr/r
      for (size_t b = 0; b < cells_.size(); ++b) values[k++] = 1.0;                   // hd/δ
      if (ha_ >= 0) values[k++] = 1.0;                                                // ha/α
      for (const Cell& q : cells_) values[k++] = 1.0 - x[q.d];                        // comp/r
      for (const Cell& q : cells_) values[k++] = -x[q.r];                             // comp/δ
      for (size_t l = 0; l < links_.size(); ++l) values[k++] = 1.0;                   // link/copy
      for (size_t l = 0; l < links_.size(); ++l) values[k++] = -1.0;                  // link/orig
   }

   void hess_structure(std::vector<int>& row, std::vector<int>& col) const override {
      row = hess_row_;
      col = hess_col_;
   }

   // Hessian of σ·f + λᵀg, lower triangle, in the order of hess_row_/hess_col_.
   void hess_values(const double* x, double obj_factor, const double* lam,
                    double* values) override {
      int k = 0;
      for (size_t i = 0; i < u_orig_.size(); ++i) values[k++] = obj_factor;          // (u,u)
      for (const Cell& q : cells_)                                                  // (θ,r)
         values[k++] = lam[q.h2x] * std::sin(x[q.th]) - lam[q.h2y] * std::cos(x[q.th]);
      for (const Cell& q : cells_)                                                  // (θ,δ)
         values[k++] = lam[q.h3x] * std::sin(x[q.th]) - lam[q.h3y] * std::cos(x[q.th]);
      for (const Cell& q : cells_) {                                                // (θ,θ)
         const double c = std::cos(x[q.th]), s = std::sin(x[q.th]);
         values[k++] = x[q.r] * (lam[q.h2x] * c + lam[q.h2y] * s)
                     + x[q.d] * (lam[q.h3x] * c + lam[q.h3y] * s)
                     + obj_factor * eps_theta;
      }
      for (const Cell& q : cells_)                                                  // (δ,r)
         values[k++] = penalty > 0.0 && penalty_hessian ? -lam[q.comp] - obj_factor * penalty
                                                        : -lam[q.comp];
      for (const auto& e : div_x_) values[k++] = e.v * lam[nodes_[e.r].h1];         // (α,qx)
      for (const auto& e : div_y_) values[k++] = e.v * lam[nodes_[e.r].h1];         // (α,qy)
      // (α,α) per tile: zero, because the weight enters linearly.  The entries
      // stay in the pattern so every α copy has a diagonal.
      for (size_t k2 = 0; k2 < tile_alpha_.size(); ++k2) values[k++] = 0.0;
      values[k++] = obj_factor * kRegAlpha;                                         // (α,α)
   }

   // The level gate and the μ-coupled continuation (see the header), plus the
   // μ-trace row for the solution file.  Collective.
   bool iteration(const Iterate& it) override {
      if (!pinned_.empty()) {   // the clean-up: no relaxation, no continuation
         if (diag && it.dual) {
            const Global G = gather(it);
            if (print) diagnose(G.it);
         }
         double comp = 0.0;
         for (const Cell& q : cells_) comp = std::max(comp, it.x[q.r] * (1.0 - it.x[q.d]));
         mu_trace.insert(mu_trace.end(), {(double)it.iter, it.mu, t, it.x[la_], dd::Comm::max(comp)});
         return true;
      }
      if (stop_at_tmin && it.iter > 0 && t <= t_min * (1.0 + 1e-9)) {
         reached_tmin = true;
         if (print) std::printf("  [t = t_min] stopping for the clean-up\n");
         return false;
      }
      if ((diag && it.dual) || (classify && it.zl && t <= t_min * (1.0 + 1e-9))) {
         const Global G = gather(it);   // on rank 0, in the global layout
         if (print && diag && it.dual) diagnose(G.it);
         if (print && classify && it.zl && t <= t_min * (1.0 + 1e-9)) classify_point(G.it);
      }
      if (relax_cells && it.zl && it.iter > 0 && t <= t_min * (1.0 + 1e-9)) flag_cells(it);
      double rw_max = 0.0;   // max r(1 − δ), for the penalty's gate
      if (penalty > 0.0) {
         double c = 0.0;
         for (const Cell& q : cells_) {
            const double r = it.x[q.r], w = 1.0 - it.x[q.d];
            c = std::max(c, std::min(r, w));
            rw_max = std::max(rw_max, r * w);
         }
         double cm[2] = {c, rw_max};
         dd::Comm::max(cm, 2);
         c = cm[0];
         rw_max = cm[1];
         comp_hist_.push_back(c);
         const size_t h = comp_hist_.size();
         if (penalty < penalty_max && c > std::pow(it.mu, 0.4) && h >= 4 &&
             c > 0.9 * comp_hist_[h - 4]) {
            penalty = std::min(penalty_max, 10.0 * penalty);
            comp_hist_.clear();
            ++penalty_raises;
            if (print) std::printf("  [penalty] max min(r, 1-d) = %.1e > mu^0.4 = %.1e, not decreasing:"
                                   " pi = %.0e\n", c, std::pow(it.mu, 0.4), penalty);
         }
      }
      if (it.iter > 0 && t <= t_min * (1.0 + 1e-9)) {
         const double gate = std::max(gate_floor, std::sqrt(t));
         if (it.inf_pr <= gate && it.inf_du <= gate && it.mu <= gate &&
             (penalty <= 0.0 || rw_max <= t_min)) {
            gate_fired = true;
            if (print) std::printf("  [level gate] t=%.2e  inf_pr=%.1e  inf_du=%.1e  mu=%.1e"
                        "  all <= sqrt(t) = %.1e: stopping\n",
                        t, it.inf_pr, it.inf_du, it.mu, gate);
            return false;
         }
      }
      double comp = 0.0;   // max r(1 − δ) at this iterate (global)
      for (const Cell& q : cells_) comp = std::max(comp, it.x[q.r] * (1.0 - it.x[q.d]));
      comp = dd::Comm::max(comp);
      double t_new = std::max(t_min, t_mu_scale * it.mu);
      if (t_new < t * t_rate) t_new = std::max(t_min, t * t_rate);
      if (t_comp_ratio > 0.0 && penalty <= 0.0) t_new = std::max(t_new, comp / t_comp_ratio);
      if (t_on_solved && !it.barrier_solved) t_new = t;
      if (t_new < t) {
         t = t_new;
         eps_theta = c_theta * t_new;
      }
      mu_trace.insert(mu_trace.end(), {(double)it.iter, it.mu, t, it.x[la_], comp});
      return true;
   }

   bool final_form() const override { return t <= t_min * (1.0 + 1e-9); }

   // r ≥ 0 (the slack of row hr, or r itself with bounds_as_rows = false) and
   // 1 − δ ≥ 0 (the upper bound of δ), per own cell
   void complementarity_bounds(char* lo, char* up) const override {
      for (const Cell& q : cells_) {
         lo[bounds_as_rows ? slack(q.hr) : q.r] = 1;
         up[q.d] = 1;
      }
   }

   // ---- the clean-up (see the header) ---------------------------------------
   // Pin the active set at pt (the final point of the continuation, local) and
   // move the multipliers into the new form, so that pt stays stationary as far
   // as it was: the product row's ξ goes into the pinned row (γ = a − ξw,
   // ν = b − ξr), and so does the multiplier of a dropped variable bound.
   // Returns the global counts of cells pinned at r = 0 and at δ = 1.
   // Returns the global counts of cells pinned at r = 0 only, δ = 1 only,
   // both, and neither.
   std::array<long, 4> begin_cleanup(NLPPoint& pt) {
      pinned_.assign(cells_.size(), 0);
      relaxed_.clear();
      const double eps = cleanup_eps > 0.0 ? cleanup_eps : std::sqrt(t_min);
      std::array<long, 4> cnt{0, 0, 0, 0};
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double r = pt.x[q.r], w = 1.0 - pt.x[q.d], xi = pt.lam[q.comp];
         char pin = r <= w ? kPinR : kPinD;
         if (r <= eps && w <= eps) pin = cleanup_biactive == 1 ? kPinR | kPinD
                                       : cleanup_biactive == 2 ? 0 : pin;
         pinned_[b] = pin;
         if (pin & kPinR) {
            pt.lam[q.hr] += xi * w - pt.zl[q.r];   // ∂/∂r: λ_hr + ξ(1−δ) − z_L,r
            pt.zl[q.r] = 0.0;
         }
         if (pin & kPinD) {
            pt.lam[q.hd] += -xi * r + pt.zu[q.d];  // ∂/∂δ: λ_hd − ξr + z_U,δ
            pt.zu[q.d] = 0.0;
         }
         if (pin == 0) {   // ξ into the bound multipliers of r ≥ 0 and δ ≤ 1
            pt.lam[q.hr] += xi * w;
            pt.zu[q.d] -= xi * r;
         }
         ++cnt[pin == kPinR ? 0 : pin == kPinD ? 1 : pin ? 2 : 3];
         pt.lam[q.comp] = 0.0;   // the product row is free now
         pt.szl[q.comp] = pt.szu[q.comp] = 0.0;
      }
      bounds_moved_ = false;
      for (long& c : cnt) c = dd::Comm::sum(c);
      return cnt;
   }
   // S-stationarity at the clean-up's final point pt: on the cells where the
   // free member is also ≤ eps (biactive), the MPCC multiplier of the pinned
   // constraint, γ = −λ_hr or ν = λ_hd, should be ≥ 0.  Global: biactive cells,
   // those with a multiplier < −tol, the smallest multiplier, max r·w.
   // Also: the largest violation of a pinned row (|r| or |w| of a pinned
   // member), and of complementarity (r·w).
   struct CleanupReport {
      long biactive = 0, wrong = 0;
      double min_mult = 0.0, max_rw = 0.0, max_pin = 0.0;
   };
   CleanupReport cleanup_report(const NLPPoint& pt, double eps, double tol) const {
      CleanupReport rep;
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double r = pt.x[q.r], w = 1.0 - pt.x[q.d];
         const char pin = pinned_[b];
         rep.max_rw = std::max(rep.max_rw, std::abs(r * w));
         if (pin & kPinR) rep.max_pin = std::max(rep.max_pin, std::abs(r));
         if (pin & kPinD) rep.max_pin = std::max(rep.max_pin, std::abs(w));
         if (r > eps || w > eps) continue;   // not biactive
         // MPCC multipliers of r ≥ 0 and w ≥ 0: the pinned rows' λ, or the
         // bound multipliers where not pinned (≥ 0 by construction)
         const double gam = pin & kPinR ? -pt.lam[q.hr] : 0.0;
         const double nu = pin & kPinD ? pt.lam[q.hd] : 0.0;
         ++rep.biactive;
         if (std::min(gam, nu) < -tol) ++rep.wrong;
         rep.min_mult = std::min({rep.min_mult, gam, nu});
      }
      rep.biactive = dd::Comm::sum(rep.biactive);
      rep.wrong = dd::Comm::sum(rep.wrong);
      rep.min_mult = dd::Comm::min(rep.min_mult);
      rep.max_rw = dd::Comm::max(rep.max_rw);
      rep.max_pin = dd::Comm::max(rep.max_pin);
      return rep;
   }

   std::string log_extra() const override {
      char b[48];
      if (penalty > 0.0) std::snprintf(b, sizeof b, "  t=%.2e pi=%.0e", t, penalty);
      else std::snprintf(b, sizeof b, "  t=%.2e", t);
      return b;
   }

   // Solution and multipliers on rank 0, in the global layout.
   void finalize(int, const double* x, const double* lam, double obj) override {
      solution = dd::Comm::gather(gvar_long(), x, n);
      multipliers = dd::Comm::gather(grow_long(), lam, mcon);
      objective = obj;
   }

   // Where is the dual residual ∇L − z_L + z_U?  Two log lines per iteration:
   //   diag  — per variable group: max |residual| and how many entries exceed √t
   //           (the level gate); groups split interior and consensus variables
   //           and copies, and the slacks by row type;
   //   worst — the cell of the largest entry: its state and multipliers.
   void diagnose(const Iterate& it) const {
      const double gate = std::max(gate_floor, std::sqrt(t));
      enum { U, UC, Q, QC, R, D, TH, A, CU, CQ, CA, SR, SD, SA, SC, NG };
      static const char* name[NG] = {"u", "u*", "q", "q*", "r", "δ", "θ", "α", "u'", "q'", "α'",
                                     "s_r", "s_δ", "s_α", "s_c"};
      double mx[NG] = {};
      long cnt[NG] = {};
      int arg[NG];
      std::fill(arg, arg + NG, -1);
      auto group_of = [&](int i) {
         if (i >= n) {
            const int row = rhr + (i - n);
            return row < rhd ? SR : row < rha ? SD : row == rha ? SA : SC;
         }
         if (i >= n_orig) {
            const int v = link_orig_[i - n_orig];
            return v < oqx ? CU : v == oa ? CA : CQ;
         }
         const bool cons = col_tile_[i] < 0;
         if (i < oqx) return cons ? UC : U;
         if (i < oR) return cons ? QC : Q;
         if (i < oD) return R;
         if (i < oTh) return D;
         if (i < oa) return TH;
         return A;
      };
      const int np = n + n_ineq;
      int worst = 0;
      for (int i = 0; i < np; ++i) {
         const double v = std::abs(it.dual[i]);
         const int gi = group_of(i);
         if (v > mx[gi]) { mx[gi] = v; arg[gi] = i; }
         if (v > gate) ++cnt[gi];
         if (v > std::abs(it.dual[worst])) worst = i;
      }
      std::printf("  diag %4d:", it.iter);
      for (int gi = 0; gi < NG; ++gi)
         if (arg[gi] >= 0 && mx[gi] > 0.0) std::printf("  %s %.1e/%ld", name[gi], mx[gi], cnt[gi]);
      std::printf("\n");
      // the cell (or node) behind the worst entry
      int cell = -1, node = -1;
      const int gw = group_of(worst);
      int v = worst >= n_orig && worst < n ? link_orig_[worst - n_orig] : worst;
      if (worst >= n) {
         const int row = rhr + (worst - n);
         cell = row < rhd ? row - rhr : row < rha ? row - rhd : row == rha ? -1 : row - rcomp;
      } else if (v < oqx) node = v - ou;
      else if (v < oqy) cell = v - oqx;
      else if (v < oR) cell = v - oqy;
      else if (v < oa) cell = (v - oR) % m_q;
      std::printf("  worst %4d: %s[%d] = %.2e", it.iter, name[gw], worst, it.dual[worst]);
      if (node >= 0)
         std::printf("  node (%d,%d) tile %d  u=%.3f  λ_h1=%.2e", node / N, node % N,
                     col_tile_[worst < n ? worst : 0], it.x[ou + node], it.lam[rh1 + node]);
      if (cell >= 0) {
         const int e = cell, si = n + (rcomp - rhr) + e;   // the comp slack of cell e
         std::printf("  cell (%d,%d) tile %d  r=%.2e 1-δ=%.2e θ=%.2f  λ: h2 %.1e,%.1e h3 %.1e,%.1e"
                     " hr %.1e hd %.1e comp %.1e  zU_δ=%.1e  s_c=%.1e zU_sc=%.1e",
                     e / nc, e % nc, col_tile_[oR + e], it.x[oR + e], 1.0 - it.x[oD + e],
                     it.x[oTh + e], it.lam[rh2x + e], it.lam[rh2y + e], it.lam[rh3x + e],
                     it.lam[rh3y + e], it.lam[rhr + e], it.lam[rhd + e], it.lam[rcomp + e],
                     it.zu[oD + e], it.x[si], it.zu[si]);
      }
      std::printf("\n");
      // what blocked the last step: the entry of [x | s] and its cell
      auto describe = [&](const char* what, int i) {
         if (i < 0) return;
         const int gi = group_of(i);
         int e = -1;
         if (i >= n) {
            const int row = rhr + (i - n);
            e = row < rhd ? row - rhr : row < rha ? row - rhd : row == rha ? -1 : row - rcomp;
         } else if (i >= oR && i < oa) {
            e = (i - oR) % m_q;
         }
         std::printf("  %s: %s[%d]", what, name[gi], i);
         if (e >= 0)
            std::printf(" cell (%d,%d) r=%.2e 1-δ=%.2e s_c=%.1e  z: δ %.1e s_r %.1e s_c %.1e",
                        e / nc, e % nc, it.x[oR + e], 1.0 - it.x[oD + e],
                        it.x[n + (rcomp - rhr) + e], it.zu[oD + e], it.zl[n + e],
                        it.zu[n + (rcomp - rhr) + e]);
      };
      std::printf("  block %4d:", it.iter);
      describe("α_pr", it.pr_block);
      describe("α_du", it.du_block);
      std::printf("\n");
   }

   // The MPCC reading of the current point (Scholtes back-translation, the
   // manuscript's certificate).  With G = r, H = 1 − δ, a the multiplier of
   // r ≥ 0, b that of δ ≤ 1 and ξ that of r(1 − δ) ≤ t,
   //     γ = a − ξ(1 − δ),   ν = b − ξ r
   // are the MPCC multipliers; ξ itself may grow like 1/√t on biactive cells
   // (Ralph & Wright 2004, eq. 33).  One line per threshold ε = c·√t (c = 3, 1;
   // corner pairs on r(1 − δ) = t sit at r = 1 − δ = √t):
   //   B   biactive cells (r ≤ ε, 1−δ ≤ ε), split by the sign class of (γ, ν) at
   //       τ = 1e-3·max|γ, ν|: S both ≥ −τ; M one |·| ≤ τ; C both < −τ; W opposite;
   //       the most negative γ and ν; max ξ on B and off B;
   //   𝒜   r ≤ ε < 1−δ: max |ν| there (0 at an exact point) and max ‖λ_h2‖;
   //   ℐ   1−δ ≤ ε < r: max |γ| there; and max ‖λ_h2‖ off 𝒜.
   bool bounds_changed() override {
      const bool b = bounds_moved_;
      bounds_moved_ = false;
      return b;
   }

   // Flag the cells with a clearly negative MPCC multiplier (see relax_cells).
   // Collective (the count of fresh flags).
   void flag_cells(const Iterate& it) {
      if (relaxed_.empty()) {
         relaxed_.assign(cells_.size(), 0);
         negative_run_.assign(cells_.size(), 0);
      }
      const double thr = std::max(relax_thr, std::pow(it.inf_du, relax_exp));
      long fresh = 0;
      for (size_t b = 0; b < cells_.size(); ++b) {
         if (relaxed_[b]) continue;
         const Cell& q = cells_[b];
         const double r = it.x[q.r], w = 1.0 - it.x[q.d];
         const double a = bounds_as_rows ? it.zl[slack(q.hr)] : it.zl[q.r];
         const double xi = it.zu[slack(q.comp)];
         const double gam = a - xi * w, nu = it.zu[q.d] - xi * r;
         if (std::min(gam, nu) >= -thr) negative_run_[b] = 0;
         else if (++negative_run_[b] >= relax_persist) {
            relaxed_[b] = 1;
            ++fresh;
         }
      }
      fresh = dd::Comm::sum(fresh);
      if (fresh == 0) return;
      relaxed_total += fresh;
      bounds_moved_ = true;
      if (print)
         std::printf("  [relax] %ld cells flagged (%ld in all), threshold %.2e\n", fresh,
                     relaxed_total, thr);
   }

   void classify_point(const Iterate& it) {
      const int sc0 = n + (rcomp - rhr);   // the comp slack of cell 0
      std::vector<double> gam(m_q), nu(m_q);
      mpcc_cells.resize(8 * (size_t)m_q);
      double scale = 0.0;
      for (int e = 0; e < m_q; ++e) {
         const double r = it.x[oR + e], w = 1.0 - it.x[oD + e];
         const double a = bounds_as_rows ? it.zl[n + e] : it.zl[oR + e];
         const double b = it.zu[oD + e], xi = it.zu[sc0 + e];
         gam[e] = a - xi * w;
         nu[e] = b - xi * r;
         scale = std::max({scale, std::abs(gam[e]), std::abs(nu[e])});
         double* row = &mpcc_cells[8 * (size_t)e];
         row[0] = r, row[1] = w, row[2] = xi, row[3] = a, row[4] = b, row[5] = gam[e],
         row[6] = nu[e], row[7] = std::hypot(it.lam[rh2x + e], it.lam[rh2y + e]);
      }
      const double tau = 1e-3 * scale;
      for (double c : {3.0, 1.0}) {
         const double eps = c * std::sqrt(t);
         long nB = 0, nA = 0, nI = 0, nS = 0, nM = 0, nC = 0, nW = 0;
         double nuA = 0, gamI = 0, gmin = 0, nmin = 0, xiB = 0, xiO = 0, lamA = 0, lamO = 0;
         for (int e = 0; e < m_q; ++e) {
            const double r = it.x[oR + e], w = 1.0 - it.x[oD + e], xi = it.zu[sc0 + e];
            const double g = gam[e], v = nu[e];
            const double lh = std::hypot(it.lam[rh2x + e], it.lam[rh2y + e]);
            const bool rs = r <= eps, ws = w <= eps;
            if (rs && ws) {
               ++nB;
               xiB = std::max(xiB, xi);
               gmin = std::min(gmin, g);
               nmin = std::min(nmin, v);
               if (g >= -tau && v >= -tau) ++nS;
               else if (std::min(std::abs(g), std::abs(v)) <= tau) ++nM;
               else if (g < 0 && v < 0) ++nC;
               else ++nW;
            } else {
               xiO = std::max(xiO, xi);
               if (rs) { ++nA; nuA = std::max(nuA, std::abs(v)); }
               else if (ws) { ++nI; gamI = std::max(gamI, std::abs(g)); }
            }
            if (rs && !ws) lamA = std::max(lamA, lh);
            else lamO = std::max(lamO, lh);
         }
         std::printf("  stat %4d ε=%.1e: B %ld [S %ld M %ld C %ld W %ld, min γ %.1e ν %.1e, ξ %.1e"
                     " (off B %.1e)]  𝒜 %ld [|ν| %.1e, λ_h2 %.1e]  ℐ %ld [|γ| %.1e]  λ_h2 off 𝒜 %.1e"
                     "  τ %.0e\n",
                     it.iter, eps, nB, nS, nM, nC, nW, gmin, nmin, xiB, xiO, nA, nuA, lamA, nI,
                     gamI, lamO, tau);
      }
   }

private:
   static constexpr double kRegAlpha = 1e-4;   // ridge on α
   static constexpr double kAlphaMax = 2e19;   // no upper bound on α in practice

   struct Tri { int r, c; double v; };         // one nonzero of a sparse operator

   std::vector<double> clean_, f_, theta_ref_;
   std::vector<char> relaxed_;   // per local cell: flagged by flag_cells()
   static constexpr char kPinR = 1, kPinD = 2;
   std::vector<char> pinned_;    // per local cell, in the clean-up: r = 0 or δ = 1
   std::vector<double> comp_hist_;   // max min(r, 1 − δ) per iteration (penalty update)
public:
   long penalty_raises = 0;
private:
   std::vector<int> negative_run_;   // ... consecutive iterations with γ or ν < −relax_thr
   bool bounds_moved_ = false;
   std::vector<Tri> Kx_, Ky_, KxT_, KyT_;      // gradient stencils and transposes (setup only)
   int n_tiles_ = 0;

   // ---- global (every rank, the same)
   std::vector<int> col_tile_;   // variable → tile, −1 for consensus variables
   std::vector<int> row_tile_;   // row      → tile (setup only)
   std::vector<int> link_copy_, link_orig_;   // linking row ℓ: copy − original = 0
   // "Effective index" tables, global indices (setup only): which variable
   // (original or copy) each row uses.
   std::vector<int> u_of_h1_;        // h1 row i  → the u  its tile uses
   std::vector<int> alpha_of_h1_;    // h1 row i  → the α  its tile uses
   std::vector<int> alpha_of_tile_;  // tile      → its α copy
   std::vector<int> qx_of_h3_, qy_of_h3_;   // h3 row e → the qx / qy its tile uses

   // ---- local (this rank's tiles and the consensus variables); all indices local
   std::vector<int> gvar_, grow_;    // local variable / row → global
   std::vector<int> vtile_, rtile_;  // their tiles
   int la_ = -1;                     // the consensus α
   int ha_ = -1, ha_alpha_ = -1;     // row ha and the α it reads (on tile 0's rank)
   int first_slack_row_ = 0;         // first local inequality row (one slack each)
   struct Node {                     // an own node i: its h1 row, the u and α it reads
      int i, h1, u, a;
   };
   struct Cell {                     // an own cell e: its variables and rows
      int e, r, d, th, qx, qy;       // qx, qy: the copies h3 reads
      int h2x, h2y, h3x, h3y, hr, hd, comp;
   };
   struct Link { int row, copy, orig; };
   std::vector<Node> nodes_;
   std::vector<Cell> cells_;
   std::vector<Link> links_;
   std::vector<int> u_orig_;          // the original u variables held here (diagonal of f)
   std::vector<int> tile_alpha_;      // the α copy of each own tile
   std::vector<Tri> div_x_, div_y_;   // Kxᵀqx, Kyᵀqy in h1: (node slot, variable, value)
   std::vector<Tri> grad_x_, grad_y_; // Kx u, Ky u in h2: (cell slot, variable, value)
   std::vector<int> jac_row_, jac_col_, hess_row_, hess_col_;
   dd::TileSum sum_;

   // The objective's term of local variable l.
   double objective_term(int l, double x) const {
      const int v = gvar_[l];
      if (v < ou + m_u) return 0.5 * (x - clean_[v - ou]) * (x - clean_[v - ou]);
      if (v == oa) return 0.5 * kRegAlpha * x * x;
      if (eps_theta != 0.0 && v >= oTh && v < oTh + m_q)
         return 0.5 * eps_theta * (x - theta_ref_[v - oTh]) * (x - theta_ref_[v - oTh]);
      return 0.0;
   }
   // The IPM's slack of local inequality row i, in [x | s].
   int slack(int i) const { return num_vars() + (i - first_slack_row_); }
   std::vector<long> gvar_long() const { return {gvar_.begin(), gvar_.end()}; }
   std::vector<long> grow_long() const { return {grow_.begin(), grow_.end()}; }

   // An iterate in the global layout, on rank 0 (for the diagnostic printouts).
   struct Global {
      std::vector<double> x, dual, lam, zl, zu;
      Iterate it{};
   };
   Global gather(const Iterate& it) const {
      const int nl = num_vars(), ml = num_rows(), np = nl + (ml - first_slack_row_);
      const long NP = n + n_ineq;
      std::vector<long> gp(np);
      for (int l = 0; l < nl; ++l) gp[l] = gvar_[l];
      for (int i = first_slack_row_; i < ml; ++i) gp[slack(i)] = n + (grow_[i] - n_eq);
      Global G;
      G.x = dd::Comm::gather(gp, it.x, NP);
      if (it.dual) G.dual = dd::Comm::gather(gp, it.dual, NP);
      if (it.lam) G.lam = dd::Comm::gather(grow_long(), it.lam, mcon);
      if (it.zl) G.zl = dd::Comm::gather(gp, it.zl, NP);
      if (it.zu) G.zu = dd::Comm::gather(gp, it.zu, NP);
      const double blk[2] = {it.pr_block >= 0 ? (double)gp[it.pr_block] : -1.0,
                             it.du_block >= 0 ? (double)gp[it.du_block] : -1.0};
      G.it = it;
      G.it.pr_block = (int)dd::Comm::max(blk[0]);
      G.it.du_block = (int)dd::Comm::max(blk[1]);
      G.it.x = G.x.data();
      G.it.dual = it.dual ? G.dual.data() : nullptr;
      G.it.lam = it.lam ? G.lam.data() : nullptr;
      G.it.zl = it.zl ? G.zl.data() : nullptr;
      G.it.zu = it.zu ? G.zu.data() : nullptr;
      return G;
   }

   // ---- the unsplit layout --------------------------------------------------
   void layout_unsplit() {
      ou = 0;
      oqx = m_u;
      oqy = m_u + m_q;
      oR = m_u + 2 * m_q;
      oD = m_u + 3 * m_q;
      oTh = m_u + 4 * m_q;
      oa = m_u + 5 * m_q;
      n = m_u + 5 * m_q + 1;

      rh1 = 0;
      rh2x = m_u;
      rh2y = m_u + m_q;
      rh3x = m_u + 2 * m_q;
      rh3y = m_u + 3 * m_q;
      rhr = m_u + 4 * m_q;
      rhd = m_u + 5 * m_q;
      rha = m_u + 6 * m_q;
      rcomp = rha + 1;
      mcon = rcomp + m_q;
      n_eq = rhr;               // h1..h3y are the equalities
      n_ineq = mcon - n_eq;
   }

   // Forward differences, both anchored at the cell's top-left node:
   //   (Kx u)_{a,b} = u[a,   b+1] − u[a, b]
   //   (Ky u)_{a,b} = u[a+1, b  ] − u[a, b]
   void build_stencils() {
      auto node = [&](int i, int j) { return i * N + j; };
      for (int a = 0; a < nc; ++a)
         for (int b = 0; b < nc; ++b) {
            const int cell = a * nc + b;
            Kx_.push_back({cell, node(a, b), -1.0});
            Kx_.push_back({cell, node(a, b + 1), 1.0});
            Ky_.push_back({cell, node(a, b), -1.0});
            Ky_.push_back({cell, node(a + 1, b), 1.0});
         }
      auto by_row_then_col = [](const Tri& p, const Tri& q) {
         return p.r != q.r ? p.r < q.r : p.c < q.c;
      };
      std::sort(Kx_.begin(), Kx_.end(), by_row_then_col);
      std::sort(Ky_.begin(), Ky_.end(), by_row_then_col);
      for (const auto& e : Kx_) KxT_.push_back({e.c, e.r, e.v});
      for (const auto& e : Ky_) KyT_.push_back({e.c, e.r, e.v});
      std::sort(KxT_.begin(), KxT_.end(), by_row_then_col);
      std::sort(KyT_.begin(), KyT_.end(), by_row_then_col);
   }

   void apply(const std::vector<Tri>& K, const double* v, int len,
              std::vector<double>& out) const {
      out.assign(len, 0.0);
      for (const auto& e : K) out[e.r] += e.v * v[e.c];
   }

   // ---- warm start ----------------------------------------------------------
   // Solve plain TV denoising  min_u ½‖u − f‖² + λ·‖∇u‖  by Chambolle–Pock (an
   // accelerated phase, then fixed steps), then lift the answer to the MPCC
   // variables.  At that point u, q already satisfy h1 almost exactly, so the IPM
   // starts next to the lower-level solution instead of at u = f.
   //
   // θ is taken from the DUAL q, not from ∇u: that makes h3 exact and leaves only
   // noise-level error on h2.
   void warm_start_chambolle_pock(double lam) {
      const double norm_K = std::sqrt(8.0);           // ‖K‖ for this stencil
      const double tau0 = 0.99 / (norm_K * lam), sig0 = 0.99 / norm_K;
      double tau = tau0, sig_lam = sig0;
      std::vector<double> u = f_, ubar = u, u_new(m_u), kxu, kyu, div;
      std::vector<double> qx(m_q, 0.0), qy(m_q, 0.0);
      for (int it = 0; it < 3000; ++it) {
         const bool accelerated = it < 300;
         if (!accelerated && tau != tau0) { tau = tau0; sig_lam = sig0; }
         // dual step, then project q onto the unit ball
         apply(Kx_, ubar.data(), m_q, kxu);
         apply(Ky_, ubar.data(), m_q, kyu);
         for (int e = 0; e < m_q; ++e) {
            qx[e] += sig_lam * kxu[e];
            qy[e] += sig_lam * kyu[e];
            const double nrm = std::max(1.0, std::hypot(qx[e], qy[e]));
            qx[e] /= nrm;
            qy[e] /= nrm;
         }
         // primal step
         div.assign(m_u, 0.0);
         for (const auto& e : KxT_) div[e.r] += e.v * qx[e.c];
         for (const auto& e : KyT_) div[e.r] += e.v * qy[e.c];
         double residual = 0.0;
         for (int i = 0; i < m_u; ++i) {
            u_new[i] = (tau * f_[i] + u[i] - tau * lam * div[i]) / (tau + 1.0);
            residual = std::max(residual, std::abs(u_new[i] - u[i]));
         }
         residual /= tau;
         // extrapolation
         if (accelerated) {
            const double th = 1.0 / std::sqrt(1.0 + 2.0 * tau);
            for (int i = 0; i < m_u; ++i) ubar[i] = u_new[i] + th * (u_new[i] - u[i]);
            tau *= th;
            sig_lam /= th;
         } else {
            for (int i = 0; i < m_u; ++i) ubar[i] = 2.0 * u_new[i] - u[i];
         }
         u = u_new;
         if (residual <= 1e-9) break;
      }

      std::vector<double> gx, gy;
      apply(Kx_, u.data(), m_q, gx);
      apply(Ky_, u.data(), m_q, gy);
      x_start.assign(n, 0.0);
      for (int i = 0; i < m_u; ++i) x_start[ou + i] = u[i];
      for (int e = 0; e < m_q; ++e) {
         x_start[oqx + e] = qx[e];
         x_start[oqy + e] = qy[e];
         x_start[oR + e] = std::hypot(gx[e], gy[e]);
         x_start[oD + e] = std::hypot(qx[e], qy[e]);
         x_start[oTh + e] = std::atan2(qy[e], qx[e]);
      }
      x_start[oa] = lam;
   }

   // ---- the consensus split -------------------------------------------------
   void split_into_consensus_form(const Partition& part) {
      n_orig = n;
      const int m_orig = mcon, n_eq_orig = n_eq;
      n_tiles_ = part.n_tiles;
      const std::vector<int>& node_tile = part.node_tile;
      const std::vector<int>& cell_tile = part.cell_tile;

      // 1. Which tiles' rows reference each original variable?
      //    u (node i): its own h1 row, plus every h2 stencil that reads it.
      //    qx/qy (cell e): its own h3 row, plus every h1 divergence that reads it.
      //    r/δ/θ: only their own cell's rows, so never shared.   α: every tile.
      std::vector<std::vector<int>> tiles(n_orig);
      auto touch = [&](int v, int k) {
         auto& list = tiles[v];
         if (std::find(list.begin(), list.end(), k) == list.end()) list.push_back(k);
      };
      for (int i = 0; i < m_u; ++i) touch(ou + i, node_tile[i]);
      for (const auto& e : Kx_) touch(ou + e.c, cell_tile[e.r]);
      for (const auto& e : Ky_) touch(ou + e.c, cell_tile[e.r]);
      for (int e = 0; e < m_q; ++e) {
         touch(oqx + e, cell_tile[e]);
         touch(oqy + e, cell_tile[e]);
         touch(oR + e, cell_tile[e]);
         touch(oD + e, cell_tile[e]);
         touch(oTh + e, cell_tile[e]);
      }
      for (const auto& e : KxT_) touch(oqx + e.c, node_tile[e.r]);
      for (const auto& e : KyT_) touch(oqy + e.c, node_tile[e.r]);
      for (int k = 0; k < n_tiles_; ++k) touch(oa, k);
      for (auto& list : tiles) std::sort(list.begin(), list.end());

      // 2. Give every shared variable one copy per tile.  Copies of one variable
      //    are contiguous, and α's copies come last so that every (α, ·) Hessian
      //    entry stays in the lower triangle.
      std::vector<int> first_copy(n_orig, -1);
      std::vector<int> copy_tile;
      col_tile_.assign(n_orig, 0);
      int next = n_orig;
      auto make_copies = [&](int v) {
         if ((int)tiles[v].size() <= 1) {          // used by one tile only: not shared
            col_tile_[v] = tiles[v].empty() ? 0 : tiles[v][0];
            return;
         }
         col_tile_[v] = -1;                        // becomes a consensus variable
         first_copy[v] = next;
         for (int k : tiles[v]) {
            link_copy_.push_back(next);
            link_orig_.push_back(v);
            copy_tile.push_back(k);
            ++next;
         }
      };
      for (int i = 0; i < m_u; ++i) make_copies(ou + i);
      for (int e = 0; e < m_q; ++e) make_copies(oqx + e);
      for (int e = 0; e < m_q; ++e) make_copies(oqy + e);
      for (int e = 0; e < m_q; ++e) {
         make_copies(oR + e);
         make_copies(oD + e);
         make_copies(oTh + e);
      }
      make_copies(oa);
      n_link = (int)link_copy_.size();
      col_tile_.resize(next);
      for (int c = 0; c < n_link; ++c) col_tile_[n_orig + c] = copy_tile[c];

      // The variable that tile k should use in place of original variable v.
      auto seen_by = [&](int v, int k) {
         if (first_copy[v] < 0) return v;
         const auto& list = tiles[v];
         for (size_t j = 0; j < list.size(); ++j)
            if (list[j] == k) return first_copy[v] + (int)j;
         return v;
      };

      // 3. New sizes.  Linking rows extend the equality block, so every
      //    inequality block shifts down by n_link.
      n = next;
      rlink = n_eq_orig;
      n_eq = n_eq_orig + n_link;
      rhr += n_link;
      rhd += n_link;
      rha += n_link;
      rcomp += n_link;
      mcon = m_orig + n_link;

      // 4. Row ownership.
      row_tile_.assign(mcon, 0);
      for (int i = 0; i < m_u; ++i) row_tile_[rh1 + i] = node_tile[i];
      for (int block : {rh2x, rh2y, rh3x, rh3y, rhr, rhd, rcomp})
         for (int e = 0; e < m_q; ++e) row_tile_[block + e] = cell_tile[e];
      row_tile_[rha] = 0;
      for (int l = 0; l < n_link; ++l) row_tile_[rlink + l] = copy_tile[l];

      // 5. Effective-index tables: each row reads its own tile's copies.
      u_of_h1_.resize(m_u);
      alpha_of_h1_.resize(m_u);
      for (int i = 0; i < m_u; ++i) {
         u_of_h1_[i] = seen_by(ou + i, node_tile[i]);
         alpha_of_h1_[i] = seen_by(oa, node_tile[i]);
      }
      alpha_of_tile_.resize(n_tiles_);
      for (int k = 0; k < n_tiles_; ++k) alpha_of_tile_[k] = seen_by(oa, k);
      qx_of_h3_.resize(m_q);
      qy_of_h3_.resize(m_q);
      for (int e = 0; e < m_q; ++e) {
         qx_of_h3_[e] = seen_by(oqx + e, cell_tile[e]);
         qy_of_h3_[e] = seen_by(oqy + e, cell_tile[e]);
      }

      for (const auto& e : KxT_) div_x_.push_back({e.r, seen_by(oqx + e.c, node_tile[e.r]), e.v});
      for (const auto& e : KyT_) div_y_.push_back({e.r, seen_by(oqy + e.c, node_tile[e.r]), e.v});
      for (const auto& e : Kx_) grad_x_.push_back({e.r, seen_by(ou + e.c, cell_tile[e.r]), e.v});
      for (const auto& e : Ky_) grad_y_.push_back({e.r, seen_by(ou + e.c, cell_tile[e.r]), e.v});

      // 6. Copies start equal to their consensus value, so the start is
      //    exactly feasible for the linking rows.
      x_start.resize(n);
      for (int l = 0; l < n_link; ++l) x_start[link_copy_[l]] = x_start[link_orig_[l]];
   }

   // ---- the local view ------------------------------------------------------
   // This rank's variables (its tiles' and the consensus ones) and rows, in
   // ascending global order; the own nodes and cells with local indices; the
   // stencils restricted to the own rows; and the patterns, in the global
   // block order restricted to the own rows (so a tile's entries come in the
   // same order on any number of ranks).  The global setup tables are freed.
   // div_x_ etc. hold global (node/cell, variable) on entry, and (slot, local
   // variable) on exit.
   void build_local(const Partition& part) {
      const int k0 = dd::Comm::first_tile(dd::Comm::rank(), n_tiles_);
      const int k1 = dd::Comm::first_tile(dd::Comm::rank() + 1, n_tiles_);
      auto own = [&](int k) { return k >= k0 && k < k1; };

      std::vector<int> lv(n, -1), lr(mcon, -1);
      for (int v = 0; v < n; ++v)
         if (col_tile_[v] < 0 || own(col_tile_[v])) {
            lv[v] = (int)gvar_.size();
            gvar_.push_back(v);
            vtile_.push_back(col_tile_[v]);
         }
      for (int r = 0; r < mcon; ++r)
         if (own(row_tile_[r])) {
            lr[r] = (int)grow_.size();
            grow_.push_back(r);
            rtile_.push_back(row_tile_[r]);
         }
      first_slack_row_ = (int)(std::lower_bound(grow_.begin(), grow_.end(), n_eq) - grow_.begin());
      la_ = lv[oa];
      if (own(row_tile_[rha])) { ha_ = lr[rha]; ha_alpha_ = lv[alpha_of_tile_[0]]; }

      std::vector<int> node_slot(m_u, -1), cell_slot(m_q, -1);
      for (int i = 0; i < m_u; ++i)
         if (own(part.node_tile[i])) {
            node_slot[i] = (int)nodes_.size();
            nodes_.push_back({i, lr[rh1 + i], lv[u_of_h1_[i]], lv[alpha_of_h1_[i]]});
         }
      for (int e = 0; e < m_q; ++e)
         if (own(part.cell_tile[e])) {
            cell_slot[e] = (int)cells_.size();
            cells_.push_back({e, lv[oR + e], lv[oD + e], lv[oTh + e], lv[qx_of_h3_[e]],
                              lv[qy_of_h3_[e]], lr[rh2x + e], lr[rh2y + e], lr[rh3x + e],
                              lr[rh3y + e], lr[rhr + e], lr[rhd + e], lr[rcomp + e]});
         }
      for (int l = 0; l < n_link; ++l)
         if (lr[rlink + l] >= 0) links_.push_back({lr[rlink + l], lv[link_copy_[l]], lv[link_orig_[l]]});
      for (int i = 0; i < m_u; ++i)
         if (lv[ou + i] >= 0) u_orig_.push_back(lv[ou + i]);
      for (int k = k0; k < k1; ++k) tile_alpha_.push_back(lv[alpha_of_tile_[k]]);
      auto restrict_to = [&](std::vector<Tri>& K, const std::vector<int>& slot) {
         std::vector<Tri> out;
         for (const Tri& e : K)
            if (slot[e.r] >= 0) out.push_back({slot[e.r], lv[e.c], e.v});
         K.swap(out);
      };
      restrict_to(div_x_, node_slot);
      restrict_to(div_y_, node_slot);
      restrict_to(grad_x_, cell_slot);
      restrict_to(grad_y_, cell_slot);

      // the patterns; jac_values and hess_values fill values in THIS order
      auto J = [&](int r, int c) { jac_row_.push_back(r); jac_col_.push_back(c); };
      for (const Node& q : nodes_) J(q.h1, q.u);
      for (const auto& e : div_x_) J(nodes_[e.r].h1, e.c);
      for (const auto& e : div_y_) J(nodes_[e.r].h1, e.c);
      for (const Node& q : nodes_) J(q.h1, q.a);
      for (const auto& e : grad_x_) J(cells_[e.r].h2x, e.c);
      for (const Cell& q : cells_) J(q.h2x, q.r);
      for (const Cell& q : cells_) J(q.h2x, q.th);
      for (const auto& e : grad_y_) J(cells_[e.r].h2y, e.c);
      for (const Cell& q : cells_) J(q.h2y, q.r);
      for (const Cell& q : cells_) J(q.h2y, q.th);
      for (const Cell& q : cells_) J(q.h3x, q.qx);
      for (const Cell& q : cells_) J(q.h3x, q.d);
      for (const Cell& q : cells_) J(q.h3x, q.th);
      for (const Cell& q : cells_) J(q.h3y, q.qy);
      for (const Cell& q : cells_) J(q.h3y, q.d);
      for (const Cell& q : cells_) J(q.h3y, q.th);
      for (const Cell& q : cells_) J(q.hr, q.r);
      for (const Cell& q : cells_) J(q.hd, q.d);
      if (ha_ >= 0) J(ha_, ha_alpha_);
      for (const Cell& q : cells_) J(q.comp, q.r);
      for (const Cell& q : cells_) J(q.comp, q.d);
      for (const Link& l : links_) J(l.row, l.copy);
      for (const Link& l : links_) J(l.row, l.orig);

      auto H = [&](int r, int c) { hess_row_.push_back(r); hess_col_.push_back(c); };
      for (int l : u_orig_) H(l, l);
      for (const Cell& q : cells_) H(q.th, q.r);
      for (const Cell& q : cells_) H(q.th, q.d);
      for (const Cell& q : cells_) H(q.th, q.th);
      for (const Cell& q : cells_) H(q.d, q.r);
      for (const auto& e : div_x_) H(nodes_[e.r].a, e.c);
      for (const auto& e : div_y_) H(nodes_[e.r].a, e.c);
      for (int l : tile_alpha_) H(l, l);
      H(la_, la_);

      std::vector<double> xs(gvar_.size());
      for (size_t l = 0; l < gvar_.size(); ++l) xs[l] = x_start[gvar_[l]];
      x_start.swap(xs);
      for (auto* v : {&row_tile_, &u_of_h1_, &alpha_of_h1_, &qx_of_h3_, &qy_of_h3_})
         std::vector<int>().swap(*v);
      for (auto* v : {&Kx_, &Ky_, &KxT_, &KyT_}) std::vector<Tri>().swap(*v);
   }
};
