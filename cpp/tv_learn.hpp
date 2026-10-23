// tv_learn.hpp — learning a (block-wise) TV weight map over a set of images,
// for the lower-level problems of operators.hpp, as one smooth NLP in consensus
// form for the domain decomposition.  tv_mpcc.hpp's problem, generalized:
//
//   min   (ℓ)·Σ_s ½‖u_s − u_clean,s‖² + ½·reg_α·‖α‖² + (ℓ)·½·ε_θ·Σ_s ‖θ_s − θ_ref,s‖²
//   s.t.  for every image s, on its nodes and cells:
//         h1  : A_s u_s − F_s + K_ξᵀ(α∘q_s) = 0       optimality of u_s
//         h2  : K_ξ u_s − r_s·(cosθ_s, sinθ_s) = 0      K_ξ u in polar form
//         h3  : q_s − δ_s·(cosθ_s, sinθ_s) = 0          q in polar form
//         hr  : r ≥ 0      hd : δ ≥ 0      ha : α_b ≥ 0 (one row per block)
//         comp: r·(1 − δ) − t ≤ 0                       relaxed complementarity
//         and the bound δ ≤ 1;
//
// CARTESIAN (cartesian_gamma = γ > 0 at construction; stage 1 of --start ipm):
// the smoothed problem, without the lifting,
//   x = [ u | qx | qy | α | copies ],   rows = [ h1 | cx | cy | links | ha ],
//         c   : q − h(K_ξu) = 0                          (cx, cy per cell)
// with h the C¹ Huber clipping of the dual (De los Reyes et al., the draft's
// h_γ with α taken out: α multiplies q in h1), c = 1/γ:
//   h(g) = c·g                                  if c|g| ≤ 1 − 1/(2c)
//        = g/|g|                                if c|g| ≥ 1 + 1/(2c)
//        = g/|g|·(1 − (c/2)(1 − c|g| + 1/(2c))²)  between,
// a C¹ version of g/max(γ, |g|): the lower level is strictly convex (u unique
// for each α), the rows are linear in q (∂c/∂q = I) and in u near g = 0.  c
// reads what h2 and h3 read, so the copies and the sharing are those of the
// lifted problem.
// ℓ = 1/S for the mean loss, 1 for the sum.  α_e = α_{b(e)} is constant on the
// blocks of learn_partition.hpp (one block: a scalar α) and shared by all
// images.  A, F and K_ξ are those of operators.hpp: K_ξ = K (TV) or M_e·K
// (directional TV), A = I, a mask's diagonal, or CᵀC (blur).
//
// With one image, A = I, K_ξ = K and one block this IS tv_mpcc.hpp's NLP, in the
// same order (variables, rows, nonzeros, and the order of every sum), so a run
// is bit-identical to tv_dd's.  That is the check of this file.
//
// LAYOUT (global; field-major, so one image is tv_mpcc.hpp's layout)
//   x    = [ u (S·N²) | qx | qy | r | δ | θ (each S·(N−1)²) | α (n_blocks) | copies ]
//   rows = [ h1 | h2x | h2y | h3x | h3y | links | hr | hd | ha (n_blocks) | comp ]
// image s owns entries s·N² + i of u and s·(N−1)² + e of each cell field.
//
// SHARING.  A variable read by rows of ≥ 2 tiles gets one copy per tile and a
// linking row, as in tv_mpcc.hpp; who reads what follows from the stencils:
//   u (node j of image s): its own h1 row, the h1 rows of the nodes within R of
//     j (R = 2w for a blur of radius w, else 0), and the h2 rows of the cells
//     (i, j), (i, j−1), (i−1, j) (K and K_ξ alike);
//   qx, qy (cell (a, b)): its own h3 row, and the h1 rows of the nodes (a, b)
//     and (a, b+1) (qx) or (a+1, b) (qy); for K_ξ both of these for both;
//   α_b: the h1 rows of the nodes next to block b (learn_partition.hpp), in
//     every image.
// The copies are numbered after the originals: u (by image, node), qx, qy (by
// image, cell), then α (by block); each run in ascending tile order.
//
// h1 sums its weight terms by α variable, Σ_g α_g·(Σ ±q): with one block every
// node has one group and the value is tv_mpcc.hpp's α·div exactly.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "comm.hpp"
#include "learn_partition.hpp"
#include "nlp.hpp"
#include "operators.hpp"
#include "pdhg.hpp"

// A start for every image: a lower-level solution u at the starting weights and its
// dual q (|q| ≤ 1), e.g. the smoothed method's (smoothed.hpp); lifted as it is.
struct StartFields {
   std::vector<std::vector<double>> u, qx, qy;
   bool theta_from_u = false;   // θ = the angle of K_ξu (the interior start, q = 0)
};

class TvLearn : public NLP {
public:
   // ---- continuation state (as tv_mpcc.hpp) -------------------------------
   double t = 1.0, t_min = 1e-5, t_mu_scale = 10.0, t_rate = 0.5, t_comp_ratio = 0.0;
   bool t_on_solved = false;
   double c_theta = 1.0, eps_theta = 0.0;
   double gate_floor = 1e-8;
   bool gate_dual = false;
   double gate_du_scale = 1.0;
   bool gate_fired = false;
   bool print = true;
   bool bounds_as_rows = true;
   // the Cartesian problem: Definition 1's Huber (De los Reyes–Valkonen; C¹, the clipping
   // g/max(γ, |g|), as smoothed.hpp) instead of the C² variant (3.2) of the top of the file
   bool cart_c1 = false;
   // ... or Definition 1's Huber in the primal-dual form of De los Reyes et al. (4.1b):
   // c = (max(γ, |g|)·q − g)/γ = 0, q its own unknown (q − g/γ for |g| < γ, as above);
   // its Jacobian reads q (the q ĝᵀ term of their semismooth Newton, unprojected: the
   // IPM's line search globalizes instead)
   bool cart_dual = false;
   // ... and with the dual projected onto |q| ≤ 1 where it enters the Jacobian and the
   // Hessian ((SSN-3) of De los Reyes et al.; the rows themselves stay exact): the
   // derivatives are then those of an inexact Newton model, not of c
   bool cart_dual_proj = false;
   // ... or the rows q − h(K_ξu) of cart_c1 / the C² Huber unchanged, with the derivative of
   // h written with the dual itself (the earlier code, ~/Downloads/bimpcc_2, utils_reg.py):
   // where h is not linear, ∂h/∂g = φ(I − ĝĝᵀ) + ψ'ĝĝᵀ with φĝĝᵀ = h ĝᵀ/s replaced by
   // q̄ĝᵀ/s, q̄ = q projected onto |q| ≤ 1 (Hintermüller–Stadler); exact where q = h(g)
   bool cart_proj = false;
   // the Hessian of the Lagrangian without the constraints' curvature (the objective's
   // only: ℓ on u, ρ_α on α); the reduced Hessian is then ‖∂u/∂α·‖² + ρ_α‖·‖², positive
   // definite, as a quasi-Newton model (the earlier code ran IPOPT with L-BFGS)
   bool cart_gn = false;
   // q, or its projection onto the unit ball (cart_dual_proj), for the derivatives
   void cart_qbar(const double* x, size_t b, double& qx, double& qy) const {
      qx = x[cells_[b].qx];
      qy = x[cells_[b].qy];
      if (!cart_dual_proj) return;
      const double n = std::hypot(qx, qy);
      if (n > 1.0) { qx /= n; qy /= n; }
   }
   // the Cartesian problem's γ (see the top of the file; 0: the lifted MPCC)
   double cart_gamma() const { return cart_gamma_; }
   // The MPCC multipliers at the last iterate with t = t_min (classify = true): per cell
   // γ = a − ξ(1 − δ), ν = b − ξ r (a, b: multipliers of r ≥ 0 and δ ≤ 1; ξ: of the
   // product row), and the class of the biactive cells (r, 1 − δ ≤ ε): S (γ, ν ≥ −τ),
   // M (one of them within τ of 0), C (both < 0), W (the rest), τ = 1e-3·max|γ, ν|;
   // for ε = 3√t and √t.  As tv_mpcc.hpp's classify_point, counted over all ranks.
   bool classify = false;
   // the lifted start's products r(1 − δ) above 1e-12, one per cell where the exact
   // complementarity fails (from the smoothed method: its Huber band 0 < |K_ξu| < γ,
   // up to γ/4; from PDHG: its inexactness), and the number of cells (all images)
   std::vector<double> start_products;
   long start_cells = 0;
   struct Classes {
      int iter = -1;
      double eps[2] = {0, 0}, tau = 0, gmin[2] = {0, 0}, nmin[2] = {0, 0}, xiB[2] = {0, 0};
      long nB[2] = {0, 0}, nS[2] = {0, 0}, nM[2] = {0, 0}, nC[2] = {0, 0}, nW[2] = {0, 0};
      long nA[2] = {0, 0}, nI[2] = {0, 0};
   } classes;
   // --endgame-diag (diag = true): per iteration, where the dual residual sits (one log
   // line: its max on the cells and off them, the cell and the entry that hold it, the
   // cells above √t, the largest ξ); the last iteration's per-cell record goes to
   // write_diag().  Read-only: the run is unchanged.
   bool diag = false;
   // --freeze-xi X (> 0, opt-in): a cell whose product multiplier ξ has been ≥ X for
   // freeze_persist consecutive iterations, while t > t_min, keeps the t of that
   // iteration in its product row, r(1 − δ) ≤ t_e, while t goes on falling elsewhere.
   // At points that are not strongly stationary ξ grows like 1/√t under any Scholtes
   // schedule (Ralph–Wright 2004; Kanzow–Schwartz 2015); freezing keeps it bounded there.
   double freeze_xi = 0.0;
   int freeze_persist = 3;
   long frozen = 0;                    // cells frozen so far (all ranks)
   double frozen_tmin = 0.0, frozen_tmax = 0.0;   // ... the range of their t_e
   // --follow-comp (on by default, as in tv_mpcc.hpp): a cell whose product row stays violated (r(1 − δ) > t_e)
   // while ξ rises and the product does not fall, at a fixed t, for follow_persist
   // consecutive iterations, is "locked" between
   // the branches r = 0 and δ = 1.  Its t_e is then raised to r(1 − δ) − s, so its row
   // holds at the current point (s ≤ 0: the row's slack), and from then on only
   // lowered, to r(1 − δ) − s each iteration: the barrier on s pulls the product
   // down and t_e follows it.  The cell is released once t_e ≤ t.
   // The test is armed only in a stall: μ and t unchanged for follow_stall iterations
   // in a row (an ordinary end phase at t = t_min is shorter; the failing runs sit at
   // a fixed (μ, t) for hundreds of iterations).
   bool follow = true;
   int follow_persist = 3, follow_stall = 20;
   // --rho-step (opt-in): curvilinear trial points that move the scaled dual, (Pα)∘δ
   // and Q(α)q, along the Newton step instead of δ and q.  With w the weight a q (copy
   // or original) or δ is multiplied by and w⁺ = w + a·dw its trial value,
   // q ← q + a(w/w⁺)dq and δ ← δ + a(w/w⁺)dδ; δ keeps the straight step where this
   // would take it within 1% of the bounds 0, 1.  On a flat block the direction
   // "α up, δ down with (Pα)δ fixed" is then followed exactly.
   bool rho_step = false;
   long rho_straight = 0;              // δ entries of trial points that kept the straight step
   long follow_events = 0;             // raises so far (all ranks)
   long follow_now = 0;                // cells with t_e > t at the last iteration
   double follow_tmax = 0.0;           // ... their largest t_e

   // ---- sizes and offsets -------------------------------------------------
   int N, nc, m_u, m_q, S, n_blocks;   // per image: side, cells per side, #nodes, #cells
   int n, mcon, n_eq, n_ineq;
   int n_link = 0, n_orig = 0;
   int ou, oqx, oqy, oR, oD, oTh, oa;  // α of block b is oa + b; u of image s, node i: ou + s·m_u + i
   int rh1, rh2x, rh2y, rh3x, rh3y, rlink, rhr, rhd, rha, rcomp;

   std::vector<double> x_start;
   std::vector<double> solution;       // finalize: x on rank 0, global
   double xi_max = 0.0, objective = 0.0;
   std::vector<double> mu_trace;       // (iter, μ, t, max α, max r(1−δ)) per iteration
   int warm_iters = 0;                 // the warm start's PDHG/CP iterations (max over images)
   double warm_residual = 0.0;         // ... its final max |u⁺ − u|/τ, and the residual of h1
   double warm_kkt = 0.0;              //     at its point (pdhg_kkt; max over images)

   // loss_scale: 1 (sum) or 1/S (mean); alpha0: the warm start's weight;
   // warm_tol / warm_max_iter: pdhg.hpp's stopping rule (when A ≠ I or K_ξ ≠ K)
   // alpha_init (optional, one value per block): start from this weight map instead
   TvLearn(const std::vector<Instance>& data, const LearnPartition& part, const Blur* blur,
           double alpha0, double loss_scale = 1.0, double warm_tol = 1e-6,
           int warm_max_iter = 20000, const std::vector<double>& alpha_init = {},
           const StartFields* fields = nullptr, double cartesian_gamma = 0.0)
       : warm_tol(warm_tol), warm_max_iter(warm_max_iter), part_(part), blur_(blur),
         sum_(part.n_tiles), scale_(loss_scale), cart_gamma_(cartesian_gamma),
         cart_(cartesian_gamma > 0.0) {
      if (data.empty()) throw std::runtime_error("no images");
      N = part.N;
      nc = N - 1;
      m_u = N * N;
      m_q = nc * nc;
      S = (int)data.size();
      n_blocks = part.n_blocks;
      if (S != part.S) throw std::runtime_error("TvLearn: partition and data disagree on S");
      for (const Instance& in : data)
         if (in.N != N) throw std::runtime_error("TvLearn: every image must be N x N");
      dtv_ = data[0].dtv();
      R_ = blur ? 2 * blur->w : 0;
      if (R_ > 0 && part.min_band() < 2 * R_)
         throw std::runtime_error("deblur: tiles must be at least 4w cells wide (fewer tiles, "
                                  "or a smaller --blur-radius)");
      for (int b = 0; b < n_blocks; ++b) atiles_.push_back(part.alpha_tiles(b));
      layout_unsplit();
      if (!alpha_init.empty() && (int)alpha_init.size() != n_blocks)
         throw std::runtime_error("--alpha-init: " + std::to_string(alpha_init.size()) +
                                  " values for " + std::to_string(n_blocks) + " blocks");
      build_local(warm_start(data, alpha0, alpha_init, fields), data);
   }

   // max over cells of r·(1 − δ) (x global)
   double max_complementarity(const double* x) const {
      double c = 0.0;
      for (int e = 0; e < S * m_q; ++e) c = std::max(c, x[oR + e] * (1.0 - x[oD + e]));
      return c;
   }

   // =========================================================================
   //  NLP interface
   // =========================================================================
   int num_vars() const override { return (int)gvar_.size(); }
   int num_rows() const override { return (int)grow_.size(); }
   int n_tiles() const override { return n_tiles_; }
   const std::vector<int>& var_tile() const override { return vtile_; }
   const std::vector<int>& row_tile() const override { return rtile_; }

   void bounds(double* xl, double* xu, double* gl, double* gu) const override {
      const int nl = num_vars(), ml = num_rows();
      for (int i = 0; i < nl; ++i) { xl[i] = -2e19; xu[i] = 2e19; }
      if (!cart_)
         for (const Cell& c : cells_) xu[c.d] = 1.0;
      for (int l : alpha_orig_) xu[l] = kAlphaMax;
      for (int i = 0; i < ml; ++i) {
         const int r = grow_[i];
         if (r < n_eq) { gl[i] = 0.0; gu[i] = 0.0; }
         else if (r < rcomp) { gl[i] = 0.0; gu[i] = 2e19; }
         else { gl[i] = -2e19; gu[i] = 0.0; }
         if (!bounds_as_rows && r >= rhr && r < rcomp) { gl[i] = -2e19; gu[i] = 2e19; }
      }
      if (!bounds_as_rows) {
         if (!cart_)
            for (const Cell& c : cells_) xl[c.r] = xl[c.d] = 0.0;
         for (int l : alpha_orig_) xl[l] = 0.0;
      }
   }

   void start(double* x) const override { std::copy(x_start.begin(), x_start.end(), x); }

   bool curvilinear() const override { return rho_step && !cart_; }
   void curve(const double* p, const double* d, double a, double* pt) override {
      for (const auto& [v, w] : rho_q_) {
         const double w1 = p[w] + a * d[w];
         if (w1 > 0.0) pt[v] = p[v] + a * (p[w] / w1) * d[v];
      }
      for (const auto& [v, w] : rho_d_) {
         const double w1 = p[w] + a * d[w];
         if (w1 <= 0.0) continue;
         const double dl = p[v] + a * (p[w] / w1) * d[v];
         if (dl >= 0.01 * p[v] && 1.0 - dl >= 0.01 * (1.0 - p[v])) pt[v] = dl;
         else ++rho_straight;
      }
   }

   double f(const double* x) override {
      for (int l = 0; l < num_vars(); ++l) {
         const double term = objective_term(l, x[l]);
         if (term != 0.0) sum_.add(vtile_[l], term);
      }
      return sum_.total();
   }

   void grad_f(const double* x, double* g) override {
      for (int l = 0; l < num_vars(); ++l) {
         const int v = gvar_[l];
         g[l] = v < ou + S * m_u                ? scaled(x[l] - ref_[l])
              : v >= oa && v < oa + n_blocks    ? kRegAlpha * x[l]
              : !cart_ && v >= oTh && v < oTh + S * m_q ? scaled(eps_theta * (x[l] - ref_[l]))
                                                : 0.0;
      }
   }

   void g(const double* x, double* g) override {
      std::vector<double> dsum(nodes_.size(), 0.0), divg(grp_alpha_.size(), 0.0);
      std::vector<double> kxu(cells_.size(), 0.0), kyu(cells_.size(), 0.0);
      for (const auto& e : data_) dsum[e.r] += e.v * x[e.c];
      for (const auto& e : div_x_) divg[e.g] += e.v * x[e.c];
      for (const auto& e : div_y_) divg[e.g] += e.v * x[e.c];
      for (const auto& e : grad_x_) kxu[e.r] += e.v * x[e.c];
      for (const auto& e : grad_y_) kyu[e.r] += e.v * x[e.c];
      for (size_t a = 0; a < nodes_.size(); ++a) {
         const Node& q = nodes_[a];
         double v = dsum[a] - fnode_[a];
         for (int gi = q.g0; gi < q.g1; ++gi) v += x[grp_alpha_[gi]] * divg[gi];
         g[q.h1] = v;
      }
      for (size_t b = 0; cart_ && b < cells_.size(); ++b) {   // c = q − h(K_ξu)
         const Cell& q = cells_[b];
         double gx, gy, phi, d1, d2;
         cart_cell(x, b, gx, gy);
         if (cart_dual) {
            const double m = std::max(cart_gamma_, std::hypot(gx, gy));
            g[q.h2x] = (m * x[q.qx] - gx) / cart_gamma_;
            g[q.h2y] = (m * x[q.qy] - gy) / cart_gamma_;
            continue;
         }
         cart_profile(std::hypot(gx, gy), phi, d1, d2);
         g[q.h2x] = x[q.qx] - phi * gx;
         g[q.h2y] = x[q.qy] - phi * gy;
      }
      for (size_t b = 0; !cart_ && b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double c = std::cos(x[q.th]), s = std::sin(x[q.th]);
         const double r = x[q.r], d = x[q.d];
         g[q.h2x] = kxu[b] - r * c;
         g[q.h2y] = kyu[b] - r * s;
         g[q.h3x] = x[q.qx] - d * c;
         g[q.h3y] = x[q.qy] - d * s;
         g[q.hr] = r;
         g[q.hd] = d;
         g[q.comp] = r * (1.0 - d) - (t_cell_.empty() ? t : std::max(t, t_cell_[b]));
      }
      for (const auto& h : ha_) g[h.first] = x[h.second];
      for (const Link& l : links_) g[l.row] = x[l.copy] - x[l.orig];
   }

   void jac_structure(std::vector<int>& row, std::vector<int>& col) const override {
      row = jac_row_;
      col = jac_col_;
   }

   void jac_values(const double* x, double* values) override {
      std::vector<double> divg(grp_alpha_.size(), 0.0);
      for (const auto& e : div_x_) divg[e.g] += e.v * x[e.c];
      for (const auto& e : div_y_) divg[e.g] += e.v * x[e.c];
      int k = 0;
      for (const auto& e : data_) values[k++] = e.v;                                  // h1/u
      for (const auto& e : div_x_) values[k++] = x[grp_alpha_[e.g]] * e.v;            // h1/qx
      for (const auto& e : div_y_) values[k++] = x[grp_alpha_[e.g]] * e.v;            // h1/qy
      for (size_t gi = 0; gi < grp_alpha_.size(); ++gi) values[k++] = divg[gi];       // h1/α
      if (cart_) {
         cart_jac(x, values, k);
         for (size_t h = 0; h < ha_.size(); ++h) values[k++] = 1.0;                   // ha/α
         for (size_t l = 0; l < links_.size(); ++l) values[k++] = 1.0;                // link/copy
         for (size_t l = 0; l < links_.size(); ++l) values[k++] = -1.0;               // link/orig
         return;
      }
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
      for (size_t h = 0; h < ha_.size(); ++h) values[k++] = 1.0;                      // ha/α
      for (const Cell& q : cells_) values[k++] = 1.0 - x[q.d];                        // comp/r
      for (const Cell& q : cells_) values[k++] = -x[q.r];                             // comp/δ
      for (size_t l = 0; l < links_.size(); ++l) values[k++] = 1.0;                   // link/copy
      for (size_t l = 0; l < links_.size(); ++l) values[k++] = -1.0;                  // link/orig
   }

   void hess_structure(std::vector<int>& row, std::vector<int>& col) const override {
      row = hess_row_;
      col = hess_col_;
   }

   void hess_values(const double* x, double obj_factor, const double* lam,
                    double* values) override {
      int k = 0;
      // ℓ·σ; written as tv_mpcc.hpp's expressions below, so that the compiler
      // contracts them (FMA) the same way
      const double fu = scale_ == 1.0 ? obj_factor : obj_factor * scale_;
      for (size_t i = 0; i < u_orig_.size(); ++i) values[k++] = fu;                  // (u,u)
      if (cart_) {
         const int k0 = k;
         cart_hess(x, lam, values, k);
         for (const auto& e : div_x_) values[k++] = e.v * lam[nodes_[e.r].h1];      // (α,qx)
         for (const auto& e : div_y_) values[k++] = e.v * lam[nodes_[e.r].h1];      // (α,qy)
         if (cart_gn) std::fill(values + k0, values + k, 0.0);
         for (size_t k2 = 0; k2 < tile_alpha_.size(); ++k2) values[k++] = 0.0;
         for (size_t k2 = 0; k2 < alpha_orig_.size(); ++k2) values[k++] = obj_factor * kRegAlpha;
         return;
      }
      for (const Cell& q : cells_)                                                  // (θ,r)
         values[k++] = lam[q.h2x] * std::sin(x[q.th]) - lam[q.h2y] * std::cos(x[q.th]);
      for (const Cell& q : cells_)                                                  // (θ,δ)
         values[k++] = lam[q.h3x] * std::sin(x[q.th]) - lam[q.h3y] * std::cos(x[q.th]);
      for (const Cell& q : cells_) {                                                // (θ,θ)
         const double c = std::cos(x[q.th]), s = std::sin(x[q.th]);
         values[k++] = x[q.r] * (lam[q.h2x] * c + lam[q.h2y] * s)
                     + x[q.d] * (lam[q.h3x] * c + lam[q.h3y] * s)
                     + fu * eps_theta;
      }
      for (const Cell& q : cells_) values[k++] = -lam[q.comp];                      // (δ,r)
      for (const auto& e : div_x_) values[k++] = e.v * lam[nodes_[e.r].h1];         // (α,qx)
      for (const auto& e : div_y_) values[k++] = e.v * lam[nodes_[e.r].h1];         // (α,qy)
      // (α,α) of every α a tile reads: zero (α enters h1 linearly); the entries
      // keep a diagonal on every α copy.  Then the ridge on the originals.
      for (size_t k2 = 0; k2 < tile_alpha_.size(); ++k2) values[k++] = 0.0;
      for (size_t k2 = 0; k2 < alpha_orig_.size(); ++k2) values[k++] = obj_factor * kRegAlpha;
   }

   // The level gate and the μ-coupled continuation (tv_mpcc.hpp's, without the
   // penalty, the clean-up and the per-cell relaxation).  Collective.
   bool iteration(const Iterate& it) override {
      if (cart_) {   // the Cartesian problem: no t, no gate; the IPM's own test
         double am = -2e19;
         for (int l : alpha_orig_) am = std::max(am, it.x[l]);
         am = dd::Comm::max(am);
         mu_trace.insert(mu_trace.end(), {(double)it.iter, it.mu, cart_gamma_, am, 0.0});
         return true;
      }
      if (classify && it.zl && it.zu && t <= t_min * (1.0 + 1e-9)) classify_point(it);
      if (it.iter > 0 && t <= t_min * (1.0 + 1e-9)) {
         const double gate = std::max(gate_floor, std::sqrt(t)), gate_du = gate_du_scale * gate;
         if (it.inf_pr <= gate && (!gate_dual || it.inf_du <= gate_du) && it.mu <= gate) {
            gate_fired = true;
            if (print && gate_dual)
               std::printf("  [level gate] t=%.2e  inf_pr=%.1e  inf_du=%.1e  mu=%.1e"
                           "  <= sqrt(t) = %.1e (inf_du: %.1e): stopping\n",
                           t, it.inf_pr, it.inf_du, it.mu, gate, gate_du);
            else if (print)
               std::printf("  [level gate] t=%.2e  inf_pr=%.1e  mu=%.1e  <= sqrt(t) = %.1e"
                           " (inf_du=%.1e, not tested): stopping\n",
                           t, it.inf_pr, it.mu, gate, it.inf_du);
            return false;
         }
      }
      if (diag && it.dual && it.zl && it.zu) diag_point(it);
      if (freeze_xi > 0.0 && it.zu && t > t_min * (1.0 + 1e-9)) freeze_cells(it);
      if (follow && it.zu) follow_cells(it);
      double cm[2] = {0.0, -2e19};   // max r(1 − δ), max α (global)
      for (const Cell& q : cells_) cm[0] = std::max(cm[0], it.x[q.r] * (1.0 - it.x[q.d]));
      for (int l : alpha_orig_) cm[1] = std::max(cm[1], it.x[l]);
      dd::Comm::max(cm, 2);
      const double comp = cm[0];
      double t_new = std::max(t_min, t_mu_scale * it.mu);
      if (t_new < t * t_rate) t_new = std::max(t_min, t * t_rate);
      if (t_comp_ratio > 0.0) t_new = std::max(t_new, comp / t_comp_ratio);
      if (t_on_solved && !it.barrier_solved) t_new = t;
      if (t_new < t) {
         t = t_new;
         eps_theta = c_theta * t_new;
      }
      mu_trace.insert(mu_trace.end(), {(double)it.iter, it.mu, t, cm[1], comp});
      return true;
   }

   bool final_form() const override {
      return cart_ || t <= t_min * (1.0 + 1e-9);
   }

   // The lower-level fields (u, q per image) and the block weights of a global x
   // (e.g. solution), the start of another problem on the same images
   StartFields fields_of(const std::vector<double>& x, std::vector<double>& alpha) const {
      StartFields F;
      F.u.resize(S);
      F.qx.resize(S);
      F.qy.resize(S);
      for (int s = 0; s < S; ++s) {
         F.u[s].assign(x.begin() + ou + (long)s * m_u, x.begin() + ou + (long)(s + 1) * m_u);
         F.qx[s].assign(x.begin() + oqx + (long)s * m_q, x.begin() + oqx + (long)(s + 1) * m_q);
         F.qy[s].assign(x.begin() + oqy + (long)s * m_q, x.begin() + oqy + (long)(s + 1) * m_q);
      }
      alpha.assign(x.begin() + oa, x.begin() + oa + n_blocks);
      return F;
   }

   void complementarity_bounds(char* lo, char* up) const override {
      if (cart_) return;
      for (const Cell& q : cells_) {
         lo[bounds_as_rows ? slack(q.hr) : q.r] = 1;
         up[q.d] = 1;
      }
   }

   std::string log_extra() const override {
      char b[48];
      std::snprintf(b, sizeof b, cart_ ? "  gamma=%.2e" : "  t=%.2e", cart_ ? cart_gamma_ : t);
      return b;
   }

   void finalize(int, const double* x, const double* lam, double obj) override {
      std::vector<int> counts;
      const std::vector<double> all = dd::Comm::gather_values(x, num_vars(), counts);
      solution.clear();
      if (dd::Comm::root()) {
         const int nr = dd::Comm::size();
         std::vector<long> at(nr, 0);
         for (int r = 1; r < nr; ++r) at[r] = at[r - 1] + counts[r - 1];
         solution.assign(n, 0.0);
         for_each_var([&](int v, int k, int) {
            if (k >= 0) {
               solution[v] = all[at[dd::Comm::owner_of(k, n_tiles_)]++];
            } else {
               solution[v] = all[at[0]];
               for (int r = 0; r < nr; ++r) ++at[r];
            }
         });
      }
      double xi = 0.0;
      if (!cart_)
         for (const Cell& q : cells_) xi = std::max(xi, std::abs(lam[q.comp]));
      xi_max = dd::Comm::max(xi);
      objective = obj;
   }

   // α per cell (the map the lower level sees), from a global x
   std::vector<double> alpha_cells(const double* x) const {
      std::vector<double> a(m_q);
      for (int e = 0; e < m_q; ++e) a[e] = x[oa + part_.block(e)];
      return a;
   }

   void classify_point(const Iterate& it) {
      std::vector<double> gam(cells_.size()), nu(cells_.size());
      double scale = 0.0;
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double r = it.x[q.r], w = 1.0 - it.x[q.d];
         const double a = bounds_as_rows ? it.zl[slack(q.hr)] : it.zl[q.r];
         const double bm = it.zu[q.d], xi = it.zu[slack(q.comp)];
         gam[b] = a - xi * w;
         nu[b] = bm - xi * r;
         scale = std::max({scale, std::abs(gam[b]), std::abs(nu[b])});
      }
      scale = dd::Comm::max(scale);
      Classes C;
      C.iter = it.iter;
      C.tau = 1e-3 * scale;
      for (int k = 0; k < 2; ++k) {
         const double eps = (k == 0 ? 3.0 : 1.0) * std::sqrt(t);
         C.eps[k] = eps;
         long cnt[7] = {0, 0, 0, 0, 0, 0, 0};   // B S M C W A I
         double mm[3] = {0.0, 0.0, 0.0};        // −min γ, −min ν, max ξ on B (max-reduced)
         for (size_t b = 0; b < cells_.size(); ++b) {
            const Cell& q = cells_[b];
            const double r = it.x[q.r], w = 1.0 - it.x[q.d], xi = it.zu[slack(q.comp)];
            const double g = gam[b], v = nu[b];
            const bool rs = r <= eps, ws = w <= eps;
            if (rs && ws) {
               ++cnt[0];
               mm[0] = std::max(mm[0], -g);
               mm[1] = std::max(mm[1], -v);
               mm[2] = std::max(mm[2], xi);
               if (g >= -C.tau && v >= -C.tau) ++cnt[1];
               else if (std::min(std::abs(g), std::abs(v)) <= C.tau) ++cnt[2];
               else if (g < 0 && v < 0) ++cnt[3];
               else ++cnt[4];
            } else if (rs) {
               ++cnt[5];
            } else if (ws) {
               ++cnt[6];
            }
         }
         for (long& c : cnt) c = dd::Comm::sum(c);
         dd::Comm::max(mm, 3);
         C.nB[k] = cnt[0]; C.nS[k] = cnt[1]; C.nM[k] = cnt[2]; C.nC[k] = cnt[3]; C.nW[k] = cnt[4];
         C.nA[k] = cnt[5]; C.nI[k] = cnt[6];
         C.gmin[k] = -mm[0]; C.nmin[k] = -mm[1]; C.xiB[k] = mm[2];
      }
      classes = C;
   }

   // --endgame-diag: where the dual residual (∇L − z_l + z_u per entry of [x | s]) sits.
   // A cell's entries are r, δ, θ, the q copies its h3 rows read, and the slacks of its
   // hr, hd and product rows; "off the cells" is everything else (u, α, the other
   // copies).  Prints one line on rank 0 and keeps the per-cell record of this
   // iteration for write_diag().  Collective.
   static constexpr int kDiagRec = 15;   // s, e, r, w, ξ, a, b, γ, ν, res, entry, t_e, streak, iter, t
   void diag_point(const Iterate& it) {
      const int nv = num_vars(), np = nv + (num_rows() - first_slack_row_);
      std::vector<char> on_cell(np, 0);
      const double gate = std::sqrt(t);
      diag_rec_.assign(kDiagRec * cells_.size(), 0.0);
      double mx[3] = {0.0, 0.0, 0.0};      // max residual on the cells, off them, max ξ
      double where[2] = {-1.0, -1.0};      // their codes (s·m_q + e)·8 + entry; ξ: s·m_q + e
      long above = 0;
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const int ent[8] = {q.r, q.d, q.th, q.qx, q.qy, slack(q.hr), slack(q.hd), slack(q.comp)};
         double res = 0.0;
         int kmax = 0;
         for (int k = 0; k < 8; ++k) {
            on_cell[ent[k]] = 1;
            const double v = std::abs(it.dual[ent[k]]);
            if (v > res) { res = v; kmax = k; }
         }
         const double r = it.x[q.r], w = 1.0 - it.x[q.d], xi = it.zu[slack(q.comp)];
         const double a = bounds_as_rows ? it.zl[slack(q.hr)] : it.zl[q.r], bm = it.zu[q.d];
         const double te = t_cell_.empty() ? t : std::max(t, t_cell_[b]);
         const int streak = xi_streak_.empty() ? 0 : xi_streak_[b];
         const double rec[kDiagRec] = {(double)q.s, (double)q.e, r, w, xi, a, bm, a - xi * w,
                                       bm - xi * r, res, (double)kmax, te, (double)streak,
                                       (double)it.iter, t};
         std::copy(rec, rec + kDiagRec, &diag_rec_[kDiagRec * b]);
         if (res > gate) ++above;
         if (res > mx[0]) { mx[0] = res; where[0] = ((double)q.s * m_q + q.e) * 8 + kmax; }
         if (xi > mx[2]) { mx[2] = xi; where[1] = (double)q.s * m_q + q.e; }
      }
      for (int i = 0; i < np; ++i)
         if (!on_cell[i]) mx[1] = std::max(mx[1], std::abs(it.dual[i]));
      const double loc[3] = {mx[0], mx[1], mx[2]};
      dd::Comm::max(mx, 3);
      // the location of each maximum: from the rank that holds it (ties: the largest code)
      double w2[2] = {loc[0] == mx[0] ? where[0] : -1.0, loc[2] == mx[2] ? where[1] : -1.0};
      dd::Comm::max(w2, 2);
      above = dd::Comm::sum(above);
      if (!print) return;
      static const char* ename[8] = {"r", "delta", "theta", "qx", "qy", "hr slack", "hd slack",
                                     "comp slack"};
      const long c0 = (long)w2[0], c1 = (long)w2[1];
      const long e0 = (c0 / 8) % m_q, e1 = c1 % m_q;
      std::printf("  [du] it %d t=%.2e: cells %.1e at (%ld; %ld,%ld) %s, off cells %.1e;"
                  " %ld cells > sqrt(t); max xi %.1e at (%ld; %ld,%ld)",
                  it.iter, t, mx[0], c0 / 8 / m_q, e0 / nc, e0 % nc, ename[c0 % 8], mx[1], above,
                  mx[2], c1 / m_q, e1 / nc, e1 % nc);
      if (freeze_xi > 0.0) std::printf("; frozen %ld", frozen);
      std::printf("\n");
   }

   // --endgame-diag FILE: the per-cell record of the last diag_point(), gathered on rank
   // 0, as CSV (one line per cell, global order of the ranks' cells).  Collective.
   void write_diag(const std::string& file) const {
      std::vector<int> counts;
      const std::vector<double> all =
          dd::Comm::gather_values(diag_rec_.data(), (int)diag_rec_.size(), counts);
      if (!dd::Comm::root()) return;
      FILE* f = std::fopen(file.c_str(), "w");
      if (!f) { std::fprintf(stderr, "cannot write %s\n", file.c_str()); return; }
      std::fprintf(f, "image,cell,i,j,r,w,xi,a,b,gamma,nu,dual_res,entry,t_cell,xi_streak,iter,t\n");
      static const char* ename[8] = {"r", "delta", "theta", "qx", "qy", "hr_slack", "hd_slack",
                                     "comp_slack"};
      for (size_t k = 0; k + kDiagRec <= all.size(); k += kDiagRec) {
         const double* v = &all[k];
         const int e = (int)v[1];
         std::fprintf(f, "%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%s,%.6e,%d,%d,%.6e\n",
                      (int)v[0], e, e / nc, e % nc, v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
                      ename[(int)v[10]], v[11], (int)v[12], (int)v[13], v[14]);
      }
      std::fclose(f);
   }

   // --freeze-xi: see freeze_xi above.  A frozen cell keeps t_e for the rest of the
   // solve; at the moment it is frozen t_e = t, so the problem does not change in that
   // iteration.  Collective.
   void freeze_cells(const Iterate& it) {
      if (t_cell_.empty()) {
         t_cell_.assign(cells_.size(), 0.0);
         xi_streak_.assign(cells_.size(), 0);
      }
      long fresh = 0;
      for (size_t b = 0; b < cells_.size(); ++b) {
         if (t_cell_[b] > 0.0) continue;
         const double xi = it.zu[slack(cells_[b].comp)];
         xi_streak_[b] = xi >= freeze_xi ? xi_streak_[b] + 1 : 0;
         if (xi_streak_[b] >= freeze_persist) {
            t_cell_[b] = t;
            ++fresh;
         }
      }
      fresh = dd::Comm::sum(fresh);
      if (fresh == 0) return;
      frozen += fresh;
      double lo = 2e19, hi = 0.0;
      for (double v : t_cell_)
         if (v > 0.0) { lo = std::min(lo, v); hi = std::max(hi, v); }
      frozen_tmin = dd::Comm::min(lo);
      frozen_tmax = dd::Comm::max(hi);
      if (print)
         std::printf("  [freeze] it %d: %ld cells with xi >= %.1e for %d iterations keep"
                     " t = %.2e (%ld frozen)\n",
                     it.iter, fresh, freeze_xi, freeze_persist, t, frozen);
   }

   // --follow-comp: see `follow` above.  Collective.
   void follow_cells(const Iterate& it) {
      if (t_cell_.empty()) t_cell_.assign(cells_.size(), 0.0);
      if (lock_streak_.empty()) {
         lock_streak_.assign(cells_.size(), 0);
         xi_prev_.assign(cells_.size(), 0.0);
         rw_prev_.assign(cells_.size(), 0.0);
      }
      stall_iters_ = it.mu == mu_prev_ && t == t_prev_ ? stall_iters_ + 1 : 0;
      mu_prev_ = it.mu;
      const bool armed = stall_iters_ >= follow_stall;
      long fresh = 0, now = 0;
      double tmax = 0.0;
      for (size_t b = 0; b < cells_.size(); ++b) {
         const Cell& q = cells_[b];
         const double rw = it.x[q.r] * (1.0 - it.x[q.d]), s = it.x[slack(q.comp)];
         const double xi = it.zu[slack(q.comp)];
         const double te = std::max(t, t_cell_[b]);
         if (t_cell_[b] > 0.0) {   // following: lower t_e to the product (never raise it here)
            t_cell_[b] = std::min(t_cell_[b], rw - s);
            if (t_cell_[b] <= t) t_cell_[b] = 0.0;   // released
         }
         // locked: violated, ξ rising, the product not falling, and t not moving.  While
         // t falls every active row is briefly violated with ξ rising (the catch-up of
         // the continuation); only a row that stays violated at a fixed t is locked.
         const bool locked =
             armed && rw > te && xi > xi_prev_[b] && rw >= rw_prev_[b] && t == t_prev_;
         lock_streak_[b] = locked ? lock_streak_[b] + 1 : 0;
         xi_prev_[b] = xi;
         rw_prev_[b] = rw;
         if (lock_streak_[b] >= follow_persist) {
            t_cell_[b] = std::max(t_cell_[b], rw - s);
            lock_streak_[b] = 0;
            ++fresh;
         }
         if (t_cell_[b] > t) { ++now; tmax = std::max(tmax, t_cell_[b]); }
      }
      fresh = dd::Comm::sum(fresh);
      follow_events += fresh;
      follow_now = dd::Comm::sum(now);
      follow_tmax = dd::Comm::max(tmax);
      t_prev_ = t;
      if (print && fresh > 0)
         std::printf("  [follow] it %d: %ld locked cells raised to their product; %ld followed,"
                     " max t_e %.2e (t = %.2e, stalled %d iterations)\n",
                     it.iter, fresh, follow_now, follow_tmax, t, stall_iters_);
   }

private:
   std::vector<int> lock_streak_;      // --follow-comp: consecutive locked iterations
   std::vector<double> xi_prev_;       // ... and the last iteration's ξ,
   std::vector<double> rw_prev_;       //     r(1 − δ)
   double t_prev_ = -1.0;              //     and t
   double mu_prev_ = -1.0;             // ... the last iteration's μ, and how many
   int stall_iters_ = 0;               //     iterations (μ, t) have not moved
   std::vector<double> diag_rec_;      // --endgame-diag: per own cell, the last record
   std::vector<double> t_cell_;        // --freeze-xi: per own cell, its frozen t (0: not frozen)
   std::vector<int> xi_streak_;        // ... consecutive iterations with ξ ≥ freeze_xi
   double warm_tol;
   int warm_max_iter;
   double cart_gamma_ = 0.0;   // the Cartesian problem's γ (0: the lifted MPCC)
   bool cart_ = false;
   struct CartCell { int var[3]; double bx[3], by[3]; };   // K_ξu at a cell = Σ (bx, by)·u
   std::vector<CartCell> cst_;                            // per own cell (Cartesian)

   // h(g) = φ(s)·g, s = |g|, the profile of the clipping (see the top of the file);
   // φ, φ', φ'' at s (inactive: φ = c, φ' = φ'' = 0)
   void cart_profile(double s, double& phi, double& d1, double& d2) const {
      const double c = 1.0 / cart_gamma_, cs = c * s;
      if (cs <= (cart_c1 ? 1.0 : 1.0 - 0.5 / c)) { phi = c; d1 = 0.0; d2 = 0.0; return; }
      double psi, p1, p2;   // ψ = φ·s and its derivatives
      if (cart_c1 || cs >= 1.0 + 0.5 / c) { psi = 1.0; p1 = 0.0; p2 = 0.0; }
      else {
         const double b = 1.0 - cs + 0.5 / c;
         psi = 1.0 - 0.5 * c * b * b;
         p1 = c * c * b;
         p2 = -c * c * c;
      }
      phi = psi / s;
      d1 = (p1 * s - psi) / (s * s);
      d2 = (p2 * s * s - 2.0 * p1 * s + 2.0 * psi) / (s * s * s);
   }
   void cart_cell(const double* x, size_t b, double& gx, double& gy) const {
      const CartCell& c = cst_[b];
      gx = 0.0;
      gy = 0.0;
      for (int j = 0; j < 3; ++j) {
         gx += c.bx[j] * x[c.var[j]];
         gy += c.by[j] * x[c.var[j]];
      }
   }
   // c = q − φ(s)g: ∂c/∂q = I, ∂c/∂u_j = −(∂h/∂g)(bx_j, by_j), ∂h/∂g = φI + (φ'/s)ggᵀ;
   // in the pattern's order
   void cart_jac(const double* x, double* values, int& k) const {
      if (cart_dual) {   // ∂c/∂u_j = (q χ ĝ·B_j − B_j)/γ, ∂c/∂q = (m/γ) I
         const double ig = 1.0 / cart_gamma_;
         for (size_t b = 0; b < cells_.size(); ++b) {
            const CartCell& c = cst_[b];
            double gx, gy;
            cart_cell(x, b, gx, gy);
            const double s = std::hypot(gx, gy), m = std::max(cart_gamma_, s);
            const double hx = s >= cart_gamma_ ? gx / s : 0.0, hy = s >= cart_gamma_ ? gy / s : 0.0;
            double qx, qy;
            cart_qbar(x, b, qx, qy);
            for (int j = 0; j < 3; ++j) values[k++] = (qx * (hx * c.bx[j] + hy * c.by[j]) - c.bx[j]) * ig;
            values[k++] = m * ig;
            for (int j = 0; j < 3; ++j) values[k++] = (qy * (hx * c.bx[j] + hy * c.by[j]) - c.by[j]) * ig;
            values[k++] = m * ig;
         }
         return;
      }
      for (size_t b = 0; b < cells_.size(); ++b) {
         const CartCell& c = cst_[b];
         double gx, gy, phi, d1, d2;
         cart_cell(x, b, gx, gy);
         const double s = std::hypot(gx, gy);
         cart_profile(s, phi, d1, d2);
         const double e = s > 0.0 ? d1 / s : 0.0;
         double jxx = phi + e * gx * gx, jxy = e * gx * gy, jyx = jxy, jyy = phi + e * gy * gy;
         if (cart_proj && s > 0.0 && d1 != 0.0) {   // φI − q̄ĝᵀ/s + ψ'ĝĝᵀ, ψ' = φ's + φ
            double qx = x[cells_[b].qx], qy = x[cells_[b].qy];
            const double n = std::hypot(qx, qy);
            if (n > 1.0) { qx /= n; qy /= n; }
            const double hx = gx / s, hy = gy / s, p1 = d1 * s + phi;
            jxx = phi - qx * hx / s + p1 * hx * hx;
            jxy = -qx * hy / s + p1 * hx * hy;
            jyx = -qy * hx / s + p1 * hx * hy;
            jyy = phi - qy * hy / s + p1 * hy * hy;
         }
         for (int j = 0; j < 3; ++j) values[k++] = -(jxx * c.bx[j] + jxy * c.by[j]);
         values[k++] = 1.0;
         for (int j = 0; j < 3; ++j) values[k++] = -(jyx * c.bx[j] + jyy * c.by[j]);
         values[k++] = 1.0;
      }
   }
   // λx·c_x + λy·c_y = λ·q − λ·h(g): (u,u) = −Bᵀ M B with M = ∇²_g(λ·h),
   //   M_bc = (φ'/s)(λ_b g_c + λ_c g_b) + (λ·g)[φ'' g_b g_c/s² + φ'(δ_bc/s − g_b g_c/s³)];
   // (q,u) = 0 (c is linear in q; the entries keep the pattern)
   void cart_hess(const double* x, const double* lam, double* values, int& k) const {
      if (cart_dual) {   // ∇²(λ·c): (u,u) = Bᵀ M B, M = χ (λ·q)/(γ s) (I − ĝĝᵀ);
                         // (q_b, u_j) = χ λ_b ĝ·B_j / γ
         const double ig = 1.0 / cart_gamma_;
         for (size_t b = 0; b < cells_.size(); ++b) {
            const CartCell& c = cst_[b];
            const Cell& q = cells_[b];
            double gx, gy;
            cart_cell(x, b, gx, gy);
            const double s = std::hypot(gx, gy);
            const double lx = lam[q.h2x], ly = lam[q.h2y];
            double mxx = 0.0, mxy = 0.0, myy = 0.0, hx = 0.0, hy = 0.0;
            if (s >= cart_gamma_) {
               hx = gx / s;
               hy = gy / s;
               double qbx, qby;
               cart_qbar(x, b, qbx, qby);
               const double kap = (lx * qbx + ly * qby) * ig / s;
               mxx = kap * (1.0 - hx * hx);
               mxy = -kap * hx * hy;
               myy = kap * (1.0 - hy * hy);
            }
            for (int j = 0; j < 3; ++j)
               for (int i = 0; i <= j; ++i)
                  values[k++] = mxx * c.bx[i] * c.bx[j] + mxy * (c.bx[i] * c.by[j] + c.by[i] * c.bx[j]) +
                                myy * c.by[i] * c.by[j];
            for (int j = 0; j < 3; ++j) values[k++] = lx * (hx * c.bx[j] + hy * c.by[j]) * ig;
            for (int j = 0; j < 3; ++j) values[k++] = ly * (hx * c.bx[j] + hy * c.by[j]) * ig;
         }
         return;
      }
      for (size_t b = 0; b < cells_.size(); ++b) {
         const CartCell& c = cst_[b];
         const Cell& q = cells_[b];
         double gx, gy, phi, d1, d2;
         cart_cell(x, b, gx, gy);
         const double s = std::hypot(gx, gy);
         cart_profile(s, phi, d1, d2);
         double mxx = 0.0, mxy = 0.0, myy = 0.0;
         if (s > 0.0 && (d1 != 0.0 || d2 != 0.0)) {
            const double lx = lam[q.h2x], ly = lam[q.h2y], lg = lx * gx + ly * gy;
            const double a = d1 / s, s2 = s * s, s3 = s2 * s;
            mxx = a * 2.0 * lx * gx + lg * (d2 * gx * gx / s2 + d1 * (1.0 / s - gx * gx / s3));
            mxy = a * (lx * gy + ly * gx) + lg * (d2 * gx * gy / s2 - d1 * gx * gy / s3);
            myy = a * 2.0 * ly * gy + lg * (d2 * gy * gy / s2 + d1 * (1.0 / s - gy * gy / s3));
         }
         for (int j = 0; j < 3; ++j)
            for (int i = 0; i <= j; ++i)
               values[k++] = -(mxx * c.bx[i] * c.bx[j] + mxy * (c.bx[i] * c.by[j] + c.by[i] * c.bx[j]) +
                               myy * c.by[i] * c.by[j]);
         for (int j = 0; j < 3; ++j) values[k++] = 0.0;
         for (int j = 0; j < 3; ++j) values[k++] = 0.0;
      }
   }
   static constexpr double kRegAlpha = 1e-4;
   static constexpr double kAlphaMax = 2e19;

   struct Tri { int r, c; double v; };          // (row slot, variable, value)
   struct DivTri { int r, c; double v; int g; };   // ... and the α group it is weighted by

   LearnPartition part_;
   const Blur* blur_;
   std::vector<std::vector<int>> atiles_;   // per block: the tiles that read it, ascending
   bool dtv_ = false;
   int R_ = 0;                              // radius of A's stencil (2w for a blur)
   std::vector<double> ref_;
   std::vector<double> fnode_;              // per own node: F
   int n_tiles_ = 0;
   std::array<std::vector<int>, 4> copy_off_;

   std::vector<int> gvar_, grow_, vtile_, rtile_;
   int first_slack_row_ = 0;
   struct Node { int s, i, h1, g0, g1; };   // groups [g0, g1) of grp_alpha_
   struct Cell {
      int s, e, r, d, th, qx, qy;
      int h2x, h2y, h3x, h3y, hr, hd, comp;
   };
   struct Link { int row, copy, orig; };
   std::vector<Node> nodes_;
   std::vector<Cell> cells_;
   std::vector<Link> links_;
   std::vector<int> grp_alpha_;             // per (node, group): the α variable
   std::vector<std::pair<int, int>> ha_;    // own rows ha: (row, the α it reads)
   std::vector<int> u_orig_, alpha_orig_;   // the original u / α variables held here
   std::vector<int> tile_alpha_;            // every α variable an own tile reads, by tile
   std::vector<std::pair<int, int>> rho_q_, rho_d_;   // --rho-step: (q or δ, its weight α)
   std::vector<Tri> data_;                  // A u in h1
   std::vector<DivTri> div_x_, div_y_;      // K_ξᵀ(α∘q) in h1: qx terms, then qy terms
   std::vector<Tri> grad_x_, grad_y_;       // K_ξ u in h2
   std::vector<int> jac_row_, jac_col_, hess_row_, hess_col_;
   dd::TileSum sum_;
   double scale_ = 1.0;                     // ℓ

   double scaled(double v) const { return scale_ == 1.0 ? v : v * scale_; }

   double objective_term(int l, double x) const {
      const int v = gvar_[l];
      if (v < ou + S * m_u) return scaled(0.5 * (x - ref_[l]) * (x - ref_[l]));
      if (v >= oa && v < oa + n_blocks) return 0.5 * kRegAlpha * x * x;
      if (!cart_ && eps_theta != 0.0 && v >= oTh && v < oTh + S * m_q)
         return scaled(0.5 * eps_theta * (x - ref_[l]) * (x - ref_[l]));
      return 0.0;
   }
   int slack(int i) const { return num_vars() + (i - first_slack_row_); }

   void layout_unsplit() {
      const int MU = S * m_u, MQ = S * m_q;
      const int ML = cart_ ? 0 : MQ;   // the lifting (r, δ, θ) and its rows h3, hr, hd, comp
      ou = 0;
      oqx = MU;
      oqy = MU + MQ;
      oR = MU + 2 * MQ;
      oD = oR + ML;
      oTh = oD + ML;
      oa = oTh + ML;
      n = oa + n_blocks;

      rh1 = 0;
      rh2x = MU;
      rh2y = MU + MQ;
      rh3x = MU + 2 * MQ;
      rh3y = rh3x + ML;
      rhr = rh3y + ML;
      rhd = rhr + ML;
      rha = rhd + ML;
      rcomp = rha + n_blocks;
      mcon = rcomp + ML;
      n_eq = rhr;
      n_ineq = mcon - n_eq;
   }

   // ---- warm start ----------------------------------------------------------
   // Each image's lower level at the starting weights (alpha0 everywhere, or the
   // map ainit), lifted to the MPCC variables (θ from the dual q: h3 exact, h2 off
   // by the solver's error).  Denoising with TV at a constant weight:
   // tv_mpcc.hpp's Chambolle–Pock, unchanged (the check against tv_dd, either
   // start); everything else: pdhg.hpp.  Given fields: their lifting, no solve.
   std::vector<double> warm_start(const std::vector<Instance>& data, double lam,
                                  const std::vector<double>& ainit, const StartFields* fields) {
      std::vector<double> x(n, 0.0);
      if (!ainit.empty()) lam = ainit[part_.block(0)];
      std::vector<double> wcell(m_q, lam);   // the start's weight per cell
      bool uniform = true;
      if (!ainit.empty())
         for (int e = 0; e < m_q; ++e) {
            wcell[e] = ainit[part_.block(e)];
            uniform = uniform && wcell[e] == lam;
         }
      for (int s = 0; s < S; ++s) {
         const Instance& in = data[s];
         std::vector<double> u, qx, qy;
         bool identity = blur_ ? blur_->w == 0 : true;   // A = I (a mask with no hole too)
         for (size_t i = 0; identity && i < in.diag.size(); ++i) identity = in.diag[i] == 1.0;
         if (fields) {
            u = fields->u[s];
            qx = fields->qx[s];
            qy = fields->qy[s];
         } else if (identity && !in.dtv() && uniform) {
            chambolle_pock(in.rhs, lam, u, qx, qy);
         } else {
            const std::vector<double>& w = wcell;
            PdhgResult P = pdhg(in, blur_, w, warm_tol, warm_max_iter);
            warm_iters = std::max(warm_iters, P.iters);
            warm_residual = std::max(warm_residual, P.residual);
            warm_kkt = std::max(warm_kkt, pdhg_kkt(in, blur_, w, P));
            u.swap(P.u);
            qx.swap(P.qx);
            qy.swap(P.qy);
         }
         std::vector<double> gx, gy;
         if (in.dtv()) apply_K(in, u, gx, gy);
         for (int i = 0; i < m_u; ++i) x[ou + s * m_u + i] = u[i];
         for (int a = 0; a < nc; ++a)
            for (int b = 0; b < nc; ++b) {
               const int e = a * nc + b, o = s * m_q + e;
               double kx, ky;
               if (in.dtv()) {
                  kx = gx[e];
                  ky = gy[e];
               } else {
                  kx = (0.0 + -1.0 * u[a * N + b]) + 1.0 * u[a * N + b + 1];
                  ky = (0.0 + -1.0 * u[a * N + b]) + 1.0 * u[(a + 1) * N + b];
               }
               x[oqx + o] = qx[e];
               x[oqy + o] = qy[e];
               if (cart_) continue;
               x[oR + o] = std::hypot(kx, ky);
               x[oD + o] = std::hypot(qx[e], qy[e]);
               x[oTh + o] = fields && fields->theta_from_u ? std::atan2(ky, kx)
                                                           : std::atan2(qy[e], qx[e]);
            }
      }
      for (int b = 0; b < n_blocks; ++b) x[oa + b] = ainit.empty() ? lam : ainit[b];
      if (!cart_) {
         start_cells = (long)S * m_q;
         for (long o = 0; o < start_cells; ++o) {
            const double c = x[oR + o] * (1.0 - x[oD + o]);
            if (c > 1e-12) start_products.push_back(c);
         }
      }
      return x;
   }

   // tv_mpcc.hpp's warm_start_chambolle_pock, the iteration only (verbatim).
   void chambolle_pock(const std::vector<double>& f, double lam, std::vector<double>& u,
                       std::vector<double>& qx, std::vector<double>& qy) {
      const double norm_K = std::sqrt(8.0);
      const double tau0 = 0.99 / (norm_K * lam), sig0 = 0.99 / norm_K;
      double tau = tau0, sig_lam = sig0;
      u = f;
      std::vector<double> ubar = u, u_new(m_u), div(m_u);
      qx.assign(m_q, 0.0);
      qy.assign(m_q, 0.0);
      int it = 0;
      for (; it < 3000; ++it) {
         const bool accelerated = it < 300;
         if (!accelerated && tau != tau0) { tau = tau0; sig_lam = sig0; }
         for (int a = 0; a < nc; ++a) {
            const double* u0 = &ubar[a * N];
            const double* u1 = u0 + N;
            for (int b = 0; b < nc; ++b) {
               const int e = a * nc + b;
               const double kx = (0.0 + -1.0 * u0[b]) + 1.0 * u0[b + 1];
               const double ky = (0.0 + -1.0 * u0[b]) + 1.0 * u1[b];
               qx[e] += sig_lam * kx;
               qy[e] += sig_lam * ky;
               const double nrm = std::max(1.0, std::hypot(qx[e], qy[e]));
               qx[e] /= nrm;
               qy[e] /= nrm;
            }
         }
         for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
               double d = 0.0;
               if (i < nc && j >= 1) d += 1.0 * qx[i * nc + j - 1];
               if (i < nc && j < nc) d += -1.0 * qx[i * nc + j];
               if (i >= 1 && j < nc) d += 1.0 * qy[(i - 1) * nc + j];
               if (i < nc && j < nc) d += -1.0 * qy[i * nc + j];
               div[i * N + j] = d;
            }
         double residual = 0.0;
         for (int i = 0; i < m_u; ++i) {
            u_new[i] = (tau * f[i] + u[i] - tau * lam * div[i]) / (tau + 1.0);
            residual = std::max(residual, std::abs(u_new[i] - u[i]));
         }
         residual /= tau;
         if (accelerated) {
            const double th = 1.0 / std::sqrt(1.0 + 2.0 * tau);
            for (int i = 0; i < m_u; ++i) ubar[i] = u_new[i] + th * (u_new[i] - u[i]);
            tau *= th;
            sig_lam /= th;
         } else {
            for (int i = 0; i < m_u; ++i) ubar[i] = 2.0 * u_new[i] - u[i];
         }
         u.swap(u_new);
         if (residual <= 1e-9) break;
      }
      warm_iters = std::max(warm_iters, std::min(it + 1, 3000));
   }

   // ---- the consensus split -------------------------------------------------
   struct TileSet {   // the distinct tiles of one variable, ascending
      int t[8], n = 0;
      void add(int k) {
         for (int i = 0; i < n; ++i)
            if (t[i] == k) return;
         if (n == 8) throw std::runtime_error("TileSet: a variable read by more than 8 tiles");
         t[n++] = k;
      }
      int pos(int k) const {
         for (int i = 0; i < n; ++i)
            if (t[i] == k) return i;
         return -1;
      }
   };
   TileSet tiles_of_u(int s, int node) const {
      const int i = node / N, j = node % N;
      TileSet ts;
      ts.add(part_.node_tile(s, i, j));
      if (i < nc && j < nc) ts.add(part_.cell_tile(s, i, j));
      if (i < nc && j >= 1) ts.add(part_.cell_tile(s, i, j - 1));
      if (i >= 1 && j < nc) ts.add(part_.cell_tile(s, i - 1, j));
      if (R_ > 0) {   // the h1 rows of the nodes within R
         const int r0 = part_.band[std::min(std::max(0, i - R_), nc - 1)];
         const int r1 = part_.band[std::min(std::min(N - 1, i + R_), nc - 1)];
         const int c0 = part_.band[std::min(std::max(0, j - R_), nc - 1)];
         const int c1 = part_.band[std::min(std::min(N - 1, j + R_), nc - 1)];
         for (int a = r0; a <= r1; ++a)
            for (int c = c0; c <= c1; ++c) ts.add(s * part_.tiles_per_image + a * part_.k + c);
      }
      std::sort(ts.t, ts.t + ts.n);
      return ts;
   }
   TileSet tiles_of_q(int s, int e, bool y) const {
      const int a = e / nc, b = e % nc;
      TileSet ts;
      ts.add(part_.cell_tile(s, a, b));
      ts.add(part_.node_tile(s, a, b));
      if (dtv_) {
         ts.add(part_.node_tile(s, a, b + 1));
         ts.add(part_.node_tile(s, a + 1, b));
      } else {
         ts.add(y ? part_.node_tile(s, a + 1, b) : part_.node_tile(s, a, b + 1));
      }
      std::sort(ts.t, ts.t + ts.n);
      return ts;
   }

   template <class F>
   void for_each_var(F fn) const {
      for (int s = 0; s < S; ++s)
         for (int i = 0; i < m_u; ++i) {
            const TileSet ts = tiles_of_u(s, i);
            fn(ou + s * m_u + i, ts.n > 1 ? -1 : ts.t[0], ou + s * m_u + i);
         }
      for (int y = 0; y < 2; ++y)
         for (int s = 0; s < S; ++s)
            for (int e = 0; e < m_q; ++e) {
               const TileSet ts = tiles_of_q(s, e, y);
               const int v = (y ? oqy : oqx) + s * m_q + e;
               fn(v, ts.n > 1 ? -1 : ts.t[0], v);
            }
      for (int o : {oR, oD, oTh})
         for (int s = 0; s < S && !cart_; ++s)
            for (int e = 0; e < m_q; ++e) fn(o + s * m_q + e, part_.cell_tile(s, e), o + s * m_q + e);
      for (int b = 0; b < n_blocks; ++b)
         fn(oa + b, atiles_[b].size() > 1 ? -1 : atiles_[b][0], oa + b);
      int c = n_orig;
      for (int s = 0; s < S; ++s)
         for (int i = 0; i < m_u; ++i) {
            const TileSet ts = tiles_of_u(s, i);
            if (ts.n > 1)
               for (int j = 0; j < ts.n; ++j) fn(c++, ts.t[j], ou + s * m_u + i);
         }
      for (int y = 0; y < 2; ++y)
         for (int s = 0; s < S; ++s)
            for (int e = 0; e < m_q; ++e) {
               const TileSet ts = tiles_of_q(s, e, y);
               if (ts.n > 1)
                  for (int j = 0; j < ts.n; ++j) fn(c++, ts.t[j], (y ? oqy : oqx) + s * m_q + e);
            }
      for (int b = 0; b < n_blocks; ++b)
         if (atiles_[b].size() > 1)
            for (int k : atiles_[b]) fn(c++, k, oa + b);
   }

   int seen_by(int v, int k) const {
      int base = -1, j = -1;
      if (v >= ou && v < ou + S * m_u) {
         const int idx = v - ou;
         const TileSet ts = tiles_of_u(idx / m_u, idx % m_u);
         if (ts.n > 1) base = copy_off_[0][idx], j = ts.pos(k);
      } else if (v >= oqx && v < oqy + S * m_q) {
         const bool y = v >= oqy;
         const int idx = v - (y ? oqy : oqx);
         const TileSet ts = tiles_of_q(idx / m_q, idx % m_q, y);
         if (ts.n > 1) base = copy_off_[1 + y][idx], j = ts.pos(k);
      } else if (v >= oa && v < oa + n_blocks) {
         const std::vector<int>& at = atiles_[v - oa];
         if (at.size() > 1) {
            const auto it = std::lower_bound(at.begin(), at.end(), k);
            base = copy_off_[3][v - oa];
            j = it != at.end() && *it == k ? (int)(it - at.begin()) : -1;
         }
      }
      return base >= 0 && j >= 0 ? n_orig + base + j : v;
   }

   // ---- the local view (as tv_mpcc.hpp's build_local) ------------------------
   void build_local(const std::vector<double>& x0, const std::vector<Instance>& data) {
      n_tiles_ = part_.n_tiles;
      const int k0 = dd::Comm::first_tile(dd::Comm::rank(), n_tiles_);
      const int k1 = dd::Comm::first_tile(dd::Comm::rank() + 1, n_tiles_);
      auto own = [&](int k) { return k >= k0 && k < k1; };
      const int MU = S * m_u, MQ = S * m_q;

      // 1. The copies
      n_orig = n;
      const int m_orig = mcon, n_eq_orig = n_eq;
      copy_off_[0].assign(MU + 1, 0);
      copy_off_[1].assign(MQ + 1, 0);
      copy_off_[2].assign(MQ + 1, 0);
      copy_off_[3].assign(n_blocks + 1, 0);
      int next = 0;
      for (int idx = 0; idx < MU; ++idx) {
         copy_off_[0][idx] = next;
         const TileSet ts = tiles_of_u(idx / m_u, idx % m_u);
         if (ts.n > 1) next += ts.n;
      }
      copy_off_[0][MU] = next;
      for (int y = 0; y < 2; ++y) {
         for (int idx = 0; idx < MQ; ++idx) {
            copy_off_[1 + y][idx] = next;
            const TileSet ts = tiles_of_q(idx / m_q, idx % m_q, y);
            if (ts.n > 1) next += ts.n;
         }
         copy_off_[1 + y][MQ] = next;
      }
      for (int b = 0; b < n_blocks; ++b) {
         copy_off_[3][b] = next;
         if (atiles_[b].size() > 1) next += (int)atiles_[b].size();
      }
      copy_off_[3][n_blocks] = next;
      n_link = next;
      n = n_orig + n_link;
      rlink = n_eq_orig;
      n_eq = n_eq_orig + n_link;
      rhr += n_link;
      rhd += n_link;
      rha += n_link;
      rcomp += n_link;
      mcon = m_orig + n_link;

      // 2. Local variables and rows, ascending
      std::vector<int> orig;
      std::vector<std::array<int, 4>> own_links;
      for_each_var([&](int v, int k, int o) {
         if (k >= 0 && !own(k)) return;
         gvar_.push_back(v);
         vtile_.push_back(k);
         orig.push_back(o);
         if (v >= n_orig) own_links.push_back({v, o, rlink + (v - n_orig), k});
      });
      auto add_row = [&](int r, int k) {
         if (!own(k)) return;
         grow_.push_back(r);
         rtile_.push_back(k);
      };
      for (int s = 0; s < S; ++s)
         for (int i = 0; i < m_u; ++i) add_row(rh1 + s * m_u + i, part_.node_tile(s, i));
      for (int block : {rh2x, rh2y, rh3x, rh3y})
         for (int s = 0; s < S && !(cart_ && block >= rh3x); ++s)
            for (int e = 0; e < m_q; ++e) add_row(block + s * m_q + e, part_.cell_tile(s, e));
      for (const auto& l : own_links) add_row(l[2], l[3]);
      for (int block : {rhr, rhd})
         for (int s = 0; s < S && !cart_; ++s)
            for (int e = 0; e < m_q; ++e) add_row(block + s * m_q + e, part_.cell_tile(s, e));
      for (int b = 0; b < n_blocks; ++b) add_row(rha + b, atiles_[b][0]);
      for (int s = 0; s < S && !cart_; ++s)
         for (int e = 0; e < m_q; ++e) add_row(rcomp + s * m_q + e, part_.cell_tile(s, e));
      auto lv = [&](int v) {
         const auto it = std::lower_bound(gvar_.begin(), gvar_.end(), v);
         return it != gvar_.end() && *it == v ? (int)(it - gvar_.begin()) : -1;
      };
      auto lr = [&](int r) {
         const auto it = std::lower_bound(grow_.begin(), grow_.end(), r);
         return it != grow_.end() && *it == r ? (int)(it - grow_.begin()) : -1;
      };
      first_slack_row_ = (int)(std::lower_bound(grow_.begin(), grow_.end(), n_eq) - grow_.begin());
      for (int b = 0; b < n_blocks; ++b)
         if (own(atiles_[b][0])) ha_.push_back({lr(rha + b), lv(seen_by(oa + b, atiles_[b][0]))});

      // 3. Own nodes (data stencil, weighted divergence) and cells (gradient)
      std::vector<std::pair<int, int>> tile_alpha;   // (tile, α variable)
      for (int s = 0; s < S; ++s) {
         const Instance& in = data[s];
         for (int node = 0; node < m_u; ++node) {
            const int k = part_.node_tile(s, node);
            if (!own(k)) continue;
            const int slot = (int)nodes_.size(), i = node / N, j = node % N;
            Node nd{s, node, lr(rh1 + s * m_u + node), (int)grp_alpha_.size(), 0};
            fnode_.push_back(in.rhs[node]);
            auto u_of = [&](int ii, int jj) { return lv(seen_by(ou + s * m_u + ii * N + jj, k)); };
            if (!in.diag.empty()) {
               data_.push_back({slot, u_of(i, j), in.diag[node]});
            } else {   // CᵀC = G ⊗ G
               for (int ii = std::max(0, i - R_); ii <= std::min(N - 1, i + R_); ++ii)
                  for (int jj = std::max(0, j - R_); jj <= std::min(N - 1, j + R_); ++jj)
                     data_.push_back({slot, u_of(ii, jj), blur_->g1(i, ii) * blur_->g1(j, jj)});
            }
            // K_ξᵀ(α∘q) at node (i, j): the cells (i−1, j), (i, j−1), (i, j)
            // ascending; for K only the nonzero terms (Kxᵀ: (i, j−1) +1,
            // (i, j) −1; Kyᵀ: (i−1, j) +1, (i, j) −1, as tv_mpcc.hpp)
            const int cells[3][2] = {{i - 1, j}, {i, j - 1}, {i, j}};
            for (int y = 0; y < 2; ++y)
               for (const auto& ce : cells) {
                  const int a = ce[0], b = ce[1];
                  if (a < 0 || b < 0 || a >= nc || b >= nc) continue;
                  const double cx = (a == i && b == j) ? -1.0 : (a == i && b + 1 == j) ? 1.0 : 0.0;
                  const double cy = (a == i && b == j) ? -1.0 : (a + 1 == i && b == j) ? 1.0 : 0.0;
                  const int e = a * nc + b;
                  double v;
                  if (in.dtv()) v = y ? cx * in.m12[e] + cy * in.m22[e] : cx * in.m11[e] + cy * in.m12[e];
                  else {
                     v = y ? cy : cx;
                     if (v == 0.0) continue;
                  }
                  const int av = lv(seen_by(oa + part_.block(e), k));
                  int gi = nd.g0;
                  while (gi < (int)grp_alpha_.size() && grp_alpha_[gi] != av) ++gi;
                  if (gi == (int)grp_alpha_.size()) {
                     grp_alpha_.push_back(av);
                     tile_alpha.push_back({k, av});
                  }
                  const int qv = lv(seen_by((y ? oqy : oqx) + s * m_q + e, k));
                  (y ? div_y_ : div_x_).push_back({slot, qv, v, gi});
               }
            if ((int)grp_alpha_.size() == nd.g0) {   // the last node reads no cell: keep
               // tv_mpcc.hpp's structural h1/α entry (value 0) with the α of its anchor cell
               const int av = lv(seen_by(oa + part_.block(std::min(i, nc - 1), std::min(j, nc - 1)), k));
               grp_alpha_.push_back(av);
               tile_alpha.push_back({k, av});
            }
            nd.g1 = (int)grp_alpha_.size();
            nodes_.push_back(nd);
         }
      }
      // the div entries must come node by node within each list (they do) and
      // the x list before the y list per node (kept by separate lists)
      for (int s = 0; s < S; ++s) {
         const Instance& in = data[s];
         for (int e = 0; e < m_q; ++e) {
            const int k = part_.cell_tile(s, e);
            if (!own(k)) continue;
            const int o = s * m_q + e;
            const int slot = (int)cells_.size(), a = e / nc, b = e % nc;
            if (cart_)
               cells_.push_back({s, e, -1, -1, -1, lv(seen_by(oqx + o, k)), lv(seen_by(oqy + o, k)),
                                 lr(rh2x + o), lr(rh2y + o), -1, -1, -1, -1, -1});
            else
               cells_.push_back({s, e, lv(oR + o), lv(oD + o), lv(oTh + o), lv(seen_by(oqx + o, k)),
                                 lv(seen_by(oqy + o, k)), lr(rh2x + o), lr(rh2y + o), lr(rh3x + o),
                                 lr(rh3y + o), lr(rhr + o), lr(rhd + o), lr(rcomp + o)});
            if (!cart_) rho_d_.push_back({lv(oD + o), lv(seen_by(oa + part_.block(e), k))});
            const int n00 = lv(seen_by(ou + s * m_u + a * N + b, k));
            const int n01 = lv(seen_by(ou + s * m_u + a * N + b + 1, k));
            const int n10 = lv(seen_by(ou + s * m_u + (a + 1) * N + b, k));
            if (cart_) {   // c's stencil: the nodes (a, b), (a, b+1), (a+1, b)
               CartCell cc{{n00, n01, n10}, {-1.0, 1.0, 0.0}, {-1.0, 0.0, 1.0}};
               if (in.dtv()) {
                  cc.bx[0] = -(in.m11[e] + in.m12[e]); cc.bx[1] = in.m11[e]; cc.bx[2] = in.m12[e];
                  cc.by[0] = -(in.m12[e] + in.m22[e]); cc.by[1] = in.m12[e]; cc.by[2] = in.m22[e];
               }
               cst_.push_back(cc);
            }
            if (in.dtv()) {   // M (Kx u, Ky u)
               grad_x_.push_back({slot, n00, -(in.m11[e] + in.m12[e])});
               grad_x_.push_back({slot, n01, in.m11[e]});
               grad_x_.push_back({slot, n10, in.m12[e]});
               grad_y_.push_back({slot, n00, -(in.m12[e] + in.m22[e])});
               grad_y_.push_back({slot, n01, in.m12[e]});
               grad_y_.push_back({slot, n10, in.m22[e]});
            } else {
               grad_x_.push_back({slot, n00, -1.0});
               grad_x_.push_back({slot, n01, 1.0});
               grad_y_.push_back({slot, n00, -1.0});
               grad_y_.push_back({slot, n10, 1.0});
            }
         }
      }
      for (const auto& l : own_links) links_.push_back({lr(l[2]), lv(l[0]), lv(l[1])});
      {  // --rho-step: the weight of every q a node reads (its tile's α), and of the
         // originals of q copies (the original α), so that the links stay linear
         std::unordered_map<int, int> orig_of, wq;
         for (const auto& l : links_) orig_of[l.copy] = l.orig;
         for (const auto* L : {&div_x_, &div_y_})
            for (const auto& e : *L) wq.emplace(e.c, grp_alpha_[e.g]);
         std::vector<std::pair<int, int>> origs;
         for (const auto& [v, w] : wq) {
            const auto iv = orig_of.find(v);
            if (iv == orig_of.end()) continue;
            const auto iw = orig_of.find(w);
            origs.push_back({iv->second, iw != orig_of.end() ? iw->second : w});
         }
         for (const auto& o : origs) wq.emplace(o.first, o.second);
         rho_q_.assign(wq.begin(), wq.end());
         std::sort(rho_q_.begin(), rho_q_.end());
      }
      for (size_t l = 0; l < gvar_.size() && gvar_[l] < ou + MU; ++l) u_orig_.push_back((int)l);
      for (int b = 0; b < n_blocks; ++b) {
         const int l = lv(oa + b);
         if (l >= 0) alpha_orig_.push_back(l);
      }
      // every row reads only variables of its own tile (nlp.hpp): a stencil
      // that reached past its sharing table would show up here
      auto check = [&](int var, int row) {
         if (var < 0 || vtile_[var] != rtile_[row])
            throw std::runtime_error("TvLearn: a row reads a variable of another tile");
      };
      for (const auto& e : data_) check(e.c, nodes_[e.r].h1);
      for (const auto& e : div_x_) check(e.c, nodes_[e.r].h1);
      for (const auto& e : div_y_) check(e.c, nodes_[e.r].h1);
      for (const Node& q : nodes_)
         for (int gi = q.g0; gi < q.g1; ++gi) check(grp_alpha_[gi], q.h1);
      for (const auto& e : grad_x_) check(e.c, cells_[e.r].h2x);
      for (const auto& e : grad_y_) check(e.c, cells_[e.r].h2y);
      for (const auto& h : ha_) check(h.second, h.first);
      for (size_t b = 0; cart_ && b < cells_.size(); ++b)
         for (int j = 0; j < 3; ++j) {
            check(cst_[b].var[j], cells_[b].h2x);
            check(cst_[b].var[j], cells_[b].h2y);
            check(cells_[b].qx, cells_[b].h2x);
            check(cells_[b].qy, cells_[b].h2y);
         }
      std::sort(tile_alpha.begin(), tile_alpha.end());
      tile_alpha.erase(std::unique(tile_alpha.begin(), tile_alpha.end()), tile_alpha.end());
      for (const auto& p : tile_alpha) tile_alpha_.push_back(p.second);

      // the patterns, in the order of jac_values / hess_values
      auto J = [&](int r, int c) { jac_row_.push_back(r); jac_col_.push_back(c); };
      for (const auto& e : data_) J(nodes_[e.r].h1, e.c);
      for (const auto& e : div_x_) J(nodes_[e.r].h1, e.c);
      for (const auto& e : div_y_) J(nodes_[e.r].h1, e.c);
      for (const Node& q : nodes_)
         for (int gi = q.g0; gi < q.g1; ++gi) J(q.h1, grp_alpha_[gi]);
      if (cart_) {   // in the order of cart_jac, then ha and the links
         for (size_t b = 0; b < cells_.size(); ++b) {
            for (int j = 0; j < 3; ++j) J(cells_[b].h2x, cst_[b].var[j]);
            J(cells_[b].h2x, cells_[b].qx);
            for (int j = 0; j < 3; ++j) J(cells_[b].h2y, cst_[b].var[j]);
            J(cells_[b].h2y, cells_[b].qy);
         }
         for (const auto& h : ha_) J(h.first, h.second);
         for (const Link& l : links_) J(l.row, l.copy);
         for (const Link& l : links_) J(l.row, l.orig);
      }
      for (const auto& e : grad_x_) if (!cart_) J(cells_[e.r].h2x, e.c);
      if (!cart_) {
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
      for (const auto& h : ha_) J(h.first, h.second);
      for (const Cell& q : cells_) J(q.comp, q.r);
      for (const Cell& q : cells_) J(q.comp, q.d);
      for (const Link& l : links_) J(l.row, l.copy);
      for (const Link& l : links_) J(l.row, l.orig);
      }

      auto H = [&](int r, int c) { hess_row_.push_back(r); hess_col_.push_back(c); };
      for (int l : u_orig_) H(l, l);
      if (cart_) {   // in the order of cart_hess, then the α entries
         for (size_t b = 0; b < cells_.size(); ++b) {
            for (int j = 0; j < 3; ++j)
               for (int i = 0; i <= j; ++i) H(cst_[b].var[j], cst_[b].var[i]);
            for (int j = 0; j < 3; ++j) H(cells_[b].qx, cst_[b].var[j]);
            for (int j = 0; j < 3; ++j) H(cells_[b].qy, cst_[b].var[j]);
         }
      }
      for (const Cell& q : cells_) if (!cart_) H(q.th, q.r);
      for (const Cell& q : cells_) if (!cart_) H(q.th, q.d);
      for (const Cell& q : cells_) if (!cart_) H(q.th, q.th);
      for (const Cell& q : cells_) if (!cart_) H(q.d, q.r);
      for (const auto& e : div_x_) H(grp_alpha_[e.g], e.c);
      for (const auto& e : div_y_) H(grp_alpha_[e.g], e.c);
      for (int l : tile_alpha_) H(l, l);
      for (int l : alpha_orig_) H(l, l);

      // 4. The start and the objective's reference values
      x_start.resize(gvar_.size());
      ref_.assign(gvar_.size(), 0.0);
      for (size_t l = 0; l < gvar_.size(); ++l) {
         const int v = gvar_[l];
         x_start[l] = x0[orig[l]];
         if (v < ou + MU) ref_[l] = data[(v - ou) / m_u].clean[(v - ou) % m_u];
         else if (!cart_ && v >= oTh && v < oTh + MQ) ref_[l] = x0[v];
      }
      for (auto& c : copy_off_) std::vector<int>().swap(c);
   }
};
