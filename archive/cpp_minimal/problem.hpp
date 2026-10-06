// problem.hpp — learning a TV-denoising weight, written as one smooth NLP.
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
// THE COMPLEMENTARITY ROW CAN BE WRITTEN TWO WAYS (Comp).  Scholtes writes it as
// above.  Its gradient (1 − δ, −r) has norm √(2t) at the knee r = 1 − δ = √t of
// the relaxation boundary, where the barrier parks most active cells, while every
// other row is O(1): the multipliers of these rows are conditioned like 1/√t and
// inf_du stalls (docs/inf_du_stall).  With s = 1 − δ, the smoothed
// Fischer–Burmeister function
//
//         φ_t(r, s) = r + s − √(r² + s² + 2t) ≤ 0
//
// describes the SAME set (for r, s ≥ 0:  r + s ≤ √(r² + s² + 2t)  ⇔  r·s ≤ t),
// but its gradient (1 − r/w, 1 − s/w), w = √(r² + s² + 2t), has norm in [1/√2, 1]
// on the boundary for every t.  Same sparsity, same row type; only the three
// evaluation routines see the difference.
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
// That is exactly the arrowhead shape arrowhead.hpp exploits.  The objective stays
// on the original indices, so it is untouched.
//
// Everything below keeps a fixed ORDER for variables, rows and nonzeros: IPOPT
// builds the KKT matrix from these arrays, and the order is the sparsity pattern.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include <limits>

#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpTNLP.hpp"
#include "IpTNLPAdapter.hpp"
#include "image.hpp"
#include "partition.hpp"

class ConsensusTV : public Ipopt::TNLP {
   using Index = Ipopt::Index;
   using Number = Ipopt::Number;

public:
   // How the complementarity row is written (see the header).
   enum class Comp { Scholtes, FischerBurmeister };

   // ---- continuation state, set by main.cpp, updated by the callback --------
   double t = 1.0;             // current relaxation level
   double t_min = 1e-4;        // the level the answer is reported at
   double t_mu_scale = 10.0;   // t follows the barrier: t = max(t_min, 10·μ)
   double t_rate = 0.5;        // ... but may at most halve per iteration
   double c_theta = 1.0;       // ε_θ = c_θ · t
   double eps_theta = 0.0;
   double gate_floor = 1e-8;   // floor on the level gate (see the callback)
   bool gate_fired = false;

   // ---- sizes and offsets -------------------------------------------------
   int N, nc, m_u, m_q;        // image side, cells per side, #nodes, #cells
   int n, mcon, n_eq, n_ineq;  // #variables, #rows, #equality rows, #inequality rows
   int n_link = 0;             // #copies = #linking rows

   // Column offsets of the ORIGINAL variables.  Copies are appended after them.
   //   x = [ u | qx | qy | r | δ | θ | α | copies... ]
   int ou, oqx, oqy, oR, oD, oTh, oa;
   // Row offsets.  Linking rows are appended to the equality block.
   int rh1, rh2x, rh2y, rh3x, rh3y, rlink, rhr, rhd, rha, rcomp;

   std::vector<double> x_start;    // Chambolle–Pock warm start
   std::vector<double> solution;   // written by finalize_solution: x ...
   std::vector<double> multipliers;//   ... the constraint multipliers λ
   double objective = 0.0;         //   ... and f(x)

   // The in-solve continuation path, one row (iter, μ, t, weight, max r(1−δ))
   // per IPOPT iteration.  weight and max r(1−δ) are NaN in restoration.
   std::vector<double> mu_trace;

   // Cumulative linear-solver counters, read once per iteration so the log can
   // show what each iteration cost.  Optional: main.cpp wires it to the solver.
   struct LinearStats { long attempts, wrong_inertia, solves, rejected, cg_iters; };
   std::function<LinearStats()> linear_stats;

   // Fallback for a failed solve.  In monotone mode μ only drops once the
   // barrier subproblem at the previous μ has converged, so the iterate seen
   // just BEFORE each drop is a converged relaxation level.  The one with the
   // smallest max r(1−δ) is kept here.
   bool have_banked = false;
   double banked_comp = 1e300, banked_t = 0.0;
   std::vector<double> banked_x;

   int n_orig = 0;                 // #variables before the copies were added

   // Build the whole NLP.  Order matters: the warm start and θ_ref are computed
   // on the unsplit problem, THEN the consensus copies are appended.
   ConsensusTV(const Image& img, const Partition& part, double sigma,
               Comp comp = Comp::Scholtes)
       : N(img.N), clean_(img.clean), f_(img.noisy), fb_(comp == Comp::FischerBurmeister) {
      nc = N - 1;
      m_u = N * N;
      m_q = nc * nc;
      layout_unsplit();
      build_stencils();
      warm_start_chambolle_pock(0.7 * sigma);
      theta_ref_.assign(x_start.begin() + oTh, x_start.begin() + oTh + m_q);
      split_into_consensus_form(part);
   }

   // Which subdomain owns each unknown of the KKT system, in IPOPT's order
   //   [ primals | slacks of the inequalities | λ of equalities | λ of inequalities ].
   // −1 marks the border: exactly the consensus variables.
   std::vector<int> kkt_owner() const {
      std::vector<int> owner;
      owner.reserve(n + 2 * n_ineq + n_eq);
      owner.insert(owner.end(), col_tile_.begin(), col_tile_.end());
      for (int r = n_eq; r < mcon; ++r) owner.push_back(row_tile_[r]);   // slacks
      for (int r = 0; r < n_eq; ++r) owner.push_back(row_tile_[r]);      // λ equality
      for (int r = n_eq; r < mcon; ++r) owner.push_back(row_tile_[r]);   // λ inequality
      return owner;
   }

   // =========================================================================
   //  IPOPT callbacks
   // =========================================================================
   bool get_nlp_info(Index& n_, Index& m_, Index& nnz_jac, Index& nnz_h,
                     IndexStyleEnum& style) override {
      n_ = n;
      m_ = mcon;
      nnz_jac = (Index)jac_row_.size();
      nnz_h = (Index)hess_row_.size();
      style = C_STYLE;
      return true;
   }

   // Only δ ≤ 1 is a variable bound.  r ≥ 0, δ ≥ 0, α ≥ 0 are ROWS (hr, hd, ha).
   bool get_bounds_info(Index, Number* xl, Number* xu, Index, Number* gl,
                        Number* gu) override {
      for (int i = 0; i < n; ++i) { xl[i] = -2e19; xu[i] = 2e19; }
      for (int e = 0; e < m_q; ++e) xu[oD + e] = 1.0;
      xu[oa] = kAlphaMax;
      for (int i = 0; i < n_eq; ++i) { gl[i] = 0.0; gu[i] = 0.0; }         // = 0
      for (int i = rhr; i < rcomp; ++i) { gl[i] = 0.0; gu[i] = 2e19; }     // ≥ 0
      for (int i = rcomp; i < mcon; ++i) { gl[i] = -2e19; gu[i] = 0.0; }   // ≤ 0
      return true;
   }

   bool get_starting_point(Index, bool init_x, Number* x, bool init_z, Number* zL,
                           Number* zU, Index, bool init_lam, Number* lam) override {
      if (init_x) for (int i = 0; i < n; ++i) x[i] = x_start[i];
      if (init_z) for (int i = 0; i < n; ++i) zL[i] = zU[i] = 0.0;
      if (init_lam) for (int i = 0; i < mcon; ++i) lam[i] = 0.0;
      return true;
   }

   bool eval_f(Index, const Number* x, bool, Number& obj) override {
      double s = 0.0;
      for (int i = 0; i < m_u; ++i) {
         const double d = x[ou + i] - clean_[i];
         s += d * d;
      }
      obj = 0.5 * s + 0.5 * kRegAlpha * x[oa] * x[oa];
      if (eps_theta != 0.0) {
         double g = 0.0;
         for (int e = 0; e < m_q; ++e) {
            const double d = x[oTh + e] - theta_ref_[e];
            g += d * d;
         }
         obj += 0.5 * eps_theta * g;
      }
      return true;
   }

   bool eval_grad_f(Index, const Number* x, bool, Number* g) override {
      for (int i = 0; i < n; ++i) g[i] = 0.0;
      for (int i = 0; i < m_u; ++i) g[ou + i] = x[ou + i] - clean_[i];
      g[oa] = kRegAlpha * x[oa];
      if (eps_theta != 0.0)
         for (int e = 0; e < m_q; ++e)
            g[oTh + e] = eps_theta * (x[oTh + e] - theta_ref_[e]);
      return true;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
      std::vector<double> div(m_u, 0.0), kxu(m_q, 0.0), kyu(m_q, 0.0);
      for (const auto& e : div_x_) div[e.r] += e.v * x[e.c];
      for (const auto& e : div_y_) div[e.r] += e.v * x[e.c];
      for (const auto& e : grad_x_) kxu[e.r] += e.v * x[e.c];
      for (const auto& e : grad_y_) kyu[e.r] += e.v * x[e.c];
      for (int i = 0; i < m_u; ++i)
         g[rh1 + i] = x[u_of_h1_[i]] - f_[i] + x[alpha_of_h1_[i]] * div[i];
      for (int e = 0; e < m_q; ++e) {
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         const double r = x[oR + e], d = x[oD + e];
         g[rh2x + e] = kxu[e] - r * c;
         g[rh2y + e] = kyu[e] - r * s;
         g[rh3x + e] = x[qx_of_h3_[e]] - d * c;
         g[rh3y + e] = x[qy_of_h3_[e]] - d * s;
         g[rhr + e] = r;
         g[rhd + e] = d;
         g[rcomp + e] = fb_ ? fb(r, 1.0 - d) : r * (1.0 - d) - t;
      }
      g[rha] = x[alpha_of_tile_[0]];
      for (int l = 0; l < n_link; ++l) g[rlink + l] = x[link_copy_[l]] - x[link_orig_[l]];
      return true;
   }

   // Values in exactly the order of jac_row_/jac_col_ (see build_structures).
   bool eval_jac_g(Index, const Number* x, bool, Index, Index nele, Index* iRow,
                   Index* jCol, Number* values) override {
      if (values == nullptr) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = jac_row_[k]; jCol[k] = jac_col_[k]; }
         return true;
      }
      std::vector<double> div(m_u, 0.0);
      for (const auto& e : div_x_) div[e.r] += e.v * x[e.c];
      for (const auto& e : div_y_) div[e.r] += e.v * x[e.c];
      Index k = 0;
      for (int i = 0; i < m_u; ++i) values[k++] = 1.0;                              // h1/u
      for (const auto& e : div_x_) values[k++] = x[alpha_of_h1_[e.r]] * e.v;        // h1/qx
      for (const auto& e : div_y_) values[k++] = x[alpha_of_h1_[e.r]] * e.v;        // h1/qy
      for (int i = 0; i < m_u; ++i) values[k++] = div[i];                           // h1/α
      for (const auto& e : grad_x_) values[k++] = e.v;                              // h2x/u
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);            // h2x/r
      for (int e = 0; e < m_q; ++e) values[k++] = x[oR + e] * std::sin(x[oTh + e]); // h2x/θ
      for (const auto& e : grad_y_) values[k++] = e.v;                              // h2y/u
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);            // h2y/r
      for (int e = 0; e < m_q; ++e) values[k++] = -x[oR + e] * std::cos(x[oTh + e]);// h2y/θ
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                              // h3x/qx
      for (int e = 0; e < m_q; ++e) values[k++] = -std::cos(x[oTh + e]);            // h3x/δ
      for (int e = 0; e < m_q; ++e) values[k++] = x[oD + e] * std::sin(x[oTh + e]); // h3x/θ
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                              // h3y/qy
      for (int e = 0; e < m_q; ++e) values[k++] = -std::sin(x[oTh + e]);            // h3y/δ
      for (int e = 0; e < m_q; ++e) values[k++] = -x[oD + e] * std::cos(x[oTh + e]);// h3y/θ
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                              // hr/r
      for (int e = 0; e < m_q; ++e) values[k++] = 1.0;                              // hd/δ
      values[k++] = 1.0;                                                            // ha/α
      if (fb_) {   // ∂φ/∂r = 1 − r/w,  ∂φ/∂δ = −∂φ/∂s = s/w − 1
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0 - x[oR + e] / fb_w(e, x);  // comp/r
         for (int e = 0; e < m_q; ++e) values[k++] = (1.0 - x[oD + e]) / fb_w(e, x) - 1.0; // comp/δ
      } else {
         for (int e = 0; e < m_q; ++e) values[k++] = 1.0 - x[oD + e];               // comp/r
         for (int e = 0; e < m_q; ++e) values[k++] = -x[oR + e];                    // comp/δ
      }
      for (int l = 0; l < n_link; ++l) values[k++] = 1.0;                           // link/copy
      for (int l = 0; l < n_link; ++l) values[k++] = -1.0;                          // link/orig
      return true;
   }

   // Hessian of the Lagrangian, lower triangle, in the order of hess_row_/hess_col_.
   bool eval_h(Index, const Number* x, bool, Number obj_factor, Index,
               const Number* lam, bool, Index nele, Index* iRow, Index* jCol,
               Number* values) override {
      if (values == nullptr) {
         for (Index k = 0; k < nele; ++k) { iRow[k] = hess_row_[k]; jCol[k] = hess_col_[k]; }
         return true;
      }
      Index k = 0;
      for (int i = 0; i < m_u; ++i) values[k++] = obj_factor;                       // (u,u)
      for (int e = 0; e < m_q; ++e)                                                 // (θ,r)
         values[k++] = lam[rh2x + e] * std::sin(x[oTh + e])
                     - lam[rh2y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e)                                                 // (θ,δ)
         values[k++] = lam[rh3x + e] * std::sin(x[oTh + e])
                     - lam[rh3y + e] * std::cos(x[oTh + e]);
      for (int e = 0; e < m_q; ++e) {                                               // (θ,θ)
         const double c = std::cos(x[oTh + e]), s = std::sin(x[oTh + e]);
         values[k++] = x[oR + e] * (lam[rh2x + e] * c + lam[rh2y + e] * s)
                     + x[oD + e] * (lam[rh3x + e] * c + lam[rh3y + e] * s)
                     + obj_factor * eps_theta;
      }
      if (fb_) {   // ∇²φ in (r, δ):  φ_rr = −(s²+2t)/w³, φ_δδ = −(r²+2t)/w³, φ_rδ = −r·s/w³
         for (int e = 0; e < m_q; ++e) {
            const double r = x[oR + e], s = 1.0 - x[oD + e], w = fb_w(e, x);
            const double w3 = w * w * w, l = lam[rcomp + e];
            values[k++] = -l * r * s / w3;                                          // (δ,r)
            values[k++] = -l * (s * s + 2.0 * t) / w3;                              // (r,r)
            values[k++] = -l * (r * r + 2.0 * t) / w3;                              // (δ,δ)
         }
      } else {
         for (int e = 0; e < m_q; ++e) values[k++] = -lam[rcomp + e];               // (δ,r)
      }
      for (const auto& e : div_x_) values[k++] = e.v * lam[rh1 + e.r];              // (α,qx)
      for (const auto& e : div_y_) values[k++] = e.v * lam[rh1 + e.r];              // (α,qy)
      // (α,α) per tile: zero, because the weight enters linearly.  The entries
      // stay in the pattern only so it matches the reference implementation.
      for (int k2 = 0; k2 < n_tiles_; ++k2) values[k++] = 0.0;
      values[k++] = obj_factor * kRegAlpha;                                         // (α,α)
      return true;
   }

   // Called once per IPOPT iteration.  Two jobs:
   //
   //  1. THE LEVEL GATE.  Once t has reached t_min, stop as soon as the primal
   //     and dual infeasibility and μ are all below √t.  A relaxation at level t
   //     is only accurate to about √t, so solving it more tightly is wasted work.
   //     Returning false makes IPOPT stop with User_Requested_Stop.
   //
   //  2. THE μ-COUPLED CONTINUATION.  Tighten the relaxation along with the
   //     barrier, t = max(t_min, 10·μ), never loosening and at most halving per
   //     iteration.  This is legal mid-solve because t enters only eval_g.
   //     Only in regular mode: in restoration IPOPT solves a different NLP with
   //     its own barrier, and following THAT μ would push t to t_min before the
   //     original problem has got there.
   //
   //  3. BOOKKEEPING for the solution file: the μ-trace row, and the banked
   //     fallback level (see have_banked).
   //
   //  4. THE LOG LINE.  An "r" after the iteration number marks restoration.
   //     a_pr/a_du are the accepted step sizes, ls the line-search trials, reg the
   //     primal regularization δ_w IPOPT added for the inertia.  The linear-solver
   //     columns count work since the previous line: fac factorize() calls
   //     (wi = rejected for wrong inertia), sol interface solves (rej = residual
   //     above 1e-2), cg CG iterations.
   bool intermediate_callback(Ipopt::AlgorithmMode mode, Index iter, Number obj,
                              Number inf_pr, Number inf_du, Number mu, Number,
                              Number reg, Number alpha_du, Number alpha_pr,
                              Index ls_trials,
                              const Ipopt::IpoptData* ip_data,
                              Ipopt::IpoptCalculatedQuantities* ip_cq) override {
      const bool regular = mode == Ipopt::RegularMode;
      if (regular && iter > 0 && t <= t_min * (1.0 + 1e-9)) {
         const double gate = std::max(gate_floor, std::sqrt(t));
         if (inf_pr <= gate && inf_du <= gate && mu <= gate) {
            gate_fired = true;
            std::printf("  [level gate] t=%.2e  inf_pr=%.1e  inf_du=%.1e  mu=%.1e"
                        "  all <= sqrt(t) = %.1e: stopping\n",
                        t, inf_pr, inf_du, mu, gate);
            return false;
         }
      }
      if (regular) {
         double t_new = std::max(t_min, t_mu_scale * mu);
         if (t_new < t * t_rate) t_new = std::max(t_min, t * t_rate);
         if (t_new < t) {
            t = t_new;
            eps_theta = c_theta * t_new;
         }
      }

      // The current iterate, in our variable order.  During restoration IPOPT
      // solves a different NLP, the casts fail, and the row records NaN.
      double weight_now = std::numeric_limits<double>::quiet_NaN();
      double comp_now = std::numeric_limits<double>::quiet_NaN();
      if (ip_cq && ip_data) {
         auto* orig = dynamic_cast<Ipopt::OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
         auto* adapter = orig ? dynamic_cast<Ipopt::TNLPAdapter*>(GetRawPtr(orig->nlp())) : nullptr;
         if (adapter && IsValid(ip_data->curr()) && IsValid(ip_data->curr()->x())) {
            if ((int)x_now_.size() != n) x_now_.assign(n, 0.0);
            adapter->ResortX(*ip_data->curr()->x(), x_now_.data());
            weight_now = x_now_[oa];
            comp_now = max_complementarity(x_now_.data());
         }
      }
      // Bank the previous iterate whenever μ drops (it converged a level).
      if (regular && std::isfinite(comp_now)) {
         if (prev_valid_ && mu < mu_prev_ * (1.0 - 1e-12) && prev_comp_ < banked_comp) {
            banked_x = prev_x_;
            banked_comp = prev_comp_;
            banked_t = prev_t_;
            have_banked = true;
         }
         prev_x_ = x_now_;
         prev_comp_ = comp_now;
         prev_t_ = t;
         prev_valid_ = true;
         mu_prev_ = mu;
      }
      mu_trace.insert(mu_trace.end(), {(double)iter, mu, t, weight_now, comp_now});

      LinearStats ls{0, 0, 0, 0, 0};
      if (linear_stats) ls = linear_stats();
      if (iter > 0) {
         std::printf("  it=%4d%c mu=%.2e  t=%.2e  inf_pr=%.1e  inf_du=%.1e  obj=%.4e"
                     "  a_pr=%.1e a_du=%.1e ls=%d reg=%.1e"
                     "  | fac=%ld wi=%ld sol=%ld rej=%ld cg=%ld\n",
                     (int)iter, regular ? ' ' : 'r', mu, t, inf_pr, inf_du, obj,
                     alpha_pr, alpha_du, (int)ls_trials, reg,
                     ls.attempts - ls_prev_.attempts,
                     ls.wrong_inertia - ls_prev_.wrong_inertia,
                     ls.solves - ls_prev_.solves, ls.rejected - ls_prev_.rejected,
                     ls.cg_iters - ls_prev_.cg_iters);
         std::fflush(stdout);
      }
      ls_prev_ = ls;
      return true;
   }

   void finalize_solution(Ipopt::SolverReturn, Index, const Number* x, const Number*,
                          const Number*, Index, const Number*, const Number* lam,
                          Number obj, const Ipopt::IpoptData*,
                          Ipopt::IpoptCalculatedQuantities*) override {
      solution.assign(x, x + n);
      multipliers.assign(lam, lam + mcon);
      objective = obj;
   }

   // max over cells of r·(1 − δ): how far from exact complementarity x is.
   double max_complementarity(const double* x) const {
      double c = 0.0;
      for (int e = 0; e < m_q; ++e) c = std::max(c, x[oR + e] * (1.0 - x[oD + e]));
      return c;
   }

private:
   // The Fischer–Burmeister row and its w = √(r² + s² + 2t) at cell e.
   double fb(double r, double s) const { return r + s - std::sqrt(r * r + s * s + 2.0 * t); }
   double fb_w(int e, const Number* x) const {
      const double r = x[oR + e], s = 1.0 - x[oD + e];
      return std::sqrt(r * r + s * s + 2.0 * t);
   }
   const bool fb_;

   static constexpr double kRegAlpha = 1e-4;   // ridge on α
   static constexpr double kAlphaMax = 2e19;   // no upper bound on α in practice

   struct Tri { int r, c; double v; };         // one nonzero of a sparse operator

   std::vector<double> clean_, f_, theta_ref_;
   std::vector<double> x_now_;                 // callback scratch
   std::vector<double> prev_x_;                // the iterate one callback ago
   double prev_comp_ = 1e300, prev_t_ = 0.0, mu_prev_ = 1e300;
   bool prev_valid_ = false;
   LinearStats ls_prev_{0, 0, 0, 0, 0};        // linear_stats at the previous line
   std::vector<Tri> Kx_, Ky_, KxT_, KyT_;      // gradient stencils and transposes
   int n_tiles_ = 0;

   // Structure of the Jacobian and the Hessian (IPOPT's triplet format).
   std::vector<int> jac_row_, jac_col_, hess_row_, hess_col_;

   // Ownership after the split.
   std::vector<int> col_tile_;   // variable → tile, −1 for consensus variables
   std::vector<int> row_tile_;   // row      → tile
   std::vector<int> link_copy_, link_orig_;   // linking row ℓ: copy − original = 0

   // "Effective index" tables: which variable (original or copy) each row uses.
   std::vector<int> u_of_h1_;        // h1 row i  → the u  its tile uses
   std::vector<int> alpha_of_h1_;    // h1 row i  → the α  its tile uses
   std::vector<int> alpha_of_tile_;  // tile      → its α copy
   std::vector<int> qx_of_h3_, qy_of_h3_;   // h3 row e → the qx / qy its tile uses
   std::vector<Tri> div_x_, div_y_;   // Kxᵀqx, Kyᵀqy in h1, remapped to copies
   std::vector<Tri> grad_x_, grad_y_; // Kx u, Ky u in h2, remapped to copies

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
   // variables.  At that point u, q already satisfy h1 almost exactly, so IPOPT
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
      for (const auto& e : KxT_) div_x_.push_back({e.r, seen_by(oqx + e.c, node_tile[e.r]), e.v});
      for (const auto& e : KyT_) div_y_.push_back({e.r, seen_by(oqy + e.c, node_tile[e.r]), e.v});
      for (const auto& e : Kx_) grad_x_.push_back({e.r, seen_by(ou + e.c, cell_tile[e.r]), e.v});
      for (const auto& e : Ky_) grad_y_.push_back({e.r, seen_by(ou + e.c, cell_tile[e.r]), e.v});
      qx_of_h3_.resize(m_q);
      qy_of_h3_.resize(m_q);
      for (int e = 0; e < m_q; ++e) {
         qx_of_h3_[e] = seen_by(oqx + e, cell_tile[e]);
         qy_of_h3_[e] = seen_by(oqy + e, cell_tile[e]);
      }

      build_structures();

      // 6. Copies start equal to their consensus value, so the start is
      //    exactly feasible for the linking rows.
      x_start.resize(n);
      for (int l = 0; l < n_link; ++l) x_start[link_copy_[l]] = x_start[link_orig_[l]];
   }

   // The nonzero pattern.  eval_jac_g and eval_h fill values in THIS order.
   void build_structures() {
      auto J = [&](int r, int c) { jac_row_.push_back(r); jac_col_.push_back(c); };
      for (int i = 0; i < m_u; ++i) J(rh1 + i, u_of_h1_[i]);
      for (const auto& e : div_x_) J(rh1 + e.r, e.c);
      for (const auto& e : div_y_) J(rh1 + e.r, e.c);
      for (int i = 0; i < m_u; ++i) J(rh1 + i, alpha_of_h1_[i]);
      for (const auto& e : grad_x_) J(rh2x + e.r, e.c);
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oR + e);
      for (int e = 0; e < m_q; ++e) J(rh2x + e, oTh + e);
      for (const auto& e : grad_y_) J(rh2y + e.r, e.c);
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oR + e);
      for (int e = 0; e < m_q; ++e) J(rh2y + e, oTh + e);
      for (int e = 0; e < m_q; ++e) J(rh3x + e, qx_of_h3_[e]);
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oD + e);
      for (int e = 0; e < m_q; ++e) J(rh3x + e, oTh + e);
      for (int e = 0; e < m_q; ++e) J(rh3y + e, qy_of_h3_[e]);
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oD + e);
      for (int e = 0; e < m_q; ++e) J(rh3y + e, oTh + e);
      for (int e = 0; e < m_q; ++e) J(rhr + e, oR + e);
      for (int e = 0; e < m_q; ++e) J(rhd + e, oD + e);
      J(rha, alpha_of_tile_[0]);
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oR + e);
      for (int e = 0; e < m_q; ++e) J(rcomp + e, oD + e);
      for (int l = 0; l < n_link; ++l) J(rlink + l, link_copy_[l]);
      for (int l = 0; l < n_link; ++l) J(rlink + l, link_orig_[l]);

      auto H = [&](int r, int c) { hess_row_.push_back(r); hess_col_.push_back(c); };
      for (int i = 0; i < m_u; ++i) H(ou + i, ou + i);
      for (int e = 0; e < m_q; ++e) H(oTh + e, oR + e);
      for (int e = 0; e < m_q; ++e) H(oTh + e, oD + e);
      for (int e = 0; e < m_q; ++e) H(oTh + e, oTh + e);
      for (int e = 0; e < m_q; ++e) {
         H(oD + e, oR + e);
         if (fb_) { H(oR + e, oR + e); H(oD + e, oD + e); }
      }
      for (const auto& e : div_x_) H(alpha_of_h1_[e.r], e.c);
      for (const auto& e : div_y_) H(alpha_of_h1_[e.r], e.c);
      for (int k = 0; k < n_tiles_; ++k) H(alpha_of_tile_[k], alpha_of_tile_[k]);
      H(oa, oa);
   }
};
