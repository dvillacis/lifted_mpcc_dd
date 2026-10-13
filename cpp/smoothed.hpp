// smoothed.hpp — the standard smoothed bilevel approach, as a baseline for the
// MPCC (tv_learn --smoothed γ).
//
//   min_{α ≥ 0}  J(α) = c_S Σ_s ½‖u_s(α) − u_clean,s‖² + ½·ρ_α·‖α‖²
//   u_s(α) = argmin_u  E_s(u) = ½uᵀA_s u − F_sᵀu + Σ_e α_{b(e)} h_γ(|(K_ξ u)_e|)
//
// with the Huber function h_γ(s) = s²/(2γ) for s ≤ γ, s − γ/2 above: TV with its
// kink rounded off on a scale γ.  The data terms, the regularizers (TV, DTV), the
// blocks and the ridge ρ_α = 1e-4 are those of tv_learn (operators.hpp,
// learn_partition.hpp, tv_learn.hpp), so the two methods solve the same upper
// level, the MPCC with the exact TV lower level and this one with h_γ.
//
//   lower level  Newton on E_s: ∇E = A u − F + K_ξᵀ(w∘ψ(K_ξ u)), ψ(g) = g/max(γ, |g|),
//                Hessian H = A + K_ξᵀ D K_ξ, per cell D_e = (w_e/γ) I for |g_e| < γ,
//                (w_e/|g_e|)(I − ĝĝᵀ) above; sparse LDLᵀ (Eigen), Armijo backtracking
//                on E; stops when ‖∇E‖∞ ≤ newton_tol.  H is positive definite (A = I,
//                a mask with ε > 0, or CᵀC whose kernel K_ξ does not share).
//   gradient     the adjoint: H p = u − u_clean with the last factorization, then
//                ∂J/∂α_b = −c_S Σ_s Σ_{e∈b} ⟨(K_ξ p_s)_e, ψ((K_ξ u_s)_e)⟩ + ρ_α α_b.
//   upper level  projected L-BFGS on α ≥ 0 (memory 10, the two-loop recursion on the
//                free variables, Armijo along the projected path; a simple stand-in
//                for L-BFGS-B, adequate for one-sided bounds).
//
// Serial; images one after the other.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include <memory>

#include "comm.hpp"
#include "ui.hpp"
#include "learn_partition.hpp"
#ifdef DD_HAVE_MUMPS
#include "mumps_block.hpp"
#endif
#include "operators.hpp"

namespace smoothed {

using SpMat = Eigen::SparseMatrix<double>;
using Vec = Eigen::VectorXd;
using Trip = Eigen::Triplet<double>;

// A and K_ξ of one instance as sparse matrices (K_ξ: 2·m_q × m_u, the x rows then the y rows)
inline void build_operators(const Instance& in, const Blur* blur, SpMat& A, SpMat& K) {
   const int N = in.N, nc = N - 1, m_u = N * N, m_q = nc * nc;
   std::vector<Trip> t;
   if (!in.diag.empty()) {
      for (int i = 0; i < m_u; ++i) t.emplace_back(i, i, in.diag[i]);
   } else {   // C'C = G ⊗ G
      const int R = 2 * blur->w;
      for (int i1 = 0; i1 < N; ++i1)
         for (int i2 = 0; i2 < N; ++i2)
            for (int j1 = std::max(0, i1 - R); j1 <= std::min(N - 1, i1 + R); ++j1)
               for (int j2 = std::max(0, i2 - R); j2 <= std::min(N - 1, i2 + R); ++j2)
                  t.emplace_back(i1 * N + i2, j1 * N + j2, blur->g1(i1, j1) * blur->g1(i2, j2));
   }
   A.resize(m_u, m_u);
   A.setFromTriplets(t.begin(), t.end());
   t.clear();
   for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b) {
         const int e = a * nc + b, n00 = a * N + b, n01 = a * N + b + 1, n10 = (a + 1) * N + b;
         const double m11 = in.dtv() ? in.m11[e] : 1.0, m12 = in.dtv() ? in.m12[e] : 0.0,
                      m22 = in.dtv() ? in.m22[e] : 1.0;
         // x row: m11·(u01 − u00) + m12·(u10 − u00); y row: m12·(u01 − u00) + m22·(u10 − u00)
         t.emplace_back(e, n00, -(m11 + m12));
         t.emplace_back(e, n01, m11);
         if (m12 != 0.0) t.emplace_back(e, n10, m12);
         t.emplace_back(m_q + e, n00, -(m12 + m22));
         if (m12 != 0.0) t.emplace_back(m_q + e, n01, m12);
         t.emplace_back(m_q + e, n10, m22);
      }
   K.resize(2 * m_q, m_u);
   K.setFromTriplets(t.begin(), t.end());
}

// One image's Huber-TV lower level: Newton, and the adjoint with the last factorization.
struct LowerLevel {
   SpMat A, K, Kt;
   Vec F, u;               // u: the current solution (warm start for the next α)
   int m_q = 0;
   double gamma = 1e-3;
   // pd: the primal-dual semismooth Newton of De los Reyes–Schönlieb–Valkonen (2017,
   // (SSN-1)–(SSN-3)) and Hintermüller–Stadler (2006): the dual p (q = w∘p, the x parts
   // then the y parts) is its own unknown, max(γ, |g|) p − g = 0; its iterate enters H
   // projected onto |p| ≤ 1 and symmetrized,
   //   D_e = (w_e/m_e)(I − χ_e ½(p̄ĝᵀ + ĝp̄ᵀ)),  m_e = max(γ, |g_e|),  χ_e = [|g_e| ≥ γ],
   // positive semidefinite since |p̄| ≤ 1.  The direction solves H δu = −∇E as before
   // (Armijo on E), and p moves with the same step along
   //   δp_e = (I − χ_e p̄ ĝᵀ) δg_e / m_e − (p_e − ψ_e).
   // With p = ψ(u), D is the exact Hessian: so at convergence p is reset to ψ(u) and
   // the factorization left for the adjoint is the exact one.
   bool pd = false;
   Vec p;
   Eigen::SimplicialLDLT<SpMat> ldlt;
   bool factorized = false;
   // The solver of H: 0 Eigen's sparse LDLᵀ, 1 MUMPS (positive definite, analysed
   // once), 2 CG with the diagonal preconditioner, 3 CG with incomplete Cholesky
   // (2, 3: inexact Newton directions, see solve(); O(nnz(H)) memory)
   int solver = 0;
   bool mumps = false;   // solver == 1
#ifdef DD_HAVE_MUMPS
   std::unique_ptr<dd::MumpsBlock> mb;
#endif
   std::vector<double> hval;   // H's lower triangle, in the order given to MUMPS
   SpMat H;                    // kept for CG
   Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cgj;
   Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                            Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<int>>>
       cgi;
   long cg_iters = 0;          // CG iterations, accumulated
   int cg_max_iter = 5000;
   // H⁻¹ b with the current factorization; for CG, to the relative residual rtol
   Vec hsolve(const Vec& b, double rtol = 1e-10) {
#ifdef DD_HAVE_MUMPS
      if (mumps) {
         Vec x = b;
         mb->solve(x.data(), 1);
         return x;
      }
#endif
      if (solver == 2) {
         cgj.setTolerance(rtol);
         const Vec x = cgj.solve(b);
         cg_iters += cgj.iterations();
         return x;
      }
      if (solver == 3) {
         cgi.setTolerance(rtol);
         const Vec x = cgi.solve(b);
         cg_iters += cgi.iterations();
         return x;
      }
      return ldlt.solve(b);
   }
   bool analysed = false;                  // the ordering and symbolic factorization of H
   // H's assembly: its pattern, A's and a 3×3 block per cell (the cell's nodes n00, n01,
   // n10, which its rows of K_ξ touch), is fixed once; each Hessian then refills the
   // values in place (no products, no temporaries)
   std::vector<double> hbase;   // A on H's pattern
   std::vector<int> hpos;       // per cell, the positions of its 3×3 block in H's values
   std::vector<double> kloc;    // per cell, K_ξ's x row then its y row on (n00, n01, n10)
   void setup_hessian() {
      const int m_u = (int)A.rows(), N = (int)std::lround(std::sqrt((double)m_u)), nc = N - 1;
      auto nodes = [&](int e, int* n) {
         const int a = e / nc, b = e % nc;
         n[0] = a * N + b;
         n[1] = a * N + b + 1;
         n[2] = (a + 1) * N + b;
      };
      std::vector<Trip> t;
      t.reserve((size_t)A.nonZeros() + 9 * (size_t)m_q);
      for (int j = 0; j < A.outerSize(); ++j)
         for (SpMat::InnerIterator it(A, j); it; ++it) t.emplace_back((int)it.row(), j, 0.0);
      int n[3];
      for (int e = 0; e < m_q; ++e) {
         nodes(e, n);
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) t.emplace_back(n[i], n[j], 0.0);
      }
      H.resize(m_u, m_u);
      H.setFromTriplets(t.begin(), t.end());
      H.makeCompressed();
      std::vector<Trip>().swap(t);
      const double* v0 = H.valuePtr();
      hbase.assign((size_t)H.nonZeros(), 0.0);
      for (int j = 0; j < A.outerSize(); ++j)
         for (SpMat::InnerIterator it(A, j); it; ++it)
            hbase[&H.coeffRef((int)it.row(), j) - v0] += it.value();
      hpos.resize(9 * (size_t)m_q);
      kloc.assign(6 * (size_t)m_q, 0.0);
      for (int e = 0; e < m_q; ++e) {
         nodes(e, n);
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) hpos[9 * (size_t)e + 3 * i + j] = (int)(&H.coeffRef(n[i], n[j]) - v0);
         for (int r = 0; r < 2; ++r)   // K_ξ's rows e and m_q + e: the columns of K_ξᵀ
            for (SpMat::InnerIterator it(Kt, r * m_q + e); it; ++it)
               for (int k = 0; k < 3; ++k)
                  if (it.row() == n[k]) kloc[6 * (size_t)e + 3 * r + k] = it.value();
      }
   }
   double t_factor = 0.0, t_solve = 0.0, t_energy = 0.0;   // seconds, accumulated
   int max_newton = 0;                                      // most iterations of one solve
   int analyses = 0;                                        // symbolic analyses of H
   using Clock = std::chrono::steady_clock;
   static double since(Clock::time_point t) {
      return std::chrono::duration<double>(Clock::now() - t).count();
   }

   double energy(const Vec& v, const std::vector<double>& w) const {
      const Vec g = K * v;
      double hsum = 0.0;
      for (int e = 0; e < m_q; ++e) {
         const double s = std::hypot(g[e], g[m_q + e]);
         hsum += w[e] * (s <= gamma ? s * s / (2.0 * gamma) : s - 0.5 * gamma);
      }
      return 0.5 * v.dot(A * v) - F.dot(v) + hsum;
   }
   // ∇E and, if hess, the factorization of H at v
   Vec gradient(const Vec& v, const std::vector<double>& w, bool hess) {
      const Vec g = K * v;
      Vec psi(2 * m_q);
      const auto th = Clock::now();
      double* hv = nullptr;
      if (hess) {   // H = A + Σ_e K_eᵀ D_e K_e, refilled in place
         if (hpos.empty()) setup_hessian();
         hv = H.valuePtr();
         std::copy(hbase.begin(), hbase.end(), hv);
      }
      for (int e = 0; e < m_q; ++e) {
         const double gx = g[e], gy = g[m_q + e], s = std::hypot(gx, gy), den = std::max(gamma, s);
         psi[e] = gx / den;
         psi[m_q + e] = gy / den;
         if (!hess) continue;
         double d11, d12, d22;
         if (pd) {   // the primal-dual D_e, with the dual projected onto |p| ≤ 1
            const double c = w[e] / den;
            d11 = d22 = c;
            d12 = 0.0;
            if (s >= gamma) {
               double px = p[e], py = p[m_q + e];
               const double pn = std::hypot(px, py);
               if (pn > 1.0) { px /= pn; py /= pn; }
               const double hx = gx / s, hy = gy / s;
               d11 = c * (1.0 - px * hx);
               d12 = -c * 0.5 * (px * hy + py * hx);
               d22 = c * (1.0 - py * hy);
            }
         } else if (s < gamma) {
            d11 = d22 = w[e] / gamma;
            d12 = 0.0;
         } else {
            const double c = w[e] / s, hx = gx / s, hy = gy / s;
            d11 = c * (1.0 - hx * hx);
            d12 = -c * hx * hy;
            d22 = c * (1.0 - hy * hy);
         }
         const double* kx = &kloc[6 * (size_t)e];
         const double* ky = kx + 3;
         const int* pos = &hpos[9 * (size_t)e];
         for (int j = 0; j < 3; ++j) {
            const double cx = d11 * kx[j] + d12 * ky[j], cy = d12 * kx[j] + d22 * ky[j];
            for (int i = 0; i < 3; ++i) hv[pos[3 * i + j]] += kx[i] * cx + ky[i] * cy;
         }
      }
      if (hess) t_factor += since(th);
      Vec wpsi(2 * m_q);
      for (int e = 0; e < m_q; ++e) {
         wpsi[e] = w[e] * psi[e];
         wpsi[m_q + e] = w[e] * psi[m_q + e];
      }
      if (hess) {
         const auto t0 = Clock::now();
         // H's pattern does not change: the ordering and symbolic analysis once, then
         // numeric factorizations
         if (solver >= 2) {   // CG: the preconditioner only
            if (solver == 2) {
               cgj.setMaxIterations(cg_max_iter);
               cgj.compute(H);
               factorized = cgj.info() == Eigen::Success;
            } else {
               cgi.setMaxIterations(cg_max_iter);
               cgi.compute(H);
               factorized = cgi.info() == Eigen::Success;
            }
            t_factor += since(t0);
            return A * v - F + Kt * wpsi;
         }
         const bool fresh = !analysed;
         if (fresh) {
            analysed = true;
            ++analyses;
         }
#ifdef DD_HAVE_MUMPS
         if (mumps) {
            hval.clear();
            std::vector<int> r, c;
            for (int j = 0; j < H.outerSize(); ++j)
               for (SpMat::InnerIterator it(H, j); it; ++it)
                  if (it.row() >= j) {
                     hval.push_back(it.value());
                     if (fresh) { r.push_back((int)it.row()); c.push_back(j); }
                  }
            if (fresh) {
               mb = std::make_unique<dd::MumpsBlock>(1);
               mb->analyze((int)H.rows(), 0, r, c);
            }
            factorized = mb->factorize(hval.data());
            t_factor += since(t0);
            return A * v - F + Kt * wpsi;
         }
#endif
         if (fresh) ldlt.analyzePattern(H);
         ldlt.factorize(H);
         factorized = ldlt.info() == Eigen::Success;
         t_factor += since(t0);
      }
      return A * v - F + Kt * wpsi;
   }
   // Newton from u: the iterations in iters; false if ‖∇E‖∞ ≤ tol was not reached.
   // Armijo backtracking on E, except once the predicted decrease −∇Eᵀd is below the
   // precision of E itself: there the full step is taken (Newton's local region; an
   // energy test could not see the progress any more).
   // the consistent dual p = ψ(u) (pd: the start of the dual, and its reset at the end)
   void dual_from_u() {
      const Vec g = K * u;
      p.resize(2 * m_q);
      for (int e = 0; e < m_q; ++e) {
         const double den = std::max(gamma, std::hypot(g[e], g[m_q + e]));
         p[e] = g[e] / den;
         p[m_q + e] = g[m_q + e] / den;
      }
   }
   // pd: p ← p + step·δp along the direction dir of u (from the iterate before the step)
   void dual_step(const Vec& dir, double step) {
      const Vec g = K * u, dg = K * dir;
      for (int e = 0; e < m_q; ++e) {
         const double gx = g[e], gy = g[m_q + e], s = std::hypot(gx, gy), m = std::max(gamma, s);
         double px = p[e], py = p[m_q + e];
         double ax = dg[e], ay = dg[m_q + e];   // (I − χ p̄ĝᵀ) δg
         if (s >= gamma) {
            const double pn = std::hypot(px, py), sc = pn > 1.0 ? 1.0 / pn : 1.0;
            const double hd = (gx * ax + gy * ay) / s;
            ax -= sc * px * hd;
            ay -= sc * py * hd;
         }
         p[e] += step * (ax / m - (px - gx / m));
         p[m_q + e] += step * (ay / m - (py - gy / m));
      }
   }
   bool solve(const std::vector<double>& w, double tol, int max_it, int& iters) {
      if (pd && p.size() != 2 * (size_t)m_q) dual_from_u();
      for (iters = 0; iters < max_it; ++iters) {
         // pd: the residual first; converged, the exact Hessian (p = ψ(u)) for the adjoint
         const Vec gr = gradient(u, w, !pd);
         if (gr.lpNorm<Eigen::Infinity>() <= tol) {
            if (pd) {
               dual_from_u();
               gradient(u, w, true);
            }
            return true;
         }
         if (pd) gradient(u, w, true);
         if (!factorized) return false;
         auto t0 = Clock::now();
         // CG: an inexact Newton direction, relative residual η = min(0.1, √‖∇E‖)
         // (superlinear convergence; loose far from the solution)
         const double eta = solver >= 2 ? std::max(1e-12, std::min(0.1, std::sqrt(gr.norm()))) : 0.0;
         const Vec dir = -hsolve(gr, eta);
         t_solve += since(t0);
         t0 = Clock::now();
         const double e0 = energy(u, w), slope = gr.dot(dir);
         double step = 1.0;
         if (-slope > 1e-13 * (1.0 + std::abs(e0)))
            for (int k = 0; k < 40; ++k, step *= 0.5)
               if (energy(u + step * dir, w) <= e0 + 1e-4 * step * slope) break;
         t_energy += since(t0);
         if (pd) dual_step(dir, step);
         u += step * dir;
         max_newton = std::max(max_newton, iters + 1);
      }
      if (pd) dual_from_u();
      const Vec gr = gradient(u, w, true);
      return gr.lpNorm<Eigen::Infinity>() <= tol;
   }
};

struct Result {
   std::vector<double> alpha;   // per block
   double J = 0.0;
   int iters = 0, evals = 0, lower_solves = 0;
   long newton_iters = 0;
   int lower_failures = 0;
   double wall = 0.0;
   bool converged = false;
   const char* stop = "iteration limit";   // why the iteration stopped
   // per image, the lower level at alpha: u and the Huber dual q = K_ξu / max(γ, |K_ξu|)
   // (A u − F + K_ξᵀ(α∘q) = 0 to Newton's tolerance, |q| ≤ 1): the MPCC's start
   std::vector<std::vector<double>> u, qx, qy;
   double newton_residual = 0.0;   // max ‖∇E‖∞ there
};

// J and ∇J over the training images; the lower levels keep their last solution as
// the warm start of the next evaluation.
// dist: the images are split over the MPI ranks in contiguous blocks (as tiles,
// Comm::first_tile); each rank solves its own lower levels, the parts of J and ∇J
// are gathered per image and summed in image order, so every rank holds the same
// J and ∇J, bit for bit the same for any number of ranks.  Otherwise this rank
// solves every image (no communication).
struct Objective {
   const std::vector<Instance>& data;
   std::vector<LowerLevel> ll;   // built only for this rank's images
   std::vector<int> blk;
   int m_q = 0, nb = 0, S = 0;
   bool dist = false;
   int s0 = 0, s1 = 0;                  // this rank's images [s0, s1)
   std::vector<int> counts, displs;     // per rank: its images (for the gathers)
   double gamma, loss_scale, newton_tol, rho = 1e-4;   // ρ_α: tv_learn.hpp's kRegAlpha
   int evals = 0, lower_solves = 0, lower_failures = 0;
   long newton_iters = 0;
   double t_adjoint = 0.0;

   Objective(const std::vector<Instance>& data_, const Blur* blur, const LearnPartition& part,
             double gamma_, double loss_scale_, double newton_tol_, bool dist_ = false)
       : data(data_), ll(data_.size()), gamma(gamma_), loss_scale(loss_scale_),
         newton_tol(newton_tol_) {   // (LowerLevel is not movable: built in place)
      const int nc = part.N - 1;
      S = (int)data.size();
      m_q = nc * nc;
      nb = part.n_blocks;
      dist = dist_ && dd::Comm::size() > 1;
      s0 = dist ? dd::Comm::first_tile(dd::Comm::rank(), S) : 0;
      s1 = dist ? dd::Comm::first_tile(dd::Comm::rank() + 1, S) : S;
      if (dist)
         for (int r = 0; r < dd::Comm::size(); ++r) {
            counts.push_back(dd::Comm::first_tile(r + 1, S) - dd::Comm::first_tile(r, S));
            displs.push_back(dd::Comm::first_tile(r, S));
         }
      blk.resize(m_q);
      for (int e = 0; e < m_q; ++e) blk[e] = part.block(e);
      for (int s = s0; s < s1; ++s) {
         build_operators(data[s], blur, ll[s].A, ll[s].K);
         ll[s].Kt = ll[s].K.transpose();
         ll[s].F = Eigen::Map<const Vec>(data[s].rhs.data(), data[s].rhs.size());
         ll[s].u = Eigen::Map<const Vec>(data[s].f.data(), data[s].f.size());   // start: the data
         ll[s].m_q = m_q;
         ll[s].gamma = gamma;
      }
   }
   // A new Huber parameter (the continuation); the lower levels keep their solutions
   // as the warm start, and H's pattern does not depend on γ
   void set_gamma(double g) {
      gamma = g;
      for (int s = s0; s < s1; ++s) ll[s].gamma = g;
   }
   // Gather per-image blocks of k values (this rank's in place) to every rank.
   void gather(std::vector<double>& all, int k) const {
      if (!dist) return;
      std::vector<int> c(counts.size()), d(counts.size());
      for (size_t r = 0; r < counts.size(); ++r) { c[r] = counts[r] * k; d[r] = displs[r] * k; }
      dd::Comm::allgatherv(all, c, d);
   }
   double operator()(const std::vector<double>& a, std::vector<double>& grad) {
      ++evals;
      std::vector<double> w(m_q);
      for (int e = 0; e < m_q; ++e) w[e] = a[blk[e]];
      // per image: its part of J, then of ∇J
      const int k = 1 + nb;
      std::vector<double> parts((size_t)S * k, 0.0);
      for (int s = s0; s < s1; ++s) {
         LowerLevel& L = ll[s];
         double* P = &parts[(size_t)s * k];
         int it = 0;
         if (!L.solve(w, newton_tol, 200, it)) ++lower_failures;
         ++lower_solves;
         newton_iters += it;
         const Vec clean = Eigen::Map<const Vec>(data[s].clean.data(), data[s].clean.size());
         const Vec res = L.u - clean;
         P[0] = loss_scale * 0.5 * res.squaredNorm();
         if (!L.factorized) continue;
         const auto t0 = LowerLevel::Clock::now();
         const Vec p = L.hsolve(res);   // H p = u − u_clean
         t_adjoint += LowerLevel::since(t0);
         const Vec Kp = L.K * p, g = L.K * L.u;
         for (int e = 0; e < m_q; ++e) {
            const double den = std::max(gamma, std::hypot(g[e], g[m_q + e]));
            P[1 + blk[e]] -= loss_scale * (Kp[e] * g[e] + Kp[m_q + e] * g[m_q + e]) / den;
         }
      }
      gather(parts, k);
      double J = 0.0;
      for (int b = 0; b < nb; ++b) J += 0.5 * rho * a[b] * a[b];
      grad.assign(nb, 0.0);
      for (int b = 0; b < nb; ++b) grad[b] = rho * a[b];
      for (int s = 0; s < S; ++s) {   // in image order, whichever rank solved it
         const double* P = &parts[(size_t)s * k];
         J += P[0];
         for (int b = 0; b < nb; ++b) grad[b] += P[1 + b];
      }
      return J;
   }
};

// The smoothed bilevel problem over the training images, from the uniform weight alpha0.
// Continuation in γ (gamma0 > gamma; Hintermüller–Wu 2015, Alg. 6.2): start at gamma0;
// once |proj grad| <= gamma_sigma·(γ/gamma0)·|proj grad|₀ (or J stalls, or the line
// search fails) set γ ← max(gamma_rate·γ, gamma), forget the L-BFGS memory and go on
// from the same α and the same lower-level solutions.  At γ = gamma the usual tests.
// max_iter counts the iterations over all levels (each change of γ takes one).
inline Result solve(const std::vector<Instance>& data, const Blur* blur, const LearnPartition& part,
                    double gamma, double alpha0, double loss_scale, bool print,
                    double tol = 1e-6, int max_iter = 100, double newton_tol = 1e-10,
                    double ftol = 1e-9, int solver = 0, bool dist = false,
                    double gamma0 = 0.0, double gamma_rate = 0.1, double gamma_sigma = 0.1,
                    bool pd_newton = false) {
   const auto t0 = std::chrono::steady_clock::now();
   const int S = (int)data.size(), nb = part.n_blocks;
   const bool cont = gamma0 > gamma;
   Objective evaluate(data, blur, part, cont ? gamma0 : gamma, loss_scale, newton_tol, dist);
   // sums and maxima over the ranks (identity without dist); every rank calls them
   auto lsum = [&](long v) { return evaluate.dist ? dd::Comm::sum(v) : v; };
   auto dmax = [&](double v) { return evaluate.dist ? dd::Comm::max(v) : v; };
   for (LowerLevel& L : evaluate.ll) {
      L.solver = solver;
      L.mumps = solver == 1;
      L.pd = pd_newton;
   }
   Result R;
   auto project = [](std::vector<double>& a) { for (double& v : a) v = std::max(v, 0.0); };
   // projected gradient: zero where a bound is active and the gradient points out
   auto proj_grad = [](const std::vector<double>& a, const std::vector<double>& g) {
      std::vector<double> pg(g);
      for (size_t i = 0; i < a.size(); ++i)
         if (a[i] <= 0.0 && g[i] > 0.0) pg[i] = 0.0;
      return pg;
   };
   auto norm_inf = [](const std::vector<double>& v) {
      double m = 0.0;
      for (double x : v) m = std::max(m, std::abs(x));
      return m;
   };
   std::vector<double> a(nb, alpha0), g;
   double J = evaluate(a, g);
   const double pg0 = std::max(norm_inf(proj_grad(a, g)), 1e-300);
   std::deque<std::vector<double>> Ss, Ys;
   const size_t mem = 10;
   if (print)
   {
      ui::kv("problem", "%d weight%s, %d image%s of %dx%d, Huber gamma %.0e", nb, nb > 1 ? "s" : "",
             S, S > 1 ? "s" : "", data[0].N, data[0].N, gamma);
      if (cont)
         ui::kv("continuation", "gamma from %.0e, x%g once |proj grad| <= %g (gamma/%.0e) x the"
                " first", gamma0, gamma_rate, gamma_sigma, gamma0);
      ui::kv("L-BFGS", "projected, to |proj grad| <= %.0e x the first, or J falling by less than"
             " %.0e (relative) over 5 iterations; at most %d iterations", tol, ftol, max_iter);
      ui::kv("lower levels", "%sNewton to |grad E| <= %.0e, %s",
             pd_newton ? "primal-dual semismooth " : "", newton_tol,
             solver == 0 ? "Eigen LDL'" : solver == 1 ? "MUMPS"
             : solver == 2 ? "CG with the diagonal preconditioner"
                           : "CG with incomplete Cholesky");
      if (evaluate.dist) ui::kv("ranks", "the images split over %d ranks", dd::Comm::size());
      std::printf(" it            J      |proj grad|   step  evals  alpha mean       min       max"
                  "   Newton its (max)  time s\n");
   }
   double last_step = 0.0;
   std::deque<double> Jhist;   // J at the last iterations (the stall test)
   int last_ls = 0;
   long newton_mark = 0;
   for (R.iters = 0; R.iters < max_iter; ++R.iters) {
      const std::vector<double> pg = proj_grad(a, g);
      int mx = 0;
      for (const LowerLevel& L : evaluate.ll) mx = std::max(mx, L.max_newton);
      mx = (int)dmax(mx);
      const long nt = lsum(evaluate.newton_iters);
      if (print) {
         double am = 0.0, amin = 1e300, amax = -1e300;
         for (double v : a) { am += v / nb; amin = std::min(amin, v); amax = std::max(amax, v); }
         const double el =
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         if (R.iters == 0)
            std::printf("%3d  %.8e  %10.3e      -  %5d   %.6f  %.6f  %.6f   %8ld (%3d)  %6.2f\n",
                        R.iters, J, norm_inf(pg), evaluate.evals, am, amin, amax, nt, mx, el);
         else
            std::printf("%3d  %.8e  %10.3e  %.0e  %5d   %.6f  %.6f  %.6f   %8ld (%3d)  %6.2f\n",
                        R.iters, J, norm_inf(pg), last_step, last_ls, am, amin, amax,
                        nt - newton_mark, mx, el);
         std::fflush(stdout);
      }
      newton_mark = nt;
      for (LowerLevel& L : evaluate.ll) L.max_newton = 0;
      // the next γ: the same α, a new J and gradient, a fresh L-BFGS memory and stall test
      auto next_gamma = [&](const char* why) {
         const double gn = std::max(gamma_rate * evaluate.gamma, gamma);
         if (print) std::printf("     gamma %.0e -> %.0e (%s)\n", evaluate.gamma, gn, why);
         evaluate.set_gamma(gn);
         J = evaluate(a, g);
         Ss.clear();
         Ys.clear();
         Jhist.clear();
         last_step = 0.0;
         last_ls = 1;
      };
      const bool coarse = evaluate.gamma > gamma;   // a continuation level before the last
      if (coarse && norm_inf(pg) <= gamma_sigma * (evaluate.gamma / gamma0) * pg0) {
         next_gamma("projected gradient");
         continue;
      }
      if (norm_inf(pg) <= tol * pg0) {
         R.converged = true;
         R.stop = "projected gradient";
         break;
      }
      // J has stopped decreasing: less than ftol·|J| over the last 5 iterations
      // (the projected gradient can stall above tol where cells cross |K_ξu| = γ)
      Jhist.push_back(J);
      if (Jhist.size() > 6) Jhist.pop_front();
      if (ftol > 0.0 && Jhist.size() == 6 &&
          Jhist.front() - J <= ftol * std::max(1.0, std::abs(J))) {
         if (coarse) {
            next_gamma("J stalled");
            continue;
         }
         R.converged = true;
         R.stop = "J stalled";
         break;
      }
      // free variables: not at a bound with the gradient pointing out
      std::vector<char> fr(nb);
      for (int i = 0; i < nb; ++i) fr[i] = !(a[i] <= 0.0 && g[i] > 0.0);
      // two-loop recursion on the free variables
      std::vector<double> q(nb);
      for (int i = 0; i < nb; ++i) q[i] = fr[i] ? g[i] : 0.0;
      const int k = (int)Ss.size();
      std::vector<double> al(k), rh(k);
      auto dotf = [&](const std::vector<double>& x, const std::vector<double>& y) {
         double s = 0.0;
         for (int i = 0; i < nb; ++i) if (fr[i]) s += x[i] * y[i];
         return s;
      };
      for (int j = k - 1; j >= 0; --j) {
         const double sy = dotf(Ss[j], Ys[j]);
         rh[j] = sy > 0.0 ? 1.0 / sy : 0.0;
         al[j] = rh[j] * dotf(Ss[j], q);
         for (int i = 0; i < nb; ++i) if (fr[i]) q[i] -= al[j] * Ys[j][i];
      }
      double scale = 1.0;
      if (k > 0) {
         const double yy = dotf(Ys[k - 1], Ys[k - 1]);
         if (yy > 0.0) scale = dotf(Ss[k - 1], Ys[k - 1]) / yy;
      } else {   // first step: move α by about 10% of its size
         double an = std::max(norm_inf(a), 1e-3), gn = std::max(norm_inf(q), 1e-300);
         scale = 0.1 * an / gn;
      }
      for (double& v : q) v *= scale;
      for (int j = 0; j < k; ++j) {
         const double b = rh[j] * dotf(Ys[j], q);
         for (int i = 0; i < nb; ++i) if (fr[i]) q[i] += Ss[j][i] * (al[j] - b);
      }
      std::vector<double> d(nb);
      for (int i = 0; i < nb; ++i) d[i] = fr[i] ? -q[i] : 0.0;
      double gd = 0.0;
      for (int i = 0; i < nb; ++i) gd += g[i] * d[i];
      if (gd >= 0.0) {   // not a descent direction: steepest descent, and forget the memory
         for (int i = 0; i < nb; ++i) d[i] = -pg[i] * scale;
         Ss.clear();
         Ys.clear();
      }
      // Armijo along the projected path
      double step = 1.0, Jn = 0.0;
      std::vector<double> an(nb), gn;
      bool ok = false;
      last_ls = 0;
      for (int ls = 0; ls < 30; ++ls, step *= 0.5) {
         for (int i = 0; i < nb; ++i) an[i] = a[i] + step * d[i];
         project(an);
         double dec = 0.0;
         for (int i = 0; i < nb; ++i) dec += g[i] * (an[i] - a[i]);
         Jn = evaluate(an, gn);
         ++last_ls;
         if (Jn <= J + 1e-4 * dec) { ok = true; break; }
      }
      last_step = step;
      if (!ok && coarse) {
         next_gamma("line search failed");
         continue;
      }
      if (!ok) {   // no decrease found: at the precision of J if the gradient is small
         R.converged = norm_inf(pg) <= 1e-3 * pg0;
         R.stop = R.converged ? "line search at the precision of J" : "line search failed";
         break;
      }
      std::vector<double> sv(nb), yv(nb);
      double sy = 0.0;
      for (int i = 0; i < nb; ++i) {
         sv[i] = an[i] - a[i];
         yv[i] = gn[i] - g[i];
         sy += sv[i] * yv[i];
      }
      if (sy > 1e-16) {
         Ss.push_back(sv);
         Ys.push_back(yv);
         if (Ss.size() > mem) { Ss.pop_front(); Ys.pop_front(); }
      }
      a = an;
      g = gn;
      J = Jn;
   }
   R.alpha = a;
   R.J = J;
   // the lower levels at the final α (the last evaluation may have been a rejected trial)
   {
      const int m_q = evaluate.m_q;
      std::vector<double> w(m_q);
      for (int e = 0; e < m_q; ++e) w[e] = a[evaluate.blk[e]];
      const int m_u = data[0].N * data[0].N, k = m_u + 2 * m_q;
      std::vector<double> flds((size_t)S * k, 0.0);   // per image: u, qx, qy
      for (int s = evaluate.s0; s < evaluate.s1; ++s) {
         LowerLevel& L = evaluate.ll[s];
         int it = 0;
         if (!L.solve(w, newton_tol, 200, it)) ++evaluate.lower_failures;
         evaluate.newton_iters += it;
         R.newton_residual = std::max(R.newton_residual,
                                      L.gradient(L.u, w, false).lpNorm<Eigen::Infinity>());
         const Vec g = L.K * L.u;
         double* F = &flds[(size_t)s * k];
         std::copy(L.u.data(), L.u.data() + m_u, F);
         for (int e = 0; e < m_q; ++e) {
            const double den = std::max(evaluate.gamma, std::hypot(g[e], g[m_q + e]));
            F[m_u + e] = g[e] / den;
            F[m_u + m_q + e] = g[m_q + e] / den;
         }
      }
      evaluate.gather(flds, k);
      R.newton_residual = dmax(R.newton_residual);
      R.u.resize(S);
      R.qx.resize(S);
      R.qy.resize(S);
      for (int s = 0; s < S; ++s) {
         const double* F = &flds[(size_t)s * k];
         R.u[s].assign(F, F + m_u);
         R.qx[s].assign(F + m_u, F + m_u + m_q);
         R.qy[s].assign(F + m_u + m_q, F + k);
      }
   }
   R.evals = evaluate.evals;
   R.lower_solves = (int)lsum(evaluate.lower_solves);
   R.lower_failures = (int)lsum(evaluate.lower_failures);
   R.newton_iters = lsum(evaluate.newton_iters);
   R.wall = dmax(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
   // time split: this rank's sums, then the slowest rank's
   double tf = 0.0, ts = 0.0, te = 0.0;
   long an = 0;
   for (const LowerLevel& L : evaluate.ll) {
      tf += L.t_factor; ts += L.t_solve; te += L.t_energy; an += L.analyses;
   }
   tf = dmax(tf);
   ts = dmax(ts);
   te = dmax(te);
   const double tadj = dmax(evaluate.t_adjoint);
   an = lsum(an);
   long cgit = 0;
   for (const LowerLevel& L : evaluate.ll) cgit += L.cg_iters;
   cgit = lsum(cgit);
   if (print) {
      ui::kv("evaluations", "%d of J: %d lower-level solves, %ld Newton iterations%s",
             R.evals, R.lower_solves, R.newton_iters, solver >= 2 ? "" : "");
      if (solver >= 2) ui::kv("CG", "%ld iterations (Newton directions and adjoints)", cgit);
      ui::kv("lower levels", "|grad E| %.1e at the end", R.newton_residual);
      if (R.lower_failures > 0)
         ui::warn("%d lower-level solves did not converge", R.lower_failures);
      ui::kv("time", "%.2f s: factorizations %.2f s (%ld symbolic analyses), Newton solves %.2f s,"
             " line-search energies %.2f s, adjoints %.2f s, the rest %.2f s%s", R.wall, tf, an, ts,
             te, tadj, R.wall - tf - ts - te - tadj, evaluate.dist ? " (slowest rank)" : "");
   }
   return R;
}

}  // namespace smoothed
