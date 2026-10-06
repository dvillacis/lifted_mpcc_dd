// restoration.hpp — the feasibility problem of IPOPT's restoration phase
// (Wächter & Biegler 2006, Sec. 3.3), as an NLP that ipm.hpp solves with the
// same interior point method and the same domain decomposition:
//
//   min  ρ Σ_r (p_r + n_r) + ζ/2 Σ_i D_i² (x_i − x̄_i)²
//   s.t. g_l ≤ g(x) − p + n ≤ g_u,   x_l ≤ x ≤ x_u,   p, n ≥ 0
//
// with x̄ the iterate where the line search failed, D_i = min(1, 1/|x̄_i|),
// ζ = √μ̄ (fixed for the whole phase).  Variables [x | p | n]: p_r and n_r
// belong to the tile of row r, so the tile structure and the border are those
// of the original problem.  The Hessian is ζD² plus the constraints' part of
// the original one (σ = 0).
//
// p and n are ELASTIC (nlp.hpp): the IPM eliminates them from its Newton
// systems, and ζD² is reported as hess_diag(), so the KKT systems of the phase
// have exactly the pattern of the original problem's, and the IPM reuses the
// original problem's decomposition (and MUMPS instances) for them.
//
// Like the original problem it is local (nlp.hpp): this rank's tiles and the
// consensus variables, with p_r, n_r for the local rows.
//
// The phase ends as soon as the iteration() hook sees a point that the
// original problem accepts (the caller's `accept`, which gets the IPM's
// [x | p | n | s] vector); the IPM then stops with status 2.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "nlp.hpp"

class RestorationNLP : public NLP {
public:
   RestorationNLP(NLP& orig, double rho, double zeta, std::vector<double> xbar,
                  std::function<bool(const double*)> accept)
       : o_(orig), rho_(rho), zeta_(zeta), xbar_(std::move(xbar)), accept_(std::move(accept)),
         n_(orig.num_vars()), m_(orig.num_rows()), sum_(orig.n_tiles()) {
      d2_.resize(n_);
      for (int i = 0; i < n_; ++i) {
         const double d = std::min(1.0, 1.0 / std::max(1e-300, std::abs(xbar_[i])));
         d2_[i] = d * d;
      }
      var_tile_ = orig.var_tile();
      for (int k = 0; k < 2; ++k)
         var_tile_.insert(var_tile_.end(), orig.row_tile().begin(), orig.row_tile().end());
   }

   bool accepted() const { return accepted_; }

   int num_vars() const override { return n_ + 2 * m_; }
   int num_rows() const override { return m_; }
   void bounds(double* xl, double* xu, double* gl, double* gu) const override {
      o_.bounds(xl, xu, gl, gu);
      for (int i = n_; i < n_ + 2 * m_; ++i) {
         xl[i] = 0.0;
         xu[i] = 1e20;
      }
   }
   void start(double* x) const override {   // unused: ipm.hpp is given the start point
      std::copy(xbar_.begin(), xbar_.end(), x);
      std::fill(x + n_, x + n_ + 2 * m_, 1.0);
   }

   double f(const double* x) override {
      for (int r = 0; r < 2 * m_; ++r) sum_.add(var_tile_[n_ + r], rho_ * x[n_ + r]);
      for (int i = 0; i < n_; ++i)
         sum_.add(var_tile_[i], 0.5 * zeta_ * d2_[i] * (x[i] - xbar_[i]) * (x[i] - xbar_[i]));
      return sum_.total();
   }
   void grad_f(const double* x, double* grad) override {
      for (int i = 0; i < n_; ++i) grad[i] = zeta_ * d2_[i] * (x[i] - xbar_[i]);
      for (int r = 0; r < 2 * m_; ++r) grad[n_ + r] = rho_;
   }
   void g(const double* x, double* g) override {
      o_.g(x, g);
      for (int r = 0; r < m_; ++r) g[r] += -x[n_ + r] + x[n_ + m_ + r];
   }
   // the original Jacobian, then p (−1) and n (+1) of every row
   void jac_structure(std::vector<int>& row, std::vector<int>& col) const override {
      o_.jac_structure(row, col);
      nnz_j_ = row.size();
      for (int r = 0; r < m_; ++r) { row.push_back(r); col.push_back(n_ + r); }
      for (int r = 0; r < m_; ++r) { row.push_back(r); col.push_back(n_ + m_ + r); }
   }
   void jac_values(const double* x, double* val) override {
      o_.jac_values(x, val);
      for (int r = 0; r < m_; ++r) val[nnz_j_ + r] = -1.0;
      for (int r = 0; r < m_; ++r) val[nnz_j_ + m_ + r] = 1.0;
   }
   // the original Hessian's pattern; its constraint part (σ = 0) ...
   void hess_structure(std::vector<int>& row, std::vector<int>& col) const override {
      o_.hess_structure(row, col);
   }
   void hess_values(const double* x, double, const double* lam, double* val) override {
      o_.hess_values(x, 0.0, lam, val);
   }
   // ... and the proximity term ζD² on the diagonal
   bool hess_diag(const double*, double sigma, double* d) override {
      for (int i = 0; i < n_; ++i) d[i] = sigma * zeta_ * d2_[i];
      std::fill(d + n_, d + n_ + 2 * m_, 0.0);
      return true;
   }
   int n_elastic() const override { return 2 * m_; }

   int n_tiles() const override { return o_.n_tiles(); }
   const std::vector<int>& var_tile() const override { return var_tile_; }
   const std::vector<int>& row_tile() const override { return o_.row_tile(); }

   bool iteration(const Iterate& it) override {
      if (it.iter > 0 && accept_(it.x)) {
         accepted_ = true;
         return false;
      }
      return true;
   }

private:
   NLP& o_;
   double rho_, zeta_;
   std::vector<double> xbar_, d2_;
   std::function<bool(const double*)> accept_;
   int n_, m_;
   dd::TileSum sum_;
   bool accepted_ = false;
   std::vector<int> var_tile_;
   mutable size_t nnz_j_ = 0;   // entries of the original Jacobian
};
