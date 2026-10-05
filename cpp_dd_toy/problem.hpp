// problem.hpp — the toy NLP, written in the structured form (1) of Lueg et al.
//
//   min  ½h² Σ_g (u_g − u_d,g)²  +  ½αh² Σ_g f_g²
//   s.t. 4u_g − Σ_{nb∈N(g)} u_nb + h²(κ u_g³ − f_g) = 0      every interior node g
//        −fmax ≤ f_g ≤ fmax
//
// on an N×N grid of interior nodes of the unit square, h = 1/(N+1), u = 0 on the
// boundary (the 5-point Laplacian, each row scaled by h²).  The u³ term makes the
// constraint nonlinear, and the Lagrangian Hessian 6κh²λ_g u_g has no fixed
// sign, so the problem is nonconvex and IPOPT's inertia correction can fire.
//
// ---------------------------------------------------------------------------
//  PARTITIONING (paper eq. 1)
// ---------------------------------------------------------------------------
// The grid is cut into P×P tiles.  Node g, its row and its control f_g belong
// to the tile containing g.  A row references u at its 4 neighbours, so u at a
// node next to a cut is used by two (or, near a corner, three) tiles.  Each
// such u becomes a COMPLICATING variable y_g (paper: y), and every tile that
// uses it gets its own LOCAL COPY (paper: y_k) tied to it by a linking row
//
//        copy − y_g = 0                                        (paper eq. 1d)
//
// owned by the copy's tile.  The objective term of u_g sits on the home tile's
// copy, so y appears in the linking rows and nowhere else: its Hessian block is
// zero, exactly as in the paper (IPOPT adds δ_w there; see schur_dd.hpp).
// Every other variable is LOCAL to one tile (paper: x_k).  With P = 1 there are
// no copies and this is the plain (monolithic) problem.
//
// ---------------------------------------------------------------------------
//  WHO OWNS WHICH KKT ROW
// ---------------------------------------------------------------------------
// All constraints are equalities and f has only variable bounds, so IPOPT's
// augmented KKT system has the unknowns  [ x | λ ]  in exactly the TNLP's
// order (no slacks: those exist only for inequality rows; no fixed variables:
// those would be removed and shift the x indices).  kkt_owner() maps each KKT
// unknown to its tile, or to −1 for the complicating variables y.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "IpTNLP.hpp"

class SemilinearDD : public Ipopt::TNLP {
   using Index = Ipopt::Index;
   using Number = Ipopt::Number;

public:
   struct Params {
      int N = 32;           // interior nodes per side
      int P = 4;            // tiles per side (P×P tiles)
      double kappa = 1.0;   // strength of the u³ term
      double alpha = 1e-4;  // control cost
      double fmax = 40.0;   // |f| ≤ fmax
      double amp = 2.0;     // u_d = amp·sin(πx)·sin(2πy)
   };

   explicit SemilinearDD(const Params& p) : p_(p), h_(1.0 / (p.N + 1)) {
      const int N = p.N, NN = N * N;
      ud_.resize(NN);
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            const double x = (j + 1) * h_, y = (i + 1) * h_;
            ud_[i * N + j] = p.amp * std::sin(M_PI * x) * std::sin(2.0 * M_PI * y);
         }

      // ---- which tiles use u_g: its home tile plus the tiles of its neighbours
      std::vector<std::vector<int>> users(NN);
      for (int g = 0; g < NN; ++g) {
         users[g].push_back(tile_of(g));
         for (int nb : neighbours(g))
            if (std::find(users[g].begin(), users[g].end(), tile_of(nb)) == users[g].end())
               users[g].push_back(tile_of(nb));
      }

      // ---- variables: per node, u (local, or one copy per user tile) and f.
      //      The complicating y come last, so the tail of x is the border.
      uvar_.assign(NN, {});
      fvar_.assign(NN, -1);
      yvar_.assign(NN, -1);
      for (int g = 0; g < NN; ++g) {
         for (int t : users[g]) uvar_[g].push_back({t, new_var(t)});
         fvar_[g] = new_var(tile_of(g));
      }
      for (int g = 0; g < NN; ++g)
         if (users[g].size() > 1) yvar_[g] = new_var(-1);

      // ---- rows: one PDE row per node, then one linking row per copy
      for (int g = 0; g < NN; ++g) row_tile_.push_back(tile_of(g));
      for (int g = 0; g < NN; ++g)
         if (yvar_[g] >= 0)
            for (auto [t, v] : uvar_[g]) {
               links_.push_back({v, yvar_[g]});
               row_tile_.push_back(t);
            }
   }

   // ---------------------------------------------------------------- partition
   int n_tiles() const { return p_.P * p_.P; }
   int n_vars() const { return (int)var_tile_.size(); }
   int n_rows() const { return (int)row_tile_.size(); }
   int n_complicating() const {
      return (int)std::count_if(yvar_.begin(), yvar_.end(), [](int v) { return v >= 0; });
   }
   int n_copies() const { return (int)links_.size(); }

   // Tile of each KKT unknown [x | λ]; −1 = complicating variable (border).
   std::vector<int> kkt_owner() const {
      std::vector<int> owner(var_tile_);
      owner.insert(owner.end(), row_tile_.begin(), row_tile_.end());
      return owner;
   }

   // ---------------------------------------------------------------- results
   std::vector<double> u, f;     // u at the home copy, f
   double objective = 0.0;
   double max_link_gap = 0.0;    // max |copy − y| at the solution
   int n_at_bound = 0;           // controls with |f| within 1e-4·fmax of the bound

   bool save_csv(const std::string& path) const {
      FILE* fp = std::fopen(path.c_str(), "w");
      if (!fp) return false;
      std::fprintf(fp, "i,j,x,y,tile,u,f,ud\n");
      const int N = p_.N;
      for (int g = 0; g < N * N; ++g)
         std::fprintf(fp, "%d,%d,%.6f,%.6f,%d,%.12e,%.12e,%.12e\n", g / N, g % N,
                      (g % N + 1) * h_, (g / N + 1) * h_, tile_of(g), u[g], f[g], ud_[g]);
      std::fclose(fp);
      return true;
   }

   // ================================================================ TNLP
   bool get_nlp_info(Index& n, Index& m, Index& nnz_jac, Index& nnz_h,
                     IndexStyleEnum& style) override {
      n = n_vars();
      m = n_rows();
      nnz_jac = 0;
      for (int g = 0; g < p_.N * p_.N; ++g) nnz_jac += 2 + (int)neighbours(g).size();
      nnz_jac += 2 * n_copies();
      nnz_h = 2 * p_.N * p_.N;   // diagonal: home u and f of every node
      style = C_STYLE;
      return true;
   }

   bool get_bounds_info(Index n, Number* xl, Number* xu, Index m, Number* gl,
                        Number* gu) override {
      for (Index v = 0; v < n; ++v) { xl[v] = -2e19; xu[v] = 2e19; }
      for (int g = 0; g < p_.N * p_.N; ++g) { xl[fvar_[g]] = -p_.fmax; xu[fvar_[g]] = p_.fmax; }
      for (Index r = 0; r < m; ++r) gl[r] = gu[r] = 0.0;   // all equalities
      return true;
   }

   bool get_starting_point(Index n, bool init_x, Number* x, bool, Number*, Number*, Index,
                           bool, Number*) override {
      if (init_x) std::fill(x, x + n, 0.0);
      return true;
   }

   bool eval_f(Index, const Number* x, bool, Number& obj) override {
      const double h2 = h_ * h_;
      obj = 0.0;
      for (int g = 0; g < p_.N * p_.N; ++g) {
         const double e = x[home_u(g)] - ud_[g], fg = x[fvar_[g]];
         obj += 0.5 * h2 * (e * e + p_.alpha * fg * fg);
      }
      return true;
   }

   bool eval_grad_f(Index n, const Number* x, bool, Number* grad) override {
      const double h2 = h_ * h_;
      std::fill(grad, grad + n, 0.0);
      for (int g = 0; g < p_.N * p_.N; ++g) {
         grad[home_u(g)] = h2 * (x[home_u(g)] - ud_[g]);
         grad[fvar_[g]] = h2 * p_.alpha * x[fvar_[g]];
      }
      return true;
   }

   bool eval_g(Index, const Number* x, bool, Index, Number* c) override {
      const int NN = p_.N * p_.N;
      const double h2 = h_ * h_;
      for (int g = 0; g < NN; ++g) {
         const int t = tile_of(g);
         const double ug = x[u_in(t, g)];
         double r = 4.0 * ug + h2 * (p_.kappa * ug * ug * ug - x[fvar_[g]]);
         for (int nb : neighbours(g)) r -= x[u_in(t, nb)];
         c[g] = r;
      }
      for (size_t l = 0; l < links_.size(); ++l)
         c[NN + l] = x[links_[l].copy] - x[links_[l].y];
      return true;
   }

   // Jacobian: PDE row g → home u, its neighbours (as seen from g's tile), f_g;
   //           linking row → +1 on the copy, −1 on y.
   bool eval_jac_g(Index, const Number* x, bool, Index, Index, Index* irow, Index* jcol,
                   Number* val) override {
      const int NN = p_.N * p_.N;
      const double h2 = h_ * h_;
      int k = 0;
      for (int g = 0; g < NN; ++g) {
         const int t = tile_of(g);
         if (val == nullptr) {
            irow[k] = g; jcol[k++] = u_in(t, g);
            for (int nb : neighbours(g)) { irow[k] = g; jcol[k++] = u_in(t, nb); }
            irow[k] = g; jcol[k++] = fvar_[g];
         } else {
            const double ug = x[u_in(t, g)];
            val[k++] = 4.0 + 3.0 * p_.kappa * h2 * ug * ug;
            for (size_t q = 0; q < neighbours(g).size(); ++q) val[k++] = -1.0;
            val[k++] = -h2;
         }
      }
      for (size_t l = 0; l < links_.size(); ++l) {
         if (val == nullptr) {
            irow[k] = NN + (int)l; jcol[k++] = links_[l].copy;
            irow[k] = NN + (int)l; jcol[k++] = links_[l].y;
         } else {
            val[k++] = 1.0;
            val[k++] = -1.0;
         }
      }
      return true;
   }

   // Hessian of σ·objective + Σ λ_r c_r: diagonal, on the home u and on f.
   bool eval_h(Index, const Number* x, bool, Number sigma, Index, const Number* lambda, bool,
               Index, Index* irow, Index* jcol, Number* val) override {
      const double h2 = h_ * h_;
      int k = 0;
      for (int g = 0; g < p_.N * p_.N; ++g) {
         const int uh = home_u(g), fg = fvar_[g];
         if (val == nullptr) {
            irow[k] = jcol[k] = uh; ++k;
            irow[k] = jcol[k] = fg; ++k;
         } else {
            val[k++] = sigma * h2 + lambda[g] * 6.0 * p_.kappa * h2 * x[uh];
            val[k++] = sigma * h2 * p_.alpha;
         }
      }
      return true;
   }

   void finalize_solution(Ipopt::SolverReturn, Index, const Number* x, const Number*,
                          const Number*, Index, const Number*, const Number*, Number obj,
                          const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) override {
      const int NN = p_.N * p_.N;
      u.resize(NN);
      f.resize(NN);
      n_at_bound = 0;
      for (int g = 0; g < NN; ++g) {
         u[g] = x[home_u(g)];
         f[g] = x[fvar_[g]];
         if (std::abs(std::abs(f[g]) - p_.fmax) <= 1e-4 * p_.fmax) ++n_at_bound;
      }
      max_link_gap = 0.0;
      for (const auto& l : links_)
         max_link_gap = std::max(max_link_gap, std::abs(x[l.copy] - x[l.y]));
      objective = obj;
   }

private:
   struct Link { int copy, y; };

   int tile_of(int g) const {
      const int i = g / p_.N, j = g % p_.N;
      return (i * p_.P / p_.N) * p_.P + (j * p_.P / p_.N);
   }
   std::vector<int> neighbours(int g) const {
      const int N = p_.N, i = g / N, j = g % N;
      std::vector<int> nb;
      if (i > 0) nb.push_back(g - N);
      if (i < N - 1) nb.push_back(g + N);
      if (j > 0) nb.push_back(g - 1);
      if (j < N - 1) nb.push_back(g + 1);
      return nb;
   }
   int new_var(int tile) {
      var_tile_.push_back(tile);
      return (int)var_tile_.size() - 1;
   }
   // The variable that tile t uses for u at node g (a copy, or the local u).
   int u_in(int t, int g) const {
      for (auto [tt, v] : uvar_[g])
         if (tt == t) return v;
      assert(false && "tile does not use this u");
      return -1;
   }
   int home_u(int g) const { return u_in(tile_of(g), g); }

   Params p_;
   double h_;
   std::vector<double> ud_;
   std::vector<int> var_tile_;                          // tile of each x, −1 for y
   std::vector<int> row_tile_;                          // tile of each row
   std::vector<std::vector<std::pair<int, int>>> uvar_; // node → (tile, var) for u
   std::vector<int> fvar_, yvar_;                       // node → f var, y var (−1: none)
   std::vector<Link> links_;
};
