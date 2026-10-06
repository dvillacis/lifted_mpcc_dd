// schur_dd.hpp — the linear algebra of Lueg, Bynum, Laird & Biegler,
// "Domain decomposition preconditioners for Schur complement systems arising in
// structured nonlinear optimization problems", Optim. Eng. 27 (2026) 555–585.
//
// Nothing here knows about IPOPT (ipopt_bridge.hpp plugs it in) or about the toy
// problem (only the owner map of the KKT unknowns and where the primals end).
// Block factorizations are in blocks.hpp, preconditioners and PCG in precond.hpp.
//
// ============================================================================
//  1. THE ARROWHEAD (paper eq. 5)
// ============================================================================
// Group the KKT unknowns by tile, the complicating variables y ("border") last:
//
//        ⎡ W_1            B_1ᵀ ⎤ ⎡u_1⎤   ⎡r_1⎤
//        ⎢      ⋱          ⋮   ⎥ ⎢ ⋮ ⎥ = ⎢ ⋮ ⎥
//        ⎢          W_K   B_Kᵀ ⎥ ⎢u_K⎥   ⎢r_K⎥
//        ⎣ B_1  ⋯   B_K    C   ⎦ ⎣u_y⎦   ⎣r_y⎦
//
// W_k: the tile's own KKT block (eq. 6).  B_k: the −1 that each linking row
// "copy − y = 0" puts on y, i.e. the paper's R̄_k = N_kᵀR_k (N_k selects the p_k
// border unknowns tile k touches).  C: the y–y block; the paper has C = 0, IPOPT
// adds δ_w there, so C = δ_w·I.
//
// ============================================================================
//  2. SCHUR COMPLEMENT (eqs. 10, 11, 14, 15)
// ============================================================================
//      S u_y = r_y − Σ_k N_k B_k W_k⁻¹ r_k                          (10)
//      S     = C + Σ_k N_k S_k N_kᵀ,   S_k = −B_k W_k⁻¹ B_kᵀ         (14), (15)
//      u_k   = W_k⁻¹ (r_k − B_kᵀ N_kᵀ u_y)                          (11)
//
// Only the small dense S_k are stored.  S itself is assembled (sparse) only for
// the direct solve and for the exact-inertia diagnostic.
//
// ============================================================================
//  3. SOLVING FOR u_y
// ============================================================================
//  · direct ("xSC"): factorize the assembled S.
//  · pcg: CG on S, applied tile by tile through the S_k (15) or by back-solves
//    (12), with a one-level preconditioner
//        bj   S_k                                (17)
//        as   S̃_k = N_kᵀ S N_k                   (18, 19)
//        asd  S_k with diag replaced by diag(S)  (20, 21)
//    and optionally a coarse term (22) with Z built from the interface "faces"
//    (the y's shared by the same set of tiles): see build_Z.
//
// ============================================================================
//  4. INERTIA (eqs. 9, 24–26)
// ============================================================================
// In(A) = Σ_k In(W_k) + In(S) (Haynsworth).  In(W_k) comes from the block
// factorization.  For S:
//  · direct: exact, from its factorization.
//  · pcg, inertia "curvature": S is assumed SPD unless CG meets pᵀSp ≤ 0 on
//    the first solve with this matrix (then the bridge refuses the matrix).
//    Not rigorous: CG can solve an indefinite S without meeting such p.
//  · pcg, inertia "exact": a factorization of S just for its inertia (diagnostic).
//  · pcg, inertia "free": no inertia; IPOPT's inertia-free curvature test
//    (neg_curv_test_tol) decides on regularization instead.
//  · --local-inertia: also require In(S_k) = (p_k,0,0) (26); here S_k is
//    singular at every interior tile corner, so this always fails.
//
// ============================================================================
//  5. PARALLELISM (OpenMP)
// ============================================================================
// Every per-tile loop runs in parallel.  Tiles never write to the same border
// entry: each writes into its own buffer, and a parallel loop over the border
// gathers through users_ (border index → the (tile, position) pairs using it).
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "blocks.hpp"
#include "precond.hpp"

namespace dd {

class SchurDD {
public:
   enum class Interface { Direct, Pcg };
   enum class Precond { None, BJ, AS, ASd };
   enum class Inertia { Curvature, Exact, Free };
   enum class CoarseSpace { None, Faces, Sides, Linear };
   enum class CoarseMode { Additive, Balanced };
   struct Options {
      Interface interface = Interface::Direct;
      Precond precond = Precond::ASd;
      bool sparse_blocks = true;       // W_k: sparse LDLᵀ (true) or dense Bunch–Kaufman
      double cg_tol = 1e-10;
      int cg_maxit = 2000;
      bool matvec_backsolve = false;   // S·v by eq. (12) instead of (15)
      Inertia inertia = Inertia::Curvature;
      bool local_inertia = false;      // enforce In(S_k) = (p_k,0,0), eq. (26)
      CoarseSpace coarse = CoarseSpace::None;    // two-level: coarse space of eq. (22), see build_Z
      CoarseMode coarse_mode = CoarseMode::Balanced;
      int dense_s_max = 2000;          // S up to this size: dense Bunch–Kaufman
      int check_max_dim = 0;           // > 0: compare with a dense solve of the full KKT
      int verbose = 0;                 // 1: per factorization, 2: + per solve, 3: + spectra of S_k
   };
   struct Stats {
      // partition (paper Table 2)
      int n_tiles = 0, dim = 0, p = 0, max_pk = 0, max_tk = 0, n_faces = 0;
      // counters
      long factorizations = 0, singular = 0, s_not_pd = 0, local_not_pd = 0;
      long coarse_failures = 0, coarse_dim = 0;
      long solves = 0, cg_iters = 0, cg_max = 0, cg_breakdowns = 0, cg_unconverged = 0;
      long checks = 0, check_inertia_mismatch = 0;
      double check_max_err = 0.0;
      long factor_nnz = 0;             // Σ nnz(L_k), last factorization (sparse blocks)
      long dense_fallbacks = 0;        // tile factorizations redone densely (zero pivot)
      // wall time (s): totals and phases
      double t_factor = 0.0, t_solve = 0.0;
      double t_blocks = 0.0, t_sdirect = 0.0, t_precond = 0.0, t_coarse = 0.0;
      double t_tile_solves = 0.0, t_interface = 0.0;
   };

   // ---------------------------------------------------------------- structure
   // owner[i]: tile of KKT unknown i, −1 = border.  Unknowns i < n_x are primal.
   // Optional coarse-space hints, per KKT unknown (only border entries are read):
   // side = which tile the unknown "belongs to" (its home), xy = its position.
   struct Hints {
      std::vector<int> side;
      std::vector<std::array<double, 2>> xy;
   };
   bool set_structure(int dim, const std::vector<int>& irow, const std::vector<int>& jcol,
                      const std::vector<int>& owner, int n_x, int n_tiles, const Options& opt,
                      const Hints* hints = nullptr) {
      if ((int)owner.size() != dim) {
         std::fprintf(stderr, "[dd] owner map has %zu entries, KKT has %d\n", owner.size(), dim);
         return false;
      }
      opt_ = opt;
      dim_ = dim;
      irow_ = irow;
      jcol_ = jcol;
      values_.assign(irow.size(), 0.0);
      tiles_ = std::vector<Tile>(n_tiles);   // Tile holds a factorization: not copyable
      c_.clear();

      // local numbering: tile unknowns in ascending KKT order, border likewise
      std::vector<int> loc(dim, -1);
      border_.clear();
      for (int i = 0; i < dim; ++i) {
         if (owner[i] < 0) {
            loc[i] = (int)border_.size();
            border_.push_back(i);
         } else {
            Tile& T = tiles_[owner[i]];
            loc[i] = (int)T.glob.size();
            T.glob.push_back(i);
            T.primal.push_back(i < n_x);
         }
      }
      const int p = (int)border_.size();

      // N_k: the border unknowns tile k touches
      for (size_t t = 0; t < irow.size(); ++t) {
         const int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0 && owner[r] != owner[c]) {
            std::fprintf(stderr, "[dd] KKT entry (%d,%d) couples tiles %d and %d\n", r, c,
                         owner[r], owner[c]);
            return false;
         }
         if ((owner[r] < 0) != (owner[c] < 0))
            tiles_[std::max(owner[r], owner[c])].nk.push_back(loc[owner[r] < 0 ? r : c]);
      }
      for (Tile& T : tiles_) {
         std::sort(T.nk.begin(), T.nk.end());
         T.nk.erase(std::unique(T.nk.begin(), T.nk.end()), T.nk.end());
      }

      // route every triplet to W_k, B_k or C
      for (size_t t = 0; t < irow.size(); ++t) {
         const int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0) {
            tiles_[owner[r]].w.push_back({loc[r], loc[c], (int)t});
         } else if (owner[r] < 0 && owner[c] < 0) {
            c_.push_back({loc[r], loc[c], (int)t});
         } else {
            Tile& T = tiles_[std::max(owner[r], owner[c])];
            const int b = owner[r] < 0 ? r : c, l = owner[r] < 0 ? c : r;
            const int a = (int)(std::lower_bound(T.nk.begin(), T.nk.end(), loc[b]) - T.nk.begin());
            T.b.push_back({a, loc[l], (int)t});
         }
      }

      // users_: border index → (tile, position in N_k)
      ustart_.assign(p + 1, 0);
      for (const Tile& T : tiles_)
         for (int i : T.nk) ++ustart_[i + 1];
      for (int i = 0; i < p; ++i) ustart_[i + 1] += ustart_[i];
      users_.assign(ustart_[p], {0, 0});
      {
         std::vector<int> fill(ustart_.begin(), ustart_.end() - 1);
         for (int k = 0; k < n_tiles; ++k)
            for (int a = 0; a < (int)tiles_[k].nk.size(); ++a)
               users_[fill[tiles_[k].nk[a]]++] = {k, a};
      }

      if (opt_.coarse != CoarseSpace::None) build_Z(hints);

      // symbolic work per tile
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < n_tiles; ++k) {
         Tile& T = tiles_[k];
         T.blk.set_pattern(opt_.sparse_blocks, (int)T.glob.size(), T.w, T.primal);
      }

      st_->n_tiles = n_tiles;
      st_->dim = dim;
      st_->p = p;
      st_->n_faces = n_faces_;
      st_->coarse_dim = Z_.cols();
      st_->max_pk = st_->max_tk = 0;
      for (const Tile& T : tiles_) {
         st_->max_pk = std::max(st_->max_pk, (int)T.nk.size());
         st_->max_tk = std::max(st_->max_tk, (int)T.glob.size());
      }
      return true;
   }

   double* values() { return values_.data(); }
   int dim() const { return dim_; }
   int negative_eigenvalues() const { return neg_; }
   // Per-tile view (for a host that regularizes tile by tile, paper Sec. 3.5).
   // After a successful factorize(): In(W_k) is right iff tile_negative(k) ==
   // tile_duals(k) (eq. 24), and s_negative() is #neg(S) when S was factorized
   // (direct mode or exact inertia), 0 otherwise.  After a failed one,
   // singular_tile() is the first tile with a zero pivot (−1: S was singular).
   int n_tiles() const { return (int)tiles_.size(); }
   int tile_negative(int k) const { return tiles_[k].blk.negative(); }
   int tile_duals(int k) const {
      return (int)std::count(tiles_[k].primal.begin(), tiles_[k].primal.end(), 0);
   }
   int s_negative() const { return negS_; }
   int singular_tile() const { return sing_tile_; }
   const Stats& stats() const { return *st_; }
   // Counters go to *sink (shared by several solver instances) instead of our own.
   void set_stats_sink(Stats* sink) { st_ = sink ? sink : &own_stats_; }

   // ---------------------------------------------------------------- factorize
   // false = singular (some W_k or, when it is factorized, S has a zero pivot).
   bool factorize() {
      const auto t0 = Clock::now();
      ++st_->factorizations;
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      fresh_ = true;
      const bool pcg_mode = opt_.interface == Interface::Pcg;

      // ---- each tile: W_k → In(W_k);  S_k = −B_k W_k⁻¹ B_kᵀ                (14)
      std::vector<char> ok(K, 1);
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < K; ++k) {
         Tile& T = tiles_[k];
         if (!T.blk.factorize(values_.data())) { ok[k] = 0; continue; }
         std::vector<Eigen::Triplet<double>> tb;
         tb.reserve(T.b.size());
         for (const Entry& e : T.b) tb.emplace_back(e[0], e[1], values_[e[2]]);
         T.B.resize((int)T.nk.size(), (int)T.glob.size());
         T.B.setFromTriplets(tb.begin(), tb.end());
         T.S = T.blk.schur(T.B);
      }
      st_->t_blocks += secs(t0);
      {
         long fb = 0;
         for (const Tile& T : tiles_) fb += T.blk.fallbacks();
         st_->dense_fallbacks += fb - fallbacks_seen_;
         fallbacks_seen_ = fb;
      }
      int negW = 0;
      negS_ = 0;
      sing_tile_ = -1;
      for (int k = 0; k < K; ++k) {
         if (!ok[k]) {
            sing_tile_ = k;
            ++st_->singular;
            if (opt_.verbose) std::printf("[dd] fact %ld: W_%d singular\n", st_->factorizations, k);
            st_->t_factor += secs(t0);
            return false;
         }
         negW += tiles_[k].blk.negative();
      }
      if (opt_.verbose >= 3) print_local_spectra();

      // ---- C (= δ_w·I here) and diag(S)
      cdiag_.setZero(p);
      coff_.clear();
      for (const Entry& e : c_) {
         if (e[0] == e[1]) cdiag_[e[0]] += values_[e[2]];
         else coff_.push_back({(double)e[0], (double)e[1], values_[e[2]]});
      }
      diagS_ = cdiag_;
      for (int i = 0; i < p; ++i)
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q)
            diagS_[i] += tiles_[users_[q][0]].S(users_[q][1], users_[q][1]);

      // ---- S itself: direct solve, or the exact-inertia diagnostic          (25)
      int negS = 0;
      const bool need_S = p > 0 && (!pcg_mode || opt_.inertia == Inertia::Exact);
      if (need_S) {
         const auto t1 = Clock::now();
         const SpMat S = assemble_S();
         const bool good = Sf_.factorize(S, opt_.dense_s_max);
         st_->t_sdirect += secs(t1);
         if (!good) {
            ++st_->singular;
            st_->t_factor += secs(t0);
            if (opt_.verbose) std::printf("[dd] fact %ld: S singular\n", st_->factorizations);
            return false;
         }
         negS = Sf_.negative();
         if (negS > 0) ++st_->s_not_pd;
      }

      if (p > 0 && pcg_mode) {
         // ---- one-level preconditioner                         (17), (19), (21)
         const auto t1 = Clock::now();
         if (opt_.precond == Precond::AS) {
#pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < K; ++k) tiles_[k].St = local_assembled(k);
         }
         M_.reset(p, opt_.precond == Precond::None ? 0 : K);
         long not_pd = 0;
         if (opt_.precond != Precond::None) {
#pragma omp parallel for schedule(dynamic) reduction(+ : not_pd)
            for (int k = 0; k < K; ++k) {
               const Tile& T = tiles_[k];
               if (T.nk.empty()) continue;
               Mat Mk;
               if (opt_.precond == Precond::AS) {
                  Mk = T.St;                                           // (18)
               } else {
                  Mk = T.S;
                  for (size_t a = 0; a < T.nk.size(); ++a) {
                     const int i = T.nk[a];
                     Mk(a, a) = opt_.precond == Precond::ASd
                                   ? diagS_[i]                         // (20)
                                   : Mk(a, a) + cdiag_[i] / mult(i);   // C split evenly
                  }
               }
               if (!M_.set(k, T.nk, Mk)) ++not_pd;
            }
         }
         M_.finalize();
         st_->local_not_pd += not_pd;
         st_->t_precond += secs(t1);

         // ---- coarse space                                                (22)
         use_coarse_ = false;
         if (opt_.coarse != CoarseSpace::None) {
            const auto t2 = Clock::now();
            use_coarse_ = build_coarse();
            if (!use_coarse_) ++st_->coarse_failures;
            st_->t_coarse += secs(t2);
         }

         // ---- optional local inertia check (26)
         if (opt_.local_inertia && opt_.inertia != Inertia::Exact) {
            bool all_spd = true;
            for (int k = 0; k < K && all_spd; ++k) {
               const Tile& T = tiles_[k];
               if (T.nk.empty()) continue;
               Mat Mk = T.S;
               for (size_t a = 0; a < T.nk.size(); ++a) Mk(a, a) += cdiag_[T.nk[a]] / mult(T.nk[a]);
               all_spd = Eigen::LLT<Mat>(Mk).info() == Eigen::Success;
            }
            if (!all_spd) negS = std::max(negS, 1);   // not a count: "not certified"
         }
      }
      neg_ = negW + negS;
      negS_ = negS;

      if (opt_.check_max_dim > 0 && dim_ <= opt_.check_max_dim) dense_reference();

      if (opt_.verbose) {
         // C_yy = δ_w in the Newton systems (1 in IPOPT's initial least-squares
         // multiplier system, which also goes through this solver)
         const double dw = p > 0 ? cdiag_.mean() : 0.0;
         std::printf("[dd] fact %3ld  C_yy=%-9.2e Σ#neg(W_k)=%d", st_->factorizations, dw, negW);
         if (need_S) std::printf("  #neg(S)=%d", negS);
         if (use_coarse_) std::printf("  coarse dim %d", coarse_.dim());
         if (opt_.check_max_dim > 0 && dim_ <= opt_.check_max_dim)
            std::printf("  | dense #neg(A)=%d", dense_neg_);
         std::printf("\n");
      }
      st_->t_factor += secs(t0);
      return true;
   }

   // ---------------------------------------------------------------- solve
   // In place.  Returns false if PCG met negative curvature (S not SPD).
   bool solve(double* rhs) {
      const auto t0 = Clock::now();
      ++st_->solves;
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      Eigen::Map<Vec> R(rhs, dim_);
      const bool checking = fresh_ && dense_lu_.rows() == dim_ && dim_ > 0;
      Vec r0;
      if (checking) r0 = R;

      // z_k = W_k⁻¹ r_k,  then r_S = r_y − Σ N_k B_k z_k                    (10)
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < K; ++k) {
         Tile& T = tiles_[k];
         T.z.resize(T.glob.size());
         for (size_t l = 0; l < T.glob.size(); ++l) T.z[l] = R[T.glob[l]];
         T.blk.solve(T.z);
         T.buf = T.B * T.z;
      }
      Vec rS(p);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p; ++i) {
         double s = R[border_[i]];
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q)
            s -= tiles_[users_[q][0]].buf[users_[q][1]];
         rS[i] = s;
      }
      const auto t1 = Clock::now();
      st_->t_tile_solves += secs(t0);

      // S u_y = r_S
      Vec uy = Vec::Zero(p);
      bool good = true;
      if (p > 0 && opt_.interface == Interface::Direct) {
         uy = rS;
         Sf_.solve(uy);
      } else if (p > 0) {
         auto applyS = [this](const Vec& v, Vec& out) { apply_S(v, out); };
         auto applyM = [this](const Vec& r, Vec& z) { apply_M(r, z); };
         const CgResult cg = pcg(applyS, applyM, rS, uy, opt_.cg_tol, opt_.cg_maxit);
         st_->cg_iters += cg.iters;
         st_->cg_max = std::max<long>(st_->cg_max, cg.iters);
         if (cg.breakdown) { ++st_->cg_breakdowns; good = false; }
         else if (cg.rel > opt_.cg_tol) ++st_->cg_unconverged;
         if (opt_.verbose >= 2)
            std::printf("[dd]      solve: pcg its=%4d  rel.res=%.1e%s\n", cg.iters, cg.rel,
                        cg.breakdown ? "  NEGATIVE CURVATURE" : "");
      }
      const auto t2 = Clock::now();
      st_->t_interface += std::chrono::duration<double>(t2 - t1).count();

      // u_k = W_k⁻¹(r_k − B_kᵀ N_kᵀ u_y)                                   (11)
      for (int i = 0; i < p; ++i) R[border_[i]] = uy[i];
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < K; ++k) {
         Tile& T = tiles_[k];
         Vec uyk(T.nk.size());
         for (size_t a = 0; a < T.nk.size(); ++a) uyk[a] = uy[T.nk[a]];
         Vec rk(T.glob.size());
         for (size_t l = 0; l < T.glob.size(); ++l) rk[l] = R[T.glob[l]];
         rk -= T.B.transpose() * uyk;
         T.blk.solve(rk);
         for (size_t l = 0; l < T.glob.size(); ++l) R[T.glob[l]] = rk[l];
      }
      st_->t_tile_solves += secs(t2);

      if (checking) {
         const Vec xd = dense_lu_.solve(r0);
         const double err = (R - xd).norm() / std::max(xd.norm(), 1e-300);
         ++st_->checks;
         st_->check_max_err = std::max(st_->check_max_err, err);
         std::printf("[dd]      check: ‖u_dd − u_dense‖/‖u_dense‖ = %.1e   #neg: dd %d, dense %d%s\n",
                     err, neg_, dense_neg_, neg_ == dense_neg_ ? "" : "   MISMATCH");
      }
      fresh_ = false;
      st_->t_solve += secs(t0);
      return good;
   }

private:
   using Clock = std::chrono::steady_clock;
   static double secs(Clock::time_point t0) {
      return std::chrono::duration<double>(Clock::now() - t0).count();
   }

   struct Tile {
      std::vector<int> glob;     // KKT index of each tile unknown
      std::vector<char> primal;  // is it a primal variable?
      std::vector<int> nk;       // border indices touched: N_k
      std::vector<Entry> w, b;   // routed triplets: W_k (r, c, slot), B_k (a, l, slot)
      TileBlock blk;             // factorization of W_k
      SpMat B;                   // B_k (p_k × t_k)
      Mat S, St;                 // S_k, and S̃_k = N_kᵀ S N_k when needed
      Vec z;                     // W_k⁻¹ r_k during a solve
      mutable Vec buf;           // per-tile output, gathered over the border
   };

   int mult(int i) const { return ustart_[i + 1] - ustart_[i]; }

   // S = C + Σ_k N_k S_k N_kᵀ (15), sparse, both triangles
   SpMat assemble_S() const {
      const int p = (int)border_.size();
      size_t nnz = p + coff_.size();
      for (const Tile& T : tiles_) nnz += T.nk.size() * T.nk.size();
      std::vector<Eigen::Triplet<double>> t;
      t.reserve(nnz);
      for (int i = 0; i < p; ++i) t.emplace_back(i, i, cdiag_[i]);
      for (const auto& e : coff_) {
         t.emplace_back((int)e[0], (int)e[1], e[2]);
         t.emplace_back((int)e[1], (int)e[0], e[2]);
      }
      for (const Tile& T : tiles_)
         for (size_t a = 0; a < T.nk.size(); ++a)
            for (size_t b = 0; b < T.nk.size(); ++b) t.emplace_back(T.nk[a], T.nk[b], T.S(a, b));
      SpMat S(p, p);
      S.setFromTriplets(t.begin(), t.end());
      return S;
   }

   // S̃_k = N_kᵀ S N_k, from the S_j of the tiles that share border unknowns
   // with tile k (paper: "local assembled Schur complement", eq. 18).
   Mat local_assembled(int k) const {
      const Tile& T = tiles_[k];
      const int pk = (int)T.nk.size();
      static thread_local std::vector<int> mark;
      if ((int)mark.size() < (int)border_.size()) mark.assign(border_.size(), -1);
      for (int a = 0; a < pk; ++a) mark[T.nk[a]] = a;
      Mat St = Mat::Zero(pk, pk);
      for (int a = 0; a < pk; ++a) {
         const int i = T.nk[a];
         St(a, a) += cdiag_[i];
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) {
            const Tile& Tj = tiles_[users_[q][0]];
            const int aj = users_[q][1];
            for (size_t bj = 0; bj < Tj.nk.size(); ++bj) {
               const int b = mark[Tj.nk[bj]];
               if (b >= 0) St(a, b) += Tj.S(aj, bj);
            }
         }
      }
      for (const auto& e : coff_) {
         const int a = mark[(int)e[0]], b = mark[(int)e[1]];
         if (a >= 0 && b >= 0) { St(a, b) += e[2]; St(b, a) += e[2]; }
      }
      for (int a = 0; a < pk; ++a) mark[T.nk[a]] = -1;
      return 0.5 * (St + St.transpose());
   }

   // The coarse basis Z (p × p₀), fixed for the whole run.  Border unknowns are
   // grouped into FACES by the set of tiles that use them (2 tiles: an edge of
   // the tile grid; 3+: near a tile corner).  Then
   //   faces   one column per face, 1 on its members (constants; Lueg eq. 22 with
   //           a partition-of-unity-free piecewise-constant N₀)
   //   sides   faces split by the side of the cut each unknown lies on (hint):
   //           spans the constant AND the jump across the cut, which is the
   //           normal derivative.  This interface operator is fourth order
   //           (eliminating f leaves α|Δu|²), so it needs that mode.
   //   linear  sides plus the centred x and y coordinates on each group (hint):
   //           the linear modes along each face.
   void build_Z(const Hints* hints) {
      const int p = (int)border_.size();
      CoarseSpace mode = opt_.coarse;
      if (mode != CoarseSpace::Faces && (hints == nullptr || hints->side.empty())) {
         std::fprintf(stderr, "[dd] coarse sides/linear need hints from the problem; using faces\n");
         mode = CoarseSpace::Faces;
      }
      std::map<std::vector<int>, int> faces, groups;
      std::vector<int> group(p);
      for (int i = 0; i < p; ++i) {
         std::vector<int> sig;
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) sig.push_back(users_[q][0]);
         std::sort(sig.begin(), sig.end());
         faces.emplace(sig, (int)faces.size());
         if (mode != CoarseSpace::Faces) sig.push_back(hints->side[border_[i]]);
         group[i] = groups.emplace(sig, (int)groups.size()).first->second;
      }
      n_faces_ = (int)faces.size();
      const int ng = (int)groups.size();
      std::vector<Eigen::Triplet<double>> zt;
      for (int i = 0; i < p; ++i) zt.emplace_back(i, group[i], 1.0);
      int ncol = ng;
      if (mode == CoarseSpace::Linear && !hints->xy.empty()) {
         std::vector<std::vector<int>> members(ng);
         for (int i = 0; i < p; ++i) members[group[i]].push_back(i);
         for (int g = 0; g < ng; ++g) {
            if (members[g].size() < 3) continue;
            for (int d = 0; d < 2; ++d) {
               double mean = 0.0;
               for (int i : members[g]) mean += hints->xy[border_[i]][d];
               mean /= members[g].size();
               double nrm = 0.0;
               for (int i : members[g]) nrm += std::pow(hints->xy[border_[i]][d] - mean, 2);
               if (nrm < 1e-24) continue;   // the face runs along the other axis
               nrm = std::sqrt(nrm / members[g].size());
               for (int i : members[g])
                  zt.emplace_back(i, ncol, (hints->xy[border_[i]][d] - mean) / nrm);
               ++ncol;
            }
         }
      }
      Z_.resize(p, ncol);
      Z_.setFromTriplets(zt.begin(), zt.end());
      Zr_ = Z_;
   }

   // S₀ = Zᵀ S Z, assembled tile by tile.
   bool build_coarse() {
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      const SpMat& Z = Z_;
      const Eigen::SparseMatrix<double, Eigen::RowMajor>& Zr = Zr_;
      const int ncol = (int)Z.cols();

      // per tile: G_k = Z_kᵀ S_k Z_k on the coarse columns it touches
      std::vector<std::vector<int>> cols(K);
      std::vector<Mat> G(K);
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < K; ++k) {
         const Tile& T = tiles_[k];
         std::vector<int>& cl = cols[k];
         for (int i : T.nk)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Zr, i); it; ++it)
               cl.push_back((int)it.col());
         std::sort(cl.begin(), cl.end());
         cl.erase(std::unique(cl.begin(), cl.end()), cl.end());
         Mat Zk = Mat::Zero(T.nk.size(), cl.size());
         for (size_t a = 0; a < T.nk.size(); ++a)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Zr, T.nk[a]); it; ++it)
               Zk(a, std::lower_bound(cl.begin(), cl.end(), (int)it.col()) - cl.begin()) = it.value();
         G[k] = Zk.transpose() * T.S * Zk;
      }
      std::vector<Eigen::Triplet<double>> st;
      for (int k = 0; k < K; ++k)
         for (size_t a = 0; a < cols[k].size(); ++a)
            for (size_t b = 0; b < cols[k].size(); ++b)
               st.emplace_back(cols[k][a], cols[k][b], G[k](a, b));
      for (int i = 0; i < p; ++i)   // Zᵀ C Z, diagonal part of C
         for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator a(Zr, i); a; ++a)
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator b(Zr, i); b; ++b)
               st.emplace_back((int)a.col(), (int)b.col(), cdiag_[i] * a.value() * b.value());
      SpMat S0(ncol, ncol);
      S0.setFromTriplets(st.begin(), st.end());
      return coarse_.build(Z, S0);
   }

   // out = S v, tile by tile: (15) with the stored S_k, or (12) by back-solves
   void apply_S(const Vec& v, Vec& out) const {
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
#pragma omp parallel for schedule(dynamic, 4)
      for (int k = 0; k < K; ++k) {
         const Tile& T = tiles_[k];
         const int pk = (int)T.nk.size();
         if (pk == 0) continue;
         Vec vk(pk);
         for (int a = 0; a < pk; ++a) vk[a] = v[T.nk[a]];
         if (opt_.matvec_backsolve) {
            Vec w = T.B.transpose() * vk;
            T.blk.solve(w);
            T.buf = -(T.B * w);
         } else {
            T.buf = T.S * vk;
         }
      }
      out.resize(p);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p; ++i) {
         double s = cdiag_[i] * v[i];
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q)
            s += tiles_[users_[q][0]].buf[users_[q][1]];
         out[i] = s;
      }
      for (const auto& e : coff_) {
         out[(int)e[0]] += e[2] * v[(int)e[1]];
         out[(int)e[1]] += e[2] * v[(int)e[0]];
      }
   }

   // z = M⁻¹ r: one level, plus the coarse term (additive or balanced)
   void apply_M(const Vec& r, Vec& z) const {
      if (!use_coarse_) { M_.apply(r, z); return; }
      Vec q;
      coarse_.apply(r, q);                       // Q r
      if (opt_.coarse_mode == CoarseMode::Additive) {
         M_.apply(r, z);
         z += q;
         return;
      }
      Vec t, y, u;                               // Pᵀ M⁻¹ P r + Q r,  P = I − S Q
      apply_S(q, t);
      M_.apply(r - t, y);
      apply_S(y, t);
      coarse_.apply(t, u);
      z = y - u + q;
   }

   // --verbose 3: the spectrum of every S_k (only positive SEMIdefinite: tile k
   // does not see a combination of copies that costs it nothing, e.g. the two
   // foreign copies in the row at an interior tile corner).
   void print_local_spectra() const {
      for (size_t k = 0; k < tiles_.size(); ++k) {
         const Mat& S = tiles_[k].S;
         if (S.size() == 0) continue;
         const Vec ev = Eigen::SelfAdjointEigenSolver<Mat>(S, Eigen::EigenvaluesOnly).eigenvalues();
         const double big = ev.cwiseAbs().maxCoeff(), tiny = 1e-10 * std::max(big, 1e-300);
         int neg = 0, zero = 0;
         for (int i = 0; i < ev.size(); ++i) {
            if (std::abs(ev[i]) <= tiny) ++zero;
            else if (ev[i] < 0) ++neg;
         }
         std::printf("[dd]      S_%-3zu p_k=%-3d eig ∈ [%10.3e, %10.3e]  #neg=%d  #(|λ|≤1e-10·max)=%d\n",
                     k, (int)S.rows(), ev[0], ev[ev.size() - 1], neg, zero);
      }
   }

   // The full KKT matrix, dense: its exact inertia and an LU for checking solves.
   void dense_reference() {
      Mat A = Mat::Zero(dim_, dim_);
      for (size_t t = 0; t < irow_.size(); ++t) {
         A(irow_[t], jcol_[t]) += values_[t];
         if (irow_[t] != jcol_[t]) A(jcol_[t], irow_[t]) += values_[t];
      }
      Eigen::SelfAdjointEigenSolver<Mat> es(A, Eigen::EigenvaluesOnly);
      const Vec ev = es.eigenvalues();
      const double tiny = 1e-12 * std::max(ev.cwiseAbs().maxCoeff(), 1.0);
      dense_neg_ = 0;
      for (int i = 0; i < ev.size(); ++i) dense_neg_ += ev[i] < -tiny;
      if (dense_neg_ != neg_) ++st_->check_inertia_mismatch;
      dense_lu_.compute(A);
   }

   Options opt_;
   int dim_ = 0, neg_ = 0, dense_neg_ = 0, n_faces_ = 0, negS_ = 0, sing_tile_ = -1;
   bool fresh_ = false, use_coarse_ = false;
   long fallbacks_seen_ = 0;
   std::vector<int> irow_, jcol_, border_, ustart_;
   SpMat Z_;                                     // coarse basis (p × p₀)
   Eigen::SparseMatrix<double, Eigen::RowMajor> Zr_;
   std::vector<std::array<int, 2>> users_;
   std::vector<double> values_;
   std::vector<Tile> tiles_;
   std::vector<Entry> c_;
   std::vector<std::array<double, 3>> coff_;   // off-diagonal C entries (none in the toy)
   Vec cdiag_, diagS_;
   SymFactor Sf_;
   Schwarz M_;
   Coarse coarse_;
   Eigen::PartialPivLU<Mat> dense_lu_;
   Stats own_stats_;
   Stats* st_ = &own_stats_;
};

}  // namespace dd
