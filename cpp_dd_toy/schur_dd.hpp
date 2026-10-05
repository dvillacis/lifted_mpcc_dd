// schur_dd.hpp — the linear algebra of Lueg, Bynum, Laird & Biegler,
// "Domain decomposition preconditioners for Schur complement systems arising in
// structured nonlinear optimization problems", Optim. Eng. 27 (2026) 555–585.
//
// Pure Eigen + LAPACK: nothing here knows about IPOPT (ipopt_bridge.hpp plugs it
// in) or about the toy problem (only the owner map of the KKT unknowns).
//
// ============================================================================
//  1. THE ARROWHEAD (paper eq. 5)
// ============================================================================
// Group the KKT unknowns by tile, the complicating variables y ("border") last.
// Tiles never couple directly, so IPOPT's KKT matrix is an arrowhead:
//
//        ⎡ W_1            B_1ᵀ ⎤ ⎡u_1⎤   ⎡r_1⎤
//        ⎢      ⋱          ⋮   ⎥ ⎢ ⋮ ⎥ = ⎢ ⋮ ⎥
//        ⎢          W_K   B_Kᵀ ⎥ ⎢u_K⎥   ⎢r_K⎥
//        ⎣ B_1  ⋯   B_K    C   ⎦ ⎣u_y⎦   ⎣r_y⎦
//
// W_k is the tile's own KKT block (eq. 6).  B_k holds the −1 that each linking
// row "copy − y = 0" puts on y.  This B_k is the paper's R̄_k = N_kᵀR_k: the p_k
// border unknowns tile k touches, which N_k selects from all p of them.  C is
// the y–y block.  The paper has C = 0 because y appears only in linear linking
// rows.  IPOPT still adds its primal regularization there, so here C = δ_w·I.
// The verbose log prints that δ_w.
//
// ============================================================================
//  2. SCHUR COMPLEMENT (eqs. 10, 11, 14, 15)
// ============================================================================
// Eliminating the tiles leaves a system in the border alone:
//
//      S u_y = r_y − Σ_k N_k B_k W_k⁻¹ r_k                          (10)
//      S     = C + Σ_k N_k S_k N_kᵀ,   S_k = −B_k W_k⁻¹ B_kᵀ         (14), (15)
//      u_k   = W_k⁻¹ (r_k − B_kᵀ N_kᵀ u_y)                          (11)
//
// Forming S_k costs p_k back-solves with W_k.  X_k = W_k⁻¹B_kᵀ is kept, so (11)
// is one back-solve plus a product.
//
// ============================================================================
//  3. SOLVING FOR u_y: DIRECT OR PCG (Sec. 3, 3.1, 3.3)
// ============================================================================
//  · direct ("xSC"): assemble S densely and factorize it.
//  · pcg: conjugate gradients on S.  The product S·v is applied tile by tile,
//    either with the dense S_k (15) or with one back-solve per tile (12).  The
//    preconditioner is one of
//        none  M = I
//        bj    M⁻¹ = Σ N_k  S_k⁻¹ N_kᵀ        block Jacobi                (17)
//        as    M⁻¹ = Σ N_k  S̃_k⁻¹ N_kᵀ,  S̃_k = N_kᵀ S N_k                 (18, 19)
//        asd   M⁻¹ = Σ N_k (S̃ᴰ_k)⁻¹ N_kᵀ, S_k with diag replaced by diag(S) (20, 21)
//    In bj and asd, C's diagonal is split evenly between the tiles that touch
//    each y, so that Σ N_k S_k N_kᵀ is still exactly S.
//
// ============================================================================
//  4. INERTIA (eqs. 9, 24–26)
// ============================================================================
// IPOPT wants exactly (#constraints) negative eigenvalues.  By Haynsworth,
//
//        In(A) = Σ_k In(W_k) + In(S),
//
// so the condition splits into In(W_k) = (n_k+p_k, m_k+p_k, 0) (24) and S
// positive definite (25).  Each W_k is factorized by dense Bunch–Kaufman
// (LAPACK dsytrf).  The paper uses MA27; both are pivoted LDLᵀ and give the
// exact inertia.  Tiles are small, so dense is fine.  For S:
//  · direct: Bunch–Kaufman on S gives #neg(S) exactly.
//  · pcg: S is not factorized.  As in the paper's experiments, a direction with
//    pᵀSp ≤ 0 in PCG is taken as proof that S is not positive definite, and
//    the bridge answers WRONG_INERTIA, if this is the first solve with this
//    matrix (see ipopt_bridge.hpp).  CG can solve an indefinite S without ever
//    meeting such a direction, so the check can miss.  --pcg-inertia exact
//    replaces it with a dense factorization of S, as a diagnostic.
//    Optionally (--local-inertia) also check In(S_k) = (p_k,0,0) (26).  That is
//    sufficient but conservative: here S_k is singular at every interior tile
//    corner, so (26) always fails.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

extern "C" {
void dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv,
             double* work, const int* lwork, int* info);
void dsytrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda,
             const int* ipiv, double* b, const int* ldb, int* info);
}

namespace dd {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

// =============================================================================
//  DenseBK — dense symmetric-indefinite LDLᵀ with Bunch–Kaufman pivoting.
//  D has 1×1 and 2×2 blocks; its inertia is that of A (Sylvester).
// =============================================================================
class DenseBK {
public:
   // Factorize A (only its lower triangle is read).  false = exactly singular.
   bool factorize(Mat A) {
      n_ = (int)A.rows();
      f_ = std::move(A);
      neg_ = 0;
      if (n_ == 0) return true;
      if (!f_.allFinite()) return false;
      ipiv_.assign(n_, 0);
      int info = 0, lwork = -1;
      double wq = 0.0;
      dsytrf_("L", &n_, f_.data(), &n_, ipiv_.data(), &wq, &lwork, &info);
      lwork = std::max(1, (int)wq);
      std::vector<double> work(lwork);
      dsytrf_("L", &n_, f_.data(), &n_, ipiv_.data(), work.data(), &lwork, &info);
      if (info != 0) return false;   // > 0: a zero pivot
      for (int i = 0; i < n_;) {
         if (ipiv_[i] > 0) {           // 1×1 block
            if (f_(i, i) < 0.0) ++neg_;
            i += 1;
         } else {                      // 2×2 block [a b; b c]
            const double a = f_(i, i), b = f_(i + 1, i), c = f_(i + 1, i + 1);
            const double det = a * c - b * b;
            if (det < 0.0) neg_ += 1;
            else if (a + c < 0.0) neg_ += 2;
            i += 2;
         }
      }
      return true;
   }
   void solve(Mat& B) const {   // in place, B ← A⁻¹B
      if (n_ == 0 || B.cols() == 0) return;
      int nrhs = (int)B.cols(), info = 0;
      dsytrs_("L", &n_, &nrhs, f_.data(), &n_, ipiv_.data(), B.data(), &n_, &info);
   }
   void solve(Vec& b) const {
      if (n_ == 0) return;
      int one = 1, info = 0;
      dsytrs_("L", &n_, &one, f_.data(), &n_, ipiv_.data(), b.data(), &n_, &info);
   }
   int negative() const { return neg_; }

private:
   int n_ = 0, neg_ = 0;
   Mat f_;
   std::vector<int> ipiv_;
};

// =============================================================================
//  Schwarz — additive one-level preconditioner  M⁻¹ r = Σ_k N_k M_k⁻¹ N_kᵀ r.
//  Each M_k is a small dense block on the border indices idx_k.  An SPD block
//  is applied by Cholesky.  Otherwise its eigenvalues are clipped to be
//  positive: CG needs a symmetric positive definite M.  No blocks = identity.
// =============================================================================
class Schwarz {
public:
   void reset(int p) {
      p_ = p;
      blocks_.clear();
   }
   // Returns whether M was SPD.
   bool add_block(const std::vector<int>& idx, const Mat& M) {
      Block b;
      b.idx = idx;
      Eigen::LLT<Mat> chol(M);
      if (chol.info() == Eigen::Success) {
         b.llt = std::move(chol);
         b.spd = true;
      } else {
         Eigen::SelfAdjointEigenSolver<Mat> es(M);
         Vec d = es.eigenvalues();
         const double scale = std::max(d.cwiseAbs().maxCoeff(), 1e-300);
         for (int i = 0; i < d.size(); ++i) d[i] = 1.0 / std::max(d[i], 1e-8 * scale);
         b.inv = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
         b.spd = false;
      }
      blocks_.push_back(std::move(b));
      return blocks_.back().spd;
   }
   void apply(const Vec& r, Vec& z) const {
      if (blocks_.empty()) { z = r; return; }
      z.setZero(p_);
      for (const Block& b : blocks_) {
         Vec rk(b.idx.size());
         for (size_t i = 0; i < b.idx.size(); ++i) rk[i] = r[b.idx[i]];
         const Vec zk = b.spd ? Vec(b.llt.solve(rk)) : Vec(b.inv * rk);
         for (size_t i = 0; i < b.idx.size(); ++i) z[b.idx[i]] += zk[i];
      }
   }

private:
   struct Block {
      std::vector<int> idx;
      Eigen::LLT<Mat> llt;
      Mat inv;
      bool spd = true;
   };
   int p_ = 0;
   std::vector<Block> blocks_;
};

// =============================================================================
//  pcg — preconditioned conjugate gradients from x = 0.
//  Stops at ‖r‖ ≤ tol·‖b‖ or maxit.  If it meets pᵀSp ≤ 0, it stops and
//  reports a breakdown: S is not positive definite (Sec. 3.5).
// =============================================================================
struct CgResult {
   int iters = 0;
   double rel = 0.0;
   bool breakdown = false;
};

template <class ApplyS>
CgResult pcg(ApplyS&& applyS, const Schwarz& M, const Vec& b, Vec& x, double tol, int maxit) {
   const int n = (int)b.size();
   CgResult res;
   x.setZero(n);
   const double bnorm = b.norm();
   if (bnorm == 0.0) return res;
   Vec r = b, z(n), Ap(n);
   M.apply(r, z);
   Vec p = z;
   double rz = r.dot(z);
   for (; res.iters < maxit; ++res.iters) {
      if (r.norm() <= tol * bnorm) break;
      applyS(p, Ap);
      const double pAp = p.dot(Ap);
      if (!(pAp > 0.0)) { res.breakdown = true; break; }
      const double a = rz / pAp;
      x += a * p;
      r -= a * Ap;
      M.apply(r, z);
      const double rz_new = r.dot(z);
      p = z + (rz_new / rz) * p;
      rz = rz_new;
   }
   res.rel = r.norm() / bnorm;
   return res;
}

// =============================================================================
//  SchurDD — the solver:
//    set_structure()  once: who owns what, routing of every triplet
//    values()         IPOPT writes the matrix entries here
//    factorize()      every new matrix: W_k, S_k, S or the preconditioner
//    solve()          every right-hand side, in place
// =============================================================================
class SchurDD {
public:
   enum class Interface { Direct, Pcg };
   enum class Precond { None, BJ, AS, ASd };
   struct Options {
      Interface interface = Interface::Direct;
      Precond precond = Precond::ASd;
      double cg_tol = 1e-10;
      int cg_maxit = 2000;
      bool matvec_backsolve = false;   // S·v by eq. (12) instead of (15)
      bool local_inertia = false;      // enforce In(S_k) = (p_k,0,0), eq. (26)
      bool pcg_exact_inertia = false;  // diagnostic: In(S) from a dense factorization,
                                       // so PCG mode gets the same inertia as direct
      int check_max_dim = 0;           // > 0: compare with a dense solve of the full KKT
      int verbose = 0;                 // 1: per factorization, 2: + per solve, 3: + spectra of S_k
   };
   struct Stats {
      // partition (paper Table 2)
      int n_tiles = 0, dim = 0, p = 0, max_pk = 0, max_tk = 0;
      // counters
      long factorizations = 0, singular = 0, s_not_pd = 0, local_not_pd = 0;
      long solves = 0, cg_iters = 0, cg_max = 0, cg_breakdowns = 0, cg_unconverged = 0;
      long checks = 0, check_inertia_mismatch = 0;
      double check_max_err = 0.0;
      double t_factor = 0.0, t_solve = 0.0;
   };

   // ---------------------------------------------------------------- structure
   bool set_structure(int dim, const std::vector<int>& irow, const std::vector<int>& jcol,
                      const std::vector<int>& owner, int n_tiles, const Options& opt) {
      if ((int)owner.size() != dim) {
         std::fprintf(stderr, "[dd] owner map has %zu entries, KKT has %d\n", owner.size(), dim);
         return false;
      }
      opt_ = opt;
      dim_ = dim;
      irow_ = irow;
      jcol_ = jcol;
      values_.assign(irow.size(), 0.0);
      tiles_.assign(n_tiles, Tile());
      c_.clear();

      // local numbering: tile unknowns in ascending KKT order, border likewise
      loc_.assign(dim, -1);
      border_.clear();
      for (int i = 0; i < dim; ++i) {
         if (owner[i] < 0) {
            loc_[i] = (int)border_.size();
            border_.push_back(i);
         } else {
            Tile& T = tiles_[owner[i]];
            loc_[i] = (int)T.glob.size();
            T.glob.push_back(i);
         }
      }
      owner_ = owner;
      const int p = (int)border_.size();

      // N_k: the border unknowns that tile k touches (through B_k)
      std::vector<int> pos(p, -1);
      for (size_t t = 0; t < irow.size(); ++t) {
         int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0 && owner[r] != owner[c]) {
            std::fprintf(stderr, "[dd] KKT entry (%d,%d) couples tiles %d and %d\n", r, c,
                         owner[r], owner[c]);
            return false;
         }
         if ((owner[r] < 0) != (owner[c] < 0)) {
            const int k = std::max(owner[r], owner[c]), b = owner[r] < 0 ? r : c;
            auto& nk = tiles_[k].nk;
            if (std::find(nk.begin(), nk.end(), loc_[b]) == nk.end()) nk.push_back(loc_[b]);
         }
      }
      for (Tile& T : tiles_) std::sort(T.nk.begin(), T.nk.end());

      // route every triplet to W_k, B_k or C
      for (size_t t = 0; t < irow.size(); ++t) {
         const int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0) {
            tiles_[owner[r]].w.push_back({loc_[r], loc_[c], (int)t});
         } else if (owner[r] < 0 && owner[c] < 0) {
            c_.push_back({loc_[r], loc_[c], (int)t});
         } else {
            const int k = std::max(owner[r], owner[c]);
            const int b = owner[r] < 0 ? r : c, l = owner[r] < 0 ? c : r;
            const auto& nk = tiles_[k].nk;
            const int a = (int)(std::lower_bound(nk.begin(), nk.end(), loc_[b]) - nk.begin());
            tiles_[k].b.push_back({a, loc_[l], (int)t});
         }
      }

      // multiplicity of each y (how many tiles touch it) for splitting C
      mult_.assign(p, 0);
      for (const Tile& T : tiles_)
         for (int a : T.nk) ++mult_[a];

      st_->n_tiles = n_tiles;
      st_->dim = dim;
      st_->p = p;
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
   const Stats& stats() const { return *st_; }
   // Counters go to *sink (shared by several solver instances) instead of our own.
   void set_stats_sink(Stats* sink) { st_ = sink ? sink : &own_stats_; }

   // ---------------------------------------------------------------- factorize
   // false = singular (some W_k or, in direct mode, S has a zero pivot).
   bool factorize() {
      const auto t0 = Clock::now();
      ++st_->factorizations;
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      fresh_ = true;

      // ---- each tile: W_k → In(W_k), X_k = W_k⁻¹B_kᵀ, S_k = −B_k X_k   (14)
      std::vector<char> ok(K, 1);
#pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < K; ++k) {
         Tile& T = tiles_[k];
         const int tk = (int)T.glob.size(), pk = (int)T.nk.size();
         Mat W = Mat::Zero(tk, tk);
         for (const auto& e : T.w) add_sym(W, e[0], e[1], values_[e[2]]);
         if (!T.W.factorize(std::move(W))) { ok[k] = 0; continue; }
         T.B = Mat::Zero(pk, tk);
         for (const auto& e : T.b) T.B(e[0], e[1]) += values_[e[2]];
         T.X = T.B.transpose();
         T.W.solve(T.X);
         T.S = -T.B * T.X;
         T.S = 0.5 * (T.S + T.S.transpose());
      }
      if (opt_.verbose >= 3) print_local_spectra();
      int negW = 0;
      for (int k = 0; k < K; ++k) {
         if (!ok[k]) {
            ++st_->singular;
            if (opt_.verbose) std::printf("[dd] fact %ld: W_%d singular\n", st_->factorizations, k);
            st_->t_factor += secs(t0);
            return false;
         }
         negW += tiles_[k].W.negative();
      }

      // ---- the border block C (= δ_w·I for the toy) and diag(S)
      Cmat_ = Mat::Zero(p, p);
      for (const auto& e : c_) add_sym(Cmat_, e[0], e[1], values_[e[2]]);
      diagS_ = Cmat_.diagonal();
      for (const Tile& T : tiles_)
         for (size_t a = 0; a < T.nk.size(); ++a) diagS_[T.nk[a]] += T.S(a, a);

      int negS = 0;
      bool spd_ok = true;
      if (p > 0 && opt_.interface == Interface::Direct) {
         // ---- xSC: assemble S (15) and factorize it; exact In(S)          (25)
         Mat S = assemble_S();
         if (!Sbk_.factorize(std::move(S))) {
            ++st_->singular;
            st_->t_factor += secs(t0);
            if (opt_.verbose) std::printf("[dd] fact %ld: S singular\n", st_->factorizations);
            return false;
         }
         negS = Sbk_.negative();
         if (negS > 0) ++st_->s_not_pd;
      } else if (p > 0) {
         // ---- PCG: build the preconditioner                (17), (19), (21)
         M_.reset(p);
         int not_pd = 0;
         if (opt_.precond == Precond::AS) {
            const Mat S = assemble_S();
            for (const Tile& T : tiles_) {
               const int pk = (int)T.nk.size();
               if (pk == 0) continue;
               Mat St(pk, pk);   // S̃_k = N_kᵀ S N_k
               for (int a = 0; a < pk; ++a)
                  for (int b = 0; b < pk; ++b) St(a, b) = S(T.nk[a], T.nk[b]);
               if (!M_.add_block(T.nk, St)) ++not_pd;
            }
         } else if (opt_.precond != Precond::None) {
            for (const Tile& T : tiles_) {
               const int pk = (int)T.nk.size();
               if (pk == 0) continue;
               Mat Mk = T.S;
               for (int a = 0; a < pk; ++a) {
                  const int i = T.nk[a];
                  Mk(a, a) = opt_.precond == Precond::ASd ? diagS_[i]   // (20)
                                                          : Mk(a, a) + Cmat_(i, i) / mult_[i];
               }
               if (!M_.add_block(T.nk, Mk)) ++not_pd;
            }
         }
         st_->local_not_pd += not_pd;
         if (opt_.pcg_exact_inertia) {
            DenseBK bk;
            if (bk.factorize(assemble_S())) negS = bk.negative();
            else negS = 1;
            if (negS > 0) ++st_->s_not_pd;
         }
         // optional local inertia check (26): every S_k (+ its share of C) SPD
         if (opt_.local_inertia) {
            for (const Tile& T : tiles_) {
               Mat Mk = T.S;
               for (size_t a = 0; a < T.nk.size(); ++a)
                  Mk(a, a) += Cmat_(T.nk[a], T.nk[a]) / mult_[T.nk[a]];
               if (Mk.size() && Eigen::LLT<Mat>(Mk).info() != Eigen::Success) spd_ok = false;
            }
            if (!spd_ok) negS = 1;   // not a count: just "In(S) cannot be certified"
         }
      }
      neg_ = negW + negS;

      if (opt_.check_max_dim > 0 && dim_ <= opt_.check_max_dim) dense_reference();

      if (opt_.verbose) {
         // C = δ_w·I in the Newton systems (1·I in IPOPT's initial least-squares
         // multiplier system, which also goes through this solver)
         const double dw = p > 0 ? Cmat_.diagonal().mean() : 0.0;
         std::printf("[dd] fact %3ld  C_yy=%-9.2e Σ#neg(W_k)=%d", st_->factorizations, dw, negW);
         if (p > 0 && (opt_.interface == Interface::Direct || opt_.pcg_exact_inertia))
            std::printf("  #neg(S)=%d", negS);
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
      const Vec r0 = R;

      // z_k = W_k⁻¹ r_k;   r_S = r_y − Σ N_k B_k z_k                      (10)
      std::vector<Vec> z(K);
      Vec rS(p);
      for (int i = 0; i < p; ++i) rS[i] = R[border_[i]];
      for (int k = 0; k < K; ++k) {
         const Tile& T = tiles_[k];
         z[k].resize(T.glob.size());
         for (size_t l = 0; l < T.glob.size(); ++l) z[k][l] = R[T.glob[l]];
         T.W.solve(z[k]);
         const Vec Bz = T.B * z[k];
         for (size_t a = 0; a < T.nk.size(); ++a) rS[T.nk[a]] -= Bz[a];
      }

      // S u_y = r_S
      Vec uy = Vec::Zero(p);
      bool good = true;
      if (p > 0 && opt_.interface == Interface::Direct) {
         uy = rS;
         Sbk_.solve(uy);
      } else if (p > 0) {
         auto applyS = [this](const Vec& v, Vec& out) { apply_S(v, out); };
         const CgResult cg = pcg(applyS, M_, rS, uy, opt_.cg_tol, opt_.cg_maxit);
         st_->cg_iters += cg.iters;
         st_->cg_max = std::max<long>(st_->cg_max, cg.iters);
         if (cg.breakdown) { ++st_->cg_breakdowns; good = false; }
         else if (cg.rel > opt_.cg_tol) ++st_->cg_unconverged;
         if (opt_.verbose >= 2)
            std::printf("[dd]      solve: pcg its=%4d  rel.res=%.1e%s\n", cg.iters, cg.rel,
                        cg.breakdown ? "  NEGATIVE CURVATURE" : "");
      }

      // u_k = W_k⁻¹(r_k − B_kᵀ N_kᵀ u_y) = z_k − X_k N_kᵀ u_y              (11)
      for (int i = 0; i < p; ++i) R[border_[i]] = uy[i];
      for (int k = 0; k < K; ++k) {
         const Tile& T = tiles_[k];
         Vec uyk(T.nk.size());
         for (size_t a = 0; a < T.nk.size(); ++a) uyk[a] = uy[T.nk[a]];
         const Vec uk = z[k] - T.X * uyk;
         for (size_t l = 0; l < T.glob.size(); ++l) R[T.glob[l]] = uk[l];
      }

      if (fresh_ && dense_lu_.rows() == dim_ && dim_ > 0) {
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
   static void add_sym(Mat& A, int i, int j, double v) {
      A(i, j) += v;
      if (i != j) A(j, i) += v;
   }

   struct Tile {
      std::vector<int> glob;                 // KKT index of each tile unknown
      std::vector<int> nk;                   // border indices touched: N_k
      std::vector<std::array<int, 3>> w, b;  // routed triplets (row, col, triplet)
      DenseBK W;                             // W_k factorization
      Mat B, X, S;                           // B_k, W_k⁻¹B_kᵀ, S_k
   };

   // S = C + Σ_k N_k S_k N_kᵀ (15), dense
   Mat assemble_S() const {
      Mat S = Cmat_;
      for (const Tile& T : tiles_)
         for (size_t a = 0; a < T.nk.size(); ++a)
            for (size_t b = 0; b < T.nk.size(); ++b) S(T.nk[a], T.nk[b]) += T.S(a, b);
      return S;
   }

   // out = S v, tile by tile: (15) with the stored S_k, or (12) by back-solves
   void apply_S(const Vec& v, Vec& out) const {
      out = Cmat_ * v;
      for (const Tile& T : tiles_) {
         const int pk = (int)T.nk.size();
         if (pk == 0) continue;
         Vec vk(pk);
         for (int a = 0; a < pk; ++a) vk[a] = v[T.nk[a]];
         Vec sk;
         if (opt_.matvec_backsolve) {
            Vec w = T.B.transpose() * vk;
            T.W.solve(w);
            sk = -(T.B * w);
         } else {
            sk = T.S * vk;
         }
         for (int a = 0; a < pk; ++a) out[T.nk[a]] += sk[a];
      }
   }

   // --verbose 3: the spectrum of every S_k.  S_k is the Hessian of tile k's
   // value function in its copies, so it is only positive SEMIdefinite when some
   // combination of copies costs the tile nothing (e.g. the two foreign copies
   // seen by the row at an interior tile corner, moved in opposite directions).
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
      for (size_t t = 0; t < irow_.size(); ++t) add_sym(A, irow_[t], jcol_[t], values_[t]);
      Eigen::SelfAdjointEigenSolver<Mat> es(A, Eigen::EigenvaluesOnly);
      const Vec ev = es.eigenvalues();
      const double tiny = 1e-12 * std::max(ev.cwiseAbs().maxCoeff(), 1.0);
      dense_neg_ = 0;
      for (int i = 0; i < ev.size(); ++i) dense_neg_ += ev[i] < -tiny;
      if (dense_neg_ != neg_) ++st_->check_inertia_mismatch;
      dense_lu_.compute(A);
   }

   Options opt_;
   int dim_ = 0, neg_ = 0, dense_neg_ = 0;
   bool fresh_ = false;
   std::vector<int> irow_, jcol_, owner_, loc_, border_, mult_;
   std::vector<double> values_;
   std::vector<Tile> tiles_;
   std::vector<std::array<int, 3>> c_;
   Mat Cmat_;
   Vec diagS_;
   DenseBK Sbk_;
   Schwarz M_;
   Eigen::PartialPivLU<Mat> dense_lu_;
   Stats own_stats_;
   Stats* st_ = &own_stats_;
};

}  // namespace dd
