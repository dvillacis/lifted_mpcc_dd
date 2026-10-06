// blocks.hpp — factorizations of one tile block W_k, and of S.
//
// What the decomposition needs from a factorization of W_k:
//   · its inertia (eq. 24),
//   · solves W_k⁻¹ r,
//   · its local Schur complement S_k = −B_k W_k⁻¹ B_kᵀ (eq. 14).
//
// Two backends:
//
//   DenseBK    dense Bunch–Kaufman (LAPACK dsytrf).  Pivoted and exact, O(t_k³).
//              The reference; fine for tiles of a few hundred unknowns.
//
//   SparseKKT  sparse LDLᵀ (Eigen, no pivoting) in a KKT-aware order.  A static
//              order must avoid zero pivots: W_k has zero diagonals on every
//              constraint row, and in this problem also on every foreign copy
//              (no Hessian entry), so the pair (copy, linking row) is [0 1; 1 0].
//              The unknowns are therefore eliminated in four levels:
//                 L1 primals with a nonzero diagonal (Hessian + δ_w + Σ)
//                 L2 constraint rows coupled to some L1 primal
//                 L3 the other primals
//                 L4 the other constraint rows
//              each level in the relative order of an AMD ordering of the whole
//              block, except that every L4 row comes right after its L3 partner.
//              Eliminating L1 puts −J H⁻¹ Jᵀ on the L2 diagonal; that in turn
//              puts a nonzero on the L3 copies.  Not on all of them: at an
//              interior tile corner two foreign copies sit in one row only, so
//              their block is rank one and the second one would get an exact zero
//              pivot (the corner null mode of S_k again).  Taking each copy
//              together with its linking row fixes that, since the pair
//              [m 1; 1 0] has determinant −1: the pivots are m, then −1/m, which
//              restores the partner's diagonal.  Since LDLᵀ is a congruence the
//              signs of D give the exact inertia (Sylvester).  A pivot that is
//              still exactly zero sends that factorization to DenseBK (TileBlock).
//              The levels depend on which diagonals are zero (δ_w switches), so
//              the order is recomputed only when they change.
//
//              S_k needs only FORWARD solves: S_k = −Yᵀ D⁻¹ Y with Y = L⁻¹ P B_kᵀ,
//              and the nonzeros of B_kᵀ (the linking rows) sit in L4, at the end
//              of the order, so Y is nonzero only in the last rows.
//
//   SparseLDLT for the assembled S (direct mode, exact inertia): Eigen LDLᵀ with
//              AMD.  S is positive definite whenever the inertia is right, so
//              no pivoting is needed there; a breakdown is reported as singular.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "mumps_block.hpp"

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
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Entry = std::array<int, 3>;   // (row, col, slot in the KKT value array)

// =============================================================================
//  DenseBK — dense symmetric-indefinite LDLᵀ with Bunch–Kaufman pivoting.
// =============================================================================
class DenseBK {
public:
   bool factorize(Mat A) {   // only the lower triangle is read; false = singular
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
      if (info != 0) return false;
      for (int i = 0; i < n_;) {
         if (ipiv_[i] > 0) {                     // 1×1 block
            if (f_(i, i) < 0.0) ++neg_;
            i += 1;
         } else {                                // 2×2 block [a b; b c]
            const double a = f_(i, i), b = f_(i + 1, i), c = f_(i + 1, i + 1);
            const double det = a * c - b * b;
            if (det < 0.0) neg_ += 1;
            else if (a + c < 0.0) neg_ += 2;
            i += 2;
         }
      }
      return true;
   }
   void solve(Mat& B) const {
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

// Eigen keeps L and D protected; this exposes L (unit lower, by columns).
template <class Ordering>
struct OpenLDLT : Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Ordering> {
   const SpMat& rawL() const { return this->m_matrix; }
};

// Count negative pivots; false if a pivot is zero or not finite.
inline bool count_pivots(const Vec& d, int& neg) {
   neg = 0;
   for (int i = 0; i < d.size(); ++i) {
      if (!std::isfinite(d[i]) || d[i] == 0.0) return false;
      if (d[i] < 0.0) ++neg;
   }
   return true;
}

// =============================================================================
//  SparseKKT — sparse LDLᵀ of a KKT block in the level order described above.
// =============================================================================
class SparseKKT {
public:
   // n unknowns; entries (r, c, slot) in local numbering, any triangle,
   // duplicates allowed; primal[i] says whether unknown i is a primal variable.
   void set_pattern(int n, const std::vector<Entry>& w, std::vector<char> primal) {
      n_ = n;
      w_ = &w;
      primal_ = std::move(primal);
      // AMD on the symmetric pattern of the whole block
      std::vector<Eigen::Triplet<double>> t;
      t.reserve(2 * w.size());
      for (const Entry& e : w) {
         t.emplace_back(e[0], e[1], 1.0);
         t.emplace_back(e[1], e[0], 1.0);
      }
      SpMat pat(n, n);
      pat.setFromTriplets(t.begin(), t.end());
      adj_.assign(n, {});
      for (const Entry& e : w)
         if (e[0] != e[1]) {
            adj_[e[0]].push_back(e[1]);
            adj_[e[1]].push_back(e[0]);
         }
      Eigen::AMDOrdering<int>::PermutationType perm;
      Eigen::AMDOrdering<int>()(pat, perm);
      amd_.assign(perm.indices().data(), perm.indices().data() + n);   // new → old
      level_.clear();
   }

   bool factorize(const double* values) {
      const std::vector<Entry>& w = *w_;
      // levels from the current diagonal
      std::vector<double> diag(n_, 0.0);
      for (const Entry& e : w)
         if (e[0] == e[1]) diag[e[0]] += values[e[2]];
      std::vector<char> lev(n_, 0);
      for (int i = 0; i < n_; ++i) lev[i] = primal_[i] ? (diag[i] != 0.0 ? 1 : 3) : 4;
      for (const Entry& e : w) {
         const int r = e[0], c = e[1];
         if (r == c) continue;
         if (lev[r] == 1 && !primal_[c]) lev[c] = 2;
         if (lev[c] == 1 && !primal_[r]) lev[r] = 2;
      }
      if (lev != level_) reorder(std::move(lev));

      // numeric values into the fixed pattern
      std::fill(A_.valuePtr(), A_.valuePtr() + A_.nonZeros(), 0.0);
      for (size_t t = 0; t < w.size(); ++t) A_.valuePtr()[pos_[t]] += values[w[t][2]];
      f_.factorize(A_);
      if (f_.info() != Eigen::Success) return false;
      if (!count_pivots(f_.vectorD(), neg_)) return false;
      // A pivot that is not zero but tiny relative to ITS OWN ROW of W_k signals
      // the same cancellation polluted by round-off; without pivoting the
      // factors would be inaccurate.  Refuse it too (TileBlock falls back).
      // The scale is per row on purpose: late in the barrier W_k legitimately
      // spans many orders of magnitude (Σ on active bounds), and a test against
      // the largest pivot of the block would refuse healthy factorizations.
      std::vector<double> rowmax(n_, 0.0);
      for (const Entry& e : w) {
         const double v = std::abs(values[e[2]]);
         rowmax[e[0]] = std::max(rowmax[e[0]], v);
         rowmax[e[1]] = std::max(rowmax[e[1]], v);
      }
      const Vec& D = f_.vectorD();
      for (int i = 0; i < n_; ++i)
         if (std::abs(D[o2n_[i]]) < 1e-14 * rowmax[i]) return false;
      return true;
   }

   int negative() const { return neg_; }

   void solve(Vec& b) const {
      Vec bp(n_);
      for (int i = 0; i < n_; ++i) bp[o2n_[i]] = b[i];
      const Vec xp = f_.solve(bp);
      for (int i = 0; i < n_; ++i) b[i] = xp[o2n_[i]];
   }

   // −B W⁻¹ Bᵀ for B (p × n, sparse), by forward solves only.
   Mat schur(const SpMat& B) const {
      const int p = (int)B.rows();
      if (p == 0) return Mat(0, 0);
      const SpMat Bt = B.transpose();
      int s = n_;
      for (int k = 0; k < Bt.outerSize(); ++k)
         for (SpMat::InnerIterator it(Bt, k); it; ++it) s = std::min(s, o2n_[it.row()]);
      const int m = n_ - s;
      RowMat Y = RowMat::Zero(m, p);   // rows s..n−1 of L⁻¹ P Bᵀ
      for (int k = 0; k < Bt.outerSize(); ++k)
         for (SpMat::InnerIterator it(Bt, k); it; ++it) Y(o2n_[it.row()] - s, k) += it.value();
      const SpMat& L = f_.rawL();
      for (int j = s; j < n_; ++j) {
         const auto yj = Y.row(j - s);
         if (yj.isZero(0.0)) continue;
         for (SpMat::InnerIterator it(L, j); it; ++it)
            if ((int)it.row() > j) Y.row(it.row() - s) -= it.value() * yj;
      }
      const Vec dinv = f_.vectorD().tail(m).cwiseInverse();
      Mat S = -(Y.transpose() * dinv.asDiagonal() * Y);
      return 0.5 * (S + S.transpose());
   }

   long factor_nnz() const { return f_.rawL().nonZeros(); }

private:
   void reorder(std::vector<char> lev) {
      level_ = std::move(lev);
      std::vector<int> order;   // new → old
      order.reserve(n_);
      for (int l = 1; l <= 2; ++l)
         for (int i : amd_)
            if (level_[i] == l) order.push_back(i);
      // L3 in AMD order, each followed by the L4 rows whose L3 partners are all placed
      std::vector<char> placed(n_, 0);
      for (int i : order) placed[i] = 1;
      for (int i : amd_) {
         if (level_[i] != 3) continue;
         order.push_back(i);
         placed[i] = 1;
         for (int j : adj_[i]) {
            if (level_[j] != 4 || placed[j]) continue;
            bool ready = true;
            for (int q : adj_[j]) ready = ready && (level_[q] != 3 || placed[q]);
            if (ready) { order.push_back(j); placed[j] = 1; }
         }
      }
      for (int i : amd_)
         if (!placed[i]) order.push_back(i);
      o2n_.assign(n_, 0);
      for (int k = 0; k < n_; ++k) o2n_[order[k]] = k;

      // pattern of the permuted lower triangle, and where each entry lands
      const std::vector<Entry>& w = *w_;
      std::vector<Eigen::Triplet<double>> t;
      t.reserve(w.size());
      for (const Entry& e : w) {
         int r = o2n_[e[0]], c = o2n_[e[1]];
         if (r < c) std::swap(r, c);
         t.emplace_back(r, c, 0.0);
      }
      A_.resize(n_, n_);
      A_.setFromTriplets(t.begin(), t.end());
      A_.makeCompressed();
      pos_.resize(w.size());
      for (size_t k = 0; k < w.size(); ++k) {
         const int r = t[k].row(), c = t[k].col();
         const int* rows = A_.innerIndexPtr();
         const int lo = A_.outerIndexPtr()[c], hi = A_.outerIndexPtr()[c + 1];
         pos_[k] = (int)(std::lower_bound(rows + lo, rows + hi, r) - rows);
      }
      f_.analyzePattern(A_);
   }

   int n_ = 0, neg_ = 0;
   const std::vector<Entry>* w_ = nullptr;
   std::vector<char> primal_, level_;
   std::vector<int> amd_, o2n_, pos_;
   std::vector<std::vector<int>> adj_;
   SpMat A_;
   OpenLDLT<Eigen::NaturalOrdering<int>> f_;
};

// =============================================================================
//  TileBlock — one W_k with one of three backends:
//
//    sparse  SparseKKT (level-ordered, unpivoted).  When its static order meets
//            a zero or tiny pivot — no static 1×1 order can rule that out: two
//            constraint rows leaning on the same L1 variable give a
//            rank-deficient −J H⁻¹ Jᵀ block, which needs 2×2 pivots — that one
//            factorization FALLS BACK to a pivoted solver, dense Bunch–Kaufman
//            or MUMPS.  The inertia stays exact instead of the IPM regularizing.
//    dense   Bunch–Kaufman on the whole block, always (the reference).
//    mumps   MUMPS on the augmented block [W_k B_kᵀ; B_k 0], which also returns
//            S_k (mumps_block.hpp).  Needs a build with DD_HAVE_MUMPS.
// =============================================================================
enum class Backend { Sparse, Dense, Mumps };

class TileBlock {
public:
   // w: the W_k entries; b: the B_k entries (border row a, tile column l, slot);
   // p: the number of border unknowns the tile touches.
   void set_pattern(Backend backend, Backend fallback, int n, const std::vector<Entry>& w,
                    std::vector<char> primal, const std::vector<Entry>& b, int p) {
      backend_ = backend;
      fallback_ = fallback;
      n_ = n;
      p_ = p;
      w_ = &w;
      b_ = &b;
      if (backend_ == Backend::Sparse) sp_.set_pattern(n, w, std::move(primal));
   }

   bool factorize(const double* values) {
      if (backend_ == Backend::Sparse) {
         if (sp_.factorize(values)) {
            used_ = Backend::Sparse;
            bk_ = DenseBK();   // release a previous dense fallback (t_k² doubles)
            return true;
         }
         ++fallbacks_;
         return factorize_with(fallback_, values);
      }
      return factorize_with(backend_, values);
   }
   int negative() const {
      return used_ == Backend::Sparse ? sp_.negative()
           : used_ == Backend::Dense  ? bk_.negative()
                                      : mumps_negative();
   }
   void solve(Vec& b) const {
      if (used_ == Backend::Sparse) sp_.solve(b);
      else if (used_ == Backend::Dense) bk_.solve(b);
      else mumps_solve(b);
   }
   Mat schur(const SpMat& B) const {
      if (used_ == Backend::Sparse) return sp_.schur(B);
      if (used_ == Backend::Mumps) return mumps_schur();
      const Mat Bd(B);
      Mat X = Bd.transpose();
      bk_.solve(X);
      Mat S = -Bd * X;
      return 0.5 * (S + S.transpose());
   }
   long fallbacks() const { return fallbacks_; }

private:
   bool factorize_with(Backend which, const double* values) {
      used_ = which;
      if (which == Backend::Mumps) return mumps_factorize(values);
      bool ok = false;
      // A dense tile costs t_k² doubles (≈100 MB at t_k = 3500).  Build them one
      // at a time: otherwise every thread holds one and the peak memory
      // multiplies by the thread count.
#pragma omp critical(dd_dense_tile)
      {
         Mat W = Mat::Zero(n_, n_);
         for (const Entry& e : *w_) {
            W(e[0], e[1]) += values[e[2]];
            if (e[0] != e[1]) W(e[1], e[0]) += values[e[2]];
         }
         ok = bk_.factorize(std::move(W));
      }
      return ok;
   }

#ifdef DD_HAVE_MUMPS
   bool mumps_factorize(const double* values) {
      if (!mu_) {   // first use: the augmented pattern, analysed once
         mu_ = std::make_unique<MumpsBlock>();
         std::vector<int> r, c;
         for (const Entry& e : *w_) { r.push_back(e[0]); c.push_back(e[1]); }
         for (const Entry& e : *b_) { r.push_back(n_ + e[0]); c.push_back(e[1]); }
         if (!mu_->analyze(n_, p_, r, c)) return false;
      }
      mval_.resize(w_->size() + b_->size());
      size_t k = 0;
      for (const Entry& e : *w_) mval_[k++] = values[e[2]];
      for (const Entry& e : *b_) mval_[k++] = values[e[2]];
      return mu_->factorize(mval_.data());
   }
   int mumps_negative() const { return mu_->negative(); }
   void mumps_solve(Vec& b) const { mu_->solve(b.data()); }
   Mat mumps_schur() const { return Eigen::Map<const Mat>(mu_->schur(), p_, p_); }
   std::unique_ptr<MumpsBlock> mu_;
   std::vector<double> mval_;
#else
   bool mumps_factorize(const double*) { return false; }
   int mumps_negative() const { return 0; }
   void mumps_solve(Vec&) const {}
   Mat mumps_schur() const { return Mat(); }
#endif

   Backend backend_ = Backend::Sparse, fallback_ = Backend::Dense, used_ = Backend::Sparse;
   int n_ = 0, p_ = 0;
   long fallbacks_ = 0;
   const std::vector<Entry>* w_ = nullptr;
   const std::vector<Entry>* b_ = nullptr;
   SparseKKT sp_;
   DenseBK bk_;
};

// =============================================================================
//  SymFactor — factorization of the assembled S: dense Bunch–Kaufman when it
//  is small (exact, pivoted), sparse LDLᵀ with AMD otherwise.
// =============================================================================
class SymFactor {
public:
   bool factorize(const SpMat& S, int dense_max) {
      dense_ = S.rows() <= dense_max;
      if (dense_) {
         const bool ok = bk_.factorize(Mat(S));   // reads the lower triangle
         neg_ = bk_.negative();
         return ok;
      }
      sp_.compute(S);
      if (sp_.info() != Eigen::Success) return false;
      return count_pivots(sp_.vectorD(), neg_);
   }
   int negative() const { return neg_; }
   void solve(Vec& b) const {
      if (dense_) bk_.solve(b);
      else b = sp_.solve(b);
   }

private:
   bool dense_ = true;
   int neg_ = 0;
   DenseBK bk_;
   Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> sp_;
};

}  // namespace dd
