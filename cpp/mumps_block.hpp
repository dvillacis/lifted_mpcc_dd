// mumps_block.hpp — one tile block W_k factorized by MUMPS (sequential), with
// the local Schur complement S_k = −B_k W_k⁻¹ B_kᵀ returned by the
// factorization itself.  Built only with DD_HAVE_MUMPS (see build.sh).
//
// MUMPS eliminates the interior of the AUGMENTED tile matrix
//
//     A_k = [ W_k   B_kᵀ ]     Schur variables = the p_k appended border rows
//           [ B_k   0    ]
//
// and returns its Schur complement 0 − B_k W_k⁻¹ B_kᵀ = S_k (ICNTL(19)): BLAS-3
// frontal kernels instead of p_k triangular solves.  It pivots (1×1 and 2×2),
// so no level order and no fallback are needed, and INFOG(12) is the number of
// negative pivots of the factored interior, i.e. #neg(W_k) — exact as long as
// no ScaLAPACK root is used (ICNTL(13) = 1).
//
// Solves with W_k use JOB=3 with ICNTL(26)=0: the interior problem of A_k, the
// Schur rows of the right-hand side zeroed on the way in and dropped on the way
// out.  The two tile solves of a Newton solve go through MUMPS's condensation
// and expansion instead (reduce(), expand(): ICNTL(26)=1, then 2): one forward
// sweep returns B_k W_k⁻¹ r_k, one backward sweep returns W_k⁻¹(r_k − B_kᵀ y),
// half the work of two full solves.
//
// PIVOT THRESHOLD.  CNTL(1) = 1e-5 instead of MUMPS's 0.01.  At 0.01 about
// 15% of the pivots of a tile were delayed (the zero-diagonal rows of the KKT
// block), which grows the fronts.  Measured (same iterations, α*, PSNR at
// every threshold):
//
//    CNTL(1)                      0.01    1e-4    1e-5    1e-6
//    cam64 4×4: flops (10⁹)       7.73    5.21    4.55    4.16
//               delayed pivots    689k    291k    119k    3.3k
//    mumps_check max rel. error   6e-12   1e-10   1e-10   1.3e-9 (fails its 1e-9)
//
// 1e-6 (IPOPT's mumps_pivtol) gains little more and costs accuracy: IPOPT
// makes up for it with iterative refinement, which tv_dd does not do.
//
// THREADS.  MUMPS's C interface is not thread-safe, even across separate
// instances: every dmumps_c call hands its arrays to the Fortran side through
// global static variables, and an instance table is shared.  Measured with
// mumps_check (MUMPS 5.9.1): concurrent calls crash or abort with "Instance
// Error ... MPI_ABORT", even with instances created serially.  So every call
// here takes one process-wide lock: MUMPS tiles are factorized and solved one
// at a time, while the rest of the per-tile work stays parallel.
//
// The layout of the returned Schur array is checked rather than assumed:
// MUMPS 5.x fills one triangle of the p×p array; self_test() compares against
// a dense reference (tv_dd runs it once at startup, mumps_check on demand).
#pragma once

#ifdef DD_HAVE_MUMPS

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include <Eigen/Dense>

#include "dmumps_c.h"

namespace dd {

class MumpsBlock {
public:
   MumpsBlock() {
      std::memset(&id_, 0, sizeof(id_));
      id_.comm_fortran = -987654;   // USE_COMM_WORLD: the sequential libseq stub
      id_.par = 1;                  // the host works
      id_.sym = 2;                  // general symmetric (indefinite)
      id_.job = -1;                 // initialize
      call();
      alive_ = infog(1) >= 0;
      icntl(1) = -1;   // error stream: off
      icntl(2) = -1;   // diagnostics: off
      icntl(3) = -1;   // global info: off
      icntl(4) = 0;    // print level: none
      icntl(13) = 1;   // no ScaLAPACK root: INFOG(12) stays exact
      icntl(24) = 1;   // detect null pivots (report singular, don't abort)
      id_.cntl[0] = 1e-5;   // CNTL(1), relative pivot threshold (see above)
      icntl(49) = 1;   // after each factorization, move the factors into an array of their
                       // own size and free the workspace (N=256 16×16, 8 ranks: peak RSS
                       // 380 → 331 MB/rank, same iterations)
   }
   ~MumpsBlock() {
      if (alive_) {
         id_.job = -2;
         call();
      }
   }
   MumpsBlock(const MumpsBlock&) = delete;
   MumpsBlock& operator=(const MumpsBlock&) = delete;

   // Pattern of the augmented matrix, 0-based, any triangle (folded to the
   // lower one here), duplicates allowed: interior 0..n−1, Schur n..n+p−1.
   // Call once; afterwards only values change (same order as r/c).
   bool analyze(int n, int p, const std::vector<int>& r, const std::vector<int>& c) {
      if (!alive_) return false;
      n_ = n;
      p_ = p;
      const int na = n + p;
      nuser_ = (int)r.size();
      irn_.clear();
      jcn_.clear();
      for (size_t t = 0; t < r.size(); ++t) {
         irn_.push_back(std::max(r[t], c[t]) + 1);
         jcn_.push_back(std::min(r[t], c[t]) + 1);
      }
      // a zero diagonal for every variable, so each is structurally present
      for (int i = 1; i <= na; ++i) { irn_.push_back(i); jcn_.push_back(i); }
      a_.assign(irn_.size(), 0.0);
      id_.n = na;
      id_.nnz = (MUMPS_INT8)irn_.size();
      id_.irn = irn_.data();
      id_.jcn = jcn_.data();
      id_.a = a_.data();
      if (p_ > 0) {
         listvar_.resize(p_);
         for (int j = 0; j < p_; ++j) listvar_[j] = n_ + 1 + j;
         schur_.assign((size_t)p_ * p_, 0.0);
         id_.size_schur = p_;
         id_.listvar_schur = listvar_.data();
         id_.schur = schur_.data();
         icntl(19) = 1;   // centralized Schur complement on the host
      } else {
         icntl(19) = 0;
      }
      id_.job = 1;
      call();
      return infog(1) >= 0 && !(infog(1) > 0 && (infog(1) & 1));   // +1: dropped entries
   }

   // Factorize with the values of the user entries (in analyze()'s order).
   // false: singular (zero or null pivot), or MUMPS could not finish.
   bool factorize(const double* user_values) {
      std::copy(user_values, user_values + nuser_, a_.begin());
      std::fill(a_.begin() + nuser_, a_.end(), 0.0);
      neg_ = 0;
      for (int attempt = 0; attempt < 6; ++attempt) {
         id_.job = 2;
         call();
         const int st = infog(1);
         if (st == -8 || st == -9 || st == -19) {   // workspace too small: retry larger
            icntl(14) = std::max(2 * icntl(14), 40);
            continue;
         }
         if (st < 0) return false;                  // −10 singular, −13 memory, ...
         if (infog(28) > 0) return false;           // null pivots
         neg_ = infog(12);
         // MUMPS fills one triangle of the p×p array: symmetrize (the other
         // triangle is checked to be zero by self_test()).
         for (int i = 0; i < p_; ++i)
            for (int j = 0; j < i; ++j) {
               double& lo = schur_[(size_t)i * p_ + j];
               double& up = schur_[(size_t)j * p_ + i];
               const double v = lo != 0.0 ? lo : up;
               lo = up = v;
            }
         return true;
      }
      return false;
   }

   int negative() const { return neg_; }
   // S_k (p × p, symmetric), valid after a successful factorize()
   const double* schur() const { return schur_.data(); }

   // In-place solve W x = b for nrhs columns (leading dimension n).
   bool solve(double* b, int nrhs = 1) {
      const int na = n_ + p_;
      buf_.assign((size_t)na * nrhs, 0.0);
      for (int k = 0; k < nrhs; ++k)
         std::copy(b + (size_t)k * n_, b + (size_t)(k + 1) * n_, buf_.begin() + (size_t)k * na);
      id_.rhs = buf_.data();
      id_.nrhs = nrhs;
      id_.lrhs = na;
      icntl(26) = 0;   // the interior problem only
      id_.job = 3;
      call();
      if (infog(1) < 0) return false;
      for (int k = 0; k < nrhs; ++k)
         std::copy(buf_.begin() + (size_t)k * na, buf_.begin() + (size_t)k * na + n_,
                   b + (size_t)k * n_);
      return true;
   }

   // Condensation (JOB=3, ICNTL(26)=1): for b on the interior (length n),
   // red = 0 − B W⁻¹ b (length p), by forward elimination only.  Must be
   // followed by expand() on this instance, with no other solve in between.
   bool reduce(const double* b, double* red) {
      const int na = n_ + p_;
      buf_.assign(na, 0.0);
      std::copy(b, b + n_, buf_.begin());
      red_.assign(p_, 0.0);
      id_.rhs = buf_.data();
      id_.nrhs = 1;
      id_.lrhs = na;
      id_.redrhs = red_.data();
      id_.lredrhs = p_;
      icntl(26) = 1;
      id_.job = 3;
      call();
      icntl(26) = 0;
      if (infog(1) < 0) return false;
      std::copy(red_.begin(), red_.end(), red);
      return true;
   }
   // Expansion (JOB=3, ICNTL(26)=2) after reduce(b): x = W⁻¹(b − Bᵀ y) for
   // the Schur unknowns y (length p); x has length n.
   bool expand(const double* y, double* x) {
      const int na = n_ + p_;
      std::copy(y, y + p_, red_.begin());
      id_.rhs = buf_.data();
      id_.nrhs = 1;
      id_.lrhs = na;
      id_.redrhs = red_.data();
      id_.lredrhs = p_;
      icntl(26) = 2;
      id_.job = 3;
      call();
      icntl(26) = 0;
      if (infog(1) < 0) return false;
      std::copy(buf_.begin(), buf_.begin() + n_, x);
      return true;
   }

   // Once per process: factorize a small KKT block with a known Schur
   // complement and inertia, and compare.  Guards against a MUMPS build whose
   // Schur layout or inertia reporting differs from what this file assumes.
   static bool self_test() {
      // W = [[2, 0, 1], [0, −1, 1], [1, 1, 0]] (1 primal of each sign, 1 dual),
      // B = [[1, 0, 0], [0, 1, 1]] → S = −B W⁻¹ Bᵀ.
      Eigen::Matrix3d W;
      W << 2, 0, 1, 0, -1, 1, 1, 1, 0;
      Eigen::Matrix<double, 2, 3> B;
      B << 1, 0, 0, 0, 1, 1;
      const Eigen::Matrix2d S_ref = -B * W.inverse() * B.transpose();
      const Eigen::Vector3d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(W).eigenvalues();
      const int neg_ref = (int)(ev.array() < 0).count();
      std::vector<int> r, c;
      std::vector<double> v;
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j <= i; ++j)
            if (W(i, j) != 0.0) { r.push_back(i); c.push_back(j); v.push_back(W(i, j)); }
      for (int a = 0; a < 2; ++a)
         for (int l = 0; l < 3; ++l)
            if (B(a, l) != 0.0) { r.push_back(3 + a); c.push_back(l); v.push_back(B(a, l)); }
      MumpsBlock m;
      if (!m.analyze(3, 2, r, c) || !m.factorize(v.data())) return false;
      double err = 0.0;
      for (int i = 0; i < 2; ++i)
         for (int j = 0; j < 2; ++j) err = std::max(err, std::abs(m.schur()[i * 2 + j] - S_ref(i, j)));
      Eigen::Vector3d x(1.0, 2.0, 3.0);
      const Eigen::Vector3d x_ref = W.inverse() * x;
      if (!m.solve(x.data())) return false;
      err = std::max(err, (x - x_ref).cwiseAbs().maxCoeff());
      // condensation and expansion
      const Eigen::Vector3d b(1.0, 2.0, 3.0);
      const Eigen::Vector2d y(0.5, -1.0);
      const Eigen::Vector2d red_ref = -B * W.inverse() * b;
      const Eigen::Vector3d xe_ref = W.inverse() * (b - B.transpose() * y);
      Eigen::Vector2d red;
      Eigen::Vector3d xe;
      if (!m.reduce(b.data(), red.data()) || !m.expand(y.data(), xe.data())) return false;
      err = std::max(err, (red - red_ref).cwiseAbs().maxCoeff());
      err = std::max(err, (xe - xe_ref).cwiseAbs().maxCoeff());
      return err < 1e-12 && m.negative() == neg_ref;
   }

   const char* version() const { return id_.version_number; }

private:
   // Every MUMPS call goes through here, under one process-wide lock.
   void call() {
      static std::mutex lock;
      std::lock_guard<std::mutex> guard(lock);
      dmumps_c(&id_);
   }
   MUMPS_INT& icntl(int i) { return id_.icntl[i - 1]; }
   int infog(int i) const { return (int)id_.infog[i - 1]; }

   DMUMPS_STRUC_C id_;
   bool alive_ = false;
   int n_ = 0, p_ = 0, nuser_ = 0, neg_ = 0;
   std::vector<MUMPS_INT> irn_, jcn_, listvar_;
   std::vector<double> a_, schur_, buf_, red_;
};

}  // namespace dd

#endif  // DD_HAVE_MUMPS
