// A thin RAII wrapper around HSL MA57 for one subdomain block W_k.
//
// Why MA57 and not Eigen: the local blocks of the arrowhead are symmetric
// INDEFINITE and, when IPOPT hands us δ_c = 0, not even quasi-definite.
// Eigen::SimplicialLDLT does no pivoting, so it is only stable on quasi-definite
// matrices — measured, it failed outright on some W_k, cost 1.2–2.7× extra IPOPT
// iterations, and broke down entirely at N=48 (see ../CLAUDE.md). MA57 is a
// Bunch–Kaufman/Duff–Reid multifrontal solver with threshold pivoting: it is
// stable on indefinite matrices and returns the inertia directly, which is exactly
// what the Haynsworth identity needs. It is also what Lueg's paper uses for the
// local blocks (MA27 there).
//
// Analysis (MA57AD) is done ONCE per block, in InitializeStructure: the sparsity
// of W_k is fixed for the whole solve, only the values change between Newton
// steps. Reusing the symbolic factorization is the one form of reuse that pays
// here (the vault records that reusing *numeric* hierarchies backfires).
//
// Matrix input: coordinate form, ONE triangle, 1-based Fortran indices. Duplicate
// entries are summed, which is what IPOPT's triplets may contain anyway.
#ifndef MA57_BLOCK_HPP
#define MA57_BLOCK_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
void ma57id_(double* cntl, int* icntl);
void ma57ad_(const int* n, const int* ne, const int* irn, const int* jcn,
             const int* lkeep, int* keep, int* iwork, const int* icntl,
             int* info, double* rinfo);
void ma57bd_(const int* n, const int* ne, const double* a, double* fact,
             const int* lfact, int* ifact, const int* lifact, const int* lkeep,
             const int* keep, int* ppos, const int* icntl, const double* cntl,
             int* info, double* rinfo);
void ma57cd_(const int* job, const int* n, const double* fact, const int* lfact,
             const int* ifact, const int* lifact, const int* nrhs, double* rhs,
             const int* lrhs, double* w, const int* lw, int* iw1,
             const int* icntl, int* info);
}

class Ma57Block {
public:
   // MA57 INFO() indices, 0-based here (Fortran INFO(k) == info_[k-1]).
   static constexpr int I_STAT = 0;     // INFO(1)  error/warning flag
   static constexpr int I_LFACT = 8;    // INFO(9)  min LFACT for MA57BD
   static constexpr int I_LIFACT = 9;   // INFO(10) min LIFACT for MA57BD
   static constexpr int I_NEIG = 23;    // INFO(24) number of negative eigenvalues
   static constexpr int I_RANK = 24;    // INFO(25) rank of the factorized matrix

   Ma57Block() {
      ma57id_(cntl_, icntl_);
      silence();
      // DD_MA57_NO_SCALING=1: skip MA57's MC64 scaling (ICNTL(15)=0). Profiling
      // (2026-07-22, N=32 k=4 direct) put MC64 at ~13% of the whole run — it
      // re-scales every block at every factorization. EXPERIMENT ONLY: scaling
      // protects pivot quality on the badly-scaled Scholtes-tail KKTs
      // (‖A‖ ~ 1e18); validate any use with DD_CHECK and a trajectory diff.
      if (std::getenv("DD_MA57_NO_SCALING")) icntl_[14] = 0;
   }

   void silence() {
      icntl_[0] = -1;   // error stream   (<0 suppresses)
      icntl_[1] = -1;   // warning stream
      icntl_[2] = -1;   // monitor stream
      icntl_[3] = -1;   // statistics stream
      icntl_[4] = 0;    // print level: none
   }

   // Symbolic analysis. irn/jcn are 1-based, one triangle. Call once.
   //
   // A full set of zero diagonal entries is APPENDED to the user's triplets (MA57
   // sums duplicates, so this is exact): it guarantees every diagonal position is
   // structurally present, which the analysis' pivot ordering keys off. Kept
   // as-is so factorizations stay bit-identical to the validated ../cpp solver.
   bool analyze(int n, const std::vector<int>& irn, const std::vector<int>& jcn) {
      n_ = n; ne_user_ = (int)irn.size(); ne_ = ne_user_ + n;
      irn_ = irn; jcn_ = jcn;
      for (int i = 1; i <= n; ++i) { irn_.push_back(i); jcn_.push_back(i); }
      a_.assign(std::max(ne_, 1), 0.0);
      lkeep_ = 5 * n_ + ne_ + std::max(n_, ne_) + 42;
      keep_.assign(lkeep_, 0);
      std::vector<int> iwork(5 * std::max(n_, 1));
      ma57ad_(&n_, &ne_, irn_.data(), jcn_.data(), &lkeep_, keep_.data(),
              iwork.data(), icntl_, info_, rinfo_);
      if (info_[I_STAT] < 0) return false;
      // Generous initial workspace; MA57BD grows it on INFO(1) = −3/−4 anyway.
      lfact_ = 2 * info_[I_LFACT] + 100;
      lifact_ = 2 * info_[I_LIFACT] + 100;
      fact_.assign(lfact_, 0.0);
      ifact_.assign(lifact_, 0);
      ppos_.assign(std::max(n_, 1), 0);
      return true;
   }

   double* values() { return a_.data(); }          // the user's ne triplets
   int nnz() const { return ne_user_; }
   int dim() const { return n_; }

   // Numeric factorization of the values currently in values(). Retries with a
   // larger workspace on the standard −3 / −4 "LFACT/LIFACT too small" returns.
   bool factorize() {
      for (int attempt = 0; attempt < 6; ++attempt) {
         ma57bd_(&n_, &ne_, a_.data(), fact_.data(), &lfact_, ifact_.data(),
                 &lifact_, &lkeep_, keep_.data(), ppos_.data(), icntl_, cntl_,
                 info_, rinfo_);
         const int st = info_[I_STAT];
         if (st == -3) { lfact_ = (int)(2.0 * lfact_); fact_.assign(lfact_, 0.0); continue; }
         if (st == -4) { lifact_ = (int)(2.0 * lifact_); ifact_.assign(lifact_, 0); continue; }
         return st >= 0;
      }
      return false;
   }

   // In-place solve A x = b for nrhs columns (column-major, leading dimension n).
   //
   // The MA57CD scratch is PERSISTENT (grow-only), not a per-call allocation:
   // the CG interface issues ~10⁶ single-RHS backsolves per run, and a fresh
   // value-initialized vector per call means zeroing ~n doubles a million times
   // (the "MA57 per-solve scratch allocation" cost the old OpenMP measurements
   // flagged). MA57 uses w/iw1 as scratch only, so reuse is safe; each block
   // object is touched by one thread at a time (OMP parallelism is ACROSS
   // blocks), so per-object storage is thread-safe.
   void solve(double* rhs, int nrhs = 1) { solve_job(1, rhs, nrhs); }

   // Partial solves with the SAME factorization: JOB=2 solves LX=B, JOB=3
   // solves DX=B, JOB=4 solves LᵀX=B (JOB=1 is the full solve, what solve()
   // forwards to). Composing 2→3→4 reproduces JOB=1 ONLY when MC64 scaling was
   // off during factorization — with scaling the partials act on the scaled
   // system. Callers composing partials must build the object with
   // scaling_off() and self-check the composition against JOB=1.
   void solve_job(int job, double* rhs, int nrhs = 1) {
      const int lw = n_ * nrhs;
      if ((int)w_.size() < lw) w_.resize(lw);
      if ((int)iw1_.size() < n_) iw1_.resize(n_);
      ma57cd_(&job, &n_, fact_.data(), &lfact_, ifact_.data(), &lifact_, &nrhs,
              rhs, &n_, w_.data(), &lw, iw1_.data(), icntl_, info_);
      // MA57CD reports errors in INFO(1); returning silently would hand the
      // caller garbage. Out-of-contract (the workspace is sized by analyze),
      // so make it loud — once, not per solve (~10⁶ backsolves/run under CG).
      if (info_[I_STAT] < 0) {
         static bool warned = false;
         if (!warned) {
            std::cerr << "[ma57] MA57CD back-solve failed: INFO(1)="
                      << info_[I_STAT] << "\n";
            warned = true;
         }
      }
   }

   // Disable MC64 scaling (ICNTL(15)=0) for THIS instance — required for
   // meaningful JOB=2/3/4 partial solves. Call before analyze/factorize.
   void scaling_off() { icntl_[14] = 0; }

   int negative_eigenvalues() const { return info_[I_NEIG]; }
   int rank() const { return info_[I_RANK]; }
   bool singular() const { return info_[I_RANK] < n_; }
   int status() const { return info_[I_STAT]; }

private:
   int n_ = 0, ne_ = 0, ne_user_ = 0, lkeep_ = 0, lfact_ = 0, lifact_ = 0;
   std::vector<int> irn_, jcn_, keep_, ifact_, ppos_, iw1_;
   std::vector<double> a_, fact_, w_;
   double cntl_[5] = {0}, rinfo_[20] = {0};
   int icntl_[20] = {0}, info_[40] = {0};
};

#endif  // MA57_BLOCK_HPP
