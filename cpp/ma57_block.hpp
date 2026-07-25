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
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
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

// Process-wide default for HSL's MC64 scaling on every block this wrapper
// creates. OFF since 2026-07-25 (see the constructor for the measurements);
// the drivers flip it back on with --ma57-scaling on.
inline bool& hsl_block_scaling_default() { static bool v = false; return v; }

class Ma57Block {
public:
   // MA57 INFO() indices, 0-based here (Fortran INFO(k) == info_[k-1]).
   static constexpr int I_STAT = 0;     // INFO(1)  error/warning flag
   static constexpr int I_NOOR = 2;     // INFO(3)  # out-of-range entries (MA57AD)
   static constexpr int I_LFACT = 8;    // INFO(9)  min LFACT for MA57BD
   static constexpr int I_LIFACT = 9;   // INFO(10) min LIFACT for MA57BD
   static constexpr int I_NEIG = 23;    // INFO(24) number of negative eigenvalues
   static constexpr int I_RANK = 24;    // INFO(25) rank of the factorized matrix

   Ma57Block() {
      ma57id_(cntl_, icntl_);
      silence();
      // ---- MC64 scaling, ICNTL(15) — DEFAULT OFF since 2026-07-25 ----------
      // MA57 re-derives the MC64 scaling at EVERY factorization, and the DD
      // solver re-factorizes every W_k at every Newton step: profiling put
      // mc64wd/dd/ed at ~10% of the whole run. Measured effect of disabling it:
      //   * trajectory unchanged where the instance is not on a knife edge
      //     (N=64 4×4 tiles: 155 iterations, same status, same PSNR, ~6%
      //     faster; N=32 6×6: ~27% faster);
      //   * per-step accuracy IMPROVES — over the Newton steps of an N=32 6×6
      //     run, DD_CHECK's max rel-err is 3.5e-9 unscaled vs 4.4e0 scaled;
      //   * it is a PREREQUISITE for --schur forward, whose JOB=2/3 partial
      //     solves only compose into JOB=1 when scaling is off.
      // Restore it with --ma57-scaling on when a badly-scaled instance needs the
      // pivot protection on the deep Scholtes tail (‖A‖ ~ 1e18).
      if (!hsl_block_scaling_default()) icntl_[14] = 0;
      if (std::getenv("DD_MA57_SCALING")) icntl_[14] = 1;
      // ---- ICNTL(13), MA57CD's BLAS2/BLAS3 multi-RHS switch ----------------
      // The S_k formation back-solves p_k ~ 20–40 RHS columns at once. At the
      // HSL default, profiling (2026-07-25, N=32 6×6) caught MA57CD on its
      // BLAS2 kernels — ma57qd/rd at 14% self time against ma57xd/yd at 3%,
      // i.e. the Schur RHS block was being solved one column at a time. Raising
      // the threshold past p_k moves it onto the BLAS3 kernels: the S_k phase
      // drops 2.37s → 1.27s at N=32 6×6 (1.53× on the whole solve) with the
      // iteration trajectory UNCHANGED (N=32 6×6: 72 its; N=64 4×4: 155 its,
      // same status and PSNR). Override with DD_MA57_ICNTL13.
      icntl_[12] = 1 << 20;
      if (const char* v = std::getenv("DD_MA57_ICNTL13")) icntl_[12] = std::atoi(v);
      // DD_MA57_ICNTL6=n: pivot-ordering choice. HSL's default is 5 (automatic
      // AMD/METIS); 4 forces METIS nested dissection, 2 forces AMD. Only
      // meaningful against a real METIS — the stock build links HSL's
      // `fakemetis` stub, so 4 is a no-op there. See cpp/README.md.
      if (const char* v = std::getenv("DD_MA57_ICNTL6")) icntl_[5] = std::atoi(v);
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
      // Out-of-range indices are a WARNING (INFO(1) = +1/+3): MA57 silently
      // drops the entries and factorizes the wrong matrix. The sign test above
      // cannot catch it — the appended diagonal below always trips the
      // duplicate warning (+2), so INFO(1) > 0 is expected — but INFO(3) holds
      // the exact out-of-range count. Any nonzero means an index-mapping bug
      // upstream (owner/slot arrays); fail loudly instead of solving the wrong
      // system. (Duplicates, INFO(4), stay legitimate: the appended diagonal
      // and IPOPT's own triplets both rely on MA57 summing them.)
      if (info_[I_NOOR] > 0) {
         std::cerr << "[ma57] analyze: " << info_[I_NOOR]
                   << " triplet(s) out of range (n=" << n_
                   << ") — index-mapping bug, refusing to factorize the "
                      "reduced matrix\n";
         return false;
      }
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
   // The doubling is computed in long: past ~1.07e9 an int doubling wraps
   // NEGATIVE and the resize throws/corrupts. LFACT/LIFACT are Fortran INTEGER
   // in MA57's ABI, so a factor that genuinely needs > INT_MAX cannot be
   // represented at all — fail loudly instead of wrapping, and turn a bad_alloc
   // on the grow into a clean false rather than a std::terminate through the
   // C/Fortran boundary (IPOPT only catches IpoptException).
   bool factorize() {
      resource_failure_ = false;
      solve_failed_ = false;   // new factorization = new solve epoch
      for (int attempt = 0; attempt < 6; ++attempt) {
         ma57bd_(&n_, &ne_, a_.data(), fact_.data(), &lfact_, ifact_.data(),
                 &lifact_, &lkeep_, keep_.data(), ppos_.data(), icntl_, cntl_,
                 info_, rinfo_);
         const int st = info_[I_STAT];
         if (st == -3 || st == -4) {
            int& len = (st == -3) ? lfact_ : lifact_;
            const long want = 2L * len;
            if (want > (long)std::numeric_limits<int>::max()) {
               std::cerr << "[ma57] workspace request " << want
                         << " exceeds MA57's 32-bit LFACT/LIFACT limit (n="
                         << n_ << ")\n";
               resource_failure_ = true;
               return false;
            }
            try {
               len = (int)want;
               if (st == -3) fact_.assign(lfact_, 0.0);
               else          ifact_.assign(lifact_, 0);
            } catch (const std::bad_alloc&) {
               std::cerr << "[ma57] out of memory growing "
                         << (st == -3 ? "LFACT" : "LIFACT") << " to " << want
                         << " (n=" << n_ << ")\n";
               resource_failure_ = true;
               return false;
            }
            continue;
         }
         return st >= 0;
      }
      // Six −3/−4 rounds without MA57BD ever running to completion: a
      // workspace/resource problem, not a property of the matrix.
      resource_failure_ = true;
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
   bool solve(double* rhs, int nrhs = 1) { return solve_job(1, rhs, nrhs); }

   // Partial solves with the SAME factorization: JOB=2 solves LX=B, JOB=3
   // solves DX=B, JOB=4 solves LᵀX=B (JOB=1 is the full solve, what solve()
   // forwards to). Composing 2→3→4 reproduces JOB=1 ONLY when MC64 scaling was
   // off during factorization — with scaling the partials act on the scaled
   // system. Callers composing partials must build the object with
   // scaling_off() and self-check the composition against JOB=1.
   bool solve_job(int job, double* rhs, int nrhs = 1) {
      // MA57CD's LW is a Fortran INTEGER, but n_·nrhs overflows int on the
      // widest multi-RHS S_k backsolves (dim_k ~1e6 × p_k ~1e3 at N=512 with
      // few subdomains): the wrapped LW is negative or silently undersized —
      // OOB scratch writes. Chunk the RHS columns (contiguous, column-major,
      // leading dimension n_) so LW stays under a sane cap; the single-call
      // fast path below is untouched for every current run size, so validated
      // trajectories are bit-identical.
      static constexpr long LW_CAP = 1L << 27;   // 2^27 doubles = 1 GB scratch
      const long lw_full = (long)n_ * nrhs;
      if (lw_full > LW_CAP) {
         const int chunk = std::max(1, (int)(LW_CAP / n_));   // 1 is always representable: lw = n_·1 fits int by type
         bool ok = true;
         for (int j0 = 0; j0 < nrhs; j0 += chunk)
            ok = solve_job(job, rhs + (std::size_t)j0 * n_,
                           std::min(chunk, nrhs - j0)) && ok;
         return ok;
      }
      const int lw = (int)lw_full;
      if ((int)w_.size() < lw) w_.resize(lw);
      if ((int)iw1_.size() < n_) iw1_.resize(n_);
      ma57cd_(&job, &n_, fact_.data(), &lfact_, ifact_.data(), &lifact_, &nrhs,
              rhs, &n_, w_.data(), &lw, iw1_.data(), icntl_, info_);
      // MA57CD reports errors in INFO(1); returning silently would hand the
      // caller garbage. Warn once (not per solve — ~10⁶ backsolves/run under
      // CG), and LATCH the failure per factorization epoch so callers that
      // batch many backsolves (the S_k formation) can poll solve_failed()
      // afterwards instead of checking every call.
      if (info_[I_STAT] < 0) {
         solve_failed_ = true;
         static bool warned = false;
         if (!warned) {
            std::cerr << "[ma57] MA57CD back-solve failed: INFO(1)="
                      << info_[I_STAT] << "\n";
            warned = true;
         }
         return false;
      }
      return true;
   }

   // Disable MC64 scaling (ICNTL(15)=0) for THIS instance — required for
   // meaningful JOB=2/3/4 partial solves. Call before analyze/factorize.
   void scaling_off() { icntl_[14] = 0; }

   int negative_eigenvalues() const { return info_[I_NEIG]; }
   int rank() const { return info_[I_RANK]; }
   bool singular() const { return info_[I_RANK] < n_; }
   int status() const { return info_[I_STAT]; }
   // True when the last factorize() failure was workspace/memory, NOT a
   // property of the matrix — the caller must not treat it as SINGULAR
   // (IPOPT's δ_w bump would re-run the same OOM in a loop).
   bool resource_failure() const { return resource_failure_; }
   // Sticky per factorization epoch: any back-solve since the last
   // factorize() reported INFO(1) < 0 (its output is garbage).
   bool solve_failed() const { return solve_failed_; }

private:
   bool resource_failure_ = false;
   bool solve_failed_ = false;
   int n_ = 0, ne_ = 0, ne_user_ = 0, lkeep_ = 0, lfact_ = 0, lifact_ = 0;
   std::vector<int> irn_, jcn_, keep_, ifact_, ppos_, iw1_;
   std::vector<double> a_, fact_, w_;
   double cntl_[5] = {0}, rinfo_[20] = {0};
   int icntl_[20] = {0}, info_[40] = {0};
};

#endif  // MA57_BLOCK_HPP
