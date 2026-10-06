// A thin RAII wrapper around HSL MA97 for one symmetric-indefinite block — the
// drop-in sibling of ma57_block.hpp for machines whose HSL install has MA97 but
// not MA57 (e.g. an HPC conda env built for IPOPT's own ma97/spral route).
// Selected by compiling with -DDD_USE_MA97 (see dd_solver.hpp / build_linux.sh);
// the default build keeps MA57, the validated reference.
//
// MA97 is HSL's supernodal multifrontal LDLᵀ with threshold partial pivoting —
// like MA57 it is stable on indefinite matrices and reports the inertia
// (info.num_neg) and rank directly, which is what the Haynsworth machinery
// consumes. It is OpenMP-parallel internally; the DD loops parallelize ACROSS
// blocks, so set OMP_NUM_THREADS=1 if nested threading over many small W_k
// ever measures as overhead.
//
// API contract (mirrors Ma57Block exactly — dd_solver.hpp uses either through
// the SymBlock alias):
//   * analyze(n, irn, jcn): 1-based coordinate triplets, one triangle, called
//     once per block. MA57 accepts entries in EITHER triangle per entry; MA97
//     wants the lower triangle (row ≥ col), so out-of-triangle entries are
//     swapped per entry — exact for a symmetric matrix, and the VALUE ORDER is
//     untouched (factorize() relies on values() matching the analyzed order).
//     The full zero diagonal is appended as in Ma57Block (duplicates are
//     summed, so this is exact).
//   * solve_job(job, ...) takes MA57's JOB convention (1 = full, 2 = L, 3 = D,
//     4 = Lᵀ) and translates to MA97's (0 = full, 1 = PL, 2 = D, 3 = (PL)ᵀ).
//     With scaling off, A = (PL) D (PL)ᵀ, so composing 2→3→4 reproduces the
//     full solve exactly as with MA57 — the signed-MINRES snapshot self-check
//     in dd_solver.hpp validates the composition at runtime either way, and
//     its D-solve bit-probe is format-free (it discovers the 1×1/2×2 pivot
//     pairing by probing, never by reading factor internals), so it carries
//     over unchanged.
//   * scaling_off() disables MC64 scaling for THIS instance (control.scaling
//     = 0) — required before analyze/factorize for meaningful partial solves.
//     The default mirrors Ma57Block: MC64 ON (control.scaling = 1), because
//     scaling protects pivot quality on the badly-scaled Scholtes-tail KKTs
//     (‖A‖ ~ 1e18). DD_MA97_NO_SCALING=1 turns it off globally (the A/B twin
//     of DD_MA57_NO_SCALING; same caveats — validate with DD_CHECK).
//
// The C interface (ma97_*_d symbols, bind(C) in the library's hsl_ma97_ciface
// module) is declared here directly rather than via <hsl_ma97d.h>, so the
// build needs only libhsl_ma97.so, not its headers. The struct layouts are the
// documented HSL_MA97 C-interface ABI (stable across 2.x — additions are
// absorbed by the trailing spare slots, and everything read here sits before
// them). If your libhsl_ma97.so lacks these symbols it was built without the
// C interface — rebuild with it (IPOPT's own ma97 route needs it too).
#ifndef MA97_BLOCK_HPP
#define MA97_BLOCK_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace ma97 {
// The documented HSL_MA97 C-interface layout, PLUS a trailing abi_margin_* pad
// on both structs. HSL's own compatibility mechanism absorbs additions into
// the documented spare slots (size stays constant across 2.x), so the margin
// should never be reached — it exists because these structs are passed to an
// arbitrary user-supplied libhsl_ma97.so, the library WRITES them (info from
// analyse/factor/solve, control from ma97_default_control), and info_ is a
// data member of Ma97Block: a non-conforming build writing past the documented
// end would corrupt the adjacent heap silently. 128 spare bytes turns that
// into corrupt-but-contained padding. An ILP64 library shifts every field
// instead — that is what the ma97_smoke ABI canary catches; run it first on
// any new machine.
struct control {
   int    f_arrays;          // ≠0: 1-based (Fortran) array indexing
   int    action;            // ≠0: continue on singularity, report rank
   int    nemin;
   double multiplier;
   int    ordering;
   int    print_level;
   int    scaling;           // 0 = none/user, 1 = MC64, 2 = MC77, 3 = MC30
   double small;
   double u;                 // pivot threshold
   int    unit_diagnostics;
   int    unit_error;
   int    unit_warning;
   long   factor_min;
   int    solve_blas3;
   long   solve_min;
   int    solve_mf;
   double consist_tol;
   int    ispare[5];
   double rspare[10];
   int    abi_margin_i[8];   // defensive margin — not part of the HSL ABI
   double abi_margin_r[8];
};
struct info {
   int    flag;              // 0 ok, <0 error, >0 warning
   int    flag68;
   int    flag77;
   int    matrix_dup;
   int    matrix_rank;
   int    matrix_outrange;
   int    matrix_missing_diag;
   int    maxdepth;
   int    maxfront;
   int    num_delay;
   long   num_factor;
   long   num_flops;
   int    num_neg;           // negative eigenvalues == In(A)_neg
   int    num_sup;
   int    num_two;
   int    ordering;
   int    stat;
   int    maxsupernode;      // spare slot in pre-2.6 libraries — never read here
   int    ispare[4];
   double rspare[10];
   int    abi_margin_i[8];   // defensive margin — not part of the HSL ABI
   double abi_margin_r[8];
};
// Freeze the compiled layout (LP64, natural alignment): the documented HSL
// portion is 216/176 bytes, the margin adds 96 to each. A failure here means
// the declarations above were edited (or an unsupported data model) — fix the
// struct, do not silence the assert; the runtime ABI canary in ma97_smoke
// checks the LIBRARY side of the same contract.
static_assert(sizeof(control) == 312, "ma97::control layout drifted");
static_assert(sizeof(info) == 272, "ma97::info layout drifted");
}  // namespace ma97

extern "C" {
void ma97_default_control_d(ma97::control* control);
void ma97_analyse_coord_d(int n, int ne, const int* row, const int* col,
                          double* val, void** akeep,
                          const ma97::control* control, ma97::info* info,
                          int* order);
void ma97_factor_d(int matrix_type, const int* ptr, const int* row,
                   const double* val, void** akeep, void** fkeep,
                   const ma97::control* control, ma97::info* info,
                   double* scale);
void ma97_solve_d(int job, int nrhs, double* x, int ldx, void** akeep,
                  void** fkeep, const ma97::control* control, ma97::info* info);
void ma97_free_akeep_d(void** akeep);
void ma97_free_fkeep_d(void** fkeep);
void ma97_finalise_d(void** akeep, void** fkeep);
}

// Process-wide default for HSL's MC64 scaling on every block this wrapper
// creates. OFF since 2026-07-25 (see the constructor for the measurements);
// the drivers flip it back on with --ma57-scaling on.
inline bool& hsl_block_scaling_default() { static bool v = false; return v; }

class Ma97Block {
public:
   Ma97Block() {
      ma97_default_control_d(&control_);
      control_.f_arrays = 1;   // the triplets stay 1-based, as for MA57
      control_.action = 1;     // continue on rank deficiency; singular() reads rank
      // MC64 scaling, mirroring Ma57Block's ICNTL(15) — DEFAULT OFF since
      // 2026-07-25 (same rationale and the same --ma57-scaling switch; see the
      // measurements in ma57_block.hpp's constructor).
      control_.scaling = hsl_block_scaling_default() ? 1 : 0;
      silence();
      if (std::getenv("DD_MA97_SCALING")) control_.scaling = 1;
      if (std::getenv("DD_MA97_NO_SCALING")) control_.scaling = 0;
   }
   ~Ma97Block() { free_keeps(); }
   Ma97Block(const Ma97Block&) = delete;
   Ma97Block& operator=(const Ma97Block&) = delete;

   void silence() {
      control_.print_level = 0;
      control_.unit_error = -1;        // <0 suppresses, as for MA57's streams
      control_.unit_warning = -1;
      control_.unit_diagnostics = -1;
   }

   // Symbolic analysis. irn/jcn are 1-based, one triangle. Call once.
   bool analyze(int n, const std::vector<int>& irn, const std::vector<int>& jcn) {
      n_ = n; ne_user_ = (int)irn.size(); ne_ = ne_user_ + n;
      irn_ = irn; jcn_ = jcn;
      for (int e = 0; e < ne_user_; ++e)               // lower triangle (row ≥ col)
         if (irn_[e] < jcn_[e]) std::swap(irn_[e], jcn_[e]);
      for (int i = 1; i <= n; ++i) { irn_.push_back(i); jcn_.push_back(i); }
      a_.assign(std::max(ne_, 1), 0.0);
      free_keeps();
      ma97_analyse_coord_d(n_, ne_, irn_.data(), jcn_.data(), nullptr, &akeep_,
                           &control_, &info_, nullptr);
      if (info_.flag < 0) return false;
      // Out-of-range indices are only a WARNING: MA97 drops the entries and
      // factorizes the wrong matrix. Unlike MA57's sign test, the exact count
      // is exposed — any nonzero means an index-mapping bug upstream; fail
      // loudly. (matrix_dup stays legitimate: the appended diagonal and
      // IPOPT's own triplets both rely on duplicates being summed.)
      if (info_.matrix_outrange > 0) {
         std::cerr << "[ma97] analyse: " << info_.matrix_outrange
                   << " triplet(s) out of range (n=" << n_
                   << ") — index-mapping bug, refusing to factorize the "
                      "reduced matrix\n";
         return false;
      }
      scale_.assign(std::max(n_, 1), 0.0);
      return true;
   }

   double* values() { return a_.data(); }          // the user's ne triplets
   int nnz() const { return ne_user_; }
   int dim() const { return n_; }

   // Numeric factorization of the values currently in values(). MA97 manages
   // its own factor storage through fkeep — no LFACT-style retry loop needed.
   // matrix_type 4 = real symmetric indefinite. The scale array is passed only
   // when MA97 computes a scaling (with control.scaling = 0 a non-NULL scale
   // would be READ as user-supplied scaling).
   bool factorize() {
      resource_failure_ = false;
      solve_failed_ = false;   // new factorization = new solve epoch
      ma97_factor_d(4, nullptr, nullptr, a_.data(), &akeep_, &fkeep_, &control_,
                    &info_, control_.scaling != 0 ? scale_.data() : nullptr);
      if (info_.flag < 0) {
         // An allocation failure sets the Fortran STAT value in info.stat —
         // classify it so the caller doesn't treat OOM as a singularity
         // (IPOPT's δ_w bump would re-run the same OOM in a loop).
         if (info_.stat != 0) {
            resource_failure_ = true;
            std::cerr << "[ma97] factor allocation failure: flag=" << info_.flag
                      << " stat=" << info_.stat << " (n=" << n_ << ")\n";
         }
         return false;
      }
      return true;
   }

   // In-place solve A x = b for nrhs columns (column-major, leading dim n).
   bool solve(double* rhs, int nrhs = 1) { return solve_job(1, rhs, nrhs); }

   // Partial solves, in MA57's JOB numbering (see the header comment): 1 full,
   // 2 forward (PL), 3 diagonal (D), 4 back ((PL)ᵀ). Composing 2→3→4
   // reproduces 1 ONLY with scaling off — same contract as Ma57Block.
   bool solve_job(int job, double* rhs, int nrhs = 1) {
      static const int ma97_job[5] = {-1, 0, 1, 2, 3};
      ma97_solve_d(ma97_job[job], nrhs, rhs, n_, &akeep_, &fkeep_, &control_,
                   &info_);
      // Warn once, LATCH per factorization epoch — same contract as Ma57Block
      // (callers batching backsolves poll solve_failed() afterwards).
      if (info_.flag < 0) {
         solve_failed_ = true;
         static bool warned = false;
         if (!warned) {
            std::cerr << "[ma97] solve failed: info.flag=" << info_.flag << "\n";
            warned = true;
         }
         return false;
      }
      return true;
   }

   // Disable MC64 scaling for THIS instance — required for meaningful
   // JOB=2/3/4 partial solves. Call before analyze/factorize.
   void scaling_off() { control_.scaling = 0; }

   int negative_eigenvalues() const { return info_.num_neg; }
   int rank() const { return info_.matrix_rank; }
   bool singular() const { return info_.matrix_rank < n_; }
   int status() const { return info_.flag; }
   // Same contract as Ma57Block: resource_failure() = the last factorize()
   // failure was allocation, not the matrix; solve_failed() = some back-solve
   // since the last factorize() reported flag < 0.
   bool resource_failure() const { return resource_failure_; }
   bool solve_failed() const { return solve_failed_; }

private:
   bool resource_failure_ = false;
   bool solve_failed_ = false;
   void free_keeps() {
      if (akeep_ && fkeep_) ma97_finalise_d(&akeep_, &fkeep_);
      else if (akeep_) ma97_free_akeep_d(&akeep_);
      else if (fkeep_) ma97_free_fkeep_d(&fkeep_);
      akeep_ = fkeep_ = nullptr;
   }

   int n_ = 0, ne_ = 0, ne_user_ = 0;
   void* akeep_ = nullptr;
   void* fkeep_ = nullptr;
   std::vector<int> irn_, jcn_;
   std::vector<double> a_, scale_;
   ma97::control control_;
   ma97::info info_ = {};
};

#endif  // MA97_BLOCK_HPP
