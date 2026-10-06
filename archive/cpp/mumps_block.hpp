// A RAII wrapper around COIN ThirdParty-Mumps for one subdomain block W_k,
// computing the local Schur contribution S_k = −B̄_k W_k⁻¹ B̄_kᵀ as a byproduct
// of the factorization (MUMPS's partial-factorization Schur feature).
//
// Why MUMPS next to MA57: in direct mode the dominant DD cost is the S_k phase
// — p_k dense multi-RHS MA57 backsolves per block (measured: backsolves are
// 97.5% of the phase; the RHS columns are 0.3–1% dense and MA57CD is
// dense-RHS-only). MUMPS eliminates the interior variables of the AUGMENTED
// local matrix
//
//     A_k = [ W_k   B̄_kᵀ ]     Schur list = the p_k appended border indices
//           [ B̄_k   0    ]
//
// and returns Schur = 0 − B̄_k W_k⁻¹ B̄_kᵀ directly — BLAS-3 frontal kernels and
// internal sparsity instead of p_k naive backsolves. That is EXACTLY the Sk_
// the arrowhead scatter consumes (same sign), and INFOG(12) gives the negative
// pivots of the factored interior part (= In(W_k)_neg), so the Haynsworth
// inertia contract is untouched.
//
// Contracts mirrored from Ma57Block (the validated reference wrapper):
//   * analysis once per block, values-only refresh per Newton step;
//   * coordinate input, ONE (lower) triangle, 1-based indices, duplicates
//     summed; a full zero diagonal is appended so every variable — including
//     Schur variables with only off-diagonal couplings — is structurally
//     present;
//   * honest singularity: null-pivot detection (ICNTL(24)=1) reports
//     singular() without masking, INFO(1)=−10 maps to the same; OOM/workspace
//     exhaustion is resource_failure(), never singularity;
//   * solve() = W_k⁻¹ on the INTERIOR problem (JOB=3 with ICNTL(26)=0 on the
//     augmented system), single or multi RHS, with the per-epoch
//     solve_failed() latch.
//
// The Schur array layout was established empirically (mumps_smoke.cpp's
// sentinel probe): MUMPS writes the lower triangle into the full p×p array
// and zeroes the upper; factorize() mirrors it to a plain symmetric matrix.
// Run mumps_smoke on every new machine/library build — it is the layout
// canary as well as the API validation gate.
//
// THREADING: unproven. MUMPS instances carry library-internal state and this
// wrapper has not been through a mumps_smoke_par-style concurrency audit (the
// MA97 lesson: concurrent ma97_factor heap-corrupts). Callers must factorize
// MumpsSchurBlock objects SERIALLY until such an audit exists.
#ifndef MUMPS_BLOCK_HPP
#define MUMPS_BLOCK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "dmumps_c.h"

class MumpsSchurBlock {
public:
   MumpsSchurBlock() {
      std::memset(&id_, 0, sizeof(id_));
      id_.comm_fortran = -987654;   // USE_COMM_WORLD (COIN libseq stub)
      id_.par = 1;                  // host works
      id_.sym = 2;                  // general symmetric indefinite
      id_.job = -1;                 // initialize
      dmumps_c(&id_);
      alive_ = (infog(1) >= 0);
      icntl(1) = -1;   // error stream    (≤0 suppresses)
      icntl(2) = -1;   // diagnostic stream
      icntl(3) = -1;   // global info stream
      icntl(4) = 0;    // print level: none
      icntl(13) = 1;   // no ScaLAPACK root — INFOG(12) inertia stays exact
      icntl(24) = 1;   // null-pivot detection: honest singular(), no hard abort
      // Scaling/ordering stay at MUMPS defaults — the MA57 record says scaling
      // OFF is a measured 6.5× blowup on the Scholtes-tail KKTs; don't mimic.
   }

   ~MumpsSchurBlock() {
      if (alive_) {
         id_.job = -2;
         dmumps_c(&id_);
      }
   }
   MumpsSchurBlock(const MumpsSchurBlock&) = delete;
   MumpsSchurBlock& operator=(const MumpsSchurBlock&) = delete;

   const char* version() const { return id_.version_number; }

   // Symbolic analysis of the augmented block. irn/jcn are the USER triplets,
   // 1-based, one (lower) triangle: interior indices 1..n_int, Schur (border)
   // indices n_int+1..n_int+pk. Call once; only values change afterwards.
   bool analyze(int n_int, int pk, const std::vector<int>& irn,
                const std::vector<int>& jcn) {
      if (!alive_) return false;
      n_int_ = n_int;
      pk_ = pk;
      const int n_aug = n_int + pk;
      ne_user_ = (int)irn.size();
      irn_.assign(irn.begin(), irn.end());
      jcn_.assign(jcn.begin(), jcn.end());
      // Appended zero diagonal (summed with any user entries — exact): every
      // variable is structurally present, as in Ma57Block.
      for (int i = 1; i <= n_aug; ++i) { irn_.push_back(i); jcn_.push_back(i); }
      ne_ = (int)irn_.size();
      a_.assign(std::max(ne_, 1), 0.0);
      id_.n = n_aug;
      id_.nnz = (MUMPS_INT8)ne_;
      id_.irn = irn_.data();
      id_.jcn = jcn_.data();
      id_.a = a_.data();
      if (pk_ > 0) {
         listvar_.resize(pk_);
         for (int j = 0; j < pk_; ++j) listvar_[j] = n_int_ + 1 + j;
         schur_.assign((std::size_t)pk_ * pk_, 0.0);
         id_.size_schur = pk_;
         id_.listvar_schur = listvar_.data();
         id_.schur = schur_.data();
         icntl(19) = 1;              // centralized Schur on the host
      } else {
         id_.size_schur = 0;
         id_.listvar_schur = nullptr;
         id_.schur = nullptr;
         icntl(19) = 0;
      }
      id_.job = 1;
      dmumps_c(&id_);
      status_ = infog(1);
      if (status_ < 0) return false;
      // Out-of-range entries are dropped silently past the warning (+1 flag):
      // that would factorize the WRONG matrix. Any hit is an index-mapping bug
      // upstream — refuse, exactly like Ma57Block's INFO(3) gate.
      if (status_ > 0 && (status_ & 1)) {
         std::cerr << "[mumps] analyze: out-of-range triplet(s) (n=" << n_aug
                   << ") — index-mapping bug, refusing to factorize the "
                      "reduced matrix\n";
         return false;
      }
      return true;
   }

   double* values() { return a_.data(); }   // the user's ne triplets
   int nnz() const { return ne_user_; }
   int dim() const { return n_int_; }       // INTERIOR dimension (= W_k's)
   int schur_dim() const { return pk_; }

   // Numeric factorization + Schur formation of the values currently in
   // values(). Retries with a larger workspace relaxation on the standard
   // "workspace too small" returns (the MA57 −3/−4 pattern).
   bool factorize() {
      resource_failure_ = false;
      solve_failed_ = false;      // new factorization = new solve epoch
      singular_ = false;
      neg_ = 0;
      nullpiv_ = 0;
      if (!alive_) { resource_failure_ = true; return false; }
      for (int attempt = 0; attempt < 6; ++attempt) {
         id_.job = 2;
         dmumps_c(&id_);
         status_ = infog(1);
         if (status_ == -8 || status_ == -9 || status_ == -19) {
            // Workspace estimate too small: raise the relaxation and retry.
            icntl(14) = std::max(2 * icntl(14), 40);
            continue;
         }
         if (status_ == -13) {     // allocation failure — resources, not math
            std::cerr << "[mumps] factorize: allocation failure (n="
                      << n_int_ + pk_ << ", INFOG(2)=" << infog(2) << ")\n";
            resource_failure_ = true;
            return false;
         }
         if (status_ == -10) {     // numerically singular — the honest answer
            singular_ = true;
            return false;
         }
         if (status_ < 0) return false;
         neg_ = infog(12);         // negative pivots of the factored interior
         nullpiv_ = infog(28);     // null pivots found by ICNTL(24)
         singular_ = nullpiv_ > 0;
         return pk_ > 0 ? finish_schur() : true;
      }
      resource_failure_ = true;    // six workspace rounds without completing
      return false;
   }

   // The symmetrized pk×pk Schur block, valid after a successful factorize().
   // Layout: symmetric, so row-/col-major coincide; entry (a,b) at a*pk+b.
   const double* schur() const { return schur_.data(); }

   // In-place solve W_k x = b on the INTERIOR problem, nrhs columns
   // (column-major, leading dimension n_int). JOB=3 with ICNTL(26)=0 solves
   // the internal problem of the augmented system; Schur-variable rows of the
   // MUMPS RHS are zeroed on the way in and dropped on the way out.
   bool solve(double* rhs, int nrhs = 1) {
      if (nrhs <= 0) return true;
      if (!alive_) { solve_failed_ = true; return false; }
      const int n_aug = n_int_ + pk_;
      // Chunk very wide RHS batches so the staging buffer stays bounded
      // (mirrors Ma57Block's LW cap; in the DD solver nrhs is 1 anyway).
      static constexpr long CAP = 1L << 27;
      if ((long)n_aug * nrhs > CAP) {
         const int chunk = std::max(1, (int)(CAP / n_aug));
         bool ok = true;
         for (int j0 = 0; j0 < nrhs; j0 += chunk)
            ok = solve(rhs + (std::size_t)j0 * n_int_,
                       std::min(chunk, nrhs - j0)) && ok;
         return ok;
      }
      if ((long)rhs_buf_.size() < (long)n_aug * nrhs)
         rhs_buf_.resize((std::size_t)n_aug * nrhs);
      for (int c = 0; c < nrhs; ++c) {
         double* dst = rhs_buf_.data() + (std::size_t)c * n_aug;
         const double* src = rhs + (std::size_t)c * n_int_;
         std::copy(src, src + n_int_, dst);
         std::fill(dst + n_int_, dst + n_aug, 0.0);
      }
      id_.rhs = rhs_buf_.data();
      id_.nrhs = nrhs;
      id_.lrhs = n_aug;
      icntl(26) = 0;               // solve the internal problem
      id_.job = 3;
      dmumps_c(&id_);
      status_ = infog(1);
      if (status_ < 0) {
         solve_failed_ = true;
         static bool warned = false;
         if (!warned) {
            std::cerr << "[mumps] solve failed: INFOG(1)=" << status_ << "\n";
            warned = true;
         }
         return false;
      }
      for (int c = 0; c < nrhs; ++c) {
         const double* src = rhs_buf_.data() + (std::size_t)c * n_aug;
         std::copy(src, src + n_int_, rhs + (std::size_t)c * n_int_);
      }
      return true;
   }

   int negative_eigenvalues() const { return neg_; }
   int rank() const { return n_int_ - nullpiv_; }
   bool singular() const { return singular_; }
   int status() const { return status_; }
   // True when the last factorize() failure was workspace/memory, NOT a
   // property of the matrix (same contract as Ma57Block).
   bool resource_failure() const { return resource_failure_; }
   // Sticky per factorization epoch: any solve since the last factorize()
   // reported INFOG(1) < 0 (its output is garbage).
   bool solve_failed() const { return solve_failed_; }

private:
   MUMPS_INT& icntl(int i) { return id_.icntl[i - 1]; }
   int infog(int i) const { return (int)id_.infog[i - 1]; }

   // MUMPS 5.5 + SYM=2 + ICNTL(19)=1 returns the LOWER triangle of the Schur
   // in the full p×p array (row-major reading: entry (i,j), j ≤ i, at i·p+j)
   // and ZEROES the upper triangle — established empirically by the sentinel
   // probe in mumps_smoke.cpp (2026-07-24; a NaN pre-fill came back 0.0 above
   // the diagonal, exact values below). Mirror lower → upper so consumers see
   // the plain symmetric matrix. mumps_smoke is the canary for any future
   // library whose layout differs — run it on every new machine/build.
   bool finish_schur() {
      const int p = pk_;
      double* s = schur_.data();
      for (int i = 0; i < p; ++i)
         for (int j = 0; j < i; ++j)
            s[(std::size_t)j * p + i] = s[(std::size_t)i * p + j];
      return true;
   }

   DMUMPS_STRUC_C id_;
   bool alive_ = false;
   bool resource_failure_ = false;
   bool solve_failed_ = false;
   bool singular_ = false;
   int n_int_ = 0, pk_ = 0, ne_ = 0, ne_user_ = 0;
   int status_ = 0, neg_ = 0, nullpiv_ = 0;
   std::vector<MUMPS_INT> irn_, jcn_, listvar_;
   std::vector<double> a_, schur_, rhs_buf_;
};

#endif  // MUMPS_BLOCK_HPP
