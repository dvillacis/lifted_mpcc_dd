// Concurrent-MA97 stress test — pins down WHICH part of the DD solver's
// threading breaks MA97 (the OMP=1 MA97_CONCURRENT=1 build segfaults at
// OMP_NUM_THREADS >= 2 on the HPC, 2026-07-23, while everything serial and the
// plain ma97_smoke pass). Mirrors dd_solver.hpp's structure exactly: analyses
// SERIAL (as InitializeStructure does), then factorizations CONCURRENT (the
// loop at factorize()), then backsolves CONCURRENT (the S_k / solve_one
// loops), each phase announced before it runs so a crash names its phase.
// Results are checked against a serial reference pass — a silent corruption
// (wrong inertia / residual) is reported as FAIL, not just a crash.
//
//   OMP=1 ./build_linux.sh ma97_smoke_par.cpp -o ma97_smoke_par
//   OMP_NUM_THREADS=8 ./ma97_smoke_par                        # expect: reproduces the crash
//   OMP_NUM_THREADS=8 OMP_STACKSIZE=512M ./ma97_smoke_par     # worker-stack theory
//   OMP_NUM_THREADS=8 OMP_STACKSIZE=512M MKL_THREADING_LAYER=SEQUENTIAL ./ma97_smoke_par
//   OMP_NUM_THREADS=8 DD_MA97_NO_SCALING=1 ./ma97_smoke_par   # MC64-under-concurrency theory
//   ./ma97_smoke_par [K] [m]                                  # blocks, dim per block (16, 500)
//
// Reading the outcome:
//   * dies in "phase A" only  -> concurrent ma97_factor is the problem; the
//     backsolve loops (97.5% of the S_k cost) can stay parallel — say so and
//     the factor loop gets serialized on its own.
//   * dies in "phase B" too   -> all concurrent MA97 is off-limits in this
//     library build; stay with the serial-across-blocks configuration.
//   * an env var fixes it     -> bake it into the run scripts.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ma97_block.hpp"

// Deterministic symmetric-indefinite tridiagonal block: diag mostly +4 with
// every 7th entry negative (guaranteed indefinite), off-diagonal -1, slight
// per-block shift so the blocks are not identical.
static void fill_block(int id, int m, std::vector<int>& irn, std::vector<int>& jcn,
                       std::vector<double>& val) {
   irn.clear(); jcn.clear(); val.clear();
   for (int i = 1; i <= m; ++i) {
      irn.push_back(i); jcn.push_back(i);
      val.push_back((i % 7 == 3 ? -2.5 : 4.0) + 0.01 * id);
      if (i > 1) { irn.push_back(i); jcn.push_back(i - 1); val.push_back(-1.0); }
   }
}

static double resid(int m, const std::vector<int>& irn, const std::vector<int>& jcn,
                    const std::vector<double>& val, const std::vector<double>& x) {
   std::vector<double> ax(m, 0.0);                       // b was all-ones
   for (size_t e = 0; e < val.size(); ++e) {
      const int i = irn[e] - 1, j = jcn[e] - 1;
      ax[i] += val[e] * x[j];
      if (i != j) ax[j] += val[e] * x[i];
   }
   double r = 0.0;
   for (int i = 0; i < m; ++i) r = std::max(r, std::fabs(ax[i] - 1.0));
   return r;
}

int main(int argc, char** argv) {
   const int K = argc > 1 ? std::atoi(argv[1]) : 16;
   const int m = argc > 2 ? std::atoi(argv[2]) : 500;
#ifdef _OPENMP
   std::printf("OpenMP: max_threads=%d\n", omp_get_max_threads());
#else
   std::printf("compiled WITHOUT OpenMP — phases run serial, rebuild with OMP=1\n");
#endif
   std::printf("K=%d blocks, dim=%d each, scaling %s\n\n", K, m,
               std::getenv("DD_MA97_NO_SCALING") ? "OFF" : "ON (MC64)");

   std::vector<std::vector<int>> irn(K), jcn(K);
   std::vector<std::vector<double>> val(K);
   std::vector<Ma97Block> blk(K);
   std::vector<int> ref_neg(K);
   int fails = 0;

   std::printf("phase 0: serial analyse of all blocks ...\n"); std::fflush(stdout);
   for (int k = 0; k < K; ++k) {
      fill_block(k, m, irn[k], jcn[k], val[k]);
      if (!blk[k].analyze(m, irn[k], jcn[k])) {
         std::printf("FAIL analyse block %d (flag=%d)\n", k, blk[k].status());
         return 1;
      }
   }

   std::printf("phase R: serial reference factor+solve ...\n"); std::fflush(stdout);
   for (int k = 0; k < K; ++k) {
      for (size_t e = 0; e < val[k].size(); ++e) blk[k].values()[e] = val[k][e];
      if (!blk[k].factorize()) { std::printf("FAIL ref factor %d\n", k); return 1; }
      ref_neg[k] = blk[k].negative_eigenvalues();
      std::vector<double> x(m, 1.0);
      blk[k].solve(x.data(), 1);
      if (resid(m, irn[k], jcn[k], val[k], x) > 1e-8) { std::printf("FAIL ref solve %d\n", k); return 1; }
   }

   // DD_PAR_SKIP_FACTOR=1 factorizes SERIALLY here, so phase B (concurrent
   // backsolves) stays reachable although concurrent factorization crashes
   // (gdb 2026-07-23: free() inside __hsl_ma97_..._rfact_block on a worker
   // thread — heap corruption in MA97's internal front memory management,
   // module-global, not curable by any env var). The split mirrors the
   // fallback dd_solver.hpp would implement: serial factor loop + parallel
   // backsolve loops (the backsolves carry 97.5% of the S_k cost).
   const bool skipA = std::getenv("DD_PAR_SKIP_FACTOR") != nullptr;
   if (skipA) {
      std::printf("phase A: SKIPPED (DD_PAR_SKIP_FACTOR=1) — serial refactorize ...\n");
      std::fflush(stdout);
      for (int k = 0; k < K; ++k) {
         for (size_t e = 0; e < val[k].size(); ++e) blk[k].values()[e] = val[k][e];
         if (!blk[k].factorize()) { std::printf("FAIL serial factor %d\n", k); ++fails; }
      }
   } else {
      std::printf("phase A: CONCURRENT factorize (the dd factorize() loop) ...\n");
      std::fflush(stdout);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int k = 0; k < K; ++k) {
         for (size_t e = 0; e < val[k].size(); ++e) blk[k].values()[e] = val[k][e];
         if (!blk[k].factorize()) {
#pragma omp critical
            { std::printf("FAIL concurrent factor %d (flag=%d)\n", k, blk[k].status()); ++fails; }
         }
      }
   }
   for (int k = 0; k < K; ++k)
      if (blk[k].negative_eigenvalues() != ref_neg[k]) {
         std::printf("FAIL inertia block %d: %d vs ref %d\n", k,
                     blk[k].negative_eigenvalues(), ref_neg[k]);
         ++fails;
      }
   std::printf("phase A: %s\n\n",
               fails ? "FAIL"
               : skipA ? "OK (serial refactorize — concurrency NOT tested)"
                       : "OK — concurrent factorize survives");

   std::printf("phase B: CONCURRENT backsolves (the S_k / solve_one loops) ...\n");
   std::fflush(stdout);
   int sfails = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int k = 0; k < K; ++k) {
      double worst = 0.0;
      for (int rep = 0; rep < 50; ++rep) {              // stress repeated backsolves
         std::vector<double> x(m, 1.0);
         blk[k].solve(x.data(), 1);
         worst = std::max(worst, resid(m, irn[k], jcn[k], val[k], x));
      }
      if (worst > 1e-8) {
#pragma omp critical
         { std::printf("FAIL concurrent solve %d (resid=%.2e)\n", k, worst); ++sfails; }
      }
   }
   fails += sfails;
   std::printf("phase B: %s\n\n", sfails ? "FAIL" : "OK — concurrent backsolves survive");

   std::printf(fails ? "STRESS: %d FAILURE(S)\n" : "STRESS: ALL OK at this thread count\n",
               fails);
   return fails ? 1 : 0;
}
