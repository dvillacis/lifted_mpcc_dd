// Standalone smoke test for ma97_block.hpp — isolates the MA97 C-interface ABI
// from the DD solver. Build and run on the machine whose libhsl_ma97.so will be
// used:
//
//   HSL_SOLVER=ma97 ./build_linux.sh ma97_smoke.cpp -o ma97_smoke
//   ./ma97_smoke
//
// Reading the output:
//   * The "defaults" line is the ABI CANARY. Sane values are roughly
//     f_arrays=0 action=1 nemin=8..32 ordering=5 scaling=0 u=0.01 small~1e-20.
//     Garbage there (huge/negative nemin, u not in [0,0.5], ordering > 8) means
//     the struct layout in ma97_block.hpp does not match this library build —
//     e.g. an HSL built with 64-bit default integers (ILP64) — and NOTHING
//     downstream can work; fix the declarations, don't debug the DD solver.
//   * Then a tiny indefinite 3×3 is analyzed/factorized/solved twice (fresh
//     values the second time, as the Newton refresh does), the inertia is
//     checked (expect 1 negative eigenvalue), and the JOB 2→3→4 partial-solve
//     composition is compared against the full solve with scaling off (the
//     signed-MINRES contract).
#include <cmath>
#include <cstdio>
#include <vector>

#include "ma97_block.hpp"

static double resid3(const double A[9], const double* x, const double* b) {
   double r = 0.0;
   for (int i = 0; i < 3; ++i) {
      double ax = 0.0;
      for (int j = 0; j < 3; ++j) ax += A[3 * i + j] * x[j];
      r = std::max(r, std::fabs(ax - b[i]));
   }
   return r;
}

int main() {
   std::printf("sizeof(ma97::control)=%zu  sizeof(ma97::info)=%zu\n",
               sizeof(ma97::control), sizeof(ma97::info));

   ma97::control c;
   ma97_default_control_d(&c);
   std::printf("defaults: f_arrays=%d action=%d nemin=%d ordering=%d scaling=%d "
               "print_level=%d u=%g small=%g\n",
               c.f_arrays, c.action, c.nemin, c.ordering, c.scaling,
               c.print_level, c.u, c.small);
   bool sane = c.nemin > 0 && c.nemin < 1000 && c.u >= 0.0 && c.u <= 0.5 &&
               c.ordering >= 0 && c.ordering <= 8;
   std::printf("ABI canary: %s\n", sane ? "PLAUSIBLE" : "GARBAGE — struct layout mismatch, stop here");
   if (!sane) return 1;

   // A = [[2,1,0],[1,-3,1],[0,1,4]]: det = -30 < 0 ⇒ exactly 1 negative eigenvalue.
   const double Afull[9] = {2, 1, 0, 1, -3, 1, 0, 1, 4};
   const std::vector<int> irn = {1, 2, 2, 3, 3};   // lower triangle, 1-based
   const std::vector<int> jcn = {1, 1, 2, 2, 3};
   const double av[5] = {2, 1, -3, 1, 4};

   int fails = 0;
   {  // default instance (MC64 scaling on, as the DD blocks use it)
      Ma97Block blk;
      if (!blk.analyze(3, irn, jcn)) { std::printf("FAIL analyze (flag=%d)\n", blk.status()); return 1; }
      for (int pass = 0; pass < 2; ++pass) {           // second pass = value refresh
         for (int e = 0; e < 5; ++e) blk.values()[e] = av[e] * (pass ? 2.0 : 1.0);
         if (!blk.factorize()) { std::printf("FAIL factorize pass %d (flag=%d)\n", pass, blk.status()); return 1; }
         double b[3] = {1, 2, 3}, x[3] = {1, 2, 3};
         blk.solve(x, 1);
         double As[9]; for (int i = 0; i < 9; ++i) As[i] = Afull[i] * (pass ? 2.0 : 1.0);
         const double r = resid3(As, x, b);
         const bool ok = r < 1e-12 && blk.negative_eigenvalues() == 1 &&
                         blk.rank() == 3 && !blk.singular();
         std::printf("pass %d: resid=%.2e  neg=%d  rank=%d  flag=%d  %s\n", pass, r,
                     blk.negative_eigenvalues(), blk.rank(), blk.status(),
                     ok ? "OK" : "FAIL");
         if (!ok) ++fails;
      }
   }
   {  // scaling_off instance: JOB 2→3→4 must reproduce JOB 1 (MINRES contract)
      Ma97Block blk;
      blk.scaling_off();
      if (!blk.analyze(3, irn, jcn)) { std::printf("FAIL analyze/off (flag=%d)\n", blk.status()); return 1; }
      for (int e = 0; e < 5; ++e) blk.values()[e] = av[e];
      if (!blk.factorize()) { std::printf("FAIL factorize/off (flag=%d)\n", blk.status()); return 1; }
      double z1[3] = {1, 2, 3}, z2[3] = {1, 2, 3};
      blk.solve_job(1, z1);
      blk.solve_job(2, z2); blk.solve_job(3, z2); blk.solve_job(4, z2);
      double d = 0.0;
      for (int i = 0; i < 3; ++i) d = std::max(d, std::fabs(z1[i] - z2[i]));
      std::printf("compose |234 - 1| = %.2e  %s\n", d, d < 1e-12 ? "OK" : "FAIL");
      if (d >= 1e-12) ++fails;
   }
   std::printf(fails ? "SMOKE: %d FAILURE(S)\n" : "SMOKE: ALL OK\n", fails);
   return fails ? 1 : 0;
}
