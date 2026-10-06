// mumps_check — run once on every machine / MUMPS build before using
// --block-solver mumps.  Three checks:
//
//   1. self_test(): a 3×3 KKT block with a known Schur complement and inertia.
//   2. accuracy: random sparse KKT tiles (indefinite Hessian, rank-full
//      Jacobian, linking rows to p border unknowns) against dense Eigen:
//      S_k = −B W⁻¹ Bᵀ, #neg(W), and a multi-RHS solve.
//   3. concurrency: many MumpsBlock instances used from OpenMP threads at once,
//      repeatedly, compared bit for bit with a serial run.  MUMPS's C interface
//      is not thread-safe (see mumps_block.hpp), so MumpsBlock serializes its
//      calls with a lock; this check verifies that callers on many threads get
//      exactly the serial results and nothing crashes.
//
//   ./mumps_check [tiles=32] [rounds=20]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mumps_block.hpp"

using Mat = Eigen::MatrixXd;

struct Tile {
   int n, p;                       // interior (primal + dual) and border sizes
   std::vector<int> r, c;          // augmented pattern (any triangle)
   std::vector<double> v;          // values
   Mat W, B;                       // dense copies for the reference
};

// A KKT tile: nx primals (diagonal Hessian of mixed sign plus a few couplings),
// m constraint rows (random sparse Jacobian with a nonzero per row), p border
// unknowns each coupled to one constraint row (a linking row).
static Tile make_tile(int nx, int m, int p, std::mt19937& g) {
   std::uniform_real_distribution<double> U(-1.0, 1.0);
   std::uniform_int_distribution<int> col(0, nx - 1);
   Tile t;
   t.n = nx + m;
   t.p = p;
   t.W = Mat::Zero(t.n, t.n);
   t.B = Mat::Zero(p, t.n);
   for (int i = 0; i < nx; ++i) t.W(i, i) = (i % 5 == 0 ? -1.0 : 1.0) * (0.5 + std::abs(U(g)));
   for (int k = 0; k < nx; ++k) {
      const int i = col(g), j = col(g);
      if (i != j) { const double a = 0.2 * U(g); t.W(i, j) += a; t.W(j, i) += a; }
   }
   for (int r = 0; r < m; ++r) {
      const int j0 = r % nx;   // guarantees full row rank
      t.W(nx + r, j0) = t.W(j0, nx + r) = 1.0 + std::abs(U(g));
      for (int k = 0; k < 3; ++k) {
         const int j = col(g);
         t.W(nx + r, j) += U(g);
         t.W(j, nx + r) = t.W(nx + r, j);
      }
   }
   for (int a = 0; a < p; ++a) t.B(a, nx + (a * 7) % m) = -1.0;
   for (int i = 0; i < t.n; ++i)
      for (int j = 0; j <= i; ++j)
         if (t.W(i, j) != 0.0) { t.r.push_back(i); t.c.push_back(j); t.v.push_back(t.W(i, j)); }
   for (int a = 0; a < p; ++a)
      for (int l = 0; l < t.n; ++l)
         if (t.B(a, l) != 0.0) { t.r.push_back(t.n + a); t.c.push_back(l); t.v.push_back(t.B(a, l)); }
   return t;
}

struct Result {
   bool ok = false;
   int neg = -1;
   std::vector<double> S, x;
};

static Result run(const Tile& t, int nrhs) {
   Result res;
   dd::MumpsBlock m;
   if (!m.analyze(t.n, t.p, t.r, t.c) || !m.factorize(t.v.data())) return res;
   res.neg = m.negative();
   res.S.assign(m.schur(), m.schur() + (size_t)t.p * t.p);
   res.x.resize((size_t)t.n * nrhs);
   for (size_t i = 0; i < res.x.size(); ++i) res.x[i] = std::sin(0.37 * (double)i);
   res.ok = m.solve(res.x.data(), nrhs);
   return res;
}

int main(int argc, char** argv) {
   const int ntiles = argc > 1 ? std::atoi(argv[1]) : 32;
   const int rounds = argc > 2 ? std::atoi(argv[2]) : 20;
   int failures = 0;

   // ---- 1. self test
   const bool st = dd::MumpsBlock::self_test();
   std::printf("1. self test (Schur layout, inertia, solve): %s\n", st ? "OK" : "FAILED");
   failures += !st;

   // ---- 2. accuracy against dense Eigen
   std::mt19937 g(7);
   std::vector<Tile> tiles;
   for (int k = 0; k < ntiles; ++k) tiles.push_back(make_tile(300 + 13 * k, 120 + 5 * k, 40, g));
   double max_err = 0.0;
   int inertia_bad = 0;
   for (const Tile& t : tiles) {
      const Result r = run(t, 3);
      if (!r.ok) { ++inertia_bad; continue; }
      const Mat S_ref = -t.B * t.W.lu().solve(t.B.transpose());
      const Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Mat>(t.W).eigenvalues();
      const int neg_ref = (int)(ev.array() < 0).count();
      Mat X0(t.n, 3);
      for (int i = 0; i < t.n * 3; ++i) X0.data()[i] = std::sin(0.37 * i);
      const Mat X_ref = t.W.lu().solve(X0);
      const Eigen::Map<const Mat> S(r.S.data(), t.p, t.p), X(r.x.data(), t.n, 3);
      max_err = std::max(max_err, (S - S_ref).cwiseAbs().maxCoeff() / std::max(1.0, S_ref.cwiseAbs().maxCoeff()));
      max_err = std::max(max_err, (X - X_ref).cwiseAbs().maxCoeff() / std::max(1.0, X_ref.cwiseAbs().maxCoeff()));
      inertia_bad += r.neg != neg_ref;
   }
   const bool acc = max_err < 1e-9 && inertia_bad == 0;
   std::printf("2. accuracy on %d random KKT tiles: max rel. error %.1e, inertia mismatches %d: %s\n",
               ntiles, max_err, inertia_bad, acc ? "OK" : "FAILED");
   failures += !acc;

   // ---- 3. concurrency (MumpsBlock serializes the MUMPS calls themselves)
   int threads = 1;
#ifdef _OPENMP
   threads = omp_get_max_threads();
#endif
   std::vector<Result> serial(ntiles);
   for (int k = 0; k < ntiles; ++k) serial[k] = run(tiles[k], 2);
   long mismatches = 0;
   for (int round = 0; round < rounds; ++round) {
      std::vector<dd::MumpsBlock> blocks(ntiles);   // serial creation
#pragma omp parallel for schedule(dynamic) reduction(+ : mismatches)
      for (int k = 0; k < ntiles; ++k) {
         const Tile& t = tiles[k];
         dd::MumpsBlock& m = blocks[k];
         Result r;
         // analyze once, then factorize + solve twice (values-only refresh)
         bool ok = m.analyze(t.n, t.p, t.r, t.c);
         for (int rep = 0; rep < 2 && ok; ++rep) {
            ok = m.factorize(t.v.data());
            r.neg = m.negative();
            r.S.assign(m.schur(), m.schur() + (size_t)t.p * t.p);
            r.x.resize((size_t)t.n * 2);
            for (size_t i = 0; i < r.x.size(); ++i) r.x[i] = std::sin(0.37 * (double)i);
            ok = ok && m.solve(r.x.data(), 2);
         }
         if (!ok || r.neg != serial[k].neg || r.S != serial[k].S || r.x != serial[k].x) ++mismatches;
      }
   }                                                // serial destruction
   const bool conc = mismatches == 0;
   std::printf("3. concurrency: %d rounds × %d tiles on %d threads (callers in parallel,"
               " MUMPS calls serialized), %ld results differ from serial: %s\n",
               rounds, ntiles, threads, mismatches,
               threads < 2 ? "NOT TESTED (1 thread; build with OMP=1)" : conc ? "OK" : "FAILED");
   failures += !conc;

   std::printf("%s\n", failures ? "MUMPS CHECK FAILED" : "MUMPS CHECK PASSED");
   return failures ? 1 : 0;
}
