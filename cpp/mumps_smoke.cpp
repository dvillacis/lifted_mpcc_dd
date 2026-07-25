// Standalone validation of MumpsSchurBlock against Eigen dense linear algebra
// — the first thing to run on any machine/library build, and the gate that
// must pass BEFORE the DD solver is allowed to use --wk-backend mumps. Checks
// every piece of MUMPS API fine print the wrapper relies on:
//
//   1. Schur == −B̄ W⁻¹ B̄ᵀ (sign, layout, the sentinel-mirroring of the
//      returned triangle),
//   2. INFOG(12) == the negative-eigenvalue count of W (the Haynsworth input),
//   3. JOB=3 + ICNTL(26)=0 == the plain interior solve W⁻¹b, single and
//      multi-RHS,
//   4. value-only refresh + refactorize (the per-Newton-step cycle),
//   5. duplicate triplets are summed,
//   6. pk = 0 degenerates to a plain factorization,
//   7. a singular W is REPORTED (singular() or a failed factorize) — never
//      silently factorized.
//
// Build & run:  ./build.sh mumps_smoke.cpp -o mumps_smoke && ./mumps_smoke
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "mumps_block.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
   std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
   if (!ok) ++failures;
}

// Deterministic values in (−1, 1) — no <random>, reproducible everywhere.
struct Lcg {
   unsigned s = 0x2545f491u;
   double operator()() {
      s = 1103515245u * s + 12345u;
      return (double)(s >> 8) / (double)(1u << 23) - 1.0;
   }
};

int neg_count(const Eigen::MatrixXd& M) {
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
   int c = 0;
   for (int i = 0; i < M.rows(); ++i)
      if (es.eigenvalues()[i] < 0) ++c;
   return c;
}

double rel_err(const Eigen::MatrixXd& got, const Eigen::MatrixXd& ref) {
   return (got - ref).norm() / std::max(ref.norm(), 1e-300);
}

}  // namespace

int main() {
   const int n = 30, p = 5;
   Lcg rng;

   // Symmetric indefinite W, safely nonsingular; moderately sparse B.
   Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, n);
   for (int i = 0; i < n; ++i)
      for (int j = 0; j <= i; ++j) {
         const double v = rng();
         if (i == j || std::abs(v) > 0.4) { W(i, j) = v; W(j, i) = v; }
      }
   W.diagonal().array() += 0.5;            // push off exact singularity…
   for (int i = 0; i < n; i += 3) W(i, i) -= 2.0;   // …but keep it indefinite
   Eigen::MatrixXd B = Eigen::MatrixXd::Zero(p, n);
   for (int r = 0; r < p; ++r)
      for (int c = 0; c < n; ++c) {
         const double v = rng();
         if (std::abs(v) > 0.55) B(r, c) = v;
      }

   // Augmented lower-triangle triplets: W lower + B̄ rows (n+r+1, c+1).
   // W(0,0) is deliberately SPLIT into two half-entries (duplicate summing).
   std::vector<int> irn, jcn;
   std::vector<double> va;
   for (int i = 0; i < n; ++i)
      for (int j = 0; j <= i; ++j)
         if (W(i, j) != 0.0) {
            if (i == 0 && j == 0) {
               irn.push_back(1); jcn.push_back(1); va.push_back(0.5 * W(0, 0));
               irn.push_back(1); jcn.push_back(1); va.push_back(0.5 * W(0, 0));
            } else {
               irn.push_back(i + 1); jcn.push_back(j + 1); va.push_back(W(i, j));
            }
         }
   for (int r = 0; r < p; ++r)
      for (int c = 0; c < n; ++c)
         if (B(r, c) != 0.0) {
            irn.push_back(n + r + 1); jcn.push_back(c + 1); va.push_back(B(r, c));
         }

   MumpsSchurBlock blk;
   std::printf("MUMPS version %s\n", blk.version());
   std::printf("-- analyze/factorize (n=%d, p=%d, ne=%zu) --\n", n, p, va.size());
   if (!blk.analyze(n, p, irn, jcn)) {
      std::printf("  [FAIL] analyze (status=%d)\n", blk.status());
      return 1;
   }
   std::copy(va.begin(), va.end(), blk.values());
   check(blk.factorize(), "factorize");
   check(!blk.singular(), "not reported singular");

   // 1. Schur block, sign and layout.
   const Eigen::MatrixXd Sref = -B * W.ldlt().solve(B.transpose());
   Eigen::Map<const Eigen::MatrixXd> Sgot(blk.schur(), p, p);
   const double serr = rel_err(Sgot, Sref);
   std::printf("     schur rel-err %.2e\n", serr);
   check(serr < 1e-10, "Schur == -B W^-1 B^T");
   check(rel_err(Sgot, Sgot.transpose()) < 1e-14, "Schur symmetrized");

   // 2. Inertia of the factored interior.
   const int nref = neg_count(W);
   std::printf("     neg pivots %d, dense count %d\n",
               blk.negative_eigenvalues(), nref);
   check(blk.negative_eigenvalues() == nref, "INFOG(12) == In(W)_neg");

   // 3. Interior solves, single and multi RHS.
   {
      Eigen::VectorXd b(n);
      for (int i = 0; i < n; ++i) b[i] = rng();
      Eigen::VectorXd x = b;
      check(blk.solve(x.data(), 1), "solve(1 rhs) returns ok");
      check((W * x - b).norm() / b.norm() < 1e-10, "solve(1 rhs) == W^-1 b");
      Eigen::MatrixXd R(n, 3);
      for (int c = 0; c < 3; ++c)
         for (int i = 0; i < n; ++i) R(i, c) = rng();
      Eigen::MatrixXd X = R;
      check(blk.solve(X.data(), 3), "solve(3 rhs) returns ok");
      check(rel_err(W * X, R) < 1e-10, "solve(3 rhs) == W^-1 R");
   }

   // 4. Value-only refresh: scale everything by s ⇒ Schur scales by s,
   //    inertia unchanged; solves track the new values.
   {
      const double s = 1.7;
      double* av = blk.values();
      for (std::size_t e = 0; e < va.size(); ++e) av[e] = s * va[e];
      check(blk.factorize(), "refactorize after value refresh");
      Eigen::Map<const Eigen::MatrixXd> S2(blk.schur(), p, p);
      check(rel_err(S2, s * Sref) < 1e-10, "refreshed Schur == s * S");
      check(blk.negative_eigenvalues() == nref, "refreshed inertia unchanged");
      Eigen::VectorXd b(n);
      for (int i = 0; i < n; ++i) b[i] = rng();
      Eigen::VectorXd x = b;
      blk.solve(x.data(), 1);
      check(((s * W) * x - b).norm() / b.norm() < 1e-10,
            "refreshed solve == (sW)^-1 b");
      // restore the original values for good measure
      std::copy(va.begin(), va.end(), blk.values());
      check(blk.factorize(), "refactorize with original values");
   }

   // 6. pk = 0: a plain factorization, no Schur machinery.
   {
      std::vector<int> wi, wj;
      std::vector<double> wv;
      for (int i = 0; i < n; ++i)
         for (int j = 0; j <= i; ++j)
            if (W(i, j) != 0.0) {
               wi.push_back(i + 1); wj.push_back(j + 1); wv.push_back(W(i, j));
            }
      MumpsSchurBlock plain;
      std::printf("-- pk = 0 --\n");
      check(plain.analyze(n, 0, wi, wj), "analyze(pk=0)");
      std::copy(wv.begin(), wv.end(), plain.values());
      check(plain.factorize(), "factorize(pk=0)");
      check(plain.negative_eigenvalues() == nref, "inertia(pk=0)");
      Eigen::VectorXd b(n);
      for (int i = 0; i < n; ++i) b[i] = rng();
      Eigen::VectorXd x = b;
      plain.solve(x.data(), 1);
      check((W * x - b).norm() / b.norm() < 1e-10, "solve(pk=0)");
   }

   // 7. Singular W must be REPORTED, not silently factorized. Zero out row/col
   //    5 of W (the appended diagonal keeps the variable structurally alive).
   {
      Eigen::MatrixXd Ws = W;
      Ws.row(5).setZero();
      Ws.col(5).setZero();
      std::vector<int> wi, wj;
      std::vector<double> wv;
      for (int i = 0; i < n; ++i)
         for (int j = 0; j <= i; ++j)
            if (Ws(i, j) != 0.0) {
               wi.push_back(i + 1); wj.push_back(j + 1); wv.push_back(Ws(i, j));
            }
      MumpsSchurBlock sing;
      std::printf("-- singular W --\n");
      check(sing.analyze(n, 0, wi, wj), "analyze(singular)");
      std::copy(wv.begin(), wv.end(), sing.values());
      const bool fok = sing.factorize();
      std::printf("     factorize=%d singular=%d status=%d rank=%d/%d\n",
                  (int)fok, (int)sing.singular(), sing.status(), sing.rank(), n);
      check(!fok || sing.singular(), "singular W reported");
   }

   std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL OK" : "FAILED",
               failures, failures == 1 ? "" : "s");
   return failures == 0 ? 0 : 1;
}
