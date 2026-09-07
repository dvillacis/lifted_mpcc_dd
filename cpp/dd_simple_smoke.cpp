// Standalone validation gate for dd_solver_simple.hpp — the analogue of
// mumps_smoke.cpp / ma97_smoke.cpp, but for the READABLE arrowhead solver.
//
// It needs NEITHER IPOPT NOR HSL: ddsimple::Arrowhead is pure Eigen, and
// DD_SIMPLE_NO_IPOPT compiles the header without its IPOPT adapter.  So this is
// the one piece of the project that builds anywhere Eigen does — run it first on
// any new machine, and after any edit to dd_solver_simple.hpp.
//
//   clang++ -std=c++17 -O2 -I$(brew --prefix eigen)/include/eigen3 -I. \
//           dd_simple_smoke.cpp -o dd_simple_smoke && ./dd_simple_smoke
//
// Two synthetic arrowhead KKT matrices are built with a KNOWN owner map, and the
// decomposition is checked against dense references that share none of its code:
//
//   A. no peel        — is the arrowhead solve the same step a dense LU gives,
//                       and is Σ_k In(W_k) + In(S) the true number of negative
//                       eigenvalues (from a dense symmetric eigendecomposition)?
//   B. with the peel  — the border also carries dual unknowns and a dense
//                       "alpha" column, so S is indefinite and CG can only run
//                       on the peeled complement.  Same two questions, plus:
//                       CG must actually carry the solves (no fallbacks).
//   C. predicted      — problem B with Options::PREDICTED: S is never assembled
//                       or factorized, and In(S) is predicted as In(T) from the
//                       tiny dense peel complement.  The prediction is checked
//                       against the DENSE EIGENVALUE COUNT of the whole matrix,
//                       which shares no code with it.
//   D. no inertia     — Options::NONE: nothing is reported at all.  The only
//                       question left is whether the STEP is still the one a
//                       dense solve gives.
//
// The dense references cost O(dim³), so the test problems are deliberately tiny.
#define DD_SIMPLE_NO_IPOPT
#include "dd_solver_simple.hpp"

#include <Eigen/Eigenvalues>
#include <cstdio>
#include <random>

using namespace ddsimple;

namespace {

// Accumulates a symmetric matrix in two forms at once: the lower-triangle
// triplet list the solver consumes, and a dense copy for the reference.
struct Builder {
   int dim;
   Eigen::MatrixXd A;
   std::vector<int> ir, jc;
   std::vector<double> vv;
   explicit Builder(int d) : dim(d), A(Eigen::MatrixXd::Zero(d, d)) {}
   void put(int i, int j, double v) {
      if (i < j) std::swap(i, j);            // keep the lower triangle
      ir.push_back(i); jc.push_back(j); vv.push_back(v);
      A(i, j) += v;
      if (i != j) A(j, i) += v;
   }
};

int reference_negatives(const Eigen::MatrixXd& A) {
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
   int n = 0;
   for (int i = 0; i < A.rows(); ++i) if (es.eigenvalues()[i] < 0.0) ++n;
   return n;
}

// ---------------------------------------------------------------------------
//  A.  K subdomains of (nx primal + nc dual), a border of nbp primal unknowns.
//      S comes out positive definite, so no peel is needed and CG runs on all
//      of it.  Run twice: once with CG, once with the direct LDLᵀ of S.
// ---------------------------------------------------------------------------
int test_no_peel() {
   const int K = 3, nx = 8, nc = 4, nbp = 5, per = nx + nc;
   const int dim = K * per + nbp;
   std::vector<int> owner(dim, -1);
   for (int k = 0; k < K; ++k)
      for (int j = 0; j < per; ++j) owner[k * per + j] = k;

   std::mt19937 rng(7);
   std::uniform_real_distribution<double> U(-1.0, 1.0);
   int failures = 0;

   for (int trial = 0; trial < 6; ++trial) {
      Builder B(dim);
      for (int k = 0; k < K; ++k) {
         const int b = k * per;
         for (int i = 0; i < nx; ++i) {                       // primal: diagonally dominant
            B.put(b + i, b + i, 3.0 + 0.5 * U(rng));
            for (int j = 0; j < i; ++j) if (U(rng) > 0.5) B.put(b + i, b + j, 0.3 * U(rng));
            for (int q = 0; q < nbp; ++q) if (U(rng) > 0.7) B.put(K * per + q, b + i, 0.4 * U(rng));
         }
         for (int c = 0; c < nc; ++c) {                       // dual rows: tiny -delta_c
            const int d = b + nx + c;
            B.put(d, d, -1e-3);
            for (int i = 0; i < nx; ++i) if (U(rng) > 0.2) B.put(d, b + i, U(rng));
            for (int q = 0; q < nbp; ++q) if (U(rng) > 0.4) B.put(K * per + q, d, U(rng));
         }
      }
      for (int q = 0; q < nbp; ++q) {                         // the corner block C
         B.put(K * per + q, K * per + q, 1.0 + 0.2 * U(rng));
         for (int r = 0; r < q; ++r) if (U(rng) > 0.5) B.put(K * per + q, K * per + r, 0.2 * U(rng));
      }

      Arrowhead::Options opt;
      opt.use_cg = (trial % 2 == 0);      // alternate CG / direct
      opt.cg_tol = 1e-12;
      opt.cg_maxit = 2000;

      Arrowhead ah;
      if (!ah.set_structure(dim, B.ir, B.jc, owner, K, opt)) { printf("  setup FAILED\n"); return 1; }
      std::copy(B.vv.begin(), B.vv.end(), ah.values());
      if (ah.factorize() != Arrowhead::OK) { printf("  trial %d: factorize FAILED\n", trial); ++failures; continue; }

      const int ref_neg = reference_negatives(B.A);
      Eigen::VectorXd rhs(dim);
      for (int i = 0; i < dim; ++i) rhs[i] = U(rng);
      const Eigen::VectorXd xref = B.A.fullPivLu().solve(rhs);

      std::vector<double> x(rhs.data(), rhs.data() + dim);
      const bool ok = ah.solve(x.data());
      Eigen::Map<Eigen::VectorXd> xv(x.data(), dim);
      const double err = (xv - xref).norm() / xref.norm();
      const double res = (B.A * xv - rhs).norm() / rhs.norm();
      const bool inertia_ok = (ah.negative_eigenvalues() == ref_neg);
      printf("  trial %d  interface=%-6s  inertia %d vs %d %s  rel-err %.2e  rel-res %.2e\n",
             trial, opt.use_cg ? "cg" : "direct", ah.negative_eigenvalues(), ref_neg,
             inertia_ok ? "MATCH" : "MISMATCH", err, res);
      if (!inertia_ok || !ok || !(res < 1e-9)) ++failures;
   }
   return failures;
}

// ---------------------------------------------------------------------------
//  B.  Same idea, but the border also carries nbd DUAL unknowns and a scalar
//      alpha that couples into every subdomain.  The duals make S indefinite,
//      so this exercises the admissibility gate, the peel and the CG path that
//      only exists because of them.
// ---------------------------------------------------------------------------
int test_peel(Arrowhead::Options::InertiaMode mode) {
   const bool exact = (mode == Arrowhead::Options::EXACT);
   const bool no_inertia = (mode == Arrowhead::Options::NONE);
   const int K = 4, nx = 10, nc = 5, nbp = 6, nbd = 3;
   const int n_primal = K * nx + nbp + 1;               // ... + the alpha column
   const int dim = n_primal + K * nc + nbd;
   const int oa = K * nx + nbp;                         // alpha's KKT index
   std::vector<int> owner(dim, -1);
   for (int k = 0; k < K; ++k) {
      for (int j = 0; j < nx; ++j) owner[k * nx + j] = k;
      for (int j = 0; j < nc; ++j) owner[n_primal + k * nc + j] = k;
   }
   auto bp = [&](int q) { return K * nx + q; };                 // border primal (kept)
   auto bd = [&](int q) { return n_primal + K * nc + q; };      // border dual   (peeled)

   std::mt19937 rng(11);
   std::uniform_real_distribution<double> U(-1.0, 1.0);
   int failures = 0;

   for (int trial = 0; trial < 4; ++trial) {
      Builder B(dim);
      for (int k = 0; k < K; ++k) {
         for (int i = 0; i < nx; ++i) {
            B.put(k * nx + i, k * nx + i, 4.0 + 0.5 * U(rng));
            for (int j = 0; j < i; ++j) if (U(rng) > 0.6) B.put(k * nx + i, k * nx + j, 0.3 * U(rng));
            B.put(oa, k * nx + i, 0.3 * U(rng));               // alpha is in EVERY subdomain
            for (int q = 0; q < nbp; ++q) if (U(rng) > 0.6) B.put(bp(q), k * nx + i, 0.4 * U(rng));
         }
         for (int c = 0; c < nc; ++c) {
            const int d = n_primal + k * nc + c;
            B.put(d, d, -1e-6);
            for (int i = 0; i < nx; ++i) if (U(rng) > 0.3) B.put(d, k * nx + i, U(rng));
            B.put(d, oa, 0.5 * U(rng));
            for (int q = 0; q < nbp; ++q) if (U(rng) > 0.6) B.put(d, bp(q), U(rng));
         }
         for (int q = 0; q < nbd; ++q)                          // border duals see every block
            for (int i = 0; i < nx; ++i) if (U(rng) > 0.6) B.put(bd(q), k * nx + i, 0.6 * U(rng));
      }
      for (int q = 0; q < nbp; ++q) {
         B.put(bp(q), bp(q), 2.0 + 0.3 * U(rng));
         B.put(oa, bp(q), 0.2 * U(rng));
      }
      B.put(oa, oa, 1.5);
      for (int q = 0; q < nbd; ++q) { B.put(bd(q), bd(q), -1e-6); B.put(bd(q), oa, 0.4 * U(rng)); }

      Arrowhead::Options opt;
      opt.use_cg = true;
      opt.inertia = mode;
      opt.n_primal = n_primal;      // -> the nbd border duals are peeled
      opt.alpha_index = oa;         // -> so is alpha
      opt.cg_tol = 1e-12;
      opt.cg_maxit = 2000;

      Arrowhead ah;
      if (!ah.set_structure(dim, B.ir, B.jc, owner, K, opt)) { printf("  setup FAILED\n"); return 1; }
      std::copy(B.vv.begin(), B.vv.end(), ah.values());
      if (ah.factorize() != Arrowhead::OK) { printf("  trial %d: factorize FAILED\n", trial); ++failures; continue; }

      const int ref_neg = no_inertia ? -1 : reference_negatives(B.A);
      double worst_err = 0, worst_res = 0;
      for (int r = 0; r < 3; ++r) {
         Eigen::VectorXd rhs(dim);
         for (int i = 0; i < dim; ++i) rhs[i] = U(rng);
         const Eigen::VectorXd xref = B.A.fullPivLu().solve(rhs);
         std::vector<double> x(rhs.data(), rhs.data() + dim);
         if (!ah.solve(x.data())) { printf("  solve reported failure\n"); ++failures; }
         Eigen::Map<Eigen::VectorXd> xv(x.data(), dim);
         worst_err = std::max(worst_err, (xv - xref).norm() / xref.norm());
         worst_res = std::max(worst_res, (B.A * xv - rhs).norm() / rhs.norm());
      }
      const Arrowhead::Stats& st = ah.stats();
      const bool inertia_ok = no_inertia || (ah.negative_eigenvalues() == ref_neg);
      if (no_inertia)
         printf("  trial %d  inertia not reported        rel-err %.2e  rel-res %.2e  "
                "cg %ld/%ld carried, %ld its\n",
                trial, worst_err, worst_res, st.solves - st.fallbacks, st.solves, st.iters);
      else
         printf("  trial %d  inertia %s %d vs dense %d %s  rel-err %.2e  "
                "rel-res %.2e  cg %ld/%ld carried\n",
                trial, exact ? "     " : "(pred)", ah.negative_eigenvalues(), ref_neg,
                inertia_ok ? "MATCH" : "MISMATCH", worst_err, worst_res,
                st.solves - st.fallbacks, st.solves);
      if (!inertia_ok || !(worst_res < 1e-8)) ++failures;
      if (st.indef_before || st.indef_after) {
         printf("    CG saw non-positive curvature in S_ff (before=%ld after=%ld)"
                " — evidence against the prediction's premise\n",
                st.indef_before, st.indef_after);
         ++failures;
      }
      // CG must have done the work: a silent fall-through to the direct solve
      // would still give the right answer and hide a broken Krylov path.
      if (st.fallbacks || st.skipped) { printf("    CG did NOT carry the solves\n"); ++failures; }
   }
   return failures;
}

}  // namespace

int main() {
   printf("A. arrowhead solve + Haynsworth inertia, no peel (SPD interface)\n");
   const int f1 = test_no_peel();
   printf("B. arrowhead solve + Haynsworth inertia, alpha + dual peel (indefinite S)\n");
   const int f2 = test_peel(Arrowhead::Options::EXACT);
   printf("C. the same, PREDICTED inertia: S never assembled, In(S) = In(T)\n");
   const int f3 = test_peel(Arrowhead::Options::PREDICTED);
   printf("D. the same, NO inertia reported\n");
   const int f4 = test_peel(Arrowhead::Options::NONE);
   const int failures = f1 + f2 + f3 + f4;
   printf(failures ? "\nFAILED: %d check(s)\n" : "\nOK: all checks passed (%d failures)\n",
          failures);
   return failures ? 1 : 0;
}
