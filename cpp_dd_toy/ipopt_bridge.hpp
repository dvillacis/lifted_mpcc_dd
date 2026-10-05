// ipopt_bridge.hpp — make SchurDD IPOPT's linear solver.
//
// IPOPT solves every Newton system through a "sparse symmetric linear solver"
// interface.  DDLinearSolver implements it on top of dd::SchurDD, and
// DDSolverBuilder makes IPOPT use it instead of MUMPS/MA27:
//
//     app->OptimizeNLP(new TNLPAdapter(problem), builder);
//
//   InitializeStructure  KKT sparsity pattern (triplets)  →  SchurDD::set_structure
//   GetValuesArrayPtr    where IPOPT writes the values     →  SchurDD::values
//   MultiSolve           factorize if new, check inertia, solve each right-hand side
//   NumberOfNegEVals     Σ #neg(W_k) + #neg(S)  (Haynsworth)
//
// If we answer SINGULAR or WRONG_INERTIA, IPOPT increases its regularization
// (δ_w on the primal diagonal, δ_c on the dual one) and asks again.  That loop
// is IPOPT's; the solver only reports.
#pragma once

#include <utility>
#include <vector>

#include "IpAlgBuilder.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTSymLinearSolver.hpp"
#include "schur_dd.hpp"

class DDLinearSolver : public Ipopt::SparseSymLinearSolverInterface {
   using Index = Ipopt::Index;
   using Number = Ipopt::Number;

public:
   // IPOPT constructs the solver itself, so the partition is handed over through
   // static state set before the solve starts.
   static void configure(std::vector<int> owner, int n_tiles, const dd::SchurDD::Options& opt) {
      owner_map() = std::move(owner);
      tiles() = n_tiles;
      options() = opt;
   }
   // Counters of every instance (main algorithm and restoration phase) pooled.
   static dd::SchurDD::Stats& stats() { static dd::SchurDD::Stats s; return s; }
   static long& wrong_inertia() { static long n = 0; return n; }
   static long& late_breakdowns() { static long n = 0; return n; }

   DDLinearSolver() { s_.set_stats_sink(&stats()); }

   bool InitializeImpl(const Ipopt::OptionsList&, const std::string&) override { return true; }
   EMatrixFormat MatrixFormat() const override { return Triplet_Format; }
   bool ProvidesInertia() const override { return true; }
   bool IncreaseQuality() override { return false; }
   Index NumberOfNegEVals() const override { return s_.negative_eigenvalues(); }

   Ipopt::ESymSolverStatus InitializeStructure(Index dim, Index nnz, const Index* ia,
                                               const Index* ja) override {
      std::vector<int> irow(nnz), jcol(nnz);
      for (Index t = 0; t < nnz; ++t) {   // IPOPT is 1-based
         irow[t] = ia[t] - 1;
         jcol[t] = ja[t] - 1;
      }
      return s_.set_structure((int)dim, irow, jcol, owner_map(), tiles(), options())
                ? Ipopt::SYMSOLVER_SUCCESS
                : Ipopt::SYMSOLVER_FATAL_ERROR;
   }

   Number* GetValuesArrayPtr() override { return s_.values(); }

   Ipopt::ESymSolverStatus MultiSolve(bool new_matrix, const Index*, const Index*, Index nrhs,
                                      Number* rhs, bool check_inertia,
                                      Index wanted_neg) override {
      if (new_matrix) {
         if (!s_.factorize()) return Ipopt::SYMSOLVER_SINGULAR;
         fresh_ = true;
      }
      if (check_inertia && s_.negative_eigenvalues() != wanted_neg) {
         ++wrong_inertia();
         return Ipopt::SYMSOLVER_WRONG_INERTIA;
      }
      for (Index c = 0; c < nrhs; ++c) {
         if (s_.solve(rhs + (size_t)c * s_.dim())) continue;
         // PCG met pᵀSp ≤ 0, so S is not positive definite (paper Sec. 3.5).
         // IPOPT can only act on that while it is still deciding whether to
         // accept this matrix: the first solve after the factorization.  A later
         // solve (iterative refinement) that breaks down keeps the CG iterate;
         // refusing there makes IPOPT abort.  So the check is not rigorous, as
         // the paper says.
         if (fresh_ && check_inertia) {
            ++wrong_inertia();
            return Ipopt::SYMSOLVER_WRONG_INERTIA;
         }
         ++late_breakdowns();
      }
      fresh_ = false;
      return Ipopt::SYMSOLVER_SUCCESS;
   }

private:
   static std::vector<int>& owner_map() { static std::vector<int> v; return v; }
   static int& tiles() { static int v = 1; return v; }
   static dd::SchurDD::Options& options() { static dd::SchurDD::Options v; return v; }

   dd::SchurDD s_;
   bool fresh_ = false;   // no solve yet with the current factorization
};

class DDSolverBuilder : public Ipopt::AlgorithmBuilder {
public:
   Ipopt::SmartPtr<Ipopt::SymLinearSolver> SymLinearSolverFactory(
      const Ipopt::Journalist&, const Ipopt::OptionsList&, const std::string&) override {
      // Called once for the main algorithm and once for the restoration phase.
      // Each gets its own instance; both use the same owner map, since IPOPT
      // condenses the restoration KKT system back to the original pattern.
      return new Ipopt::TSymLinearSolver(new DDLinearSolver(), nullptr);   // no scaling
   }
};
