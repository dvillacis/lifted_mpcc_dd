// ipopt_bridge.hpp — make arrowhead.hpp IPOPT's linear solver.
//
// IPOPT solves every Newton system through a "sparse symmetric linear solver"
// interface.  DDLinearSolver implements that interface on top of dd::Arrowhead,
// and DDSolverBuilder tells IPOPT to use it instead of its built-in solver:
//
//     app->OptimizeNLP(new TNLPAdapter(problem), new DDSolverBuilder());
//
// What IPOPT asks of a linear solver, and how each request is answered:
//
//   InitializeStructure   the KKT sparsity pattern  →  Arrowhead::set_structure
//   GetValuesArrayPtr     where to write the values →  Arrowhead::values
//   MultiSolve            factorize if the matrix is new, check the inertia,
//                         then solve for each right-hand side
//   NumberOfNegEVals      the inertia               →  #neg(W_k) summed + #neg(T)
//
// If a factorization is refused (SINGULAR) or its inertia is not the one IPOPT
// wants (WRONG_INERTIA), IPOPT adds a diagonal regularization and asks again.
#pragma once

#include <utility>
#include <vector>

#include "IpAlgBuilder.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpTSymLinearSolver.hpp"
#include "arrowhead.hpp"

class DDLinearSolver : public Ipopt::SparseSymLinearSolverInterface {
   using Index = Ipopt::Index;
   using Number = Ipopt::Number;

public:
   // IPOPT constructs the solver itself, so the partition is handed over through
   // static state set by main.cpp before the solve starts.
   static void configure(std::vector<int> owner, int n_tiles,
                         const dd::Arrowhead::Options& opt) {
      owner_map() = std::move(owner);
      tiles() = n_tiles;
      options() = opt;
   }
   static dd::Arrowhead::Stats& stats() { static dd::Arrowhead::Stats s; return s; }

   bool InitializeImpl(const Ipopt::OptionsList&, const std::string&) override { return true; }
   EMatrixFormat MatrixFormat() const override { return Triplet_Format; }
   bool ProvidesInertia() const override { return true; }
   bool IncreaseQuality() override { return false; }
   Index NumberOfNegEVals() const override { return a_.negative_eigenvalues(); }

   Ipopt::ESymSolverStatus InitializeStructure(Index dim, Index nnz, const Index* ia,
                                               const Index* ja) override {
      std::vector<int> irow(nnz), jcol(nnz);
      for (Index t = 0; t < nnz; ++t) {   // IPOPT is 1-based
         irow[t] = ia[t] - 1;
         jcol[t] = ja[t] - 1;
      }
      return a_.set_structure((int)dim, std::move(irow), std::move(jcol), owner_map(),
                              tiles(), options())
                ? Ipopt::SYMSOLVER_SUCCESS
                : Ipopt::SYMSOLVER_FATAL_ERROR;
   }

   Number* GetValuesArrayPtr() override { return a_.values(); }

   Ipopt::ESymSolverStatus MultiSolve(bool new_matrix, const Index*, const Index*,
                                      Index nrhs, Number* rhs, bool check_inertia,
                                      Index wanted_neg) override {
      Ipopt::ESymSolverStatus status = Ipopt::SYMSOLVER_SUCCESS;
      if (new_matrix && a_.factorize() != dd::Arrowhead::OK) {
         status = Ipopt::SYMSOLVER_SINGULAR;
      } else if (check_inertia && a_.negative_eigenvalues() != wanted_neg) {
         status = Ipopt::SYMSOLVER_WRONG_INERTIA;
         ++wrong_inertia_;
      } else {
         for (Index c = 0; c < nrhs; ++c)
            if (!a_.solve(rhs + (size_t)c * a_.dim())) {
               status = Ipopt::SYMSOLVER_SINGULAR;
               break;
            }
      }
      stats() = a_.stats();
      stats().wrong_inertia = wrong_inertia_;
      return status;
   }

private:
   static std::vector<int>& owner_map() { static std::vector<int> v; return v; }
   static int& tiles() { static int v = 1; return v; }
   static dd::Arrowhead::Options& options() { static dd::Arrowhead::Options v; return v; }

   dd::Arrowhead a_;
   long wrong_inertia_ = 0;
};

class DDSolverBuilder : public Ipopt::AlgorithmBuilder {
public:
   Ipopt::SmartPtr<Ipopt::SymLinearSolver> SymLinearSolverFactory(
      const Ipopt::Journalist&, const Ipopt::OptionsList&, const std::string&) override {
      // One solver instance for the whole run, so warm starts carry over.
      static Ipopt::SmartPtr<Ipopt::SparseSymLinearSolverInterface> solver =
         new DDLinearSolver();
      return new Ipopt::TSymLinearSolver(solver, nullptr);
   }
};
