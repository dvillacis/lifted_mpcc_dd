// nlp.hpp — what the interior point method (ipm.hpp) needs from a problem.
//
//   min f(x)   s.t.  g_l ≤ g(x) ≤ g_u,   x_l ≤ x ≤ x_u
//
// Rows with g_l = g_u are equalities; the others get a slack inside the IPM.
// Derivatives are sparse in triplet form (0-based): the Jacobian of g, and the
// Hessian of the Lagrangian σ·f + λᵀg, one triangle.  Values are returned in
// the order of the structure.
//
// For the domain decomposition every variable and every row names its tile;
// variable tile −1 marks a consensus (complicating) variable.  No row may mix
// variables of two different tiles.
//
// LOCAL VIEW.  With MPI, each rank holds only its part of the problem: the
// variables and rows of its tiles (dd::Comm::first_tile) and all consensus
// variables, which every rank holds, in the same order.  Every size, vector
// and index below is in this local numbering, and the derivatives are those
// of the local rows.  So that each rank can evaluate its part alone:
//   · f = Σ_k f_k(x_k) + f_0(y): every term depends on the variables of one
//     tile, or on consensus variables only.  f() returns the GLOBAL value,
//     summed with dd::TileSum (bit-identical for any number of ranks); the
//     gradient and the Hessian of f_0 are formed identically on every rank;
//   · the Hessian has no entry between two consensus variables except those
//     of f_0 (rows may involve consensus variables only linearly).
// The IPM forms all global quantities (norms, sums, step lengths) itself.
#pragma once

#include <string>
#include <vector>

#include "comm.hpp"

// A primal-dual point in the NLP's own layout (local): variables and their
// bound multipliers; per row, the slack (inequality rows; unused for
// equalities), its bound multipliers and λ.  The IPM can export its final
// point in this form and warm-start from one, also for a modified problem
// (other bounds, rows turned into equalities).
struct NLPPoint {
   std::vector<double> x, zl, zu;         // per variable
   std::vector<double> s, szl, szu, lam;  // per row
   double mu = 0.0;                       // the barrier parameter it was at
};

class NLP {
public:
   virtual ~NLP() = default;

   // ---- sizes, bounds, start
   virtual int num_vars() const = 0;
   virtual int num_rows() const = 0;
   virtual void bounds(double* xl, double* xu, double* gl, double* gu) const = 0;
   virtual void start(double* x) const = 0;

   // ---- functions and derivatives
   virtual double f(const double* x) = 0;   // the global value (collective)
   virtual void grad_f(const double* x, double* grad) = 0;
   virtual void g(const double* x, double* g) = 0;
   virtual void jac_structure(std::vector<int>& row, std::vector<int>& col) const = 0;
   virtual void jac_values(const double* x, double* val) = 0;
   virtual void hess_structure(std::vector<int>& row, std::vector<int>& col) const = 0;
   virtual void hess_values(const double* x, double sigma, const double* lam, double* val) = 0;

   // ---- optional structure the IPM exploits
   // Elastic variables: the LAST n_elastic() variables each enter exactly one
   // row, linearly, and have no Hessian entry (the restoration phase's p and
   // n).  The IPM eliminates them from its Newton systems: they move into the
   // row diagonal, and the KKT pattern is that of the problem without them.
   virtual int n_elastic() const { return 0; }
   // A diagonal part of the Hessian of σ·f, kept out of hess_structure (so that
   // the pattern stays that of another problem): d[i] for every variable.
   // false: there is none.
   virtual bool hess_diag(const double* /*x*/, double /*sigma*/, double* /*d*/) { return false; }
   // The bounds of complementarity variables (MPCC): lo[i] / up[i] = 1 if the
   // lower / upper bound of entry i of the IPM's [x | s] (one slack per
   // inequality row, in row order) is one.  For Raghunathan & Biegler's
   // modified complementarity block (IPM::Options::vw_mu).
   virtual void complementarity_bounds(char* /*lo*/, char* /*up*/) const {}

   // ---- partition
   virtual int n_tiles() const = 0;                        // all tiles, on all ranks
   virtual const std::vector<int>& var_tile() const = 0;   // n entries, −1 = consensus
   virtual const std::vector<int>& row_tile() const = 0;   // m entries

   // ---- hooks
   // Called once per IPM iteration, after the optimality errors are computed and
   // before the step.  The problem may change itself here (the MPCC tightens its
   // relaxation); the IPM re-evaluates afterwards.  Returning false stops.
   // x is the IPM's primal vector [x | s]: the n variables, then one slack per
   // inequality row (g_l ≠ g_u) in row order.  dual = ∇L − z_L + z_U on the same
   // vector (its max is inf_du), lam the row multipliers in row order, zl/zu the
   // bound multipliers of [x | s].  All local; the scalars are global.  The
   // hook is called on every rank (collective calls are allowed), and what it
   // returns, and bounds_changed(), must agree on all ranks.
   struct Iterate {
      int iter;
      double obj, inf_pr, inf_du, mu, reg;
      const double* x;
      const double* dual = nullptr;
      const double* lam = nullptr;
      const double* zl = nullptr;
      const double* zu = nullptr;
      bool barrier_solved = false;        // E_μ ≤ κ_ε·μ: the barrier problem at this μ is solved
      int pr_block = -1, du_block = -1;   // last step: the entry of [x | s] whose bound
                                          // set α_pr / α_du (−1: none, full step, or
                                          // on another rank)
   };
   virtual bool iteration(const Iterate&) { return true; }
   // False while the problem is still being changed towards its final form (a
   // continuation): the IPM does not stop at its tolerance then.
   virtual bool final_form() const { return true; }
   // True right after iteration() moved some bounds (bounds() changed): the IPM
   // re-reads them.  Only LOOSENING keeps the iterate interior; a bound that
   // moved inward past the iterate is met by pushing the iterate back inside.
   virtual bool bounds_changed() { return false; }
   // Appended to the IPM's log line (e.g. the relaxation level).
   virtual std::string log_extra() const { return {}; }
   virtual void finalize(int /*status*/, const double* /*x*/, const double* /*lam*/,
                         double /*obj*/) {}
};
