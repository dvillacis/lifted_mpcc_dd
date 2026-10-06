// Which IPOPT phase the current factorization belongs to, as seen by the TNLP.
//
// IPOPT's feasibility restoration evaluates the ORIGINAL problem's Hessian with
// obj_factor = 0 (its own objective replaces the original one), and a regular
// iteration never does. The TNLP records that in eval_h, which IPOPT calls
// before the factorizations of each iteration, and the linear solver reads it
// (--drop-corner-reg in dd_solve_2d, see dd_solver_simple.hpp). A plain flag is
// enough: IPOPT is single-threaded at this level.
#ifndef IPOPT_PHASE_HPP
#define IPOPT_PHASE_HPP

#include <functional>
#include <vector>

namespace ipopt_phase {
inline bool& restoration() {
   static bool r = false;
   return r;
}

// DDS_VERIFY_WK: the current IPOPT iterate (TNLP representation, unscaled),
// recorded by the TNLP's intermediate_callback, and a hook the linear solver
// calls with the W_k blocks it assembled for the next factorization — so the
// driver can rebuild them from the TNLP's own derivatives and compare.
struct Iterate {
   std::vector<double> x, z_L, z_U, lambda;
   int iter = -1;
   bool fresh = false;   // set by the callback, cleared once verified
};
inline Iterate& iterate() {
   static Iterate it;
   return it;
}
// The barrier parameter of the current IPOPT iterate, recorded by the TNLP's
// intermediate_callback (--block-inertia scales its δ_C^k with μ^{1/4}, as
// IPOPT does its own δ_c). Negative until the first callback.
inline double& mu() {
   static double m = -1.0;
   return m;
}
struct WkEntry { int gi, gj; double v; };   // global KKT indices, as stored
using WkHook = std::function<void(const std::vector<std::vector<WkEntry>>&)>;
inline WkHook& wk_hook() {
   static WkHook h;
   return h;
}
}  // namespace ipopt_phase

#endif  // IPOPT_PHASE_HPP
