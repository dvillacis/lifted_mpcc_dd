// save.hpp — write a run to a NumPy .npz archive.
//
// The archive holds the instance (clean and noisy image), the solution, a
// one-row run summary ("levels") and the per-iteration μ-trace, so a plot needs
// nothing else.  Only the ORIGINAL variables are written: at a feasible point
// every copy equals its consensus variable.  Field names and shapes are those
// python/plot_slurm.py reads.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "image.hpp"
#include "npz_writer.hpp"
#include "tv_mpcc.hpp"

// The run summary (one row of "levels").
struct RunSummary {
   double t;         // relaxation level at the end
   int status;       // IPM status (ipm.hpp: 0 converged, 1 iteration limit,
                     // 2 level gate, 3 acceptable, −1 failure)
   int iters;        // IPM iterations
   double comp_res;  // max r·(1 − δ) at the final iterate
   double weight;    // α at the final iterate
   double obj;       // objective at the final iterate
   double xi_max;    // largest complementarity multiplier |λ_comp|
   bool converged;
};

inline bool save_npz(const std::string& file, const TvMpcc& p, const Image& img,
                     const RunSummary& run, int nsub, double sigma) {
   const std::vector<double>& x = p.solution;
   const size_t N = (size_t)p.N, nc = (size_t)p.nc;
   npz::Writer w(file);
   if (!w.ok()) { std::fprintf(stderr, "cannot write %s\n", file.c_str()); return false; }

   w.scalar_i("N", p.N);
   w.scalar_i("nsub", nsub);
   w.scalar_i("n_var", p.n_orig);
   w.scalar_d("sigma", sigma);
   w.scalar_d("t_last", run.t);
   w.scalar_i("weight_exp", 0);   // linear weight
   w.scalar_i("averaged", 0);     // one-sided stencil
   w.scalar_i("consensus", 1);

   w.array_d("u_clean", img.clean.data(), {N, N});
   w.array_d("f", img.noisy.data(), {N, N});

   w.array_d("u", x.data() + p.ou, {N, N});
   w.array_d("qx", x.data() + p.oqx, {nc, nc});
   w.array_d("qy", x.data() + p.oqy, {nc, nc});
   w.array_d("r", x.data() + p.oR, {nc, nc});
   w.array_d("delta", x.data() + p.oD, {nc, nc});
   w.array_d("theta", x.data() + p.oTh, {nc, nc});
   w.scalar_d("alpha", x[p.oa]);
   w.scalar_d("weight", x[p.oa]);
   w.array_d("x", x.data(), {(size_t)p.n_orig});

   const std::vector<double> row = {run.t,        (double)run.status, (double)run.iters,
                                    run.comp_res, run.weight,         run.obj,
                                    run.xi_max,   run.converged ? 1.0 : 0.0};
   w.array_d("levels", row.data(), {1, 8});
   w.array_d("mu_trace", p.mu_trace.data(), {p.mu_trace.size() / 5, 5});

   if (!w.close()) { std::fprintf(stderr, "failed writing %s\n", file.c_str()); return false; }
   return true;
}
