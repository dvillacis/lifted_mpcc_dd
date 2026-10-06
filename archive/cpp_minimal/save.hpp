// save.hpp — write the solution file, in the same two formats as ../cpp.
//
//   --save-solution run.npz   NumPy archive of named, shaped arrays (recommended)
//   --save-solution run.txt   the older positional text format
//
// Both hold the instance (clean and noisy image), the solution, the one-row
// continuation summary and the μ-trace, so a plot needs nothing else.  Only the
// ORIGINAL variables are written: at a feasible point every copy equals its
// consensus variable, so the leading n_orig entries are the whole solution.
// The files are byte-identical to what ../cpp/dd_solve_2d writes for the same run,
// so python/plot_2d.py and the other readers work unchanged.
#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "npz_writer.hpp"
#include "problem.hpp"

// One row of the continuation summary.  The μ-coupled run is one IPOPT solve,
// hence one row.
struct Level {
   double t;         // relaxation level at the end
   int status;       // IPOPT return status
   int iters;        // IPOPT iterations
   double comp_res;  // max r·(1 − δ) at the final iterate
   double weight;    // α at the final iterate
   double obj;       // objective at the final iterate
   double xi_max;    // largest complementarity multiplier |λ_comp|
   bool converged;
};

inline void save_npz(const std::string& file, const ConsensusTV& p, const Image& img,
                     const Level& level, const std::vector<double>& x, double t_last,
                     int nsub, double sigma) {
   const size_t N = (size_t)p.N, nc = (size_t)p.nc;
   npz::Writer w(file);
   if (!w.ok()) { std::cerr << "cannot write " << file << "\n"; return; }

   w.scalar_i("N", p.N);
   w.scalar_i("nsub", nsub);
   w.scalar_i("n_var", p.n_orig);
   w.scalar_d("sigma", sigma);
   w.scalar_d("t_last", t_last);
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

   const std::vector<double> row = {level.t,        (double)level.status, (double)level.iters,
                                    level.comp_res, level.weight,         level.obj,
                                    level.xi_max,   level.converged ? 1.0 : 0.0};
   w.array_d("levels", row.data(), {1, 8});
   w.array_d("mu_trace", p.mu_trace.data(), {p.mu_trace.size() / 5, 5});

   if (!w.close()) std::cerr << "failed writing " << file << "\n";
}

inline void save_txt(const std::string& file, const ConsensusTV& p, const Image& img,
                     const Level& level, const std::vector<double>& x, double t_last,
                     int nsub, double sigma) {
   std::ofstream out(file);
   if (!out) { std::cerr << "cannot write " << file << "\n"; return; }
   out << std::setprecision(17);
   // header: #variables, #levels, t_last, nsub, weight_exp
   out << p.n_orig << " " << 1 << " " << t_last << " " << nsub << " " << 0 << "\n";
   out << level.t << " " << level.status << " " << level.iters << " " << level.comp_res
       << " " << level.weight << " " << level.obj << " " << level.xi_max << " "
       << (level.converged ? 1 : 0) << "\n";
   for (int i = 0; i < p.n_orig; ++i) out << x[i] << (i + 1 < p.n_orig ? ' ' : '\n');
   // the instance: N, averaged, sigma, then the clean and noisy images
   out << p.N << " " << 0 << " " << sigma << "\n";
   for (const std::vector<double>* v : {&img.clean, &img.noisy})
      for (size_t i = 0; i < v->size(); ++i) out << (*v)[i] << (i + 1 < v->size() ? ' ' : '\n');
   // the μ-trace: row count, column count, then one row per iteration
   out << p.mu_trace.size() / 5 << " " << 5 << "\n";
   for (size_t i = 0; i + 4 < p.mu_trace.size(); i += 5) {
      out << (long)p.mu_trace[i];
      for (size_t j = 1; j < 5; ++j) out << " " << p.mu_trace[i + j];
      out << "\n";
   }
}

// Choose the format from the extension: .txt → text, anything else → .npz.
inline void save_solution(const std::string& file, const ConsensusTV& p, const Image& img,
                          const Level& level, const std::vector<double>& x, double t_last,
                          int nsub, double sigma) {
   const bool txt = file.size() > 4 && file.compare(file.size() - 4, 4, ".txt") == 0;
   if (txt) save_txt(file, p, img, level, x, t_last, nsub, sigma);
   else save_npz(file, p, img, level, x, t_last, nsub, sigma);
}
