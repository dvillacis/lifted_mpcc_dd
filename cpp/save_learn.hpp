// save_learn.hpp — tv_learn's outputs.
//
//   save_pair_npz   one .npz per image, with the keys of save.hpp (so that
//                   python/plot_slurm.py renders it unchanged), plus the weight
//                   map alpha_map (per cell), the block side, the operator and
//                   the mask.  A training image carries the MPCC's own solution;
//                   a held-out image the lower level solved at the learned map
//                   (pdhg.hpp), lifted as the warm start is (r = |K_ξ u|,
//                   δ = |q|, θ from q).  "alpha"/"weight" are the scalar α, or
//                   the mean of the map.
//   save_report     the run record as JSON, one row per image (training and
//                   held-out), in the format python/plot_dataset.py reads.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "npz_writer.hpp"
#include "operators.hpp"

struct LearnSummary {
   double t;
   int status, iters;
   double comp_res, weight, obj, xi_max;
   bool converged;
};

// The fields of one image, in the global layout of a single image.
struct PairFields {
   std::vector<double> u, qx, qy, r, delta, theta;
};

inline int op_code(const OpConfig& c) { return c.deblur() ? 2 : c.inpaint() ? 1 : 0; }

inline bool save_pair_npz(const std::string& file, const Instance& in, const PairFields& p,
                          const std::vector<double>& alpha_cells, const LearnSummary& run,
                          const std::vector<double>& mu_trace, int nsub, int block, double sigma,
                          const OpConfig& cfg, int n_var) {
   const size_t N = (size_t)in.N, nc = N - 1;
   npz::Writer w(file);
   if (!w.ok()) { std::fprintf(stderr, "cannot write %s\n", file.c_str()); return false; }
   w.scalar_i("N", in.N);
   w.scalar_i("nsub", nsub);
   w.scalar_i("n_var", n_var);
   w.scalar_d("sigma", sigma);
   w.scalar_d("t_last", run.t);
   w.scalar_i("weight_exp", 0);
   w.scalar_i("averaged", 0);
   w.scalar_i("consensus", 1);
   w.scalar_i("block", block);
   w.scalar_i("op", op_code(cfg));   // 0 denoise, 1 inpaint, 2 deblur
   w.scalar_i("dtv", in.dtv() ? 1 : 0);
   w.scalar_i("seed", (int)in.seed);

   w.array_d("u_clean", in.clean.data(), {N, N});
   w.array_d("f", in.f.data(), {N, N});
   if (!in.mask.empty()) w.array_d("mask", in.mask.data(), {N, N});
   w.array_d("u", p.u.data(), {N, N});
   w.array_d("qx", p.qx.data(), {nc, nc});
   w.array_d("qy", p.qy.data(), {nc, nc});
   w.array_d("r", p.r.data(), {nc, nc});
   w.array_d("delta", p.delta.data(), {nc, nc});
   w.array_d("theta", p.theta.data(), {nc, nc});
   w.array_d("alpha_map", alpha_cells.data(), {nc, nc});
   w.scalar_d("alpha", run.weight);
   w.scalar_d("weight", run.weight);

   const std::vector<double> row = {run.t,        (double)run.status, (double)run.iters,
                                    run.comp_res, run.weight,         run.obj,
                                    run.xi_max,   run.converged ? 1.0 : 0.0};
   w.array_d("levels", row.data(), {1, 8});
   w.array_d("mu_trace", mu_trace.data(), {mu_trace.size() / 5, 5});
   if (!w.close()) { std::fprintf(stderr, "failed writing %s\n", file.c_str()); return false; }
   return true;
}

// The lifted fields of a lower-level solution (u, q) from pdhg.hpp.
inline PairFields lift_fields(const Instance& in, const std::vector<double>& u,
                              const std::vector<double>& qx, const std::vector<double>& qy) {
   PairFields p;
   p.u = u;
   p.qx = qx;
   p.qy = qy;
   std::vector<double> gx, gy;
   apply_K(in, u, gx, gy);
   const size_t m = qx.size();
   p.r.resize(m);
   p.delta.resize(m);
   p.theta.resize(m);
   for (size_t e = 0; e < m; ++e) {
      p.r[e] = std::hypot(gx[e], gy[e]);
      p.delta[e] = std::hypot(qx[e], qy[e]);
      p.theta[e] = std::atan2(qy[e], qx[e]);
   }
   return p;
}

struct ReportRow {
   std::string role, path;
   int index = 0;
   unsigned seed = 0;
   double psnr_noisy = NAN, psnr_mpcc = NAN, psnr_rof = NAN, lower_kkt = NAN;
};

inline std::string json_num(double v) {
   if (!std::isfinite(v)) return "null";
   char b[32];
   std::snprintf(b, sizeof b, "%.10g", v);
   return b;
}
inline std::string json_str(const std::string& s) {
   std::string o = "\"";
   for (char c : s) {
      if (c == '"' || c == '\\') o += '\\';
      o += c;
   }
   return o + "\"";
}

// The run record.  "Q_alpha" is the scalar α (or the mean of the map), the key
// plot_dataset.py prints; "alpha_blocks" the learned block values.
inline bool save_report(const std::string& file, const std::string& tag, int S, int N, int nsub,
                        int block, double sigma, const OpConfig& cfg, const std::string& loss,
                        int iters, int status, double wall, double alpha_mean,
                        const std::vector<double>& alpha_blocks,
                        const std::vector<ReportRow>& rows, const std::string& start = "uniform",
                        double stage1_wall = 0.0) {
   FILE* fp = std::fopen(file.c_str(), "w");
   if (!fp) { std::fprintf(stderr, "cannot write %s\n", file.c_str()); return false; }
   std::fprintf(fp, "{\n  \"tag\": %s,\n  \"solver\": \"dd\",\n  \"interface\": \"cg\",\n",
                json_str(tag).c_str());
   std::fprintf(fp, "  \"S\": %d,\n  \"N\": %d,\n  \"nsub\": %d,\n  \"block\": %d,\n", S, N, nsub,
                block);
   std::fprintf(fp, "  \"sigma\": %s,\n  \"op\": %s,\n  \"dtv\": %s,\n  \"loss\": %s,\n",
                json_num(sigma).c_str(), json_str(cfg.op).c_str(), cfg.dtv ? "true" : "false",
                json_str(loss).c_str());
   std::fprintf(fp, "  \"iters\": %d,\n  \"status\": %d,\n  \"wall_s\": %s,\n", iters, status,
                json_num(wall).c_str());
   std::fprintf(fp, "  \"start\": %s,\n  \"stage1_wall_s\": %s,\n", json_str(start).c_str(),
                json_num(stage1_wall).c_str());
   std::fprintf(fp, "  \"Q_alpha\": %s,\n  \"alpha_blocks\": [", json_num(alpha_mean).c_str());
   for (size_t b = 0; b < alpha_blocks.size(); ++b)
      std::fprintf(fp, "%s%s", b ? ", " : "", json_num(alpha_blocks[b]).c_str());
   std::fprintf(fp, "],\n  \"pairs\": [\n");
   for (size_t i = 0; i < rows.size(); ++i) {
      const ReportRow& r = rows[i];
      std::fprintf(fp,
                   "    {\"role\": %s, \"path\": %s, \"index\": %d, \"seed\": %u, \"psnr_noisy\": %s,"
                   " \"psnr_mpcc\": %s, \"psnr_rof\": %s, \"lower_kkt\": %s}%s\n",
                   json_str(r.role).c_str(), json_str(r.path).c_str(), r.index, r.seed,
                   json_num(r.psnr_noisy).c_str(), json_num(r.psnr_mpcc).c_str(),
                   json_num(r.psnr_rof).c_str(), json_num(r.lower_kkt).c_str(),
                   i + 1 < rows.size() ? "," : "");
   }
   std::fprintf(fp, "  ]\n}\n");
   std::fclose(fp);
   return true;
}
