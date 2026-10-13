// operators.hpp — the lower-level problems tv_learn can learn a weight for.
//
//   u(α) = argmin_u  D(u) + Σ_e α_e |(K_ξ u)_e|
//
// The data term D(u) = ½uᵀAu − Fᵀu is one of
//   denoise   A = I,                      F = f                 (f = u_clean + noise)
//   inpaint   A = diag(m + ε(1 − m)),     F = m∘f + ε(1 − m)∘f̃  (f = m∘(u_clean + noise),
//             m ∈ {0, 1} the observed pixels, f̃ the mean of the observed values: the
//             ε term keeps D strongly convex on the holes without pulling them to 0)
//   deblur    A = CᵀC,                    F = Cᵀf               (f = C u_clean + noise)
// with C a Gaussian blur, reflected at the border (symmetric, half-sample).
// The regularizer is TV (K_ξ = K, the forward differences of tv_mpcc.hpp) or
// directional TV: (K_ξ u)_e = M_e (Kx u, Ky u)_e with M_e = I − γ ξ_e ξ_eᵀ, ξ_e =
// ∇v / √(|∇v|² + η²) from a smoothed guide image v (Ehrhardt & Betcke 2016).
//
// The optimality condition of u, the row h1 of TvLearn, is  A u − F + K_ξᵀ(α∘q) = 0.
//
// Pixels are stored row by row (image.hpp); cells (a, b) of the (N−1)² grid as
// a·(N−1) + b, anchored at their top-left node.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "image.hpp"

// A 2-D Gaussian blur with reflected borders, separable: C = c ⊗ c with the 1-D
// N×N matrix c, c[p][refl(p + k)] += κ_k, |k| ≤ w.  Reflection keeps every
// entry within w of the diagonal, so c is banded (w) and G = cᵀc too (2w).
struct Blur {
   int N = 0, w = 0;
   std::vector<double> kappa;   // 2w + 1 weights, sum 1
   std::vector<double> c;       // N × (2w+1): c[p][j] at p·(2w+1) + (j − p + w)
   std::vector<double> G;       // N × (4w+1): G[i][j] at i·(4w+1) + (j − i + 2w)

   Blur() = default;
   Blur(int N_, double sigma, int radius) : N(N_), w(radius) {
      if (w < 0) w = (int)std::ceil(2.0 * sigma);
      kappa.assign(2 * w + 1, 0.0);
      double s = 0.0;
      for (int k = -w; k <= w; ++k) s += kappa[k + w] = std::exp(-0.5 * k * k / (sigma * sigma));
      for (double& v : kappa) v /= s;
      const int cw = 2 * w + 1, gw = 4 * w + 1;
      c.assign((size_t)N * cw, 0.0);
      for (int p = 0; p < N; ++p)
         for (int k = -w; k <= w; ++k) c[(size_t)p * cw + (refl(p + k) - p + w)] += kappa[k + w];
      G.assign((size_t)N * gw, 0.0);
      for (int p = 0; p < N; ++p)
         for (int a = std::max(0, p - w); a <= std::min(N - 1, p + w); ++a)
            for (int b = std::max(0, p - w); b <= std::min(N - 1, p + w); ++b)
               G[(size_t)a * gw + (b - a + 2 * w)] += c1(p, a) * c1(p, b);
   }
   int refl(int p) const { return p < 0 ? -p - 1 : p >= N ? 2 * N - p - 1 : p; }
   double c1(int p, int j) const {   // the 1-D c, 0 off the band
      const int o = j - p + w;
      return o < 0 || o > 2 * w ? 0.0 : c[(size_t)p * (2 * w + 1) + o];
   }
   double g1(int i, int j) const {   // the 1-D G = cᵀc, 0 off the band
      const int o = j - i + 2 * w;
      return o < 0 || o > 4 * w ? 0.0 : G[(size_t)i * (4 * w + 1) + o];
   }
   // out = C u (transpose = false) or Cᵀ u, on an N×N image
   void apply(const std::vector<double>& u, std::vector<double>& out, bool transpose) const {
      std::vector<double> tmp((size_t)N * N, 0.0);
      out.assign((size_t)N * N, 0.0);
      auto m = [&](int p, int j) { return transpose ? c1(j, p) : c1(p, j); };
      for (int i = 0; i < N; ++i)          // along the rows: tmp[i][p] = Σ_j m(p, j) u[i][j]
         for (int p = 0; p < N; ++p) {
            double s = 0.0;
            for (int j = std::max(0, p - w); j <= std::min(N - 1, p + w); ++j)
               s += m(p, j) * u[(size_t)i * N + j];
            tmp[(size_t)i * N + p] = s;
         }
      for (int p = 0; p < N; ++p)          // along the columns
         for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int i = std::max(0, p - w); i <= std::min(N - 1, p + w); ++i)
               s += m(p, i) * tmp[(size_t)i * N + j];
            out[(size_t)p * N + j] = s;
         }
   }
};

struct OpConfig {
   std::string op = "denoise";   // denoise | inpaint | deblur
   double blur_sigma = 1.0;
   int blur_radius = -1;         // < 0: ⌈2σ_b⌉
   double mask_frac = 0.5;       // fraction of MISSING pixels
   double mask_eps = 1e-3;
   unsigned mask_seed = 1000;    // + the image's noise seed
   bool dtv = false;
   double dtv_gamma = 0.9, dtv_eta = 0.05, dtv_smooth = 1.0;
   std::string guide = "clean";  // clean | data: the image ξ is taken from
   bool deblur() const { return op == "deblur"; }
   bool inpaint() const { return op == "inpaint"; }
};

// One training (or held-out) pair and its lower-level data.
struct Instance {
   std::string path;
   int index = 0;                 // position in its listing (seeds the noise and the mask)
   unsigned seed = 0;
   int N = 0;
   std::vector<double> clean;     // u_clean
   std::vector<double> f;         // the observation (0 on the holes of a mask)
   std::vector<double> diag;      // A's diagonal when A is diagonal (denoise, inpaint)
   std::vector<double> rhs;       // F
   std::vector<double> mask;      // m (inpaint), else empty
   std::vector<double> m11, m12, m22;   // M_e per cell (DTV), else empty
   bool dtv() const { return !m11.empty(); }
};

// The weights M_e = I − γ ξξᵀ of directional TV from guide image v.
inline void dtv_weights(Instance& in, const std::vector<double>& guide, const OpConfig& cfg) {
   const int N = in.N, nc = N - 1;
   std::vector<double> v = guide;
   if (cfg.dtv_smooth > 0.0) Blur(N, cfg.dtv_smooth, (int)std::ceil(3.0 * cfg.dtv_smooth)).apply(guide, v, false);
   in.m11.assign((size_t)nc * nc, 1.0);
   in.m12.assign((size_t)nc * nc, 0.0);
   in.m22.assign((size_t)nc * nc, 1.0);
   for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b) {
         const double gx = v[a * N + b + 1] - v[a * N + b], gy = v[(a + 1) * N + b] - v[a * N + b];
         const double nrm = std::sqrt(gx * gx + gy * gy + cfg.dtv_eta * cfg.dtv_eta);
         const double xx = gx / nrm, xy = gy / nrm;
         const int e = a * nc + b;
         in.m11[e] = 1.0 - cfg.dtv_gamma * xx * xx;
         in.m12[e] = -cfg.dtv_gamma * xx * xy;
         in.m22[e] = 1.0 - cfg.dtv_gamma * xy * xy;
      }
}

// Decode, observe, and set up A, F (and M_e).  The noise is drawn exactly as
// image.hpp's load_image draws it, so a denoising instance equals tv_dd's.
// guide: "clean" or "data" (held-out pairs use "data": no ground truth there).
inline Instance make_instance(const std::string& path, int index, int N, double sigma,
                              unsigned seed, const OpConfig& cfg, const Blur* blur,
                              const std::string& guide) {
   Instance in;
   in.path = path;
   in.index = index;
   in.seed = seed;
   in.N = N;
   in.clean = load_image(path, N, 0.0, seed).clean;
   const size_t n = in.clean.size();
   std::vector<double> cu = in.clean;
   if (cfg.deblur()) blur->apply(in.clean, cu, false);
   in.f.resize(n);
   std::mt19937 rng(seed);
   std::normal_distribution<double> gauss(0.0, 1.0);
   for (size_t i = 0; i < n; ++i) in.f[i] = std::min(1.0, std::max(0.0, cu[i] + sigma * gauss(rng)));
   if (cfg.inpaint()) {
      std::mt19937 mrng(cfg.mask_seed + seed);
      std::uniform_real_distribution<double> uni(0.0, 1.0);
      in.mask.resize(n);
      double sum = 0.0;
      long cnt = 0;
      for (size_t i = 0; i < n; ++i) {
         in.mask[i] = uni(mrng) >= cfg.mask_frac ? 1.0 : 0.0;
         if (in.mask[i] > 0.0) { sum += in.f[i]; ++cnt; } else in.f[i] = 0.0;
      }
      if (cnt == 0) throw std::runtime_error("inpainting: no observed pixel");
      const double fbar = sum / cnt, eps = cfg.mask_eps;
      in.diag.resize(n);
      in.rhs.resize(n);
      for (size_t i = 0; i < n; ++i) {
         const double m = in.mask[i];
         in.diag[i] = m + eps * (1.0 - m);
         in.rhs[i] = m * in.f[i] + eps * (1.0 - m) * fbar;
      }
   } else if (cfg.deblur()) {
      blur->apply(in.f, in.rhs, true);
   } else {
      in.diag.assign(n, 1.0);
      in.rhs = in.f;
   }
   if (cfg.dtv) {
      // a data guide of an inpainting instance is filled first (the observed mean
      // on the holes, F/A), or the directions would follow the mask
      std::vector<double> data = in.f;
      if (cfg.inpaint())
         for (size_t i = 0; i < n; ++i)
            if (in.mask[i] == 0.0) data[i] = in.rhs[i] / in.diag[i];
      dtv_weights(in, guide == "clean" ? in.clean : data, cfg);
   }
   return in;
}

// (K_ξ u)_e at cell e = (a, b): M_e (u[a,b+1] − u[a,b], u[a+1,b] − u[a,b]).
inline void apply_K(const Instance& in, const std::vector<double>& u, std::vector<double>& gx,
                    std::vector<double>& gy) {
   const int N = in.N, nc = N - 1;
   gx.resize((size_t)nc * nc);
   gy.resize((size_t)nc * nc);
   for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b) {
         const int e = a * nc + b;
         const double kx = u[a * N + b + 1] - u[a * N + b], ky = u[(a + 1) * N + b] - u[a * N + b];
         if (in.dtv()) {
            gx[e] = in.m11[e] * kx + in.m12[e] * ky;
            gy[e] = in.m12[e] * kx + in.m22[e] * ky;
         } else {
            gx[e] = kx;
            gy[e] = ky;
         }
      }
}

// out = K_ξᵀ (px, py) on the nodes
inline void apply_KT(const Instance& in, const std::vector<double>& px, const std::vector<double>& py,
                     std::vector<double>& out) {
   const int N = in.N, nc = N - 1;
   out.assign((size_t)N * N, 0.0);
   for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b) {
         const int e = a * nc + b;
         double vx = px[e], vy = py[e];   // Mᵀp (M symmetric)
         if (in.dtv()) {
            vx = in.m11[e] * px[e] + in.m12[e] * py[e];
            vy = in.m12[e] * px[e] + in.m22[e] * py[e];
         }
         out[a * N + b] -= vx + vy;
         out[a * N + b + 1] += vx;
         out[(a + 1) * N + b] += vy;
      }
}

// out = A u
inline void apply_A(const Instance& in, const Blur* blur, const std::vector<double>& u,
                    std::vector<double>& out) {
   if (!in.diag.empty()) {
      out.resize(u.size());
      for (size_t i = 0; i < u.size(); ++i) out[i] = in.diag[i] * u[i];
   } else {
      std::vector<double> cu;
      blur->apply(u, cu, false);
      blur->apply(cu, out, true);
   }
}
