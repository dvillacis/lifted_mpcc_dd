// learn_util.hpp — small helpers of tv_learn: PSNR, peak memory, and a
// finite-difference check of an NLP's derivatives (--check-derivatives).
#pragma once

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "nlp.hpp"

inline double psnr(const std::vector<double>& clean, const double* u) {
   double mse = 0.0;
   for (size_t i = 0; i < clean.size(); ++i) mse += (clean[i] - u[i]) * (clean[i] - u[i]);
   mse /= (double)clean.size();
   return mse == 0.0 ? 1e9 : 10.0 * std::log10(1.0 / mse);
}

// SSIM (Wang, Bovik, Sheikh & Simoncelli 2004) of u against clean, both N×N in [0, 1]:
// 11×11 Gaussian window (σ = 1.5), K1 = 0.01, K2 = 0.03, mean over the windows that
// fit inside the image (no padding).
inline double ssim(const std::vector<double>& clean, const double* u, int N) {
   const int R = 5;
   double w[2 * R + 1], ws = 0.0;
   for (int k = -R; k <= R; ++k) ws += w[k + R] = std::exp(-0.5 * k * k / (1.5 * 1.5));
   for (double& v : w) v /= ws;
   const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
   double sum = 0.0;
   long cnt = 0;
   for (int i = R; i < N - R; ++i)
      for (int j = R; j < N - R; ++j) {
         double mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
         for (int a = -R; a <= R; ++a)
            for (int b = -R; b <= R; ++b) {
               const double ww = w[a + R] * w[b + R];
               const double x = clean[(i + a) * N + j + b], y = u[(i + a) * N + j + b];
               mx += ww * x;
               my += ww * y;
               sxx += ww * x * x;
               syy += ww * y * y;
               sxy += ww * x * y;
            }
         sxx -= mx * mx;
         syy -= my * my;
         sxy -= mx * my;
         sum += ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2));
         ++cnt;
      }
   return cnt ? sum / cnt : NAN;
}

inline double peak_rss_mb() {
   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
   return ru.ru_maxrss / 1048576.0;
#else
   return ru.ru_maxrss / 1024.0;
#endif
}

// Central differences (h = 1e-6) at the start point, perturbed, against the
// analytic ∇f, J and the Hessian of σf + λᵀg (random λ, σ = 1; the Hessian is
// tested on random directions, H·v against the difference of ∇L).  Single
// rank only (f() is collective, the others are local).  Returns the largest
// relative error max |fd − an| / max(1, |an|).
inline double check_derivatives(NLP& nlp, unsigned seed = 1, int n_cols = 300, int n_dirs = 5) {
   const int n = nlp.num_vars(), m = nlp.num_rows();
   std::mt19937 rng(seed);
   std::uniform_real_distribution<double> uni(-1.0, 1.0);
   std::vector<double> x(n);
   nlp.start(x.data());
   for (double& v : x) v += 1e-2 * uni(rng);
   const double h = 1e-6;

   std::vector<int> jr, jc, hr, hc;
   nlp.jac_structure(jr, jc);
   nlp.hess_structure(hr, hc);
   std::vector<double> jv(jr.size()), hv(hr.size()), gf(n), lam(m);
   nlp.jac_values(x.data(), jv.data());
   nlp.grad_f(x.data(), gf.data());
   for (double& l : lam) l = uni(rng);
   nlp.hess_values(x.data(), 1.0, lam.data(), hv.data());

   std::vector<std::vector<std::pair<int, double>>> col(n);   // J by column
   for (size_t k = 0; k < jr.size(); ++k) col[jc[k]].push_back({jr[k], jv[k]});

   double err_g = 0.0, err_j = 0.0, err_h = 0.0;
   std::vector<double> gp(m), gm(m), xp, xm;
   std::uniform_int_distribution<int> pick(0, n - 1);
   for (int t = 0; t < std::min(n, n_cols); ++t) {
      const int j = n <= n_cols ? t : pick(rng);
      xp = x;
      xm = x;
      xp[j] += h;
      xm[j] -= h;
      const double fd = (nlp.f(xp.data()) - nlp.f(xm.data())) / (2 * h);
      err_g = std::max(err_g, std::abs(fd - gf[j]) / std::max(1.0, std::abs(gf[j])));
      nlp.g(xp.data(), gp.data());
      nlp.g(xm.data(), gm.data());
      std::vector<double> an(m, 0.0);
      for (const auto& e : col[j]) an[e.first] += e.second;
      for (int r = 0; r < m; ++r) {
         const double d = (gp[r] - gm[r]) / (2 * h);
         err_j = std::max(err_j, std::abs(d - an[r]) / std::max(1.0, std::abs(an[r])));
      }
   }
   // ∇L = ∇f + Jᵀλ at a point
   auto gradL = [&](const std::vector<double>& p) {
      std::vector<double> g(n), v(jr.size());
      nlp.grad_f(p.data(), g.data());
      nlp.jac_values(p.data(), v.data());
      for (size_t k = 0; k < jr.size(); ++k) g[jc[k]] += v[k] * lam[jr[k]];
      return g;
   };
   for (int d = 0; d < n_dirs; ++d) {
      std::vector<double> v(n);
      for (double& a : v) a = uni(rng);
      std::vector<double> an(n, 0.0);
      for (size_t k = 0; k < hr.size(); ++k) {
         an[hr[k]] += hv[k] * v[hc[k]];
         if (hr[k] != hc[k]) an[hc[k]] += hv[k] * v[hr[k]];
      }
      xp = x;
      xm = x;
      for (int i = 0; i < n; ++i) { xp[i] += h * v[i]; xm[i] -= h * v[i]; }
      const std::vector<double> lp = gradL(xp), lm = gradL(xm);
      for (int i = 0; i < n; ++i) {
         const double fd = (lp[i] - lm[i]) / (2 * h);
         err_h = std::max(err_h, std::abs(fd - an[i]) / std::max(1.0, std::abs(an[i])));
      }
   }
   std::printf("derivative check (central differences, h = %.0e, %d columns, %d Hessian"
               " directions): grad f %.1e   J %.1e   H %.1e\n",
               h, std::min(n, n_cols), n_dirs, err_g, err_j, err_h);
   return std::max({err_g, err_j, err_h});
}
