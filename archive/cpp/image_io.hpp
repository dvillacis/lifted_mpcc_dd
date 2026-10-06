// Shared data loader for the C++ TV solvers: feeds either the synthetic phantom
// text file (the original format) OR a real image file (PNG/JPEG/BMP/...) decoded
// in-process via the vendored stb_image. Used by all four TNLPs' load().
//
//   .txt  → parse "N \n N² clean \n N² noisy" (unchanged; phantoms/reproducibility).
//   image → stb_image decode → grayscale → center-crop to square → bilinear resize
//           to size×size → normalize [0,1] → u_clean; then add Gaussian noise
//           f = clip(u_clean + σ·g, 0, 1). C-order (i = row*N + col), N = size.
//
// Decisions baked in (see plan): square output (center-crop), clean+synthetic-noise.
// Caveats: these solvers are small-N tools — keep --size ≤ ~48–64. The C++ <random>
// noise does NOT match NumPy's, so an image run is not byte-reproducible against the
// Python phantom (irrelevant for real images).
#ifndef IMAGE_IO_HPP
#define IMAGE_IO_HPP

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace image_io {

struct Opts {
   int size = 0;            // target side length N for image input (ignored for .txt)
   double sigma = 0.1;      // Gaussian noise std for synthetic noisy copy
   unsigned seed = 0;       // noise RNG seed
   // Intensity scaling. `true` (the default, and the historical behaviour every
   // recorded number here was measured under) min–max stretches the crop to [0,1];
   // `false` divides by 255, which is what the Python side does
   // (lifted_mpcc_unitball_v2.load_image). Min–max is contrast-dependent, so it
   // changes the optimal weight — set it false when comparing against Python.
   bool minmax = true;
   // Downsampling filter. `false` (the default, historical) point-samples with
   // bilinear interpolation, which ALIASES badly at large reduction factors: at
   // 512→16 it keeps whatever 4 pixels it lands on and throws the rest away, so the
   // "clean" image carries spurious high-frequency detail and TV denoising has less
   // to remove. `true` averages over the whole source footprint, which is what
   // PIL's BILINEAR does when reducing (it scales the filter support), and is what
   // the Python side therefore produces. Measured on cameraman 512→16: 0.0457 /
   // +2.55 dB point-sampled vs 0.0744 / +4.68 dB area-averaged.
   bool area_downsample = false;
};

inline bool ends_with(const std::string& s, const std::string& suf) {
   return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// bilinear sample of a S×S buffer at (fy,fx) in input-pixel coordinates
inline double bilerp(const std::vector<double>& g, int S, double fy, double fx) {
   fy = std::min(std::max(fy, 0.0), (double)(S - 1));
   fx = std::min(std::max(fx, 0.0), (double)(S - 1));
   int y0 = (int)std::floor(fy), x0 = (int)std::floor(fx);
   int y1 = std::min(y0 + 1, S - 1), x1 = std::min(x0 + 1, S - 1);
   double wy = fy - y0, wx = fx - x0;
   double a = g[y0 * S + x0], b = g[y0 * S + x1], c = g[y1 * S + x0], d = g[y1 * S + x1];
   return (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d);
}

// Parse the original phantom text file: N, then N² clean, then N² noisy.
inline void load_txt(const std::string& fn, int& N,
                     std::vector<double>& uclean, std::vector<double>& f) {
   std::ifstream in(fn);
   if (!in) throw std::runtime_error("cannot open " + fn);
   in >> N;
   int mm = N * N;
   uclean.resize(mm); f.resize(mm);
   for (int i = 0; i < mm; ++i) in >> uclean[i];
   for (int i = 0; i < mm; ++i) in >> f[i];
}

// Decode a real image and build (u_clean, noisy f) at size×size.
inline void load_image(const std::string& fn, const Opts& opt, int& N,
                       std::vector<double>& uclean, std::vector<double>& f) {
   if (opt.size <= 0)
      throw std::runtime_error("image input requires --size N (target side length)");
   int w, h, ch;
   unsigned char* img = stbi_load(fn.c_str(), &w, &h, &ch, 1);   // force 1 channel (luma)
   if (!img) throw std::runtime_error("stb_image could not decode " + fn);

   // center-crop to square S = min(w,h), to float [0,255]
   int S = std::min(w, h);
   int roff = (h - S) / 2, coff = (w - S) / 2;
   std::vector<double> g((size_t)S * S);
   for (int r = 0; r < S; ++r)
      for (int c = 0; c < S; ++c)
         g[(size_t)r * S + c] = (double)img[(size_t)(r + roff) * w + (c + coff)];
   stbi_image_free(img);

   // bilinear resize S×S → N×N  (N = opt.size); pixel-center mapping
   N = opt.size;
   int m = N * N;
   uclean.resize(m);
   double scale = (double)S / N;
   if (opt.area_downsample && scale > 1.0) {
      // box average over each output pixel's source footprint (see Opts)
      for (int oy = 0; oy < N; ++oy)
         for (int ox = 0; ox < N; ++ox) {
            const int y0 = (int)std::floor(oy * scale), y1 = (int)std::ceil((oy + 1) * scale);
            const int x0 = (int)std::floor(ox * scale), x1 = (int)std::ceil((ox + 1) * scale);
            double acc = 0.0;
            int cnt = 0;
            for (int r = std::max(0, y0); r < std::min(S, y1); ++r)
               for (int c = std::max(0, x0); c < std::min(S, x1); ++c) {
                  acc += g[(size_t)r * S + c];
                  ++cnt;
               }
            uclean[(size_t)oy * N + ox] = cnt ? acc / cnt : 0.0;
         }
   } else {
      for (int oy = 0; oy < N; ++oy)
         for (int ox = 0; ox < N; ++ox) {
            double fy = (oy + 0.5) * scale - 0.5, fx = (ox + 0.5) * scale - 0.5;
            uclean[(size_t)oy * N + ox] = bilerp(g, S, fy, fx);
         }
   }

   // normalize to [0,1]: min–max stretch (default) or a plain /255 like Python
   if (opt.minmax) {
      double lo = uclean[0], hi = uclean[0];
      for (double v : uclean) { lo = std::min(lo, v); hi = std::max(hi, v); }
      double den = (hi > lo) ? (hi - lo) : 1.0;
      for (double& v : uclean) v = (hi > lo) ? (v - lo) / den : 0.0;
   } else {
      for (double& v : uclean) v = std::min(1.0, std::max(0.0, v / 255.0));
   }

   // synthetic noisy copy: f = clip(u_clean + σ·N(0,1), 0, 1)
   f.resize(m);
   std::mt19937 gen(opt.seed);
   std::normal_distribution<double> nd(0.0, 1.0);
   for (int i = 0; i < m; ++i)
      f[i] = std::min(1.0, std::max(0.0, uclean[i] + opt.sigma * nd(gen)));
}

// Branch on the path: ".txt" → phantom parse; else → image decode.
inline void load_data(const std::string& path, const Opts& opt, int& N,
                      std::vector<double>& uclean, std::vector<double>& f) {
   if (ends_with(path, ".txt")) load_txt(path, N, uclean, f);
   else                         load_image(path, opt, N, uclean, f);
}

} // namespace image_io
#endif
