// image.hpp — turn a picture file into a denoising instance.
//
//   decode → grayscale → centre-crop to a square → box-average down to N×N
//   → divide by 255                                          = the clean image
//   clean + σ·(Gaussian noise), clipped to [0, 1]            = the noisy image
//
// Pixels are stored row by row: pixel (row, col) is entry row·N + col.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

struct Image {
   int N = 0;
   std::vector<double> clean;   // the ground truth the weight is learned against
   std::vector<double> noisy;   // the data the lower level denoises
};

inline Image load_image(const std::string& path, int N, double sigma, unsigned seed) {
   int w, h, channels;
   unsigned char* px = stbi_load(path.c_str(), &w, &h, &channels, 1);  // 1 = grayscale
   if (!px) throw std::runtime_error("cannot decode image: " + path);

   // Centre-crop to an S×S square.
   const int S = std::min(w, h);
   const int row0 = (h - S) / 2, col0 = (w - S) / 2;
   std::vector<double> square((size_t)S * S);
   for (int r = 0; r < S; ++r)
      for (int c = 0; c < S; ++c)
         square[(size_t)r * S + c] = (double)px[(size_t)(r + row0) * w + (c + col0)];
   stbi_image_free(px);

   const double scale = (double)S / N;
   if (!(scale > 1.0))
      throw std::runtime_error("--size must be smaller than the image side");

   // Downsample by averaging every source pixel under each output pixel.
   Image img;
   img.N = N;
   img.clean.resize((size_t)N * N);
   for (int oy = 0; oy < N; ++oy)
      for (int ox = 0; ox < N; ++ox) {
         const int y0 = (int)std::floor(oy * scale), y1 = (int)std::ceil((oy + 1) * scale);
         const int x0 = (int)std::floor(ox * scale), x1 = (int)std::ceil((ox + 1) * scale);
         double sum = 0.0;
         int count = 0;
         for (int r = std::max(0, y0); r < std::min(S, y1); ++r)
            for (int c = std::max(0, x0); c < std::min(S, x1); ++c) {
               sum += square[(size_t)r * S + c];
               ++count;
            }
         img.clean[(size_t)oy * N + ox] = count ? sum / count : 0.0;
      }
   for (double& v : img.clean) v = std::min(1.0, std::max(0.0, v / 255.0));

   // Add Gaussian noise.
   img.noisy.resize(img.clean.size());
   std::mt19937 rng(seed);
   std::normal_distribution<double> gauss(0.0, 1.0);
   for (size_t i = 0; i < img.clean.size(); ++i)
      img.noisy[i] = std::min(1.0, std::max(0.0, img.clean[i] + sigma * gauss(rng)));
   return img;
}
