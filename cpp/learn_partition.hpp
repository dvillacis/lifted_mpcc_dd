// learn_partition.hpp — the tiles and the α blocks of tv_learn.
//
// Every image of the training set has the same N×N grid and the same k×k
// tiles; tile (a, c) of image s gets id s·k² + a·k + c (image-major), so that
// dd::Comm::first_tile hands each rank whole images where it can.
//
// The weight α is block-constant on the CELLS: block (I, J) holds the cells
// (a, b) with a / B = I, b / B = J, and is shared by all images.  B = 0 means
// one block (a scalar α).  With B ≥ 2 the tile cuts are moved to the nearest
// multiple of B, so that no block is cut by a tile boundary: a block is then
// read by the tiles of its cells and, through the nodes on its bottom and right
// edges, by at most three more.  B ≤ 1 leaves the cuts of partition.hpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct LearnPartition {
   int N, nc, k, S, B;     // image side, cells per side, tiles per side, images, block side
   int tiles_per_image, n_tiles;
   int nb, n_blocks;       // blocks per side, #blocks (1 for a scalar α)
   std::vector<int> cut;   // k + 1 cell-line cuts
   std::vector<int> band;  // cell line → its band, nc entries

   LearnPartition(int N_, int k_, int S_, int B_)
       : N(N_), nc(N_ - 1), k(k_), S(S_), B(B_), tiles_per_image(k_ * k_), n_tiles(S_ * k_ * k_) {
      cut.assign(k + 1, 0);
      const double step = (double)nc / (double)k;
      for (int j = 0; j < k; ++j) cut[j] = (int)(j * step);   // as partition.hpp
      cut[k] = nc;
      if (B >= 2) {
         for (int j = 1; j < k; ++j) cut[j] = (int)std::lround((double)cut[j] / B) * B;
         for (int j = 1; j <= k; ++j)
            if (cut[j] <= cut[j - 1])
               throw std::runtime_error("--block: too few blocks for the tiles (an empty tile)");
      }
      band.assign(nc, 0);
      for (int a = 0; a < k; ++a)
         for (int i = cut[a]; i < cut[a + 1]; ++i) band[i] = a;
      nb = B >= 1 ? (nc + B - 1) / B : 1;
      n_blocks = nb * nb;
   }

   // the narrowest band, in cells
   int min_band() const {
      int w = nc;
      for (int a = 0; a < k; ++a) w = std::min(w, cut[a + 1] - cut[a]);
      return w;
   }

   // Tiles of image s: cell (i, j) / cell e = i·nc + j, node (i, j) / node v = i·N + j.
   int cell_tile(int s, int i, int j) const { return s * tiles_per_image + band[i] * k + band[j]; }
   int cell_tile(int s, int e) const { return cell_tile(s, e / nc, e % nc); }
   int node_tile(int s, int i, int j) const {
      return cell_tile(s, std::min(i, nc - 1), std::min(j, nc - 1));
   }
   int node_tile(int s, int v) const { return node_tile(s, v / N, v % N); }

   // the block of cell (i, j) / of cell e
   int block(int i, int j) const { return B >= 1 ? (i / B) * nb + j / B : 0; }
   int block(int e) const { return block(e / nc, e % nc); }

   // The tiles whose h1 rows read α of block bl, ascending: those of the nodes
   // (a, b), (a, b+1), (a+1, b) of every cell (a, b) of the block (for K and
   // for K_ξ alike).  For rows a0..a1−1 and columns b0..b1−1 of cells these
   // are the nodes of rows a0..a1−1 with columns b0..b1, and of row a1 with
   // columns b0..b1−1.  A scalar α is read by every tile.
   std::vector<int> alpha_tiles(int bl) const {
      std::vector<int> t;
      if (B < 1) {
         for (int q = 0; q < n_tiles; ++q) t.push_back(q);
         return t;
      }
      const int a0 = (bl / nb) * B, b0 = (bl % nb) * B;
      const int a1 = std::min(a0 + B, nc), b1 = std::min(b0 + B, nc);
      auto bands = [&](int lo, int hi, std::vector<int>& out) {   // node lines lo..hi
         out.clear();
         for (int a = band[std::min(lo, nc - 1)]; a <= band[std::min(hi, nc - 1)]; ++a)
            out.push_back(a);
      };
      std::vector<int> r1, c1, r2, c2;
      bands(a0, a1 - 1, r1);
      bands(b0, b1, c1);
      bands(a1, a1, r2);
      bands(b0, b1 - 1, c2);
      std::vector<int> local;
      for (int a : r1)
         for (int c : c1) local.push_back(a * k + c);
      for (int a : r2)
         for (int c : c2) local.push_back(a * k + c);
      std::sort(local.begin(), local.end());
      local.erase(std::unique(local.begin(), local.end()), local.end());
      for (int s = 0; s < S; ++s)
         for (int q : local) t.push_back(s * tiles_per_image + q);
      return t;
   }
};
