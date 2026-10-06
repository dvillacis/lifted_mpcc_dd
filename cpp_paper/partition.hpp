// partition.hpp — cut the image into k×k rectangular tiles, the subdomains.
//
// The (N−1)×(N−1) grid of CELLS is split along k+1 evenly spaced lines in each
// direction; tile (a, c) gets id a·k + c.  Each NODE (pixel) then joins the tile
// of the cell it anchors under the one-sided difference stencil: node (i, j)
// belongs with cell (i, j) — the cell whose top-left corner it is — clamped to
// the grid.
#pragma once

#include <algorithm>
#include <vector>

struct Partition {
   int N, nc, k, n_tiles;
   std::vector<int> cell_tile;   // (N−1)² entries
   std::vector<int> node_tile;   // N² entries

   Partition(int N_, int k_) : N(N_), nc(N_ - 1), k(k_), n_tiles(k_ * k_) {
      std::vector<int> cut(k + 1, 0);
      const double step = (double)nc / (double)k;
      for (int j = 0; j < k; ++j) cut[j] = (int)(j * step);
      cut[k] = nc;

      cell_tile.assign(nc * nc, 0);
      for (int a = 0; a < k; ++a)
         for (int c = 0; c < k; ++c)
            for (int i = cut[a]; i < cut[a + 1]; ++i)
               for (int j = cut[c]; j < cut[c + 1]; ++j)
                  cell_tile[i * nc + j] = a * k + c;

      node_tile.assign(N * N, 0);
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            const int ci = std::min(i, nc - 1);
            const int cj = std::min(j, nc - 1);
            node_tile[i * N + j] = cell_tile[ci * nc + cj];
         }
   }
};
