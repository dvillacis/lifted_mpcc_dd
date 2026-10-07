// partition.hpp — cut the image into k×k rectangular tiles, the subdomains.
//
// The (N−1)×(N−1) grid of CELLS is split along k+1 evenly spaced lines in each
// direction; tile (a, c) gets id a·k + c.  Each NODE (pixel) then joins the tile
// of the cell it anchors under the one-sided difference stencil: node (i, j)
// belongs with cell (i, j) — the cell whose top-left corner it is — clamped to
// the grid.  Only the band of each grid line is stored (O(N)); the tile of a
// cell or node is computed from it.
#pragma once

#include <algorithm>
#include <vector>

struct Partition {
   int N, nc, k, n_tiles;
   std::vector<int> band;   // grid line (cell row or column) → its band, nc entries

   Partition(int N_, int k_) : N(N_), nc(N_ - 1), k(k_), n_tiles(k_ * k_) {
      std::vector<int> cut(k + 1, 0);
      const double step = (double)nc / (double)k;
      for (int j = 0; j < k; ++j) cut[j] = (int)(j * step);
      cut[k] = nc;
      band.assign(nc, 0);
      for (int a = 0; a < k; ++a)
         for (int i = cut[a]; i < cut[a + 1]; ++i) band[i] = a;
   }
   // the tile of cell (i, j) / of cell e = i·nc + j
   int cell_tile(int i, int j) const { return band[i] * k + band[j]; }
   int cell_tile(int e) const { return cell_tile(e / nc, e % nc); }
   // the tile of node (i, j) / of node v = i·N + j
   int node_tile(int i, int j) const { return cell_tile(std::min(i, nc - 1), std::min(j, nc - 1)); }
   int node_tile(int v) const { return node_tile(v / N, v % N); }
};
