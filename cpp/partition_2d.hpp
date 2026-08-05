// The 2D cell/node partition geometry, shared by dd_solve_2d.cpp (one image) and
// dd_solve_dataset.cpp (one image per training pair).
//
// Extracted VERBATIM from dd_solve_2d.cpp so the two drivers cannot drift: the
// linspace bounds, the tile/strip cell ownership and the anchor rule are the
// measured geometry, validated against ../lifted_mpcc_2d.py's own partition.
// Nothing formulation-specific lives here — kkt_owner stays in each driver,
// since it reads that driver's TNLP layout.
#ifndef PARTITION_2D_HPP
#define PARTITION_2D_HPP

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Partition of the (N−1)² cells. TILE = k×k (mirrors ../lifted_mpcc_2d.py). STRIP =
// k horizontal row-strips (cut ONE direction only): then no cell sits at a 4-way
// cross corner, so no cell has both qx AND qy on the border, so the cut-corner dual
// rank-deficiency never arises — the 2D generalization of the working 1D DD, at the
// price of a wider interface and only k subdomains.
// ---------------------------------------------------------------------------
struct Partition2D {
   int N, nc, k, n_sub;
   bool striped;
   std::vector<int> bounds, cell_owner, node_owner;

   Partition2D(int N_, int k_, bool striped_ = false)
       : N(N_), nc(N_ - 1), k(k_), n_sub(striped_ ? k_ : k_ * k_), striped(striped_) {
      // np.linspace(0, nc, k+1).astype(int) — truncated, last one exact.
      bounds.assign(k + 1, 0);
      const double step = (double)nc / (double)k;
      for (int j = 0; j < k; ++j) bounds[j] = (int)(j * step);
      bounds[k] = nc;

      cell_owner.assign(nc * nc, 0);
      if (striped) {
         // Only the ROW index chooses the strip; columns are never cut.
         for (int a = 0; a < k; ++a)
            for (int i = bounds[a]; i < bounds[a + 1]; ++i)
               for (int j = 0; j < nc; ++j) cell_owner[i * nc + j] = a;
      } else {
         for (int a = 0; a < k; ++a)
            for (int c = 0; c < k; ++c)
               for (int i = bounds[a]; i < bounds[a + 1]; ++i)
                  for (int j = bounds[c]; j < bounds[c + 1]; ++j)
                     cell_owner[i * nc + j] = a * k + c;
      }

      // anchor rule: node (i,j) → cell (i−1,j−1), clamped — the cell the node
      // anchors under the one-sided stencil
      node_owner.assign(N * N, 0);
      for (int i = 0; i < N; ++i)
         for (int j = 0; j < N; ++j) {
            const int ci = std::min(std::max(i - 1, 0), nc - 1);
            const int cj = std::min(std::max(j - 1, 0), nc - 1);
            node_owner[i * N + j] = cell_owner[ci * nc + cj];
         }
   }
};

#endif  // PARTITION_2D_HPP
