// comm.hpp — the little MPI the decomposition needs, or a single-rank stand-in
// when built without MPI (then every call is trivial and nothing changes).
//
// Model: the TILES are distributed.  Rank r owns the contiguous tile range
// [first(r), first(r+1)) and holds only their variables, rows and KKT
// unknowns, plus the consensus variables, which every rank holds and updates
// identically.  Ranks exchange per-tile data with all-gathers (and, for the
// local Schur complements, an all-to-all between neighbours) — never with MPI
// sums of floating-point values.  Every global sum is formed by TileSum: one
// partial per tile, added in tile order.  So every rank computes bit-identical
// global values, and the run is bit-identical for any number of ranks (no rank
// can drift from the others and deadlock on a different iteration count).
#pragma once

#include <algorithm>
#include <vector>

#ifdef DD_HAVE_MPI
#include <mpi.h>
#endif

namespace dd {

struct Comm {
   static int rank() {
#ifdef DD_HAVE_MPI
      int r = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
      return r;
#else
      return 0;
#endif
   }
   static int size() {
#ifdef DD_HAVE_MPI
      int s = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &s);
      return s;
#else
      return 1;
#endif
   }
   static bool root() { return rank() == 0; }

   // First tile of rank r (r = size() gives K): contiguous blocks of tiles.
   static int first_tile(int r, int K) { return (int)((long)K * r / size()); }
   // The rank that owns tile k.
   static int owner_of(int k, int K) {
      int r = (int)((long)k * size() / K);
      while (first_tile(r + 1, K) <= k) ++r;
      while (first_tile(r, K) > k) --r;
      return r;
   }

   // All-gather of variable-length blocks: rank r contributes counts[r] values
   // starting at all[displs[r]] (already in place); afterwards every rank has
   // the whole of `all`.
   template <class T>
   static void allgatherv(std::vector<T>& all, const std::vector<int>& counts,
                          const std::vector<int>& displs) {
#ifdef DD_HAVE_MPI
      if (size() > 1)
         MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all.data(), counts.data(),
                        displs.data(), type<T>(), MPI_COMM_WORLD);
#else
      (void)all, (void)counts, (void)displs;
#endif
   }
   // Every rank sends send[sdispls[r] .. + scounts[r]) to rank r and receives
   // rcounts[r] values from it into recv[rdispls[r] ..).
   static void alltoallv(const std::vector<double>& send, const std::vector<int>& scounts,
                         const std::vector<int>& sdispls, std::vector<double>& recv,
                         const std::vector<int>& rcounts, const std::vector<int>& rdispls) {
#ifdef DD_HAVE_MPI
      if (size() > 1)
         MPI_Alltoallv(send.data(), scounts.data(), sdispls.data(), MPI_DOUBLE, recv.data(),
                       rcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#else
      (void)send, (void)scounts, (void)sdispls, (void)recv, (void)rcounts, (void)rdispls;
#endif
   }

   // Largest / smallest value over the ranks (exact, so order does not matter).
   static double max(double v) { reduce(&v, 1, true); return v; }
   static double min(double v) { reduce(&v, 1, false); return v; }
   static void max(double* v, int n) { reduce(v, n, true); }
   static void min(double* v, int n) { reduce(v, n, false); }
   static long sum(long v) {
#ifdef DD_HAVE_MPI
      long s = v;
      if (size() > 1) MPI_Allreduce(&v, &s, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      return s;
#else
      return v;
#endif
   }
   static bool any(bool b) { return sum(b ? 1 : 0) > 0; }

   // Gather values with global indices on rank 0: there, the result has n
   // entries, entry idx[i] = val[i] (entries held by several ranks — the
   // consensus variables — are the same on all of them).  Empty elsewhere.
   static std::vector<double> gather(const std::vector<long>& idx, const double* val, long n) {
      const long cnt = (long)idx.size();
#ifdef DD_HAVE_MPI
      const int nr = size();
      if (nr > 1) {
         std::vector<int> counts(nr), displs(nr, 0);
         int c = (int)cnt;
         MPI_Gather(&c, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
         for (int r = 1; r < nr; ++r) displs[r] = displs[r - 1] + counts[r - 1];
         const int total = root() ? displs[nr - 1] + counts[nr - 1] : 0;
         std::vector<long> all_idx(std::max(total, 1));
         std::vector<double> all_val(std::max(total, 1));
         MPI_Gatherv(idx.data(), c, MPI_LONG, all_idx.data(), counts.data(), displs.data(),
                     MPI_LONG, 0, MPI_COMM_WORLD);
         MPI_Gatherv(val, c, MPI_DOUBLE, all_val.data(), counts.data(), displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
         if (!root()) return {};
         std::vector<double> out(n, 0.0);
         for (int i = 0; i < total; ++i) out[all_idx[i]] = all_val[i];
         return out;
      }
#endif
      std::vector<double> out(n, 0.0);
      for (long i = 0; i < cnt; ++i) out[idx[i]] = val[i];
      return out;
   }

private:
#ifdef DD_HAVE_MPI
   template <class T> static MPI_Datatype type();
#endif
   static void reduce(double* v, int n, bool is_max) {
#ifdef DD_HAVE_MPI
      if (size() > 1)
         MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, is_max ? MPI_MAX : MPI_MIN, MPI_COMM_WORLD);
#else
      (void)v, (void)n, (void)is_max;
#endif
   }
};

#ifdef DD_HAVE_MPI
template <> inline MPI_Datatype Comm::type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype Comm::type<long>() { return MPI_LONG; }
template <> inline MPI_Datatype Comm::type<int>() { return MPI_INT; }
#endif

// =============================================================================
//  TileSum — global sums that come out the same for any number of ranks.
//  Every term goes to the partial sum of its tile; tile −1 is the consensus
//  part, which every rank computes identically.  total() all-gathers the
//  partials of the owned tiles, adds them in tile order and the consensus
//  partial last.  Terms of one tile must be added in an order that does not
//  depend on the ranks (e.g. the local order, which follows the global one).
//  `width` sums are formed at once (one all-gather).
// =============================================================================
class TileSum {
public:
   explicit TileSum(int n_tiles, int width = 1)
       : K_(n_tiles), w_(width), part_((size_t)(K_ + 1) * w_, 0.0) {}
   void clear() { std::fill(part_.begin(), part_.end(), 0.0); }
   // tile −1: the consensus part
   void add(int tile, double v, int j = 0) { part_[(size_t)(tile < 0 ? K_ : tile) * w_ + j] += v; }
   // Collective: out[j] = Σ_k partial_k[j] (tile order) + consensus[j]; clears.
   void total(double* out) {
      const int nr = Comm::size();
      if (nr > 1) {
         std::vector<int> counts(nr), displs(nr);
         for (int r = 0; r < nr; ++r) {
            displs[r] = Comm::first_tile(r, K_) * w_;
            counts[r] = Comm::first_tile(r + 1, K_) * w_ - displs[r];
         }
         Comm::allgatherv(part_, counts, displs);   // the consensus slot stays local
      }
      for (int j = 0; j < w_; ++j) {
         double s = 0.0;
         for (int k = 0; k <= K_; ++k) s += part_[(size_t)k * w_ + j];
         out[j] = s;
      }
      clear();
   }
   double total() {
      double s;
      total(&s);
      return s;
   }

private:
   int K_, w_;
   std::vector<double> part_;
};

}  // namespace dd
