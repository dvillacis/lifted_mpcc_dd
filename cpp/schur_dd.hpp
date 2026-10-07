// schur_dd.hpp — the Newton systems of the IPM by Schur-complement domain
// decomposition (Lueg, Bynum, Laird & Biegler, Optim. Eng. 27 (2026) 555–585).
//
// Block factorizations are in blocks.hpp, the preconditioner and PCG in
// precond.hpp.  Nothing here knows about the problem: only the owner map of the
// KKT unknowns and where the primals end.
//
// ============================================================================
//  1. THE ARROWHEAD (paper eq. 5)
// ============================================================================
// Group the KKT unknowns by tile, the consensus variables y ("border") last:
//
//        ⎡ W_1            B_1ᵀ ⎤ ⎡u_1⎤   ⎡r_1⎤
//        ⎢      ⋱          ⋮   ⎥ ⎢ ⋮ ⎥ = ⎢ ⋮ ⎥
//        ⎢          W_K   B_Kᵀ ⎥ ⎢u_K⎥   ⎢r_K⎥
//        ⎣ B_1  ⋯   B_K    C   ⎦ ⎣u_y⎦   ⎣r_y⎦
//
// W_k: the tile's own KKT block (eq. 6).  B_k: the −1 that each linking row
// "copy − y = 0" puts on y, i.e. the paper's R̄_k = N_kᵀR_k (N_k selects the p_k
// border unknowns tile k touches).  C: the y–y block (the objective's curvature
// on y; the IPM adds no regularization there).
//
// ============================================================================
//  2. SCHUR COMPLEMENT (eqs. 10, 11, 14, 15)
// ============================================================================
//      S u_y = r_y − Σ_k N_k B_k W_k⁻¹ r_k                          (10)
//      S     = C + Σ_k N_k S_k N_kᵀ,   S_k = −B_k W_k⁻¹ B_kᵀ         (14), (15)
//      u_k   = W_k⁻¹ (r_k − B_kᵀ N_kᵀ u_y)                          (11)
//
// Only the small dense S_k of a rank's own tiles are stored.  S itself is
// assembled (sparse) only for the direct solve.
//
// ============================================================================
//  3. SOLVING FOR u_y, AND THE INERTIA
// ============================================================================
// In(A) = Σ_k In(W_k) + In(S) (Haynsworth); In(W_k) comes from the block
// factorizations (eq. 24).  For S:
//  · pcg (default): CG on S, applied tile by tile through the S_k (15), with
//    the additive Schwarz preconditioner on the local assembled S̃_k (19), or
//    (precond = ASd) the diagonally assembled one (20, 21): tile k's own S_k
//    with its diagonal replaced by that of S.  S is taken as positive definite
//    unless CG meets pᵀSp ≤ 0 (the paper's Sec. 3.5 check), or, with AS, some
//    S̃_k is not: S̃_k = N_kᵀ S N_k is a principal submatrix of S, so its
//    Cholesky factorization failing proves S is not positive definite, and
//    the solve is refused before any tile solve or CG iteration.  In both
//    cases solve() returns false.  (In all runs measured, every S̃_k that was
//    not SPD was followed by a CG breakdown anyway.)  The preconditioner is
//    built at the first solve after a factorization, so that a factorization
//    refused for In(W_k) costs no S̃_k exchange and no Cholesky.
//  · direct: factorize the assembled S; its inertia is exact (eq. 25).
//
// The IPM refuses every factorization with a wrong In(W_k) (eq. 24).  When it
// says so (factorize(true)), every rank stops factorizing its tiles as soon as
// any rank meets one with a wrong inertia, or a singular one: the rest would be
// thrown away (StopSignal, comm.hpp).
//
// ============================================================================
//  4. PARALLELISM (MPI over tiles, OpenMP within a rank)
// ============================================================================
// The tiles are split into contiguous ranges, one per MPI rank (comm.hpp), and
// a rank HOLDS ONLY ITS OWN TILES: the KKT system it is given (set_structure)
// is its local one, the unknowns of its tiles plus the border, which every
// rank holds and solves for identically.  A rank factorizes its own W_k,
// forms its S_k and does its tile solves.  Per-tile data is exchanged:
//  · the border contributions of every tile, in each solve and each PCG
//    product, by all-gathers (share()); the border sums are then formed on
//    every rank in the serial (tile) order (users_), so the results are
//    bit-identical to a single-rank run;
//  · for the Schwarz blocks S̃_k, from every tile j that shares border unknowns
//    with an owned tile k, the part of S_j on those shared unknowns
//    (exchange_S(): neighbours send an edge block, distant tiles only the
//    entry of α, which every tile shares).  In direct mode every rank needs
//    the whole of S, and every S_k is all-gathered instead.
//
// Within a rank the per-tile loops run under OpenMP.  Tiles never write to the
// same border entry: each writes into its own buffer, and a parallel loop over
// the border gathers through users_ (border index → its (tile, position) pairs).
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "blocks.hpp"
#include "comm.hpp"
#include "precond.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dd {

// MPI calls inside OpenMP loops are made by the main thread only (FUNNELED).
inline bool main_thread() {
#ifdef _OPENMP
   return omp_get_thread_num() == 0;
#else
   return true;
#endif
}

class SchurDD {
public:
   enum class Interface { Direct, Pcg };
   enum class Precond { AS, ASd };
   struct Options {
      Interface interface = Interface::Pcg;
      Precond precond = Precond::AS;   // PCG: S̃_k (19) or S_k with diag(S) (21)
      Backend blocks = Backend::Sparse;     // W_k: sparse / dense / mumps (blocks.hpp)
      Backend fallback = Backend::Dense;    // where a failed sparse factorization goes
      double cg_tol = 1e-10;
      int cg_maxit = 2000;
      int dense_s_max = 2000;      // direct mode: S up to this size by dense Bunch–Kaufman
      int verbose = 0;             // 1: one line per factorization, 2: + per PCG solve
   };
   struct Stats {
      int n_tiles = 0, dim = 0, p = 0, max_pk = 0, max_tk = 0;
      long factorizations = 0, singular = 0, s_not_pd = 0, fallbacks = 0;
      long solves = 0, cg_iters = 0, cg_max = 0, cg_breakdowns = 0, cg_unconverged = 0;
      long s_tilde_indefinite = 0;   // solves refused: some S̃_k not SPD, so S not PD (AS)
      double t_factor = 0.0, t_solve = 0.0;                    // wall time (s)
      double t_blocks = 0.0, t_sdirect = 0.0, t_precond = 0.0;
      double t_tile_solves = 0.0, t_interface = 0.0;
   };

   // ---------------------------------------------------------------- structure
   // The LOCAL KKT system of this rank.  owner[i]: tile of KKT unknown i (one
   // of this rank's tiles), −1 = border; the border unknowns must be the same,
   // in the same order, on every rank.  Unknowns i < n_x are primal.
   // Collective.
   bool set_structure(int dim, const std::vector<int>& irow, const std::vector<int>& jcol,
                      const std::vector<int>& owner, int n_x, int n_tiles, const Options& opt) {
      if ((int)owner.size() != dim) {
         std::fprintf(stderr, "[dd] owner map has %zu entries, KKT has %d\n", owner.size(), dim);
         return false;
      }
      opt_ = opt;
      dim_ = dim;
      hash_ = pattern_hash(dim, irow, jcol, owner, n_x, n_tiles);
      values_.assign(irow.size(), 0.0);
      tiles_ = std::vector<Tile>(n_tiles);   // Tile holds a factorization: not copyable
      c_.clear();
      k0_ = Comm::first_tile(Comm::rank(), n_tiles);
      k1_ = Comm::first_tile(Comm::rank() + 1, n_tiles);

      // local numbering: tile unknowns in ascending KKT order, border likewise
      std::vector<int> loc(dim, -1);
      border_.clear();
      bool foreign = false;
      for (int i = 0; i < dim; ++i) {
         if (owner[i] < 0) {
            loc[i] = (int)border_.size();
            border_.push_back(i);
         } else if (owner[i] < k0_ || owner[i] >= k1_) {
            foreign = true;
         } else {
            Tile& T = tiles_[owner[i]];
            loc[i] = (int)T.glob.size();
            T.glob.push_back(i);
            T.primal.push_back(i < n_x);
         }
      }
      const int p = (int)border_.size();
      if (Comm::any(foreign) || Comm::max((double)p) != Comm::min((double)p)) {
         if (Comm::root())
            std::fprintf(stderr, "[dd] a rank holds another rank's tile, or the border differs\n");
         return false;
      }

      // N_k: the border unknowns tile k touches
      bool coupled = false;
      for (size_t t = 0; t < irow.size(); ++t) {
         const int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0 && owner[r] != owner[c]) {
            std::fprintf(stderr, "[dd] KKT entry (%d,%d) couples tiles %d and %d\n", r, c,
                         owner[r], owner[c]);
            coupled = true;
            break;
         }
         if ((owner[r] < 0) != (owner[c] < 0))
            tiles_[std::max(owner[r], owner[c])].nk.push_back(loc[owner[r] < 0 ? r : c]);
      }
      if (Comm::any(coupled)) return false;
      for (int k = k0_; k < k1_; ++k) {
         Tile& T = tiles_[k];
         std::sort(T.nk.begin(), T.nk.end());
         T.nk.erase(std::unique(T.nk.begin(), T.nk.end()), T.nk.end());
      }

      // route every triplet to W_k, B_k or C
      for (size_t t = 0; t < irow.size(); ++t) {
         const int r = irow[t], c = jcol[t];
         if (owner[r] >= 0 && owner[c] >= 0) {
            tiles_[owner[r]].w.push_back({loc[r], loc[c], (int)t});
         } else if (owner[r] < 0 && owner[c] < 0) {
            c_.push_back({loc[r], loc[c], (int)t});
         } else {
            Tile& T = tiles_[std::max(owner[r], owner[c])];
            const int b = owner[r] < 0 ? r : c, l = owner[r] < 0 ? c : r;
            const int a = (int)(std::lower_bound(T.nk.begin(), T.nk.end(), loc[b]) - T.nk.begin());
            T.b.push_back({a, loc[l], (int)t});
         }
      }

      // every rank learns every tile's N_k (border-sized), its size and its
      // number of duals (for the inertia test)
      lenP_.assign(n_tiles, 0);
      lenS_.assign(n_tiles, 0);
      duals_.assign(n_tiles, 0);
      std::vector<int> tk(n_tiles, 0);
      for (int k = k0_; k < k1_; ++k) {
         lenP_[k] = (int)tiles_[k].nk.size();
         duals_[k] = (int)std::count(tiles_[k].primal.begin(), tiles_[k].primal.end(), 0);
         tk[k] = (int)tiles_[k].glob.size();
      }
      share_int(lenP_);
      share_int(duals_);
      share_int(tk);
      {
         std::vector<int> off(n_tiles + 1, 0);
         for (int k = 0; k < n_tiles; ++k) off[k + 1] = off[k] + lenP_[k];
         std::vector<int> all(std::max(off[n_tiles], 1));
         for (int k = k0_; k < k1_; ++k) std::copy(tiles_[k].nk.begin(), tiles_[k].nk.end(), all.begin() + off[k]);
         share_blocks(all, off);
         for (int k = 0; k < n_tiles; ++k)
            if (k < k0_ || k >= k1_) tiles_[k].nk.assign(all.begin() + off[k], all.begin() + off[k + 1]);
      }
      for (int k = 0; k < n_tiles; ++k) lenS_[k] = lenP_[k] * lenP_[k];

      // users_: border index → (tile, position in N_k)
      ustart_.assign(p + 1, 0);
      for (const Tile& T : tiles_)
         for (int i : T.nk) ++ustart_[i + 1];
      for (int i = 0; i < p; ++i) ustart_[i + 1] += ustart_[i];
      users_.assign(ustart_[p], {0, 0});
      {
         std::vector<int> fill(ustart_.begin(), ustart_.end() - 1);
         for (int k = 0; k < n_tiles; ++k)
            for (int a = 0; a < (int)tiles_[k].nk.size(); ++a)
               users_[fill[tiles_[k].nk[a]]++] = {k, a};
      }
      tile_neg_.assign(n_tiles, 0);

      // flat_: every tile's border output, tile k at foff_[k] (p_k entries);
      // users_[q] is at uoff_[q].  One all-gather spreads it.
      foff_.assign(n_tiles + 1, 0);
      for (int k = 0; k < n_tiles; ++k) foff_[k + 1] = foff_[k] + lenP_[k];
      flat_.assign(std::max(foff_[n_tiles], 1), 0.0);
      uoff_.resize(users_.size());
      for (size_t q = 0; q < users_.size(); ++q) uoff_[q] = foff_[users_[q][0]] + users_[q][1];
      fcounts_.resize(Comm::size());
      fdispl_.resize(Comm::size());
      for (int r = 0; r < Comm::size(); ++r) {
         fdispl_[r] = foff_[Comm::first_tile(r, n_tiles)];
         fcounts_[r] = foff_[Comm::first_tile(r + 1, n_tiles)] - fdispl_[r];
      }
      if (opt_.interface == Interface::Pcg) plan_exchange_S();

      // symbolic work on this rank's tiles
#pragma omp parallel for schedule(dynamic)
      for (int k = k0_; k < k1_; ++k) {
         Tile& T = tiles_[k];
         T.blk.set_pattern(opt_.blocks, opt_.fallback, (int)T.glob.size(), T.w, T.primal, T.b,
                           (int)T.nk.size(), &T.S);
      }

      st_->n_tiles = n_tiles;
      st_->dim = (int)(Comm::sum((long)(dim - p)) + p);
      st_->p = p;
      st_->max_pk = st_->max_tk = 0;
      for (int k = 0; k < n_tiles; ++k) {
         st_->max_pk = std::max(st_->max_pk, lenP_[k]);
         st_->max_tk = std::max(st_->max_tk, tk[k]);
      }
      return true;
   }

   // Was set_structure() called with this structure (on every rank)?  Then
   // the decomposition can be reused for it.  Collective.
   bool same_structure(int dim, const std::vector<int>& irow, const std::vector<int>& jcol,
                       const std::vector<int>& owner, int n_x, int n_tiles) const {
      return !Comm::any(pattern_hash(dim, irow, jcol, owner, n_x, n_tiles) != hash_);
   }

   double* values() { return values_.data(); }
   int dim() const { return dim_; }
   const Stats& stats() const { return *st_; }
   void set_stats_sink(Stats* sink) { st_ = sink ? sink : &own_stats_; }

   // Per-tile inertia, for the IPM's regularization.  After a successful
   // factorize(): In(W_k) is right iff tile_negative(k) == tile_duals(k)
   // (eq. 24); s_negative() is #neg(S) in direct mode, 0 in PCG mode.
   int n_tiles() const { return (int)tiles_.size(); }
   int tile_negative(int k) const { return tile_neg_[k]; }
   int tile_duals(int k) const { return duals_[k]; }
   int s_negative() const { return negS_; }
   // After factorize(true) returned true: the factorization stopped at a tile
   // with a wrong In(W_k), and is to be refused (tile_negative() is then not
   // meaningful).
   bool inertia_stopped() const { return inertia_stopped_; }

   // ---------------------------------------------------------------- factorize
   // false = singular (some W_k or, in direct mode, S has a zero pivot).
   // stop_on_inertia: the caller refuses any factorization with a wrong
   // In(W_k) (#neg(W_k) ≠ #duals(W_k), eq. 24).  Every rank then stops as soon
   // as some rank meets a tile that is singular or has a wrong inertia, and a
   // true return with inertia_stopped() reports either.
   bool factorize(bool stop_on_inertia = false) {
      const auto t0 = Clock::now();
      ++st_->factorizations;
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      const bool pcg_mode = opt_.interface == Interface::Pcg;
      inertia_stopped_ = false;
      precond_ready_ = false;

      // ---- this rank's tiles: W_k → In(W_k);  S_k = −B_k W_k⁻¹ B_kᵀ        (14)
      // per tile: status (0 singular, 1 factorized, 2 wrong In(W_k), 3 skipped),
      // #neg(W_k), fallbacks
      std::vector<long> info(3 * K, 0);
      std::atomic<bool> stop(false), found(false);   // stop: any rank; found: this one
      // The first bad tile on ANY rank stops every rank (StopSignal): the
      // main thread passes a bad tile of this rank on and looks for the other
      // ranks'.  Only ranks that found one raise (drain() counts on that).
      auto poll = [&] {
         if (!stop_on_inertia || !main_thread()) return;
         if (found.load(std::memory_order_relaxed)) stop_signal_.raise();
         else if (!stop.load(std::memory_order_relaxed) && stop_signal_.seen()) stop = true;
      };
#pragma omp parallel for schedule(dynamic)
      for (int k = k0_; k < k1_; ++k) {
         Tile& T = tiles_[k];
         info[3 * k + 2] = T.blk.fallbacks();
         poll();
         if (stop.load(std::memory_order_relaxed)) {
            info[3 * k] = 3;
            continue;
         }
         const bool good = T.blk.factorize(values_.data());
         info[3 * k] = good;
         info[3 * k + 1] = good ? T.blk.negative() : 0;
         info[3 * k + 2] = T.blk.fallbacks();
         if (stop_on_inertia && (!good || T.blk.negative() != duals_[k])) {
            if (good) info[3 * k] = 2;
            found = true;
            stop = true;
            poll();
            continue;
         }
         if (!T.blk.needs_w() && !T.w.empty()) std::vector<Entry>().swap(T.w);   // analysed
         if (!good || T.blk.schur_in_place()) continue;   // MUMPS: S_k is in T.S already
         std::vector<Eigen::Triplet<double>> tb;
         tb.reserve(T.b.size());
         for (const Entry& e : T.b) tb.emplace_back(e[0], e[1], values_[e[2]]);
         T.B.resize((int)T.nk.size(), (int)T.glob.size());
         T.B.setFromTriplets(tb.begin(), tb.end());
         const Mat Sk = T.blk.schur(T.B);
         T.S = Sk;   // a copy: a move would swap out the storage a MUMPS fallback writes into
      }
      // a bad tile found by a worker thread after the main thread's last poll
      // is still passed on, so that drain() can count on every bad rank
      bool local_bad = false;
      for (int k = k0_; k < k1_; ++k) local_bad = local_bad || info[3 * k] == 0 || info[3 * k] == 2;
      if (stop_on_inertia && local_bad) stop_signal_.raise();
      share_long(info, 3);
      bool all_ok = true, wrong = false;
      for (int k = 0; k < K; ++k) {
         all_ok = all_ok && info[3 * k] == 1;
         wrong = wrong || info[3 * k] == 2;
      }
      if (stop_on_inertia) {   // receive the other bad ranks' signals
         int others = 0;
         for (int r = 0; r < Comm::size(); ++r) {
            if (r == Comm::rank()) continue;
            bool bad = false;
            for (int k = Comm::first_tile(r, K); k < Comm::first_tile(r + 1, K); ++k)
               bad = bad || info[3 * k] == 0 || info[3 * k] == 2;
            others += bad;
         }
         stop_signal_.drain(others);
      }
      // direct: every rank assembles S from every S_k (PCG: the parts the
      // S̃_k need are exchanged when the preconditioner is built)
      if (all_ok && !pcg_mode)
         share(lenS_, [&](int k, double* d) { std::copy(tiles_[k].S.data(), tiles_[k].S.data() + lenS_[k], d); },
               [&](int k, const double* d) { tiles_[k].S = Eigen::Map<const Mat>(d, lenP_[k], lenP_[k]); });
      st_->t_blocks += secs(t0);
      {
         long fb = 0;
         for (int k = 0; k < K; ++k) fb += info[3 * k + 2];
         st_->fallbacks += fb - fallbacks_seen_;
         fallbacks_seen_ = fb;
      }
      // With stop_on_inertia, which bad tile is met first depends on timing,
      // so a singular W_k is reported like a wrong In(W_k): refused (for the
      // caller both mean "raise δ").
      if (stop_on_inertia && !all_ok) wrong = true;
      int negW = 0;
      negS_ = 0;
      for (int k = 0; k < K && !wrong; ++k) {
         if (!info[3 * k]) {
            ++st_->singular;
            if (opt_.verbose && Comm::root())
               std::printf("[dd] fact %ld: W_%d singular\n", st_->factorizations, k);
            st_->t_factor += secs(t0);
            return false;
         }
         tile_neg_[k] = (int)info[3 * k + 1];
         negW += tile_neg_[k];
      }
      if (wrong) {   // stopped at a wrong In(W_k) or a singular W_k: the caller refuses it
         inertia_stopped_ = true;
         if (opt_.verbose && Comm::root())
            std::printf("[dd] fact %3ld  stopped: wrong In(W_k) or singular W_k\n", st_->factorizations);
         st_->t_factor += secs(t0);
         return true;
      }

      // ---- C
      cdiag_.setZero(p);
      coff_.clear();
      for (const Entry& e : c_) {
         if (e[0] == e[1]) cdiag_[e[0]] += values_[e[2]];
         else coff_.push_back({(double)e[0], (double)e[1], values_[e[2]]});
      }

      if (p > 0 && !pcg_mode) {
         // ---- direct: assemble S (15) and factorize it; exact In(S)        (25)
         const auto t1 = Clock::now();
         const bool good = Sf_.factorize(assemble_S(), opt_.dense_s_max);
         st_->t_sdirect += secs(t1);
         if (!good) {
            ++st_->singular;
            st_->t_factor += secs(t0);
            if (opt_.verbose && Comm::root())
               std::printf("[dd] fact %ld: S singular\n", st_->factorizations);
            return false;
         }
         negS_ = Sf_.negative();
         if (negS_ > 0) ++st_->s_not_pd;
      }
      // PCG: the preconditioner is built by the first solve (build_precond())

      if (opt_.verbose && Comm::root())
         std::printf("[dd] fact %3ld  Σ#neg(W_k)=%d%s\n", st_->factorizations, negW,
                     pcg_mode ? "" : ("  #neg(S)=" + std::to_string(negS_)).c_str());
      st_->t_factor += secs(t0);
      return true;
   }

   // ---------------------------------------------------------------- solve
   // In place.  Returns false if S is seen not to be positive definite: PCG
   // met negative curvature, or (AS) some S̃_k is not SPD.
   bool solve(double* rhs) {
      const int p = (int)border_.size();
      const bool pcg_mode = opt_.interface == Interface::Pcg;
      if (p > 0 && pcg_mode && !precond_ready_) build_precond();
      const auto t0 = Clock::now();
      if (p > 0 && pcg_mode && s_tilde_indefinite_) {
         // a principal submatrix of S is not positive definite, so S is not
         // (eq. 25): the answer a CG breakdown would give, without the solve
         ++st_->s_tilde_indefinite;
         if (opt_.verbose >= 2 && Comm::root())
            std::printf("[dd]      some S~_k not SPD: S not positive definite\n");
         st_->t_solve += secs(t0);
         return false;
      }
      ++st_->solves;
      Eigen::Map<Vec> R(rhs, dim_);

      // z_k = W_k⁻¹ r_k,  then r_S = r_y − Σ N_k B_k z_k                    (10)
      // (MUMPS tiles: B_k z_k by condensation, forward sweep only)
#pragma omp parallel for schedule(dynamic)
      for (int k = k0_; k < k1_; ++k) {
         Tile& T = tiles_[k];
         T.z.resize(T.glob.size());
         for (size_t l = 0; l < T.glob.size(); ++l) T.z[l] = R[T.glob[l]];
         if (T.blk.can_reduce()) {
            T.blk.reduce(T.z, flat_.data() + foff_[k]);
            continue;
         }
         T.blk.solve(T.z);
         if (lenP_[k] > 0) flat(k).noalias() = T.B * T.z;
      }
      gather_flat();
      Vec rS(p);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p; ++i) {
         double s = R[border_[i]];
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) s -= flat_[uoff_[q]];
         rS[i] = s;
      }
      const auto t1 = Clock::now();
      st_->t_tile_solves += secs(t0);

      // S u_y = r_S
      Vec uy = Vec::Zero(p);
      bool good = true;
      if (p > 0 && opt_.interface == Interface::Direct) {
         uy = rS;
         Sf_.solve(uy);
      } else if (p > 0) {
         auto applyS = [this](const Vec& v, Vec& out) { apply_S(v, out); };
         auto applyM = [this](const Vec& r, Vec& z) { apply_M(r, z); };
         const CgResult cg = pcg(applyS, applyM, rS, uy, opt_.cg_tol, opt_.cg_maxit);
         st_->cg_iters += cg.iters;
         st_->cg_max = std::max<long>(st_->cg_max, cg.iters);
         if (cg.breakdown) { ++st_->cg_breakdowns; good = false; }
         else if (cg.rel > opt_.cg_tol) ++st_->cg_unconverged;
         if (opt_.verbose >= 2 && Comm::root())
            std::printf("[dd]      pcg its=%4d  rel.res=%.1e%s\n", cg.iters, cg.rel,
                        cg.breakdown ? "  NEGATIVE CURVATURE" : "");
      }
      const auto t2 = Clock::now();
      st_->t_interface += std::chrono::duration<double>(t2 - t1).count();

      // u_k = W_k⁻¹(r_k − B_kᵀ N_kᵀ u_y)                                   (11)
      // (MUMPS tiles: by expansion of the condensation above, backward sweep only)
      for (int i = 0; i < p; ++i) R[border_[i]] = uy[i];
#pragma omp parallel for schedule(dynamic)
      for (int k = k0_; k < k1_; ++k) {
         Tile& T = tiles_[k];
         Vec uyk(T.nk.size());
         for (size_t a = 0; a < T.nk.size(); ++a) uyk[a] = uy[T.nk[a]];
         Vec rk;
         if (T.blk.can_reduce()) {
            T.blk.expand(uyk, rk);
         } else {
            rk.resize(T.glob.size());
            for (size_t l = 0; l < T.glob.size(); ++l) rk[l] = R[T.glob[l]];
            if (uyk.size() > 0) rk -= T.B.transpose() * uyk;
            T.blk.solve(rk);
         }
         for (size_t l = 0; l < T.glob.size(); ++l) R[T.glob[l]] = rk[l];
      }
      st_->t_tile_solves += secs(t2);
      st_->t_solve += secs(t0);
      return good;
   }

   // The border part of a product with the coupling entries B_k, summed over
   // all tiles in tile order (the same on every rank):
   //     out[i] = Σ_k Σ_{(a,l,s) ∈ B_k, N_k(a) = i} val(s)·x(l)
   // val(s): the value of KKT entry s (return 0 to leave it out); x(j): the
   // tile-side unknown, by local KKT index.  Collective.
   template <class Val, class X>
   void border_product(Val val, X x, Vec& out) const {
      for (int k = k0_; k < k1_; ++k) {
         const Tile& T = tiles_[k];
         double* o = flat_.data() + foff_[k];
         std::fill(o, o + lenP_[k], 0.0);
         for (const Entry& e : T.b) o[e[0]] += val(e[2]) * x(T.glob[e[1]]);
      }
      gather_flat();
      const int p = (int)border_.size();
      out.resize(p);
      for (int i = 0; i < p; ++i) {
         double s = 0.0;
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) s += flat_[uoff_[q]];
         out[i] = s;
      }
   }
   // Local KKT index of border unknown i.
   int border_index(int i) const { return border_[i]; }
   int border_size() const { return (int)border_.size(); }

private:
   using Clock = std::chrono::steady_clock;
   static unsigned long long pattern_hash(int dim, const std::vector<int>& irow,
                                          const std::vector<int>& jcol,
                                          const std::vector<int>& owner, int n_x, int n_tiles) {
      unsigned long long h = 1469598103934665603ULL;   // FNV-1a
      auto mix = [&h](long long v) { h = (h ^ (unsigned long long)v) * 1099511628211ULL; };
      mix(dim), mix(n_x), mix(n_tiles), mix((long long)irow.size());
      for (int v : irow) mix(v);
      for (int v : jcol) mix(v);
      for (int v : owner) mix(v);
      return h;
   }
   static double secs(Clock::time_point t0) {
      return std::chrono::duration<double>(Clock::now() - t0).count();
   }

   // What tile j contributes to S̃_k: S_j on the border unknowns N_k ∩ N_j,
   // at positions a (in N_k) and b (in N_j), both ascending.
   struct Part {
      int j;
      std::vector<int> a, b;
      Mat blk;                   // received (j ≠ k); for j = k, S_k itself is used
   };
   struct Tile {
      std::vector<int> glob;     // KKT index of each tile unknown
      std::vector<char> primal;  // is it a primal variable?
      std::vector<int> nk;       // border indices touched: N_k
      std::vector<Entry> w, b;   // routed triplets: W_k (r, c, slot), B_k (a, l, slot)
      TileBlock blk;             // factorization of W_k
      SpMat B;                   // B_k (p_k × t_k)
      Mat S;                     // S_k (this rank's tiles; all of them in direct mode).
                                 // A MUMPS tile's factorization writes it in place:
                                 // never resize an own tile's S.
      std::vector<Part> parts;   // S̃_k: the tiles sharing border unknowns with k (PCG)
      Vec sdiag;                 // diag(S_k), all tiles (ASd)
      Vec z;                     // W_k⁻¹ r_k during a solve
   };

   // S = C + Σ_k N_k S_k N_kᵀ (15), sparse, both triangles
   SpMat assemble_S() const {
      const int p = (int)border_.size();
      size_t nnz = p + 2 * coff_.size();
      for (const Tile& T : tiles_) nnz += T.nk.size() * T.nk.size();
      std::vector<Eigen::Triplet<double>> t;
      t.reserve(nnz);
      for (int i = 0; i < p; ++i) t.emplace_back(i, i, cdiag_[i]);
      for (const auto& e : coff_) {
         t.emplace_back((int)e[0], (int)e[1], e[2]);
         t.emplace_back((int)e[1], (int)e[0], e[2]);
      }
      for (const Tile& T : tiles_)
         for (size_t a = 0; a < T.nk.size(); ++a)
            for (size_t b = 0; b < T.nk.size(); ++b) t.emplace_back(T.nk[a], T.nk[b], T.S(a, b));
      SpMat S(p, p);
      S.setFromTriplets(t.begin(), t.end());
      return S;
   }

   // ---- PCG: the additive Schwarz blocks S̃_k = N_kᵀ S N_k                (18, 19)
   //      or, ASd, S_k with its diagonal replaced by diag(S)                 (20, 21)
   // Built by the first solve after a factorization, so that the factorizations
   // the IPM refuses for In(W_k) cost no exchange and no block factorizations
   // (counted in the factorization's time).  AS: whether some S̃_k is not SPD,
   // which proves S is not (its principal submatrix).  Collective.
   void build_precond() {
      const auto t0 = Clock::now();
      const int p = (int)border_.size();
      const int K = (int)tiles_.size();
      const bool as = opt_.precond == Precond::AS;
      if (as) exchange_S();
      M_.reset(K);
      if (!as) {
         // diag(S) = diag(C) + Σ_k N_k diag(S_k), in tile order: every rank
         // needs the diagonals of all S_k (one value per border unknown each)
         for (int k = k0_; k < k1_; ++k) tiles_[k].sdiag = tiles_[k].S.diagonal();
         share(lenP_, [&](int k, double* d) { std::copy(tiles_[k].sdiag.data(), tiles_[k].sdiag.data() + lenP_[k], d); },
               [&](int k, const double* d) { tiles_[k].sdiag = Eigen::Map<const Vec>(d, lenP_[k]); });
         diagS_.resize(p);
         for (int i = 0; i < p; ++i) {
            double s = cdiag_[i];
            for (int q = ustart_[i]; q < ustart_[i + 1]; ++q)
               s += tiles_[users_[q][0]].sdiag[users_[q][1]];
            diagS_[i] = s;
         }
      }
      int indefinite = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : indefinite)
      for (int k = k0_; k < k1_; ++k) {
         const Tile& T = tiles_[k];
         if (T.nk.empty()) continue;
         bool spd;
         if (as) {
            spd = M_.set(k, T.nk, local_assembled(k), false);
         } else {   // not a submatrix of S: clip to SPD, as CG needs
            Mat Mk = T.S;
            for (size_t a = 0; a < T.nk.size(); ++a) Mk(a, a) = diagS_[T.nk[a]];
            spd = M_.set(k, T.nk, Mk, true);
         }
         if (!spd) ++indefinite;
      }
      s_tilde_indefinite_ = as && Comm::any(indefinite > 0);
      precond_ready_ = true;
      const double dt = secs(t0);
      st_->t_precond += dt;
      st_->t_factor += dt;
   }

   // S̃_k = N_kᵀ S N_k, from the parts of the S_j of the tiles that share
   // border unknowns with tile k (paper: "local assembled Schur complement",
   // eq. 18).  Every entry adds its contributions in tile order.
   Mat local_assembled(int k) const {
      const Tile& T = tiles_[k];
      const int pk = (int)T.nk.size();
      Mat St = Mat::Zero(pk, pk);
      for (int a = 0; a < pk; ++a) St(a, a) += cdiag_[T.nk[a]];
      for (const Part& P : T.parts) {
         const Mat& M = P.j == k ? T.S : P.blk;
         const int c = (int)P.a.size();
         for (int y = 0; y < c; ++y)
            for (int x = 0; x < c; ++x)
               St(P.a[x], P.a[y]) += P.j == k ? M(P.b[x], P.b[y]) : M(x, y);
      }
      static thread_local std::vector<int> mark;
      if ((int)mark.size() < (int)border_.size()) mark.assign(border_.size(), -1);
      for (int a = 0; a < pk; ++a) mark[T.nk[a]] = a;
      for (const auto& e : coff_) {
         const int a = mark[(int)e[0]], b = mark[(int)e[1]];
         if (a >= 0 && b >= 0) { St(a, b) += e[2]; St(b, a) += e[2]; }
      }
      for (int a = 0; a < pk; ++a) mark[T.nk[a]] = -1;
      return 0.5 * (St + St.transpose());
   }

   // ---- the parts of S_j that the Schwarz blocks need (PCG mode) ----------
   // For every owned tile k: the tiles j sharing border unknowns with it, in
   // tile order, with the shared positions.  For every owned j: what goes to
   // which rank.  Messages to rank r hold, for its tiles k in order and then j
   // in order, S_j on N_k ∩ N_j (column-major).
   void plan_exchange_S() {
      const int K = (int)tiles_.size(), nr = Comm::size(), me = Comm::rank();
      std::vector<int> slot(K, -1);
      auto common = [&](int k, std::vector<Part>& out) {   // N_k ∩ N_j for all j, tile order
         out.clear();
         const Tile& T = tiles_[k];
         std::vector<int> js;
         for (int a = 0; a < (int)T.nk.size(); ++a)
            for (int q = ustart_[T.nk[a]]; q < ustart_[T.nk[a] + 1]; ++q) {
               const int j = users_[q][0];
               if (slot[j] < 0) { slot[j] = (int)js.size(); js.push_back(j); out.push_back({j, {}, {}, Mat()}); }
               out[slot[j]].a.push_back(a);
               out[slot[j]].b.push_back(users_[q][1]);
            }
         for (int j : js) slot[j] = -1;
         std::sort(out.begin(), out.end(), [](const Part& x, const Part& y) { return x.j < y.j; });
      };
      for (int k = k0_; k < k1_; ++k) common(k, tiles_[k].parts);
      // sends: for every tile k of another rank, the owned j it shares unknowns with
      sends_.assign(nr, {});
      std::vector<Part> tmp;
      for (int r = 0; r < nr; ++r) {
         if (r == me) continue;
         for (int k = Comm::first_tile(r, K); k < Comm::first_tile(r + 1, K); ++k) {
            common(k, tmp);
            for (Part& P : tmp)
               if (P.j >= k0_ && P.j < k1_) sends_[r].push_back({P.j, {}, std::move(P.b), Mat()});
         }
      }
   }
   void exchange_S() {
      const int K = (int)tiles_.size(), nr = Comm::size();
      std::vector<int> scount(nr, 0), sdispl(nr, 0), rcount(nr, 0), rdispl(nr, 0);
      for (int r = 0; r < nr; ++r)
         for (const Part& P : sends_[r]) scount[r] += (int)(P.b.size() * P.b.size());
      for (int k = k0_; k < k1_; ++k)
         for (const Part& P : tiles_[k].parts)
            if (P.j < k0_ || P.j >= k1_) rcount[Comm::owner_of(P.j, K)] += (int)(P.a.size() * P.a.size());
      for (int r = 1; r < nr; ++r) {
         sdispl[r] = sdispl[r - 1] + scount[r - 1];
         rdispl[r] = rdispl[r - 1] + rcount[r - 1];
      }
      std::vector<double> sbuf(std::max(sdispl[nr - 1] + scount[nr - 1], 1));
      std::vector<double> rbuf(std::max(rdispl[nr - 1] + rcount[nr - 1], 1));
      for (int r = 0; r < nr; ++r) {
         int q = sdispl[r];
         for (const Part& P : sends_[r]) {
            const Mat& S = tiles_[P.j].S;
            for (int y : P.b)
               for (int x : P.b) sbuf[q++] = S(x, y);
         }
      }
      Comm::alltoallv(sbuf, scount, sdispl, rbuf, rcount, rdispl);
      std::vector<int> cur(rdispl);
      for (int k = k0_; k < k1_; ++k)
         for (Part& P : tiles_[k].parts) {
            if (P.j == k) continue;
            const int c = (int)P.a.size();
            P.blk.resize(c, c);
            if (P.j >= k0_ && P.j < k1_) {   // both tiles here
               const Mat& S = tiles_[P.j].S;
               for (int y = 0; y < c; ++y)
                  for (int x = 0; x < c; ++x) P.blk(x, y) = S(P.b[x], P.b[y]);
            } else {
               int& q = cur[Comm::owner_of(P.j, K)];
               for (int y = 0; y < c; ++y)
                  for (int x = 0; x < c; ++x) P.blk(x, y) = rbuf[q++];
            }
         }
   }

   // out = S v, tile by tile through the stored S_k (15)
   void apply_S(const Vec& v, Vec& out) const {
      const int p = (int)border_.size();
#pragma omp parallel for schedule(dynamic, 4)
      for (int k = k0_; k < k1_; ++k) {
         const Tile& T = tiles_[k];
         const int pk = (int)T.nk.size();
         if (pk == 0) continue;
         static thread_local Vec vk;
         vk.resize(pk);
         for (int a = 0; a < pk; ++a) vk[a] = v[T.nk[a]];
         flat(k).noalias() = T.S * vk;
      }
      gather_flat();
      out.resize(p);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p; ++i) {
         double s = cdiag_[i] * v[i];
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) s += flat_[uoff_[q]];
         out[i] = s;
      }
      for (const auto& e : coff_) {
         out[(int)e[0]] += e[2] * v[(int)e[1]];
         out[(int)e[1]] += e[2] * v[(int)e[0]];
      }
   }

   // z = M⁻¹ r: this rank's Schwarz blocks, then the sum over the border in
   // the serial order (z_i = Σ over the tiles touching i of their output).
   void apply_M(const Vec& r, Vec& z) {
      M_.apply_blocks(r, k0_, k1_, flat_.data(), foff_.data());
      gather_flat();
      const int p = (int)border_.size();
      z.resize(p);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < p; ++i) {
         double s = 0.0;
         for (int q = ustart_[i]; q < ustart_[i + 1]; ++q) s += flat_[uoff_[q]];
         z[i] = s;
      }
   }

   // ---- MPI exchange of per-tile data (no-ops on one rank) ----------------
   // Every rank ends with every tile's block of len[k] doubles: the owners
   // pack theirs, an all-gather spreads them, the other ranks unpack.
   template <class Pack, class Unpack>
   void share(const std::vector<int>& len, Pack pack, Unpack unpack) const {
      const int nr = Comm::size();
      if (nr == 1) return;
      const int K = (int)tiles_.size();
      std::vector<int> off(K + 1, 0);
      for (int k = 0; k < K; ++k) off[k + 1] = off[k] + len[k];
      std::vector<double> all(std::max(off[K], 1));
      for (int k = k0_; k < k1_; ++k) pack(k, all.data() + off[k]);
      std::vector<int> counts(nr), displs(nr);
      for (int r = 0; r < nr; ++r) {
         displs[r] = off[Comm::first_tile(r, K)];
         counts[r] = off[Comm::first_tile(r + 1, K)] - displs[r];
      }
      Comm::allgatherv(all, counts, displs);
      for (int k = 0; k < K; ++k)
         if (k < k0_ || k >= k1_) unpack(k, all.data() + off[k]);
   }
   // The per-tile border outputs: tile k writes flat(k) (p_k entries); after
   // gather_flat() every rank has every tile's (no allocation, no copies).
   Eigen::Map<Vec> flat(int k) const { return Eigen::Map<Vec>(flat_.data() + foff_[k], lenP_[k]); }
   void gather_flat() const { Comm::allgatherv(flat_, fcounts_, fdispl_); }
   // m integers per tile (filled by the owners).
   void share_long(std::vector<long>& v, int m) const {
      const int nr = Comm::size();
      if (nr == 1) return;
      const int K = (int)tiles_.size();
      std::vector<int> counts(nr), displs(nr);
      for (int r = 0; r < nr; ++r) {
         displs[r] = m * Comm::first_tile(r, K);
         counts[r] = m * Comm::first_tile(r + 1, K) - displs[r];
      }
      Comm::allgatherv(v, counts, displs);
   }

   // One int per tile (filled by the owners).
   void share_int(std::vector<int>& v) const {
      std::vector<int> off(v.size() + 1);
      for (size_t k = 0; k <= v.size(); ++k) off[k] = (int)k;
      share_blocks(v, off);
   }
   // Tile k's block is all[off[k] .. off[k+1]) (filled by the owner).
   void share_blocks(std::vector<int>& all, const std::vector<int>& off) const {
      const int nr = Comm::size();
      if (nr == 1) return;
      const int K = (int)tiles_.size();
      std::vector<int> counts(nr), displs(nr);
      for (int r = 0; r < nr; ++r) {
         displs[r] = off[Comm::first_tile(r, K)];
         counts[r] = off[Comm::first_tile(r + 1, K)] - displs[r];
      }
      Comm::allgatherv(all, counts, displs);
   }

   Options opt_;
   unsigned long long hash_ = 0;   // of the structure (same_structure())
   int dim_ = 0, negS_ = 0;
   bool inertia_stopped_ = false;      // last factorize(true) stopped at a bad W_k
   StopSignal stop_signal_;            // factorize(true): the first bad tile stops every rank
   bool precond_ready_ = false;        // the preconditioner is that of the last factorization
   bool s_tilde_indefinite_ = false;   // ... and some S̃_k of it is not SPD (AS)
   int k0_ = 0, k1_ = 0;                        // this rank's tiles
   std::vector<int> lenP_, lenS_;               // per tile: p_k, p_k²
   std::vector<int> tile_neg_, duals_;          // #neg(W_k), #duals of W_k: every tile
   std::vector<std::vector<Part>> sends_;       // per rank: the parts of owned S_j it needs
   long fallbacks_seen_ = 0;
   std::vector<int> border_, ustart_;
   std::vector<int> foff_, uoff_, fcounts_, fdispl_;   // flat_ layout (set_structure)
   mutable std::vector<double> flat_;                   // per-tile border outputs
   std::vector<std::array<int, 2>> users_;
   std::vector<double> values_;
   std::vector<Tile> tiles_;
   std::vector<Entry> c_;
   std::vector<std::array<double, 3>> coff_;   // off-diagonal C entries
   Vec cdiag_, diagS_;
   SymFactor Sf_;
   Schwarz M_;
   Stats own_stats_;
   Stats* st_ = &own_stats_;
};

}  // namespace dd
