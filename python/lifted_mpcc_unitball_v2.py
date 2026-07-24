r"""Lifted TV-denoising MPCC, unit-ball dual — v2 with numerical-performance fixes.

Same formulation, variable/constraint layout, continuation driver, reporting rule
and certificate as ``lifted_mpcc_unitball.py`` (see its module docstring for the
full math; the two files are meant to be A/B-comparable level by level). This
file applies the following numerics/performance fixes:

  1. **Consistent objective/gradient/Hessian.** The ½·reg_alpha·α² ridge now
     appears in ``gradient()`` (it was commented out), in the exact Hessian's
     (α,α) entry, and in the certificate's R_α row. Previously the objective
     contained a term whose gradient IPOPT thought was zero (≈2e-4 at the σ=0.1
     optimum — two orders above the 1e-6 tol), polluting the line search
     precisely where small-t steps are tiny.
  2. **Real barrier warm-start between levels (opt-in — measured MIXED).**
     ``--dual-warmstart`` passes the previous multipliers, arms
     ``warm_start_init_point``, sets ``mu_init`` to the previous level's final
     barrier parameter (clamped to [1e-9, 1e-1]) and shrinks the warm-start
     push options to 1e-8, instead of restarting the barrier at μ = 0.1 every
     level. Measured across the instance matrix it is a wash-to-loss: −22%
     iterations on phantom σ=0.2 N=32 (396 vs 506), but 2.2× on cameraman
     σ=0.2 (2146 vs 968 — the frozen-small μ makes the final level grind when
     α still has to move) and 1.45× at N=48 (1532 vs 1056). It therefore stays
     OFF by default; the always-on primal warm-start plus the other fixes
     already beat v1.
  3. **MA57 MC64 scaling, size-based auto (``--ma57-scaling auto``, default:
     on iff m = N² ≥ 1600, i.e. N ≥ 40).** MC64 targets the badly-scaled KKT
     systems of the small-t tail where ‖ξ‖∞ ~ 1/t. Measured: at N=48 it turns
     the documented tail stall into FULL-schedule convergence to t=2.2e-4,
     α=−2.658/+6.92 dB ≈ the exact-TV optimum, under plain L-BFGS (v1:
     restoration failure at t=8.1e-3, ξ→10², reports the loose overfit
     α=−2.45/+8.16; unscaled v2: fails at t=7.3e-4, ξ→700). At N=32 it never
     pays: ≈2× per-factorization cost (33 vs 16 ms/iter), no iteration savings,
     and everything converges without it. NOTE a per-level trigger (arming MC64
     only for t ≤ 1e-2) was tried and does NOT work: the tail rescue is
     path-dependent — scaling must shape the trajectory from the first level,
     or the tail fails from the same incoming iterate (measured; don't re-try).
     ``on``/``off`` force it globally. ``--mu-strategy adaptive`` is likewise
     exposed for the loose levels.
  4. **Per-x memoized callback intermediates** (e^α, cos θ, sin θ, div q) shared
     by constraints/jacobian/hessian — removes the redundant sparse matvecs and
     trig evaluations (they were each computed up to three times per iterate).
     The memo depends on x only, so it stays valid when t changes.
  5. **Hybrid accelerated Chambolle–Pock warm start with a residual stop.**
     The ROF primal ½‖u−f‖² is 1-strongly convex, so a γ-accelerated phase
     (O(1/k²) in u and in the complementarity) runs first, then fixed steps
     polish the h1 fixed-point consistency (where acceleration's τ→0 lags at
     O(1/k)), stopping on the true h1 residual ‖u₊−u‖∞/τ ≤ tol — which is free,
     since the primal prox gives u₊−f+λ·div q = −(u₊−u)/τ exactly. Measured on
     the default instance: 350 iterations for h1 = 1.0e-7 / comp = 1.7e-5 vs
     v1's fixed 3000 iterations for h1 = 2.1e-7 / comp = 1.5e-4 (v1's ‖Δu‖-based
     test never fires). ``cp-scan`` grid points also warm-start from the
     previous grid solution, and the operator transposes are hoisted out.
  6. **One cyipopt.Problem reused across all levels.** The Scholtes t moved into
     the comp constraint *value* (row = r·(1−δ) − t with fixed upper bound 0, an
     additive constant — Jacobian/Hessian unchanged), so cl/cu never change and
     per-level options are updated in place on a single Problem object.
  7. **fd-check via directional derivatives** (sparse assembly — no dense
     m_con×n / n×n matrices, so it is fast and memory-light at any N), and it
     now also validates the OBJECTIVE gradient — the gap that let fix 1's
     inconsistency slip through v1's three verification layers. Sparse COO
     assembly *sums* duplicate (row, col) entries, closing v1's fancy-indexing
     ``J[rows, cols] += vals`` trap (which silently drops duplicates).

  Experimental (off by default): ``--xi-rescale`` scales the warm-started comp
  multipliers by t_prev/t before each level (ξ ~ 1/t along the Scholtes path).
  ``--t-update mu`` replaces the geometric outer loop with ONE IPOPT solve whose
  Scholtes t is slaved to the barrier parameter each iteration,
  t = max(t_min, c·μ) with c = ``--t-mu-scale`` (> 0), tightening only — an
  externally-applied interior relaxation à la Raghunathan–Biegler, possible here
  because t enters the comp row additively (Jacobian/Hessian are t-free). See
  :func:`solve_mu_coupled`.

Run:   uv run python lifted_mpcc_unitball_v2.py                 # cameraman, N=32
       uv run python lifted_mpcc_unitball_v2.py --fd-check      # validate derivatives
       uv run python lifted_mpcc_unitball_v2.py --N 48          # MC64 auto-on: full tail
       uv run python lifted_mpcc_unitball_v2.py --t-update mu   # ONE solve, t slaved to μ
       uv run python lifted_mpcc_unitball_v2.py --dual-warmstart --xi-rescale  # A/B levers
"""

from __future__ import annotations

import argparse
import os
import signal
import time

import cyipopt
import numpy as np
import scipy.sparse as sp

# HSL MA57 shared library. Overridable via the HSLLIB env var (set it to the
# cluster's Linux .so — e.g. libcoinhsl.so or libhsl_ma57.so — so you don't edit
# source on the HPC); defaults to this machine's dylib. If the resolved path
# exists we use ma57, otherwise we fall back to MUMPS (ships with cyipopt).
HSLLIB = os.environ.get(
    "HSLLIB", "/Users/davidvillacis/src/hsl/hsl_ma57-5.3.2/src/.libs/libhsl_ma57.dylib"
)

# Bundled natural test image (the classic 512×512 cameraman photograph, exported
# from skimage.data.camera()). Resolved relative to this file so the default run
# works from any working directory.
DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "cameraman.png")


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
def make_phantom(N: int, sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A piecewise-constant cartoon image (ideal for TV) plus a noisy copy.

    Returns flattened (C-order) ``(u_clean, f)`` normalized to roughly [0, 1].
    Kept for A/B comparisons against the synthetic case; the default run uses a
    natural image (see ``load_image``).
    """
    u = np.full((N, N), 0.2)
    a, b = N // 4, 3 * N // 4
    u[a:b, a:b] = 0.8
    c, d = N // 3, 2 * N // 3
    u[c:d, c:d] = 0.5
    rng = np.random.default_rng(seed)
    f = np.clip(u + sigma * rng.standard_normal((N, N)), 0.0, 1.0)
    return u.ravel(), f.ravel()


def load_image(path: str, N: int, sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load a natural image, center-crop to a square, resize to N×N, add noise.

    Grayscale → center-crop to the largest centered square → bilinear resize to
    ``N×N`` → normalize to [0, 1] → ``u_clean``; then ``f = clip(u + σ·N(0,1))``.
    C-order flatten (index = row·N + col) matches ``grad_operators``. The square
    crop is the only geometry assumption (the lift/operators are built for N×N).
    """
    from PIL import Image

    img = Image.open(path).convert("L")  # 8-bit grayscale
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((N, N), Image.BILINEAR)
    u = np.asarray(img, dtype=float) / 255.0
    rng = np.random.default_rng(seed)
    f = np.clip(u + sigma * rng.standard_normal((N, N)), 0.0, 1.0)
    return u.ravel(), f.ravel()


def grad_operators(N: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Forward-difference gradient with Neumann (no-flux) boundary.

    For C-order flattening (index = row·N + col): ``Kx`` differences along
    columns (x), ``Ky`` along rows (y). Both are (N²×N²) sparse.
    """
    e = np.ones(N)
    D = sp.diags([-e, e[:-1]], [0, 1], shape=(N, N), format="lil")
    D[N - 1, :] = 0.0  # Neumann: zero gradient at the far boundary
    D = D.tocsr()
    Id = sp.identity(N, format="csr")
    Kx = sp.kron(Id, D, format="csr")
    Ky = sp.kron(D, Id, format="csr")
    return Kx, Ky


# ---------------------------------------------------------------------------
# The lifted MPCC (unit-ball formulation) as a cyipopt problem object
# ---------------------------------------------------------------------------
class LiftedTVMPCC:
    """cyipopt problem object for the unit-ball lifted TV-MPCC (v2).

    Identical formulation to v1 with two mechanical changes:

    * The Scholtes level ``t`` lives on the object (``t_comp``) and enters the
      comp row as the additive constant ``r·(1−δ) − t ≤ 0`` (fixed upper bound
      0), so the driver can reuse one cyipopt.Problem across all levels. Being
      additive, it changes no derivative.
    * The x-dependent intermediates shared by constraints/jacobian/hessian
      (e^α, cos θ, sin θ, div q) are memoized per x via ``_common`` — they are
      t-independent, so the memo survives level changes.
    """

    def __init__(self, f, u_clean, N, reg_alpha=1e-4):
        self.f = np.asarray(f, float)
        self.u_clean = np.asarray(u_clean, float)
        self.N = N
        self.m = m = N * N
        self.reg_alpha = reg_alpha

        # Scholtes level: the comp row is r·(1−δ) − t_comp ≤ 0 with fixed upper
        # bound 0; the continuation driver writes t_comp per level (fix 6).
        self.t_comp = 0.0

        # TR (Tikhonov gauge) ridge ½·eps_theta·‖θ − θ_ref‖² — the D1 fix for the
        # angle-gauge indeterminacy (θ_i undetermined where r_i = δ_i = 0). Off by
        # default; the driver sets eps_theta = c_θ·t per level so the bias → 0.
        self.eps_theta = 0.0
        self.theta_ref = np.zeros(m)

        # Per-solve telemetry, (re)set by the driver / intermediate callback.
        self.n_iter = 0
        self.n_reg = 0   # iters with δ_w > 0 (Hessian inertia corr.; see intermediate)
        self.n_rest = 0  # iters in IPOPT's restoration phase (alg_mod == 1)
        self.inf_pr = float("nan")
        self.inf_du = float("nan")
        self.mu_last = float("nan")  # final barrier μ of the last solve (fix 2)
        self._interrupt = False  # set by the driver's SIGINT handler (see intermediate)

        # μ-coupled t update (--t-update mu): with t_mu_scale > 0 the intermediate
        # callback slaves the Scholtes level to the barrier parameter,
        # t = max(t_floor, t_mu_scale·μ), monotonically non-increasing. t_hist
        # records (iter, t) at each change for reporting.
        self.t_mu_scale = 0.0
        self.t_floor = 0.0
        self.c_theta_live = 0.0  # keeps the TR ridge eps_theta = c_θ·t in sync
        self.t_hist = []
        # μ-coupled progress printing: the single solve is otherwise silent under
        # print_level 0 (no per-level rows), so at N=48 a hard instance looks like
        # a hang. The driver sets these; intermediate prints a live row every
        # ``_mu_progress_every`` iters and warns if t stays pinned (μ not moving →
        # t not tightening, the near-singular-first-level failure mode).
        self._mu_progress_every = 0
        self._mu_stall_iters = 0
        self._mu_stall_warned = False

        self._cx = None  # memo key for _common (fix 4)

        self.Kx, self.Ky = grad_operators(N)
        self.KxT, self.KyT = self.Kx.T.tocsr(), self.Ky.T.tocsr()

        # Column offsets of each variable block, and the scalar α index. No slack:
        # δ carries the unit-ball radius directly and the comp row is written on δ.
        self.off = {
            "u": 0, "qx": m, "qy": 2 * m, "r": 3 * m,
            "delta": 4 * m, "theta": 5 * m, "alpha": 6 * m,
        }
        # Row offsets of each constraint block. The non-negativities r ≥ 0 and
        # δ ≥ 0 are explicit inequality rows (hr, hd) rather than variable box
        # bounds; only δ ≤ 1 and the α box remain as bounds. ``comp`` stays LAST
        # (the driver treats every row from ``comp`` on as the one-sided r·(1−δ)≤t).
        # System is 6N²+1 vars / 8N² rows.
        self.roff = {
            "h1": 0, "h2x": m, "h2y": 2 * m, "h3x": 3 * m,
            "h3y": 4 * m, "hr": 5 * m, "hd": 6 * m, "comp": 7 * m,
        }
        self.n = 6 * m + 1
        self.m_con = 8 * m
        self._rows, self._cols = self._build_structure()
        self._hrows, self._hcols = self._build_hess_structure()

    # ---- slicing helpers --------------------------------------------------
    def _blk(self, x, name):
        s = self.off[name]
        return x[s : s + self.m]

    def _divq(self, x):
        """The divergence Kxᵀ q_x + Kyᵀ q_y (length m) — appears in h1 and its
        α-derivatives.  ∂h1/∂α = e^α·div q, ∂²h1/∂α² carries the same vector."""
        return self.KxT @ self._blk(x, "qx") + self.KyT @ self._blk(x, "qy")

    def _common(self, x):
        """Memoized per-x intermediates ``(e^α, cos θ, sin θ, div q)`` shared by
        constraints/jacobian/hessian (fix 4). Keyed on x by value; t-independent,
        so level changes never invalidate it."""
        if self._cx is None or not np.array_equal(x, self._cx):
            self._cx = np.array(x, dtype=float, copy=True)
            theta = self._blk(self._cx, "theta")
            self._cvals = (
                float(np.exp(self._cx[self.off["alpha"]])),
                np.cos(theta),
                np.sin(theta),
                self._divq(self._cx),
            )
        return self._cvals

    # ---- objective --------------------------------------------------------
    def objective(self, x):
        u = self._blk(x, "u")
        a = x[self.off["alpha"]]
        obj = 0.5 * np.sum((u - self.u_clean) ** 2) + 0.5 * self.reg_alpha * a * a
        if self.eps_theta:  # TR gauge ridge ½·eps_theta·‖θ − θ_ref‖²
            d = self._blk(x, "theta") - self.theta_ref
            obj += 0.5 * self.eps_theta * np.dot(d, d)
        return obj

    def gradient(self, x):
        g = np.zeros(self.n)
        g[self.off["u"] : self.off["u"] + self.m] = self._blk(x, "u") - self.u_clean
        # Ridge gradient restored (fix 1) — matches the ½·reg_alpha·α² term in
        # ``objective`` and the obj_factor·reg_alpha (α,α) entry in ``hessian``.
        g[self.off["alpha"]] = self.reg_alpha * x[self.off["alpha"]]
        if self.eps_theta:  # ∂/∂θ of the TR ridge = eps_theta·(θ − θ_ref)
            s = self.off["theta"]
            g[s : s + self.m] = self.eps_theta * (self._blk(x, "theta") - self.theta_ref)
        return g

    # ---- constraints ------------------------------------------------------
    def constraints(self, x):
        ea, c, s, divq = self._common(x)
        u, qx, qy = self._blk(x, "u"), self._blk(x, "qx"), self._blk(x, "qy")
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        return np.concatenate(
            [
                u - self.f + ea * divq,        # h1  (← e^α)
                self.Kx @ u - r * c,           # h2x
                self.Ky @ u - r * s,           # h2y
                qx - delta * c,                # h3x
                qy - delta * s,                # h3y
                r,                             # hr : r ≥ 0      (explicit inequality)
                delta,                         # hd : δ ≥ 0      (explicit inequality)
                r * (1.0 - delta) - self.t_comp,  # comp: r·(1 − δ) − t ≤ 0 (Scholtes)
            ]
        )

    # ---- Jacobian ---------------------------------------------------------
    def _build_structure(self):
        """Precompute the (row, col) index arrays of every Jacobian nonzero.

        The 20 pieces below are assembled in the SAME order by ``jacobian``;
        constant blocks (Kx, Ky, identities) have their values cached here. Note
        piece 4 is the dense ∂h1/∂α column (the whole α-coupling of the KKT
        system); the comp row is written on δ (∂comp/∂r = 1 − δ, ∂comp/∂δ = −r);
        and the explicit non-negativity rows hr, hd are identity diagonals
        (∂hr/∂r = ∂hd/∂δ = 1, pieces 19–20). The −t in the comp row is additive
        and contributes nothing here.
        """
        m, off, ro = self.m, self.off, self.roff
        idx = np.arange(m)

        def diag(roff, coff):
            return roff + idx, coff + idx

        Kx, Ky = self.Kx.tocoo(), self.Ky.tocoo()
        KxT, KyT = self.KxT.tocoo(), self.KyT.tocoo()
        self._ones = np.ones(m)
        self._Kx, self._Ky = Kx.data, Ky.data
        self._KxT, self._KyT = KxT.data, KyT.data

        pieces = [
            diag(ro["h1"], off["u"]),                       # 1  h1/∂u   = I
            (ro["h1"] + KxT.row, off["qx"] + KxT.col),      # 2  h1/∂qx  = e^α·Kxᵀ
            (ro["h1"] + KyT.row, off["qy"] + KyT.col),      # 3  h1/∂qy  = e^α·Kyᵀ
            (ro["h1"] + idx, np.full(m, off["alpha"])),     # 4  h1/∂α   = e^α·div q
            (ro["h2x"] + Kx.row, off["u"] + Kx.col),        # 5  h2x/∂u  = Kx
            diag(ro["h2x"], off["r"]),                      # 6  h2x/∂r  = -cosθ
            diag(ro["h2x"], off["theta"]),                  # 7  h2x/∂θ  =  r sinθ
            (ro["h2y"] + Ky.row, off["u"] + Ky.col),        # 8  h2y/∂u  = Ky
            diag(ro["h2y"], off["r"]),                      # 9  h2y/∂r  = -sinθ
            diag(ro["h2y"], off["theta"]),                  # 10 h2y/∂θ  = -r cosθ
            diag(ro["h3x"], off["qx"]),                     # 11 h3x/∂qx = I
            diag(ro["h3x"], off["delta"]),                  # 12 h3x/∂δ  = -cosθ
            diag(ro["h3x"], off["theta"]),                  # 13 h3x/∂θ  =  δ sinθ
            diag(ro["h3y"], off["qy"]),                     # 14 h3y/∂qy = I
            diag(ro["h3y"], off["delta"]),                  # 15 h3y/∂δ  = -sinθ
            diag(ro["h3y"], off["theta"]),                  # 16 h3y/∂θ  = -δ cosθ
            diag(ro["comp"], off["r"]),                     # 17 comp/∂r = 1 − δ
            diag(ro["comp"], off["delta"]),                 # 18 comp/∂δ = -r
            diag(ro["hr"], off["r"]),                       # 19 hr/∂r  = 1
            diag(ro["hd"], off["delta"]),                   # 20 hd/∂δ  = 1
        ]
        rows = np.concatenate([p[0] for p in pieces]).astype(np.int64)
        cols = np.concatenate([p[1] for p in pieces]).astype(np.int64)
        return rows, cols

    def jacobianstructure(self):
        return self._rows, self._cols

    def jacobian(self, x):
        ea, c, s, divq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        return np.concatenate(
            [
                self._ones,        # 1  h1/∂u
                ea * self._KxT,    # 2  h1/∂qx  = e^α·Kxᵀ
                ea * self._KyT,    # 3  h1/∂qy  = e^α·Kyᵀ
                ea * divq,         # 4  h1/∂α   = e^α·div q
                self._Kx,          # 5  h2x/∂u
                -c,                # 6  h2x/∂r
                r * s,             # 7  h2x/∂θ
                self._Ky,          # 8  h2y/∂u
                -s,                # 9  h2y/∂r
                -r * c,            # 10 h2y/∂θ
                self._ones,        # 11 h3x/∂qx
                -c,                # 12 h3x/∂δ
                delta * s,         # 13 h3x/∂θ
                self._ones,        # 14 h3y/∂qy
                -s,                # 15 h3y/∂δ
                -delta * c,        # 16 h3y/∂θ
                1.0 - delta,       # 17 comp/∂r = 1 − δ
                -r,                # 18 comp/∂δ = -r
                self._ones,        # 19 hr/∂r  = 1
                self._ones,        # 20 hd/∂δ  = 1
            ]
        )

    # ---- exact Lagrangian Hessian (lower triangle) ------------------------
    # H = σ_f·∇²J + Σ_k λ_k ∇²c_k. Objective → I on u, reg_alpha on (α,α)
    # (+ eps_theta·I on θ under TR). Nonlinear constraints and their per-pixel
    # Hessians:
    #   h1 : bilinear in (α, q).  ∂²/∂qx∂α = e^α·Kxᵀ·(·) , ∂²/∂qy∂α = e^α·Kyᵀ·(·),
    #        ∂²/∂α² = e^α·⟨λ_h1, div q⟩.  (α is the trailing scalar, so these land
    #        in the α row/column of the lower triangle.)
    #   h2x: ∂²/∂r∂θ= sinθ, ∂²/∂θ²= r cosθ ;  h2y: ∂²/∂r∂θ=−cosθ, ∂²/∂θ²= r sinθ
    #   h3x: ∂²/∂δ∂θ= sinθ, ∂²/∂θ²= δ cosθ ;  h3y: ∂²/∂δ∂θ=−cosθ, ∂²/∂θ²= δ sinθ
    #   comp: r·(1 − δ), ∂²/∂r∂δ = −1 → (δ,r) cross = −ξ (indefinite, eigenvalues
    #        ±ξ; lower triangle since off[δ]=4m > off[r]=3m).
    def _build_hess_structure(self):
        m, off = self.m, self.off
        idx = np.arange(m)
        alpha = off["alpha"]
        rows = np.concatenate(
            [
                off["u"] + idx,       # (u,u)   diag
                off["theta"] + idx,   # (θ,r)
                off["theta"] + idx,   # (θ,δ)
                off["theta"] + idx,   # (θ,θ)   diag
                off["delta"] + idx,   # (δ,r)   comp cross (indefinite)
                np.full(m, alpha),    # (α,qx)  h1 bilinear cross
                np.full(m, alpha),    # (α,qy)  h1 bilinear cross
                [alpha],              # (α,α)   h1 curvature + reg_alpha ridge
            ]
        )
        cols = np.concatenate(
            [
                off["u"] + idx, off["r"] + idx, off["delta"] + idx,
                off["theta"] + idx, off["r"] + idx,
                off["qx"] + idx, off["qy"] + idx, [alpha],
            ]
        )
        return rows.astype(np.int64), cols.astype(np.int64)

    def hessianstructure(self):
        return self._hrows, self._hcols

    def hessian(self, x, lagrange, obj_factor):
        m, ro = self.m, self.roff
        ea, c, s, divq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        l1 = lagrange[ro["h1"] : ro["h1"] + m]
        l2x, l2y = lagrange[ro["h2x"] : ro["h2x"] + m], lagrange[ro["h2y"] : ro["h2y"] + m]
        l3x, l3y = lagrange[ro["h3x"] : ro["h3x"] + m], lagrange[ro["h3y"] : ro["h3y"] + m]
        xi = lagrange[ro["comp"] : ro["comp"] + m]
        H_uu = obj_factor * self._ones
        H_tr = l2x * s - l2y * c                                    # (θ,r)
        H_td = l3x * s - l3y * c                                    # (θ,δ)
        H_tt = r * (l2x * c + l2y * s) + delta * (l3x * c + l3y * s)  # (θ,θ)
        H_tt = H_tt + obj_factor * self.eps_theta                   # TR ridge (if on)
        H_dr = -xi                                                  # (δ,r) comp cross
        # h1 bilinear curvature: ∂²(λ_h1·h1) couples α with qx, qy and itself.
        H_aqx = ea * (self.Kx @ l1)                                 # (α,qx)
        H_aqy = ea * (self.Ky @ l1)                                 # (α,qy)
        # (α,α): h1 curvature + the objective ridge (fix 1).
        H_aa = np.array([ea * float(np.dot(l1, divq)) + obj_factor * self.reg_alpha])
        return np.concatenate([H_uu, H_tr, H_td, H_tt, H_dr, H_aqx, H_aqy, H_aa])

    # ---- per-NLP-solve telemetry (reset by the driver before each solve) ---
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.n_iter = iter_count
        self.inf_pr = inf_pr
        self.inf_du = inf_du  # final (scaled) dual infeasibility
        self.mu_last = mu     # final barrier parameter → next level's mu_init (fix 2)
        # regularization_size is IPOPT's δ_w (primal/Hessian inertia correction,
        # 'lg(rg)'). Incomplete D1 detector under L-BFGS (PD by construction); the
        # dual/Jacobian δ_c is not exposed by IPOPT. Use θ-drift + restoration too.
        if regularization_size > 0.0:
            self.n_reg += 1
        if alg_mod == 1:
            self.n_rest += 1
        # μ-coupled Scholtes level (--t-update mu): slave t to the barrier,
        # t = max(t_floor, c·μ), TIGHTENING ONLY — the monotone guard keeps
        # adaptive-μ oscillations and the restoration phase (whose μ here is the
        # restoration problem's) from loosening the comp row mid-solve. IPOPT is
        # not told the NLP changed: the accepted iterate's cached constraint
        # values lag the update by one iteration — a mild, one-sided perturbation
        # the next evaluation absorbs (the Jacobian/Hessian are t-free).
        if self.t_mu_scale > 0.0:
            t_new = max(self.t_floor, self.t_mu_scale * mu)
            tightened = t_new < self.t_comp
            if tightened:
                self.t_comp = t_new
                if self.c_theta_live > 0.0:
                    self.eps_theta = self.c_theta_live * t_new  # TR ridge ∝ t
                self.t_hist.append((iter_count, t_new))
            # Live progress + stall watch (only when the driver armed it, i.e. under
            # print_level 0 where IPOPT is otherwise silent). A run whose t never
            # tightens off the first level is stuck at the incoming relaxation with
            # μ frozen — the near-singular-KKT first-level failure; flag it so the
            # long single solve is not mistaken for a hang (Ctrl-C still works).
            ev = self._mu_progress_every
            if ev:
                self._mu_stall_iters = 0 if tightened else self._mu_stall_iters + 1
                if iter_count % ev == 0:
                    print(f"    [mu-coupled] it={iter_count:>4d}  μ={mu:.2e}  "
                          f"t={self.t_comp:.2e}  inf_pr={inf_pr:.1e}  "
                          f"inf_du={inf_du:.1e}  obj={obj_value:.4e}", flush=True)
                if (not self._mu_stall_warned and self._mu_stall_iters >= 100
                        and self.t_comp > self.t_floor * (1 + 1e-9)):
                    self._mu_stall_warned = True
                    print(f"    [mu-coupled] ⚠ t pinned at {self.t_comp:.2e} for "
                          f"{self._mu_stall_iters} iters (μ not decreasing, "
                          f"inf_du={inf_du:.1e}) — near-singular first level; the "
                          f"barrier is stuck so t cannot tighten. Consider the "
                          f"noise-aware default α₀, --c-theta (D1 gauge ridge), or "
                          f"--hess-update exact. Ctrl-C to stop.", flush=True)
        # Cooperative interruption: the driver's SIGINT handler sets ``_interrupt``;
        # returning False here asks IPOPT to stop THIS solve at the current iterate
        # (status = User_Requested_Stop) instead of raising a KeyboardInterrupt
        # mid-C-call. The driver then reports the last completed level.
        return not self._interrupt


# ---------------------------------------------------------------------------
# Initial point and bounds
# ---------------------------------------------------------------------------
def chambolle_pock_rof(f, Kx, Ky, KxT, KyT, lam, n_iter=3000, tol=1e-7,
                       n_accel=300, u0=None, qx0=None, qy0=None):
    """Hybrid Chambolle–Pock for ROF  min_u ½‖u−f‖² + lam·‖∇u‖_{2,1}.

    Returns ``(u, qx, qy)`` with the per-pixel dual on the **unit ball**
    ‖(qxᵢ,qyᵢ)‖ ≤ 1.  At convergence  u = f − lam·(Kxᵀ qx + Kyᵀ qy)  with the
    complementarity r ⊥ (1 − δ) — i.e. *exactly* the unit-ball MPCC lower-level
    system at e^α = lam, so a CP solve is a consistent, approximately-feasible
    warm start for the whole lifted problem (h1 exact, comp ≈ 0).

    v2 (fix 5) — hybrid schedule with a residual stop:

    * **Phase 1 (≤ n_accel iters)**: the primal is 1-strongly convex, so the
      γ-accelerated variant (Chambolle & Pock 2011, Alg. 2, γ = 1:
      θ = 1/√(1+2τ), τ ← θτ, σ ← σ/θ) converges u and the complementarity
      O(1/k²) — but its h1 consistency lags at O(1/k) because τ → 0.
    * **Phase 2**: fixed steps (θ = 1) polish the h1 fixed point, which
      contracts fast once u is near the solution.
    * **Stop on the true h1 residual**: the primal prox gives, exactly,
      u₊ − f + lam·div q = −(u₊ − u)/τ, so ``‖u₊−u‖∞/τ ≤ tol`` (absolute) tests
      the h1 feasibility the MPCC warm start needs — at zero extra cost. (v1's
      relative ‖Δu‖ test never fired; it always ran all 3000 iterations.)

    Optional ``(u0, qx0, qy0)`` warm-starts the iteration (used by
    :func:`cp_scan` to chain across the α grid); the transposes are passed in
    rather than rebuilt per call.
    """
    tau0 = 0.99 / (np.sqrt(8.0) * lam)  # τ₀·lam = σ₀·lam = 0.99/√8 (‖∇‖² ≤ 8, 2D)
    sig_lam0 = 0.99 / np.sqrt(8.0)      # σ·lam — the dual step applied to K ū
    tau, sig_lam = tau0, sig_lam0
    u = f.copy() if u0 is None else np.asarray(u0, float).copy()
    ubar = u.copy()
    qx = np.zeros_like(f) if qx0 is None else np.asarray(qx0, float).copy()
    qy = np.zeros_like(f) if qy0 is None else np.asarray(qy0, float).copy()
    for k in range(n_iter):
        accel = k < n_accel
        if not accel and tau != tau0:
            tau, sig_lam = tau0, sig_lam0  # phase switch (τσ invariant ⇒ exact reset)
        qx += sig_lam * (Kx @ ubar)    # dual ascent (σ·lam = sig_lam) …
        qy += sig_lam * (Ky @ ubar)
        nrm = np.maximum(1.0, np.hypot(qx, qy))  # … + projection onto unit ball
        qx /= nrm
        qy /= nrm
        u_new = (tau * f + u - (tau * lam) * (KxT @ qx + KyT @ qy)) / (tau + 1.0)  # prox
        h1_res = float(np.max(np.abs(u_new - u))) / tau  # = ‖u₊−f+lam·div q‖∞
        if accel:
            th = 1.0 / np.sqrt(1.0 + 2.0 * tau)  # γ = 1 (strong convexity of ½‖·−f‖²)
            ubar = u_new + th * (u_new - u)
            tau *= th                            # τσ‖lam·K‖² ≤ 0.99² is invariant
            sig_lam /= th
        else:
            ubar = 2.0 * u_new - u               # θ = 1 extrapolation
        u = u_new
        if h1_res <= tol:
            break
    return u, qx, qy


def cp_scan(prob: LiftedTVMPCC, alpha_center: float, half_width: float = 2.0, n: int = 9):
    """Coarse Chambolle–Pock pre-scan of the bilevel loss
    ``L(α) = ½‖u_CP(e^α) − u_clean‖²`` on ``n`` log-spaced weights spanning
    ``alpha_center ± half_width``; returns the argmin ``(α*, u, qx, qy)`` — the CP
    lower-level solution to warm-start from.

    Legitimate because CP is an *exact* lower-level solve and ``u_clean`` is the
    training target, so ``L(α)`` here is the true (unimodal) bilevel objective.
    v2 (fix 5): consecutive grid points warm-start from the previous CP solution
    (adjacent weights have nearby solutions), on top of the accelerated solver.
    """
    alphas = alpha_center + np.linspace(-half_width, half_width, n)
    best = None
    u = qx = qy = None
    for a in alphas:
        u, qx, qy = chambolle_pock_rof(
            prob.f, prob.Kx, prob.Ky, prob.KxT, prob.KyT, float(np.exp(a)),
            u0=u, qx0=qx, qy0=qy,
        )
        loss = 0.5 * float(np.sum((u - prob.u_clean) ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(a), u, qx, qy)
    return best[1], best[2], best[3], best[4]


def initial_point(prob: LiftedTVMPCC, alpha0: float, init: str = "cp") -> np.ndarray:
    """Warm start for the lifted MPCC.

    ``cold``: u = f, q = 0. With u = f we read off the polar gradient (r, θ); q = 0
    gives δ = 0, so the N-side 1 − δ = 1. h1 holds (q = 0 kills the e^α·div q term);
    only the complementarity r·(1 − δ) = r is violated. This sits on the
    *no-regularization* manifold and biases the continuation toward the spurious
    small-α (near-noisy) branch, especially in large dimension.

    ``cp`` (default): run Chambolle–Pock ROF at e^α₀ to get a lower-level primal–
    dual pair (u*, q*), then lift it: u = u*, q = q*, δ = |q*|, and — since at the
    ROF solution q ∥ ∇u where |∇u|>0 — take **θ from the dual** (θ = ∠q*) so h3 is
    exact and h1 holds by CP optimality; r = |∇u*|. Lands ~4 orders of magnitude
    closer to complementarity feasibility (r·(1−δ) ≈ 0) *inside the good-denoising
    basin*, curing the spurious near-noisy convergence.

    ``cp-scan``: like ``cp`` but the CP weight is chosen by :func:`cp_scan` — a
    coarse sweep of L(α) around α₀ — so the seed sits at the true bilevel optimum's
    basin without depending on α₀ being well-placed. ``α₀`` then only centres the
    scan window.
    """
    m, off = prob.m, prob.off
    x = np.zeros(prob.n)
    if init in ("cp", "cp-scan"):
        if init == "cp-scan":
            alpha0, u, qx, qy = cp_scan(prob, alpha0)   # α* = argmin L(α) on the grid
        else:
            u, qx, qy = chambolle_pock_rof(
                prob.f, prob.Kx, prob.Ky, prob.KxT, prob.KyT, float(np.exp(alpha0))
            )
        x[off["u"] : off["u"] + m] = u
        x[off["qx"] : off["qx"] + m] = qx
        x[off["qy"] : off["qy"] + m] = qy
        x[off["r"] : off["r"] + m] = np.hypot(prob.Kx @ u, prob.Ky @ u)
        x[off["delta"] : off["delta"] + m] = np.hypot(qx, qy)
        x[off["theta"] : off["theta"] + m] = np.arctan2(qy, qx)  # dual angle → h3 exact
        x[off["alpha"]] = alpha0
        return x
    # cold: u = f, q = 0
    x[off["u"] : off["u"] + m] = prob.f
    gx, gy = prob.Kx @ prob.f, prob.Ky @ prob.f
    x[off["r"] : off["r"] + m] = np.hypot(gx, gy)
    x[off["theta"] : off["theta"] + m] = np.arctan2(gy, gx)
    # qx = qy = 0, delta = 0 already (zeros)
    x[off["alpha"]] = alpha0
    return x


def bounds(prob: LiftedTVMPCC, alpha_lo=-15.0, alpha_hi=15.0):
    """Variable bounds. Only δ ≤ 1 and the α box remain as bounds.

    The non-negativities r ≥ 0 and δ ≥ 0 are NOT box bounds here — they are the
    explicit inequality rows ``hr``/``hd`` in ``constraints`` (their bounds are set
    by the driver: ``0 ≤ r, δ ≤ +∞``). δ ≤ 1 (the unit-ball radius, N-side of the
    complementarity) is kept as a box. To experiment with the D2 fix r ≥ ε_r,
    raise the lower bound of the ``hr`` rows in ``solve_scholtes``.
    """
    m, off, n = prob.m, prob.off, prob.n
    xl = np.full(n, -2.0e19)
    xu = np.full(n, 2.0e19)
    xu[off["delta"] : off["delta"] + m] = 1.0  # δ ≤ 1 (unit-ball radius)
    xl[off["alpha"]] = alpha_lo
    xu[off["alpha"]] = alpha_hi
    return xl, xu


# ---------------------------------------------------------------------------
# Scholtes ε-continuation driver
# ---------------------------------------------------------------------------
def solve_scholtes(
    prob, x0, schedule, *, linear_solver, tol=1e-8, tol_factor=0.1, max_iter=1500,
    dual_warmstart=False, c_theta=0.0, hess_update="bfgs", mu_strategy="monotone",
    xi_rescale=False, ma57_scaling="auto", print_level=0, verbose=True,
):
    """Solve a sequence of relaxed NLPs with r·w ≤ t for t ↓ along ``schedule``.

    * **Tolerance coupling** — level ``t`` is solved only to
      ``max(tol, tol_factor·t)`` (the Scholtes path is itself O(√t)-accurate).
    * **Primal + dual + barrier warm-start (fix 2)** — carry the previous iterate;
      with ``dual_warmstart`` (default) also pass the previous multipliers, arm
      ``warm_start_init_point``, set ``mu_init`` to the previous level's final
      barrier parameter (clamped to [1e-9, 1e-1]) and shrink the warm-start push
      options to 1e-8 — so the barrier continues where it left off instead of
      restarting at μ = 0.1 every level. ``--xi-rescale`` additionally scales the
      comp-row multipliers by t_prev/t (ξ ~ 1/t along the Scholtes path).
    * **One Problem object (fix 6)** — the Scholtes t enters through
      ``prob.t_comp`` (comp row = r·(1−δ) − t ≤ 0, fixed upper bound 0), so a
      single cyipopt.Problem is reused across all levels; only options change.
    * **Reported iterate = tightest complementarity-faithful converged level.** We
      target the MPCC (t→0), so among levels IPOPT actually solves (status ∈ {0,1})
      we report the one with the smallest ``r·w`` — the most complementarity-faithful
      bilevel point. We do NOT report the smallest *loss*: on a natural image the
      loss *rises* as t↓ (exact-TV optimality is only an approximate prior), so the
      loosest, barely-constrained level fits u_clean best but is not a real
      lower-level solution. Watch the per-level ``α``/``obj`` columns to see this
      loose-vs-tight tension directly (and the loosest level's junk α).
    * **Stop-at-first-failure** — halt at the first level IPOPT does not solve
      (status ∉ {0,1}); the small-t tail is where the D2/D3 degeneracy bites.
    * **Interruptible (Ctrl-C)** — a SIGINT asks IPOPT (via the intermediate
      callback) to stop the current solve cleanly, then the loop halts and reports
      the best *completed* level; a partial, interrupted level is never adopted. A
      second Ctrl-C force-quits. The original SIGINT handler is always restored.
    * **TR gauge ridge (optional)** — with ``c_theta > 0`` add ½·(c_θ·t)·‖θ−θ_ref‖²
      to the objective each level (weight ∝ t, so it vanishes as t↓0).

    Returns ``(best_x, best_info, best_t, total_iter, best_n_reg, best_n_rest)``.
    """
    m, off = prob.m, prob.off
    xl, xu = bounds(prob)
    cl = np.full(prob.m_con, 0.0)
    cu = np.full(prob.m_con, 0.0)
    # hr, hd are the explicit non-negativities 0 ≤ r, δ ≤ +∞. The comp rows are
    # one-sided with FIXED upper bound 0: r·(1−δ) − t ≤ 0, t on prob.t_comp.
    cu[prob.roff["hr"] : prob.roff["comp"]] = 2.0e19
    cl[prob.roff["comp"] :] = -2.0e19

    use_ma57 = (linear_solver == "ma57") and os.path.exists(HSLLIB)
    if linear_solver == "ma57" and not use_ma57:
        print(f"  [warn] {HSLLIB} not found → falling back to MUMPS")
        linear_solver = "mumps"
    # MC64 scaling (fix 3): must be armed for the WHOLE continuation or not at
    # all — a per-level trigger was tried and fails (the tail rescue is
    # path-dependent; see module docstring). 'auto' = on iff m ≥ 1600 (N ≥ 40),
    # calibrated on N=32 (never pays) vs N=48 (required for the tail).
    mc64 = ma57_scaling in (True, "on") or (ma57_scaling == "auto" and prob.m >= 1600)
    if verbose and use_ma57:
        print(f"  MA57 MC64 scaling: {'on' if mc64 else 'off'}  (--ma57-scaling "
              f"{ma57_scaling if isinstance(ma57_scaling, str) else 'on'})")

    if verbose:
        print(
            f"  {'t':>9} {'tol':>8} {'μ0':>8} {'iters':>6} {'δw':>4} {'rest':>4} "
            f"{'status':>7} {'comp_res':>10} {'max|ξ|':>10} {'α':>8} {'obj':>11}"
        )

    x = x0.copy()
    warm = None  # (mult_g, mult_x_L, mult_x_U, t of the producing level)
    best = None  # (comp_res, x, info, t, n_reg, inf_du, n_rest) of the reported level
    info = None
    comp_res = float("nan")
    total_iter = 0

    # ---- cooperative Ctrl-C interruption ---------------------------------
    # First SIGINT flips prob._interrupt; the intermediate callback then returns
    # False, stopping the current IPOPT solve cleanly (User_Requested_Stop, status
    # 5) instead of raising KeyboardInterrupt mid-C-call. The loop then reports the
    # best *completed* level. A second SIGINT restores the previous handler and
    # re-raises for a hard quit. Only installable from the main thread.
    prob._interrupt = False

    def _on_sigint(signum, frame):
        if prob._interrupt:  # second Ctrl-C → hard quit
            signal.signal(signal.SIGINT, _old_sigint)
            raise KeyboardInterrupt
        prob._interrupt = True
        print("\n  [interrupt] Ctrl-C — stopping after the current IPOPT iteration; "
              "will report the best completed level. Ctrl-C again to force-quit.",
              flush=True)

    try:
        _old_sigint = signal.signal(signal.SIGINT, _on_sigint)
        _sig_installed = True
    except (ValueError, OSError):  # not in the main thread → no cooperative stop
        _old_sigint, _sig_installed = None, False

    nlp = None
    try:
        # One Problem for the whole continuation (fix 6); options are updated in
        # place per level below.
        nlp = cyipopt.Problem(
            n=prob.n, m=prob.m_con, problem_obj=prob, lb=xl, ub=xu, cl=cl, cu=cu
        )
        nlp.add_option("sb", "yes")
        nlp.add_option("print_level", int(print_level))
        nlp.add_option("max_iter", max_iter)
        nlp.add_option("acceptable_iter", 10)
        nlp.add_option("mu_strategy", mu_strategy)
        nlp.add_option("linear_solver", linear_solver)
        if hess_update == "exact":
            nlp.add_option("hessian_approximation", "exact")  # prob.hessian callback
        else:
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 25)
            nlp.add_option("limited_memory_update_type", hess_update)  # bfgs / sr1
        if use_ma57:
            nlp.add_option("hsllib", HSLLIB)
            nlp.add_option("ma57_automatic_scaling", "yes" if mc64 else "no")
        # Warm-start geometry (fix 2): when warm_start_init_point is armed, keep
        # the supplied primal/dual point instead of pushing it into the interior.
        for opt in ("warm_start_bound_push", "warm_start_bound_frac",
                    "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
                    "warm_start_mult_bound_push"):
            nlp.add_option(opt, 1e-8)

        for t in schedule:
            prob.t_comp = t               # comp row: r·(1−δ) − t ≤ 0
            prob.eps_theta = c_theta * t  # TR gauge ridge weight (0 = off)
            tol_t = max(tol, tol_factor * t)
            nlp.add_option("tol", tol_t)
            nlp.add_option("acceptable_tol", max(tol, 10 * tol_t))

            prob.n_iter = 0
            prob.n_reg = 0
            prob.n_rest = 0
            mu0 = None
            if dual_warmstart and warm is not None:
                nlp.add_option("warm_start_init_point", "yes")
                mu_prev = prob.mu_last  # final μ of the previous level's solve
                mu0 = float(np.clip(mu_prev if np.isfinite(mu_prev) else 1e-4,
                                    1e-9, 1e-1))
                nlp.add_option("mu_init", mu0)
                mg_ws, zl_ws, zu_ws, t_prev = warm
                if xi_rescale and t_prev > 0.0:
                    mg_ws = mg_ws.copy()
                    mg_ws[prob.roff["comp"] :] *= t_prev / t  # ξ ~ 1/t on the path
                x, info = nlp.solve(x, lagrange=mg_ws, zl=zl_ws, zu=zu_ws)
            else:
                x, info = nlp.solve(x)
            _mg = info.get("mult_g")
            if _mg is not None and len(_mg):
                warm = (_mg, info.get("mult_x_L"), info.get("mult_x_U"), t)
            total_iter += prob.n_iter

            r = x[off["r"] : off["r"] + m]
            w = 1.0 - x[off["delta"] : off["delta"] + m]  # N-side of the comp: 1 − δ
            comp_res = float(np.max(r * w)) if m else 0.0
            a_k = float(x[off["alpha"]])
            xi_max = (
                float(np.max(np.abs(_mg[prob.roff["comp"] :])))
                if (_mg is not None and len(_mg)) else float("nan")
            )
            if verbose:
                mu0_s = f"{mu0:.1e}" if mu0 is not None else "-"
                print(
                    f"  {t:>9.1e} {tol_t:>8.1e} {mu0_s:>8} {prob.n_iter:>6d} "
                    f"{prob.n_reg:>4d} {prob.n_rest:>4d} {info['status']:>7d} "
                    f"{comp_res:>10.2e} {xi_max:>10.2e} {a_k:>8.3f} "
                    f"{info['obj_val']:>11.4e}"
                )

            converged = info["status"] in (0, 1)
            if converged and (best is None or comp_res < best[0]):
                best = (comp_res, x.copy(), info, t, prob.n_reg, prob.inf_du, prob.n_rest)
            if prob._interrupt:
                if verbose:
                    if best is not None:
                        print("  [stop] interrupted by user — reporting the best "
                              f"completed level (t={best[3]:.1e}).")
                    else:
                        print("  [stop] interrupted before any level completed — "
                              "reporting the partial iterate (NOT converged).")
                break
            if not converged:
                if verbose:
                    print(
                        f"  [stop] t={t:.1e} did not converge (status {info['status']}); "
                        f"halting — small-t degeneracy. Reporting best good-branch level."
                    )
                break
    finally:
        if nlp is not None:
            nlp.close()
        if _sig_installed:
            signal.signal(signal.SIGINT, _old_sigint)  # restore the prior handler

    if best is None:  # no level converged (first-level failure or early interrupt)
        best = (comp_res, x, info, t, prob.n_reg, prob.inf_du, prob.n_rest)
    prob.inf_du = best[5]  # expose the reported level's dual infeasibility to certify
    return best[1], best[2], best[3], total_iter, best[4], best[6]


# ---------------------------------------------------------------------------
# μ-coupled continuation driver (--t-update mu, experimental)
# ---------------------------------------------------------------------------
def solve_mu_coupled(
    prob, x0, *, linear_solver, t_mu_scale=10.0, t_min=1e-4, tol=1e-8,
    tol_factor=0.1, max_iter=1500, c_theta=0.0, hess_update="bfgs",
    mu_strategy="monotone", ma57_scaling="auto", print_level=0, verbose=True,
):
    """ONE IPOPT solve with the Scholtes t slaved to the barrier parameter μ.

    Instead of the outer loop of NLP solves over a geometric t-schedule, the
    ``intermediate`` callback sets, every iteration,

        t  =  max(t_min, c·μ),          c = t_mu_scale > 0,

    monotonically non-increasing — so the complementarity relaxation and the
    barrier are driven down *together* (the interior-relaxation coupling of
    Raghunathan & Biegler, applied externally: t enters the comp row additively,
    so the Jacobian/Hessian never change and IPOPT only sees a between-iteration
    shift of one inequality bound). With monotone μ (mu_init = 0.1) the path
    starts at t = c/10 (default c = 10 → t₀ = 1, matching the geometric
    schedule's first level) and bottoms out on the t_min floor as μ → 0; the
    floor is essential — without it t → c·μ_final ≈ c·tol lands in the exact-
    complementarity degenerate regime the continuation exists to avoid.

    The single solve runs at ``max(tol, tol_factor·t_min)`` — the same coupling
    the geometric mode applies to its final level, for a fair A/B. Ctrl-C stops
    the solve cleanly via the intermediate callback (partial iterate reported,
    marked NOT converged by its status). Returns the same tuple as
    :func:`solve_scholtes`:  ``(x, info, t_final, n_iter, n_reg, n_rest)``.
    """
    m, off = prob.m, prob.off
    xl, xu = bounds(prob)
    cl = np.full(prob.m_con, 0.0)
    cu = np.full(prob.m_con, 0.0)
    cu[prob.roff["hr"] : prob.roff["comp"]] = 2.0e19
    cl[prob.roff["comp"] :] = -2.0e19

    use_ma57 = (linear_solver == "ma57") and os.path.exists(HSLLIB)
    if linear_solver == "ma57" and not use_ma57:
        print(f"  [warn] {HSLLIB} not found → falling back to MUMPS")
        linear_solver = "mumps"
    mc64 = ma57_scaling in (True, "on") or (ma57_scaling == "auto" and prob.m >= 1600)
    if verbose and use_ma57:
        print(f"  MA57 MC64 scaling: {'on' if mc64 else 'off'}  (--ma57-scaling "
              f"{ma57_scaling if isinstance(ma57_scaling, str) else 'on'})")

    # Arm the coupling: the intermediate callback owns t from here on. Seed t at
    # c·mu_init (adaptive μ re-picks μ₀ itself; the first callback then syncs t
    # downward — the monotone guard ignores a larger adaptive μ₀).
    mu0 = 0.1
    prob.t_mu_scale = float(t_mu_scale)
    prob.t_floor = float(t_min)
    prob.c_theta_live = float(c_theta)
    prob.t_comp = max(t_min, t_mu_scale * mu0)
    prob.eps_theta = c_theta * prob.t_comp
    prob.t_hist = [(0, prob.t_comp)]
    prob.n_iter = 0
    prob.n_reg = 0
    prob.n_rest = 0
    # Live per-iteration progress (the single solve prints no per-level rows).
    # Only when IPOPT is otherwise silent (print_level 0) and verbose, so we do
    # not double-log against IPOPT's own trace.
    prob._mu_progress_every = 25 if (verbose and int(print_level) == 0) else 0
    prob._mu_stall_iters = 0
    prob._mu_stall_warned = False

    tol_solve = max(tol, tol_factor * t_min)

    if verbose:
        print(
            f"  {'t_final':>9} {'tol':>8} {'μ0':>8} {'iters':>6} {'δw':>4} {'rest':>4} "
            f"{'status':>7} {'comp_res':>10} {'max|ξ|':>10} {'α':>8} {'obj':>11}"
        )

    # ---- cooperative Ctrl-C interruption (same contract as solve_scholtes) --
    prob._interrupt = False

    def _on_sigint(signum, frame):
        if prob._interrupt:  # second Ctrl-C → hard quit
            signal.signal(signal.SIGINT, _old_sigint)
            raise KeyboardInterrupt
        prob._interrupt = True
        print("\n  [interrupt] Ctrl-C — stopping after the current IPOPT iteration; "
              "the partial iterate will be reported (NOT converged). "
              "Ctrl-C again to force-quit.", flush=True)

    try:
        _old_sigint = signal.signal(signal.SIGINT, _on_sigint)
        _sig_installed = True
    except (ValueError, OSError):
        _old_sigint, _sig_installed = None, False

    nlp = None
    try:
        nlp = cyipopt.Problem(
            n=prob.n, m=prob.m_con, problem_obj=prob, lb=xl, ub=xu, cl=cl, cu=cu
        )
        nlp.add_option("sb", "yes")
        nlp.add_option("print_level", int(print_level))
        nlp.add_option("max_iter", max_iter)
        nlp.add_option("acceptable_iter", 10)
        nlp.add_option("mu_strategy", mu_strategy)
        nlp.add_option("mu_init", mu0)  # read by monotone; adaptive picks its own
        nlp.add_option("linear_solver", linear_solver)
        nlp.add_option("tol", tol_solve)
        nlp.add_option("acceptable_tol", max(tol, 10 * tol_solve))
        if hess_update == "exact":
            nlp.add_option("hessian_approximation", "exact")
        else:
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 25)
            nlp.add_option("limited_memory_update_type", hess_update)
        if use_ma57:
            nlp.add_option("hsllib", HSLLIB)
            nlp.add_option("ma57_automatic_scaling", "yes" if mc64 else "no")
        x, info = nlp.solve(x0.copy())
    finally:
        if nlp is not None:
            nlp.close()
        if _sig_installed:
            signal.signal(signal.SIGINT, _old_sigint)
        prob.t_mu_scale = 0.0  # disarm: later evaluations must not move t
        prob._mu_progress_every = 0

    t_final = prob.t_comp
    r = x[off["r"] : off["r"] + m]
    w = 1.0 - x[off["delta"] : off["delta"] + m]
    comp_res = float(np.max(r * w)) if m else 0.0
    _mg = info.get("mult_g")
    xi_max = (
        float(np.max(np.abs(_mg[prob.roff["comp"] :])))
        if (_mg is not None and len(_mg)) else float("nan")
    )
    if verbose:
        print(
            f"  {t_final:>9.1e} {tol_solve:>8.1e} {mu0:>8.1e} {prob.n_iter:>6d} "
            f"{prob.n_reg:>4d} {prob.n_rest:>4d} {info['status']:>7d} "
            f"{comp_res:>10.2e} {xi_max:>10.2e} {float(x[off['alpha']]):>8.3f} "
            f"{info['obj_val']:>11.4e}"
        )
        hist = prob.t_hist
        fmt = lambda e: f"{e[0]}:{e[1]:.1e}"  # noqa: E731
        shown = (", ".join(map(fmt, hist)) if len(hist) <= 12 else
                 ", ".join(map(fmt, hist[:8])) + ", …, " + ", ".join(map(fmt, hist[-3:])))
        floor = "  [t_min floor active]" if t_final <= t_min * (1 + 1e-12) else ""
        print(f"  t path (iter:t, {len(hist)} values): {shown}{floor}")
        if info["status"] not in (0, 1):
            print(f"  [stop] μ-coupled solve did not converge (status {info['status']}); "
                  f"reporting the final iterate as-is.")

    return x, info, t_final, prob.n_iter, prob.n_reg, prob.n_rest


# ---------------------------------------------------------------------------
def psnr(clean, recon, data_range=1.0):
    mse = float(np.mean((clean - recon) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)


def save_triptych(path, u_clean, f, u, N):
    """Write a clean │ noisy │ recon side-by-side PNG (each panel N×N, 1px gap)."""
    from PIL import Image

    def panel(v):
        return (np.clip(v, 0.0, 1.0).reshape(N, N) * 255).astype(np.uint8)

    gap = np.full((N, 2), 255, np.uint8)
    strip = np.concatenate([panel(u_clean), gap, panel(f), gap, panel(u)], axis=1)
    Image.fromarray(strip).save(path)


# ---------------------------------------------------------------------------
# Original-MPCC multiplier recovery + stationarity classification
# ---------------------------------------------------------------------------
def certify_stationarity(prob, x, info, t, *, cert_c=3.0, sign_tol=1e-7, mult_warn=1e3):
    """Recover the original-MPCC multipliers and classify the stationarity type.

    The lifted complementarity here is  ``0 ≤ r ⊥ (1 − δ) ≥ 0``, the pair
    ``M := r ≥ 0``, ``N := 1 − δ ≥ 0``.  Write ``w := 1 − δ`` for the N-side; its
    bound multiplier is ``z_w := z_{δ,hi}`` (the multiplier of the box ``δ ≤ 1``).
    IPOPT returns the relaxed-NLP multipliers; the Scholtes back-translation
    recovers the candidate MPCC multipliers for M/N::

        γ_i = z_r,i − ξ_i · w_i          (multiplier of  M = r)
        ν_i = z_w,i − ξ_i · r_i          (multiplier of  N = 1 − δ)

    with ``z_r`` the multiplier of ``r ≥ 0`` (here the explicit inequality row
    ``hr``, so ``z_r = −λ_hr`` from ``mult_g``), ``z_w = z_{δ,hi}`` that of the box
    ``δ ≤ 1``, and ``ξ ≥ 0`` the multiplier of ``r·(1 − δ) ≤ t``.  The δ ≥ 0
    multiplier ``μ`` likewise comes from the ``hd`` row (``μ = −λ_hd``).  The
    α-stationarity row carries the reg_alpha ridge (fix 1) and, with the q-rows,
    the e^α from h1:

        R_qx = e^α·(Kx·λ_h1) + λ_h3x
        R_qy = e^α·(Ky·λ_h1) + λ_h3y
        R_α  = reg_alpha·α + e^α·⟨λ_h1, div q⟩ − z_{α,lo} + z_{α,hi}

    and the index-set corner cap for w, since ``w = 1 − δ ≤ 1``:

        ε_r = min(c√t, ½·max r),   ε_w = min(c√t, ½)
        𝓐 = {r ≤ ε_r, w > ε_w}   𝓘 = {r > ε_r, w ≤ ε_w}   𝓑 = {r ≤ ε_r, w ≤ ε_w}
        𝓝 = {r > ε_r, w > ε_w}  (unresolved).

    The biactive sign test on 𝓑 names the type (W / C / M / S). Returns a dict of
    everything printed by :func:`print_certificate`.
    """
    m, off, ro = prob.m, prob.off, prob.roff
    eps = cert_c * np.sqrt(max(t, 0.0))

    mg = np.asarray(info["mult_g"], float)
    zL = np.asarray(info["mult_x_L"], float)
    zU = np.asarray(info["mult_x_U"], float)

    def grow(name):
        s = ro[name]
        return mg[s : s + m]

    lam_h1 = grow("h1")
    lam_h2x, lam_h2y = grow("h2x"), grow("h2y")
    lam_h3x, lam_h3y = grow("h3x"), grow("h3y")
    xi = grow("comp")  # ξ ≥ 0
    # r ≥ 0 and δ ≥ 0 are explicit inequality rows (hr, hd), so their
    # non-negativity multipliers come from mult_g, not the bound arrays. For a
    # lower-bounded (g ≥ 0) row IPOPT returns λ ≤ 0 entering ∇L as +λ·∂g, so the
    # bound-equivalent multipliers (z ≥ 0) are the negatives:
    #   z_r = −λ_hr  (mult of r ≥ 0),   z_δ = −λ_hd  (mult of δ ≥ 0).
    z_r = -grow("hr")
    z_delta = -grow("hd")

    def zrow(z, name):
        s = off[name]
        return z[s : s + m]

    z_alpha_lo, z_alpha_hi = zL[off["alpha"]], zU[off["alpha"]]

    r = prob._blk(x, "r")
    delta, theta = prob._blk(x, "delta"), prob._blk(x, "theta")
    # N-side value w = 1 − δ and its bound multiplier z_w = z_{δ,hi} (the
    # multiplier of the box δ ≤ 1, i.e. of 1 − δ ≥ 0).
    w = 1.0 - delta
    z_w = zrow(zU, "delta")
    a_val = x[off["alpha"]]
    ea = np.exp(a_val)
    c, s = np.cos(theta), np.sin(theta)

    # --- back-translation: candidate MPCC multipliers (Scholtes ∂c = (w, r)) --
    gamma = z_r - xi * w   # multiplier of M = r
    nu = z_w - xi * r      # multiplier of N = 1 − δ  (= w)
    mu = z_delta           # multiplier of δ ≥ 0

    # --- measured index sets (M=r, N=w) with scale-aware, capped thresholds --
    # w = 1 − δ ≤ 1, so ε_w is capped at ½ (its hard ceiling), ε_r at ½·max r.
    r_scale = float(np.max(r)) if r.size else 1.0
    eps_r = min(eps, 0.5 * r_scale)
    eps_w = min(eps, 0.5)  # w ∈ [0, 1]
    eps_capped = (eps_r < eps) or (eps_w < eps)
    r_small, w_small = r <= eps_r, w <= eps_w
    setA = r_small & ~w_small   # active   : r≈0, w>0  (δ<1) → ν_i should be 0
    setI = ~r_small & w_small   # inactive : r>0, w≈0  (δ≈1) → γ_i should be 0
    setB = r_small & w_small    # biactive : r≈0, w≈0  (δ≈1) → sign test lives here
    setN = ~r_small & ~w_small  # neither: complementarity unresolved at this t
    setT = (r <= eps_r) & (delta <= eps_w)  # D1 gauge set (r≈δ≈0)

    # --- W-stationarity residual: ‖∇_x 𝓛_MPCC‖∞ over every variable block --
    Kx, Ky, KxT, KyT = prob.Kx, prob.Ky, prob.KxT, prob.KyT
    divq = KxT @ prob._blk(x, "qx") + KyT @ prob._blk(x, "qy")
    R_u = (prob._blk(x, "u") - prob.u_clean) + lam_h1 + KxT @ lam_h2x + KyT @ lam_h2y
    R_qx = ea * (Kx @ lam_h1) + lam_h3x          # ← e^α from h1
    R_qy = ea * (Ky @ lam_h1) + lam_h3y          # ← e^α from h1
    R_r = -c * lam_h2x - s * lam_h2y - gamma
    # δ-stationarity: the comp/δ term and the δ ≤ 1 bound enter through ν
    # (= z_{δ,hi} − ξ·r); μ = z_{δ,lo} is the δ ≥ 0 multiplier.
    R_delta = -c * lam_h3x - s * lam_h3y - mu + nu
    R_theta = r * s * lam_h2x - r * c * lam_h2y + delta * s * lam_h3x - delta * c * lam_h3y
    # α-stationarity: h1 coupling + the reg_alpha ridge (fix 1) + the α box.
    R_alpha = (prob.reg_alpha * float(a_val) + ea * float(np.dot(lam_h1, divq))
               - z_alpha_lo + z_alpha_hi)
    stat_res = max(
        float(np.max(np.abs(b))) if b.size else 0.0
        for b in (R_u, R_qx, R_qy, R_r, R_delta, R_theta, np.array([R_alpha]))
    )

    # --- biactive sign classification (per-pixel, with a sign tolerance) ----
    scale = max(1.0, float(np.max(np.abs(gamma))), float(np.max(np.abs(nu))))
    tau = sign_tol * scale
    gB, nB = gamma[setB], nu[setB]
    nb = int(setB.sum())
    if nb == 0:
        cls = "S"  # 𝓑 = ∅ ⇒ W = C = M = S coincide
        viol = {"C": 0, "M": 0, "S": 0}
        quad = {"both+": 0, "both-": 0, "mixed": 0, "on-axis": 0}
    else:
        pos_g, neg_g = gB > tau, gB < -tau
        pos_n, neg_n = nB > tau, nB < -tau
        zero_g, zero_n = ~pos_g & ~neg_g, ~pos_n & ~neg_n
        s_ok = (gB >= -tau) & (nB >= -tau)
        m_ok = zero_g | zero_n | (pos_g & pos_n)
        c_ok = ~((pos_g & neg_n) | (neg_g & pos_n))
        viol = {"S": int(np.sum(~s_ok)), "M": int(np.sum(~m_ok)), "C": int(np.sum(~c_ok))}
        quad = {
            "both+": int(np.sum(pos_g & pos_n)),
            "both-": int(np.sum(neg_g & neg_n)),
            "mixed": int(np.sum((pos_g & neg_n) | (neg_g & pos_n))),
            "on-axis": int(np.sum(zero_g | zero_n)),
        }
        cls = "W"
        if viol["C"] == 0:
            cls = "C"
        if viol["M"] == 0:
            cls = "M"
        if viol["S"] == 0:
            cls = "S"

    # --- degeneracy gates: s_d (averaged) plus a raw-magnitude gate --------
    s_max = 100.0
    s_d = max(s_max, (np.sum(np.abs(mg)) + np.sum(np.abs(zL)) + np.sum(np.abs(zU)))
              / (prob.m_con + prob.n)) / s_max
    xi_max = float(np.max(np.abs(xi))) if xi.size else 0.0
    # equality-constraint multiplier norm (h1..h3y, before the hr/hd/comp rows)
    lam_inf = float(np.max(np.abs(mg[: ro["hr"]]))) if ro["hr"] else 0.0
    gn_inf = max(
        float(np.max(np.abs(gamma))) if gamma.size else 0.0,
        float(np.max(np.abs(nu))) if nu.size else 0.0,
    )
    mult_inf = max(lam_inf, gn_inf, xi_max)
    theta_drift = float(np.max(np.abs(theta - prob.theta_ref))) if theta.size else 0.0

    return {
        "eps": eps, "eps_r": eps_r, "eps_w": eps_w, "eps_capped": eps_capped,
        "ea": float(ea), "t": t,
        "nA": int(setA.sum()), "nI": int(setI.sum()), "nB": nb,
        "nN": int(setN.sum()), "nT": int(setT.sum()), "setB": setB,
        "class": cls, "viol": viol, "quad": quad, "gamma": gamma, "nu": nu,
        "supp_gamma_on_I": float(np.max(np.abs(gamma[setI]))) if setI.any() else 0.0,
        "supp_nu_on_A": float(np.max(np.abs(nu[setA]))) if setA.any() else 0.0,
        "stat_res": stat_res, "inf_du": float(getattr(prob, "inf_du", float("nan"))),
        "s_d": s_d, "lam_inf": lam_inf, "xi_max": xi_max, "gn_inf": gn_inf,
        "mult_inf": mult_inf, "degenerate": bool(mult_inf > mult_warn),
        "mult_warn": mult_warn, "theta_drift": theta_drift,
        "alpha_active": bool(z_alpha_lo > tau or z_alpha_hi > tau), "tau": tau,
    }


def print_certificate(cert, n_reg, n_rest):
    """Pretty-print the stationarity certificate produced by certify_stationarity."""
    names = {
        "W": "Weak (KKT of TNLP only)",
        "C": "Clarke (γ·ν ≥ 0 on 𝓑)",
        "M": "Mordukhovich (γ·ν=0 ∨ both>0 on 𝓑)",
        "S": "Strong (γ,ν ≥ 0 on 𝓑 ⇔ KKT of the MPCC)",
    }
    v = cert
    print("─" * 64)
    print("  MPCC stationarity certificate (recovered multipliers)")
    cap = "  [capped at ½·scale]" if v["eps_capped"] else ""
    print(
        f"  corner scale  : ε = c·√t = {v['eps']:.2e}  →  r≤{v['eps_r']:.2e}, "
        f"w≤{v['eps_w']:.2e}{cap}   (w ≤ 1, e^α = {v['ea']:.3f}, t = {v['t']:.1e})"
    )
    print(
        f"  index sets    : |𝓐|={v['nA']}  |𝓘|={v['nI']}  |𝓑|={v['nB']} (biactive)  "
        f"|𝓝|={v['nN']} (unresolved)   |𝓣|={v['nT']} (D1 gauge: r≈δ≈0)"
    )
    vac = (
        "   [𝓑 empty ⇒ vacuous: W=C=M=S coincide; check |𝓝| & comp_res]"
        if v["nB"] == 0 else ""
    )
    print(f"  ⇒ STATIONARITY: {v['class']}-stationary — {names[v['class']]}{vac}")
    if v["nB"]:
        q = v["quad"]
        print(
            f"     biactive (γ,ν) quadrants: both+={q['both+']}  on-axis={q['on-axis']}  "
            f"both−={q['both-']}  mixed={q['mixed']}"
        )
        print(
            f"     biactive violations: C={v['viol']['C']}  M={v['viol']['M']}  "
            f"S={v['viol']['S']}   (0 ⇒ that class holds; sign tol={v['tau']:.1e})"
        )
    print(
        f"  support resid : max|γ| on 𝓘 = {v['supp_gamma_on_I']:.2e}   "
        f"max|ν| on 𝓐 = {v['supp_nu_on_A']:.2e}   (should be ≈ 0)"
    )
    print(
        f"  W-stat resid  : ‖∇𝓛_MPCC‖∞ = {v['stat_res']:.2e}   "
        f"(IPOPT inf_du = {v['inf_du']:.2e})"
    )
    flag = (
        f"  ⚠ ≳ {v['mult_warn']:.0e} — near-singular KKT (D2/D4); s_d's average hides it"
        if v["degenerate"] else ""
    )
    print(
        f"  cert gates    : ‖λ‖∞ = {v['lam_inf']:.2e}  ‖(γ,ν)‖∞ = {v['gn_inf']:.2e}  "
        f"‖ξ‖∞ = {v['xi_max']:.2e}  s_d = {v['s_d']:.2f}{flag}"
    )
    print(
        f"  D1 gauge      : θ-drift ‖θ−θ_ref‖∞ = {v['theta_drift']:.2e} rad "
        f"({v['theta_drift'] / np.pi:.1f}π)   |𝓣| = {v['nT']}   (large ⇒ wandering)"
    )
    print(
        f"  solver flags  : δ_w-corr iters = {n_reg} (incomplete under L-BFGS)   "
        f"restoration iters = {n_rest}   α at bound: {v['alpha_active']}"
    )


# ---------------------------------------------------------------------------
# Finite-difference validation of the analytic derivatives (directional, fix 7)
# ---------------------------------------------------------------------------
def fd_check(prob, x, seed=0, n_dirs=8):
    """Directional FD validation of the gradient, Jacobian and exact Hessian.

    v2 (fix 7): instead of dense m_con×n / n×n comparisons (≈400 MB each at
    N=32), compare J·v, gᵀv and H·v against central differences along ``n_dirs``
    random unit directions — any wrong entry shows up in every random direction
    with probability 1. The sparse COO assembly *sums* duplicate (row, col)
    entries, matching IPOPT's triplet convention (v1's dense fancy-indexing
    ``+=`` silently dropped duplicates). Also validates the OBJECTIVE gradient,
    which v1 never checked (the gap that hid the reg_alpha inconsistency).
    Returns ``(jac_err, grad_err, hess_err)``.
    """
    rng = np.random.default_rng(seed)
    x = x.astype(float).copy()
    n, mcon = prob.n, prob.m_con
    h = 1e-6

    rows, cols = prob.jacobianstructure()
    J = sp.coo_matrix((prob.jacobian(x), (rows, cols)), shape=(mcon, n)).tocsr()
    g = prob.gradient(x)

    obj_factor = float(rng.uniform(0.5, 1.5))
    lam = rng.standard_normal(mcon)
    hrows, hcols = prob.hessianstructure()
    Hl = sp.coo_matrix(
        (prob.hessian(x, lam, obj_factor), (hrows, hcols)), shape=(n, n)
    ).tocsr()  # lower triangle
    H = Hl + Hl.T - sp.diags(Hl.diagonal())  # symmetrize

    jac_err = grad_err = hess_err = 0.0
    for _ in range(n_dirs):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        xp, xm = x + h * v, x - h * v
        jac_err = max(jac_err, float(np.max(np.abs(
            (prob.constraints(xp) - prob.constraints(xm)) / (2 * h) - J @ v))))
        grad_err = max(grad_err, abs(float(
            (prob.objective(xp) - prob.objective(xm)) / (2 * h) - g @ v)))
        hess_err = max(hess_err, float(np.max(np.abs(
            (_lag_grad_dense(prob, xp, lam, obj_factor)
             - _lag_grad_dense(prob, xm, lam, obj_factor)) / (2 * h) - H @ v))))
    return jac_err, grad_err, hess_err


def _lag_grad_dense(prob, xx, lam, obj_factor):
    """∇_x [ obj_factor·J(x) + λᵀ c(x) ] via the analytic Jacobian (for the FD
    Hessian's outer difference)."""
    rows, cols = prob.jacobianstructure()
    vals = prob.jacobian(xx)
    JTlam = np.zeros(prob.n)
    np.add.at(JTlam, cols, vals * lam[rows])
    return obj_factor * prob.gradient(xx) + JTlam


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Lifted TV-MPCC denoising — unit-ball dual formulation (v2, "
                    "numerical-performance fixes)."
    )
    p.add_argument("--data", default=DEFAULT_IMAGE,
                   help="natural image file (PNG/JPEG/…); default = bundled cameraman")
    p.add_argument("--phantom", action="store_true",
                   help="use the synthetic piecewise-constant phantom instead of --data")
    p.add_argument("--N", type=int, default=32, help="image side length (problem is 6N²+1 vars / 8N² rows)")
    p.add_argument("--sigma", type=float, default=0.1, help="Gaussian noise std "
                   "(0.1 default: TV meaningfully denoises the natural image here)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha0", type=float, default=None,
                   help="initial log-strength α (default: noise-aware log(0.7·σ), "
                   "calibrated to the natural-image TV optimum; also centres the "
                   "cp-scan window)")
    p.add_argument("--init", default="cp", choices=["cold", "cp", "cp-scan"],
                   help="warm start: 'cp' (default; Chambolle–Pock ROF denoise at "
                   "e^α₀ → lower-level primal–dual point, in the good-denoising "
                   "basin); 'cp-scan' (pick the CP weight by argmin L(α) over a "
                   "coarse grid around α₀ — instance-robust); or 'cold' (u=f, q=0; "
                   "drifts to the spurious near-noisy branch — kept for A/B)")
    p.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha")
    p.add_argument("--c-theta", type=float, default=0.0, dest="c_theta",
                   help="TR (Tikhonov gauge) strength c_θ: adds ½·(c_θ·t)·‖θ−θ_ref‖² "
                   "to the objective each level (D1 fix; weight ∝ t). 0 = off.")
    p.add_argument("--t0", type=float, default=1.0, help="first Scholtes level")
    p.add_argument("--t-min", type=float, default=1e-4, dest="t_min",
                   help="smallest relaxation level (default 1e-4; push to 1e-8 for the tail)")
    p.add_argument("--factor", type=float, default=0.3, help="t reduction per level")
    p.add_argument("--t-update", default="geometric", choices=["geometric", "mu"],
                   dest="t_update",
                   help="continuation driver: 'geometric' (default: outer loop of "
                   "NLP solves over t0·factor^k ≥ t-min) or 'mu' (experimental: "
                   "ONE IPOPT solve with the Scholtes t slaved to the barrier, "
                   "t = max(t-min, c·μ), c = --t-mu-scale; ignores --t0/--factor/"
                   "--dual-warmstart/--xi-rescale)")
    p.add_argument("--t-mu-scale", type=float, default=10.0, dest="t_mu_scale",
                   help="positive scale c in the μ-coupled update t = max(t-min, c·μ) "
                   "(default 10: with mu_init = 0.1 the path starts at t = 1, "
                   "matching the geometric schedule's first level)")
    p.add_argument("--tol", type=float, default=1e-6, help="final per-level KKT tolerance (floored)")
    p.add_argument("--max-iter", type=int, default=1500, dest="max_iter",
                   help="IPOPT iteration cap per level")
    p.add_argument("--dual-warmstart", action=argparse.BooleanOptionalAction,
                   default=False, dest="dual_warmstart",
                   help="warm-start multipliers AND the barrier (mu_init = previous "
                   "level's final μ) between levels; measured MIXED (helps phantom "
                   "σ=0.2 N=32, makes cameraman σ=0.2 / N=48 tail levels grind), "
                   "so OFF by default")
    p.add_argument("--mu-strategy", default="monotone", choices=["monotone", "adaptive"],
                   dest="mu_strategy",
                   help="IPOPT barrier strategy (adaptive: A/B lever for loose levels)")
    p.add_argument("--xi-rescale", action="store_true", dest="xi_rescale",
                   help="experimental: scale warm-started comp multipliers by "
                   "t_prev/t between levels (ξ ~ 1/t along the Scholtes path)")
    p.add_argument("--linear-solver", default="ma57", choices=["ma57", "mumps"],
                   dest="linear_solver")
    p.add_argument("--ma57-scaling", default="auto", choices=["auto", "on", "off"],
                   dest="ma57_scaling",
                   help="MA57 MC64 automatic scaling for the whole continuation: "
                   "'auto' (default) = on iff N ≥ 40 — measured: it cures the "
                   "N=48 small-t tail stall (‖ξ‖∞ ~ 1/t) but ~doubles the "
                   "per-factorization cost and never pays at N=32")
    p.add_argument("--hess-update", default="bfgs", choices=["bfgs", "sr1", "exact"],
                   dest="hess_update",
                   help="Hessian: bfgs (L-BFGS, PD, default), sr1 (indefinite), or exact "
                   "(analytic Lagrangian Hessian — needed to push the small-t tail).")
    p.add_argument("--cert-c", type=float, default=3.0, dest="cert_c",
                   help="corner-scale factor c for the certificate (ε = c·√t, capped)")
    p.add_argument("--fd-check", action="store_true", dest="fd_check",
                   help="finite-difference validate gradient/Jacobian/Hessian, then exit")
    p.add_argument("--save-recon", default=None, dest="save_recon",
                   help="write a clean│noisy│recon triptych PNG to this path")
    p.add_argument("--print-level", type=int, default=0, dest="print_level",
                   help="IPOPT print_level (5 for per-iteration output)")
    args = p.parse_args()

    # Noise-aware default log-strength: e^α₀ ≈ 0.7·σ, calibrated to the cameraman
    # TV optimum (σ=0.1→α*≈−2.6, σ=0.2→α*≈−2.0). Only used if --alpha0 is omitted.
    a0_auto = args.alpha0 is None
    if a0_auto:
        args.alpha0 = float(np.log(0.7 * args.sigma))

    if args.phantom:
        u_clean, f = make_phantom(args.N, args.sigma, args.seed)
        src = f"phantom (synthetic {args.N}×{args.N})"
    else:
        u_clean, f = load_image(args.data, args.N, args.sigma, args.seed)
        src = f"{os.path.basename(args.data)} ({args.N}×{args.N})"

    prob = LiftedTVMPCC(f, u_clean, args.N, reg_alpha=args.reg_alpha)

    if args.fd_check:
        x0 = initial_point(prob, args.alpha0, init="cold")  # cold: fast, no CP solve
        # Perturb off the feasible seed so every derivative block is exercised.
        rng = np.random.default_rng(args.seed)
        x0 = x0 + 0.05 * rng.standard_normal(prob.n)
        x0[prob.off["delta"] : prob.off["delta"] + prob.m] = np.abs(
            x0[prob.off["delta"] : prob.off["delta"] + prob.m]
        )
        prob.t_comp = 0.1  # exercise the comp row's additive −t (no derivative effect)
        jac_err, grad_err, hess_err = fd_check(prob, x0, seed=args.seed)
        print(f"FD check (N={args.N}, directional):  max|(J−J_fd)v| = {jac_err:.2e}   "
              f"|(g−g_fd)ᵀv| = {grad_err:.2e}   max|(H−H_fd)v| = {hess_err:.2e}")
        ok = jac_err < 1e-6 and grad_err < 1e-6 and hess_err < 1e-5
        print("  ⇒ " + ("PASS" if ok else "FAIL") + " (thresholds 1e-6 / 1e-6 / 1e-5)")
        return

    # Geometric schedule t0, factor·t0, … ≥ t_min.
    schedule, t = [], args.t0
    while t >= args.t_min:
        schedule.append(t)
        t *= args.factor

    if args.t_update == "mu":
        if args.t_mu_scale <= 0.0:
            p.error("--t-mu-scale must be > 0")
        if args.dual_warmstart or args.xi_rescale:
            print("  [note] --t-update mu is a single solve: "
                  "--dual-warmstart/--xi-rescale are ignored")
        mode = f"t-update=μ-coupled(c={args.t_mu_scale:g})"
    else:
        mode = f"levels={len(schedule)}"
    gauge = f"TR(c_θ={args.c_theta:g})" if args.c_theta > 0 else "none"
    ws = ("dual+μ" + ("+ξ/t" if args.xi_rescale else "")) if args.dual_warmstart else "primal"
    print(
        f"Lifted TV-MPCC (unit-ball v2)   source={src}  m={prob.m}  n={prob.n}  "
        f"{mode}  solver={args.linear_solver}  hess={args.hess_update}  "
        f"μ-strat={args.mu_strategy}  warm={ws}  gauge={gauge}  init={args.init}"
    )
    x0 = initial_point(prob, args.alpha0, init=args.init)
    prob.theta_ref = x0[prob.off["theta"] : prob.off["theta"] + prob.m].copy()
    a0_eff = float(x0[prob.off["alpha"]])  # cp-scan may override α₀ with argmin L(α)
    a0_src = ("cp-scan argmin L(α)" if args.init == "cp-scan"
              else "noise-aware log(0.7·σ)" if a0_auto else "explicit --alpha0")
    print(f"  init α₀ = {a0_eff:.3f}  (e^α₀ = {np.exp(a0_eff):.4f})   [{a0_src}]")

    t_start = time.perf_counter()
    if args.t_update == "mu":
        x, info, best_t, total_iter, best_n_reg, best_n_rest = solve_mu_coupled(
            prob, x0,
            linear_solver=args.linear_solver, t_mu_scale=args.t_mu_scale,
            t_min=args.t_min, tol=args.tol, max_iter=args.max_iter,
            c_theta=args.c_theta, hess_update=args.hess_update,
            mu_strategy=args.mu_strategy, ma57_scaling=args.ma57_scaling,
            print_level=args.print_level,
        )
    else:
        x, info, best_t, total_iter, best_n_reg, best_n_rest = solve_scholtes(
            prob, x0, schedule,
            linear_solver=args.linear_solver, tol=args.tol, max_iter=args.max_iter,
            dual_warmstart=args.dual_warmstart, c_theta=args.c_theta,
            hess_update=args.hess_update, mu_strategy=args.mu_strategy,
            xi_rescale=args.xi_rescale, ma57_scaling=args.ma57_scaling,
            print_level=args.print_level,
        )
    wall = time.perf_counter() - t_start

    u = x[: prob.m]
    alpha = x[prob.off["alpha"]]
    msg = info["status_msg"]
    msg = msg.decode() if isinstance(msg, bytes) else msg
    print("─" * 64)
    t_note = ("final coupled level, t = max(t_min, c·μ)" if args.t_update == "mu"
              else "smallest r·w among converged levels")
    print(f"  best level t  : {best_t:.1e}  ({t_note})")
    print(f"  status        : {info['status']} ({msg})")
    print(f"  total IPOPT it: {total_iter}   wall: {wall:.1f} s")
    print(f"  α* / e^α*     : {alpha:.4f} / {np.exp(alpha):.4f}")
    print(f"  PSNR noisy    : {psnr(u_clean, f):.2f} dB")
    print(
        f"  PSNR recon    : {psnr(u_clean, u):.2f} dB  "
        f"(+{psnr(u_clean, u) - psnr(u_clean, f):.2f})"
    )

    cert = certify_stationarity(prob, x, info, best_t, cert_c=args.cert_c)
    print_certificate(cert, best_n_reg, best_n_rest)
    print("─" * 64)

    if args.save_recon:
        save_triptych(args.save_recon, u_clean, f, u, args.N)
        print(f"  saved clean│noisy│recon triptych → {args.save_recon}")


if __name__ == "__main__":
    main()
