r"""Lifted TV-denoising MPCC — unit-ball dual on a **staggered (cell-centred) grid**.

Fork of ``lifted_mpcc_unitball.py`` (the frozen v1 baseline) that changes ONE
thing: the finite-difference discretization.  v1 uses square N²×N² forward
differences with an artificial Neumann zero-row; this file uses the whiteboard
discretization — **u on the N×N nodes, the dual/lifted fields on the (N−1)²
cell centres** — so the gradient operators are *rectangular* and Kᵀ is the true
discrete adjoint with no zero-row hack.

Everything else (unit-ball dual, Q(α)=e^α in the state row, no slack, explicit
r≥0 / δ≥0 rows, Scholtes ε-continuation, stationarity certificate) is v1's.

============================================================================
The discretization
============================================================================
Nodes carry ``u`` (and the data ``f``, ``u_clean``); cells carry the gradient
and the dual.  For an N×N node grid there are (N−1)×(N−1) cells::

    u₁ ──────── u₂ ──────── u₃          node (i,j) → index i·N + j      (i,j < N)
    │  ∘ q₍₀,₀₎  │  ∘ q₍₀,₁₎ │          cell (I,J) → index I·(N−1) + J  (I,J < N−1)
    u₄ ──────── u₅ ──────── u₆
    │  ∘ q₍₁,₀₎  │  ∘ q₍₁,₁₎ │          m_u = N²   nodes  (u)
    u₇ ──────── u₈ ──────── u₉          m_q = (N−1)²  cells (q, r, δ, θ)

Both whiteboard stencils are implemented, selected by ``--stencil``.  The
**default is ``onesided``** — one difference per cell, both components anchored
at the cell's bottom-right node so that ``q_x`` and ``q_y`` stay co-located (the
polar lift needs them at the same point, which is also why the true MAC
staggering is not an option here)::

    (K_x u)_{I,J} = u_{I+1,J+1} − u_{I+1,J}          K_x = Sh ⊗ D
    (K_y u)_{I,J} = u_{I+1,J+1} − u_{I,J+1}          K_y = D ⊗ Sh

On the 3×3 example above that is exactly the board's

    K_x u = ( u₅−u₄ ,  u₆−u₅ ,  u₈−u₇ ,  u₉−u₈ )
    K_y u = ( u₅−u₂ ,  u₆−u₃ ,  u₈−u₅ ,  u₉−u₆ )

``--stencil averaged`` gives the other board variant, the ½ stencil (the gradient
of the bilinear interpolant at the cell centre, each component averaging the
cell's two parallel edges)::

    (K_x u)_{I,J} = ½[ (u_{I,J+1} − u_{I,J}) + (u_{I+1,J+1} − u_{I+1,J}) ]   K_x = A ⊗ D
    (K_y u)_{I,J} = ½[ (u_{I+1,J} − u_{I,J}) + (u_{I+1,J+1} − u_{I,J+1}) ]   K_y = D ⊗ A

    i.e. on the 3×3 grid   K_x u = ½( u₅−u₄ + u₂−u₁ ,  u₆−u₅ + u₃−u₂ ,  … )

with ``D`` the (N−1)×N difference, ``A`` the two-point average, and ``Sh`` the
"take node J+1" selector.  All are (N−1)² × N².

**Why ``onesided`` is the default.**  Averaging two parallel differences
multiplies the operator symbol by ``cos(ξ⊥/2)``.  That annihilates the Nyquist
mode, so ``ker[K_x;K_y]`` gains a **pure checkerboard** on top of the constants
and ``TV(checkerboard) = 0`` at any α; and it damps ‖K‖ from 2.83 to 2.00, i.e.
uniformly weakens the TV penalty at high frequency — exactly where the noise
lives.  Measured (exact-TV CP optimum, cameraman): averaged reaches +5.56 dB at
N=32 / +5.79 at N=48, one-sided +6.31 / +6.65, v1's square operator +6.46 / +6.88.
So ``onesided`` recovers almost all of v1's denoising power while keeping the
rectangular, no-fictitious-row structure.  Its own defect is far milder: the
single corner node that lies outside every stencil (``ker`` dim 2, the extra mode
a one-node spike) is worth 0.002 dB.

The diagonal/Roberts stencil is **not** a third option — rotating the diagonal
pair back to (x, y) reproduces ``averaged`` *exactly* (isotropic TV is invariant
under rotation in the (g_x, g_y) plane), checkerboard and all.

Consequences vs v1's square operators:

  • **No Neumann zero-row.**  v1 zeroes the last row of the 1-D difference to
    make it square; that row is a fictitious constraint ``(∇u)_last = 0`` and it
    makes Kᵀ *not* the adjoint of a consistent gradient.  Here K is genuinely
    rectangular, ``dim(K_xᵀ) = N² × (N−1)²``, and −Kᵀ is the exact discrete
    divergence with the natural (no-flux) boundary condition built in.
  • **Fields live on different meshes.**  ``u`` has length m_u = N², while
    ``q_x, q_y, r, δ, θ`` have length m_q = (N−1)².  So the variable and
    constraint blocks are no longer all the same size — see below.
  • **The mesh spacing is not applied** (h = 1): K is pure differences, as on the
    board and as in v1, so α remains directly comparable with v1's α.

============================================================================
The bilevel problem (unchanged from v1)
============================================================================
Upper level:   min_α  ½‖u(α) − u_clean‖²  +  ½·reg_alpha·α²
Lower level:   u(α) = argmin_u ½‖u − f‖² + e^α·‖∇u‖_{2,1}

lifted through the primal–dual system with the shared polar angle θ:

    (K_x u)_i = r_i cos θ_i,   (K_y u)_i = r_i sin θ_i      (r_i = ‖∇u_i‖ ≥ 0)
    (q_x)_i   = δ_i cos θ_i,   (q_y)_i   = δ_i sin θ_i      (δ_i = ‖q_i‖ ≥ 0)

with Q(α)=e^α in the **state row** (so the dual ball is the plain box δ ≤ 1 and
the α-free slack w = 1 − δ is redundant and eliminated).

============================================================================
Variables and constraints
============================================================================
    x = [ u | q_x | q_y | r | δ | θ | α ]
        lengths  m_u | m_q | m_q | m_q | m_q | m_q | 1        n = m_u + 5·m_q + 1

    equalities  h(x) = 0
      h1  :  u − f + e^α·(Kxᵀ q_x + Kyᵀ q_y)     length m_u   (state, ← e^α)
      h2x :  Kx u − r·cos θ                       length m_q
      h2y :  Ky u − r·sin θ                       length m_q
      h3x :  q_x − δ·cos θ                        length m_q
      h3y :  q_y − δ·sin θ                        length m_q
    inequalities
      hr  :  r ≥ 0                                length m_q
      hd  :  δ ≥ 0                                length m_q
      comp:  r_i · (1 − δ_i) ≤ t                  length m_q   (Scholtes)

                                            m_con = m_u + 7·m_q

Remaining variable bounds: δ ≤ 1 and the α box.  ``comp`` stays LAST (the driver
treats every row from ``comp`` on as the one-sided r·(1−δ) ≤ t).

============================================================================
Difference from v1 beyond the mesh: consistent α-derivatives
============================================================================
v1's ``objective`` carries ½·reg_alpha·α² but its ``gradient`` has the matching
term commented out (and the exact Hessian's (α,α) block and the certificate's
R_α lack it too) — a deliberate, documented inconsistency kept frozen there.
It is **fixed here** (gradient, ``obj_factor·reg_alpha`` in (α,α), ``reg_alpha·α``
in R_α), and ``--fd-check`` now also validates the objective gradient.  For an
exactly like-for-like A/B against v1, run both with ``--reg-alpha 0``.

Run:   uv run python lifted_mpcc_unitball_staggered.py               # cameraman, N=32
       uv run python lifted_mpcc_unitball_staggered.py --fd-check    # validate derivatives
       uv run python lifted_mpcc_unitball_staggered.py --N 48 --hess-update exact
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
    C-order flatten (node index = row·N + col) matches ``grad_operators``.
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


def grad_operators(N: int, stencil: str = "onesided") -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Staggered cell-centred gradient: nodes → cells, **rectangular**.

    Maps node values (C-order, index = row·N + col, length N²) to cell-centred
    gradient components (index = cellrow·(N−1) + cellcol, length (N−1)²).  Both
    returned operators are ((N−1)² × N²) sparse.  There is **no Neumann
    zero-row**: the operator is genuinely rectangular, so ``−Kᵀ`` is the exact
    discrete divergence carrying the natural no-flux boundary condition.  Unit
    mesh spacing (h = 1).  Two stencils, both from the whiteboard:

    ``onesided`` (default) — one difference per cell, both components anchored at
    the cell's bottom-right node (so ``q_x``/``q_y`` stay co-located, as the polar
    lift requires)::

        (Kx u)_{I,J} = u_{I+1,J+1} − u_{I+1,J}          Kx = Sh ⊗ D
        (Ky u)_{I,J} = u_{I+1,J+1} − u_{I,J+1}          Ky = D ⊗ Sh

    ``averaged`` — the ½ stencil, i.e. the gradient of the bilinear interpolant
    at the cell centre (each component averages the cell's two parallel edges)::

        (Kx u)_{I,J} = ½[(u_{I,J+1} − u_{I,J}) + (u_{I+1,J+1} − u_{I+1,J})]   Kx = A ⊗ D
        (Ky u)_{I,J} = ½[(u_{I+1,J} − u_{I,J}) + (u_{I+1,J+1} − u_{I,J+1})]   Ky = D ⊗ A

    **``onesided`` is the default because ``averaged`` is a measurably weaker TV
    prior.**  Averaging two parallel differences multiplies the operator symbol
    by ``cos(ξ⊥/2)``, which (a) annihilates the Nyquist mode outright — so
    ``ker[Kx;Ky]`` gains a **pure checkerboard** on top of the constants, and
    ``TV(checkerboard) = 0`` at any α — and (b) damps ‖K‖ from 2.83 to 2.00, i.e.
    uniformly weakens the penalty at high frequency, exactly where the noise
    lives.  Measured cost: ≈0.8 dB (see the module docstring).  ``onesided``
    keeps v1's full spectrum; its only defect is that one corner node falls
    outside every stencil (``ker`` dim 2, the extra mode a single-node spike),
    which measures at 0.002 dB.

    Note the diagonal/Roberts stencil is **not** a third option: rotating the
    diagonal pair back to (x, y) reproduces ``averaged`` exactly (isotropic TV is
    invariant under rotation in the (g_x, g_y) plane), checkerboard and all.
    """
    if N < 2:
        raise ValueError("N ≥ 2 required (the staggered grid needs at least one cell)")
    e = np.ones(N - 1)
    # D : R^N → R^{N-1}, one-sided difference (no Neumann row — K is rectangular)
    D = sp.diags([-e, e], [0, 1], shape=(N - 1, N), format="csr")
    if stencil == "onesided":
        # Sh : R^N → R^{N-1}, pick node J+1 (anchors both components at the same node)
        Sh = sp.hstack([sp.csr_matrix((N - 1, 1)), sp.identity(N - 1)], format="csr")
        return sp.kron(Sh, D, format="csr"), sp.kron(D, Sh, format="csr")
    if stencil == "averaged":
        # A : R^N → R^{N-1}, two-point average onto the same cell centres
        A = sp.diags([0.5 * e, 0.5 * e], [0, 1], shape=(N - 1, N), format="csr")
        return sp.kron(A, D, format="csr"), sp.kron(D, A, format="csr")
    raise ValueError(f"unknown stencil {stencil!r} (expected 'onesided' or 'averaged')")


# ---------------------------------------------------------------------------
# The lifted MPCC (unit-ball, staggered grid) as a cyipopt problem object
# ---------------------------------------------------------------------------
class LiftedTVMPCC:
    """cyipopt problem object for the staggered unit-ball lifted TV-MPCC.

    Same formulation as ``lifted_mpcc_unitball.LiftedTVMPCC`` — Q(α)=e^α in the
    state row h1 (a live ∂/∂α column), plain unit box δ ≤ 1, no slack, explicit
    r≥0/δ≥0 rows — but on the staggered mesh: ``u`` on the m_u = N² nodes and
    ``q_x, q_y, r, δ, θ`` on the m_q = (N−1)² cell centres, with rectangular
    ``Kx, Ky : R^{m_u} → R^{m_q}``.  Every block therefore has one of two
    lengths (``self.blen`` records which), and h1 alone is m_u long.

    The Scholtes level ``t`` is not part of this object; it enters only through
    the upper bound on the complementarity rows set by the continuation driver.
    The optional TR (Tikhonov gauge) ridge weight ``eps_theta`` is written onto
    the object per level and perturbs only ``objective``/``gradient`` (+ the θ
    Hessian block under ``--hess-update exact``).
    """

    def __init__(self, f, u_clean, N, reg_alpha=1e-4, stencil="onesided"):
        self.f = np.asarray(f, float)
        self.u_clean = np.asarray(u_clean, float)
        self.N = N
        self.stencil = stencil
        self.m_u = m_u = N * N            # nodes  : u, f, u_clean, h1
        self.m_q = m_q = (N - 1) * (N - 1)  # cells  : q, r, δ, θ and h2/h3/hr/hd/comp
        self.reg_alpha = reg_alpha

        # TR (Tikhonov gauge) ridge ½·eps_theta·‖θ − θ_ref‖² — the D1 fix for the
        # angle-gauge indeterminacy (θ_i undetermined where r_i = δ_i = 0). Off by
        # default; the driver sets eps_theta = c_θ·t per level so the bias → 0.
        self.eps_theta = 0.0
        self.theta_ref = np.zeros(m_q)

        # Per-solve telemetry, (re)set by the driver / intermediate callback.
        self.n_iter = 0
        self.n_reg = 0   # iters with δ_w > 0 (Hessian inertia corr.; see intermediate)
        self.n_rest = 0  # iters in IPOPT's restoration phase (alg_mod == 1)
        self.inf_pr = float("nan")
        self.inf_du = float("nan")
        self._interrupt = False  # set by the driver's SIGINT handler (see intermediate)

        self.Kx, self.Ky = grad_operators(N, stencil)  # (m_q × m_u)
        self.KxT, self.KyT = self.Kx.T.tocsr(), self.Ky.T.tocsr()  # (m_u × m_q)

        # Column offsets of each variable block, and the scalar α index. u is the
        # only node-length block; everything else is cell-length. No slack: δ
        # carries the unit-ball radius directly and the comp row is written on δ.
        self.off = {
            "u": 0, "qx": m_u, "qy": m_u + m_q, "r": m_u + 2 * m_q,
            "delta": m_u + 3 * m_q, "theta": m_u + 4 * m_q, "alpha": m_u + 5 * m_q,
        }
        # Block lengths, so ``_blk`` can slice either mesh.
        self.blen = {
            "u": m_u, "qx": m_q, "qy": m_q, "r": m_q,
            "delta": m_q, "theta": m_q,
        }
        # Row offsets. h1 is node-length (m_u); the rest are cell-length (m_q).
        # The non-negativities r ≥ 0 and δ ≥ 0 are explicit inequality rows
        # (hr, hd) rather than variable box bounds; only δ ≤ 1 and the α box
        # remain as bounds. ``comp`` stays LAST (the driver treats every row from
        # ``comp`` on as the one-sided r·(1−δ) ≤ t).
        self.roff = {
            "h1": 0, "h2x": m_u, "h2y": m_u + m_q, "h3x": m_u + 2 * m_q,
            "h3y": m_u + 3 * m_q, "hr": m_u + 4 * m_q, "hd": m_u + 5 * m_q,
            "comp": m_u + 6 * m_q,
        }
        self.rlen = {
            "h1": m_u, "h2x": m_q, "h2y": m_q, "h3x": m_q,
            "h3y": m_q, "hr": m_q, "hd": m_q, "comp": m_q,
        }
        self.n = m_u + 5 * m_q + 1
        self.m_con = m_u + 7 * m_q
        self._rows, self._cols = self._build_structure()
        self._hrows, self._hcols = self._build_hess_structure()

    # ---- slicing helpers --------------------------------------------------
    def _blk(self, x, name):
        s = self.off[name]
        return x[s : s + self.blen[name]]

    def _rblk(self, v, name):
        """Slice a constraint-length vector (multipliers, residuals) by row block."""
        s = self.roff[name]
        return v[s : s + self.rlen[name]]

    def _divq(self, x):
        """The divergence Kxᵀ q_x + Kyᵀ q_y (length m_u) — appears in h1 and its
        α-derivatives.  ∂h1/∂α = e^α·div q, ∂²h1/∂α² carries the same vector."""
        return self.KxT @ self._blk(x, "qx") + self.KyT @ self._blk(x, "qy")

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
        g[self.off["u"] : self.off["u"] + self.m_u] = self._blk(x, "u") - self.u_clean
        # Consistent with ``objective`` (v1 leaves this term out — see module docstring).
        g[self.off["alpha"]] = self.reg_alpha * x[self.off["alpha"]]
        if self.eps_theta:  # ∂/∂θ of the TR ridge = eps_theta·(θ − θ_ref)
            s = self.off["theta"]
            g[s : s + self.m_q] = self.eps_theta * (self._blk(x, "theta") - self.theta_ref)
        return g

    # ---- constraints ------------------------------------------------------
    def constraints(self, x):
        u, qx, qy = self._blk(x, "u"), self._blk(x, "qx"), self._blk(x, "qy")
        r, delta, theta = self._blk(x, "r"), self._blk(x, "delta"), self._blk(x, "theta")
        a = x[self.off["alpha"]]
        c, s = np.cos(theta), np.sin(theta)
        ea = np.exp(a)
        return np.concatenate(
            [
                u - self.f + ea * (self.KxT @ qx + self.KyT @ qy),  # h1  (m_u, ← e^α)
                self.Kx @ u - r * c,   # h2x
                self.Ky @ u - r * s,   # h2y
                qx - delta * c,        # h3x
                qy - delta * s,        # h3y
                r,                     # hr : r ≥ 0      (explicit inequality)
                delta,                 # hd : δ ≥ 0      (explicit inequality)
                r * (1.0 - delta),     # comp: r·(1 − δ) ≤ t (Scholtes)
            ]
        )

    # ---- Jacobian ---------------------------------------------------------
    def _build_structure(self):
        """Precompute the (row, col) index arrays of every Jacobian nonzero.

        The 20 pieces below are assembled in the SAME order by ``jacobian``;
        constant blocks (Kx, Ky, identities) have their values cached here. Note
        piece 4 is the dense ∂h1/∂α column (length m_u — the whole α-coupling of
        the KKT system); the comp row is written on δ (∂comp/∂r = 1 − δ,
        ∂comp/∂δ = −r); and the explicit non-negativity rows hr, hd are identity
        diagonals (∂hr/∂r = ∂hd/∂δ = 1, pieces 19–20).

        Unlike v1, the Kx/Ky blocks are rectangular (m_q × m_u) and the KxT/KyT
        blocks are (m_u × m_q), so the diagonals come in two lengths.
        """
        m_u, m_q, off, ro = self.m_u, self.m_q, self.off, self.roff
        iu, iq = np.arange(m_u), np.arange(m_q)

        def qdiag(roff, coff):  # cell-length diagonal (m_q)
            return roff + iq, coff + iq

        Kx, Ky = self.Kx.tocoo(), self.Ky.tocoo()
        KxT, KyT = self.KxT.tocoo(), self.KyT.tocoo()
        self._ones_u = np.ones(m_u)
        self._ones_q = np.ones(m_q)
        self._Kx, self._Ky = Kx.data, Ky.data
        self._KxT, self._KyT = KxT.data, KyT.data

        pieces = [
            (ro["h1"] + iu, off["u"] + iu),                 # 1  h1/∂u   = I  (m_u)
            (ro["h1"] + KxT.row, off["qx"] + KxT.col),      # 2  h1/∂qx  = e^α·Kxᵀ
            (ro["h1"] + KyT.row, off["qy"] + KyT.col),      # 3  h1/∂qy  = e^α·Kyᵀ
            (ro["h1"] + iu, np.full(m_u, off["alpha"])),    # 4  h1/∂α   = e^α·div q
            (ro["h2x"] + Kx.row, off["u"] + Kx.col),        # 5  h2x/∂u  = Kx
            qdiag(ro["h2x"], off["r"]),                     # 6  h2x/∂r  = -cosθ
            qdiag(ro["h2x"], off["theta"]),                 # 7  h2x/∂θ  =  r sinθ
            (ro["h2y"] + Ky.row, off["u"] + Ky.col),        # 8  h2y/∂u  = Ky
            qdiag(ro["h2y"], off["r"]),                     # 9  h2y/∂r  = -sinθ
            qdiag(ro["h2y"], off["theta"]),                 # 10 h2y/∂θ  = -r cosθ
            qdiag(ro["h3x"], off["qx"]),                    # 11 h3x/∂qx = I
            qdiag(ro["h3x"], off["delta"]),                 # 12 h3x/∂δ  = -cosθ
            qdiag(ro["h3x"], off["theta"]),                 # 13 h3x/∂θ  =  δ sinθ
            qdiag(ro["h3y"], off["qy"]),                    # 14 h3y/∂qy = I
            qdiag(ro["h3y"], off["delta"]),                 # 15 h3y/∂δ  = -sinθ
            qdiag(ro["h3y"], off["theta"]),                 # 16 h3y/∂θ  = -δ cosθ
            qdiag(ro["comp"], off["r"]),                    # 17 comp/∂r = 1 − δ
            qdiag(ro["comp"], off["delta"]),                # 18 comp/∂δ = -r
            qdiag(ro["hr"], off["r"]),                      # 19 hr/∂r  = 1
            qdiag(ro["hd"], off["delta"]),                  # 20 hd/∂δ  = 1
        ]
        rows = np.concatenate([p[0] for p in pieces]).astype(np.int64)
        cols = np.concatenate([p[1] for p in pieces]).astype(np.int64)
        return rows, cols

    def jacobianstructure(self):
        return self._rows, self._cols

    def jacobian(self, x):
        r, delta, theta = self._blk(x, "r"), self._blk(x, "delta"), self._blk(x, "theta")
        a = x[self.off["alpha"]]
        c, s = np.cos(theta), np.sin(theta)
        ea = np.exp(a)
        divq = self._divq(x)
        return np.concatenate(
            [
                self._ones_u,      # 1  h1/∂u
                ea * self._KxT,    # 2  h1/∂qx  = e^α·Kxᵀ
                ea * self._KyT,    # 3  h1/∂qy  = e^α·Kyᵀ
                ea * divq,         # 4  h1/∂α   = e^α·div q
                self._Kx,          # 5  h2x/∂u
                -c,                # 6  h2x/∂r
                r * s,             # 7  h2x/∂θ
                self._Ky,          # 8  h2y/∂u
                -s,                # 9  h2y/∂r
                -r * c,            # 10 h2y/∂θ
                self._ones_q,      # 11 h3x/∂qx
                -c,                # 12 h3x/∂δ
                delta * s,         # 13 h3x/∂θ
                self._ones_q,      # 14 h3y/∂qy
                -s,                # 15 h3y/∂δ
                -delta * c,        # 16 h3y/∂θ
                1.0 - delta,       # 17 comp/∂r = 1 − δ
                -r,                # 18 comp/∂δ = -r
                self._ones_q,      # 19 hr/∂r  = 1
                self._ones_q,      # 20 hd/∂δ  = 1
            ]
        )

    # ---- exact Lagrangian Hessian (lower triangle) ------------------------
    # H = σ_f·∇²J + Σ_k λ_k ∇²c_k. Objective → I on u, reg_alpha on (α,α)
    # (+ eps_theta·I on θ under TR). Nonlinear constraints and their Hessians:
    #   h1 : bilinear in (α, q).  ∂²/∂qx∂α = e^α·Kx·λ_h1 (length m_q, since Kx is
    #        m_q×m_u and λ_h1 is m_u), likewise for qy; ∂²/∂α² = e^α·⟨λ_h1, div q⟩.
    #   h2x: ∂²/∂r∂θ= sinθ, ∂²/∂θ²= r cosθ ;  h2y: ∂²/∂r∂θ=−cosθ, ∂²/∂θ²= r sinθ
    #   h3x: ∂²/∂δ∂θ= sinθ, ∂²/∂θ²= δ cosθ ;  h3y: ∂²/∂δ∂θ=−cosθ, ∂²/∂θ²= δ sinθ
    #   comp: r·(1 − δ), ∂²/∂r∂δ = −1 → (δ,r) cross = −ξ (indefinite, eigenvalues
    #        ±ξ; lower triangle since off[δ] > off[r]).
    def _build_hess_structure(self):
        m_u, m_q, off = self.m_u, self.m_q, self.off
        iu, iq = np.arange(m_u), np.arange(m_q)
        alpha = off["alpha"]
        rows = np.concatenate(
            [
                off["u"] + iu,        # (u,u)   diag           (m_u)
                off["theta"] + iq,    # (θ,r)                  (m_q)
                off["theta"] + iq,    # (θ,δ)
                off["theta"] + iq,    # (θ,θ)   diag
                off["delta"] + iq,    # (δ,r)   comp cross (indefinite)
                np.full(m_q, alpha),  # (α,qx)  h1 bilinear cross
                np.full(m_q, alpha),  # (α,qy)  h1 bilinear cross
                [alpha],              # (α,α)   h1 curvature + reg_alpha ridge
            ]
        )
        cols = np.concatenate(
            [
                off["u"] + iu, off["r"] + iq, off["delta"] + iq,
                off["theta"] + iq, off["r"] + iq,
                off["qx"] + iq, off["qy"] + iq, [alpha],
            ]
        )
        return rows.astype(np.int64), cols.astype(np.int64)

    def hessianstructure(self):
        return self._hrows, self._hcols

    def hessian(self, x, lagrange, obj_factor):
        off = self.off
        r, delta, theta = self._blk(x, "r"), self._blk(x, "delta"), self._blk(x, "theta")
        a = x[off["alpha"]]
        ea = np.exp(a)
        c, s = np.cos(theta), np.sin(theta)
        l1 = self._rblk(lagrange, "h1")
        l2x, l2y = self._rblk(lagrange, "h2x"), self._rblk(lagrange, "h2y")
        l3x, l3y = self._rblk(lagrange, "h3x"), self._rblk(lagrange, "h3y")
        xi = self._rblk(lagrange, "comp")
        H_uu = obj_factor * self._ones_u
        H_tr = l2x * s - l2y * c                                      # (θ,r)
        H_td = l3x * s - l3y * c                                      # (θ,δ)
        H_tt = r * (l2x * c + l2y * s) + delta * (l3x * c + l3y * s)  # (θ,θ)
        H_tt = H_tt + obj_factor * self.eps_theta                     # TR ridge (if on)
        H_dr = -xi                                                    # (δ,r) comp cross
        # h1 bilinear curvature: ∂²(λ_h1·h1) couples α with qx, qy and itself.
        H_aqx = ea * (self.Kx @ l1)                                   # (α,qx)  (m_q)
        H_aqy = ea * (self.Ky @ l1)                                   # (α,qy)  (m_q)
        H_aa = np.array([ea * float(np.dot(l1, self._divq(x)))
                         + obj_factor * self.reg_alpha])              # (α,α)
        return np.concatenate([H_uu, H_tr, H_td, H_tt, H_dr, H_aqx, H_aqy, H_aa])

    # ---- per-NLP-solve telemetry (reset by the driver before each solve) ---
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.n_iter = iter_count
        self.inf_pr = inf_pr
        self.inf_du = inf_du  # final (scaled) dual infeasibility
        # regularization_size is IPOPT's δ_w (primal/Hessian inertia correction,
        # 'lg(rg)'). Incomplete D1 detector under L-BFGS (PD by construction); the
        # dual/Jacobian δ_c is not exposed by IPOPT. Use θ-drift + restoration too.
        if regularization_size > 0.0:
            self.n_reg += 1
        if alg_mod == 1:
            self.n_rest += 1
        # Cooperative interruption: the driver's SIGINT handler sets ``_interrupt``;
        # returning False here asks IPOPT to stop THIS solve at the current iterate
        # (status = User_Requested_Stop) instead of raising a KeyboardInterrupt
        # mid-C-call. The driver then reports the last completed level.
        return not self._interrupt


# ---------------------------------------------------------------------------
# Initial point and bounds
# ---------------------------------------------------------------------------
def chambolle_pock_rof(f, Kx, Ky, lam, n_iter=3000, tol=1e-9):
    """Chambolle–Pock for ROF denoising  min_u ½‖u−f‖² + lam·‖∇u‖_{2,1}.

    Returns ``(u, qx, qy)`` with ``u`` on the nodes (length m_u) and the dual on
    the cells (length m_q), per-cell on the **unit ball** ‖(qxᵢ,qyᵢ)‖ ≤ 1.  At
    convergence  u = f − lam·(Kxᵀ qx + Kyᵀ qy)  with the complementarity
    r ⊥ (1 − δ) — i.e. *exactly* the unit-ball MPCC lower-level system at
    e^α = lam.  So a CP solve is a consistent, approximately-feasible warm start
    for the whole lifted problem (h1 exact, comp ≈ 0).

    The step size uses ‖K‖ ≤ √8, which still holds for the averaged staggered
    operator (‖A ⊗ D‖ ≤ ‖A‖·‖D‖ ≤ 1·2, so ‖K‖² ≤ ‖Kx‖² + ‖Ky‖² ≤ 8) — it is in
    fact conservative here, since averaging strictly shrinks the spectrum.
    """
    KxT, KyT = Kx.T.tocsr(), Ky.T.tocsr()
    m_q = Kx.shape[0]
    step = 0.99 / np.sqrt(8.0)          # τ·lam = σ·lam = step (‖∇‖ ≤ √8, 2D)
    tau = step / lam
    u = f.copy()
    ubar = f.copy()
    qx = np.zeros(m_q)
    qy = np.zeros(m_q)
    for _ in range(n_iter):
        qx = qx + step * (Kx @ ubar)   # dual ascent (σ·lam = step) …
        qy = qy + step * (Ky @ ubar)
        nrm = np.maximum(1.0, np.hypot(qx, qy))  # … + projection onto unit ball
        qx /= nrm
        qy /= nrm
        u_new = (tau * f + u - step * (KxT @ qx + KyT @ qy)) / (tau + 1.0)  # primal prox
        ubar = 2.0 * u_new - u         # θ = 1 extrapolation
        if np.linalg.norm(u_new - u) <= tol * (np.linalg.norm(u_new) + 1e-30):
            u = u_new
            break
        u = u_new
    return u, qx, qy


def cp_scan(prob: LiftedTVMPCC, alpha_center: float, half_width: float = 2.0, n: int = 9):
    """Coarse Chambolle–Pock pre-scan of the bilevel loss
    ``L(α) = ½‖u_CP(e^α) − u_clean‖²`` on ``n`` log-spaced weights spanning
    ``alpha_center ± half_width``; returns the argmin ``(α*, u, qx, qy)`` — the CP
    lower-level solution to warm-start from.

    Legitimate because CP is an *exact* lower-level solve and ``u_clean`` is the
    training target, so ``L(α)`` here is the true (unimodal) bilevel objective.
    This seeds the MPCC essentially at the optimum's basin **instance-independently**
    — no reliance on a good hand-picked α₀ — at the cost of ``n`` cheap CP solves.
    """
    alphas = alpha_center + np.linspace(-half_width, half_width, n)
    best = None
    for a in alphas:
        u, qx, qy = chambolle_pock_rof(prob.f, prob.Kx, prob.Ky, float(np.exp(a)))
        loss = 0.5 * float(np.sum((u - prob.u_clean) ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(a), u, qx, qy)
    return best[1], best[2], best[3], best[4]


def initial_point(prob: LiftedTVMPCC, alpha0: float, init: str = "cp") -> np.ndarray:
    """Warm start for the lifted MPCC.

    ``cold``: u = f, q = 0. With u = f we read off the polar cell gradient (r, θ);
    q = 0 gives δ = 0, so the N-side 1 − δ = 1. h1 holds (q = 0 kills the
    e^α·div q term); only the complementarity r·(1 − δ) = r is violated. This sits
    on the *no-regularization* manifold and biases the continuation toward the
    spurious small-α (near-noisy) branch, especially in large dimension.

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
    m_u, m_q, off = prob.m_u, prob.m_q, prob.off
    x = np.zeros(prob.n)
    if init in ("cp", "cp-scan"):
        if init == "cp-scan":
            alpha0, u, qx, qy = cp_scan(prob, alpha0)   # α* = argmin L(α) on the grid
        else:
            u, qx, qy = chambolle_pock_rof(prob.f, prob.Kx, prob.Ky, float(np.exp(alpha0)))
        x[off["u"] : off["u"] + m_u] = u
        x[off["qx"] : off["qx"] + m_q] = qx
        x[off["qy"] : off["qy"] + m_q] = qy
        x[off["r"] : off["r"] + m_q] = np.hypot(prob.Kx @ u, prob.Ky @ u)
        x[off["delta"] : off["delta"] + m_q] = np.hypot(qx, qy)
        x[off["theta"] : off["theta"] + m_q] = np.arctan2(qy, qx)  # dual angle → h3 exact
        x[off["alpha"]] = alpha0
        return x
    # cold: u = f, q = 0
    x[off["u"] : off["u"] + m_u] = prob.f
    gx, gy = prob.Kx @ prob.f, prob.Ky @ prob.f
    x[off["r"] : off["r"] + m_q] = np.hypot(gx, gy)
    x[off["theta"] : off["theta"] + m_q] = np.arctan2(gy, gx)
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
    m_q, off, n = prob.m_q, prob.off, prob.n
    xl = np.full(n, -2.0e19)
    xu = np.full(n, 2.0e19)
    xu[off["delta"] : off["delta"] + m_q] = 1.0  # δ ≤ 1 (unit-ball radius)
    xl[off["alpha"]] = alpha_lo
    xu[off["alpha"]] = alpha_hi
    return xl, xu


# ---------------------------------------------------------------------------
# Scholtes ε-continuation driver
# ---------------------------------------------------------------------------
def solve_scholtes(
    prob, x0, schedule, *, linear_solver, tol=1e-8, tol_factor=0.1, max_iter=1500,
    dual_warmstart=False, c_theta=0.0, hess_update="bfgs", cert_c=3.0,
    print_level=0, verbose=True,
):
    """Solve a sequence of relaxed NLPs with r·w ≤ t for t ↓ along ``schedule``.

    * **Tolerance coupling** — level ``t`` is solved only to
      ``max(tol, tol_factor·t)`` (the Scholtes path is itself O(√t)-accurate).
    * **Primal + (optional) dual warm-start** — carry the previous iterate; with
      ``dual_warmstart`` also pass the previous multipliers and arm
      ``warm_start_init_point`` (pympcc's default; ~1.8× fewer iters).
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
    m_q, off = prob.m_q, prob.off
    xl, xu = bounds(prob)
    cl = np.full(prob.m_con, 0.0)
    cu = np.full(prob.m_con, 0.0)
    # hr, hd are the explicit non-negativities 0 ≤ r, δ ≤ +∞.
    cu[prob.roff["hr"] : prob.roff["comp"]] = 2.0e19
    cl[prob.roff["comp"] :] = -2.0e19  # complementarity rows are one-sided: r·(1−δ) ≤ t

    use_ma57 = (linear_solver == "ma57") and os.path.exists(HSLLIB)
    if linear_solver == "ma57" and not use_ma57:
        print(f"  [warn] {HSLLIB} not found → falling back to MUMPS")
        linear_solver = "mumps"

    if verbose:
        print(
            f"  {'t':>9} {'tol':>8} {'iters':>6} {'δw':>4} {'rest':>4} "
            f"{'status':>7} {'comp_res':>10} {'max|ξ|':>10} {'α':>8} {'obj':>11}"
        )

    x = x0.copy()
    warm = None
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

    try:
        for t in schedule:
            prob.eps_theta = c_theta * t  # TR gauge ridge weight (0 = off)
            tol_t = max(tol, tol_factor * t)
            cu_t = cu.copy()
            cu_t[prob.roff["comp"] :] = t  # r·w ≤ t

            nlp = cyipopt.Problem(
                n=prob.n, m=prob.m_con, problem_obj=prob, lb=xl, ub=xu, cl=cl, cu=cu_t
            )
            nlp.add_option("sb", "yes")
            nlp.add_option("print_level", int(print_level))
            nlp.add_option("tol", tol_t)
            nlp.add_option("acceptable_tol", max(tol, 10 * tol_t))
            nlp.add_option("acceptable_iter", 10)
            nlp.add_option("max_iter", max_iter)
            if hess_update == "exact":
                nlp.add_option("hessian_approximation", "exact")  # prob.hessian callback
            else:
                nlp.add_option("hessian_approximation", "limited-memory")
                nlp.add_option("limited_memory_max_history", 25)
                nlp.add_option("limited_memory_update_type", hess_update)  # bfgs / sr1
            nlp.add_option("mu_strategy", "monotone")
            nlp.add_option("linear_solver", linear_solver)
            if use_ma57:
                nlp.add_option("hsllib", HSLLIB)
            if dual_warmstart and warm is not None:
                nlp.add_option("warm_start_init_point", "yes")

            prob.n_iter = 0
            prob.n_reg = 0
            prob.n_rest = 0
            if dual_warmstart and warm is not None:
                x, info = nlp.solve(x, lagrange=warm[0], zl=warm[1], zu=warm[2])
            else:
                x, info = nlp.solve(x)
            nlp.close()
            _mg = info.get("mult_g")
            if _mg is not None and len(_mg):
                warm = (_mg, info.get("mult_x_L"), info.get("mult_x_U"))
            total_iter += prob.n_iter

            r = x[off["r"] : off["r"] + m_q]
            w = 1.0 - x[off["delta"] : off["delta"] + m_q]  # N-side of the comp: 1 − δ
            comp_res = float(np.max(r * w)) if m_q else 0.0
            a_k = float(x[off["alpha"]])
            xi_max = (
                float(np.max(np.abs(_mg[prob.roff["comp"] :])))
                if (_mg is not None and len(_mg)) else float("nan")
            )
            if verbose:
                print(
                    f"  {t:>9.1e} {tol_t:>8.1e} {prob.n_iter:>6d} {prob.n_reg:>4d} "
                    f"{prob.n_rest:>4d} {info['status']:>7d} {comp_res:>10.2e} "
                    f"{xi_max:>10.2e} {a_k:>8.3f} {info['obj_val']:>11.4e}"
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
        if _sig_installed:
            signal.signal(signal.SIGINT, _old_sigint)  # restore the prior handler

    if best is None:  # no level converged (first-level failure or early interrupt)
        best = (comp_res, x, info, t, prob.n_reg, prob.inf_du, prob.n_rest)
    prob.inf_du = best[5]  # expose the reported level's dual infeasibility to certify
    return best[1], best[2], best[3], total_iter, best[4], best[6]


# ---------------------------------------------------------------------------
def psnr(clean, recon, data_range=1.0):
    mse = float(np.mean((clean - recon) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)


def save_triptych(path, u_clean, f, u, N):
    """Write a clean │ noisy │ recon side-by-side PNG (each panel N×N, 1px gap).

    All three are node fields, so they reshape to the full N×N node grid.
    """
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
    ``M := r ≥ 0``, ``N := 1 − δ ≥ 0`` — both **cell** quantities (length m_q).
    Write ``w := 1 − δ`` for the N-side; its bound multiplier is ``z_w := z_{δ,hi}``
    (the multiplier of the box ``δ ≤ 1``).  IPOPT returns the relaxed-NLP
    multipliers; the Scholtes back-translation recovers the candidate MPCC
    multipliers for M/N::

        γ_i = z_r,i − ξ_i · w_i          (multiplier of  M = r)
        ν_i = z_w,i − ξ_i · r_i          (multiplier of  N = 1 − δ)

    with ``z_r`` the multiplier of ``r ≥ 0`` (here the explicit inequality row
    ``hr``, so ``z_r = −λ_hr`` from ``mult_g``), ``z_w = z_{δ,hi}`` that of the box
    ``δ ≤ 1``, and ``ξ ≥ 0`` the multiplier of ``r·(1 − δ) ≤ t``.  The δ ≥ 0
    multiplier ``μ`` likewise comes from the ``hd`` row (``μ = −λ_hd``).

    The stationarity residual mixes the two meshes: ``λ_h1`` is node-length, so
    ``Kx·λ_h1`` (cell-length) appears in the q-rows and ``Kxᵀ·λ_h2x``
    (node-length) in the u-row::

        R_u  = (u − u_clean) + λ_h1 + Kxᵀλ_h2x + Kyᵀλ_h2y        (m_u)
        R_qx = e^α·(Kx·λ_h1) + λ_h3x                             (m_q)
        R_qy = e^α·(Ky·λ_h1) + λ_h3y                             (m_q)
        R_α  = e^α·⟨λ_h1, div q⟩ + reg_alpha·α − z_{α,lo} + z_{α,hi}

    and the index-set corner cap for w, since ``w = 1 − δ ≤ 1``:

        ε_r = min(c√t, ½·max r),   ε_w = min(c√t, ½)
        𝓐 = {r ≤ ε_r, w > ε_w}   𝓘 = {r > ε_r, w ≤ ε_w}   𝓑 = {r ≤ ε_r, w ≤ ε_w}
        𝓝 = {r > ε_r, w > ε_w}  (unresolved).

    The biactive sign test on 𝓑 names the type (W / C / M / S). Returns a dict of
    everything printed by :func:`print_certificate`.
    """
    m_q, off = prob.m_q, prob.off
    ro = prob.roff
    eps = cert_c * np.sqrt(max(t, 0.0))

    mg = np.asarray(info["mult_g"], float)
    zL = np.asarray(info["mult_x_L"], float)
    zU = np.asarray(info["mult_x_U"], float)

    lam_h1 = prob._rblk(mg, "h1")                                     # node-length
    lam_h2x, lam_h2y = prob._rblk(mg, "h2x"), prob._rblk(mg, "h2y")   # cell-length
    lam_h3x, lam_h3y = prob._rblk(mg, "h3x"), prob._rblk(mg, "h3y")
    xi = prob._rblk(mg, "comp")  # ξ ≥ 0
    # r ≥ 0 and δ ≥ 0 are explicit inequality rows (hr, hd), so their
    # non-negativity multipliers come from mult_g, not the bound arrays. For a
    # lower-bounded (g ≥ 0) row IPOPT returns λ ≤ 0 entering ∇L as +λ·∂g, so the
    # bound-equivalent multipliers (z ≥ 0) are the negatives:
    #   z_r = −λ_hr  (mult of r ≥ 0),   z_δ = −λ_hd  (mult of δ ≥ 0).
    z_r = -prob._rblk(mg, "hr")
    z_delta = -prob._rblk(mg, "hd")

    z_alpha_lo, z_alpha_hi = zL[off["alpha"]], zU[off["alpha"]]

    r = prob._blk(x, "r")
    delta, theta = prob._blk(x, "delta"), prob._blk(x, "theta")
    # N-side value w = 1 − δ and its bound multiplier z_w = z_{δ,hi} (the
    # multiplier of the box δ ≤ 1, i.e. of 1 − δ ≥ 0).
    w = 1.0 - delta
    z_w = zU[off["delta"] : off["delta"] + m_q]
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
    R_qx = ea * (Kx @ lam_h1) + lam_h3x          # ← e^α from h1 (node → cell)
    R_qy = ea * (Ky @ lam_h1) + lam_h3y          # ← e^α from h1
    R_r = -c * lam_h2x - s * lam_h2y - gamma
    # δ-stationarity: the comp/δ term and the δ ≤ 1 bound enter through ν
    # (= z_{δ,hi} − ξ·r); μ = z_{δ,lo} is the δ ≥ 0 multiplier.
    R_delta = -c * lam_h3x - s * lam_h3y - mu + nu
    R_theta = r * s * lam_h2x - r * c * lam_h2y + delta * s * lam_h3x - delta * c * lam_h3y
    R_alpha = (ea * float(np.dot(lam_h1, divq)) + prob.reg_alpha * a_val
               - z_alpha_lo + z_alpha_hi)        # ← from h1 + the reg-α ridge
    stat_res = max(
        float(np.max(np.abs(b))) if b.size else 0.0
        for b in (R_u, R_qx, R_qy, R_r, R_delta, R_theta, np.array([R_alpha]))
    )

    # --- biactive sign classification (per-cell, with a sign tolerance) -----
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

    # α-at-bound test. v1 tests the bound multipliers alone against τ = sign_tol·
    # scale; that is a false positive here. IPOPT leaves O(μ) barrier residuals in
    # BOTH z_{α,lo} and z_{α,hi} even for a wide-open box, and because this
    # formulation's multipliers are well scaled (‖(γ,ν)‖∞ ≈ 0.3 rather than v1's
    # ≈ 200) the scale-relative τ no longer masks them. Require the iterate to
    # actually sit at the bound as well — a genuinely active box shows one large
    # multiplier and the other ≈ 0, not two comparable residuals.
    a_lo, a_hi = bounds(prob)[0][off["alpha"]], bounds(prob)[1][off["alpha"]]
    a_span = max(1.0, a_hi - a_lo)
    at_lo = (a_val - a_lo) <= 1e-6 * a_span
    at_hi = (a_hi - a_val) <= 1e-6 * a_span
    alpha_active = bool((at_lo and z_alpha_lo > tau) or (at_hi and z_alpha_hi > tau))

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
        "alpha_active": alpha_active, "tau": tau,
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
# Finite-difference validation of the objective gradient, Jacobian, Hessian
# ---------------------------------------------------------------------------
def fd_check(prob, x, seed=0):
    """Compare the analytic derivatives against central finite differences.

    Guards the rectangular-mesh blocks (the two-length diagonals, Kx/Kxᵀ shapes,
    the node-length ∂h1/∂α column, and the cell-length (α,q) Hessian crosses) as
    well as the α–q curvature. Unlike v1's version this ALSO differences the
    objective, so the reg-α gradient/Hessian terms are validated.

    Returns ``(grad_err, jac_err, hess_err)``.
    """
    rng = np.random.default_rng(seed)
    x = x.astype(float).copy()
    n, mcon = prob.n, prob.m_con
    h = 1e-6

    # Objective gradient.
    g = prob.gradient(x)
    gfd = np.zeros(n)
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        gfd[j] = (prob.objective(xp) - prob.objective(xm)) / (2 * h)
    grad_err = float(np.max(np.abs(g - gfd)))

    # Dense analytic Jacobian from the sparse (row, col, val) triplet.
    rows, cols = prob.jacobianstructure()
    vals = prob.jacobian(x)
    J = np.zeros((mcon, n))
    J[rows, cols] += vals  # += in case a structural (row,col) repeats
    # Central-difference Jacobian.
    Jfd = np.zeros((mcon, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        Jfd[:, j] = (prob.constraints(xp) - prob.constraints(xm)) / (2 * h)
    jac_err = float(np.max(np.abs(J - Jfd)))

    # Exact Hessian of the Lagrangian at a random (obj_factor, λ).
    obj_factor = float(rng.uniform(0.5, 1.5))
    lam = rng.standard_normal(mcon)
    hrows, hcols = prob.hessianstructure()
    hvals = prob.hessian(x, lam, obj_factor)
    H = np.zeros((n, n))
    H[hrows, hcols] += hvals  # lower triangle
    H = H + H.T - np.diag(np.diag(H))  # symmetrize

    Hfd = np.zeros((n, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        Hfd[:, j] = (_lag_grad_dense(prob, xp, lam, obj_factor)
                     - _lag_grad_dense(prob, xm, lam, obj_factor)) / (2 * h)
    hess_err = float(np.max(np.abs(H - Hfd)))
    return grad_err, jac_err, hess_err


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
        description="Lifted TV-MPCC denoising — unit-ball dual, staggered "
                    "(cell-centred) discretization with rectangular Kx, Ky."
    )
    p.add_argument("--data", default=DEFAULT_IMAGE,
                   help="natural image file (PNG/JPEG/…); default = bundled cameraman")
    p.add_argument("--phantom", action="store_true",
                   help="use the synthetic piecewise-constant phantom instead of --data")
    p.add_argument("--N", type=int, default=32,
                   help="image side length in NODES (u lives on N², the dual/lifted "
                        "fields on the (N−1)² cells: N²+5(N−1)²+1 vars / "
                        "N²+7(N−1)² rows)")
    p.add_argument("--sigma", type=float, default=0.1, help="Gaussian noise std "
                   "(0.1 default: TV meaningfully denoises the natural image here)")
    p.add_argument("--stencil", default="onesided", choices=["onesided", "averaged"],
                   help="cell gradient stencil: 'onesided' (default; one difference "
                        "per cell anchored at a common node — keeps v1's full "
                        "spectrum) or 'averaged' (the ½ bilinear stencil — carries a "
                        "checkerboard kernel and damps ‖K‖ 2.83→2.00, costing ≈0.8 dB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha0", type=float, default=None,
                   help="initial log-strength α (default: noise-aware log(0.7·σ), "
                   "calibrated to v1's natural-image TV optimum; the averaged "
                   "stencil shifts this slightly — use --init cp-scan if unsure. "
                   "Also centres the cp-scan window)")
    p.add_argument("--init", default="cp", choices=["cold", "cp", "cp-scan"],
                   help="warm start: 'cp' (default; Chambolle–Pock ROF denoise at "
                   "e^α₀ → lower-level primal–dual point, in the good-denoising "
                   "basin); 'cp-scan' (pick the CP weight by argmin L(α) over a "
                   "coarse grid around α₀ — instance-robust); or 'cold' (u=f, q=0; "
                   "drifts to the spurious near-noisy branch — kept for A/B)")
    p.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha",
                   help="½·reg_alpha·α² ridge on the upper level (consistent in the "
                        "gradient/Hessian/certificate here, unlike v1; use 0 for an "
                        "exactly like-for-like A/B against v1)")
    p.add_argument("--c-theta", type=float, default=0.0, dest="c_theta",
                   help="TR (Tikhonov gauge) strength c_θ: adds ½·(c_θ·t)·‖θ−θ_ref‖² "
                   "to the objective each level (D1 fix; weight ∝ t). 0 = off.")
    p.add_argument("--t0", type=float, default=1.0, help="first Scholtes level")
    p.add_argument("--t-min", type=float, default=1e-4, dest="t_min",
                   help="smallest relaxation level (default 1e-4; push to 1e-8 for the tail)")
    p.add_argument("--factor", type=float, default=0.3, help="t reduction per level")
    p.add_argument("--tol", type=float, default=1e-6, help="final per-level KKT tolerance (floored)")
    p.add_argument("--max-iter", type=int, default=1500, dest="max_iter",
                   help="IPOPT iteration cap per level")
    p.add_argument("--dual-warmstart", action="store_true", dest="dual_warmstart",
                   help="also warm-start multipliers between levels (pympcc default)")
    p.add_argument("--linear-solver", default="ma57", choices=["ma57", "mumps"],
                   dest="linear_solver")
    p.add_argument("--hess-update", default="bfgs", choices=["bfgs", "sr1", "exact"],
                   dest="hess_update",
                   help="Hessian: bfgs (L-BFGS, PD, default), sr1 (indefinite), or exact "
                   "(analytic Lagrangian Hessian — needed to push the small-t tail).")
    p.add_argument("--cert-c", type=float, default=3.0, dest="cert_c",
                   help="corner-scale factor c for the certificate (ε = c·√t, capped)")
    p.add_argument("--fd-check", action="store_true", dest="fd_check",
                   help="finite-difference validate the objective gradient, Jacobian "
                        "and Hessian, then exit")
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

    prob = LiftedTVMPCC(f, u_clean, args.N, reg_alpha=args.reg_alpha, stencil=args.stencil)

    if args.fd_check:
        x0 = initial_point(prob, args.alpha0, init="cold")  # cold: fast, no CP solve
        # Perturb off the feasible seed so every derivative block is exercised.
        rng = np.random.default_rng(args.seed)
        x0 = x0 + 0.05 * rng.standard_normal(prob.n)
        d0, d1 = prob.off["delta"], prob.off["delta"] + prob.m_q
        x0[d0:d1] = np.abs(x0[d0:d1])
        grad_err, jac_err, hess_err = fd_check(prob, x0, seed=args.seed)
        print(f"FD check (N={args.N}, m_u={prob.m_u}, m_q={prob.m_q}):  "
              f"max|g − g_fd| = {grad_err:.2e}   max|J − J_fd| = {jac_err:.2e}   "
              f"max|H − H_fd| = {hess_err:.2e}")
        ok = grad_err < 1e-6 and jac_err < 1e-5 and hess_err < 1e-4
        print("  ⇒ " + ("PASS" if ok else "FAIL") + " (thresholds 1e-6 / 1e-5 / 1e-4)")
        return

    # Geometric schedule t0, factor·t0, … ≥ t_min.
    schedule, t = [], args.t0
    while t >= args.t_min:
        schedule.append(t)
        t *= args.factor

    gauge = f"TR(c_θ={args.c_theta:g})" if args.c_theta > 0 else "none"
    print(
        f"Lifted TV-MPCC (unit-ball, staggered)   source={src}  "
        f"m_u={prob.m_u} (nodes)  m_q={prob.m_q} (cells)  n={prob.n}  "
        f"m_con={prob.m_con}  levels={len(schedule)}  stencil={args.stencil}  "
        f"solver={args.linear_solver}  hess={args.hess_update}  gauge={gauge}  "
        f"init={args.init}"
    )
    x0 = initial_point(prob, args.alpha0, init=args.init)
    prob.theta_ref = x0[prob.off["theta"] : prob.off["theta"] + prob.m_q].copy()
    a0_eff = float(x0[prob.off["alpha"]])  # cp-scan may override α₀ with argmin L(α)
    a0_src = ("cp-scan argmin L(α)" if args.init == "cp-scan"
              else "noise-aware log(0.7·σ)" if a0_auto else "explicit --alpha0")
    print(f"  init α₀ = {a0_eff:.3f}  (e^α₀ = {np.exp(a0_eff):.4f})   [{a0_src}]")

    t_start = time.perf_counter()
    x, info, best_t, total_iter, best_n_reg, best_n_rest = solve_scholtes(
        prob, x0, schedule,
        linear_solver=args.linear_solver, tol=args.tol, max_iter=args.max_iter,
        dual_warmstart=args.dual_warmstart, c_theta=args.c_theta,
        hess_update=args.hess_update, cert_c=args.cert_c,
        print_level=args.print_level,
    )
    wall = time.perf_counter() - t_start

    u = x[: prob.m_u]
    alpha = x[prob.off["alpha"]]
    msg = info["status_msg"]
    msg = msg.decode() if isinstance(msg, bytes) else msg
    print("─" * 64)
    print(f"  best level t  : {best_t:.1e}  (smallest r·w among converged levels)")
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
