r"""2D lifted TV-MPCC on a **staggered node/cell grid** — with domain decomposition.

The 2D sibling of ``lifted_mpcc_1d.py``: same staggered whiteboard mesh, same
self-contained domain-decomposition probe (partition → arrowhead → local Schur →
Haynsworth inertia → direct interface solve), same graphics — but on an N×N
image instead of a signal.

``u`` lives on the ``m_u = N²`` **nodes**, the dual and lifted fields
``qx, qy, r, δ, θ`` on the ``m_q = (N−1)²`` **cell centres**, so the gradient
operators are genuinely rectangular ``(m_q × m_u)`` with **no Neumann
zero-row** and ``−Kᵀ`` is the exact discrete divergence::

    u₁ ──────── u₂ ──────── u₃          node (i,j) → i·N + j        (i,j < N)
    │  ∘ q₍₀,₀₎  │  ∘ q₍₀,₁₎ │          cell (I,J) → I·(N−1) + J    (I,J < N−1)
    u₄ ──────── u₅ ──────── u₆
    │  ∘ q₍₁,₀₎  │  ∘ q₍₁,₁₎ │          m_u = N²      nodes  (u, f, u_clean, h1)
    u₇ ──────── u₈ ──────── u₉          m_q = (N−1)²  cells  (q, r, δ, θ, everything else)

Two stencils, both from the board (``--stencil``):

``onesided`` (default) — one difference per cell, both components anchored at the
cell's bottom-right node so ``qx``/``qy`` stay co-located (the polar lift needs
them at the same point)::

    (Kx u)_{I,J} = u_{I+1,J+1} − u_{I+1,J}          Kx = Sh ⊗ D
    (Ky u)_{I,J} = u_{I+1,J+1} − u_{I,J+1}          Ky = D ⊗ Sh

``averaged`` — the ½ stencil (gradient of the bilinear interpolant at the cell
centre), ``Kx = A ⊗ D``, ``Ky = D ⊗ A``.

Relation to the rest of the repo
--------------------------------
``lifted_mpcc_unitball_staggered.py`` already implements this *formulation*
(it is v1's numerics on this mesh). This file is a deliberate **self-contained
fork** of it, so the two will drift — do not "fix" either to match the other.
What the fork adds, and why it exists:

* **Domain decomposition on the staggered mesh** (the new work). ``dd_kkt.py``
  cannot be reused: it hardcodes the uniform ``6m+1`` / ``8m`` layout
  (``np.tile(dd.sub_of_pixel, 6)``, KKT dim ``17m+1``). Here ``u`` and the cell
  fields live on different meshes, so both the layout and the interface geometry
  are different.
* **Graphics** — solution fields, the domain map, the continuation path.
* **Both weights**, as in ``lifted_mpcc_1d.py``: ``--weight linear`` (default,
  the board's bare ``α`` in the state row plus an explicit ``ha : α ≥ 0`` row,
  as in ``lifted_mpcc_unitball_linalpha.py``) and ``--weight exp`` (v2's
  ``e^α``, α boxed). One code path via ``Q, Q', Q''``.
* **v2 numerics** that ``staggered.py`` (a v1 fork) lacks: the hybrid
  accelerated CP warm start with an h1-residual stop, per-x memoized callbacks,
  a single reused ``cyipopt.Problem`` with ``t`` on ``prob.t_comp``, and a
  ``--fd-check`` that also validates the objective gradient.

The formulation
---------------
Variables ``x = [u (m_u) | qx | qy | r | δ | θ (m_q each) | α]``,
``n = m_u + 5·m_q + 1``. Rows, ``comp`` LAST::

    h1  (m_u) : u − f + Q(α)·(Kxᵀqx + Kyᵀqy) = 0        the state row
    h2x (m_q) : Kx u − r ∘ cos θ = 0                    polar, primal side
    h2y (m_q) : Ky u − r ∘ sin θ = 0
    h3x (m_q) : qx   − δ ∘ cos θ = 0                    polar, dual side
    h3y (m_q) : qy   − δ ∘ sin θ = 0
    hr, hd    : r ≥ 0,  δ ≥ 0                           explicit inequality ROWS
    ha  (1)   : α ≥ 0                                   (linear weight only)
    comp(m_q) : r ∘ (1 − δ) ≤ t                         Scholtes relaxation

plus the box ``δ ≤ 1`` (the unit ball). ``n_eq = m_u + 4·m_q``,
``n_ineq = 3·m_q [+1]``, KKT dim ``n + 2·n_ineq + n_eq``.

Domain decomposition
--------------------
``--nsub k`` splits the ``(N−1)²`` **cells** into ``k×k`` contiguous tiles
(``k²`` subdomains — ``k`` is per-direction, as in ``dd_structure.build``).

**Node ownership uses the ``anchor`` rule**: node ``(i,j)`` belongs to the tile
of cell ``(i−1, j−1)`` (clamped), i.e. the cell the node anchors under the
one-sided stencil. This is not cosmetic — measured against the naive
"node follows cell ``i``" rule:

    stencil    N,k     forward      anchor
    onesided   16,2    p =  90      p =  60
    onesided   32,4    p = 538      p = 364
    averaged   16,2    p =  90      p =  90   (no change)

Under ``anchor`` only **one** dual component crosses each cut (``qx`` at
vertical cuts, ``qy`` at horizontal), reproducing the structural-zero behaviour
``dd_structure.py`` documents for the uniform grid, and ``p ≈ 4N(k−1)`` matches
the uniform rule's size. ``averaged`` cannot benefit — its stencil genuinely
touches all four nodes of the cell — so it carries a ~1.5× larger interface on
top of its known ≈0.8 dB weaker TV prior.

The complicating set is **computed from the real Jacobian sparsity** (a column is
complicating iff the rows it appears in are owned by ≥2 tiles) rather than
hand-derived: that is stencil-agnostic and self-validating, and is the same
cross-check ``dd_structure.py --self-test`` runs against its analytic rule.

When the probe runs
-------------------
Three clocks, all fed by the same :func:`dd_probe`, because the border rule above
reads jacobian *structure* — it is independent of ``x``, ``t`` and ``μ``, so
``owner`` is built once and every probe rebuilds only ``H``, ``J``, ``Σ``, ``δ_c``:

1. **Every Scholtes level** (default; ``--no-dd-path`` to disable). The claim is
   about the *path* — as ``t ↓ 0`` the biactive set grows, ``Σ`` blows up, and
   nothing forces ``S`` to stay SPD — so one sample at the end cannot separate
   "DD works here" from "it happened to work there". Each level is probed at its
   converged iterate: smallest μ, largest ``Σ``, the worst-conditioned point of
   that level.
2. **Every Newton iteration** (``--dd-iters STRIDE``, off by default because it
   costs one DD solve per probed iteration). This is the honest comparison
   against IPOPT: correction happens at the *intermediate* iterates, which is
   also the only place a real DD linear solver would ever be called. Two things
   make it valid — ``get_current_violations`` gives the **exact** ``Σ = z/s``
   instead of the central-path ``z²/μ`` (``--dd-sigma-mu`` for the A/B), and the
   callback's off-by-one is *measured* rather than assumed (see
   :func:`report_dd_iters`).
3. **The reported iterate** — the original probe, with the full timing and
   fill breakdown.

Measured N=12 k=2 through t=3.0e-1 (a run where IPOPT regularizes hard, δw 44/46
of 70 iterations): **142/142** probed iterates hit the inertia target with ``S``
SPD and Haynsworth MATCH, and our δ_w *decision* agrees with IPOPT's on 123/140
under the +1 alignment (vs 111/142 unshifted, which is how the shift is settled).

Usage
-----
    uv run python lifted_mpcc_2d.py                       # cameraman N=16, k=2
    uv run python lifted_mpcc_2d.py --N 32 --nsub 4
    uv run python lifted_mpcc_2d.py --dd-iters 1          # the per-Newton-iteration probe
    uv run python lifted_mpcc_2d.py --no-dd-path          # only the reported iterate
    uv run python lifted_mpcc_2d.py --stencil averaged    # the wider-interface A/B
    uv run python lifted_mpcc_2d.py --weight exp          # e^α, for the staggered A/B
    uv run python lifted_mpcc_2d.py --fd-check
    uv run python lifted_mpcc_2d.py --save-plot sol.png --save-domains dom.png
    uv run python lifted_mpcc_2d.py --N 96 --nsub 12 --s-format dense  # the old dense-S A/B

The interface
-------------
``S`` is assembled and factorized **sparse** by default (``--s-format sparse``):
its pattern is the union of the ``N_k×N_k`` cliques, measured 23.4% dense at
N=32 k=4 and **1.45% at N=128 k=16**, where the old dense ``p×p`` array cost
417 MB and a 1.35e11-flop ``LDLᵀ``. ``In(S)`` comes from SuperLU's symmetric
mode (see ``Arrowhead.factor_S``), so no dense factorization is ever formed.
"""

from __future__ import annotations

import argparse
import os
import signal
import time

import cyipopt
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from lifted_mpcc_unitball_staggered import (
    DEFAULT_IMAGE, HSLLIB, load_image, make_phantom, psnr,
)

# Above this KKT dimension the *monolithic* dense LDLᵀ cross-check of the
# Haynsworth identity is skipped (it is O(dim²) memory: 121 MB at N=16, but
# 2.2 GB at N=32). The distributed inertia — dense LDLᵀ per W_k — still runs.
DENSE_CHECK_MAX_DIM = 6000


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def grad_operators_2d(N: int, stencil: str = "onesided"):
    """Staggered cell-centred gradient ``(Kx, Ky) : R^{N²} → R^{(N−1)²}``.

    Rectangular by construction — no Neumann zero-row — so ``−Kᵀ`` is the exact
    discrete divergence with the natural no-flux boundary condition. Unit mesh
    spacing (h = 1), as on the board, so α stays comparable with the other
    scripts. See the module docstring for the two stencils.
    """
    if N < 2:
        raise ValueError("N ≥ 2 required (the staggered grid needs at least one cell)")
    e = np.ones(N - 1)
    D = sp.diags([-e, e], [0, 1], shape=(N - 1, N), format="csr")
    if stencil == "onesided":
        Sh = sp.hstack([sp.csr_matrix((N - 1, 1)), sp.identity(N - 1)], format="csr")
        return sp.kron(Sh, D, format="csr"), sp.kron(D, Sh, format="csr")
    if stencil == "averaged":
        A = sp.diags([0.5 * e, 0.5 * e], [0, 1], shape=(N - 1, N), format="csr")
        return sp.kron(A, D, format="csr"), sp.kron(D, A, format="csr")
    raise ValueError(f"unknown stencil {stencil!r} (expected 'onesided' or 'averaged')")


def k_norm_sq(stencil: str) -> float:
    """``‖K‖²`` for the CP step size. ``onesided`` inherits the usual 2D bound 8;
    ``averaged`` multiplies the symbol by ``cos(ξ⊥/2)``, which shrinks it to 4."""
    return 8.0 if stencil == "onesided" else 4.0


# ---------------------------------------------------------------------------
# the problem object
# ---------------------------------------------------------------------------
class Lifted2DMPCC:
    """cyipopt problem object for the staggered 2D lifted TV-MPCC.

    ``u`` is the only node-length block (``m_u = N²``); ``qx, qy, r, δ, θ`` and
    every row except ``h1``/``ha`` are cell-length (``m_q = (N−1)²``). See the
    module docstring for the full row/column layout.
    """

    def __init__(self, f, u_clean, N, *, weight="linear", stencil="onesided",
                 reg_alpha=1e-4, w_max=float("inf")):
        self.f = np.asarray(f, float)
        self.u_clean = np.asarray(u_clean, float)
        self.N = N
        self.m_u = m_u = N * N
        self.m_q = m_q = (N - 1) * (N - 1)
        self.weight = weight
        self.stencil = stencil
        self.reg_alpha = reg_alpha
        self.w_max = w_max            # optional cap on the linear weight (see bounds)
        self.has_ha = weight == "linear"

        # Scholtes level: the comp row is r·(1−δ) − t_comp ≤ 0 with fixed upper
        # bound 0, so one cyipopt.Problem is reused across the whole continuation
        # (t is additive, hence absent from every derivative).
        self.t_comp = 0.0
        # TR gauge ridge ½·eps_theta·‖θ − θ_ref‖² — the D1 fix for the angle-gauge
        # indeterminacy (θ undetermined where r = δ = 0). The driver sets
        # eps_theta = c_θ·t per level so the bias vanishes as t ↓ 0.
        self.eps_theta = 0.0
        self.theta_ref = np.zeros(m_q)

        self.n_iter = self.n_reg = self.n_rest = 0
        self.inf_pr = self.inf_du = self.mu_last = float("nan")
        self._interrupt = False
        self._cx = None
        # per-Newton-iteration probe plumbing (see ``intermediate``): the hook and
        # the live cyipopt.Problem, both installed by ``solve_scholtes``.
        self._on_iter = None
        self._nlp = None

        self.Kx, self.Ky = grad_operators_2d(N, stencil)
        self.KxT, self.KyT = self.Kx.T.tocsr(), self.Ky.T.tocsr()

        self.off = {"u": 0, "qx": m_u, "qy": m_u + m_q, "r": m_u + 2 * m_q,
                    "delta": m_u + 3 * m_q, "theta": m_u + 4 * m_q,
                    "alpha": m_u + 5 * m_q}
        self.blen = {"u": m_u, "qx": m_q, "qy": m_q, "r": m_q,
                     "delta": m_q, "theta": m_q}
        self.n = m_u + 5 * m_q + 1

        ro, o = {}, 0
        for name, size in (("h1", m_u), ("h2x", m_q), ("h2y", m_q), ("h3x", m_q),
                           ("h3y", m_q), ("hr", m_q), ("hd", m_q),
                           ("ha", 1 if self.has_ha else 0), ("comp", m_q)):
            ro[name] = o
            o += size
        self.roff = ro
        self.rlen = {"h1": m_u, "h2x": m_q, "h2y": m_q, "h3x": m_q, "h3y": m_q,
                     "hr": m_q, "hd": m_q, "ha": 1 if self.has_ha else 0,
                     "comp": m_q}
        self.m_con = o
        self.n_eq = ro["hr"]                  # h1|h2x|h2y|h3x|h3y
        self.n_ineq = self.m_con - self.n_eq  # hr|hd|[ha]|comp

        self._rows, self._cols = self._build_structure()
        self._hrows, self._hcols = self._build_hess_structure()

    # ---- the weight Q(α) and its derivatives ------------------------------
    def Q(self, a):
        return float(a) if self.weight == "linear" else float(np.exp(a))

    def dQ(self, a):
        return 1.0 if self.weight == "linear" else float(np.exp(a))

    def d2Q(self, a):
        return 0.0 if self.weight == "linear" else float(np.exp(a))

    def alpha_of_weight(self, w):
        return float(w) if self.weight == "linear" else float(np.log(w))

    def weight_of_alpha(self, a):
        return float(a) if self.weight == "linear" else float(np.exp(a))

    # ---- slicing helpers --------------------------------------------------
    def _blk(self, x, name):
        s = self.off[name]
        return x[s : s + self.blen[name]]

    def _rblk(self, v, name):
        s = self.roff[name]
        return v[s : s + self.rlen[name]]

    def _divq(self, x):
        return self.KxT @ self._blk(x, "qx") + self.KyT @ self._blk(x, "qy")

    def _common(self, x):
        """Memoized per-x intermediates ``(Q, Q', Q'', cos θ, sin θ, div q)``.

        All are t-independent, so the memo survives Scholtes level changes.
        """
        if self._cx is None or not np.array_equal(x, self._cx):
            self._cx = np.array(x, dtype=float, copy=True)
            a = self._cx[self.off["alpha"]]
            th = self._blk(self._cx, "theta")
            self._cvals = (self.Q(a), self.dQ(a), self.d2Q(a), np.cos(th),
                           np.sin(th), self._divq(self._cx))
        return self._cvals

    # ---- objective --------------------------------------------------------
    def objective(self, x):
        u = self._blk(x, "u")
        a = x[self.off["alpha"]]
        obj = 0.5 * np.sum((u - self.u_clean) ** 2) + 0.5 * self.reg_alpha * a * a
        if self.eps_theta:
            d = self._blk(x, "theta") - self.theta_ref
            obj += 0.5 * self.eps_theta * np.dot(d, d)
        return obj

    def gradient(self, x):
        g = np.zeros(self.n)
        g[: self.m_u] = self._blk(x, "u") - self.u_clean
        g[self.off["alpha"]] = self.reg_alpha * x[self.off["alpha"]]
        if self.eps_theta:
            s = self.off["theta"]
            g[s : s + self.m_q] = self.eps_theta * (self._blk(x, "theta")
                                                    - self.theta_ref)
        return g

    # ---- constraints ------------------------------------------------------
    def constraints(self, x):
        Qa, _, _, c, s, divq = self._common(x)
        u, qx, qy = self._blk(x, "u"), self._blk(x, "qx"), self._blk(x, "qy")
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        out = [u - self.f + Qa * divq,     # h1  (m_u)  state row
               self.Kx @ u - r * c,        # h2x
               self.Ky @ u - r * s,        # h2y
               qx - delta * c,             # h3x
               qy - delta * s,             # h3y
               r,                          # hr : r ≥ 0
               delta]                      # hd : δ ≥ 0
        if self.has_ha:
            out.append(np.array([x[self.off["alpha"]]]))   # ha : α ≥ 0
        out.append(r * (1.0 - delta) - self.t_comp)        # comp (Scholtes), LAST
        return np.concatenate(out)

    # ---- Jacobian ---------------------------------------------------------
    def _build_structure(self):
        """Index arrays of every Jacobian nonzero.

        The pieces below are assembled in the SAME order by ``jacobian`` — the
        value arrays are concatenated positionally, not matched by key, so the
        two lists must stay aligned (run ``--fd-check`` after any edit). Piece 4
        is the dense ``∂h1/∂α`` column (length m_u), the whole α-coupling of the
        KKT system. Kx/Ky are rectangular (m_q × m_u) and KxT/KyT are
        (m_u × m_q), so the diagonals come in two lengths.
        """
        m_u, m_q, off, ro = self.m_u, self.m_q, self.off, self.roff
        iu, iq = np.arange(m_u), np.arange(m_q)

        def qdiag(roff, coff):            # cell-length diagonal
            return roff + iq, coff + iq

        Kx, Ky = self.Kx.tocoo(), self.Ky.tocoo()
        KxT, KyT = self.KxT.tocoo(), self.KyT.tocoo()
        self._ones_u, self._ones_q = np.ones(m_u), np.ones(m_q)
        self._Kx, self._Ky = Kx.data, Ky.data
        self._KxT, self._KyT = KxT.data, KyT.data

        pieces = [
            (ro["h1"] + iu, off["u"] + iu),                # 1  h1/∂u   = I  (m_u)
            (ro["h1"] + KxT.row, off["qx"] + KxT.col),     # 2  h1/∂qx  = Q(α)·Kxᵀ
            (ro["h1"] + KyT.row, off["qy"] + KyT.col),     # 3  h1/∂qy  = Q(α)·Kyᵀ
            (ro["h1"] + iu, np.full(m_u, off["alpha"])),   # 4  h1/∂α   = Q'(α)·div q
            (ro["h2x"] + Kx.row, off["u"] + Kx.col),       # 5  h2x/∂u  = Kx
            qdiag(ro["h2x"], off["r"]),                    # 6  h2x/∂r  = −cos θ
            qdiag(ro["h2x"], off["theta"]),                # 7  h2x/∂θ  =  r sin θ
            (ro["h2y"] + Ky.row, off["u"] + Ky.col),       # 8  h2y/∂u  = Ky
            qdiag(ro["h2y"], off["r"]),                    # 9  h2y/∂r  = −sin θ
            qdiag(ro["h2y"], off["theta"]),                # 10 h2y/∂θ  = −r cos θ
            qdiag(ro["h3x"], off["qx"]),                   # 11 h3x/∂qx = I
            qdiag(ro["h3x"], off["delta"]),                # 12 h3x/∂δ  = −cos θ
            qdiag(ro["h3x"], off["theta"]),                # 13 h3x/∂θ  =  δ sin θ
            qdiag(ro["h3y"], off["qy"]),                   # 14 h3y/∂qy = I
            qdiag(ro["h3y"], off["delta"]),                # 15 h3y/∂δ  = −sin θ
            qdiag(ro["h3y"], off["theta"]),                # 16 h3y/∂θ  = −δ cos θ
            qdiag(ro["hr"], off["r"]),                     # 17 hr/∂r   = 1
            qdiag(ro["hd"], off["delta"]),                 # 18 hd/∂δ   = 1
        ]
        if self.has_ha:
            pieces.append((np.array([ro["ha"]]), np.array([off["alpha"]])))  # 19 ha/∂α
        pieces += [
            qdiag(ro["comp"], off["r"]),                   # 20 comp/∂r = 1 − δ
            qdiag(ro["comp"], off["delta"]),               # 21 comp/∂δ = −r
        ]
        rows = np.concatenate([p[0] for p in pieces]).astype(np.int64)
        cols = np.concatenate([p[1] for p in pieces]).astype(np.int64)
        return rows, cols

    def jacobianstructure(self):
        return self._rows, self._cols

    def jacobian(self, x):
        Qa, dQa, _, c, s, divq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        vals = [self._ones_u,      # 1  h1/∂u
                Qa * self._KxT,    # 2  h1/∂qx
                Qa * self._KyT,    # 3  h1/∂qy
                dQa * divq,        # 4  h1/∂α
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
                self._ones_q,      # 17 hr/∂r
                self._ones_q]      # 18 hd/∂δ
        if self.has_ha:
            vals.append(np.array([1.0]))   # 19 ha/∂α
        vals += [1.0 - delta,      # 20 comp/∂r
                 -r]               # 21 comp/∂δ
        return np.concatenate(vals)

    # ---- exact Lagrangian Hessian (lower triangle) ------------------------
    # H = σ_f·∇²J + Σ_k λ_k ∇²c_k. Nonlinear rows and their curvature:
    #   h1  : bilinear in (α, q) → (α,qx) = Q'(α)·Kx λ_h1 (length m_q, since Kx is
    #         m_q×m_u and λ_h1 is m_u), likewise (α,qy);
    #         (α,α) = Q''(α)·⟨λ_h1, div q⟩ — identically 0 for the LINEAR weight,
    #         which therefore gets its only (α,α) curvature from the reg-α ridge.
    #   h2x : ∂²/∂r∂θ =  sin θ, ∂²/∂θ² =  r cos θ ;  h2y : −cos θ,  r sin θ
    #   h3x : ∂²/∂δ∂θ =  sin θ, ∂²/∂θ² =  δ cos θ ;  h3y : −cos θ,  δ sin θ
    #   comp: r·(1−δ) → ∂²/∂r∂δ = −1 ⇒ (δ,r) = −ξ, the indefinite MPCC cross
    #         (eigenvalues ±ξ).
    # Column order is u < qx < qy < r < δ < θ < α, so all land in the lower triangle.
    def _build_hess_structure(self):
        m_u, m_q, off = self.m_u, self.m_q, self.off
        iu, iq = np.arange(m_u), np.arange(m_q)
        a = off["alpha"]
        rows = np.concatenate([
            off["u"] + iu,         # (u,u)   objective identity
            off["theta"] + iq,     # (θ,r)
            off["theta"] + iq,     # (θ,δ)
            off["theta"] + iq,     # (θ,θ)
            off["delta"] + iq,     # (δ,r)   comp cross (indefinite)
            np.full(m_q, a),       # (α,qx)  h1 bilinear cross
            np.full(m_q, a),       # (α,qy)  h1 bilinear cross
            [a],                   # (α,α)
        ])
        cols = np.concatenate([
            off["u"] + iu, off["r"] + iq, off["delta"] + iq, off["theta"] + iq,
            off["r"] + iq, off["qx"] + iq, off["qy"] + iq, [a],
        ])
        return rows.astype(np.int64), cols.astype(np.int64)

    def hessianstructure(self):
        return self._hrows, self._hcols

    def hessian(self, x, lagrange, obj_factor):
        Qa, dQa, d2Qa, c, s, divq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        l1 = self._rblk(lagrange, "h1")
        l2x, l2y = self._rblk(lagrange, "h2x"), self._rblk(lagrange, "h2y")
        l3x, l3y = self._rblk(lagrange, "h3x"), self._rblk(lagrange, "h3y")
        xi = self._rblk(lagrange, "comp")

        H_uu = obj_factor * self._ones_u
        H_tr = l2x * s - l2y * c                                       # (θ,r)
        H_td = l3x * s - l3y * c                                       # (θ,δ)
        H_tt = (r * (l2x * c + l2y * s) + delta * (l3x * c + l3y * s)
                + obj_factor * self.eps_theta)                         # (θ,θ)
        H_dr = -xi                                                     # (δ,r)
        H_aqx = dQa * (self.Kx @ l1)                                   # (α,qx)
        H_aqy = dQa * (self.Ky @ l1)                                   # (α,qy)
        H_aa = np.array([d2Qa * float(np.dot(l1, divq))
                         + obj_factor * self.reg_alpha])               # (α,α)
        return np.concatenate([H_uu, H_tr, H_td, H_tt, H_dr, H_aqx, H_aqy, H_aa])

    # ---- telemetry --------------------------------------------------------
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.n_iter = iter_count
        self.inf_pr, self.inf_du, self.mu_last = inf_pr, inf_du, mu
        if regularization_size > 0.0:
            self.n_reg += 1
        if alg_mod == 1:
            self.n_rest += 1

        # Per-Newton-iteration hook. ``get_current_iterate`` is only legal from
        # inside this callback (Ipopt ≥ 3.14 / cyipopt ≥ 1.1) and hands back a
        # dict with exactly the keys ``dd_probe`` wants — x, mult_g, mult_x_L,
        # mult_x_U — so the probe is a drop-in on the live iterate;
        # ``get_current_violations`` supplies the complementarity products that
        # buy the exact Σ = z/s.
        #
        # ALIGNMENT: this callback fires at the *end* of iteration ``iter_count``,
        # so ``x`` here is the iterate the NEXT step will be computed from, while
        # ``regularization_size`` is the δ_w Ipopt already used for the step just
        # taken. Our δ_w at this x is therefore the prediction for the *next*
        # callback's ``regularization_size``; the shift is resolved empirically in
        # :func:`report_dd_iters` rather than asserted here.
        if self._on_iter is not None and self._nlp is not None:
            try:
                it = self._nlp.get_current_iterate()
                if it is not None:
                    vi = self._nlp.get_current_violations()
                    self._on_iter(dict(it), None if vi is None else dict(vi),
                                  iter_count, mu, regularization_size, alg_mod)
            except Exception as exc:      # a probe must never kill the solve
                print(f"  [warn] iterate probe failed at iter {iter_count}: "
                      f"{type(exc).__name__}: {exc}")

        return not self._interrupt   # False → IPOPT stops cleanly (status 5)


# ---------------------------------------------------------------------------
# warm start: 2D Chambolle–Pock ROF
# ---------------------------------------------------------------------------
def chambolle_pock_2d(f, Kx, Ky, KxT, KyT, lam, *, knorm2=8.0, n_iter=3000,
                      tol=1e-9, n_accel=300, u0=None, qx0=None, qy0=None):
    """Hybrid Chambolle–Pock for ROF ``min_u ½‖u−f‖² + lam·‖∇u‖_{2,1}``.

    Returns ``(u, qx, qy)`` with ``u`` on the nodes and the dual on the cells,
    per-cell on the **unit ball** ``‖(qx,qy)‖ ≤ 1``. At convergence
    ``u = f − lam·(Kxᵀqx + Kyᵀqy)`` with ``r ⊥ (1−δ)`` — exactly this MPCC's
    lower-level system at ``Q(α) = lam``, so the CP pair lifts to an
    almost-feasible warm start (h1 ≈ 0, h3 exact).

    v2's hybrid schedule (which ``staggered.py`` predates): an accelerated phase
    (γ = 1, the primal is 1-strongly convex) then fixed steps that polish the h1
    fixed point, stopping on the exact residual identity
    ``‖u₊−u‖∞/τ = ‖h1‖∞``. ``knorm2`` is stencil-dependent (8 one-sided, 4
    averaged — averaging multiplies the symbol by ``cos(ξ⊥/2)``).
    """
    kn = np.sqrt(knorm2)
    tau0 = 0.99 / (kn * lam)
    sig_lam0 = 0.99 / kn
    tau, sig_lam = tau0, sig_lam0
    u = f.copy() if u0 is None else np.asarray(u0, float).copy()
    ubar = u.copy()
    m_q = Kx.shape[0]
    qx = np.zeros(m_q) if qx0 is None else np.asarray(qx0, float).copy()
    qy = np.zeros(m_q) if qy0 is None else np.asarray(qy0, float).copy()
    for k in range(n_iter):
        accel = k < n_accel
        if not accel and tau != tau0:
            tau, sig_lam = tau0, sig_lam0          # phase switch (τσ invariant)
        qx += sig_lam * (Kx @ ubar)
        qy += sig_lam * (Ky @ ubar)
        nrm = np.maximum(1.0, np.hypot(qx, qy))    # projection onto the unit ball
        qx /= nrm
        qy /= nrm
        u_new = (tau * f + u - (tau * lam) * (KxT @ qx + KyT @ qy)) / (tau + 1.0)
        h1_res = float(np.max(np.abs(u_new - u))) / tau
        if accel:
            th = 1.0 / np.sqrt(1.0 + 2.0 * tau)
            ubar = u_new + th * (u_new - u)
            tau *= th
            sig_lam /= th
        else:
            ubar = 2.0 * u_new - u
        u = u_new
        if h1_res <= tol:
            break
    return u, qx, qy


def cp_scan(prob, w0, half_width=2.0, n_grid=9):
    """Coarse CP sweep of ``L(w) = ½‖u_CP(w) − u_clean‖²`` on a log grid around
    ``w0``; returns the argmin ``(w*, u, qx, qy)``. Legitimate because CP is an
    exact lower-level solve and ``u_clean`` is the training target, so this is
    the true (unimodal) bilevel objective."""
    best = None
    u = qx = qy = None
    kn2 = k_norm_sq(prob.stencil)
    for w in w0 * np.exp(np.linspace(-half_width, half_width, n_grid)):
        u, qx, qy = chambolle_pock_2d(prob.f, prob.Kx, prob.Ky, prob.KxT, prob.KyT,
                                      float(w), knorm2=kn2, u0=u, qx0=qx, qy0=qy)
        loss = 0.5 * float(np.sum((u - prob.u_clean) ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(w), u.copy(), qx.copy(), qy.copy())
    return best[1], best[2], best[3], best[4]


def initial_point(prob, w0, init="cp"):
    """Warm start. ``w0`` is the WEIGHT (not α).

    ``cp``/``cp-scan`` lift a Chambolle–Pock lower-level solution: ``u, qx, qy``
    from CP, ``δ = |q|``, ``r = |∇u|``, and — since at the ROF solution
    ``q ∥ ∇u`` where ``|∇u| > 0`` — **θ from the dual** (``θ = ∠q``), which makes
    h3 exact while h1 holds by CP optimality.

    Taking θ from the dual rather than from ``∇u`` is the right way round, and
    the 1D sibling measured why: the reverse puts the error on
    ``h3 = q − |q|·(direction)``, i.e. **O(1) wherever the cell is flat** (a flat
    cell still carries a unit-size dual), and nothing converges. This way the
    error lands on ``h2 = ∇u − |∇u|·(direction)``, which is at noise level
    exactly where the two disagree.

    ``cold`` is ``u = f, q = 0``: it sits on the no-regularization manifold and
    is this repo's known route to the spurious near-noisy branch.
    """
    m_u, m_q, off = prob.m_u, prob.m_q, prob.off
    x = np.zeros(prob.n)
    if init in ("cp", "cp-scan"):
        if init == "cp-scan":
            w0, u, qx, qy = cp_scan(prob, w0)
        else:
            u, qx, qy = chambolle_pock_2d(prob.f, prob.Kx, prob.Ky, prob.KxT,
                                          prob.KyT, float(w0),
                                          knorm2=k_norm_sq(prob.stencil))
    else:                                     # cold
        u, qx, qy = prob.f.copy(), np.zeros(m_q), np.zeros(m_q)

    gx, gy = prob.Kx @ u, prob.Ky @ u
    x[off["u"] : off["u"] + m_u] = u
    x[off["qx"] : off["qx"] + m_q] = qx
    x[off["qy"] : off["qy"] + m_q] = qy
    x[off["r"] : off["r"] + m_q] = np.hypot(gx, gy)
    x[off["delta"] : off["delta"] + m_q] = np.hypot(qx, qy)
    if init == "cold":
        x[off["theta"] : off["theta"] + m_q] = np.arctan2(gy, gx)   # q = 0 ⇒ no angle
    else:
        x[off["theta"] : off["theta"] + m_q] = np.arctan2(qy, qx)   # dual angle
    x[off["alpha"]] = prob.alpha_of_weight(w0)
    return x


def bounds(prob, alpha_lo=-15.0, alpha_hi=15.0):
    """Variable bounds. Only ``δ ≤ 1`` (the unit ball) is a box for every mode;
    ``r ≥ 0``, ``δ ≥ 0`` and — under the linear weight — ``α ≥ 0`` are explicit
    rows instead.

    The linear weight is unbounded above by default (``w_max = inf``), matching
    ``lifted_mpcc_unitball_linalpha.py``'s "no α box at all"; the exponential
    weight keeps v2's box, which caps it implicitly at ``e^15``. ``--w-max`` is
    an opt-in safety net for the loose end of the schedule, where the argument
    for a cap is real: at a vacuous level ``q`` is pinned only by ``|q| = δ ≤ 1``
    while ``α·Kᵀq`` is not pinned at all, so the upper level can push ``α → ∞``
    opposed only by the tiny ``½·reg_alpha·α²`` ridge.
    """
    xl = np.full(prob.n, -2.0e19)
    xu = np.full(prob.n, 2.0e19)
    xu[prob.off["delta"] : prob.off["delta"] + prob.m_q] = 1.0
    if prob.has_ha:
        if np.isfinite(prob.w_max):
            xu[prob.off["alpha"]] = prob.w_max
    else:
        xl[prob.off["alpha"]], xu[prob.off["alpha"]] = alpha_lo, alpha_hi
    return xl, xu


def constraint_bounds(prob):
    """``(cl, cu)``: equalities at 0, ``hr/hd/ha`` as ``0 ≤ · ≤ +∞``, and the comp
    rows one-sided with a FIXED upper bound 0 (``t`` rides inside the row on
    ``prob.t_comp``, additively, so it changes no derivative — that is what lets
    a single Problem serve every level)."""
    cl = np.zeros(prob.m_con)
    cu = np.zeros(prob.m_con)
    cu[prob.roff["hr"] : prob.roff["comp"]] = 2.0e19
    cl[prob.roff["comp"] :] = -2.0e19
    return cl, cu


# ---------------------------------------------------------------------------
# Scholtes ε-continuation
# ---------------------------------------------------------------------------
def solve_scholtes(prob, x0, schedule, *, linear_solver="ma57", tol=1e-8,
                   tol_factor=0.1, max_iter=1500, hess_update="exact",
                   dual_warmstart=False, c_theta=0.0, print_level=0, verbose=True,
                   on_level=None, on_iter=None):
    """Solve the relaxed NLPs for ``t ↓`` along ``schedule``.

    Level ``t`` is solved to ``max(tol, tol_factor·t)``; one ``cyipopt.Problem``
    is reused (``t`` lives on ``prob.t_comp``); the reported iterate is the
    **tightest converged level** (smallest ``r·(1−δ)``), not the smallest loss —
    a loose level overfits ``u_clean`` and is not a real lower-level solution;
    the loop stops at the first level IPOPT fails. Ctrl-C stops the current solve
    cleanly and reports the best completed level.

    ``on_level(record, x, info, mu)`` is called once per attempted level, right
    after it finishes and *before* the iterate is overwritten by the next level —
    that hook is what lets the DD probe run along the whole path instead of only
    on the reported iterate. ``on_iter(iterate, viol, iter_count, mu, regu,
    alg_mod)`` is the finer clock: it fires from ``prob.intermediate`` at **every
    Newton iteration**, which is where IPOPT actually performs its inertia
    correction and where a real DD linear solver would be called. Both are
    diagnostics: an exception inside either is caught and reported, never allowed
    to kill the continuation.

    Returns ``(x, info, t, total_iter, mu_last, history)``.
    """
    off, m_q = prob.off, prob.m_q
    xl, xu = bounds(prob)
    cl, cu = constraint_bounds(prob)

    use_ma57 = (linear_solver == "ma57") and os.path.exists(HSLLIB)
    if linear_solver == "ma57" and not use_ma57:
        print(f"  [warn] {HSLLIB} not found → falling back to MUMPS")
        linear_solver = "mumps"

    if verbose:
        print(f"  {'t':>9} {'tol':>8} {'iters':>6} {'δw':>4} {'rest':>4} "
              f"{'status':>7} {'comp_res':>10} {'max|ξ|':>10} {'weight':>9} {'obj':>11}")

    x = x0.copy()
    warm, best, info = None, None, None
    comp_res, total_iter, t = float("nan"), 0, schedule[-1]
    history = []

    prob._interrupt = False

    def _on_sigint(signum, frame):
        if prob._interrupt:
            signal.signal(signal.SIGINT, _old_sigint)
            raise KeyboardInterrupt
        prob._interrupt = True
        print("\n  [interrupt] Ctrl-C — stopping after the current IPOPT iteration.",
              flush=True)

    try:
        _old_sigint = signal.signal(signal.SIGINT, _on_sigint)
        _sig = True
    except (ValueError, OSError):
        _old_sigint, _sig = None, False

    nlp = None
    try:
        nlp = cyipopt.Problem(n=prob.n, m=prob.m_con, problem_obj=prob,
                              lb=xl, ub=xu, cl=cl, cu=cu)
        prob._nlp, prob._on_iter = nlp, on_iter   # per-iteration probe plumbing
        nlp.add_option("sb", "yes")
        nlp.add_option("print_level", int(print_level))
        nlp.add_option("max_iter", int(max_iter))
        nlp.add_option("acceptable_iter", 10)
        nlp.add_option("mu_strategy", "monotone")
        nlp.add_option("linear_solver", linear_solver)
        if hess_update == "exact":
            nlp.add_option("hessian_approximation", "exact")
        else:
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 25)
            nlp.add_option("limited_memory_update_type", hess_update)
        if use_ma57:
            nlp.add_option("hsllib", HSLLIB)
            # MC64 scaling: v2 measured this as required for the N ≥ 40 tail and
            # never paying below it (≈2× per-factorization cost).
            nlp.add_option("ma57_automatic_scaling",
                           "yes" if prob.m_u >= 1600 else "no")
        for opt in ("warm_start_bound_push", "warm_start_bound_frac",
                    "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
                    "warm_start_mult_bound_push"):
            nlp.add_option(opt, 1e-8)

        for t in schedule:
            prob.t_comp = t
            prob.eps_theta = c_theta * t
            tol_t = max(tol, tol_factor * t)
            nlp.add_option("tol", tol_t)
            nlp.add_option("acceptable_tol", max(tol, 10 * tol_t))
            prob.n_iter = prob.n_reg = prob.n_rest = 0

            if dual_warmstart and warm is not None:
                nlp.add_option("warm_start_init_point", "yes")
                nlp.add_option("mu_init", float(np.clip(prob.mu_last, 1e-9, 1e-1)))
                x, info = nlp.solve(x, lagrange=warm[0], zl=warm[1], zu=warm[2])
            else:
                x, info = nlp.solve(x)
            mg = info.get("mult_g")
            if mg is not None and len(mg):
                warm = (mg, info.get("mult_x_L"), info.get("mult_x_U"))
            total_iter += prob.n_iter

            r = x[off["r"] : off["r"] + m_q]
            w = 1.0 - x[off["delta"] : off["delta"] + m_q]
            comp_res = float(np.max(r * w)) if m_q else 0.0
            xi_max = (float(np.max(np.abs(mg[prob.roff["comp"] :])))
                      if (mg is not None and len(mg)) else float("nan"))
            history.append({
                "t": t, "status": int(info["status"]), "iters": prob.n_iter,
                "comp_res": comp_res, "xi_max": xi_max, "obj": info["obj_val"],
                "weight": prob.weight_of_alpha(x[off["alpha"]]),
                "converged": info["status"] in (0, 1),
            })
            if verbose:
                print(f"  {t:>9.1e} {tol_t:>8.1e} {prob.n_iter:>6d} {prob.n_reg:>4d} "
                      f"{prob.n_rest:>4d} {info['status']:>7d} {comp_res:>10.2e} "
                      f"{xi_max:>10.2e} "
                      f"{prob.weight_of_alpha(x[off['alpha']]):>9.4f} "
                      f"{info['obj_val']:>11.4e}")

            if on_level is not None:
                try:
                    on_level(history[-1], x, info, prob.mu_last)
                except Exception as exc:      # a probe must never kill the solve
                    print(f"  [warn] level probe failed at t={t:.1e}: "
                          f"{type(exc).__name__}: {exc}")

            converged = info["status"] in (0, 1)
            if converged and (best is None or comp_res < best[0]):
                best = (comp_res, x.copy(), info, t, prob.mu_last)
            if prob._interrupt:
                if verbose:
                    print("  [stop] interrupted — reporting the best completed level.")
                break
            if not converged:
                if verbose:
                    print(f"  [stop] t={t:.1e} did not converge "
                          f"(status {info['status']}); halting.")
                break
    finally:
        prob._nlp = prob._on_iter = None       # the handle dies with the solve
        if nlp is not None:
            nlp.close()
        if _sig:
            signal.signal(signal.SIGINT, _old_sigint)

    if best is None:
        best = (comp_res, x, info, t, prob.mu_last)
    return best[1], best[2], best[3], total_iter, best[4], history


# ---------------------------------------------------------------------------
# derivative check
# ---------------------------------------------------------------------------
def fd_check(prob, x, seed=0, n_dirs=8):
    """Directional FD validation of the gradient, Jacobian and exact Hessian
    (``J·v``, ``gᵀv``, ``H·v`` against central differences along random unit
    directions — any wrong entry shows up in every random direction with
    probability 1). Returns ``(jac_err, grad_err, hess_err)``."""
    rng = np.random.default_rng(seed)
    x = x.astype(float).copy()
    h = 1e-6
    rows, cols = prob.jacobianstructure()
    J = sp.coo_matrix((prob.jacobian(x), (rows, cols)),
                      shape=(prob.m_con, prob.n)).tocsr()
    g = prob.gradient(x)
    obj_factor = float(rng.uniform(0.5, 1.5))
    lam = rng.standard_normal(prob.m_con)
    hr, hc = prob.hessianstructure()
    Hl = sp.coo_matrix((prob.hessian(x, lam, obj_factor), (hr, hc)),
                       shape=(prob.n, prob.n)).tocsr()
    H = Hl + Hl.T - sp.diags(Hl.diagonal())

    def lag_grad(xx):
        vals = prob.jacobian(xx)
        JTlam = np.zeros(prob.n)
        np.add.at(JTlam, cols, vals * lam[rows])
        return obj_factor * prob.gradient(xx) + JTlam

    je = ge = he = 0.0
    for _ in range(n_dirs):
        v = rng.standard_normal(prob.n)
        v /= np.linalg.norm(v)
        xp, xm = x + h * v, x - h * v
        je = max(je, float(np.max(np.abs(
            (prob.constraints(xp) - prob.constraints(xm)) / (2 * h) - J @ v))))
        ge = max(ge, abs(float(
            (prob.objective(xp) - prob.objective(xm)) / (2 * h) - g @ v)))
        he = max(he, float(np.max(np.abs(
            (lag_grad(xp) - lag_grad(xm)) / (2 * h) - H @ v))))
    return je, ge, he


# ---------------------------------------------------------------------------
# domain decomposition: partition, KKT, arrowhead
# ---------------------------------------------------------------------------
class Partition2D:
    """Partition of the ``(N−1)²`` cells into ``k×k`` contiguous tiles.

    ``k`` is per-direction, so there are ``k²`` subdomains (same convention as
    ``dd_structure.build``).

    **Node ownership — the ``anchor`` rule**: node ``(i,j)`` belongs to the tile
    of cell ``(i−1, j−1)`` (clamped to the grid), i.e. the cell the node anchors
    under the one-sided stencil. See the module docstring for the measured
    interface sizes; the naive "node follows cell ``i``" rule inflates ``p`` by
    ~1.5× because it makes *both* dual components cross every cut.
    """

    def __init__(self, N: int, k: int):
        self.N, self.nc, self.k = N, N - 1, k
        self.n_sub = k * k
        b = np.linspace(0, self.nc, k + 1).astype(int)
        self.cell_bounds = b
        self.cell_owner_2d = np.empty((self.nc, self.nc), dtype=int)
        for a in range(k):
            for c in range(k):
                self.cell_owner_2d[b[a]:b[a + 1], b[c]:b[c + 1]] = a * k + c
        self.cell_owner = self.cell_owner_2d.ravel()
        idx = np.clip(np.arange(N) - 1, 0, self.nc - 1)          # anchor rule
        self.node_owner_2d = self.cell_owner_2d[np.ix_(idx, idx)]
        self.node_owner = self.node_owner_2d.ravel()
        # cut lines, in node coordinates (the first node past each cut)
        self.cut_nodes = [int(c) for c in b[1:-1]]

    def tile_cell_range(self, t: int):
        """``(rows, cols)`` slices of the cell block owned by tile ``t``."""
        b = self.cell_bounds
        a, c = divmod(t, self.k)
        return (b[a], b[a + 1]), (b[c], b[c + 1])


def kkt_owner(prob, part: Partition2D):
    """Label every KKT index with its subdomain, or ``-1`` for the border.

    Rows are never duplicated: each is owned by its node's or cell's tile (the
    scalar ``ha`` row goes to tile 0 — its slack and multiplier are *dual*
    directions, and putting them on the border would make ``S`` indefinite by
    construction, defeating the δ_w loop that runs on that signal).

    The **border is derived from the Jacobian sparsity**: a primal column is
    complicating iff the rows it appears in are owned by ≥2 tiles. That is the
    definition, it is stencil-agnostic, and it needs no hand-derived geometry —
    the same cross-check ``dd_structure.py --self-test`` runs against its
    analytic rule. ``α`` is border by construction (its ``h1`` column is dense).

    Returns ``(owner, col_owner, row_owner)``.
    """
    m_u, m_q, off, ro = prob.m_u, prob.m_q, prob.off, prob.roff
    co, no = part.cell_owner, part.node_owner

    row_owner = np.empty(prob.m_con, dtype=int)
    row_owner[ro["h1"] : ro["h1"] + m_u] = no
    for name in ("h2x", "h2y", "h3x", "h3y", "hr", "hd", "comp"):
        row_owner[ro[name] : ro[name] + m_q] = co
    if prob.has_ha:
        row_owner[ro["ha"]] = 0

    col_owner = np.empty(prob.n, dtype=int)
    col_owner[off["u"] : off["u"] + m_u] = no
    for name in ("qx", "qy", "r", "delta", "theta"):
        col_owner[off[name] : off[name] + m_q] = co
    col_owner[off["alpha"]] = -1

    jr, jc = prob.jacobianstructure()
    pattern = sp.coo_matrix((np.ones(len(jr)), (jr, jc)),
                            shape=(prob.m_con, prob.n)).tocsc()
    for c in range(prob.n):
        rs = pattern.indices[pattern.indptr[c] : pattern.indptr[c + 1]]
        if rs.size and len(set(row_owner[rs].tolist())) > 1:
            col_owner[c] = -1

    # KKT ordering: primal | slacks (ineq rows) | λ_c (eq rows) | λ_d (ineq rows)
    ineq_owner = row_owner[prob.n_eq :]
    owner = np.concatenate([col_owner, ineq_owner, row_owner[: prob.n_eq],
                            ineq_owner])
    return owner, col_owner, row_owner


def _sigma(z, mu):
    """One barrier diagonal from the central-path identity ``z·s = μ`` ⇒
    ``Σ = z²/μ``.

    Never the literal ``z/s``: cyipopt returns the primal iterate but keeps its
    own slacks, and at a converged level ``x`` sits exactly on the relaxed
    bound, so ``z/s`` divides by ~0 and yields a 1e306-scale KKT.
    """
    return z * z / max(mu, 1e-300)


def _sigma_exact(z, sz):
    """The **exact** barrier diagonal ``Σ = z/s``, written ``z²/(s·z)`` so that
    the only input needed is the complementarity product IPOPT already publishes
    through ``get_current_violations`` (``compl_x_L`` = ``(x−x_L)·z_L`` etc.).

    Valid only at an *intermediate* iterate, where IPOPT keeps ``s`` strictly
    interior; the floor guards the exactly-zero entries of unbounded components,
    which are masked out by the caller anyway.
    """
    return z * z / np.maximum(np.abs(sz), 1e-300)


def augmented_kkt(prob, x, mult_g, mult_x_L, mult_x_U, xl, xu, mu, *,
                  delta_w=0.0, delta_c=0.0, compl=None):
    """IPOPT's symmetric augmented KKT matrix at ``x``::

        [[H+Σ_x+δ_w,  0,      A_cᵀ,    A_dᵀ  ],
         [0,          Σ_s+δ_w, 0,      −I    ],
         [A_c,        0,      −δ_c I,   0    ],
         [A_d,       −I,       0,     −δ_c I ]]

    square of size ``n + 2·n_ineq + n_eq``.

    ``Σ`` has two modes. By default ``Σ = z²/μ`` — the *central-path* form, forced
    on us at converged iterates because cyipopt returns the primal point but keeps
    its own slacks, and there ``x`` sits exactly on the relaxed bound so ``z/s``
    divides by ~0. When ``compl`` is given — the dict from
    ``Problem.get_current_violations()``, live inside an intermediate callback —
    the **exact** ``Σ = z/s = z²/(s·z)`` is used instead, since ``s·z`` is
    precisely what ``compl_x_L``/``compl_x_U``/``compl_g`` report. The two agree
    to the extent the iterate is on the central path (``s·z ≈ μ``), which is why
    the default is the right approximation at a converged level and the wrong one
    at an intermediate Newton iterate.
    """
    n, n_eq, n_in = prob.n, prob.n_eq, prob.n_ineq
    hr, hc = prob.hessianstructure()
    L = sp.coo_matrix((prob.hessian(x, mult_g, 1.0), (hr, hc)), shape=(n, n)).tocsr()
    H = (L + L.T - sp.diags(L.diagonal())).tocsr()

    jr, jc = prob.jacobianstructure()
    J = sp.coo_matrix((prob.jacobian(x), (jr, jc)), shape=(prob.m_con, n)).tocsr()
    A_c, A_d = J[:n_eq], J[n_eq:]

    sig_x = np.zeros(n)
    fl, fu = xl > -1e19, xu < 1e19
    lam_d = mult_g[n_eq:]
    n_low = n_in - prob.m_q
    if compl is None:
        sig_x[fl] += _sigma(mult_x_L[fl], mu)
        sig_x[fu] += _sigma(mult_x_U[fu], mu)
        # Σ_s for hr|hd|[ha]|comp. hr/hd/ha are lower-bounded rows (a ``g ≥ 0``
        # row returns λ ≤ 0, so the bound-equivalent multiplier is −λ); comp is
        # upper bounded, with v_U = ξ.
        sig_s = np.concatenate([_sigma(-lam_d[:n_low], mu),
                                _sigma(lam_d[n_low:], mu)])
    else:
        # |·| throughout: Σ = z/s is a nonnegative quantity, and taking magnitudes
        # sidesteps having to re-derive IPOPT's sign convention for each of the
        # one-sided row families above.
        cl = np.abs(np.asarray(compl["compl_x_L"], float))
        cu = np.abs(np.asarray(compl["compl_x_U"], float))
        cg = np.abs(np.asarray(compl["compl_g"], float))[n_eq:]
        sig_x[fl] += _sigma_exact(mult_x_L[fl], cl[fl])
        sig_x[fu] += _sigma_exact(mult_x_U[fu], cu[fu])
        sig_s = np.concatenate([_sigma_exact(-lam_d[:n_low], cg[:n_low]),
                                _sigma_exact(lam_d[n_low:], cg[n_low:])])

    I_d = sp.identity(n_in, format="csr")
    return sp.bmat([
        [(H + sp.diags(sig_x + delta_w)).tocsr(), None, A_c.T, A_d.T],
        [None, sp.diags(sig_s + delta_w), sp.csr_matrix((n_in, n_eq)), -I_d],
        [A_c, None, -delta_c * sp.identity(n_eq), None],
        [A_d, -I_d, None, -delta_c * I_d],
    ], format="csr")


class Arrowhead:
    """Bordered block-diagonal view: ``W_k = A[loc_k,loc_k]``,
    ``B_k = A[bord,loc_k]``, ``C = A[bord,bord]``, ``S = C − Σ_k B_k W_k⁻¹ B_kᵀ``.

    The permutation is exact — no reformulation — so the border carries real
    Jacobian entries and ``C`` is nonzero.

    **The local Schur complement is restricted to the rows each subdomain
    actually sees** (``schur="restricted"``, the default). Only ``p_k`` of the
    ``p`` border rows of ``B_k`` are nonzero — subdomain ``k`` touches just the
    cut segments on its own boundary — and if row ``j`` of ``B_k`` vanishes then
    so do row and column ``j`` of ``S_k``. So forming ``S_k`` costs ``p_k`` dense
    backsolves, not ``p``, and the result is scattered into ``S`` through the
    index list ``N_k``. This is exact, not an approximation, and it is the
    difference between ``Σ_k p_k`` and ``n_sub·p`` backsolves — measured
    **7.8× fewer** at N=32 k=4 and **17.2× fewer** at k=6, on the phase that was
    86–90% of the DD cost. (``dd_kkt.py`` does the same via ``Bbar``/``Nk``.)

    ``schur="dense"`` keeps the naive full-``p`` version for A/B.

    **The interface itself is kept sparse** (``s_format="sparse"``, the default).
    ``S`` is the union of the ``N_k×N_k`` cliques plus ``C``, and every entry of
    ``C`` lies inside some clique, so its pattern is known and thin — measured
    **1.45% of the dense triangle at N=128 k=16** (23.3% at N=32 k=4, 5.8% at
    N=64 k=8). Densifying it costs 417 MB and a 1.35e11-flop ``LDLᵀ`` at N=128 to
    carry ~98.5% structural zeros. The blocks are accumulated as COO triplets
    (duplicates summed by ``tocsc``) and factorized by SuperLU; ``s_format=
    "dense"`` keeps the old ``p×p`` array for A/B. (The C++ solver has always
    done this — ``cpp/dd_solver.hpp:421-451`` notes that handing MA57 the dense
    triangle "dominated the whole solve at N=64".)
    """

    def __init__(self, A, owner, n_sub, schur="restricted", s_format="sparse",
                 w_inertia="dense"):
        self.A = A
        self.schur = schur
        self.s_format = s_format
        self.w_inertia = w_inertia
        self.loc = [np.flatnonzero(owner == k) for k in range(n_sub)]
        self.bord = np.flatnonzero(owner == -1)
        self.p = len(self.bord)
        self.W = [A[np.ix_(i, i)].tocsc() for i in self.loc]
        self.B = [A[np.ix_(self.bord, i)].tocsr() for i in self.loc]
        # N_k = the border rows subdomain k actually touches; B̄_k = B_k[N_k].
        self.Nk = [np.flatnonzero(np.diff(Bk.indptr) > 0) for Bk in self.B]
        self.Bbar = [Bk[idx] for Bk, idx in zip(self.B, self.Nk)]
        self.p_k = [len(idx) for idx in self.Nk]
        Cb = A[np.ix_(self.bord, self.bord)]
        self.C = Cb.tocsr() if s_format == "sparse" else np.asarray(Cb.todense())
        self.lu = self.S = self.S_lu = self.inS = None
        self.S_singular = False
        self.t_sfact = self.t_winertia = 0.0
        self.n_w_fallback = 0

    def check_block_diagonal(self) -> bool:
        """No two subdomains may share a nonzero — the whole point of the
        complicating-variable rule."""
        for i in range(len(self.loc)):
            for j in range(i + 1, len(self.loc)):
                if self.A[np.ix_(self.loc[i], self.loc[j])].nnz:
                    return False
        return True

    def factorize(self):
        t0 = time.perf_counter()
        self.lu = [spla.splu(Wk) for Wk in self.W]
        self.t_factor = time.perf_counter() - t0
        self.fill = sum(lu.L.nnz + lu.U.nnz for lu in self.lu)

    def local_schur(self):
        """``S = C − Σ_k B̄_kᵀ-scattered (B̄_k W_k⁻¹ B̄_kᵀ)`` — ``p_k`` backsolves
        per subdomain under the default restriction, ``p`` under ``dense``.

        Assembly follows ``s_format``: COO triplets into a sparse ``S`` (the
        default) or scatter-add into a dense ``p×p`` array.
        """
        t0 = time.perf_counter()
        self.S_k, tgt = [], []            # tgt[k] = border rows S_k scatters to
        if self.schur == "dense":
            allp = np.arange(self.p)
            for lu, Bk in zip(self.lu, self.B):
                self.S_k.append(-(Bk @ lu.solve(np.asarray(Bk.T.todense()))))
                tgt.append(allp)
        else:
            for lu, Bb, idx in zip(self.lu, self.Bbar, self.Nk):
                if not len(idx):
                    self.S_k.append(np.zeros((0, 0)))
                    tgt.append(idx)
                    continue
                # p_k backsolves, not p
                self.S_k.append(-(Bb @ lu.solve(np.asarray(Bb.T.todense()))))
                tgt.append(idx)

        if self.s_format == "dense":
            self.S = self.C.copy()
            for Sk, idx in zip(self.S_k, tgt):
                if len(idx):
                    self.S[np.ix_(idx, idx)] += Sk      # scatter-add by N_k
        else:
            live = [(Sk, i) for Sk, i in zip(self.S_k, tgt) if len(i)]
            if live:
                rows = np.concatenate([np.repeat(i, len(i)) for _, i in live])
                cols = np.concatenate([np.tile(i, len(i)) for _, i in live])
                data = np.concatenate([Sk.ravel() for Sk, _ in live])
            else:
                rows = cols = np.zeros(0, dtype=int)
                data = np.zeros(0)
            # duplicate (i,j) pairs across subdomains are summed by tocsc()
            blocks = sp.coo_matrix((data, (rows, cols)), shape=(self.p, self.p))
            self.S = (self.C + blocks).tocsc()
        self.S_lu = self.inS = None
        self.S_singular = False
        self.t_schur = time.perf_counter() - t0

    def factor_S(self):
        """Factorize the interface and record ``In(S)`` — memoized.

        Sparse mode takes both the factorization and the inertia from one
        symmetric-mode SuperLU call (see ``_sym_lu`` for why the pivot signs are
        ``In(S)``).

        A structurally singular ``S`` makes SuperLU raise; that is recorded as
        ``In(S) = (0, 0, p)``, which fails both tests in ``find_delta_w`` and so
        correctly drives δ_w up.
        """
        if self.inS is not None:
            return
        t0 = time.perf_counter()
        if self.s_format == "dense":
            self.inS = _inertia(self.S)
        else:
            self.S_lu, self.inS = _sym_lu(self.S)
            if self.S_lu is None:
                self.S_singular = True
                self.inS = (0, 0, self.p)
        self.t_sfact += time.perf_counter() - t0

    def S_nnz(self):
        return (int(np.count_nonzero(self.S)) if self.s_format == "dense"
                else int(self.S.nnz))

    def S_dense(self):
        """Densified interface — for plotting only, never for the solve."""
        return (self.S if self.s_format == "dense"
                else np.asarray(self.S.todense()))

    def solve(self, rhs):
        """``r_S = r_y − Σ_k B_k W_k⁻¹ r_k``, one dense interface solve, then
        ``Δx_k = W_k⁻¹(r_k − B_kᵀ Δy)`` — one backsolve each, parallel. Under the
        restriction only the ``N_k`` entries of the border are touched."""
        r_loc = [rhs[i] for i in self.loc]
        r_S = rhs[self.bord].copy()
        if self.schur == "dense":
            for lu, Bk, rk in zip(self.lu, self.B, r_loc):
                r_S -= Bk @ lu.solve(rk)
        else:
            for lu, Bb, idx, rk in zip(self.lu, self.Bbar, self.Nk, r_loc):
                if len(idx):
                    r_S[idx] -= Bb @ lu.solve(rk)
        if self.s_format == "dense":
            dy = np.linalg.solve(self.S, r_S)
        else:
            self.factor_S()
            if self.S_singular:
                raise np.linalg.LinAlgError("interface S is singular")
            dy = self.S_lu.solve(r_S)
        out = np.empty(self.A.shape[0])
        out[self.bord] = dy
        if self.schur == "dense":
            for i, lu, Bk, rk in zip(self.loc, self.lu, self.B, r_loc):
                out[i] = lu.solve(rk - Bk.T @ dy)
        else:
            for i, lu, Bb, idx, rk in zip(self.loc, self.lu, self.Bbar, self.Nk,
                                          r_loc):
                out[i] = lu.solve(rk - (Bb.T @ dy[idx] if len(idx) else 0.0))
        return out

    def inertia(self):
        """``(Σ_k In(W_k) + In(S), [In(W_k)], In(S))`` — Haynsworth additivity
        used as a *computation*: the inertia IPOPT needs for its correction loop
        is assembled from the small local blocks plus ``S``, without ever
        factorizing the full KKT."""
        t0 = time.perf_counter()
        if self.w_inertia == "dense":
            inW = [_inertia(np.asarray(Wk.todense())) for Wk in self.W]
        else:
            inW, self.n_w_fallback = [], 0
            for Wk in self.W:
                _, got = _sym_lu(Wk)
                if got is None:              # singular or asymmetric pivoting
                    self.n_w_fallback += 1
                    got = _inertia(np.asarray(Wk.todense()))
                inW.append(got)
        self.t_winertia += time.perf_counter() - t0
        self.factor_S()
        return tuple(sum(c) for c in zip(*inW, self.inS)), inW, self.inS


def _sym_lu(M):
    """``(factorization, inertia)`` of a sparse symmetric ``M`` from **SuperLU in
    symmetric mode** — the sparse analogue of ``_inertia``, and never a dense
    matrix.

    With the diagonal-pivot threshold at 0 and a symmetric fill-reducing ordering
    SuperLU takes its pivots from the diagonal, so the row and column
    permutations coincide (checked below). ``P M Pᵀ = L U`` is then a
    *congruence* with unit-lower ``L``, which forces ``U = D Lᵀ``, so
    ``sign(diag U)`` is ``In(M)`` by Sylvester's law of inertia. Same pivot-sign
    principle as ``_inertia``, still never eigenvalues, and the same
    factorization then serves the solve.

    Returns ``(None, None)`` when the result cannot be trusted — SuperLU raised
    on an exactly singular factor, or it pivoted off the diagonal so the
    congruence argument no longer holds. Callers decide what to do: ``factor_S``
    treats it as "not SPD, raise δ_w"; ``inertia`` falls back to a dense
    ``LDLᵀ`` for that block.

    **Do not drop the ``perm_r == perm_c`` check.** Without it this silently
    becomes an unpivoted LU on an indefinite matrix, which is exactly the failure
    the C++ hit with ``Eigen::LDLT`` (485–491 negatives where the truth was 512).

    An exact zero on the diagonal forces SuperLU off it, so that case is rejected
    up front rather than paying for a factorization that will be thrown away.
    Measured as a predictor of the ``perm_r != perm_c`` outcome over δ_w sweeps
    at N=16 k=4, 24 k=3, 32 k=4: **no false negatives** (it never lets an
    untrustworthy factorization through — the guard is still the real check) and
    one false positive per sweep, i.e. it is conservative in the safe direction.
    """
    if (M.diagonal() == 0.0).any():
        return None, None
    try:
        lu = spla.splu(M.tocsc(), diag_pivot_thresh=0.0,
                       permc_spec="MMD_AT_PLUS_A",
                       options=dict(SymmetricMode=True))
    except RuntimeError:                        # "Factor is exactly singular"
        return None, None
    if not np.array_equal(lu.perm_r, lu.perm_c):
        return None, None
    d = lu.U.diagonal()
    return lu, (int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()))


def _inertia(M):
    """Inertia from **LDLᵀ pivot signs** (Bunch–Kaufman), never eigenvalues: with
    ``Σ ~ z²/μ`` these matrices reach ``‖A‖ ~ 1e18``, so ``eigvalsh`` only
    resolves down to ``ε‖A‖`` and every small dual eigenvalue gets a random
    sign. This is also what IPOPT's own inertia test uses (MA57 pivot signs)."""
    _, d, _ = sla.ldl(M)
    n = d.shape[0]
    pos = neg = zer = 0
    i = 0
    while i < n:
        if i + 1 < n and d[i, i + 1] != 0.0:
            w = np.linalg.eigvalsh(d[i:i + 2, i:i + 2])
            i += 2
        else:
            w = np.array([d[i, i]])
            i += 1
        pos += int((w > 0).sum())
        neg += int((w < 0).sum())
        zer += int((w == 0).sum())
    return pos, neg, zer


def target_inertia(prob):
    """IPOPT's inertia condition ``In(A) = (n_var + n_ineq, n_eq + n_ineq, 0)``
    (Wächter–Biegler eq. 9): every primal and slack direction positive, every
    multiplier direction negative, nothing singular."""
    return (prob.n + prob.n_ineq, prob.n_eq + prob.n_ineq, 0)


def find_delta_w(build, prob, d0=1e-4, kappa=8.0, d_max=1e10):
    """IPOPT's inertia-correction loop (Alg. IC), driven by Haynsworth: raise
    ``δ_w`` — from ``1e-4``, ×8, IPOPT's own ``δ̄_w`` and ``κ_w⁺`` — until the
    *distributed* inertia ``Σ_k In(W_k) + In(S)`` hits the target and ``S`` is
    SPD."""
    target, delta_w, trials = target_inertia(prob), 0.0, 0
    while True:
        arrow = build(delta_w)
        got, _, inS = arrow.inertia()
        trials += 1
        if got == target and inS == (arrow.p, 0, 0):
            return delta_w, arrow, trials, True
        if delta_w > d_max:
            return delta_w, arrow, trials, False
        delta_w = d0 if delta_w == 0.0 else delta_w * kappa


def dd_probe(prob, owner, n_sub, x, info, mu, *, seed=0, check_solve=True,
             max_dense=DENSE_CHECK_MAX_DIM, compl=None, delta_c=None,
             schur="restricted", s_format="sparse", w_inertia="dense"):
    """The full arrowhead/Haynsworth probe of the KKT **at one iterate**.

    Split out of ``main`` so the same probe serves three clocks: every Scholtes
    level, every Newton iteration, and the reported iterate. Only the numerics
    depend on where we are — ``owner``, hence ``loc``/``bord``/``p`` and the whole
    block structure, comes from ``kkt_owner``'s Jacobian-*sparsity* rule, which is
    structural and so independent of ``x``, ``t`` and ``μ``. The caller builds it
    once (``main`` already did, for the bookkeeping print) and every probe reuses
    it; what is rebuilt per call is ``H(x,λ)``, ``J(x)``, ``Σ`` and ``δ_c``.

    ``δ_c`` defaults to IPOPT's Alg. IC value ``√ε·μ^¼``, applied unconditionally
    — IPOPT itself only reaches for it on a *detected* rank-deficient Jacobian, so
    ``delta_c=0.0`` is the A/B lever for whether ours pre-empts a δ_w IPOPT needs.
    Measured in 1D: δ_c=0 is much worse (agreement 84/107 vs 97/108, and three
    iterates where δ_w runs to 1e10 without correcting) — don't ship it as default.

    ``compl`` — the ``get_current_violations()`` dict, available only inside an
    intermediate callback — switches ``Σ`` from the central-path ``z²/μ`` to the
    exact ``z/s``. Pass it whenever it exists: at an intermediate iterate the two
    are genuinely different matrices.

    Returns ``(arrow, record)``; the record is one row of the path table.
    """
    xl, xu = bounds(prob)
    mg = np.asarray(info["mult_g"], float)
    zl = np.asarray(info["mult_x_L"], float)
    zu = np.asarray(info["mult_x_U"], float)
    if delta_c is None:
        delta_c = float(np.sqrt(np.finfo(float).eps) * max(mu, 0.0) ** 0.25)
    delta_c = float(delta_c)

    def build(dw):
        A = augmented_kkt(prob, x, mg, zl, zu, xl, xu, mu,
                          delta_w=dw, delta_c=delta_c, compl=compl)
        arrow = Arrowhead(A, owner, n_sub, schur=schur, s_format=s_format,
                          w_inertia=w_inertia)
        arrow.factorize()
        arrow.local_schur()
        return arrow

    delta_w, arrow, trials, corrected = find_delta_w(build, prob)
    tot, inW, inS = arrow.inertia()
    dim = arrow.A.shape[0]
    rec = {
        "t": prob.t_comp, "mu": mu, "delta_c": delta_c, "delta_w": delta_w,
        "trials": trials, "corrected": corrected, "p": arrow.p,
        "inertia": tot, "inW": inW, "inS": inS, "target": target_inertia(prob),
        "spd": inS == (arrow.p, 0, 0), "fill": arrow.fill,
        "decoupled": arrow.check_block_diagonal(),
        "sigma": "exact z/s" if compl is not None else "z²/μ",
        "haynsworth": None, "err": float("nan"),
    }
    rec["hit"] = tot == rec["target"]
    # O(dim²) — 2.2 GB at N=32, hence the guard the monolithic check already had.
    if dim <= max_dense:
        rec["haynsworth"] = tot == _inertia(np.asarray(arrow.A.todense()))
    if check_solve:
        rng = np.random.default_rng(seed)
        rhs = rng.standard_normal(dim)
        ref = spla.splu(arrow.A.tocsc()).solve(rhs)
        rec["err"] = float(np.linalg.norm(arrow.solve(rhs) - ref)
                           / np.linalg.norm(ref))
    return arrow, rec


def report_dd_path(records):
    """The per-Scholtes-level DD table.

    Probing only the reported iterate tests a *point*; the claim being made is
    about the *path*, and the path is where the degeneracy bites — as ``t ↓ 0``
    the biactive set grows, ``Σ`` blows up, and nothing forces ``S`` to stay SPD.
    Each level is probed at its converged iterate, i.e. at its smallest μ and so
    its largest ``Σ``: the worst-conditioned point of that level.
    """
    if not records:
        return
    print("\n  domain decomposition along the Scholtes path "
          f"(probed at each level's converged iterate, p={records[0]['p']}):")
    print(f"  {'t':>9} {'mu':>9} {'δ_c':>9} {'δ_w':>9} {'tri':>4} "
          f"{'In(S)':>16} {'SPD':>4} {'target':>7} {'Haynsw':>8} {'rel-err':>9}")
    for r in records:
        hay = {True: "MATCH", False: "MISMATCH", None: "skipped"}[r["haynsworth"]]
        print(f"  {r['t']:>9.1e} {r['mu']:>9.1e} {r['delta_c']:>9.1e} "
              f"{r['delta_w']:>9.1e} {r['trials']:>4d} {str(r['inS']):>16} "
              f"{'yes' if r['spd'] else 'NO':>4} "
              f"{'hit' if r['hit'] else 'MISSED':>7} {hay:>8} {r['err']:>9.1e}"
              f"{'' if r['converged'] else '   [level failed]'}")


def report_dd_iters(records, stride=1, *, max_rows=30):
    """Print the per-Newton-iteration DD table and settle the δ_w alignment.

    This is the honest comparison against IPOPT's inertia correction: the
    per-*level* table only ever sees converged iterates, where IPOPT has long
    since stopped regularizing, so it reports ``δ_w = 0`` even on runs whose
    ``δw`` column is nonzero. Correction happens at the intermediate Newton
    iterates, and that is also the only place a real DD linear solver would ever
    be called.

    **The off-by-one is measured, not assumed.** ``intermediate`` fires at the
    *end* of iteration ``k``: ``regularization_size`` is the δ_w IPOPT used for
    the step just taken, while ``get_current_iterate`` returns the point that step
    landed on — so our δ_w at record ``k`` is a prediction of IPOPT's value at
    record ``k+1``. Both shifts are scored on decision agreement ("was any
    regularization needed"), over consecutive-iteration pairs inside one level,
    and the better one is reported. Only meaningful at ``stride == 1``.
    """
    if not records:
        return
    n_reg_ip = sum(r["regu"] > 0 for r in records)
    n_reg_dd = sum(r["delta_w"] > 0 for r in records)
    print(f"\n  per-Newton-iteration probe: {len(records)} iterate(s) "
          f"(stride {stride}, Σ = {records[0]['sigma']}), IPOPT regularized "
          f"{n_reg_ip}, DD asked for δ_w > 0 at {n_reg_dd}")

    bad = [r for r in records if not r["hit"] or not r["spd"]
           or r["haynsworth"] is False]
    print(f"    inertia target hit + S SPD + Haynsworth: "
          f"{len(records) - len(bad)}/{len(records)}")

    if stride == 1:
        def score(shift):
            hit = tot = 0
            for a, b in zip(records, records[shift:]):
                if shift and (a["t"] != b["t"] or b["iter"] != a["iter"] + 1):
                    continue
                tot += 1
                hit += int((a["delta_w"] > 0) == (b["regu"] > 0))
            return hit, tot
        s0, s1 = score(0), score(1)
        best = "+1 (probe predicts the NEXT callback)" if (
            s1[1] and s1[0] * s0[1] >= s0[0] * s1[1]) else "0 (same callback)"
        print(f"    δ_w decision agreement with IPOPT:  shift 0: {s0[0]}/{s0[1]}"
              f"   shift +1: {s1[0]}/{s1[1]}   → alignment {best}")

    rows = [r for r in records if r["delta_w"] > 0 or r["regu"] > 0
            or not r["hit"] or not r["spd"]]
    if not rows:
        print("    no iterate needed regularization on either side — "
              "nothing to tabulate")
        return
    shown = rows[:max_rows]
    print(f"    iterates where either side regularized (or DD missed) "
          f"[{len(shown)} of {len(rows)}]:")
    print(f"      {'t':>9} {'iter':>5} {'mu':>9} {'IPOPT δ_w':>10} "
          f"{'DD δ_w':>10} {'tri':>4} {'SPD':>4} {'target':>7} {'Haynsw':>8}")
    for r in shown:
        hay = {True: "MATCH", False: "MISMATCH", None: "skipped"}[r["haynsworth"]]
        print(f"      {r['t']:>9.1e} {r['iter']:>5d} {r['mu']:>9.1e} "
              f"{r['regu']:>10.1e} {r['delta_w']:>10.1e} {r['trials']:>4d} "
              f"{'yes' if r['spd'] else 'NO':>4} "
              f"{'hit' if r['hit'] else 'MISSED':>7} {hay:>8}"
              f"{'   [restoration]' if r['alg_mod'] else ''}")


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plt(show: bool):
    """Import pyplot with a headless backend unless we are actually showing."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def index_sets(r, w, eps):
    """Classify each cell by the MPCC complementarity ``0 ≤ r ⊥ w ≥ 0``,
    ``w := 1 − δ``, at resolution ``eps``.

    Returns an integer map: ``0`` inactive (flat: ``r ≈ 0``, dual strictly inside
    the ball), ``1`` active (a jump: ``r > 0`` with the dual pinned to
    ``δ = 1``), ``2`` **biactive** — both sides ≈ 0, the degenerate corner where
    the MPCC constraint qualifications fail and which decides the
    C-/M-/S-stationarity distinction.

    The three sets are exhaustive at any feasible point once ``eps ≥ √t``:
    ``r > eps`` and ``w > eps`` would give ``r·w > eps² ≥ t``, violating the
    Scholtes row. That is exactly why the corner is resolved at the ``O(√t)``
    scale and not at machine zero — δ never reaches 1 (the barrier keeps it
    interior and the relaxation only forces ``1−δ ≲ t/r``), so a ``1e-6`` test
    would classify nothing as active.
    """
    act = (r > eps) & (w <= eps)
    bi = (r <= eps) & (w <= eps)
    out = np.zeros(r.shape, dtype=int)
    out[act] = 1
    out[bi] = 2
    return out


def plot_solution(prob, x, t_last, history, *, path=None, show=False, title=None,
                  cuts=None, mu_trace=None):
    """Twelve-panel view of the solution on the staggered mesh.

    Row 1 — the ``N×N`` **node** mesh: clean, noisy, reconstruction, error.
    Row 2 — the ``(N−1)²`` **cell** mesh: the lift ``r = |∇u|``, the dual radius
    ``δ = |q|``, the MPCC index sets, and the polar angle ``θ``.
    Row 3 — the MPCC itself: the dual vector field over the reconstruction, the
    complementarity scatter, the per-cell residual ``r(1−δ)``, and the
    continuation path.

    The two meshes differ by one in each direction — that is the staggering,
    made visible; each title says which mesh it is on.

    Everything corner-related is judged at the ``O(√t)`` scale via the repo's cap
    ``ε_w = min(3√t, ½)`` (see :func:`index_sets`), never at machine zero.

    ``cuts`` — optional ``(cut_rows, cut_cols)`` of the DD partition, overlaid as
    dashed lines on every image panel (the 2D analogue of the 1D figure's
    subdomain bands). The value convention is shared with :func:`plot_domains`:
    ``c`` = the first cell/node index AFTER the cut, line drawn at ``c − 0.5`` —
    valid on both meshes at once because the cell after a cut and the node after
    it share the index under the anchor rule. ``None`` leaves the figure exactly
    as before.

    ``mu_trace`` — optional ``(n, 3)`` array of ``(iter, μ, t)`` per IPOPT
    iteration from a μ-coupled single solve (the C++ driver's ``--t-update mu``,
    the default there). When given, the continuation panel draws the IN-SOLVE
    homotopy — ``t`` stepping down with the barrier over the iterations — since
    there are no outer levels to tabulate (only the final ``t`` is a converged
    point; intermediate ``t`` values are passed through, never certified).
    ``None`` (the Python driver, geometric C++ runs) keeps the per-level panel.
    """
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
    plt = _plt(show)
    N, nc, off = prob.N, prob.N - 1, prob.off
    m_q = prob.m_q
    u = x[: prob.m_u].reshape(N, N)
    clean = prob.u_clean.reshape(N, N)
    noisy = prob.f.reshape(N, N)
    r = x[off["r"] : off["r"] + m_q].reshape(nc, nc)
    d = x[off["delta"] : off["delta"] + m_q].reshape(nc, nc)
    qx = x[off["qx"] : off["qx"] + m_q].reshape(nc, nc)
    qy = x[off["qy"] : off["qy"] + m_q].reshape(nc, nc)
    th = x[off["theta"] : off["theta"] + m_q].reshape(nc, nc)
    w = 1.0 - d
    comp = r * w
    eps_w = min(3.0 * np.sqrt(max(t_last, 0.0)), 0.5)
    iset = index_sets(r, w, eps_w)
    # D1 gauge set: θ is undetermined where r ≈ 0 AND δ ≈ 0 (both h2 and h3 lose
    # their θ dependence). Note this is NOT the biactive set, which is r ≈ 0 with
    # δ ≈ 1 — the two degeneracies live at opposite ends of the ball.
    gauge = (r <= eps_w) & (d <= eps_w)

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 12.0))
    vmin, vmax = float(min(clean.min(), noisy.min(), u.min())), \
                 float(max(clean.max(), noisy.max(), u.max()))
    for ax, img, ttl in zip(
            axes[0], (clean, noisy, u),
            (f"clean $u^\\dagger$   ({N}$\\times${N} nodes)",
             f"noisy $f$   {psnr(prob.u_clean, prob.f):.2f} dB",
             f"recon $u$   {psnr(prob.u_clean, x[:prob.m_u]):.2f} dB")):
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    err = u - clean
    lim = float(np.abs(err).max()) or 1.0
    im = axes[0, 3].imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[0, 3].set_title(f"$u - u^\\dagger$   (max {lim:.3f})", fontsize=9)
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(r, cmap="magma")
    axes[1, 0].set_title(f"$r = |\\nabla u|$   ({nc}$\\times${nc} cells)", fontsize=9)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(d, cmap="viridis", vmin=0.0, vmax=1.0)
    n_sat = int((d >= 1.0 - eps_w).sum())
    if 0 < n_sat < d.size:
        ax.contour(d, levels=[1.0 - eps_w], colors="w", linewidths=0.8)
    ax.set_title(f"$\\delta = |q|$   saturated {n_sat}/{d.size}\n"
                 f"(white: $1-\\epsilon_w$, $\\epsilon_w$={eps_w:.1e})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- MPCC index sets ---------------------------------------------------
    ax = axes[1, 2]
    iso_cmap = ListedColormap(["#2c7fb8", "#f0c419", "#d7191c"])
    ax.imshow(iset, cmap=iso_cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3))
    n_in, n_ac, n_bi = [int((iset == v).sum()) for v in (0, 1, 2)]
    handles = [plt.Line2D([], [], marker="s", ls="", ms=8, mfc=c, mec="none", label=l)
               for c, l in zip(iso_cmap.colors,
                               (f"inactive: $w>\\epsilon_w$ ({n_in})",
                                f"active: $r>\\epsilon_w\\geq w$ ({n_ac})",
                                f"biactive: both $\\leq\\epsilon_w$ ({n_bi})"))]
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", framealpha=0.9)
    ax.set_title(f"MPCC index sets at $\\epsilon_w$={eps_w:.1e}\n"
                 f"biactive = degenerate corner ({n_bi / max(iset.size, 1):.0%})",
                 fontsize=9)

    # --- polar angle θ, with the D1 gauge set masked -----------------------
    ax = axes[1, 3]
    th_m = np.ma.masked_where(gauge, np.mod(th, 2 * np.pi))
    cmap_c = plt.get_cmap("twilight").with_extremes(bad="0.75")
    im = ax.imshow(th_m, cmap=cmap_c, vmin=0.0, vmax=2 * np.pi)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0, np.pi, 2 * np.pi])
    cb.ax.set_yticklabels(["0", "$\\pi$", "2$\\pi$"])
    ax.set_title(f"$\\theta = \\angle q$   (cyclic)\n"
                 f"grey: gauge undetermined, $r=\\delta=0$ ({int(gauge.sum())})",
                 fontsize=9)

    # --- dual vector field over the reconstruction -------------------------
    ax = axes[2, 0]
    ax.imshow(u, cmap="gray", vmin=vmin, vmax=vmax)
    step = max(1, int(np.ceil(nc / 20)))   # ~20 arrows per side, else they mat
    I, J = np.mgrid[0:nc:step, 0:nc:step]
    ax.quiver(J + 0.5, I + 0.5, qx[::step, ::step], qy[::step, ::step],
              color="C1", angles="xy", scale_units="xy", scale=1.4, width=0.004)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    sub = "every cell" if step == 1 else f"every {step}th cell"
    ax.set_title(f"dual field $q=(q_x,q_y)$ on $u$\n"
                 f"($q \\parallel \\nabla u$ where $|\\nabla u|>0$; {sub})",
                 fontsize=9)

    # --- the complementarity scatter --------------------------------------
    ax = axes[2, 1]
    # Floor only what is exactly zero, and set the window from the data — a fixed
    # 1e-12 decade floor wastes most of the axis on empty space.
    both = np.concatenate([r.ravel(), w.ravel()])
    pos_both = both[both > 0]
    flo = max(float(pos_both.min()) / 10 if pos_both.size else 1e-12, 1e-16)
    n_zero = int((both <= 0).sum())
    rr = np.maximum(r.ravel(), flo)
    ww = np.maximum(w.ravel(), flo)
    for v, col, lab in zip((0, 1, 2), iso_cmap.colors,
                           ("inactive", "active", "biactive")):
        sel = iset.ravel() == v
        if sel.any():
            ax.plot(rr[sel], ww[sel], ".", ms=3, color=col, label=lab, alpha=0.8)
    hi = float(max(rr.max(), ww.max())) * 2
    grid = np.logspace(np.log10(flo), np.log10(hi), 200)
    if t_last > 0:                     # the Scholtes relaxation boundary r·w = t
        ax.plot(grid, t_last / grid, "k-", lw=1.2, label=f"$r\\,w = t$")
    ax.axvline(eps_w, color="0.5", ls=":", lw=1.0)
    ax.axhline(eps_w, color="0.5", ls=":", lw=1.0, label="$\\epsilon_w$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(flo / 2, hi)
    ax.set_ylim(flo / 2, hi)
    ax.set_xlabel("$r = |\\nabla u|$", fontsize=8)
    ax.set_ylabel("$w = 1-\\delta$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    # What this panel is for: points ON the line r·w = t mean the Scholtes row is
    # ACTIVE — the iterate sits on the relaxation boundary, i.e. O(√t) away from
    # the true corner — whereas points collapsing onto the two axes (the classic
    # MPCC "L") mean exact complementarity. Watch the cloud migrate from the line
    # to the axes as t ↓ 0.
    n_on = int((comp >= 0.5 * t_last).sum())
    zmsg = f", {n_zero} exact 0 clipped" if n_zero else ""
    ax.set_title(f"complementarity $0 \\leq r \\perp w \\geq 0$\n"
                 f"{n_on}/{m_q} on the relaxation $rw=t${zmsg}", fontsize=9)

    ax = axes[2, 2]
    pos = comp[comp > 0]
    lo = max(pos.min() if pos.size else t_last * 1e-7, t_last * 1e-7)
    im = ax.imshow(np.maximum(comp, lo), cmap="cividis",
                   norm=LogNorm(vmin=lo, vmax=max(comp.max(), t_last)))
    ax.set_title(f"$r(1-\\delta)$   $\\leq t$ = {t_last:.1e}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in (axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3], axes[2, 0],
               axes[2, 2]):
        ax.set_xticks([])
        ax.set_yticks([])

    # --- DD partition overlay (opt-in) -------------------------------------
    if cuts is not None:
        cut_rows, cut_cols = cuts
        for ax in (*axes[0], *axes[1], axes[2, 0], axes[2, 2]):
            for c in cut_cols:
                ax.axvline(c - 0.5, color="C3", ls="--", lw=0.7, alpha=0.75)
            for c in cut_rows:
                ax.axhline(c - 0.5, color="C3", ls="--", lw=0.7, alpha=0.75)

    ax = axes[2, 3]
    if mu_trace is not None and len(mu_trace) and mu_trace.shape[1] >= 5:
        # μ-coupled single solve: the continuation happens INSIDE the solve —
        # t = max(t_min, c·μ) tightened every iteration — so the panel keeps
        # the geometric version's story (weight converging, complementarity
        # tightening under the dotted t) but on the ITERATION axis, since t
        # plateaus at the floor while the solution still moves. weight/comp
        # are NaN on restoration-phase iterations and plot as honest gaps.
        # Only the final iterate is a converged point.
        it, mus, ts, ws, cr = (mu_trace.T)[:5]
        ax.plot(it, ws, "-", color="C0", lw=1.4, label="weight $Q(\\alpha)$")
        ax.set_xlabel("IPOPT iteration", fontsize=8)
        ax.set_ylabel("weight", color="C0", fontsize=8)
        ax.tick_params(axis="y", labelcolor="C0", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax2 = ax.twinx()
        with np.errstate(invalid="ignore"):
            crp = np.where(cr > 0, cr, np.nan)   # log axis: mask 0/NaN
        ax2.plot(it, crp, "-", color="C2", lw=1.2)
        ax2.plot(it, ts, ":", lw=1.1, color="0.5", drawstyle="steps-post")
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$\max\, r(1-\delta)$  (dotted: $t$)", color="C2",
                       fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C2", labelsize=7)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("continuation path ($\\mu$-coupled, in-solve: "
                     "$t=\\max(t_{\\min},\\, c\\,\\mu)$)", fontsize=9)
    elif mu_trace is not None and len(mu_trace):
        # early 3-column (iter, μ, t) trace: no per-iterate weight/comp — draw
        # the schedule itself.
        it, mus, ts = (mu_trace.T)[:3]
        ax.plot(it, ts, drawstyle="steps-post", color="C0", lw=1.5,
                label="Scholtes $t$")
        ax.plot(it, mus, "-", color="C2", lw=1.0, alpha=0.85,
                label="barrier $\\mu$")
        ax.axhline(ts[-1], color="0.5", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("IPOPT iteration", fontsize=8)
        ax.set_ylabel("$t$, $\\mu$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("continuation path ($\\mu$-coupled, in-solve: "
                     "$t=\\max(t_{\\min},\\, c\\,\\mu)$)", fontsize=9)
    else:
        if history:
            ts = np.array([h["t"] for h in history])
            ws = np.array([h["weight"] for h in history])
            cr = np.array([h["comp_res"] for h in history])
            okm = np.array([h["converged"] for h in history])
            ax.plot(ts, ws, "-o", ms=4, color="C0", label="weight $Q(\\alpha)$")
            if (~okm).any():
                ax.plot(ts[~okm], ws[~okm], "x", ms=10, mew=2, color="C3",
                        label="level failed")
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("Scholtes level $t$   (tightening →)", fontsize=8)
            ax.set_ylabel("weight", color="C0", fontsize=8)
            ax.tick_params(axis="y", labelcolor="C0", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax2 = ax.twinx()
            ax2.plot(ts, cr, "-s", ms=4, color="C2")
            ax2.plot(ts, ts, ":", lw=1.0, color="0.5")
            ax2.set_yscale("log")
            ax2.set_ylabel(r"$\max\, r(1-\delta)$  (dotted: $t$)", color="C2",
                           fontsize=8)
            ax2.tick_params(axis="y", labelcolor="C2", labelsize=7)
            ax.legend(fontsize=7, loc="best")
        ax.set_title("continuation path", fontsize=9)

    fig.suptitle(title or "", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_arrowhead(prob, col_owner, arrow, *, path=None, show=False, title=None):
    """The DD structure: the KKT before and after the arrowhead permutation.

    Left is IPOPT's augmented KKT in its natural ordering; middle is the *same
    matrix* permuted by ``[loc_1 … loc_k | border]`` — no reformulation, so the
    picture is the honest bordered block-diagonal with the border arms carrying
    real Jacobian entries. Right is the assembled interface matrix ``S``, the
    dense ``p×p`` system the method actually has to solve; its border ordering
    is by column index, so it splits into ``u | qx | qy | α`` groups, drawn with
    separators (``p`` is far too large here for per-variable tick labels).
    """
    plt = _plt(show)
    perm = np.concatenate(arrow.loc + [arrow.bord])
    Ap = arrow.A[perm][:, perm]
    ms = max(0.08, min(2.5, 400.0 / arrow.A.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    axes[0].spy(arrow.A, markersize=ms, color="C0")
    axes[0].set_title(f"augmented KKT, natural order\n{arrow.A.shape[0]}$\\times$"
                      f"{arrow.A.shape[0]}, nnz={arrow.A.nnz}", fontsize=9)

    ax = axes[1]
    ax.spy(Ap, markersize=ms, color="C0")
    o = 0
    for k, idx in enumerate(arrow.loc):
        s = len(idx)
        ax.add_patch(plt.Rectangle((o - .5, o - .5), s, s, fill=False,
                                   ec="C2", lw=1.0))
        if len(arrow.loc) <= 4:
            ax.text(o + s / 2, o + s / 2, f"$W_{{{k + 1}}}$", color="C2",
                    fontsize=11, ha="center", va="center")
        o += s
    ax.add_patch(plt.Rectangle((o - .5, -.5), arrow.p, o, fill=False, ec="C3", lw=1.2))
    ax.add_patch(plt.Rectangle((-.5, o - .5), o, arrow.p, fill=False, ec="C3", lw=1.2))
    ax.set_title(f"permuted to arrowhead — {len(arrow.loc)} blocks + border "
                 f"$p$={arrow.p}\n(pure permutation, border in red)", fontsize=9)

    # --- the interface matrix, grouped by variable kind --------------------
    ax = axes[2]
    S = np.abs(arrow.S_dense())
    logS = np.where(S > 0, np.log10(np.maximum(S, 1e-300)), np.nan)
    cmap = plt.get_cmap("viridis").with_extremes(bad="0.9")
    hi = np.nanmax(logS) if np.isfinite(logS).any() else 0.0
    im = ax.imshow(logS, cmap=cmap, vmin=hi - 12, vmax=hi)
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}|S|$  (grey = structural 0)")
    # border indices are ordered by column index: u | qx | qy | α
    bord = arrow.bord
    m_u, m_q, off = prob.m_u, prob.m_q, prob.off
    edges, labs = [], []
    # The trailing λ group is empty for this script's own probe (its border is
    # purely primal, by the complicating-column rule) and non-empty only when the
    # arrowhead comes from cpp/dd_solve_2d, which additionally promotes the
    # (λ_h3x, λ_h3y) pair of each cut-corner cell — 2(k−1)² dual indices ≥ n. The
    # `if cnt:` guard below drops it in the former case, so the probe's figure is
    # unchanged; without the group the separators would silently mis-align.
    for name, lo_c, hi_c in (("$u$", 0, m_u),
                             ("$q_x$", off["qx"], off["qx"] + m_q),
                             ("$q_y$", off["qy"], off["qy"] + m_q),
                             (r"$\alpha$", off["alpha"], off["alpha"] + 1),
                             (r"$\lambda_{h3}$", prob.n,
                              prob.n + 2 * prob.n_ineq + prob.n_eq)):
        cnt = int(((bord >= lo_c) & (bord < hi_c)).sum())
        if cnt:
            edges.append(cnt)
            labs.append(name)
    pos = np.cumsum([0] + edges)
    for b in pos[1:-1]:
        ax.axvline(b - 0.5, color="w", lw=1.0)
        ax.axhline(b - 0.5, color="w", lw=1.0)
    ax.set_xticks([(pos[i] + pos[i + 1] - 1) / 2 for i in range(len(edges))])
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_yticks([(pos[i] + pos[i + 1] - 1) / 2 for i in range(len(edges))])
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_title(f"interface $S = C - \\sum_k B_k W_k^{{-1}} B_k^\\top$\n"
                 f"{arrow.p}$\\times${arrow.p}, blocks "
                 f"{'|'.join(str(e) for e in edges)}", fontsize=9)

    fig.suptitle(title or "", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_domains(prob, part, col_owner, arrow=None, *, path=None, show=False,
                 title=None, promoted=None):
    """The domain map — the 2D analogue of the 1D subdomain bands.

    Left: the tiling drawn on both meshes at once (nodes as the background
    image, cell centres at the half-integer offsets), with the **border nodes**
    and **border cells** highlighted. This is the picture of the
    complicating-variable rule: the cut lines pick up the first ``u`` past each
    cut and the last ``q`` before it. Right: the per-subdomain block sizes, which
    is where the load imbalance shows (the clamped anchor rule gives the first
    tile fewer nodes).

    ``part`` is duck-typed: it needs ``node_owner_2d`` plus either ``cut_nodes``
    (the k×k tile case — cuts in both directions) or explicit ``cut_rows`` /
    ``cut_cols`` and a ``label`` (the C++ adapter's dump-derived partition, which
    is how a ``--partition strip`` run draws only its horizontal cuts).
    ``promoted`` optionally marks the cells whose dual pairs the C++ solver
    promoted to the border (cut-corner cells; empty for this script's own probe).
    """
    plt = _plt(show)
    N, nc, off = prob.N, prob.N - 1, prob.off
    m_u, m_q = prob.m_u, prob.m_q
    bn = np.flatnonzero(col_owner[: m_u] < 0)                        # border nodes
    bqx = np.flatnonzero(col_owner[off["qx"] : off["qx"] + m_q] < 0)  # border qx cells
    bqy = np.flatnonzero(col_owner[off["qy"] : off["qy"] + m_q] < 0)
    cut_rows = list(getattr(part, "cut_rows", getattr(part, "cut_nodes", [])))
    cut_cols = list(getattr(part, "cut_cols", getattr(part, "cut_nodes", [])))
    label = getattr(part, "label", f"{part.k}$\\times${part.k} tiles")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.imshow(part.node_owner_2d, cmap="tab20", alpha=0.35, interpolation="nearest")
    for c in cut_cols:                             # cut lines, in node coordinates
        ax.axvline(c - 0.5, color="C3", ls="--", lw=1.2)
    for c in cut_rows:
        ax.axhline(c - 0.5, color="C3", ls="--", lw=1.2)
    # cells sit at the half-integer offsets relative to the node image
    ax.plot((bqx % nc) + 0.5, (bqx // nc) + 0.5, "s", ms=3.5, mfc="none",
            mec="C0", mew=1.0, label=f"border $q_x$ cell ({len(bqx)})")
    ax.plot((bqy % nc) + 0.5, (bqy // nc) + 0.5, "D", ms=3.0, mfc="none",
            mec="C2", mew=1.0, label=f"border $q_y$ cell ({len(bqy)})")
    ax.plot(bn % N, bn // N, "o", ms=3.0, color="C3",
            label=f"border node $u$ ({len(bn)})")
    if promoted is not None and len(promoted):
        promoted = np.asarray(promoted)
        ax.plot((promoted % nc) + 0.5, (promoted // nc) + 0.5, "*", ms=9,
                mfc="C1", mec="k", mew=0.4,
                label=f"promoted-dual cell ({len(promoted)})")
    # The honest p is the arrowhead's when one is given (it includes the promoted
    # dual entries); the primal count is the probe's own rule.
    p_true = arrow.p if arrow is not None else len(bn) + len(bqx) + len(bqy) + 1
    ax.set_title(f"domain map — {label} on "
                 f"{nc}$\\times${nc} cells / {N}$\\times${N} nodes\n"
                 f"p = {p_true} (incl. $\\alpha$)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    if arrow is not None:
        dims = [len(i) for i in arrow.loc]
        ax.bar(range(len(dims)), dims, color="C0")
        ax.axhline(np.mean(dims), color="C3", ls="--", lw=1.2,
                   label=f"mean {np.mean(dims):.0f}")
        ax.set_xticks(range(len(dims)))
        if len(dims) > 16:
            ax.tick_params(axis="x", labelsize=6, rotation=90)
        ax.set_xlabel("subdomain $k$")
        ax.set_ylabel("local block dim  $|W_k|$")
        imbalance = (max(dims) - min(dims)) / max(np.mean(dims), 1.0)
        ax.set_title(f"local block sizes — border p={arrow.p}, "
                     f"imbalance {imbalance:.0%}\n"
                     f"(total {sum(dims)} + {arrow.p} = {arrow.A.shape[0]})",
                     fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no arrowhead (--no-dd)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.5")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title or "", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="2D lifted TV-MPCC on a staggered node/cell grid, with "
                    "domain decomposition and graphics.")
    ap.add_argument("--data", default=DEFAULT_IMAGE, help="image to denoise")
    ap.add_argument("--phantom", action="store_true",
                    help="use the synthetic phantom instead of --data")
    ap.add_argument("--N", type=int, default=16, help="nodes per side (cells = N−1)")
    ap.add_argument("--nsub", type=int, default=2,
                    help="tiles per direction (k² subdomains)")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stencil", default="onesided", choices=["onesided", "averaged"])
    ap.add_argument("--weight", choices=["linear", "exp"], default="linear",
                    help="Q(α) = α (board, with an α ≥ 0 row) or e^α")
    ap.add_argument("--alpha0", type=float, default=None,
                    help="initial weight (linear) or log-weight (exp); default 0.7·σ")
    ap.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha")
    ap.add_argument("--w-max", type=float, default=float("inf"), dest="w_max",
                    help="optional upper cap on the linear weight (default none)")
    ap.add_argument("--c-theta", type=float, default=0.0, dest="c_theta",
                    help="TR gauge ridge ½·(c_θ·t)·‖θ − θ_ref‖² (the D1 fix)")
    ap.add_argument("--init", choices=["cp", "cp-scan", "cold"], default="cp")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--t-min", type=float, default=1e-4, dest="t_min")
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=1500, dest="max_iter")
    ap.add_argument("--linear-solver", default="ma57", dest="linear_solver")
    ap.add_argument("--hess-update", default="exact", dest="hess_update",
                    choices=["exact", "bfgs", "sr1"])
    ap.add_argument("--dual-warmstart", action="store_true", dest="dual_warmstart")
    ap.add_argument("--print-level", type=int, default=0, dest="print_level")
    ap.add_argument("--no-solve", action="store_true", dest="no_solve")
    ap.add_argument("--no-dd", action="store_true", dest="no_dd",
                    help="skip the domain-decomposition probe entirely")
    ap.add_argument("--no-dd-path", action="store_false", dest="dd_path",
                    help="probe only the reported iterate, not every "
                         "continuation level (the path table is the point, "
                         "so it is on by default)")
    ap.add_argument("--dd-iters", type=int, default=0, dest="dd_iters",
                    metavar="STRIDE",
                    help="also probe every STRIDE-th IPOPT iteration, where the "
                         "inertia correction actually happens (1 = every "
                         "iteration; 0 = off, the default — this is the "
                         "expensive clock, and in 2D it is expensive: budget "
                         "roughly one extra DD solve per probed iteration)")
    ap.add_argument("--dd-sigma-mu", action="store_true", dest="dd_sigma_mu",
                    help="per-iteration probe: use the central-path Σ = z²/μ "
                         "instead of the exact Σ = z/s from "
                         "get_current_violations (an A/B lever — it makes the "
                         "probe factorize a different matrix from IPOPT's)")
    ap.add_argument("--dd-delta-c", type=float, default=-1.0, dest="dd_delta_c",
                    metavar="VAL",
                    help="per-iteration probe: override δ_c (default <0 = "
                         "IPOPT's √ε·μ^¼). Use 0 to test whether our "
                         "unconditional δ_c is pre-empting δ_w — measured much "
                         "worse in 1D, so this is a diagnostic, not a setting")
    ap.add_argument("--schur", default="restricted",
                    choices=["restricted", "dense"],
                    help="local Schur: p_k backsolves (default) or the naive "
                         "full-p version, for A/B")
    ap.add_argument("--s-format", default="sparse", dest="s_format",
                    choices=["sparse", "dense"],
                    help="interface S: sparse COO/SuperLU (default) or the old "
                         "dense p×p array + LAPACK, for A/B")
    ap.add_argument("--w-inertia", default="dense", dest="w_inertia",
                    choices=["dense", "sparse"],
                    help="In(W_k): always dense LDLᵀ (default) or sparse SuperLU "
                         "pivot signs with a per-block dense fallback — 23× "
                         "faster when δ_w > 0, but falls back on every block at "
                         "δ_w = 0, which is the accepted trial at convergence")
    ap.add_argument("--fd-check", action="store_true", dest="fd_check")
    ap.add_argument("--save-plot", default=None, dest="save_plot",
                    help="write the 8-panel solution figure to this PNG")
    ap.add_argument("--save-domains", default=None, dest="save_domains",
                    help="write the domain-map figure to this PNG")
    ap.add_argument("--save-dd-plot", default=None, dest="save_dd_plot",
                    help="write the arrowhead / interface-matrix figure to this PNG")
    ap.add_argument("--show", action="store_true",
                    help="display the figures instead of only saving")
    args = ap.parse_args()

    N, k = args.N, args.nsub
    if N < 3 or k < 1 or k > N - 1:
        raise SystemExit("need N ≥ 3 and 1 ≤ nsub ≤ N−1")
    w0 = args.alpha0 if args.alpha0 is not None else 0.7 * args.sigma
    if args.weight == "exp" and args.alpha0 is not None:
        w0 = float(np.exp(args.alpha0))          # --alpha0 is the log-weight
    if w0 <= 0.0:
        raise SystemExit("the initial weight must be > 0")

    if args.phantom:
        u_clean, f = make_phantom(N, args.sigma, args.seed)
        src = "phantom"
    else:
        u_clean, f = load_image(args.data, N, args.sigma, args.seed)
        src = os.path.basename(args.data)
    prob = Lifted2DMPCC(f, u_clean, N, weight=args.weight, stencil=args.stencil,
                        reg_alpha=args.reg_alpha, w_max=args.w_max)
    part = Partition2D(N, k)
    kkt_dim = prob.n + 2 * prob.n_ineq + prob.n_eq

    print(f"2D lifted TV-MPCC (staggered)   {src}  N={N}  nodes={prob.m_u}  "
          f"cells={prob.m_q}  tiles={k}×{k}")
    print(f"  vars={prob.n} (m_u + 5·m_q + 1)   rows={prob.m_con} "
          f"({prob.n_eq} eq + {prob.n_ineq} ineq)   KKT dim={kkt_dim}")
    print(f"  stencil = {args.stencil}   weight Q(α) = "
          f"{'α' if args.weight == 'linear' else 'e^α'}"
          f"{'  (+ explicit row ha: α ≥ 0)' if prob.has_ha else '  (α boxed)'}"
          f"   initial weight = {w0:.4f}   init = {args.init}\n")

    owner, col_owner, _ = kkt_owner(prob, part)
    n_bord = int((owner == -1).sum())
    nb_u = int((col_owner[: prob.m_u] < 0).sum())
    nb_qx = int((col_owner[prob.off["qx"] : prob.off["qx"] + prob.m_q] < 0).sum())
    nb_qy = int((col_owner[prob.off["qy"] : prob.off["qy"] + prob.m_q] < 0).sum())
    print(f"  partition: {part.n_sub} subdomains, complicating p={n_bord}  "
          f"(u {nb_u}, qx {nb_qx}, qy {nb_qy}, α 1)")
    print(f"    node ownership = anchor rule (node (i,j) → cell (i−1,j−1));  "
          f"4N(k−1) = {4 * N * (k - 1)} for reference\n")

    x0 = initial_point(prob, w0, init=args.init)
    prob.theta_ref = x0[prob.off["theta"] : prob.off["theta"] + prob.m_q].copy()

    if args.fd_check:
        je, ge, he = fd_check(prob, x0)
        print(f"  fd-check at the warm start:  J={je:.2e}  g={ge:.2e}  H={he:.2e}\n")

    if args.no_solve:
        return

    schedule, t = [], args.t0
    while t >= args.t_min:
        schedule.append(t)
        t *= args.factor

    # `owner` was already built above, before the continuation, and it is
    # structural — kkt_owner's border rule reads jacobian*structure*, so it is
    # independent of x, t and μ. That is what makes probing every level (and
    # every Newton iteration) cost only the numerics.
    dd_path, dd_iters = [], []
    dd_kw = dict(schur=args.schur, s_format=args.s_format,
                 w_inertia=args.w_inertia)

    def probe_level(record, x_l, info_l, mu_l):
        _, rec = dd_probe(prob, owner, part.n_sub, x_l, info_l, mu_l,
                          seed=args.seed, **dd_kw)
        rec["converged"] = record["converged"]
        dd_path.append(rec)

    def probe_iter(iterate, viol, iter_count, mu_i, regu, alg_mod):
        if iter_count % args.dd_iters:
            return
        # No LU cross-check here: exactness of the permutation is a structural
        # property already certified once per level, and at 2D sizes the
        # monolithic splu is the expensive part. What this clock is for is δ_w vs
        # IPOPT's own, so pay only for the inertia loop. `viol` buys the exact
        # Σ = z/s — without it we would be regularizing a *different* matrix from
        # the one IPOPT factorizes, and the comparison would be void.
        _, rec = dd_probe(prob, owner, part.n_sub,
                          np.asarray(iterate["x"], float), iterate, mu_i,
                          seed=args.seed, check_solve=False,
                          compl=None if args.dd_sigma_mu else viol,
                          delta_c=None if args.dd_delta_c < 0 else args.dd_delta_c,
                          **dd_kw)
        rec.update(iter=int(iter_count), regu=float(regu), alg_mod=int(alg_mod))
        dd_iters.append(rec)

    print("  continuation:")
    x, info, t_last, n_it, mu, history = solve_scholtes(
        prob, x0, schedule, linear_solver=args.linear_solver, tol=args.tol,
        max_iter=args.max_iter, hess_update=args.hess_update,
        dual_warmstart=args.dual_warmstart, c_theta=args.c_theta,
        print_level=args.print_level,
        on_level=None if (args.no_dd or not args.dd_path) else probe_level,
        on_iter=None if (args.no_dd or args.dd_iters <= 0) else probe_iter)

    report_dd_path(dd_path)
    report_dd_iters(dd_iters, args.dd_iters)

    a = float(x[prob.off["alpha"]])
    u = x[: prob.m_u]
    print(f"\n  reported level t={t_last:.1e} ({n_it} iterations total):  "
          f"α* = {a:.6f}   weight = {prob.weight_of_alpha(a):.6f}")
    print(f"    PSNR noisy {psnr(u_clean, f):.2f} dB → recon {psnr(u_clean, u):.2f} dB "
          f"({psnr(u_clean, u) - psnr(u_clean, f):+.2f})")

    stamp = (f"2D staggered lifted TV-MPCC — {src} N={N}, {k}×{k} tiles, "
             f"σ={args.sigma}, {args.stencil}, "
             f"Q(α)={'α' if args.weight == 'linear' else 'e^α'}, "
             f"t={t_last:.1e}, weight={prob.weight_of_alpha(a):.4f}")

    arrow = None
    if not args.no_dd:
        print(f"\n  domain-decomposition solve of the KKT at the reported level "
              f"t={t_last:.1e}:")
        arrow, rec = dd_probe(prob, owner, part.n_sub, x, info, mu,
                              seed=args.seed, **dd_kw)
        print(f"    inertia loop: δ_w={rec['delta_w']:.1e} after "
              f"{rec['trials']} trial(s)"
              f"{'' if rec['corrected'] else '  [NOT corrected]'}"
              f"   δ_c={rec['delta_c']:.1e}")
        dims = [len(i) for i in arrow.loc]
        print(f"    arrowhead: {part.n_sub} blocks of dim {dims}, border p={arrow.p}"
              f"   (total {sum(dims) + arrow.p} = {arrow.A.shape[0]})")
        print(f"    blocks are mutually decoupled: {rec['decoupled']}")
        # p_k = border rows each subdomain actually sees. The local Schur costs
        # Σ_k p_k backsolves, not n_sub·p — the phase that dominated before.
        n_bs = (sum(arrow.p_k) if args.schur == "restricted"
                else part.n_sub * arrow.p)
        print(f"    local Schur ({args.schur}): p_k {min(arrow.p_k)}–{max(arrow.p_k)} "
              f"(mean {np.mean(arrow.p_k):.1f}) ⇒ {n_bs} backsolves "
              f"({sum(arrow.p_k)} restricted vs {part.n_sub * arrow.p} full-p);  "
              f"factor {arrow.t_factor * 1e3:.1f} ms, S_k {arrow.t_schur * 1e3:.1f} ms")
        nz = arrow.S_nnz()
        fill = (f", fill nnz(L)+nnz(U)={arrow.S_lu.L.nnz + arrow.S_lu.U.nnz}"
                if arrow.S_lu is not None else "")
        print(f"    interface ({args.s_format}): S is {arrow.p}×{arrow.p}, "
              f"nnz={nz} = {100.0 * nz / (arrow.p ** 2):.2f}% dense "
              f"({arrow.p ** 2 * 8 / 2 ** 20:.1f} MB if densified){fill};  "
              f"S-fact {arrow.t_sfact * 1e3:.1f} ms")
        wmem = sum(len(i) ** 2 for i in arrow.loc) * 8 / 2 ** 20
        print(f"    In(W_k) ({args.w_inertia}): {arrow.t_winertia * 1e3:.1f} ms"
              f"{f', {arrow.n_w_fallback} block(s) fell back to dense' if arrow.n_w_fallback else ''}"
              f"   ({wmem:.1f} MB if every W_k were densified)")

        print(f"    direct interface solve: rel-err vs monolithic LU = "
              f"{rec['err']:.2e}")

        if rec["haynsworth"] is not None:
            print(f"    Haynsworth  Σ In(W_k)+In(S)={rec['inertia']}  vs "
                  f"monolithic In(A): "
                  f"{'MATCH' if rec['haynsworth'] else 'MISMATCH'}")
        else:
            print(f"    Haynsworth  Σ In(W_k)+In(S)={rec['inertia']}   "
                  f"[monolithic dense cross-check SKIPPED: dim "
                  f"{arrow.A.shape[0]} > {DENSE_CHECK_MAX_DIM}, would need "
                  f"{arrow.A.shape[0] ** 2 * 8 / 2 ** 30:.1f} GB]")
        print(f"    IPOPT target {rec['target']}  "
              f"{'hit' if rec['hit'] else 'MISSED'};  "
              f"S is {'SPD' if rec['spd'] else 'NOT SPD ' + str(rec['inS'])}")

    if args.save_plot or args.show:
        plot_solution(prob, x, t_last, history, path=args.save_plot,
                      show=args.show, title=stamp,
                      cuts=(part.cut_nodes, part.cut_nodes))
        if args.save_plot:
            print(f"\n  saved solution figure → {args.save_plot}")
    if args.save_domains or args.show:
        plot_domains(prob, part, col_owner, arrow, path=args.save_domains,
                     show=args.show, title=stamp)
        if args.save_domains:
            print(f"  saved domain map → {args.save_domains}")
    if args.save_dd_plot or args.show:
        if arrow is None:
            print("  [skip] arrowhead figure needs the DD probe (drop --no-dd)")
        else:
            plot_arrowhead(prob, col_owner, arrow, path=args.save_dd_plot,
                           show=args.show, title=stamp)
            if args.save_dd_plot:
                print(f"  saved arrowhead figure → {args.save_dd_plot}")


if __name__ == "__main__":
    main()
