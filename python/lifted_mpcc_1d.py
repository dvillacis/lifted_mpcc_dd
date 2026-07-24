"""1D lifted TV-MPCC on a **staggered node/edge grid** — the whiteboard example.

This is the 1D setting drawn on the board: the primal ``u`` lives on the ``n``
grid **nodes**, the dual ``q`` lives on the ``n−1`` **edges** between them, and
the difference operator is therefore the *rectangular*

    K ∈ R^{(n−1)×n},   (K u)_e = u_{e+1} − u_e,
    Kᵀ ∈ R^{n×(n−1)},  (Kᵀq)_i = q_{i−1} − q_i     (q_{−1} = q_{n−1} = 0)

exactly as written out for ``n = 4`` on the board::

            q₁   q₂   q₃                     ⎛−1  1  0  0⎞          ⎛−1  0  0⎞
      ──┬────┬────┬────┬──           K u  =  ⎜ 0 −1  1  0⎟ u ,  Kᵀ = ⎜ 1 −1  0⎟
        u₁   u₂   u₃   u₄                    ⎝ 0  0 −1  1⎠          ⎜ 0  1 −1⎟
                                                                    ⎝ 0  0  1⎠

and the state row is the board's ``u − f + α·Kᵀq = 0``.

This is **not** what ``dd_demo_1d.py`` does. That script fakes 1D inside the 2D
code by keeping a square ``n×n`` ``Kx`` whose last row is zeroed (a Neumann end)
and ``Ky ≡ 0``, so every block is length ``n``. Here the staggering is real:
``u`` has ``n`` entries while the dual/lift blocks have ``n−1``, there is no
dead Neumann column in ``Kᵀ``, and the layout is ``n + 5(n−1) + 1`` vars /
``n + 7(n−1) + 1`` rows. (It is the 1D sibling of
``lifted_mpcc_unitball_staggered.py``, which does the same for the 2D grid.)

The lift
--------
The lower-level problem is 1D ROF, ``min_u ½‖u−f‖² + α‖K u‖₁``, whose optimality
system is ``u − f + α Kᵀq = 0`` with ``|q_e| ≤ 1`` and ``q_e = sign((Ku)_e)``
wherever ``(Ku)_e ≠ 0``. In the repo's unit-ball polar form, with ``r = |Ku|``,
``δ = |q|`` and the board's angle ``θ``::

    h1  : u − f + Q(α)·Kᵀqx = 0                (n)     the state row
    h2x : K u − r ∘ cos θ   = 0                (n−1)   polar, primal side
    h2y :     − r ∘ sin θ   = 0                (n−1)   ← the 1D y-gradient is 0
    h3x : qx  − δ ∘ cos θ   = 0                (n−1)   polar, dual side
    h3y : qy  − δ ∘ sin θ   = 0                (n−1)
    hr  : r ≥ 0,  hd : δ ≥ 0,  ha : α ≥ 0              explicit inequality ROWS
    comp: r ∘ (1 − δ) ≤ t                      (n−1)   Scholtes relaxation, LAST

plus the box ``δ ≤ 1`` (the unit ball, in 1D the interval ``[−1,1]``). The
solution has ``sin θ = 0`` and ``qy ≡ 0``, so ``θ ∈ {0, π}`` carries the sign of
the jump — the 1D degeneration of the 2D polar lift, exactly as on the board.

**Why the y-rows are kept, and why ``qy`` is load-bearing.** They look redundant
(they only ever say ``sin θ = 0``), and the tempting simplification is to drop
``θ`` for a signed direction ``c`` with an explicit row ``c² = 1``. That was
implemented and **measured to fail** — keep it out:

* ``c² = 1`` makes the direction *discrete*. An interior-point method cannot
  move ``c`` from ``+1`` to ``−1`` without passing through ``c = 0``, which the
  row forbids, so the entire sign pattern is frozen at whatever the warm start
  guessed. Any edge whose dual must change sign along the continuation then
  makes the level genuinely infeasible: measured, ``n = 7`` solves but ``n = 64``
  returns ``Infeasible_Problem_Detected`` at *every* level. With ``θ`` the sign
  flip is a continuous rotation through ``θ = π/2``, where ``r`` passes through
  0 — the physically correct route.
* Dropping ``qy`` but keeping the row ``δ sin θ = 0`` does not work either: at
  the solution ``sin θ = 0`` the gradients of ``r sin θ`` and ``δ sin θ`` are
  both pure ``θ`` directions, so the two rows become **linearly dependent** and
  LICQ fails. The ``qy`` column is what keeps ``h3y`` independent of ``h2y``.

So ``qy`` is not a ghost: the lift carries the full plane dual ``(qx, qy)``, and
the 1D geometry is what forces ``qy ≡ 0``. The angle gauge D1 (``θ``
undetermined where ``r = δ = 0``) survives, as in 2D — ``--c-theta`` adds the TR
ridge for it.

The weight
----------
``--weight linear`` (default, as on the board) puts the bare ``α`` in the state
row and carries the explicit row ``ha : α ≥ 0``, like
``lifted_mpcc_unitball_linalpha.py``; ``--alpha0`` is then the weight itself.
``--weight exp`` uses ``Q(α) = e^α`` with an α box and no ``ha`` row, like
``lifted_mpcc_unitball_v2.py``; ``--alpha0`` is then the log-weight. Both share
one code path through ``Q, Q', Q''``. Note ``--w-max``: the linear weight needs
an explicit cap (see :func:`bounds`).

Domain decomposition
--------------------
The right half of the board: partition the ``n−1`` edges into ``k`` contiguous
subdomains and give each node to the subdomain owning its outgoing edge. Each
subdomain then carries the *local* state equation drawn on the board — for
``n = 7, k = 2``::

    Ω₁ : (u₁ u₂ u₃ u₄)ᵀ − f_{Ω₁} + α K₁ᵀ (q₁ q₂ q₃)ᵀ = 0     K₁ᵀ : 4×3
    Ω₂ : (u₄ u₅ u₆ u₇)ᵀ − f_{Ω₂} + α K₂ᵀ (q₄ q₅ q₆)ᵀ = 0     K₂ᵀ : 4×3

with the interface node ``u₄`` appearing in **both** — the board's
``N₁u = (u_{Γ₁}; u_{Γ₁∩Γ₂})``, the restriction ``(I 0)``. Summing the two local
rows reproduces the global ``h1`` row at ``u₄``, which is precisely why the
complicating (border) variables are

    * the **first ``u`` past each cut** (the shared interface node),
    * the **last ``qx`` before each cut** (the edge whose ``h1`` reaches across),
    * plus the scalar ``α``,

i.e. ``p = 2(k−1) + 1``, the same rule ``dd_structure.py`` derives in 2D.
(``qy`` never crosses a cut: it appears only in the edge-local ``h3y``.) The
script assembles IPOPT's augmented KKT at the reported iterate, permutes it to
arrowhead form (pure permutation — no reformulation), forms the local Schur
complements ``S_k = −B̄_k W_k⁻¹ B̄_kᵀ``, assembles ``S``, solves the interface
directly, and checks Haynsworth additivity ``In(A) = Σ_k In(W_k) + In(S)``
against IPOPT's inertia target, running IPOPT's own δ_w correction loop on the
*distributed* inertia.

That probe runs at **every** Scholtes level, not only at the end: the claim
being tested ("the arrowhead is exact and Haynsworth reproduces IPOPT's inertia
on this problem") is a claim about the whole relaxation path, and the path is
where the degeneracy bites — the biactive set grows, ``Σ = z²/μ`` blows up, and
``S`` is not obliged to stay SPD as ``t ↓ 0``. A single sample at the end cannot
tell "DD works here" from "DD happened to work at that point". It costs almost
nothing because the *structure* is ``t``-independent — ``kkt_owner`` sees only
the partition and the index layout, so ``loc/bord/p`` are computed once and only
the numerics (``H(x,λ)``, ``J(x)``, ``Σ``, ``δ_c = √ε·μ^¼``) are rebuilt per
level. Each level is probed at its *converged* iterate, i.e. at its smallest μ
and so its largest ``Σ`` — the worst-conditioned point of that level.
``--no-dd-path`` restores final-iterate-only probing.

``--dd-iters STRIDE`` adds the finer clock: a probe at every STRIDE-th **Newton
iteration**, which is where IPOPT actually corrects inertia (a run can show its
``δw`` column nonzero while every *level* probe reports ``δ_w = 0``) and the only
place a real DD linear solver would ever be called. Two things make that
comparison valid: ``get_current_violations`` supplies the **exact** ``Σ = z/s``
in place of the central-path ``z²/μ``, and the callback's off-by-one is measured
rather than assumed (see :func:`report_dd_iters`).

Everything here is a **probe**: it rebuilds IPOPT's KKT after the fact and never
supplies a step. Making the arrowhead the actual linear solver is impossible from
Python (IPOPT's ``SparseSymLinearSolverInterface`` is C++), so the real thing lives
in ``cpp/dd_solve_1d.cpp`` + ``cpp/mpcc_1d_tnlp.hpp``: the same formulation as an
``Ipopt::TNLP`` with ``DDArrowheadSolver`` producing every Newton step. Run
``cpp/dump_data_1d.py`` first — it exports this script's data, its lifted CP warm
start and :func:`kkt_owner`'s border map, so the two sides solve the *same*
problem from the *same* point. Measured: identical solution and identical
per-level iteration counts.

Usage
-----
    uv run python lifted_mpcc_1d.py                       # n=7, k=2 — the board
    uv run python lifted_mpcc_1d.py --n 64 --nsub 4       # a real 1D denoising run
    uv run python lifted_mpcc_1d.py --n 64 --nsub 4 --t-min 1e-6   # 12-level DD path table
    uv run python lifted_mpcc_1d.py --no-dd-path          # probe the reported iterate only
    uv run python lifted_mpcc_1d.py --show-operators --no-solve   # bookkeeping only
    uv run python lifted_mpcc_1d.py --fd-check            # validate the derivatives
    uv run python lifted_mpcc_1d.py --weight exp --init cp-scan   # the A/B levers

    # graphics (matplotlib): a 4-panel solution figure and the DD structure
    uv run python lifted_mpcc_1d.py --save-plot sol.png --save-dd-plot dd.png
    uv run python lifted_mpcc_1d.py --n 64 --nsub 4 --show
"""

from __future__ import annotations

import argparse
import os
import signal

import cyipopt
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from lifted_mpcc_unitball_v2 import HSLLIB, psnr


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_signal(n: int, sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant test signal + Gaussian noise.

    Small ``n`` (the board's 7-node example) gets a single jump so the printed
    vectors stay readable; larger ``n`` gets the usual multi-plateau signal.
    """
    u = np.zeros(n)
    if n < 12:
        u[: n // 2] = 0.25
        u[n // 2 :] = 0.80
    else:
        for lo, hi, v in [(0.00, 0.20, 0.2), (0.20, 0.35, 0.8), (0.35, 0.55, 0.5),
                          (0.55, 0.70, 0.95), (0.70, 1.01, 0.35)]:
            u[int(lo * n):int(hi * n)] = v
    rng = np.random.default_rng(seed)
    return u, np.clip(u + sigma * rng.standard_normal(n), 0.0, 1.0)


def grad_operator_1d(n: int) -> sp.csr_matrix:
    """The board's rectangular forward difference ``K ∈ R^{(n−1)×n}``.

    No Neumann padding row: there are ``n−1`` edges between ``n`` nodes, full
    stop, so every column of ``Kᵀ`` is nonzero (contrast ``dd_demo_1d.py``,
    whose square ``Kx`` leaves a dead last column).
    """
    e = np.ones(n - 1)
    return sp.diags([-e, e], [0, 1], shape=(n - 1, n), format="csr")


# ---------------------------------------------------------------------------
# the problem object
# ---------------------------------------------------------------------------
class Lifted1DMPCC:
    """cyipopt problem object for the staggered 1D lifted TV-MPCC.

    Variables ``x = [u (n) | qx | qy | r | δ | θ | α]`` with the five middle
    blocks of length ``mE = n−1``, so ``n_var = n + 5·mE + 1``.

    Rows, in order (``comp`` LAST, as everywhere in this repo)::

        h1 (n) | h2x | h2y | h3x | h3y | hr | hd | [ha (1)] | comp     (mE each)

    ``ha`` (the row ``α ≥ 0``) is present iff ``weight == "linear"``. Only ``h1``
    is node-length; every other block is edge-length.
    """

    def __init__(self, f, u_clean, *, weight="linear", reg_alpha=1e-4,
                 w_max=float("inf")):
        self.f = np.asarray(f, float)
        self.u_clean = np.asarray(u_clean, float)
        self.n_nodes = n = len(self.f)
        self.n_edges = mE = n - 1
        self.weight = weight
        self.reg_alpha = reg_alpha
        self.w_max = w_max          # optional cap on the linear weight (see ``bounds``)
        self.has_ha = weight == "linear"

        # Scholtes level: the comp row is r·(1−δ) − t_comp ≤ 0 with fixed upper
        # bound 0, so one cyipopt.Problem is reused across the whole continuation
        # (t is additive, hence absent from every derivative).
        self.t_comp = 0.0
        # TR gauge ridge ½·eps_theta·‖θ − θ_ref‖² — the D1 fix for the angle-gauge
        # indeterminacy (θ undetermined where r = δ = 0). The driver sets
        # eps_theta = c_θ·t per level, so the bias vanishes as t ↓ 0.
        self.eps_theta = 0.0
        self.theta_ref = np.zeros(mE)

        self.n_iter = self.n_reg = self.n_rest = 0
        self.inf_pr = self.inf_du = self.mu_last = float("nan")
        self._interrupt = False
        self._cx = None
        # per-Newton-iteration probe plumbing (see ``intermediate``): the hook and
        # the live cyipopt.Problem, both installed by ``solve_scholtes``.
        self._on_iter = None
        self._nlp = None

        self.K = grad_operator_1d(n)
        self.KT = self.K.T.tocsr()

        self.off = {"u": 0, "qx": n, "qy": n + mE, "r": n + 2 * mE,
                    "delta": n + 3 * mE, "theta": n + 4 * mE, "alpha": n + 5 * mE}
        self.n = n + 5 * mE + 1

        ro, o = {}, 0
        for name, size in (("h1", n), ("h2x", mE), ("h2y", mE), ("h3x", mE),
                           ("h3y", mE), ("hr", mE), ("hd", mE),
                           ("ha", 1 if self.has_ha else 0), ("comp", mE)):
            ro[name] = o
            o += size
        self.roff = ro
        self.m_con = o
        self.n_eq = ro["hr"]                  # h1|h2x|h2y|h3x|h3y are the equalities
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
        size = self.n_nodes if name == "u" else self.n_edges
        return x[s : s + size]

    def _common(self, x):
        """Memoized per-x intermediates ``(Q, Q', Q'', cos θ, sin θ, Kᵀqx)``.

        All are t-independent, so the memo survives Scholtes level changes.
        """
        if self._cx is None or not np.array_equal(x, self._cx):
            self._cx = np.array(x, dtype=float, copy=True)
            a = self._cx[self.off["alpha"]]
            th = self._blk(self._cx, "theta")
            self._cvals = (self.Q(a), self.dQ(a), self.d2Q(a), np.cos(th),
                           np.sin(th), self.KT @ self._blk(self._cx, "qx"))
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
        g[: self.n_nodes] = self._blk(x, "u") - self.u_clean
        g[self.off["alpha"]] = self.reg_alpha * x[self.off["alpha"]]
        if self.eps_theta:
            s = self.off["theta"]
            g[s : s + self.n_edges] = self.eps_theta * (self._blk(x, "theta")
                                                        - self.theta_ref)
        return g

    # ---- constraints ------------------------------------------------------
    def constraints(self, x):
        Qa, _, _, c, s, KTq = self._common(x)
        u, qx, qy = self._blk(x, "u"), self._blk(x, "qx"), self._blk(x, "qy")
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        out = [u - self.f + Qa * KTq,     # h1  (n)   state row
               self.K @ u - r * c,        # h2x       polar, primal side
               -r * s,                    # h2y       (K_y ≡ 0 in 1D)
               qx - delta * c,            # h3x       polar, dual side
               qy - delta * s,            # h3y
               r,                         # hr : r ≥ 0
               delta]                     # hd : δ ≥ 0
        if self.has_ha:
            out.append(np.array([x[self.off["alpha"]]]))   # ha : α ≥ 0
        out.append(r * (1.0 - delta) - self.t_comp)        # comp (Scholtes), LAST
        return np.concatenate(out)

    # ---- Jacobian ---------------------------------------------------------
    def _build_structure(self):
        """Index arrays of every Jacobian nonzero.

        The pieces below are assembled in the SAME order by ``jacobian`` — the
        value arrays are concatenated positionally, not matched by key, so the
        two lists must stay aligned (run ``--fd-check`` after any edit). Piece 3
        is the dense ``∂h1/∂α`` column, the whole α-coupling of the KKT system.
        """
        n, mE, off, ro = self.n_nodes, self.n_edges, self.off, self.roff
        idn, ide = np.arange(n), np.arange(mE)
        K, KT = self.K.tocoo(), self.KT.tocoo()
        self._Kv, self._KTv = K.data, KT.data
        self._one_n, self._one_e = np.ones(n), np.ones(mE)

        pieces = [
            (ro["h1"] + idn, off["u"] + idn),             # 1  h1/∂u   = I
            (ro["h1"] + KT.row, off["qx"] + KT.col),      # 2  h1/∂qx  = Q(α)·Kᵀ
            (ro["h1"] + idn, np.full(n, off["alpha"])),   # 3  h1/∂α   = Q'(α)·Kᵀqx
            (ro["h2x"] + K.row, off["u"] + K.col),        # 4  h2x/∂u  = K
            (ro["h2x"] + ide, off["r"] + ide),            # 5  h2x/∂r  = −cos θ
            (ro["h2x"] + ide, off["theta"] + ide),        # 6  h2x/∂θ  =  r sin θ
            (ro["h2y"] + ide, off["r"] + ide),            # 7  h2y/∂r  = −sin θ
            (ro["h2y"] + ide, off["theta"] + ide),        # 8  h2y/∂θ  = −r cos θ
            (ro["h3x"] + ide, off["qx"] + ide),           # 9  h3x/∂qx = I
            (ro["h3x"] + ide, off["delta"] + ide),        # 10 h3x/∂δ  = −cos θ
            (ro["h3x"] + ide, off["theta"] + ide),        # 11 h3x/∂θ  =  δ sin θ
            (ro["h3y"] + ide, off["qy"] + ide),           # 12 h3y/∂qy = I
            (ro["h3y"] + ide, off["delta"] + ide),        # 13 h3y/∂δ  = −sin θ
            (ro["h3y"] + ide, off["theta"] + ide),        # 14 h3y/∂θ  = −δ cos θ
            (ro["hr"] + ide, off["r"] + ide),             # 15 hr/∂r   = 1
            (ro["hd"] + ide, off["delta"] + ide),         # 16 hd/∂δ   = 1
        ]
        if self.has_ha:
            pieces.append((np.array([ro["ha"]]), np.array([off["alpha"]])))  # 17 ha/∂α
        pieces += [
            (ro["comp"] + ide, off["r"] + ide),           # 18 comp/∂r = 1 − δ
            (ro["comp"] + ide, off["delta"] + ide),       # 19 comp/∂δ = −r
        ]
        rows = np.concatenate([p[0] for p in pieces]).astype(np.int64)
        cols = np.concatenate([p[1] for p in pieces]).astype(np.int64)
        return rows, cols

    def jacobianstructure(self):
        return self._rows, self._cols

    def jacobian(self, x):
        Qa, dQa, _, c, s, KTq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        vals = [self._one_n,       # 1  h1/∂u
                Qa * self._KTv,    # 2  h1/∂qx
                dQa * KTq,         # 3  h1/∂α
                self._Kv,          # 4  h2x/∂u
                -c,                # 5  h2x/∂r
                r * s,             # 6  h2x/∂θ
                -s,                # 7  h2y/∂r
                -r * c,            # 8  h2y/∂θ
                self._one_e,       # 9  h3x/∂qx
                -c,                # 10 h3x/∂δ
                delta * s,         # 11 h3x/∂θ
                self._one_e,       # 12 h3y/∂qy
                -s,                # 13 h3y/∂δ
                -delta * c,        # 14 h3y/∂θ
                self._one_e,       # 15 hr/∂r
                self._one_e]       # 16 hd/∂δ
        if self.has_ha:
            vals.append(np.array([1.0]))   # 17 ha/∂α
        vals += [1.0 - delta,      # 18 comp/∂r
                 -r]               # 19 comp/∂δ
        return np.concatenate(vals)

    # ---- exact Lagrangian Hessian (lower triangle) ------------------------
    # H = σ_f·∇²J + Σ_k λ_k ∇²c_k. Nonlinear rows and their curvature:
    #   h1  : bilinear in (α, qx) → (α,qx) = Q'(α)·K λ_h1,
    #         (α,α) = Q''(α)·⟨λ_h1, Kᵀqx⟩ — identically 0 for the LINEAR weight,
    #         which therefore gets its only (α,α) curvature from the reg-α ridge.
    #   h2x : ∂²/∂r∂θ =  sin θ, ∂²/∂θ² =  r cos θ ;  h2y : −cos θ,  r sin θ
    #   h3x : ∂²/∂δ∂θ =  sin θ, ∂²/∂θ² =  δ cos θ ;  h3y : −cos θ,  δ sin θ
    #   comp: r·(1−δ) → ∂²/∂r∂δ = −1 ⇒ (δ,r) = −ξ, the indefinite MPCC cross
    #         (eigenvalues ±ξ).
    # Column order is u < qx < qy < r < δ < θ < α, so all of these land in the
    # lower triangle as written.
    def _build_hess_structure(self):
        n, mE, off = self.n_nodes, self.n_edges, self.off
        idn, ide = np.arange(n), np.arange(mE)
        a = off["alpha"]
        rows = np.concatenate([
            off["u"] + idn,        # (u,u)  objective identity
            off["theta"] + ide,    # (θ,r)
            off["theta"] + ide,    # (θ,δ)
            off["theta"] + ide,    # (θ,θ)
            off["delta"] + ide,    # (δ,r)  comp cross (indefinite)
            np.full(mE, a),        # (α,qx) h1 bilinear cross
            [a],                   # (α,α)
        ])
        cols = np.concatenate([
            off["u"] + idn, off["r"] + ide, off["delta"] + ide, off["theta"] + ide,
            off["r"] + ide, off["qx"] + ide, [a],
        ])
        return rows.astype(np.int64), cols.astype(np.int64)

    def hessianstructure(self):
        return self._hrows, self._hcols

    def hessian(self, x, lagrange, obj_factor):
        n, mE, ro = self.n_nodes, self.n_edges, self.roff
        Qa, dQa, d2Qa, c, s, KTq = self._common(x)
        r, delta = self._blk(x, "r"), self._blk(x, "delta")
        l1 = lagrange[ro["h1"] : ro["h1"] + n]
        l2x = lagrange[ro["h2x"] : ro["h2x"] + mE]
        l2y = lagrange[ro["h2y"] : ro["h2y"] + mE]
        l3x = lagrange[ro["h3x"] : ro["h3x"] + mE]
        l3y = lagrange[ro["h3y"] : ro["h3y"] + mE]
        xi = lagrange[ro["comp"] : ro["comp"] + mE]

        H_uu = obj_factor * self._one_n
        H_tr = l2x * s - l2y * c                                      # (θ,r)
        H_td = l3x * s - l3y * c                                      # (θ,δ)
        H_tt = (r * (l2x * c + l2y * s) + delta * (l3x * c + l3y * s)
                + obj_factor * self.eps_theta)                        # (θ,θ)
        H_dr = -xi                                                    # (δ,r)
        H_aq = dQa * (self.K @ l1)                                    # (α,qx)
        H_aa = np.array([d2Qa * float(np.dot(l1, KTq))
                         + obj_factor * self.reg_alpha])              # (α,α)
        return np.concatenate([H_uu, H_tr, H_td, H_tt, H_dr, H_aq, H_aa])

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
        # mult_x_U — so the probe is a drop-in on the live iterate.
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
# warm start: 1D Chambolle–Pock ROF
# ---------------------------------------------------------------------------
def chambolle_pock_1d(f, K, KT, lam, n_iter=3000, tol=1e-9, n_accel=300,
                      u0=None, q0=None):
    """Hybrid Chambolle–Pock for 1D ROF ``min_u ½‖u−f‖² + lam·‖K u‖₁``.

    Returns ``(u, q)`` with the edge dual in the 1D unit ball ``|q_e| ≤ 1``. At
    convergence ``u = f − lam·Kᵀq`` with ``r ⊥ (1−δ)`` — i.e. exactly this
    MPCC's lower-level system at ``Q(α) = lam``, so the CP pair lifts to an
    almost-feasible warm start (h1 ≈ 0, h3 exact).

    Same hybrid schedule as v2: an accelerated phase (γ = 1, the primal is
    1-strongly convex) then fixed steps that polish the h1 fixed point, stopping
    on the exact residual identity ``‖u₊−u‖∞/τ = ‖h1‖∞``. In 1D ``‖K‖² ≤ 4``
    (vs 8 in 2D), which sets the step sizes.
    """
    tau0 = 0.99 / (2.0 * lam)     # τ₀·lam = σ₀·lam = 0.99/2, since ‖K‖² ≤ 4
    sig_lam0 = 0.99 / 2.0
    tau, sig_lam = tau0, sig_lam0
    u = f.copy() if u0 is None else np.asarray(u0, float).copy()
    ubar = u.copy()
    q = np.zeros(K.shape[0]) if q0 is None else np.asarray(q0, float).copy()
    for k in range(n_iter):
        accel = k < n_accel
        if not accel and tau != tau0:
            tau, sig_lam = tau0, sig_lam0          # phase switch (τσ invariant)
        q = np.clip(q + sig_lam * (K @ ubar), -1.0, 1.0)   # ascent + unit-ball proj
        u_new = (tau * f + u - (tau * lam) * (KT @ q)) / (tau + 1.0)
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
    return u, q


def cp_scan(prob, w0, half_width=2.0, n_grid=9):
    """Coarse CP sweep of ``L(w) = ½‖u_CP(w) − u_clean‖²`` on a log grid around
    ``w0``; returns the argmin ``(w*, u, q)``. Legitimate because CP is an exact
    lower-level solve and ``u_clean`` is the training target, so this is the true
    (unimodal) bilevel objective."""
    best, u, q = None, None, None
    for w in w0 * np.exp(np.linspace(-half_width, half_width, n_grid)):
        u, q = chambolle_pock_1d(prob.f, prob.K, prob.KT, float(w), u0=u, q0=q)
        loss = 0.5 * float(np.sum((u - prob.u_clean) ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(w), u.copy(), q.copy())
    return best[1], best[2], best[3]


def initial_point(prob, w0, init="cp"):
    """Warm start. ``w0`` is the WEIGHT (not α).

    ``cp``/``cp-scan`` lift a Chambolle–Pock lower-level solution: ``u, qx`` from
    CP, ``δ = |qx|``, ``qy = 0``, ``r = |Ku|``, and ``θ ∈ {0, π}`` taken from the
    sign of the **dual** (as in 2D, where θ = ∠q).

    Taking θ from the dual rather than from ``Ku`` matters. It makes h3x exact
    and keeps ``r ≥ 0`` by construction, leaving the whole error on
    ``h2x = Ku − |Ku|·cos θ``, which is ``0`` where the two signs agree and
    ``2|Ku|`` where they do not — and they only disagree on flat edges, where
    ``|Ku|`` is at noise level. Doing it the other way round (θ from ``Ku``,
    ``r = cos θ·Ku``) looks equally reasonable and is much worse: it puts the
    error on ``h3x = qx − |qx|·cos θ = 2qx``, i.e. **O(1)** on every flat edge,
    since a flat edge still carries a definite unit-size dual. Measured, that
    left the n=64 warm start at ``h3x = 1.9`` and no level ever converged.

    ``cold`` is ``u = f, q = 0``: it sits on the no-regularization manifold and
    is this repo's known route to the spurious near-noisy branch.
    """
    off, mE = prob.off, prob.n_edges
    x = np.zeros(prob.n)
    if init in ("cp", "cp-scan"):
        if init == "cp-scan":
            w0, u, q = cp_scan(prob, w0)
        else:
            u, q = chambolle_pock_1d(prob.f, prob.K, prob.KT, float(w0))
    else:                                     # cold
        u, q = prob.f.copy(), np.zeros(mE)

    Ku = prob.K @ u
    sgn = np.where(np.abs(q) > 1e-12, np.sign(q), 0.0)
    fallback = np.where(Ku != 0.0, np.sign(Ku), 1.0)
    sgn = np.where(sgn == 0.0, fallback, sgn)
    x[off["u"] : off["u"] + prob.n_nodes] = u
    x[off["qx"] : off["qx"] + mE] = q
    # qy = 0 already
    x[off["r"] : off["r"] + mE] = np.abs(Ku)
    x[off["delta"] : off["delta"] + mE] = np.abs(q)
    x[off["theta"] : off["theta"] + mE] = np.where(sgn > 0, 0.0, np.pi)
    x[off["alpha"]] = prob.alpha_of_weight(w0)
    return x


def bounds(prob, alpha_lo=-15.0, alpha_hi=15.0):
    """Variable bounds. Only ``δ ≤ 1`` (the unit ball) is a box for every mode;
    ``r ≥ 0``, ``δ ≥ 0`` and — under the linear weight — ``α ≥ 0`` are explicit
    rows instead.

    The linear weight is unbounded above by default (``w_max = inf``), matching
    ``lifted_mpcc_unitball_linalpha.py``'s "no α box at all"; the exponential
    weight keeps v2's box, which caps it implicitly at ``e^15``.

    ``--w-max`` exists as an opt-in safety net for the loose end of the
    schedule, where the argument for a cap is real even though it does not bite
    in practice: at a vacuous level ``qx`` is pinned only by ``|qx| = δ ≤ 1``
    while ``α·Kᵀqx`` is not pinned at all, so the upper level can drive ``u``
    onto ``u_clean`` (up to a constant — ``range(Kᵀ) = 1^⊥``) by taking
    ``α → ∞``, opposed only by the tiny ``½·reg_alpha·α²`` ridge. That runaway
    was observed at n=64 (``α → 5.7e5``, the solve never leaving t = 1) but
    **only under a broken warm start**; with the h3x-exact lift of
    :func:`initial_point` the first level lands at ``α ≈ 0.24`` and an
    effectively infinite cap reaches the same optimum. Don't cite the old
    number as a property of the formulation.
    """
    xl = np.full(prob.n, -2.0e19)
    xu = np.full(prob.n, 2.0e19)
    xu[prob.off["delta"] : prob.off["delta"] + prob.n_edges] = 1.0
    if prob.has_ha:
        if np.isfinite(prob.w_max):          # α ≥ 0 is the row ha; cap from above
            xu[prob.off["alpha"]] = prob.w_max
    else:
        xl[prob.off["alpha"]], xu[prob.off["alpha"]] = alpha_lo, alpha_hi
    return xl, xu


def constraint_bounds(prob):
    """``(cl, cu)``: equalities at 0, ``hr/hd/ha`` as ``0 ≤ · ≤ +∞``, and the
    comp rows one-sided with a FIXED upper bound 0 (``t`` rides inside the row on
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
                   tol_factor=0.1, max_iter=1000, hess_update="exact",
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
    on the reported iterate. ``on_iter(iterate, iter_count, mu, regu, alg_mod)``
    is the finer clock: it fires from ``prob.intermediate`` at **every Newton
    iteration**, which is where IPOPT actually performs its inertia correction and
    where a real DD linear solver would be called. Both are diagnostics: an
    exception inside either is caught and reported, never allowed to kill the
    continuation.

    Returns ``(x, info, t, total_iter, mu_last, history)``, where ``history`` is
    one dict per attempted level (for :func:`plot_solution`'s continuation panel).
    """
    off, mE = prob.off, prob.n_edges
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

            r = x[off["r"] : off["r"] + mE]
            w = 1.0 - x[off["delta"] : off["delta"] + mE]
            comp_res = float(np.max(r * w)) if mE else 0.0
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
class Partition1D:
    """Contiguous partition of the ``n−1`` edges into ``k`` subdomains.

    Nodes follow their outgoing edge (the last node follows the last edge), so
    each subdomain owns a contiguous run of nodes and edges. The **complicating
    (border) columns** are, per cut: the first ``u`` past the cut (the shared
    interface node) and the last ``qx`` before it, plus the scalar ``α``.
    """

    def __init__(self, n_nodes: int, n_sub: int):
        self.n_nodes, self.n_edges, self.n_sub = n_nodes, n_nodes - 1, n_sub
        bnd = np.linspace(0, self.n_edges, n_sub + 1).astype(int)
        self.edge_ranges = [(bnd[k], bnd[k + 1]) for k in range(n_sub)]
        self.edge_owner = np.empty(self.n_edges, dtype=int)
        for k, (a, b) in enumerate(self.edge_ranges):
            self.edge_owner[a:b] = k
        self.node_owner = np.empty(self.n_nodes, dtype=int)
        self.node_owner[: self.n_edges] = self.edge_owner
        self.node_owner[self.n_edges] = self.edge_owner[-1]
        # cut j sits between subdomain j and j+1, at edge index e_j
        self.cut_edges = [self.edge_ranges[k][1] - 1 for k in range(n_sub - 1)]
        self.cut_nodes = [e + 1 for e in self.cut_edges]

    def nodes_of(self, k: int) -> np.ndarray:
        """Nodes of the LOCAL state equation of Ω_k — its own nodes **plus** the
        interface node past its last edge (the board's ``N_k u``)."""
        a, b = self.edge_ranges[k]
        return np.arange(a, b + 1)

    def edges_of(self, k: int) -> np.ndarray:
        a, b = self.edge_ranges[k]
        return np.arange(a, b)


def local_state_rows(prob, part: Partition1D, k: int) -> str:
    """The board's local state equation for Ω_k, printed symbolically."""
    nodes, edges = part.nodes_of(k), part.edges_of(k)
    KkT = np.asarray(grad_operator_1d(len(nodes)).T.todense()).astype(int)
    lines = [f"    Ω{k + 1}: nodes u{nodes[0] + 1}..u{nodes[-1] + 1}   "
             f"edges q{edges[0] + 1}..q{edges[-1] + 1}   "
             f"K{k + 1}ᵀ : {KkT.shape[0]}×{KkT.shape[1]}"]
    for i, node in enumerate(nodes):
        terms = " ".join(f"{v:+d}·q{e + 1}" for v, e in zip(KkT[i], edges) if v)
        lines.append(f"      u{node + 1} − f{node + 1} + α·({terms or '0':>16s}) = 0")
    return "\n".join(lines)


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

    square of size ``n_var + 2·n_ineq + n_eq``.

    ``Σ`` has two modes. By default ``Σ = z²/μ`` — the *central-path* form, forced
    on us at converged iterates because cyipopt returns the primal point but keeps
    its own slacks, and there ``x`` sits exactly on the relaxed bound so ``z/s``
    divides by ~0. When ``compl`` is given — the dict from
    ``Problem.get_current_violations()``, live inside an intermediate callback —
    the **exact** ``Σ = z/s = z²/(s·z)`` is used instead, since ``s·z`` is
    precisely what ``compl_x_L``/``compl_x_U``/``compl_g`` report. The two agree
    to the extent the iterate is on the central path (``s·z ≈ μ``), which is why
    the default is the right approximation at a converged level and the wrong one
    at an intermediate Newton iterate — where IPOPT is off the path and the
    difference is exactly what its inertia correction reacts to.
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
    n_low = n_in - prob.n_edges
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


def kkt_owner(prob, part: Partition1D) -> np.ndarray:
    """Label every KKT index with its subdomain, or ``-1`` for the border.

    Columns are node/edge-indexed and so inherit an owner; slacks and multipliers
    inherit their row's node or edge. **Only primal columns are ever border** —
    the cut ``u``/``qx`` and the global α. In particular the ``ha`` row
    (``α ≥ 0``), whose only column is the border α, is given to subdomain 0 like
    any other row: its slack and multiplier are *dual* directions, and putting
    them on the border would make ``S`` indefinite by construction
    (``In(S) = (p−1, 1, 0)``), defeating the δ_w loop that runs on that signal.
    """
    n, mE, off, ro = prob.n_nodes, prob.n_edges, prob.off, prob.roff
    eo, no = part.edge_owner, part.node_owner
    dim = prob.n + 2 * prob.n_ineq + prob.n_eq
    owner = np.empty(dim, dtype=int)

    owner[off["u"] : off["u"] + n] = no
    for name in ("qx", "qy", "r", "delta", "theta"):
        owner[off[name] : off[name] + mE] = eo
    owner[off["alpha"]] = -1                              # α is global

    s0 = prob.n                       # slacks (inequality rows only)
    c0 = s0 + prob.n_ineq             # λ_c  (equality rows)
    d0 = c0 + prob.n_eq               # λ_d  (inequality rows)
    ineq0 = ro["hr"]
    for base in (s0, d0):
        for name in ("hr", "hd", "comp"):
            j = base + ro[name] - ineq0
            owner[j : j + mE] = eo
        if prob.has_ha:
            owner[base + ro["ha"] - ineq0] = 0            # dual dirs stay local
    owner[c0 + ro["h1"] : c0 + ro["h1"] + n] = no
    for name in ("h2x", "h2y", "h3x", "h3y"):
        owner[c0 + ro[name] : c0 + ro[name] + mE] = eo

    for e, node in zip(part.cut_edges, part.cut_nodes):
        owner[off["u"] + node] = -1      # first u past the cut
        owner[off["qx"] + e] = -1        # last qx before the cut
    return owner


class Arrowhead:
    """Bordered block-diagonal view: ``W_k = A[loc_k,loc_k]``,
    ``B_k = A[bord,loc_k]``, ``C = A[bord,bord]``, ``S = C − Σ_k B_k W_k⁻¹ B_kᵀ``.

    The permutation is exact — no reformulation — so the border carries real
    Jacobian entries and ``C`` is nonzero. (For readability the full ``p`` border
    rows of ``B_k`` are used; only the ``p_k`` rows subdomain ``k`` actually sees
    are nonzero, which is what makes this scale.)
    """

    def __init__(self, A, owner, n_sub):
        self.A = A
        self.loc = [np.flatnonzero(owner == k) for k in range(n_sub)]
        self.bord = np.flatnonzero(owner == -1)
        self.p = len(self.bord)
        self.W = [A[np.ix_(i, i)].tocsc() for i in self.loc]
        self.B = [A[np.ix_(self.bord, i)].tocsr() for i in self.loc]
        self.C = np.asarray(A[np.ix_(self.bord, self.bord)].todense())
        self.lu = self.S = None

    def check_block_diagonal(self) -> bool:
        """No two subdomains may share a nonzero — the whole point of the
        complicating-variable rule."""
        for i in range(len(self.loc)):
            for j in range(i + 1, len(self.loc)):
                if self.A[np.ix_(self.loc[i], self.loc[j])].nnz:
                    return False
        return True

    def factorize(self):
        self.lu = [spla.splu(Wk) for Wk in self.W]
        self.fill = sum(lu.L.nnz + lu.U.nnz for lu in self.lu)

    def local_schur(self):
        self.S = self.C.copy()
        self.S_k = []
        for lu, Bk in zip(self.lu, self.B):
            Sk = -(Bk @ lu.solve(np.asarray(Bk.T.todense())))
            self.S_k.append(Sk)
            self.S += Sk

    def solve(self, rhs):
        """``r_S = r_y − Σ_k B_k W_k⁻¹ r_k``, one dense interface solve, then
        ``Δx_k = W_k⁻¹(r_k − B_kᵀ Δy)`` — one backsolve each, parallel."""
        r_loc = [rhs[i] for i in self.loc]
        r_S = rhs[self.bord].copy()
        for lu, Bk, rk in zip(self.lu, self.B, r_loc):
            r_S -= Bk @ lu.solve(rk)
        dy = np.linalg.solve(self.S, r_S)
        out = np.empty(self.A.shape[0])
        out[self.bord] = dy
        for i, lu, Bk, rk in zip(self.loc, self.lu, self.B, r_loc):
            out[i] = lu.solve(rk - Bk.T @ dy)
        return out

    def inertia(self):
        """``(Σ_k In(W_k) + In(S), [In(W_k)], In(S))`` — Haynsworth additivity
        used as a *computation*: the inertia IPOPT needs for its correction loop
        is assembled from the small local blocks plus ``S``, without ever
        factorizing the full KKT."""
        inW = [_inertia(np.asarray(Wk.todense())) for Wk in self.W]
        inS = _inertia(self.S)
        return tuple(sum(c) for c in zip(*inW, inS)), inW, inS


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
             max_dense=3000, compl=None, delta_c=None):
    """The full arrowhead/Haynsworth probe of the KKT **at one iterate**.

    Split out of ``main`` so the same probe can run at every continuation level
    and again on the reported iterate. Only the numerics depend on the level:
    ``owner`` — hence ``loc``, ``bord``, ``p``, and the whole block structure —
    is fixed by the partition and the index layout, so it is built once by the
    caller and reused; what moves with ``t`` is ``H(x,λ)``, ``J(x)``,
    ``Σ = z²/μ`` and ``δ_c``.

    ``δ_c`` is IPOPT's Alg. IC value ``√ε·μ^¼`` at *this* level's μ: probing with
    ``δ_c = 0`` hands the lab a matrix IPOPT itself would never factorize.

    ``compl`` — the ``get_current_violations()`` dict, available only from inside
    an intermediate callback — switches ``Σ`` from the central-path ``z²/μ`` to
    the exact ``z/s``. Pass it whenever it exists: at an intermediate iterate the
    two are genuinely different matrices.

    Returns ``(arrow, record)``; the record is one row of the path table.
    """
    xl, xu = bounds(prob)
    mg = np.asarray(info["mult_g"], float)
    zl = np.asarray(info["mult_x_L"], float)
    zu = np.asarray(info["mult_x_U"], float)
    # Default = IPOPT's Alg. IC value, applied unconditionally. Note IPOPT itself
    # only reaches for δ_c when it *detects* a rank-deficient Jacobian, so our
    # unconditional use is a simplification that can pre-empt a δ_w IPOPT would
    # have needed — ``delta_c=0.0`` is the A/B lever that tests exactly that.
    if delta_c is None:
        delta_c = float(np.sqrt(np.finfo(float).eps) * max(mu, 0.0) ** 0.25)
    delta_c = float(delta_c)

    def build(dw):
        A = augmented_kkt(prob, x, mg, zl, zu, xl, xu, mu,
                          delta_w=dw, delta_c=delta_c, compl=compl)
        arrow = Arrowhead(A, owner, n_sub)
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
    # The monolithic dense cross-check is O(dim²) — a cheap luxury in 1D, guarded
    # anyway so a large --n cannot turn a diagnostic into the bottleneck.
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
          f"{'In(S)':>14} {'SPD':>4} {'target':>7} {'Haynsw':>8} {'rel-err':>9}")
    for r in records:
        hay = {True: "MATCH", False: "MISMATCH", None: "skipped"}[r["haynsworth"]]
        print(f"  {r['t']:>9.1e} {r['mu']:>9.1e} {r['delta_c']:>9.1e} "
              f"{r['delta_w']:>9.1e} {r['trials']:>4d} {str(r['inS']):>14} "
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
          f"(stride {stride}), IPOPT regularized {n_reg_ip}, DD asked for "
          f"δ_w > 0 at {n_reg_dd}")

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
          f"{'DD δ_w':>10} {'tri':>4} {'SPD':>4} {'target':>7} {'Haynsw':>7}")
    for r in shown:
        hay = {True: "MATCH", False: "MISMATCH", None: "—"}[r["haynsworth"]]
        print(f"      {r['t']:>9.1e} {r['iter']:>5d} {r['mu']:>9.1e} "
              f"{r['regu']:>10.1e} {r['delta_w']:>10.1e} {r['trials']:>4d} "
              f"{'yes' if r['spd'] else 'NO':>4} "
              f"{'hit' if r['hit'] else 'MISSED':>7} {hay:>7}"
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


def _shade_subdomains(ax, part, ymin=0.0, ymax=1.0):
    """Alternating background bands per subdomain + dashed interface lines.

    Subdomain ``k`` owns nodes ``[a, b]`` and ``k+1`` owns ``[b, c]``: they meet
    *at* the node ``b``, which is why the band edges coincide with the dashed
    interface lines rather than overlapping.
    """
    for k, (a, b) in enumerate(part.edge_ranges):
        ax.axvspan(a, b, color=("0.92" if k % 2 == 0 else "0.98"), zorder=0)
    for c in part.cut_nodes:
        ax.axvline(c, color="C3", ls="--", lw=1.0, alpha=0.8, zorder=1)


def plot_solution(prob, part, x, t_last, history, *, path=None, show=False,
                  title=None):
    """Four-panel view of the solution on the staggered grid.

    * **nodes** — clean / noisy / reconstruction, with the subdomain bands and
      the interface nodes marked (the board's ``N_k`` picture);
    * **edges** — the lift ``r = |Ku|`` against the dual radius ``δ = |q|``,
      drawn at the half-integer edge positions so the staggering is visible;
      the MPCC structure reads off directly: ``δ = 1`` (dual saturated) exactly
      where ``r > 0`` (a jump), ``δ < 1`` where the signal is flat;
    * **complementarity** — ``r·(1−δ)`` per edge against the level ``t``;
    * **continuation** — the Scholtes path, weight and ``max r·(1−δ)`` vs ``t``.
    """
    plt = _plt(show)
    n, mE, off = prob.n_nodes, prob.n_edges, prob.off
    u = x[:n]
    r = x[off["r"] : off["r"] + mE]
    d = x[off["delta"] : off["delta"] + mE]
    xn = np.arange(n)
    xe = np.arange(mE) + 0.5              # edges sit between the nodes
    # "Saturated" must be judged against the Scholtes level, not machine zero: δ
    # never reaches 1 (the barrier keeps it interior, and the relaxation only
    # forces 1−δ ≲ t/r — measured max δ = 0.99985 at t=2.2e-4, so a 1e-6 test
    # marks *nothing* active). Use the repo's own corner cap ε_w = min(c√t, ½),
    # the same O(√t) width the certificate uses to classify the corner.
    eps_w = min(3.0 * np.sqrt(max(t_last, 0.0)), 0.5)
    active = d >= 1.0 - eps_w             # dual on the ball ⇒ the jump set

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    ax = axes[0, 0]
    _shade_subdomains(ax, part)
    ax.plot(xn, prob.u_clean, color="0.45", lw=2.2, label="clean $u^\\dagger$",
            zorder=2)
    ax.plot(xn, prob.f, "o", ms=4, color="C3", alpha=0.75, label="noisy $f$",
            zorder=3)
    ax.plot(xn, u, "-o", ms=3.5, lw=1.6, color="C0", label="recon $u$", zorder=4)
    ax.set_xlabel("node $i$")
    ax.set_ylabel("$u$")
    ax.set_title(f"nodes — PSNR {psnr(prob.u_clean, prob.f):.2f} → "
                 f"{psnr(prob.u_clean, u):.2f} dB")
    if part.cut_nodes:                    # proxy handle: bands + dashed lines
        ax.plot([], [], color="C3", ls="--", lw=1.0,
                label=f"interface node ({len(part.cut_nodes)})")
    ax.legend(fontsize=8, loc="best")

    ax = axes[0, 1]
    _shade_subdomains(ax, part)
    ax.vlines(xe, 0.0, r, color="C0", lw=1.6, label=r"$r=|Ku|$", zorder=3)
    ax.plot(xe, r, "o", ms=3.5, color="C0", zorder=4)
    ax.set_xlabel("edge $e$  (half-integer positions)")
    ax.set_ylabel(r"$r = |Ku|$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    # δ as markers, not a line: consecutive edges are not interpolatable, and at
    # n=64 a connecting line is pure zigzag. Filled = saturated (δ=1, the dual is
    # on the ball) vs open = interior — i.e. the active set, read directly.
    if mE <= 24:
        ax2.plot(xe, d, "-", lw=0.9, color="C1", alpha=0.5, zorder=4)
    ax2.plot(xe[~active], d[~active], "s", ms=4, mfc="none", mec="C1", mew=1.2,
             zorder=5, label=r"interior ($\delta < 1-\epsilon_w$)")
    ax2.plot(xe[active], d[active], "s", ms=4, color="C1", zorder=6,
             label=rf"saturated ({active.sum()}/{mE})")
    ax2.axhline(1.0, color="C1", ls=":", lw=1.0)
    ax2.axhspan(1.0 - eps_w, 1.15, color="C1", alpha=0.08, zorder=0)
    ax2.set_ylabel(r"$\delta = |q|$   (ball radius)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=7, loc="lower right")
    ax.set_title(r"edges — the lift: $\delta\!\approx\!1$ (filled, band "
                 rf"$\epsilon_w\!=\!{eps_w:.1e}$) $\Leftrightarrow$ a jump $r>0$")

    ax = axes[1, 0]
    _shade_subdomains(ax, part)
    comp = r * (1.0 - d)
    ax.vlines(xe, 1e-20, np.maximum(comp, 1e-20), color="C2", lw=1.6, zorder=3)
    ax.plot(xe, np.maximum(comp, 1e-20), "o", ms=3.5, color="C2", zorder=4)
    ax.axhline(t_last, color="k", ls="--", lw=1.2,
               label=f"Scholtes level $t$ = {t_last:.1e}")
    ax.set_yscale("log")
    # Clamp to ~7 decades below the level: a couple of numerically-zero edges
    # would otherwise stretch the axis over 10+ decades and flatten everything.
    pos = comp[comp > 0]
    lo = max(min(pos.min() if pos.size else t_last, t_last) / 10, t_last * 1e-7)
    ax.set_ylim(lo, max(t_last, comp.max()) * 10)
    ax.set_xlabel("edge $e$")
    ax.set_ylabel(r"$r\,(1-\delta)$")
    ax.set_title("complementarity residual per edge")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 1]
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
        ax.invert_xaxis()                 # the continuation runs left → right
        ax.set_xlabel("Scholtes level $t$   (tightening →)")
        ax.set_ylabel("weight", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(ts, cr, "-s", ms=4, color="C2")
        ax2.plot(ts, ts, ":", lw=1.0, color="0.5")
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$\max\, r(1-\delta)$  (dotted: $t$)", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")
        ax.legend(fontsize=8, loc="best")
    ax.set_title("continuation path")

    fig.suptitle(title or "", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_arrowhead(prob, part, arrow, *, path=None, show=False, title=None):
    """The DD structure: the KKT before and after the arrowhead permutation.

    Left is IPOPT's augmented KKT in its natural ordering; middle is the *same
    matrix* permuted by ``[loc_1 … loc_k | border]`` — no reformulation, so the
    picture is the honest bordered block-diagonal with the border arms carrying
    real Jacobian entries. Right is the assembled interface matrix ``S``, whose
    size ``p = 2(k−1)+1`` is what the DD method actually has to solve densely.
    """
    plt = _plt(show)
    perm = np.concatenate(arrow.loc + [arrow.bord])
    Ap = arrow.A[perm][:, perm]
    ms = max(0.15, min(2.5, 400.0 / arrow.A.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    axes[0].spy(arrow.A, markersize=ms, color="C0")
    axes[0].set_title(f"augmented KKT, natural order\n{arrow.A.shape[0]}$\\times$"
                      f"{arrow.A.shape[0]}, nnz={arrow.A.nnz}", fontsize=9)

    ax = axes[1]
    ax.spy(Ap, markersize=ms, color="C0")
    o = 0
    for k, idx in enumerate(arrow.loc):                 # block outlines
        s = len(idx)
        ax.add_patch(plt.Rectangle((o - .5, o - .5), s, s, fill=False,
                                   ec="C2", lw=1.3))
        ax.text(o + s / 2, o + s / 2, f"$W_{{{k + 1}}}$", color="C2", fontsize=11,
                ha="center", va="center")
        o += s
    ax.add_patch(plt.Rectangle((o - .5, -.5), arrow.p, o, fill=False, ec="C3", lw=1.3))
    ax.add_patch(plt.Rectangle((-.5, o - .5), o, arrow.p, fill=False, ec="C3", lw=1.3))
    ax.set_title(f"permuted to arrowhead — {len(arrow.loc)} blocks + border "
                 f"$p$={arrow.p}\n(pure permutation, border in red)", fontsize=9)

    ax = axes[2]
    # Mask the structural zeros: with them included the colour scale runs to
    # log10(1e-30) and the real dynamic range washes out to one flat colour.
    S = np.abs(arrow.S)
    logS = np.where(S > 0, np.log10(np.maximum(S, 1e-300)), np.nan)
    cmap = plt.get_cmap("viridis").with_extremes(bad="0.9")
    hi = np.nanmax(logS) if np.isfinite(logS).any() else 0.0
    im = ax.imshow(logS, cmap=cmap, vmin=hi - 12, vmax=hi)
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}|S|$  (grey = structural 0)")
    names = ([f"$u_{{{c + 1}}}$" for c in part.cut_nodes]
             + [f"$q_{{{e + 1}}}$" for e in part.cut_edges] + [r"$\alpha$"])
    if arrow.p <= 20 and len(names) == arrow.p:
        ax.set_xticks(range(arrow.p))
        ax.set_yticks(range(arrow.p))
        ax.set_xticklabels(names, fontsize=7, rotation=90)
        ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"interface matrix $S = C - \\sum_k B_k W_k^{{-1}} B_k^\\top$\n"
                 f"{arrow.p}$\\times${arrow.p}, SPD after the $\\delta_w$ loop",
                 fontsize=9)

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
# printing
# ---------------------------------------------------------------------------
def show_operators(prob):
    n, mE = prob.n_nodes, prob.n_edges
    K = np.asarray(prob.K.todense()).astype(int)
    KT = np.asarray(prob.KT.todense()).astype(int)
    print(f"  K  ({mE}×{n}, nnz={prob.K.nnz})   (Ku)_e = u_{{e+1}} − u_e")
    for e in range(mE):
        print(f"    h2x[{e + 1}] | " + " ".join(f"{v:+2d}" if v else " ." for v in K[e]))
    print(f"  Kᵀ ({n}×{mE})   (Kᵀq)_i = q_{{i−1}} − q_i   ← what the state row carries")
    for i in range(n):
        print(f"    h1 [{i + 1}] | " + " ".join(f"{v:+2d}" if v else " ." for v in KT[i]))
    dead = [c for c in range(mE) if not KT[:, c].any()]
    print(f"  all-zero columns of Kᵀ: {dead}   (none — the grid is staggered, "
          f"not Neumann-padded)\n")


def show_partition(prob, part):
    print(f"  partition of {prob.n_edges} edges / {prob.n_nodes} nodes into "
          f"{part.n_sub} subdomains")
    print("    node  :  " + " ".join(f"{i + 1:2d}" for i in range(prob.n_nodes)))
    print("    owner :  " + " ".join(f"{o + 1:2d}" for o in part.node_owner))
    print("    edge  :    " + " ".join(f"{i + 1:2d}" for i in range(prob.n_edges)))
    print("    owner :    " + " ".join(f"{o + 1:2d}" for o in part.edge_owner))
    print("\n  local state equations (the board's N_k u − N_k f + α K_kᵀ q_k = 0):")
    for k in range(part.n_sub):
        print(local_state_rows(prob, part, k))
    border = ([f"u{c + 1}" for c in part.cut_nodes]
              + [f"q{e + 1}" for e in part.cut_edges] + ["α"])
    print(f"\n    interface nodes (in both local systems): "
          f"{[f'u{c + 1}' for c in part.cut_nodes]}")
    print(f"    complicating vector y = [{', '.join(border)}]   p = {len(border)} "
          f"= 2(k−1)+1\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="1D lifted TV-MPCC on a staggered node/edge grid "
                    "(the whiteboard example), with domain decomposition.")
    ap.add_argument("--n", type=int, default=7, help="number of NODES (edges = n−1)")
    ap.add_argument("--nsub", type=int, default=2, help="number of subdomains")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha0", type=float, default=None,
                    help="initial weight (linear) or log-weight (exp); default 0.7·σ")
    ap.add_argument("--weight", choices=["linear", "exp"], default="linear",
                    help="Q(α) = α (board, with an α ≥ 0 row) or e^α")
    ap.add_argument("--reg-alpha", type=float, default=1e-4, dest="reg_alpha")
    ap.add_argument("--w-max", type=float, default=float("inf"), dest="w_max",
                    help="optional upper cap on the linear weight (default none, "
                         "as in linalpha); a safety net for vacuous levels")
    ap.add_argument("--c-theta", type=float, default=0.0, dest="c_theta",
                    help="TR gauge ridge ½·(c_θ·t)·‖θ − θ_ref‖² (the D1 fix)")
    ap.add_argument("--init", choices=["cp", "cp-scan", "cold"], default="cp")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--t-min", type=float, default=1e-4, dest="t_min")
    ap.add_argument("--factor", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=1000, dest="max_iter")
    ap.add_argument("--linear-solver", default="ma57", dest="linear_solver")
    ap.add_argument("--hess-update", default="exact", dest="hess_update",
                    choices=["exact", "bfgs", "sr1"])
    ap.add_argument("--dual-warmstart", action="store_true", dest="dual_warmstart")
    ap.add_argument("--print-level", type=int, default=0, dest="print_level")
    ap.add_argument("--show-operators", action="store_true", dest="show_ops")
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
                         "expensive clock)")
    ap.add_argument("--dd-sigma-mu", action="store_true", dest="dd_sigma_mu",
                    help="per-iteration probe: use the central-path Σ = z²/μ "
                         "instead of the exact Σ = z/s from "
                         "get_current_violations (an A/B lever — it makes the "
                         "probe factorize a different matrix from IPOPT's)")
    ap.add_argument("--dd-delta-c", type=float, default=-1.0, dest="dd_delta_c",
                    metavar="VAL",
                    help="per-iteration probe: override δ_c (default <0 = "
                         "IPOPT's √ε·μ^¼). Use 0 to test whether our "
                         "unconditional δ_c is pre-empting δ_w")
    ap.add_argument("--fd-check", action="store_true", dest="fd_check")
    ap.add_argument("--save-plot", default=None, dest="save_plot",
                    help="write the 4-panel solution figure to this PNG")
    ap.add_argument("--save-dd-plot", default=None, dest="save_dd_plot",
                    help="write the arrowhead/interface structure figure to this PNG")
    ap.add_argument("--show", action="store_true",
                    help="display the figures interactively instead of only saving")
    args = ap.parse_args()

    n, k = args.n, args.nsub
    if n < 3 or k < 1 or k > n - 1:
        raise SystemExit("need n ≥ 3 and 1 ≤ nsub ≤ n−1")
    w0 = args.alpha0 if args.alpha0 is not None else 0.7 * args.sigma
    if args.weight == "exp" and args.alpha0 is not None:
        w0 = float(np.exp(args.alpha0))          # --alpha0 is the log-weight
    if w0 <= 0.0:
        raise SystemExit("the initial weight must be > 0")

    u_clean, f = make_signal(n, args.sigma, args.seed)
    prob = Lifted1DMPCC(f, u_clean, weight=args.weight, reg_alpha=args.reg_alpha,
                        w_max=args.w_max)
    part = Partition1D(n, k)

    print(f"1D lifted TV-MPCC (staggered)   nodes={n}  edges={prob.n_edges}  "
          f"subdomains={k}")
    print(f"  vars={prob.n} (n + 5(n−1) + 1)   rows={prob.m_con} "
          f"({prob.n_eq} eq + {prob.n_ineq} ineq)   "
          f"KKT dim={prob.n + 2 * prob.n_ineq + prob.n_eq}")
    print(f"  weight Q(α) = {'α' if args.weight == 'linear' else 'e^α'}"
          f"{'  (+ explicit row ha: α ≥ 0)' if prob.has_ha else '  (α boxed)'}"
          f"   initial weight = {w0:.4f}   init = {args.init}\n")

    if args.show_ops or n <= 12:
        show_operators(prob)
    show_partition(prob, part)

    x0 = initial_point(prob, w0, init=args.init)
    prob.theta_ref = x0[prob.off["theta"] : prob.off["theta"] + prob.n_edges].copy()

    if args.fd_check:
        je, ge, he = fd_check(prob, x0)
        print(f"  fd-check at the warm start:  J={je:.2e}  g={ge:.2e}  H={he:.2e}\n")

    if args.no_solve:
        return

    schedule, t = [], args.t0
    while t >= args.t_min:
        schedule.append(t)
        t *= args.factor

    # The DD structure is t-independent — it depends on the partition and the
    # index layout only — so it is built once, outside the continuation, and
    # every level reuses it. That is what makes probing the whole path cheap.
    owner = kkt_owner(prob, part)

    dd_path, dd_iters = [], []

    def probe_level(record, x_l, info_l, mu_l):
        _, rec = dd_probe(prob, owner, k, x_l, info_l, mu_l, seed=args.seed)
        rec["converged"] = record["converged"]
        dd_path.append(rec)

    def probe_iter(iterate, viol, iter_count, mu_i, regu, alg_mod):
        if iter_count % args.dd_iters:
            return
        # No LU cross-check here: exactness of the permutation is a structural
        # property already certified once per level. What this clock is for is
        # δ_w vs IPOPT's own, so pay only for the inertia loop. `viol` buys the
        # exact Σ = z/s — without it we would be regularizing a *different*
        # matrix from the one IPOPT factorizes, and the comparison would be void.
        _, rec = dd_probe(prob, owner, k, np.asarray(iterate["x"], float),
                          iterate, mu_i, seed=args.seed, check_solve=False,
                          compl=None if args.dd_sigma_mu else viol,
                          delta_c=None if args.dd_delta_c < 0 else args.dd_delta_c)
        rec.update(iter=int(iter_count), regu=float(regu),
                   alg_mod=int(alg_mod))
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

    off, mE = prob.off, prob.n_edges
    u = x[: n]
    a = float(x[off["alpha"]])
    print(f"\n  reported level t={t_last:.1e} ({n_it} iterations total):  "
          f"α* = {a:.6f}   weight = {prob.weight_of_alpha(a):.6f}")
    print(f"    PSNR noisy {psnr(u_clean, f):.2f} dB → recon {psnr(u_clean, u):.2f} dB "
          f"({psnr(u_clean, u) - psnr(u_clean, f):+.2f})")
    if n <= 24:
        r = x[off["r"] : off["r"] + mE]
        d = x[off["delta"] : off["delta"] + mE]
        q = x[off["qx"] : off["qx"] + mE]
        print("    nodes   u_clean:", " ".join(f"{v:6.2f}" for v in u_clean))
        print("            noisy f:", " ".join(f"{v:6.2f}" for v in f))
        print("            recon u:", " ".join(f"{v:6.2f}" for v in u))
        print("    edges   qx     :", "   " + " ".join(f"{v:6.2f}" for v in q))
        print("            r=|Ku| :", "   " + " ".join(f"{v:6.2f}" for v in r))
        print("            δ=|q|  :", "   " + " ".join(f"{v:6.2f}" for v in d))
        print("            r(1−δ) :", "   " + " ".join(f"{v:6.2f}" for v in r * (1 - d)),
              "  ← complementarity, ≤ t")

    stamp = (f"1D staggered lifted TV-MPCC — n={n}, k={k}, σ={args.sigma}, "
             f"Q(α)={'α' if args.weight == 'linear' else 'e^α'}, "
             f"t={t_last:.1e}, weight={prob.weight_of_alpha(a):.4f}")
    if args.save_plot or args.show:
        plot_solution(prob, part, x, t_last, history, path=args.save_plot,
                      show=args.show, title=stamp)
        if args.save_plot:
            print(f"  saved solution figure → {args.save_plot}")

    if args.no_dd:
        return

    # ---- the arrowhead on the reported iterate ---------------------------
    print(f"\n  domain-decomposition solve of the KKT at the reported level "
          f"t={t_last:.1e}:")
    arrow, rec = dd_probe(prob, owner, k, x, info, mu, seed=args.seed)
    print(f"    inertia loop: δ_w={rec['delta_w']:.1e} after {rec['trials']} "
          f"trial(s){'' if rec['corrected'] else '  [NOT corrected]'}"
          f"   δ_c={rec['delta_c']:.1e}")
    dims = [len(i) for i in arrow.loc]
    print(f"    arrowhead: {k} blocks of dim {dims}, border p={arrow.p}   "
          f"(total {sum(dims) + arrow.p} = {arrow.A.shape[0]})")
    print(f"    blocks are mutually decoupled: {rec['decoupled']}")
    print(f"    direct interface solve: rel-err vs monolithic LU = {rec['err']:.2e}")
    hay = {True: "MATCH", False: "MISMATCH", None: "skipped (dim)"}[rec["haynsworth"]]
    print(f"    Haynsworth  Σ In(W_k)+In(S)={rec['inertia']}  "
          f"vs monolithic In(A): {hay}")
    print(f"    IPOPT target {rec['target']}  {'hit' if rec['hit'] else 'MISSED'};  "
          f"S is {'SPD' if rec['spd'] else 'NOT SPD ' + str(rec['inS'])}")

    if args.save_dd_plot or args.show:
        plot_arrowhead(prob, part, arrow, path=args.save_dd_plot, show=args.show,
                       title=stamp)
        if args.save_dd_plot:
            print(f"    saved arrowhead figure → {args.save_dd_plot}")


if __name__ == "__main__":
    main()
