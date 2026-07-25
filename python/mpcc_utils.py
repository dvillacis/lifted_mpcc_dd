"""Generic utilities for the lifted TV-denoising MPCC — data generation for the
C++ domain-decomposition solver (``../cpp``).

This module is a *self-contained extraction* of the pieces the reproduction
helpers (``dump_data*.py``) need from the project's Python reference
implementation: the synthetic/real instance builders, the staggered
discretization, the lifted 1D/2D MPCC problem objects, the Chambolle–Pock warm
start, and the domain-decomposition partition + KKT owner-map rule. The full
reference solvers (IPOPT continuation driver, certificate, plotting) are not
part of this archival package.

Public API
----------
    make_signal(n, sigma, seed)            → (u_clean, f)         1D synthetic
    make_phantom(N, sigma, seed)           → (u_clean, f)         2D synthetic
    load_image(path, N, sigma, seed)       → (u_clean, f)         2D from image
    Lifted1DMPCC, Partition1D              staggered-1D problem + partition
    Lifted2DMPCC, Partition2D              staggered-2D problem + partition
    initial_point(prob, w0, init=...)      → x0    (dispatches on problem type)
    kkt_owner(prob, part)                  → owner map (1D) / (owner, col, _) (2D)

The 1D and 2D ``initial_point`` / ``kkt_owner`` / ``cp_scan`` are kept as
separate ``*_1d`` / ``*_2d`` implementations (they operate on different meshes);
the two public names dispatch on the problem class.
"""

import numpy as np
import scipy.sparse as sp

# ======================================================================
# Instance builders (from lifted_mpcc_unitball_v2)
# ======================================================================
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

# ======================================================================
# Staggered 1D: problem, partition, Chambolle–Pock, owner map
# ======================================================================
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


def cp_scan_1d(prob, w0, half_width=2.0, n_grid=9):
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


def initial_point_1d(prob, w0, init="cp"):
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
            w0, u, q = cp_scan_1d(prob, w0)
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


def kkt_owner_1d(prob, part: Partition1D) -> np.ndarray:
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

# ======================================================================
# Staggered 2D: problem, partition, Chambolle–Pock, owner map
# ======================================================================
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


def cp_scan_2d(prob, w0, half_width=2.0, n_grid=9):
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


def initial_point_2d(prob, w0, init="cp"):
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
            w0, u, qx, qy = cp_scan_2d(prob, w0)
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


def kkt_owner_2d(prob, part: Partition2D):
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

# ======================================================================
# Public dispatchers (1D vs 2D by problem type)
# ======================================================================
def initial_point(prob, *args, **kwargs):
    if isinstance(prob, Lifted2DMPCC):
        return initial_point_2d(prob, *args, **kwargs)
    return initial_point_1d(prob, *args, **kwargs)


def kkt_owner(prob, *args, **kwargs):
    if isinstance(prob, Lifted2DMPCC):
        return kkt_owner_2d(prob, *args, **kwargs)
    return kkt_owner_1d(prob, *args, **kwargs)

