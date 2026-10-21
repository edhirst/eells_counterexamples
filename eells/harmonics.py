"""Spherical harmonics on S^3 and the G-symmetric bases of Proposition 2.2.

Jacobi basis (Lachieze-Rey), orthonormal for the volume 2 pi^2 of S^3:

    Y_{m,n,j}(z, w) = c_{|m|,|n|,j} z^{(m)} w^{(n)} P_j^{(|m|,|n|)}(1 - 2|z|^2),
    c_{a,b,j}^2     = (2j+a+b+1) (j+a+b)! j! / (2 pi^2 (j+a)! (j+b)!),     l = |m| + |n| + 2j,

with z^{(m)} = z^m (m >= 0), zbar^{|m|} (m < 0).  Y_{m,n,j} is the restriction of a harmonic
homogeneous polynomial of degree l, so  Delta Y = -l(l+2) Y  on S^3.

Lemma 2.1:  e^{i theta}.Y = e^{i(m-2n) theta} Y,   kappa.Y = conj(Y),   epsilon.Y = (-1)^{m+n} Y.

Proposition 2.2 (G-equivariant vector fields, L^2-orthonormal):
    U-modes  (Re Y_{2n-4,n,j}, Im Y_{2n-4,n,j}, 0),   n even,
    V-modes  sqrt(2) (0, 0, Re Y_{2n,n,j}),            n odd >= 1,
and G-invariant scalars  Y_{0,0,j},  sqrt(2) Re Y_{2n,n,j}  (n even >= 2).

**Evaluation.**  On the grid of :class:`eells.grid.OrbitGrid` the points are
``(z, w) = (sqrt(s), sqrt(1-s) e^{i phi})`` with ``z`` real, so with ``rs = sqrt(s)``,
``rc = sqrt(1-s)``, ``a = |m|``, ``b = |n|``, ``P = P_j^{(a,b)}(1-2s)``, ``P' = dP/dx``:

    Y = c rs^a rc^b P e^{i n phi}.

Writing the real gradient on R^4 = C^2 in the coordinates (x1, y1, x2, y2) = (Re z, Im z,
Re w, Im w) via Wirtinger derivatives (d/dx1 = d_z + d_zbar, d/dy1 = i (d_z - d_zbar), and
likewise for w) one gets, at the grid points,

    d_{x1} Y = c rc^b [ a rs^{a-1} P - 4 rs^{a+1} P' ] e^{i n phi},
    d_{y1} Y = i c rc^b  m rs^{a-1} P                e^{i n phi},
    d_{x2} Y = c rs^a P  b rc^{b-1}                   e^{i (n - sgn n) phi},
    d_{y2} Y = i c rs^a P n rc^{b-1}                  e^{i (n - sgn n) phi},

(the terms with rs^{a-1}, rc^{b-1} are absent when a = 0, b = 0).  The tangential gradient
on S^3 is  grad f - (x . grad f) x,  with x = (rs, 0, rc cos phi, rc sin phi) the grid point.

Everything below runs on float arrays or on rigorous :class:`eells.ball.Ball` arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

import numpy as np

from . import arith as ar
from .ball import Ball


@dataclass(frozen=True, order=True)
class Mode:
    """One basis element: kind 'U' (vector, n even), 'V' (vector, n odd) or 'S' (invariant scalar)."""
    l: int
    kind: str
    n: int
    j: int

    @property
    def m(self):
        return 2 * self.n - 4 if self.kind == "U" else 2 * self.n

    @property
    def a(self):
        return abs(self.m)

    @property
    def b(self):
        return abs(self.n)

    def label(self):
        return f"{self.kind}(m={self.m},n={self.n},j={self.j})"


def vector_modes(D):
    """The basis (Psi_k) of Proposition 2.2 with degree l_k <= D, sorted by (l, kind, n, j)."""
    out = []
    for n in range(-D, D + 1):
        if n % 2 == 0:
            a, b = abs(2 * n - 4), abs(n)
            for j in range((D - a - b) // 2 + 1):
                if a + b + 2 * j <= D:
                    out.append(Mode(a + b + 2 * j, "U", n, j))
    for n in range(1, D + 1, 2):
        a, b = 2 * n, n
        for j in range((D - a - b) // 2 + 1):
            if a + b + 2 * j <= D:
                out.append(Mode(a + b + 2 * j, "V", n, j))
    return sorted(out)


def scalar_modes(D):
    """G-invariant scalar harmonics of degree <= D (Proposition 2.2), n even >= 0, m = 2n."""
    out = []
    for n in range(0, D + 1, 2):
        a, b = 2 * n, n
        for j in range((D - a - b) // 2 + 1):
            if a + b + 2 * j <= D:
                out.append(Mode(a + b + 2 * j, "S", n, j))
    return sorted(out)


def norm_constant_squared_times_2pi2(a, b, j):
    """The rational number q with c_{a,b,j}^2 = q / (2 pi^2)."""
    return Fraction((2 * j + a + b + 1) * factorial(j + a + b) * factorial(j),
                    factorial(j + a) * factorial(j + b))


def jacobi_table(a, b, jmax, x):
    """[P_0^{(a,b)}(x), ..., P_jmax^{(a,b)}(x)] by the three-term recurrence (any backend)."""
    one = ar.zeros_like(x) + 1.0
    P = [one]
    if jmax >= 1:
        P.append((x - 1) * (0.5 * (a + b + 2)) + (a + 1))
    for k in range(1, jmax):
        q = 2 * k + a + b
        c1 = 2 * (k + 1) * (k + a + b + 1) * q
        alpha = (q + 1) * (q + 2) * q
        beta = (q + 1) * (a * a - b * b)
        c3 = 2 * (k + a) * (k + b) * (q + 2)
        assert max(c1, alpha, abs(beta), c3) < 2 ** 53
        P.append(((x * alpha + beta) * P[k] - P[k - 1] * c3) * (1.0 / c1) if not ar.is_ball(x)
                 else ((x * alpha + beta) * P[k] - P[k - 1] * c3) / Ball.exact(float(c1)))
    return P


class RigorousJacobiTables:
    """P_j^{(a,b)}(x_i) at the Chebyshev-Lobatto nodes as balls, for the rigorous backend.

    The three-term recurrence suffers from interval wrapping (the enclosure widths grow like
    (1 + sqrt 2)^j), so it is run in mpmath interval arithmetic with 80 + j_max/2 decimal digits
    (the wrapping costs at most log10(3) digits per step) and the results are rounded outward
    to double-precision balls.  Tables are cached on disk by (Ns, a, b, j_max)."""

    def __init__(self, Ns, cache_dir=None):
        import mpmath as mp
        self.Ns = Ns
        self.cache_dir = cache_dir
        self._mem = {}
        iv = mp.iv
        old = iv.prec
        iv.dps = 220
        try:
            self.xs = [iv.cos(iv.pi * i / Ns) for i in range(Ns + 1)]
            self.xs[0] = iv.mpf(1)
            if Ns % 2 == 0:
                self.xs[Ns // 2] = iv.mpf(0)
            self.xs[Ns] = iv.mpf(-1)
        finally:
            iv.prec = old

    def table(self, a, b, jmax):
        """[P_0, ..., P_jmax] at all nodes, each a Ball of shape (Ns + 1,)."""
        import mpmath as mp
        key = (a, b, jmax)
        if key in self._mem:
            return self._mem[key]
        fn = None
        if self.cache_dir is not None:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
            fn = os.path.join(self.cache_dir, f"jacobi_Ns{self.Ns}_a{a}_b{b}_j{jmax}.npz")
            if os.path.exists(fn):
                d = np.load(fn)
                tab = [Ball(d["m"][k], d["r"][k]) for k in range(jmax + 1)]
                self._mem[key] = tab
                return tab
        iv = mp.iv
        old = iv.prec
        iv.dps = 80 + jmax // 2
        try:
            rows = []
            for x in self.xs:
                P = [iv.mpf(1)]
                if jmax >= 1:
                    P.append((a + 1) + (a + b + 2) * (x - 1) / 2)
                for k in range(1, jmax):
                    q = 2 * k + a + b
                    c1 = 2 * (k + 1) * (k + a + b + 1) * q
                    alpha = (q + 1) * (q + 2) * q
                    beta = (q + 1) * (a * a - b * b)
                    c3 = 2 * (k + a) * (k + b) * (q + 2)
                    P.append(((x * alpha + beta) * P[k] - P[k - 1] * c3) / c1)
                rows.append(P)
            tab = [iv_list_to_ball([rows[i][k] for i in range(self.Ns + 1)]) for k in range(jmax + 1)]
        finally:
            iv.prec = old
        if fn is not None:
            np.savez(fn, m=np.stack([t.m for t in tab]), r=np.stack([t.r for t in tab]))
        self._mem[key] = tab
        return tab


def iv_list_to_ball(xs):
    """Fast outward rounding of a list of mpmath intervals to a Ball array."""
    import mpmath as mp
    from .ball import _up
    ms, rs = [], []
    ctx = mp.mp
    with mp.workprec(mp.iv.prec + 64):
        for x in xs:
            lo_t, hi_t = x._mpi_                      # exact binary endpoints
            a, b = ctx.make_mpf(lo_t), ctx.make_mpf(hi_t)
            m = float((a + b) / 2)                    # nearest double
            mm = ctx.mpf(m)                           # exact
            rs.append(_up(float(max(abs(mm - a), abs(b - mm)))))
            ms.append(m)
    return Ball(np.array(ms), np.array(rs))


def jacobi_explicit(a, b, j, s):
    """P_j^{(a,b)}(1 - 2 s) by the explicit sum of the paper (used only in the tests)."""
    from math import comb
    return sum((-1) ** k * comb(j + a, j - k) * comb(j + a + b + k, k) * s ** k for k in range(j + 1))


class HarmonicEvaluator:
    """Evaluates basis modes (values, tangential gradients, Laplacians) on an OrbitGrid."""

    def __init__(self, grid, max_degree=None, cache_dir=None):
        self.g = grid
        self.rig = grid.rigorous
        self._pow = {"s": {}, "c": {}}
        self._jac = {}
        self.x = grid.x
        self.max_degree = max_degree
        if self.rig:
            self._two_pi2 = 2 * grid.pi * grid.pi
            self._tables = RigorousJacobiTables(grid.Ns, cache_dir)

    # ---- cached radial ingredients
    def power(self, which, k):
        """rs^k or rc^k (binary powering, so that the ball radius grows only like log k)."""
        tab = self._pow[which]
        if k not in tab:
            base = self.g.rs if which == "s" else self.g.rc
            tab[k] = base ** k if self.rig else base ** k
        return tab[k]

    def jacobi(self, a, b, j):
        """P_j^{(a,b)}(x_i) at the nodes.  The table of a family (a, b) is built up to the largest
        j occurring for a mode of degree <= max_degree (or j if larger)."""
        key = (a, b)
        if key not in self._jac or len(self._jac[key]) <= j:
            jmax = j
            if self.max_degree is not None:
                jmax = max(j, (self.max_degree - a - b) // 2)
            if self.rig:
                self._jac[key] = self._tables.table(a, b, jmax)
            else:
                self._jac[key] = jacobi_table(a, b, jmax, self.x)
        return self._jac[key][j]

    def norm_constant(self, mode):
        """c_{a,b,j}, times sqrt(2) for V-modes and for scalar modes with n >= 2."""
        q = norm_constant_squared_times_2pi2(mode.a, mode.b, mode.j)
        if mode.kind == "V" or (mode.kind == "S" and mode.n >= 2):
            q = 2 * q
        if self.rig:
            return (Ball.from_fraction(q) / self._two_pi2).sqrt()
        return float(np.sqrt(float(q) / (2 * np.pi ** 2)))

    # ---- the complex harmonic and its ambient partial derivatives at the grid points
    def _radial(self, mode):
        a, b, j, m, n = mode.a, mode.b, mode.j, mode.m, mode.n
        c = self.norm_constant(mode)
        P = self.jacobi(a, b, j)
        dP = self.jacobi(a + 1, b + 1, j - 1) * (0.5 * (j + a + b + 1)) if j > 0 else None
        rs_a, rc_b = self.power("s", a), self.power("c", b)
        R = rs_a * rc_b * P * c
        # d_{x1}:  c rc^b [ a rs^{a-1} P - 4 rs^{a+1} P' ]
        X1 = rc_b * c
        t = None
        if a > 0:
            t = self.power("s", a - 1) * P * a
        if dP is not None:
            t2 = self.power("s", a + 1) * dP * (-4.0)
            t = t2 if t is None else t + t2
        X1 = X1 * t if t is not None else ar.zeros_like(R)
        # d_{y1}:  i c rc^b m rs^{a-1} P
        Y1 = rc_b * self.power("s", a - 1) * P * (c * m) if a > 0 else ar.zeros_like(R)
        # d_{x2}, d_{y2}:  c rs^a P b rc^{b-1},  i c rs^a P n rc^{b-1}   (times e^{i(n - sgn n) phi})
        if b > 0:
            base = rs_a * P * self.power("c", b - 1) * c
            X2, Y2 = base * b, base * n
        else:
            X2 = Y2 = ar.zeros_like(R)
        return R, X1, Y1, X2, Y2

    def complex_harmonic(self, mode):
        """(Re Y, Im Y) and the ambient partials [(Re, Im) for x1, y1, x2, y2] on the grid."""
        R, X1, Y1, X2, Y2 = (v.reshape(-1, 1) for v in self._radial(mode))
        n = mode.n
        nn = n - (1 if n > 0 else -1 if n < 0 else 0)
        cn, sn = self.g.cos_n_phi(n).reshape(1, -1), self.g.sin_n_phi(n).reshape(1, -1)
        cnn, snn = self.g.cos_n_phi(nn).reshape(1, -1), self.g.sin_n_phi(nn).reshape(1, -1)
        val = (R * cn, R * sn)
        grad = [(X1 * cn, X1 * sn),          # d_{x1} Y = X1 e^{in phi}
                (-(Y1 * sn), Y1 * cn),       # d_{y1} Y = i Y1 e^{in phi}
                (X2 * cnn, X2 * snn),        # d_{x2} Y = X2 e^{i(n - sgn n) phi}
                (-(Y2 * snn), Y2 * cnn)]     # d_{y2} Y = i Y2 e^{i(n - sgn n) phi}
        return val, grad

    def tangential(self, grad_f):
        """Tangential gradient on S^3:  grad f - (x . grad f) x  at the grid point x.
        (The expression differentiated above is the non-homogeneous extension
        c z^{(m)} w^{(n)} P_j(1 - 2|z|^2) of Y, which agrees with Y on S^3; its ambient gradient
        differs from that of the homogeneous extension by a normal vector only.)"""
        X = self.g.X
        radial = grad_f[0] * X[0] + grad_f[2] * X[2] + grad_f[3] * X[3]      # X[1] = Im z = 0
        return [grad_f[0] - radial * X[0], grad_f[1], grad_f[2] - radial * X[2], grad_f[3] - radial * X[3]]

    def evaluate(self, modes):
        """Fields Psi_k for the vector modes: returns (val, grad, lap) with shapes
        val [k, c, i, b],  grad [k, a, c, i, b] (tangential partial d_a of component c),
        lap [k, c, i, b] = -l(l+2) val."""
        vals, grads, laps = [], [], []
        zero = ar.zeros(self.g.shape, self.rig)
        for md in modes:
            (re, im), grad = self.complex_harmonic(md)
            if md.kind == "U":
                comps = [re, im, zero]
                gcomps = [self.tangential([g[0] for g in grad]),
                          self.tangential([g[1] for g in grad]),
                          [zero] * 4]
            elif md.kind == "V":
                comps = [zero, zero, re]
                gcomps = [[zero] * 4, [zero] * 4, self.tangential([g[0] for g in grad])]
            else:
                raise ValueError("vector modes only")
            vals.append(ar.stack(comps))
            grads.append(ar.stack([ar.stack([gcomps[c][a] for c in range(3)]) for a in range(4)]))
            laps.append(vals[-1] * float(-md.l * (md.l + 2)))
        return ar.stack(vals), ar.stack(grads), ar.stack(laps)

    def evaluate_scalar(self, mode, unnormalised=False):
        """Value of a G-invariant scalar mode (kind 'S'): Re Y (times sqrt 2 for n >= 2), or, with
        ``unnormalised=True``,  Re(z^{2n} w^n) P_j^{(2n,n)}(1 - 2 s)  with coefficient 1."""
        assert mode.kind == "S"
        (re, _), _ = self.complex_harmonic(mode)
        if unnormalised:
            re = re / self.norm_constant(mode)
        return re
