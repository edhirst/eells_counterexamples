"""The orbit-space grid, exact quadrature and rigorous sup/inf bounds for G-invariant scalars.

**Orbit space.**  Every point of S^3 lies on the circle orbit of a point
``(z, w) = (sqrt(s), sqrt(1-s) e^{i phi})`` with ``s = |z|^2 in [0,1]`` and ``phi in [0, 2 pi)``
(use the circle action to make ``z`` real and non-negative).  A circle-invariant function is
therefore a function ``f(s, phi)``, and (volume element ``ds dpsi_1 dpsi_2 / 2`` on S^3)

    int_{S^3} f dvol = pi * int_0^1 ds int_0^{2 pi} f(s, phi) dphi.

**Polynomial structure.**  A G-invariant polynomial of degree ``p`` in ``(z, zbar, w, wbar)``
is, in the variables ``x = 1 - 2 s in [-1, 1]`` and ``phi``, a polynomial of degree ``<= p/2`` in
``x`` times trigonometric polynomials of degree ``<= p/3`` in ``phi`` with even frequencies only
(a monomial ``z^a zbar^b w^c wbar^d`` is circle-invariant iff ``a - b = 2(c - d)``, and
epsilon-invariant iff ``a + b + c + d`` is even; it equals ``s^{(a+b)/2} (1-s)^{(c+d)/2}
e^{i (c-d) phi}``).

**Grid.**  Chebyshev-Lobatto nodes ``x_i = cos(pi i / Ns)``, ``i = 0..Ns``, in ``x`` and
equispaced nodes ``phi_b = 2 pi b / Nphi`` in ``phi``.  Clenshaw-Curtis weights integrate
polynomials of degree ``<= Ns`` in ``x`` exactly; the trapezoidal rule integrates trigonometric
polynomials exactly as long as no nonzero frequency is a multiple of ``Nphi``.  Hence
``integrate(f, deg)`` is exact whenever ``deg <= 2 Ns`` and ``deg < 3 Nphi``.

**Chebyshev-Fourier analysis.**  For the same grid, Chebyshev interpolation in ``x`` (exact for
degree ``<= Ns``) and the discrete Fourier transform in ``phi`` (exact for frequencies
``< Nphi/2``) give the exact expansion coefficients of a G-invariant polynomial.  From the
coefficients, ``ChebFourier.bounds`` returns a rigorous enclosure of ``[min f, max f]`` over S^3
by evaluating the expansion on a fine grid and adding a Lipschitz remainder
(``|T_k'| <= k^2``, ``|d/dphi cos(n phi)| <= n``).

Every array is indexed ``[i, b]`` (node in ``x`` or ``s``, node in ``phi``).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import arith as ar
from .ball import Ball, cos_sin_table, pi_ball, asball


class OrbitGrid:
    def __init__(self, Ns, Nphi, rigorous=False):
        assert Ns >= 2 and Nphi >= 2
        self.Ns, self.Nphi, self.rigorous = Ns, Nphi, rigorous
        c2, s2 = cos_sin_table(2 * Ns, rigorous)          # cos(pi p / Ns) = cos(2 pi p / (2 Ns))
        self._cos_pi_over_Ns = c2
        self.x = c2[: Ns + 1]                               # x_i = cos(pi i / Ns), i = 0..Ns
        half = ar.const(0.5, rigorous)
        self.s = (1 - self.x) * half                        # s_i = |z|^2
        self.rs = ar.sqrt(self.s, nonneg=True)                           # |z|   = sqrt(s)
        self.rc = ar.sqrt((1 + self.x) * half, nonneg=True)              # |w|   = sqrt(1 - s)
        self.cphi, self.sphi = cos_sin_table(Nphi, rigorous)
        self.pi = pi_ball() if rigorous else np.pi
        # Chebyshev analysis matrix:  c_k = sum_i D[k, i] f(x_i)  for the interpolant
        # f(x) = sum_{k=0}^{Ns} c_k T_k(x);   D[k, i] = (2/Ns) e_i e_k cos(pi i k / Ns),
        # e_0 = e_Ns = 1/2, otherwise 1.
        pref = [[Fraction(2, Ns) * _edge(i, Ns) * _edge(k, Ns) for i in range(Ns + 1)] for k in range(Ns + 1)]
        idx = np.array([[(i * k) % (2 * Ns) for i in range(Ns + 1)] for k in range(Ns + 1)])
        if rigorous:
            pm = Ball.from_fractions([q for row in pref for q in row]).reshape(Ns + 1, Ns + 1)
            self.D = pm * c2[idx]
        else:
            self.D = np.array([[float(q) for q in row] for row in pref]) * c2[idx]
        # Clenshaw-Curtis weights on [0, 1]:  int_0^1 g ds = (1/2) int_{-1}^{1} g dx
        #   = sum_{k even} c_k / (1 - k^2)   (int_{-1}^{1} T_k = 2/(1-k^2) for even k, 0 for odd k)
        ik = [Fraction(1, 1 - k * k) if k % 2 == 0 else Fraction(0) for k in range(Ns + 1)]
        ikv = Ball.from_fractions(ik) if rigorous else np.array([float(q) for q in ik])
        self.w_s = (ikv.reshape(-1, 1) * self.D).sum(axis=0)          # (Ns+1,)
        # full weights  omega[i, b] = pi * w_i * (2 pi / Nphi)  for  int_{S^3} f = sum omega f
        scale = (2 * self.pi * self.pi) * ar.const(Fraction(1, Nphi), rigorous) if rigorous \
            else 2 * np.pi ** 2 / Nphi
        self.omega = (self.w_s * scale).reshape(-1, 1) * ar.ones((1, Nphi), rigorous)
        # ambient coordinates of the grid points  x = (Re z, Im z, Re w, Im w)
        rs2 = self.rs.reshape(-1, 1)
        rc2 = self.rc.reshape(-1, 1)
        self.X = [rs2 * ar.ones((1, Nphi), rigorous), ar.zeros((Ns + 1, Nphi), rigorous),
                  rc2 * self.cphi.reshape(1, -1), rc2 * self.sphi.reshape(1, -1)]
        self.shape = (Ns + 1, Nphi)

    # ------------------------------------------------------------------ helpers
    def cos_n_phi(self, n):
        """cos(n phi_b) for b = 0..Nphi-1 (exact index arithmetic, n any integer)."""
        return self.cphi[(n * np.arange(self.Nphi)) % self.Nphi]

    def sin_n_phi(self, n):
        return self.sphi[(n * np.arange(self.Nphi)) % self.Nphi]

    def exact_for(self, deg):
        """Is the quadrature exact for G-invariant polynomials of degree ``deg``?"""
        return deg <= 2 * self.Ns and deg < 3 * self.Nphi

    def analysis_exact_for(self, deg):
        """Are the Chebyshev-Fourier coefficients exact for degree ``deg``?"""
        return deg // 2 <= self.Ns and 2 * (deg // 3) < self.Nphi

    # ------------------------------------------------------------------ quadrature
    def integrate(self, f, deg):
        """int_{S^3} f dvol for a G-invariant polynomial f (values on the grid) of degree <= deg."""
        assert self.exact_for(deg), f"grid ({self.Ns},{self.Nphi}) is not exact for degree {deg}"
        return (self.omega * f).sum()

    def project(self, f, deg):
        """Same as integrate but for several functions stacked along leading axes."""
        assert self.exact_for(deg)
        return (self.omega * f).sum(axis=-1).sum(axis=-1)

    # ------------------------------------------------------------------ analysis
    def analyse(self, f, deg):
        """Exact Chebyshev-Fourier coefficients of a G-invariant polynomial of degree <= deg."""
        assert self.analysis_exact_for(deg), \
            f"grid ({self.Ns},{self.Nphi}) cannot resolve the coefficients of degree {deg}"
        K = deg // 2 + 1          # Chebyshev degrees 0..deg//2
        N = deg // 3 + 1          # frequencies 0..deg//3
        rig = self.rigorous
        idx = (np.arange(N).reshape(1, -1) * np.arange(self.Nphi).reshape(-1, 1)) % self.Nphi   # [b, n]
        Cc = self.cphi[idx]
        Cs = self.sphi[idx]
        two_over = ar.const(Fraction(2, self.Nphi), rig)
        fa = (f @ Cc) * two_over                       # [i, n]:  a_n(s_i) = (2/Nphi) sum_b f cos(n phi_b)
        fb = (f @ Cs) * two_over
        half = ar.const(0.5, rig)
        if rig:
            scale = Ball(np.where(np.arange(N) == 0, 0.5, 1.0))
        else:
            scale = np.where(np.arange(N) == 0, 0.5, 1.0)
        fa = fa * scale.reshape(1, -1)                 # a_0 = (1/Nphi) sum_b f
        Dk = self.D[:K]
        A = Dk @ fa                                     # [k, n]
        B = Dk @ fb
        return ChebFourier(A, B, self.pi, rig)


def _edge(i, N):
    return Fraction(1, 2) if i in (0, N) else Fraction(1)


class ChebFourier:
    """f(x, phi) = sum_{k, n} T_k(x) ( A[k, n] cos(n phi) + B[k, n] sin(n phi) ),  x = 1 - 2 s."""

    def __init__(self, A, B, pi, rigorous):
        self.A, self.B, self.pi, self.rigorous = A, B, pi, rigorous
        self.K, self.N = A.shape

    def integral_S3(self):
        """int_{S^3} f dvol = 2 pi^2 sum_{k even} A[k, 0] / (1 - k^2)."""
        ik = [Fraction(1, 1 - k * k) if k % 2 == 0 else Fraction(0) for k in range(self.K)]
        ikv = Ball.from_fractions(ik) if self.rigorous else np.array([float(q) for q in ik])
        return (self.A[:, 0] * ikv).sum() * (2 * self.pi * self.pi)

    def amplitudes(self):
        return ar.sqrt(ar.sq(self.A) + ar.sq(self.B), nonneg=True)

    def lipschitz(self):
        """Bounds L_x >= sup |df/dx| and L_phi >= sup |df/dphi| from the coefficients."""
        amp = self.amplitudes()
        k2 = (np.arange(self.K) ** 2).astype(float).reshape(-1, 1)
        nn = np.arange(self.N).astype(float).reshape(1, -1)
        Lx = (amp * k2).sum()
        Lp = (amp * nn).sum()
        return Lx, Lp

    def evaluate(self, N1, N2):
        """Values on the fine grid x_i = cos(pi i / N1) (i = 0..N1), phi_b = 2 pi b / N2."""
        rig = self.rigorous
        c2, _ = cos_sin_table(2 * N1, rig)
        T = c2[(np.arange(N1 + 1).reshape(-1, 1) * np.arange(self.K).reshape(1, -1)) % (2 * N1)]   # [i, k]
        cp, sp = cos_sin_table(N2, rig)
        idx = (np.arange(self.N).reshape(-1, 1) * np.arange(N2).reshape(1, -1)) % N2             # [n, b]
        return (T @ self.A) @ cp[idx] + (T @ self.B) @ sp[idx]

    def bounds(self, N1=2048, N2=512):
        """Rigorous (lower bound of min f, upper bound of max f) over S^3 (floats: plain min/max)."""
        vals = self.evaluate(N1, N2)
        if not self.rigorous:
            return float(vals.min()), float(vals.max())
        Lx, Lp = self.lipschitz()
        # every x in [-1, 1] is within pi/(2 N1) of a node, every phi within pi/N2 of a node
        rem = (Lx * self.pi * ar.const(Fraction(1, 2 * N1), True) + Lp * self.pi * ar.const(Fraction(1, N2), True)).upper()
        return float(vals.lower().min() - rem), float(vals.upper().max() + rem)

    def coefficient_sum_bound(self):
        """sup |f| <= sum of amplitudes (a cruder but grid-free bound)."""
        return ar.upper(self.amplitudes().sum())
