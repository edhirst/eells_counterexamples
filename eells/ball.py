"""Rigorous midpoint-radius ("ball") arithmetic on numpy float64 arrays.

A ``Ball(m, r)`` represents, elementwise, the set of real numbers x with |x - m| <= r.  Every
operation returns a ball containing all results obtainable from members of its operands, so a
chain of ball operations started from exact inputs encloses the exact mathematical value.

Rounding is accounted for with the standard a-priori error model of IEEE-754 binary64
round-to-nearest arithmetic (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
sections 2.2 and 3.1):

* a single operation  fl(a o b)  (o in +, -, *, /) and a single sqrt satisfy
      |fl(x) - x| <= u |fl(x)| + eta,          u = 2^-53,  eta = 2^-1074  (underflow);
* a sum or dot product of k terms, in any order and with or without fused multiply-add,
  satisfies  |fl(sum) - sum| <= gamma_k * sum |terms| + k eta,   gamma_k = k u / (1 - k u).

The radii themselves are computed in floating point from non-negative quantities.  Each
computed radius is inflated by the factor (1 + 2^-40) (resp. (1 + 4 gamma_{k+4}) after a
k-term reduction) and 1e-300 is added; this dominates the relative error of evaluating the
radius with fewer than ~1000 operations and every underflow term.  No directed rounding is
needed, so the module runs on any IEEE-754 platform.

Only +, -, *, /, sqrt, integer powers, sums and matrix products are provided; this is all the
certificate uses.  Transcendental constants (pi, cos, sin of rational multiples of pi) enter
through ``from_mp_interval`` from mpmath interval arithmetic.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

U = 2.0 ** -53          # unit roundoff of binary64
ETA = 1e-300            # absorbs all underflow terms (each <= 2^-1074 ~ 4.9e-324)
INFL = 1.0 + 2.0 ** -40


def gamma(k):
    """Higham's gamma_k for a k-term floating-point reduction."""
    k = float(k)
    return k * U / (1.0 - k * U)


def _up(r):
    """Upper bound for a non-negative quantity computed with at most ~1000 rounded operations."""
    return r * INFL + ETA


def _up_add(r):
    """Upper bound of a radius after an addition/subtraction (no underflow term is needed: the
    rounded sum of two doubles is exact whenever it is subnormal)."""
    return r * INFL


def _up_k(r, k):
    """Upper bound for a non-negative k-term sum/dot product computed in floating point."""
    return r * (1.0 + 4.0 * gamma(k + 4)) + (k + 4) * ETA


def val_up(x, k=3):
    """Upper bound of the exact value y of a quantity whose floating-point evaluation x was
    obtained from exact doubles with at most k (<= 1000) rounded operations and no cancellation
    (x = y * prod(1 + delta_i), |delta_i| <= u):  |y - x| <= ((1-u)^-k - 1)|x| < (k + 1/2) u |x|.
    The returned float is fl(x + fl(fl((k+2) u |x|) + ETA)); its three roundings cost less than
    (3/2 + 3(k+2)u) u |x| in total, so the result is at least x + (k + 1/2) u |x| >= y when x is
    normal, while ETA covers the subnormal range."""
    return x + ((k + 2) * U * np.abs(x) + ETA)


def val_down(x, k=3):
    """Lower bound, see :func:`val_up`."""
    return x - ((k + 2) * U * np.abs(x) + ETA)


class Ball:
    """Elementwise set {x : |x - m| <= r} on numpy arrays (broadcasting as numpy does)."""

    __array_priority__ = 1000   # make  ndarray * Ball  dispatch to Ball.__rmul__

    def __init__(self, m, r=None):
        if isinstance(m, Ball):
            self.m, self.r = m.m, m.r
            return
        self.m = np.array(m, dtype=float)
        if r is None:
            self.r = np.zeros_like(self.m)
        else:
            self.r = np.array(np.broadcast_to(np.asarray(r, dtype=float), self.m.shape))
        if not (np.all(np.isfinite(self.m)) and np.all(np.isfinite(self.r)) and np.all(self.r >= 0)):
            raise FloatingPointError("non-finite midpoint or radius in a ball")

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def exact(x):
        """Ball with zero radius around a float (or array of floats)."""
        return Ball(np.asarray(x, dtype=float))

    @staticmethod
    def from_fraction(q):
        """Ball enclosing an exact rational number (the radius is computed exactly, then rounded up)."""
        q = Fraction(q)
        mid = float(q)                                   # correctly rounded quotient
        rad = float(abs(q - Fraction(mid)))               # exact rational error, rounded to nearest
        return Ball(np.array(mid), np.array(_up(rad)))

    @staticmethod
    def from_fractions(qs):
        bs = [Ball.from_fraction(q) for q in qs]
        return Ball(np.array([b.m for b in bs]), np.array([b.r for b in bs]))

    @staticmethod
    def from_interval(lo, hi):
        """Ball enclosing the interval [lo, hi] given as exact rationals."""
        lo, hi = Fraction(lo), Fraction(hi)
        assert lo <= hi
        mid = float((lo + hi) / 2)
        rad = float(max(hi - Fraction(mid), Fraction(mid) - lo))
        return Ball(np.array(mid), np.array(_up(rad)))

    @staticmethod
    def from_mp_interval(x):
        """Ball enclosing an mpmath interval (``mpmath.iv.mpf``) or a list of them.

        The endpoints are read from their exact binary representation, so the enclosure does
        not depend on the mpmath working precision in force when this is called."""
        if isinstance(x, (list, tuple)):
            bs = [Ball.from_mp_interval(v) for v in x]
            return Ball(np.array([b.m for b in bs]), np.array([b.r for b in bs]))
        lo_t, hi_t = x._mpi_
        return Ball.from_interval(_mpf_tuple_to_fraction(lo_t), _mpf_tuple_to_fraction(hi_t))

    # ------------------------------------------------------------------ container behaviour
    @property
    def shape(self):
        return self.m.shape

    @property
    def ndim(self):
        return self.m.ndim

    @property
    def size(self):
        return self.m.size

    def __len__(self):
        return len(self.m)

    def __getitem__(self, k):
        return Ball(self.m[k], self.r[k])

    def reshape(self, *shape):
        return Ball(self.m.reshape(*shape), self.r.reshape(*shape))

    def flatten(self):
        return Ball(self.m.ravel(), self.r.ravel())

    @property
    def T(self):
        return Ball(self.m.T, self.r.T)

    def transpose(self, *axes):
        return Ball(self.m.transpose(*axes), self.r.transpose(*axes))

    def __repr__(self):
        return f"Ball(m={self.m!r}, r={self.r!r})"

    # ------------------------------------------------------------------ rigorous bounds
    def upper(self):
        """Elementwise upper bound of the set (a float >= m + r)."""
        return val_up(self.m + self.r, 1)

    def lower(self):
        """Elementwise lower bound of the set (a float <= m - r)."""
        return val_down(self.m - self.r, 1)

    def absmax(self):
        """Elementwise upper bound of |x|."""
        return val_up(np.abs(self.m) + self.r, 1)

    def absmin(self):
        """Elementwise lower bound of |x| (0 where the ball contains 0)."""
        return np.maximum(val_down(np.abs(self.m) - self.r, 1), 0.0)

    def abs(self):
        """Ball enclosing |x|."""
        return Ball.from_bounds(self.absmin(), self.absmax())

    @staticmethod
    def from_bounds(lo, hi):
        """Ball enclosing the float intervals [lo, hi] (elementwise, lo <= hi)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        assert np.all(lo <= hi)
        m = 0.5 * lo + 0.5 * hi
        return Ball(m, _up(np.maximum(hi - m, m - lo)))

    def contains_zero(self):
        return np.abs(self.m) <= self.r

    # ------------------------------------------------------------------ arithmetic
    def __neg__(self):
        return Ball(-self.m, self.r)

    def __pos__(self):
        return self

    def __add__(self, o):
        o = asball(o)
        m = self.m + o.m
        return Ball(m, _up_add(self.r + o.r + U * np.abs(m)))

    __radd__ = __add__

    def __sub__(self, o):
        o = asball(o)
        m = self.m - o.m
        return Ball(m, _up_add(self.r + o.r + U * np.abs(m)))

    def __rsub__(self, o):
        return asball(o) - self

    def __mul__(self, o):
        o = asball(o)
        m = self.m * o.m
        r = np.abs(self.m) * o.r + self.r * np.abs(o.m) + self.r * o.r + U * np.abs(m)
        r = _up(r)
        # an exact zero factor gives an exact zero product (no underflow term)
        exact_zero = ((self.m == 0) & (self.r == 0)) | ((o.m == 0) & (o.r == 0))
        if np.any(exact_zero):
            r = np.where(exact_zero, 0.0, r)
        return Ball(m, r)

    __rmul__ = __mul__

    def inv(self):
        """Ball enclosing 1/x; requires that no element contains 0."""
        am = np.abs(self.m)
        lo = val_down((am - self.r) * am, 2)              # lower bound of (|m| - r)|m|
        if np.any(am - self.r <= 0) or np.any(lo <= 0):
            raise ZeroDivisionError("ball contains 0")
        m = 1.0 / self.m
        # |1/x - 1/m| = |x - m| / (|x| |m|) <= r / ((|m| - r)|m|)
        return Ball(m, _up(self.r / lo + U * np.abs(m)))

    def __truediv__(self, o):
        return self * asball(o).inv()

    def __rtruediv__(self, o):
        return asball(o) * self.inv()

    def sqrt(self, nonneg=False):
        """Ball enclosing sqrt(x).  With ``nonneg=True`` the caller asserts that the enclosed exact
        quantity is non-negative (a sum of squares, |z|^2, ...) and the ball is first intersected
        with [0, inf); otherwise a ball with negative part raises.  Uses the monotonicity of sqrt
        and the correct rounding of numpy's sqrt on the two endpoints."""
        lo, hi = self.lower(), self.upper()
        if np.any(lo < 0):
            if not nonneg:
                raise ValueError("sqrt of a ball with negative part")
            lo = np.maximum(lo, 0.0)
        exact_zero = (self.m == 0) & (self.r == 0)
        out = Ball.from_bounds(val_down(np.sqrt(lo), 1), val_up(np.sqrt(hi), 1))
        if np.any(exact_zero):
            out = Ball(np.where(exact_zero, 0.0, out.m), np.where(exact_zero, 0.0, out.r))
        return out

    def sq(self):
        """Ball enclosing x^2 (never negative, unlike x * x when the ball straddles 0)."""
        lo, hi = self.absmin(), self.absmax()
        return Ball.from_bounds(np.maximum(val_down(lo * lo, 1), 0.0), val_up(hi * hi, 1))

    def __pow__(self, k):
        assert isinstance(k, (int, np.integer)) and k >= 0
        out = Ball(np.ones_like(self.m))
        base = self
        while k:
            if k & 1:
                out = out * base
            base = base * base
            k >>= 1
        return out

    # ------------------------------------------------------------------ reductions
    def sum(self, axis=None):
        """Rigorous sum along an axis (or of all elements)."""
        if axis is None:
            k = self.m.size
        else:
            k = self.m.shape[axis]
        m = self.m.sum(axis=axis)
        r = self.r.sum(axis=axis) + gamma(k) * np.abs(self.m).sum(axis=axis)
        return Ball(m, _up_k(r, k))

    def __matmul__(self, o):
        return matmul(self, o)

    def __rmatmul__(self, o):
        return matmul(asball(o), self)


def asball(x):
    return x if isinstance(x, Ball) else Ball(np.asarray(x, dtype=float))


def matmul(A, B):
    """Rigorous product of ball matrices (or stacks of matrices), using BLAS for the midpoints.

    The exact set product is contained in  Am Bm +- (|Am| Br + Ar |Bm| + Ar Br); the floating-
    point evaluation of Am Bm with k inner terms adds at most gamma_k |Am| |Bm| + k eta."""
    A, B = asball(A), asball(B)
    k = A.m.shape[-1]
    m = A.m @ B.m
    aA, aB = np.abs(A.m), np.abs(B.m)
    rad = aA @ (B.r + gamma(k) * aB)
    if np.any(A.r):
        rad = rad + A.r @ (aB + B.r)
    return Ball(m, _up_k(rad, k))


def stack(balls, axis=0):
    balls = [asball(b) for b in balls]
    return Ball(np.stack([b.m for b in balls], axis), np.stack([b.r for b in balls], axis))


def concatenate(balls, axis=0):
    balls = [asball(b) for b in balls]
    return Ball(np.concatenate([b.m for b in balls], axis), np.concatenate([b.r for b in balls], axis))


def zeros(shape):
    return Ball(np.zeros(shape))


def ones(shape):
    return Ball(np.ones(shape))


# ---------------------------------------------------------------------- matrix norms (upper bounds)
def norm_1(M):
    """Upper bound of the matrix 1-norm (maximum absolute column sum)."""
    a = asball(M).absmax()
    return float(np.max(_up_k(a.sum(axis=0), a.shape[0])))


def norm_inf(M):
    """Upper bound of the matrix infinity-norm (maximum absolute row sum)."""
    a = asball(M).absmax()
    return float(np.max(_up_k(a.sum(axis=1), a.shape[1])))


def norm_frobenius(M):
    """Upper bound of the Frobenius norm."""
    a = asball(M).absmax().ravel()
    s = _up_k(float(np.dot(a, a)), a.size)
    return float(_up(np.sqrt(s)))


def _norm_2_direct(M):
    return min(float(val_up(np.sqrt(val_up(norm_1(M) * norm_inf(M), 1)), 1)), norm_frobenius(M))


def norm_2(M, squarings=2):
    """Upper bound of the spectral norm.

    ||M||_2 <= min( sqrt(||M||_1 ||M||_inf), ||M||_F )  directly, and, sharper, through the Gram
    matrix G = M M^T (or M^T M, whichever is smaller):  ||M||_2^2 = rho(G) <= ||G^{2^q}||^{1/2^q}
    for any induced norm, so  ||M||_2 <= ||G^{2^q}||^{1/2^{q+1}}  with q = `squarings`.  G is
    rescaled by an exact power of two before squaring to avoid overflow."""
    import math
    M = asball(M)
    best = _norm_2_direct(M)
    if M.m.ndim != 2 or squarings <= 0:
        return best
    G = matmul(M, M.T) if M.m.shape[0] <= M.m.shape[1] else matmul(M.T, M)
    exponent = 0                                     # ||G^{2^q}||_2 <= direct(G_q) * 2^exponent
    for i in range(squarings):
        n = norm_inf(G)
        if n == 0:
            return 0.0 if best == 0 else min(best, 0.0)
        e = math.frexp(n)[1]                          # 2^(e-1) <= n < 2^e, exact integer
        if abs(e) > 900:
            return best                               # scaling would leave the double range
        G = Ball(np.ldexp(G.m, -e), np.ldexp(G.r, -e))   # exact rescaling by 2^-e
        exponent = (exponent + e) * 2                 # G <- (G 2^-e)^2
        G = matmul(G, G)
    # ||M||_2 <= (direct(G_q) 2^exponent)^(1/2^(q+1)) = direct(G_q)^(1/2^(q+1)) 2^(k) 2^(r/2^(q+1)),
    # exponent = 2^(q+1) k + r,  0 <= r < 2^(q+1)
    p = 2 ** (squarings + 1)
    k, r = divmod(exponent, p)
    if abs(k) > 900:
        return best
    val = _norm_2_direct(G)
    frac = float(2 ** r)                              # exact
    for _ in range(squarings + 1):
        val = float(val_up(np.sqrt(val), 1))
        frac = float(val_up(np.sqrt(frac), 1))
    val = float(val_up(val * frac, 1)) * (2.0 ** k)   # the last factor is an exact power of two
    return min(best, val)


def norm_2_vector(v):
    """Upper bound of the Euclidean norm of a ball vector."""
    a = asball(v).absmax().ravel()
    return float(_up(np.sqrt(_up_k(float(np.dot(a, a)), a.size))))


# ---------------------------------------------------------------------- mpmath helpers
def _mpf_tuple_to_fraction(t):
    """Exact rational value of an mpmath raw mpf tuple (sign, mantissa, exponent, bitcount)."""
    sign, man, exp, _ = t
    man = int(man)
    if exp >= 0:
        q = Fraction(man * 2 ** exp)
    else:
        q = Fraction(man, 2 ** (-exp))
    return -q if sign else q


def mp_interval_to_fractions(x):
    lo_t, hi_t = x._mpi_
    return _mpf_tuple_to_fraction(lo_t), _mpf_tuple_to_fraction(hi_t)


def pi_ball():
    """Ball enclosing pi."""
    import mpmath as mp
    with mp.workprec(200):
        iv = mp.iv
        old = iv.prec
        iv.prec = 200
        try:
            return Ball.from_mp_interval(iv.pi)
        finally:
            iv.prec = old


def cos_sin_table(N, rigorous=True):
    """Enclosures (or float values) of cos(2 pi p / N), sin(2 pi p / N) for p = 0, ..., N-1."""
    if not rigorous:
        t = 2 * np.pi * np.arange(N) / N
        return np.cos(t), np.sin(t)
    import mpmath as mp
    iv = mp.iv
    old = iv.prec
    iv.prec = 120
    try:
        c = [iv.cos(2 * iv.pi * p / N) for p in range(N)]
        s = [iv.sin(2 * iv.pi * p / N) for p in range(N)]
        C, S = Ball.from_mp_interval(c), Ball.from_mp_interval(s)
    finally:
        iv.prec = old
    # exact values at the multiples of pi/2 (so that, e.g., s = (1 - cos 0)/2 is exactly 0):
    # the quarter turn q (= 0, 1, 2, 3) is the grid point p = q N / 4 when q N is divisible by 4
    for q, (cv, sv) in ((0, (1.0, 0.0)), (1, (0.0, 1.0)), (2, (-1.0, 0.0)), (3, (0.0, -1.0))):
        if (q * N) % 4 == 0:
            p = q * N // 4
            C.m[p], C.r[p], S.m[p], S.r[p] = cv, 0.0, sv, 0.0
    return C, S
