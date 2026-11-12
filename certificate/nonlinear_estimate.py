"""Proposition 3.4 and Theorem 2.22: the nonlinear constant and the Newton-Kantorovich check.

    C_nl = 4 ||Wbar||_inf (c_inf^3 + 2 c_inf^2 c_4^2),
    k(r) = C_nl r (3 ||Wbar||_X^2 + 3 r ||Wbar||_X + r^2),
    conditions:  M k(r) < 1,   M (eps + k(r) r) <= r,   C_homotopy - c_inf r > 0.
"""
from __future__ import annotations

from eells.ball import Ball
from eells.reference import sobolev_constants


def nonlinear_constant(W_sup):
    """C_nl (upper bound) from W_sup >= ||Wbar||_{C^0} and the exact c_inf, c_4."""
    c_inf, c_4, _, _ = sobolev_constants(True)
    C = 4.0 * Ball.exact(W_sup) * (c_inf * c_inf * c_inf + 2.0 * c_inf * c_inf * c_4 * c_4)
    return float(C.upper())


def norm_X(modes, values):
    """||Wbar||_X = ( sum_k (l_k + 1)^4 a_k^2 )^{1/2}  (upper bound)."""
    s = Ball.exact(0.0)
    for m, v in zip(modes, values):
        s = s + Ball.exact(float(v)) * Ball.exact(float(v)) * float((m.l + 1) ** 4)
    return float(s.sqrt().upper())


def k_of_r(Cnl, nX, r):
    C, n, rr = Ball.exact(Cnl), Ball.exact(nX), Ball.exact(r)
    return C * rr * (3.0 * n * n + 3.0 * rr * n + rr * rr)


def newton_kantorovich(M, eps, Cnl, nX, C_hom, r=None):
    """Choose r (default: 2 M eps rounded up to two significant decimal digits) and verify the
    conditions for that exact decimal number (the value printed in Table A.1)."""
    from fractions import Fraction
    from .rounding import round_up
    c_inf = sobolev_constants(True)[0]
    r_str = round_up(2.0 * M * eps, 2) if r is None else str(r)
    rb = Ball.from_fraction(Fraction(r_str))
    Mb, eb = Ball.exact(M), Ball.exact(eps)
    C, n = Ball.exact(Cnl), Ball.exact(nX)
    k = C * rb * (3.0 * n * n + 3.0 * rb * n + rb * rb)
    Mk = Mb * k
    lhs = Mb * (eb + k * rb)
    hom = Ball.exact(C_hom) - c_inf * rb
    ok = {"Mk_lt_1": float(Mk.upper()) < 1, "M_eps_plus_kr_le_r": float((lhs - rb).upper()) <= 0,
          "homotopy_margin_positive": float(hom.lower()) > 0}
    return {"r": r_str, "k": float(k.upper()), "Mk": float(Mk.upper()), "M_eps_plus_kr": float(lhs.upper()),
            "homotopy_margin": float(hom.lower()), "conditions": ok, "all_ok": all(ok.values())}
