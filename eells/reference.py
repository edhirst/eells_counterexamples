"""The reference field V_q of Section 2.3 (eq. referencemap), the standard Hopf map (used for
numerical checks only) and the Sobolev constants of Propositions 2.5 and 2.6."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import arith as ar
from .ball import Ball, pi_ball


def reference_field(grid, q=2):
    """V_q = ( w^2 - zbar^{2q}/4,  Re(z^q w) )  as a real 3-vector on the grid (degree 2q).

    V_q = pi(z^q / 2, w) for the extended Hopf fibration pi(z, w) = (w^2 - zbar^2, 2 Re(z w)) of
    Definition 2.11, so V_q / |V_q| has Hopf invariant q (Lemma 2.12, Proposition 2.14).
    At (z, w) = (sqrt s, sqrt(1-s) e^{i phi}):  w^2 = (1-s) e^{2 i phi},  zbar^{2q} = s^q,
    Re(z^q w) = s^{q/2} sqrt(1-s) cos phi.   Its norm is |w|^2 + |z|^{2q}/4, which lies in (0, 1]."""
    rig = grid.rigorous
    one_minus_s = (grid.rc * grid.rc).reshape(-1, 1)
    rs_q = (grid.rs ** q).reshape(-1, 1)
    c2, s2 = grid.cos_n_phi(2).reshape(1, -1), grid.sin_n_phi(2).reshape(1, -1)
    c1 = grid.cos_n_phi(1).reshape(1, -1)
    quarter = ar.const(Fraction(1, 4), rig)
    return [one_minus_s * c2 - (rs_q * rs_q) * quarter * ar.ones((1, grid.Nphi), rig),
            one_minus_s * s2,
            (rs_q * grid.rc.reshape(-1, 1)) * c1]


def reference_norm_min(q):
    """min over S^3 of |V_q| = 1 - s + s^q / 4 (s = |z|^2): 1/4 for q <= 4 (at s = 1), less for q >= 5."""
    s = np.linspace(0, 1, 100001)
    return float((1 - s + s ** q / 4).min())


def extended_hopf_map(z, w, q=2):
    """The standard Hopf map h(a, b) = (2 a conj(b), |a|^2 - |b|^2) / (|a|^2 + |b|^2) evaluated at
    (a, b) = (z^q, w) (numpy complex arrays; used only for numerical checks).  It is not the paper's
    extended Hopf fibration pi of Definition 2.11; on S^3, pi = h o A for the isometry
    A(z, w) = ((w + zbar)/sqrt 2, (wbar - z)/sqrt 2) of R^4 (tests/test_hopf_q1.py)."""
    a, b = z ** q, w
    den = np.abs(a) ** 2 + np.abs(b) ** 2
    zeta = 2 * a * np.conj(b) / den
    return np.stack([zeta.real, zeta.imag, (np.abs(a) ** 2 - np.abs(b) ** 2) / den])


# The analytic constants of Propositions 2.5 and 2.6 (as LaTeX, for the tables).
SOBOLEV_CONSTANTS_TEX = {
    "c_inf": r"\sqrt{\frac{\pi^2/6-5/4}{2\pi^2}}",
    "c_4": r"\left(\frac{2}{9\pi^2}\right)^{1/4}",
    "c_5": r"\frac{2\sqrt2}{9}",
    "beta": r"(2\pi^2)^{-1/4}",
}


def sobolev_constants(rigorous=True):
    """The exact constants of Propositions 2.5 and 2.6:

        c_inf = sqrt((pi^2/6 - 5/4) / (2 pi^2))       (||u||_{C^0} <= c_inf ||u||_X),
        c_4   = (2 / (9 pi^2))^{1/4}                  (||grad u||_{L^4} <= c_4 ||u||_X),
        c_5   = 2 sqrt(2) / 9                         (||du||_{L^2} <= c_5 ||u||_X; the maximum of
                                                       l(l+2)/(l+1)^4 over l >= 2 is attained at l = 2),
        beta  = (2 pi^2)^{-1/4}                       (||xi||_{L^4} <= beta ||xi||_{H^1}).

    They do not depend on the Hopf invariant q (the lowest degree in X is l = 2 for every q).
    Returns (c_inf, c_4, c_5, beta) as balls enclosing the exact values, or as floats."""
    if not rigorous:
        return (float(np.sqrt((np.pi ** 2 / 6 - 1.25) / (2 * np.pi ** 2))),
                float((2 / (9 * np.pi ** 2)) ** 0.25), float(2 * np.sqrt(2) / 9),
                float((2 * np.pi ** 2) ** -0.25))
    pi = pi_ball()
    pi2 = pi * pi
    c_inf = ((pi2 * Ball.from_fraction(Fraction(1, 6)) - 1.25) / (pi2 * 2.0)).sqrt()
    c_4 = (Ball.from_fraction(Fraction(2, 9)) / pi2).sqrt().sqrt()
    c_5 = Ball.from_fraction(Fraction(8, 81)).sqrt()
    beta = 1.0 / (pi2 * 2.0).sqrt().sqrt()
    return c_inf, c_4, c_5, beta
