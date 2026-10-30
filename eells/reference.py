"""The reference field of Proposition 2.7, the extended Hopf map (Proposition 2.6) and the
Sobolev constants of Propositions 2.3 and 2.4."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import arith as ar
from .ball import Ball, pi_ball


def reference_field(grid):
    """V_0 = ( w^2 - zbar^4,  2 Re(z^2 w) )  as a real 3-vector on the grid (degree 4).

    At (z, w) = (sqrt s, sqrt(1-s) e^{i phi}):  w^2 - zbar^4 = (1-s) e^{2 i phi} - s^2,
    2 Re(z^2 w) = 2 s sqrt(1-s) cos phi.   Its norm is 1 - |z|^2 |w|^2 (eq. homotopynorm)."""
    s = grid.s.reshape(-1, 1)
    one_minus_s = (grid.rc * grid.rc).reshape(-1, 1)
    c2, s2 = grid.cos_n_phi(2).reshape(1, -1), grid.sin_n_phi(2).reshape(1, -1)
    c1 = grid.cos_n_phi(1).reshape(1, -1)
    return [one_minus_s * c2 - (s * s) * ar.ones((1, grid.Nphi), grid.rigorous),
            one_minus_s * s2,
            (s * grid.rc.reshape(-1, 1)) * c1 * 2.0]


def extended_hopf_map(z, w):
    """h(a, b) = (2 a conj(b), |a|^2 - |b|^2) / (|a|^2 + |b|^2) evaluated at (a, b) = (z^2, w)
    (numpy complex arrays; used only for numerical checks of Proposition 2.6)."""
    a, b = z ** 2, w
    den = np.abs(a) ** 2 + np.abs(b) ** 2
    zeta = 2 * a * np.conj(b) / den
    return np.stack([zeta.real, zeta.imag, (np.abs(a) ** 2 - np.abs(b) ** 2) / den])


def sobolev_constants(rigorous=True):
    """c_inf = sqrt((pi^2/6 - 5/4) / (2 pi^2)),  c_4 = (2 / (9 pi^2))^{1/4}  (Proposition 2.3) and
    c_5 = 2 sqrt(2) / 9 = sqrt(max_{l >= 2} l (l + 2) / (l + 1)^4)  (Proposition 2.4,
    ||du||_{L^2} <= c_5 ||u||_X; the maximum is attained at l = 2, the lowest degree in X).

    Returns (c_inf, c_4, c_5) as floats, or as balls enclosing the exact constants."""
    if not rigorous:
        return (float(np.sqrt((np.pi ** 2 / 6 - 1.25) / (2 * np.pi ** 2))),
                float((2 / (9 * np.pi ** 2)) ** 0.25), float(2 * np.sqrt(2) / 9))
    pi = pi_ball()
    pi2 = pi * pi
    c_inf = ((pi2 * Ball.from_fraction(Fraction(1, 6)) - 1.25) / (pi2 * 2.0)).sqrt()
    c_4 = (Ball.from_fraction(Fraction(2, 9)) / pi2).sqrt().sqrt()
    c_5 = Ball.from_fraction(Fraction(8, 81)).sqrt()
    return c_inf, c_4, c_5
