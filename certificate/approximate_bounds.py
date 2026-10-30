"""Proposition 3.1: rigorous bounds for the approximate solution Wbar.

* residual:   ||F(0)||_{L^2} <= C_residual     (F(0) = -(chi/2) Pi Q is a polynomial; its
              squared norm is integrated exactly on the quadrature grid)
* energy:     C_lower <= E(Wbar/|Wbar|) <= C_upper, with
                  E = (1/2) int N / |Wbar|^4,   N = sum_a |Wbar x d_a Wbar|^2 = sum_a (|Wbar|^2 |d_a Wbar|^2 - (Wbar . d_a Wbar)^2),
              written with delta = |Wbar|^4 - 1 as  1/(1+delta) = 1 - delta + delta^2/(1+delta):
                  E = (1/2) [ int N - int N delta ] + err,   |err| <= (1/2) (sup delta^2 / (1 - sup|delta|)) int N,
              the two integrals being exact (polynomial integrands) and sup|delta| a rigorous bound.
* homotopy:   min Wbar . V_0 >= C_homotopy  (Chebyshev-Fourier bound of the degree D + 4 polynomial).
* min |Wbar|: a lower bound (Chebyshev-Fourier bound of |Wbar|^2), used for C_energyparameter in
              Proposition 4.2 (energy_estimate.py).
"""
from __future__ import annotations

import numpy as np

from eells import arith as ar
from eells.ball import Ball
from eells.equation import OperatorCoefficients, dot
from eells.reference import reference_field


def residual_bound(W, chi, grid, deg):
    """Upper bound of ||F(0)||_{L^2}; returns (bound, ball of the squared norm)."""
    op = OperatorCoefficients(W, chi)
    F0 = op.residual()
    sq = grid.integrate(dot(F0, F0), 2 * deg["F0"])
    return float(sq.sqrt(nonneg=True).upper()), sq


def energy_bounds(W_quad, grid_quad, W_bound, grid_bound, deg, D):
    """Rigorous enclosure [E_lower, E_upper] of E(Wbar/|Wbar|) and of E / (8 pi^2)."""
    Wv, dW = W_quad.val, W_quad.grad
    rho = dot(Wv, Wv)
    N = ar.sq(dW[0][0]) * 0.0
    for a in range(4):
        N = N + rho * dot(dW[a], dW[a]) - ar.sq(dot(Wv, dW[a]))
    delta = rho * rho - 1.0
    I1 = grid_quad.integrate(N, deg["energy"])
    I2 = grid_quad.integrate(N * delta, deg["energy"] + 4 * D)
    # sup |delta| on the bounding grid
    rho_b = dot(W_bound.val, W_bound.val)
    delta_b = rho_b * rho_b - 1.0
    lo, hi = grid_bound.analyse(delta_b, 4 * D).bounds()
    dmax = max(abs(lo), abs(hi))
    assert dmax < 1
    dmax_b = Ball.exact(dmax)
    err = (dmax_b * dmax_b / (1.0 - dmax_b)) * I1 * 0.5
    E = (I1 - I2) * 0.5
    E_lower, E_upper = float((E - err).lower()), float((E + err).upper())
    pi2 = grid_quad.pi * grid_quad.pi * 8.0
    e8 = (E - err) / pi2, (E + err) / pi2
    return {"E_lower": E_lower, "E_upper": E_upper,
            "E_over_8pi2_lower": float(e8[0].lower()), "E_over_8pi2_upper": float(e8[1].upper()),
            "sup_abs_norm4_minus_1": dmax, "int_N": I1, "int_N_delta": I2}


def homotopy_bound(W_bound, grid_bound, D):
    """Lower bound C_homotopy of  Wbar . (w^2 - zbar^4, 2 Re(z^2 w))  over S^3 (and an upper bound)."""
    V0 = reference_field(grid_bound)
    f = dot(W_bound.val, V0)
    lo, hi = grid_bound.analyse(f, D + 4).bounds(N1=8192, N2=1024)
    return lo, hi


def min_norm_bound(W_bound, grid_bound, D):
    """Lower bound of min_{S^3} |Wbar| (and an upper bound of max |Wbar|), from the Chebyshev-Fourier
    bounds of the degree 2D polynomial |Wbar|^2."""
    rho = dot(W_bound.val, W_bound.val)
    lo, hi = grid_bound.analyse(rho, 2 * D).bounds()
    lo_b, hi_b = Ball.exact(max(lo, 0.0)).sqrt(), Ball.exact(max(hi, 0.0)).sqrt()
    return float(lo_b.lower()), float(hi_b.upper())
