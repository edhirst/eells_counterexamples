"""Proposition 4.2: the Dirichlet energy of the harmonic map u = (Wbar + v) / |Wbar + v|.

Let ubar = Wbar / |Wbar| and w = ubar - u.  Since u is harmonic and |u| = |ubar| = 1,
    E(ubar) = E(u) + (1/2) ||dw||_{L^2}^2 - (1/2) int |du|^2 |w|^2,
so that, provided ||w||_{C^0} < 1,
    E(ubar) - (1/2) ||dw||_{L^2}^2  <=  E(u)  <=  E(ubar) / (1 - ||w||_{C^0}^2).
With ||v||_X <= r the constants of the proposition are
    C_energyparameter = min |Wbar| - c_inf r        (<= |Wbar + t v| for t in [0, 1]; it is positive
                                                     because C_homotopy - c_inf r > 0 and |V_0| <= 1),
    ||w||_{C^0}  <= 2 c_inf r / (C_energyparameter + c_inf r)         (|a/|a| - b/|b|| <= 2 |a - b| / |a|),
    ||dw||_{L^2} <= (c_5 r / C_energyparameter) (1 + 4 c_inf ||Wbar||_X / C_energyparameter)
                                       (|d phi_X| <= 1/|X|,  |d^2 phi_X| <= 4/|X|^2  for phi(X) = X/|X|),
    C_energy-lower2 = C_energy-lower - (c_5 r)^2 / (2 C_energyparameter^2) (1 + 4 c_inf ||Wbar||_X / C_energyparameter)^2,
    C_energy-upper2 = C_energy-upper (C_energyparameter + c_inf r)^2
                      / (C_energyparameter^2 - 3 c_inf^2 r^2 + 2 c_inf r C_energyparameter),
the last line being  C_energy-upper / (1 - (2 c_inf r / (C_energyparameter + c_inf r))^2); it requires
C_energyparameter > c_inf r, i.e. the bound on ||w||_{C^0} to be < 1.
"""
from __future__ import annotations

from fractions import Fraction

from eells.ball import Ball
from eells.reference import sobolev_constants


def energy_of_harmonic_map(E_lower, E_upper, min_W, norm_X, r_str):
    """The quantities of Proposition 4.2 for the certified radius r (an exact decimal string).

    E_lower <= E(Wbar/|Wbar|) <= E_upper,  min_W <= min |Wbar|,  norm_X >= ||Wbar||_X.  Every
    formula is monotone in these inputs in the direction in which they are rounded, so evaluating
    it on the rounded bounds (in ball arithmetic, with the balls of c_inf and c_5) gives rigorous
    bounds: C_energyparameter and C_energy-lower2 are lower bounds, the rest upper bounds."""
    c_inf, _, c_5, _ = sobolev_constants(True)
    rb = Ball.from_fraction(Fraction(r_str))
    a = c_inf * rb                                                   # >= ||v||_{C^0}
    C8 = Ball.exact(min_W) - a                                       # C_energyparameter
    assert C8.lower() > 0, "min |Wbar| - c_inf r <= 0"
    w_sup = 2.0 * a / (C8 + a)                                       # >= ||w||_{C^0}
    margin = C8 - a                                                  # > 0  <=>  w_sup < 1
    ok = float(margin.lower()) > 0
    dw = (c_5 * rb / C8) * (1.0 + 4.0 * c_inf * Ball.exact(norm_X) / C8)    # >= ||dw||_{L^2}
    C6 = Ball.exact(E_lower) - dw * dw * 0.5
    C7 = Ball.exact(E_upper) * (C8 + a) * (C8 + a) / (C8 * C8 - 3.0 * a * a + 2.0 * a * C8) if ok else None
    return {"c_5": float(c_5.upper()), "min_abs_Wbar": min_W, "C_energyparameter": float(C8.lower()),
            "w_C0_bound": float(w_sup.upper()), "dw_L2_bound": float(dw.upper()),
            "C_energy_lower2": float(C6.lower()), "C_energy_upper2": float(C7.upper()) if ok else None,
            "conditions": {"w_C0_lt_1": ok}, "all_ok": ok}
