"""Proposition 3.5: the elementary estimates for the radial map phi(X) = X/|X| and the energy constants."""
import numpy as np

from certificate.energy_estimate import energy_of_harmonic_map
from eells.reference import sobolev_constants


def _dphi(X, Y):
    """d phi_X(Y) = (Y - (X.Y) X / |X|^2) / |X|."""
    n = np.linalg.norm(X)
    return (Y - X.dot(Y) * X / n ** 2) / n


def test_radial_map_estimates():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = rng.normal(size=3), rng.normal(size=3)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        # |a/|a| - b/|b|| <= 2 |a - b| / |a|
        assert np.linalg.norm(a / na - b / nb) <= 2 * np.linalg.norm(a - b) / na + 1e-12
        X, Y, Z = rng.normal(size=3) * rng.uniform(0.1, 10), rng.normal(size=3), rng.normal(size=3)
        nX = np.linalg.norm(X)
        # d phi_X is the derivative of phi at X, and |d phi_X| <= 1 / |X|  (eq. radialbound)
        h = 1e-6 * nX
        fd = ((X + h * Y) / np.linalg.norm(X + h * Y) - (X - h * Y) / np.linalg.norm(X - h * Y)) / (2 * h)
        assert np.linalg.norm(fd - _dphi(X, Y)) < 1e-6 * np.linalg.norm(Y) / nX
        assert np.linalg.norm(_dphi(X, Y)) <= np.linalg.norm(Y) / nX + 1e-12
        # |d^2 phi_X(Y, Z)| <= 4 |Y| |Z| / |X|^2  (eq. radialbound; the sharp constant is 2/sqrt(3))
        d2 = (_dphi(X + h * Z, Y) - _dphi(X - h * Z, Y)) / (2 * h)
        assert np.linalg.norm(d2) <= 4 * np.linalg.norm(Y) * np.linalg.norm(Z) / nX ** 2


def test_energy_constants_follow_the_formulas():
    E_lo, E_hi, minW, nX, r_str = 125.2744885848975, 125.2744885856999, 0.9999999992, 69.058, "9.5e-7"
    en = energy_of_harmonic_map(E_lo, E_hi, minW, nX, r_str)
    c_inf, _, c_5, _ = sobolev_constants(False)
    r = float(r_str)
    C8 = minW - c_inf * r
    C6 = E_lo - (c_5 * r) ** 2 / (2 * C8 ** 2) * (1 + 4 * c_inf * nX / C8) ** 2
    C7 = E_hi * (C8 + c_inf * r) ** 2 / (C8 ** 2 - 3 * (c_inf * r) ** 2 + 2 * c_inf * r * C8)
    w = 2 * c_inf * r / (C8 + c_inf * r)
    assert abs(en["c_5"] - 2 * np.sqrt(2) / 9) < 1e-15 and en["c_5"] >= 2 * np.sqrt(2) / 9 - 1e-16
    assert abs(C7 - E_hi / (1 - w ** 2)) < 1e-9                            # = E(ubar) / (1 - ||w||^2)
    assert C8 - 1e-12 <= en["C_energyparameter"] <= C8 + 1e-16            # rigorous lower bounds ...
    assert C6 - 1e-9 <= en["C_energy_lower2"] <= C6 + 1e-12
    assert C7 - 1e-12 <= en["C_energy_upper2"] <= C7 + 1e-9                # ... and upper bounds
    assert w - 1e-16 <= en["w_C0_bound"] <= w + 1e-12
    assert en["all_ok"] and en["w_C0_bound"] < 1
    assert en["C_energy_lower2"] <= E_lo and en["C_energy_upper2"] >= E_hi
    # the upper bound needs min |Wbar| > 2 c_inf r (||w||_{C^0} < 1): otherwise the certificate fails
    bad = energy_of_harmonic_map(E_lo, E_hi, 0.3, nX, "1.5")
    assert not bad["all_ok"] and bad["C_energy_upper2"] is None and bad["w_C0_bound"] >= 1
