"""Hopf invariant q = 1: the construction reproduces the Hopf map.

H(z, w) = (w^2 - zbar^2, 2 Re(z w)) is G_1-equivariant, has |H| = 1 on S^3, and is the Hopf map
(a, b) -> (2 a bbar, |a|^2 - |b|^2) composed with the isometry (z, w) -> (a, bbar),
a = (w + zbar)/sqrt 2, b = (w - zbar)/sqrt 2, of R^4.  In the basis of Proposition 2.4 (q = 1) it is
pi sqrt(2/3) (U_{0,2,0} - U_{-2,0,0} + V_{1,1,0}) (Y_{0,2,0} = sqrt(3/(2 pi^2)) w^2,
Y_{-2,0,0} = sqrt(3/(2 pi^2)) zbar^2, sqrt 2 Re Y_{1,1,0} = (sqrt 6/pi) Re(z w)).  Its energy density
|dH|^2/2 is 4 everywhere, so E(H) = 8 pi^2.
"""
import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator, vector_modes
from eells.fields import field_from_coefficients, save_coefficients
from eells.equation import OperatorCoefficients, dot
from numerics.solve_approximate import solve
from certificate.certify import certify

C = np.pi * np.sqrt(2 / 3)
HOPF = {("U", 2, 0): C, ("U", 0, 0): -C, ("V", 1, 0): C}


def _H(z, w):
    zeta = w ** 2 - np.conj(z) ** 2
    return np.stack([zeta.real, zeta.imag, 2 * (z * w).real])


def hopf_coefficients(D):
    modes = vector_modes(D, 1)
    return modes, np.array([HOPF.get((m.kind, m.n, m.j), 0.0) for m in modes])


def test_H_is_an_equivariant_isometric_copy_of_the_hopf_map():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 200)); x /= np.linalg.norm(x, axis=0)
    z, w = x[0] + 1j * x[1], x[2] + 1j * x[3]
    H = _H(z, w)
    assert np.abs(np.linalg.norm(H, axis=0) - 1).max() < 1e-14
    a, b = (w + np.conj(z)) / np.sqrt(2), (w - np.conj(z)) / np.sqrt(2)
    hopf = lambda a, b: np.stack([(2 * a * np.conj(b)).real, (2 * a * np.conj(b)).imag, abs(a) ** 2 - abs(b) ** 2])
    assert np.abs(H - hopf(a, np.conj(b))).max() < 1e-14
    # (z, w) -> (a, bbar) is a real-linear isometry of R^4
    cols = []
    for zz, ww in ((1, 0), (1j, 0), (0, 1), (0, 1j)):
        aa, bb = (ww + np.conj(zz)) / np.sqrt(2), np.conj((ww - np.conj(zz)) / np.sqrt(2))
        cols.append([aa.real, aa.imag, bb.real, bb.imag])
    A = np.array(cols).T
    assert np.abs(A.T @ A - np.eye(4)).max() < 1e-15
    # equivariance under the generators of G_1
    th = 0.7
    zeta = (H[0] + 1j * H[1]) * np.exp(-2j * th)
    assert np.abs(_H(np.exp(1j * th) * z, np.exp(-1j * th) * w) - np.stack([zeta.real, zeta.imag, H[2]])).max() < 1e-14
    assert np.abs(_H(np.conj(z), np.conj(w)) - H * np.array([1, -1, 1]).reshape(3, 1)).max() < 1e-14
    assert np.abs(_H(z, -w) - H * np.array([1, 1, -1]).reshape(3, 1)).max() < 1e-14


def test_H_in_the_basis_and_its_energy_density():
    modes, vals = hopf_coefficients(6)
    g = OrbitGrid.for_quadrature(14, 1)
    W = field_from_coefficients(modes, vals, HarmonicEvaluator(g, max_degree=6))
    z = g.rs.reshape(-1, 1) + 0j
    w = g.rc.reshape(-1, 1) * np.exp(2j * np.pi * np.arange(g.Nphi) / g.Nphi)
    assert np.abs(np.stack(W.val) - _H(z, w)).max() < 1e-14
    rho = dot(W.val, W.val)
    dens = sum(rho * dot(W.grad[a], W.grad[a]) - dot(W.val, W.grad[a]) ** 2 for a in range(4)) / rho ** 2
    assert np.abs(dens - 8).max() < 1e-12                           # |dH|^2 = 8
    F0 = OperatorCoefficients(W).residual()
    assert max(np.abs(f).max() for f in F0) < 1e-12                 # H solves F(0) = 0


def test_solver_converges_to_the_hopf_map():
    for D in (2, 8):
        _, a, history, diag = solve(D, 1, log=lambda *args: None)
        _, exact = hopf_coefficients(D)
        assert np.abs(a - exact).max() < 1e-14
        assert abs(diag["energy_over_8pi2"] - 1) < 1e-13 and len(history) <= 8


def test_certificate_for_the_hopf_map(tmp_path):
    modes, vals = hopf_coefficients(2)
    path = tmp_path / "Wbar_q1_D2.json"
    save_coefficients(path, modes, vals)
    out = certify(path, 10, out_dir=str(tmp_path), log=lambda *args: None)
    assert out["passed"]
    en, eu = out["details"]["energy"], out["details"]["energy_of_u"]
    assert en["E_over_8pi2_lower"] <= 1 <= en["E_over_8pi2_upper"]
    assert eu["C_energy_lower2"] <= 8 * np.pi ** 2 <= eu["C_energy_upper2"]
