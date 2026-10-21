"""The map F, its linearisation and the expanded coefficients P, M_a, M_0 (Section 2.2, 3.2)."""
import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import vector_modes, HarmonicEvaluator
from eells.fields import field_from_coefficients
from eells.equation import OperatorCoefficients, F_direct, L_direct, dot


def _setup(seed=1, D=10):
    rng = np.random.default_rng(seed)
    modes = vector_modes(D)
    g = OrbitGrid(30, 24, False)
    H = HarmonicEvaluator(g, max_degree=D)
    ls = np.array([m.l for m in modes])
    aW = rng.standard_normal(len(modes)) * np.exp(-0.3 * ls); aW[0] = 1.0
    ah = rng.standard_normal(len(modes)) * np.exp(-0.2 * ls)
    W = field_from_coefficients(modes, aW, H)
    h = field_from_coefficients(modes, ah, H)
    chi = 1.0 + 0.1 * g.s.reshape(-1, 1) * np.ones((1, g.Nphi))
    return g, H, modes, W, h, chi


def test_expanded_linearisation_equals_direct_form():
    g, H, modes, W, h, chi = _setup()
    op = OperatorCoefficients(W, chi)
    Lc, Ld = op.apply(h), L_direct(W, chi, h)
    scale = max(np.abs(x).max() for x in Ld)
    assert max(np.abs(Lc[c] - Ld[c]).max() for c in range(3)) < 1e-13 * scale


def test_linearisation_is_derivative_of_F():
    g, H, modes, W, h, chi = _setup(seed=3)
    op = OperatorCoefficients(W, chi)
    L = op.apply(h)
    errs = []
    for t in (1e-3, 1e-4):
        Fp, Fm = F_direct(W, chi, h.scale(t)), F_direct(W, chi, h.scale(-t))
        fd = [(Fp[c] - Fm[c]) / (2 * t) for c in range(3)]
        errs.append(max(np.abs(fd[c] - L[c]).max() for c in range(3)))
    assert errs[1] < errs[0] / 50 and errs[1] < 1e-7      # O(t^2) central-difference error


def test_residual_is_tangential_and_matches_definition():
    g, H, modes, W, h, chi = _setup(seed=5)
    op = OperatorCoefficients(W, chi)
    F0, R = F_direct(W, chi), op.residual()
    assert max(np.abs(F0[c] - R[c]).max() for c in range(3)) < 1e-14
    assert np.abs(dot(F0, W.val)).max() < 1e-14


def test_principal_symbol_and_gauge_term():
    g, H, modes, W, h, chi = _setup(seed=7)
    op = OperatorCoefficients(W, chi)
    rho = dot(W.val, W.val)
    assert np.allclose(op.P, chi * rho ** 2)
    # the gauge term is the normal part of L h minus the normal part of the cross-product part
    v, gr, lp = H.evaluate(modes[:3])
    gauge = op.gauge_batch(v, lp)
    inner = (v * np.stack(W.val)).sum(axis=1) - (lp * np.stack(W.val)).sum(axis=1)
    assert np.allclose(gauge, (chi * rho * inner)[:, None] * np.stack(W.val)[None])
