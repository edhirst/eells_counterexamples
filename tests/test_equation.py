"""The map F, its linearisation, the expanded coefficients P, M_a, M_0 and the adjoint L^*
(Sections 2.2 and 3.3), for several Hopf invariants q."""
import numpy as np
import pytest

from eells.grid import OrbitGrid
from eells.harmonics import vector_modes, HarmonicEvaluator
from eells.fields import field_from_coefficients
from eells.equation import OperatorCoefficients, F_direct, L_direct, degrees, dot


def _setup(seed=1, D=10, q=2):
    rng = np.random.default_rng(seed)
    modes = vector_modes(D, q)
    g = OrbitGrid(30, 24, False, q)
    H = HarmonicEvaluator(g, max_degree=D)
    ls = np.array([m.l for m in modes])
    aW = rng.standard_normal(len(modes)) * np.exp(-0.3 * ls); aW[0] = 1.0
    ah = rng.standard_normal(len(modes)) * np.exp(-0.2 * ls)
    W = field_from_coefficients(modes, aW, H)
    h = field_from_coefficients(modes, ah, H)
    return g, H, modes, W, h


@pytest.mark.parametrize("q", [2, 3, 5])
def test_expanded_linearisation_equals_direct_form(q):
    g, H, modes, W, h = _setup(q=q)
    op = OperatorCoefficients(W)
    Lc, Ld = op.apply(h), L_direct(W, h)
    scale = max(np.abs(x).max() for x in Ld)
    assert max(np.abs(Lc[c] - Ld[c]).max() for c in range(3)) < 1e-13 * scale


def test_linearisation_is_derivative_of_F():
    g, H, modes, W, h = _setup(seed=3)
    op = OperatorCoefficients(W)
    L = op.apply(h)
    errs = []
    for t in (1e-3, 1e-4):
        Fp, Fm = F_direct(W, h.scale(t)), F_direct(W, h.scale(-t))
        fd = [(Fp[c] - Fm[c]) / (2 * t) for c in range(3)]
        errs.append(max(np.abs(fd[c] - L[c]).max() for c in range(3)))
    assert errs[1] < errs[0] / 50 and errs[1] < 1e-7      # O(t^2) central-difference error


def test_residual_is_tangential_and_matches_definition():
    g, H, modes, W, h = _setup(seed=5)
    op = OperatorCoefficients(W)
    F0, R = F_direct(W), op.residual()
    assert max(np.abs(F0[c] - R[c]).max() for c in range(3)) < 1e-14
    assert np.abs(dot(F0, W.val)).max() < 1e-14


def test_principal_symbol_and_gauge_term():
    g, H, modes, W, h = _setup(seed=7)
    op = OperatorCoefficients(W)
    rho = dot(W.val, W.val)
    assert np.allclose(op.P, rho ** 2)
    v, gr, lp = H.evaluate(modes[:3])
    gauge = op.gauge_batch(v, lp)
    inner = (v * np.stack(W.val)).sum(axis=1) - (lp * np.stack(W.val)).sum(axis=1)
    assert np.allclose(gauge, (rho * inner)[:, None] * np.stack(W.val)[None])


@pytest.mark.parametrize("q", [2, 3, 6])
def test_adjoint(q):
    """<Psi_a, L Psi_b> = <L^* Psi_a, Psi_b> by exact quadrature (degree 2l + 4D + 2)."""
    D, l = 10, 12
    modes = vector_modes(D, q)
    rng = np.random.default_rng(q)
    aW = rng.normal(size=len(modes)) / (1 + np.array([m.l for m in modes]))
    g = OrbitGrid.for_quadrature(2 * (l + degrees(D, q)["Kp"]), q)
    H = HarmonicEvaluator(g, max_degree=l)
    op = OperatorCoefficients(field_from_coefficients(modes, aW, H))
    low = vector_modes(l, q)
    v, gr, lp = H.evaluate(low)
    A = np.einsum("kcib,lcib,ib->kl", v, op.apply_batch(v, gr, lp), g.omega)
    As = np.einsum("kcib,lcib,ib->kl", op.apply_adjoint_batch(v, gr, lp), v, g.omega)
    assert np.abs(A - As).max() < 1e-13 * np.abs(A).max()
