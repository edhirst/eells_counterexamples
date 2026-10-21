"""Proposition 2.2 (orthonormality), the gradient formulas (Green's identity, tangentiality),
the sup bound (l+1)/sqrt(2 pi^2), and the Jacobi-polynomial conventions of the paper."""
import numpy as np
from scipy.special import eval_jacobi

from eells.grid import OrbitGrid
from eells.harmonics import vector_modes, scalar_modes, HarmonicEvaluator, jacobi_table, jacobi_explicit, Mode


def test_mode_counts_and_selection_rules():
    for D in (2, 3, 12, 40):
        ms = vector_modes(D)
        assert all(m.l == abs(m.m) + abs(m.n) + 2 * m.j <= D for m in ms)
        assert all((m.kind == "U" and m.n % 2 == 0 and m.m == 2 * m.n - 4) or
                   (m.kind == "V" and m.n % 2 == 1 and m.n >= 1 and m.m == 2 * m.n) for m in ms)
        assert all(m.l >= 2 for m in ms)
    assert len(vector_modes(40)) == 217
    assert all(m.n % 2 == 0 and m.m == 2 * m.n for m in scalar_modes(10))


def test_jacobi_recurrence_matches_scipy_and_paper_formula():
    x = np.linspace(-1, 1, 9)
    for a, b in ((0, 0), (4, 2), (13, 5), (40, 20)):
        P = jacobi_table(a, b, 8, x)
        for j in range(9):
            ref = eval_jacobi(j, a, b, x)
            assert np.allclose(P[j], ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
            if j <= 5:
                assert np.allclose(P[j], jacobi_explicit(a, b, j, (1 - x) / 2), rtol=1e-11, atol=1e-11 * np.abs(ref).max())


def test_orthonormality_green_identity_tangentiality_supbound():
    D = 14
    g = OrbitGrid(D + 2, 2 * D // 3 + 2, False)
    H = HarmonicEvaluator(g, max_degree=D)
    modes = vector_modes(D)
    val, grad, lap = H.evaluate(modes)
    w = g.omega
    G = np.einsum("kcib,lcib,ib->kl", val, val, w)
    assert np.abs(G - np.eye(len(modes))).max() < 1e-12
    Gg = np.einsum("kacib,lacib,ib->kl", grad, grad, w)
    ll = np.array([m.l * (m.l + 2) for m in modes])
    assert np.abs(Gg - np.diag(ll)).max() < 1e-9
    X = np.stack(g.X)
    assert np.abs(np.einsum("kacib,aib->kcib", grad, X)).max() < 1e-12
    assert np.allclose(lap, -ll.reshape(-1, 1, 1, 1) * val)
    sup = np.sqrt((val ** 2).sum(axis=1)).max(axis=(1, 2))
    bound = np.array([(m.l + 1) / np.sqrt(2 * np.pi ** 2) * (np.sqrt(2) if m.kind == "V" else 1) for m in modes])
    assert np.all(sup <= bound * (1 + 1e-12))


def test_rigorous_backend_encloses_float_values():
    D = 10
    gf, gb = OrbitGrid(12, 9, False), OrbitGrid(12, 9, True)
    Hf, Hb = HarmonicEvaluator(gf, max_degree=D), HarmonicEvaluator(gb, max_degree=D)
    modes = vector_modes(D)
    vf, gf_, _ = Hf.evaluate(modes)
    vb, gb_, _ = Hb.evaluate(modes)
    assert np.all(vb.lower() - 1e-12 <= vf) and np.all(vf <= vb.upper() + 1e-12)
    assert np.all(gb_.lower() - 1e-10 <= gf_) and np.all(gf_ <= gb_.upper() + 1e-10)
    assert vb.r.max() < 1e-13 and gb_.r.max() < 1e-10
