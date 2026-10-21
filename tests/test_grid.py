"""Quadrature exactness and the rigorous sup/inf bounds of the orbit-space grid."""
import numpy as np

from eells.grid import OrbitGrid


def _test_function(g):
    s = g.s.reshape(-1, 1)
    c2, c4 = g.cos_n_phi(2).reshape(1, -1), g.cos_n_phi(4).reshape(1, -1)
    # (z,w)-degree 16: s^5 (degree 10) times cos 2 phi (needs w^2-type factors, degree 6)
    return s ** 5 * (1 + c2) + s ** 2 * c4, 16


def test_integral_exact_float_and_ball():
    exact = 2 * np.pi ** 2 / 6          # int s^5 = 1/6, the oscillating parts integrate to 0
    for rig in (False, True):
        g = OrbitGrid(8, 11, rig)
        f, deg = _test_function(g)
        v = g.integrate(f, deg)
        if rig:
            assert v.lower() <= exact <= v.upper() and v.r < 1e-12
        else:
            assert abs(v - exact) < 1e-13
        cf = g.analyse(f, deg)
        w = cf.integral_S3()
        if rig:
            assert w.lower() <= exact <= w.upper()
        else:
            assert abs(w - exact) < 1e-13


def test_clenshaw_curtis_weights_exact():
    g = OrbitGrid(9, 4, False)
    for p in range(10):
        assert abs((g.w_s * g.s ** p).sum() - 1 / (p + 1)) < 1e-14


def test_bounds_enclose_min_max():
    g = OrbitGrid(8, 11, True)
    f, deg = _test_function(g)
    lo, hi = g.analyse(f, deg).bounds(N1=4096, N2=512)
    # dense float evaluation for reference
    s = np.linspace(0, 1, 2001).reshape(-1, 1); phi = np.linspace(0, 2 * np.pi, 1001).reshape(1, -1)
    ref = s ** 5 * (1 + np.cos(2 * phi)) + s ** 2 * np.cos(4 * phi)
    assert lo <= ref.min() and hi >= ref.max()
    assert ref.min() - lo < 0.05 and hi - ref.max() < 0.05
