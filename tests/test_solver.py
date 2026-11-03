"""The solver's Galerkin residual R(a) and its Jacobian (finite-difference check)."""
import numpy as np
import pytest

from numerics.solve_approximate import GalerkinSystem
from eells.reference import reference_field
from eells.equation import dot


@pytest.mark.parametrize("q", [2, 3])
def test_solver_jacobian_matches_finite_differences(q):
    sys_ = GalerkinSystem(8, q, kappa=10.0, chunk_points=200)
    rng = np.random.default_rng(0)
    V = reference_field(sys_.grid, q)
    nV = np.sqrt(dot(V, V))
    a = sys_.project([v / nV for v in V]) + 0.05 * rng.standard_normal(len(sys_.modes))
    res, W, op = sys_.residual(a)
    J = sys_.matrix(W)
    h = rng.standard_normal(len(sys_.modes))
    errs = []
    for t in (1e-4, 1e-5):
        rp = sys_.residual(a + t * h)[0]
        rm = sys_.residual(a - t * h)[0]
        errs.append(np.linalg.norm((rp - rm) / (2 * t) - J @ h))
    assert errs[0] < 1e-5 * np.linalg.norm(J @ h) and errs[1] < errs[0]
