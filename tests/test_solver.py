"""The solver's Galerkin residual R(a) and its Jacobian (finite-difference check)."""
import numpy as np

from numerics.solve_approximate import GalerkinSystem
from eells.reference import reference_field
from eells.equation import dot


def test_solver_jacobian_matches_finite_differences():
    sys_ = GalerkinSystem(8, kappa=10.0)
    rng = np.random.default_rng(0)
    V0 = sys_.V0
    nV = np.sqrt(dot(V0, V0))
    a = sys_.project([v / nV for v in V0]) + 0.05 * rng.standard_normal(len(sys_.modes))
    res, W, op, F0 = sys_.residual(a)
    J = sys_.matrix(op)
    h = rng.standard_normal(len(sys_.modes))
    errs = []
    for t in (1e-4, 1e-5):
        rp = sys_.residual(a + t * h)[0]
        rm = sys_.residual(a - t * h)[0]
        errs.append(np.linalg.norm((rp - rm) / (2 * t) - J @ h))
    assert errs[0] < 1e-5 * np.linalg.norm(J @ h) and errs[1] < errs[0]
