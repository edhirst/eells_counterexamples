"""certificate/index_estimate.py: the negative-definiteness test, the perturbation constant delta,
and the enclosure Q = Qt +- |E| of the form (with the rational coefficients 1/|W|^2, 1/|W|^4)
by its polynomial proxy, checked on W = V_0 = (w^2 - zbar^4, 2 Re(z^2 w)) (|V_0| = 1 - |z|^2|w|^2 is
far from 1, so the error term is O(1) and really tested) against a direct floating-point
evaluation of Q_W."""
import numpy as np

from eells.ball import Ball
from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator, Mode, norm_constant_squared_times_2pi2
from eells.fields import field_from_coefficients
from eells.reference import sobolev_constants
from eells.equation import cross
from eells.sectors import sector_modes, complex_field, real_part
from certificate import index_estimate as ie


def test_negative_definite():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    M = -(A @ A.T) - np.eye(6)
    ok, b = ie.negative_definite(Ball(M, 1e-12 * np.ones((6, 6))))
    assert ok and b < 0
    assert not ie.negative_definite(Ball(-M))[0]
    V, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    M3 = V @ np.diag([-3, -2, -1, -1, -0.5, 0.25]) @ V.T
    assert not ie.negative_definite(Ball(M3))[0]
    assert not ie.negative_definite(Ball(M, 10.0 * np.ones((6, 6))))[0]      # radii larger than the gap


def test_perturbation_constant_formula():
    nX, C8, r = 69.06, 0.99999986, 9.5e-7
    pert, _ = ie.perturbation_constant("9.5e-7", nX, C8)
    c_inf, c_4, _, beta = sobolev_constants(False)
    n = nX + r
    delta = r * (2 * beta * (3 * c_inf * c_4 * n / C8 ** 2 + c_4 / C8) + 12 * beta ** 2 * c_inf * c_4 ** 2 * n ** 2 / C8 ** 3
                 + 4 * beta ** 2 * c_4 ** 2 * n / C8 ** 2 + 4 * c_inf / C8)
    assert pert["delta"] >= delta * (1 - 1e-14) and abs(pert["delta"] - delta) < 1e-10 * delta
    assert pert["beta"] >= beta * (1 - 1e-15) and abs(pert["beta"] - beta) < 1e-14


def _reference_modes():
    """V_0 = (w^2 - zbar^4, 2 Re(z^2 w)) in the basis of Proposition 2.2."""
    c = lambda a, b: float(np.sqrt(float(norm_constant_squared_times_2pi2(a, b, 0)) / (2 * np.pi ** 2)))
    modes = [Mode(2, "U", 2, 0), Mode(4, "U", 0, 0), Mode(3, "V", 1, 0)]
    values = [1.0 / c(0, 2), -1.0 / c(4, 0), np.sqrt(2.0) / c(2, 1)]
    return modes, values


def _true_Q(W, f, df):
    """Q_W^h(f, f) with the exact coefficients, in floating point: f, df complex arrays."""
    Wv, dW = np.stack(W.val), W.grad
    rho = (Wv ** 2).sum(axis=0)
    omega = [np.stack(cross(list(Wv), dW[a])) for a in range(4)]
    tot = 0.0
    for a in range(4):
        E = df[a] - np.stack(cross(list(omega[a]), list(f))) / rho
        B = (omega[a] * f).sum(axis=0) / rho
        tot = tot + (np.abs(E) ** 2).sum(axis=0) - np.abs(B) ** 2
    S = (Wv * f).sum(axis=0) / np.sqrt(rho)
    tot = tot + np.abs(S) ** 2
    return tot


def test_proxy_enclosure_on_reference_field():
    D, L = 4, 6
    degs = ie.integrand_degrees(D, L)
    Ns, Nphi = -(-degs["max"] // 2), degs["max"] // 3 + 1
    gq, gf = OrbitGrid(Ns, Nphi, True), OrbitGrid(Ns, Nphi, False)
    gb = OrbitGrid(D, 2 * ((2 * D) // 3) + 1, True)
    Hq, Hf, Hb = HarmonicEvaluator(gq, max_degree=L), HarmonicEvaluator(gf, max_degree=L), HarmonicEvaluator(gb, max_degree=D)
    modes, values = _reference_modes()
    Wq, Wf, Wb = (field_from_coefficients(modes, values, H) for H in (Hq, Hf, Hb))
    z = gf.rs.reshape(-1, 1) + 0j
    w = gf.rc.reshape(-1, 1) * np.exp(2j * np.pi * np.arange(gf.Nphi) / gf.Nphi)
    zeta = w ** 2 - np.conj(z) ** 4
    V0 = [zeta.real, zeta.imag, 2 * (z ** 2 * w).real]
    assert max(np.abs(Wf.val[c] - V0[c]).max() for c in range(3)) < 1e-13
    kappa1, kappa2, s = ie.rho_bounds(Wb, gb, D)
    # sup |rho - 1| = 7/16 exactly; the rigorous bound adds a Lipschitz remainder (~1e-3)
    assert 7 / 16 <= s < 7 / 16 + 1e-2 and float(kappa1.upper()) >= 7 / 9 and float(kappa2.upper()) >= (7 / 16) * (2 + 7 / 16) / (9 / 16) ** 2
    Wv, dW = Wq.val, Wq.grad
    omega = [cross(Wv, dW[a]) for a in range(4)]
    rng = np.random.default_rng(5)
    for nu, p in ((2, 1), (0, -1), (3, -1)):
        sm = sector_modes(nu, p, L)
        coeffs = rng.normal(size=len(sm)) + 1j * rng.normal(size=len(sm))
        val, grad = complex_field(sm, coeffs, Hq)
        valf, gradf = complex_field(sm, coeffs, Hf)
        if nu == 0:                     # the real test field Re f
            vr, gr = real_part(val, grad)
            z = Ball(np.zeros(gq.shape))
            val = [(v, z) for v in vr]
            grad = [[(g, z) for g in row] for row in gr]
            f = np.stack([v[0] for v in valf]).astype(complex)
            df = np.stack([np.stack([g[0] for g in row]) for row in gradf]).astype(complex)
        else:
            f = np.stack([v[0] + 1j * v[1] for v in valf])
            df = np.stack([np.stack([g[0] + 1j * g[1] for g in row]) for row in gradf])
        q = ie.FieldQuantities(val, grad, Wv, omega)
        ints = ie.pair_integrals(gq, q, q, degs)
        norms = ie._norms(ints)
        Q, Hh, Qt, e = ie.enclose_Q(ints, norms, norms, kappa1, kappa2)
        truth = gf.integrate(_true_Q(Wf, f, df), degs["max"])          # not exact (rational integrand) but accurate
        assert Q[0].lower() <= truth <= Q[0].upper()
        assert abs(truth - float(Qt[0].m)) <= e and abs(truth - float(Qt[0].m)) > 1e-3 * abs(truth)
        assert abs(float(Q[1].m)) <= 1e-9 and abs(float(Hh[1].m)) <= 1e-9          # Q^h(f, f) real
        h_coef = sum((1.0 + m.l * (m.l + 2)) * abs(c) ** 2 for m, c in zip(sm, coeffs))
        if nu == 0:
            # Re f = (f + conj f)/2 has H^1 norm^2 = (||f||^2 + Re <f, conj f>)/2, computed directly instead
            h_coef = gf.integrate((np.abs(f) ** 2).sum(axis=0) + (np.abs(df) ** 2).sum(axis=(0, 1)), degs["T1"])
        assert Hh[0].lower() - 1e-9 <= h_coef <= Hh[0].upper() + 1e-9
