"""The sector bookkeeping of eells/sectors.py: completeness of the modes, the circle character
and the antipodal parity (checked pointwise against a direct evaluation of Y_{m,n,j}),
orthonormality, the H^1 diagonal, exactness of the orbit-space quadrature for same-sector
integrands and the vanishing of the bilinear (weight 2 nu) and cross-sector integrals."""
import numpy as np
from scipy.special import eval_jacobi

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator, norm_constant_squared_times_2pi2
from eells.sectors import sector_modes, complex_field, make_mode, _cfactors


# ---------------------------------------------------------------- direct evaluation at arbitrary points
def Y_direct(md, z, w):
    """Y_{m,n,j}(z, w) = c z^{(m)} w^{(n)} P_j^{(|m|,|n|)}(1 - 2|z|^2) for complex arrays z, w."""
    c = np.sqrt(float(norm_constant_squared_times_2pi2(md.a, md.b, md.j)) / (2 * np.pi ** 2))
    zm = z ** md.m if md.m >= 0 else np.conj(z) ** (-md.m)
    wn = w ** md.n if md.n >= 0 else np.conj(w) ** (-md.n)
    return c * zm * wn * eval_jacobi(md.j, md.a, md.b, 1 - 2 * np.abs(z) ** 2)


def mode_direct(md, z, w):
    """The C^3-valued mode at (z, w): array [3, ...]."""
    Y = Y_direct(md, z, w)
    out = np.zeros((3,) + Y.shape, dtype=complex)
    for comp, fac in enumerate(_cfactors(md.kind, 1.0, 0.0, np.sqrt(0.5))):
        if fac is not None:
            out[comp] = (fac[0] + 1j * fac[1]) * Y
    return out


def field_direct(modes, coeffs, z, w):
    return sum(c * mode_direct(m, z, w) for m, c in zip(modes, coeffs))


def full_s3_grid(ns=12, n1=16, n2=16):
    """Points (z, w) = (sqrt s e^{i psi1}, sqrt(1-s) e^{i psi2}) and weights for the volume
    element ds dpsi1 dpsi2 / 2 (exact for polynomials of degree < 2 ns in s and < n1, n2 in the angles)."""
    xs, ws = np.polynomial.legendre.leggauss(ns)
    s = 0.5 * (xs + 1)
    ws = 0.5 * ws
    p1 = 2 * np.pi * np.arange(n1) / n1
    p2 = 2 * np.pi * np.arange(n2) / n2
    S, P1, P2 = np.meshgrid(s, p1, p2, indexing="ij")
    z = np.sqrt(S) * np.exp(1j * P1)
    w = np.sqrt(1 - S) * np.exp(1j * P2)
    W = np.broadcast_to(ws.reshape(-1, 1, 1), S.shape) * (2 * np.pi / n1) * (2 * np.pi / n2) / 2
    return z.ravel(), w.ravel(), W.ravel()


def rotate(theta, z, w):
    return np.exp(1j * theta) * z, np.exp(-2j * theta) * w


def act_target(theta, f):
    """e^{i theta}.(zeta, t) = (e^{-4 i theta} zeta, t) on complex 3-vectors [3, ...] (C-linear)."""
    zeta = f[0] + 1j * f[1]
    zetap = f[0] - 1j * f[1]
    zeta, zetap = np.exp(-4j * theta) * zeta, np.exp(4j * theta) * zetap
    return np.stack([(zeta + zetap) / 2, (zeta - zetap) / (2j), f[2]])


# ---------------------------------------------------------------- tests
def test_sector_modes_are_complete_and_distinct():
    L = 7
    allm = []
    for nu in range(-(3 * L + 4), 3 * L + 5):          # |nu| = |2n - m -+ 4| <= 3L + 4
        for p in (1, -1):
            ms = sector_modes(nu, p, L)
            assert all(m.nu == nu and m.parity == p and m.l <= L for m in ms)
            allm += ms
    assert len(set(allm)) == len(allm)
    assert len(allm) == 3 * sum((l + 1) ** 2 for l in range(L + 1))
    assert all(m == make_mode(m.kind, m.m, m.n, m.j) for m in allm)


def test_character_and_parity_pointwise():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(4, 20))
    v /= np.linalg.norm(v, axis=0)
    z, w = v[0] + 1j * v[1], v[2] + 1j * v[3]
    for md in (make_mode("P", -3, 1, 2), make_mode("M", 5, -1, 0), make_mode("T", -2, 2, 1),
               make_mode("P", 0, 2, 0), make_mode("T", 1, 0, 3)):
        f = mode_direct(md, z, w)
        for theta in (0.3, 1.7):
            zt, wt = rotate(theta, z, w)
            lhs = mode_direct(md, zt, wt)                       # f(e^{i theta} y)
            rhs = np.exp(-1j * md.nu * theta) * act_target(theta, f)
            assert np.abs(lhs - rhs).max() < 1e-12
        fm = mode_direct(md, -z, -w)
        assert np.abs(fm - md.parity * np.stack([f[0], f[1], -f[2]])).max() < 1e-12


def test_orthonormality_h1_diagonal_and_quadrature_exactness():
    L = 8
    g = OrbitGrid(L + 2, 2 * L // 3 + 2, False)
    H = HarmonicEvaluator(g, max_degree=L)
    z, w, Wt = full_s3_grid(ns=L + 2, n1=2 * L + 3, n2=2 * L + 3)
    for nu, p in ((0, 1), (0, -1), (1, -1), (2, 1), (3, -1), (5, 1)):
        modes = sector_modes(nu, p, L)
        k = len(modes)
        vals, grads = [], []
        for md in modes:
            val, grad = complex_field([md], [1.0], H)
            vals.append(np.stack([v[0] + 1j * v[1] for v in val]))                       # [3, i, b]
            grads.append(np.stack([np.stack([x[0] + 1j * x[1] for x in row]) for row in grad]))  # [4, 3, i, b]
        V = np.stack(vals)
        Gr = np.stack(grads)
        M = np.einsum("kcib,lcib,ib->kl", np.conj(V), V, g.omega)
        T1 = np.einsum("kacib,lacib,ib->kl", np.conj(Gr), Gr, g.omega)
        ll = np.array([m.l * (m.l + 2) for m in modes], dtype=float)
        assert np.abs(M - np.eye(k)).max() < 1e-12
        assert np.abs(T1 - np.diag(ll)).max() < 1e-9
        # same-sector sesquilinear integrals agree with a full S^3 quadrature; bilinear ones vanish
        sub = modes[: min(6, k)]
        F = np.stack([mode_direct(m, z, w) for m in sub])                                # [k, 3, N]
        Mfull = np.einsum("kcx,lcx,x->kl", np.conj(F), F, Wt)
        assert np.abs(Mfull - np.eye(len(sub))).max() < 1e-10
        Bil = np.einsum("kcx,lcx,x->kl", F, F, Wt)
        if nu != 0:
            assert np.abs(Bil).max() < 1e-10
    # cross-sector integrals vanish
    f1 = mode_direct(make_mode("P", -3, 1, 1), z, w)
    f2 = mode_direct(make_mode("T", 2, 0, 1), z, w)       # nu = -2
    f3 = mode_direct(make_mode("P", -5, 1, 0), z, w)      # nu = 3, parity -1
    f4 = mode_direct(make_mode("M", 1, 0, 1), z, w)       # nu = 3, parity +1
    assert abs(np.einsum("cx,cx,x->", np.conj(f1), f2, Wt)) < 1e-10
    assert abs(np.einsum("cx,cx,x->", np.conj(f3), f4, Wt)) < 1e-10


def test_complex_field_rigorous_encloses_float():
    L = 6
    gf, gb = OrbitGrid(9, 7, False), OrbitGrid(9, 7, True)
    Hf, Hb = HarmonicEvaluator(gf, max_degree=L), HarmonicEvaluator(gb, max_degree=L)
    modes = sector_modes(2, 1, L)
    rng = np.random.default_rng(3)
    coeffs = rng.normal(size=len(modes)) + 1j * rng.normal(size=len(modes))
    vf, gfl = complex_field(modes, coeffs, Hf)
    vb, gbl = complex_field(modes, coeffs, Hb)
    for c in range(3):
        for k in range(2):
            assert np.all(vb[c][k].lower() - 1e-12 <= vf[c][k]) and np.all(vf[c][k] <= vb[c][k].upper() + 1e-12)
            assert vb[c][k].r.max() < 1e-12
            for a in range(4):
                assert np.all(gbl[a][c][k].lower() - 1e-10 <= gfl[a][c][k]) and np.all(gfl[a][c][k] <= gbl[a][c][k].upper() + 1e-10)
