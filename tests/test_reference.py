"""Propositions 2.5 and 2.6 and Section 2.3: Sobolev constants, the Hopf map and the
reference field V_q = (w^2 - zbar^{2q}/4, Re(z^q w))."""
import numpy as np
import pytest

from eells.grid import OrbitGrid
from eells.harmonics import vector_modes
from eells.reference import reference_field, extended_hopf_map, sobolev_constants, reference_norm_min


def _V(q, z, w):
    """V_q from its definition at points (z, w) of C^2 (independent of the grid code)."""
    zeta = w ** 2 - np.conj(z) ** (2 * q) / 4
    return np.stack([zeta.real, zeta.imag, (z ** q * w).real])


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5, 6])
def test_reference_field_values_and_norm(q):
    g = OrbitGrid(40, 32, False, q)
    V = np.stack(reference_field(g, q))
    z = g.rs.reshape(-1, 1) + 0j
    w = g.rc.reshape(-1, 1) * np.exp(2j * np.pi * np.arange(g.Nphi) / g.Nphi)
    assert np.abs(V - _V(q, z, w)).max() < 1e-14
    s = g.s.reshape(-1, 1)
    norm = np.sqrt((V ** 2).sum(axis=0))
    assert np.abs(norm - (1 - s + s ** q / 4)).max() < 1e-14              # |V_q| = |w|^2 + |z|^{2q}/4
    assert norm.max() <= 1 + 1e-14 and norm.min() >= reference_norm_min(q) - 1e-9 > 0
    if q <= 4:
        assert abs(reference_norm_min(q) - 0.25) < 1e-9                    # 1/4 <= |V_q| <= 1 (attained at w = 0)


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5, 6])
def test_reference_field_is_equivariant(q):
    """V_q(g.x) = g.V_q(x) for the generators of G_q, so that Wbar . V_q is G_q-invariant."""
    rng = np.random.default_rng(q)
    x = rng.normal(size=(4, 50)); x /= np.linalg.norm(x, axis=0)
    z, w = x[0] + 1j * x[1], x[2] + 1j * x[3]
    th = 0.7
    V = _V(q, z, w)
    zeta = (V[0] + 1j * V[1]) * np.exp(-2j * q * th)
    assert np.abs(_V(q, np.exp(1j * th) * z, np.exp(-1j * q * th) * w) - np.stack([zeta.real, zeta.imag, V[2]])).max() < 1e-14
    assert np.abs(_V(q, np.conj(z), np.conj(w)) - V * np.array([1, -1, 1]).reshape(3, 1)).max() < 1e-14
    assert np.abs(_V(q, z, -w) - V * np.array([1, 1, -1]).reshape(3, 1)).max() < 1e-14


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5, 6])
def test_kappa_h_is_orthogonal_to_reference_field(q):
    """V_q is pointwise orthogonal to kappa.h(z^q, 2w), a check of the formula for V_q.  (The paper
    does not use this identity: it obtains H(V_q/|V_q|) = q from V_q = pi(z^q/2, w), Lemma 2.12 and
    Proposition 2.14.)"""
    g = OrbitGrid(40, 32, False, q)
    V = reference_field(g, q)
    z = g.rs.reshape(-1, 1) + 0j
    w = g.rc.reshape(-1, 1) * np.exp(2j * np.pi * np.arange(g.Nphi) / g.Nphi)
    h = extended_hopf_map(z, 2 * w, q)
    kh = np.stack([h[0], -h[1], h[2]])          # kappa acts on (zeta, t) by conjugating zeta
    assert np.abs(sum(kh[c] * V[c] for c in range(3))).max() < 1e-14


def test_sobolev_constants():
    cf = sobolev_constants(False)
    cb = sobolev_constants(True)
    assert abs(cf[0] - np.sqrt((np.pi ** 2 / 6 - 1.25) / (2 * np.pi ** 2))) < 1e-15
    assert abs(cf[1] - (2 / (9 * np.pi ** 2)) ** 0.25) < 1e-15
    assert abs(cf[2] - 2 * np.sqrt(2) / 9) < 1e-15
    assert abs(cf[3] - (2 * np.pi ** 2) ** -0.25) < 1e-15
    assert all(b.lower() <= f <= b.upper() and b.r < 1e-15 for f, b in zip(cf, cb))
    # the series behind c_inf: sum_{l>=2} 1/(l+1)^2 = pi^2/6 - 5/4
    assert abs(sum(1 / (l + 1) ** 2 for l in range(2, 200000)) - (np.pi ** 2 / 6 - 1.25)) < 1e-5
    # c_5^2 = max_{l>=2} l(l+2)/(l+1)^4 = 8/81, attained at l = 2 (Proposition 2.6)
    assert max(l * (l + 2) / (l + 1) ** 4 for l in range(2, 1000)) == 8 / 81
    # the constants assume that X contains no modes of degree 0 or 1, for every q
    assert all(min(m.l for m in vector_modes(40, q)) == 2 for q in (1, 2, 3, 4, 5, 6))
