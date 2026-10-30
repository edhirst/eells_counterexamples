"""Propositions 2.3, 2.4, 2.6, 2.7: Sobolev constants, the Hopf map and the reference field."""
import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import vector_modes
from eells.reference import reference_field, extended_hopf_map, sobolev_constants


def test_reference_field_norm_identity():
    g = OrbitGrid(40, 32, False)
    V0 = reference_field(g)
    s = g.s.reshape(-1, 1)
    assert np.abs(np.sqrt(sum(v * v for v in V0)) - (1 - s * (1 - s))).max() < 1e-14
    assert np.sqrt(sum(v * v for v in V0)).max() <= 1 + 1e-14          # the lemma sup |V_0| = 1


def test_kappa_h_is_orthogonal_to_reference_field():
    g = OrbitGrid(40, 32, False)
    V0 = reference_field(g)
    z = g.rs.reshape(-1, 1) + 0j
    w = g.rc.reshape(-1, 1) * np.exp(1j * 2 * np.pi * np.arange(g.Nphi) / g.Nphi)
    h = extended_hopf_map(z, w)
    kh = np.stack([h[0], -h[1], h[2]])          # kappa acts on (zeta, t) by conjugating zeta
    assert np.abs(sum(kh[c] * V0[c] for c in range(3))).max() < 1e-14


def test_sobolev_constants():
    cf = sobolev_constants(False)
    cb = sobolev_constants(True)
    assert abs(cf[0] - np.sqrt((np.pi ** 2 / 6 - 1.25) / (2 * np.pi ** 2))) < 1e-15
    assert abs(cf[1] - (2 / (9 * np.pi ** 2)) ** 0.25) < 1e-15
    assert abs(cf[2] - 2 * np.sqrt(2) / 9) < 1e-15
    assert all(b.lower() <= f <= b.upper() and b.r < 1e-15 for f, b in zip(cf, cb))
    # the series behind c_inf: sum_{l>=2} 1/(l+1)^2 = pi^2/6 - 5/4
    assert abs(sum(1 / (l + 1) ** 2 for l in range(2, 200000)) - (np.pi ** 2 / 6 - 1.25)) < 1e-5
    # c_5^2 = max_{l>=2} l(l+2)/(l+1)^4 = 8/81, attained at l = 2 (Proposition 2.4)
    assert max(l * (l + 2) / (l + 1) ** 4 for l in range(2, 1000)) == 8 / 81
    # both constants assume that X contains no modes of degree 0 or 1
    assert min(m.l for m in vector_modes(40)) == 2
