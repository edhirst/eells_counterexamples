"""Negative directions of the quadratic form Q_Wbar of Section 5 (floating point).

In each sector S_{nu,p} (eells/sectors.py) the Hermitian Galerkin matrix

    Q_{kl} = Q^h_Wbar(Phi_k, Phi_l),     Q_W(xi) = int |d xi - (W x dW) x xi / |W|^2|^2
                                                  - |(W x dW) . xi|^2 / |W|^4 + (W . xi)^2 / |W|^2,

of the L^2-orthonormal modes Phi_k of degree <= L is assembled by exact quadrature and
diagonalised.  Its negative eigenvalues approximate the negative eigenvalues of the Jacobi
operator of ubar = Wbar/|Wbar| (the L^2 mass matrix is the identity, Proposition 2.2 / Lemma 2.1),
and the H^1 Gram matrix is diag(1 + l_k (l_k + 2)).  The eigenvectors with clearly negative
eigenvalue are written to a coefficient file; the certificate (certificate/index_estimate.py)
re-verifies, in ball arithmetic, that they span a negative-definite subspace of Q_W for the
harmonic map W = Wbar + v.  Near-zero eigenvalues are the Jacobi fields of the isometries (eight
of them: sectors (0,+) x 2, (+-1,+), (+-3,+), (+-4,-)); they are reported but not stored.

For nu = 0 the sector is conjugation invariant and the stored field is made real
(f -> (f + conj f)/2 after fixing the phase); for nu != 0 the complex field f yields the two real
directions Re f and Im f.

**Integer coefficients.**  The test fields only need to be negative directions, not exact
eigenvectors.  Each eigenvector is therefore rotated (nu != 0: its largest coefficient made real
and positive), scaled so that its largest coefficient is ``--scale`` (default 100) and rounded to
Gaussian integers; zero coefficients are dropped.  Rounding is symmetric, so for nu = 0 the
rounded field is still real.  The stored fields are these short integer combinations (Table
tab:index-fields of the paper), and the certificate uses them exactly.

The index part of the paper concerns the Hopf invariant q = 2 (the group G of Section 2.1).

Usage:  python -m numerics.index_basis --coefficients data/Wbar_q2_D40.json --max-degree 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator
from eells.fields import field_from_coefficients, load_coefficients
from eells.equation import dot, cross
from eells.sectors import sector_modes, complex_field, make_mode, save_index_basis


def index_grid(D, L, rigorous=False):
    """Grid exact for the integrands of Q_Wbar(f, g) with deg f, deg g <= L:  |(Wbar x dWbar) x f|^2
    has degree 4D + 2 + 2L (the largest)."""
    p = 4 * D + 2 + 2 * L
    return OrbitGrid(-(-p // 2), p // 3 + 1, rigorous), p


class SectorGalerkin:
    """Galerkin matrices of Q_Wbar in the modes of one sector."""

    def __init__(self, W, grid, H, batch=32):
        self.grid, self.H, self.batch = grid, H, batch
        Wv, dW = W.val, W.grad
        self.Wv = np.stack(Wv)
        self.rho = dot(Wv, Wv)
        self.omega = np.stack([np.stack(cross(Wv, dW[a])) for a in range(4)])          # [a, c, i, b]
        self.inv_rho = 1.0 / self.rho
        self.inv_sqrt_rho = 1.0 / np.sqrt(self.rho)
        self.w = grid.omega

    def features(self, modes):
        """Per mode: E = d f - (omega x f)/rho [4,3], Bq = (omega . f)/rho [4], S = (W . f)/sqrt(rho),
        flattened with the quadrature weights so that Q = F^H diag(sign) F."""
        out = []
        for md in modes:
            val, grad = complex_field([md], [1.0], self.H)
            f = np.stack([v[0] + 1j * v[1] for v in val])                               # [3, i, b]
            df = np.stack([np.stack([g[0] + 1j * g[1] for g in row]) for row in grad])    # [4, 3, i, b]
            oxf = np.stack([np.stack(cross(self.omega[a], f)) for a in range(4)])
            E = df - oxf * self.inv_rho
            Bq = np.einsum("acib,cib->aib", self.omega, f) * self.inv_rho
            S = np.einsum("cib,cib->ib", self.Wv, f) * self.inv_sqrt_rho
            sw = np.sqrt(self.w)
            out.append(np.concatenate([(E * sw).ravel(), (Bq * sw).ravel(), (S * sw).ravel()]))
        return np.stack(out)

    def matrices(self, modes, log=print):
        k = len(modes)
        G = self.grid.omega.size
        nE, nB = 12 * G, 4 * G
        sign = np.concatenate([np.ones(nE), -np.ones(nB), np.ones(G)])
        F = np.empty((k, nE + nB + G), dtype=complex)
        for i in range(0, k, self.batch):
            F[i:i + self.batch] = self.features(modes[i:i + self.batch])
        Q = np.conj(F) @ (sign.reshape(-1, 1) * F.T)
        Q = 0.5 * (Q + np.conj(Q.T))
        h = np.array([1.0 + m.l * (m.l + 2) for m in modes])
        return Q, h


def conj_index(modes):
    """Index map k -> k' with conj(Phi_k) = Phi_{k'} (P <-> M, T -> T, (m, n) -> (-m, -n))."""
    pos = {m: i for i, m in enumerate(modes)}
    swap = {"P": "M", "M": "P", "T": "T"}
    return np.array([pos[make_mode(swap[m.kind], -m.m, -m.n, m.j)] for m in modes])


def realify(modes, c):
    """For nu = 0: the coefficients of the real field (f + conj f)/2 (phase fixed to maximise
    its norm), renormalised to unit L^2 norm."""
    ci = conj_index(modes)
    best = None
    for phase in (1.0, 1j, np.exp(0.25j * np.pi), np.exp(-0.25j * np.pi)):
        d = phase * c
        r = 0.5 * (d + np.conj(d[ci]))
        if best is None or np.linalg.norm(r) > np.linalg.norm(best):
            best = r
    best = best / np.linalg.norm(best)
    assert np.abs(best - np.conj(best[ci])).max() < 1e-12
    return best


def integer_coefficients(modes, c, scale, real):
    """Rotate/scale c so that its largest coefficient is ``scale`` and round to Gaussian integers
    (symmetric rounding keeps the conjugation symmetry of a real field)."""
    k = int(np.argmax(np.abs(c)))
    if real:
        c = c * (scale / abs(c[k]))
    else:
        c = c * (np.conj(c[k]) / abs(c[k])) * (scale / abs(c[k]))
    ci = np.round(c.real) + 1j * np.round(c.imag)
    if real:
        cj = conj_index(modes)
        assert np.array_equal(ci, np.conj(ci[cj])), "rounding broke the conjugation symmetry"
    return ci


def rayleigh(Q, h, c):
    num = float(np.real(np.conj(c) @ Q @ c))
    return num / float(np.real(np.conj(c) @ c)), num / float(np.real(np.conj(c) @ (h * c)))


def compute_basis(coeff_path, L, nu_max=6, threshold=-0.05, batch=32, scale=100, log=print):
    t0 = time.time()
    modes, values, _ = load_coefficients(coeff_path)
    assert modes[0].q == 2, "the index certificate concerns the Hopf invariant 2"
    D = max(m.l for m in modes)
    grid, p = index_grid(D, L)
    H = HarmonicEvaluator(grid, max_degree=max(D, L))
    W = field_from_coefficients(modes, values, H)
    log(f"Wbar degree {D}, modes up to degree L = {L}, grid {grid.Ns + 1} x {grid.Nphi} (exact to degree {p})  ({time.time() - t0:.0f}s)")
    sg = SectorGalerkin(W, grid, H, batch)
    fields, sectors = [], []
    for nu in range(nu_max + 1):
        for p_ in (-1, 1):
            sm = sector_modes(nu, p_, L)
            Q, h = sg.matrices(sm)
            lam, V = np.linalg.eigh(Q)
            hs = 1.0 / np.sqrt(h)
            mu = np.linalg.eigvalsh(hs.reshape(-1, 1) * Q * hs.reshape(1, -1))
            neg = [i for i in range(len(lam)) if lam[i] < threshold]
            zero = [float(x) for x in lam if threshold <= x < 1e-2]
            info = {"nu": nu, "parity": p_, "dim": len(sm), "lowest_eigenvalues_L2": [float(x) for x in lam[:4]],
                    "lowest_eigenvalues_H1": [float(x) for x in mu[:4]], "negative": len(neg), "near_zero": zero}
            sectors.append(info)
            log(f"  sector ({nu:d},{'+' if p_ > 0 else '-'}): dim {len(sm):4d}, lowest L2 eigenvalues {np.array2string(lam[:4], precision=4)}, "
                f"H1 quotients {np.array2string(mu[:3], precision=4)}  ({time.time() - t0:.0f}s)")
            for i in neg:
                c = V[:, i]
                if nu == 0:
                    c = realify(sm, c)
                if scale:
                    c = integer_coefficients(sm, c, scale, real=(nu == 0))
                r2, rh = rayleigh(Q, h, c)
                keep = [k for k in range(len(sm)) if c[k] != 0]
                name = f"f_{len(fields) + 1}"
                fields.append({"name": name, "nu": nu, "parity": p_, "modes": [sm[k] for k in keep],
                               "coeffs": c[keep],
                               "diagnostics": {"L2_eigenvalue": float(lam[i]), "L2_rayleigh": r2, "H1_rayleigh": rh,
                                               "real": nu == 0, "nonzero_coefficients": len(keep)}})
                log(f"    -> {name}: L2 eigenvalue {lam[i]:.6f}; with {len(keep)} {'integer ' if scale else ''}coefficients "
                    f"Rayleigh quotients L2 {r2:.6f}, H1 {rh:.6f}")
    n_real = sum(1 if f["nu"] == 0 else 2 for f in fields)
    log(f"{len(fields)} complex fields = {n_real} real negative directions  ({time.time() - t0:.0f}s)")
    meta = {"coefficients": str(coeff_path), "degree_Wbar": D, "max_degree": L, "nu_max": nu_max, "scale": scale,
            "grid": {"Ns": grid.Ns, "Nphi": grid.Nphi, "exact_degree": p}, "threshold": threshold,
            "sectors": sectors, "real_directions": n_real, "seconds": time.time() - t0}
    return fields, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coefficients", default="data/Wbar_q2_D40.json")
    ap.add_argument("--max-degree", type=int, default=8, help="degree L of the test fields")
    ap.add_argument("--nu-max", type=int, default=6, help="sectors nu = 0..nu_max are scanned")
    ap.add_argument("--scale", type=int, default=100, help="largest coefficient after rounding to Gaussian integers "
                                                           "(0: keep the float eigenvectors)")
    ap.add_argument("--out", default="data/index_basis_q2.json", help="output JSON")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    fields, meta = compute_basis(args.coefficients, args.max_degree, args.nu_max, batch=args.batch, scale=args.scale)
    out = Path(args.out)
    save_index_basis(out, fields, meta)
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in meta.items() if k != "sectors"}, indent=1))


if __name__ == "__main__":
    main()
