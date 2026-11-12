"""Sectors of C^3-valued fields under the circle action and the antipodal map (Appendix C, q = 2).

G acts on fields xi: S^3 -> R^3 (or C^3) by  (g.xi)(x) = g xi(g^{-1} x), with the actions of
Section 2.1 for q = 2:  e^{i theta}.(z, w) = (e^{i theta} z, e^{-2 i theta} w),  e^{i theta}.(zeta, t) =
(e^{-4 i theta} zeta, t),  epsilon.(z, w) = -(z, w),  epsilon.(zeta, t) = (zeta, -t), where
`epsilon` denotes here the antipodal element, which the paper writes e^{i pi} epsilon (the paper's
epsilon is (z, w) -> (z, -w), called S in eells/harmonics.py), and
zeta = xi_1 + i xi_2 and t = xi_3.  For complex fields zeta = xi_1 + i xi_2 and
zeta' = xi_1 - i xi_2 are independent, and e^{i theta} multiplies zeta' by e^{4 i theta}.

**Modes.**  From the Jacobi harmonics Y = Y_{m,n,j} of Lemma 2.3 (e^{i theta}.Y = e^{i(m-2n)theta} Y,
epsilon.Y = (-1)^{m+n} Y = (-1)^l Y, l = |m| + |n| + 2j) build the C^3-valued fields

    P_{m,n,j} = (Y, -iY, 0) / sqrt 2     (zeta = sqrt2 Y, zeta' = 0),
    M_{m,n,j} = (Y,  iY, 0) / sqrt 2     (zeta = 0, zeta' = sqrt2 Y),
    T_{m,n,j} = (0, 0, Y).

Then  e^{i theta}.xi = e^{i nu theta} xi  and  epsilon.xi = p xi  with

    nu(P) = 2n - m - 4,   nu(M) = 2n - m + 4,   nu(T) = 2n - m,
    p(P) = p(M) = (-1)^l,  p(T) = -(-1)^l.

(e^{i theta}.P)(x) = e^{i theta}.P(e^{-i theta} x): the component zeta picks up e^{-4 i theta}
from the target and e^{-i(m-2n) theta} from the source.)  The modes are L^2-orthonormal; those
with given (nu, p) and degree <= L span the degree-<= L part of the sector

    S_{nu,p} = { xi in H^1(S^3, C^3) : e^{i theta}.xi = e^{i nu theta} xi,  epsilon.xi = p xi }.

Complex conjugation maps S_{nu,p} onto S_{-nu,p} (conj P_{m,n,j} = M_{-m,-n,j},
conj T_{m,n,j} = T_{-m,-n,j}), so S_{0,p} is conjugation invariant and its real fields are the
real sector; the G-symmetric fields of Proposition 2.4 are the real, kappa-even fields of S_{0,+}.

**Orthogonality.**  A G-invariant symmetric bilinear form (Q_W of eq. (quadraticform) for an
equivariant W, the L^2 and the H^1 products) vanishes between different sectors: if B(g.xi, g.eta)
= B(xi, eta), xi in S_{nu,p}, eta in S_{nu',p'} and (nu, p) != (nu', p'), then
B(xi, eta) = e^{i(nu + nu') theta} B(xi, eta) for all theta (or = p p' B(xi, eta)), so
B(xi, eta) = 0 unless nu' = -nu and p' = p; for the sesquilinear extension
B^h(f, g) := B(conj f, g) this means B^h(f, g) = 0 unless (nu', p') = (nu, p).

**Real fields from a complex sector.**  For f, g in S_{nu,p} with nu != 0, the bilinear value
B(f, g) has weight 2 nu and vanishes, hence, with q = B^h(f, g) = A + iB,

    B(Re f, Re g) = B(Im f, Im g) = A/2,     B(Re f, Im g) = -B(Im f, Re g) = B/2.

So the real Gram matrix of (Re f_1, .., Re f_k, Im f_1, .., Im f_k) is (1/2) [[A, B], [-B, A]] for
the Hermitian matrix A + iB of the f_i, and (Re f, Im f) are B-orthogonal with equal values.
For nu = 0 the test fields are taken real (xi = Re f).

**Quadrature.**  For f, g in the same sector, conj(f).g, <df, dg>, conj(W.f)(W.g), ... are
circle- and epsilon-invariant polynomials (a real orthogonal map of R^3, extended C-linearly,
preserves conj(a).b), so the orbit-space quadrature of eells/grid.py is exact for them.
Fields are stored as pairs (re, im) of real arrays, components [c], tangential partials [a][c].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import arith as ar
from .ball import Ball
from .harmonics import HarmonicEvaluator

KINDS = ("P", "M", "T")


@dataclass(frozen=True, order=True)
class GMode:
    """A mode P_{m,n,j}, M_{m,n,j} or T_{m,n,j} (kind 'P', 'M', 'T'), l = |m| + |n| + 2j."""
    l: int
    kind: str
    m: int
    n: int
    j: int

    @property
    def a(self):
        return abs(self.m)

    @property
    def b(self):
        return abs(self.n)

    @property
    def nu(self):
        return 2 * self.n - self.m - {"P": 4, "M": -4, "T": 0}[self.kind]

    @property
    def parity(self):
        return (-1) ** self.l * (-1 if self.kind == "T" else 1)

    def label(self):
        return f"{self.kind}(m={self.m},n={self.n},j={self.j})"


def make_mode(kind, m, n, j):
    assert kind in KINDS and j >= 0
    return GMode(abs(m) + abs(n) + 2 * j, kind, m, n, j)


def sector_modes(nu, parity, L):
    """All modes of the sector S_{nu, parity} of degree <= L, sorted by (l, kind, m, n, j)."""
    out = []
    for kind, shift in (("P", 4), ("M", -4), ("T", 0)):
        for n in range(-L, L + 1):
            m = 2 * n - nu - shift                 # nu = 2n - m - shift
            base = abs(m) + abs(n)
            if base > L:
                continue
            md = make_mode(kind, m, n, 0)
            if md.parity != parity:
                continue
            for j in range((L - base) // 2 + 1):
                out.append(make_mode(kind, m, n, j))
    return sorted(out)


def _cfactors(kind, cr, ci, r2):
    """Complex factors (re, im) multiplying Y in the three components of  c * mode,
    c = cr + i ci;  r2 = 1/sqrt2 (ball or float)."""
    if kind == "P":            # (c Y, -i c Y, 0)/sqrt2
        return [(cr * r2, ci * r2), (ci * r2, -(cr * r2)), None]
    if kind == "M":            # (c Y,  i c Y, 0)/sqrt2
        return [(cr * r2, ci * r2), (-(ci * r2), cr * r2), None]
    return [None, None, (cr, ci)]


def complex_field(modes, coeffs, evaluator: HarmonicEvaluator):
    """sum_k coeffs[k] * mode_k as a complex field on the evaluator's grid.

    coeffs: complex numbers (their real and imaginary parts are exact doubles).  Returns
    (val, grad) with val = [(re, im) for c in 0..2] and grad[a][c] = (re, im) the tangential
    partial d_a of component c (any backend)."""
    rig = evaluator.rig
    shape = evaluator.g.shape
    z = lambda: ar.zeros(shape, rig)
    val = [[z(), z()] for _ in range(3)]
    grad = [[[z(), z()] for _ in range(3)] for _ in range(4)]
    r2 = Ball.from_fraction(Fraction(1, 2)).sqrt() if rig else float(np.sqrt(0.5))
    for md, c in zip(modes, coeffs):
        cr, ci = float(np.real(c)), float(np.imag(c))
        if rig:
            cr, ci = Ball.exact(cr), Ball.exact(ci)
        (Yr, Yi), g = evaluator.complex_harmonic(md)
        gr = evaluator.tangential([p[0] for p in g])
        gi = evaluator.tangential([p[1] for p in g])
        for comp, fac in enumerate(_cfactors(md.kind, cr, ci, r2)):
            if fac is None:
                continue
            fr, fi = fac                                   # (fr + i fi)(Yr + i Yi)
            val[comp][0] = val[comp][0] + fr * Yr - fi * Yi
            val[comp][1] = val[comp][1] + fr * Yi + fi * Yr
            for a in range(4):
                grad[a][comp][0] = grad[a][comp][0] + fr * gr[a] - fi * gi[a]
                grad[a][comp][1] = grad[a][comp][1] + fr * gi[a] + fi * gr[a]
    return val, grad


def real_part(val, grad):
    """The real field Re f: components val[c] and partials grad[a][c] as real arrays."""
    return [v[0] for v in val], [[g[0] for g in row] for row in grad]


# ---------------------------------------------------------------------------- coefficient files
def save_index_basis(path, fields, meta=None):
    """fields: list of dicts {name, nu, parity, modes, coeffs (complex), diagnostics}."""
    data = {"description": "test fields xi for the index lower bound (Proposition 3.2 and Appendix C): each field is "
                           "sum_k c_k Phi_k over the modes P_{m,n,j} = (Y,-iY,0)/sqrt2, M_{m,n,j} = (Y,iY,0)/sqrt2, "
                           "T_{m,n,j} = (0,0,Y) of its sector S_{nu,parity}; c_k = re + i im with exact doubles. "
                           "For nu = 0 the test field is Re(sum_k c_k Phi_k); for nu != 0 the two real test fields "
                           "are the real and imaginary parts.",
            "fields": [{"name": f["name"], "nu": int(f["nu"]), "parity": int(f["parity"]),
                        "degree": max(m.l for m in f["modes"]),
                        "diagnostics": f.get("diagnostics", {}),
                        "modes": [{"kind": m.kind, "m": m.m, "n": m.n, "j": m.j, "l": m.l,
                                   "re": float(np.real(c)), "im": float(np.imag(c))}
                                  for m, c in zip(f["modes"], f["coeffs"])]}
                       for f in fields]}
    if meta:
        data["meta"] = meta
    Path(path).write_text(json.dumps(data, indent=1))


def load_index_basis(path):
    """Read and validate a basis file: every mode must lie in the field's sector, the modes of a
    field must be distinct and the coefficients finite."""
    data = json.loads(Path(path).read_text())
    fields = []
    for f in data["fields"]:
        nu, p = int(f["nu"]), int(f["parity"])
        modes, coeffs = [], []
        for c in f["modes"]:
            md = make_mode(c["kind"], int(c["m"]), int(c["n"]), int(c["j"]))
            assert md.l == int(c["l"]), f"wrong degree in {c}"
            assert md.nu == nu and md.parity == p, f"mode {c} is not in the sector ({nu}, {p})"
            re, im = float(c["re"]), float(c["im"])
            assert np.isfinite(re) and np.isfinite(im)
            modes.append(md)
            coeffs.append(complex(re, im))
        assert len(set(modes)) == len(modes), "duplicate modes"
        assert max(m.l for m in modes) == int(f["degree"]), "header degree does not match the modes"
        fields.append({"name": f["name"], "nu": nu, "parity": p, "modes": modes,
                       "coeffs": np.array(coeffs, dtype=complex), "diagnostics": f.get("diagnostics", {})})
    return fields, data
