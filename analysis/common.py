"""Shared tools for the figures: the maps u_q = Wbar_q / |Wbar_q| : S^3 -> S^2 (q = 1..6) read from
data/, evaluated anywhere on S^3, their fibres, the circle orbits, the energy density, the
stereographic projection S^3 -> R^3 and the Gauss linking number.

Conventions (as in eells/harmonics.py and eells/grid.py).  A point of S^3 is (z, w) in C^2.  The
circle of G_q acts by (e^{i th} z, e^{-i q th} w) on S^3 and by zeta -> e^{-2 i q th} zeta on the
target (zeta, t) in S^2 (zeta in C, t in R).  Every orbit meets the slice
{(sqrt s, sqrt(1-s) e^{i phi})} (z real >= 0) exactly once, so the orbit space S^3/S^1 is the
2-sphere with height x = 1 - 2s = |w|^2 - |z|^2 and longitude phi = arg(z^q w); in these coordinates
dvol_{S^3} = pi ds dphi, so equal areas on this sphere are equal volumes in S^3 (Archimedes).
G_q-invariant quantities (the height t of u_q, the energy density) are functions on it.
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eells.fields import field_from_coefficients, load_coefficients  # noqa: E402
from eells.grid import OrbitGrid  # noqa: E402
from eells.harmonics import HarmonicEvaluator, jacobi_table, norm_constant_squared_times_2pi2  # noqa: E402

QS = (1, 2, 3, 4, 5, 6)
FIG_DIR = ROOT / "analysis" / "figures"
CACHE_DIR = ROOT / "analysis" / "cache"


def coefficient_file(q):
    """The certified coefficient file data/Wbar_q<q>_D<D>.json (the largest D present)."""
    files = sorted(ROOT.glob(f"data/Wbar_q{q}_D*.json"), key=lambda p: int(p.stem.split("_D")[1]))
    if not files:
        raise FileNotFoundError(f"no data/Wbar_q{q}_D*.json")
    return files[-1]


class Solution:
    """Wbar_q = sum_k a_k Psi_k, evaluated at arbitrary points (plain floating point)."""

    def __init__(self, q):
        self.q = q
        self.path = coefficient_file(q)
        self.modes, self.values, _ = load_coefficients(self.path)
        assert self.modes[0].q == q
        self.D = max(m.l for m in self.modes)
        # group the modes by (kind, n): they share a = |m|, b = |n| and differ in the Jacobi index j
        self.groups = {}
        for m, v in zip(self.modes, self.values):
            c2 = float(norm_constant_squared_times_2pi2(m.a, m.b, m.j)) * (2 if m.kind == "V" else 1)
            g = self.groups.setdefault((m.kind, m.n), (m.a, m.b, [], []))
            g[2].append(m.j)
            g[3].append(v * np.sqrt(c2 / (2 * np.pi ** 2)))

    # ---- values
    def slice_W(self, s, phi):
        """Wbar at the slice points (sqrt s, sqrt(1-s) e^{i phi}); returns shape (3,) + s.shape."""
        s = np.asarray(s, dtype=float)
        phi = np.broadcast_to(np.asarray(phi, dtype=float), s.shape)
        x = 1 - 2 * s
        rs, rc = np.sqrt(np.clip(s, 0, 1)), np.sqrt(np.clip(1 - s, 0, 1))
        W = np.zeros((3,) + s.shape)
        for (kind, n), (a, b, js, cs) in self.groups.items():
            P = jacobi_table(a, b, max(js), x)
            R = sum(c * P[j] for j, c in zip(js, cs)) * rs ** a * rc ** b
            if kind == "U":
                W[0] += R * np.cos(n * phi)
                W[1] += R * np.sin(n * phi)
            else:
                W[2] += R * np.cos(n * phi)
        return W

    def W(self, z, w):
        """Wbar(z, w) at arbitrary points of S^3 via the slice and the equivariance:
        (z, w) = e^{i th}.(|z|, e^{i q th} w) with th = arg z, so phi = q arg z + arg w."""
        z, w = np.asarray(z, dtype=complex), np.asarray(w, dtype=complex)
        th = np.angle(z)
        Ws = self.slice_W(np.abs(z) ** 2, self.q * th + np.angle(w))
        zeta = (Ws[0] + 1j * Ws[1]) * np.exp(-2j * self.q * th)
        return np.stack([zeta.real, zeta.imag, Ws[2]])

    def u(self, z, w):
        W = self.W(z, w)
        return W / np.linalg.norm(W, axis=0)

    def direct_W(self, z, w):
        """Wbar(z, w) from the definition Y_{m,n,j} = c z^(m) w^(n) P_j(1 - 2|z|^2), without the
        slice (a check of :meth:`W`)."""
        z, w = np.asarray(z, dtype=complex), np.asarray(w, dtype=complex)
        x = 1 - 2 * np.abs(z) ** 2
        out = np.zeros((3,) + z.shape)
        for m, v in zip(self.modes, self.values):
            c = np.sqrt(float(norm_constant_squared_times_2pi2(m.a, m.b, m.j)) / (2 * np.pi ** 2))
            zm = z ** m.m if m.m >= 0 else np.conj(z) ** (-m.m)
            wn = w ** m.n if m.n >= 0 else np.conj(w) ** (-m.n)
            Y = c * zm * wn * jacobi_table(m.a, m.b, m.j, x)[m.j]
            if m.kind == "U":
                out[0] += v * Y.real
                out[1] += v * Y.imag
            else:
                out[2] += v * np.sqrt(2) * Y.real
        return out


# ---------------------------------------------------------------------------- orbit-space data
def orbit_data(sol, Ns=240, Nphi=480, log=print):
    """Wbar and the energy density e = |du|^2 / 2 of u = Wbar/|Wbar| on the orbit-space grid
    x_i = cos(pi i / Ns) (uniform in latitude on the orbit sphere), phi_b = 2 pi b / Nphi, with the
    Clenshaw-Curtis x trapezoid weights omega (sum omega = 2 pi^2).  Cached in analysis/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fn = CACHE_DIR / f"orbit_q{sol.q}_D{sol.D}_{Ns}x{Nphi}.npz"
    if fn.exists():
        d = np.load(fn)
        return {k: d[k] for k in d.files}
    g = OrbitGrid(Ns, Nphi, False, sol.q)
    H = HarmonicEvaluator(g, max_degree=sol.D)
    W = np.zeros((3, Ns + 1, Nphi))
    e = np.zeros((Ns + 1, Nphi))
    for rows in g.row_chunks(6000):
        F = field_from_coefficients(sol.modes, sol.values, H, batch=32, rows=rows)
        V = np.stack(F.val)
        rho = (V ** 2).sum(axis=0)
        N = sum(rho * (np.stack(F.grad[a]) ** 2).sum(axis=0) - ((V * np.stack(F.grad[a])).sum(axis=0)) ** 2
                for a in range(4))
        W[:, rows] = V
        e[rows] = 0.5 * N / rho ** 2
    out = {"x": g.x, "phi": 2 * np.pi * np.arange(Nphi) / Nphi, "W": W, "e": e, "omega": g.omega}
    np.savez_compressed(fn, **out)
    log(f"  q={sol.q}: orbit grid {Ns + 1} x {Nphi} evaluated, cached in {fn.relative_to(ROOT)}")
    return out


def height_and_zeta(W):
    n = np.linalg.norm(W, axis=0)
    return W[2] / n, (W[0] + 1j * W[1]) / n


# ---------------------------------------------------------------------------- pole fibres and orbits
def pole_point(sol):
    """s* in (0, 1) with zeta(s*, phi = 0) = 0, and the sign of t there.  On phi = 0 the map is
    kappa-symmetric, so zeta is real; the zeros of zeta are the orbits mapped to the poles: the
    orbit through (s*, 0) and its image under S (phi -> phi + pi, t -> -t)."""
    from scipy.optimize import brentq
    s = np.linspace(1e-9, 1 - 1e-9, 20000)
    W = sol.slice_W(s, 0 * s)
    f = W[0] / np.linalg.norm(W, axis=0)
    idx = np.nonzero(f[:-1] * f[1:] <= 0)[0]
    assert len(idx) >= 1 and idx[-1] - idx[0] <= 1, f"expected one zero of zeta on phi = 0, found {len(idx)}"
    g = lambda ss: sol.slice_W(np.array([ss]), np.array([0.0]))[0, 0]
    s_star = brentq(g, s[idx[0]], s[idx[-1] + 1], xtol=1e-15)
    Wp = sol.slice_W(np.array([s_star]), np.array([0.0]))[:, 0]
    return s_star, float(np.sign(Wp[2]))


def count_zeta_zeros(data):
    """Number of zeros of zeta on the orbit sphere (local minima of |zeta| below 0.05 on the grid)."""
    _, zeta = height_and_zeta(data["W"])
    a = np.abs(zeta)
    ext = np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)
    pad = np.concatenate([np.full((1, ext.shape[1]), np.inf), ext, np.full((1, ext.shape[1]), np.inf)])
    c = pad[1:-1, 1:-1]
    is_min = np.ones_like(c, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_min &= c <= pad[1 + di:pad.shape[0] - 1 + di, 1 + dj:pad.shape[1] - 1 + dj]
    return int((is_min & (c < 0.05)).sum())


def orbit(q, s, phi, n=1000, closed=True):
    """The circle orbit through the slice point (sqrt s, sqrt(1-s) e^{i phi}), as (z, w) arrays."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=not closed)
    return np.exp(1j * th) * np.sqrt(s), np.exp(-1j * q * th) * np.sqrt(1 - s) * np.exp(1j * phi)


# ---------------------------------------------------------------------------- general fibres
def level_loop(sol, data, t0, n=400, newton=4):
    """The level curve {t = t0} on the orbit sphere around the pole point (t0 != 0), as n points
    (s_i, phi_i) on the slice, refined by Newton steps on the exact height."""
    import contourpy
    from matplotlib.path import Path as MPath
    t, _ = height_and_zeta(data["W"])
    x, phi = data["x"], data["phi"]
    phi_c = 0.0 if t0 > 0 else np.pi
    rel = (phi - phi_c + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(rel)
    rel, T = rel[order], t[:, order]
    rel = np.concatenate([rel, rel[:1] + 2 * np.pi])
    T = np.concatenate([T, T[:, :1]], axis=1)
    gen = contourpy.contour_generator(x=rel, y=x[::-1], z=T[::-1])
    s_star, _ = pole_point(sol)
    centre = (0.0, 1 - 2 * s_star)
    loops = [L for L in gen.lines(t0) if len(L) > 8 and np.allclose(L[0], L[-1])
             and MPath(L).contains_point(centre)]
    assert loops, f"no closed level curve t = {t0} around the pole point"
    L = min(loops, key=len)[:-1]
    # resample uniformly in arc length (in the (phi, x) chart), then refine on the exact height
    seg = np.linalg.norm(np.diff(np.vstack([L, L[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, cum[-1], n, endpoint=False)
    Lc = np.vstack([L, L[:1]])
    P = np.stack([np.interp(tt, cum, Lc[:, 0]), np.interp(tt, cum, Lc[:, 1])], axis=1)
    ph, xx = P[:, 0] + phi_c, P[:, 1]
    h = 1e-7

    def height(xv, pv):
        Wv = sol.slice_W((1 - xv) / 2, pv)
        return Wv[2] / np.linalg.norm(Wv, axis=0)

    for _ in range(newton):
        f = height(xx, ph) - t0
        gx = (height(xx + h, ph) - height(xx - h, ph)) / (2 * h)
        gp = (height(xx, ph + h) - height(xx, ph - h)) / (2 * h)
        g2 = gx ** 2 + gp ** 2
        xx, ph = xx - f * gx / g2, ph - f * gp / g2
    err = np.abs(height(xx, ph) - t0).max()
    return (1 - xx) / 2, ph, err


def fibre(sol, data, t0, alpha0, n=400):
    """The fibre u_q^{-1}(y) over y = (sqrt(1-t0^2) e^{i alpha0}, t0), t0 != 0, as a list of closed
    curves [(z, w), ...].  On the level loop {t = t0} of the orbit sphere the point
    e^{i th}.(slice point) lies on the fibre iff  arg zeta(slice point) - 2 q th = alpha0  (mod 2 pi);
    following the 2q branches th = (arg zeta - alpha0)/(2q) + k pi/q around the loop closes them up
    into gcd(d, 2q) curves, d the winding number of zeta around the loop.  Also returns the largest
    |u(point) - y| over the curves (a check)."""
    q = sol.q
    s, ph, _ = level_loop(sol, data, t0, n)
    Wl = sol.slice_W(s, ph)
    A = np.unwrap(np.angle(Wl[0] + 1j * Wl[1]))
    total = A[-1] - A[0] + np.angle(np.exp(1j * (A[0] - A[-1])))     # change of arg zeta once round the loop
    d = int(round(total / (2 * np.pi)))
    g = gcd(abs(d), 2 * q) if d else 2 * q
    ntrav = 2 * q // g if d else 1
    curves = []
    for k in range(g):
        th = np.concatenate([(A + 2 * np.pi * d * r - alpha0) / (2 * q) + k * np.pi / q for r in range(ntrav)])
        ss, pp = np.tile(s, ntrav), np.tile(ph, ntrav)
        curves.append((np.exp(1j * th) * np.sqrt(ss), np.exp(-1j * q * th) * np.sqrt(1 - ss) * np.exp(1j * pp)))
    y = np.array([np.sqrt(1 - t0 ** 2) * np.cos(alpha0), np.sqrt(1 - t0 ** 2) * np.sin(alpha0), t0])
    err = max(np.abs(sol.u(z, w) - y.reshape(3, 1)).max() for z, w in curves)
    return curves, d, err


# ---------------------------------------------------------------------------- R^3 and linking
def stereo(z, w):
    """Stereographic projection of S^3 from (z, w) = (0, i) onto R^3: (Re z, Im z, Re w)/(1 - Im w).
    The circle z = 0 becomes the vertical axis and the tori |z| = const tori of revolution about it."""
    return np.stack([z.real, z.imag, w.real]) / (1 - w.imag)


def linking_number(P, Q):
    """Gauss linking integral of two closed polygons P, Q (shape (3, N), last point != first)."""
    dP = np.roll(P, -1, axis=1) - P
    dQ = np.roll(Q, -1, axis=1) - Q
    mP = P + dP / 2
    mQ = Q + dQ / 2
    r = mP[:, :, None] - mQ[:, None, :]
    cr = np.cross(dP.T[:, None, :], dQ.T[None, :, :])          # (N, M, 3)
    num = np.einsum("nmc,cnm->nm", cr, r)
    return float((num / np.linalg.norm(r, axis=0) ** 3).sum() / (4 * np.pi))


def selfcheck(sol, log=print):
    """Pointwise check of the slice evaluation against the direct formula, and |Wbar| ~ 1."""
    rng = np.random.default_rng(sol.q)
    X = rng.normal(size=(4, 200))
    X /= np.linalg.norm(X, axis=0)
    z, w = X[0] + 1j * X[1], X[2] + 1j * X[3]
    W1, W2 = sol.W(z, w), sol.direct_W(z, w)
    err = np.abs(W1 - W2).max()
    nrm = np.abs(np.linalg.norm(W1, axis=0) - 1).max()
    log(f"  q={sol.q} (D={sol.D}): slice vs direct evaluation {err:.1e}, max ||Wbar| - 1| = {nrm:.1e}")
    assert err < 1e-9
    return err
