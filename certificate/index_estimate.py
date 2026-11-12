"""Theorem 1.3 (q = 2): nine certified negative directions of the second variation (index >= 9),
through Propositions 2.19, 2.20, 3.2 and Lemma 2.21.

For W nowhere zero and xi in H^1(S^3, R^3) the paper's form (eq. quadraticform) is

    Q_W(xi) = int |d xi - (W x dW) x xi / |W|^2|^2 - |(W x dW) . xi|^2 / |W|^4 + (W . xi)^2 / |W|^2,

with Ind Q_W = Ind Q_u for u = W/|W| harmonic.  With omega_a = Wbar x d_a Wbar, A xi = omega x xi,
B xi = omega . xi and rho = |Wbar|^2, its symmetric bilinear form at Wbar is

    Q_Wbar(xi, eta) = int <d xi, d eta> - (<d xi, A eta> + <A xi, d eta>)/rho
                        + (<A xi, A eta> - (B xi)(B eta))/rho^2 + (Wbar . xi)(Wbar . eta)/rho.

**Polynomial proxy.**  Writing 1/rho = 1 + e_1 and 1/rho^2 = 1 + e_2,  Q_Wbar = Qt + E, where Qt
is the same expression with rho = 1 (all integrands are polynomials, integrated exactly by the
orbit-space quadrature) and, by Cauchy-Schwarz,

    |E(xi, eta)| <= kappa_1 (||d xi|| ||A eta|| + ||A xi|| ||d eta|| + ||Wbar.xi|| ||Wbar.eta||)
                  + kappa_2 (||A xi|| ||A eta|| + ||B xi|| ||B eta||),
    kappa_1 = sup |1 - rho| / min rho >= sup |e_1|,   kappa_2 = sup |1 - rho^2| / min rho^2 >= sup |e_2|,

with L^2 norms that are themselves exact polynomial integrals (sup |rho - 1| is a Chebyshev-
Fourier bound).  The same holds for the sesquilinear extension Q^h(f, g) = Q(conj f, g).

**Test fields.**  The file data/index_basis_q2.json (numerics/index_basis.py) holds, for the
sectors (0,-), (1,-), (2,+), (2,-), (3,-) of eells/sectors.py, one polynomial field f each; the
real test fields are xi = f (real, nu = 0) and Re f, Im f (nu != 0), nine in all.  By the
invariance of Q_Wbar and of the H^1 product under G, different sectors are orthogonal, and for
nu != 0 the real 2 x 2 block of (Re f, Im f) is (1/2) Q^h(f, f) I_2 (eells/sectors.py).  Hence
the matrices G_ab = Q_Wbar(xi_a, xi_b) and H_ab = <xi_a, xi_b>_{H^1} of Lemma 2.21 are
diagonal; they are nevertheless assembled as 9 x 9 ball matrices and the test of the lemma,
G + delta H negative definite, is made with a general routine (Sylvester's law of inertia:
V^T (G + delta H) V has negative Gershgorin discs for a float matrix V).

**Perturbation.**  For the harmonic map W = Wbar + v, ||v||_X <= r, Proposition 2.20
gives |Q_W(xi) - Q_Wbar(xi)| <= delta ||xi||_{H^1}^2 with

    delta = r [ 2 beta ( 3 c_inf c_4 (||Wbar||_X + r) / C^2 + c_4 / C )
                + 12 beta^2 c_inf c_4^2 (||Wbar||_X + r)^2 / C^3
                + 4 beta^2 c_4^2 (||Wbar||_X + r) / C^2 + 4 c_inf / C ],   beta = (2 pi^2)^{-1/4},

C = C_energyparameter = min |Wbar| - c_inf r (a lower bound of min |W|; the variable C8 below), and
beta the constant
of ||xi||_{L^4} <= beta ||xi||_{H^1} on S^3.  The lemma then gives: if G + delta H is negative
definite, the nine fields span a negative-definite subspace of Q_W, so Ind Q_u >= 9
(Proposition 2.19).

Besides the literal check the certificate reports the largest t (a decimal) with G + t H
negative definite, i.e. Q_Wbar(xi) <= -t ||xi||_{H^1}^2 on the span; lambda^* = -t is the bound
of Proposition 3.2, and lambda^* + delta < 0.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np

from eells import arith as ar
from eells.ball import Ball, matmul
from eells.equation import dot, cross
from eells.reference import sobolev_constants
from eells.sectors import complex_field, real_part, load_index_basis
from .rounding import round_down


# ---------------------------------------------------------------------------- constants
def perturbation_constant(r_str, nX, C8):
    """delta of Proposition 2.20 (ball arithmetic; nX and C8 = C_energyparameter are the certified bounds)."""
    c_inf, c_4, _, beta = sobolev_constants(True)
    rb = Ball.from_fraction(Fraction(r_str))
    n = Ball.exact(nX) + rb
    C = Ball.exact(C8)
    C2, C3 = C * C, C * C * C
    term1 = 2.0 * beta * (3.0 * c_inf * c_4 * n / C2 + c_4 / C)
    term2 = 12.0 * beta * beta * c_inf * c_4 * c_4 * n * n / C3
    term3 = 4.0 * beta * beta * c_4 * c_4 * n / C2
    term4 = 4.0 * c_inf / C
    delta = rb * (term1 + term2 + term3 + term4)
    return {"delta": float(delta.upper()), "beta": float(beta.upper()), "r": r_str, "norm_X": nX, "C_energyparameter": C8,
            "terms": [float(t.upper()) for t in (term1, term2, term3, term4)]}, delta


def rho_bounds(W_bound, grid_bound, D):
    """kappa_1 >= sup |1/rho - 1|, kappa_2 >= sup |1/rho^2 - 1| for rho = |Wbar|^2 (degree 2D)."""
    rho = dot(W_bound.val, W_bound.val)
    lo, hi = grid_bound.analyse(rho - 1.0, 2 * D).bounds()
    s = Ball.exact(max(abs(lo), abs(hi)))                 # sup |rho - 1|
    assert float(s.upper()) < 1
    one_minus = 1.0 - s                                   # <= min rho
    kappa1 = s / one_minus
    kappa2 = s * (2.0 + s) / (one_minus * one_minus)      # |1 - rho^2| = |1 - rho| (1 + rho) <= s (2 + s)
    return kappa1, kappa2, float(s.upper())


# ---------------------------------------------------------------------------- complex helpers
# a complex array is a pair (re, im); a complex 3-vector a list of three pairs
def _cross_real(omega, f):
    """omega x f for a real 3-vector omega and a complex 3-vector f."""
    re = cross(omega, [p[0] for p in f])
    im = cross(omega, [p[1] for p in f])
    return [(re[c], im[c]) for c in range(3)]


def _dot_real(omega, f):
    return dot(omega, [p[0] for p in f]), dot(omega, [p[1] for p in f])


def _hprod(u, v, acc):
    """acc += conj(u) . v for complex 3-vectors (pointwise)."""
    re, im = acc
    for (ur, ui), (vr, vi) in zip(u, v):
        re = re + ur * vr + ui * vi
        im = im + ur * vi - ui * vr
    return re, im


def _hprod1(u, v, acc):
    re, im = acc
    (ur, ui), (vr, vi) = u, v
    return re + ur * vr + ui * vi, im + ur * vi - ui * vr


class FieldQuantities:
    """Pointwise quantities of one complex field: val, grad, A f (per a), B f (per a), Wbar . f."""

    def __init__(self, val, grad, Wv, omega):
        self.val, self.grad = val, grad
        self.A = [_cross_real(omega[a], val) for a in range(4)]
        self.B = [_dot_real(omega[a], val) for a in range(4)]
        self.S = _dot_real(Wv, val)


def pair_integrals(quad, f: FieldQuantities, g: FieldQuantities, degs):
    """The complex integrals T1 = <df,dg>, T2 = <df,Ag> + <Af,dg>, T3 = <Af,Ag>, T4 = (Bf,Bg),
    T5 = (Wbar.f, Wbar.g), M = (f, g) (sesquilinear, conj on f), each as a pair of balls."""
    zero = (ar.zeros(quad.shape, quad.rigorous), ar.zeros(quad.shape, quad.rigorous))
    T1, T2, T3, T4 = zero, zero, zero, zero
    for a in range(4):
        T1 = _hprod(f.grad[a], g.grad[a], T1)
        T2 = _hprod(f.grad[a], g.A[a], _hprod(f.A[a], g.grad[a], T2))
        T3 = _hprod(f.A[a], g.A[a], T3)
        T4 = _hprod1(f.B[a], g.B[a], T4)
    T5 = _hprod1(f.S, g.S, zero)
    M = _hprod(f.val, g.val, zero)
    I = lambda pair, d: (quad.integrate(pair[0], d), quad.integrate(pair[1], d))
    return {"T1": I(T1, degs["T1"]), "T2": I(T2, degs["T2"]), "T3": I(T3, degs["T3"]),
            "T4": I(T4, degs["T4"]), "T5": I(T5, degs["T5"]), "M": I(M, degs["M"])}


def integrand_degrees(D, L):
    return {"T1": 2 * L + 2, "T2": 2 * D + 2 * L + 2, "T3": 4 * D + 2 + 2 * L, "T4": 4 * D + 2 + 2 * L,
            "T5": 2 * D + 2 * L, "M": 2 * L, "max": 4 * D + 2 + 2 * L}


def _norms(diag):
    """L^2 norms ||df||, ||Af||, ||Bf||, ||Wbar.f|| from the diagonal integrals (real parts)."""
    return {k: diag[k][0].sqrt(nonneg=True) for k in ("T1", "T3", "T4", "T5")}


def enclose_Q(ints, nf, ng, kappa1, kappa2):
    """(Q^h(f,g), H^h(f,g)) as pairs of balls, Q = Qt +- |E|."""
    T1, T2, T3, T4, T5, M = (ints[k] for k in ("T1", "T2", "T3", "T4", "T5", "M"))
    Qt = (T1[0] - T2[0] + T3[0] - T4[0] + T5[0], T1[1] - T2[1] + T3[1] - T4[1] + T5[1])
    err = kappa1 * (nf["T1"] * ng["T3"] + nf["T3"] * ng["T1"] + nf["T5"] * ng["T5"]) \
        + kappa2 * (nf["T3"] * ng["T3"] + nf["T4"] * ng["T4"])
    e = float(err.upper())
    Q = tuple(Ball.from_bounds(float(q.lower()) - e, float(q.upper()) + e) for q in Qt)
    H = (M[0] + T1[0], M[1] + T1[1])
    return Q, H, Qt, e


# ---------------------------------------------------------------------------- negative definiteness
def negative_definite(M: Ball):
    """Rigorous test that every symmetric matrix in the ball matrix M is negative definite:
    S = V^T M V for the float eigenvector matrix V of mid(M); if all Gershgorin discs of S lie in
    (-inf, 0) then S, hence M (Sylvester's law of inertia; V is then invertible), is negative
    definite.  Returns (ok, largest Gershgorin upper bound of S)."""
    _, V = np.linalg.eigh(0.5 * (M.m + M.m.T))
    Vb = Ball.exact(V)
    S = matmul(matmul(Vb.T, M), Vb)
    k = S.m.shape[0]
    bounds = []
    for i in range(k):
        off = sum(float(S[i, j].absmax()) for j in range(k) if j != i)
        bounds.append(float((S[i, i] + Ball.exact(off)).upper()))
    return all(b < 0 for b in bounds), max(bounds)


# ---------------------------------------------------------------------------- main routine
def index_lower_bound(Wq, quad, Hq, Wb, bound, D, basis_path, r_str, nX, C8, log=print):
    """The index certificate (Theorem 1.3): returns a dict with G, H (9 x 9 balls), delta, the checks."""
    t0 = time.time()
    fields, data = load_index_basis(basis_path)
    L = max(max(m.l for m in f["modes"]) for f in fields)
    degs = integrand_degrees(D, L)
    assert quad.exact_for(degs["max"]), f"quadrature grid is not exact for degree {degs['max']}"
    kappa1, kappa2, sup_rho = rho_bounds(Wb, bound, D)
    log(f"[5]   {len(fields)} complex test fields of degree <= {L} from {basis_path};  sup||Wbar|^2 - 1| <= {sup_rho:.3e},  "
        f"kappa_1 <= {float(kappa1.upper()):.3e},  kappa_2 <= {float(kappa2.upper()):.3e}")
    pert, delta_b = perturbation_constant(r_str, nX, C8)
    delta = pert["delta"]
    log(f"[5]   beta <= {pert['beta']:.6f},  delta <= {delta:.4e}   (terms {', '.join(f'{t:.3e}' for t in pert['terms'])})")
    Wv, dW = Wq.val, Wq.grad
    omega = [cross(Wv, dW[a]) for a in range(4)]
    # group the fields by sector
    sectors = {}
    for f in fields:
        sectors.setdefault((f["nu"], f["parity"]), []).append(f)
    real_names, blocks, per_field = [], [], []
    for (nu, p), fs in sorted(sectors.items()):
        qs = []
        for f in fs:
            val, grad = complex_field(f["modes"], f["coeffs"], Hq)
            if nu == 0:                                   # the real test field xi = Re f
                vr, gr = real_part(val, grad)
                z = ar.zeros(quad.shape, True)
                val = [(v, z) for v in vr]
                grad = [[(g, z) for g in row] for row in gr]
            qs.append(FieldQuantities(val, grad, Wv, omega))
        k = len(fs)
        diag = [pair_integrals(quad, q, q, degs) for q in qs]
        norms = [_norms(d) for d in diag]
        Qh = [[None] * k for _ in range(k)]
        Hh = [[None] * k for _ in range(k)]
        for i in range(k):
            for j in range(i, k):
                ints = diag[i] if i == j else pair_integrals(quad, qs[i], qs[j], degs)
                Q, H, Qt, e = enclose_Q(ints, norms[i], norms[j], kappa1, kappa2)
                Qh[i][j], Hh[i][j] = Q, H
                if i != j:                                # Hermitian: conj
                    Qh[j][i] = (Q[0], -Q[1])
                    Hh[j][i] = (H[0], -H[1])
                if i == j:
                    f = fs[i]
                    # orthonormality + Green: ||f||_{H^1}^2 = sum (1 + l(l+2)) |c_k|^2 (f real for nu = 0)
                    h_coef = sum((1.0 + m.l * (m.l + 2)) * abs(c) ** 2 for m, c in zip(f["modes"], f["coeffs"]))
                    rq = float((Q[0] / H[0]).upper())
                    per_field.append({"name": f["name"], "nu": nu, "parity": p, "degree": max(m.l for m in f["modes"]),
                                      "Q_mid": float(Qt[0].m), "Q_radius": float(Qt[0].r), "proxy_error": e,
                                      "H1_mid": float(H[0].m), "H1_from_coefficients": h_coef,
                                      "L2_mid": float(ints["M"][0].m),
                                      "L2_rayleigh": float((Q[0] / ints["M"][0]).m), "H1_rayleigh_upper": rq,
                                      "diagnostics": f.get("diagnostics", {})})
                    log(f"      {f['name']} ({nu:d},{'+' if p > 0 else '-'}): Q = {float(Qt[0].m):.9f} (radius {float(Qt[0].r):.1e}, proxy error {e:.1e}),  "
                        f"||xi||_H1^2 = {float(H[0].m):.9f} (from coefficients {h_coef:.9f}),  "
                        f"Rayleigh L2 {float((Q[0] / ints['M'][0]).m):.6f},  H1 <= {rq:.6f}  ({time.time() - t0:.0f}s)")
        # real blocks
        if nu == 0:
            Gb = Ball(np.array([[Qh[i][j][0].m for j in range(k)] for i in range(k)]),
                      np.array([[Qh[i][j][0].r for j in range(k)] for i in range(k)]))
            Hb = Ball(np.array([[Hh[i][j][0].m for j in range(k)] for i in range(k)]),
                      np.array([[Hh[i][j][0].r for j in range(k)] for i in range(k)]))
            names = [f["name"] for f in fs]
        else:
            def realify(Mh):
                A = Ball(np.array([[Mh[i][j][0].m for j in range(k)] for i in range(k)]),
                         np.array([[Mh[i][j][0].r for j in range(k)] for i in range(k)]))
                B = Ball(np.array([[Mh[i][j][1].m for j in range(k)] for i in range(k)]),
                         np.array([[Mh[i][j][1].r for j in range(k)] for i in range(k)]))
                top = Ball(np.concatenate([A.m, B.m], axis=1), np.concatenate([A.r, B.r], axis=1))
                bot = Ball(np.concatenate([-B.m, A.m], axis=1), np.concatenate([B.r, A.r], axis=1))
                return Ball(np.concatenate([top.m, bot.m], axis=0), np.concatenate([top.r, bot.r], axis=0)) * 0.5
            Gb, Hb = realify(Qh), realify(Hh)
            names = [f"Re {f['name']}" for f in fs] + [f"Im {f['name']}" for f in fs]
        blocks.append((Gb, Hb))
        real_names += names
    n = len(real_names)
    Gm, Gr, Hm, Hr = (np.zeros((n, n)) for _ in range(4))
    pos = 0
    for Gb, Hb in blocks:
        k = Gb.m.shape[0]
        Gm[pos:pos + k, pos:pos + k], Gr[pos:pos + k, pos:pos + k] = Gb.m, Gb.r
        Hm[pos:pos + k, pos:pos + k], Hr[pos:pos + k, pos:pos + k] = Hb.m, Hb.r
        pos += k
    G, H = Ball(Gm, Gr), Ball(Hm, Hr)
    ok_lemma, g_lemma = negative_definite(G + H * delta_b)
    # the largest decimal t with G + t H negative definite (from the diagonal quotients)
    quot = [float((-(G[i, i] / H[i, i])).lower()) for i in range(n)]
    t = round_down(min(quot), 4)
    ok_t, g_t = negative_definite(G + H * Ball.from_fraction(Fraction(t)))
    if not ok_t:
        t = round_down(0.99 * min(quot), 4)
        ok_t, g_t = negative_definite(G + H * Ball.from_fraction(Fraction(t)))
    h_pos, _ = negative_definite(-H)
    log(f"[5]   real directions: {', '.join(real_names)}")
    log(f"[5]   G + delta H negative definite: {ok_lemma} (largest Gershgorin bound {g_lemma:.4e});  "
        f"G + t H negative definite for t = {t}: {ok_t};  H positive definite: {h_pos}  ({time.time() - t0:.0f}s)")
    return {"basis": str(basis_path), "max_degree": L, "n_real_directions": n, "real_directions": real_names,
            "sup_abs_rho_minus_1": sup_rho, "kappa_1": float(kappa1.upper()), "kappa_2": float(kappa2.upper()),
            "perturbation": pert, "fields": per_field,
            "G_mid": Gm.tolist(), "G_radius": Gr.tolist(), "H_mid": Hm.tolist(), "H_radius": Hr.tolist(),
            "diagonal_H1_quotients_lower": quot, "t": t,
            "conditions": {"G_plus_deltaH_negative_definite": ok_lemma, "G_plus_tH_negative_definite": ok_t,
                           "H_positive_definite": h_pos},
            "gershgorin_bound_lemma": g_lemma, "all_ok": ok_lemma and ok_t and h_pos, "seconds": time.time() - t0}
