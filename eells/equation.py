"""The equation F(v) = 0 of Section 2.2, its linearisation (eq. linearisation) and the
expanded coefficients used in the linear estimate of Section 3.2.

Notation: W = Wbar + v, Delta the (non-positive) Laplacian of S^3, d_a (a = 0..3) tangential
partial derivatives along the ambient axes, so that  sum_a d_a f d_a g = <grad f, grad g>  and
Delta(f g) = f Delta g + g Delta f + 2 sum_a d_a f d_a g.

    F(v) = chi [ (1/2) Wbar x ( W x (3|W|^2 Delta W - Delta(|W|^2 W)) )
                 + |Wbar|^2 (Wbar . (1 - Delta) v) Wbar ].

**Expanded linearisation.**  With  Q = 3|Wbar|^2 Delta Wbar - Delta(|Wbar|^2 Wbar)
  = 2|Wbar|^2 Delta Wbar - 2 (Wbar . Delta Wbar + |grad Wbar|^2) Wbar - 4 sum_a (Wbar . d_a Wbar) d_a Wbar
and  Pi = |Wbar|^2 I - Wbar Wbar^T  (so that Wbar x (Wbar x v) = -Pi v,  Pi Wbar = 0), the
linearisation DF(0) h of eq. (linearisation) reads

    L h = -P Delta h + sum_a M_a d_a h + M_0 h,

    P   = chi |Wbar|^4,
    M_a = 2 chi [ (Pi d_a Wbar) Wbar^T + (Wbar . d_a Wbar) Pi ],
    M_0 = (chi/2) [ (Wbar . Q) I - Q Wbar^T ]
          - chi Pi [ 2 Delta Wbar Wbar^T - 2 sum_a d_a Wbar d_a Wbar^T - (Wbar . Delta Wbar + |grad Wbar|^2) I ]
          + chi |Wbar|^2 Wbar Wbar^T.

(Derivation: expand the Laplacians of products in (linearisation) by the product rule; the
second-order terms combine to  -chi (Pi^2 + |Wbar|^2 Wbar Wbar^T) Delta h = -chi |Wbar|^4 Delta h.)
The residual is  F(0) = (chi/2) Wbar x (Wbar x Q) = -(chi/2) Pi Q.

With R = L + Delta the first- and zeroth-order coefficient matrices are M_a and M_0 and the
second-order coefficient is (P - 1)(-Delta); eta, b_1, b_0 of Section 3.2 bound
sup |P - 1|,  sup sum_a ||M_a||_F^2,  sup ||M_0||_F^2.

Polynomial degrees (Wbar of degree D, chi of degree D_chi), see :func:`degrees`.
"""
from __future__ import annotations

from . import arith as ar
from .fields import Field


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def degrees(D, Dchi):
    """Polynomial degrees of the quantities built from Wbar (degree D) and chi (degree Dchi)."""
    return {"W": D, "dW": D + 1, "LW": D, "chi": Dchi,
            "rho": 2 * D,                     # |Wbar|^2
            "Q": 3 * D + 2,
            "F0": Dchi + 5 * D + 2,           # residual F(0)
            "P": Dchi + 4 * D,
            "M": Dchi + 4 * D + 1,            # each M_a
            "M0": Dchi + 4 * D + 2,
            "Kp": Dchi + 4 * D + 2,           # L Psi has degree l + Kp as evaluated here
            "K": Dchi + 4 * D,                # band width of the paper (K = 4 deg Wbar + deg chi)
            "energy": 4 * D + 2,              # sum_a |Wbar x d_a Wbar|^2
            "V0": 4}


class OperatorCoefficients:
    """P, M_a (a = 0..3), M_0 of the expanded linearisation at Wbar, on the grid."""

    def __init__(self, W: Field, chi):
        Wv, dW, LW = W.val, W.grad, W.lap
        rho = dot(Wv, Wv)
        WdW = [dot(Wv, dW[a]) for a in range(4)]
        gradsq = ar.sq(dW[0][0]) * 0.0
        for a in range(4):
            for c in range(3):
                gradsq = gradsq + ar.sq(dW[a][c])
        WLW = dot(Wv, LW)
        Pi = [[(rho if c == d else 0.0) - Wv[c] * Wv[d] for d in range(3)] for c in range(3)]
        Q = [LW[c] * (rho * 2.0) - Wv[c] * ((WLW + gradsq) * 2.0) for c in range(3)]
        for a in range(4):
            for c in range(3):
                Q[c] = Q[c] - dW[a][c] * (WdW[a] * 4.0)
        WQ = dot(Wv, Q)
        self.rho, self.gradsq, self.WLW, self.Pi, self.Q, self.WQ = rho, gradsq, WLW, Pi, Q, WQ
        self.val_W = Wv
        self.chi = chi
        self.P = chi * rho * rho
        # M_a = 2 chi [ (Pi d_a Wbar) Wbar^T + (Wbar . d_a Wbar) Pi ]
        self.M = []
        for a in range(4):
            PidW = [Pi[c][0] * dW[a][0] + Pi[c][1] * dW[a][1] + Pi[c][2] * dW[a][2] for c in range(3)]
            self.M.append([[(PidW[c] * Wv[d] + Pi[c][d] * WdW[a]) * (chi * 2.0) for d in range(3)] for c in range(3)])
        # M_0
        S = [[LW[c] * Wv[d] * 2.0 - ((WLW + gradsq) if c == d else 0.0) for d in range(3)] for c in range(3)]
        for a in range(4):
            for c in range(3):
                for d in range(3):
                    S[c][d] = S[c][d] - dW[a][c] * dW[a][d] * 2.0
        self.M0 = []
        for c in range(3):
            row = []
            for d in range(3):
                PiS = Pi[c][0] * S[0][d] + Pi[c][1] * S[1][d] + Pi[c][2] * S[2][d]
                val = ((WQ if c == d else 0.0) - Q[c] * Wv[d]) * (chi * 0.5) - PiS * chi + Wv[c] * Wv[d] * (chi * rho)
                row.append(val)
            self.M0.append(row)

    def residual(self):
        """F(0) = -(chi/2) Pi Q."""
        return [-(self.Pi[c][0] * self.Q[0] + self.Pi[c][1] * self.Q[1] + self.Pi[c][2] * self.Q[2]) * (self.chi * 0.5)
                for c in range(3)]

    def apply(self, h: Field):
        """L h = -P Delta h + sum_a M_a d_a h + M_0 h."""
        out = []
        for c in range(3):
            v = -(self.P * h.lap[c])
            for d in range(3):
                v = v + self.M0[c][d] * h.val[d]
                for a in range(4):
                    v = v + self.M[a][c][d] * h.grad[a][d]
            out.append(v)
        return out

    def apply_batch(self, val, grad, lap):
        """L applied to a stack of fields: val [k, c, i, b], grad [k, a, c, i, b], lap [k, c, i, b]."""
        out = []
        for c in range(3):
            v = -(lap[:, c] * self.P)
            for d in range(3):
                v = v + val[:, d] * self.M0[c][d]
                for a in range(4):
                    v = v + grad[:, a, d] * self.M[a][c][d]
            out.append(v)
        return ar.stack(out, axis=1)

    def outer_cross_batch(self, val):
        """(chi/2) h x (W x Q) for a stack of fields h: the derivative of the residual
        (chi/2) Wbar x (Wbar x Q) with respect to its *outer* factor Wbar (needed when Wbar itself is
        the unknown, as in the solver; DF(0) keeps that factor fixed)."""
        Wv, Q = self.val_W, self.Q
        WxQ = cross(Wv, Q)
        return ar.stack([(val[:, 1] * WxQ[2] - val[:, 2] * WxQ[1]) * (self.chi * 0.5),
                         (val[:, 2] * WxQ[0] - val[:, 0] * WxQ[2]) * (self.chi * 0.5),
                         (val[:, 0] * WxQ[1] - val[:, 1] * WxQ[0]) * (self.chi * 0.5)], axis=1)

    def gauge_batch(self, val, lap):
        """The gauge term  chi |Wbar|^2 (Wbar . (1 - Delta) h) Wbar  for a stack of fields h."""
        Wv = self.val_W
        inner = val[:, 0] * Wv[0] + val[:, 1] * Wv[1] + val[:, 2] * Wv[2] \
            - (lap[:, 0] * Wv[0] + lap[:, 1] * Wv[1] + lap[:, 2] * Wv[2])
        inner = inner * (self.chi * self.rho)
        return ar.stack([inner * Wv[c] for c in range(3)], axis=1)

    def frobenius_squares(self):
        """(sum_a ||M_a||_F^2,  ||M_0||_F^2)  pointwise (degrees 2(Dchi+4D+1), 2(Dchi+4D+2))."""
        m1 = ar.sq(self.M[0][0][0]) * 0.0
        m0 = ar.sq(self.M0[0][0]) * 0.0
        for c in range(3):
            for d in range(3):
                m0 = m0 + ar.sq(self.M0[c][d])
                for a in range(4):
                    m1 = m1 + ar.sq(self.M[a][c][d])
        return m1, m0


# ---------------------------------------------------------------------------- direct forms (tests)
def laplacian_of_product(f, df, Lf, g, dg, Lg):
    """Delta(f g) = f Delta g + g Delta f + 2 sum_a d_a f d_a g for scalars."""
    out = f * Lg + g * Lf
    for a in range(4):
        out = out + df[a] * dg[a] * 2.0
    return out


def F_direct(W: Field, chi, v: Field = None):
    """F(v) straight from its definition (eq. F-equation), with W = Wbar + v."""
    Wt = W if v is None else W + v
    rho = dot(Wt.val, Wt.val)
    drho = [dot(Wt.val, Wt.grad[a]) * 2.0 for a in range(4)]
    Lrho = dot(Wt.val, Wt.lap) * 2.0
    for a in range(4):
        Lrho = Lrho + dot(Wt.grad[a], Wt.grad[a]) * 2.0
    # Delta(|W|^2 W)_c = |W|^2 Delta W_c + Delta|W|^2 W_c + 2 sum_a d_a|W|^2 d_a W_c
    LrhoW = [laplacian_of_product(rho, drho, Lrho, Wt.val[c], [Wt.grad[a][c] for a in range(4)], Wt.lap[c]) for c in range(3)]
    Qw = [Wt.lap[c] * (rho * 3.0) - LrhoW[c] for c in range(3)]
    first = cross(W.val, cross(Wt.val, Qw))
    out = [f * 0.5 for f in first]
    if v is not None:
        rhob = dot(W.val, W.val)
        gauge = dot(W.val, [v.val[c] - v.lap[c] for c in range(3)]) * rhob
        out = [out[c] + W.val[c] * gauge for c in range(3)]
    return [o * chi for o in out]


def L_direct(W: Field, chi, h: Field):
    """DF(0) h straight from eq. (linearisation), expanding the Laplacians by the product rule."""
    Wv, dW, LW = W.val, W.grad, W.lap
    rho = dot(Wv, Wv)
    drho = [dot(Wv, dW[a]) * 2.0 for a in range(4)]
    Lrho = dot(Wv, LW) * 2.0
    for a in range(4):
        Lrho = Lrho + dot(dW[a], dW[a]) * 2.0
    Wh = dot(Wv, h.val)
    dWh = [dot(dW[a], h.val) + dot(Wv, h.grad[a]) for a in range(4)]
    LWh = dot(LW, h.val) + dot(Wv, h.lap)
    for a in range(4):
        LWh = LWh + dot(dW[a], h.grad[a]) * 2.0
    # Delta( 2 (W.h) W + |W|^2 h )
    T = [laplacian_of_product(Wh, dWh, LWh, Wv[c], [dW[a][c] for a in range(4)], LW[c]) * 2.0
         + laplacian_of_product(rho, drho, Lrho, h.val[c], [h.grad[a][c] for a in range(4)], h.lap[c]) for c in range(3)]
    LrhoW = [laplacian_of_product(rho, drho, Lrho, Wv[c], [dW[a][c] for a in range(4)], LW[c]) for c in range(3)]
    Q = [LW[c] * (rho * 3.0) - LrhoW[c] for c in range(3)]
    S = [LW[c] * (Wh * 6.0) + h.lap[c] * (rho * 3.0) - T[c] for c in range(3)]
    inner = cross(h.val, Q)
    WS = cross(Wv, S)
    first = cross(Wv, [inner[c] + WS[c] for c in range(3)])
    gauge = dot(Wv, [h.val[c] - h.lap[c] for c in range(3)]) * rho
    return [(first[c] * 0.5 + Wv[c] * gauge) * chi for c in range(3)]
