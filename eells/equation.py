"""The equation F(v) = 0 of Section 2.2, its linearisation (eq. linearisation), the expanded
coefficients used in the linear estimate of Section 3.3 and the L^2-adjoint of the linearisation.

Notation: W = Wbar + v, Delta the (non-positive) Laplacian of S^3, d_a (a = 0..3) tangential
partial derivatives along the ambient axes, so that  sum_a d_a f d_a g = <grad f, grad g>  and
Delta(f g) = f Delta g + g Delta f + 2 sum_a d_a f d_a g.

    F(v) = (1/2) Wbar x ( W x (3|W|^2 Delta W - Delta(|W|^2 W)) ) + |Wbar|^2 (Wbar . (1 - Delta) v) Wbar.

**Expanded linearisation.**  With  Q = 3|Wbar|^2 Delta Wbar - Delta(|Wbar|^2 Wbar)
  = 2|Wbar|^2 Delta Wbar - 2 (Wbar . Delta Wbar + |grad Wbar|^2) Wbar - 4 sum_a (Wbar . d_a Wbar) d_a Wbar
and  Pi = |Wbar|^2 I - Wbar Wbar^T  (so that Wbar x (Wbar x v) = -Pi v,  Pi Wbar = 0), the
linearisation DF(0) h of eq. (linearisation) reads

    L h = -P Delta h + sum_a M_a d_a h + M_0 h,

    P   = |Wbar|^4,
    M_a = 2 [ (Pi d_a Wbar) Wbar^T + (Wbar . d_a Wbar) Pi ],
    M_0 = (1/2) [ (Wbar . Q) I - Q Wbar^T ]
          - Pi [ 2 Delta Wbar Wbar^T - 2 sum_a d_a Wbar d_a Wbar^T - (Wbar . Delta Wbar + |grad Wbar|^2) I ]
          + |Wbar|^2 Wbar Wbar^T.

(Derivation: expand the Laplacians of products in (linearisation) by the product rule; the
second-order terms combine to  -(Pi^2 + |Wbar|^2 Wbar Wbar^T) Delta h = -|Wbar|^4 Delta h.)
The residual is  F(0) = (1/2) Wbar x (Wbar x Q) = -(1/2) Pi Q.

With R = L + Delta the first- and zeroth-order coefficient matrices are M_a and M_0 and the
second-order coefficient is (P - 1)(-Delta); eta, b_1, b_0 of Section 3.3 bound
sup |P - 1|,  sup sum_a ||M_a||_F^2,  sup ||M_0||_F^2.

**Adjoint.**  Writing  Pp u = (1/2) Wbar x (Wbar x u) = -(1/2) Pi u  (symmetric) and
N = 2 Wbar Wbar^T + |Wbar|^2 I, eq. (linearisation) is the sum of the six terms
(1/2) Wbar x (h x Q),  6 (Wbar . h) Pp Delta Wbar,  3 |Wbar|^2 Pp Delta h,  -Pp Delta(N h),
|Wbar|^2 (Wbar . h) Wbar  and  -|Wbar|^2 (Wbar . Delta h) Wbar, so that, integrating by parts on S^3,

    L^* g = (1/2) [ (Wbar . Q) g - (Q . g) Wbar ] + 6 (Delta Wbar . Pp g) Wbar + 3 Delta(|Wbar|^2 Pp g)
            - N Delta(Pp g) + |Wbar|^2 (Wbar . g) Wbar - Delta(|Wbar|^2 (Wbar . g) Wbar),

with <g, L h>_{L^2} = <L^* g, h>_{L^2}.  With  F_1 = |Wbar|^2 Pp,  F_2 = Pp,  F_3 = |Wbar|^2 Wbar Wbar^T
and  Delta(F g) = F Delta g + 2 sum_a d_a F d_a g + (Delta F) g  this is

    L^* g = -|Wbar|^4 Delta g + sum_a B_a d_a g + C_0 g,
    B_a = 2 (3 d_a F_1 - N d_a F_2 - d_a F_3),
    C_0 = (1/2) [ (Wbar . Q) I - Wbar Q^T ] + 6 Wbar (Pp Delta Wbar)^T + 3 Delta F_1 - N Delta F_2 + F_3 - Delta F_3

(the second-order coefficient 3 F_1 - N F_2 - F_3 equals -|Wbar|^4 I because Wbar Wbar^T Pp = 0).
The derivatives of the entries of F_1, F_2, F_3 (products of components of Wbar) are evaluated by
the product rule from values, tangential gradients and Laplacians ("jets") of the factors.

Polynomial degrees (Wbar of degree D), see :func:`degrees`.
"""
from __future__ import annotations

from . import arith as ar
from .fields import Field


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def degrees(D, q=2):
    """Polynomial degrees (as evaluated, in the ambient coordinates) of the quantities built from
    Wbar of degree D, for the Hopf invariant q."""
    return {"W": D, "dW": D + 1, "LW": D,
            "rho": 2 * D,                     # |Wbar|^2
            "Q": 3 * D + 2,
            "F0": 5 * D + 2,                  # residual F(0)
            "P": 4 * D,
            "M": 4 * D + 1,                   # each M_a
            "M0": 4 * D + 2,
            "Kp": 4 * D + 2,                  # L Psi and L^* Psi have degree l + Kp as evaluated here
            "K": 4 * D,                       # band width of the paper (K = 4 deg Wbar)
            "energy": 4 * D + 2,              # sum_a |Wbar x d_a Wbar|^2
            "V": 2 * q}                       # the reference field V_q


# ---------------------------------------------------------------------------- jets
# A jet of a scalar function f is (f, [d_0 f, .., d_3 f], Delta f); jets multiply by the product rule.
def jet_mul(f, g):
    fv, fg, fL = f
    gv, gg, gL = g
    dot_g = fg[0] * gg[0] + fg[1] * gg[1] + fg[2] * gg[2] + fg[3] * gg[3]
    return (fv * gv, [fv * gg[a] + gv * fg[a] for a in range(4)], fv * gL + gv * fL + dot_g * 2.0)


def jet_add(f, g):
    return (f[0] + g[0], [f[1][a] + g[1][a] for a in range(4)], f[2] + g[2])


def jet_sub(f, g):
    return (f[0] - g[0], [f[1][a] - g[1][a] for a in range(4)], f[2] - g[2])


def jet_scale(f, t):
    return (f[0] * t, [x * t for x in f[1]], f[2] * t)


class OperatorCoefficients:
    """P, M_a (a = 0..3), M_0 of the expanded linearisation at Wbar, on the grid (or on a block of
    grid rows), and the adjoint L^*."""

    def __init__(self, W: Field):
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
        self.W = W
        self.rho, self.gradsq, self.WLW, self.Pi, self.Q, self.WQ = rho, gradsq, WLW, Pi, Q, WQ
        self.val_W = Wv
        self.P = rho * rho
        # M_a = 2 [ (Pi d_a Wbar) Wbar^T + (Wbar . d_a Wbar) Pi ]
        self.M = []
        for a in range(4):
            PidW = [Pi[c][0] * dW[a][0] + Pi[c][1] * dW[a][1] + Pi[c][2] * dW[a][2] for c in range(3)]
            self.M.append([[(PidW[c] * Wv[d] + Pi[c][d] * WdW[a]) * 2.0 for d in range(3)] for c in range(3)])
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
                row.append(((WQ if c == d else 0.0) - Q[c] * Wv[d]) * 0.5 - PiS + Wv[c] * Wv[d] * rho)
            self.M0.append(row)
        self._adj = None

    def adjoint_coefficients(self):
        """(B_a, C_0) of L^* g = -|Wbar|^4 Delta g + sum_a B_a d_a g + C_0 g (module docstring)."""
        if self._adj is not None:
            return self._adj
        W = self.W
        Wv = W.val
        jW = [(Wv[c], [W.grad[a][c] for a in range(4)], W.lap[c]) for c in range(3)]
        jr = jet_add(jet_add(jet_mul(jW[0], jW[0]), jet_mul(jW[1], jW[1])), jet_mul(jW[2], jW[2]))
        F2 = [[jet_scale(jet_mul(jW[c], jW[d]) if c != d else jet_sub(jet_mul(jW[c], jW[c]), jr), 0.5)
               for d in range(3)] for c in range(3)]
        F1 = [[jet_mul(jr, F2[c][d]) for d in range(3)] for c in range(3)]
        F3 = [[jet_mul(jr, jet_mul(jW[c], jW[d])) for d in range(3)] for c in range(3)]
        rho = self.rho
        N = [[Wv[c] * Wv[d] * 2.0 + (rho if c == d else 0.0) for d in range(3)] for c in range(3)]

        def NF(F, part, a=None):          # (N F)_{cd} for the gradient (part 1, index a) or Laplacian (part 2)
            get = (lambda x: x[1][a]) if part == 1 else (lambda x: x[2])
            return [[N[c][0] * get(F[0][d]) + N[c][1] * get(F[1][d]) + N[c][2] * get(F[2][d]) for d in range(3)]
                    for c in range(3)]

        B = []
        for a in range(4):
            NdF2 = NF(F2, 1, a)
            B.append([[(F1[c][d][1][a] * 3.0 - NdF2[c][d] - F3[c][d][1][a]) * 2.0 for d in range(3)] for c in range(3)])
        NLF2 = NF(F2, 2)
        LW = W.lap
        PpLW = [F2[c][0][0] * LW[0] + F2[c][1][0] * LW[1] + F2[c][2][0] * LW[2] for c in range(3)]
        C0 = [[((self.WQ if c == d else 0.0) - Wv[c] * self.Q[d]) * 0.5 + Wv[c] * PpLW[d] * 6.0
               + F1[c][d][2] * 3.0 - NLF2[c][d] + F3[c][d][0] - F3[c][d][2] for d in range(3)] for c in range(3)]
        self._adj = (B, C0)
        return self._adj

    def residual(self):
        """F(0) = -(1/2) Pi Q."""
        return [-(self.Pi[c][0] * self.Q[0] + self.Pi[c][1] * self.Q[1] + self.Pi[c][2] * self.Q[2]) * 0.5
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

    def apply_adjoint_batch(self, val, grad, lap):
        """L^* (the L^2-adjoint of L) applied to a stack of fields (same layout as apply_batch)."""
        B, C0 = self.adjoint_coefficients()
        out = []
        for c in range(3):
            v = -(lap[:, c] * self.P)
            for d in range(3):
                v = v + val[:, d] * C0[c][d]
                for a in range(4):
                    v = v + grad[:, a, d] * B[a][c][d]
            out.append(v)
        return ar.stack(out, axis=1)

    def outer_cross_batch(self, val):
        """(1/2) h x (W x Q) for a stack of fields h: the derivative of the residual
        (1/2) Wbar x (Wbar x Q) with respect to its *outer* factor Wbar (needed when Wbar itself is
        the unknown, as in the solver; DF(0) keeps that factor fixed)."""
        Wv, Q = self.val_W, self.Q
        WxQ = cross(Wv, Q)
        return ar.stack([(val[:, 1] * WxQ[2] - val[:, 2] * WxQ[1]) * 0.5,
                         (val[:, 2] * WxQ[0] - val[:, 0] * WxQ[2]) * 0.5,
                         (val[:, 0] * WxQ[1] - val[:, 1] * WxQ[0]) * 0.5], axis=1)

    def gauge_batch(self, val, lap):
        """The gauge term  |Wbar|^2 (Wbar . (1 - Delta) h) Wbar  for a stack of fields h."""
        Wv = self.val_W
        inner = val[:, 0] * Wv[0] + val[:, 1] * Wv[1] + val[:, 2] * Wv[2] \
            - (lap[:, 0] * Wv[0] + lap[:, 1] * Wv[1] + lap[:, 2] * Wv[2])
        inner = inner * self.rho
        return ar.stack([inner * Wv[c] for c in range(3)], axis=1)

    def frobenius_squares(self):
        """(sum_a ||M_a||_F^2,  ||M_0||_F^2)  pointwise (degrees 2(4D+1), 2(4D+2))."""
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


def F_direct(W: Field, v: Field = None):
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
    return out


def L_direct(W: Field, h: Field):
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
    return [first[c] * 0.5 + Wv[c] * gauge for c in range(3)]
