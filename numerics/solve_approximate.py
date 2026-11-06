"""Newton-Galerkin construction of the approximate solution Wbar (Section 3.1).

Wbar is sought, for the Hopf invariant q, in the span of the basis fields Psi_k of
Proposition 2.2 with degree l_k <= D, as a zero of the Galerkin system

        R_k(a) := < Psi_k ,  F_{Wbar}(0) + kappa (|Wbar|^2 - 1) Wbar >_{L^2} = 0,   l_k <= D,

where F_{Wbar}(0) = (1/2) Wbar x (Wbar x (3|Wbar|^2 Delta Wbar - Delta(|Wbar|^2 Wbar))) is the
residual of the paper's equation (F-equation).  The residual F(0) is tangential
(F(0) . Wbar = 0) and the term kappa (|Wbar|^2 - 1) Wbar is normal, so a zero of R has, up to the
Galerkin truncation error, both a vanishing tension-type residual and |Wbar| = 1; the normal
term removes the degeneracy of the tangential residual under pointwise rescalings of Wbar.
The Jacobian of R is the Galerkin matrix of
    L - G + (1/2) h x (Wbar x Q) + kappa (2 Wbar Wbar^T + (|Wbar|^2 - 1)),
where L is the linearisation (linearisation), G its gauge term and the cross-product term the
derivative with respect to the outer factor Wbar of the residual (1/2) Wbar x (Wbar x Q)
(DF(0) holds that factor fixed), so Newton's method converges quadratically.  All Galerkin
integrands have degree <= 6D + 2 and are integrated exactly; the matrices are accumulated over
blocks of grid rows.

Starting point: the L^2-projection of V_q / |V_q| for the reference field
V_q = (w^2 - zbar^{2q}/4, Re(z^q w)) (eells/reference.py), or (--start) a coefficient file of the
same q, e.g. the solution at a lower degree (continuation in D).

This script is plain floating point (no rigour); the certificate re-verifies everything.

Usage:  python -m numerics.solve_approximate --hopf 2 --degree 40
        python -m numerics.solve_approximate --hopf 3 --degree 80 --start data/Wbar_q3_D64.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator, vector_modes
from eells.fields import field_from_coefficients, save_coefficients, load_coefficients
from eells.equation import OperatorCoefficients, dot
from eells.reference import reference_field


class GalerkinSystem:
    """Galerkin residual and Jacobian of the system above on a grid exact to degree 6D + 2."""

    def __init__(self, D, q, batch=48, kappa=10.0, chunk_points=20000):
        self.D, self.q, self.kappa, self.batch = D, q, kappa, batch
        self.deg = 6 * D + 2
        self.grid = OrbitGrid.for_quadrature(self.deg, q)
        self.H = HarmonicEvaluator(self.grid, max_degree=D)
        self.modes = vector_modes(D, q)
        self.chunks = self.grid.row_chunks(chunk_points)

    def field(self, a):
        return field_from_coefficients(self.modes, a, self.H, self.batch)

    def project(self, f):
        """L^2 projection of a 3-vector field (list of 3 arrays on the grid) onto the basis."""
        out = np.zeros(len(self.modes))
        for rows in self.chunks:
            fw = np.stack([fc[rows] for fc in f]) * self.grid.omega[rows]
            for i in range(0, len(self.modes), self.batch):
                v = self.H.evaluate(self.modes[i:i + self.batch], rows)[0]
                out[i:i + self.batch] += np.einsum("kcib,cib->k", v, fw)
        return out

    def residual(self, a):
        """R(a) = P_D [ F(0) + kappa (|W|^2 - 1) W ]; also returns the field and the operator."""
        W = self.field(a)
        op = OperatorCoefficients(W)
        F0 = op.residual()
        defect = op.rho - 1.0
        res = self.project([F0[c] + self.kappa * defect * W.val[c] for c in range(3)])
        return res, W, op

    def matrix(self, W):
        """Jacobian of R: Galerkin matrix of  L - G + (1/2) h x (W x Q) + kappa (2 W W^T + (|W|^2 - 1) I)."""
        d = len(self.modes)
        J = np.zeros((d, d))
        for rows in self.chunks:
            op = OperatorCoefficients(W.rows(rows))
            Wv = op.val_W
            defect = op.rho - 1.0
            w = self.grid.omega[rows]
            V, T = [], []
            for i in range(0, d, self.batch):
                v, g, lp = self.H.evaluate(self.modes[i:i + self.batch], rows)
                LPsi = op.apply_batch(v, g, lp) - op.gauge_batch(v, lp) + op.outer_cross_batch(v)
                Wh = v[:, 0] * Wv[0] + v[:, 1] * Wv[1] + v[:, 2] * Wv[2]
                for c in range(3):
                    LPsi[:, c] += self.kappa * (2.0 * Wh * Wv[c] + defect * v[:, c])
                V.append((v * w).reshape(len(v), -1))
                T.append(LPsi.reshape(len(v), -1))
            J += np.concatenate(V) @ np.concatenate(T).T
        return J

    def diagnostics(self, a, W):
        """Indicative values (the integrals of the rational energy density and of |F(0)|^2, of
        degree > 6D + 2, are not exact on this grid; the certificate computes them rigorously)."""
        op = OperatorCoefficients(W)
        F0 = op.residual()
        rho = op.rho
        N = sum(rho * dot(W.grad[k], W.grad[k]) - dot(W.val, W.grad[k]) ** 2 for k in range(4))
        energy = 0.5 * float((self.grid.omega * N / rho ** 2).sum())
        L2res = np.sqrt(float((self.grid.omega * dot(F0, F0)).sum()))
        hom = dot(W.val, reference_field(self.grid, self.q))
        normX = np.sqrt(sum((m.l + 1) ** 4 * c ** 2 for m, c in zip(self.modes, a)))
        return {"energy_over_8pi2": energy / (8 * np.pi ** 2), "residual_L2": float(L2res),
                "sup_abs_norm2_minus_1": float(np.abs(rho - 1).max()), "min_Wbar_dot_V": float(hom.min()),
                "norm_X": float(normX), "sup_norm": float(np.sqrt(rho).max())}


def solve(D, q, max_iter=40, tol=1e-13, log=print, start=None, kappa=10.0):
    sys_ = GalerkinSystem(D, q, kappa=kappa)
    g = sys_.grid
    log(f"Hopf invariant q={q}, degree D={D}: {len(sys_.modes)} modes, grid {g.Ns + 1} x {g.Nphi} (exact to degree {sys_.deg})")
    if start is None:
        V = reference_field(g, q)
        nV = np.sqrt(dot(V, V))
        a = sys_.project([v / nV for v in V])
        log("start: projection of V/|V|")
    else:
        a = start.copy()
    history = []
    t0 = time.time()
    res, W, _ = sys_.residual(a)
    for it in range(max_iter):
        rn = np.linalg.norm(res)
        diag = sys_.diagnostics(a, W)
        history.append({"iteration": it, "galerkin_residual": float(rn), **diag})
        log(f"  it {it:2d}  |R| = {rn:.3e}   ||F(0)||_L2 ~ {diag['residual_L2']:.3e}   "
            f"E/8pi^2 ~ {diag['energy_over_8pi2']:.12f}   sup||W|^2-1| = {diag['sup_abs_norm2_minus_1']:.2e}   "
            f"min W.V = {diag['min_Wbar_dot_V']:.4f}   ({time.time() - t0:.0f}s)")
        if rn < tol:
            break
        J = sys_.matrix(W)
        delta = np.linalg.solve(J, -res)
        step = 1.0
        while True:
            a_try = a + step * delta
            res_try, W_try, _ = sys_.residual(a_try)
            if np.linalg.norm(res_try) <= (1 - 1e-4 * step) * rn or step < 1e-3:
                break
            step *= 0.5
        a, res, W = a_try, res_try, W_try
        log(f"        step {step:.3g}, |delta| = {np.linalg.norm(delta):.3e}")
    diag = sys_.diagnostics(a, W)
    return sys_, a, history, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hopf", type=int, default=2, help="Hopf invariant q")
    ap.add_argument("--degree", type=int, default=40, help="degree D of Wbar")
    ap.add_argument("--out", type=str, default=None, help="output JSON (default data/Wbar_q{q}_D{D}.json)")
    ap.add_argument("--start", type=str, default=None, help="coefficient JSON to start from (truncated/extended to D)")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-13)
    ap.add_argument("--kappa", type=float, default=10.0, help="weight of the normalisation term")
    args = ap.parse_args()
    q = args.hopf
    out = Path(args.out or f"data/Wbar_q{q}_D{args.degree}.json")
    start = None
    if args.start:
        m0, v0, _ = load_coefficients(args.start)
        assert m0[0].q == q, "the start file is for a different Hopf invariant"
        lookup = dict(zip(m0, v0))
        start = np.array([lookup.get(m, 0.0) for m in vector_modes(args.degree, q)])
    sys_, a, history, diag = solve(args.degree, q, args.max_iter, args.tol, start=start, kappa=args.kappa)
    meta = {"solver": "numerics/solve_approximate.py (Newton-Galerkin with weak normalisation)",
            "kappa": args.kappa, "start": args.start or "projection of V/|V|",
            "grid": {"Ns": sys_.grid.Ns, "Nphi": sys_.grid.Nphi, "exact_degree": sys_.deg},
            "iterations": len(history), "final": diag, "history": history}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefficients(out, sys_.modes, a, meta)
    print(f"wrote {out}")
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
