"""Newton-Galerkin construction of the approximate solution Wbar (Section 3.1).

Wbar is sought in the span of the basis fields Psi_k of Proposition 2.2 with degree l_k <= D,
as a zero of the Galerkin system

        R_k(a) := < Psi_k ,  F_{Wbar}(0) + kappa (|Wbar|^2 - 1) Wbar >_{L^2} = 0,   l_k <= D,

where F_{Wbar}(0) = (1/2) Wbar x (Wbar x (3|Wbar|^2 Delta Wbar - Delta(|Wbar|^2 Wbar))) is the
residual of the paper's equation (F-equation) with chi = 1.  The residual F(0) is tangential
(F(0) . Wbar = 0) and the term kappa (|Wbar|^2 - 1) Wbar is normal, so a zero of R has, up to the
Galerkin truncation error, both a vanishing tension-type residual and |Wbar| = 1; the normal
term removes the degeneracy of the tangential residual under pointwise rescalings of Wbar.
The Jacobian of R is the Galerkin matrix of
    L - G + (1/2) h x (Wbar x Q) + kappa (2 Wbar Wbar^T + (|Wbar|^2 - 1)),
where L is the linearisation (linearisation), G its gauge term and the cross-product term the
derivative with respect to the outer factor Wbar of the residual (1/2) Wbar x (Wbar x Q)
(DF(0) holds that factor fixed), so Newton's method converges quadratically.

Starting point: the L^2-projection of V_0 / |V_0|, with V_0 = (w^2 - zbar^4, 2 Re(z^2 w)) the
reference field of Proposition 2.7.

This script is plain floating point (no rigour); the certificate re-verifies everything.

Usage:  python -m numerics.solve_approximate --degree 40 --out data/Wbar_D40.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import Mode, HarmonicEvaluator, vector_modes
from eells.fields import field_from_coefficients, save_coefficients, save_scalar_coefficients, load_coefficients
from eells.equation import OperatorCoefficients, degrees, dot
from eells.reference import reference_field


def solver_grid(D, Dchi=0):
    """Grid exact for the Galerkin integrands  Psi_a . L Psi_b  (degree 2D + Kp) and for
    |F(0)|^2 (degree 2 (Dchi + 5D + 2))."""
    deg = degrees(D, Dchi)
    p = max(2 * deg["F0"], 2 * D + deg["Kp"])
    Ns = -(-p // 2)
    Nphi = p // 3 + 1
    return OrbitGrid(Ns, Nphi, rigorous=False), p


class GalerkinSystem:
    """Basis values on the grid and the Galerkin residual/matrix of the equation at Wbar."""

    def __init__(self, D, batch=64, kappa=10.0):
        self.D = D
        self.kappa = kappa
        self.grid, self.deg = solver_grid(D)
        self.H = HarmonicEvaluator(self.grid, max_degree=D)
        self.modes = vector_modes(D)
        self.batch = batch
        self.omega = self.grid.omega
        t = time.time()
        self.val = np.concatenate([self.H.evaluate(self.modes[i:i + batch])[0]
                                   for i in range(0, len(self.modes), batch)])
        self.setup_seconds = time.time() - t
        self.chi = np.ones(self.grid.shape)
        self.V0 = reference_field(self.grid)

    def field(self, a):
        return field_from_coefficients(self.modes, a, self.H, self.batch)

    def project(self, f):
        """L^2 projection of a 3-vector field (list of 3 arrays) onto the basis."""
        return np.einsum("kcib,cib,ib->k", self.val, np.stack(f), self.omega)

    def residual(self, a):
        """R(a) = P_D [ F(0) + kappa (|W|^2 - 1) W ]; also returns the field, the operator and F(0)."""
        W = self.field(a)
        op = OperatorCoefficients(W, self.chi)
        F0 = op.residual()
        defect = op.rho - 1.0
        res = self.project([F0[c] + self.kappa * defect * W.val[c] for c in range(3)])
        return res, W, op, F0

    def matrix(self, op):
        """Jacobian of R: Galerkin matrix of  L - G + (1/2) h x (W x Q) + kappa (2 W W^T + (|W|^2 - 1) I)."""
        d = len(self.modes)
        J = np.empty((d, d))
        Wv = op.val_W
        defect = op.rho - 1.0
        for i in range(0, d, self.batch):
            v, g, lp = self.H.evaluate(self.modes[i:i + self.batch])
            LPsi = op.apply_batch(v, g, lp) - op.gauge_batch(v, lp) + op.outer_cross_batch(v)
            Wh = v[:, 0] * Wv[0] + v[:, 1] * Wv[1] + v[:, 2] * Wv[2]
            for c in range(3):
                LPsi[:, c] += self.kappa * (2.0 * Wh * Wv[c] + defect * v[:, c])
            J[:, i:i + self.batch] = np.einsum("kcib,lcib,ib->kl", self.val, LPsi, self.omega)
        return J

    def diagnostics(self, a, W=None, F0=None):
        if W is None:
            _, W, _, F0 = self.residual(a)
        rho = dot(W.val, W.val)
        N = sum(dot(W.val, W.val) * dot(W.grad[k], W.grad[k]) - dot(W.val, W.grad[k]) ** 2 for k in range(4))
        energy = 0.5 * self.grid.integrate(N / rho ** 2, self.deg)   # not exact (rational integrand)
        L2res = np.sqrt(self.grid.integrate(dot(F0, F0), self.deg))
        hom = dot(W.val, self.V0)
        normX = np.sqrt(sum((m.l + 1) ** 4 * c ** 2 for m, c in zip(self.modes, a)))
        return {"energy_over_8pi2": float(energy / (8 * np.pi ** 2)),
                "residual_L2": float(L2res),
                "sup_abs_norm2_minus_1": float(np.abs(rho - 1).max()),
                "min_Wbar_dot_V0": float(hom.min()),
                "norm_X": float(normX),
                "sup_norm": float(np.sqrt(rho).max())}


def solve(D, max_iter=40, tol=1e-13, log=print, start=None, kappa=10.0):
    sys_ = GalerkinSystem(D, kappa=kappa)
    g = sys_.grid
    log(f"degree D={D}: {len(sys_.modes)} modes, grid {g.Ns + 1} x {g.Nphi} (exact to degree {sys_.deg}), "
        f"setup {sys_.setup_seconds:.1f}s")
    if start is None:
        V0 = sys_.V0
        nV = np.sqrt(dot(V0, V0))
        a = sys_.project([v / nV for v in V0])
        log("start: projection of V_0/|V_0|")
    else:
        a = start.copy()
    history = []
    t0 = time.time()
    res, W, op, F0 = sys_.residual(a)
    for it in range(max_iter):
        rn = np.linalg.norm(res)
        diag = sys_.diagnostics(a, W, F0)
        history.append({"iteration": it, "galerkin_residual": float(rn), **diag})
        log(f"  it {it:2d}  |R| = {rn:.3e}   ||F(0)||_L2 = {diag['residual_L2']:.3e}   "
            f"E/8pi^2 = {diag['energy_over_8pi2']:.12f}   sup||W|^2-1| = {diag['sup_abs_norm2_minus_1']:.2e}   "
            f"min W.V0 = {diag['min_Wbar_dot_V0']:.4f}   ({time.time() - t0:.0f}s)")
        if rn < tol:
            break
        J = sys_.matrix(op)
        delta = np.linalg.solve(J, -res)
        step = 1.0
        while True:
            a_try = a + step * delta
            res_try, W_try, op_try, F0_try = sys_.residual(a_try)
            if np.linalg.norm(res_try) <= (1 - 1e-4 * step) * rn or step < 1e-3:
                break
            step *= 0.5
        a = a_try
        res, W, op, F0 = res_try, W_try, op_try, F0_try
        log(f"        step {step:.3g}, |delta| = {np.linalg.norm(delta):.3e}")
    diag = sys_.diagnostics(a, W, F0)
    return sys_, a, history, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degree", type=int, default=40, help="degree D of Wbar")
    ap.add_argument("--out", type=str, default=None, help="output JSON (default data/Wbar_D{D}.json)")
    ap.add_argument("--start", type=str, default=None, help="coefficient JSON to start from (truncated/extended to D)")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-13)
    ap.add_argument("--kappa", type=float, default=10.0, help="weight of the normalisation term")
    args = ap.parse_args()
    out = Path(args.out or f"data/Wbar_D{args.degree}.json")
    start = None
    if args.start:
        m0, v0, _ = load_coefficients(args.start)
        lookup = dict(zip(m0, v0))
        start = np.array([lookup.get(m, 0.0) for m in vector_modes(args.degree)])
    sys_, a, history, diag = solve(args.degree, args.max_iter, args.tol, start=start, kappa=args.kappa)
    meta = {"solver": "numerics/solve_approximate.py (Newton-Galerkin with weak normalisation, chi = 1)",
            "kappa": args.kappa,
            "grid": {"Ns": sys_.grid.Ns, "Nphi": sys_.grid.Nphi, "exact_degree": sys_.deg},
            "iterations": len(history), "final": diag, "history": history}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_coefficients(out, sys_.modes, a, meta)
    chi_path = out.parent / "chi_constant.json"
    if not chi_path.exists():
        save_scalar_coefficients(chi_path, [Mode(0, "S", 0, 0)], [1.0], {"note": "chi = 1 (degree 0)"})
    print(f"wrote {out} and {chi_path}")
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
