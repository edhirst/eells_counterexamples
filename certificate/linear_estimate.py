"""Section 3.3: the linear estimate.

* coefficient bounds   eta >= sup | |Wbar|^4 - 1 |,  b_1^2 >= sup sum_a ||M_a||_F^2,
                       b_0^2 >= sup ||M_0||_F^2   (Chebyshev-Fourier bounds on the bounding grid),
                       and  ||Wbar||_{C^0}  (for C_nl of Proposition 3.2).
* gamma = (1 - eta)(l+1)(l+3) - b_1 sqrt((l+1)(l+3)) - b_0 > 0,    ||D^{-1}||_{L^2 -> L^2} <= 1/gamma.
* the finite block  A = (<Psi_a, L Psi_b>)_{l_a, l_b <= l}  by exact quadrature in ball arithmetic.
* ||A^{-1}||_2 <= sqrt(||Z||_1 ||Z||_inf) / (1 - sqrt(||E||_1 ||E||_inf)),  E = I - Z A,  Z ~ mid(A)^{-1}.
* B and C.  Delta preserves the degree and the multipliers of L have degree <= K = 4 deg Wbar, so
  B (high -> low) and C (low -> high) are finite.  Their norms are obtained from Gram matrices
  of the low modes: with P_low the L^2-projection onto H_{<= l} and u in H_{<= l},
      ||C u||^2 = ||L u||^2 - ||P_low L u||^2 = u^T (G_L - A^T A) u,        G_L  = (<L Psi_a, L Psi_b>),
      ||B^T u||^2 = ||L^* u||^2 - ||P_low L^* u||^2 = u^T (G_L* - A A^T) u,   G_L* = (<L^* Psi_a, L^* Psi_b>),
  where L^* is the L^2-adjoint of L (eells/equation.py; L and L^* map G-equivariant fields to
  G-equivariant fields, and <Psi_a, L^* Psi_b> = A_{ba}).  The Gram matrices are integrated
  exactly (degree 2(l + 4 deg Wbar + 2)), so  ||C||^2 = lambda_max(G_L - A^T A)  and
  ||B||^2 = lambda_max(G_L* - A A^T)  are bounded rigorously; likewise
      ||A^{-1} B|| <= ||Z B|| / (1 - ||E||),     ||Z B||^2 = lambda_max(Z (G_L* - A A^T) Z^T).
  (This replaces the entry-by-entry computation of B and C, whose number of entries grows like
  (l + 4 deg Wbar)^2 l^2; the norms are the same.)
* Schur complement S = A - B D^{-1} C:  ||S u|| >= (1 - ||A^{-1} B|| ||D^{-1}|| ||C||) ||u|| / ||A^{-1}||, and
      ||L^{-1}||_{L^2->L^2} <= ||D^{-1}|| + ||A^{-1}|| sqrt((1 + ||B||^2 ||D^{-1}||^2)(1 + ||C||^2 ||D^{-1}||^2))
                                             / (1 - ||A^{-1} B|| ||D^{-1}|| ||C||).
* M = 2/(1-eta) + (1 + 2 b_0/(1-eta) + b_1^2/(1-eta)^2) ||L^{-1}||_{L^2->L^2}  >=  ||L^{-1}||_{Y -> X}.
"""
from __future__ import annotations

import time

import numpy as np

from eells import arith as ar
from eells.ball import Ball, matmul, concatenate, norm_1, norm_inf, norm_2, val_up
from eells.equation import OperatorCoefficients
from eells.harmonics import vector_modes


def coefficient_bounds(W_bound, grid_bound, deg, log=print):
    """eta, b_1, b_0 of Section 3.3 and sup |Wbar| (rigorous upper bounds)."""
    op = OperatorCoefficients(W_bound)
    lo, hi = grid_bound.analyse(op.P - 1.0, deg["P"]).bounds(*fine_grid(deg["P"], grid_bound.q))
    eta = max(abs(lo), abs(hi))
    m1, m0 = op.frobenius_squares()
    _, s1 = grid_bound.analyse(m1, 2 * deg["M"]).bounds(*fine_grid(2 * deg["M"], grid_bound.q))
    _, s0 = grid_bound.analyse(m0, 2 * deg["M0"]).bounds(*fine_grid(2 * deg["M0"], grid_bound.q))
    b1 = float(val_up(np.sqrt(max(s1, 0.0)), 1))
    b0 = float(val_up(np.sqrt(max(s0, 0.0)), 1))
    _, sw = grid_bound.analyse(op.rho, deg["rho"]).bounds(*fine_grid(deg["rho"], grid_bound.q))
    W_sup = float(val_up(np.sqrt(max(sw, 0.0)), 1))
    # non-rigorous grid maxima, for information
    info = {"eta_grid": float(np.abs(ar.midpoint(op.P - 1.0)).max()),
            "b1_grid": float(np.sqrt(ar.midpoint(m1).max())), "b0_grid": float(np.sqrt(ar.midpoint(m0).max()))}
    log(f"  eta <= {eta:.3e} (grid {info['eta_grid']:.3e}),  b1 <= {b1:.4f} (grid {info['b1_grid']:.4f}),  "
        f"b0 <= {b0:.4f} (grid {info['b0_grid']:.4f}),  ||Wbar||_inf <= {W_sup:.6f}")
    return {"eta": eta, "b1": b1, "b0": b0, "W_sup": W_sup, **info}


def fine_grid(deg, q):
    """Fine evaluation grid (N1, N2) for ChebFourier.bounds, scaled with the degree so that the
    Lipschitz remainder stays small."""
    return max(2048, 8 * (deg // 2)), max(512, 8 * (deg // (q + 1)))


def gamma_bound(eta, b1, b0, l):
    """Lower bound of gamma = (1 - eta)(l+1)(l+3) - b1 sqrt((l+1)(l+3)) - b0 (ball arithmetic)."""
    lam = float((l + 1) * (l + 3))
    g = (1.0 - Ball.exact(eta)) * lam - Ball.exact(b1) * Ball.exact(lam).sqrt() - Ball.exact(b0)
    return float(g.lower())


def lambda_max(S, squarings=4):
    """Upper bound of the largest eigenvalue of a symmetric positive semi-definite ball matrix
    (= its spectral norm, bounded through repeated squaring, eells.ball.norm_2)."""
    S = (S + S.T) * 0.5                     # encloses the (symmetric) exact matrix; rounding is accounted for
    return norm_2(S, squarings=squarings)


def finite_blocks(W_quad, grid_quad, H, l, q, deg, batch=32, chunk_points=6000, log=print):
    """The ball matrices A, G_L - A^T A (= C^T C) and G_L* - A A^T (= B B^T) of the low modes,
    accumulated block by block over the rows of the quadrature grid."""
    t0 = time.time()
    low = vector_modes(l, q)
    d = len(low)
    assert grid_quad.exact_for(2 * (l + deg["Kp"])), "quadrature grid not exact for the Gram matrices"
    A = GL = GA = None
    chunks = grid_quad.row_chunks(chunk_points)
    for ic, rows in enumerate(chunks):
        op = OperatorCoefficients(W_quad.rows(rows))
        w = grid_quad.omega[rows].reshape(1, 1, -1)
        V, L, La = [], [], []
        for i in range(0, d, batch):
            v, g, lp = H.evaluate(low[i:i + batch], rows)
            k = len(low[i:i + batch])
            V.append(v.reshape(k, 3, -1))
            L.append(op.apply_batch(v, g, lp).reshape(k, 3, -1))
            La.append(op.apply_adjoint_batch(v, g, lp).reshape(k, 3, -1))
        V, L, La = concatenate(V), concatenate(L), concatenate(La)
        Vw = (V * w).reshape(d, -1)
        Lw = (L * w).reshape(d, -1)
        Law = (La * w).reshape(d, -1)
        L, La = L.reshape(d, -1), La.reshape(d, -1)
        dA, dGL, dGA = matmul(Vw, L.T), matmul(Lw, L.T), matmul(Law, La.T)
        A = dA if A is None else A + dA
        GL = dGL if GL is None else GL + dGL
        GA = dGA if GA is None else GA + dGA
        if ic % max(1, len(chunks) // 8) == 0 or ic == len(chunks) - 1:
            log(f"    grid rows {rows.stop}/{grid_quad.Ns + 1} ({time.time() - t0:.0f}s)")
    CtC = GL - matmul(A.T, A)
    BBt = GA - matmul(A, A.T)
    nC = float(val_up(np.sqrt(lambda_max(CtC)), 1))
    nB = float(val_up(np.sqrt(lambda_max(BBt)), 1))
    info = {"d_low": d, "max_degree_coupled": l + deg["K"],
            "normB_float": float(np.sqrt(max(np.linalg.eigvalsh(0.5 * (BBt.m + BBt.m.T)).max(), 0.0))),
            "normC_float": float(np.sqrt(max(np.linalg.eigvalsh(0.5 * (CtC.m + CtC.m.T)).max(), 0.0))),
            "max_radius_A": float(A.r.max()), "max_radius_CtC": float(CtC.r.max()), "max_radius_BBt": float(BBt.r.max()),
            "chunks": len(chunks), "seconds": time.time() - t0}
    log(f"  A block {d}x{d};  ||B|| <= {nB:.4f} (float {info['normB_float']:.4f}),  ||C|| <= {nC:.4f} "
        f"(float {info['normC_float']:.4f})  [coupled modes up to degree {l + deg['K']}, {time.time() - t0:.0f}s]")
    return A, BBt, nB, nC, info


def inverse_bound(A, log=print):
    """||a^{-1}||_2 <= sqrt(||Z||_1 ||Z||_inf) / (1 - sqrt(||E||_1 ||E||_inf)) for every a in A."""
    Z = np.linalg.inv(A.m)
    E = Ball(np.eye(A.m.shape[0])) - matmul(Ball(Z), A)
    nE = float(val_up(np.sqrt(val_up(norm_1(E) * norm_inf(E), 1)), 1))
    assert nE < 1, f"||E|| = {nE} >= 1: the finite block is not certified invertible"
    nZ = float(val_up(np.sqrt(val_up(norm_1(Ball(Z)) * norm_inf(Ball(Z)), 1)), 1))
    bound = float((Ball.exact(nZ) / (1.0 - Ball.exact(nE))).upper())
    sv = np.linalg.svd(A.m, compute_uv=False)
    log(f"  ||E|| <= {nE:.3e},  ||Z|| <= {nZ:.4f}  =>  ||A^-1|| <= {bound:.4f}   (1/sigma_min(mid A) = {1 / sv.min():.4f})")
    return bound, Z, nE, {"normE": nE, "normZ": nZ, "sigma_min_midA": float(sv.min()), "sigma_max_midA": float(sv.max())}


def L2_inverse_bound(invA, Z, nE, BBt, nB, nC, gamma, log=print):
    """||L^{-1}||_{L^2 -> L^2} by the Schur complement with ||A^{-1} B|| <= ||Z B|| / (1 - ||E||)."""
    Zb = Ball(Z)
    nZB = float(val_up(np.sqrt(lambda_max(matmul(matmul(Zb, BBt), Zb.T))), 1))
    AB = Ball.exact(nZB) / (1.0 - Ball.exact(nE))                # ||A^{-1} B||
    invD = 1.0 / Ball.exact(gamma)
    theta = AB * invD * Ball.exact(nC)                           # ||A^{-1} B|| ||D^{-1}|| ||C||
    assert theta.upper() < 1, f"Schur bound not certified: ||A^-1 B|| ||D^-1|| ||C|| = {theta.upper()}"
    BD = Ball.exact(nB) * invD
    DC = Ball.exact(nC) * invD
    Linv = invD + Ball.exact(invA) * ((1.0 + BD * BD) * (1.0 + DC * DC)).sqrt() / (1.0 - theta)
    log(f"  ||Z B|| <= {nZB:.4e},  ||A^-1 B|| <= {float(AB.upper()):.4e},  ||A^-1 B|| ||D^-1|| ||C|| <= {float(theta.upper()):.4e}")
    return float(Linv.upper()), {"normZB": nZB, "normA_inv_B": float(AB.upper()), "theta": float(theta.upper())}


def M_bound(eta, b1, b0, L2inv):
    """M >= ||L^{-1}||_{Y -> X} (injectivity estimate)."""
    e, b1b, b0b, Linv = Ball.exact(eta), Ball.exact(b1), Ball.exact(b0), Ball.exact(L2inv)
    one = 1.0 - e
    M = 2.0 / one + (1.0 + 2.0 * b0b / one + b1b * b1b / (one * one)) * Linv
    return float(M.upper())
