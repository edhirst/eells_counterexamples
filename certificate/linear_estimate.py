"""Section 3.2: the linear estimate.

* coefficient bounds   eta >= sup |chi |Wbar|^4 - 1|,  b_1^2 >= sup sum_a ||M_a||_F^2,
                       b_0^2 >= sup ||M_0||_F^2   (Chebyshev-Fourier bounds on the bounding grid),
                       and  ||chi Wbar||_{C^0}.
* gamma = (1 - eta)(l+1)(l+3) - b_1 sqrt((l+1)(l+3)) - b_0 > 0,    ||D^{-1}||_{L^2 -> L^2} <= 1/gamma.
* the finite blocks  A = (<Psi_a, L Psi_b>)_{l_a, l_b <= l},
                     B: l_a <= l < l_b <= l + K,   C: l_b <= l < l_a <= l + K,  K = 4 deg Wbar + deg chi,
  by exact quadrature in ball arithmetic;  ||B||_2, ||C||_2 <= min(sqrt(||.||_1 ||.||_inf), ||.||_F).
* ||A^{-1}||_2 <= sqrt(||Z||_1 ||Z||_inf) / (1 - sqrt(||E||_1 ||E||_inf)),  E = I - Z A,  Z ~ mid(A)^{-1}.
* Schur complement S = A - B D^{-1} C.  Two bounds are computed:
  (literal, as in the draft)
      ||S^{-1}|| <= 1 / (||A^{-1}||^{-1} - ||B|| ||D^{-1}|| ||C||),
  (weighted) using the Laplacian weight on the high modes: the lower bound for D gives
      ||D h||_{L^2} >= (gamma / lambda_{l+1}) ||Delta h||_{L^2},  lambda_{l+1} = (l+1)(l+3),
  i.e.  ||(-Delta) D^{-1}||_{L^2->L^2} <= lambda_{l+1}/gamma, and (-Delta)^{-1} is diagonal on
  the high modes with entries 1/lambda_b, lambda_b = l_b (l_b + 2); hence
      theta := ||A^{-1} B D^{-1} C|| <= ||A^{-1} B Lambda^{-1}|| (lambda_{l+1}/gamma) ||C||,
      ||A^{-1} B Lambda^{-1}|| <= ||Z B Lambda^{-1}|| / (1 - ||E||),
      ||S^{-1}|| <= ||A^{-1}|| / (1 - theta),        ||B D^{-1}|| <= ||B Lambda^{-1}|| lambda_{l+1}/gamma.
  The literal bound fails for this problem (||B||, ||C|| grow like b_1 l while gamma ~ l^2, so
  ||B|| ||C|| / gamma ~ b_1^2 >> ||A^{-1}||^{-1}); the weighted bound exploits that B D^{-1} C acts
  on the highest low modes, where A is large.
* ||L^{-1}||_{L^2->L^2} <= ||D^{-1}|| + ||S^{-1}|| sqrt((1 + ||B D^{-1}||^2)(1 + ||D^{-1} C||^2)).
* M = 2/(1-eta) + (1 + 2 b_0/(1-eta) + b_1^2/(1-eta)^2) ||L^{-1}||_{L^2->L^2}  >=  ||L^{-1}||_{Y -> X}.
"""
from __future__ import annotations

import time

import numpy as np

from eells import arith as ar
from eells.ball import Ball, matmul, norm_1, norm_inf, norm_2, val_up
from eells.equation import OperatorCoefficients, dot
from eells.harmonics import vector_modes


def coefficient_bounds(W_bound, chi_bound, grid_bound, deg, log=print):
    op = OperatorCoefficients(W_bound, chi_bound)
    lo, hi = grid_bound.analyse(op.P - 1.0, deg["P"]).bounds()
    eta = max(abs(lo), abs(hi))
    m1, m0 = op.frobenius_squares()
    _, s1 = grid_bound.analyse(m1, 2 * deg["M"]).bounds()
    _, s0 = grid_bound.analyse(m0, 2 * deg["M0"]).bounds()
    b1 = float(val_up(np.sqrt(max(s1, 0.0)), 1))
    b0 = float(val_up(np.sqrt(max(s0, 0.0)), 1))
    _, sc = grid_bound.analyse(chi_bound * chi_bound * op.rho, 2 * deg["chi"] + 2 * deg["W"]).bounds()
    chiW = float(val_up(np.sqrt(max(sc, 0.0)), 1))
    # non-rigorous grid maxima, for information
    info = {"eta_grid": float(np.abs(ar.midpoint(op.P - 1.0)).max()),
            "b1_grid": float(np.sqrt(ar.midpoint(m1).max())), "b0_grid": float(np.sqrt(ar.midpoint(m0).max()))}
    log(f"  eta <= {eta:.3e} (grid {info['eta_grid']:.3e}),  b1 <= {b1:.4f} (grid {info['b1_grid']:.4f}),  "
        f"b0 <= {b0:.4f} (grid {info['b0_grid']:.4f}),  ||chi Wbar||_inf <= {chiW:.6f}")
    return {"eta": eta, "b1": b1, "b0": b0, "chiW_sup": chiW, **info}


def gamma_bound(eta, b1, b0, l):
    """Lower bound of gamma = (1 - eta)(l+1)(l+3) - b1 sqrt((l+1)(l+3)) - b0 (ball arithmetic)."""
    lam = float((l + 1) * (l + 3))
    g = (1.0 - Ball.exact(eta)) * lam - Ball.exact(b1) * Ball.exact(lam).sqrt() - Ball.exact(b0)
    return float(g.lower())


def finite_blocks(W_quad, chi_quad, grid_quad, H, l, deg, batch=48, log=print):
    """A (ball matrix), ||B||_2 and ||C||_2 (upper bounds), with the mode lists."""
    t0 = time.time()
    K = deg["K"]
    low = vector_modes(l)
    high = [m for m in vector_modes(l + K) if m.l > l]
    d, dh = len(low), len(high)
    op = OperatorCoefficients(W_quad, chi_quad)
    omega = grid_quad.omega
    G = omega.size
    p_block = 2 * l + K + deg["Kp"]
    assert grid_quad.exact_for(p_block)
    # low modes: values (test functions) and L Psi (weighted)
    val_low, LPsi_low = [], []
    for i in range(0, d, batch):
        v, g, lp = H.evaluate(low[i:i + batch])
        val_low.append(v)
        LPsi_low.append(op.apply_batch(v, g, lp) * omega)
    val_low = ar.stack([x for b in val_low for x in [b[k] for k in range(len(b))]])   # (d, 3, i, b)
    LPsi_low = ar.stack([x for b in LPsi_low for x in [b[k] for k in range(len(b))]])
    VL = val_low.reshape(d, 3 * G)
    WL = LPsi_low.reshape(d, 3 * G)
    A = matmul(VL, WL.T)
    log(f"  A block {d}x{d} assembled ({time.time() - t0:.0f}s)")
    # high modes in batches: C rows (values) and B columns (L Psi)
    Bm = np.zeros((d, dh)); Br = np.zeros((d, dh))
    Cm = np.zeros((dh, d)); Cr = np.zeros((dh, d))
    for i in range(0, dh, batch):
        mk = high[i:i + batch]
        v, g, lp = H.evaluate(mk)
        nb = len(mk)
        Cblk = matmul(v.reshape(nb, 3 * G), WL.T)
        Cm[i:i + nb], Cr[i:i + nb] = Cblk.m, Cblk.r
        LPsi = (op.apply_batch(v, g, lp) * omega).reshape(nb, 3 * G)
        Bblk = matmul(VL, LPsi.T)
        Bm[:, i:i + nb], Br[:, i:i + nb] = Bblk.m, Bblk.r
        if (i // batch) % 10 == 0:
            log(f"    high modes {i + nb}/{dh} ({time.time() - t0:.0f}s)")
    B, C = Ball(Bm, Br), Ball(Cm, Cr)
    nB, nC = norm_2(B), norm_2(C)
    lam_high = np.array([float(m.l * (m.l + 2)) for m in high])
    B_lam = B * Ball.exact(lam_high).inv().reshape(1, -1)             # B (-Delta)^{-1} on the high modes
    nB_lam = norm_2(B_lam)
    info = {"d_low": d, "d_high": dh, "max_degree": l + K, "normB_lambda": nB_lam,
            "normB_float": float(np.linalg.norm(Bm, 2)), "normC_float": float(np.linalg.norm(Cm, 2)),
            "normB_frobenius": float(np.linalg.norm(Bm)), "normC_frobenius": float(np.linalg.norm(Cm)),
            "max_radius_A": float(A.r.max()), "max_radius_B": float(Br.max()), "max_radius_C": float(Cr.max()),
            "seconds": time.time() - t0}
    log(f"  ||B|| <= {nB:.4f} (spectral mid {info['normB_float']:.4f}),  ||C|| <= {nC:.4f} (spectral mid {info['normC_float']:.4f})  "
        f"[{dh} high modes up to degree {l + K}, {time.time() - t0:.0f}s]")
    return A, B, C, B_lam, nB, nC, nB_lam, info


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


def L2_inverse_bound_literal(invA, nB, nC, gamma):
    """||L^{-1}||_{L^2 -> L^2} by the literal Schur-complement formula of the draft (may fail)."""
    invD = 1.0 / Ball.exact(gamma)
    b, c, a = Ball.exact(nB), Ball.exact(nC), Ball.exact(invA)
    denom = 1.0 / a - b * invD * c
    if denom.lower() <= 0:
        return None
    num = ((1.0 + b * b * invD * invD) * (1.0 + c * c * invD * invD)).sqrt()
    return float((invD + num / denom).upper())


def L2_inverse_bound_weighted(invA, Z, nE, B_lam, nB_lam, nC, gamma, l, log=print):
    """||L^{-1}||_{L^2 -> L^2} by the Schur complement with the Laplacian-weighted coupling bound."""
    lam = float((l + 1) * (l + 3))
    nZB = norm_2(matmul(Ball(Z), B_lam))                       # ||Z B Lambda^{-1}||
    ratio = Ball.exact(lam) / Ball.exact(gamma)                  # ||(-Delta) D^{-1}||
    AB = Ball.exact(nZB) / (1.0 - Ball.exact(nE))                # ||A^{-1} B (-Delta)^{-1}||
    theta = AB * ratio * Ball.exact(nC)
    assert theta.upper() < 1, f"weighted Schur bound not certified: theta = {theta.upper()}"
    invS = Ball.exact(invA) / (1.0 - theta)
    BD = Ball.exact(nB_lam) * ratio                              # ||B D^{-1}||
    DC = Ball.exact(nC) / Ball.exact(gamma)                      # ||D^{-1} C||
    invD = 1.0 / Ball.exact(gamma)
    Linv = invD + invS * ((1.0 + BD * BD) * (1.0 + DC * DC)).sqrt()
    log(f"  ||Z B Lambda^-1|| <= {nZB:.4e},  ||A^-1 B (-Delta)^-1|| <= {float(AB.upper()):.4e},  "
        f"||(-Delta) D^-1|| <= {float(ratio.upper()):.4f},  theta <= {float(theta.upper()):.4e},  "
        f"||S^-1|| <= {float(invS.upper()):.4f},  ||B D^-1|| <= {float(BD.upper()):.4f},  ||D^-1 C|| <= {float(DC.upper()):.4f}")
    return float(Linv.upper()), {"normZBlam": nZB, "normA_inv_B_lam": float(AB.upper()), "lambda_over_gamma": float(ratio.upper()),
                                 "theta": float(theta.upper()), "normS_inv": float(invS.upper()),
                                 "normBD_inv": float(BD.upper()), "normD_invC": float(DC.upper())}


def M_bound(eta, b1, b0, L2inv):
    """M >= ||L^{-1}||_{Y -> X} (injectivity estimate)."""
    e, b1b, b0b, Linv = Ball.exact(eta), Ball.exact(b1), Ball.exact(b0), Ball.exact(L2inv)
    one = 1.0 - e
    M = 2.0 / one + (1.0 + 2.0 * b0b / one + b1b * b1b / (one * one)) * Linv
    return float(M.upper())
