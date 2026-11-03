"""The certificate: rigorous verification of every constant of Table A.1 (master-table) of
"A counterexample to Eells' conjecture via a computer-assisted proof", for the approximate
solution Wbar of Hopf invariant q read from a coefficient file (q = 2 is the paper's main case;
q = 3, ..., 6 are the family of Remark rem:family).

Inputs
    --coefficients  JSON with the coefficients of Wbar in the basis of Proposition 2.2 (its header
                    gives the Hopf invariant q)
    --cutoff        the mode cutoff l of the linear estimate (Section 3.3)
    --alpha         the scaling of the reference field V_{q,alpha} (default: REFERENCE_ALPHA[q])

Steps (all arithmetic rigorous: ball arithmetic of eells/ball.py, exact quadrature of eells/grid.py)
    1. grids: a quadrature grid exact for every integrand that is integrated, and a bounding grid
       whose Chebyshev-Fourier coefficients are exact for every function whose sup/inf is bounded;
    2. Proposition 3.1: C_residual = eps, the energy interval, C_homotopy  (approximate_bounds.py);
    3. Section 3.3: eta, b_1, b_0, gamma, ||A^{-1}||, ||B||, ||C||, ||A^{-1} B||, ||L^{-1}||, M
       (linear_estimate.py);
    4. Proposition 3.2 and Theorem 2.10: C_nl, r, and the inequalities M k < 1, M(eps + k r) <= r,
       C_homotopy - c_inf r > 0  (nonlinear_estimate.py);
    5. Proposition 4.2: min |Wbar|, C_energyparameter and the energy enclosure
       [C_energy-lower2, C_energy-upper2] of the harmonic map, with ||w||_{C^0} < 1  (energy_estimate.py);
    6. Section 5, for q = 2 (if data/index_basis_q2.json exists, see numerics/index_basis.py): the
       constants delta and lambda^* and the matrices G, H of the nine test fields, with the check
       that G + delta H is negative definite, so that the Morse index of the harmonic map is >= 9
       (index_estimate.py).
The Sobolev constants c_inf, c_4, c_5, beta of Propositions 2.3 and 2.4 enter as balls enclosing
their exact values (eells/reference.py).
Outputs: certificate/output/<name>.json (all bounds, rounded outward) and <name>.log.

Usage:  python -u -m certificate.certify --coefficients data/Wbar_q2_D40.json --cutoff 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator
from eells.fields import field_from_coefficients, load_coefficients
from eells.equation import degrees
from eells.reference import REFERENCE_ALPHA, SOBOLEV_CONSTANTS_TEX
from . import approximate_bounds as ab
from . import linear_estimate as le
from . import nonlinear_estimate as ne
from . import energy_estimate as ee
from . import index_estimate as ie
from .rounding import round_up, round_down


def grids(D, q, l):
    """(quadrature grid, bounding grid) exact for all integrands / all bounded functions."""
    deg = degrees(D, q)
    p_quad = max(2 * deg["F0"], 2 * (l + deg["Kp"]), deg["energy"] + 4 * D)
    quad = OrbitGrid.for_quadrature(p_quad, q, rigorous=True)
    p_bound = max(2 * deg["M0"], 4 * D, D + 2 * q)
    bound = OrbitGrid.for_analysis(p_bound, q, rigorous=True)
    assert quad.exact_for(p_quad) and bound.analysis_exact_for(p_bound)
    return quad, bound, deg


def certify(coeff_path, l, alpha=None, out_dir="certificate/output", name=None, log=print, index_basis=None,
            chunk_points=6000):
    t0 = time.time()
    modes, values, data = load_coefficients(coeff_path)
    q = modes[0].q
    alpha = alpha or REFERENCE_ALPHA[q]
    D = max(m.l for m in modes)                      # validated against the header by the loader
    name = name or f"q{q}_D{D}_l{l}"
    cache = Path(out_dir) / "cache"
    log(f"Hopf invariant q = {q}:  Wbar of degree {D}, {len(modes)} coefficients;  cutoff l = {l};  reference V_(q,alpha), alpha = {alpha}")
    quad, bound, deg = grids(D, q, l)
    log(f"quadrature grid {quad.Ns + 1} x {quad.Nphi}, bounding grid {bound.Ns + 1} x {bound.Nphi}  ({time.time() - t0:.0f}s)")
    Hq = HarmonicEvaluator(quad, max_degree=max(D, l), cache_dir=str(cache))
    Hb = HarmonicEvaluator(bound, max_degree=D, cache_dir=str(cache))
    Wq = field_from_coefficients(modes, values, Hq, batch=16)
    Wb = field_from_coefficients(modes, values, Hb, batch=16)
    log(f"Wbar evaluated on both grids ({time.time() - t0:.0f}s)")
    out = {"inputs": {"coefficients": str(coeff_path), "hopf_invariant": q, "deg_Wbar": D, "cutoff_l": l, "alpha": alpha,
                      "quadrature_grid": [quad.Ns + 1, quad.Nphi], "bounding_grid": [bound.Ns + 1, bound.Nphi]}}

    # ---- Proposition 3.1
    eps, sq = ab.residual_bound(Wq, quad, deg)
    log(f"[3.1] ||F(0)||_L2 <= {eps:.4e}   (ball of the square: mid {float(sq.m):.3e}, radius {float(sq.r):.1e})  ({time.time() - t0:.0f}s)")
    energy = ab.energy_bounds(Wq, quad, Wb, bound, deg, D)
    log(f"[3.1] energy of Wbar/|Wbar|: [{energy['E_lower']:.15f}, {energy['E_upper']:.15f}]  "
        f"= 8 pi^2 x [{energy['E_over_8pi2_lower']:.15f}, {energy['E_over_8pi2_upper']:.15f}]  ({time.time() - t0:.0f}s)")
    hom_lo, hom_hi = ab.homotopy_bound(Wb, bound, D, q, alpha)
    log(f"[3.1] min Wbar . V >= {hom_lo:.6f}   (max <= {hom_hi:.6f})  ({time.time() - t0:.0f}s)")
    nX = ne.norm_X(modes, values)

    # ---- Section 3.3
    cb = le.coefficient_bounds(Wb, bound, deg, log=log)
    gamma = le.gamma_bound(cb["eta"], cb["b1"], cb["b0"], l)
    log(f"[3.3] gamma >= {gamma:.4f}  (l = {l})  ({time.time() - t0:.0f}s)")
    assert gamma > 0, "gamma <= 0: increase the cutoff l"
    A, BBt, nB, nC, binfo = le.finite_blocks(Wq, quad, Hq, l, q, deg, chunk_points=chunk_points, log=log)
    invA, Z, nE, ainfo = le.inverse_bound(A, log=log)
    L2inv, sinfo = le.L2_inverse_bound(invA, Z, nE, BBt, nB, nC, gamma, log=log)
    M = le.M_bound(cb["eta"], cb["b1"], cb["b0"], L2inv)
    log(f"[3.3] ||L^-1||_(L2->L2) <= {L2inv:.4f},   ||L^-1||_(Y->X) <= M = {M:.4f}  ({time.time() - t0:.0f}s)")

    # ---- Proposition 3.2 / Theorem 2.10
    Cnl = ne.nonlinear_constant(cb["W_sup"])
    nk = ne.newton_kantorovich(M, eps, Cnl, nX, hom_lo)
    log(f"[4]   C_nl <= {Cnl:.5f},  ||Wbar||_X <= {nX:.4f},  r = {nk['r']}:  M k = {nk['Mk']:.3e} < 1 {nk['conditions']['Mk_lt_1']},  "
        f"M(eps + k r) = {nk['M_eps_plus_kr']:.3e} <= r {nk['conditions']['M_eps_plus_kr_le_r']},  "
        f"C_hom - c_inf r = {nk['homotopy_margin']:.5f} > 0 {nk['conditions']['homotopy_margin_positive']}")

    # ---- Proposition 4.2
    minW_lo, maxW_hi = ab.min_norm_bound(Wb, bound, D)
    en = ee.energy_of_harmonic_map(energy["E_lower"], energy["E_upper"], minW_lo, nX, nk["r"])
    log(f"[4.2] min |Wbar| >= {minW_lo:.10f} (max |Wbar| <= {maxW_hi:.10f}),  C_energyparameter >= {en['C_energyparameter']:.10f},  "
        f"||w||_C0 <= {en['w_C0_bound']:.3e} < 1 {en['conditions']['w_C0_lt_1']},  ||dw||_L2 <= {en['dw_L2_bound']:.3e}")
    log(f"[4.2] energy of the harmonic map u: [{en['C_energy_lower2']:.15f}, "
        + (f"{en['C_energy_upper2']:.15f}]" if en["all_ok"] else "no upper bound (||w||_C0 bound >= 1)]") + f"  ({time.time() - t0:.0f}s)")

    # ---- Section 5 (q = 2): nine negative directions (index >= 9), when the test fields exist
    ix = None
    basis = index_basis if index_basis is not None else "data/index_basis_q2.json"
    if q == 2 and Path(basis).exists() and nk["all_ok"] and en["all_ok"]:
        Hix = HarmonicEvaluator(quad, max_degree=max(m.l for f in ie.load_index_basis(basis)[0] for m in f["modes"]),
                                cache_dir=str(cache))
        ix = ie.index_lower_bound(Wq, quad, Hix, Wb, bound, D, basis, nk["r"], nX, en["C_energyparameter"], log=log)
        log(f"[5]   nine negative directions of Q_W certified: {ix['all_ok']}  (Morse index >= 9)  ({time.time() - t0:.0f}s)")
    elif q == 2:
        log(f"[5]   index step skipped (no basis file {basis}, or an earlier step failed)")
    passed = nk["all_ok"] and en["all_ok"] and (ix is None or ix["all_ok"])
    log(f"CERTIFICATE {'PASSED' if passed else 'FAILED'}  ({time.time() - t0:.0f}s)")

    table = [  # (label, latex symbol, value string, kind)
        ("q", r"$H$", str(q), "exact"),
        ("deg Wbar", r"$\deg\bar W$", str(D), "exact"),
        ("l", r"$l$", str(l), "exact"),
        ("alpha", r"$\alpha_q$", str(alpha), "exact"),
        ("c_inf", r"$c_\infty$", SOBOLEV_CONSTANTS_TEX["c_inf"], "exact"),
        ("c_4", r"$c_4$", SOBOLEV_CONSTANTS_TEX["c_4"], "exact"),
        ("c_5", r"$c_5$", SOBOLEV_CONSTANTS_TEX["c_5"], "exact"),
        ("beta", r"$\beta$", SOBOLEV_CONSTANTS_TEX["beta"], "exact"),
        ("C_residual", r"$\Cr{const:residual}$", round_up(eps, 4), "upper"),
        ("C_energy-lower", r"$\Cr{const:energy-lower}$", round_down(energy["E_lower"], 16), "lower"),
        ("C_energy-upper", r"$\Cr{const:energy-upper}$", round_up(energy["E_upper"], 16), "upper"),
        ("C_homotopy", r"$\Cr{const:homotopy}$", round_down(hom_lo, 4), "lower"),
        ("C_nl", r"$\Cr{const:non-linear}$", round_up(Cnl, 4), "upper"),
        ("C_energy-lower2", r"$\Cr{const:energy-lower2}$", round_down(en["C_energy_lower2"], 16), "lower"),
        ("C_energy-upper2", r"$\Cr{const:energy-upper2}$", round_up(en["C_energy_upper2"], 16) if en["all_ok"] else "--", "upper"),
        ("C_energyparameter", r"$\Cr{const:energyparameter}$", round_down(en["C_energyparameter"], 8), "lower"),
        ("norm_X(Wbar)", r"$\|\bar W\|_X$", round_up(nX, 5), "upper"),
        ("sup |Wbar|", r"$\|\bar W\|_{C^0}$", round_up(cb["W_sup"], 6), "upper"),
        ("eta", r"$\eta$", round_up(cb["eta"], 4), "upper"),
        ("b_1", r"$b_1$", round_up(cb["b1"], 4), "upper"),
        ("b_0", r"$b_0$", round_up(cb["b0"], 4), "upper"),
        ("gamma", r"$\gamma$", round_down(gamma, 4), "lower"),
        ("||A^-1||", r"$\|A^{-1}\|_{L^2\to L^2}$", round_up(invA, 4), "upper"),
        ("||B||", r"$\|B\|_{L^2\to L^2}$", round_up(nB, 4), "upper"),
        ("||C||", r"$\|C\|_{L^2\to L^2}$", round_up(nC, 4), "upper"),
        ("||A^-1 B||", r"$\|A^{-1}B\|_{L^2\to L^2}$", round_up(sinfo["normA_inv_B"], 4), "upper"),
        ("||L^-1||_L2", r"$\|\mathscr L^{-1}\|_{L^2\to L^2}$", round_up(L2inv, 4), "upper"),
        ("M", r"$\|\mathscr L^{-1}\|_{Y\to X}$", round_up(M, 4), "upper"),
        ("r", r"$r$", nk["r"], "exact"),
        ("Mk", r"$Mk$", round_up(nk["Mk"], 3), "upper"),
        ("M(eps+kr)", r"$M(\varepsilon+kr)$", round_up(nk["M_eps_plus_kr"], 3), "upper"),
        ("w_C0", r"$\|w\|_{C^0}$", round_up(en["w_C0_bound"], 3), "upper"),
    ]
    if ix is not None:
        table += [("delta", r"$\delta$", round_up(ix["perturbation"]["delta"], 4), "upper"),
                  ("lambda*", r"$\lambda^*$", "-" + ix["t"], "upper")]
    out["table_A1"] = [{"name": a, "latex": b, "value": c, "kind": d} for a, b, c, d in table]
    out["details"] = {"residual_square": {"mid": float(sq.m), "radius": float(sq.r)},
                      "energy": {k: (float(v.m) if hasattr(v, "m") else v) for k, v in energy.items()},
                      "homotopy": {"min_lower": hom_lo, "max_upper": hom_hi, "alpha": alpha}, "coefficient_bounds": cb,
                      "gamma": gamma, "blocks": binfo, "inverse": ainfo, "schur": sinfo, "L2_inverse": L2inv, "M": M,
                      "nonlinear": {"C_nl": Cnl, "norm_X": nX}, "newton_kantorovich": nk,
                      "min_abs_Wbar": {"min_lower": minW_lo, "max_upper": maxW_hi}, "energy_of_u": en, "index": ix,
                      "seconds": time.time() - t0}
    out["passed"] = passed
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {out_dir}/{name}.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coefficients", default="data/Wbar_q2_D40.json")
    ap.add_argument("--cutoff", type=int, default=40, help="mode cutoff l of the linear estimate")
    ap.add_argument("--alpha", type=int, default=None, help="scaling of the reference field (default REFERENCE_ALPHA[q])")
    ap.add_argument("--out-dir", default="certificate/output")
    ap.add_argument("--name", default=None)
    ap.add_argument("--chunk-points", type=int, default=6000, help="grid points per block in the Gram accumulation")
    ap.add_argument("--index-basis", default=None, help="test fields of Section 5 (q = 2; default data/index_basis_q2.json)")
    args = ap.parse_args()
    out = certify(args.coefficients, args.cutoff, args.alpha, args.out_dir, args.name, index_basis=args.index_basis,
                  chunk_points=args.chunk_points)
    print()
    for row in out["table_A1"]:
        print(f"  {row['name']:>22s}  {row['kind']:>5s}  {row['value']}")
    if not out["passed"]:
        print("CERTIFICATE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
