"""The certificate: rigorous verification of every constant of Table A.1 (master-table) of
"A counterexample to Eells' conjecture via a computer-assisted proof".

Inputs
    --coefficients  JSON with the coefficients of Wbar in the basis of Proposition 2.2
    --chi           JSON with the G-invariant polynomial chi (default: chi = 1)
    --cutoff        the mode cutoff l of the linear estimate (Section 3.3)

Steps (all arithmetic rigorous: ball arithmetic of eells/ball.py, exact quadrature of eells/grid.py)
    1. grids: a quadrature grid exact for every integrand that is integrated, and a bounding grid
       whose Chebyshev-Fourier coefficients are exact for every function whose sup/inf is bounded;
    2. Proposition 3.1: C_residual = eps, the energy interval, C_homotopy  (approximate_bounds.py);
    3. Section 3.3: eta, b_1, b_0, gamma, ||A^{-1}||, ||B||, ||C||, ||L^{-1}||, M  (linear_estimate.py);
    4. Proposition 3.2 and Theorem 2.10: C_nl, r, and the inequalities M k < 1, M(eps + k r) <= r,
       C_homotopy - c_inf r > 0  (nonlinear_estimate.py);
    5. Proposition 4.2: c_5, min |Wbar|, C_energyparameter and the energy enclosure
       [C_energy-lower2, C_energy-upper2] of the harmonic map, with ||w||_{C^0} < 1  (energy_estimate.py).
Outputs: certificate/output/<name>.json (all bounds, rounded outward) and <name>_tableA1.tex.

Usage:  python -m certificate.certify --coefficients data/Wbar_D40.json --cutoff 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from eells.grid import OrbitGrid
from eells.harmonics import HarmonicEvaluator, Mode
from eells.fields import field_from_coefficients, load_coefficients, load_scalar_coefficients, scalar_from_coefficients
from eells.equation import degrees
from . import approximate_bounds as ab
from . import linear_estimate as le
from . import nonlinear_estimate as ne
from . import energy_estimate as ee
from .rounding import round_up, round_down, sci


def grids(D, Dchi, l):
    """(quadrature grid, bounding grid) exact for all integrands / all bounded functions."""
    deg = degrees(D, Dchi)
    p_quad = max(2 * deg["F0"], 2 * l + deg["K"] + deg["Kp"], deg["energy"] + 4 * D)
    quad = OrbitGrid(-(-p_quad // 2), p_quad // 3 + 1, rigorous=True)
    p_bound = max(2 * deg["M0"], 4 * D, D + 4, 2 * deg["chi"] + 2 * D)
    bound = OrbitGrid(p_bound // 2, 2 * (p_bound // 3) + 1, rigorous=True)
    assert quad.exact_for(p_quad) and bound.analysis_exact_for(p_bound)
    return quad, bound, deg


def certify(coeff_path, chi_path, l, out_dir="certificate/output", name=None, batch=48, log=print):
    t0 = time.time()
    modes, values, data = load_coefficients(coeff_path)
    D = max(m.l for m in modes)                      # validated against the header by the loader
    chi_modes, chi_values, _ = load_scalar_coefficients(chi_path)
    Dchi = max(m.l for m in chi_modes)
    name = name or f"D{D}_l{l}"
    cache = Path(out_dir) / "cache"
    log(f"Wbar: degree {D}, {len(modes)} coefficients;  chi: degree {Dchi};  cutoff l = {l}")
    quad, bound, deg = grids(D, Dchi, l)
    log(f"quadrature grid {quad.Ns + 1} x {quad.Nphi}, bounding grid {bound.Ns + 1} x {bound.Nphi}  ({time.time() - t0:.0f}s)")
    Hq = HarmonicEvaluator(quad, max_degree=l + deg["K"], cache_dir=str(cache))
    Hb = HarmonicEvaluator(bound, max_degree=D, cache_dir=str(cache))
    Wq = field_from_coefficients(modes, values, Hq)
    Wb = field_from_coefficients(modes, values, Hb)
    chiq = scalar_from_coefficients(chi_modes, chi_values, Hq)
    chib = scalar_from_coefficients(chi_modes, chi_values, Hb)
    log(f"Wbar evaluated on both grids ({time.time() - t0:.0f}s)")
    out = {"inputs": {"coefficients": str(coeff_path), "chi": str(chi_path), "deg_Wbar": D, "deg_chi": Dchi, "cutoff_l": l,
                      "quadrature_grid": [quad.Ns + 1, quad.Nphi], "bounding_grid": [bound.Ns + 1, bound.Nphi]}}

    # ---- Proposition 3.1
    eps, sq = ab.residual_bound(Wq, chiq, quad, deg)
    log(f"[3.1] ||F(0)||_L2 <= {eps:.4e}   (ball of the square: mid {float(sq.m):.3e}, radius {float(sq.r):.1e})  ({time.time() - t0:.0f}s)")
    energy = ab.energy_bounds(Wq, quad, Wb, bound, deg, D)
    log(f"[3.1] energy of Wbar/|Wbar|: [{energy['E_lower']:.15f}, {energy['E_upper']:.15f}]  "
        f"= 8 pi^2 x [{energy['E_over_8pi2_lower']:.15f}, {energy['E_over_8pi2_upper']:.15f}]  ({time.time() - t0:.0f}s)")
    hom_lo, hom_hi = ab.homotopy_bound(Wb, bound, D)
    log(f"[3.1] min Wbar . V0 >= {hom_lo:.6f}   (max <= {hom_hi:.6f})  ({time.time() - t0:.0f}s)")
    nX = ne.norm_X(modes, values)

    # ---- Section 3.3
    cb = le.coefficient_bounds(Wb, chib, bound, deg, log=log)
    gamma = le.gamma_bound(cb["eta"], cb["b1"], cb["b0"], l)
    log(f"[3.3] gamma >= {gamma:.4f}  (l = {l})")
    assert gamma > 0, "gamma <= 0: increase the cutoff l"
    A, B, C, B_lam, nB, nC, nB_lam, binfo = le.finite_blocks(Wq, chiq, quad, Hq, l, deg, batch=batch, log=log)
    invA, Z, nE, ainfo = le.inverse_bound(A, log=log)
    L2inv_literal = le.L2_inverse_bound_literal(invA, nB, nC, gamma)
    log(f"[3.3] literal Schur bound ||A^-1||^-1 - ||B|| ||D^-1|| ||C|| = {1 / invA - nB * nC / gamma:.4f}: "
        + (f"||L^-1|| <= {L2inv_literal:.4f}" if L2inv_literal else "NOT certified (negative)"))
    L2inv_w, winfo = le.L2_inverse_bound_weighted(invA, Z, nE, B_lam, nB_lam, nC, gamma, l, log=log)
    log(f"[3.3] weighted Schur bound: ||L^-1||_(L2->L2) <= {L2inv_w:.4f}")
    L2inv, sinfo = le.L2_inverse_bound_paper(invA, Z, nE, B, nB, nC, gamma, log=log)
    M = le.M_bound(cb["eta"], cb["b1"], cb["b0"], L2inv)
    log(f"[3.3] ||L^-1||_(L2->L2) <= {L2inv:.4f} (paper's Schur bound with ||A^-1 B||),   ||L^-1||_(Y->X) <= M = {M:.4f}  ({time.time() - t0:.0f}s)")

    # ---- Proposition 3.2 / Theorem 2.10
    Cnl, c_inf, c_4 = ne.nonlinear_constant(cb["chiW_sup"])
    nk = ne.newton_kantorovich(M, eps, Cnl, nX, hom_lo, c_inf)
    log(f"[4]   C_nl <= {Cnl:.5f},  ||Wbar||_X <= {nX:.4f},  r = {nk['r']}:  M k = {nk['Mk']:.3e} < 1 {nk['conditions']['Mk_lt_1']},  "
        f"M(eps + k r) = {nk['M_eps_plus_kr']:.3e} <= r {nk['conditions']['M_eps_plus_kr_le_r']},  "
        f"C_hom - c_inf r = {nk['homotopy_margin']:.5f} > 0 {nk['conditions']['homotopy_margin_positive']}")

    # ---- Proposition 4.2
    minW_lo, maxW_hi = ab.min_norm_bound(Wb, bound, D)
    en = ee.energy_of_harmonic_map(energy["E_lower"], energy["E_upper"], minW_lo, nX, nk["r"])
    log(f"[4.2] min |Wbar| >= {minW_lo:.10f} (max |Wbar| <= {maxW_hi:.10f}),  C_energyparameter >= {en['C_energyparameter']:.10f},  "
        f"||w||_C0 <= {en['w_C0_bound']:.3e} < 1 {en['conditions']['w_C0_lt_1']},  ||dw||_L2 <= {en['dw_L2_bound']:.3e},  c_5 <= {en['c_5']:.6f}")
    log(f"[4.2] energy of the harmonic map u: [{en['C_energy_lower2']:.15f}, "
        + (f"{en['C_energy_upper2']:.15f}]" if en["all_ok"] else "no upper bound (||w||_C0 bound >= 1)]") + f"  ({time.time() - t0:.0f}s)")
    passed = nk["all_ok"] and en["all_ok"]
    log(f"CERTIFICATE {'PASSED' if passed else 'FAILED'}  ({time.time() - t0:.0f}s)")

    table = [  # (label, latex symbol, value string, kind)
        ("deg Wbar", r"$\deg\bar W$", str(D), "exact"),
        ("deg chi", r"$\deg\chi$", str(Dchi), "exact"),
        ("l", r"$l$", str(l), "exact"),
        ("c_inf", r"$c_\infty$", round_up(c_inf, 6), "upper"),
        ("c_4", r"$c_4$", round_up(c_4, 6), "upper"),
        ("c_5", r"$c_5$", round_up(en["c_5"], 6), "upper"),
        ("C_residual", r"$\Cr{const:residual}$", round_up(eps, 4), "upper"),
        ("C_energy-lower", r"$\Cr{const:energy-lower}$", round_down(energy["E_lower"], 16), "lower"),
        ("C_energy-upper", r"$\Cr{const:energy-upper}$", round_up(energy["E_upper"], 16), "upper"),
        ("C_homotopy", r"$\Cr{const:homotopy}$", round_down(hom_lo, 4), "lower"),
        ("C_nl", r"$\Cr{const:non-linear}$", round_up(Cnl, 4), "upper"),
        ("C_energy-lower2", r"$\Cr{const:energy-lower2}$", round_down(en["C_energy_lower2"], 16), "lower"),
        ("C_energy-upper2", r"$\Cr{const:energy-upper2}$", round_up(en["C_energy_upper2"], 10) if en["all_ok"] else "--", "upper"),
        ("C_energyparameter", r"$\Cr{const:energyparameter}$", round_down(en["C_energyparameter"], 8), "lower"),
        ("norm_X(Wbar)", r"$\|\bar W\|_X$", round_up(nX, 5), "upper"),
        ("sup |chi Wbar|", r"$\|\chi\bar W\|_{C^0}$", round_up(cb["chiW_sup"], 6), "upper"),
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
    ]
    out["table_A1"] = [{"name": a, "latex": b, "value": c, "kind": d} for a, b, c, d in table]
    out["details"] = {"residual_square": {"mid": float(sq.m), "radius": float(sq.r)}, "energy": {k: (float(v.m) if hasattr(v, "m") else v) for k, v in energy.items()},
                      "homotopy": {"min_lower": hom_lo, "max_upper": hom_hi}, "coefficient_bounds": cb, "gamma": gamma,
                      "blocks": binfo, "inverse": ainfo, "schur_paper": sinfo, "L2_inverse": L2inv,
                      "schur_weighted": winfo, "L2_inverse_weighted_schur": L2inv_w,
                      "L2_inverse_literal_schur": L2inv_literal, "M": M, "nonlinear": {"C_nl": Cnl, "c_inf": c_inf, "c_4": c_4, "norm_X": nX},
                      "newton_kantorovich": nk, "min_abs_Wbar": {"min_lower": minW_lo, "max_upper": maxW_hi},
                      "energy_of_u": en, "seconds": time.time() - t0}
    out["passed"] = passed
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    Path(out_dir, f"{name}_tableA1.tex").write_text(latex_table(table, D, Dchi, l, nk, en))
    log(f"wrote {out_dir}/{name}.json and {out_dir}/{name}_tableA1.tex")
    return out


def latex_table(table, D, Dchi, l, nk, en):
    rows = []
    for name, sym, val, kind in table:
        rel = {"upper": r"$\le$", "lower": r"$\ge$", "exact": "$=$"}[kind]
        v = sci(val)
        if name == "C_residual":
            rows.append(f"{sym} & $=\\varepsilon\\le {v}$\\\\")
        elif name == "M":
            rows.append(f"{sym} & $\\le M={v}$\\\\")
        else:
            rows.append(f"{sym} & {rel} ${v}$\\\\")
    body = "\n".join(rows)
    chi = r" and $\chi\equiv 1$" if Dchi == 0 else f" and $\\deg\\chi={Dchi}$"
    checks = (f"$Mk={sci(round_up(nk['Mk'], 3))}<1$, $M(\\varepsilon+kr)={sci(round_up(nk['M_eps_plus_kr'], 3))}\\le r$, "
              f"$\\Cr{{const:homotopy}}-c_\\infty r>0$ and $\\|w\\|_{{C^0}}\\le{sci(round_up(en['w_C0_bound'], 3))}<1$")
    caption = (f"Constants for the certificate, for $\\deg\\bar W={D}$, $l={l}${chi}; upper bounds are rounded up and "
               f"lower bounds down. With these constants {checks}.")
    return (f"% generated by certificate/certify.py (deg Wbar = {D}, cutoff l = {l}); every value is an outward-rounded rigorous bound\n"
            "\\begin{table}[H]\n\\centering\n\\begin{tabular}{lc}\n\\toprule\nVariable & Certified value or bound\\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{master-table}}\n\\end{{table}}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coefficients", default="data/Wbar_D40.json")
    ap.add_argument("--chi", default="data/chi_constant.json")
    ap.add_argument("--cutoff", type=int, default=40, help="mode cutoff l of the linear estimate")
    ap.add_argument("--out-dir", default="certificate/output")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    out = certify(args.coefficients, args.chi, args.cutoff, args.out_dir, args.name, args.batch)
    print()
    for row in out["table_A1"]:
        print(f"  {row['name']:>22s}  {row['kind']:>5s}  {row['value']}")
    if not out["passed"]:
        print("CERTIFICATE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
