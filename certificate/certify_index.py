"""Stand-alone run of the index certificate of Section 5 (certificate/index_estimate.py), q = 2.

Reads the certified constants r, ||Wbar||_X and C_energyparameter from an existing certificate
JSON (certificate/certify.py writes them), builds grids that are exact for the integrands of
Q_Wbar on the test fields, and verifies that the stored fields span a negative-definite subspace.
The full certificate (certify.py) runs the same step when the basis file exists.

Usage:  python -m certificate.certify_index --coefficients data/Wbar_q2_D40.json \
            --basis data/index_basis_q2.json --certificate certificate/output/q2_D40_l40.json
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
from eells.sectors import load_index_basis
from . import index_estimate as ie


def index_grids(D, L):
    """Quadrature grid exact for degree 4D + 2 + 2L and a bounding grid resolving degree 2D."""
    p = ie.integrand_degrees(D, L)["max"]
    quad = OrbitGrid.for_quadrature(p, 2, rigorous=True)
    bound = OrbitGrid.for_analysis(2 * D, 2, rigorous=True)
    assert quad.exact_for(p) and bound.analysis_exact_for(2 * D)
    return quad, bound


def run(coeff_path, basis_path, cert_path, out_dir="certificate/output", name=None, log=print):
    t0 = time.time()
    modes, values, _ = load_coefficients(coeff_path)
    assert modes[0].q == 2, "the index certificate concerns the Hopf invariant 2"
    D = max(m.l for m in modes)
    fields, _ = load_index_basis(basis_path)
    L = max(max(m.l for m in f["modes"]) for f in fields)
    cert = json.loads(Path(cert_path).read_text())
    det = cert["details"]
    r_str, nX, C8 = det["newton_kantorovich"]["r"], det["nonlinear"]["norm_X"], det["energy_of_u"]["C_energyparameter"]
    assert det["newton_kantorovich"]["all_ok"] and det["energy_of_u"]["all_ok"], \
        "the existence part of the certificate must have passed (r, ||Wbar||_X, C_8 are taken from it)"
    quad, bound = index_grids(D, L)
    cache = Path(out_dir) / "cache"
    Hq = HarmonicEvaluator(quad, max_degree=max(D, L), cache_dir=str(cache))
    Hb = HarmonicEvaluator(bound, max_degree=D, cache_dir=str(cache))
    Wq = field_from_coefficients(modes, values, Hq)
    Wb = field_from_coefficients(modes, values, Hb)
    log(f"Wbar degree {D}, test fields of degree <= {L}; quadrature grid {quad.Ns + 1} x {quad.Nphi}, bounding grid "
        f"{bound.Ns + 1} x {bound.Nphi};  r = {r_str}, ||Wbar||_X <= {nX:.4f}, C_8 >= {C8:.8f}  ({time.time() - t0:.0f}s)")
    ix = ie.index_lower_bound(Wq, quad, Hq, Wb, bound, D, basis_path, r_str, nX, C8, log=log)
    log(f"INDEX CERTIFICATE {'PASSED' if ix['all_ok'] else 'FAILED'}: nine negative directions, Ind >= 9  ({time.time() - t0:.0f}s)")
    name = name or f"q2_D{D}_index"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, f"{name}.json").write_text(json.dumps({"inputs": {"coefficients": str(coeff_path), "basis": str(basis_path),
                                                                     "certificate": str(cert_path)}, "index": ix,
                                                          "passed": ix["all_ok"]}, indent=1, default=float))
    log(f"wrote {out_dir}/{name}.json")
    return ix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coefficients", default="data/Wbar_q2_D40.json")
    ap.add_argument("--basis", default="data/index_basis_q2.json")
    ap.add_argument("--certificate", default="certificate/output/q2_D40_l40.json")
    ap.add_argument("--out-dir", default="certificate/output")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    ix = run(args.coefficients, args.basis, args.certificate, args.out_dir, args.name)
    if not ix["all_ok"]:
        print("INDEX CERTIFICATE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
