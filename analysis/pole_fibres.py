"""Figure: the linked pole fibres u_q^{-1}(north) and u_q^{-1}(south), q = 1..6.

Both are circle orbits (the zeros of zeta on the orbit sphere are the two points (s*, 0) and
(s*, pi)), i.e. two parallel (1, q) curves on the torus |z|^2 = s*: the torus link T(2, 2q), with
linking number q = the Hopf invariant (q = 1: the Hopf link).  Drawn in the stereographic
projection from (0, i), where the tori |z| = const are tori of revolution about the vertical axis.
The script checks that both curves are mapped to the poles and computes their Gauss linking number.

Usage:  python -m analysis.pole_fibres          (writes analysis/figures/pole_fibres.pdf, and a PNG
                                                 of it for the README)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common import FIG_DIR, QS, Solution, linking_number, orbit, pole_point, stereo  # noqa: E402
from analysis.render import Camera, Scene, finish  # noqa: E402
from analysis.style import NORTH, SOUTH, setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(FIG_DIR / "pole_fibres.pdf"))
    ap.add_argument("--png-dpi", type=int, default=130, help="resolution of the PNG copy (0: no PNG)")
    ap.add_argument("--azim", type=float, default=30.0, help="camera azimuth for q >= 2 (degrees)")
    ap.add_argument("--elev", type=float, default=55.0, help="camera elevation for q >= 2 (degrees)")
    args = ap.parse_args()
    setup()
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.7))
    for ax, q in zip(axes.flat, QS):
        sol = Solution(q)
        s_star, sign = pole_point(sol)
        n = 400 * (q + 1)
        zN, wN = orbit(q, s_star, 0.0, n)
        zS, wS = orbit(q, s_star, np.pi, n)
        if sign < 0:
            zN, wN, zS, wS = zS, wS, zN, wN
        errN = np.abs(sol.u(zN, wN) - np.array([[0], [0], [1]])).max()
        errS = np.abs(sol.u(zS, wS) - np.array([[0], [0], [-1]])).max()
        PN, PS = stereo(zN, wN), stereo(zS, wS)
        lk = linking_number(PN, PS)
        rho = np.sqrt((1 - s_star) / s_star)               # minor radius of the projected torus
        r = min(0.11, 0.3 * rho)
        # q = 1: the two Hopf circles lie in the planes x1 = +-x3, so look along x1 to see both open
        cam = Camera(20.0, 14.0) if q == 1 else Camera(args.azim, args.elev)
        sc = Scene()
        sc.add_tube(PN.T, r, NORTH, sides=16)
        sc.add_tube(PS.T, r, SOUTH, sides=16)
        lo, hi = sc.draw(ax, cam)
        finish(ax, lo, hi)
        ax.set_title(rf"$q={q}$", pad=13)
        ax.text(0.5, 1.0, rf"linking number ${abs(round(lk))}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        print(f"q={q}: s*={s_star:.6f}, |u - north| <= {errN:.1e}, |u - south| <= {errS:.1e}, Gauss linking integral {lk:+.5f}")
        assert errN < 1e-9 and errS < 1e-9 and abs(abs(lk) - q) < 0.05
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.25)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    print(f"wrote {args.out}")
    if args.png_dpi:
        png = str(Path(args.out).with_suffix(".png"))
        fig.savefig(png, dpi=args.png_dpi)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
