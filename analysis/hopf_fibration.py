"""Figure: the generalised Hopf fibration of u_q, q = 1..6.

The fibres u_q^{-1}(y) over N points y equally spaced (in polar angle) along one meridian of S^2,
from the north pole to the south pole, in the stereographic projection from (0, i) (as in
pole_fibres.py).  The first and last fibres are the pole fibres; the colours run from red (north) to
blue (south), with a legend giving the latitude of each colour.  The fibres sweep out the
preimage of the meridian, an annulus bounded by the pole fibres (the link T(2, 2q)) that twists q
times.  N is even, so the equator, where the level curve on the orbit sphere runs through the
exceptional orbits z = 0 and w = 0, is avoided.

The fibre through the projection point, over u_q(0, i) = (-1, 0, 0), goes through infinity; the
default meridian, longitude 0, stays at least 90 degrees away from that point, so all fibres are
bounded.  With --classical-q1 the q = 1 panel is drawn in the classical view instead: S^3 is first
rotated by the isometry hopf_view (north fibre -> unit circle, south fibre -> vertical axis) and the
frame is cropped at radius --crop1.  Every fibre is computed on the orbit sphere (analysis/common.py:
level curve of the height, then the 2q branches of the circle angle) and checked to map to y, and
consecutive fibres are checked to link q times.

Usage:  python -m analysis.hopf_fibration      (writes analysis/figures/hopf_fibration.pdf)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.common import (FIG_DIR, QS, Solution, fibre, linking_number, orbit, orbit_data,  # noqa: E402
                             pole_point, stereo)
from analysis.render import Camera, Scene, finish, resample_closed  # noqa: E402
from analysis.style import NORTH, SOUTH, setup  # noqa: E402

NS_CMAP = LinearSegmentedColormap.from_list(
    "north_south", [NORTH, (0.96, 0.55, 0.18), (0.93, 0.80, 0.22), (0.42, 0.72, 0.36), (0.16, 0.66, 0.68), SOUTH])


def hopf_view(z, w):
    """The isometry (z, w) -> ((z + wbar)/sqrt 2, (w - zbar)/sqrt 2) of S^3, used for q = 1: it maps
    the north fibre of the Hopf map (w^2 - zbar^2, 2 Re(z w)) to the circle w = 0 and the south fibre
    to the circle z = 0, which the projection from (0, i) sends to the vertical axis."""
    return (z + np.conj(w)) / np.sqrt(2), (w - np.conj(z)) / np.sqrt(2)


def meridian_points(N, longitude=0.0):
    """N points (t_k, alpha) equally spaced in polar angle along one meridian, from the north pole
    (k = 0) to the south pole (k = N - 1).  The default meridian, longitude 0, keeps every point at
    least 90 degrees from u_q(0, i) = (-1, 0, 0), whose fibre passes through the projection point."""
    th = np.linspace(0, np.pi, N)
    t = np.cos(th)
    t[0], t[-1] = 1.0, -1.0
    return [(float(tk), longitude) for tk in t]


def latitude_label(t0):
    lat = 90.0 - np.degrees(np.arccos(np.clip(t0, -1, 1)))
    if abs(abs(lat) - 90) < 1e-9:
        return "north pole" if lat > 0 else "south pole"
    return f"{abs(lat):.1f}\u00b0{'N' if lat > 0 else 'S'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(FIG_DIR / "hopf_fibration.pdf"))
    ap.add_argument("--points", type=int, default=8,
                    help="number of points on the meridian, poles included (even, so that the equator t = 0, "
                         "where the level curve on the orbit sphere passes through the exceptional orbits, is avoided)")
    ap.add_argument("--longitude", type=float, default=0.0, help="longitude of the meridian (radians)")
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--elev", type=float, default=55.0)
    ap.add_argument("--azim1", type=float, default=20.0, help="camera azimuth for q = 1 (as in pole_fibres.py)")
    ap.add_argument("--elev1", type=float, default=14.0, help="camera elevation for q = 1")
    ap.add_argument("--crop1", type=float, default=3.2, help="half-width of the q = 1 frame in the classical view")
    ap.add_argument("--classical-q1", action="store_true", help="draw q = 1 in the classical view (hopf_view, cropped)")
    args = ap.parse_args()
    setup()
    assert args.points % 2 == 0, "use an even number of points"
    pts = meridian_points(args.points, args.longitude)
    colours = [NS_CMAP(k / (len(pts) - 1))[:3] for k in range(len(pts))]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.7))
    for ax, q in zip(axes.flat, QS):
        sol = Solution(q)
        data = orbit_data(sol)
        s_star, sign = pole_point(sol)
        assert sign > 0
        classical = q == 1 and args.classical_q1
        cam = Camera(args.azim1, args.elev1) if q == 1 else Camera(args.azim, args.elev)
        view = hopf_view if classical else (lambda z, w: (z, w))
        curves, drawn, worst = [], [], 0.0
        for (t0, a), col in zip(pts, colours):
            if abs(t0) == 1.0:                  # the pole fibres are the orbits through (s*, 0), (s*, pi)
                z, w = orbit(q, s_star, 0.0 if t0 > 0 else np.pi, 400 * (q + 1))
                err = np.abs(sol.u(z, w) - np.array([[0], [0], [t0]])).max()
            else:
                cs, d, err = fibre(sol, data, t0, a, n=500)
                assert len(cs) == 1, "fibre not connected"
                z, w = cs[0]
            worst = max(worst, err)
            curves.append((stereo(z, w), col))
            if not (classical and t0 == -1.0):   # in the classical view the south fibre is the vertical axis
                drawn.append((stereo(*view(z, w)), col))
        lks = [linking_number(resample_closed(curves[k][0], 2500), resample_closed(curves[k + 1][0], 2500))
               for k in range(len(curves) - 1)]
        ext = 2 * args.crop1 if classical else np.ptp(np.concatenate([c for c, _ in drawn], axis=1), axis=1).max()
        r = 0.0065 * ext
        sc = Scene()
        if classical:
            h = 1.5 * args.crop1
            sc.add_tube(np.stack([np.zeros(300), np.zeros(300), np.linspace(-h, h, 300)], axis=1), r, colours[-1],
                        sides=10, shine=0.6, closed=False)
        for P, col in drawn:
            L = np.linalg.norm(np.diff(np.concatenate([P, P[:, :1]], axis=1), axis=0), axis=0).sum()
            n = int(np.clip(L / (0.7 * r), 400, 14000))
            sc.add_tube(resample_closed(P, n).T, r, col, sides=10, shine=0.6)
        lo, hi = sc.draw(ax, cam, lw=0.2)
        if classical:
            lo, hi = -np.full(2, args.crop1), np.full(2, args.crop1)
        finish(ax, lo, hi)
        ax.set_title(rf"$q={q}$", pad=2)
        print(f"q={q}: {len(curves)} fibres, max |u - y| = {worst:.1e}, linking numbers of consecutive fibres "
              f"in [{min(lks):+.3f}, {max(lks):+.3f}], extent {ext:.2f}")
        assert worst < 1e-9 and all(abs(abs(v) - q) < 0.05 for v in lks)
    handles = [Line2D([0], [0], color=c, lw=3, solid_capstyle="butt") for c in colours]
    fig.legend(handles, [latitude_label(t0) for t0, _ in pts], loc="lower center", ncol=len(pts), frameon=False,
               fontsize=8, handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07, wspace=0.04, hspace=0.1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
