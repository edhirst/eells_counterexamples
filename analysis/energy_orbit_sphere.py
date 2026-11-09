"""Figure: the energy density of u_q on the sphere of circle orbits S^3/S^1, q = 1..6.

The energy density e = |du_q|^2/2 is G_q-invariant, so it is a function on the orbit space.  Every
orbit meets the slice {(sqrt s, sqrt(1-s) e^{i phi})} once, and the orbit space is drawn as the unit
sphere with height h = 2s - 1 = |z|^2 - |w|^2 (top: the orbit w = 0, bottom: the orbit z = 0, the
exceptional orbit with stabiliser Z_q) and longitude phi = arg(z^q w).  Since dvol = pi ds dphi on
S^3, equal areas on this sphere are equal volumes of S^3 (Archimedes), so the colour, e divided by its
mean E/(2 pi^2), averages to 1 over each sphere.  The red and blue dots are the orbits u_q^{-1}(north)
(at phi = 0) and u_q^{-1}(south) (at phi = pi); the view looks at the meridian phi = pi/2, so both are
visible (for q = 1 they lie on the equator, on the outline).  The symmetries kappa and S of G_q make e
invariant under phi -> -phi and phi -> phi + pi, so the visible half of each sphere shows all of e.
For the Hopf map (q = 1) the density is constant.

Usage:  python -m analysis.energy_orbit_sphere     (writes analysis/figures/energy_orbit_sphere.pdf)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common import FIG_DIR, QS, Solution, orbit_data, pole_point  # noqa: E402
from analysis.render import Camera, Scene, finish, pole_marker, sphere_vertices  # noqa: E402
from analysis.style import ENERGY_CMAP, NORTH, SOUTH, energy_colorbar, energy_norm, setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(FIG_DIR / "energy_orbit_sphere.pdf"))
    ap.add_argument("--azim", type=float, default=90.0)
    ap.add_argument("--elev", type=float, default=35.0)
    args = ap.parse_args()
    setup()
    cmap, norm = plt.get_cmap(ENERGY_CMAP), energy_norm()
    cam = Camera(args.azim, args.elev)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.9))
    for ax, q in zip(axes.flat, QS):
        sol = Solution(q)
        d = orbit_data(sol)
        e, om = d["e"], d["omega"]
        E = float((om * e).sum())
        rel = e / (E / (2 * np.pi ** 2))
        # faces between the nodes: geometric mean of the four corners
        lr = np.log(rel)
        lr = np.concatenate([lr, lr[:, :1]], axis=1)
        face = np.exp((lr[:-1, :-1] + lr[1:, :-1] + lr[:-1, 1:] + lr[1:, 1:]) / 4)
        V = sphere_vertices(-d["x"], d["phi"])          # height 2s - 1 = -x
        sc = Scene()
        sc.add_surface(V, cmap(norm(face))[..., :3], shine=0.25)
        s_star, sign = pole_point(sol)
        h = 2 * s_star - 1
        assert sign > 0                                  # u^{-1}(north) at phi = 0, u^{-1}(south) at phi = pi
        lo, hi = sc.draw(ax, cam, lw=0.15, ambient=0.78, diffuse=0.22, specular=0.08)
        for ph, col in ((0.0, NORTH), (np.pi, SOUTH)):
            pole_marker(ax, cam, [np.sqrt(1 - h * h) * np.cos(ph), np.sqrt(1 - h * h) * np.sin(ph), h], col)
        finish(ax, lo, hi, pad=0.03)
        ax.set_title(rf"$q={q}$", pad=1)
        ax.text(0.5, -0.01, rf"$E/8\pi^2={E / (8 * np.pi ** 2):.4f}$,  $\max\,e/\bar e={rel.max():.2f}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        print(f"q={q}: E/8pi^2 = {E / (8 * np.pi ** 2):.8f} (grid quadrature), e/mean in [{rel.min():.3f}, {rel.max():.3f}], "
              f"pole orbits at height {h:.4f}")
    energy_colorbar(fig, [0.25, 0.1, 0.5, 0.022], r"energy density $e=\frac{1}{2}|du_q|^2$ relative to its mean $\bar e=E/2\pi^2$")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.2, wspace=0.04, hspace=0.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
