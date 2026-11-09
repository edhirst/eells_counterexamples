"""Figure: the energy of u_q distributed over the target S^2, q = 1..6.

The push-forward of the energy measure e dvol (e = |du_q|^2/2) to S^2 has the density
e_*(y) = sum over the fibre of e/J (coarea formula, J the normal Jacobian of u_q), i.e. the energy
carried by the fibre over y.  By the equivariance of u_q it depends only on the height t of y
(every map of the family commutes with the rotations about the vertical axis); the figure shows
the profile t -> e_*(t) / mean for q = 1..6.  Since the area of S^2 between heights t and
t + dt is 2 pi dt,  e_*(t) = (1/2 pi) d/dt  int_{u_3 <= t} e dvol;  it is computed from the
orbit-space grid (upsampled 4 x 16 times, see :func:`upsample`) by binning in t and smoothing with a
Gaussian of width sigma (reflected at t = +-1), and normalised by its mean E/(4 pi).  For the Hopf
map e_* is constant (= 2 pi).  For q >= 2, e_* is singular at the
equator: near the exceptional orbit z = 0 the height is t ~ |z|^q cos(phi) while the volume is ~ |z|^2,
so e_*(t) ~ |t|^{2/q - 1} (log|t| for q = 2) as t -> 0 (the fitted slopes approach these exponents);
the plotted peak is therefore set by sigma.

Usage:  python -m analysis.energy_target_sphere     (writes analysis/figures/energy_target_sphere.pdf)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common import FIG_DIR, QS, Solution, height_and_zeta, orbit_data  # noqa: E402
from analysis.style import setup  # noqa: E402


def upsample(f, mx, mphi):
    """Values on the orbit grid (Chebyshev-Lobatto in x, uniform in phi) -> values on the grid with
    mx times as many x-intervals and mphi times as many phi-nodes, by Chebyshev interpolation in x
    and trigonometric interpolation in phi (exact for the polynomial Wbar; the energy density and the
    height are smooth, so the error is far below the resolution of the figure)."""
    from scipy.fft import dct, irfft, rfft
    N, Nphi = f.shape[0] - 1, f.shape[1]
    F = rfft(f, axis=1)
    f = irfft(F, n=Nphi * mphi, axis=1) * mphi
    c = dct(f, type=1, axis=0) / N
    c[0] /= 2
    c[-1] /= 2
    M = N * mx
    T = np.cos(np.pi * np.outer(np.arange(M + 1), np.arange(N + 1)) / M)
    return T @ c


def clenshaw_curtis_s(M):
    """Weights w_i with int_0^1 g ds ~ sum_i w_i g(s_i), s_i = (1 - cos(pi i / M))/2."""
    e = np.ones(M + 1)
    e[0] = e[-1] = 0.5
    k = np.arange(M + 1)
    D = (2.0 / M) * np.outer(e, e) * np.cos(np.pi * np.outer(k, k) / M)
    kk = k.astype(float)
    ik = np.zeros(M + 1)
    ik[k % 2 == 0] = 1.0 / (1.0 - kk[k % 2 == 0] ** 2)
    return ik @ D


def pushforward_density(d, ts, sigma, mx=4, mphi=16):
    """e_*(t) at the heights ts and the energy E = int e dvol: the energy measure binned by the height
    t of u_q on an upsampled orbit grid, smoothed with a Gaussian of width sigma (reflected at +-1)."""
    t, _ = height_and_zeta(d["W"])
    tf, ef = upsample(t, mx, mphi), upsample(d["e"], mx, mphi)
    M, Nphi = tf.shape[0] - 1, tf.shape[1]
    omega = np.pi * clenshaw_curtis_s(M)[:, None] * (2 * np.pi / Nphi)
    E = float((omega * ef).sum())
    nb = 4000
    hist, edges = np.histogram(np.clip(tf, -1, 1), bins=nb, range=(-1, 1), weights=omega * ef)
    centres = (edges[:-1] + edges[1:]) / 2
    dens = hist / (2 * np.pi * (2.0 / nb))
    K = lambda v: np.exp(-0.5 * (v / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    es = np.array([(dens * (K(t0 - centres) + K(t0 - (2 - centres)) + K(t0 - (-2 - centres)))).sum() * (2.0 / nb)
                   for t0 in ts])
    return es, E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(FIG_DIR / "energy_target_sphere.pdf"))
    ap.add_argument("--sigma", type=float, default=0.005, help="kernel width in t")
    args = ap.parse_args()
    setup()
    ts = np.linspace(-1, 1, 401)
    fig, prof = plt.subplots(figsize=(4.8, 3.2))
    lines = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(QS)))
    for k, q in enumerate(QS):
        sol = Solution(q)
        es, E = pushforward_density(orbit_data(sol), ts, args.sigma)
        rel = es / (E / (4 * np.pi))
        total = np.trapz(es, ts) * 2 * np.pi / E
        print(f"q={q}: e_*/mean in [{rel.min():.3f}, {rel.max():.3f}] (max at t = {ts[rel.argmax()]:+.3f}), "
              f"int e_* dA / E = {total:.6f}")
        assert abs(total - 1) < 2e-3
        prof.plot(ts, rel, color=lines[k], lw=1.2, label=rf"$q={q}$", ls="--" if q == 1 else "-")
    prof.set_yscale("log")
    prof.set_ylim(1 / 4, 32)
    prof.set_xlim(-1, 1)
    prof.set_yticks([1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32])
    prof.set_yticklabels([r"$\frac{1}{4}$", r"$\frac{1}{2}$", "1", "2", "4", "8", "16", "32"])
    prof.minorticks_off()
    prof.set_xlabel(r"height $t$ on $S^2$")
    prof.set_ylabel(r"$e_*(t)\,/\,\bar e_*$")
    prof.legend(loc="upper right", frameon=False, fontsize=8)
    prof.tick_params(labelsize=8)
    for sp in ("top", "right"):
        prof.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
