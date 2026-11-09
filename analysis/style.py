"""Common figure style."""
import matplotlib

NORTH = (0.84, 0.25, 0.27)      # u^{-1}(north pole)
SOUTH = (0.18, 0.47, 0.71)      # u^{-1}(south pole)


def setup():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
    })


# energy density relative to its mean, on a log scale symmetric about 1 (the Hopf map is uniformly 1)
ENERGY_CMAP = "RdBu_r"
ENERGY_RANGE = 16.0


def energy_norm():
    from matplotlib.colors import LogNorm
    return LogNorm(vmin=1 / ENERGY_RANGE, vmax=ENERGY_RANGE)


def energy_colorbar(fig, rect, label):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes(rect)
    sm = ScalarMappable(norm=energy_norm(), cmap=ENERGY_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
    cb.set_ticks(ticks)
    cb.set_ticklabels([r"$\frac{1}{16}$", r"$\frac{1}{8}$", r"$\frac{1}{4}$", r"$\frac{1}{2}$", "1", "2", "4", "8", "16"])
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.set_label(label, fontsize=8.5)
    plt.setp(cb.outline, linewidth=0.5)
    return cb
