# Figures of the maps $u_q$, $q=1,\dots,6$

Scripts that draw the maps $u_q=\bar W_q/|\bar W_q|:S^3\to S^2$ for the certified coefficient files
`data/Wbar_q<q>_D<D>.json`. For $q=1$ this map is the Hopf map; see the main README.
Run them from the repository root. They need `matplotlib` in addition to the packages of
`environment/requirements.txt` (`pip install matplotlib`).

```bash
python -m analysis                       # all four figures (about 2 min on the first run)
python -m analysis.pole_fibres           # each one separately; --help lists the options
python -m analysis.hopf_fibration
python -m analysis.energy_orbit_sphere
python -m analysis.energy_target_sphere
```

The PDFs are written to `analysis/figures/`. The 3D pictures are rasterised at 400 dpi; the text is vector.
`pole_fibres` also writes `pole_fibres.png` (130 dpi), the image at the top of the main README.
The values of $\bar W_q$ and its energy density on the orbit-space grid are cached in
`analysis/cache/` (git-ignored).

| figure | what it shows |
|---|---|
| `pole_fibres.pdf` | $u_q^{-1}(\text{north})$ (red) and $u_q^{-1}(\text{south})$ (blue) in $\mathbb R^3$, with their linking number under each $q$. Both are circle orbits, two parallel $(1,q)$ curves on a torus, so together they form the torus link $T(2,2q)$, with linking number $q$. For $q=1$ this is the Hopf link. |
| `hopf_fibration.pdf` | Fibres over 8 points equally spaced along one meridian of $S^2$, from the north pole to the south pole; the first and last are the pole fibres. The colour runs from red (north) to blue (south); the legend gives the latitude of each colour. The fibres sweep out the preimage of the meridian, an annulus bounded by the pole link $T(2,2q)$ that twists $q$ times; consecutive fibres link $q$ times. |
| `energy_orbit_sphere.pdf` | Energy density $e=\frac12\lvert du_q\rvert^2$ relative to its mean $E/2\pi^2$, on the sphere of circle orbits $S^3/S^1$ (log colour scale). The red and blue dots are the orbits $u_q^{-1}(\text{north})$ and $u_q^{-1}(\text{south})$. |
| `energy_target_sphere.pdf` | The profile $t\mapsto e_*(t)/\bar e_*$: the energy carried by the fibre over a point of height $t$ of the target $S^2$, relative to its mean $E/4\pi$ (by symmetry $e_*$ depends only on $t$). |

## Geometry used

* **Orbit sphere.** The circle of $G_q$ acts by $(e^{i\theta}z,e^{-iq\theta}w)$ on $S^3$ and by
  $\zeta\mapsto e^{-2iq\theta}\zeta$ on the target.
  * Every orbit meets the slice $\{(\sqrt s,\sqrt{1-s}\,e^{i\varphi})\}$ exactly once. So $u_q$ is determined by its values there, and $u_q(z,w)$ is evaluated from the slice using the equivariance (`analysis/common.py`).
  * The orbit space is a 2-sphere, with height $|z|^2-|w|^2=2s-1$ and longitude $\varphi=\arg(z^qw)$.
  * Since $d\mathrm{vol}_{S^3}=\pi\,ds\,d\varphi$, equal areas on this sphere are equal volumes in $S^3$ (Archimedes).
  * The height of $u_q$ and its energy density are invariant under the circle, so they are functions on this sphere.
* **Fibres.**
  * A pole fibre is the circle orbit through a zero of $\zeta$ on the orbit sphere. There are exactly two such zeros, $(s^*,0)$ and $(s^*,\pi)$.
  * The fibre over $y=(\sqrt{1-t_0^2}e^{i\alpha_0},t_0)$ consists of the points $e^{i\theta}\cdot p$, for $p$ on the level curve $\{t=t_0\}$ of the orbit sphere, with $\arg\zeta(p)-2q\theta=\alpha_0$.
  * The $2q$ branches of $\theta$ close up into one curve, because $\zeta$ winds once around the level curve.
* **$\mathbb R^3$.**
  * The pictures in $\mathbb R^3$ use stereographic projection from $(z,w)=(0,i)$. There the tori $|z|=$ const are tori of revolution about the vertical axis, the image of the circle $z=0$.
  * For $q\ge2$ this point is as far as possible from the north fibre.
  * Its image $u_q(0,i)=(-1,0,0)$ is the same for every $q$. The fibre over that point passes through infinity; the meridian used in `hopf_fibration.pdf` (longitude $0$) stays at least $90^\circ$ from it, so all the fibres drawn are bounded.
  * With `--classical-q1`, the $q=1$ panel is drawn in the classical view: $S^3$ is first rotated by the isometry $(z,w)\mapsto((z+\bar w)/\sqrt2,(w-\bar z)/\sqrt2)$, which makes the north fibre the unit circle and the south fibre the vertical axis, and the frame is cropped.

## Checks made by the scripts

The scripts stop with an error if any of these fails:
* **Evaluation:** evaluating $\bar W_q$ through the slice agrees with the direct formula $Y_{m,n,j}=c\,z^{(m)}w^{(n)}P_j(1-2|z|^2)$ to about $10^{-14}$.
* **Pole fibres:** they map to the poles to $10^{-13}$.
* **Other fibres:** every computed fibre satisfies $|u_q-y|\le10^{-13}$.
* **Linking numbers:** the Gauss linking integrals come out as $\pm q$ to within $10^{-3}$ (the two pole fibres; consecutive fibres along the meridian).
* **Energy:** the energies obtained from the grid agree with the certified ones to 8 digits.

## Observations

* **Where the energy sits.** For $q\ge2$ the energy concentrates at the orbit $w=0$, where $u_q=(c\bar z^{2q},0)$ winds $2q$ times round the equator.
  * The maximum of $e$ is $\approx4.1q^2$: $16.2, 36.7, 65.4, 102.3, 147.4$ for $q=2,\dots,6$, i.e. $2.6$ to $13.9$ times the mean.
  * The minimum, $e=2$, is at the exceptional orbit $z=0$.
  * The Hopf map has constant density $e=4$.
  * $e$ depends mainly on the height of the orbit sphere.
* **Fibres over the equator.** For $q\ge2$ the density $e_*$ on the target is singular at the equator.
  * Near $z=0$ the height of $u_q$ is $\sim|z|^q\cos\varphi$, while the volume is $\sim|z|^2$.
  * Hence $e_*(t)\sim|t|^{2/q-1}$ ($\log|t|$ for $q=2$). The fitted slopes approach these exponents.
  * The peak drawn in `energy_target_sphere.pdf` is therefore set by the smoothing width $\sigma$ (default $0.005$, option `--sigma`).
