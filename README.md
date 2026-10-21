<p align="center">
  <img src="Eells_image.png" alt="An eel winding between two spheres" width="100%">
</p>

# A counterexample to Eells' conjecture: numerics and computer-assisted certificate

This repository contains the code behind the paper *A counterexample to Eells' conjecture via a
computer-assisted proof* (E. Hirst, D. Platt, H. N. Sá Earp, J. Stein): the construction of the
approximate harmonic map $\bar W:S^3\to\mathbb R^3$ and the rigorous verification of every
constant in Table A.1 of the paper (the Newton–Kantorovich certificate that a genuine
$G$-equivariant harmonic map $S^3\to S^2$ of Hopf invariant $2$ exists near $\bar W/|\bar W|$).

Everything is specific to the Hopf-invariant-$2$ construction of the paper: the symmetry group
$G=O(2)\times\mathbb Z_2$ of Section 2.1, the equivariant spherical-harmonic basis of
Proposition 2.2, the equation $F(v)=0$ of Section 2.2 and the estimates of Section 3.  The code
follows the notation of the paper; each module says which statement it implements.

## Layout

```
eells/            library (runs on plain floats for the numerics and on rigorous balls for the certificate)
  ball.py           midpoint–radius interval arithmetic on numpy arrays (the only source of rigour)
  grid.py           orbit-space grid (|z|^2, phi): exact Clenshaw–Curtis × trapezoid quadrature,
                    Chebyshev–Fourier coefficients and rigorous sup/inf bounds of G-invariant polynomials
  harmonics.py      Jacobi basis Y_{m,n,j}, Lemma 2.1, the G-symmetric modes of Proposition 2.2,
                    values / tangential gradients / Laplacians on the grid
  fields.py         Wbar = sum_k a_k Psi_k from a coefficient file; the polynomial chi
  equation.py       the map F (eq. F-equation), its linearisation (eq. linearisation) and the
                    expanded coefficients P, M_a, M_0 used in Section 3.2
  reference.py      the reference field V_0 of Proposition 2.7, the Hopf map, the Sobolev constants
numerics/         non-rigorous floating point
  solve_approximate.py   Newton–Galerkin construction of Wbar (Section 3.1)
  make_tables.py         LaTeX table of the coefficients of Wbar (appendix)
certificate/      rigorous verification of Table A.1
  approximate_bounds.py  Proposition 3.1: residual, energy, homotopy constant
  linear_estimate.py     Section 3.2: eta, b_1, b_0, gamma, the blocks A, B, C, ||L^{-1}||, M
  nonlinear_estimate.py  Proposition 3.3 and Theorem 2.9: C_nl, the radius r and the final inequalities
  certify.py             driver: runs all steps, writes certificate/output/<name>.json and <name>_tableA1.tex
data/             coefficient files Wbar_D*.json (exact IEEE doubles), chi_constant.json, solver logs
tests/            unit tests (basis orthonormality, Green's identity, operator identities, enclosures, ...)
environment/      requirements.txt and setup instructions
```

## Setup

See [environment/README.md](environment/README.md). In short:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
python -m pytest tests          # ~15 s
```

## Running the code

All commands are run from the repository root.

1. **Approximate solution** (floating point, a few minutes).  Finds $\bar W$ of degree $D$ in the
   span of the basis fields of Proposition 2.2 by Newton's method on the Galerkin system
   $\langle\Psi_k,\,F(0)+\kappa(|\bar W|^2-1)\bar W\rangle=0$, starting from $V_0/|V_0|$, and writes
   the coefficients (exact doubles) together with diagnostics:

   ```bash
   python -m numerics.solve_approximate --degree 40 --out data/Wbar_D40.json
   ```

2. **Certificate** (rigorous, ~10–30 minutes depending on the degree).  Verifies every row of
   Table A.1 for the given $\bar W$, $\chi$ and cutoff $l$, and checks the Newton–Kantorovich
   inequalities $Mk<1$, $M(\varepsilon+kr)\le r$ and $C_{\text{homotopy}}-c_\infty r>0$:

   ```bash
   python -u -m certificate.certify --coefficients data/Wbar_D40.json --chi data/chi_constant.json --cutoff 40
   ```

   Output: `certificate/output/D40_l40.json` (all bounds and intermediate quantities) and
   `certificate/output/D40_l40_tableA1.tex` (Table A.1, every entry rounded outward).

3. **Appendix tables**:

   ```bash
   python -m numerics.make_tables --coefficients data/Wbar_D40.json
   ```

## How the rigour works

* Every quantity of Table A.1 is computed in the ball arithmetic of `eells/ball.py`, which
  encloses rounding errors of IEEE double arithmetic by the standard a-priori bounds (no
  directed rounding, so any platform/BLAS is fine).  The few transcendental constants ($\pi$,
  $\cos$, $\sin$ at rational multiples of $\pi$, the Jacobi-polynomial tables) are enclosed with
  mpmath interval arithmetic at high precision and rounded outward.
* Integrals of $G$-invariant polynomials over $S^3$ are reduced to the orbit space
  $(|z|^2,\varphi)$ and computed by a Clenshaw–Curtis × trapezoidal rule that is **exact** for the
  polynomial degree at hand (the degree of every integrand is tracked in `eells/equation.py`), so
  the only error is rounding, which the balls enclose.
* Suprema and infima (the constants $\eta,b_1,b_0$, $\|\chi\bar W\|_{C^0}$, $C_{\text{homotopy}}$,
  the energy error) come from the exact Chebyshev–Fourier coefficients of the polynomial in
  question: the expansion is evaluated on a fine grid and a Lipschitz remainder is added.
* The coefficients of $\bar W$ are IEEE doubles and are used exactly; the published table is the
  polynomial that was certified.

A note on Section 3.2: the literal Schur-complement bound
$\|A^{-1}\|^{-1}-\|B\|\|D^{-1}\|\|C\|$ is negative for this problem (the coupling norms grow like
$b_1 l$ while $\gamma\sim l^2$), so the certificate uses the weighted form
$\|A^{-1}BD^{-1}C\|\le\|A^{-1}B(-\Delta)^{-1}\|\,\|(-\Delta)D^{-1}\|\,\|C\|$ with
$\|(-\Delta)D^{-1}\|\le(l+1)(l+3)/\gamma$ (which follows from the same lower bound for $D$);
both are reported in the log, see `certificate/linear_estimate.py`.

## License

MIT, see [LICENSE](LICENSE).
