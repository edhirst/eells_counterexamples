# Data files

* `Wbar_q<q>_D<D>.json` — coefficients $a_k$ of the approximate solution $\bar W_q=\sum_k a_k\Psi_k$
  of Hopf invariant $q$ and degree $D$ in the orthonormal basis of Proposition 2.2 (for $q\ne2$:
  Remark rem:family), one entry per basis field: type U/V, $n$, $j$, degree $l$, value.  For type U
  the field is $(\mathrm{Re}\,Y_{m,n,j},\mathrm{Im}\,Y_{m,n,j},0)$ with $m=qn-2q$, $n$ even; for type V
  it is $\sqrt2(0,0,\mathrm{Re}\,Y_{m,n,j})$ with $m=qn$, $n$ odd.  The values are IEEE doubles and
  are used exactly by the certificate.  The header gives `hopf_invariant` and `degree`; the `meta`
  field records the solver settings, the Newton history and floating-point diagnostics.

  | file | $q$ | $D$ | coefficients |
  |---|---|---|---|
  | `Wbar_q2_D40.json`  | 2 | 40  | 217  |
  | `Wbar_q3_D80.json`  | 3 | 80  | 630  |
  | `Wbar_q4_D112.json` | 4 | 112 | 969  |
  | `Wbar_q5_D152.json` | 5 | 152 | 1482 |
  | `Wbar_q6_D184.json` | 6 | 184 | 1847 |

  Produced by `numerics/solve_approximate.py`: for $q=2$ directly from the projection of $V_2/|V_2|$,
  for $q\ge3$ by continuation in the degree from a low-degree solution in `start/` (see below).
* `start/Wbar_q<q>_D<D>.json` — the intermediate solutions of the continuation: the first one of
  each $q$ is computed from the projection of $V_q/|V_q|$, each further one starts from the previous
  degree (`--start`).
* `*.log` — the solver logs.
* `index_basis_q2.json` — the five test fields $f_1,\dots,f_5$ of Proposition prop:indexbound
  ($q=2$): for each, its sector $(\nu,p)$ and its Gaussian-integer coefficients on the modes
  $P_{m,n,j},M_{m,n,j},T_{m,n,j}$ (eq. basisPMT) of degree $\le8$ (Table tab:index-fields).
  Produced by `python -m numerics.index_basis`.
