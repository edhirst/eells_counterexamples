# Data files

* `Wbar_D<D>.json` — coefficients $a_k$ of the approximate solution $\bar W=\sum_k a_k\Psi_k$ of
  degree $D$ in the orthonormal basis of Proposition 2.2 (each entry: kind U/V, $n$, $j$, degree
  $l$, value).  The values are IEEE doubles and are used exactly by the certificate.  The `meta`
  field records the solver settings, the Newton history and floating-point diagnostics
  (energy, residual, $\sup||\bar W|^2-1|$, $\min\bar W\cdot V_0$, $\|\bar W\|_X$).
  Produced by `python -m numerics.solve_approximate --degree <D>`.
* `chi_constant.json` — the auxiliary polynomial $\chi\equiv1$ (degree 0), in the unnormalised
  invariant basis $\mathrm{Re}(z^{2n}w^n)P_j^{(2n,n)}(1-2|z|^2)$.
* `solve_D<D>.log` — solver logs.
