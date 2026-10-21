# Environment

The code is pure Python and needs only NumPy, SciPy (used in the tests, for independent
reference values of the Jacobi polynomials) and mpmath (high-precision interval arithmetic for
the few transcendental constants: pi, cosines and sines of rational multiples of pi, and the
Jacobi-polynomial tables).  Python 3.10 or newer is required.

## Virtual environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows:  .venv\Scripts\activate
pip install --upgrade pip
pip install -r environment/requirements.txt
```

Check the installation by running the unit tests (about one minute):

```bash
python -m pytest tests
```

All scripts are run from the repository root as modules (`python -m numerics.solve_approximate`,
`python -m certificate.certify`, ...), so no installation of the `eells` package is needed.

## Reproducibility notes

* The rigorous part of the code (`eells/ball.py`) relies only on IEEE-754 double precision with
  round-to-nearest and on the error bounds for sums/dot products stated there; it does not use
  directed rounding, so it runs unchanged on any platform (the BLAS used by NumPy may sum in any
  order, with or without fused multiply-add, which the bounds allow).
* The `certificate/output/cache/` directory stores the Jacobi-polynomial tables (computed once
  per grid size with mpmath); delete it to force recomputation.
* Tested with Python 3.10.13, NumPy 2.2, SciPy 1.15, mpmath 1.3 on macOS (Apple silicon).
