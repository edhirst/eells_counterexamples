"""Vector fields on S^3 sampled on the grid, built from basis coefficients.

A :class:`Field` stores, for a vector field h = sum_k a_k Psi_k,
    val[c]      the three components,
    grad[a][c]  the tangential partial derivative d_a h_c along the ambient axis a (a = 0..3),
    lap[c]      the Laplacian Delta h_c = -sum_k a_k l_k (l_k + 2) Psi_k.
Degrees (as polynomials in the ambient coordinates): val D, grad D + 1, lap D.

Coefficient files are JSON lists of {kind, n, j, l, value} with a header giving the Hopf
invariant q and the degree; the values are IEEE doubles and are used exactly by the certificate
(``Ball.exact``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import arith as ar
from .ball import Ball
from .harmonics import Mode, HarmonicEvaluator


class Field:
    def __init__(self, val, grad, lap):
        self.val, self.grad, self.lap = val, grad, lap

    def __add__(self, o):
        return Field([self.val[c] + o.val[c] for c in range(3)],
                     [[self.grad[a][c] + o.grad[a][c] for c in range(3)] for a in range(4)],
                     [self.lap[c] + o.lap[c] for c in range(3)])

    def scale(self, t):
        return Field([v * t for v in self.val], [[g * t for g in row] for row in self.grad], [v * t for v in self.lap])

    def rows(self, rows):
        """The field on the block ``rows`` of grid rows."""
        return Field([v[rows] for v in self.val], [[g[rows] for g in row] for row in self.grad], [v[rows] for v in self.lap])


def field_from_coefficients(modes, values, evaluator: HarmonicEvaluator, batch=64, rows=slice(None)):
    """sum_k values[k] Psi_{modes[k]} as a Field (any backend; values are exact doubles), on the
    whole grid or on the grid rows ``rows``."""
    rig = evaluator.rig
    shape = (len(range(evaluator.g.Ns + 1)[rows]), evaluator.g.Nphi)
    val = [ar.zeros(shape, rig) for _ in range(3)]
    grad = [[ar.zeros(shape, rig) for _ in range(3)] for _ in range(4)]
    lap = [ar.zeros(shape, rig) for _ in range(3)]
    for start in range(0, len(modes), batch):
        mk = modes[start:start + batch]
        v, g, lp = evaluator.evaluate(mk, rows)
        coef = np.asarray(values[start:start + batch], dtype=float).reshape(-1, 1, 1)
        coef = Ball.exact(coef) if rig else coef
        for c in range(3):
            val[c] = val[c] + (v[:, c] * coef).sum(axis=0)
            lap[c] = lap[c] + (lp[:, c] * coef).sum(axis=0)
            for a in range(4):
                grad[a][c] = grad[a][c] + (g[:, a, c] * coef).sum(axis=0)
    return Field(val, grad, lap)


# ---------------------------------------------------------------------------- coefficient files
def save_coefficients(path, modes, values, meta=None):
    q = modes[0].q
    assert all(m.q == q for m in modes)
    data = {"description": "coefficients a_k of Wbar = sum_k a_k Psi_k in the orthonormal basis of "
                           "Proposition 2.2 for the Hopf invariant q (U: (Re Y_{qn-2q,n,j}, Im Y_{qn-2q,n,j}, 0), "
                           "n even; V: sqrt(2) (0, 0, Re Y_{qn,n,j}), n odd); values are exact IEEE doubles",
            "hopf_invariant": q,
            "degree": max(m.l for m in modes),
            "coefficients": [{"kind": m.kind, "n": m.n, "j": m.j, "l": m.l, "value": float(v)}
                             for m, v in zip(modes, values)]}
    if meta:
        data["meta"] = meta
    Path(path).write_text(json.dumps(data, indent=1))


def load_coefficients(path):
    """Read and validate a coefficient file.  Every mode must be one of the basis fields of
    Proposition 2.2 for the file's Hopf invariant q (U with n even, V with n odd >= 1, j >= 0,
    l = |m| + |n| + 2j), the modes must be distinct (the coefficient-squares formula for
    ||Wbar||_X needs orthonormality), the values finite, and the header degree must equal the
    largest degree present."""
    data = json.loads(Path(path).read_text())
    q = int(data.get("hopf_invariant", 2))
    assert q >= 2
    modes, values = [], []
    for c in data["coefficients"]:
        kind, n, j, l, v = c["kind"], int(c["n"]), int(c["j"]), int(c["l"]), float(c["value"])
        assert kind in ("U", "V"), f"unknown mode kind {kind}"
        assert j >= 0 and np.isfinite(v), f"invalid mode {c}"
        assert (kind == "U" and n % 2 == 0) or (kind == "V" and n % 2 == 1 and n >= 1), f"mode {c} violates the selection rule"
        m = Mode(l, kind, n, j, q)
        assert m.l == abs(m.m) + abs(m.n) + 2 * m.j, f"wrong degree in {c}"
        modes.append(m)
        values.append(v)
    assert len(set(modes)) == len(modes), "duplicate modes"
    assert int(data["degree"]) == max(m.l for m in modes), "header degree does not match the coefficients"
    return modes, np.array(values, dtype=float), data
