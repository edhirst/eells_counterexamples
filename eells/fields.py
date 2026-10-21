"""Vector fields on S^3 sampled on the grid, built from basis coefficients.

A :class:`Field` stores, for a vector field h = sum_k a_k Psi_k,
    val[c]      the three components,
    grad[a][c]  the tangential partial derivative d_a h_c along the ambient axis a (a = 0..3),
    lap[c]      the Laplacian Delta h_c = -sum_k a_k l_k (l_k + 2) Psi_k.
Degrees (as polynomials in the ambient coordinates): val D, grad D + 1, lap D.

Coefficient files are JSON lists of {kind, n, j, l, value}; the values are IEEE doubles and
are used exactly by the certificate (``Ball.exact``).
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


def field_from_coefficients(modes, values, evaluator: HarmonicEvaluator, batch=64):
    """sum_k values[k] Psi_{modes[k]} as a Field (any backend; values are exact doubles)."""
    rig = evaluator.rig
    shape = evaluator.g.shape
    val = [ar.zeros(shape, rig) for _ in range(3)]
    grad = [[ar.zeros(shape, rig) for _ in range(3)] for _ in range(4)]
    lap = [ar.zeros(shape, rig) for _ in range(3)]
    for start in range(0, len(modes), batch):
        mk = modes[start:start + batch]
        v, g, lp = evaluator.evaluate(mk)
        coef = np.asarray(values[start:start + batch], dtype=float).reshape(-1, 1, 1)
        coef = Ball.exact(coef) if rig else coef
        for c in range(3):
            val[c] = val[c] + (v[:, c] * coef).sum(axis=0)
            lap[c] = lap[c] + (lp[:, c] * coef).sum(axis=0)
            for a in range(4):
                grad[a][c] = grad[a][c] + (g[:, a, c] * coef).sum(axis=0)
    return Field(val, grad, lap)


def scalar_from_coefficients(modes, values, evaluator: HarmonicEvaluator, unnormalised=True):
    """G-invariant scalar  sum_k values[k] * (unnormalised scalar harmonic of modes[k])."""
    out = ar.zeros(evaluator.g.shape, evaluator.rig)
    for md, v in zip(modes, values):
        out = out + evaluator.evaluate_scalar(md, unnormalised=unnormalised) * float(v)
    return out


# ---------------------------------------------------------------------------- coefficient files
def save_coefficients(path, modes, values, meta=None):
    data = {"description": "coefficients a_k of Wbar = sum_k a_k Psi_k in the orthonormal basis of "
                           "Proposition 2.2 (U: (Re Y_{2n-4,n,j}, Im Y_{2n-4,n,j}, 0), n even; "
                           "V: sqrt(2) (0, 0, Re Y_{2n,n,j}), n odd); values are exact IEEE doubles",
            "degree": max(m.l for m in modes),
            "coefficients": [{"kind": m.kind, "n": m.n, "j": m.j, "l": m.l, "value": float(v)}
                             for m, v in zip(modes, values)]}
    if meta:
        data["meta"] = meta
    Path(path).write_text(json.dumps(data, indent=1))


def load_coefficients(path):
    data = json.loads(Path(path).read_text())
    modes = [Mode(c["l"], c["kind"], c["n"], c["j"]) for c in data["coefficients"]]
    values = np.array([c["value"] for c in data["coefficients"]], dtype=float)
    for m, c in zip(modes, data["coefficients"]):
        assert m.l == abs(m.m) + abs(m.n) + 2 * m.j == c["l"]
    return modes, values, data


def save_scalar_coefficients(path, modes, values, meta=None):
    data = {"description": "coefficients of the G-invariant polynomial chi = sum chi_{n,j} "
                           "Re(z^{2n} w^n) P_j^{(2n,n)}(1 - 2|z|^2)  (n even >= 0; unnormalised basis)",
            "degree": max(m.l for m in modes),
            "coefficients": [{"n": m.n, "j": m.j, "l": m.l, "value": float(v)} for m, v in zip(modes, values)]}
    if meta:
        data["meta"] = meta
    Path(path).write_text(json.dumps(data, indent=1))


def load_scalar_coefficients(path):
    data = json.loads(Path(path).read_text())
    modes = [Mode(c["l"], "S", c["n"], c["j"]) for c in data["coefficients"]]
    values = np.array([c["value"] for c in data["coefficients"]], dtype=float)
    return modes, values, data
