"""Backend dispatch: the geometric code runs unchanged on numpy float arrays (fast, non-rigorous
numerics) and on :class:`eells.ball.Ball` arrays (rigorous certificate).  Only the handful of
operations below differ between the two backends."""
import numpy as np

from .ball import Ball, stack as _ball_stack


def is_ball(x):
    return isinstance(x, Ball)


def sqrt(x, nonneg=False):
    """Square root; ``nonneg=True`` asserts the exact quantity is >= 0 (see Ball.sqrt)."""
    return x.sqrt(nonneg=nonneg) if is_ball(x) else np.sqrt(x)


def sq(x):
    """x^2, never negative."""
    return x.sq() if is_ball(x) else x * x


def stack(xs, axis=0):
    return _ball_stack(xs, axis) if any(is_ball(x) for x in xs) else np.stack(xs, axis)


def zeros_like(x):
    return Ball(np.zeros(x.shape)) if is_ball(x) else np.zeros(x.shape)


def zeros(shape, rigorous):
    return Ball(np.zeros(shape)) if rigorous else np.zeros(shape)


def ones(shape, rigorous):
    return Ball(np.ones(shape)) if rigorous else np.ones(shape)


def const(c, rigorous):
    """A scalar constant given exactly as a float (or integer)."""
    return Ball.exact(float(c)) if rigorous else float(c)


def upper(x):
    """Upper bound (ball) or value (float)."""
    return x.upper() if is_ball(x) else np.asarray(x, dtype=float)


def lower(x):
    return x.lower() if is_ball(x) else np.asarray(x, dtype=float)


def midpoint(x):
    return x.m if is_ball(x) else np.asarray(x, dtype=float)


def to_float(x):
    """Midpoint as a Python float / numpy array (for diagnostics only)."""
    return np.asarray(x.m if is_ball(x) else x, dtype=float)
