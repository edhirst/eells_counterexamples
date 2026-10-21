"""Ball arithmetic: every operation must enclose the exact result (checked against exact rationals)."""
from fractions import Fraction

import numpy as np

from eells.ball import Ball, matmul, norm_2, pi_ball, cos_sin_table


def test_enclosure_of_rational_arithmetic():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = Fraction(int(rng.integers(-10**6, 10**6)), int(rng.integers(1, 10**4))), Fraction(int(rng.integers(-10**6, 10**6)), int(rng.integers(1, 10**4)))
        A, B = Ball.from_fraction(a), Ball.from_fraction(b)
        for exact, ball in ((a + b, A + B), (a - b, A - B), (a * b, A * B), (a * a, A.sq())):
            assert ball.lower() <= exact <= ball.upper()
        if b != 0:
            q = A / B
            assert q.lower() <= a / b <= q.upper()
        if a >= 0:
            s = A.sqrt()
            assert Fraction(float(s.lower())) ** 2 <= a <= Fraction(float(s.upper())) ** 2


def test_exact_zero_stays_exact():
    z = Ball(np.array([1.0])) - Ball(np.array([1.0]))
    assert z.m[0] == 0 and z.r[0] == 0
    assert (z * Ball(np.array([1e300]))).r[0] == 0
    assert z.sqrt(nonneg=True).r[0] == 0


def test_sum_and_matmul_enclose():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 40)); B = rng.standard_normal((40, 20))
    exactish = np.array([[sum(Fraction(float(A[i, k])) * Fraction(float(B[k, j])) for k in range(40)) for j in range(20)] for i in range(30)], dtype=object)
    P = matmul(Ball(A), Ball(B))
    lo, hi = P.lower(), P.upper()
    for i in range(30):
        for j in range(20):
            assert Fraction(float(lo[i, j])) <= exactish[i, j] <= Fraction(float(hi[i, j]))
    s = Ball(A).sum(axis=1)
    rows = [sum(Fraction(float(x)) for x in A[i]) for i in range(30)]
    assert all(Fraction(float(s.lower()[i])) <= rows[i] <= Fraction(float(s.upper()[i])) for i in range(30))


def test_norm_bounds():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((25, 60))
    true = np.linalg.norm(M, 2)
    assert true <= norm_2(Ball(M)) <= 1.2 * true


def test_constants():
    p = pi_ball()
    assert p.lower() < np.pi < p.upper() and p.r < 1e-15
    c, s = cos_sin_table(12)
    assert abs(c.m[3]) == 0 and c.r[3] == 0 and s.m[3] == 1.0
    assert c.lower()[1] <= np.cos(2 * np.pi / 12) <= c.upper()[1]
