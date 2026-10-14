"""Coefficient tables of the factorized K=3 ReLU step.

These are numerical constants, not code: the ReLU Hermite/Wick coefficients
w(k, p) = E[d^k relu(Z)^p], Z ~ N(mu, sigma^2), in the unified form
const * sigma^e * (P1(alpha) phi(alpha) + P2(alpha) Phi(alpha)), alpha = mu / sigma, and the
diagram-summation term table of the K3-simple algorithm (Wu et al., arXiv:2605.05179,
S.4.3) restricted to the output slices (1,), (2,), (3,), (4,), (1,1), (2,1), (2,2). They are
the tables of ARC's reference generator as published (pruned at relative contribution
1e-5) by 504aldo's K=3 estimator (github.com/504aldo/whest-p2-cumulant-k3, MIT licence),
together with that estimator's fitted per-layer kappa4 regeneration constants (LAM, REF_R).
"""

WICK_PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2),
              (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (6, 1), (6, 2),
              (7, 1)]

# (const, sigma exponent, P1 coefficients, P2 coefficients), polynomials in alpha, ascending
WICK_UNIFIED = [
    (1.0, 1, [1.0], [0.0, 1.0]),
    (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]),
    (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
    (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]),
    (1.0, 0, [], [1.0]),
    (1.0, 1, [2.0], [0.0, 2.0]),
    (1.0, -1, [1.0], []),
    (2.0, 0, [], [1.0]),
    (1.0, -2, [0.0, -1.0], []),
    (2.0, -1, [1.0], []),
    (6.0, 0, [], [1.0]),
    (1.0, 1, [24.0], [0.0, 24.0]),
    (1.0, -3, [-1.0, 0.0, 1.0], []),
    (2.0, -2, [0.0, -1.0], []),
    (6.0, -1, [1.0], []),
    (24.0, 0, [], [1.0]),
    (1.0, -4, [0.0, 3.0, 0.0, -1.0], []),
    (2.0, -3, [-1.0, 0.0, 1.0], []),
    (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []),
    (2.0, -4, [0.0, 3.0, 0.0, -1.0], []),
    (1.0, -6, [0.0, -15.0, 0.0, 10.0, 0.0, -1.0], []),
]
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = -6, 4

# term = (slice a, slice b, wick column left, wick column right (None for d=1), coefficient)
TERM_SPECS = {
    (1,): [('ones1', 'ones1', 0, None, 1.0),
           ('d3', 'ones1', 8, None, 1.0 / 6.0),
           ('g4', 'ones1', 12, None, 1.0 / 24.0),
           ('d3', 'd3', 18, None, 1.0 / 72.0),
           ('d3', 'g4', 20, None, 1.0 / 144.0)],
    (1, 1): [('c_off', 'ones2', 4, 4, 1.0),
             ('d21T', 'ones2', 4, 6, 1.0),
             ('c_off', 'c_off', 6, 6, 0.5),
             ('wk4m', 'ones2', 6, 6, 0.25),
             ('d3row', 'c_off', 4, 12, 1.0 / 3.0),
             ('c_off', 'd21T', 6, 8, 1.0),
             ('d3row', 'd21T', 4, 16, 1.0 / 6.0),
             ('g4row', 'c_off', 4, 16, 1.0 / 12.0),
             ('d3row', 'd21', 6, 12, 1.0 / 6.0),
             ('d21T', 'd21T', 6, 12, 0.25),
             ('c_off', 'wk4m', 8, 8, 0.25),
             ('d21T', 'd21', 8, 8, 0.25),
             ('g4row', 'd21T', 4, 18, 1.0 / 24.0),
             ('d3row', 'wk4m', 6, 16, 1.0 / 12.0),
             ('g4row', 'd21', 6, 16, 1.0 / 24.0),
             ('d21T', 'wk4m', 8, 12, 0.25),
             ('g4row', 'wk4m', 6, 18, 1.0 / 48.0),
             ('wk431', 'ones2', 4, 8, 1.0 / 3.0)],
    (2,): [('ones1', 'ones1', 1, None, 1.0),
           ('d3', 'ones1', 9, None, 1.0 / 6.0),
           ('g4', 'ones1', 13, None, 1.0 / 24.0)],
    (2, 1): [('c_off', 'ones2', 5, 4, 1.0),
             ('d21T', 'ones2', 5, 6, 0.5),
             ('d21', 'ones2', 7, 4, 0.5),
             ('c_off', 'c_off', 7, 6, 0.5),
             ('wk4m', 'ones2', 7, 6, 0.25),
             ('d3row', 'c_off', 5, 12, 1.0 / 6.0),
             ('c_off', 'd21T', 7, 8, 0.5),
             ('c_off', 'd21', 9, 6, 0.5),
             ('c_off', 'd3col', 13, 4, 1.0 / 6.0),
             ('d3row', 'd21T', 5, 16, 1.0 / 12.0),
             ('g4row', 'c_off', 5, 16, 1.0 / 24.0),
             ('d3row', 'd21', 7, 12, 1.0 / 12.0),
             ('d21T', 'd21T', 7, 12, 0.125),
             ('c_off', 'wk4m', 9, 8, 0.25),
             ('d21T', 'd21', 9, 8, 0.25),
             ('c_off', 'g4col', 17, 4, 1.0 / 24.0),
             ('d21', 'd3col', 17, 4, 1.0 / 12.0),
             ('g4row', 'd21T', 5, 18, 1.0 / 48.0),
             ('g4row', 'd21', 7, 16, 1.0 / 48.0),
             ('d21', 'g4col', 19, 4, 1.0 / 48.0),
             ('wk431', 'ones2', 5, 8, 1.0 / 6.0),
             ('wk431', 'ones2', 9, 4, 1.0 / 6.0)],
    (2, 2): [('c_off', 'ones2', 5, 5, 1.0),
             ('d21T', 'ones2', 5, 7, 1.0),
             ('c_off', 'c_off', 7, 7, 0.5),
             ('wk4m', 'ones2', 7, 7, 0.25),
             ('d3row', 'c_off', 5, 13, 1.0 / 3.0),
             ('c_off', 'd21T', 7, 9, 1.0),
             ('d3row', 'd21T', 5, 17, 1.0 / 6.0),
             ('g4row', 'c_off', 5, 17, 1.0 / 12.0),
             ('d21T', 'd21T', 7, 13, 0.25),
             ('c_off', 'wk4m', 9, 9, 0.25),
             ('g4row', 'd21T', 5, 19, 1.0 / 24.0),
             ('wk431', 'ones2', 5, 9, 1.0 / 3.0)],
    (3,): [('ones1', 'ones1', 2, None, 1.0),
           ('d3', 'ones1', 10, None, 1.0 / 6.0),
           ('g4', 'ones1', 14, None, 1.0 / 24.0)],
    (4,): [('ones1', 'ones1', 3, None, 1.0),
           ('d3', 'ones1', 11, None, 1.0 / 6.0),
           ('g4', 'ones1', 15, None, 1.0 / 24.0)],
}

# kappa4 regeneration: pre-activation off-diagonal core lambda_l * C_off (fitted per layer,
# 8 public MLPs; times the 0.95 global scale) and the reference ratio of the adaptive rule
LAM = [c * 0.95 for c in (4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02,
                          1.0828e-02, 1.0589e-02, 1.0048e-02, 9.6483e-03, 9.1720e-03,
                          8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03,
                          7.2588e-03)]
REF_R = [6.58815e-03, 8.18414e-03, 8.53136e-03, 8.38859e-03, 8.10153e-03, 7.74287e-03,
         7.35951e-03, 6.91093e-03, 6.55752e-03, 6.17892e-03, 5.83953e-03, 5.56824e-03,
         5.32459e-03, 5.05165e-03, 4.77181e-03]
METRIC_C = 2.0
