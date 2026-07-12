import numpy as np
from ddorm_coverage.targets import ddo_target, score_gradient, ddo_cross_entropy
from ddorm_coverage.coverage import approximate_full_target, support_coverage, support_gap_from_coverage


def test_ddo_target_normalized():
    s = np.array([[0.0, 1.0, 2.0]])
    r = np.array([[1.0, 0.0, -1.0]])
    p, q = ddo_target(s, r, lam=1.0)
    assert np.allclose(p.sum(), 1.0)
    assert np.allclose(q.sum(), 1.0)
    assert np.all(q > 0)


def test_gradient_sums_zero():
    s = np.random.default_rng(0).normal(size=(2, 4))
    _, q = ddo_target(s, np.ones_like(s))
    g = score_gradient(s, q)
    assert np.allclose(g.sum(axis=1), 0.0)


def test_coverage_gap_nonnegative():
    s = np.array([[0.0, 1.0, 2.0, 3.0]])
    r = np.array([[0.0, 0.1, 0.2, 2.0]])
    _, q_pool = approximate_full_target(s, r, lam=1.0)
    rho = support_coverage(q_pool, np.array([[2, 3]]))
    gap = support_gap_from_coverage(rho, lam=1.0)
    assert rho[0] > 0
    assert gap[0] >= 0


def test_ce_finite():
    s = np.array([[0.0, 1.0]])
    _, q = ddo_target(s, np.array([[1.0, 0.0]]))
    ce = ddo_cross_entropy(s, q)
    assert np.isfinite(ce).all()
