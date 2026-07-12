import numpy as np
from ddorm_coverage.targets import ddo_target, score_gradient
from ddorm_coverage.coverage import approximate_full_target, support_coverage, support_gap_from_coverage

rng = np.random.default_rng(0)
s = rng.normal(size=(3, 5))
r = rng.normal(size=(3, 5))
p, q = ddo_target(s, r, lam=0.7, tau=1.0)
assert np.allclose(p.sum(axis=1), 1.0)
assert np.allclose(q.sum(axis=1), 1.0)
g = score_gradient(s, q)
assert g.shape == s.shape
_, q_pool = approximate_full_target(s, r, lam=0.7)
rho = support_coverage(q_pool, np.array([[0,1],[1,2],[2,3]]))
gap = support_gap_from_coverage(rho, lam=0.7)
print("p", p.shape, "q", q.shape, "rho", rho, "gap", gap)
