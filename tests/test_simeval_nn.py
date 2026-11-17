import numpy as np
from simeval.nn import PathPolicy, NF, PH, GL

def _feats(r, N=7, K=4):
    X = r.normal(size=(N, NF)); A = r.random((N, N)); A = A / A.sum(1, keepdims=True)
    P = r.random((K, N)); P /= P.sum(1, keepdims=True); Phi = r.normal(size=(K, PH)); glob = r.normal(size=GL)
    mask = np.array([1, 1, 1, 0], bool); return X, A, P, Phi, glob, mask

def test_gradcheck():
    for ug in (True, False):
        r = np.random.default_rng(1); pol = PathPolicy(use_graph=ug, seed=3)
        for k in pol.p: pol.p[k] = pol.p[k] + r.normal(0, .05, pol.p[k].shape)
        f = _feats(r); pi, v, c = pol.forward(*f); a = 1; adv, ret = .7, .3; lo = float(np.log(pi[a]) - .1)
        g = pol.grad(c, a, adv, lo, ret)
        for k in pol.p:
            if pol.p[k].size == 0 or not np.any(g[k]): continue
            idx = tuple(np.unravel_index(r.integers(pol.p[k].size), pol.p[k].shape)); e = 1e-6
            old = pol.p[k][idx]
            pol.p[k][idx] = old + e; l1 = pol.loss(pol.forward(*f)[2], a, adv, lo, ret)
            pol.p[k][idx] = old - e; l2 = pol.loss(pol.forward(*f)[2], a, adv, lo, ret); pol.p[k][idx] = old
            num = (l1 - l2) / (2 * e)
            assert abs(num - g[k][idx]) < 1e-5 + 1e-3 * abs(num), (ug, k, num, g[k][idx])
