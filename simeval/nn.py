"""nn.py — dependency-free (numpy) learners: PPO path policy with GraphSAGE encoder (or flat MLP) and a DQN.

Why numpy: the evaluation sandbox has no torch/CUDA (same as the target AMD RX580 workstation). Gradients are
written by hand and verified against finite differences in tests/test_simeval_nn.py.
"""
from __future__ import annotations
import numpy as np

NF, HID, PH, GL = 4, 16, 5, 4          # node feats, hidden, path feats, global feats


class Adam:
    def __init__(s, params, lr=1e-3, b1=.9, b2=.999):
        s.p, s.lr, s.b1, s.b2, s.t = params, lr, b1, b2, 0
        s.m = {k: np.zeros_like(v) for k, v in params.items()}; s.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(s, g, clip=1.0):
        s.t += 1
        n = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        sc = min(1.0, clip / (n + 1e-8))
        for k in s.p:
            gk = g[k] * sc
            s.m[k] = s.b1 * s.m[k] + (1 - s.b1) * gk; s.v[k] = s.b2 * s.v[k] + (1 - s.b2) * gk ** 2
            s.p[k] -= s.lr * (s.m[k] / (1 - s.b1 ** s.t)) / (np.sqrt(s.v[k] / (1 - s.b2 ** s.t)) + 1e-8)


def _softmax(z, mask):
    z = np.where(mask, z, -1e9); z = z - z.max(); e = np.exp(z) * mask
    return e / e.sum()


class PathPolicy:
    """Scores K candidate paths for one flow. use_graph=True: 2-layer GraphSAGE (mean aggregator) over the topology
    -> mean-pooled path embedding; use_graph=False: flat MLP on path+global features only (ablation A5)."""

    def __init__(s, use_graph=True, seed=0):
        r = np.random.default_rng(seed); s.use_graph = use_graph
        w = lambda *sh: r.normal(0, 1 / np.sqrt(sh[0]), sh)
        s.p = dict(Ws1=w(NF, HID), Wn1=w(NF, HID), b1=np.zeros(HID), Ws2=w(HID, HID), Wn2=w(HID, HID), b2=np.zeros(HID),
                   Wz=w(HID, HID) * (1 if use_graph else 0), Wp=w(PH, HID), Wg=w(GL, HID), bh=np.zeros(HID),
                   wo=w(HID, 1)[:, 0] * .1, bo=np.zeros(1), wvg=w(HID, 1)[:, 0] * (.1 if use_graph else 0),
                   wvs=w(GL, 1)[:, 0] * .1, bv=np.zeros(1))
        s.opt = None

    # ---- forward ----
    def forward(s, X, A, P, Phi, glob, mask):
        p = s.p; c = {}
        if s.use_graph:
            AX = A @ X; pre1 = X @ p["Ws1"] + AX @ p["Wn1"] + p["b1"]; H1 = np.maximum(pre1, 0)
            AH1 = A @ H1; pre2 = H1 @ p["Ws2"] + AH1 @ p["Wn2"] + p["b2"]; H2 = np.maximum(pre2, 0)
            Z = P @ H2; gpool = H2.mean(0); c.update(AX=AX, H1=H1, AH1=AH1, H2=H2, Z=Z)
        else:
            Z = np.zeros((P.shape[0], HID)); gpool = np.zeros(HID)
        pre = Z @ p["Wz"] + Phi @ p["Wp"] + (glob @ p["Wg"])[None, :] + p["bh"]; Hk = np.tanh(pre)
        sc = Hk @ p["wo"] + p["bo"][0]
        pi = _softmax(sc, mask); v = float(gpool @ p["wvg"] + glob @ p["wvs"] + p["bv"][0])
        c.update(X=X, A=A, P=P, Phi=Phi, glob=glob, mask=mask, Hk=Hk, pi=pi, gpool=gpool)
        return pi, v, c

    # ---- backward: grads of  L = -clipsurr + cv*(v-R)^2 - ce*H ----
    def grad(s, c, a, adv, logp_old, ret, clip=.2, cv=.5, ce=.01):
        p, pi, mask = s.p, c["pi"], c["mask"]; g = {k: np.zeros_like(v) for k, v in p.items()}
        K = len(pi); logp = np.log(pi[a] + 1e-12); r = np.exp(logp - logp_old)
        unclipped = (adv >= 0 and r < 1 + clip) or (adv < 0 and r > 1 - clip)
        dlogp = -adv * r if unclipped else 0.0
        onehot = np.zeros(K); onehot[a] = 1
        dz = dlogp * (onehot - pi)
        H = -(pi * np.log(pi + 1e-12)).sum(); dz += ce * pi * (np.log(pi + 1e-12) + H)
        dz = dz * mask
        v = float(c["gpool"] @ p["wvg"] + c["glob"] @ p["wvs"] + p["bv"][0]); dv = 2 * cv * (v - ret)
        Hk = c["Hk"]; g["wo"] = Hk.T @ dz; g["bo"] = np.array([dz.sum()])
        dpre = (dz[:, None] * p["wo"][None, :]) * (1 - Hk ** 2)
        g["Wp"] = c["Phi"].T @ dpre; g["Wg"] = np.outer(c["glob"], dpre.sum(0)); g["bh"] = dpre.sum(0)
        g["wvs"] = c["glob"] * dv; g["bv"] = np.array([dv])
        if s.use_graph:
            g["Wz"] = c["Z"].T @ dpre; dZ = dpre @ p["Wz"].T; dH2 = c["P"].T @ dZ
            g["wvg"] = c["gpool"] * dv; dH2 = dH2 + (dv / c["H2"].shape[0]) * p["wvg"][None, :]
            d2 = dH2 * (c["H2"] > 0); g["Ws2"] = c["H1"].T @ d2; g["Wn2"] = c["AH1"].T @ d2; g["b2"] = d2.sum(0)
            dH1 = d2 @ p["Ws2"].T + c["A"].T @ (d2 @ p["Wn2"].T)
            d1 = dH1 * (c["H1"] > 0); g["Ws1"] = c["X"].T @ d1; g["Wn1"] = c["AX"].T @ d1; g["b1"] = d1.sum(0)
        return g

    def loss(s, c, a, adv, logp_old, ret, clip=.2, cv=.5, ce=.01):
        pi = c["pi"]; r = np.exp(np.log(pi[a] + 1e-12) - logp_old)
        sur = min(r * adv, np.clip(r, 1 - clip, 1 + clip) * adv)
        v = float(c["gpool"] @ s.p["wvg"] + c["glob"] @ s.p["wvs"] + s.p["bv"][0])
        H = -(pi * np.log(pi + 1e-12)).sum()
        return -sur + cv * (v - ret) ** 2 - ce * H

    def act(s, feats, greedy=False, rng=None):
        pi, v, c = s.forward(*feats)
        a = int(np.argmax(pi)) if greedy else int((rng or np.random).choice(len(pi), p=pi))
        return a, float(np.log(pi[a] + 1e-12)), v, pi

    def update(s, batch, lr=2e-3, epochs=4, mb=64, rng=None):
        """batch: list of dict(feats, a, logp, adv, ret)."""
        rng = rng or np.random.default_rng(0)
        if s.opt is None: s.opt = Adam(s.p, lr=lr)
        adv = np.array([b["adv"] for b in batch]); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for b, a_ in zip(batch, adv): b["advn"] = a_
        for _ in range(epochs):
            idx = rng.permutation(len(batch))
            for i in range(0, len(idx), mb):
                gs = None
                for j in idx[i:i + mb]:
                    b = batch[j]; _, _, c = s.forward(*b["feats"])
                    g = s.grad(c, b["a"], b["advn"], b["logp"], b["ret"])
                    gs = g if gs is None else {k: gs[k] + g[k] for k in g}
                n = len(idx[i:i + mb]); s.opt.step({k: v / n for k, v in gs.items()})


class DQN:
    def __init__(s, din=8, nact=6, hid=64, seed=0):
        r = np.random.default_rng(seed); w = lambda a, b: r.normal(0, np.sqrt(2 / a), (a, b))
        s.p = dict(W1=w(din, hid), b1=np.zeros(hid), W2=w(hid, hid), b2=np.zeros(hid), W3=w(hid, nact) * .1, b3=np.zeros(nact))
        s.tp = {k: v.copy() for k, v in s.p.items()}; s.opt = Adam(s.p, lr=1e-3); s.nact = nact
        s.buf = []; s.rng = r; s.eps = 1.0

    @staticmethod
    def fwd(p, x):
        h1 = np.maximum(x @ p["W1"] + p["b1"], 0); h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        return h2 @ p["W3"] + p["b3"], (x, h1, h2)

    def q(s, x): return s.fwd(s.p, x[None])[0][0]

    def act(s, x, greedy=False):
        if not greedy and s.rng.random() < s.eps: return int(s.rng.integers(s.nact))
        return int(np.argmax(s.q(x)))

    def remember(s, tr):
        s.buf.append(tr)
        if len(s.buf) > 50000: s.buf.pop(0)

    def train_step(s, bs=64, gamma=.9):
        if len(s.buf) < bs: return
        idx = s.rng.integers(len(s.buf), size=bs); B = [s.buf[i] for i in idx]
        x = np.array([b[0] for b in B]); a = np.array([b[1] for b in B]); r = np.array([b[2] for b in B])
        x2 = np.array([b[3] for b in B]); d = np.array([b[4] for b in B], float)
        q2 = s.fwd(s.tp, x2)[0].max(1); y = r + gamma * q2 * (1 - d)
        q, (xi, h1, h2) = s.fwd(s.p, x); err = np.zeros_like(q); err[np.arange(bs), a] = np.clip(q[np.arange(bs), a] - y, -1, 1) / bs
        g = {"W3": h2.T @ err, "b3": err.sum(0)}
        dh2 = (err @ s.p["W3"].T) * (h2 > 0); g["W2"] = h1.T @ dh2; g["b2"] = dh2.sum(0)
        dh1 = (dh2 @ s.p["W2"].T) * (h1 > 0); g["W1"] = xi.T @ dh1; g["b1"] = dh1.sum(0)
        s.opt.step(g, clip=5.0)

    def sync(s): s.tp = {k: v.copy() for k, v in s.p.items()}
