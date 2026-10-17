"""bias.py <tag>: (1) per-layer error profile (label MSE minus the label noise pk2/n, per layer, pooled);
(2) final-layer residual-bias regression on the chain's own last-layer stats, fit on one half of the
networks and scored on the other (both ways)."""
import sys, os, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
from fm.mlps import load_split
lab = {d["seed"]: d for s in ("train", "val") for d in load_split(s, min_samples=1)}
tag = sys.argv[1]
D = {}
for sd in sorted(lab):
    f = f"/tmp/claude-0/acc/{tag}/cap_{sd}.npz"
    if os.path.exists(f): D[sd] = dict(np.load(f))
seeds = sorted(D); print("networks", len(seeds))
L = 16
# (1) per-layer profile
prof = np.zeros((L, 3))
for sd in seeds:
    t = lab[sd]["means"]; nn = lab[sd]["n"]
    p = np.load(f"/tmp/claude-0/acc/{tag}/pred_{sd}.npy")
    for l in range(L):
        se = ((p[l] - t[l]) ** 2).mean(); nz = (D[sd][f"pk2_{l}"] - D[sd][f"pk1_{l}"].astype(np.float64) ** 2).mean() / nn
        prof[l] += (se, nz, ((p[l] - t[l]) ** 2).mean() - nz)
prof /= len(seeds)
chk = np.mean([lab[sd]["noise_last"].mean() for sd in seeds])
print("last-layer noise: pk2/n %.3e vs noise_last %.3e" % (prof[-1, 1], chk))
for l in range(L):
    print("layer %2d  raw-se %.3e  noise %.3e  excess %.3e  (excess/mean-square-pred %.2e)" % (l, *prof[l], prof[l, 2] / np.mean([(np.load(f'/tmp/claude-0/acc/{tag}/pred_{sd}.npy')[l] ** 2).mean() for sd in seeds[:5]])))
# (2) residual bias at the last layer
import math
class norm:
    pdf = staticmethod(lambda a: np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi))
    cdf = staticmethod(lambda a: 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2))))
def feats(z):
    l = L - 1
    mu, var = z[f"mu_{l}"].astype(np.float64), z[f"var_{l}"].astype(np.float64)
    s = np.sqrt(var); a = mu / s; ph = norm.pdf(a)
    s3 = z[f"D3_{l}"] / s ** 3; s4 = z[f"g4_{l}"] / s ** 4
    base = s * ph
    X = np.stack([base, base * a, base * a * a, base * s3, base * s3 * a, base * s3 * a * a,
                  base * s4, base * s4 * a, base * s4 * a * a, base * s3 * s3, s * norm.cdf(a), mu], 1)
    return X
X, Y, G = [], [], []
for sd in seeds:
    p = np.load(f"/tmp/claude-0/acc/{tag}/pred_{sd}.npy")[-1]
    X.append(feats(D[sd])); Y.append(lab[sd]["means"][-1] - p); G.append(np.full(1024, sd))
X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
nz = {sd: lab[sd]["noise_last"].mean() for sd in seeds}
def score(m, beta):
    tot0 = tot1 = 0.0
    for sd in np.unique(G[m]):
        k = G == sd
        e0 = (Y[k] ** 2).mean() - nz[sd]; e1 = ((Y[k] - X[k] @ beta) ** 2).mean() - nz[sd]
        tot0 += e0; tot1 += e1
    return 100 * (tot1 - tot0) / tot0
for cols, name in (([0, 1, 2], "phi poly"), ([0, 1, 2, 3, 4, 5], "+s3"), (list(range(9)), "+s4"), (list(range(12)), "all")):
    out = []
    for split in (lambda s: s % 2 == 0, lambda s: s % 2 == 1, lambda s: s >= 1032 or s < 1016, lambda s: (s % 3) == 0):
        tr = np.array([split(s) for s in G])
        Xa = X[:, cols]
        beta = np.linalg.lstsq(Xa[tr], Y[tr], rcond=None)[0]
        bfull = np.zeros(X.shape[1]); bfull[cols] = beta
        out.append((score(~tr, bfull), score(tr, bfull)))
    print("%-9s held-out dMSE %s   (in-sample %s)" % (name, " ".join("%+.2f%%" % o[0] for o in out), " ".join("%+.2f%%" % o[1] for o in out)))
