"""fit_corr.py <dump_tag> <fit ids spec> <test ids spec> [feature set] [ridge]: ridge final-layer mean correction
y = truth - pk1v on dumped last-layer features; reports pooled held-out dMSE (sum over nets) with halves."""
import sys, os, glob, numpy as np
LAB = {"m": np.load("/tmp/claude-0/hfwork/mini.npz"), "f": np.load("/tmp/claude-0/hfwork/full.npz")}
def ids(spec):
    out = []
    for part in spec.split(","):
        p, r = part[0], part[1:]
        a, b = (r.split("-") + [r])[:2]
        out += [f"{p}{i}" for i in range(int(a), int(b) + 1)]
    return out
def truth(key):
    z = LAB[key[0]]; w = np.where(z["mlp_id"] == int(key[1:]))[0]
    return None if len(w) == 0 else z["means"][w[0]][-1].astype(np.float64)
FS = {
 "base13": lambda d: [d["mu"], d["var"], d["sigma"], d["alpha"], np.abs(d["alpha"]), d["phi"], d["Phi"], d["pk1v"],
                      d["D3"], d["K4_sigma"], d["sigma"] * d["phi"]],
 "blend": lambda d: [d["phi"] * (d["D3"] - d["D3raw"]), d["phi"] * d["alpha"] * (d["g4row"] - d["g4raw"]), d["phi"] * d["D3"], d["pk1v"]],
 "rich": lambda d: [d["mu"], d["var"], d["sigma"], d["alpha"], np.abs(d["alpha"]), d["phi"], d["Phi"], d["pk1v"],
                    d["D3"], d["K4_sigma"], d["sigma"] * d["phi"], d["k4_prev"], d["K4_vec"], d["pk2v"], d["pk3v"],
                    d["pk4v"], d["D3"] * d["phi"], d["D3"] * d["alpha"] * d["phi"], d["K4_vec"] * d["phi"],
                    d["k4_prev"] * d["phi"], d["sigma"] * d["phi"] * d["alpha"] ** 2, d["D3"] ** 2 * d["phi"]],
}
def load(tag, keys, fs):
    X, Y, P, K = [], [], [], []
    for k in keys:
        p = f"/tmp/claude-0/acc/{tag}/dump_{k}.npz"
        t = truth(k)
        if not os.path.exists(p) or t is None: continue
        d = dict(np.load(p))
        F = np.stack([np.ones_like(d["mu"])] + FS[fs](d), 1)
        X.append(F); Y.append(t - d["pk1v"]); P.append(d["pk1v"]); K.append(k)
    return X, Y, K
tag, fspec, tspec = sys.argv[1:4]
fs = sys.argv[4] if len(sys.argv) > 4 else "base13"
lams = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1e-6,1e-4,1e-2,1").split(",")]
Xf, Yf, Kf = load(tag, ids(fspec), fs); Xt, Yt, Kt = load(tag, ids(tspec), fs)
A = np.concatenate(Xf); y = np.concatenate(Yf)
mu, sd = A[:, 1:].mean(0), A[:, 1:].std(0) + 1e-30
Z = lambda X: np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], 1)
Az = Z(A); G = Az.T @ Az; g = Az.T @ y
print(f"fit nets {len(Kf)} test nets {len(Kt)} features {A.shape[1]} ({fs})")
base = np.array([np.mean(t ** 2) for t in Yt])
for lam in lams:
    b = np.linalg.solve(G + lam * len(y) * np.eye(len(g)) * np.r_[0, np.ones(len(g) - 1)], g)
    new = np.array([np.mean((t - Z(X) @ b) ** 2) for X, t in zip(Xt, Yt)])
    fit = np.mean([np.mean((t - Z(X) @ b) ** 2) / 1 for X, t in zip(Xf, Yf)]) / np.mean([np.mean(t ** 2) for t in Yf]) - 1
    d = new - base; ev = np.array([int(k[1:]) % 2 == 0 for k in Kt])
    rng = np.random.default_rng(0); bs = [d[i].sum() / base[i].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(500))]
    print(f"lam {lam:g}: test pooled {d.sum()/base.sum():+.2%} (se {np.std(bs):.2%}) better {np.sum(d<0)}/{len(d)} "
          f"even {d[ev].sum()/base[ev].sum():+.2%} odd {d[~ev].sum()/base[~ev].sum():+.2%} | fit-set {fit:+.2%} | base raw {base.mean():.3e}")
np.save(f"/tmp/claude-0/sc/fit/beta_{tag}_{fs}.npy", np.stack([b, np.r_[0, mu], np.r_[1, sd]]))
