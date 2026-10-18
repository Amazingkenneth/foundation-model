"""fitlin.py <data tag> <train spec> <test spec> [lams]: linear carried corrector.
(a) per-layer ridge of -C_l on [1, X_l]; (b) joint ridge of the final error on the propagated features
G_l X_l (G_l = Phi_L W_L^T ... Phi_{l+1} W_{l+1}^T); both scored on the test networks by propagating
delta_l = X_l beta_l with s_l = Phi_l*(W_l^T s_{l-1}) + delta_l."""
import sys, os, glob, numpy as np
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
from prep import ids
tag, trs, tes = sys.argv[1:4]
lams = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1e-3,1e-1,10").split(",")]
D = f"/tmp/claude-0/sc/corr/data_{tag}"
def load(k):
    p = f"{D}/{k}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None
def prop_feats(z, Ws):
    """G_l X_l for every l: propagate each layer's feature block to the output layer."""
    X = z["X"].astype(np.float64); Phi = z["Phi"].astype(np.float64); L, n, nf = X.shape
    Xa = np.concatenate([np.ones((L, n, 1)), X], 2)
    out = np.zeros((L, n, nf + 1))
    for l in range(L):
        g = Xa[l]
        for m in range(l + 1, L):
            g = Phi[m][:, None] * (Ws[m].T @ g)
        out[l] = g
    return out
def stats(X):
    return X.reshape(-1, X.shape[-1]).std(0) + 1e-30
tr = [k for k in ids(trs) if os.path.exists(f"{D}/{k}.npz")]; te = [k for k in ids(tes) if os.path.exists(f"{D}/{k}.npz")]
print(f"train {len(tr)} test {len(te)}", flush=True)
# ---- (a) per-layer ridge on created errors
L = 16; Z0 = load(tr[0]); nf = Z0["X"].shape[2] + 1
G = np.zeros((L, nf, nf)); g = np.zeros((L, nf)); sc = None
for k in tr:
    z = load(k); X = np.concatenate([np.ones((L, z["X"].shape[1], 1)), z["X"].astype(np.float64)], 2)
    if sc is None: sc = np.r_[1.0, stats(X[:, :, 1:])]
    Xs = X / sc
    for l in range(L):
        G[l] += Xs[l].T @ Xs[l]; g[l] += Xs[l].T @ (-z["C"][l].astype(np.float64))
# ---- (b) joint final-objective ridge on propagated features
cache = {}
GG = np.zeros((L * nf, L * nf)); gg = np.zeros(L * nf)
for k in tr:
    z = load(k); Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]
    PF = prop_feats(z, Ws) / sc                      # (L, n, nf)
    A = PF.transpose(1, 0, 2).reshape(PF.shape[1], -1)
    GG += A.T @ A; gg += A.T @ (-z["E"][-1].astype(np.float64))
print("fitted", flush=True)
te_data = []
for k in te:
    z = load(k); Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]
    te_data.append((k, z, Ws, prop_feats(z, Ws) / sc))
for lam in lams:
    Ba = np.stack([np.linalg.solve(G[l] + lam * np.trace(G[l]) / nf * 1e-3 * np.eye(nf), g[l]) for l in range(L)])
    Bb = np.linalg.solve(GG + lam * np.trace(GG) / GG.shape[0] * 1e-3 * np.eye(GG.shape[0]), gg).reshape(L, nf)
    ra, rb, base = [], [], []
    for k, z, Ws, PF in te_data:
        eL = z["E"][-1].astype(np.float64)
        sa = (PF * Ba[:, None, :]).sum((0, 2)); sb = (PF * Bb[:, None, :]).sum((0, 2))
        base.append(np.mean(eL ** 2)); ra.append(np.mean((eL + sa) ** 2)); rb.append(np.mean((eL + sb) ** 2))
    base, ra, rb = map(np.array, (base, ra, rb))
    ev = np.array([int(k[1:]) % 2 == 0 for k, *_ in te_data])
    f = lambda r: f"{r.sum()/base.sum()-1:+.2%} (even {r[ev].sum()/base[ev].sum()-1:+.2%} odd {r[~ev].sum()/base[~ev].sum()-1:+.2%}, better {np.sum(r<base)}/{len(r)})"
    print(f"lam {lam:g}: (a) per-layer {f(ra)} | (b) joint {f(rb)}", flush=True)
np.savez(f"/tmp/claude-0/sc/corr/lin_{tag}.npz", Bb=Bb, Ba=Ba, sc=sc)
