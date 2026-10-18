"""fitmlp.py <data tag> <train spec> <test spec> [epochs] [hidden] [out]: per-layer created-error MLP,
delta_l = f([X_l standardized, layer one-hot]) fit to -C_l with layer weights rho^(15-l); scored on the test nets
by propagating delta through Phi_l W_l^T (final-layer MSE ratio)."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
from prep import ids
torch.set_num_threads(int(os.environ.get("NT", "3")))
tag, trs, tes = sys.argv[1:4]
torch.manual_seed(int(os.environ.get("SEED", "0")))
EP = int(sys.argv[4]) if len(sys.argv) > 4 else 10; H = int(sys.argv[5]) if len(sys.argv) > 5 else 64
OUT = sys.argv[6] if len(sys.argv) > 6 else f"/tmp/claude-0/sc/corr/mlp_{tag}.pt"
RHO = float(os.environ.get("RHO", "0.75"))
# WMODE=gain: layer weights = measured propagation gain to the final layer (gain.py on m0-5), not rho^(15-l)
GAIN = [0.031, 0.052, 0.076, 0.107, 0.139, 0.180, 0.223, 0.282, 0.349, 0.412, 0.502, 0.580, 0.656, 0.765, 0.883, 1.0]
WMODE = os.environ.get("WMODE", "rho"); ACT = os.environ.get("ACT", "gelu"); EPS = int(os.environ.get("SEED", "0"))
D = f"/tmp/claude-0/sc/corr/data_{tag}"
tr = [k for k in ids(trs) if os.path.exists(f"{D}/{k}.npz")]; te = [k for k in ids(tes) if os.path.exists(f"{D}/{k}.npz")]
print(f"train {len(tr)} test {len(te)}", flush=True)
L = 16
def feats(z):
    X = z["X"].astype(np.float32); n = X.shape[1]
    oh = np.broadcast_to(np.eye(L, dtype=np.float32)[:, None, :], (L, n, L))
    return np.concatenate([X, oh], 2)
Xs, Ys, Ws_ = [], [], []
for k in tr:
    z = np.load(f"{D}/{k}.npz"); Xs.append(feats(z)); Ys.append(-z["C"].astype(np.float32))
X = np.concatenate(Xs, 1); Y = np.concatenate(Ys, 1); del Xs, Ys   # (L, N, nf), (L, N)
Xf = X.reshape(-1, X.shape[2]); s1 = np.zeros(Xf.shape[1]); s2 = np.zeros(Xf.shape[1])
for i in range(0, Xf.shape[0], 1 << 20):
    c = Xf[i:i + (1 << 20)].astype(np.float64); s1 += c.sum(0); s2 += (c * c).sum(0)
mu = (s1 / Xf.shape[0]).astype(np.float32); sd = (np.sqrt(np.maximum(s2 / Xf.shape[0] - (s1 / Xf.shape[0]) ** 2, 0)) + 1e-12).astype(np.float32)
mu[-L:] = 0; sd[-L:] = 1
ysd = float(Y.std())
wl = np.array(GAIN if WMODE == "gain" else [RHO ** (L - 1 - l) for l in range(L)], np.float32)
Xf -= mu; Xf /= sd; Xt = torch.from_numpy(Xf)                   # in place: X is not used after this
nX = X.shape[1]; del X, Xf
Yt = torch.from_numpy((Y / ysd).reshape(-1).astype(np.float32))
Wt = torch.from_numpy(np.repeat(wl, nX))
nf = Xt.shape[1]
A = torch.nn.ReLU if ACT == "relu" else torch.nn.GELU
net = torch.nn.Sequential(torch.nn.Linear(nf, H), A(), torch.nn.Linear(H, H), A(),
                          torch.nn.Linear(H, H), A(), torch.nn.Linear(H, 1))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
N = Xt.shape[0]; bs = 8192
for ep in range(EP):
    perm = torch.randperm(N); tot = 0.0
    for i in range(0, N, bs):
        idx = perm[i:i + bs]
        p = net(Xt[idx]).squeeze(1)
        loss = (Wt[idx] * (p - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    for g in opt.param_groups: g["lr"] *= 0.8
    print(f"epoch {ep}: train weighted loss {tot / N:.4f} (var-normalised; 1 = no fit)", flush=True)
torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, ysd=ysd, H=H, nf=nf, act=ACT), OUT)
# ---- evaluation by propagation
base, corr = [], []; PL0 = np.zeros(L); PL1 = np.zeros(L)
with torch.no_grad():
    for k in te:
        z = np.load(f"{D}/{k}.npz"); Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]
        F = (feats(z) - mu) / sd
        dl = net(torch.from_numpy(F.reshape(-1, nf).astype(np.float32))).numpy().reshape(L, -1).astype(np.float64) * ysd
        Phi = z["Phi"].astype(np.float64); s = np.zeros(dl.shape[1]); Cz = z["C"].astype(np.float64)
        for l in range(L):
            a, b = Cz[l].copy(), Cz[l] + dl[l]
            for m in range(l + 1, L):
                a = Phi[m] * (Ws[m].T @ a); b = Phi[m] * (Ws[m].T @ b)
            PL0[l] += np.sum(a ** 2); PL1[l] += np.sum(b ** 2)
        for l in range(L):
            s = (Phi[l] * (Ws[l].T @ s) if l > 0 else 0.0) + dl[l]
        eL = z["E"][-1].astype(np.float64); base.append(np.mean(eL ** 2)); corr.append(np.mean((eL + s) ** 2))
base, corr = np.array(base), np.array(corr)
ev = np.array([int(k[1:]) % 2 == 0 for k in te])
print("per-layer propagated created-error change:", " ".join("%+.1f%%" % (100 * (PL1[l] / PL0[l] - 1)) for l in range(L)))
print(f"TEST final-layer dMSE {corr.sum()/base.sum()-1:+.2%} (even {corr[ev].sum()/base[ev].sum()-1:+.2%} "
      f"odd {corr[~ev].sum()/base[~ev].sum()-1:+.2%}), better {np.sum(corr < base)}/{len(te)}", flush=True)
