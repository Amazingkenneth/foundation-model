"""fitmlp2.py <data tag> <train> <test> <stage1 model.pt> [epochs] [hidden] [out]: two-stage corrector with the
incoming correction inc_l = Phi_l * (W_l^T s_{l-1}) as extra inputs (inc, inc/sigma). Stage-1 model gives s for
the training features; the test score runs the recursion self-consistently with the stage-2 model."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
from prep import ids
torch.set_num_threads(int(os.environ.get("NT", "3"))); torch.manual_seed(0)
tag, trs, tes, m1 = sys.argv[1:5]
EP = int(sys.argv[5]) if len(sys.argv) > 5 else 12; H = int(sys.argv[6]) if len(sys.argv) > 6 else 128
OUT = sys.argv[7] if len(sys.argv) > 7 else f"/tmp/claude-0/sc/corr/mlp2_{tag}.pt"
RHO = float(os.environ.get("RHO", "0.75")); L = 16
D = f"/tmp/claude-0/sc/corr/data_{tag}"
tr = [k for k in ids(trs) if os.path.exists(f"{D}/{k}.npz")]; te = [k for k in ids(tes) if os.path.exists(f"{D}/{k}.npz")]
def mk(nf, H):
    return torch.nn.Sequential(torch.nn.Linear(nf, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(),
                               torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, 1))
c1 = torch.load(m1, weights_only=False); net1 = mk(c1["nf"], c1["H"]); net1.load_state_dict(c1["state"])
def base_feats(z):
    X = z["X"].astype(np.float32); n = X.shape[1]
    return np.concatenate([X, np.broadcast_to(np.eye(L, dtype=np.float32)[:, None, :], (L, n, L))], 2)
def run(z, Ws, f):
    """recursion with a per-layer predictor f(l, inc) -> delta_l (n,); returns s_L and the inc features."""
    Phi = z["Phi"].astype(np.float64); sig = z["X"][:, :, 2].astype(np.float64)
    s = None; incs = []
    for l in range(L):
        inc = Phi[l] * (Ws[l].T @ s) if s is not None else np.zeros(Phi.shape[1])
        incs.append(np.stack([inc, inc / np.maximum(sig[l], 1e-6)], 1))
        d = f(l, incs[-1]); s = d if s is None else inc + d
    return s, np.stack(incs)
def f1(z):
    F = (base_feats(z) - c1["mu"]) / c1["sd"]
    with torch.no_grad():
        dl = net1(torch.from_numpy(F.reshape(-1, c1["nf"]).astype(np.float32))).numpy().reshape(L, -1).astype(np.float64) * c1["ysd"]
    return lambda l, inc: dl[l]
Xs, Ys = [], []
for k in tr:
    z = np.load(f"{D}/{k}.npz"); Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]
    _, inc = run(z, Ws, f1(z))
    Xs.append(np.concatenate([base_feats(z), inc.astype(np.float32)], 2)); Ys.append(-z["C"].astype(np.float32))
print(f"train {len(tr)} test {len(te)} (stage-1 s computed)", flush=True)
X = np.concatenate(Xs, 1); Y = np.concatenate(Ys, 1); nf = X.shape[2]
mu = X.reshape(-1, nf).mean(0); sd = X.reshape(-1, nf).std(0) + 1e-12; mu[24:40] = 0; sd[24:40] = 1
ysd = float(Y.std()); wl = np.array([RHO ** (L - 1 - l) for l in range(L)], np.float32)
Xt = torch.from_numpy(((X - mu) / sd).reshape(-1, nf).astype(np.float32)); Yt = torch.from_numpy((Y / ysd).reshape(-1))
Wt = torch.from_numpy(np.repeat(wl, X.shape[1])); del X, Xs
net = mk(nf, H); opt = torch.optim.Adam(net.parameters(), lr=2e-3); N = Xt.shape[0]; bs = 8192
for ep in range(EP):
    perm = torch.randperm(N); tot = 0.0
    for i in range(0, N, bs):
        idx = perm[i:i + bs]; p = net(Xt[idx]).squeeze(1)
        loss = (Wt[idx] * (p - Yt[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    for g in opt.param_groups: g["lr"] *= 0.8
    print(f"epoch {ep}: {tot / N:.4f}", flush=True)
torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, ysd=ysd, H=H, nf=nf), OUT)
base, corr, corr1 = [], [], []
with torch.no_grad():
    for k in te:
        z = np.load(f"{D}/{k}.npz"); Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]
        BF = base_feats(z)
        def f2(l, inc):
            x = (np.concatenate([BF[l], inc.astype(np.float32)], 1) - mu) / sd
            return net(torch.from_numpy(x.astype(np.float32))).numpy()[:, 0].astype(np.float64) * ysd
        s2, _ = run(z, Ws, f2); s1, _ = run(z, Ws, f1(z))
        eL = z["E"][-1].astype(np.float64); base.append(np.mean(eL ** 2)); corr.append(np.mean((eL + s2) ** 2)); corr1.append(np.mean((eL + s1) ** 2))
base, corr, corr1 = map(np.array, (base, corr, corr1)); ev = np.array([int(k[1:]) % 2 == 0 for k in te])
print(f"TEST stage1 {corr1.sum()/base.sum()-1:+.2%} | stage2 {corr.sum()/base.sum()-1:+.2%} (even {corr[ev].sum()/base[ev].sum()-1:+.2%} odd {corr[~ev].sum()/base[~ev].sum()-1:+.2%}), better {np.sum(corr<base)}/{len(te)}", flush=True)
