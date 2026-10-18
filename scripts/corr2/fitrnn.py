"""fitrnn.py <data tag> <train spec> <test spec> <epochs> <H> <S> <out>: the carried corrector with a learned S-dim
per-neuron state carried through W (18254-style). At layer l the MLP reads [X_l standardized, layer one-hot,
W_l^T state_{l-1} / sqrt(n * mean W^2)] and returns delta_l (fit to -C_l, layer weights = propagation gain or rho^(15-l))
and state_l. S = 0 is the plain per-layer corrector trained the same way (one network per step).
Weights are regenerated per network (hf_weights) in DataLoader workers."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from prep import ids
tag, trs, tes = sys.argv[1:4]
EP, H, S = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]); OUT = sys.argv[7]
NT = int(os.environ.get("NT", "2")); NW = int(os.environ.get("NW", "2")); torch.set_num_threads(NT)
torch.manual_seed(int(os.environ.get("SEED", "0")))
WMODE = os.environ.get("WMODE", "gain"); RHO = float(os.environ.get("RHO", "0.75")); ACT = os.environ.get("ACT", "gelu")
LR = float(os.environ.get("LR", "2e-3")); DEC = float(os.environ.get("DEC", "0.8"))
GAIN = [0.031, 0.052, 0.076, 0.107, 0.139, 0.180, 0.223, 0.282, 0.349, 0.412, 0.502, 0.580, 0.656, 0.765, 0.883, 1.0]
L = 16
D = f"/tmp/claude-0/sc/corr/data_{tag}"
tr = [k for k in ids(trs) if os.path.exists(f"{D}/{k}.npz")]; te = [k for k in ids(tes) if os.path.exists(f"{D}/{k}.npz")]
print(f"train {len(tr)} test {len(te)} H {H} S {S} {WMODE} {ACT}", flush=True)
# standardisation from (up to) 200 training networks
acc = []
for k in tr[:200]:
    z = np.load(f"{D}/{k}.npz"); acc.append((z["X"].astype(np.float64).reshape(-1, z["X"].shape[2]), z["C"].astype(np.float64).ravel()))
Xa = np.concatenate([a for a, _ in acc]); mu = Xa.mean(0).astype(np.float32); sd = (Xa.std(0) + 1e-12).astype(np.float32)
ysd = float(np.concatenate([b for _, b in acc]).std()); del acc, Xa
nx = mu.shape[0]
wl = torch.tensor(GAIN if WMODE == "gain" else [RHO ** (L - 1 - l) for l in range(L)], dtype=torch.float32)


class DS(torch.utils.data.Dataset):
    def __init__(self, keys): self.keys = keys
    def __len__(self): return len(self.keys)
    def __getitem__(self, i):
        from regen import hf_weights
        z = np.load(f"{D}/{self.keys[i]}.npz")
        X = ((z["X"].astype(np.float32) - mu) / sd).astype(np.float32)
        Ws = np.stack(hf_weights(int(z["seed"])))                              # (L, n, n) float32
        return (torch.from_numpy(X), torch.from_numpy((-z["C"] / ysd).astype(np.float32)), torch.from_numpy(Ws),
                torch.from_numpy(z["Phi"].astype(np.float32)), torch.from_numpy(z["E"].astype(np.float32)))


A = torch.nn.ReLU if ACT == "relu" else torch.nn.GELU


class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.f = torch.nn.Sequential(torch.nn.Linear(nx + L + S, H), A(), torch.nn.Linear(H, H), A(),
                                  torch.nn.Linear(H, H), A(), torch.nn.Linear(H, 1 + S))
    def forward(s, X, Ws):                                                    # X (L, n, nx), Ws (L, n, n)
        n = X.shape[1]; eye = torch.eye(L); out = []; st = None
        for l in range(L):
            parts = [X[l], eye[l].expand(n, L)]
            if S:
                cin = torch.zeros(n, S) if st is None else (Ws[l].T @ st) * Ws_scale(Ws[l])
                parts.append(cin)
            o = s.f(torch.cat(parts, 1)); out.append(o[:, 0])
            if S: st = o[:, 1:]
        return torch.stack(out)                                               # (L, n)


def Ws_scale(W):                                                              # W^T st has ~ n * mean(W^2) x |st|^2
    return 1.0 / torch.sqrt(W.shape[0] * (W * W).mean())


net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR)
dl = torch.utils.data.DataLoader(DS(tr), batch_size=None, shuffle=True, num_workers=NW, persistent_workers=True, prefetch_factor=2)
for ep in range(EP):
    tot, tb = 0.0, 0.0
    for X, Y, Ws, Phi, E in dl:
        p = net(X, Ws)
        loss = ((p - Y) ** 2).mean(1) @ wl / wl.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); tb += float(((Y ** 2).mean(1) @ wl / wl.sum()).item())
    for g in opt.param_groups: g["lr"] *= DEC
    print(f"epoch {ep}: weighted loss / no-fit {tot / tb:.4f}", flush=True)
torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, ysd=ysd, H=H, S=S, nx=nx, act=ACT), OUT)
base, corr = [], []
with torch.no_grad():
    for X, Y, Ws, Phi, E in torch.utils.data.DataLoader(DS(te), batch_size=None, shuffle=False, num_workers=NW):
        dlt = net(X, Ws).double() * ysd; Phi = Phi.double(); W64 = Ws.double(); s = torch.zeros(X.shape[1], dtype=torch.float64)
        for l in range(L):
            s = (Phi[l] * (W64[l].T @ s) if l > 0 else 0.0) + dlt[l]
        eL = E[-1].double(); base.append(float((eL ** 2).mean())); corr.append(float(((eL + s) ** 2).mean()))
base, corr = np.array(base), np.array(corr)
ev = np.array([int(k[1:]) % 2 == 0 for k in te])
print(f"TEST final-layer dMSE {corr.sum()/base.sum()-1:+.2%} (even {corr[ev].sum()/base[ev].sum()-1:+.2%} "
      f"odd {corr[~ev].sum()/base[~ev].sum()-1:+.2%}), better {np.sum(corr < base)}/{len(te)}", flush=True)
