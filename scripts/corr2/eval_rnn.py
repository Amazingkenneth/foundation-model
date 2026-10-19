"""eval_rnn.py <model.pt> <data tag> <keys file or spec>: held-out final-layer dMSE of a fitrnn model on given networks."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
def ids(spec):
    out = []
    for part in spec.split(","):
        p, r = part[0], part[1:]; a, b = (r.split("-") + [r])[:2]
        out += [f"{p}{i}" for i in range(int(a), int(b) + 1)]
    return out
mp, tag, spec = sys.argv[1:4]
keys = open(spec).read().split() if os.path.exists(spec) else ids(spec)
ck = torch.load(mp, weights_only=False); L = 16; H, S, nx = ck["H"], ck["S"], ck["nx"]; S2 = ck.get("S2", 0); S3 = ck.get("S3", 0)
A = torch.nn.ReLU if ck.get("act") == "relu" else torch.nn.GELU
f = torch.nn.Sequential(torch.nn.Linear(nx + L + S + S2 + S3, H), A(), torch.nn.Linear(H, H), A(), torch.nn.Linear(H, H), A(),
                        torch.nn.Linear(H, 1 + S + S2 + S3))
f.load_state_dict({k[2:]: v for k, v in ck["state"].items() if k.startswith("f.")})
D = f"/tmp/claude-0/sc/corr/data_{tag}"; base, corr, kk, es, ss, GG, bb = [], [], [], [], [], [], []
LAYERFIT = os.environ.get("LAYERFIT", "0") == "1"   # per-layer scales alpha_l, fit on one half, tested on the other
torch.set_num_threads(2)
with torch.no_grad():
    for k in keys:
        if not os.path.exists(f"{D}/{k}.npz"): continue
        z = np.load(f"{D}/{k}.npz"); Ws = [torch.from_numpy(w) for w in hf_weights(int(z["seed"]))]
        Xr = z["X"]
        if "names" in ck and "names" in z.files:     # a DROP-trained model reads its kept columns by name
            nm = [str(x) for x in z["names"]]; Xr = Xr[..., [nm.index(c) for c in ck["names"]]]
        X = torch.from_numpy(((Xr.astype(np.float32) - ck["mu"]) / ck["sd"]).astype(np.float32)); n = X.shape[1]
        Phi = z["Phi"].astype(np.float64); s = np.zeros(n); st = st2 = None; P = np.zeros((L, n))   # P[l]: delta_l propagated to L
        for l in range(L):
            parts = [X[l], torch.eye(L)[l].expand(n, L)]
            if S: parts.append(torch.zeros(n, S) if st is None else (Ws[l].T @ st) / torch.sqrt(n * (Ws[l] ** 2).mean()))
            if S2:
                W2 = Ws[l] * Ws[l]; parts.append(torch.zeros(n, S2) if st2 is None else (W2.T @ st2) / (W2.sum() / n))
            o = f(torch.cat(parts, 1)); d = o[:, 0].double().numpy() * ck["ysd"]
            if S: st = o[:, 1:1 + S]
            if S2: st2 = o[:, 1 + S:1 + S + S2]
            Wt = Ws[l].T.double().numpy()
            s = (Phi[l] * (Wt @ s) if l > 0 else 0.0) + d
            if LAYERFIT:
                if l > 0: P[:l] = (Wt @ P[:l].T).T * Phi[l]
                P[l] = d
        eL = z["E"][-1].astype(np.float64); base.append(np.mean(eL ** 2)); corr.append(np.mean((eL + s) ** 2)); kk.append(k)
        es.append(np.mean(eL * s)); ss.append(np.mean(s * s))
        if LAYERFIT: GG.append(P @ P.T / n); bb.append(P @ eL / n)
base, corr = np.array(base), np.array(corr); ev = np.array([int(k[1:]) % 2 == 0 for k in kk])
print(f"{os.path.basename(mp)} on {len(kk)} nets: dMSE {corr.sum()/base.sum()-1:+.2%} (even {corr[ev].sum()/base[ev].sum()-1:+.2%} odd {corr[~ev].sum()/base[~ev].sum()-1:+.2%}) better {np.sum(corr<base)}/{len(kk)}")
es, ss = np.array(es), np.array(ss)
for a in [float(x) for x in os.environ.get("ALPHAS", "").split(",") if x]:   # scaled correction a * s_L
    c = base + 2 * a * es + a * a * ss
    print(f"  alpha {a:.2f}: dMSE {c.sum()/base.sum()-1:+.2%} (even {c[ev].sum()/base[ev].sum()-1:+.2%} odd {c[~ev].sum()/base[~ev].sum()-1:+.2%})")
if LAYERFIT:
    GG, bb = np.array(GG), np.array(bb); tot = 0.0
    for h, nm in ((ev, "even"), (~ev, "odd")):
        a = -np.linalg.solve(GG[~h].sum(0), bb[~h].sum(0))         # fit on the other half
        c = base[h] + 2 * bb[h] @ a + np.einsum("i,nij,j->n", a, GG[h], a); tot += c.sum()
        print(f"  layer alphas fit on the other half, test {nm}: dMSE {c.sum()/base[h].sum()-1:+.2%}  "
              f"alpha {np.array2string(a, precision=2, max_line_width=200)}")
    print(f"  layer alphas cross-fit total dMSE {tot/base.sum()-1:+.2%}")
