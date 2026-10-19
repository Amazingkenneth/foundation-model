"""torch_pred_rnn.py <model.pt> <tag> <key>: torch-side corrected final prediction of a fitrnn model (parity check)."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr2")
from regen import hf_weights
mp, tag, key = sys.argv[1:4]
ck = torch.load(mp, weights_only=False); L = 16; H, S, nx = ck["H"], ck["S"], ck["nx"]; S2 = ck.get("S2", 0); S3 = ck.get("S3", 0)
A = torch.nn.ReLU if ck.get("act") == "relu" else torch.nn.GELU
f = torch.nn.Sequential(torch.nn.Linear(nx + L + S + S2 + S3, H), A(), torch.nn.Linear(H, H), A(), torch.nn.Linear(H, H), A(),
                        torch.nn.Linear(H, 1 + S + S2 + S3))
f.load_state_dict({k[2:]: v for k, v in ck["state"].items()})
z = np.load(f"/tmp/claude-0/sc/corr/data_{tag}/{key}.npz"); Ws = [torch.from_numpy(w) for w in hf_weights(int(z["seed"]))]
X = torch.from_numpy(((z["X"].astype(np.float32) - ck["mu"]) / ck["sd"]).astype(np.float32)); n = X.shape[1]
Phi = z["Phi"].astype(np.float64); s = np.zeros(n); st = None; st2 = None; st3 = None
with torch.no_grad():
    for l in range(L):
        parts = [X[l], torch.eye(L)[l].expand(n, L)]
        if S:
            parts.append(torch.zeros(n, S) if st is None else (Ws[l].T @ st) / torch.sqrt(n * (Ws[l] ** 2).mean()))
        if S2:
            W2 = Ws[l] * Ws[l]
            parts.append(torch.zeros(n, S2) if st2 is None else (W2.T @ st2) / (W2.sum() / n))
        if S3:
            W3 = Ws[l] * Ws[l] * Ws[l]
            parts.append(torch.zeros(n, S3) if st3 is None else (W3.T @ st3) / torch.sqrt((W3 * W3).sum() / n))
        o = f(torch.cat(parts, 1)); d = o[:, 0].double().numpy() * ck["ysd"]
        if S: st = o[:, 1:1 + S]
        if S2: st2 = o[:, 1 + S:1 + S + S2]
        if S3: st3 = o[:, 1 + S + S2:]
        s = (Phi[l] * (Ws[l].T.double().numpy() @ s) if l > 0 else 0.0) + d
p = np.load(f"/tmp/claude-0/acc/{tag}/pred_{key}.npy")[-1].astype(np.float64) + s
np.save(f"/tmp/claude-0/sc/corr2/torchpred_{key}.npy", p); print("delta rms %.3e" % np.sqrt(np.mean(s ** 2)))
