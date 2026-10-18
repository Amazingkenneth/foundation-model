"""torch_pred.py <model.pt> <data tag> <key>: corrected final prediction of the torch pipeline (for parity)."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
ck = torch.load(sys.argv[1], weights_only=False); H = ck["H"]; nf = ck["nf"]
net = torch.nn.Sequential(torch.nn.Linear(nf, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(),
                          torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, 1)); net.load_state_dict(ck["state"])
z = np.load(f"/tmp/claude-0/sc/corr/data_{sys.argv[2]}/{sys.argv[3]}.npz"); L = 16
X = z["X"].astype(np.float32); n = X.shape[1]
F = np.concatenate([X, np.broadcast_to(np.eye(L, dtype=np.float32)[:, None, :], (L, n, L))], 2)
F = (F - ck["mu"]) / ck["sd"]
with torch.no_grad(): dl = net(torch.from_numpy(F.reshape(-1, nf).astype(np.float32))).numpy().reshape(L, n) * ck["ysd"]
Ws = [np.asarray(w, dtype=np.float64) for w in hf_weights(int(z["seed"]))]; Phi = z["Phi"].astype(np.float64)
s = np.zeros(n)
for l in range(L): s = (Phi[l] * (Ws[l].T @ s) if l > 0 else 0.0) + dl[l]
P = np.load(f"/tmp/claude-0/acc/{sys.argv[2]}/pred_{sys.argv[3]}.npy")[-1].astype(np.float64)
np.save(f"/tmp/claude-0/sc/corr/torchpred_{sys.argv[3]}.npy", P + s); print("delta rms %.3e" % np.sqrt(np.mean(s ** 2)))
