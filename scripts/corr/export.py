"""export.py <model.pt> <out corr.npz>: torch MLP -> numpy weights for the cn62 estimator."""
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], weights_only=False); st = ck["state"]
lin = sorted({k.split(".")[0] for k in st if k.endswith("weight")}, key=int)
out = {"mu": ck["mu"].astype(np.float32), "sd": ck["sd"].astype(np.float32), "ysd": np.float32(ck["ysd"])}
for i, l in enumerate(lin):
    out[f"W{i}"] = st[f"{l}.weight"].numpy().T.astype(np.float32); out[f"b{i}"] = st[f"{l}.bias"].numpy().astype(np.float32)
np.savez(sys.argv[2], **out); print({k: v.shape for k, v in out.items()})
