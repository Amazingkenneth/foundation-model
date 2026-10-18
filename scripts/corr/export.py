"""export.py <model.pt> <out corr.npz>: torch MLP -> numpy weights for the cn62 estimator."""
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], weights_only=False); st = ck["state"]
lin = sorted({k.split(".")[0] for k in st if k.endswith("weight")}, key=int)
out = {"mu": np.ascontiguousarray(ck["mu"], dtype=np.float32), "sd": np.ascontiguousarray(ck["sd"], dtype=np.float32), "ysd": np.asarray(ck["ysd"], dtype=np.float32)}
for i, l in enumerate(lin):
    out[f"W{i}"] = np.ascontiguousarray(st[f"{l}.weight"].numpy().T, dtype=np.float32); out[f"b{i}"] = np.ascontiguousarray(st[f"{l}.bias"].numpy(), dtype=np.float32)
np.savez(sys.argv[2], **out); print({k: v.shape for k, v in out.items()})
