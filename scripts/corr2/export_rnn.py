"""export_rnn.py <model.pt> <out corr.npz>: fitrnn / fitmlp3 torch model -> corr.npz (C order) for the cn62/cn63
estimators. Inputs are [24 features, 16 one-hot, S state]; mu / sd are padded with 0 / 1 past the 24 features."""
import sys, numpy as np, torch
ck = torch.load(sys.argv[1], weights_only=False); st = ck["state"]
pre = "f." if any(k.startswith("f.") for k in st) else ""
lin = sorted({int(k[len(pre):].split(".")[0]) for k in st if k.startswith(pre) and k.endswith("weight")})
W0 = st[f"{pre}{lin[0]}.weight"].numpy().T; nin = W0.shape[0]
mu = np.zeros(nin, np.float32); sd = np.ones(nin, np.float32)
m = np.asarray(ck["mu"], np.float32); s = np.asarray(ck["sd"], np.float32)
mu[:len(m)] = m; sd[:len(s)] = s
out = {"mu": np.ascontiguousarray(mu), "sd": np.ascontiguousarray(sd), "ysd": np.asarray(ck["ysd"], dtype=np.float32)}
for i, l in enumerate(lin):
    out[f"W{i}"] = np.ascontiguousarray(st[f"{pre}{l}.weight"].numpy().T, dtype=np.float32)
    out[f"b{i}"] = np.ascontiguousarray(st[f"{pre}{l}.bias"].numpy(), dtype=np.float32)
if ck.get("act", "gelu") == "relu": out["relu"] = np.ones(1, np.float32)
if ck.get("S2", 0): out["nS2"] = np.array([float(ck["S2"])], np.float32)
if ck.get("S3", 0): out["nS3"] = np.array([float(ck["S3"])], np.float32)
V4_FEATS = ("Rho4r", "Rhosr", "Rho2max", "D21qr", "D21qc", "W431a", "W4ma", "ipr_var", "ipr_pk2v")
if "names" in ck and "Rho2r" in list(ck["names"]):   # prep_v4 data: v3 + a subset of the v4 features, named
    nm = list(ck["names"]); assert "Rho2r" in nm and "wRho2r" in nm, nm
    exp = [k for k in nm if k not in V4_FEATS]
    assert len(exp) == 28, exp          # all v3 features kept, in order
    out["v3"] = np.ones(1, np.float32)
    if any(k in V4_FEATS for k in nm): out["v4"] = np.array([float(k in nm) for k in V4_FEATS], np.float32)
elif int(ck.get("nx", 24)) == 28: out["v3"] = np.ones(1, np.float32)   # v3 inputs (20 local + 8 carried features)
if ck.get("root"):
    assert [tuple(x) for x in ck["root"]] == [(7, 2), (8, 3), (9, 4)], ck["root"]
    out["root"] = np.ones(1, np.float32)   # pk2v, pk3v, pk4v as signed square / cube / fourth roots
np.savez(sys.argv[2], **out); print({k: v.shape for k, v in out.items()})
