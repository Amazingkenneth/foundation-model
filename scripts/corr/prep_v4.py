"""prep.py <tag> <ids spec>: per network, created errors and features for the carried corrector.
Writes /tmp/claude-0/sc/corr/data_<tag>/<key>.npz with
  X (16, n, nf) features, C (16, n) created error c_l = e_l - Phi_l*(W_l^T e_{l-1}), E (16, n) chain error e_l,
  Phi (16, n). The features at layer l: local chain stats, plus the previous layer's stats carried through W_l^T
  and (for variances) through (W_l*W_l)^T."""
import sys, os, numpy as np
sys.path.insert(0, "/tmp/claude-0/hfwork")
from regen import hf_weights
LAB = {"m": np.load("/tmp/claude-0/hfwork/mini.npz"), "f": np.load("/tmp/claude-0/hfwork/full.npz")}
LOC = ["mu", "var", "sigma", "alpha", "phi", "Phi", "pk1v", "pk2v", "pk3v", "pk4v", "D3", "g4row", "k4_prev", "K4_vec",
       "D21r", "D21c", "D21a", "Co2r", "Coar", "Rho2r",    # v3: + pre-activation correlation structure
       "Rho4r", "Rhosr", "Rho2max", "D21qr", "D21qc", "W431a", "W4ma"]   # v4 recorder stats
CAR = ["pk1v", "pk2v", "D3", "g4row", "D21r"]          # previous layer, carried through W^T
CARW = ["pk2v", "var", "Rho2r"]                         # previous layer, carried through (W*W)^T (v3: + Rho2r)
IPR = ["var", "pk2v"]   # v4: participation ratio of the incoming variance sum_i W_ij^2 v_i (W^4 v^2 / (W^2 v)^2)
def ids(spec):
    out = []
    for part in spec.split(","):
        p, r = part[0], part[1:]; a, b = (r.split("-") + [r])[:2]
        out += [f"{p}{i}" for i in range(int(a), int(b) + 1)]
    return out
tag, spec = sys.argv[1], sys.argv[2]
od = f"/tmp/claude-0/sc/corr/data_{tag}_v4"; os.makedirs(od, exist_ok=True)
for key in ids(spec):
    dp = f"/tmp/claude-0/acc/{tag}/ldump_{key}.npz"; pp = f"/tmp/claude-0/acc/{tag}/pred_{key}.npy"
    if not (os.path.exists(dp) and os.path.exists(pp)) or os.path.exists(f"{od}/{key}.npz"): continue
    z = LAB[key[0]]; w = np.where(z["mlp_id"] == int(key[1:]))[0]
    if len(w) == 0: continue
    T = z["means"][w[0]].astype(np.float64); P = np.load(pp).astype(np.float64); d = np.load(dp)
    Ws = [np.asarray(x, dtype=np.float64) for x in hf_weights(z["mlp_seed"][w[0]])]
    L, n = P.shape
    E = P - T; Phi = d["Phi"].astype(np.float64)
    C = np.empty_like(E)
    for l in range(L):
        prop = Phi[l] * (Ws[l].T @ E[l - 1]) if l > 0 else 0.0
        C[l] = E[l] - prop
    feats = []
    for l in range(L):
        f = [d[k][l] if k in d else np.zeros(n) for k in LOC]
        if l > 0:
            W2 = Ws[l] ** 2; W4 = W2 * W2
            f += [Ws[l].T @ d[k][l - 1] for k in CAR] + [W2.T @ d[k][l - 1] for k in CARW]
            for k in IPR:
                v = np.maximum(d[k][l - 1], 0.0); a = W2.T @ v
                f.append((W4.T @ (v * v)) / np.maximum(a * a, 1e-300))
        else:
            f += [np.zeros(n)] * (len(CAR) + len(CARW) + len(IPR))
        feats.append(np.stack(f, 1))
    np.savez(f"{od}/{key}.npz", X=np.stack(feats).astype(np.float32), C=C.astype(np.float32), E=E.astype(np.float32),
             Phi=Phi.astype(np.float32), seed=z["mlp_seed"][w[0]],
             names=np.array(LOC + ["c" + k for k in CAR] + ["w" + k for k in CARW] + ["ipr_" + k for k in IPR]))
    print(key, "ok", flush=True)
