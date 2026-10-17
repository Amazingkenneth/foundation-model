"""jfit_hf.py <base> <tag>:<delta> ...: linear-response recalibration on PUBLIC networks (1e9-sample truth).
J_k = (p_k - p0)/delta per network/neuron (final layer); fit x by least squares (joint and per knob); held-out
dMSE on id-parity halves and on a 50/50 random split. Optional env HOLD=<tag-prefix list> not used."""
import sys, os, glob, numpy as np
LAB = {"m": np.load("/tmp/claude-0/hfwork/mini.npz")}
if os.path.exists("/tmp/claude-0/hfwork/full.npz"):
    LAB["f"] = np.load("/tmp/claude-0/hfwork/full.npz")
def truth(key):
    z = LAB[key[0]]; i = int(np.where(z["mlp_id"] == int(key[1:]))[0][0])
    return z["means"][i][-1].astype(np.float64), z["avg_var"][i] / 1e9
base = sys.argv[1]; ks = [a.split(":") for a in sys.argv[2:]]
P = lambda t, k: np.load(f"/tmp/claude-0/acc/{t}/pred_{k}.npy")[-1].astype(np.float64) if os.path.exists(f"/tmp/claude-0/acc/{t}/pred_{k}.npy") else None
keys = sorted(os.path.basename(p)[5:-4] for p in glob.glob(f"/tmp/claude-0/acc/{base}/pred_*.npy"))
keys = [k for k in keys if k[0] in "mf" and all(P(t, k) is not None for t, _ in ks)]
E0, J, NZ = {}, {}, {}
for k in keys:
    p0 = P(base, k); t, nz = truth(k); E0[k] = t - p0; NZ[k] = nz
    J[k] = np.stack([(P(tg, k) - p0) / float(d) for tg, d in ks], 1)
print("networks", len(keys), "knobs", [t for t, _ in ks])
def fit(S, cols):
    A = sum(J[s][:, cols].T @ J[s][:, cols] for s in S); b = sum(J[s][:, cols].T @ E0[s] for s in S)
    return np.linalg.solve(A + 1e-30 * np.eye(len(cols)), b)
def dmse(S, cols, x):
    num = sum(((E0[s] - J[s][:, cols] @ x) ** 2).mean() - (E0[s] ** 2).mean() for s in S)
    den = sum((E0[s] ** 2).mean() - NZ[s] for s in S)
    return 100 * num / den
rng = np.random.default_rng(1)
perm = rng.permutation(len(keys))
splits = {"parity": [int(k[1:]) % 2 == 0 for k in keys], "rand": [perm[i] < len(keys) // 2 for i in range(len(keys))],
          "split": [k[0] == "m" for k in keys]}
allc = list(range(len(ks)))
for name, cols in [("joint", allc)] + [(ks[c][0], [c]) for c in allc]:
    x = fit(keys, cols)
    out = []
    for sn, m in splits.items():
        A = [s for s, f in zip(keys, m) if f]; B = [s for s, f in zip(keys, m) if not f]
        if not A or not B: continue
        xa, xb = fit(A, cols), fit(B, cols)
        out.append("%s %+.2f/%+.2f" % (sn, dmse(B, cols, xa), dmse(A, cols, xb)))
    print("%-10s x=%s in-sample %+.2f%% | held-out: %s" % (name, np.array2string(x, precision=3), dmse(keys, cols, x), "  ".join(out)))
