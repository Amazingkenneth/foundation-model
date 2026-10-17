"""eval_hf.py <base_tag> <tag>...: paired final-layer MSE on PUBLIC WhestBench networks (1e9-sample truth) that both
tags have. pooled dMSE = sum(d)/sum(base) with bootstrap se; halves by id parity; absolute raw MSE of the base."""
import sys, os, glob, numpy as np
LAB = {"m": np.load("/tmp/claude-0/hfwork/mini.npz")}
if os.path.exists("/tmp/claude-0/hfwork/full.npz"):
    LAB["f"] = np.load("/tmp/claude-0/hfwork/full.npz")
def truth(key):
    z = LAB[key[0]]; i = int(np.where(z["mlp_id"] == int(key[1:]))[0][0])
    return z["means"][i][-1].astype(np.float64), z["avg_var"][i] / 1e9
base = sys.argv[1]
keys = sorted(os.path.basename(p)[5:-4] for p in glob.glob(f"/tmp/claude-0/acc/{base}/pred_*.npy"))
keys = [k for k in keys if k[0] in "mf"]
rng = np.random.default_rng(0)
for tag in sys.argv[2:] or [base]:
    rows = []
    for k in keys:
        fq = f"/tmp/claude-0/acc/{tag}/pred_{k}.npy"
        if not os.path.exists(fq): continue
        p = np.load(f"/tmp/claude-0/acc/{base}/pred_{k}.npy")[-1]; q = np.load(fq)[-1]
        t, nz = truth(k)
        a = ((p - t) ** 2).mean() - nz; b = ((q - t) ** 2).mean() - nz
        rows.append((k, a, b - a, np.sqrt(((q - p) ** 2).mean())))
    if not rows: print(tag, "no common networks"); continue
    A = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows]); mv = np.array([r[3] for r in rows])
    pooled = D.sum() / A.sum()
    bs = [D[ix].sum() / A[ix].sum() for ix in (rng.integers(0, len(D), len(D)) for _ in range(2000))]
    ev = np.array([int(r[0][1:]) % 2 == 0 for r in rows])
    h = lambda m: 100 * D[m].sum() / A[m].sum() if m.any() else float("nan")
    print("%-14s n=%3d pooled %+6.2f%% (se %.2f)  better %d/%d | even %+.2f%% odd %+.2f%% | move %.1e | base raw %.3e" % (
        tag, len(rows), 100 * pooled, 100 * np.std(bs), int((D < 0).sum()), len(rows), h(ev), h(~ev), mv.mean(), A.mean()))
