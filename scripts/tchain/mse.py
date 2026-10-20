import sys, os, glob, numpy as np
z = np.load("/tmp/claude-0/hfwork/mini.npz")
tags = sys.argv[1:]
keys = sorted(set.intersection(*[set(os.path.basename(p)[5:-4] for p in glob.glob(f"/tmp/claude-0/acc/{t}/pred_m*.npy")) for t in tags]), key=lambda k: int(k[1:]))
print("nets:", ",".join(keys))
res = {t: [] for t in tags}
for k in keys:
    j = int(np.where(z["mlp_id"] == int(k[1:]))[0][0]); tr = z["means"][j][-1].astype(np.float64); nz = z["avg_var"][j] / 1e9
    for t in tags:
        p = np.load(f"/tmp/claude-0/acc/{t}/pred_{k}.npy")[-1]
        res[t].append(((p - tr) ** 2).mean() - nz)
b = np.array(res[tags[0]])
for t in tags:
    a = np.array(res[t])
    print("%-16s mean %.4e  ratio-to-first %.4f  geo %.4f  better %d/%d" % (t, a.mean(), a.sum() / b.sum(), np.exp(np.mean(np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)))), int((a < b).sum()), len(a)))
