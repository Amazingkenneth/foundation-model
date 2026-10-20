"""perlayer.py ID src...: per-layer MSE (raw) of predictions vs 1e9 labels; src = acc tag or lean:<name>"""
import sys, os, numpy as np
z = np.load("/tmp/claude-0/hfwork/mini.npz")
i = int(sys.argv[1]); j = int(np.where(z["mlp_id"] == i)[0][0]); gt = z["means"][j].astype(np.float64)
rows = []
for s in sys.argv[2:]:
    p = (f"/tmp/claude-0/lean_ext/pred/{s[5:]}/pred_m{i}.npy" if s.startswith("lean:")
         else f"/tmp/claude-0/acc/{s}/pred_m{i}.npy")
    if not os.path.exists(p):
        print("missing", s); continue
    q = np.load(p).astype(np.float64)
    rows.append((s, [((q[l] - gt[l]) ** 2).mean() for l in range(16)]))
print("layer " + " ".join("%7d" % l for l in (1, 3, 5, 7, 9, 11, 13, 14, 15)))
for s, m in rows:
    print("%-14s" % s[:14] + " ".join("%7.2e" % m[l] for l in (1, 3, 5, 7, 9, 11, 13, 14, 15)))
