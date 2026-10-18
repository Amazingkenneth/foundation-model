"""gain.py <tag> <ids>: per-layer propagation gain to the final layer, g_l = E||J_{L<-l} u||^2 / E||u||^2 for white u,
with J the corrector's propagation s_l = Phi_l * (W_l^T s_{l-1}); also the share of the final error per layer."""
import sys, numpy as np
sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/sc/corr")
from regen import hf_weights
from prep import ids
tag, spec = sys.argv[1], sys.argv[2]
G, S = [], []
for k in ids(spec):
    z = np.load(f"/tmp/claude-0/sc/corr/data_{tag}/{k}.npz"); Ws = hf_weights(int(z["seed"]))
    Phi = z["Phi"].astype(np.float64); C = z["C"].astype(np.float64); E = z["E"].astype(np.float64); L, n = Phi.shape
    U = np.random.default_rng(0).standard_normal((n, 64))
    g = []
    for l in range(L):
        s = U.copy()
        for m in range(l + 1, L):
            s = Phi[m][:, None] * (Ws[m].T.astype(np.float64) @ s)
        g.append((s ** 2).sum() / (U ** 2).sum())
    # propagated share of each layer's created error in the final error
    sh = []
    for l in range(L):
        s = C[l].copy()
        for m in range(l + 1, L):
            s = Phi[m] * (Ws[m].T.astype(np.float64) @ s)
        sh.append(np.mean(s ** 2))
    G.append(g); S.append(np.array(sh) / np.mean(E[-1] ** 2))
G = np.array(G).mean(0); S = np.array(S).mean(0)
print("gain g_l   :", " ".join("%.3f" % x for x in G))
print("ratio g_l/g_{l+1}:", " ".join("%.2f" % (G[l] / G[l + 1]) for l in range(len(G) - 1)))
print("share of final MSE (no cross terms):", " ".join("%.3f" % x for x in S), "sum %.3f" % S.sum())
