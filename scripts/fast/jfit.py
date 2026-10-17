"""jfit.py <base> <tag>:<delta> ...: linear-response recalibration. For each knob run (knob moved by delta
from the base), J_k = (p_k - p0)/delta per network/neuron (last layer). Predicted e(x) = e0 - J x with
e0 = label - p0; fits x (joint and one knob at a time) by least squares over neurons of the chosen networks,
then scores the pooled noise-corrected dMSE on held-out networks (seed-parity and thirds)."""
import sys, os, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
from fm.mlps import load_split
lab = {d["seed"]: d for s in ("train", "val") for d in load_split(s, min_samples=1)}
base = sys.argv[1]; ks = [a.split(":") for a in sys.argv[2:]]
def P(tag, sd):
    f = f"/tmp/claude-0/acc/{tag}/pred_{sd}.npy"
    return np.load(f)[-1].astype(np.float64) if os.path.exists(f) else None
seeds = [sd for sd in sorted(lab) if P(base, sd) is not None and all(P(t, sd) is not None for t, _ in ks)]
E0, J, nz = {}, {}, {}
for sd in seeds:
    p0 = P(base, sd); E0[sd] = lab[sd]["means"][-1] - p0; nz[sd] = lab[sd]["noise_last"].mean()
    J[sd] = np.stack([(P(t, sd) - p0) / float(d) for t, d in ks], 1)
print("networks", len(seeds), "knobs", [t for t, _ in ks])
def fit(S, cols):
    A = sum(J[s][:, cols].T @ J[s][:, cols] for s in S); b = sum(J[s][:, cols].T @ E0[s] for s in S)
    return np.linalg.solve(A + 1e-30 * np.eye(len(cols)), b)
def dmse(S, cols, x):
    num = sum(((E0[s] - J[s][:, cols] @ x) ** 2).mean() - (E0[s] ** 2).mean() for s in S)
    den = sum((E0[s] ** 2).mean() - nz[s] for s in S)
    return 100 * num / den
splits = {"parity": [s % 2 == 0 for s in seeds], "third0": [s % 3 == 0 for s in seeds],
          "third1": [s % 3 == 1 for s in seeds], "lowhigh": [s < 1032 for s in seeds]}
allc = list(range(len(ks)))
for name, cols in [("joint", allc)] + [(ks[c][0], [c]) for c in allc]:
    x = fit(seeds, cols)
    out = []
    for sn, m in splits.items():
        A = [s for s, f in zip(seeds, m) if f]; B = [s for s, f in zip(seeds, m) if not f]
        xa, xb = fit(A, cols), fit(B, cols)
        out.append("%s %+.2f/%+.2f" % (sn, dmse(B, cols, xa), dmse(A, cols, xb)))
    print("%-10s x=%s in-sample %+.2f%% | held-out: %s" % (name, np.array2string(x, precision=3), dmse(seeds, cols, x), "  ".join(out)))
