"""eval75.py <base_tag> <tag>...: paired label MSE over every network both tags have (fast-mode runs on all 75
labeled MLPs). pooled dMSE = sum(d)/sum(base) (leaderboard weighting) with bootstrap se; split into the old
tuning-12 (hard six + held-out six), the other networks, and seed-parity halves (a consistency check)."""
import sys, os, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
from fm.mlps import load_split
lab = {d["seed"]: d for s in ("train", "val") for d in load_split(s, min_samples=1)}
S1 = [1015, 1002, 5010, 1005, 1008, 1010, 5008, 1007, 1012, 1006, 5002, 1001]
base = sys.argv[1]
rng = np.random.default_rng(0)
def load(tag, sd):
    f = f"/tmp/claude-0/acc/{tag}/pred_{sd}.npy"
    return np.load(f)[-1] if os.path.exists(f) else None
for tag in sys.argv[2:]:
    rows = []
    for sd in sorted(lab):
        p, q = load(base, sd), load(tag, sd)
        if p is None or q is None: continue
        t = lab[sd]["means"][-1]; nz = lab[sd]["noise_last"].mean()
        a, b = ((p - t) ** 2).mean() - nz, ((q - t) ** 2).mean() - nz
        rows.append((sd, a, b - a, np.sqrt(((q - p) ** 2).mean())))
    if not rows: continue
    r = np.array(rows); a, d = r[:, 1], r[:, 2]
    boots = [d[i].sum() / a[i].sum() for i in (rng.integers(0, len(d), len(d)) for _ in range(4000))]
    def part(m):
        return (100 * d[m].sum() / a[m].sum()) if m.any() else float("nan")
    m1 = np.isin(r[:, 0], S1); ev = (r[:, 0] % 2 == 0)
    print("%-12s n=%2d pooled %+.2f%% (se %.2f)  better %2d/%d | tune12 %+.2f%%  rest %+.2f%% | even %+.2f%%  odd %+.2f%% | move %.1e" % (
        tag, len(r), 100 * d.sum() / a.sum(), 100 * np.std(boots), int((d < 0).sum()), len(r),
        part(m1), part(~m1), part(ev), part(~ev), np.sqrt((r[:, 3] ** 2).mean())))
