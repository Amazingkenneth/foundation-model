# Generate ARC current AUGMENT (augment=True) term tables in our estimator's TERM_SPECS format.
import sys, pprint, math
sys.path.insert(0, "/tmp/claude-0/arcpkg/pydeps"); sys.path.insert(0, "/tmp/claude-0/arcpkg/dl")
sys.path.insert(0, "/tmp/claude-0/ext/whest-p2-cumulant-k3/lean")
import torch; torch.set_default_dtype(torch.float64)
from mlp_kprop.kprop_harmonic import get_all_terms_iso
from mlp_kprop.partitions import vec_part_coef, check_vec_partition
import lean_k3_aug as LA
torch.set_num_threads(1)
terms_iso = get_all_terms_iso(k_max=3, d_max=4, augment=True)
dense = []
for int_part, vd in terms_iso.items():
    if len(int_part) > 2:
        continue
    for vec_part, count in vd.items():
        d = len(int_part)
        k_axes = check_vec_partition(vec_part, d)
        coef = count * vec_part_coef(vec_part, divide_fac=True)
        factors = tuple((sum(v), v) for v in vec_part)
        wick_kp = tuple((int(k_axes[ax]), int(int_part[ax])) for ax in range(d))
        dense.append((tuple(int_part), factors, float(coef), wick_kp))
print("dense terms", len(dense))
pairs = {(1, 1), (2, 1), (1, 2), (3, 1)}
for t in dense:
    pairs.update(t[3])
# keep every pair the estimator hard-codes (marg2 indices etc.)
for kp in [(5, 1), (5, 2), (6, 1), (6, 2), (3, 2)]:
    pairs.add(kp)
WP = sorted(pairs)
idx = {kp: i for i, kp in enumerate(WP)}
wp = LA.build_wick_polys()
def unified(k, p):
    if k < p:
        c1, c2 = wp[(p, k)]
        return (1.0, p - k, list(c1), list(c2))
    m = k - p + 1
    const = float(math.factorial(p))
    if m == 1:
        return (const, 0, [], [1.0])
    sign = (-1.0) ** (m - 2)
    return (const, -(m - 1), [sign * c for c in LA.HE_COEFS[m - 2]], [])
WU = [unified(k, p) for (k, p) in WP]
print("pairs", len(WP), "alpha deg", max(max(len(u[2]), len(u[3])) for u in WU) - 1,
      "sig exps", min(u[1] for u in WU), max(u[1] for u in WU))
BASE2 = {(1, 1): "c_off", (2, 1): "d21", (1, 2): "d21T", (2, 2): "wk4m", (3, 0): "d3col", (0, 3): "d3row",
         (4, 0): "g4col", (0, 4): "g4row", (3, 1): "wk431", (1, 3): "wk431T"}
BASE1 = {(3,): "d3", (4,): "g4"}
SPECS = {}
for ip, factors, coef, wkp in dense:
    d = len(ip)
    vecs = [f[1] for f in factors]
    if d == 2:
        if len(vecs) == 3:
            assert all(v == (1, 1) for v in vecs), vecs
            a, b = "c3", "ones2"
        else:
            a = BASE2[vecs[0]] if len(vecs) >= 1 else "ones2"
            b = BASE2[vecs[1]] if len(vecs) == 2 else "ones2"
        spec = (a, b, idx[wkp[0]], idx[wkp[1]], coef)
    else:
        assert len(vecs) <= 2, vecs
        a = BASE1[vecs[0]] if len(vecs) >= 1 else "ones1"
        b = BASE1[vecs[1]] if len(vecs) == 2 else "ones1"
        spec = (a, b, idx[wkp[0]], None, coef)
    SPECS.setdefault(ip, []).append(spec)
for k in sorted(SPECS):
    print(k, len(SPECS[k]))
pk2k = LA.build_pk_to_k_table(aug=True)
print("PK2K (3,1)", pk2k[(3, 1)])
with open(sys.argv[1], "w") as f:
    f.write("# ARC mlp_cumulant_propagation (MIT) current main, get_all_terms_iso(k_max=3, d_max=4, augment=True),\n")
    f.write("# converted to the estimator's TERM_SPECS format by gen_aug_specs.py (no pruning).\n")
    for name, obj in [("WICK_PAIRS_A", WP), ("WICK_UNIFIED_A", WU), ("TERM_SPECS_A", SPECS)]:
        f.write(f"{name} = " + pprint.pformat(obj, width=110) + "\n\n")
print("wrote", sys.argv[1])
