import sys, pprint
sys.path.insert(0, "/tmp/claude-0/arcpkg/pydeps"); sys.path.insert(0, "/tmp/claude-0/arcpkg/dl")
sys.path.insert(0, "/tmp/claude-0/ext/whest-p2-cumulant-k3/lean")
import torch; torch.set_default_dtype(torch.float64)
from mlp_kprop.kprop_harmonic import get_all_terms_iso
from mlp_kprop.partitions import vec_part_coef, check_vec_partition, vector_partitions
from mlp_kprop.cumulants import _pK_to_K_coef
import lean_k3_aug as LA
torch.set_num_threads(1)
t = get_all_terms_iso(k_max=3, d_max=4, augment=True)
T211 = []
for vec_part, count in t[(2, 1, 1)].items():
    if set(vec_part) == {(1, 1, 1)}:
        continue
    k_axes = check_vec_partition(vec_part, 3)
    coef = count * vec_part_coef(vec_part, divide_fac=True)
    factors = tuple((sum(v), v) for v in vec_part)
    wick_kp = tuple((int(k_axes[ax]), int((2, 1, 1)[ax])) for ax in range(3))
    T211.append(((2, 1, 1), factors, float(coef), wick_kp))
for r in T211: print(r)
P211 = []
for vpart in vector_partitions((2, 1, 1)):
    if any(tuple(v) == (1, 1, 1) for v in vpart):
        continue
    coef = vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart)
    if coef == 0: continue
    P211.append((tuple(vpart), float(coef)))
print("PK2K211:"); [print(x) for x in P211]
with open(sys.argv[1], "w") as f:
    f.write("# ARC current main, augment=True: capped (2,1,1) production terms (the (1,1,1) kappa3 term excluded)\n")
    f.write("TERM_TABLE_211_A = " + pprint.pformat(T211, width=110) + "\n\n")
    f.write("PK2K211_A = " + pprint.pformat(P211, width=110) + "\n")
