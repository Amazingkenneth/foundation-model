"""runfast_hf_ldump.py (saves per-layer chain statistics to ldump_<key>.npz)
runfast_hf.py <dir> <tag> <ids>: fast mode (plain matmul instead of Strassen) on PUBLIC WhestBench v2-phase2
networks, ids like m12 (mini mlp_id 12) or f345 (full mlp_id 345); weights regenerated from mlp_seed."""
import os, sys, importlib.util, numpy as np, time
sys.path.insert(0, "/home/user/foundation-model"); sys.path.insert(0, "/tmp/claude-0/hfwork")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from regen import hf_weights
d, tag = sys.argv[1], sys.argv[2]; ids = sys.argv[3].split(",")
LAB = {"m": np.load("/tmp/claude-0/hfwork/mini.npz")}
if os.path.exists("/tmp/claude-0/hfwork/full.npz"):
    LAB["f"] = np.load("/tmp/claude-0/hfwork/full.npz")
SN = "strassen4" if os.path.exists(os.path.join(d, "strassen4.py")) else "strassen2"
sspec = importlib.util.spec_from_file_location(SN, os.path.join(d, SN + ".py"))
s2 = importlib.util.module_from_spec(sspec); sys.modules[SN] = s2; sspec.loader.exec_module(s2)
class _LT:
    def __init__(self, X, lev, w, tag="T", mb=None):
        self.X = X; self.lev = 0; self.c = []
def _mm(X, R, lev, out, ksum=False, mb=None):
    Xa = X.X if isinstance(X, _LT) else X
    if ksum:
        fnp.copyto(out, fnp.sum(fnp.matmul(Xa, R), axis=0, keepdims=True))
    else:
        fnp.matmul(Xa, R, out=out)
    return out
s2.LeftTree = _LT; s2.mm = _mm
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
assert mod.LeftTree is _LT and mod._smm is _mm
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
os.makedirs(f"/tmp/claude-0/acc/{tag}", exist_ok=True)
for key in ids:
    out = f"/tmp/claude-0/acc/{tag}/pred_{key}.npy"
    if os.path.exists(out): continue
    z = LAB[key[0]]; i = int(np.where(z["mlp_id"] == int(key[1:]))[0][0])
    t0 = time.time()
    ws = [fnp.asarray(w) for w in hf_weights(z["mlp_seed"][i])]
    mod._LDUMP.clear()
    with flops.BudgetContext(flop_budget=2**43, quiet=True) as ctx:
        p = np.asarray(est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**43), dtype=np.float64)
    _L = mod._LDUMP; _ks = sorted(set(k for d in _L for k in d))
    np.savez(out.replace('pred_', 'ldump_').replace('.npy', '.npz'), **{k: np.stack([d.get(k, np.zeros(1024, np.float32)) for d in _L]) for k in _ks})
    np.save(out, p); print(tag, key, "%.0fs" % (time.time() - t0), flush=True)
