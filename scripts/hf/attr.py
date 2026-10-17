# per-layer error attribution on public mini networks: e_l = pred - truth; propagated part J_l e_{l-1},
# J_l = diag(Phi_l) W_l^T (h_l = relu(h_{l-1} W_l)); created = e_l - J_l e_{l-1}
import os, sys, numpy as np
sys.path.insert(0, "/home/user/foundation-model"); sys.path.insert(0, "/tmp/claude-0/hfwork"); sys.path.insert(0, "/tmp/claude-0/acc")
os.environ["XSAVEPHI"] = "1"
import importlib.util, flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from regen import hf_weights
d = "/tmp/claude-0/sc/xs"
sspec = importlib.util.spec_from_file_location("strassen2", os.path.join(d, "strassen2.py"))
s2 = importlib.util.module_from_spec(sspec); sys.modules["strassen2"] = s2; sspec.loader.exec_module(s2)
class _LT:
    def __init__(self, X, lev, w, tag="T", mb=None): self.X = X; self.lev = 0; self.c = []
def _mm(X, R, lev, out, ksum=False, mb=None):
    Xa = X.X if isinstance(X, _LT) else X
    if ksum: fnp.copyto(out, fnp.sum(fnp.matmul(Xa, R), axis=0, keepdims=True))
    else: fnp.matmul(Xa, R, out=out)
    return out
s2.LeftTree = _LT; s2.mm = _mm
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
z = np.load("/tmp/claude-0/hfwork/mini.npz")
tot = {"e": np.zeros(16), "prop": np.zeros(16), "new": np.zeros(16), "cross": np.zeros(16)}
ids = [int(x) for x in sys.argv[1].split(",")]
for mid in ids:
    est = mod.Estimator(); est.setup(C()); est._phis = []
    i = int(np.where(z["mlp_id"] == mid)[0][0])
    ws = hf_weights(z["mlp_seed"][i])
    with flops.BudgetContext(flop_budget=2**43, quiet=True):
        p = np.asarray(est.predict(MLP(width=1024, depth=16, weights=[fnp.asarray(w) for w in ws], seed=0), 2**43), dtype=np.float64)
    t = z["means"][i].astype(np.float64); e = p - t
    phis = est._phis
    for l in range(16):
        tot["e"][l] += (e[l] ** 2).mean()
        if l == 0: continue
        pr = phis[l] * (e[l - 1] @ ws[l].astype(np.float64))          # linear propagation of the previous error
        nw = e[l] - pr
        tot["prop"][l] += (pr ** 2).mean(); tot["new"][l] += (nw ** 2).mean(); tot["cross"][l] += 2 * (pr * nw).mean()
    print("done", mid, len(phis), flush=True)
k = len(ids)
for l in range(16):
    print("layer %2d  MSE %.2e  propagated %.2e  created %.2e  2<prop,new> %+.2e" % (l, tot["e"][l] / k, tot["prop"][l] / k, tot["new"][l] / k, tot["cross"][l] / k))
