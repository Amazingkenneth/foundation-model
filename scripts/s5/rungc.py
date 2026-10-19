# residual and GC time inside one predict(); argv: dir [nogc]
import os, sys, time, gc, importlib.util, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d = sys.argv[1]; nogc = len(sys.argv) > 2
sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
ws = [fnp.asarray(w) for w in make_weights(1)]
G = {"t": 0.0, "n": [0, 0, 0], "t0": 0.0}
def cb(phase, info):
    if phase == "start": G["t0"] = time.perf_counter()
    else: G["t"] += time.perf_counter() - G["t0"]; G["n"][info["generation"]] += 1
gc.callbacks.append(cb)
for rep in range(2):
    G["t"] = 0.0; G["n"] = [0, 0, 0]
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        if nogc: gc.disable()
        p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
        if nogc: gc.enable()
    print("rep %d wall %.1f residual %.3f s | GC %.3f s in %s collections (gen0/1/2)" % (rep, time.time() - t0, ctx.residual_wall_time_s, G["t"], G["n"]), flush=True)
