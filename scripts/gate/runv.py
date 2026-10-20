# predictions + C/B + wall + VmHWM for one candidate on given seeds (in-process 0.12.3, one thread)
import os, sys, time, importlib.util, numpy as np, resource
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
d, tag = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]
spec = importlib.util.spec_from_file_location("e", os.path.join(d, "estimator.py"))
mod = importlib.util.module_from_spec(spec); sys.modules["e"] = mod; spec.loader.exec_module(mod)
class C: seed = 0; submission_dir = d
est = mod.Estimator(); est.setup(C())
os.makedirs(f"/tmp/claude-0/acc123/{tag}", exist_ok=True)
for sd in seeds:
    ws = [fnp.asarray(w) for w in make_weights(sd)]
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        p = np.asarray(est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41), dtype=np.float64)
    dt = time.time() - t0
    np.save(f"/tmp/claude-0/acc123/{tag}/pred_{sd}.npy", p)
    hwm = [l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1]
    print(tag, sd, "C/B %.5f wall %.1f VmHWM %.2f GB" % (ctx.flops_used / 2**41, dt, int(hwm) / 1e6), flush=True)
