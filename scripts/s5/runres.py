# in-process residual (non-flopscope) wall per predict() for candidate dirs, seed 1
import os, sys, time, importlib.util, numpy as np
sys.path.insert(0, "/home/user/foundation-model")
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
from fm.mlps import make_weights
ws = None
for i, d in enumerate(sys.argv[1:]):
    sys.path.insert(0, d)
    for m in [m for m in list(sys.modules) if m.startswith("strassen")]: del sys.modules[m]
    spec = importlib.util.spec_from_file_location(f"e{i}", os.path.join(d, "estimator.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[f"e{i}"] = mod; spec.loader.exec_module(mod)
    class C: seed = 0; submission_dir = d
    est = mod.Estimator(); est.setup(C())
    ws = ws or [fnp.asarray(w) for w in make_weights(1)]
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        p = est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**41)
    print("%-55s wall %.1f residual %.3f s C/B %.5f" % (d[-40:], time.time() - t0, ctx.residual_wall_time_s, ctx.flops_used / 2**41), flush=True)
    sys.path.remove(d)
