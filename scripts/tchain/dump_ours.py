# dump_ours.py <dir> <id> <out.npz>: run our estimator (fast mode) with XDUMP=1 and save per-layer arrays
import os, sys, importlib.util, numpy as np, pickle
os.environ["XDUMP"] = "1"
sys.argv = [sys.argv[0], sys.argv[1], "dumptag_unused", "m" + sys.argv[2], sys.argv[3]]
d, key, out = sys.argv[1], sys.argv[3], sys.argv[4]
src = open("/tmp/claude-0/acc/runfast_hf.py").read()
src = src.split("os.makedirs(f\"/tmp/claude-0/acc/{tag}\"")[0]   # setup only
exec(compile(src, "runfast_setup", "exec"))
z = LAB["m"]; i = int(np.where(z["mlp_id"] == int(key[1:]))[0][0])
ws = [fnp.asarray(w) for w in hf_weights(z["mlp_seed"][i])]
with flops.BudgetContext(flop_budget=2**43, quiet=True):
    p = np.asarray(est.predict(MLP(width=1024, depth=16, weights=ws, seed=0), 2**43), dtype=np.float64)
pickle.dump(dict(dumps=mod.XDUMPS, pred=p), open(out, "wb"))
print("dumped", len(mod.XDUMPS), "layers")
