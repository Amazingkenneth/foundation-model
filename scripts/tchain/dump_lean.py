# dump_lean.py 'JSON kwargs' id out.pkl: patched lean with debug_state
import os, sys, json, pickle
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/lean_ext")
import lean_k3_aug_ext as LX
kw = json.loads(sys.argv[1]); i = int(sys.argv[2])
w = np.load(f"/tmp/claude-0/lean_ext/w/w_m{i}.npy")
dbg = []
pred = LX.lean_k3_predict(w, statics=LX.build_static_coefs(1024), debug_state=dbg, **kw).numpy()
L = int(os.environ.get("DUMP_L", "2"))
out = [{k: (None if v is None else v.numpy()) for k, v in d.items()} for d in dbg[:L + 1]]
pickle.dump(dict(dumps=out, pred=pred), open(sys.argv[3], "wb"))
print("dumped", len(out))
