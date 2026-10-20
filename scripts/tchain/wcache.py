# cache public mini weights as float32 .npy (x@w convention), ids from argv
import sys, numpy as np
sys.path.insert(0, "/tmp/claude-0/hfwork")
from regen import hf_weights
z = np.load("/tmp/claude-0/hfwork/mini.npz")
for i in [int(a) for a in sys.argv[1:]]:
    j = int(np.where(z["mlp_id"] == i)[0][0])
    ws = hf_weights(int(z["mlp_seed"][j]))
    np.save(f"/tmp/claude-0/cub/w_m{i}.npy", np.stack(ws).astype(np.float32))
    print("cached", i, flush=True)
