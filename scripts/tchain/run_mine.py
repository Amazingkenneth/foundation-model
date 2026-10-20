"""run_mine.py NAME 'JSON kwargs' ID[,ID...]: patched lean chain with arbitrary kwargs -> pred/NAME/pred_m<id>.npy"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
sys.path.insert(0, "/tmp/claude-0/lean_ext")
import lean_k3_aug_ext as LX
name, kw = sys.argv[1], json.loads(sys.argv[2])
for k in ("age_rank",):
    if k in kw: kw[k] = [tuple(x) for x in kw[k]]
ids = [int(x) for x in sys.argv[3].split(",")]
lab = np.load("/tmp/claude-0/hfwork/mini.npz")
statics = LX.build_static_coefs(1024)
os.makedirs(f"/tmp/claude-0/lean_ext/pred/{name}", exist_ok=True)
for i in ids:
    out = f"/tmp/claude-0/lean_ext/pred/{name}/pred_m{i}.npy"
    if os.path.exists(out): continue
    w = np.load(f"/tmp/claude-0/lean_ext/w/w_m{i}.npy")
    gt = lab["means"][i].astype(np.float64); nz = float(lab["avg_var"][i]) / 1e9
    t0 = time.perf_counter(); eps = []
    pred = LX.lean_k3_predict(w, statics=statics, eps_out=eps, **kw).numpy()
    np.save(out, pred.astype(np.float32))
    print(f"{name} m{i}: final {float(np.mean((pred[-1]-gt[-1])**2))-nz:.4e} t={time.perf_counter()-t0:.0f}s", flush=True)
