# Read ONLY the label columns (mlp_id, mlp_seed, all_layer_means, avg_variance) of the public WhestBench parquet
# shards over HTTP range requests (pyarrow column projection); never downloads the weights column.
import sys, json, io, numpy as np, pyarrow.parquet as pq, fsspec
split = sys.argv[1]; shards = [int(s) for s in sys.argv[2].split(",")]; nsh = int(sys.argv[3])
out = sys.argv[4]
base = "https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/data"
fs = fsspec.filesystem("https", block_size=1 << 20, client_kwargs={"trust_env": True})
rows = {}
for s in shards:
    url = f"{base}/{split}-{s:05d}-of-{nsh:05d}.parquet"
    try:
        f = fs.open(url, "rb")
    except FileNotFoundError:
        print("missing", url, flush=True); continue
    with f:
        pf = pq.ParquetFile(f)
        if s == shards[0]:
            print("schema:", pf.schema_arrow, "row groups", pf.num_row_groups, "rows", pf.metadata.num_rows, flush=True)
        t = pf.read(columns=["mlp_id", "mlp_seed", "all_layer_means", "avg_variance"])
    d = t.to_pydict()
    for i in range(len(d["mlp_id"])):
        m = np.asarray(d["all_layer_means"][i], dtype=np.float32)
        rows[int(d["mlp_id"][i])] = dict(seed=int(d["mlp_seed"][i]), means=m, avg_var=float(d["avg_variance"][i]))
    print(split, s, len(d["mlp_id"]), "rows", flush=True)
ids = sorted(rows)
np.savez(out, mlp_id=np.array(ids), mlp_seed=np.array([rows[i]["seed"] for i in ids], dtype=np.int64),
         means=np.stack([np.stack(rows[i]["means"]) if rows[i]["means"].dtype == object else rows[i]["means"] for i in ids]),
         avg_var=np.array([rows[i]["avg_var"] for i in ids]))
print("saved", out, len(ids))
