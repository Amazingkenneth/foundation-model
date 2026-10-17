# Local gate on the public WhestBench v2-phase2 networks (1e9-sample truth)

* `hf_labels.py <split> <shards> <n_shards> <out.npz>`: reads ONLY `mlp_id`, `mlp_seed`, `all_layer_means`,
  `avg_variance` from `hf://aicrowd/arc-whestbench-public-2026@v2-phase2` (pyarrow column projection over HTTP range
  requests; needs huggingface.co reachable). mini = 7 shards (100 MLPs), full = shards 0-62 (1,000 MLPs).
* `regen.py`: `hf_weights(mlp_seed)` regenerates the weights exactly like `whest dataset bake`
  (`derive_seed_streams(seed)[0]` -> `sample_mlp`). Checked against the truth with a 2e4-sample forward pass.
* `runfast_hf.py <dir> <tag> m0,m1,...|f0,...`: fast mode (plain matmul for Strassen) on public networks;
  `gohf.sh` runs it on 4 processes.
* `eval_hf.py <base> <tag>...`: paired final-layer MSE against the 1e9-sample truth; `jfit_hf.py`: linear-response
  gain fit with held-out halves.
The scripts expect the label files at /tmp/claude-0/hfwork/{mini,full}.npz and write predictions under /tmp/claude-0/acc.
