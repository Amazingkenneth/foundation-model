"""Regenerate public WhestBench v2-phase2 weights from mlp_seed, exactly like `whest dataset bake`
(whestbench.dataset: weight_ss = derive_seed_streams(seed)[0]; sample_mlp(width, depth, default_rng(weight_ss)))."""
import numpy as np
import flopscope.numpy as fnp
from whestbench.seeds import derive_seed_streams
from whestbench.generation import sample_mlp


def hf_weights(mlp_seed, width=1024, depth=16):
    wss, _, _ = derive_seed_streams(int(mlp_seed))
    mlp = sample_mlp(width, depth, fnp.random.default_rng(wss))
    return [np.asarray(w, dtype=np.float32) for w in mlp.weights]
