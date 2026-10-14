"""Deterministic suite-shape MLPs and Monte Carlo dataset access."""

from __future__ import annotations

import glob
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def make_weights(seed: int, width: int = 1024, depth: int = 16) -> list[np.ndarray]:
    """He-normal weights N(0, 2/width), float32, stored (n_in, n_out) like whestbench:
    the forward pass is h <- relu(h @ W)."""
    rng = np.random.default_rng([seed, 104729])
    scale = np.float32(np.sqrt(2.0 / width))
    return [(rng.standard_normal((width, width), dtype=np.float32) * scale) for _ in range(depth)]


def load_split(split: str, data_dir: str = DATA_DIR, min_samples: int = 1) -> list[dict]:
    """All finished MLPs of a split: seed, MC means (depth, width), samples, final-layer
    MC noise variance per neuron (var / n)."""
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, split, "*.npz"))):
        if p.endswith(".tmp.npz"):
            continue
        z = np.load(p)
        n = int(z["n"])
        if n < min_samples:
            continue
        means = z["sums"] / n
        var_last = z["sq_last"] / n - means[-1] ** 2
        out.append(dict(seed=int(z["seed"]), means=means.astype(np.float64), n=n,
                        noise_last=var_last / n, path=p))
    return out
