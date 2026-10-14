"""Monte Carlo ground truth for training / validation MLPs (Algorithm 2's mu_MC^(l)).

Each MLP is a Phase 2 suite-shape network (width 1024, depth 16, He-normal weights
N(0, 2/n), float32, no biases, x -> relu(x @ W)), regenerated deterministically from its
integer seed, so the dataset stores only seeds and Monte Carlo statistics.

For every MLP the script accumulates, in float64, the per-layer sums of the post-ReLU
activations over N(0, I) inputs and the final-layer sum of squares (for the noise
estimate). Statistics are checkpointed to ``<out>/<split>/<seed>.npz`` after every
block, so an interrupted run resumes and a later run with a larger ``--samples`` tops
an existing file up instead of starting over.

    python scripts/gen_data.py --split train --seeds 1000-1127 --samples 500000
    python scripts/gen_data.py --split val   --seeds 5000-5023 --samples 1500000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fm.mlps import make_weights  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def mc_one(seed: int, n_samples: int, path: str, width: int, depth: int,
           chunk: int, block: int) -> None:
    done = 0
    sums = np.zeros((depth, width), dtype=np.float64)
    sq_last = np.zeros(width, dtype=np.float64)
    if os.path.exists(path):
        z = np.load(path)
        done = int(z["n"])
        sums = z["sums"].astype(np.float64)
        sq_last = z["sq_last"].astype(np.float64)
        if done >= n_samples:
            return
    ws = make_weights(seed, width, depth)
    # The sampling stream is keyed by (seed, samples already drawn), so a top-up run
    # never reuses inputs.
    rng = np.random.default_rng([seed, 7919, done])
    t0 = time.time()
    while done < n_samples:
        nb = min(block, n_samples - done)
        bsum = np.zeros((depth, width), dtype=np.float64)
        bsq = np.zeros(width, dtype=np.float64)
        for start in range(0, nb, chunk):
            m = min(chunk, nb - start)
            x = rng.standard_normal((m, width), dtype=np.float32)
            for li, w in enumerate(ws):
                x = x @ w
                np.maximum(x, 0.0, out=x)
                bsum[li] += x.sum(axis=0, dtype=np.float64)
            bsq += np.einsum("ij,ij->j", x, x, dtype=np.float64)
        sums += bsum
        sq_last += bsq
        done += nb
        tmp = path + ".tmp.npz"
        np.savez(tmp, n=np.int64(done), sums=sums, sq_last=sq_last, seed=np.int64(seed),
                 width=np.int64(width), depth=np.int64(depth))
        os.replace(tmp, path)
    dt = time.time() - t0
    means = sums / done
    var_last = sq_last / done - means[-1] ** 2
    print(f"seed {seed}: n={done} {dt:.0f}s  mean(var_L)={var_last.mean():.4f}  "
          f"MC noise MSE_L~{var_last.mean() / done:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 1000-1127 or 1,2,3")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--block", type=int, default=131072)
    args = ap.parse_args()
    d = os.path.join(args.out, args.split)
    os.makedirs(d, exist_ok=True)
    for s in parse_seeds(args.seeds):
        mc_one(s, args.samples, os.path.join(d, f"{s}.npz"), args.width, args.depth,
               args.chunk, args.block)


if __name__ == "__main__":
    main()
