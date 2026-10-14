"""Train the learned ReLU operator F_theta (Algorithm 2).

Stage 1 trains the input corrections (f_self, f_msg1, f_msg2, f_in, f_pair); stage 2 adds
the output corrections (f_out, f_cov, f_last, output layers initialized to zero) and trains
everything. Each step runs Algorithm 1 on one training network and backpropagates
L = sum_l lambda_l (1/n) ||mu_hat^(l) - mu_MC^(l)||^2 (lambda_L = 1, lambda_{l<L} = 0.1)
through all layers. The operator acts only from layer ``--active-from`` on (default: 1-based
layers 9..15 + the last layer, the paper's v5b), so the chain state entering that layer is
independent of theta: it is computed once per network and cached on disk, and gradients
flow through every layer after it. The epoch with the lowest validation MSE is kept.

    python scripts/train.py --stage 1 --epochs 8 --out runs/s1
    python scripts/train.py --stage 2 --epochs 8 --init runs/s1/best.pt --out runs/s2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from fm.mlps import DATA_DIR, load_split  # noqa: E402
from fm.model import (OperatorParams, fm_core, make_chain, make_operator,  # noqa: E402
                      torch_weights)


def cached_state(chain, seed, start, cache_dir):
    p = os.path.join(cache_dir, f"{seed}_{start}.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False)
    ws = torch_weights(seed)
    with torch.no_grad():
        _, st = chain.run(ws, hooks=None, stop=start)
    tmp = p + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, p)
    return st


class Capture(fm_core.Operator):
    """Operator with zero corrections that records the raw features (for standardization)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []

    def std(self, cols):
        X = torch.stack(cols, 1)
        if len(cols) == 14:
            self.rows.append(X.detach())
        return torch.clamp((X - self.xmean[None, :len(cols)]) / self.xstd[None, :len(cols)], -8.0, 8.0)


def run_net(chain, B, op, d, start, cache_dir):
    st = cached_state(chain, d["seed"], start, cache_dir)
    ws = torch_weights(d["seed"])
    rows, _ = chain.run(ws, hooks=op, state=st, start=start)
    return torch.stack(rows, 0)          # (L - start, n)


def loss_fn(pred, target, L, start):
    lam = torch.full((L - start,), 0.1)
    lam[-1] = 1.0
    err = ((pred - target) ** 2).mean(dim=1)
    return (lam * err).sum(), err.detach()


def evaluate(chain, B, params, data, start, cache_dir, active):
    op = None if params is None else make_operator(B, params, active)
    out = []
    with torch.no_grad():
        for d in data:
            pred = run_net(chain, B, op, d, start, cache_dir)
            tgt = torch.from_numpy(d["means"][start:]).float()
            e = ((pred.double() - tgt.double()) ** 2).mean(dim=1)
            out.append((e[-1].item(), e[-1].item() - float(d["noise_last"].mean())))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--active-from", type=int, default=8, help="0-based first corrected layer")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "s1"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    L, n = 16, 1024
    start = args.active_from
    active = list(range(start, L - 1))
    train = load_split("train")
    val = load_split("val")
    if args.train_limit:
        train = train[:args.train_limit]
    if args.val_limit:
        val = val[:args.val_limit]
    print(f"train {len(train)} val {len(val)}  active layers (0-based) {active} + last", flush=True)
    B, chain = make_chain(n, L)

    if args.init:
        ck = torch.load(args.init, weights_only=False)
        params = OperatorParams(nets=ck["nets"])
        params.load_state_dict(ck["state"])
    else:
        params = OperatorParams(nets=fm_core.STAGE1, seed=args.seed)
        # fixed feature standardization from the uncorrected chain on training networks
        cap_p = OperatorParams(nets=fm_core.STAGE1, seed=args.seed)
        with torch.no_grad():
            for nm in cap_p.nets:
                cap_p.p[f"{nm}__W3"].zero_()
            cap = Capture(B, cap_p.param_dict(), cap_p.xmean, cap_p.xstd, active, cap_p.nets)
            for d in train[:4]:
                run_net(chain, B, cap, d, start, cache_dir)
            X = torch.cat(cap.rows, 0)
            params.xmean.copy_(X.mean(0))
            params.xstd.copy_(X.std(0).clamp_min(1e-6))
        print("feature mean", params.xmean.numpy().round(4).tolist(), flush=True)
        print("feature std ", params.xstd.numpy().round(4).tolist(), flush=True)
    if args.stage == 2:
        params.add_stage2(seed=args.seed + 1)
    train_names = fm_core.STAGE1 if args.stage == 1 else tuple(params.nets)
    trainable = [p for k, p in params.p.items() if k.split("__")[0] in train_names]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    # precompute the cached states (one uncorrected run of layers < start per network)
    t0 = time.time()
    for d in train + val:
        cached_state(chain, d["seed"], start, cache_dir)
    print(f"cached states ready ({time.time() - t0:.0f}s)", flush=True)

    base_val = evaluate(chain, B, None, val, start, cache_dir, active)
    cur_val = evaluate(chain, B, params, val, start, cache_dir, active)
    print(f"val final-layer MSE: analytic {base_val[0]:.4e} (noise-corr {base_val[1]:.4e}); "
          f"init {cur_val[0]:.4e} (noise-corr {cur_val[1]:.4e})", flush=True)
    best = cur_val[0]
    hist = [dict(epoch=0, val=cur_val[0], val_corr=cur_val[1], base=base_val[0])]
    torch.save(dict(nets=params.nets, state=params.state_dict()), os.path.join(args.out, "best.pt"))
    params.export(os.path.join(args.out, "fm_weights.npz"), active)
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(train))
        tl, tf = [], []
        for i in order:
            d = train[i]
            op = make_operator(B, params, active)
            pred = run_net(chain, B, op, d, start, cache_dir)
            tgt = torch.from_numpy(d["means"][start:]).float()
            loss, err = loss_fn(pred, tgt, L, start)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tl.append(loss.item())
            tf.append(err[-1].item() - float(d["noise_last"].mean()))
        v = evaluate(chain, B, params, val, start, cache_dir, active)
        hist.append(dict(epoch=ep, train_loss=float(np.mean(tl)), train_corr=float(np.mean(tf)),
                         val=v[0], val_corr=v[1], base=base_val[0], time=time.time() - t0))
        tag = ""
        if v[0] < best:
            best = v[0]
            tag = "  * best"
            torch.save(dict(nets=params.nets, state=params.state_dict()), os.path.join(args.out, "best.pt"))
            params.export(os.path.join(args.out, "fm_weights.npz"), active)
        torch.save(dict(nets=params.nets, state=params.state_dict()), os.path.join(args.out, "last.pt"))
        print(f"epoch {ep}: train loss {np.mean(tl):.4e} (final noise-corr {np.mean(tf):.4e})  "
              f"val {v[0]:.4e} (noise-corr {v[1]:.4e}, analytic {base_val[1]:.4e})  "
              f"{time.time() - t0:.0f}s{tag}", flush=True)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()
