# A Learned Foundation Model for the ReLU Step

Implementation of *"A Learned Foundation Model for the ReLU Step"* (Option B, width 1024,
depth 16) for the ARC White-Box Estimation Challenge 2026: a factorized K=3 cumulant chain
whose ReLU step is a learned operator

    F_theta(u~, v~) = F_an((u~, v~) + Delta_in) + Delta_out

with ten small MLPs (two hidden layers of width 64, GELU, 114,427 parameters) shared by all
neurons, pairs, layers and networks.

## Layout

| path | what |
|---|---|
| `submission/` | **the shipped folder** (pure stdlib + flopscope): `estimator.py` (entry point), `fm_core.py` (chain + operator, backend-agnostic), `backend_fnp.py` (flopscope backend: pooled buffers, Strassen-Winograd families), `fm_tables.py` (Wick/term coefficient tables), `fm_weights.npz` (trained theta) |
| `fm/` | training side: `backend_torch.py` (same chain on torch, autograd), `model.py` (parameters, init, export), `mlps.py` (suite-shape MLPs, MC data access) |
| `scripts/gen_data.py` | Monte Carlo ground truth mu_MC^(l) for training/validation MLPs (resumable, top-up) |
| `scripts/train.py` | Algorithm 2 (stage 1: input corrections; stage 2: + output corrections) |
| `scripts/eval_fnp.py` | the submission under flopscope on local ground truth: C/B, residual time, MSE |
| `scripts/parity.py` | torch chain vs flopscope chain (with and without the operator) |
| `scripts/smoke.py` | allowed-imports check, non-suite shapes, weights load |
| `scripts/package.sh`, `scripts/submit.sh`, `scripts/install-hooks.sh` | packaging / submission automation |
| `.github/workflows/submit.yml` | CI: validate + package + (optionally) submit on every push that touches `submission/` |

## The algorithm (what is where)

**Linear step** (`Chain.layer`, first half): exact third-order transport. The third cumulant
is a sum of hub sources (one born per ReLU layer) with dense legs (A, P) for young sources,
a shared transported rank-384 basis for sources older than 4 transports and a nested
rank-224 sub-basis beyond 7; thin residual / feedback legs ride along. The (3,) and (2,1)
slices kappa_iii = D3 and kappa_iic = D21 are the interface to the ReLU step
(`Chain.dslices`). The kappa4 diagonal is regenerated each layer from Sigma~ and the previous
kappa_iiii (`dG`, adaptive lambda), the (3,1) slice as kappa_iiij = lambda C_off.

**ReLU step**: `F_an` is the analytic Hermite/Wick update (`Chain.wick`, `term1`, `term2`,
the pK -> K conversion and the birth of the new source). The learned corrections are
`fm_core.Operator`, in the order of Algorithm 1:

1. `f_self` (12 -> 2): kappa~_iiii += f s^4, kappa~_iii += 0.1 f s^3; completes x_i (14).
2. `f_msg1` (14 -> 16) and 3. `f_msg2` (174 -> 16): two rounds of messages over ten fixed
   edge weights E_c in {rho, rho^2, rho^3, |rho|, t, t^T, t^2, (t^T)^2, rho t, rho t^T}
   (rho_ij = Sigma~_ij/(s_i s_j), t_ij = kappa~_iij/(s_i^2 s_j)), each normalized by its mean
   absolute value (and by n), giving x_msg in R^334.
4. `f_in` (334 -> 2): kappa~_iiii += f s^4, kappa~_iii += f s^3.
5. `f_pair` (2 x (14 -> 36)): kappa~_iiij += sum_c B_c(i,j) <l_c(x_i), r_c(x_j)> s_i s_j^3 over
   nine pair channels — n^2 pair values from n network evaluations.
6. `f_out` (334 -> 6): mu += 1e-3 f s, Sigma_ii *= 1 + 1e-3 f, kappa_iiii += 1e-2 f sigma^4,
   and the new third-cumulant term: its diagonal += 1e-2 f sigma^3, its two factor
   coefficients (the Y-leg scaling w2 and the M-leg A-coefficient e of the newborn source)
   *= 1 + 1e-2 f.
7. `f_cov` (2 x (14 -> 36)): Sigma_ij += 1e-3 (f(i,j) + f(j,i))/2 sigma_i sigma_j.
8. `f_last` (14 -> 1, last layer): mu^(L) = F_an(u~)_mu + 1e-3 f s.

Steps 1-7 run at the corrected layers (default: 1-based layers 9..15, the paper's "v5b"),
`f_self` and `f_last` at the last layer. The 14 features x_i: standardized mean, skewness,
kurtosis, depth, eight carried third/fourth-cumulant summaries (inherited kurtosis
sum_a W_ai^4 kappa_aaaa / s^4, the diagonal-transport estimate of kappa~_iiii, inherited
skewness, the regenerated part of kappa~_iiii, input-weighted kurtosis and skewness of the
previous layer, the diagonal share of the variance, 100 lambda), and the skewness and
kurtosis after `f_self`.

**One code path.** `fm_core` is written against a backend interface; the submission runs
it on flopscope (`backend_fnp`), training on torch (`fm/backend_torch.py`). `scripts/parity.py`
checks the two agree.

## Reproduce

```bash
pip install -r requirements-train.txt
# 1. ground truth (resumable; --samples tops up existing files)
python scripts/gen_data.py --split train --seeds 1000-1599 --samples 1000000
python scripts/gen_data.py --split val   --seeds 5000-5049 --samples 4000000
# 2. train (Algorithm 2)
python scripts/train.py --stage 1 --epochs 8 --out runs/s1
python scripts/train.py --stage 2 --epochs 8 --init runs/s1/best.pt --out runs/s2
cp runs/s2/fm_weights.npz submission/
# 3. check
python scripts/parity.py
python scripts/eval_fnp.py --split val
# 4. package / submit
scripts/package.sh
AICROWD_API_KEY=... scripts/submit.sh
```

## Automated submission

* **CI** (`.github/workflows/submit.yml`): every push to `main`/`master`/`claude/**` that
  changes `submission/` (or the packaging scripts) validates the estimator, runs the smoke
  checks, packages the folder (uploaded as a workflow artifact) and submits it to AIcrowd
  if the repository secret `AICROWD_API_KEY` is set. Put `[skip submit]` in the commit
  message to package without submitting; `workflow_dispatch` runs it by hand. Mind the
  10 submissions / team / UTC day.
* **Local**: `scripts/install-hooks.sh` installs a pre-push hook that packages whenever a
  push carries changes to `submission/`, and submits too with `FM_AUTO_SUBMIT=1 git push`.
