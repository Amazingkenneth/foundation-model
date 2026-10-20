# A Learned Foundation Model for the ReLU Step

Implementation of *"A Learned Foundation Model for the ReLU Step"* (Option B, width 1024,
depth 16) for the ARC White-Box Estimation Challenge 2026: a factorized K=3 cumulant chain
whose ReLU step is a learned operator

    F_theta(u~, v~) = F_an((u~, v~) + Delta_in) + Delta_out

with ten small MLPs (two hidden layers of width 64, GELU, 114,427 parameters) shared by all
neurons, pairs, layers and networks.

## Current estimator (Oct 20)

The best graded build is **`candidates/v5b_cn69`** (337211, 3.3201e-9), and its 4-level robust twin is
`candidates/v5b_cn69r` (337220, 3.5448e-9). Both are 504aldo's factorized K=3 chain with our cost engineering and
a learned carried corrector. The shipped estimator is not the `submission/` design described above.

- **[`docs/ALGORITHM.md`](docs/ALGORITHM.md)**: verified spec of cn69 (task and limits, chain structure, measured
  cost breakdown, learned corrector, closed ideas, open levers, validation gate, file map).
- **[`docs/PROGRESS_2026-10-19.md`](docs/PROGRESS_2026-10-19.md)**: status, nominations and plan to the deadline.
- [`scripts/gate/README.md`](scripts/gate/README.md): the pre-submission gate tools.

## Layout

| path | what |
|---|---|
| `submission/` | **the shipped folder** (pure stdlib + flopscope): `estimator.py` (entry point), `fm_core.py` (chain + operators, backend-agnostic), `backend_fnp.py` (flopscope backend: pooled buffers, chunked Strassen-Winograd families), `fm_tables.py` (Wick/term coefficient tables), `fm_config.json` (Strassen depths), `k34.npz` + `mp.npz` (the team's trained v5b operator), `LICENSE_504aldo` (MIT, the chain math) |
| `fm/` | training side: `backend_torch.py` (same chain on torch, autograd), `model.py` (parameters, init, export), `mlps.py` (suite-shape MLPs, MC data access) |
| `scripts/gen_data.py` | Monte Carlo ground truth mu_MC^(l) for training/validation MLPs (resumable, top-up) |
| `scripts/train.py` | Algorithm 2 (stage 1: input corrections; stage 2: + output corrections) |
| `scripts/eval_fnp.py` | the submission under flopscope on local ground truth: C/B, residual time, MSE |
| `scripts/parity.py` | torch chain vs flopscope chain (with and without the operator) |
| `scripts/smoke.py` | allowed-imports check, non-suite shapes, weights load |
| `scripts/capcheck.py` | largest array allocated by `predict()` (the grader rejects arrays > 100 MiB), C/B, residual |
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

## The shipped operator: the team's v5b

`submission/k34.npz` + `submission/mp.npz` are the trained weights of the team's v5b
submission (`learned_relu_step_v5b_mp_l8_strassen5`), run by `fm_core.OperatorV5B` on this
repository's chain. `estimator.py` uses `fm_weights.npz` (the paper's ten-network operator)
if present, else the v5b pair, else the analytic chain. The v5b operator:

* `f_self` (layers >= 2): 12 features (alpha, skewness, kappa4 m dG / s^4, own-kurtosis
  transport, off-diagonal variance share, depth, five Stein-type terms of the newborn A leg,
  the transported kappa4 "star" of the older sources) -> 64 -> 64 -> 2; g4 += o_0,
  g3 += 0.1 o_1.
* message passing (layers >= 8, not the last): x = (raw features, g4', g3', 1) in R^15; net A
  (15 -> 192 -> 192 -> 88: the first message and 9 x 4 left / right pair factors), net B
  (175 -> 64 -> 64 -> 16), net O (335 -> 64 -> 64 -> 2: the residual on g4', g3'); ten edge
  matrices (rho, rho^2, t, t^T, rho t, rho t^T, rho^3, |rho|, t^2, (t^T)^2) scaled by
  n / sum |E|; the pair term adds sum_c B_c (L_c R_c^T) s_i s_j^3 to the (3,1) slice over nine
  channels (1, the chain's own kappa_iiij / (s_i s_j^3), t t^T, rho, rho^2, t, t^T, rho t, rho t^T).
* births keep the uncorrected kappa4 diagonal and scale the corrected D3, as v5b does.

Checked against the team's own v5b estimator on the 11 local validation networks (1e6-sample
targets), noise-corrected final-layer MSE:

| | MSE | C/B | MSE x C/B |
|---|---|---|---|
| analytic chain (previous default) | 1.54e-8 | 0.327 | 5.0e-9 |
| this chain + v5b operator, previous Strassen depths | 1.331e-8 | 0.331 | 4.41e-9 |
| **this chain + v5b operator, shipped depths** | **1.345e-8** | **0.2665** | **3.58e-9** |
| the team's v5b estimator | 1.326e-8 | 0.2517 | 3.34e-9 |

At the previous depths the port agrees with v5b to ~1e-6 relative per layer; the deeper
float32 Strassen of the shipped setting moves predictions by ~1e-5 relative (+1% MSE).

## Cost engineering (flopscope backend)

* **Strassen-Winograd** recursion for every n^3 product family (the transport of the young
  legs + the newborn A leg + the covariance in one batched product, the hub contraction of
  the young sources, the shared-basis formings and contractions, the covariance sandwich,
  and the joiner's n x n x 384 range-finder / projection / rotation products), written as
  ordinary flopscope ops. Each level is either *compact* (quadrant gather + one 7x4
  coefficient matmul per operand, one 4x7 matmul + scatter for the assembly: 6 ops) or
  *grouped* (products stored in the slot order M5 M6 M2 M3 M0 M1 M4 so every pair of
  combinations is one strided op: 15 ops).
* **The 100 MiB per-array cap** (graded runs reject larger arrays): every batched product
  runs in balanced batch chunks sized so that its deepest combination stack
  (batch x 1.75^depth x operand) stays under 88 MiB (`CAP_FLOATS`); the A and P legs are kept
  as two (k, n, n) stacks (one (k, 2, n, n) stack would be 120 MiB at k = 15).
  `scripts/capcheck.py` reports the largest array: 85.8 MiB for the shipped setting.
* **Fewer ops per chunk** (from v5b's Strassen): a family's shared left operand (the layer
  weight, a basis) has its combinations computed once and reused by every chunk, and the
  deepest level forms its seven products one at a time (`FUSE`), so the deepest stacks are 7x
  smaller and each chunk holds twice the planes.
* The residual cap is spent on per-op dispatch and view creation, not arithmetic: every
  large result is written into a persistent pooled buffer, every view of a pooled buffer is
  cached, the nonlinear term program is folded into ~10 matrix classes, and the first
  `predict()` of a process runs one level shallower.
* Measured in-process on this repository's 4-core container with the v5b operator (calls 2
  and 3 of a process; flopscope 0.12):

  | `fm_config.json` levels (family, hub, form, inner, sym, join) | C/B | residual / MLP | wall / MLP |
  |---|---|---|---|
  | 2, 2, 1, 1, 0, 0 (previous default) | 0.331 | 0.42-0.46 s | 26-28 s |
  | 4, 4, 3, 3, 3, 3 | 0.272 | 0.79-0.82 s | 61 s |
  | **5, 4, 3, 3, 3, 3 (default)** | **0.2665** | **0.91-0.95 s** | **74-80 s** |
  | 5, 5, 3, 3, 3, 3 | 0.264 | 1.03-1.17 s | 87 s |
  | the team's v5b estimator (graded fine) | 0.2517 | 1.19-1.22 s | 124-138 s |

  The grader's per-op cost is far lower than this container's: 504aldo's V29 (same Strassen
  depth, C/B 0.2526, live on the leaderboard) reports ~0.29 s there, and v5b, its descendant,
  measures ~1.2 s here, so this container is ~4x slower in residual; the default keeps ~25%
  margin under v5b's local residual (and 80 s vs 130 s wall, also with one BLAS thread, as
  on the grader). `whest run --runner subprocess` here therefore flags every MLP (it applies
  the 400 ms cap to this container's clock); judge by the graded `residual_wall_time_s`.
  Edit `submission/fm_config.json` (no code change) to move along the table.
* Where the FLOPs go (default, fractions of the budget): the leaf products of the transport
  family and the young hub 0.107, the shared-basis formings / contractions ~0.05, the joiner
  ~0.02, its dense r x r Gram (384 x 1024 x 384) 0.007, the term program and messages ~0.005,
  copies / additions ~0.01. Family by family this matches v5b except the hub, which v5b runs
  one level deeper (5), and its slightly leaner joiner and copies.

## Graded submissions (Oct 15) and what the grader enforces

| submission | folder | graded score | raw MSE | C/B |
|---|---|---|---|---|
| team v5b (leaderboard row CLG, before) | -- | 4.4e-9 | 1.75e-8 | 0.2518 |
| 336284 (+ an earlier identical one) | `candidates/v5b_rfhub` (v5b, RLEV 4, min blocks 24/28, joiner hub on Strassen) | 4.2513e-9 | 1.7537e-8 | 0.2424 |
| 336295 | **`candidates/v5b_rg`** (rfhub + the n x 384 Gram / basis products on Strassen) | **4.1705e-9** | 1.7534e-8 | 0.2378 |
| 336265 / 336269 / 336281 | level-6 hub / family variants (s6, h6, ga) | failed ("Error while scoring") | | |
| 336359 | `candidates/v5b_rgc192` (rg + tier-2 rank 192 + covariance at 4 levels) | 4.1702e-9 | 1.7873e-8 | 0.2333 |
| 336313 | rg with batch-3 Strassen chunks (fewer ops, 1.5x buffers) | 0.0226 (a few networks zeroed: memory) | | 0.2377 |
| 336410 | `candidates/v5b_cn2` (dead-at-birth source columns dropped, young / joiner / tier-1 products on the live block, flat Strassen pools) | 4.0540e-9 | 1.7637e-8 | 0.2299 |
| 336414 | **`candidates/v5b_cn3`** (cn2 + next-layer transports contract over live rows only) | **3.9756e-9** | 1.7637e-8 | 0.2254 |
| 336447 | `candidates/v5b_cn6` (cn3 + warm-started basis range finders, per-chunk young widths, 32-step rows, covariance at 4 levels) | 3.9041e-9 | 1.7575e-8 | 0.2221 |
| 336448 | `candidates/v5b_cn7` (cn6 + age gates 5 / 8 at the age-law ranks 320 / 192) | 3.8118e-9 | 1.7443e-8 | 0.2185 |
| 336459 | **`candidates/v5b_cn8`** (cn7 + oversampled D21 feedback range finder, 16+16 -> 16) | **3.8110e-9** | 1.7365e-8 | 0.2195 |
| 336483 | **`candidates/v5b_cn9`** (cn8 + tier-1 basis rebuilt on alternate joins, tier 2 from age 9 at rank 160) | **3.7487e-9** | 1.7391e-8 | 0.2156 |
| 336484 | `candidates/v5b_cn10` (cn9 + tier-1 rank 288) | 3.7699e-9 | 1.7916e-8 | 0.2104 |
| 336496 | **`candidates/v5b_cn11a`** (cn9 + rows at Phi < 0.01) | **3.6860e-9** | 1.7367e-8 | 0.2122 |
| 336497 | `candidates/v5b_cn11b` (cn10 + rows at Phi < 0.01) | 3.7086e-9 | 1.7897e-8 | 0.2072 |
| 336544 | **`candidates/v5b_cn21`** (cn11a + D21 feedback rank 2) | **3.5816e-9** | 1.7950e-8 | 0.1995 |
| 336605 | `candidates/v5b_cn35` (cn21 + exact per-chunk hub narrowing) | 3.5756e-9 | 1.7950e-8 | 0.1992 |
| 336607 | **`candidates/v5b_cn36`** (cn35 + S21 residual rank 8) | **3.5275e-9** | 1.8053e-8 | 0.1954 |
| 336609 | **`candidates/v5b_cn37`** (cn35 + S21 residual rank 4) | **3.5262e-9** | 1.8235e-8 | 0.1934 |
| 336610 | `candidates/v5b_cn40` (cn35 + (3,1) kappa4 slice term at b2 = -0.5) | 3.6151e-9 | 1.8160e-8 | 0.1991 |
| 336612 | `candidates/v5b_cn41` (cn35 + learned MP pair correction x0.4) | 3.6782e-9 | 1.8478e-8 | 0.1991 |
| 336627 | `candidates/v5b_cn44` (cn36 + young families at 6 Strassen levels) | failed ("Error while scoring", 4th level-6 failure: level 6 is out) | | ~0.194 |
| 336664 | `candidates/v5b_cn49b` (cn37 + marg2 + D21 feedback oversampling 8; 75-network gate -0.8% MSE, -0.2% C) | **3.5121e-9** | 1.8204e-8 | 0.1929 |
| 336695 | `candidates/v5b_cn50` (cn49b + columns Phi < 0.006, rows Phi < 0.012; gate -1.5% MSE, +0.7% C vs cn49b) | 3.5175e-9 | 1.8116e-8 | 0.1942 |
| 336767 | **`candidates/v5b_cn53b`** (cn49b + Strassen-Winograd kernel + slab buffer pool without billed reshapes; exact) | **3.4863e-9** | 1.8205e-8 | 0.1915 |
| 336768 | **`candidates/v5b_cn53`** (cn50 + the same; exact) | **3.4907e-9** | 1.8116e-8 | 0.1927 |
| 336770 | **`candidates/v5b_cn54b`** (cn53b + nonlinear term program accumulated elementwise; exact) | **3.4544e-9** | 1.8202e-8 | 0.1898 |
| 336850 | `candidates/v5b_cn54` (cn53 + elementwise term program; exact; cn50 masks) | 3.4588e-9 | 1.8112e-8 | 0.1910 |
| 336851 | `candidates/v5b_cn55b` (cn54b + Gram cores for the joiner/mover weighted cores; exact) | failed ("Error while scoring") | | |
| 336852 | `candidates/v5b_cn55` (cn54 + Gram cores; exact; cn50 masks) | failed ("Error while scoring") | | |
| 336887 | `candidates/v5b_cn58` (cn54b + RW_TAU 0.02; local prediction ~3.437e-9: 300 public nets +0.46% MSE, -0.94% C) | **0.0502** (score = raw MSE: every network zeroed at full budget) | 0.0502 | |
| 336962 | `candidates/v5b_cn59` (cn54b + RW_TAU 0.02 + residual rank 6; local prediction ~3.43e-9) | **0.0310** (score = raw MSE: ~62% of networks zeroed; cn58 0.0502 = all) | 0.0310 | |
| 336965 | `candidates/v5b_cn54b` resubmitted unchanged (grader-health control; graded 3.4544e-9 as 336770 on Oct 17) | **0.0351** (~70% of networks zeroed) | 0.0351 | |
| 336966 | **`candidates/v5b_cn61`** (cn54b + residual rank 6 + late dense window from layer 14; local: 300 public nets -1.27% MSE, +0.6..0.9% C) | **3.4404e-9** | 1.8012e-8 | 0.1910 |
| 336968 | `candidates/v5b_cn61r` (cn61 at 4 Strassen levels on the young families: in-process wall 81.6 s vs 106.9 s, C/B 0.2035 vs 0.1906) | **3.6743e-9** (clean) | 1.8017e-8 | 0.2039 |
| 336975 | `candidates/v5b_cn60` (cn61 + RW_TAU 0.02; 100 mini vs cn61: +0.63% MSE, -0.94% C, net -0.3%) | 3.4395e-9 (clean; = cn61 within noise) | 1.8177e-8 | 0.1892 |
| 336979 | `candidates/v5b_cn62` (first upload) | failed ("Smoke test failed": corr.npz had Fortran-order arrays, which the client npz reader rejects) | | |
| 336980 | `candidates/v5b_cn62` (cn61 + carried created-error corrector, MLP on 297 public full networks; held-out mini -2.02% MSE, +0.54% C; prediction ~3.39e-9) | 0.0204 (~40% of networks zeroed: grader load, wall +3.5% vs cn61) | 0.0204 | |
| 336984 | **`candidates/v5b_cn62r`** (cn61r + the carried corrector, 297-network MLP) | **3.6369e-9** (clean; raw -1.51% vs cn61r) | **1.7745e-8** | 0.2050 |
| 337009 | `candidates/v5b_cn63` (cn61 + carried-state corrector: S=32 learned state through W, 3x128 GELU on 1000 public full nets; in-chain mini -3.89% MSE, +0.85% C; prediction ~3.33-3.37e-9) | **0.0220** (~44% of networks zeroed: the 5-level wall under grader load, as cn62's 0.0204) | 0.0220 | |
| 337017 | **`candidates/v5b_cn63r`** (cn63 at 4 Strassen levels on the young families; C/B 0.2059 on the emulator; prediction ~3.57-3.60e-9) | **3.5984e-9** (clean; raw -2.83% vs cn61r, local -3.89%) | **1.7507e-8** | 0.2055 |
| 337030 | **`candidates/v5b_cn64`** (cn61 + carried-state corrector S=32 + W*W channel S2=16, ReLU, 1000 public full nets; in-chain mini -4.58% MSE, +0.85% C; lazy-view Strassen kernel: -23% slices) | **3.3386e-9** (clean; new best; raw -3.78% vs cn61, local -4.58%) | **1.7332e-8** | 0.1926 |
| 337037 | **`candidates/v5b_cn64r`** (cn64 at 4 Strassen levels on the young families; emulator C/B 0.2060) | **3.5636e-9** (clean; best robust grade) | 1.7336e-8 | 0.2056 |
| 337053 | **`candidates/v5b_cn66`** (cn64 + 16-epoch corrector: in-chain mini -4.72% MSE; PK2 views made once per layer: -6.2% client/server round trips, bitwise-identical chain) | **3.3341e-9** (clean; new best) | **1.7309e-8** | 0.1926 |
| 337060 | **`candidates/v5b_cn67r`** (cn66 at 4 Strassen levels + persistent PK2 buffers: bitwise = cn66r, round trips -9.7%; emulator C/B 0.2060) | **3.5588e-9** (clean; best robust grade) | 1.7313e-8 | 0.2056 |
| 337134 | **`candidates/v5b_cn68`** (cn67 + corrector inputs for each neuron's pre-activation correlation structure: in-chain mini -4.85% vs cn66 -4.72%; round trips 43.1k) | **3.3268e-9** (clean; new best; raw -4.15% vs cn61) | **1.7264e-8** | 0.1927 |
| 337147 | **`candidates/v5b_cn68r`** (cn68 at 4 Strassen levels; emulator C/B 0.2060) | **3.5510e-9** (clean; best robust grade) | 1.7268e-8 | 0.2056 |
| 337211 | **`candidates/v5b_cn69`** (cn68 with a small corrector, H 64 / S 16 / S2 8: in-chain mini -4.71% vs -4.85%, emulator C/B 0.1922 vs 0.1931, round trips 43.1k) | **3.3201e-9** (clean; new best) | 1.7309e-8 | 0.1918 |
| 337220 | **`candidates/v5b_cn69r`** (cn69 at 4 Strassen levels; emulator C/B 0.2051) | **3.5448e-9** (clean; best robust grade) | 1.7313e-8 | 0.2048 |
| 337265 | **`candidates/v5b_cn71`** (cn69 + closure knobs refit end to end: all 100 mini -0.50%, held-out odd 50 -0.59%; C/B 0.19145; round trips 43.3k) | **3.3183e-9** (clean; new best, -0.05% vs cn69) | 1.7297e-8 | 0.1918 |
| 337275 | **`candidates/v5b_cn71r`** (cn71 at 4 Strassen levels) | **3.5407e-9** (clean; best robust grade) | 1.7290e-8 | 0.2048 |

Prepared for Oct 17 (all on cn11a; emulator ok at 205-262 s on the loaded container, VmHWM 3.46-3.48 GB,
smoke shapes finite): `v5b_cn12` (S21 residual rank 8, C/B 0.209), `v5b_cn13` (rank 4, 0.208),
`v5b_cn14` (D21 feedback rank 8, 0.206), `v5b_cn15` (both rank 8, 0.202), `v5b_cn18` (both rank 4,
0.196, VmHWM 3.45 GB), `v5b_cn19` (both rank 2, 0.193), `v5b_cn20` / `v5b_cn21` (feedback rank 4 / 2
at residual rank 16, 0.201 / 0.199; the feedback rank moves the chain at chaos level), `v5b_cn22` /
`v5b_cn23` (feedback rank 2 with residual rank 8 / 4, 0.196 / 0.194). Oct 17 wave 1: `v5b_cn35`
(cn21 + exact per-chunk hub narrowing) and `v5b_cn36`-`v5b_cn39` (cn35 + residual rank 8 / 4 / 2 / 1),
chosen with a label test on the six hardest local MLPs (docs section 18). All pass the emulator (156-177 s
on the fast container), VmHWM 3.44-3.48 GB, smoke shapes finite. `v5b_cn16` (exact
tier-2 eigensolve) measured as a wash and is not planned; the D21 age gate (cn17) is closed
(old-source D21 is load-bearing, docs section 15).

The cn line (Oct 16, `docs/experiments_2026-10-16.md`): neurons with Phi(alpha) < 0.01 at a
source's birth get no columns in that source (a 15-23% cut of the source width from layer 8 on),
and the post-wick rows of neurons with Phi < 0.002 leave the next transports' inner dimension.
Both are exact up to the chain's float32 chaos level (relabelling rg's own network moves the
output as much). Strassen scratch is now one flat buffer per role, which also halves the
resident peak (rg 6.12 GB -> cn2 3.33 GB, cn3 3.50 GB).

**For the final (private) selection (Oct 20, 05:12 UTC): 337265 (`v5b_cn71`, 3.3183e-9, best clean grade) and 337275 (`v5b_cn71r`, 3.5407e-9: the same estimator on the 4-level robust chain; 4-level builds have graded clean every time).**
earlier advice: 336414 (`v5b_cn3`) and 336295 (`v5b_rg`)): both
graded with no failed network (cn3: -4.7% vs rg; row compaction is exactly free -- cn2 and cn3
have the same raw MSE to 4 digits -- and the dead-column mask costs +0.6% raw for -3.3% C/B).

What the grader enforces that in-process flopscope does not (reproduced with
`scripts/grader_emulator.sh`, which runs an estimator against `flopscope-server` /
`flopscope-client` 0.12.0 like the grader):

* every op is an IPC round trip, so wall and residual time scale with the op count far more
  than in-process: the failed level-6 variants ran ~150 s in-process like rfhub, but 215-220
  s under the emulator; rfhub / rg run 145-178 s there and pass. Treat ~175 s emulator wall
  as the ceiling. Confirmed on Oct 17: cn44 (level 6 on the young families, 239 s quiet against cn36's 202 s)
  failed like the earlier level-6 builds. The grader zeroes any network over 120 s of wall
  (Artyom_Mazur 18249, A.2), and one zeroed network is the whole entry. Deeper Strassen buys FLOPs
  with wall time, and our wall is already at the cap.
* a per-result array cap (`FLOPSCOPE_MAX_ARRAY_BYTES`): a 134,873,088-byte result passes, a
  137,682,944-byte one fails (the server default, 100 MiB, is lower than the grader's).
* memory, per network: rg / rfhub peak at 6.0-6.1 GB resident (VmPeak 7.7-7.8 GB) and pass;
  the batch-3 variant peaks at 7.0 GB (VmPeak 8.8 GB) and lost a few networks. The emulator
  does not enforce this; check `VmHWM` in-process. Most of rg's peak was duplicate
  shape-keyed Strassen pools: the flat per-role pools of `candidates/v5b_cn*/strassen2.py`
  bring it to 3.3-3.5 GB.
* client/server argument encoding: tuples arrive as lists (`fnp.sum(x, axis=(0, 2))` fails);
  `submission/` was fixed for this (it had never been graded).
* billing (flopscope 0.12.3, unchanged from 0.12.0 for our chain): `fnp.reshape` / `ravel` are billed
  at one FLOP per element, while slicing, `x[None]`, `swapaxes`, `broadcast_to` and `fnp.empty` are free;
  the 4-operand term-program einsums bill 6.8-8.3 n^2 per term against 3 n^2 elementwise. Both are fixed
  in `candidates/v5b_cn53*` / `v5b_cn54*` (strassen4: max-shape slabs + Winograd form; -2.4% C/B, same
  predictions). The daily submission limit is 5.

On the client runtime `submission/` (this repository's backend) is far lighter than v5b
(deep16: 92-97 s, residual 0.75 s), but even at family depth 6 it stays at C/B 0.255, and
its MSE is ~1.4% above v5b's, so the v5b line is the one to submit. Submit a folder with
`SUBMISSION_DIR=candidates/v5b_rg scripts/package.sh && whest submit dist/<artifact>`, or
from CI (Actions -> package-and-submit -> Run workflow, folder `candidates/v5b_rg`; needs
the `AICROWD_API_KEY` secret). Evaluation of the teammates' notes 13-15:
`docs/notes_13-15_evaluation.md`.

## Next steps (team meeting)

* **Option B, accelerate the linear part.** Done here: deeper chunked Strassen everywhere,
  the joiner and the covariance sandwich on Strassen, LeftTree reuse, fused last level
  (C/B 0.331 -> 0.2665 at equal accuracy). Open: the top teams spend 10-13% of the budget;
  the next gains need structure rather than faster dense products, e.g. recycling the
  products shared by the first-, second- and third-order transports (W C W^T and W A_s share
  the left factor W; the D3 / D21 slices reuse the hub legs), and lower-rank young legs.
* **More training data for the ReLU operator.** `scripts/gen_data.py` is resumable and
  tops up existing files; `scripts/train.py --v5b submission` fine-tunes the v5b operator
  end to end (torch, checkpointed through layers 1..15) and writes `k34.npz` + `mp.npz`
  to ship. With the 64 local networks the target noise (~1.4e-7 per network) dominates the
  analytic error, so retraining only pays with many more networks and samples.

## Results in this repository (paper operator, before the v5b port)

* **Analytic chain** (no operator): final-layer MSE against the local Monte Carlo ground
  truth, corrected for the MC noise of the targets, **1.54e-8** on 11 validation MLPs
  (1e6 samples each) -- the same for the torch chain and the flopscope submission
  (`scripts/parity.py`: per-layer agreement ~1e-6 relative, final-layer MSE gap ~5e-11).
  For reference the PDF quotes 2.23e-8 for the analytic K=3 chain on the public suite.
* **Learned operator** (the paper's ten networks): the full Algorithm 2 pipeline runs (both
  stages, checkpointed backprop through layers 9..16, export, flopscope inference with
  parity to torch). Trained here on **64** networks with 5e5-sample targets (the paper: 600
  networks) the corrections do **not** generalize: the target noise (~1.4e-7 per network)
  is ~10x the analytic error, and the validation MSE rises (1.55e-8 -> 1.63e-8 -> 1.92e-8
  over two stage-1 epochs) while the training MSE falls. So no `fm_weights.npz` is shipped;
  the submission runs the team's v5b operator instead. Dropping a better `fm_weights.npz`
  into `submission/` takes precedence with no code change.

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
#    or fine-tune the shipped v5b operator: writes runs/v5b/k34.npz + mp.npz
python scripts/train.py --v5b submission --epochs 4 --lr 2e-5 --out runs/v5b
# 3. check
python scripts/parity.py
python scripts/capcheck.py
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
