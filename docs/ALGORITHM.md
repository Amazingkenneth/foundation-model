# cn69: algorithm spec (ARC White-Box Estimation Challenge 2026, Phase 2)

What `candidates/v5b_cn69` computes, what it costs, how it was validated, and what has already been tried.
It is written for someone (human or AI) who wants to improve it without re-running the last week. Numbers come
from code, the 0.12.3 profiler run of Oct 20 (`scripts/gate/prof_fn.py`), graded submissions, and the experiment
logs in `docs/final_phase_2026-10-17.md`. That file is the full record; the section numbers cited below are its own.

**Update Oct 21, evening:** the current best is **337588 cn79 3.2790e-9** (raw 1.7311e-8, C/B 0.1894): cn78 plus
scale-mode surrogates for the dropped final rows (section 6). Twin cn79r is 337606.

**Update Oct 21:** cn78 was **337555 3.2888e-9** (raw 1.7301e-8, C/B 0.1901): cn71 plus the
final-layer row restriction of section 6 (-0.89% graded, predicted 3.290e-9). Its 4-level twin **337564 cn78r 3.5089e-9**.

**Update Oct 20:** the then best was **337265 cn71 3.3183e-9** (raw 1.7297e-8). Its twin **337275 cn71r
3.5407e-9** is the robust nomination. cn71 = cn69 + closure knobs refit end to end (Section 2a); everything below
holds for both.

Graded: **337211 cn69 3.3201e-9** (raw MSE 1.7309e-8, C/B 0.1918). Its 4-level twin **337220 cn69r 3.5448e-9**
(raw 1.7313e-8, C/B 0.2048). Leaderboard on Oct 19: #1 1.30e-9 (raw 1.10e-8, C/B 0.122); we are about #57.

---

## 1. Task and limits

| item | value |
|---|---|
| network | ReLU MLP, width n = 1024, depth L = 16, Gaussian input; weights from `fm.mlps.make_weights` / the WhestBench bake |
| output | per-layer post-ReLU means, shape (L, n). Only the final layer is scored |
| score | mean over networks of final-layer MSE x max(0.1, C/B), with B = 2^41 FLOPs metered by flopscope |
| interface | `class Estimator(BaseEstimator)` with `setup(ctx)` and `predict(mlp, budget)`; data files are pickle-free, C-contiguous `.npz` (the client npz reader rejects Fortran order) |
| grader runtime | flopscope-client **0.12.3**; every array op is a synchronous IPC round trip to a flopscope server |
| caps per network | **120 s wall** and **0.4 s residual** (client Python time outside metered ops). A network over a cap is zeroed. One zeroed network adds ~0.009 to the mean score; ours is 3e-9 |
| array cap | ~134.9 MB per server-side result (134,873,088 B passed, 137,682,944 B failed) |
| smoke test | non-suite shapes (widths 4..256, depths 2..8) must return finite outputs; learned parts run only at 1024 x 16 |
| private ranking | fresh MLPs with private seeds on 0.12.3; up to 2 nominations by Oct 22 23:59 UTC; submissions close Oct 21 23:59 UTC |

What failed graded runs in the past: Strassen level 6 (wall cap; 4 failures), loaded-grader days (identical
code lost 40-70% of its networks), Fortran-order arrays in corr.npz, and the corrector at a non-suite shape. The
4-level twin (`cn69r`, young families at 4 Strassen levels) has graded clean every time; the 5-level builds have
graded clean since the residual cuts of cn64.

## 2. Chain structure

cn69 = 504aldo's MIT V25/V29 factorized K = 3 cumulant chain (`LICENSE_504aldo`; docstring of `estimator.py`
lines 1-146), plus the team's learned kappa operators from its v5b submission (`k34.npz`, `mp.npz`; see the
README), plus our cost engineering (cn3-cn67), plus
our learned carried corrector (`corr.npz`, Section 5). All defaults below are code defaults in cn69 (env knobs in
parentheses); no env vars are set when graded.

State carried from layer to layer: the mean `mu`, the covariance `C` of the layer input, and **third-cumulant
sources**. Each ReLU creates one newborn source. A source born at layer b carries dense legs A, P (n x n) and an
M leg written as `M = P diag(s) + 3 A diag(e) + Z L^T`: a thin residual Z of rank `R_RES` = 6 (`CN_RRES`), plus
thin D21-feedback legs F1/F2 of rank `R_FB` = 2 (`CN_RFB`, oversampling 8 `CN_FBOS`).

Per layer `li` (`_predict_core`, sections marked `# ---- ... ----`):

1. **linear** (l.792). `C_pre = W C W^T` (`_cpre_sym`, 4 Strassen levels), `var = diag(C_pre)`, `C_off` = off-diagonal.
   Every source's legs are transported by `WD = W diag(w1_prev)` (`_mmc`, batched Strassen at `SLEV = 5` levels,
   chunks of `SB = 2`). The newborn's A leg is `W (w1 * C_off)` and its P leg is `W`.
   - *Age gate* (`V21_AGE_OLD` = 5, one older from layer 14 on via `CN_AGE6FROM` = 14): sources older than the gate
     are confined to a shared tier-1 basis `Qc` of rank `R_OLD` = 320. The basis comes from a randomized range finder
     on the weighted leg Gram, rebuilt on every other join (`CN_KEEP=alt`, warm start). Sources older than
     `V24_AGE_OLD2` = 9 move to a nested tier-2 sub-basis `U` of rank `R_OLD2` = 160. Tier legs are formed from
     static factors (`_mmc2`, `_sd`; `RLEV = 4` levels, batch `T1B = 6` under the array cap).
   - *Row mask*: rows of neurons with Phi(alpha) < `RW_TAU` = 0.01 are dropped from the next transports, in steps of
     32 with at least `CN_MIN` = 768 rows kept.
2. **WK slices** (l.1057). `_dslices` forms the per-neuron kappa3 diagonal **D3** and the (2,1) slice **D21**
   (n x n, kappa3(z_i, z_i, z_c)):
   `D21 = sum_k [LA_k A_k^T + LP_k P_k^T] + thin/feed terms`, with `LA = 2 (A o P) w2 + (P o P) e` and
   `LP = (A o A) w2 + (P o P) s/3 + (2/3) M o P`. Young sources go through `_hub2` at 5 levels; tier sources go
   through the basis (`inner Qc^T`, 4 levels).
   - kappa4 is **memoryless** (V17/V25): `G = diag(g) + lambda C_off`. Lambda is the previous layer's table value
     (`LAM x 0.95`), rescaled online by `clip((mean dG0 / mean var) / REF_R, 0.5, 2)`. The transported diagonal is
     `dG = (W*W) g + lambda (var - (W*W) var_prev)`, and `g4row = 2 dG`.
   - The **learned k34 step** (12 -> 64 -> 64 -> 2 GELU, the team's v5b operator) corrects the standardized D3 and g4row at
     every layer with sources (li >= 1).
   - **Message passing** (`mp.npz`: MLPs A 15->192->192->88, B 175->64->64->16, O 335->64->64->2) rewrites D3, g4row
     and the (3,1) slice `wk431` from li >= `MP_LI_MIN` = 8, reading `C_off` and D21.
3. **wick matrix + nonlin term program** (l.1148-1267). Wick coefficients `w(k,p)` per neuron from alpha = mu/sigma,
   in the unified form `const sigma^e (P1(alpha) phi + P2(alpha) Phi)`. The post-ReLU mean (`pk1v`) and covariance
   come from a fixed term table (`TERM_SPECS`, pair terms of arity 1/2 up to (2,2)) consuming C_off, D3, D21, g4row,
   wk4m and wk431. Includes the marginal-Edgeworth tail fix `XMARG2` = 1; the C0^3 Mehler term `XC3` = 0 is off.
4. **learned carried corrector** (l.1286-1360). See Section 5.
5. **new source birth** (l.1390). `a_b = w1 o C_off` with dead-at-birth columns (Phi < `CM_TAU` = 0.01) zeroed and
   live columns stored first (`perm_list`), plus the Rres range finder (rank 6) and the D21 feedback range finder
   (rank 2).
6. **final layer** (V19 trim): only the mean (var, D3, kappa4 diagonal; no C_off / D21 / pair program). Output =
   `pk1v + s_L` (corrector). The old ridge rider is off (`V17_NO_CORR=1`).

The 4-level twin differs in one line: `SLEV = 4` (`CN_SLEV`) on the young families.

## 2a. cn71: closure knobs refit end to end (Oct 20)

`candidates/v5b_cn71/estimator.py` = cn69 + a `KN` table (l.461-468), applied in the WK-slice block right after the
learned k34 / MP steps (l.1129-1140) and at birth (feed, l.1562). Bands: E = layers < 8, L = 8-14, F = final layer.

| knob | value | what |
|---|---|---|
| D3 E / L / F | 0.9958 / 0.9975 / 1.0027 | consumed kappa3 diagonal |
| g4row E / L / F | 1.0140 / 0.9966 / 0.9990 | consumed kappa4 diagonal (and wk4m, built from it) |
| wk431 E / L | 1.1349 / 0.9140 | (3,1) slice |
| D21 E / L | 0.9975 / 0.9946 | (2,1) slice as consumed (after MP) |
| LF | 0.7996 | kappa4 -> kappa3 feed `c1_b = lambda w1` of the newborn (suite shape only) |
| S22 | -0.2752 | added SSC-type (2,2) structure `S22 x lambda x METRIC_C x C_off o C_off` |

How the values were found:
- Fit: a finite-difference Jacobian (+5% per knob) of the final means of cn69 *with its corrector*, on 24 even mini
  networks, followed by one ridge Gauss-Newton step (rho 1e-2). The linear prediction (-0.88%) was realised exactly.
- Validation: all 100 mini -0.50% +- 0.13; the 50 odd networks, never in the fit, -0.59%.
- Grader: -0.07% raw (transfer ~0.15x, within ~2 sigma on 50 networks).
- Cost: +0.02% C, +0.3% round trips.

Round 2 (17 knobs re-linearised at cn71, plus a variance-scaled (3,1) term, a gate-weighted (2,2) term, (2,2)
multipliers and a late-only feed knob) found nothing: every knob is at its optimum. Tools: `sc/c72` (env `XKN`),
`sc/c73`, and `/tmp/claude-0/acc/knobsolve*.py` (see `docs/final_phase_2026-10-17.md` 8.15).

## 3. Cost breakdown (measured)

cn69, seed 1, in-process flopscope 0.12.3, `scripts/gate/prof_fn.py`: **C/B 0.19142 = 4.21e11 FLOPs** (the client/
server emulator bills 0.1922; the grader 0.1918).

| function (innermost estimator frame) | share of C | what |
|---|---|---|
| `_hub` | **39.7%** | D21 hub contractions: young sources (Strassen 5 levels), tier inner products (4 levels), range-finder hub |
| `_mmc` | **36.1%** | leg transports `WD @ leg` of young sources, the newborn's A leg, `W C` |
| `_predict_core` (inline) | 7.3% | linear-section inline matmuls 5.5%, nonlin term program 0.64%, **corrector 0.42%**, birth range finders 0.38% |
| `_mmc2` | 4.5% | tier-1/2 leg formation and rotation |
| `_dslices` | 3.6% | D3 / D21 einsums and thin terms |
| `_cpre_sym` | 3.6% | the `W C W^T` sandwich |
| `_sd` | 2.8% | basis products (`... Qc^T`) |
| `_mp_step`, `_mlp3`, `_star`, `_k34_step` | 1.3%, 0.7%, 0.2%, 0.2% | learned kappa operators |

By op: matmul 86.7%, einsum 3.7%, add 2.9%, subtract 2.6%, multiply 1.5%, copyto 1.2%, QR 0.65%.

Runtime:
- In-process wall is 102-112 s; the local client/server emulator measures 222 s with a 1.42 s residual. Locally
  measured absolute residuals over-predict the grader (forum 18255: 0.39-0.45 s locally vs 0.319 s official), and cn69 grades clean.
- **43,136 client round trips per MLP** steady state (49,945 cold): add 9.5k, subtract 8.5k, `__getitem__` 7.6k,
  copyto 5.0k, multiply 4.7k, matmul 2.9k.
- VmHWM 4.19 GB.

Elasticities on exact labels (100 mini networks): every cheap-side knob step costs 1.5-3% MSE per 1% C, and every
expensive-side step buys about 0.6% MSE per 1% C. The truncation share of the MSE is only ~4.5% (all confinement
and masks off: -4.47%). **The remaining error is closure error, not truncation.**

## 4. Error anatomy (exact labels)

- **Error accumulates steadily with depth.** Mini net 0, final-layer contributions by layer 2-15: 1.0 ... 18.1e-9.
  Measured propagation gains of a created error to the output: 0.031 at layer 0 up to 1.0 at layer 15 (`GAIN` in
  `fitrnn.py`).
- **The final layer itself creates about 22% of the MSE.** The rest is propagated.
- **Bias is negligible.** Any post-hoc final-layer fix tops out at 0.4-1.6% (per-network oracle fits).
- **Per-neuron gains are nearly uniform.** Within a layer they vary by 6-27% (CV) and are uncorrelated with the
  created errors, so a per-layer gain-weighted loss is the right objective.
- **Label noise:** 1e9-sample truth gives ~avg_var / 1e9, about 0.1-0.7e-9 per layer, well below our errors.

## 5. Learned carried corrector (ours; `corr.npz`, code l.1286-1360)

The corrector predicts each layer's **created mean error**
`c_l = e_l - Phi_l o (W_l e_{l-1})` (where `e_l` is the chain's mean error at layer l). Its outputs are propagated
by the chain's linearized map and added once, at the output:

```
s_0 = delta_0;   s_l = Phi_l o (W_l s_{l-1}) + delta_l;   prediction_L = pk1v_L + s_L
```

One shared MLP serves every layer (inputs standardized by `mu`/`sd`; ReLU; 3 hidden layers of H = 64; output
`1 + S + S2` = 25). `delta_l = out[0] * ysd`; `out[1:17]` is the next state, `out[17:25]` is the next W*W state.

Inputs per neuron (92 -> 68 dims in cn69):

| group | features |
|---|---|
| local (20) | mu, var, sigma, alpha, phi, Phi, pk1v, pk2v, pk3v, pk4v, D3, g4row, k4_prev, K4_vec, rowsum D21, colsum D21, rowsum abs(D21), rowsum C_off^2, rowsum abs(C_off), rowsum rho^2 (rho_ij^2 = C_ij^2 / (var_i var_j)) |
| previous layer through W (5) | pk1v, pk2v, D3, g4row, rowsum D21 |
| previous layer through W*W (3) | pk2v, var, rowsum rho^2 |
| layer one-hot (16) | |
| state S = 16 | `(W s_{l-1}) / sqrt(sum(W*W) / n)` |
| state S2 = 8 | `((W*W) s2_{l-1}) / (sum(W*W) / n)` |

Training (`scripts/corr2/fitrnn.py`):
- **Command:** `S2=8 ACT=relu DEC=0.85 fitrnn.py ld61c_v3 f0-999 m0-99 16 64 16 <out>`.
- **Optimizer:** Adam with LR 2e-3, decayed 0.85 per epoch, 16 epochs, one network per step.
- **Loss:** gain-weighted per-layer MSE of `delta` against `-c_l / ysd`. Gradients run through the states
  (backprop through the 16 layers).
- **Data:** chain dumps of the 1,000 public full-split networks with their 1e9-sample labels. Held out: 98 mini
  networks (`keys_m98.txt`, m81 and m93 excluded).
- cn69 ships the raw final weights (the EMA weights gave -4.74% vs -4.76%).

Measured history (held-out mini dMSE vs the uncorrected chain; graded raw MSE vs the uncorrected cn61 1.8012e-8, or vs
cn61r 1.8017e-8 for the 4-level twins):

| step | held-out | graded |
|---|---|---|
| per-layer MLP, no state (974 nets) | -2.23% | cn62r -1.51% (an earlier 297-net model) |
| + 32-d state through W | -3.92% | cn63r -2.83% |
| + W*W state (S2 16), ReLU | -4.60% | cn64 -3.78% |
| + 16 epochs (decay 0.85) | -4.72% | cn66 -3.90% |
| + v3 inputs (rho^2, C_off row sums) | -4.90% (in chain -4.85%) | cn68 -4.15% |
| **small: H 128 -> 64, S 32 -> 16, S2 16 -> 8** | **-4.76% (in chain -4.71%)**, C -0.46% | **cn69 -3.90%, score -0.20%** |

Grader transfer of the local gain is 0.73-0.86x. Parity between the in-chain corrector and the torch pipeline is
2.3e-6 max abs on m50.

## 6. Closed ideas (measured; do not re-run without new evidence)

dMSE is on exact labels (mini networks unless stated); dC is the 0.12.3 bill.

**Chain cost / truncation knobs:**

| change | dMSE | dC | verdict |
|---|---|---|---|
| AGE_OLD 4 / 6 | +10.56% / -2.54% | -5.10% / +4.08% | both lose |
| R_OLD 288 / 352 | +2.75% / -1.69% | -0.91% / +2.67% | lose |
| AGE_OLD2 8 / 10 | +1.75% / -0.66% | -1.16% / +0.97% | lose |
| R_OLD2 128 | +1.58% | -0.68% | loses |
| CM_TAU 0.02 | +3.70% | -1.42% | loses |
| RW_TAU 0.02 / 0.03 | +0.60% / +2.31% | -0.94% / -1.54% | 0.02 graded neutral (cn60); rows are capped at 768 anyway |
| CN_RRES 2 / 8 | +3.81% / -1.34% | -0.50% / +1.00% | 6 is the optimum |
| CN_RFB 1; feedback legs off; CN_KEEP all | +0.49%; +7.80%; +2.55% | -0.48%; -1.81%; -1.68% | lose |
| two-sided newborn column mask: also drop columns with Phi > 0.99 at birth, floor 512 (`CM_SAT=0.01 CN_MINC=512`, scratch c76, 12 mini nets) | +110% | -5.2% | no: the saturated-on columns carry the kappa4 -> kappa3 feed (weight w1 ~ 1), only their newborn terms (w2 ~ phi) vanish |
| dead rows / columns / saturated rows of the D21 hub | +7.9..+55% / +8.9% / +14.6% | | they carry real content |
| low-rank D21, dropping old sources | x4 MSE | | out |
| Strassen level 6 (young families) | | -0.9..-1.2% | failed on the grader 4 times (wall) |
| alternative-basis Strassen, rank-48 <4,4,4> | | ~0 / -4% of leaf products | extra ops break the residual cap; ~0 net at 1024 / 5 levels |

**Closure / physics** (forum sources in brackets):

| change | result | verdict |
|---|---|---|
| (3,1) two-term law [18246] | +1.4..+4.0% | our learned MP already carries that slice |
| per-neuron metric diag(W W^T) [18248, e30_a] | +0.06% +- 1.35 | absorbed by the learned operator |
| C0^3 Mehler pair term (= 18252's "missing C_off^3/6 H3 H3") | -0.2% +- 1.9 | null |
| scale-mode extraction with a carried kappa4 core [18248] | ported Oct 20 (`docs/leaders_2026-10-20.md`): carried G + K211 (newborn) + extraction 0.25 on ARC's augment table, learned ops off: +14.3% vs the cn71 chain on 8 mini nets (0/8); the learned k34/MP break on it (x50) | no (needs retrained operators + corrector) |
| spherical cubature, 8,192 orthonormal-frame points [marius_binner P1] | 4.5e-6 final MSE on m70 (iid: 7.3e-6) | no (400x short) |
| tiers restructured by the age-rank law: ages 3-5 in a rank-640 basis, ages >= 6 at 320 (`V21_AGE_OLD=2 V21_R_OLD=640 V24_AGE_OLD2=5 V24_R_OLD2=320`) | C/B **0.246** (cn71 0.191), wall 187 s in process | no (tier legs at rank 640 cost more than dense young legs) |
| skip the final layer's source work (`V19_NO_SRC_LAST=1`, D3 := 0 at the last layer; probe path fixed to transport C) | C/B 0.1834 (-4.2%), chain-only MSE x11.7 on m70 | no (the final kappa3 diagonal is essential) |
| MP from layer 6 instead of 8 | +1.8% +- 3.1 | out |
| MP pair strength x0.4 | -2.2% on 6 tuning nets, +3.1% on 6 held-out | selection effect |
| post-Jul-31 term tables (Edgeworth fix) | worse for lambda-law chains (il.miscusi) | no |
| **E1 kappa4 pendant-edge term** [18252 s3.1] | chain-only: scale 0.25 -1.15%, 0.5 -1.41% (even nets); 0.45 -0.78% (odd nets); layers 11-14 only -0.42% | real gain (~-1.1%), but our D21 is not `M W^T`, so exact E1 needs one more dense product per layer (~+0.26% C each, +3.4% for all layers): **net loss**. Screen: `scripts/gate/e1_screen` |
| young-atom pruning by norm score [18254] | | our dead-at-birth column mask + 768 floor already covers it |
| final-layer ridge rider, per-network scale | null (bias negligible) | off |

**Final-layer row restriction (Oct 21, scratch c77 emulation / c78 implementation, mini m0-23, base = cn71):**
the last layer consumes D3 only through the Edgeworth term of the mean, `dE[relu z] = -kappa3 alpha phi(alpha) / (6 sigma^2)`,
so rows far in either tail need little of it. Emulation = compute everything, zero D3 on the dropped rows.

| rows without D3 | dMSE | note |
|---|---|---|
| Phi outside [0.01, 0.99] (~50% of rows) | +31.9% (se 1.3) | far too many |
| Phi outside [0.001, 0.999] (~36%) | +0.75% (se 0.11); low side only +0.32%, high side only +0.42% | both tails cost the same |
| the same, star feature of the k34 step also zeroed there (what a restricted build computes) | +0.78% (se 0.11) | star costs nothing in the tails |
| lowest 36% by sigma abs(alpha) phi(alpha), star zeroed | +1.29% (se 0.29) | worse: drops rows near alpha = 0, where the k34 star (kappa4) input matters |
| dropped rows keep the newborn source's own D3 share (full-height legs, O(n^2)) | m0: recovers ~12% of the move | not worth it (old sources dominate) |

The implementation (`ROWF`, `ROWMODE`; `scripts/rowf/patch_c78.py` on cn71) ranks the final rows by min(Phi, 1 - Phi)
from W C (formed first), keeps the top `ceil((1 - ROWF) n / 32) * 32`, and runs the young / tier-1 / tier-2 transports,
the thin legs and the D3 slices on those rows only (row-restricted LeftTrees, leaf block 16 so all levels stay).
The newborn A leg stays full height for the k34 Stein features. It reproduces the emulation bit for bit (m0, max abs 0.0)
and with `ROWF=0` reproduces cn71 bit for bit.

| ROWF (rows dropped) | dMSE (mini) | C/B (0.12.3, seed 1) | net |
|---|---|---|---|
| 0 (cn71) | 0 | 0.19145 | |
| 0.27 (256) | **+0.06% (se 0.02), 100 nets, halves +0.06 / +0.06** | **0.18970 (-0.91%)** | **~-0.85%** -> cn78 |
| 0.31 (288) | +0.27% (se 0.10), 24 nets | ~-1.03% | ~-0.8% |
| 0.36 (352) | +0.78% (emulation, tau 0.001) | 0.18904 (-1.26%) | ~-0.6% |

**cn78** (`candidates/v5b_cn78`, ROWF 0.27 band): real Strassen m0 finite, deviation from fast mode as cn71's (5.7e-6);
VmHWM 4.24 GB; emulator C/B 0.1905 (cn71 0.1922), residual 1.037 s (cn71 1.039 s), smoke shapes finite; 43,422 round
trips (cn71 43,279, +0.3%). Predicted score 3.3183e-9 x (1 + 0.85 x 0.0006) x 0.18970 / 0.19145 = **3.290e-9**.
Submitted Oct 21 16:10 UTC as **337555: graded 3.2888e-9, clean** (raw 1.7301e-8 = +0.02% vs cn71, C/B 0.1901 =
-0.89%; the prediction was off by 0.04%). `candidates/v5b_cn78r` (4-level twin: bill 0.20247, emulator C/B 0.2033,
residual 1.166 s vs cn71r 1.116 s measured back to back, smoke finite, real Strassen m0 1.6043e-8) is **337564:
graded 3.5089e-9, clean** (raw 1.7295e-8; cn71r 3.5407e-9, -0.90%).

**Scale-mode surrogates for the dropped rows (cn79, Oct 21 evening).** A probe of the final layer (m0-m2, ROWF 0):
the standardized skew `g3 = D3 / sigma^3` is almost a function of alpha, **corr(g3, alpha) = 0.95**, `g3 ~ 0.036 alpha`
for |alpha| > 1 with a within-bin spread of only ~0.035. This is the coherent norm mode of a bias-free net
(`z = (1 + delta) y`; to first order `kappa3(z) = kappa3(y) + 6 eps mu sigma^2`, eps = Var delta ~ 0.006 here, far above the
input's 1/(2n)). By source group, the final D3 of the tier-2 / tier-1 / older young / newborn sources is 88% / 87% / 75-80%
/ 16-20% scale mode. The k34 star feature is almost constant: `star / sigma^4 = -0.014 +- 0.003`, uncorrelated with alpha.
So zeroing the dropped rows put both k34 inputs off their manifold (that is what blew up the 50% drop). cn79 fills
them instead: `D3 = c mu sigma^2` with c least-squares fitted on the kept rows, and `star = mean(star / sigma^4) sigma^4`
(`ROWFILL` 1 = D3, 2 = D3 + star; `scripts/rowf/patch_c79.py`). Mini m0-23 vs cn71 (fast mode):

| rows dropped | no fill | D3 fill | D3 + star fill | C/B |
|---|---|---|---|---|
| 256 (25%, cn78) | +0.06% | | | 0.18970 |
| 352 (34.4%) | +0.78% (emulation) | **-0.04% (se 0.03), 16/24** | +0.01% (10 nets; paired vs D3 fill +0.00%) | 0.18904 |
| 384 (37.5%) | | | +0.18% (se 0.13, 10 nets) | 0.18882 |
| 448 (43.75%) | | | +1.11% (se 0.39, 14 nets) | 0.18838 |
| 512 (50%) | +32% (emulation) | +5.63% | +5.31% (10 nets) | 0.18793 |

The fill's floor is the residual (non-scale) skew of the rows near the cut: the neurons' own skew transported
diagonally, `(W o W o W) kappa3(h_15)`, explains only corr 0.17 of it. Paired vs cn78 on m0-23 the 34.4% fill is
-0.15% (se 0.07) MSE and -0.35% C. **cn79 = cn78 + fill at ROWF 0.36 (672 rows kept).**

**cn79 gate** (`candidates/v5b_cn79`, ROWF 0.36, ROWFILL 2): all 100 mini **-0.06% (se 0.03) vs cn78**, 65/100 better,
halves -0.05 / -0.08 (vs cn71 -0.00%); real Strassen m0 finite (1.6004e-8, deviation from fast mode 5.7e-6 as cn78);
0.12.3 bill **0.18904** (cn78 0.18970), VmHWM 4.18 GB; emulator C/B 0.1898, residual 0.995 s (cn78 1.037 s, both under
load), smoke shapes finite; 43,448 round trips (cn78 43,422). Predicted 3.3183e-9 x 0.18904 / 0.19145 = **3.277e-9**.
Submitted 18:22 UTC as **337588: graded 3.2790e-9, clean** (raw 1.7311e-8, C/B 0.1894; -0.30% vs cn78, -1.18% vs cn71;
the prediction was off by +0.07%). Twin `candidates/v5b_cn79r` (bill 0.20176, emulator C/B 0.2026, residual 0.886 s, smoke
finite, real Strassen m0 1.6040e-8) submitted 19:12 UTC as **337606**.

Closed the same evening:
- **No D21 hub at layer 14** (`XD21ALL=14`, scratch c79): +172% (6 nets); **no old-source D21 at layer 14**: +42%. The
  newborn residual `S21 = K21 - rep21` is a difference of two large terms, and errors in D21 do not cancel in it.
- **Final-layer scale-mode extraction** (evaluate the mean from y's cumulants, `var_y = (var - eps mu^2) / (1 + eps)`,
  `kappa3_y = (D3 - 6 eps mu var_y) / (1 + 3 eps)`, `kappa4_y` by the first-order formula, eps from the D3 fit): +0.82%
  with the corrector, +0.49% chain-only (m0, m1). The chain's own final cumulants are better used as they are.
- **Replacing the tier sources' final D3 by their scale-mode fit**: their residual (0.006 / 0.012 sigma^3) applies to all
  kept rows, est. ~+0.7-1% MSE for ~-0.9% C. Not tried.
- **Corrector ensemble**: the corrector costs 0.43% of C (bill without it 0.18889); a second member would need > 0.5pp
  held-out gain (seed spread 0.15pp).



| variant | result |
|---|---|
| H 192 (200 nets) | -3.32% [-3.44%] |
| S 64 (200 nets) | -2.55% [-2.62%] |
| S2 32 (1000 nets) | -4.53% [-4.72%] |
| backward state | -3.35% [-3.44%] |
| 24 epochs | -4.74% [-4.72%] |
| incoming correction as an input | -1.60% [-1.72%] |
| end-to-end term, normalized per network | -1.05% |
| pooled final-MSE fine-tune [18255's terminal objective] | -4.79% [-4.90%]; training loss falls, test worsens |
| global output scale a | optimum ~0.97 (1.0: -4.90%) |
| per-layer scales, cross-fit | -4.86% [-4.90%] |
| clip prediction at 0 | +-0.0000% |
| v4 inputs (rho^4, signed rho, max rho^2, D21 squared mass, kappa4-slice mass, participation ratios) | -3.50% / -3.69% [-3.58 / -3.73% over two seeds]; training loss unchanged |
| root transform of pk2v/pk3v/pk4v (20-70 sigma tails) | +0.075pp (two seeds) |
| EMA of weights | -4.74% [-4.76%] |
| per-layer input standardisation (Susupro's recipe) | 200 nets -3.85% [-3.73 / -3.58]; **1000 nets -4.76% [-4.76%]** |
| per-layer output scale (targets normalised per layer, same objective) | -3.76% [-3.73 / -3.58] |
| tanh on the carried states | -3.64% |
| incoming-correction inputs `Phi o (W s)`, `W s` | -3.43% (TEST, m0-99) [-3.67 / -3.51] |
| mixed loss 30% per layer + 70% transported (mliston 18170) | **-1.65%** [-3.73%] |
| closure knobs, round 2 (17, incl. the SSC (3,1) and gate-weighted (2,2) terms) | at the optimum, <= 0.03% per knob |
| network context: per-layer means over neurons of the standardized inputs (CTX1) / + pooled carried states (CTX2), 200 nets | CTX1 -3.79%; CTX2 -3.96 / -3.68 / -3.90% over 3 seeds (mean -3.85%) [-3.73 / -3.58%], better-count 85-88/100 [92]; ~+0.1% C for the larger first layer, so ~0 net |

Seed noise: about 0.15pp at 200 training networks.

## 7. Open levers (ranked by expected value per day)

1. **Corrector trained on all 1,098 labelled networks** (adds the 98 mini).
   - Status: built as `candidates/v5b_cn72` (cn71 code + this corrector) and graded 3.3218e-9 (+0.10% vs cn71,
     noise). Closed.
   - Expected: about -0.07pp MSE, by the corrector's measured learning curve (-0.3..-0.5pp per doubling).
   - Catch: no held-out set remains, so the grade is the validation.
2. **Fewer client round trips.**
   - Method: batch same-shape product pairs over a leading axis of 2, and pair the Strassen combinations
     (forum 18255: 28.6k -> 24.4k ops cut the official residual median from 0.384 to 0.339 s).
   - Value: score-neutral. It buys residual margin, so a 5-level build could replace the 4-level twin as the second
     nomination.
3. **Fine-tune k34 + MP on exact labels.**
   - Why: the team trained them on the v5b chain's states, and ours has drifted (tiers, masks, rank-2 feedback).
   - Needs a differentiable replica of the chain. Never attempted, so the gain is unknown.
   - Every closure change must also go after these learned steps.
4. **Restructure D21 as `M W^T`** (as in 18252's v31/v33).
   - Would make E1 (~-1.1%) and the "edge history" (-10% raw at higher cost in 18252) cheap.
   - For us it costs one dense product per layer unless the hub can run on pre-transport legs without a separate
     transport. Multi-day, uncertain.
5. **The leaders' structural gap.** Top teams run raw 1.1-1.36e-8 at C/B 0.107-0.128, against our 1.73e-8 at 0.192.
   - Their old tier has no dense legs, and they use cheaper exact pieces.
   - About 76% of our bill is young-source hubs and transports, so that is where structural work would have to land.
6. **More corrector data from our own labels.**
   - 2^25 samples per network (validated against the 1e9 truth in 18255) is ~1e15 FLOPs per network.
   - Out of reach on CPU here.

## 8. Validation gate (every build, in this order; tools in `scripts/gate`, `scripts/hf`, `scripts/corr2`)

| step | tool | pass |
|---|---|---|
| 1. held-out accuracy (torch) | `scripts/corr2/eval_rnn.py <model> ld61c_v3 scripts/corr2/keys_m98.txt` | better than the base after the transfer factor: graded MSE ratio ~ 1 + 0.85 d_local |
| 2. torch vs chain parity, m50 | `scripts/corr2/gate_corr.sh` (step 1) | max abs diff <= ~3e-6 |
| 3. in-chain fast mode, 100 mini | `scripts/hf/gohf.sh` + `scripts/hf/eval_hf.py ld61 <base> <tag>` | pooled dMSE with se and halves; consistent with step 1 |
| 4. real Strassen kernels, m0 | `scripts/gate/runstr_hf.py` (0.12.3 venv) | finite; close to fast mode |
| 5. 0.12.3 bill and memory | `scripts/gate/runv.py <dir> <tag> 1` | C/B as predicted; VmHWM <= ~4.2 GB |
| 6. client/server emulator + smoke shapes | `scripts/gate/gate_emu.sh` | suite MLP ok; all 8 smoke shapes finite |
| 7. round trips | `scripts/gate/run_cnt.sh` | <= 43.1k per MLP for a 5-level build (cn68/cn69 level) |
| 8. submit | `scripts/package.sh` + `whest submit` | the 5-level build alone; the 4-level twin after it grades clean |

Score prediction used so far: `score_new = score_base x (1 + 0.85 d_local) / (1 + 0.85 d_base) x C_new / C_base`.
For cn69 it gave 3.316e-9 against a graded 3.3201e-9 (+0.12%).

## 9. File map

| path | what |
|---|---|
| `candidates/v5b_cn69/estimator.py` | the estimator (1,855 lines). Knobs l.450-490 and l.610-620, `_predict_core` l.674-1592 (corrector l.1286), `_k34_step` l.1608, `_mp_step` l.1669, `_dslices` l.1710 |
| `candidates/v5b_cn69/strassen4.py` | batched Winograd-form Strassen kernel (`LeftTree`, `mm`) with lazy slab views |
| `candidates/v5b_cn69/k34.npz`, `mp.npz` | the team's learned kappa3/kappa4 operators (v5b submission) |
| `candidates/v5b_cn69/corr.npz` | our corrector (mu/sd 68, W0..W3, b0..b3, `relu`, `nS2` = 8, `v3`, ysd) |
| `candidates/v5b_cn69r/` | the same with `SLEV = 4` (robust twin) |
| `scripts/hf/` | labels (`hf_labels.py`), weight regeneration (`regen.py`), fast-mode runners (`runfast_hf.py`, `runfast_hf_ldump.py`, `gohf.sh`), paired MSE (`eval_hf.py`), dump recorder (`recorder_xs_d3/`) |
| `scripts/corr/prep_v3.py` | dumps -> per-network training arrays (X, C, E, Phi) |
| `scripts/corr2/` | corrector training, eval, export, build and gate (`fitrnn.py`, `eval_rnn.py`, `torch_pred_rnn.py`, `export_rnn.py`, `build_cn66.sh`, `gate_corr.sh`, `keys_m98.txt`) |
| `scripts/gate/` | real Strassen, billing, emulator, round trips, profiler, E1 screen, submission poller (see its README) |
| `docs/final_phase_2026-10-17.md` | the experiment log of Oct 17-20 (sections 6-8) |
| `docs/PROGRESS_2026-10-19.md` | status and plan to the deadline |

Reproducing the corrector from scratch:
1. Labels: run `hf_labels.py` for mini and full.
2. Dumps: run `runfast_hf_ldump.py` with `recorder_xs_d3`. Copy `k34.npz` and `mp.npz` from cn69, and set
   `XMARG2=1 CN_FBOS=8 CM_TAU=0.01 RW_TAU=0.01 CN_RRES=6 XAGE6FROM=14`: the recorder's own defaults are older, and only
   these settings give cn69's chain (see `scripts/hf/queue_ld3c.sh`).
3. Prep: `prep_v3.py ld61c <ids>`.
4. Train: `fitrnn.py` as in Section 5.
5. Export: `export_rnn.py`.
6. Build: `SRC=c68 build_cn66.sh` (the source directory must hold cn68's `estimator.py`, `strassen4.py`, `k34.npz`,
   `mp.npz` and `LICENSE_504aldo`).
7. Gate: Section 8.
