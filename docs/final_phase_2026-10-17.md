# Final phase (Oct 17 evening -> Oct 21 close): situation, strategy, exact cost cuts

## 1. What changed on Oct 17 (forum 18251, 08:28 UTC)

* Evaluators moved to **flopscope v0.12.3** (fixes undercounted broadcast work and truncated-SVD costs; affected
  submissions are re-evaluated).
* Submissions close **21 Oct 2026, 23:59 UTC**. Nominations for the private re-evaluation close **22 Oct, 23:59 UTC**:
  Submissions page -> "Your team" -> "Select", up to two. With no nomination the two best public scores are used.
* Prize ranking comes **only** from the private re-evaluation: fresh MLPs from the Phase 2 distribution, private seeds,
  flopscope v0.12.3. Algorithmic write-ups are due 28 Oct.

**Our bill under v0.12.3 is unchanged.** Same code, 0.12.3 server/client emulator, MLP 1000: cn37 0.1953, cn49b 0.1948,
cn50 0.1956, cn36 0.1972 (the 0.12.0 emulator gave the same numbers); in-process MLP 5000 cn50 0.1944 both ways. The
poll API still returns the original grades (cn49b 3.5121e-9, cn50 3.5175e-9).

**Leaderboard quirk.** The board ranks and displays scores at two significant figures and keeps the *first*
submission that reached the displayed value: our row still shows 336607 (cn36, the first 3.5e-9), not cn49b. A
better entry only shows once it rounds lower (< 3.45e-9). This is display only; the private ranking is what counts.

Board (Oct 17 16:16 UTC, 150 teams): #1 1.3e-9 (raw 1.10e-8, C/B 0.122), #10 1.9e-9, the ten e30_a-engine teams at
3.1e-9 (raw 2.03-2.06e-8, 0.1517), us #54 at 3.5e-9 (raw 1.81e-8, 0.195). Our raw error is the best in our band; our
cost is the highest.

## 2. Strategy for the last four days

1. **Exact cost cuts first.** The score is MSE x C/B. An accuracy change shows ~1/3 of its local gain on the board
   (section 12 of the Oct 17 log) and carries selection risk; an exact cost cut (same predictions up to rounding)
   carries 1:1 to the private set. So every remaining exact saving is worth a slot.
2. **Never risk a failed private run.** One network over 120 s of wall zeroes the entry. Nominate only builds that
   graded clean on the public set, and keep their emulator wall at or below the wall of builds that graded clean.
   Memory stays under ~4.5 GB VmHWM (6.1 GB passes, 7.0 GB failed).
3. **Accuracy knobs only with a 75-network all-halves-consistent gain of >= ~3%.** Everything tunable is at its
   optimum (tiers, masks, thin legs, learned-operator gains); the move-squared rule explains why cuts lose.
4. **Nominate the two best expected private scores.** Our candidates are siblings (same chain), so diversification
   is worth little; the two best graded builds are the right pair.

## 3. Exact cost cuts found today (flopscope billing, values unchanged up to float32 rounding)

Measured with an op-level profile under v0.12.3 (`/tmp/claude-0/acc123/prof123.py`):

* **`fnp.reshape` is billed at one FLOP per element** (indexing, `x[None]`, `swapaxes`, `broadcast_to` and
  `fnp.empty` are free). The Strassen buffer pool reshaped a view of one flat buffer for every new shape: ~0.9% of
  C/B. strassen4 keeps one max-shape slab per role (split only where a slab would pass the 134.9 MB per-array cap)
  and hands out free slices. (Fresh arrays per shape bill the same but churn ~34 GB of server memory per network:
  +13% emulator wall, rejected.)
* **Strassen-Winograd form**: 15 additions per level instead of 18 (the batched levels save one output addition,
  the fused leaf three): -0.46% C/B.
* **The nonlinear term program** (`"tij,ti,tj,tg->gij"` and `"ij,ti,tj,tg->gij"` einsums) bills 6.8-8.3 n^2 per
  term; `diag(u) M diag(v)` accumulated elementwise bills 3 n^2 (a term product shared by consecutive terms is
  formed once): -0.9% C/B, and -12% in-process wall.
* Checked and **not** worth anything: the hub's batch sum (billed like an add already), the remaining 3-operand
  einsums (billed at matmul rates), merging mirrored (1,1)/(2,2) terms (the tables list each mirror pair once).

| build | base | change | C/B MLP 5000 in-process | emulator C/B (MLP 1000) | emulator wall | VmHWM | final-layer move vs base (rms) |
|---|---|---|---|---|---|---|---|
| cn50 | | | 0.19436 | 0.1956 | 188-197 s | 3.43 GB | |
| cn52 | cn50 | Winograd + fresh arrays | 0.19139 | 0.1926 | **214 s** | 3.79 GB | 1.4e-6 |
| cn52w | cn50 | Winograd only | | 0.1947 | 185 s | | 1.4e-6 |
| cn53 | cn50 | Winograd + slab pool | 0.19148 (-1.48%) | 0.1927 | 196 / 191 s | 4.29 GB | 1.4e-6 |
| cn53b | cn49b | Winograd + slab pool | 0.19028 (-1.45%) | 0.1919 | 190 s | 4.29 GB | 1.3e-6 |
| **cn54** | cn53 | + elementwise term program | **0.18976 (-2.37% vs cn50)** | **0.1910** | 187 s | 4.2 GB | 1.5e-6 |
| **cn54b** | cn53b | + elementwise term program | | **0.1901 (-2.4% vs cn49b)** | 179 s | | |

All four pass the four-shape smoke test (finite at 64x8, 256x32, 128x16, 1024x4). A final-layer move of 1.5e-6 rms is
~1e-4 of the MSE (rms error ~1.3e-4): the predictions are the same.

Expected grades: cn54b ~3.43e-9 (cn49b 3.5121e-9 x 0.976), cn54 ~3.43e-9 (cn50 3.5175e-9 x 0.976).

## 4. Slot plan

* The limit is **5 submissions per UTC day** (the sixth on Oct 17 was refused: "Submission limit reached ... resets at
  00:00 UTC"), so 20 slots remain for Oct 18-21.
* Oct 17: 336664 (cn49b), 336695 (cn50), **336767 (cn53b), 336768 (cn53), 336770 (cn54b)**; cn54 goes first on Oct 18.
* Oct 18-21: only exact cuts or 75-network gains of >= ~3%; keep >= 3 slots for the last day.
* Nomination (by Oct 22 23:59 UTC, on the website): the two best graded among cn54b / cn54 (fall back to
  cn53b / cn53, then cn49b / cn50).

## 5. Blocked: the public "mini" split with 1e9-sample truth

The starter kit's public dataset (`hf://aicrowd/arc-whestbench-public-2026@v2-phase2`, split `mini`, 100 MLPs with
baked N = 1e9 ground truth) would add 100 networks with essentially exact labels to the local gate (our labels carry
MC noise ~1.2-1.9e-7 per network, about 8x the error). huggingface.co is blocked by this environment's network
policy (proxy 403). It needs `huggingface.co` in the environment's allowed domains.

## 6. What the forum write-ups say about cheaper and better chains, and what we tested (Oct 17 evening)

How the cheaper teams got there, as far as the public record shows:

* **504aldo (18218, our base's author):** the leaders' bill equals V29 minus its old tier; "whatever carries their
  old-source content costs them ~nothing". The grader runs ~50 networks per process (1 cold + 49 steady), on one
  physical core. ngmitam (same thread) shipped V29 with Strassen leaves of 8 (7 levels) on 100 networks with 0
  failures: -2.9% C locally, -0.9% on the grader (amortisation of first-network costs, the same effect that halved
  our reshape saving on the grader).
* **Artyom_Mazur (18249, PDF):** grader wall 91 s median / 98.8 s max for his chain against the 120 s cap; the
  leaders spend 77-95 s. One more Strassen level everywhere doubles his wall, so he deepened only the five sites
  with the best "C saved per second of wall". His largest late gains were *consistency cuts* (remove a term on both
  sides or not at all): matched dead-row + dead-column cuts, saturated rows of D removed with the kappa4 level they
  carried restored by one mat-vec (E2, x0.943 raw), and marg2.
* **il.miscusi (18248, PDF):** e30_a (V29 descendant at C/B 0.152) uses the **per-neuron metric diag(W W^T)** in
  place of the constant 2 in the kappa4 diagonal, the (2,2) slice, the (3,1) slot and the kappa4 -> kappa3 feed
  (x0.981 on the grader with two other changes). In ARC's lean AUGMENT chain without scale-mode extraction the true
  metric is worth 18% of raw error.

| idea (source) | test | result | verdict |
|---|---|---|---|
| D21 hub on live rows only (dead rows out; our own) | zero D21 rows with Phi < 0.002 / 0.012, six hard networks | +7.9% / +49% | out: dead rows carry real content (wick tails with alpha^k, the MP edge channel) |
| dead rows + columns together | Phi < 0.002 | +55% | out (matches the earlier +28-84%) |
| dead columns only | Phi < 0.002 | six hard networks -1.4% +- 2.3%; **75 networks +8.88% +- 2.68% (all splits +)**, move 3.7e-5 | out: the move-squared rule predicted ~+8.5%; the hard-six reading was the selection effect |
| E2, saturated rows of D21 (Artyom) | Phi > 0.994 / 0.998, naive | +14.6% / +3.9% | out: in our chain those rows reach kappa4 through K22 row sums of Hadamard products (no cheap restoration) and feed the MP edges |
| per-neuron metric (e30_a) | `XMET` (diag, (2,2), feed), `XMET431` (+ (3,1)), 75 networks | +0.06% +- 1.35% / +0.03% +- 1.35% (halves disagree) | null: our learned operator already absorbs it; cn56 not shipped |
| (1/4) a3 b3 C0 o K22 diagram (Artyom) | | already in our term table (`('c_off', 'wk4m', 8, 8, 0.25)`) | -- |

### 6.1 Literature and web search (Oct 17 evening)

* **Sponsor paper and code** (arXiv 2605.05179, `alignment-research-center/mlp_cumulant_propagation`): the repo also
  has a factorized K = 4 chain (`factor_k4.py`); 504aldo measured per-source kappa4 legs at >= 0.95 x B, out of reach.
  504aldo's reply on 18246 (Oct 17): V29's (and so our) term tables come from the reference at commit 6e80f7f, before
  the Jul 31 Edgeworth-truncation change (paper eq. 13-14). il.miscusi measured that a lambda-law chain is *better*
  on the old program (2.200e-8 vs 2.395e-8 with the current tables), so switching tables would hurt us.
* **Fast matrix multiplication:** rational <4,4,4;48> (Dumas, Pernet, Sedoglavic, arXiv 2506.13242; accurate
  variant 2603.18699; "into practice" 2609.12027) saves 48/49 per two Strassen levels (-4.0% of leaf products when it
  replaces four of our five levels). Only the alternative-basis forms keep the additions low, and they need ~48
  separate small ops per level instead of Strassen's 22 batched ones: with hundreds of products per network that
  breaks the 400 ms residual cap. Not pursued.
* **Exact Gaussian moment matching** (Kuang & Lin, arXiv 2601.22307, NeurIPS 2026) and Wright et al. 2024 (AISTATS):
  exact or series covariance through ReLU. Our truncated Mehler terms already sit far below our error at
  correlations of a few percent (C0^3 measured null); our error is the non-Gaussian closure, not the covariance.
* **Statistical-physics corrections** (TAP / Onsager / Plefka, Edgeworth bounds for finite-width networks): all
  averaged over weights (odd cumulants vanish) or for other models; no quenched single-network method found,
  matching 504aldo's triage (18218 dead end #24).
* **Not reachable from this environment:** arxiv.org, alignment.org, lesswrong / greaterwrong and huggingface.co are
  blocked by the network policy (proxy 403); only search snippets were available for them.

### 6.2 The public dataset is reachable now (huggingface.co enabled)

`aicrowd/arc-whestbench-public-2026@v2-phase2` has a `mini` split (100 MLPs) and a `full` split (1,000 MLPs, disjoint
seeds), each with **1e9-sample means at every layer**. Only the label columns are read (pyarrow over HTTP range
requests, ~1 MB per shard; `/tmp/claude-0/hfwork/hf_labels.py`); weights are regenerated from `mlp_seed` with
whestbench's own bake recipe (`derive_seed_streams(seed)[0]` -> `sample_mlp`), checked against the truth by a
2e4-sample forward pass (final-layer rms difference 1.8e-3 = the MC noise). Tools: `runfast_hf.py`, `gohf.sh`,
`sc/eval_hf.py`, `sc/jfit_hf.py`. Our own 5e5-sample labelling was stopped (redundant).

First reading, all layers of mini network 0 (cn50 settings, fast mode): the final-layer raw MSE is 1.806e-8 (the
leaderboard reads 1.81e-8), and the error **accumulates steadily from layer 2 on** (1.0, 1.8, 2.7, 3.9, 5.0, 6.2,
7.0, 8.8, 9.9, 11.0, 12.9, 15.2, 15.8, 18.1e-9 at layers 2-15, label noise 0.1-0.7e-9) with a negligible mean bias.
With our 5e5-sample labels the early layers had looked error-free. Every layer's closure matters, which is what a
per-layer recalibration on exact labels can address (queued: linear-response runs for k34 / MP / lambda gains by layer
group on the mini split, validated on the full split).

### 6.3 Where the error is created (exact labels, 6 mini networks, cn50 settings, `scripts/hf/attr.py`)

Split of each layer's error e_l = pred - truth into the part propagated from the previous layer through the
linearised map (diag(Phi_l) W_l^T e_{l-1}) and the part created at the layer:

| layer | 1 | 2 | 3 | 5 | 8 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MSE (e-9) | 0.68 | 0.94 | 1.60 | 3.54 | 7.08 | 9.59 | 11.0 | 13.0 | 15.1 | 17.3 | **20.1** |
| propagated | 0.42 | 0.44 | 0.66 | 2.02 | 4.82 | 6.78 | 8.05 | 9.84 | 11.7 | 13.3 | 15.4 |
| created | 0.25 | 0.52 | 0.88 | 1.52 | 2.24 | 2.80 | 3.04 | 3.12 | 3.50 | 3.89 | **4.40** |

(Layers 0-3 include the label noise, var_l / 1e9 = 0.3-0.7e-9.) The propagation contracts the previous error by
0.8-0.9 per layer, the created part is uncorrelated with it (cross terms ~0), and the creation grows steadily with
depth: the final layer creates 22% of the final MSE and the last three about half. No single step is faulty; the
closure loses a little more at every layer as the cumulants grow.

| further idea tested (fast mode) | result | verdict |
|---|---|---|
| Artyom's young kappa4 "3+2" tree read into the kappa4 diagonal (12 sum_j W_aj (d2/d1)_j (W D1 C)_aj X(a,j), X = the young hub paired with the pre-transport legs; factor 12 re-derived) | before k34: -0.27% +- 0.81% (six hard networks, move 1.1e-5); after the MP: +3.5% +- 2.8% | out: k34 re-corrects it; our learned operator already carries the young kappa4 content his chain lacked |
| per-neuron metric (3,1) slot too (`XMET431`) | +0.03% +- 1.35% (75 networks) | null |

### 6.4 Zero-cost knobs on the public mini split (100 networks, exact labels, cn50 settings)

| knob | dMSE | halves | cost | net |
|---|---|---|---|---|
| MP from layer 7 | -0.43% +- 0.20% (59/100) | -0.55 / -0.31 | +0.29% C | -0.14% |
| BETA 2.0 | +0.02% +- 0.06% | | 0 | null |
| BETA 1.5 | -0.04% +- 0.03% | | 0 | null |

The exact-label gate resolves small moves to 0.03-0.2% (the 75-network gate: 0.2-1.3%). BETA 1.5's -0.19% on our own
labels and MP7's -0.99% were mostly noise; both knobs are essentially flat. Mini's base raw MSE is 1.90e-8 (board
1.81e-8).

### 6.5 Forum 18250 (DD_team, "exact compute reductions") checked against our chain

Their two exact cuts do not transfer. B1b (skip the first identity transport, W.T @ I = W.T) is what our newborn P leg
already does: it is W itself (copied or column-permuted), never a product. B1a (store one transported block for two
legs that differ only by birth column coefficients, tree-A = I diag(q2) vs B = I) needs two legs that are
column-scalings of each other; our newborn legs are W C_off and W, which are not, and older P legs are distinct suffix
products. Their K3-history ablation (keep only the newest H generations: 12-17x worse) agrees with our closed
"drop old sources" yardstick.

## 7. Idea sweep on exact public labels (Oct 17 night)

| idea | test | result | verdict |
|---|---|---|---|
| half-precision billing | flopscope 0.12.3, 256^3 matmul | float16 billed 2 n^3, same as float32 (float64 4 n^3) | closed |
| final-layer ridge mean correction (the shipped-off 13-feature delta, and a 23-feature set) | fit on mini m0-49, test m50-99 and swapped | held out -0.02..-0.09% (se 0.05-0.12%); even in-sample only -0.24% | closed |
| blend of corrected / raw corrector output (D3 vs D3raw, g4 vs g4raw) | fit f0-99 test f100-199 and swapped | -0.00% / +0.01% (se 0.01%) | closed: corrector strength already optimal |
| 23-feature ridge on 200 full nets | test mini m0-99 | -0.04% (se 0.03%) | closed |
| ceiling of any post-hoc final-layer fix | per-network oracle fits on 100 mini nets | common offset 0.4% of MSE, oracle scale 1.2%, oracle 5-feature per-network fit 1.6% | the final error is neuron-specific closure error, not bias |

Dump hook: `scripts/hf/runfast_hf_dump.py` (saves the last layer's per-neuron state); fit: `scripts/hf/fit_corr.py`.
So the remaining levers are inside the chain (truncation knobs, closure), not on its output.

Billing of single knob steps (Strassen build cn55b + rank-schedule hook, seed 1, base C/B 0.18886):

| knob | C/B | dC |
|---|---|---|
| R_OLD 288 / 352 | 0.18715 / 0.19390 | -0.91% / +2.67% |
| R_OLD2 128 / 192 | 0.18757 / 0.19088 | -0.68% / +1.07% |
| AGE_OLD 4 / 6 | 0.17922 / 0.19656 | -5.10% / +4.08% |
| AGE_OLD2 8 / 10 | 0.18666 / 0.19070 | -1.16% / +0.97% |
| CM_TAU 0.005 / 0.02 | 0.19115 / 0.18618 | +1.21% / -1.42% |
| rank slope +32 | 0.19055 | +0.89% |

**Truncation share (exact labels, mini m0-23, cn50 settings).** Turning off all old-source confinement (V21_NO_CONFINE=1)
and both dead-neuron masks (CM_TAU = RW_TAU = 0) lowers final MSE by only **-4.47% +- 0.61%** (22/24 better, halves
-4.28 / -4.65). All of our cost-side truncation together costs ~4.5% of the MSE; the gap to the 1.1-1.35e-8 raw band
of the leaders (30-40%) is closure error. Cost knobs can only pay where their marginal damage stays well under their
compute saving; accuracy has to come from the closure.

**Closure bet (step 3) deprioritized.** The (3,1) laws were already tested on this chain and hurt (+1.4..+4.0%;
brainstorm doc row 1). A Monte Carlo oracle for exact last-layer moments would need about 2.5e7 samples per network
just to resolve the mean to the 1e-4 level that matters (about 5 h per network on one core). The layer split it would
give is already in section 6.3: the final layer creates 22% of the MSE, the rest is propagated. The remaining days go
to the frontier tune (cn58) on exact labels.

Submitted Oct 18 02:05 UTC: 336850 cn54, 336851 cn55b, 336852 cn55.

### 7.1 Knob steps at cn55b settings (exact labels, 100 mini networks, vs k55_base)

| knob step | dMSE (halves) | dC | net | verdict |
|---|---|---|---|---|
| AGE_OLD 4 | +10.56% +- 0.55 (52 nets; 0/52 better) | -5.10% | +5.5% | out |
| CM_TAU 0.02 | +3.70% +- 0.26 (+3.81 / +3.59) | -1.42% | +2.3% | out |
| R_OLD 288 | +2.75% +- 0.24 (+2.71 / +2.79) | -0.91% | +1.8% | out |
| AGE_OLD2 8 | +1.75% +- 0.17 (96 nets) | -1.16% | +0.6% | out |
| R_OLD2 128 | +1.58% +- 0.12 (+1.77 / +1.38) | -0.68% | +0.9% | out |
| cn50 masks (0.006/0.012) instead of cn55b's | -0.50% +- 0.14 | +0.64% | +0.14% | cn55b's masks stay |

Every cheapening step loses, at 1.5-3.0% MSE per 1% C. cn55b sits on the cheap side of the knob optimum. The
expensive steps (AGE_OLD 6, R_OLD 352, AGE_OLD2 10, masks 0.005) are queued; their ceiling is the 4.5% truncation
share.

### 7.2 External sources re-read (Oct 18)

- **18248 (il.miscusi), the leaders' recipe.** The carried kappa4 r=1 core plus scale-mode extraction: fit eps per
  layer from the kappa3 (i,i,j) slice, then remove eps m m^T from C, 2 eps Sym(C x m) from kappa3 and
  4 eps Pair3(C x C) from kappa4. It is worth x0.38 inside a carried chain. It *hurts* memoryless lambda chains. The
  gain needs ARC's current term program. Their carried build bills 0.301 vs 0.152 and scored worse (5.18e-9 vs
  3.06e-9).
- **Sponsor paper 2605.05179 v2, section 4.2 / S.4.2 (augmented algorithm).** It tracks the kappa4 harmonic parts
  that propagate in O(n^K), which improves the MSE constant at equal leading FLOPs. The factored version lives in the
  ARC code. The off-diagonal r=1 core's birth is a partial-trace contraction of the post-ReLU kappa4. Our chain only
  forms the post-ReLU *diagonal* core (K(4,) + K(2,2) row sums) and regenerates the off-diagonal one as lambda C_off.
- **18219:** compute below the 0.1 floor buys nothing (we are at 0.19, so cost cuts still count). **18200:** a blend
  with Monte Carlo needs comparable errors; final-layer var 0.075 means about 4e6 samples (about 60 B) to match
  1.9e-8, so it is dead. **18216:** fitted constants absorb accumulated drift, not one-step skew; this matches our null
  ridge result.

**E4 (cheap carried kappa4 core) closed without a build.** Doing it honestly needs three things: the AUGMENT birth
rule (a post-ReLU partial-trace contraction that is not in our term program), ARC's current term table (measured
worse in our lambda chain), and a retrained learned k34/MP operator, whose inputs would change. 18248's own carried
build lost on score at twice the cost. That cannot be built and validated before Oct 21.

## 8. Oct 18 morning: grades, failures, leaderboard, weaknesses

**Grades.** 336850 cn54 **3.4588e-9** (raw 1.8112e-8, C/B 0.1910). 336851 cn55b and 336852 cn55 **failed** ("Error
while scoring"). Both failures are the Gram-core builds (`_gram2`: `einsum("ik,jk->ij", G, G)`, a SymmetricTensor
result). The only non-Gram build of the wave graded. Not reproduced locally:
- cn55b runs clean on a fresh **0.12.3** client/server emulator: suite shape wall 201.6 s, residual 1.22 s; control
  cn54b 215.6 s / 1.37 s on the same loaded machine.
- All non-suite shapes (4x2 ... 256x4) are finite on 0.12.3 client/server and in process.
- The Gram weights are positive (min 9 / 1), so sqrt cannot make NaN.

The cause is unknown (grader-side handling of the same-operand symmetric einsum is the prime suspect; the chain's
own F45 note says repeated same-object operands once crashed flopscope). **Decision: no `_gram2` in any further
build.** Its -0.6% is not worth a third failed slot.

**Version slip found.** `/tmp/claude-0/venv` is flopscope **0.12.1**. The Oct 17 night billing sweep (section 7
table) ran on it. The grader bills 0.12.3 (`/tmp/claude-0/venv123`). Every dC that drives a build decision is
re-measured on 0.12.3 before use. The accuracy runs (fast mode) are unaffected.

**Leaderboard (Oct 18 03:00 UTC, top 30).**

| team | score | raw | C/B |
|---|---|---|---|
| marius_binner | 1.3e-9 | 1.10e-8 | 0.122 |
| J2W | 1.4e-9 | 1.36e-8 | 0.107 |
| oliver_chen / mliston / Luna | 1.5e-9 | 1.20-1.35e-8 | 0.115-0.129 |
| AndreasHad04 | 1.6e-9 | 1.33e-8 | 0.123 |
| lode_dockx | 2.0e-9 | **1.97e-8** | **0.104** |
| Puffi | 2.0e-9 | 1.24e-8 | 0.164 |
| a_s6 / oqaris | 2.5-2.6e-9 | 2.0e-8 | 0.126-0.131 |
| **us (cn54b)** | 3.45e-9 | **1.82e-8** | **0.190** |

Our raw error is in the top-30 band, but our cost is the highest of anyone within 2x of our score. Teams at our
accuracy or worse run at 0.10-0.13. **The weakness is cost, by about 1.6x, not accuracy.**

**Where our 0.19 goes (0.12.3 profile, cn54).**
- young D21 hub `_hub2` over the dense young sources: **31.5%**;
- young leg transports: 23%;
- old tiers (formings, rotations, inner hubs): about 20%;
- the newborn: 7.5%;
- C_pre: 3.6%.

The hub is the honest kappa3 (i,i,j) contraction: LA = 2 (A o P) w2 + (P o P) e enters as Hadamard products. So
it has no carried-Gram form, and low-rank D21, older-young D21 drops and dropping old sources (x4 MSE) were all
closed by measurement before.

**The cost frontier on exact labels.** Expensive-side steps all buy accuracy at about 0.6% MSE per 1% C:

| step | dMSE | dC |
|---|---|---|
| AGE_OLD 6 | -2.54% | +4.08% |
| R_OLD 352 | -1.69% | +2.67% |

The cheap-side steps cost 1.5-3.0% MSE per 1% C. The exception is **RW_TAU** (dead rows dropped from the next
layer's transports):

| step | dMSE | dC | net |
|---|---|---|---|
| RW_TAU 0.02 | +0.60% +- 0.17 | -0.94% | **-0.34%** |

That clears the -0.3% gate. And the row drop is capped: rows go in steps of 32 with at most n - CN_MIN = 256
dropped, while deep layers already have about 252 rows with Phi < 0.01 (280 / 312 at 0.02 / 0.04). So the cap
binds from RW_TAU ~0.02 on. New knob `XRMIN` (a separate row floor, default 768) is in the scratch builds; RW_TAU
0.03 / 0.04 and floor 640 are queued (billing on 0.12.3 and accuracy).

**Row-mask frontier (Oct 18, 0.12.3 billing on cn55b's chain; accuracy on 100 mini, exact labels).**

| RW_TAU | C/B | dC | dMSE | net |
|---|---|---|---|---|
| 0.01 (cn54b) | 0.18886 | -- | -- | -- |
| **0.02** | 0.18708 | -0.94% | +0.60% +- 0.17 | **-0.34%** |
| 0.03 | 0.18596 | -1.54% | +2.31% +- 0.26 | +0.77% |
| 0.04 / 0.06 / 0.08 | 0.18499 / 0.18409 / 0.18307 | -2.05 / -2.53 / -3.07% | (not run: the slope past 0.02 is 2.8%/1%) | out |
| 0.04 / 0.06 with row floor 640 | 0.18534 / 0.18437 | more than the 768 floor | -- | out: below 768 rows the Strassen recursion pays more than the rows save |

Other 0.12.3 billing: feedback legs off -1.81%, CN_KEEP all -1.68%, CN_RRES 2 -0.50%, CN_RFB 1 -0.48% (accuracy runs
queued). 0.12.3 bills this chain exactly like 0.12.1 (base 0.18886 in both). Expensive-side closures: AGE_OLD2 10
-0.66% MSE / +0.97% C (net +0.31%), so every expensive step loses.

**cn58** = cn54b with RW_TAU 0.02 (one default changed, no Gram cores). It is not submitted until the 200-network
full-split check and the 0.12.3 emulator run pass.

Cheap-side closures (100 mini, exact labels; dC on 0.12.3):

| step | dMSE | dC | net |
|---|---|---|---|
| feedback legs off | +7.80% +- 0.33 | -1.81% | +6.0% |
| CN_KEEP all | +2.55% +- 0.18 | -1.68% | +0.9% |
| CN_RRES 2 | +3.81% +- 0.30 | -0.50% | +3.3% |
| CN_RFB 1 | +0.49% +- 0.05 | -0.48% | +0.01% |

Only RW_TAU 0.02 pays. cn58 stays a single-knob change.

**Gate for cn58 (RW_TAU 0.02).** Mini: +0.60% +- 0.17. Mini + full f0-199 (300 networks): **+0.46% +- 0.10**
(halves +0.52 / +0.40), so the full split alone is about +0.39%. dC -0.94% (0.12.3). Net about -0.5%; the
prediction is 3.4544e-9 x 0.9952 ~ 3.437e-9. The 0.12.3 client/server emulator gives C/B 0.1883 (cn54b 0.1901), and
wall and residual are on a par with cn54b. Submitted 05:51 UTC as **336887**.

**336887 cn58 graded 0.0502 (Oct 18 ~07:00 UTC).** Score equals the raw MSE (0.050185), so the compute factor was
1.0: every network failed and was scored as a zero prediction at full budget. Nothing local reproduces it:
- The real Strassen cn58 on public m0 has MSE 1.837e-8 (cn54b 1.814e-8), finite.
- Smoke shapes (4x2 ... 256x4) are finite on 0.12.3.
- The 0.12.3 client/server emulator runs it clean. On an idle machine its wall is 150.8 / 138.8 s vs cn54b's
  153.7 / 136.3 s (the 120 s grader cap maps to ~175 s here), with residual 1.27 / 1.15 s vs 1.25 / 1.11 s.

Together with this morning's two unexplained "Error while scoring" results (cn55b, cn55, submitted in the same
minute as cn54, which graded), the evidence points to grader-side trouble on Oct 18, not to the RW_TAU change.
**Plan:** hold today's last slot. Retry cn58 unchanged later (if it grades, it is the expected ~3.44e-9; if it fails
again on a healthy grader, it is code-specific and cn58 is dropped). Nominations default to the safe graded pair
cn54b (336770, 3.4544e-9) + cn54 (336850, 3.4588e-9) until a newer build grades cleanly.

**Lesson for the gate:** the fast-mode accuracy runs bypass the Strassen kernels, and the emulator / billing runs
never compared predictions. Every candidate now also gets one real-Strassen run on a public network with labels
(`/tmp/claude-0/acc/runstr_hf.py`) before it is submitted.

**E3, zero-cost scalars (100 mini, exact labels, cn55b settings; one-sided steps).**

| probe | dMSE |
|---|---|
| METRIC_C x1.1 / x0.9 | +10.99% / +8.37% (parabola minimum at x0.993, worth -0.04%) |
| lambda x1.1, layers <=4 / 5-9 / >=10 | -0.07% +- 0.03 / +0.27% +- 0.06 / +0.05% +- 0.04 |
| learned gains x1.2: g3a / g3b / g4a / g4b | +0.05 / +0.12 / +0.04 / **+0.93%** |
| learned gains x1.2: mna / mnb / mp | +0.18 / +0.28 / +0.23% |

The learned operator and the lambda table sit at their optimum to within ~0.1% everywhere except possibly g4b.
g4b x0.85: +1.07% +- 0.19, so g4b also sits at a sharp minimum. **E3 closed: no free gain.**

**Long-process robustness (0.12.3 client/server emulator, idle machine, cn54b, 6 networks in one process).**
Wall 167.4 s for the cold first network, then 150.7 / 144.2 / 144.1 / 147.7 / 145.0 s; residual 1.67 s, then
1.24-1.32 s. There is no growth with the number of networks (forum 18238's op-log accumulation does not show), so
there is about a 15% margin under the ~175 s emulator wall that maps to the grader's 120 s cap.

### 8.1 Targeted spending (Oct 18 midday; 100 mini, exact labels; dC on 0.12.3)

Two leads from the sweep: the residual rank had the steepest cheap-side rate (rank 2: +3.81% MSE for -0.50% C),
and the error created per layer grows 9x with depth (section 6.3). The scratch hooks: `XAGE6FROM=L` (skip the
tier-1 join at layer L so that source stays dense to the end; the tier-2 gate shifts by the same offset),
`XAGE4UNTIL=L` (join one layer earlier up to L), and `XRWLATE=L,tau` (row-mask threshold by depth).

| step | dMSE | dC | net |
|---|---|---|---|
| **CN_RRES 8** | **-1.34% +- 0.23** (71/100; halves -1.49 / -1.20) | +1.00% | **-0.34%** |
| dense window from layer 13 | -0.68% +- 0.07 (82/100) | +0.57% | -0.11% |
| dense window from layer 12 / 10 | (m0 alone: -2.25% for L=10) | +1.55% / +3.04% | dropped: cost grows faster than the gain |
| dense window from layer 14 | queued | +0.10% | |
| join earlier up to layer 8 | m0 +9.05% | -2.89% | out: the early ages matter most |
| CN_RRES 6 / 12 | queued | +0.50% / queued | |
| RW_TAU 0.03 below layer 10, 0.02 from it | queued | -1.46% | |
| RW_TAU 0.01 below layer 10, 0.03 from it | (m0 +0.71%) | -0.52% | |

Costs add: window from 13 + RW_TAU 0.02 bills -0.40% = +0.57 - 0.94. The cn59 candidate is RW_TAU 0.02 + CN_RRES 8,
mini and full split queued.

**cn59 candidate = cn54b + RW_TAU 0.02 + CN_RRES 8** (two defaults; no Gram cores).
- Mini: -0.70% +- 0.29 MSE (64/100; halves -0.86 / -0.55).
- 0.12.3 bill: 0.18896 vs 0.18886 (+0.05%), so the net is about -0.65%.
- Real-Strassen label check on m0 / m1: MSE 1.8217e-8 / 1.8306e-8 vs fast mode 1.8202e-8 / 1.8252e-8 (max
  prediction difference 8e-6), finite.
- Full split f0-199 queued. Rank 12 bills +2.00%.

**Residual rank curve.**

| CN_RRES | dMSE | dC | net |
|---|---|---|---|
| 2 | +3.81% | -0.50% | +3.3% |
| 4 (shipped) | -- | -- | -- |
| **6** | **-0.97% +- 0.18** (72/100) | +0.50% | **-0.47%** |
| 8 | -1.34% +- 0.23 | +1.00% | -0.34% |
| 12 | (not run) | +2.00% | |

So cn59 switches to **RW_TAU 0.02 + CN_RRES 6** (predicted -0.37% MSE at -0.44% C, about -0.8% net).

### 8.2 Where the compute goes vs where the error is created (per layer)

Per-layer bill of cn54b on 0.12.3 (`scripts/hf/perlayer_cost.py`, % of B, total 18.96%), set against the error
created per layer from section 6.3 (exact labels, e-9):

| layer | 1 | 2 | 3 | 5 | 8 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| % of B | 0.29 | 0.50 | 0.72 | 1.15 | 1.52 | 1.64 | 1.72 | 1.69 | 1.75 | 1.74 | **0.75** |
| created error | 0.25 | 0.52 | 0.88 | 1.52 | 2.24 | 2.80 | 3.04 | 3.12 | 3.50 | 3.89 | **4.40** |

The cost ramps with the number of carried sources and flattens at about 1.7% per layer from layer 9 on. The
created error keeps growing. The **final layer creates 22% of the final MSE on 4% of the bill**: it is mean-only
(trimmed), and its error is the closure error of the mean formula (the D3 / kappa4-diagonal inputs and the
truncated Edgeworth series). Layers 0-4 cost 2.5% of B together.

**cn59 gate.**
- Full f0-199: -0.36% +- 0.16 MSE (115/200; halves -0.17 / -0.54).
- 0.12.3 bill: 0.18802 (-0.44%); emulator C/B 0.1893 vs cn54b 0.1901.
- Net about -0.8%, so the prediction is 3.4544e-9 x 0.992 ~ 3.43e-9.
- Submitted 14:54 UTC as **336962**, in today's last slot, in place of the cn58 retry: it tests the same grader
  question with a better expected score.

**Late dense window from layer 14 (`XAGE6FROM=14`).** -0.49% +- 0.05 MSE (81/100; halves -0.55 / -0.43) for +0.10%
C, **net -0.39%**. This is the efficient end of the depth-targeted spend:

| window from layer | dMSE | dC | net |
|---|---|---|---|
| 13 | -0.68% | +0.57% | -0.11% |
| **14** | **-0.49%** | **+0.10%** | **-0.39%** |

So most of the dense-window value sits in the last two layers. The other probes billed on 0.12.3: CN_RRES 5 +0.25%,
full last-layer stream +5.09%, CN_RFB 3 +0.48% (accuracy queued).

**cn60 = cn59 + age gate 6 from layer 14** (code default `CN_AGE6FROM=14`; the tier-2 gate shifts by the same
offset). The smoke shapes are finite, and the real-Strassen m0 MSE is 1.797e-8 (cn59 fast mode 1.819e-8). The mini
and full-split gate is queued.

**336962 cn59 graded 0.0310** (score = raw MSE again). An all-zero network contributes about 0.050 (cn58), so about
62% of cn59's networks failed and 38% graded. The two builds with RW_TAU 0.02 fail per network on the grader; the
RW_TAU 0.01 builds (cn54, cn54b) never did. A partial, per-network failure points at a resource that grows over the
grader's ~50-network process. In process, the Strassen pool's cached views grow with every network (cn59: 649 ->
900 entries after 2 networks; cn54b 624 -> 828). RW_TAU 0.02 makes more distinct row counts, so more distinct
shapes. Under test: the server-side memory of these views on the 0.12.3 client/server emulator over 8 networks.

**The pool-growth hypothesis is ruled out.** On the 0.12.3 client/server emulator, cn59 and cn54b run side by side
for 2 networks: server RSS 3.17 -> 3.22 GB and 3.19 -> 3.25 GB (high-water mark 4.16 / 4.18 GB, flat), wall
196 / 168 s and 197 / 171 s, residual 1.30 / 1.17 s and 1.36 / 1.20 s. The builds are indistinguishable locally. The
remaining explanation is a slow or loaded grader today pushing networks past the 120 s cap: our chain has only
about 15% wall margin on the emulator. That fits the timeline:
- cn54 graded at 02:05 while the same-minute cn55b / cn55 errored;
- cn58 lost 100% of its networks at 05:51;
- cn59 lost ~62% at 14:54.

A grader-specific effect of RW_TAU 0.02 cannot be excluded, so the next build avoids it: **cn61 = cn54b + residual
rank 6 + late dense window from layer 14, RW_TAU 0.01** (predicted net -0.8%: -0.47% and -0.39%). Its mini /
full-split gate runs first in the queue.

### 8.3 New forum write-ups (Oct 18) and what they change

- **18252 (Nurlanggs, V29 line).** Same failure mode as ours: "the same file graded clean (#336225) and then lost
  about 10% of the suite to zero-prediction fallbacks (#336349) a day later". Going from 6 to 5 Strassen levels
  (local wall 96 -> 60 s, +4% q) made it grade clean. Uploads 1.15-1.35x slower than their base sat 80-90 min and
  returned "Evaluation could not complete". That supports **grader load x thin wall margin** as the cause of our
  cn58 / cn59 failures. Their oracle budget: exact var + d3 + g4 at layers 8-15 leaves 0.13-0.19 of the MSE. They
  also find that the all-distinct kappa3 needs rank ~3n/4 and full depth, so the per-source leg representation
  ends near 3.8e-9 for them.
- **18254 (Susupro, #336537: 3.145e-9, raw 1.853e-8, C/B 0.169; graded median wall 85.5 s, max 102.1 s).**
  1. Full K=4 at 1024x16 is 11.8x below augmented K=3 (beyond the budget).
  2. Inside the budget: D21 column cut at w1 < 0.014 (x0.965). For us, dead-column cuts were measured at +8.9%:
     our learned operator reads those columns.
  3. Young-atom pruning by a norm score (x0.91-0.96). For us it is capped by the 768-column Strassen floor.
  4. **A learned corrector carried through W: raw x0.887 on the grader, lower on all 50 networks.** A shared MLP
     reads per-neuron stats, stats carried through W and W*W, and a 32-d carried state. It predicts each layer's
     created error, which is propagated by Phi W^T and added to the output; the chain is unchanged. Trained on 1000
     public full + 2048 other networks. Their dead end 11 (final-layer readout, x0.979) matches our null ridge.
- **SLEV 4 (one Strassen level less on the young families) for wall margin:** C/B 0.2035 vs 0.1893 (+7.5%) and an
  emulator wall of 167.7 s, not clearly below the 5-level builds. For us it is not the cheap margin it was for
  18252. Closed.

**Now building the carried corrector (task 50).** Per-layer chain statistics at cn61 settings are being dumped for
mini + full f0-999 (`/tmp/claude-0/sc/xs_d2`, `runfast_hf_ldump.py`). The created-error targets and carried
features come from `sc/corr/prep.py`. First a linear version: per-layer ridge, and a joint ridge on the final error
through the propagated features (`sc/corr/fitlin.py`), validated on held-out networks.

### 8.4 The grader is the problem today (Oct 18 evening)

**The control 336965 (cn54b resubmitted unchanged; 3.4544e-9 as 336770 on Oct 17) graded 0.0351: about 70% of its
networks were zeroed.** So the cn58 / cn59 failures are not about RW_TAU 0.02. Today's grader runs our chain past
its per-network limits.

The scale of it: Susupro's C/B 0.169 chain graded at median wall 85.5 s and max 102.1 s against the 120 s cap
(18254). Scaled to our C/B 0.19 that is a median of ~96 s and a max of ~115 s, so even on a healthy grader we run
close to the cap, and a 1.2x slowdown zeroes most networks. 18252 saw the same pattern and fixed it with one
Strassen level fewer.

Measured in process (0.12.3, seed 1, one thread):

| build | C/B | wall | VmHWM |
|---|---|---|---|
| cn61 (young families at 5 levels) | 0.19060 | 106.9 s | 4.25 GB |
| **cn61r (4 levels)** | 0.20350 (+6.8%) | **81.6 s (-24%)** | 3.63 GB |

cn61r matches cn61 to 3e-6 (MSE 1.7800e-8 vs 1.7806e-8 on m0). It was submitted at 17:15 as **336968**, a
robustness probe under the current grader load. If it grades clean while cn54b does not, the wall margin is the
cause, and a 4-level build is the safe second nomination (about +6% score, ~25% wall headroom).

Op counts per network (cn54 profile): 35k flopscope ops, of which 18k are add / subtract (Strassen-Winograd
combinations) and 5k copyto. The young D21 hub issues 7k of them.

**336966 cn61 graded 3.4404e-9** (raw 1.8012e-8, C/B 0.1910), clean: the new best. Against cn54b, raw is -1.04%
(local -1.27%) and C/B +0.6%, so the score is -0.4%. It graded in the same hour that the unchanged cn54b control
lost ~70% of its networks, so the failures are intermittent per job (machine / load), not deterministic per build.
Nominations: cn61 (336966) + cn54b (336770), or cn61r as the hedge if it grades clean.

**336968 cn61r graded 3.6743e-9, clean** (raw 1.8017e-8 = cn61's, C/B 0.2039 = +6.8% as measured). **Nominations:
cn61 (best, 3.4404e-9) + cn61r (robust, 3.6743e-9).** If the private re-run meets a loaded grader, the 4-level build
keeps ~25% wall headroom.

**Corrector progress (held-out mini, trained on full f0-199, 157 networks):** 3x64 MLP -1.28% (80/98), 3x128 MLP
**-1.72%** (81/98; halves -1.49 / -1.96). The full 1000-network dump finishes ~21:30 UTC.

**cn60 = cn61 + RW_TAU 0.02.** 100 mini vs cn61: +0.63% +- 0.18 MSE (halves +0.64 / +0.62), -0.94% C, net -0.31%
(vs base: -0.85% MSE at -0.34% C). Submitted 18:12 UTC as **336975**.

**Corrector variants (train full f0-199, test mini):** the incoming correction J_l s_{l-1} as an input (two-stage)
gives -1.60% vs -1.72% without it, so it is dropped. MLP billing per network (0.12.3): H=64 GELU 0.16% of C, H=128
GELU 0.45% (ReLU 0.09 / 0.30%). In chain, the H=64 build costs +0.25% C (0.19108 vs 0.19060 on the same seed).

**Corrector scaling with training data (3x128 GELU MLP, 7 carried features, test = held-out mini):**

| training networks (full split) | held-out mini |
|---|---|
| 37 (mini only) | -1.01% |
| 157 | -1.72% (81/98) |
| 297 | **-2.02%** (82/98; halves -1.72 / -2.33) |

About -0.3% per doubling, so ~-2.5% is expected at 1000 networks. Carrying all 17 statistics (v2, 21 carried
inputs) gives -1.81% vs -1.72% at 157 networks: noise level, so it is not adopted. The final model trains on all
~1000 full networks once the dump finishes (~22:00 UTC). Then cn62 (cn61 + corrector) and its 4-level twin go
through the gate for Oct 19.

**cn62 (early corrector, 297 training networks) submitted 18:46 UTC as 336979.**
- 0.12.3 bill 0.19162 (+0.54% vs cn61), VmHWM 4.21 GB (releasing the per-layer references fixed the 5.81 GB of
  the first draft), in-process wall 110.6 s vs 106.9 s.
- Real-Strassen label check vs cn61: m0 1.730e-8 vs 1.781e-8 (-2.8%), m60 1.784e-8 vs 1.855e-8 (-3.8%).
- Parity with the torch pipeline: 2.2e-6.
- The package validator caught the corrector running at a non-suite shape; it now runs only at 1024 x 16, and
  every smoke shape is finite.
- Prediction: 3.4404e-9 x 0.980 x 1.0054 ~ 3.39e-9.

**336975 cn60 graded 3.4395e-9 clean** (raw 1.8177e-8, +0.92% vs cn61; C/B 0.1892, -0.94%). On the grader the
RW_TAU 0.02 step is neutral (local -0.3%). cn61 stays the main nomination: it ties, and it avoids RW_TAU 0.02.

**336979 cn62 failed "Smoke test failed".** Reproduced on the 0.12.3 client/server emulator: the flopscope client's
npz reader raises "fortran-order arrays unsupported" (export wrote transposed torch weights). The export now writes
C-contiguous arrays. cn62 then runs clean on the emulator (C/B 0.1924, residual 1.37 s) and on all smoke shapes in
client/server mode. Resubmitted 18:56 UTC as **336980**. **Gate amendment:** every build must also pass the 0.12.3
client/server emulator at the suite shape and at the smoke shapes (`emu123/drive_shapes.py`) before upload.

**336980 cn62 graded 0.0204** (~40% of networks zeroed): the same grader-load pattern; its in-process wall is +3.5%
over cn61. **cn62r (cn61r + the corrector) submitted 19:37 UTC as 336984** with today's last slot. It is the
corrector's clean-grade test on the robust chain (emulator clean, C/B 0.2054; smoke shapes finite; m0 1.7319e-8 vs
cn61r 1.7800e-8). Prediction: raw ~0.98 x 1.8017e-8, score ~3.62e-9.

**336984 cn62r graded 3.6369e-9, clean.** Raw 1.7745e-8 vs cn61r's 1.8017e-8 is **-1.51% on the grader**
(held-out mini -2.02% locally), so the carried corrector transfers. C/B 0.2050 (+0.54%), score -1.0% vs cn61r, and
raw 1.7745e-8 is our best yet. Nominations: cn61 (336966) + cn62r (336984, robust). Next: the 1000-network
corrector (locally ~-2.5%) in cn62 / cn62r on Oct 19.

### 8.5 The 974-network corrector (Oct 18 night)

**Final per-layer corrector (3x128 GELU, rho 0.75, 12 epochs) on all 974 prepared full networks (f0-999):** held-out
mini -2.23% in torch (85/98 better; halves -2.01 / -2.46), against -2.02% for the 297-network model. In chain
(cn62, fast mode, all 100 mini networks): **-2.25% +- 0.20** (87/100 better; halves -2.03 / -2.48). Parity with the
torch pipeline 2.2e-6 on m50. Real Strassen kernels on m0: 1.7228e-8 vs fast 1.7245e-8 vs cn61 1.7806e-8 (-3.2%).
The weights replace `corr.npz` in both `v5b_cn62` and `v5b_cn62r` (same shapes, C order). The 297-network file is
kept at `/tmp/claude-0/sc/corr/bak297/corr.npz`.

Rounding note: the leaderboard shows two significant digits and ties equal displays. Our 3.4404e-9 shows as 3.4e-9
(rank 59 tie at 22:00 UTC); the next display step (3.3e-9, rank 57) needs < 3.35e-9, i.e. -2.6% from cn61. The
974-network cn62 predicts ~3.38-3.40e-9 (the grader's corrector transfer was 0.75x the local gain for the 297 model).

**Layer weights from the measured propagation gain.** g_l = E|J_{L<-l} u|^2 / E|u|^2 for white u through
s -> Phi * (W^T s), mean over m0-5 (`sc/corr2/gain.py`): 0.031 0.052 0.076 0.107 0.139 0.180 0.223 0.282 0.349 0.412
0.502 0.580 0.656 0.765 0.883 1. The per-layer ratio is 0.85-0.88 from layer 9 on, so rho 0.75 underweights the middle
layers by 2-3x. The propagated created errors without cross terms sum to 0.978 of the final MSE (shares rise from 0.1%
at layer 0 to 22.3% at layer 15): the layers' contributions are nearly uncorrelated, so a per-layer loss weighted by
g_l is close to the end-to-end objective.

**Gate of the 974-network cn62 / cn62r (both pass):** 0.12.3 client/server emulator at the suite shape: cn62 C/B
0.1924, residual 1.34 s; cn62r C/B 0.2054, residual 1.13 s; every smoke shape finite on both. Real Strassen m0:
cn62r 1.7249e-8 (cn61r 1.7800e-8, -3.1%). 0.12.3 bill (runv, seed 1): cn62 0.19162 (wall 106.1 s, VmHWM 4.23 GB).

### 8.6 A learned state carried through W (18254's 32-d carried state)

`sc/corr2/fitrnn.py`: at layer l the MLP reads [24 features, layer one-hot, W_l^T state_{l-1} / sqrt(n mean W^2)] and
returns [delta_l, state_l] (S = 32). It is trained through depth on whole networks (weights regenerated per network in
DataLoader workers), with the per-layer loss weighted by the measured gain g_l.

| train f0-199 (200 nets), 3x128 GELU, 12 epochs, test held-out mini | held-out dMSE | better |
|---|---|---|
| S = 0 (same trainer, one network per step) | -1.44% (halves -1.30 / -1.58) | 84/98 |
| **S = 32** | **-2.44%** (halves -2.26 / -2.62) | 86/98 |
| S = 32, layer weights rho^(15-l) (rho 0.75) instead of g_l | -1.94% (halves -1.70 / -2.18) | 84/98 |
| S = 32, ReLU MLP (bills ~0.15% C less than exact GELU) | **-2.62%** (halves -2.41 / -2.84) | 81/98 |
| S = 32, ReLU, loss = per-layer + end-to-end final error (both normalised) | -1.05% (halves -0.96 / -1.15) | 67/98 |

The end-to-end term hurts (one noisy scalar per network); the per-layer loss weighted by g_l stays.

**S = 32 GELU on all 1000 full networks (12 epochs, ~65 min): held-out mini -3.92%** (87/98; halves -3.64 /
-4.21), against -2.23% for the 974-network per-layer model. Built as `candidates/v5b_cn63` (cn62 + the state; the
`sc/c63` estimator) and its 4-level twin `v5b_cn63r` (`sc/corr2/build_cn63.sh`). In chain vs torch on m50: 2.3e-6.
The ReLU version on 1000 networks is training.

At 200 training networks the state beats the 974-network per-layer model (-2.23%). The in-chain version (`sc/c63`,
`self._corrS` from W3's width; W @ state per layer) matches torch to 2.3e-6 on m50. 0.12.3 bill 0.19221 vs cn62
0.19162 (+0.31% C), in-process wall 108.7 s vs 106.1 s. The 974-network S = 32 model is training (~64 min).

**Gate of cn63 / cn63r (pass).** Real Strassen m0: cn63 1.6675e-8, cn63r 1.6699e-8 (cn62 1.7228e-8, cn61 1.7806e-8:
-6.4% vs cn61). 0.12.3 client/server emulator at the suite shape: cn63 C/B 0.1930 (residual 1.29 s), cn63r 0.2059
(1.16 s); every smoke shape finite on both. Prediction for cn63: raw -2.9..-3.9% vs cn61 (grader transfer 0.75-1.0 of
local), C +0.85%, score ~3.33-3.37e-9.
In chain on all 100 mini networks (fast mode): **cn63 -3.89% +- 0.30** (88/100; halves -3.66 / -4.12), cn62 -2.25%.

**337009 = cn63, submitted 00:00 UTC Oct 19** (first slot of the day, alone, so that our own jobs do not load the
grader together). The stale 00:04 reminder (cn62 plan) was deleted.

Width of the state (200 training nets, ReLU, gain weights): S = 64 -2.55% (86/98) vs S = 32 -2.62%: no gain for twice
the state product. S stays 32.

**A second state channel carried through W*W (variance-like messages)**, S2 = 16 next to S = 32 (200 training nets,
ReLU, gain weights): **-3.44%** (86/98; halves -2.88 / -4.03) against -2.62% without it. The training loss is lower
at every epoch from the 3rd on. In chain: `(W*W) @ state2 / (sum(W*W) / n)` per layer (+0.12% C at S2 = 16).
The 1000-network S = 32 + S2 = 16 ReLU model is training (00:27 UTC); the ReLU S = 32-only run was stopped for it.
A third channel through W*W*W (S3 = 16, kappa3-like messages, scaled by 1 / sqrt(sum W^6 / n)) is on test.

**337009 cn63 graded 0.0220**: ~44% of its networks zeroed (score = raw MSE; all-zero would be ~0.050). Second
5-level corrector build lost to the per-network wall cap (cn62 336980: 0.0204). The 4-level builds (cn61r, cn62r)
graded clean both times; the 5-level cn61 / cn60 graded clean on Oct 18 afternoon. The 5-level chain plus the
corrector (in-process 108.7 s vs cn61 106.9 s) sits too close to the cap under the current grader load.
Next: a Strassen level per family (young transport / newborn trees T, young D21 hub H; `sc/c63w`, CN_SLEVT /
CN_SLEVH) to find a wall cut cheaper than the full 4-level twin (+6.8% C).

Third channel through W*W*W (S3 = 16 next to S = 32, S2 = 16; 200 nets): -3.17% (86/98) vs -3.44% without it.
Training loss a little lower, held-out worse: dropped.
**337017 = cn63r submitted 00:41 UTC** (robust 4-level twin; its grade measures the state corrector's transfer).

**Strassen level per family (cn63, 0.12.3 bill and in-process wall, seed 1, container under training load):**

| transport / newborn trees T | young D21 hub H | C/B | wall |
|---|---|---|---|
| 5 | 5 (cn63) | 0.19221 | 113.4 s |
| 5 | 4 | 0.19856 (+3.3%) | 94.8 s (-16%) |
| 4 | 5 | 0.19875 (+3.4%) | 92.5 s (-18%) |
| 4 | 4 (cn63r) | 0.20510 (+6.7%) | 82.6 s (-27%) |

The trade is linear: each family costs ~+3.3% C per ~-17% wall, so a half-way build only ties cn61's score once the
corrector's gain is counted. The useful pair stays the full 5-level chain (best score when it grades clean) and the
4-level twin (robust hedge).

Width of the W*W channel: S2 = 32 -3.22% (84/98) vs S2 = 16 -3.44%: S2 stays 16.

### 8.7 strassen5: the kernel's wall is strided memory traffic

Profile of one cn63 prediction (0.12.3, in-process, one thread; the container was under training load): 85 s of 124 s
inside numpy (`_call_numpy_impl`), of which the pointwise wrappers take ~75 s. Inside the Strassen kernel: fused leaf
43.5 s, left combos 11.1 s, right combos 14.0 s, assembly 14.8 s (~83 s of 111 s). The corrector block is 0.35 s.
Micro-benchmark: an add over 32-float quadrant rows costs 2.4 ns per element against 1.1 contiguous (2.6x at 64-float
rows); a batched 32x32 matmul with a strided right operand and output 11.1 ms against 7.9 contiguous.

`strassen5.py` keeps strassen4's arithmetic operation by operation, but below the top level every plane is stored in
the recursive block layout (..., (2, 2) * r, m >> r, k >> r), so every quadrant at every level is a contiguous
sub-block. The top level reads the operands' quadrants through split views (fnp.reshape: one FLOP per element of
X, R and out; the transpose is free). Shared left trees cache one combo set per level count.

| test | result |
|---|---|
| 10 shapes (trees, ksum hub form, mb 20 / 24, levels 1-5, widths 768 / 960) | **bitwise identical** to strassen4; per call -15..-37% time, bill +0.20..+0.36% |
| one full prediction (cn63 + the S2 corrector, seed 1) | **bitwise identical** predictions; C/B 0.19223 -> 0.19269 (+0.24%); wall 107.3 -> **92.0 s (-14%)**; VmHWM 4.20 -> 4.94 GB |

That is most of the 4-level twin's wall margin (82.6 s) at +0.24% C instead of +6.7%.

**strassen5 in the client/server emulator: little gain.** Back to back on the loaded container: strassen5c 180.6 s
vs strassen4 184.1 s (-2%), residual 2.0 s vs 1.1 s. In-process compute is not what the client/server wall is made
of; strassen5 stays out of the submitted builds for now.

**Two container restarts (01:23, 01:45 UTC)** killed the S2 training at epoch 6; `fitrnn.py` now writes a per-epoch
checkpoint and resumes from it.

**337017 cn63r graded 3.5984e-9, clean** (raw 1.7507e-8 = -2.83% vs cn61r, local -3.89%: transfer 0.73, as cn62r's
0.75). Best robust grade (cn62r 3.6369e-9).

### 8.8 The failure mode is probably the 0.4 s residual cap, not the 120 s wall

whestbench's graded limits (`whestbench/cli.py`, `runner.py`): wall 120 s per MLP, **residual wall time 0.4 s per
predict() call** ("non-flopscope time ... Exceeding it zeroes that MLP's predictions"), memory 8192 MB, setup 5 s.
Residual = wall - flopscope backend - flopscope dispatch time; flopscope's own per-op bookkeeping after its timer
and every `__getitem__` land in it. Measured here (0.12.3, in-process):

| item | residual |
|---|---|
| one op | ~25-50 us (mild growth with size and ndim) |
| one slice of a flopscope array | ~1.6-1.9 us (26 us only for arrays carrying symmetry tags, e.g. from fnp.ones) |
| cn63 per predict: 35,149 ops + 23,182 slices | 1.4-1.6 s (GC 0.13 s of it) |

Same load, same seed: cn61 1.62 s, cn62 1.54, cn63 1.47, **cn63r 1.31** (4 levels: 32,518 ops), cn63k (strassen5c,
36,709 ops) 1.59. Emulator residuals: cn62 1.335 vs cn62r 1.134 s, cn63 1.29 vs cn63r 1.16 s, while the emulator
walls of the 5- and 4-level builds are about equal. Our builds graded clean at ~1.1-1.3 s here, so the grader
runs ~3x faster and our 5-level builds sit near the cap; load pushes a share of MLPs over it. The 4-level builds
have ~10-15% less residual and have always graded clean. The levers are fewer ops and fewer slices at the same
FLOPs:
- lazy pooled views (`strassen4.py` in cn64): slices 23,182 -> 17,946 (-23%), bitwise-identical predictions, same
  bill;
- Strassen batch chunks of 3 (fits the ~135 MB array cap at 5 levels): ops -5.5%, bill +0.27%;
- the PK2 term program issues ~2.1k ops and ~4.2k slices (2-3 ops + 3 slices per term); batching terms is next.
GC: disabling it inside predict only defers one large collection into the same window (no gain); `gc.freeze()`
stays.

### 8.9 cn64: S = 32 + W*W channel S2 = 16, ReLU, 1000 networks

**Held-out mini -4.60%** (91/98; halves -4.31 / -4.90), cn63's model -3.92%. Built as `v5b_cn64` / `v5b_cn64r`
(estimator from `sc/c63`, lazy-view `strassen4.py`). In chain vs torch on m50: 2.2e-6. Real Strassen m0: cn64
1.6296e-8 (cn63 1.6675e-8, cn61 1.7806e-8: -8.5%).

**Residual is per flopscope op, ~40-45 us here, independent of ndim (3-D to 7-D) and nearly of size.** Measured
with the machine otherwise idle (seed 1): cn63 1.529 s, cn64 (lazy views, -23% slices) 1.514 s, cn63r 1.344 s,
cn64r 1.343 s. The slices were cheap after all; the 4-level builds' lead is their op count (32.5k vs 35.1k) at
~41-44 us per op. Batch chunks of 3 plus a fused PK2 einsum per term (cn65 scratch: 34.0k numpy calls, C/B
0.19310, +0.46%) measured 1.50 s: inside the noise, not adopted. Flopscope's per-op cost in-process is ~350 us,
of which ~29 us is residual (bookkeeping after its timer stops), so only fewer ops would move it.

**cn64r gate:** emulator C/B 0.2060, residual 1.19 s, smoke shapes finite; real Strassen m0 1.6318e-8.
**337030 = cn64 submitted 03:14 UTC Oct 19.**

Hidden width with the two channels (200 nets, ReLU): H = 192 -3.32% (82/98) vs H = 128 -3.44%: no gain. The 1000-net
S = 32 + S2 = 16 model is retrained for 16 epochs at a slower decay (0.85 per epoch; the 12-epoch loss was still
falling).

**337030 cn64 graded 3.3386e-9, clean: the new best** (raw 1.7332e-8 = -3.78% vs cn61's 1.8012e-8, local -4.58%:
transfer 0.83; C/B 0.1926). Below 3.35e-9, so the leaderboard shows 3.3e-9 (from 3.4e-9). It graded at ~03:30-03:50
UTC, when the queue was slow but the run clean. **337037 = cn64r submitted 03:50 UTC** as the robust nomination.
Nominations now: cn64 (337030) + cn63r (337017), cn64r to replace cn63r once graded.

**337037 cn64r graded 3.5636e-9, clean** (raw 1.7336e-8 = cn64's; C/B 0.2056). **Nominations: cn64 (337030) + cn64r
(337037).**

Backward state (a second pass from the last layer down through W, SBW = 16, next to S = 32 and S2 = 16; 200 nets):
-3.35% (85/98) vs -3.44% without it: dropped.

### 8.10 What the residual clock counts on the grader (public forum 18202, our client/server counts)

Forum 18202 (an entrant's own grader data, August): the 0.4 s residual gate is enforced on the live evaluator; an
estimator at ~21k calls per MLP went from 5/50 RESIDUAL failures to a clean pass by removing client-side sync
points. With flopscope-client 0.12.3 every op is a synchronous round trip billed as dispatch, so the residual is the
client's own Python around each round trip. Counted in our 0.12.3 client/server emulator (one MLP, steady state):
**cn63 makes 49,354 round trips, of which 13,900 are `__getitem__`** (slices are server round trips in client mode;
in-process they are free), 9.5k add, 8.5k subtract, 5.0k copyto, 4.7k multiply, 2.8k matmul. Client residual
1.25-1.31 s here, ~26 us per round trip. So the slice count does matter on the grader, and the lazy views of cn64
are a real cut there (counts below).

Round trips per MLP in the 0.12.3 client/server emulator (steady state; the cold first MLP of a process in brackets):

| build | round trips | slices | residual here |
|---|---|---|---|
| cn63 (failed on the grader) | 49,354 (59,729) | 13,900 | 1.25 s |
| cn64 (graded clean) | 47,595 (54,164) | 12,145 | 1.18 s |
| cn64r (graded clean) | 44,905 (50,781) | 12,066 | 1.08 s |
| **c66 = cn64 + PK2 views made once per layer** (bitwise-identical predictions, same bill) | **44,639 (51,208)** | 9,189 | 1.15 s |
| c66 + batch chunks of 3 | 42,924 (49,743) | 9,432 | 1.06 s |

The residual is ~24-25 us per round trip here. PK2: the term loop made ~6 slices per term (`WL2[t][:, None]`,
`WR2[t][None, :]`, `PK2[g]`); the broadcast stacks and the PK2 planes are now viewed once per layer (2 slices per
term). Batch chunks of 3 bill +0.27% (FLOPs by op: matmul +1.26e9, because a chunk runs at its widest source's
live width); not adopted. cn66 = cn64 + the PK2 change (+ the 16-epoch corrector if it is better).

### 8.11 cn66

16-epoch corrector (decay 0.85; final training loss 0.9620 vs 0.9642): held-out mini -4.72% (89/98; halves -4.61 /
-4.83) vs -4.60%. **cn66 = cn64 + PK2 views once per layer + this corrector.** In chain vs torch on m50 2.3e-6;
in chain on 100 mini **-4.72% +- 0.33** (91/100; cn64 -4.58%); real Strassen m0 1.6380e-8 (finite); emulator
C/B 0.1930, smoke shapes finite. **Submitted 05:37 UTC as 337053.**

Client-side slice round trips of c66 by call site (steady MLP, 9,189): Strassen `_quads` (the operands' quadrant views
of every call) 3,608; the PK2 term loop 1,423 (2 per term); pool plane views 1,064; pool slab slices 313; the
rest under 160 each. **c67 / cn67 = cn66 + the PK2 coefficient stacks in persistent per-mode buffers** (matmul with
`out=`; their row and broadcast views made once per process): **43,001 round trips (cold 49,810)**, slices 7,551,
bitwise-identical predictions (real Strassen m0 all layers 0.0), same bill. Emulator C/B 0.1930, residual 1.26 s,
smoke shapes finite. Round trips: cn64 47,595 -> cn66 44,639 -> **cn67 43,001 (-9.7%)**, below cn64r's 44,905.

**337053 cn66 graded 3.3341e-9, clean: the new best** (raw 1.7309e-8 = -3.90% vs cn61; local -4.72%: transfer 0.83;
C/B 0.1926). **337060 = cn67r submitted 06:22 UTC** (the robust nomination with every round-trip cut).
Leaderboard 06:15 UTC: rank 57 (3.3e-9 display; tie with two teams), up from 59.

MP step (the learned message passing on layers 8-15) off: C/B 0.18678 vs 0.19060 (-2.0%); accuracy on 30 mini in
progress (a win only if the corrector, retrained on MP-off dumps, recovers most of it).
