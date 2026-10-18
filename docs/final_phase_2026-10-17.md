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
