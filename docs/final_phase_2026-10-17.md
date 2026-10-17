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
| dead columns only | Phi < 0.002 | -1.4% +- 2.3% (5/6), move 3.8e-5 | 75-network screen queued; it would cut ~4-5% of C |
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
