# Brainstorm and critical review, Oct 16-17 (before the last submission day)

## 1. Where we stand against the board (public leaderboard, Oct 16 22:08 UTC, 150 teams on pages 1-3)

| team | adjusted | raw MSE | C/B | first entry | note |
|---|---|---|---|---|---|
| marius_binner (#1) | 1.30e-9 | 1.10e-8 | 0.122 | Jun 5 | 277 entries |
| J2W (#2) | 1.40e-9 | 1.36e-8 | 0.107 | Jul 10 | 468 entries |
| oliver_chen (#3) | 1.50e-9 | 1.20e-8 | 0.129 | **Oct 2** | 106 entries: top 3 in two weeks |
| mliston / Luna (#3) | 1.50e-9 | 1.34-1.35e-8 | 0.115 | Jun 14 / Sep 23 | |
| mk1981 (#7) | 1.70e-9 | 1.65e-8 | 0.106 | **Oct 2** | |
| Masato (#7) | 1.70e-9 | 1.32e-8 | 0.132 | **Oct 4** | 33 entries |
| e30_a clones (#37, six teams) | 3.1e-9 | 2.03-2.06e-8 | 0.1517 | Oct | LordProcciuto's public MIT engine |
| **us (#56)** | **3.6e-9** | **1.79e-8** | **0.1996** | Oct 1 | cn21 |

The gap to #1 is 2.75x: 1.63x in cost and 1.63x in raw error. Teams that started in October reached the
top 10 within days, so a strong approach is learnable quickly from public material plus original work.
Our raw error is already better than the public engines' (e30_a 2.03e-8, V29 2.13e-8). Our cost is
the worst of every team within 2x of our score.

## 2. What the public write-ups (Oct 15-16) say about where accuracy lives

* **kai_leal_gonzalez (18246):** the exact (3,1) slice of kappa4, kappa(z_i,z_i,z_i,z_j), lowers V29's final MSE
  by 18% (all 23 networks). A two-term law K31_ij ~ b1 v_i S_ij + b2 kappa3_i D21_ij / v_i (b2 = 1.17-1.41
  fitted at layers 4-14, b2 = 1 for a quadratic perturbation of a Gaussian) gave -3.1% in a pre-registered
  77-network test (8 lines, 4.4e7 FLOPs). Our chain models this slice like V29 (wk431 = (m/2) lambda C_off)
  plus a learned pair correction from layer 8 on, so the kappa3 * D21 / v term is missing below layer 8 and
  only partly learnable above. **Being tested (section 4).**
* **Artyom_Mazur (18249):** the missing accuracy one order up is the young kappa4, "above all the tree that
  hands kappa3 on through one covariance edge"; a reader of those trees cuts raw error to x0.70. kai's b2 term
  is the cheapest instance of that tree. Also: per-neuron correctors fail on k >= 3 chains (errors are random
  projections), and 72% of what carrying kappa3 buys lives in its all-distinct part.
* **il.miscusi (18248):** removing the global scale mode (homogeneity) makes a *carried* kappa4 core pay (x0.38 vs the
  same chain without extraction); in memoryless-kappa4 chains (V29, e30_a) extraction hurts. Our chain is
  memoryless plus learned: extraction is not a drop-in. It is the same mode we found in the label noise
  (section 14 of the experiments log).
* **504aldo (18218):** the leaders' bill equals a V29-type chain minus its old tier, and their raw error is
  below the "K3 + memoryless kappa4" ceiling (~1.94e-8 on the LB scale): they carry kappa4 content beyond the
  memoryless core and old kappa3 content at almost no cost. Nobody has published how.

## 3. Critical review of our own process

1. **The yardstick misled us twice** (tier rank 288, feedback rank 2): MLPs 5000/5001 are among the easiest
   local networks (label MSE ~1e-9 vs a leaderboard mean of 1.7e-8), and the damage of a perturbation comes from
   its alignment with the existing error, which easy networks barely have. Fixed: the six-network hard test
   (1015, 1002, 5010, 1005, 1008, 1010) reproduces cn21's leaderboard damage. Remaining weakness: se 0.5-2.4%,
   growing with the size of the move; per-MLP swings up to +-10% for residual-rank changes.
2. **Cost work has hit diminishing returns.** Since Oct 15 every structural cut (old D21 drop, low-rank D21,
   low-rank a_b, saturated columns, rank-1 feedback, rows 0.02, columns 0.015) was closed by measurement. The
   remaining cost levers are thin legs only (~6% of C, mostly harvested). The leaders' advantage is
   structural (no dense old tier), not a constant factor we can polish away in a day.
3. **Accuracy has not been worked on since the learned operator was ported.** All our raw-MSE changes since
   cn3 come from truncation effects (+-3%). The public evidence says the large remaining errors are kappa4
   content (the (3,1) slice: 18%; the young kappa4 trees: 30%). That is where the next gains are.
4. **Slot use:** probes with expected gains of 1-2% each are the right size only when the outcome cannot be
   predicted locally. With the hard test now predictive, fewer leaderboard probes and more local screening.
5. **Final selection:** cn21 and cn11a are the same chain family; picking two siblings is fine for expected
   score but gives no diversification. A candidate with a different closure (e.g. the (3,1) law) is a
   reasonable second pick only if it is better on the public set.
6. **Learned operator drift:** k34 + MP were trained by 504aldo on his chain's states. Our chain now differs
   (tiers, masks, feedback rank 2). Any closure change that moves the inputs of the MP (wk431, D21) may be
   partly undone or amplified by it, so closure additions go after the learned steps.

## 4. Ideas, ranked by expected value per day of work

| # | idea | basis | expected | cost / risk | status |
|---|---|---|---|---|---|
| 1 | (3,1) law: wk431 += b2 D3_j D21_ji / var_j after the learned steps | 18246, pre-registered -3.1% on V29 | -1..-3% raw | ~0 C; orientation must match | **hurts here**: b2 = 1 / 0.5 / transposed: +3.4 / +1.4 / +4.0% (se 0.6-1.4%); our learned operator already carries this slice. Pending: below layer 8 only, variance-weighted lambda term, b2 = -0.5 |
| 2 | residual-rank ladder (cn36-cn39) | hard test: neutral +-2% | -1..-3% score | 5 slots | wave 1 |
| 3 | b2 per layer ~1.2-1.4 and a refit of the lambda table (b1) | 18246 free fit | -0.5..-1% | needs labels; overfit risk | after 1 |
| 4 | young-kappa4 tree reader (kappa3 through one covariance edge) as an n^2 per-layer term | 18249: x0.70 in a float64 chain | up to -10..-30% raw | derivation + implementation > 1 day; their representation paid back what it bought | not today |
| 5 | scale-mode extraction with a carried kappa4 core | 18248 | x0.38 in their chain | needs a carried kappa4 core: +77 units (their cost) | no |
| 6 | more MC samples for the six hard networks (overnight) | label se is the bottleneck of local screening | 2x sharper tests | ~1 h CPU per 2e6 samples per network | if CPU is free |
| 7 | old tier without dense legs (the leaders' bill) | 18218 section 6 | -30% C | research; no public method | no |
| 8 | deeper Strassen (young families at 6 levels where the array cap allows) | ngmitam (18218 #6): V29 at leaf 8 graded fine, -3.5% | **-1.2% C measured** (0.1992 -> 0.1968 on MLP 5000; 64-blocks eat part of it) | exact; the earlier level-6 failures match the per-array cap exactly ((7^5, 2, 32, 32) float32 = 137,682,944 B), not wall time | scratch xs6 ready; emulator check pending |
| 9 | strength of the learned corrections (k34, MP node, MP pair) | 504aldo trained them on his chain; ours has drifted (tiers, masks, rank-2 feedback) | **found -2.2%** | 3 knobs, local screening only | MP pair x0.8 / 0.6 / 0.4 / 0: -0.9 / -1.7 / -2.2 / -2.6% (all six MLPs improve at 0.4-0.8); MP node x0.8 +0.4%; k34 x0.8 / 1.2: +4.4 / -0.9%; lambda x0.9 / 1.1: +0.5 / -0.1%. **cn41 (pair x0.4) submitted 336612**; held-out check on six more MLPs queued |
| 9b | negative (3,1) term, b2 = -0.5 (undo the same overshoot) | the b2 > 0 response was linear | -1.1% | substitutes for 9 (pair 0.8 + b2 -0.5 = pair 0.6 alone: -1.7%) | **cn40 submitted 336610** |
| 10 | message passing from layer 6 instead of 8 | MP is the strongest accuracy component | closed | out-of-distribution inputs at layers 6-7 | +1.8 +- 3.1% |

## 5. What the night changed

* The public closure fix (kai's (3,1) law) does not transfer: our learned operator already carries that slice,
  and in fact overshoots it. Measuring that overshoot gave a cleaner lever: the MP pair correction is ~2.5x too
  strong in this chain (x0.4 reads -2.2%, every hard MLP improves). Likely cause: the MP reads D21 features,
  and our D21 differs from the training chain's (tiers, masks, rank-2 feedback). This is a calibration of a
  learned component to the chain it now runs in, not a new closure.
* The wave-1 leaderboard set (cn35 base, residual ranks 8 / 4, cn40, cn41) measures every lever we have on
  the public 100 networks. Wave 2 combines the winners: residual rank x pair strength x k34 x1.15 (if the
  combination test holds) x level-6 Strassen (-1% C, exact).

## 6. After the held-out check: what the night really showed, and ideas for later

**The pair-strength gain was a selection effect.** On the six tuning networks the MP pair correction at x0.6 / 0.4 / 0 read
-1.7 / -2.2 / -2.6% (every network improved). On six held-out networks the same settings read +1.8 / +3.1 / +7.4%.
The tuning set was chosen as the six hardest networks, and the overshoot lives in two of them (1008, 1010). Two
process rules follow:
* tune on one set of networks, confirm on another, and only then spend a slot (we submitted cn40 / cn41 before
  the confirmation; they are bets);
* "every tuning network improved" is not evidence of generality when the tuning networks are selected by an
  extreme statistic (hardness).

**Ideas worth pursuing with more data or time (ranked):**

1. **Recalibrate the learned operator to the chain it now runs in.** The strength scans show the k34 / MP
   corrections are miscalibrated per network: too strong on some (1008, 1010), too weak or right on others.
   A fine-tune of k34 + MP on our chain's own rollouts (scripts/train.py --v5b supports it) needs clean labels:
   the 5e5-sample labels are ~4x noisier than the error. Sources: generate 2-4e6-sample labels for 50-100
   networks (~1 GPU-day), or the CC BY 4.0 Monte Carlo oracles Artyom_Mazur will publish after the close.
2. **Per-network adaptive operator strength.** If a cheap statistic of the chain state predicts where the MP
   pair term overshoots (candidates: the share of D21 energy in its dominant pair, dead fraction at late
   layers, mean |skew|), a strength that follows it would keep the tuning-set gain (-2..-8% on the networks
   that overshoot) without the held-out loss. Needs ~50 networks to fit and validate.
3. **The young kappa4 tree (Artyom_Mazur 18249: x0.70 raw in a float64 chain).** The tree that hands kappa3 on
   through one covariance edge. Our learned operator carries part of it (the (3,1) law double-counts), so the
   right test is an ablation inside our chain: compute the tree's contribution to the next layer's (3,1) and
   (2,2) slices exactly for one layer and measure how much the operator already captures.
4. **The old tier without dense legs.** The leaders' bill equals ours minus the old tier (504aldo 18218). Our
   measurements rule out low-rank D21, dropping old D21 and CP-style merges; the remaining door is a different
   representation of old kappa3 (not legs), which is research, not tuning.
5. **Deeper Strassen everywhere the array cap allows** (cn44 tests level 6 on the young families on the real
   grader: -0.9% C, wall +19%). If the grader accepts it, level 6 for the hub's full-width chunks at batch 1
   and level 5 for the rank-320 families are the next -1%.
6. **More MC samples for the twelve local test networks** (2-4e6 each). The label-noise term caps local
   screening at ~0.5-2%; four times the samples halves it, which would have caught the selection effect
   above with fewer networks.

## 7. Leaderboard outcome of the night's work (Oct 17, 02:00 UTC)

| submission | score | verdict |
|---|---|---|
| cn37 (residual rank 4) | **3.5262e-9** | best; -1.6% vs cn21 |
| cn36 (residual rank 8) | 3.5275e-9 | same within 0.04% |
| cn35 (exact hub fix) | 3.5756e-9 | as predicted (-0.17%) |
| cn40 (negative (3,1) term) | 3.6151e-9 | lost (+1.2% raw): selection effect |
| cn41 (MP pair x0.4) | 3.6782e-9 | lost (+2.9% raw): selection effect |
| cn44 (rank 8 + level-6 Strassen) | failed ("Error while scoring") | 4th level-6 failure: level 6 is out |

Net: the board position moved 3.58 -> 3.53e-9; the two accuracy bets were the price of skipping the held-out
confirmation before submitting. Level 6 failed on the grader a fourth time and is closed.

## 8. How the top of the board gets there: a second reading (Oct 17, 03:00 UTC)

Board on Oct 16: the first four score 1.3-1.6e-9 with raw 1.10-1.35e-8 at 11-12.5% of B. We sit at raw 1.82e-8 at
19.3%. They are x0.6-0.7 on error and x0.6 on cost at once. What the public write-ups pin down:

**Accuracy: the leaders' raw level is the level of a carried kappa4 core with the scale mode removed.**
- il.miscusi (18248, PDF Table 5), lean float64 vehicle, 100 networks: ARC's reference AUGMENT chain (kappa4 carried as
  an r=1 matrix core G, current term program) is 0.80 of V29's error. With coherent scale-mode extraction it is
  **0.58 of V29**, i.e. ~1.24e-8 on the leaderboard scale: exactly the leaders' raw band. The gain shape is
  "all-layer and multiplicative, flat from layer 3", which is also the shape the e30_a write-up reads off the
  leaders' public telemetry (0.60-0.72 on all layers).
- Artyom (18249, PDF section 7): the missing accuracy of a k=3 chain is the *young* kappa4. It is mostly the "3+2" tree
  that hands the whole kappa3 on through one covariance edge, born in the last two layers. A reader of those trees is
  x0.70 of his raw (≈1.42e-8). In his representation it costs about what it buys (+5-7% FLOPs, and +23% flopscope
  calls, which crossed the 400 ms residual cap on 16/100 networks).
- Both say the same thing from two sides: kappa4 needs memory, and the memory must not carry the scale mode.

**Our chain against that picture (measured today):**
- The scale mode is most of our D21. MLP 1015, cn37: the share of D21 energy explained by the scale pattern
  S_ij = var_i mu_j + 2 mu_i C_ij is 0.28 at layer 1, rising to 0.82 at depth, with accumulated eps ~5e-3 (il.miscusi:
  0.004-0.005). The old tiers' part is 0.35 -> 0.82, the young sources' 0.58-0.73.
- So the old tiers cannot be replaced by a scalar eps: their non-mode residual is still ~20% of D21's norm, and
  D21 has to be right to ~2% (504aldo's eps^2 law). Closed.
- Removing the mode in a memoryless chain hurts, as il.miscusi measured in V29 and e30_a: the exact spherical input
  (x = E|x| u, the sphere's kappa4 at layer 0; layer-0 means match to 5e-7) is **+35%** in cn37. Our lambda law
  and learned operator are calibrated to the mode, so mode removal and memoryless kappa4 do not mix. Closed.
- Per-neuron correctors of the final layer fail on 75 networks too (held-out -9..+16%), as Artyom's random-projection
  result predicts. Closed.

**Cost: the leaders' bill is below V29's young tier alone.** Levers in public: Strassen to leaf 8 (ngmitam, graded
-3.5%; our level-6 builds fail on the grader for a reason we have not isolated), matched dead-row and dead-column cuts
(we have them), and a band basis that rides the transport and is never rebuilt (Artyom: x0.893 score; ours is
rebuilt on alternate joins). Nobody has published a kappa3 state that is not a list of sources; the leaders
seem to have one.

**What is portable in a day, and being screened on all 75 labeled networks (fast mode):**
- C0^3 (the third Mehler term, (1/6) a3 b3 C0^3, one of Artyom's two missing augmented pair diagrams; O(n^2)).
  Smoke test on two hard networks: -1.2%.
- marg2 (Artyom's consistency rule: keep the k=2 marginal Edgeworth corrections only in the dead tail, alpha <= -2;
  zero FLOPs). Smoke test: -1.5%.
- the scale-mixture K22 pattern (+2 v C0^2 on top of the dyadic (g4_i + g4_j)/6), the pattern Artyom found the
  step's diagrams need.
- MP at the final layer (-1.4% on one network, but +5.0% C: unlikely), AGE_OLD 4 (-5.2% C), learned-operator gains.

**Not portable in a day:** the carried kappa4 core with extraction. il.miscusi's production port costs 77 units
(+25% of their bill: G transport 29, capped (2,1,1) production 32, K211 11) for x0.854, which did not beat
their own best build on score. In our chain the learned operator already takes part of that gain, so the trade is
worse. The young-tree reader is the better research bet for after the deadline.
