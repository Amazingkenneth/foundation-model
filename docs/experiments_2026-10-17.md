# Experiments 2026-10-17: fast screening on all 75 labeled networks

Base: `candidates/v5b_cn37` (graded 3.5262e-9, raw 1.8235e-8, C/B 0.1934), best on the leaderboard.

## 1. A 10x faster local gate (scripts/fast)

The selection effect of Oct 16 (cn40 / cn41 tuned on six hard networks, both lost on the
leaderboard) came from screening on too few networks; a 24-network run of the real op stream took
~1.5 h. `scripts/fast/runfast.py` replaces every Strassen call (`strassen2.mm`, `LeftTree`) by a
plain batched matmul: the same arithmetic in a different float32 rounding order. On MLP 1015 it
moves the prediction by 6.5e-7 rms (chaos level) and the noise-corrected label MSE by -0.02%,
while one network takes ~60 s on one core instead of ~8 min on two. `goall.sh` runs a variant on
all 75 labeled networks (64 train at 5e5 samples, 11 validation at 1e6) in four processes;
`eval75.py` gives the pooled paired dMSE (sum of per-network MSE changes over the sum of base
MSEs, the leaderboard weighting) with a bootstrap se and four splits (old tuning-12 vs the rest,
even vs odd seeds) as a consistency check. Costs are not measured in fast mode; they come from
one real run on MLP 5000.

## 2. Where the cost goes (cn37, MLP 5000, C/B 0.1935)

Per-call-site profile (`scripts/fast/prof.py`): matmul 84.7% of C, einsum 4.7%, add/subtract 6.2%
(mostly Strassen combos), copies and reshapes 1.8%, QR 0.7%. By helper: the young-source D21 hubs
(`_hub`, 39%) and young-source transports (`_mmc`, 36%) are ~75% of the bill; the tier-1/2 leg
formation and rotations (`_mmc2`) 4%, the covariance sandwich 3.6%, the basis products (`_sd`) 3%.
Every remaining exact lever is in the Strassen kernel: level 6 for the young families (cn44,
-0.9%; failed on the grader: its emulator wall, 239 s, is over the ~175 s ceiling that maps to the grader's 120 s
per-network wall cap), and a Winograd form of the combos would save ~1/15 of the 6% addition overhead (~0.4%,
not worth a slot).

Cost elasticities of the remaining structural knobs (real op stream, MLP 5000; C/B of cn37 0.1935):

| change | C/B | dC |
|---|---|---|
| AGE_OLD 4 (one young source fewer, tier 1 one source more) | 0.1834 | -5.2% |
| AGE_OLD 6 | 0.2011 | +3.9% |
| tier-1 rank 384 | 0.2035 | +5.2% |
| tier-2 rank 192 | 0.1955 | +1.0% |
| tier-2 from age 10 | 0.1953 | +0.9% |
| full machinery at the final layer (MP there too) | 0.2032 | +5.0% |

## 3. Where the error is created, and what a per-neuron corrector can see (75 networks)

Per-layer label MSE minus the label noise (noise per neuron = post-ReLU variance / samples, from the chain's own
second moment; it reproduces the dataset's last-layer noise to 0.1%): the excess is unmeasurable (|excess| < 1e-8 against
noise 3-13e-7) through layer 7 and grows from layer 8 on: 0.5 / 4.6 / 6.8 / 5.5 / 6.0 / 8.0 / 10.7 / 12.8e-9 at layers
8-15. The error is made in the second half of the network, where the MP corrector runs.

A final-layer correction e_i ~ f(alpha_i, sigma_i, kappa3_i / sigma^3, kappa4_i / sigma^4) fitted on half the networks
does not transfer (held-out dMSE +1.2 / +0.9 / -7.1 / +0.4% for four splits with the phi-polynomial, +11 / +16 / -22 /
+2.5% with all features). Artyom_Mazur's random-projection result (18249, Proposition 1) predicts exactly this.
Closed.

## 4. The scale mode in our chain

il.miscusi (18248) shows that a common random scale on a layer's activations leaves every later mean unchanged and
inflates the carried cumulants along fixed patterns, for kappa3 2 eps Sym(C x m). Diagnostic on MLP 1015 (cn37, fast
mode): the share of the D21 slice explained by the pattern S_ij = var_i mu_j + 2 mu_i C_ij (pre-activation):

| layer | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| all sources, R^2 | 0.28 | 0.61 | 0.73 | 0.77 | 0.80 | 0.82 | 0.82 | 0.82 |
| accumulated eps | 1.7e-3 | 2.8e-3 | 3.5e-3 | 4.1e-3 | 4.6e-3 | 5.0e-3 | 5.1e-3 | 5.2e-3 |
| old tiers only, R^2 | | | | 0.50 | 0.64 | 0.76 | 0.80 | 0.82 |
| young sources only, R^2 | | | 0.73 | 0.72 | 0.71 | 0.68 | 0.61 | 0.58 |

Most of D21 is the scale mode. But the old tiers cannot be replaced by a scalar eps: their non-mode residual is
~20% of D21's norm at depth, and D21 must be right to ~2% (504aldo's eps^2 law). Removing the mode from a chain
whose kappa4 has no memory hurts, as il.miscusi found for V29 and e30_a. The exact spherical input (x = E|x| u: input
covariance (E|x|^2/n) I, the sphere's kappa4 -(2/(n+2)) Pair3(C x C) at layer 0) matches the Gaussian chain's layer-0
means to 5e-7, and then makes cn37 **+35%** worse (two networks: +25 / +45%). Our lambda law and learned operator are
calibrated to the mode. Closed.

## 5. Resolution of the 75-network gate

Any structural change moves each network's MSE by about +-10% (chaos-like: the move's correlation with the existing
error changes sign from network to network). The label-noise part of a paired difference is ~0.15% over 75 networks,
so the gate's standard error is the network spread, ~1.1% at 75 networks. Single changes worth 1% cannot be confirmed
locally, and the 100-network leaderboard has the same resolution. Changes are submitted only when they, or a bundle of them,
clear ~-2% with both seed-parity halves agreeing.

## 6. Smoke tests of portable step-level fixes (fast mode, MLPs 1015 + 1002, vs cn37)

| change | 1002 | 1015 | note |
|---|---|---|---|
| C0^3 Mehler term ((1/6) a3 b3 C0^3, all three pair channels) | -3.0% | +0.6% | 28 of 75 networks: -0.2% +- 1.9%; stopped (null) |
| marg2 (k=2 marginal Edgeworth corrections only for alpha <= -2) | -0.7% | -2.4% | **75 networks: -0.53% +- 0.47%** (40/75; held-out halves -0.4% on average): small, free |
| scale-mixture K22 pattern (+2 v C0^2 on the dyadic law) | -0.6% | -1.7% | 75 networks: +1.08% +- 1.02% (fitted strength ~0): closed |
| exact spherical input (layer 0) | +44.5% | +24.6% | closed (memoryless kappa4) |
| MP at the final layer (full last layer) | | -1.4% | +5.0% C: closed |
| **(3,1) slice read fixed in the (2,1) channel** | -3.9% | -6.4% | see below |
| D21 hub on live rows/columns only (Phi >= 0.01 / 0.001) | | +84% / +28% | closed: dead neurons' D21 entries are load-bearing (the MP reads them) |

The (3,1) read: kai_leal_gonzalez (18246, section 5) found that the (2,1) update reads the K13-oriented slot
transposed in its (3,2)-(1,1) entry (third derivative on i needs kappa(i,i,i,j) = wk431^T). It is harmless while
the slot is symmetric (V29: lambda C_off). Ours is not: the learned MP pair term Dl * outer(s1, s3) is added at layers
>= 8. Fixing the read moves the prediction 2.6e-5 rms (large), and the per-network effect spreads from -4% to
+22% over the first 16 networks. The MP pair network may have been trained with the transposed read and adapted
to it. The 75-network screen decided against it: **+8.1% +- 4.1% after 36 networks (better on 12/36)**, so it was
stopped there. The MP pair network evidently adapted to the transposed read (it is learned on the chain as shipped), so
the "correct" read is wrong for this operator. cn46 (built, emulator ok at C/B 0.1953, VmHWM 3.49 GB, smoke finite) is not
submitted. A warning about the gate: on the old tuning-12 set alone the same change read **-3.9%** (the rest +12.1%). That is the
cn40/cn41 selection trap again, and only the full set catches it.

## 7. Why cost cuts and blind fixes lose: the move-squared rule

For a variant with prediction move D = q - p and current error e = p - t, the paired change is
d = |D|^2 + 2 <e, D> (per network, per neuron mean). The first term is a sure loss; the second is a
correlation that changes sign from network to network. A truncation (cost cut) has no reason to correlate with
e, so its expected effect is +|D|^2 / MSE:

| variant | move (rms) | |D|^2 / MSE | measured (75-network gate) | dC |
|---|---|---|---|---|
| AGE_OLD 4 | 4.9e-5 | ~+18% | +15.3% +- 8.3% (24 networks) | -5.2% |
| marg2 | 9.0e-6 | ~+0.6% | -0.53% +- 0.47% (75 networks) | 0 |
| (3,1) read fix | 2.6e-5 | ~+5% | +8.1% +- 4.1% (36 networks) | 0 |
| C0^3 | 7.3e-6 | ~+0.4% | -0.2% +- 1.9% (28 networks) | ~0 |

With MSE ~1.3e-8 (rms error ~1.1e-4), a cost cut of x% pays only if it moves the prediction by less than about
sqrt(x/100) x 1.1e-4 rms: 2.5e-5 for a 5% cut. AGE_OLD 4 moves twice that. An accuracy change must anti-correlate
with the error systematically to beat its own |D|^2. That is what a fitted gain does (the linear-response fit
below), and what a blind correctness fix (the (3,1) read, against an operator trained on the old read) does
not do. On the old tuning-12 networks (the hardest, largest MSE) the |D|^2 term is relatively small, so every
variant looks better there than on the rest (the (3,1) fix -3.9% vs +12.1%, AGE_OLD 4 -6.2% vs +28%). That is the
cn40/cn41 selection effect, explained.

## 8. Where the wall goes (cn37, MLP 5000, in-process, one thread: 138.7 s)

Time between consecutive flopscope op records, attributed to the call site (`scripts/fast/prof2.py`): young D21
hubs 15-23%, young transports 13%, but **~18% of the wall in the nonlinear term program's 4-operand einsums**
(`"tij,ti,tj,tg->gij"` over up to 18 n x n planes, and the per-matrix groups) that bill ~0.1% of C. numpy runs them
unoptimised. Rewritten as elementwise accumulation they would free ~25-30 s of emulator wall, which is about what
level 6 on the young families costs (+37 s, -0.9% C).

Measured (cn47 = cn37 with those einsums as two broadcast scalings plus one matmul over the term axis; values to 5e-7):
in-process MLP 1015 43 s against ~60 s, but under the grader emulator (server/client IPC), back to back, **253.8 s against
266.4 s (-4.7%)**, at C/B 0.1957 against 0.1953 (+0.2%: the matmul form bills a little more). Level 6 adds ~19%, so
cn48 = cn47 + level 6 would sit ~13% above cn37's wall, past the cap. Neither is submitted. The rewrite stays in the
local screening copy, where it speeds the fast mode by ~25%.

## 9. Learned-operator recalibration by linear response (75 networks)

One run per knob (knob moved by delta), J = (p_knob - p_base) / delta per network and neuron. The least-squares
step x along the J's minimises the pooled label MSE. Held-out: fit on one half (seed parity, thirds, low/high
seeds), score on the other (`scripts/fast/jfit.py`).

| knob (run) | measured at the run | fitted step | held-out |
|---|---|---|---|
| marg2 (strength 1) | -0.53% +- 0.47% | x = 0.93 | -0.4% on average |
| K22 pattern (strength 1) | +1.08% +- 1.02% | x = 0.04 | ~0 |
| k34 kappa3 + kappa4 corrections x1.2 | +2.30% +- 0.94% | gain 0.97 | ~0 |
| MP node correction x1.2 | +1.07% +- 0.63% | gain 0.97 | ~0 |
| MP pair correction x0.8 | +0.37% +- 0.36% | gain 1.0 | ~0 |
| lambda x1.1 | +1.00% +- 0.53% | x0.997 | ~0 |
| all five jointly | | | overfits: held-out +0.2..+2.7% |

Every learned or fitted scale (k34, MP node, MP pair, lambda) sits at its optimum for our chain; recalibration has nothing to give. The overnight hard-six reading "x1.2: -0.9%" was the
selection effect again.

Small exact-ish cost trims (real op stream, MLP 5000, cn37 0.1935): one tier-2 range-finder pass 0.1934 (-0.05%),
feedback oversampling 8 0.1931 (-0.2%), residual rank 2 0.1926 (-0.5%), no feedback oversampling 0.1926 (-0.5%).
The bundle marg2 + residual rank 2 + oversampling 8 is screened directly.

## 10. Submitted: cn49b (336664)

| variant (vs cn37, 75 networks) | dMSE | dC | score (MSE x C) |
|---|---|---|---|
| marg2 | -0.53% +- 0.47% | 0 | -0.5% |
| marg2 + residual rank 2 + oversampling 8 | +0.61% +- 2.65% (move 2.4e-5) | -0.7% | noise; rank 2 moves too much |
| **marg2 + oversampling 8 (cn49b)** | **-0.79% +- 0.46%** | **-0.2%** | **~-1.0%** |
| oversampling 8 against marg2 alone | -0.26% +- 0.13% (all four halves -0.23..-0.28%) | -0.2% | |

cn49b: cn49 (cn37 + marg2, built with `build_step.py`; emulator ok at C/B 0.1953, VmHWM 3.46 GB, smoke finite) with the feedback
sketch oversampling at 8. Fast-mode values match the screen to 6e-7. Submitted 05:58 UTC as 336664: **graded 3.5121e-9**
(raw 1.8204e-8, C/B 0.1929): -0.40% vs cn37 (raw -0.17%, C -0.23%). The new best; the local -0.8% raw is inside the
leaderboard's ~0.5% noise.

Small-move knobs resolve finely: the gate's se scales with the move (0.13% at a 2.5e-6 move), so a sweep of the
remaining small knobs is queued on top of cn49b (oversampling 0 / 4, tier-2 range-finder passes 1 / 3, row and column dead
thresholds 0.008 / 0.012). A knob is accepted only if all four splits agree in sign.

| knob (on cn49b's settings) | dMSE vs oversampling 8 | dC |
|---|---|---|
| oversampling 0 | +0.31% +- 0.28% | -0.3% |
| oversampling 4 | +0.22% +- 0.19% | ~-0.15% |
| oversampling 16 (cn37's) | +0.26% +- 0.13% | +0.2% |

Oversampling 8 is the best of the four (the shape is not monotone, so part of the 16 -> 8 gain may be chance; its
-0.2% C is certain).

| knob (on cn49b's settings) | dMSE vs cn49b | dC (MLP 5000, cn49b 0.1931) | verdict |
|---|---|---|---|
| tier-2 range finder 1 pass | +0.58% +- 0.25% | -0.05% | out |
| tier-2 range finder 3 passes | -0.08% +- 0.11% | ~+0.05% | null |
| rows Phi < 0.008 | +0.00% +- 0.33% | ~+0.15% | out |
| columns Phi < 0.008 | -0.80% +- 0.49% (all halves -) | +0.73% (0.1945) | net ~0 |
| columns Phi < 0.006 | -1.45% +- 0.78% (all halves -) | +0.88% (0.1948) | net -0.6% |
| rows Phi < 0.012 | -0.15% +- 0.36% | -0.21% (0.1927) | net -0.4% |
| rows Phi < 0.015 | +0.50% +- 0.65% | -0.41% (0.1923) | ~0 |
| columns Phi < 0.012 | +0.24% +- 0.52% | -0.26% (0.1926) | ~0 |
| **columns 0.006 + rows 0.012 (cn50)** | **-1.46% +- 0.98% (all halves -)** | **+0.67% (0.1944)** | **net ~-0.8%** |
| columns Phi < 0.004 | -1.15% +- 0.93% | +1.35% (0.1957) | ~+0.2% |

cn50 checks: emulator ok, VmHWM 3.46 GB, smoke finite, values match the screen to 6e-7; quiet back-to-back emulator
wall 219.6 / 220.9 s against cn49b's 226.7 / 228.5 s (3% faster: more rows compacted). Submitted 09:15 UTC as 336695:
**graded 3.5175e-9** (raw 1.8116e-8, -0.48% vs cn49b; C/B 0.1942, +0.64%): a tie with cn49b inside the leaderboard's noise.
The local -1.46% raw shrank to -0.48% on the public set, the same shrinkage as cn49b (-0.79% -> -0.17%).

## 11. Tier knobs on cn50's settings (75 networks, vs cn50; cost on MLP 5000, cn50 0.1944)

| knob | dMSE | dC | verdict |
|---|---|---|---|
| tier-1 basis never rebuilt (CN_KEEP=all) | +5.40% +- 1.28% (move 2.4e-5, move^2 rule ~+4.4%) | -1.65% | out |
| tier 2 from age 8 | +3.20% +- 1.03% | -1.13% | out |
| tier-2 rank 144 | +1.71% +- 0.52% | -0.2% | out |
| tier-2 rank 192 | -1.07% +- 0.44% (48/75) | ~+1.0% | net ~0 |
| tier 2 from age 10 | -0.65% +- 0.41% | ~+0.9% | net ~+0.3% |

Every tier knob sits at its optimum in both directions.

## 12. Calibration: local gate against the leaderboard

| candidate | local raw change (75 networks) | leaderboard raw change | ratio |
|---|---|---|---|
| cn49b vs cn37 | -0.79% +- 0.46% | -0.17% | 0.2 |
| cn50 vs cn49b | -1.46% +- 0.98% | -0.48% | 0.3 |

The leaderboard sees about a third of the local gain. Both picks were the best of a sweep (winner's curse), and the local
labels (5e5-1e6 samples) differ from the 1e9-sample public truth. Artyom_Mazur reports the same ratio for fitted scalars
(18249, section 6.5: "a scalar fitted on ten networks arrives at a third of its strength on a hundred"). Rule for
the last slots: only a local net gain of ~3% or more is worth a submission.

## 13. Where the day ends

Leaderboard: cn49b **3.5121e-9** (best), cn50 3.5175e-9 (lowest raw, 1.8116e-8), cn37 3.5262e-9. Final selection:
cn49b + cn50 (README). Two of today's slots were left unused, because no candidate clears the ~3% local bar that the
1/3 shrinkage implies.

Measured today on 75 networks (fast gate), with the conclusion each one supports:
* every learned or fitted scale is at its optimum: k34 kappa3/kappa4, MP node, MP pair, lambda;
* every truncation knob is at its optimum in both directions: AGE_OLD, tier-2 age and rank, tier-1 rebuild schedule,
  residual rank, feedback oversampling, row and column thresholds. The move-squared rule predicts the losing side from
  label-free moves alone;
* closure patches from the public write-ups do not transfer to a memoryless-kappa4 chain with a learned operator:
  C0^3 null, K22 scale-mixture pattern +1%, spherical input +35%, the (3,1) read fix +8%. Only marg2 transfers (small);
* deeper Strassen is wall-capped (cn44 and the earlier level-6 builds failed at the 120 s per-network cap), and the
  term-program einsum rewrite frees only 4.7% of emulator wall.

What would move the board, all of it beyond a day's work: a carried kappa4 core with scale-mode extraction
(il.miscusi 18248: x0.66 in a lean chain, but +25% cost in production), the young-kappa4 "3+2" tree reader (Artyom 18249:
x0.70 raw, cost about what it buys in their representation), and a kappa3 state that is not a list of sources (what
the leaders' bill implies; unpublished).

## 14. The teammate's note-16 submission (rg-era base) against our line

Their archive (estimator diff against `v5b_rg`): R_FB 1, R_RES 8, the covariance sandwich at 5 Strassen levels (leaf 16),
and the young D21 hub's LA/LP stacks in one batch-4 Strassen call (HUB_SB = 4), on 504aldo's original tier law (age 4 /
384, age 7 / 224) without dead-neuron masks. Ported to our line (cn50 settings):

| change | dMSE (75 networks) | dC (MLP 5000, cn50 0.1944) | verdict |
|---|---|---|---|
| R_FB 2 -> 1 | +1.47% +- 0.72% (32 networks, all halves +) | -0.46% (0.1935) | out (our cn21-era reading holds) |
| R_RES 8 | (we are at 4; 8 vs 4 tied on the leaderboard, cn36 vs cn37) | | |
| covariance level 5 + batch-4 hub where the cap allows (cn51) | exact (rounding) | -0.1% (0.1942); emulator 0.1953 vs 0.1956 | wall +4% (193.6 / 190.4 s vs 187.9 / 180.2 s): not worth the cap risk |
| batch-4 hub alone on cn49b (cn49c) | exact | emulator 0.1950 vs 0.1948 (+0.1%: the staging copies cost more than the saved recombination) | out |

The batch-4 hub needs care in our kernel: at 5 levels the depth-3 combo stacks are (7^4, 4, n/16, w/16) float32, and those exceed
the grader's 134.9 MB per-array cap for w > 864 (157 MB at full width). cn51 therefore uses batch 4 only on narrowed chunks
(w <= 864: 132.8 MB). The teammate's rg-based file runs the hub at full width, so its batch-4 calls may hit that cap
on the grader. That is worth checking against their grade.

Nothing in the note transfers profitably to cn49b / cn50: our line already has the cheap thin legs (R_FB 2 is better than 1
here), and the kernel changes are a wash once the per-array cap and the wall cap are respected. The final pair stays
cn49b + cn50.
