# Experiments 2026-10-16: knob elasticities and dead neurons

Base: `candidates/v5b_rg` (graded 4.1705e-9, raw 1.753e-8, C/B 0.2378). Local numbers are
paired against rg's own predictions on the 11 validation MLPs (seeds 5000-5010, 1e6-sample
Monte Carlo targets, noise-corrected final-layer MSE) unless noted. Label noise variance is
6-10x the MSE (per-network MSE has a 26% standard error over 11 MLPs), so only paired
differences mean anything locally; their standard error is 2-5%.

## 1. Knob elasticities: the tiers sit at their optimum

The score is MSE x C, so a knob is worth moving while the elasticity e = dlnMSE/dlnC is below
-1 (more cost buys more than proportionally less error) or above -1 in the other direction.

| change | C/B | MSE vs rg | e | verdict |
|---|---|---|---|---|
| AGE_OLD 3 (earlier measurement) | -3.8% | +17% | -4.2 | worse |
| AGE_OLD 5 (sources stay dense one more layer) | +3.6% | -3.2% +- 2.8% | -0.9 | neutral |
| R_OLD 320 (tier-1 rank) | -3.4% | +7.2% +- 4.7% | -2.1 | worse |
| tier-2 rank 192 (graded, 336359) | -1.9% | +1.9% (LB) | -1.0 | neutral |

Every measured direction has e ~ -1 or steeper on the downside: the age gates and ranks are
at the score optimum (504aldo's age law: rank 3n/8 at age 4, one notch lower is a cliff).
Knob moves cannot buy rank; only a representation change can.

## 2. Dead neurons: which parts of the chain they can leave

At depth the pre-activation means spread (alpha = mu/sigma quantiles at layer 12: 5% -4.8,
95% +5.0), so 16-23% of the neurons have Phi(alpha) < 1e-2 from layer 8 on (none at layers
0-1, the input is centred). A ReLU that almost never fires has h ~ 0, so in exact arithmetic
it carries almost nothing. Masking experiment (same cost, zeros where a pruned chain would skip
work), Phi < 1e-2:

| what is zeroed for dead neurons | MSE vs rg | reading |
|---|---|---|
| covariance rows/cols of C (post-ReLU) | 5x worse | second order must be exact to ~1e-4 relative; 1% of the variance is not negligible |
| D21 rows/cols at the ReLU step | 14x worse | the Wick coefficients of tail neurons are large (alpha^k phi(alpha) ~ O(1) for k = 5 at alpha = -3) and cancel against D21: the expansion needs these entries |
| post-wick leg rows (inner dimension of the next transports), C exact | +2.6% +- 1.6% (on top of the column mask) | rows scaled by w1 < 0.01 still carry ~1% of D21 (eps^2 law: +2.6% MSE ~ eps = 1.1%) |
| **a source's columns for neurons dead at its birth** | **-2.4% +- 2.1% (11 MLPs, 8 of 11 better)** | see below |

Update after 22 more MLPs (train seeds 1000-1021, which the learned operator was fitted on):
the column mask is +2.4% +- 2.2% over all 33, i.e. val -2.4%, train ~+5%. So it is not a
demonstrated accuracy gain; read it as neutral within noise (possibly the operator, fitted
on the unmasked chain for those very MLPs, prefers the unmasked inputs there). What follows
was written after the first 11 MLPs and is kept as the hypothesis it was:

The column result is the only one that is an improvement, and it is an improvement in
accuracy, not just a free cut. The prediction moves by rms 1.7e-5 (the error itself is rms
~1.2e-4); a random move of that size would add ~3e-10 of MSE, the measured change is -3.2e-10,
so the removed content is anti-correlated with the error. Mechanism: a newborn source's
column a carries the third cumulant created at neuron a's kink, weighted by the Gaussian-closure
derivatives w2 = E[ReLU''] ~ phi(alpha)/sigma, the residual skew S3c of the unit and the (1,2)
Wick coefficient. For a unit 2.3+ sigma below its kink these are Edgeworth-tail quantities:
the series' relative error grows like He_k(alpha) in the tail, so the closure injects noise
there rather than signal. Dropping them is tail truncation of the expansion.

Consequences used below: (i) the mask is free accuracy (`candidates/v5b_cm`), and (ii) a
source's dead columns are then exact zeros for its whole life, so every product that contracts
over or produces those columns (young transports, young D21 hub, old-tier formings, the joiner)
can run on the live block only (`candidates/v5b_cn`).

## 3. The chain is chaotic at float32 level (why local comparisons are so noisy)

Same estimator (rg), MLP 5000, three perturbations that change nothing in exact arithmetic or
almost nothing:

| perturbation | max abs change of the final means | MSE of this MLP |
|---|---|---|
| none | -- | 1.442e-9 |
| hidden neurons of layer 1 relabelled (equivalent network) | 5e-5 | 1.359e-9 |
| hidden neurons of layer 5 relabelled | 7e-5 | 1.518e-9 |
| W_1 multiplied by (1 + 1e-7 N(0,1)) entrywise | 4e-5 | 1.094e-9 |

A rounding-level perturbation grows ~2x per layer to ~4e-5 at the output (the error itself
is rms ~1.2e-4 on an average MLP). Relabelling also changes the range-finder sketches (they
are slices of the layer weights), which adds sketch dependence on top. Consequences: (i) a
column relabelling or any float reordering moves the label-based per-MLP MSE by up to +-25% on
the best-predicted MLPs, so local paired tests resolve only a few percent (correction, section
14: that swing is the cross term with the labels' coherent noise. Against clean labels a
chaos-level change delta moves an MLP's MSE by |delta|^2 ~ 1.2e-10 plus a cross term
2 e.delta/n of sd ~1e-10, and the 100-MLP mean by ~0.05%. So leaderboard differences of
~1% between variants are real);
(ii) the expected cost of this noise on the mean MSE is its variance, ~1e-10 = 0.6%, so it is
not a lever (504aldo measured float64 as +0.004%); (iii) "bitwise parity" is the wrong test
for an exact rewrite of this chain: compare against the chaos level instead.

## 4. v5b_cn / v5b_cn2: the live-column chain

`v5b_cn`: every source stores its columns live-first (an exact relabelling of its birth
neurons, with all column-indexed vectors and the k34 feature sums permuted alike) and
records its live count rounded up to 32 (>= 768). The young-leg transports and the young D21
hub run on the first max(live) columns of the young stack; the tails stay zero. `v5b_cn2`
also runs the joiner (range finder, projection, Gram) on its live block and the tier-1
formings and inner contraction on the tier's live block. Both match `v5b_cm` to the chaos
level (max 4-6e-5 on the final means, same as relabelling rg's own network).

Strassen scratch pools are now one flat buffer per role with a reshaped view per shape (a
role is scratch inside one call). Narrowed shapes therefore add no memory; as a side effect
the duplicate pools of the old shape-keyed scheme are gone:

| | C/B (2 MLPs) | emulator wall (cold / warm) | VmHWM |
|---|---|---|---|
| v5b_rg (graded 4.1705e-9) | 0.2378 | 145-178 s | 6.12 GB |
| v5b_cn2 | 0.2303-0.2318 | 164 / 147 s | **3.33 GB** |

rg was at the grader's ~6.1 GB pass line; cn2 has 2.8 GB of headroom.

## 5. Threshold frontier and refinements (11 validation MLPs)

| variant | C/B | MSE vs rg | reading |
|---|---|---|---|
| cn3 (columns Phi < 0.01, rows Phi < 0.002) | 0.2258-0.2279 | -2.9% +- 2.1% | submitted (336414) |
| cn4 (columns Phi < 0.02, rows Phi < 0.01, narrowest block 704) | 0.2201-0.2208 | +2.8% (+5.8% +- 3.3% vs cn3) | score ~3% worse than cn3: neurons with Phi in [0.002, 0.02] still carry ~1% of D21 |
| cn5 (cn3, each batch-2 chunk of the young stack on its own width) | 0.2250-0.2271 | exact (chaos-level diffs) | -0.35%: kept for a later candidate |
| cn3 + AGE_OLD 5 | 0.2337 | (rg: -3.2% +- 2.8%) | +3.5% cost, still neutral |
| cn3 + R_OLD 448 | 0.2386 | -- | +5.7% cost |

Where the final MSE lives (rg, 75 local MLPs, per-neuron noise-corrected): output neurons
with mean 0.01-2 carry 96% of it (0.1-0.5: 37%), saturated outputs (mean > 2, 7% of the
neurons) 2%, dead ones nothing. The first moment is therefore accurate; the error is the
variance and the third/fourth cumulants at the kink, i.e. the closure, not the
bookkeeping of sources.

## 6. Graded (public leaderboard, 100 MLPs, noise-free)

| submission | score | raw MSE | C/B |
|---|---|---|---|
| 336295 v5b_rg | 4.1705e-9 | 1.7534e-8 | 0.2378 |
| 336410 v5b_cn2 | 4.0540e-9 | 1.7637e-8 | 0.2299 |
| 336414 v5b_cn3 | **3.9756e-9** | 1.7637e-8 | 0.2254 |

Attribution: rows are exactly free (cn2 and cn3 agree to 4 digits in raw MSE); the dead-column
mask costs +0.6% raw (local: val -2.4%, train +5%, all within noise) for -3.3% C/B.

## 7. The joiner's range finder: warm start (research: randomized range finders, subspace iteration)

The tier-1 basis is rebuilt at every join by a one-pass randomized range finder (Halko,
Martinsson, Tropp 2011) on the weighted leg Gram G = A dA A^T + P dP P^T + Qp S Qp^T, with
a fixed sketch (the first 384 columns of the next layer's weight) and no oversampling. Captured
offline at layers 7/9/12/14 of MLP 5000 (cn3), relative projection error of the joiner's legs
and of the old content (in span Qp, core S):

| construction (rank 384) | cost | A (l7 / l14) | P (l7 / l14) | old content (l7 / l14) |
|---|---|---|---|---|
| top-384 eigenvectors of G (optimal for G) | -- | 0.0196 / 0.0077 | 0.0510 / 0.0335 | 0.0118 / 0.0038 |
| shipped: orth(G Om), Om = weight slice | 1 | 0.0241 / 0.0096 | 0.0598 / 0.0407 | 0.0178 / 0.0057 |
| **warm start: orth(G Qp)** | **1** | 0.0218 / 0.0085 | 0.0558 / 0.0375 | **0.0102 / 0.0034** |
| Nystrom on [Qp, 32 random] -> 384 | ~1.1 + eig | 0.0206 / 0.0080 | 0.0530 / 0.0348 | 0.0110 / 0.0036 |
| two passes (QPASS 2) | 2 | 0.0204 / 0.0081 | 0.0529 / 0.0351 | 0.0130 / 0.0041 |
| shipped at rank 352 / 320 | 0.92 / 0.83 | 0.034 / 0.048 (l7) | 0.082 / 0.110 | 0.028 / 0.042 |

Trace energy is useless as a yardstick here (the top 384 eigenvalues hold 99.97% of tr G); the
2% D21 tolerance lives in the tail. Why the warm start works: every source's legs are suffix
products of the same transport chain, and products of random matrices concentrate on a common
leading (Lyapunov) subspace, so the previous basis already contains most of the joiner's
dominant directions, and it keeps the old content, which the shipped sketch re-projects with a
fresh random sketch at every join (old-content loss per join -43%). It does not buy a rank cut:
one rank notch (384 -> 352) costs +40% error, the warm start saves 10-40%.

Downstream, label-free: rms distance of the final means to a no-truncation reference (same
chain, all sources dense, C/B 0.262-0.266) on MLPs 5000/5001: shipped 3.56e-5 / 2.79e-5, warm
3.09e-5 / 2.37e-5 (chaos floor ~1e-5). Truncation is ~5% of the MSE (504aldo: +4.6% raw), so
-15% of truncation error is worth ~1.5% of MSE at zero cost. The nested tier-2 range finder is
warm-started from its own previous sub-basis as well.

## 8. Age-law tier shift, scored against the no-truncation chain

All warm-started; rms distance of the final means to the all-dense chain (MLPs 5000 / 5001) and C/B:

| variant | distance | C/B |
|---|---|---|
| cn6 cold sketch (= cn5 numerics) | 3.56e-5 / 2.79e-5 | 0.2221 / 0.2234 |
| cn6 (warm) | 3.09e-5 / 2.37e-5 | 0.2221 / 0.2234 |
| R_OLD 320 | 5.27e-5 / 4.47e-5 | 0.2104 / 0.2135 |
| R_OLD 320, R_OLD2 192 | 5.62e-5 / 4.85e-5 | 0.2067 / 0.2098 |
| AGE_OLD 5, R_OLD 320 | 2.80e-5 / 2.28e-5 | 0.2203 / 0.2231 |
| **AGE_OLD 5, R_OLD 320, AGE_OLD2 8, R_OLD2 192 (cn7)** | **2.72e-5 / 2.23e-5** | **0.2183 / 0.2211** |
| same two shifts without the warm start | 3.18e-5 / 2.51e-5 | -- |

A lower rank at the same age falls off the cliff; the same lower rank one age later is better
than the shipped tiers, because legs concentrate with every transport (rank 3n/8 at age 4 ->
5n/16 at age 5, 7n/32 at age 7 -> ~3n/16 at age 8). The age-law shift and the warm start
stack (-10% and -13% of the distance alone, -22% together). The label-based local test of the
same shift read +2.5% +- 4.4%: noise, the deterministic yardstick resolves it.

Emulator wall, measured back to back on a loaded container (absolute numbers drift +50-60%
over the day): cn3 243 / 208 s, cn6 239 / 225 s, cn7 255 / 228 s, i.e. cn7 +5-10% over cn3
(which graded fine). VmHWM 3.45-3.50 GB.

Frontier around cn7 (same yardstick, C/B relative to cn7; truncation error^2 relative to the
shipped sketch w0): tier-2 rank 176 / 160 at age 8: -0.6% / -1.2% C, 0.77 / 0.94; age 9 at 160:
-0.2%, 0.62; tier-1 rank 288 / 352: -2.3% / +2.4%, 1.09 / 0.42; ages 6 / 9 at 256 / 160: -0.3%,
0.64; keeping the transported basis at every join (no range finder): -3.1%, 1.27; at every other
join: -1.5%, 0.74. Calibration: rms chaos between equivalent runs is 1.0-1.2e-5, the truncation
distance 2.2-3.6e-5, i.e. truncation error^2 ~5-9e-10 per MLP, ~3-5% of the leaderboard MSE.
With that, every neighbour of cn7 scores within +-0.7% (rank 352: +1.3%): cn7 sits at the
optimum of the tier family. Four MLPs (5000-5003): cn7's truncation error^2 is 0.58-0.77 of
the shipped one.

## 9. D21 feedback legs: oversampling

The rank-16 D21 feedback (V18) is a 16-column sketch with one power iteration and no
oversampling. Distance to a rank-64 feedback reference (MLPs 5000 / 5001): shipped 1.59 / 1.48e-5,
16 oversampled to 32 then truncated by SVD 1.48 / 1.35e-5, rank 32 1.03 / 0.99e-5 (chaos floor
~1.1e-5): the feedback truncation is a smaller error than the tiers. MSE moves monotonically
with feedback quality on both MLPs (5001: 6.32 / 5.80 / 5.75 / 5.57e-9 for 16 / 16-oversampled
/ 32 / 64), but per-MLP MSE is chaos-sensitive. Rank 32 costs +6.6% (score-negative, as
504aldo found); oversampling costs ~0.2%: cn8.

## 10. Graded: warm start, age-law tiers, feedback oversampling

| submission | score | raw MSE | C/B |
|---|---|---|---|
| 336414 v5b_cn3 | 3.9756e-9 | 1.7637e-8 | 0.2254 |
| 336447 v5b_cn6 (warm start, exact polish) | 3.9041e-9 | 1.7575e-8 | 0.2221 |
| 336448 v5b_cn7 (age gates 5 / 8, ranks 320 / 192) | 3.8118e-9 | 1.7443e-8 | 0.2185 |
| 336459 v5b_cn8 (feedback oversampled) | **3.8110e-9** | 1.7365e-8 | 0.2195 |

Calibration of the yardstick: cn3 -> cn7 cut the truncation error^2 by 39% (ratio to the
shipped sketch 1.0 -> 0.61) and the raw MSE by 1.1%, so truncation is only ~2.8% of the
leaderboard MSE (cn3 -> cn6 alone reads 1.4%, cn6 -> cn7 5.5%: noisy two-MLP proxies). The
feedback oversampling bought -0.45% raw for +0.43% C: neutral. Consequence: at a 2.8%
share, a cheaper old tier is worth it whenever 1% of cost raises the error^2 ratio by less than
~0.36; the measured moves (keep the basis on alternate joins 0.09 per 1%, tier-1 rank 288 0.21,
tier-2 rank 160 0.27, keep the basis always 0.21) all qualify.

## 11. Cheaper old tier (cn9, cn10)

Score model: score ~ (1 + s (rho - rho_cn7)) x C with rho = truncation error^2 / shipped and the
truncation share s bracketed 1.4-5.5% by the graded runs. Variants (2 MLPs; rho, C vs cn7,
score change at s = 1.4 / 2.8 / 5.5%):

| variant | rho | C | score |
|---|---|---|---|
| cn7 | 0.61 | 1 | 0 |
| basis rebuilt on alternate joins (kalt) | 0.74 | 0.985 | -1.3 / -1.2 / -0.8% |
| kalt + tier 2 from age 9 at 160 (**cn9**) | 0.73 | 0.983 | -1.6 / -1.4 / -1.1% |
| kalt + tier-1 rank 288 | 1.29 | 0.964 | -2.7 / -1.8 / 0.0% |
| kalt + 288 + age 9 / 160 (**cn10**, measured as cn10) | ~1.3 | 0.962 | ~ -2.8 / -1.9 / -0.1% |
| basis never rebuilt (kall) | 1.28 | 0.969 | -2.2 / -1.3 / +0.5% |
| tier-1 256 / tier-2 128 / kall + 256 + 128 | 1.98 / 1.81 / 4.98 | 0.955 / 0.975 / 0.908 | ... / +12.6% at s = 5.5% |

Grader cap: with tier 2 from age 9, tier 1 holds four sources and its staged batch of 8 makes
a depth-2 Strassen scratch of (343, 8, 128, 128) = 179.8 MB > 134.9 MB (cn7's batch of 6 was
exactly 134,873,088 bytes, the largest size known to pass). Tier-1 families now chunk at 6.
Emulator 218-248 s on the loaded container (cn8 246 s alongside), VmHWM 3.44-3.46 GB.

## 12. Row and column thresholds on the yardstick (references: the same chain without row
compaction / without the column mask; MLPs 5000 / 5001; chaos floor ~1.0-1.1e-5)

| setting | distance to its exact reference | C/B (vs cn8 0.2193 / 0.2221) |
|---|---|---|
| no row compaction (reference) | -- | 0.2274 / 0.2286 |
| rows Phi < 0.002 (shipped; exactly free on the leaderboard: cn2 = cn3 raw) | 0.93 / 0.75e-5 | 0.2193 / 0.2221 |
| rows Phi < 0.005 | 0.96 / 0.81e-5 | 0.2178 / 0.2202 |
| rows Phi < 0.01 | 1.22 / 1.03e-5 | 0.2162 / 0.2184 |
| rows Phi < 0.02 | 2.04 / 2.07e-5 | 0.2144 / 0.2160 |
| no column mask (reference) | -- | 0.2287 / 0.2302 |
| columns Phi < 0.01 (shipped) | 1.69 / 1.38e-5 | 0.2193 / 0.2221 |
| columns Phi < 0.02 | 3.59 / 2.64e-5 | 0.2165 / 0.2187 |

Calibration: the column mask's excess squared distance (~1.2e-10 above the chaos floor) matches
its measured leaderboard cost (+0.6% raw = 1.1e-10); for the tier changes the factor was ~0.6.
So the yardstick predicts leaderboard MSE changes to within a factor ~1.5. Rows at 0.01: ~+0.3%
MSE for -1.5% C (net ~-1.2%); rows at 0.02: +1.7% for -2.5%; columns at 0.02: ~+5% for -1.4%
(why cn4 lost). The earlier masking test that read rows at 0.01 as +2.6% +- 1.6% was noise.

Graded: 336496 v5b_cn11a (cn9 + rows at 0.01) **3.6860e-9**, raw 1.7367e-8 (cn9 1.7391e-8: the
row cut costs nothing measurable), C/B 0.2122 (-1.6%): -1.7% vs cn9, -11.6% vs rg. 336497
v5b_cn11b (cn10 + rows at 0.01) 3.7086e-9, raw 1.7897e-8, C/B 0.2072. The yardstick's +0.3% for
rows at 0.01 was an overestimate (row compaction moves the chain mostly at the chaos level).

Graded: 336483 v5b_cn9 **3.7487e-9** (raw 1.7391e-8, C/B 0.2156: -1.6% vs cn8, -10.1% vs rg);
336484 v5b_cn10 3.7699e-9 (raw 1.7916e-8, C/B 0.2104). cn9's changes read as predicted (+0.15%
raw for -1.8% C, implied truncation share ~1.2%), but tier-1 rank 288 cost +3.0% raw for -2.4% C
(implied share ~5%): the two-MLP yardstick underestimates the step below 320, which is close to
the rank cliff on the leaderboard's harder MLPs. Keep 320.

## 13. S21 residual leg rank (6 validation MLPs, labels; per-MLP MSE swings +-25% from chaos)

| R_RES | C/B (MLP 5000) | mean MSE vs 16 |
|---|---|---|
| 16 (shipped) | 0.2075 | -- |
| 8 | 0.2037 | -0.2% (per MLP -43 .. +11%) |
| 4 | 0.2018 | -3.0% (per MLP -45 .. +24%) |

Inconclusive locally; 504aldo's ladder (16 / 32 / 64 -> 2.35 / 2.39 / 2.41e-8) also falls toward
lower rank. A leaderboard probe.

## 14. The per-network "scale error" is label noise (closes note 16 item 9)

By ReLU homogeneity the final layer is h_L(x) = |x| g(x/|x|), and the network maps input
directions to output scales that differ by ~9% rms. So the Monte Carlo labels carry a noise
mode along mu that is coherent across all 1024 neurons. Measured on independent pilot samples
(32k inputs per MLP), the label noise of the least-squares scale factor
c_m = (p . t)/(p . p) has sd 1.2e-4 at 5e5 samples (8.6e-5 at 1e6). The independent-neuron
model gives 1.9e-5. The observed spread of c_m over 33 networks (cn3 predictions) is
1.07e-4: all of it is label noise. The "oracle -38%" of a per-network rescale was the fit
absorbing this mode; it also turns negative after the noise floor is subtracted (-143% on
cn3), which only correlated noise can do. No chain error is left to correct along mu.

For local A/B tests this sets the floor. The noise term of a paired difference,
2 (q - p) . (mu - t) / n, has sd ~5e-10 per MLP for a typical chaos-level |q - p|. The
covariance of h_L has one 8-11% mode and a flat bulk (3.5%, 3.3%, 3.0%, ...), so projecting
out the top 1-16 noise PCs leaves the per-MLP spread at +-10-20% (R_RES 8 / 4 vs 16 on six
validation MLPs: k = 0 / 1 / 4 / 16 PCs removed -> -0.2 / +0.1 / +0.5 / +1.4% and
-3.1 / -1.2 / +1.6 / +2.1%, se ~2-3%). Effects under ~3% need the leaderboard, or the
deterministic yardstick when the change truncates something a reference can hold exactly.

## 15. Old-source D21 is load-bearing; exact tier-2 eigensolve is a wash

Yardstick (rms distance of the final means to the all-dense chain, MLPs 5000 / 5001; chaos floor
~1.0-1.2e-5):

| variant | distance | C/B (5000) |
|---|---|---|
| cn9 | 3.03 / 2.72e-5 | 0.2154 |
| cn11a | 3.27 / 2.80e-5 | 0.2125 |
| cn16 = cn11a + tier-2 sub-basis from an exact eigh of its 320 x 320 core (instead of 2 warm passes) | 3.19 / 2.76e-5 | 0.2129 |
| cn16 without the tier-2 D21 hub (ages >= 10 still in D3) | 11.1 / 9.3e-5 | 0.2093 |
| cn16 without the tier-1 and tier-2 D21 hubs | 39.1 / 38.3e-5 | 0.2076 |

* The warm two-pass subspace iteration is already within ~4% (truncation error^2) of the
  optimal tier-2 subspace. At a 1-5% truncation share that is <= 0.2% of MSE, against +0.2% C:
  not submitted.
* Old sources cannot shed their (2,1) slice. Without the tier-2 hub the truncation error^2
  grows ~12x, roughly +60% of the leaderboard MSE. Without both old hubs it grows ~150x. The
  cost-saving gate `v5b_cn17` (CN_D21AGE) is closed. (Small-width smoke runs showed the same:
  at n = 128 the gate at age 5 moves the final mean by 2.5e-3.) The old tiers' D21 reaches
  the means through the covariance update (the (2,1) terms of the term program), and
  covariance errors are not damped. So the old third-order content cannot be retired
  cheaply. Any cut there must keep its D21 accurate.

## 16. Thin legs: one dominant singular pair on a flat bulk

Singular values of the S21 residual Rres (compressed to rank R_RES at birth) and of D21 (fed back
at rank R_FB), MLP 5000, chain = cn11a:

| layer | Rres sv 1..4, 8, 16, 32 | D21 sv 1..4, 8, 16, 32 | top-1 / top-16 share of the top-64 energy (Rres, D21) |
|---|---|---|---|
| 3 | 0.372 0.080 0.039 0.036, 0.032, 0.029, 0.025 | 4.94 0.53 0.50 0.48, 0.45, 0.41, 0.34 | 0.74 / 0.86, 0.75 / 0.85 |
| 8 | 0.257 0.033 0.029 0.026, 0.023, 0.018, 0.013 | 4.77 0.58 0.50 0.48, 0.41, 0.33, 0.23 | 0.81 / 0.91, 0.82 / 0.92 |
| 13 | 0.191 0.023 0.021 0.020, 0.017, 0.012, 0.009 | 4.20 0.53 0.50 0.48, 0.36, 0.26, 0.17 | 0.83 / 0.93, 0.85 / 0.95 |

Both are one coherent rank-1 component plus a bulk that decays only slowly (the bulk carries
15-25% of the top-64 energy and continues past 64). A rank-r range finder takes the dominant pair
and r - 1 bulk directions, each worth ~0.1-0.4% of the energy. Ranks 4 / 8 / 16 / 32 / 64 therefore
differ only by tiny slices of the bulk, which matches 504aldo's flat ladders (R_RES 16/32/64 ->
2.35/2.39/2.41e-8; R_FB 8/16/32 -> 2.17/2.14/2.14e-8: the 8 -> 16 step is within their two-dump
noise). The thin-leg einsums scale with r and are ~6% of C, so cn12-cn15 and `v5b_cn18`
(R_RES 4 + R_FB 4) go to the leaderboard.

How far each rank moves the chain (rms change of the final means vs cn11a, chaos floor 1.0-1.2e-5;
distance to the all-dense chain in brackets, cn11a 3.27 / 2.80e-5):

| variant | MLP 5000 | MLP 5001 | C/B (5000) |
|---|---|---|---|
| R_FB 8 (cn14) | 1.16e-5 (3.31e-5) | 1.04e-5 (2.88e-5) | 0.2052 |
| R_RES 4 (cn13) | 2.88e-5 (3.96e-5) | 3.35e-5 (4.28e-5) | 0.2068 |
| R_RES 4 + R_FB 4 (cn18) | 3.18e-5 (4.10e-5) | 3.64e-5 (4.41e-5) | 0.1959 |
| R_RES 8 vs 16 (older base, six MLPs) | 1.4-2.0e-5 | | |

R_FB 8 is free (chaos level). The residual rank is not: R_RES 8 moves the chain ~1e-10 in squared
terms (<= 0.8% of the leaderboard MSE) and R_RES 4 ~6e-10 (<= 3.5%), in a direction the yardstick
cannot sign (the reference is itself at rank 16, and 504aldo's ladder suggests the higher ranks are
not closer to the truth). The leaderboard decides: cn14 (R_FB 8), cn12 / cn13 (R_RES 8 / 4),
cn15 (8 + 8), cn18 (4 + 4).

Feedback ladder on the same yardstick (R_RES 16; move vs cn11a / distance to the all-dense chain,
MLPs 5000 / 5001):

| R_FB | move vs cn11a | distance | C/B (5000) |
|---|---|---|---|
| 16 (cn11a) | -- | 3.27 / 2.80e-5 | 0.2125 |
| 8 (cn14) | 1.16 / 1.04e-5 | 3.31 / 2.88e-5 | 0.2052 |
| 4 (cn20) | 1.55 / 1.35e-5 | 3.52 / 2.96e-5 | 0.2015 |
| 2 (cn21) | 1.69 / 1.49e-5 | 3.54 / 3.00e-5 | 0.1997 |
| R_RES 2 + R_FB 2 (cn19) | 4.14 / 4.81e-5 | 4.83 / 5.35e-5 | 0.1931 |

The feedback legs only need the dominant D21 pair: rank 2 moves the chain ~1e-10 in squared terms
(<= 0.6% of the leaderboard MSE) for -6.0% C. The residual rank is what moves the chain. cn19
passes the emulator at 160-174 s (the container is fast again after a restart), VmHWM 3.44 GB.

## 17. Rank-1 feedback, an exact hub fix, and three closed column / D21 ideas (cn21-cn26)

* Feedback rank 1 vs 2 moves the chain 0.6e-5 (below chaos). Rank 0 (no feedback) moves it
  3.6-4.0e-5. The D21 feedback is a rank-1 mechanism: only the coherent D21 direction feeds
  the next layer's legs.
* The young D21 hub narrowed per chunk only when every young source was narrower than n, and
  sources born at layers 0-3 are full width. Narrowing per chunk regardless is exact (moves
  6-8e-7) and saves 0.25% C. `v5b_cn26` = cn21 + rank-1 feedback + this fix: C/B 0.198-0.200
  (-0.7% vs cn21), emulator 161-165 s, VmHWM 3.47 GB.
* Saturated-on columns (Phi > 0.99; 165-276 per source from layer 8, as many as the dead ones)
  are not free, although w2 / S3c / e ~ phi(alpha) are as small as for dead columns. The kappa4
  feed pair (c1 = w1 lambda, c2) and the feedback rows (R1T, R2T, Phi-independent) still read
  them. Masking Phi > 0.999 / 0.99 moves the chain 1.1 / 1.5e-4 for -2.5 / -4% C. Closed.
* The old-source D21 needs its bulk. Its spectrum is one dominant pair (top-4 = 62-85% of the
  energy), yet truncating it to rank 1 / 4 / 32 moves the chain 2.4 / 2.2 / 1.2e-4. The term
  program reads D21 entry by entry; a rank-k randomized hub is closed.

## 18. Graded: feedback rank 2 (cn21), and why the yardstick undershoots

336544 v5b_cn21 (cn11a + R_FB 2) **3.5816e-9**: C/B 0.1995 (-6.0%), raw 1.7950e-8 (+3.4% vs cn11a's
1.7367e-8), so -2.8% vs cn11a and -14.1% vs rg. The yardstick (MLPs 5000 / 5001, rms move 1.5-1.7e-5
against a chaos floor of 1.1e-5) had put the raw cost at <= 0.9%. That is the second underestimate
after tier rank 288. The reason is the choice of networks: 5000 / 5001 are among the easiest local
networks (label MSE ~1e-9 vs a leaderboard mean of 1.74e-8). If a perturbation acts like noise
uncorrelated with the error (dMSE = |delta|^2), the leaderboard move behind +3.4% is rms ~2.4e-5, about
2x the easy networks' net move. Consequences: feedback rank 4-8 is likely as good or better than 2
(less raw damage per C), rank 1 (cn26) likely slightly worse, and every residual-rank cut is suspect.
Re-measured on the six hardest local networks (1015, 1002, 5010, 1005, 1008, 1010), with labels.

Hard-network label test (MLPs 1015, 1002, 5010, 1005, 1008, 1010: label MSE 2.8-4.3e-8, about the
leaderboard level; paired against cn11a; the noise term scales with the move, so se ~0.5% for a
1e-5 move):

| variant | move rms | label dMSE | per MLP | C vs cn11a |
|---|---|---|---|---|
| R_FB 8 (cn14) | 1.00e-5 | +1.7 +- 0.5% | +1.0 +2.6 +0.2 +2.2 +1.0 +3.6 | -3.4% |
| R_FB 4 (cn20) | 1.36e-5 | +1.8 +- 0.8% | | -5.2% |
| R_FB 2 (cn21; leaderboard +3.4%) | 1.51e-5 | +2.2 +- 0.6% | +1.1 +3.4 -0.2 +3.0 +2.9 +3.7 | -6.0% |
| R_RES 8 (cn12) | 1.87e-5 | -1.0 +- 1.4% | | -1.2% |
| R_RES 4 (cn13) | 2.69e-5 | +0.3 +- 2.0% | +2.7 +1.1 +4.7 +2.1 -1.7 -10.0 | -2.1% |
| R_RES 2 | 3.80e-5 | -0.4 +- 2.4% (first two MLPs alone: +7.2%) | | -2.6% |
| R_FB 1 vs R_FB 2 | 0.58e-5 | +0.7 +- 0.2% | | -0.5% |
| rows Phi < 0.02 vs 0.01 (on cn21) | 1.61e-5 | +1.3 +- 0.6% | | -0.8% |
| columns Phi < 0.015 vs 0.01 (on cn21) | 1.73e-5 | +1.4 +- 1.1% | | -0.6% |

This test reproduces cn21's leaderboard damage in sign and roughly in size, unlike the easy-network
yardstick. The feedback damage comes in the first cut (16 -> 8) and grows slowly after it, so
rank 2 is near the best tradeoff. The moves are the same size as on the easy networks; what differs
is that a feedback cut moves the chain along its own error. The residual rank reads neutral from 8
down to 2 (+-1.4-2.4%, per-MLP swings up to +-10%), so the leaderboard decides: Oct 17 wave 1 is
cn35 (= cn21 + the exact hub fix) and cn36-cn39 (residual rank 8 / 4 / 2 / 1). Feedback rank 1 is
worse than 2, and rows at 0.02 / columns at 0.015 cost more than they save; all three are out.
Wall check, back to back on the same container state: cn21 (graded) 173.9 s, cn39 175.0 s.
