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
column relabelling or any float reordering moves per-MLP MSE by up to +-25% on the
best-predicted MLPs, so local paired tests resolve only a few percent even with clean labels;
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
