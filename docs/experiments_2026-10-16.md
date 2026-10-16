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
