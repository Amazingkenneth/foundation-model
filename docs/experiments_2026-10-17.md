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
-0.9%), and a Winograd form of the combos would save ~1/15 of the 6% addition overhead (~0.4%,
not worth a slot).

Cost elasticities of the remaining structural knobs (real op stream, MLP 5000; C/B of cn37 0.1935):

| change | C/B | dC |
|---|---|---|
| AGE_OLD 4 (one young source fewer, tier 1 one source more) | 0.1834 | -5.2% |
