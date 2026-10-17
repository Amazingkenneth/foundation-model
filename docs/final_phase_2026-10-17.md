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
| per-neuron metric (e30_a) | `XMET` (diag, (2,2), feed), `XMET431` (+ (3,1)) | screening | |
| (1/4) a3 b3 C0 o K22 diagram (Artyom) | | already in our term table (`('c_off', 'wk4m', 8, 8, 0.25)`) | -- |
