# An AI agent in an algorithmic competition: synthesis and self-critique

**Case:** ARC White-Box Estimation Challenge 2026 (AIcrowd), Phase 2, Oct 14-22, 2026.
**Agent:** Claude Code in a cloud container, in one continuous session directed by one team member (the user).
**Purpose:** raw material for research on how AI and LLM agents help, or fail to help, in research and algorithmic
competitions. It covers what was asked, what was built and tried, every submission, the problems met, a
critical self-assessment, and a sketch of what better auto-research would look like.

**Sources:**
- the full session transcript: 25,744 records, 45 user messages, 12 context compactions;
- 223 commits on `claude/wonderful-turing-y343j0`;
- the experiment logs in `docs/`;
- the grader's results for our own submission IDs.

All times are UTC. Numbers are copied from the logs, not re-derived.

**Caveat:** the agent wrote this about its own work, so the self-critique may be biased. Section 11 tries to be
adversarial, but anything load-bearing should be checked against the raw logs. The transcript sits outside the
repo (`/root/.claude/projects/...jsonl`), in an ephemeral container. It also holds incidental third-party data (see
section 10.5), so it should not be published as-is.

Companion file: `docs/submissions_phase2.csv`, a machine-readable ledger of all 58 submissions.

---

## 0. Summary

- **Outcome.** The best graded score went from the team's starting **4.4e-9** (raw MSE 1.75e-8, compute fraction C/B
  0.2518) to **3.2790e-9** (cn79, raw 1.7311e-8, C/B 0.1894). That is **-25.5%**, and almost all of it is compute:
  C/B -24.8%, raw error -1.1%. Public rank moved from about 74 to about 57 (the board shows two significant
  figures). The leaders held at 1.3-1.4e-9 throughout, so the gap to first place was about 2.5x at the end.
- **Volume.**
  - **Submissions:** 58 in 7 days, of which 45 graded clean, 6 lost some or all networks, and 7 failed outright.
    24 set a new best.
  - **Code and data:** 92 candidate builds, 223 commits, about 3,600 lines of experiment logs.
  - **Experiments:** on the order of 150 measured, each closed with a number.
- **Where the gain came from:**
  - about -7% from exact cost engineering (Strassen kernels, billing-aware rewrites);
  - about -15% from approximate cost cuts with measured accuracy trades (dead-neuron masks, low-rank tiers, thin-leg
    ranks);
  - about -4% from accuracy work, mostly a learned carried corrector whose design came from a public write-up;
  - about -1.2% from a final-layer analysis on the last day.
- **What the agent was good at:**
  - building faithful local replicas of an opaque grader, and diagnosing its failures;
  - fast, careful porting of public ideas;
  - disciplined measurement: paired tests, held-out halves, calibration of local gains against the grader;
  - honest reporting, including failures and corrections of its own claims;
  - documentation.
- **What it was weak at:**
  - **Calibrated persistence.** It declared the search exhausted at least five times. Each time, a non-technical
    push from the user ("try harder") was followed by real gains, including the last two bests, within hours.
  - **Strategic breadth.** Most effort went into local perturbations of the inherited design. The structural question
    (how the leaders are 2.5x better) was attacked only on day 6, after the user pushed.
  - **Evaluation-first ordering.** Noise-free public labels were reached only on day 4. Earlier, noisy labels and
    easy-network yardsticks misled several decisions.
  - **Parallelism.** Only 3 subagents were used in 8 days; experiments were queued serially on 4 CPU cores.
  - **Operational hygiene.** Recurrent self-kills with `pkill -f`, OOM kills, and 92 copy-paste build directories.
- **What the human contributed:**
  - goals and access: the competition token, network permissions, the team's code and notes;
  - persistence and direction, through 45 short messages, almost none of them technical;
  - two plan rejections ("validate locally before spending slots");
  - final authority over nominations.
- **The ideal (section 12):** an evaluation-first, multi-track research system with these properties:
  - a calibrated exploration budget, and stopping rules that are not "my idea list is empty";
  - objective-backward sensitivity analysis: what does the score actually consume?;
  - parallel sandboxes and a structured lab notebook;
  - hygiene enforced at the tool level;
  - humans kept for goals, access, risk and irreversible actions.

---

## 1. The setting

### 1.1 The task, in one paragraph

Each test case is a random bias-free ReLU MLP, 1024 wide and 16 deep, with Gaussian input. The estimator must
predict the expected post-ReLU activation of every neuron at the final layer, under a FLOP budget B = 2^41 that is
metered by a library (flopscope).

- **Score:** mean over networks of final-layer MSE x max(0.1, C/B). A 1% compute cut is worth exactly as much as
  a 1% error cut.
- **Hard caps per network:**
  - 120 s wall time;
  - 0.4 s "residual" time (Python outside metered ops);
  - about 135 MB per array, and 8 GB of memory.

  A network that breaks a cap is zeroed. One zeroed network costs about 0.009 of score, against our 3e-9, so
  robustness is existential.
- **Slots and ranking:** about 10 submissions per day. The public leaderboard uses 100 networks with noise-free
  truth. The private ranking re-runs up to 2 nominated submissions on fresh networks.

### 1.2 Glossary (terms used below)

| term | meaning |
|---|---|
| chain | the estimator: propagates mean, covariance C, third cumulant (D3 diagonal, D21 slice) and a regenerated fourth-cumulant term layer by layer. It descends from the sponsor's ARC reference and 504aldo's MIT "V25/V29" factorized K=3 chain |
| sources | each layer's newborn third-cumulant content, kept as dense legs while young and compressed into shared low-rank bases ("tiers") when old |
| k34 / MP | the team's learned ReLU-step operators (the "v5b" submission), trained before this session |
| corrector | the agent's learned per-neuron MLP that predicts each layer's created mean error and carries it forward (design after forum post 18254) |
| C/B | billed FLOPs / budget |
| raw MSE | final-layer MSE before the compute factor |
| twin (`r` build) | the same estimator with Strassen at 4 instead of 5 levels: +6.8% compute, 24% less wall time; used as a robust second nomination |
| gate | the local checks every build passes before upload (section 7) |
| fast mode | local evaluation with plain matmuls in place of Strassen (about 10x faster) |

### 1.3 People and roles

- **The user:** a member of a student team ("I am one of the 'other students'"). The user directed the session,
  provided access and teammates' material, and owns nominations and the write-up.
- **Teammates:**
  - wrote the original "Option B" PDF (a learned ReLU-step "foundation model");
  - held a meeting whose summary was shared;
  - produced the v5b implementation that became the code base;
  - produced notes 13-16, improvement brainstorms. Notes 13 and 14 are titled "twenty-agent brainstorms", so the
    team itself used AI agents;
  - shared one submitted code variant on Oct 17.
- **The agent:** implemented, measured, submitted (with delegated authority: "You should make submissions whenever
  possible"), documented, and drafted the write-up.

### 1.4 The agent's working environment

- **Container:** an ephemeral cloud container with 4 CPU cores, about 16 GB of usable memory (inferred from OOM kills) and no GPU.
  - It was restarted several times, which killed background jobs.
  - The network went through a policy proxy. aicrowd.com, huggingface.co and arXiv were blocked at first and
    enabled by the user on request.
- **Tools:** shell, file edit/write, background tasks and monitors, web search and fetch, subagents, and scheduled
  self-wake-ups. The wake-ups were used to submit at the 00:00 UTC slot reset.
- **Permission classifier:** an auto-mode classifier screened actions. It denied cloning another participant's
  public repository ("Untrusted Code Integration"), and the agent did not work around it.
- **Git hook:** a stop hook demanded that every change be committed and pushed. This produced 223 commits, all
  tagged `[skip submit]` so that the CI workflow would not submit.
- **One session, 7.5 days:** 4,085 model requests, about 4.2M output tokens and about 1.77B cache-read tokens.
  - 12 context compactions, so the agent's working memory was rebuilt from lossy summaries 12 times.
  - Client-reported cost estimate: about $619. Recorded thinking time: about 7.5 h. Model API time: about 15 h.
    Tool execution time: about 54 h.
  - Assistant activity fell in 130 of the 179 elapsed hours.

---

## 2. Headline numbers

| | team start (v5b) | agent's first best (rg, Oct 15) | final best (cn79, Oct 21) | leaders (#1, Oct 20) |
|---|---|---|---|---|
| score | 4.4e-9 | 4.1705e-9 | **3.2790e-9** | 1.3e-9 |
| raw MSE | 1.75e-8 | 1.7534e-8 | 1.7311e-8 | 1.10e-8 |
| C/B | 0.2518 | 0.2378 | 0.1894 | 0.122 |
| public rank | ~74 | 70 | ~57 (3.3e-9 displayed) | 1 |

The new-best sequence, all graded: 4.2513 → 4.1705 → 4.1702 → 4.0540 → 3.9756 → 3.9041 → 3.8118 → 3.8110 → 3.7487 →
3.6860 → 3.5816 → 3.5756 → 3.5275 → 3.5262 → 3.5121 → 3.4863 → 3.4544 → 3.4395 → 3.3386 → 3.3341 → 3.3268 → 3.3201
→ 3.3183 → 3.2888 → 3.2790 (e-9).

---

## 3. Timeline

| day (UTC) | what the user asked | what happened | best at day end |
|---|---|---|---|
| **Oct 14** | "Fully replicate the algorithm described in the PDF ... Compose scripts for automating zip/submissions ... Fully test out your work before announcing it's complete." | The agent wrote its own chain + learned-operator framework (`submission/`, `fm/`): flopscope and torch backends, Strassen, MC label generation, training, CI submit workflow and tests. Training at local scale did not beat the analytic chain (local score ~5e-9 at C/B 0.327), and the agent said so. Network policy blocked submission. **Never graded.** The next day's emulator showed it would have crashed under the grader's client (a tuple-axis bug): "fully tested" was true only for a test environment that differed from production. | none |
| **Oct 15** | Meeting summary + teammate's v5b tarball; notes 13-15 + token: "evaluate ... make submissions whenever possible"; access enabled. | The agent ported v5b's operator, matched it to 1e-6, then **switched the code base to the teammate's v5b**. It evaluated notes 13-15: the "10-50% cost reduction" came mostly from parts of the literal PDF that the code never ran. First 7 submissions. Three level-6 Strassen variants failed opaquely, so the agent **built a grader emulator** (client/server flopscope) plus memory probes. Together they reproduced the failures and pinned three limits: wall time, memory, and a 135 MB per-array cap. Three of note 16's 17 items were evaluated the same night. The per-network scale correction was later found to be label noise. Dead-neuron pruning was estimated at "1-2%" and not attempted that night. | 4.1705e-9 (rg) |
| **Oct 16** | "Analyze ... compare to the top ... 10 submissions"; "draw inspirations from research works ... make better inference about why they worked". | The agent first said "the cheap levers ... are used up". It then delivered the **biggest single day (-14%)**: | 3.5816e-9 (cn21) |
| | | - dead-neuron live-column/row chain (cn2/cn3), which was in fact note 16's item 1, realized at ~5x the agent's own estimate; | |
| | | - flat Strassen pools (memory 6.1 → 3.3 GB); | |
| | | - warm-started randomized range finders (Halko-Martinsson-Tropp); | |
| | | - the age-rank tier law; | |
| | | - feedback oversampling; | |
| | | - alternate basis rebuilds; | |
| | | - thin-leg rank cuts. | |
| | | It also found that float32 chaos makes local comparisons noisy, and that local label noise lies along the ReLU homogeneity (scale) mode. 11 submissions; two more were refused by the daily limit. | |
| **Oct 17** | Many pushes; teammate's code note; "behave like a very experienced competition participant"; forum links; network for more sites. | Two probes tuned on six hard networks lost on the board, a **selection effect**. After that the agent kept held-out halves for every decision. Other work: | 3.4544e-9 (cn54b) |
| | | - a 10x faster 75-network gate and the "move² rule" (expected damage of a perturbation ≈ \|D\|²/MSE); | |
| | | - exact kernel wins (Winograd form, slab pool, elementwise term program); | |
| | | - level-6 failed a 4th time; | |
| | | - forum post 18251 (read by the agent) revealed a billing update (flopscope 0.12.3), a **deadline extension to Oct 21**, and the nomination procedure; | |
| | | - huggingface.co enabled gave the **public 1,100-network dataset with 1e9-sample labels**, which became the gate's ground truth. | |
| **Oct 18** | Plan mode, rejected twice: "More local experiments that try to validate ideas before submitting to waste the submission slots. Seek for more external inspirations". "Note that there are errors while scoring today". | **Grader under load:** identical code (cn54b, resubmitted as a control) lost 70% of networks. That separated grader failures from code failures. The agent introduced the **robust 4-level twin** policy. Exact-label sweeps mapped every knob: cheap side 1.5-3% MSE per 1% C, expensive side 0.6%. First **carried corrector** (forum 18254 design). One upload failed the smoke test on Fortran-order arrays, a gate gap that was then closed. | 3.4395e-9 (cn60) / 3.4404e-9 (cn61) |
| **Oct 19** | "Keep improving ... validate locally first"; "look at other teams ... external sources"; "Check the forum". | Corrector iterations: carried state through W, a W∘W channel, 16 epochs, correlation inputs, then a smaller model (held-out -2.2% → -4.9%, graded raw -3.9%). Residual-time engineering cut client round trips 9% (lazy views, persistent buffers). All 5-level builds graded clean from cn64 on. | 3.3201e-9 (cn69) |
| **Oct 20** | "Create docs/ALGORITHM.md ... that another AI could use"; "exhaustive cracking options ... arXiv"; "Can you think of a better algorithm that the leaders may use?"; **"don't think that work is heavy and costly. Please really act as a responsible, worthy talent ..."**; "yes, start the layer-by-layer comparison now". | Spec + gate tools moved from /tmp into the repo. End-to-end Gauss-Newton refit of 12 closure knobs (local -0.5%, graded -0.05%: in-sample optimism). The agent reconstructed the leaders' likely construction (carried κ4 core + scale-mode extraction) and **built it** after the user's push. A layer-by-layer diff against a float64 reference found the culprit (our rank-2 feedback leg, +17%), but the construction cost about 2x our bill at equal accuracy. Closed, with evidence. Also measured: structured-sampling cubature (400x short), tier restructure, final-layer skip. The agent recommended stopping experiments. | 3.3183e-9 (cn71) |
| **Oct 21** | "draft the write-up"; "10 submissions today ... capture the last chances"; agent: leave slots unused; **"Please try harder and think of genuinely creative approaches"**; **"keep pushing. You should be confident that something is waiting for discovery"**. | Write-up drafted; it caught and fixed an attribution error. Then, within 8 hours of the two pushes, two first-principles finds: | **3.2790e-9 (cn79)** |
| | | - the final layer consumes D3 only through an Edgeworth term weighted by φ(α), so skip the tails (cn78, -0.89% graded, predicted to 0.04%); | |
| | | - the skipped rows' D3 is ~95% the network's scale mode, so fill it for free (cn79, -0.30%, predicted to 0.07%). | |
| | | Push to first upload took ~45-55 min each. | |
| **Oct 22** | "/compact"; this synthesis. | Nominations recommended: 337588 (cn79) + 337606 (cn79r). | |

---

## 4. Every submission

The full ledger is in `docs/submissions_phase2.csv`, with columns: time, ID, build, change, status, score,
raw MSE, C/B and note. Summary:

| category | count | IDs |
|---|---|---|
| graded clean | 45 | everything not listed below |
| failed: level-6 Strassen over the wall cap | 4 | 336265, 336269, 336281, 336627 |
| partial: memory over limit | 1 | 336313 (0.0226) |
| failed or partial: grader load / residual cap | 7 | 336851, 336852 ("Error while scoring"), 336887 (0.0502, all zeroed), 336962 (0.0310), **336965 (unchanged control, 0.0351)**, 336980 (0.0204), 337009 (0.0220) |
| failed: smoke test (Fortran-order arrays in an `.npz`) | 1 | 336979 |
| refused by the daily limit | 2 builds | cn22, cn23 (403 on Oct 16) |

Use of the leaderboard as an oracle:
- **Days 1-3: probes.** The leaderboard was used as the measurement instrument. Examples: rank cliffs (cn10 R_OLD
  288: +3% raw), the residual-rank sign (cn36/37), and the level-6 safety question (cn44).
- **From day 4: gate first, then confirmation.** Decisions moved to the local exact-label gate, and the leaderboard
  confirmed them. The calibration of local gains against graded ones was tracked throughout:

| change type | local gain → graded gain | transfer |
|---|---|---|
| knob tweaks tuned on hard networks (cn40/cn41) | -2.2% → +3% / +2.9% raw | negative: selection effect |
| small-move bundle (cn49b) | -0.79% → -0.17% | ~0.2-0.3 |
| exact kernel (cn53) | -1.48% C → -0.75% C | ~0.5: first-network amortization on the grader |
| learned corrector (cn62r-cn69) | -2.2..-4.9% → -1.5..-4.15% raw | 0.73-0.86 |
| in-sample knob refit (cn71) | -0.50% (held-out -0.59%) → -0.05% | ~0.1-0.15 |
| final-layer row restriction (cn78) / fill (cn79) | predicted 3.290 / 3.277e-9 → 3.2888 / 3.2790e-9 | ≈1.0 (cost-dominated, accuracy-neutral) |

**Lesson for research:** cost cuts transfer almost exactly, while accuracy gains shrink by a factor that depends on
how the change was selected. Anything fitted or selected on the evaluation set shrinks the most.

---

## 5. What moved the score, by track

Each factor is the ratio of graded new-best scores across the steps listed.

| track | steps (builds) | graded factor | kind | idea source |
|---|---|---|---|---|
| exact Strassen and billing engineering | v5b → rfhub → rg | 0.948 | exact | agent (from reading the billing rules and v5b's kernels) |
| dead-neuron live-column chain + live rows | rg → cn3; cn9 → cn11a | 0.953, 0.983 | trade (≈free) | teammates' note 16 item 1; agent's design |
| range finders and tier schedule | cn3 → cn9 | 0.943 | trade | agent (randomized NLA literature: warm start, oversampling); 504aldo's age-rank law |
| thin-leg ranks + exact hub narrowing | cn11a → cn37 | 0.957 | trade | agent (spectra: one dominant pair on a flat bulk); 504aldo's rank ladders |
| marginal tail rule + oversampling | cn37 → cn49b | 0.996 | trade | Artyom_Mazur (forum 18249) |
| Winograd kernel, slab pool, elementwise term program | cn49b → cn54b | 0.984 | exact | agent (billing forensics) |
| residual rank 6 + late dense window | cn54b → cn61 | 0.996 | trade | agent (targeted spending by depth) |
| carried learned corrector | cn61 → cn69 | 0.965 | learned | Susupro (forum 18254) design; agent's loss, inputs and size |
| end-to-end closure-knob refit | cn69 → cn71 | 0.9995 | fitted | Puffi / Susupro ideas; agent's Gauss-Newton |
| final-layer row restriction + scale-mode fill | cn71 → cn79 | 0.988 | trade + physics | agent (first-principles; the scale mode itself from Hanin-Nica / il.miscusi 18248) |
| **total** | | **0.745** | | |

Cost engineering explains about 22 of the 25.5 points. The raw error stayed within 1.73-1.82e-8 the whole week:
- truncations added about 3-4% error;
- the corrector removed about 4%.

The structural accuracy gap to the leaders (×0.64-0.79 raw) was never closed.

---

## 6. Everything tried and closed

Each line is a measured result. "dMSE" is on exact labels (100 public mini networks) unless noted, and dC is the
billed compute change. The fuller tables are `docs/ALGORITHM.md` sections 6-7,
`docs/experiments_2026-10-16.md`, `docs/experiments_2026-10-17.md` and `docs/final_phase_2026-10-17.md`.

### 6.1 Chain truncation and cost knobs (all at their optimum by Oct 18)

- **AGE_OLD 4 / 6:** +10.6% / -2.5% MSE for -5.1% / +4.1% C.
- **R_OLD 288 / 352:** +2.75% / -1.69% for -0.91% / +2.67%. Rank 288 was also graded: past the cliff.
- **AGE_OLD2 8 / 10; R_OLD2 128 / 160 / 192:** all lose, or tie (rgc192).
- **Masks:**
  - CM_TAU 0.015 / 0.02: +1.4% / +3.7%;
  - RW_TAU 0.02: graded neutral;
  - RW_TAU 0.03: +2.3%.
- **Thin legs:**
  - R_FB 1: +0.5..+1.5%;
  - feedback off: +7.8%;
  - R_RES 2: +3.8%;
  - R_RES 8: -1.3% MSE for +1.0% C;
  - 6 is the optimum.
- **Old-source handling:**
  - drop old-source D21: ×4 MSE;
  - low-rank D21 / a_b: out;
  - tier-2 exact eigensolve: wash;
  - tier restructure by the age-rank law: C/B 0.246;
  - depth-graded rank slope: loses.
- **Dead or saturated content:**
  - D21 hub dead rows / columns / saturated rows: +7.9..+55%;
  - saturated-at-birth column mask: **+110%**. Saturated neurons carry the κ4→κ3 feed.
- **Final layer:**
  - skip all source work: ×11.7;
  - skip D21 at layer 14: +172%; skip old-source D21 at layer 14: +42%.

### 6.2 Kernels, billing and runtime

- **Won:** Strassen on more families, Winograd form, slab pools without billed reshapes, the elementwise term
  program, Gram einsums (billed at half price), lazy views and persistent buffers (round trips 49.4k → 43.1k).
- **Lost:**
  - level-6 Strassen: 4 grader failures, wall cap;
  - bigger chunks: memory;
  - a fresh-array pool: +13% wall;
  - a 48-multiplication 4x4 scheme and alternative bases: extra ops break the residual cap.
- **Billing facts learned:** float16 is billed as float32; matmul gets no symmetric discount; reshape is billed per
  element; `take` costs 4 per element. So none of these is a lever.

### 6.3 Closure physics (mostly ports of public write-ups)

- **No gain:**
  - the (3,1) two-term κ4 law (18246): +1.4..+4.0%;
  - the per-neuron metric diag(WWᵀ) (18248, e30_a): null;
  - the C0³ Mehler pair term: null;
  - the young κ4 "3+2" tree reader (18249): +3.5% after the learned operator;
  - newer term tables: worse;
  - BETA exponent: flat;
  - MP from layer 6 or 7: ≈0 net;
  - (2,)-order Edgeworth term: negligible;
  - final-layer scale-mode extraction: +0.5..+0.8%.
- **E1 κ4 pendant-edge term (18252):** -1.1% raw, but it needs one more dense product per layer in this chain:
  net loss.
- **The leaders' construction (carried κ4 core + K211 + scale-mode extraction, 18248):**
  - untruncated reference: 0.70 of our error;
  - our port: parity at C/B 0.40 vs our 0.19;
  - the bottleneck, found by layer-by-layer diff: the rank-2 feedback leg tuned on the old chain.
  - Closed for the deadline.
- **Structured-sampling cubature** (Phase 1 winner's method): 4.5e-6 at C/B 0.12, about 400x short.

### 6.4 Learned components

- **Corrector ladder:** -2.23% → state through W -3.92% → W∘W channel -4.60% → 16 epochs -4.72% → v3
  correlation inputs -4.90% → small model -4.76% at -0.46% C.
- **Variants that did not beat it:**
  - capacity: H 192, S 64, S2 32, a backward state, 24 epochs; an ensemble was assessed but not built (each member costs 0.43% C);
  - objectives: a pooled end-to-end objective (18255), mixed transported loss (mliston 18170: much worse, -1.65%),
    gain-weighting changes, and the corrector trained on all 1,098 networks (graded noise);
  - inputs: incoming-correction inputs, v4 inputs, root transforms, network-context inputs (~0 net);
  - Susupro's recipes (per-layer standardisation, per-layer output scale, tanh state), EMA, output scale α,
    clipping.
- **Never attempted:** fine-tuning the team's k34/MP operators on the drifted chain. It needed a differentiable
  replica of the production chain that was never built (task #42, left pending all week).

### 6.5 Statistics and calibration findings

- **Per-network scale error is label noise.** The "oracle -38%" from rescaling each network was the fit absorbing
  Monte Carlo noise, which is coherent along the ReLU homogeneity mode.
- **The float32 chain is chaotic.** Equivalent runs differ by about 1e-5 rms, so tiny-move comparisons need many
  networks.
- **Easy-network yardsticks underestimate damage** (move² rule). Hard-network tuning sets overestimate gains
  (selection effect).
- **Error anatomy:**
  - truncation is only ~4.5% of the MSE; the rest is closure error;
  - the final layer creates 22% of the MSE;
  - created error grows ~9x with depth;
  - bias is negligible.

---

## 7. Infrastructure the agent built

The infrastructure decided what could be learned. It is listed here because most of it was built in reaction to a
failure, which is a point for section 11.

| tool | why it was built | built on |
|---|---|---|
| grader emulator (flopscope client/server, caps) | 3 opaque level-6 failures | Oct 15 |
| VmHWM / array-cap / round-trip / residual probes | memory and cap failures; residual-cap failures under load | Oct 15-19 |
| paired-MSE evaluators with halves and bootstrap se | noisy single-network comparisons | Oct 16 |
| hard-network and held-out label tests | the easy-network yardstick misled; then the selection effect | Oct 16-17 |
| fast mode (plain matmuls) + 75-network gate | full runs too slow to screen ideas | Oct 17 |
| exact public labels (HF dataset, 1e9 samples) + weight regeneration | MC labels too noisy (resolution ~3-6%) | Oct 17 evening (blocked by network policy until then) |
| real-Strassen label check | the fast gate bypasses the kernels that ship | Oct 18 |
| smoke-shape emulator check | Fortran-order `.npz` failure | Oct 18 |
| 0.12.3 billing replica + per-function profiler | evaluator update | Oct 17-20 |
| chain-dump recorder + corrector training / eval / export / parity pipeline | learned corrector | Oct 18-19 |
| Jacobian / Gauss-Newton knob fitter | end-to-end refit | Oct 20 |
| T-chain prototype + layer-by-layer differ vs a float64 reference | leaders' construction | Oct 20 |
| own-ID poller + watcher; scheduled slot-reset submits | grading latency 0.5-2 h; daily limits | Oct 15-18 |
| packaging, submit and CI workflow; stop-hook-driven commits | the original request | Oct 14 |

Gate tools lived only in `/tmp` until the user asked, on Oct 20, to move them into `scripts/`. Container restarts
had already killed jobs several times.

---

## 8. Where ideas came from

| origin | examples | shipped gain |
|---|---|---|
| agent: first principles / measurement | live-column chain design, warm-start range finder, thin-leg spectra, row/column thresholds, billing forensics, residual-time engineering, robust twins, final-layer row restriction, scale-mode fill | most of the cost track; the last two bests |
| agent: literature | Halko-Martinsson-Tropp randomized range finders; Hanin-Nica / Roberts-Yaida-Hanin gain mode; Strassen-Winograd | range-finder steps; diagnosis of label noise |
| public forum write-ups of other teams | marg2 (18249), carried corrector (18254), scale-mode extraction (18248), E1 (18252), (3,1) law (18246), knob-refit ideas (Puffi, Susupro), billing hints (18250, 18218) | corrector (-3.5%), marg2; most others closed |
| 504aldo's public MIT chain and findings | the base chain, age-rank law, rank ladders | base; tier steps |
| team (pre-session) | v5b code base and learned operators k34/MP | the starting point |
| team notes 13-16 (partly AI-generated) and teammate code | dead-neuron pruning, last-layer trimming, cleaner labels, better validation, training on the chain's own states | indirectly: pruning (-7% C), validation, and training on dumps were all realized; last-layer trimming was realized only on day 7 |
| the user directly | access, persistence, "validate locally first", "build the leaders' construction", "layer-by-layer comparison" | the process changes behind several gains |

**Teammates' AI-generated estimates against measurement:**

| claim (note) | claimed | agent's first estimate | measured |
|---|---|---|---|
| 13/14: "10-50% further cost reduction" | 10-50% | mostly not applicable (it targeted parts of the literal PDF the code never ran) | confirmed: not applicable |
| 16.1: skip dead and saturated neurons | 25-75 of ~260 units (10-29%) | "about 1-2%" | dead: about -7% C at ~0 raw (between the two estimates); saturated: +110% MSE (wrong) |
| 13.2 / 14.I02: final-layer pruning; 16.4: trim the last two layers | -4..-15 units (16.4) | 13/14: "already in" (v5b's last layer has no D21 and no birth, a surface match); 16.4 was never evaluated separately | realized on Oct 21 as row restriction + fill: -1.26% C (~2.5 units). Six days late |
| 16.6: retire the old-source tier | 107 units | | closed: old-source D21 is load-bearing |
| 16.9: per-network scale correction | oracle -38% | | label-noise artefact |
| 16.12: train on the chain's own states | | | the corrector on chain dumps: the largest accuracy gain |
| teammate code (Oct 17): feedback rank 1, etc. | -3% local | | on our line: rank 1 loses, the rest already present; flagged an array-cap risk in their code |

**Reading:** the AI-generated brainstorms were a good map of where to look and a poor guide to how much each
idea would pay. The agent's own priors were also miscalibrated, in the pessimistic direction. Measurement fixed
both.

---

## 9. Human-AI interaction

### 9.1 The user's 45 messages

| kind | count (approx.) | examples |
|---|---|---|
| task with attachments (PDF, meeting summary, code, notes, token) | 6 | "Fully replicate the algorithm ..."; "Take reference of my teammate's research note ..." |
| access grants | 4 | "Permission enabled for aicrowd access."; "Network access for the sites you listed has been enabled" (sent 3 times because of interrupts) |
| persistence / encouragement / "try harder, be creative" | ~18 | "You are in a good spot!"; "I know it's hard but keep pushing. You should be confident that something is waiting for discovery" |
| strategy questions | ~7 | "what would be the next steps for pushing our ranking further" (x2); "Can you think of a better algorithm that the leaders may use?" |
| slot reminders | 4 | "7 submissions remaining today" |
| concrete directives | ~6 | "Create docs/ALGORITHM.md ..."; "yes, start the layer-by-layer comparison now"; "draft the write-up" |
| process corrections (plan rejections, "validate locally") | 4 | "More local experiments that try to validate ideas before submitting to waste the submission slots" |

The user almost never supplied technical content of their own. Technical input came from teammates' notes and
code, public write-ups, and the agent. The user's leverage was access, direction, and refusing to accept "done".

### 9.2 Premature stops and what followed

| when | the agent's position | the user's response | what followed |
|---|---|---|---|
| Oct 15 20:58 | used 7 of 10 slots and "stopped there, because the remaining depth-tuning gains are under 1%" | (note 16 delivered) | the next day: -14% from new ideas |
| Oct 16 01:55 | "The cheap levers within this method are now used up ... Moving up meaningfully needs a structural change that I can't build and validate before the Oct 17 deadline." (at 4.17e-9) | "Try to test out more experiments by drawing inspirations from research works ..." | same day: cn2 → cn21, 4.17 → 3.58e-9. The final result was 21% better than the score at which this was said |
| Oct 20 06:34 | "Why there's little left ..." | "Can you think of a better algorithm that the leaders may use?" | reconstruction of the leaders' method |
| Oct 20 14:00 | "It can't be built and validated before Oct 21 23:59 UTC." | "don't think that work is heavy and costly. Please really act as a responsible, worthy talent ..." | built and diagnosed in ~8 h (rank-2 feedback culprit). No shippable gain, but a real finding, recorded |
| Oct 20 23:42 | "My recommendation for the last ~24 hours ... is to stop experimenting." | "draft the write-up. Make the final efforts" | write-up |
| Oct 21 12:57 | "My recommendation is to leave the remaining 10 slots unused." | "Please try harder and think of genuinely creative approaches" | cn78 (-0.89%) uploaded 53 min after the message |
| Oct 21 17:34 | reported cn78; nominations | "keep pushing. You should be confident that something is waiting for discovery" | cn79 (-0.30%) uploaded 46 min after the message |

The pattern held at every one of these points: a non-technical push was followed by progress. The agent's
stopping decisions were not wrong about the ideas it had in hand. They were wrong about the size of the idea space
it had not yet generated (section 11.1).

### 9.3 Where the agent's caution was right

- **Gating and slot discipline.** The user also wanted this: the two plan rejections asked for more local
  validation, not fewer gates. After Oct 18, every shipped accuracy change transferred in the predicted direction.
- **The robust-twin policy.** The 4-level twins graded clean 10 times out of 10, while 5-level builds lost networks on
  loaded days.
- **Refusing to submit noise-level variants for nomination.** A lucky public draw cannot help the private
  re-evaluation. However, the agent conflated *submitting* with *nominating*. Exploring with free slots never
  forced a nomination change, so the option value of more exploration was underweighted.

### 9.4 Interaction friction

- **Plan mode.** Plans were rejected twice because they front-loaded submissions. The user wanted wide local
  brainstorming first.
- **Interrupts misread.** After two user interrupts on Oct 17, the agent treated a re-sent message as needing
  confirmation ("I'm still waiting for your go-ahead"), so the user had to send it a third time.
- **Report density.** Status reports were long and number-dense: good for audit, heavy for a phone reader.
  Recommendations such as nominations were repeated in every report, which kept them clear.
- **Delegated authority.** The agent decided and executed 58 submissions with no per-submission approval. The user
  kept nominations and the write-up's team name and disclosure. This split worked: no slot was wasted on a build
  that failed the gate for a reason the gate could detect.

---

## 10. Issues encountered

### 10.1 Grader opacity and environment mismatch

- The grader returned only a score or "Error while scoring", with no per-network timing.
  - Three failure classes had to be reverse-engineered with a local emulator: wall cap, memory, and array cap.
    Later came the residual cap under load.
  - Local "fully tested" runs did not represent the client/server runtime: day 1's build would have crashed.
- **Grader load days:** identical code lost 40-70% of its networks. This was settled only by spending a slot on an
  unchanged control (336965), which was good scientific practice.
- **Mid-competition change:** an evaluator update (flopscope 0.12.3) and a deadline extension, discovered by
  reading the forum.

### 10.2 Evaluation noise and bias

- **Labels:** local MC labels at 5e5-1e6 samples had about 3-6% resolution, with noise coherent along the scale
  mode. The noise-free public labels became reachable only on day 4: the network policy blocked huggingface.co,
  and the agent did not push to unblock it early.
- **Proxy sets:** an easy-network proxy undershot damage, and a hard-network tuning set overshot gains (selection
  effect, two graded losses).
- **In-sample fitting:** the knob refit transferred at ~0.15. Training on the held-out set (cn72) left the grade as
  the only validation.

### 10.3 Agent operational errors

- **Self-kills:** `pkill -f` / `pgrep -f` patterns matched the agent's own shell, killing or SIGSTOPping it about
  8 times. The lesson was written into memory after each occurrence and still recurred across compactions.
- **OOM kills** from running too many jobs in a 16 GB cgroup (dumps + training + billing).
- **Running scripts edited in place;** stale waiters matching their own `pgrep`; background waiters hitting time
  limits; monitors re-armed by hand.
- **Builds and patches:**
  - a build script with a relative path silently produced a build without its corrector file (caught by a parity
    check);
  - patches that did not apply because an `&&` chain stopped early.
- **Clock misreads and time-estimate slips.** The rule became "always check `date -u`".
- **Probe bugs** that produced false findings: an in-place zero-diagonal gave a fake "gain mode 2.5x too large";
  a symmetric-array microbenchmark gave a wrong slice-cost conclusion. Both were caught by follow-up checks.

### 10.4 Knowledge and documentation errors

- **Attribution.** The team's own learned operators were credited to 504aldo in `ALGORITHM.md` until Oct 21.
  The same error survived in `docs/brainstorm_2026-10-17.md` until this synthesis, which fixed it, along with
  gendered pronouns for people whose pronouns are not stated.
- **Write-up overclaims** ("raw unchanged", "every step exact") were caught on self-review and replaced by
  raw-MSE and kind columns.
- **A wrong number in the docs** (a cn9 C/B) was corrected.
- **Reconstructing state.** The 12 compactions meant reconstructing state from summaries. Facts survived well (the
  summaries were detailed); procedural lessons survived less well (see the self-kills).

### 10.5 Integrity and safety events

- **Status API leak.** The AIcrowd status API returned other participants' submission records when the agent
  queried nearby IDs to recover a lost ID of its own (Oct 16). The agent stopped, reported it, and thereafter
  polled only its own IDs. This data is in the transcript, which is why the transcript should not be published raw.
- **Wrong-ID watcher.** A watcher was once started on 4 IDs that were not ours (Oct 18). It was killed
  immediately, its output went to /dev/null, and the agent reported it.
- **Cloning denied.** Cloning a participant's public repo was denied by the permission classifier, and the agent
  respected the denial. Public forum write-ups and PDFs were read and credited.
- **Credentials:** the competition token was used only through an environment variable read from the uploaded
  file, and never written to the repo or to logs.
- **Untrusted inputs:** downloaded archives were extracted into separate directories, and code that read them was
  run in isolated mode.

---

## 11. Critical self-assessment

This list is ordered by estimated impact on the result.

### 11.1 The stopping rule was miscalibrated: "my list is empty" was treated as "the space is empty"

The agent closed ideas carefully. Each closure was a measurement, and the closed-ideas tables are the most
reusable output of the week. But the ideas it *generated* were mostly:
- local perturbations of the inherited design (knob ±1, thresholds, ranks);
- ports of public write-ups.

The two last-day gains came from a different mode, objective-backward analysis:
- *What does the score actually consume at the final layer?* Means only, and D3 enters only through
  φ(α)·κ3/σ².
- *What structure does the dropped quantity have?* About 95% of it is the scale mode.

Both questions were available from day 1. Notes 13 and 14 even proposed final-layer pruning on Oct 15, and the agent
marked it "already in" on a surface match (v5b's last layer skips D21 and births). Note 16 item 4 ("trim the last
two layers") then went unevaluated. A systematic *value-of-computation map* would likely have surfaced
this, and possibly other cases, days earlier:
- for every intermediate, the sensitivity of the final score to it (adjoint or perturbation);
- set against its cost.

**What would have been better:**
- Before declaring exhaustion, run a forced divergent pass:
  - for each component, ask what its consumer needs (precision, rows, rank, layers);
  - for each quantity, ask what structure it has (a dominant mode, a function of a cheap statistic);
  - for each constraint, ask whether it is real.
- Price exploration correctly. Free slots plus hours of time mean the option value is positive, and submitting is
  not nominating.

### 11.2 Strategy was exploit-heavy; the structural gap was attacked late

From Oct 16 onward the leaders' 2.5x advantage was known to be structural, and the agent's own docs said so on
Oct 16 and 17. The only structural attempt, the carried-κ4 "T-chain", started on Oct 20, after the user pushed,
and was then closed for lack of time. A standing exploration track from day 3 might have reached the conclusion in
time to act on it. For example, a subagent could have built the lean reference and the port in parallel while the
main thread harvested knobs. The conclusion was to retrain the learned operators on the new chain and find a
cheaper κ3 engine. The agent's sequential, single-thread habit (3 subagents in 8 days) made this impossible.

### 11.3 Evaluation came second when it should have come first

- **Labels.** The 1,100-network public dataset with noise-free labels existed from the start; only the network
  policy blocked it. Asking the user to enable it on day 1 would have removed three days of noisy decisions: the
  rank-288 cliff, the misread masks, and the selection-effect losses.
- **Gate steps.** The emulator, real-Strassen check, smoke-shape check and round-trip budget were all built after
  a graded failure revealed the need. An ideal system builds the production replica before the first upload.

### 11.4 The learned components were never re-fitted to the chain they ran in

The team's k34/MP operators were trained on an older chain. All week the agent tuned around them:
- operator gain knobs;
- discovering that public closure fixes were "absorbed" or "broken" by them;
- the T-chain blowing up 50x with them on.

Building a differentiable replica of the production chain (task #42) was the enabling investment, and it was never
made. That was a judgment call under time pressure, but it locked out a whole class of improvements.

### 11.5 Compute and parallelism were left on the table

The work ran on 4 CPU cores and 16 GB, with serial queues and frequent OOM interference, for about 54 h of tool
time. Some ideas were infeasible on this hardware:
- higher-sample labels;
- a K=4 teacher (needs a 96 GB GPU per the forum);
- larger corrector sweeps.

The agent stated the limits but did not ask for a bigger environment. Parallel subagents in isolated worktrees
could also have run independent tracks.

### 11.6 Code and experiment hygiene

- **Build sprawl.** There are 92 candidate directories, each a full copy of a ~1,850-line estimator, plus patch
  scripts and environment-variable hooks in scratch copies under `/tmp`. This was traceable, because every build is
  committed and every patch scripted. But it made "which knobs does build X have?" a forensic question, and a
  stale-default recorder nearly produced mismatched training data.
- **Better:** one parameterized estimator, configs, and bitwise-equivalence tests.
- **No queryable store.** Experiment results lived in markdown tables, with no store to query or dedupe against.

### 11.7 Process slips that recurred despite being "learned"

Some lessons were recorded in the agent's own summaries and then violated again after a compaction:
- self-kill by pattern;
- OOM from parallel jobs;
- editing running scripts.

Procedural knowledge kept in natural-language memory is not reliable. It needs enforcement in the tools (a hook
that refuses `pkill -f`, a job scheduler with a memory budget).

### 11.8 Things that went right and should be kept

- **Honest negative results:**
  - the day-1 "does not reach 335910" report;
  - "nothing submitted";
  - corrections of its own claims (K211 cost, attribution, overclaims).
- **Controls:**
  - the unchanged-code control submission that separated grader faults from code faults;
  - parity checks that caught a missing file;
  - bit-for-bit reproduction checks for every "exact" change.
- **Transfer calibration:** local-to-graded ratios were tracked and used for predictions. The last two predictions
  were within 0.04-0.07%.
- **Fast execution of good ideas:** from a user push to a gated upload in under an hour (cn78, cn79).
- **Reproducible handoff:** `docs/ALGORITHM.md` was written so "another AI could use it": spec, closed ideas, open
  levers, gate, file map.
- **Integrity:** own-ID-only polling after the leak; respected the classifier; credited sources.

### 11.9 A counterfactual, with low confidence

With exact labels and the emulator on day 1, a value-of-computation map by day 3, and a parallel structural track,
the shippable result might plausibly have reached about 3.1-3.2e-9: earlier final-layer work, fewer lost probes,
maybe one more structural cut. Closing the 2.5x gap needed a rebuilt κ3 engine plus retrained learned operators,
probably with GPU-scale data. That is unlikely within this week on this hardware for any agent.

---

## 12. What an ideal auto-research system would look like

Each component below names the failure in this case that it would have prevented.

1. **Production replica first.**
   - Before any idea work: an exact local replica of the grader (runtime mode, caps, billing, smoke tests) and the
     cleanest available labels.
   - Treat "fully tested" as "tested in the production replica".
   - *Prevents:* the day-1 latent crash, 4 level-6 failures, the Fortran-order failure, and three days of noisy
     decisions.
2. **A calibrated measurement layer.**
   - Every experiment logged with a code hash, config, metric, standard error, split (tune / held-out / halves) and
     selection history.
   - Automatic shrinkage of predicted gains by change type: exact ≈1.0, learned ≈0.8, selected ≈0.2, fitted
     in-sample ≈0.1.
   - Submission decisions on expected value net of shrinkage.
   - *Prevents:* the selection-effect losses and the over-optimistic knob refit.
3. **Objective-backward sensitivity as a standard tool.**
   - An adjoint or perturbation map of the score with respect to every intermediate (rows, ranks, layers, terms),
     next to the cost profile.
   - Look for low-value computation and for structure in what is computed: dominant modes, functional
     dependencies.
   - *Prevents:* six days of missing the final-layer cut.
4. **Portfolio management with an explicit exploration budget.**
   - Parallel tracks: exploit (knobs, kernels), port (public methods), structural (rebuild), learned (training).
   - Each track with its own budget and a review cadence.
   - Stopping rules based on measured diminishing returns *and* a mandatory divergent-generation pass. Never
     "my list is empty".
   - Option value priced in: free slots and idle compute make exploring cheap; submitting is not nominating.
   - *Prevents:* premature stops, and the late structural attempt.
5. **Parallel, isolated workers with a shared lab notebook.**
   - A lead agent assigns tracks to subagents in isolated worktrees and sandboxes.
   - Results go to a structured store (a queryable closed-ideas registry with dedupe), not markdown.
   - Claims that matter get an adversarial reviewer agent, so overclaims and attribution errors are caught before
     they ship.
   - *Prevents:* serial queues, rediscovery, and doc drift.
6. **Hygiene enforced by tools, not memory.**
   - A job scheduler with CPU and memory budgets.
   - Hooks that block self-matching kill patterns and in-place edits of running scripts.
   - One parameterized code base with bitwise-equivalence tests; builds generated from configs.
   - Durable artifacts: tools in the repo, checkpoints that survive container restarts.
   - *Prevents:* the self-kills, OOMs, lost jobs and build sprawl.
7. **Resource awareness and escalation.**
   - The system should ask for what the problem needs: network access to the labelled dataset on day 1, a GPU for
     label generation and teacher training, a bigger memory allotment.
   - It should estimate the value of each before asking.
   - *Prevents:* the CPU-bound ceiling on learned components.
8. **Provenance and integrity built in.**
   - Every idea and code fragment tagged with its source (own, public write-up, licensed code, teammate), feeding
     the write-up's credits automatically.
   - Hard isolation from other participants' data, even when an API leaks it.
   - Credentials never in logs.
   - *Prevents:* the attribution errors; it handled the leak correctly already.
9. **A human role focused where it matters.** Humans set objectives and constraints, grant access, add team
   context, and own irreversible or public actions (nominations, posts). Persistence should *not* need a human:
   the system should not need "try harder" to search further. The fact that it did here is the clearest gap
   between this agent and an ideal one.
10. **Communication that fits the medium.** Decision-ready summaries (the state, the one recommendation, what is
    needed from the human), with detail linked rather than inlined. A live dashboard of experiments and
    submissions would replace most status messages.

**A rough autonomy scale, for placing this case.**
- L1: the AI writes code on request.
- L2: it runs experiments it is told to run.
- L3: it chooses and runs experiments and submissions within human-set goals, with humans supplying access,
  direction and persistence.
- L4: it manages its own research portfolio and resources, asks for escalation when needed, and needs humans
  only for goals and irreversible actions.
- L5: it sets research goals.

This session sat at a solid L3. The gaps to L4 are exactly items 3-7 above.

---

## 13. Open questions for research

1. **Persistence prompts.** Is the effect of "try harder" prompts a real information signal (the human knows the
   field has slack), a change in the agent's internal stopping threshold, or chance? Pre-registered push / no-push
   comparisons on comparable problems could tell.
2. **Generation versus verification.** How much of the gain is attributable to the agent's idea generation, as
   opposed to its implementation and verification of others' ideas? Section 8 is a first cut. A finer attribution
   at the level of individual commits is possible from the logs.
3. **AI-generated brainstorms.** How well calibrated are they ("twenty-agent brainstorms", the agent's own
   estimates) on the size of an improvement, as opposed to its direction? Here: good direction, factor 2-5 off in
   magnitude, in both directions.
4. **Compaction.** What is the effect of context compaction on procedural reliability, as opposed to factual
   recall? Here: facts held, procedures regressed.
5. **Proxy oracles.** How should an agent value a scarce, slow, noisy external oracle (graded slots) against cheap
   local proxies, when the proxies are biased by selection?
6. **Ethics and process.** What norms are needed when agents interact with competition infrastructure: API leaks,
   participant code, disclosure of AI use in write-ups?
7. **Hardware.** Would the same agent with a GPU and parallel workers have closed the structural gap, or is the gap
   one of insight? The day-6 reconstruction suggests mostly insight plus retraining, both of which a stronger
   system could attempt.

---

## Appendix A: the user's messages, verbatim

Long pasted lists are shortened where marked; attachments are shown by file name.

1. Oct 14 14:23: "@Foundation_Model_Oct_13.pdf Fully replicate the algorithm described in the PDF. This solution corresponds to submission https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/335910 performance after adding additional fixes and boosts. Compose scripts for automating zip/submissions upon updating the git repo scripts. Fully test out your work before announcing it's complete. Direct push to the repo"
2. Oct 14 22:53: "@Meeting_summary.pdf As a team we had a meeting to discuss the next steps. I am one of the "other students". Take reference of that and try to make use of the discussions @learned_relu_step_v5b_mp_l8_strassen5.tar.gz Take reference of another code attempting to implement the proposed algorithm. Try to analyze it and take advantage of it regarding parts where its implementation did well"
3. Oct 15 01:57: "@13_Improvements_to_Option_B_from_twenty_agent_brainstorms.md @14_Twenty-agent_improvements_to_Option_B.md @15_Reconstruction_of_the_leading_Phase_2_estimators_from_forum_and_leaderboard_evidence.md @aicrowd_token My teammate tells me that these approaches may could get 10-50% further cost reduction. Evaluate that and consider incorporating them into our solution. Research about next steps. You should make submissions whenever possible to push top ranks in leaderboard."
4. Oct 15 14:45: "You should be able to submit now. Permission enabled for aicrowd access."
5. Oct 15 20:58: "@16_Consolidated_improvements_to_the_implemented_Option_B_estimator.md Teammate's progress and suggestions here. You may take reference and try to improve the algorithm with that." [followed by note 16's 17-item list: skip dead/saturated neurons; exact rewrite of the third-order memory; billing engineering; trim the last two layers; rank schedule by layer; retire the old-source tier; cleaner labels; better validation; scale correction; audit the analytic ReLU step; better fourth-order terms; train on the chain's own states; teacher chain; restructure the corrections; better inputs; capacity; final-layer calibration]
6. Oct 16 00:08: "Analyze the current progress. Compare them to the top of the leaderboard. What are the things that would worth trying and what are the things holding potential for pushing ranking further more? Think comprehensively and plan with exhaustive thoughts. 10 submissions remaining today"
7. Oct 16 01:55: "Try to test out more experiments by drawing inspirations from research works on relevant contexts. Be thoughtful on our approaches and make better inference about why they worked, why some experiments didn't work, and try to make more submissions based on better designs of algorithms."
8. Oct 16 06:26: (the same) + "Try to make more careful, thoughtful attempts by sophisticated reasoning about potential areas to improve. You're already doing great job! Try hard please"
9. Oct 16 16:15: "Please note that there are still 3 remaining submissions available today. So please try hard and be creative in finding innovative ways. You are in a good spot!"
10. Oct 16 22:07: "Please brainstorm for more ideas that may turn out to be helpful later with more performance data. Comparing against the top teams, especially those who are able to move up ranks quickly, may be helpful. You may refer to relevant resources and researches on adjacent fields for inspirations about what might work. Try to be more comprehensive, creative, and critical at spotting issues in current workflow and submissions and algorithms and Please really try to make some more progress"
11. Oct 17 02:40: "Please try hard to find more innovative ways that knock off the limitations in our current approaches. Try to think harder regarding what hides possibilities and from the perspective of first principles. Make more thoughtful attempts please. Make more local experiments before submitting solutions so that we can be confident that uploaded submissions would push performance higher."
12. Oct 17 03:08: "Keep going. Try hard to brainstorm for more ideas to attack the problem in different angles. Try to think of refactor possibilities, especially try to figure out how the top teams are able to achieve that high from the similar base algorithm idea share from the aicrowd discussion forum"
13. Oct 17 14:00: "@submission-20261017-011512.tar.gz Take reference of my teammate's research note, along with submitted code, for inspiration on pushing the result and ranking:" [teammate note: built off the v5b code, compute 25% → 23%, 3% local improvement; what worked (feedback rank 16→1, residual rank 16→8, deeper covariance Strassen, D21 batch 2→4) and nine things that did not]
14. Oct 17 16:11: "For the final submissions, please try to be considerate and thoughtful regarding what to do next. Conduct more research and try exhaustively for the pushes of potential upward ranking. Try to behave like an very experienced competition participant and think about what are the best to do"
15. Oct 17 19:01: "Keep going, push for more improvements by inspecting on current approach and trying hard to find out innovative ways for major and minor improvements. Consider make more experiments on potential areas of enhancement and refactor"
16. Oct 17 19:14: "Consider taking reference of forum's writeups for inspiration and directions of improvements possible. Try to make guesses about how other competitors reach such ideas on attacking the problem with more efficient resources. Keep going"
17. Oct 17 19:26: "Please actively search for information for clues for improving algorithm further, both on aicrowd and the internet and relevant scholarly research." + links to forum posts 18248 and 18249
18. Oct 17 19:33 / 19:33 / 19:37: "Network access for the sites you listed has been enabled. Try to seek for more inspirations and draw useful reasonings" (three times; two interrupts in between)
19. Oct 17 21:40: "Keep going, push for more improvements by inspecting on current approach and trying hard to find out innovative ways for major and minor improvements. Consider refactoring and make independently verifiable edits. Especially seek inspirations from external sources"
20. Oct 17 21:49 and Oct 18 02:18: "what would be the next steps for pushing our ranking further" (plan mode). Plan rejections: "assign more tasks before testing it out. I mean to brainstorm more ideas and experiment them locally"; "More local experiments that try to validate ideas before submitting to waste the submission slots. Seek for more external inspirations"
21. Oct 18 03:12: "Compare the current leaderboard. Try to seek for potential for improvements on algorithms by carefully analyzing weaknesses and unnecessary items and by referring to other teams' performances. Note that there are errors while scoring today's submission. Take account of this and try to brainstorm for more ideas and improvement strategies. Keep going with brilliant mind that generates excellent thoughts of enhancements!"
22. Oct 18 12:50: "Please try more experiments by thinking hard about them. Refer to current leaderboard to find trends of how other teams improve results" (plan approved)
23. Oct 18 14:15: "Analyze the current situations we are facing. Where does improvement potential hid? What is making unnecessary compute and would the accuracy has more room to improve?"
24. Oct 18 16:32: "7 submissions remaining today. Please conduct extensive research that covers comprehensively on areas to further improve and make attempts with confidence after validating locally."
25. Oct 19 17:53: "Keep improving with the remaining slots and validate locally first"
26. Oct 19 20:34: "Please look at other teams' performance and try to seek inspirations from comprehensive analysis of our approach and look for external sources to max utilize the final two slots today"
27. Oct 19 20:55: "Check the forum and discussions for more inspirations"
28. Oct 20 00:33: "Create docs/ALGORITHM.md, a clean, verified spec of the current cn69 estimator that another AI could use to improve it. ..." (sections listed; also PROGRESS doc, gate tools into scripts/, links, commit)
29. Oct 20 01:31: "Please try more experiments by thinking hard and proposing exhaustive cracking options about them. Refer to current leaderboard to find trends of how other teams improve results. Take reference of arXiv etc research papers and recent progress on similar challenges"
30. Oct 20 13:51: "Can you think of a better algorithm that the leaders may use?"
31. Oct 20 15:04: "don't think that work is heavy and costly. Please really act as a responsible, worthy talent that would really try to brainstorm about top teams' constructions"
32. Oct 20 22:34: "yes, start the layer-by-layer comparison now"
33. Oct 20 22:54: "Please try to find experiments to try and make good use of the remaining time"
34. Oct 21 12:37: "draft the write-up. Make the final efforts"
35. Oct 21 12:51: "You still have 10 submissions today. Try to brainstorm and capture the last chances"
36. Oct 21 15:17: "Please try harder and think of genuinely creative approaches"
37. Oct 21 17:36: "Please try to brainstorm for more ideas. I know it's hard but keep pushing. You should be confident that something is waiting for discovery"
38. Oct 22 01:29: (this request) "Help me compose a synthesis that features everything we discussed, every improvements/attempts we made in this conversation and this repo and every submission to the competition. ..."

Also recorded: automatic stop-hook messages ("There are untracked/uncommitted changes ... commit and push"), about
10 of them; 245 task-completion notifications; one manual `/compact`.

## Appendix B: session statistics

| item | value |
|---|---|
| elapsed | Oct 14 14:23 → Oct 22 01:29 (179 h); assistant active in 130 distinct hours |
| model requests | 4,085; ~4.23M output tokens; ~1.77B cache-read tokens; ~18.7M cache-write tokens |
| recorded thinking time | ~7.5 h over 3,799 turns |
| tool calls | Bash 3,443; Monitor 163; TaskUpdate 131; TaskCreate 74; Write 59; Edit 58; Read 27; TaskStop 24; ToolSearch 21; WebSearch 17; WebFetch 7; ExitPlanMode 6; Agent (subagents) 3; scheduled-task calls 15 |
| background task outputs | 545 |
| context compactions | 12 (11 automatic, 1 manual) |
| commits | 223 (Oct 14: 4, 15: 5, 16: 42, 17: 69, 18: 38, 19: 37, 20: 16, 21: 12) |
| candidate builds | 92 directories |
| docs | 10 files in `docs/` + README, ~3,600 lines before this one |
| submissions | 58: 45 clean, 6 partial, 7 failed; 24 new bests |
| client-reported cost estimate | ~$619 (API-equivalent; client-side estimate) |

## Appendix C: where things are

| path | what |
|---|---|
| `docs/ALGORITHM.md` | verified spec of the final estimator line, cost breakdown, error anatomy, closed ideas, open levers, gate |
| `docs/WRITEUP_phase2.md` | competition write-up draft (team name and disclosure to be filled in) |
| `docs/experiments_2026-10-16.md`, `docs/experiments_2026-10-17.md`, `docs/final_phase_2026-10-17.md` | day-by-day experiment logs |
| `docs/brainstorm_2026-10-17.md`, `docs/leaders_2026-10-20.md`, `docs/status_2026-10-16.md`, `docs/notes_13-15_evaluation.md`, `docs/PROGRESS_2026-10-19.md` | reviews, leaderboard analyses, teammate-note evaluations, status |
| `docs/submissions_phase2.csv` | all 58 submissions |
| `candidates/v5b_*` | every build (cn79 / cn79r are the recommended nominations) |
| `scripts/gate`, `scripts/hf`, `scripts/corr*`, `scripts/tchain`, `scripts/rowf`, `scripts/fast` | gate tools, exact-label pipeline, corrector training, leaders'-construction prototype, final-layer patches, fast mode |
| `submission/`, `fm/` | the day-1 replication framework (never graded) |
