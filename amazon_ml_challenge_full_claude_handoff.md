# Amazon ML Challenge 2026 — WAR ROOM
# FULL HANDOFF CONTEXT FOR CLAUDE OPUS XHIGH
## Business Entity Resolution — Strategic Red-Team Research Mission

> **Read this entire document before answering.**
>
> You are being brought into an ongoing 72-hour competition project. You are NOT starting from scratch.
> The purpose of this document is to transfer the relevant project history, measured results, current hypotheses, previous red-team findings, and the exact unresolved question.
>
> Your job is to independently challenge the current thinking and identify the **highest-value next experiment**.

---

# 0. TEAM / ROLES

Human:
- Final decision maker.
- Runs the competition and decides what gets executed.

ChatGPT:
- Strategic brain.
- Reasoning, experiment design, result interpretation, debugging guidance, decision support.
- Should challenge weak assumptions rather than blindly agree.

Google Antigravity:
- Primary coding/execution agent.
- Runs experiments and reports results.

Claude Opus XHigh:
- Independent red-team reviewer/research strategist.
- Your role in this handoff is to deeply investigate the current bottleneck and prevent wasted experimentation.

Project principle:

> **Measured experiments > random experimentation.**

For every experiment consider:
1. Expected improvement
2. Time required
3. Compute required
4. Risk
5. Information gained

We are under a 72-hour deadline. We normally want only **1–3 highest-value next actions**, not 20 ideas.

During the final 12 hours:
- prioritize reliability
- prioritize proven improvements
- avoid risky architecture changes
- protect enough time for final submission

---

# 1. COMPETITION PROBLEM

Business Entity Resolution.

Inputs:
- S1 = reference entities
- S2 and S3 = source entity tables

For every S1 entity, predict **all matching S2 and S3 IDs**.

Possible outcomes:
- zero matches
- one match
- multiple matches

The final candidate set is also part of the submission:
- `candidate_pairs.tsv`
- Every final predicted match must exist in the candidate set.

Metric:
- **S1-level macro F0.5**
- beta = 0.5
- precision is more valuable than recall

Important metric behavior:
- True singleton/no-match + empty prediction = 1.0
- True singleton/no-match + any false match = 0.0
- False merges are expensive.

Competition constraints:
- Train contains US + India.
- Test contains US + India + **France**.
- France is unseen in training.
- External lookup/data augmentation is strictly prohibited.
- No commercial APIs, government databases, geocoding, external business databases, or internet augmentation.
- Models must satisfy the competition's license/size constraints (MIT/Apache 2.0, <=8B parameters).

---

# 2. DATASET SCALE / INITIAL EDA

Train:
- S1: 2,206,821
- S2: 5,034,616
- S3: 5,285,603
- GT rows: 2,206,821
- Matched pairs: 7,638,365

Test:
- S1: 1,732,544
- S2: 4,887,273
- S3: 5,082,316

Important train structure:
- 5.58% S1 are true empty/no-match cases.
- Mean matched records/S1 = 3.46
- Median = 3
- Maximum = 11
- 80.48% have both S2 and S3 matches.
- 6.48% only S2.
- 7.45% only S3.
- Every S2/S3 links to at most one S1 in train.
- 0 cross-country matches observed.
- ~26.6% S2 records unmatched/distractors.
- ~25.4% S3 records unmatched/distractors.

Missingness:
- Name is always populated.
- Addresses missing ~3.3% of train S2/S3.
- Addresses missing ~2.7% of test S2/S3.

Country:
- Train S1 ~60% US / ~40% India.
- Test S1 ~38.3% US / ~46.8% India / ~15% France.
- France is therefore a substantial test population, not a tiny edge case.

Noise observed:
- abbreviations
- legal suffix variation
- DBA/trade names
- punctuation differences
- word-order changes
- typos
- transliteration
- partial addresses
- reordered addresses
- landmark-style addresses
- missing fields
- Indic scripts / mixed scripts
- accented characters
- severe typos
- phonetic substitutions
- brand/domain aliases

---

# 3. EARLY RESEARCH / EDA FINDINGS

## 3.1 Exact matching is weak

Streaming normalization used:
- NFC
- lowercase
- punctuation -> space
- hyphen -> space
- null removal
- whitespace collapse

Runtime was ~21 minutes for ~12.5M records.

Findings:
- ~78% of true pairs have no exact normalized-name match.
- Exact name+address finds only ~2.78% of S2 true pairs.
- Exact name+address finds ~0.01% of S3 true pairs.
- Adding country to exact matching adds negligible disambiguation.
- Name+address is highly unique when present.

Name uniqueness:
- S1: ~60.72% unique; ~39.28% repeated.
- S2: ~72.12% unique.
- S3: ~73.41% unique.
- Maximum S1 name frequency: 253.
- Name+country uniqueness adds only ~0.03% S1, ~0.08% S2, ~0.13% S3.

Exact-name true-pair coverage:
- S1→S2 ~21.44%
- S1→S3 ~22.23%
- US roughly ~25.8%
- India roughly ~14.9–16.8%

India:
- more duplicate names
- lower exact-name recall
- more transliteration ambiguity

Conclusion:
> Exact name is not a sufficient backbone. Country is useful as a hard partition but weak as a disambiguator. Address is valuable but noisy. Fuzzy token/character retrieval is necessary.

---

# 4. RETRIEVAL DEVELOPMENT

## RECON-03 — n4addr retrieval

Initial implementation had a bug:
- S2 candidates were appended first and then `cands[:K]` was used.
- This starved S3 candidates.
- The implementation was corrected before trusting results.

Corrected n4addr recall:

| n4addr candidates | Candidate recall |
|---|---:|
| 20 S2 + 20 S3 | 91.46% |
| 50 S2 + 50 S3 | 93.02% |
| 100 S2 + 100 S3 | 93.99% |

At 50+50:
- S2: 93.83%
- S3: 92.48%
- US: 94.65%
- India: 90.58%

At 100+100:
- S2: 94.71%
- S3: 93.57%
- US: 95.42%
- India: 91.85%

Missing-address target-record recall at 100+100 was ~52.39%.
Entity-level recall for S1s with empty targets was ~83.24%.

Common-name behavior @200:
- freq 1: 95.51%
- freq 2–5: 93.17%
- freq 6–20: 90.56%
- freq 21–100: 89.70%
- freq >=101: 94.72%

Interpretation:
> Retrieval is strong enough to move forward, but absolutely not solved.

---

# 5. RETRIEVAL IMPROVEMENT — RECON-04

Baseline:
- n4addr 50 S2 + 50 S3

Word-token name-only fallback:
- +10 S2 +10 S3 gave ~+0.79% additional recall
- +20 S2 +20 S3 gave ~+0.99%

Char4 name-only fallback:
- standalone recall ~48.38%
- exclusive recovery +1.53 points
- union ~94.55%
- average candidates ~114/S1

Detailed result:
- baseline ~93.10%
- exclusive recovery ~+1.41 points
- union ~94.50%

Union:
- US ~95.57%
- India ~92.92%
- S2 ~94.89%
- S3 ~94.14%

Missing-address target recovery:
- improved by ~17.52 points net
- union ~66.78%

Current retrieval configuration:
- n4addr: 50 S2 + 50 S3
- char4 name-only: 10 S2 + 10 S3
- ~114 unique candidates/S1

Critical interpretation:
> ~94.5% is candidate recall on a validation sample. It is NOT the final competition score.

---

# 6. RECON-05 — FIRST SCORER BASELINE

Validation design:
- 3,995 S1 sample
- 1,994 train S1
- 2,001 held-out validation S1
- 50/50 S1 split

Candidate pool:
- ~114 candidates/S1

Features:
- character name similarities
- token name similarities
- address similarities
- missingness
- name rarity
- source indicator
- other basic pair-level signals
- ~22 total features in current implementation

Model:
- Logistic Regression
- StandardScaler
- threshold = 0.58

Validation methodology:
- candidate generation unsupervised
- features label-free
- validation labels not used for scaling/model/threshold selection

Runtime:
- ~12.8 min total
- ~12.6 min candidate generation/caching
- ~8.5 sec features
- ~1.2 sec LR training/tuning

Validation results:

| Metric | Result |
|---|---:|
| Pair precision | 86.56% |
| Candidate-pair recall | 75.83% |
| End-to-end recall | 71.37% |
| Pair F0.5 | 84.18% |
| FPR | 0.346% |
| FNR | 24.17% |
| TP | 4,939 |
| Predicted pairs | 5,706 |
| Retrieved true | 6,513 |
| Total GT | 6,920 |
| **S1 macro F0.5** | **77.59** |

S1-level:
- US: 80.08
- India: 73.87
- mean predicted matches/S1: 2.85
- median: 3
- empty predictions: 78/2001 = 3.90%
- true singleton/no-match: 112/2001 = 5.60%

Score distributions:

True candidate pairs:
- mean .758
- median .9221
- P25 .6057
- P75 .9861
- P90 .9962

False candidate pairs:
- mean .0072
- median .0003
- P75 .0012
- P90 .0052
- P99 .1419

Additional:
- 98.13% of false pairs score <.05
- 53.23% of true pairs score >=.90
- 70.93% of true pairs score >.70

Failure modes observed:
1. missing-address collapse
2. singleton/no-match false positives
3. India transliteration ambiguity
4. shared-address clones scoring extremely high (~.9963)

---

# 7. CLAUDE OPUS FIRST RED-TEAM — IMPORTANT CONTEXT

An earlier Claude Opus XHigh red-team review analyzed RECON-05.

Main conclusions:

### Retrieval
- Exact name is not the backbone.
- Country hard partition is sound.
- n4addr + char4 ~94% candidate recall is a good foundation.

### Metric interpretation
- Pair F0.5 84.18% is conditioned on candidate recall.
- Honest end-to-end pair F0.5 was estimated around ~83%.
- S1 macro F0.5 = 77.59 remains the key internal metric.
- Overall 77.59 ± roughly ~1.3 points because validation sample is only ~2,001 S1.
- Subgroup estimates with n~100–200 are noisy.

### Validation audit
Opus judged validation methodology partially trustworthy:
- candidate generation unsupervised
- full candidate generation for validation
- features label-free
- S1 split clean given at-most-one-parent structure
- singleton F0.5 implementation appears correct

Critical check:
- verify threshold .58 was tuned only on the training fold
- restate pair F0.5 correctly
- acknowledge France is absent from validation

### Recall decomposition

Retrieved true:
- 6,513 / 6,920 = ~94.12%

Scorer TP:
- 4,939 / 6,513 = ~75.83%

End-to-end recall:
- 4,939 / 6,920 = ~71.37%

Recall gap:
- retrieval loss: ~5.88 percentage points
- scorer loss: ~22.75 percentage points

Thus scorer losses dominate raw recall loss by roughly 4:1.

However, Opus made an important F0.5 point:
- precision gains are more valuable than equivalent recall gains around this operating point
- entity-level macro metric concentrates precision damage

### Opus recommendation at that point

Opus suggested:
1. duplicate-claim resolution + margin-aware no-match gate
2. interaction-augmented LR before LightGBM

It specifically said:
- do NOT immediately do LightGBM before entity-level analysis
- do NOT expand retrieval broadly yet
- do NOT optimize pair F0.5 instead of S1 macro F0.5
- do NOT blindly hard-code at-most-one-parent

Expected lever estimates were qualitative:
- entity-level FP control: potentially +1 to +3 S1 F0.5
- shared-address/missing-address feature: potentially +0.5 to +1.5
- missing-address retrieval: likely <1 overall
- France hidden loss remained unknown

---

# 8. RECON-06 — FOLLOW-UP TO OPUS

We then executed Opus's recommended entity-level experiment.

This result is extremely important because it **falsified the proposed duplicate/gating branch**.

## 8.1 Threshold audit

Code:
- `recon05_baseline_scorer.py`
- threshold grid `np.arange(.10,.96,.02)`
- best threshold .58

Audit:
- threshold selected using `train_s1_ids` only
- validation used only afterward
- StandardScaler fit on X_train only
- **100% clean / no leakage**

## 8.2 Duplicate-claim structural test

At threshold .58:

- number of S2 records predicted for >1 S1 = **0**
- number of S3 records predicted for >1 S1 = **0**
- total unique predicted candidates = 5,706
- 0/2,001 validation S1 affected
- 0/767 FPs attributable to duplicate claims

Raw retrieval pool:
- 10,521 candidates appeared in >1 S1 query
- 99.8% scored <.05
- exactly zero duplicate claims survived threshold .58

Conclusion:
> Duplicate-claim resolution is NOT the current bottleneck.

## 8.3 Singleton/no-match FP breakdown

True singleton/no-match:
- 112 entities

Entities receiving >=1 FP:
- 76

Total singleton FPs:
- 111

Breakdown:

| FP type | Share |
|---|---:|
| Duplicate claims | 0% |
| Name match + address disjoint distractors | 4.5% |
| Shared-address conflicts | 17.1% |
| Other/combined token overlap (partial name tokens + city/state) | 78.4% |

FP scores:
- min .5830
- P25 .8314
- median .9009
- P75 .9424
- max .9919

This is the key discovery:

> **The singleton false positives are often HIGH-CONFIDENCE, not weak threshold-adjacent errors.**

## 8.4 Post-processing tests

All post-processing parameters were tuned only on the 1,994 training S1 and evaluated on untouched 2,001 validation S1.

Validation baseline:
- **77.59**

Results:
- duplicate top-S1 variants: 77.59
- max-score gate <.62: 77.57
- <.65: 77.48
- <.70: 77.45
- <.75: 77.05
- solitary gates also degraded

Best:
- **77.59**
- predictions removed: 0
- true matches lost: 0
- FPs removed: 0
- no improvement

Runtime:
- ~23.6 sec feature extraction
- ~1.2 sec scoring
- ~0.4 sec grid
- CPU <2GB

Antigravity concluded:
> Abandon entity-level duplicate resolution/post-processing; proceed to GBDT/LightGBM candidate scoring.

But this last sentence is a **recommendation**, not a proven conclusion.

---

# 9. CURRENT STRATEGIC INTERPRETATION

This is where we need YOU to be skeptical.

Strong evidence:

### Observed fact 1
Retrieval candidate recall is ~94.5%.

### Observed fact 2
End-to-end recall is ~71.37%.

### Observed fact 3
S1 macro F0.5 is 77.59.

### Observed fact 4
Singleton/no-match FPs are high confidence:
- median .9009
- P75 .9424
- max .9919

### Observed fact 5
Simple threshold/no-match gates degraded overall macro F0.5.

### Observed fact 6
Duplicate claims caused zero surviving FPs.

### Current hypothesis
The scorer is a major bottleneck.

### But unresolved hypothesis
There are TWO competing explanations:

**H1 — Model-capacity problem**
> LR is too simple to model nonlinear interactions among name, address, rarity, missingness, source, etc. A stronger tree model such as LightGBM should improve discrimination.

**H2 — Representation/information problem**
> The current ~22 features do not contain enough information to distinguish the high-confidence false positives from true matches. A stronger model may improve interactions but cannot recover information that was never represented.

This distinction is the reason for this research mission.

---

# 10. THE BIG QUESTION

## Do the current ~22 features contain enough information?

We need to know whether the problem is:

> **“Our model is too weak.”**

or:

> **“Our representation is too weak.”**

A LightGBM run is only high-value if the first explanation is plausible.

We do NOT want to burn several hours on random model upgrades without understanding this.

---

# 11. WHAT YOU MUST INVESTIGATE

## A. Feature sufficiency vs model capacity

Determine whether current features plausibly contain enough information to separate:
- true matches
- high-confidence false matches

especially singleton/no-match FPs scoring 0.90–0.99.

Investigate:

1. If two candidate pairs have nearly identical current feature vectors but opposite labels, what does that imply?
2. Are current features expressive enough to encode distinctiveness?
3. Can they encode candidate competition?
4. Can they encode cardinality?
5. Can they encode S2/S3 agreement?
6. Can they encode retrieval rank?
7. Can they encode top-vs-second margin?
8. Can they encode token/address rarity?
9. Can they encode whether a candidate is plausible relative to all candidates for the same S1?
10. Are there structural signals already in S1/S2/S3 that the current pair representation discards?

If possible, reason about what the false-positive archetypes imply about missing information.

---

# 12. INVESTIGATE THE DECISION FORMULATION

Do not assume independent pair classification is the correct formulation.

Analyze whether this is better understood as:

- independent pair classification
- set prediction
- cardinality prediction + candidate selection
- adaptive thresholding
- top-k selection
- calibrated probability + expected F0.5 optimization
- structured prediction
- graph matching / constrained optimization
- another formulation

Important:
- each S1 may have zero, one, or many matches
- train max is 11
- every S2/S3 links to at most one S1 in train
- but do NOT blindly assume the at-most-one-parent property is guaranteed on test

Investigate whether:

> **Predicting the number of matches for an S1, then selecting candidates**

could outperform a single global pair threshold.

---

# 13. F0.5-SPECIFIC ANALYSIS

The real metric is **S1-level macro F0.5**, not pairwise classification F0.5.

Analyze:

1. If candidate probabilities are calibrated, what decision rule maximizes expected F0.5 for one S1?
2. Should we select a set rather than independently threshold pairs?
3. Is top-k selection useful?
4. Can k be predicted?
5. Can top-vs-second margin determine k?
6. How should true-empty entities be handled?
7. Is there a principled reason a model with worse pairwise metrics could still produce better S1 macro F0.5?
8. Could per-entity calibration matter more than pair calibration?

If you derive formulas, keep them practical and tied to this competition.

---

# 14. “FREE INFORMATION” ALREADY IN THE DATA

We cannot use external data.

Look for information already available in the supplied dataset but not represented in the ~22 pair features.

Especially investigate:

## Retrieval information
- n4addr rank
- char4 rank
- combined rank
- retrieval score
- rank percentile
- which retrieval method found the candidate

## Candidate competition
For each S1:
- best candidate score
- second-best candidate score
- top-vs-second margin
- score distribution
- number of plausible candidates
- number of high-similarity candidates
- source-specific candidate counts

## Cardinality
- candidate count
- candidate count by source
- train relationship between feature patterns and actual match count

## Cross-source agreement
- S2/S3 independent support
- whether evidence converges on the same entity
- whether multiple records reinforce one another

## Rarity
- token frequency
- address-token frequency
- city frequency
- state frequency
- combinations of tokens
- source-specific rarity

## Shared-address structure
- how many records share an address?
- when is shared address legitimate?
- when does it create false merges?
- does shared-address cluster size predict true match cardinality?

## Source behavior
- S2 vs S3 systematic differences
- source-specific missingness
- source-specific error modes

## Graph structure
Potentially model S1/S2/S3 as a graph.

Investigate:
- connected components
- shared name/address token relationships
- source-to-source relationships
- whether graph consistency provides information missing from pairwise features

Do NOT invent a graph solution just because it sounds sophisticated. Require evidence.

---

# 15. HIGH-CONFIDENCE SINGLETON FP DEEP DIVE

This is probably the most valuable failure mode.

Known facts:
- 112 true singleton/no-match S1
- 76 got >=1 FP
- 111 total FPs
- median FP score .901
- P75 .942
- max .992
- 78.4% are combined/partial-token overlap
- 17.1% are shared-address conflicts

Investigate:

- Are generic tokens causing the problem?
- Are partial name tokens too influential?
- Is city/state overlap dominating?
- Is address evidence too weak?
- Is the model missing “distinctiveness”?
- Is there no feature saying “this evidence is common”?
- Does candidate rank help?
- Does top-vs-second margin help?
- Does candidate count help?
- Does S2/S3 consistency help?
- Is the model confusing “plausible business” with “same business”?

Most importantly:

> What is the cheapest diagnostic that can tell us WHY these high-confidence FPs look so convincing?

---

# 16. FRANCE / OUT-OF-DISTRIBUTION CONCERN

France is ~15% of test S1 and absent from train.

Analyze:

1. Which current features are likely country-agnostic?
2. Which rely on US/India distributions?
3. Could rarity features shift badly?
4. Does country hard partition remain safe?
5. Can we simulate OOD behavior using existing US/India data?
6. Can we create artificial distribution shifts from train?
7. What diagnostics should be run before trusting 77.59 as representative of test?

Do not invent France performance.

---

# 17. IMPORTANT METRIC DECOMPOSITION

Validation:

- Retrieved true = 6,513
- Total GT = 6,920
- Retrieval ceiling = 6,513 / 6,920 = ~94.12%
- Scorer TP = 4,939
- Scorer efficiency = 4,939 / 6,513 = ~75.83%
- End-to-end recall = 4,939 / 6,920 = ~71.37%

Recall gap:
- retrieval loss ≈5.88 points
- scorer loss ≈22.75 points

Therefore scoring is the dominant raw recall bottleneck.

BUT:
- competition metric is F0.5
- S1 macro averaging makes false-positive-heavy entities especially important

Do not simply say “maximize recall.”

---

# 18. PREVIOUS OPUS RECOMMENDATION — NOW FALSIFIED

Previous Opus recommended:

1. duplicate-claim resolution + margin-aware no-match gate
2. interaction-augmented LR before LightGBM

We tested #1.

Result:
- duplicate claims at final threshold: ZERO
- no-match gates: all degraded overall score
- no improvement over 77.59

Therefore:

> **Do not re-recommend duplicate-claim resolution or simple threshold/no-match gating unless you identify genuinely new evidence.**

This is an important part of the project history.

---

# 19. DO NOT BLINDLY FOLLOW THESE ASSUMPTIONS

Challenge all of them.

### Assumption A
“LR is weak because it is linear.”

Alternative:
- representation may be insufficient.

### Assumption B
“Singleton FPs can be fixed by a no-match threshold.”

RECON-06 argues against this.

### Assumption C
“Retrieval should be improved before scoring.”

Retrieval ceiling is already ~94%.

### Assumption D
“LightGBM will automatically solve the problem.”

A stronger model cannot recover missing information.

### Assumption E
“Pairwise F0.5 is the right proxy.”

Actual metric is S1 macro F0.5.

### Assumption F
“At-most-one-parent in train can be hard-coded.”

Do not assume this is guaranteed at test.

### Assumption G
“US/India validation tells us test performance.”

France is 15% of test and unseen.

---

# 20. YOUR REQUIRED OUTPUT

Do NOT give a generic literature dump.

Do NOT give 20 experiments.

We need a decision.

## Section 1 — Verdict

Directly answer:

> Are the current ~22 features likely sufficient?

Choose:
- likely sufficient
- likely insufficient
- uncertain

If uncertain, state the **single most important diagnostic** that resolves the uncertainty.

---

## Section 2 — Three deepest insights

Exactly 3.

For each:
- Observed fact
- Inference
- Hypothesis

Do not blur these categories.

---

## Section 3 — One game-changing hypothesis

Give exactly one.

It must:
- be testable
- potentially change architecture/strategy
- be grounded in this dataset's evidence

---

## Section 4 — ONE cheap diagnostic experiment

Design exactly one.

Target:
- <=30 minutes if possible
- CPU-friendly
- existing validation split
- no external data
- no label leakage

The experiment must help distinguish:

> feature insufficiency vs model-capacity insufficiency

Specify:
1. exact data/features
2. exact train/validation protocol
3. exact measurement
4. success criterion
5. failure criterion
6. interpretation of each possible outcome

---

## Section 5 — ONE medium experiment

Choose exactly one.

Possible categories:
- interaction-augmented LR
- LightGBM/GBDT
- rank/margin feature augmentation
- cardinality model
- top-k/set decision model
- structured/graph model
- another approach

But choose only after reasoning.

---

## Section 6 — Expected value

For each recommended experiment give:

- Expected improvement: low / medium / high
- Time: minutes / hours
- Compute: CPU / GPU
- Implementation risk: low / medium / high
- Information gain: low / medium / high

Avoid fake precision.

---

## Section 7 — What NOT to do

Maximum 3 items.

Explain why each is low-value/risky now.

---

# 21. EXTERNAL RESEARCH STANDARD

If you use external methodological knowledge:

- identify the source/concept
- distinguish source-derived claims from your own inference
- only include literature that changes our experimental decision

Priority:
1. measured results in this document
2. structural properties of this dataset
3. mathematical reasoning about F0.5/set decisions
4. relevant entity-resolution methodology
5. external literature only where it resolves a real uncertainty

Do NOT recommend:
- external business databases
- APIs
- geocoding
- web lookup
- commercial data
- internet augmentation
- giant models
- random architecture changes

The goal is not to sound sophisticated.

The goal is to find the **highest-value next experiment**.

---

# 22. FINAL DECISION FORMAT

End your response with exactly this kind of actionable conclusion:

> **Run X next because Y.**
>
> **If result A → do Z.**
>
> **If result B → do W.**

We are in a 72-hour competition. Every experiment must earn its time.

