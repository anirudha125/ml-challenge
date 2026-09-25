# AMAZON ML CHALLENGE — LIVE PROGRESS

## CURRENT STATUS

Hours elapsed: 1h
Hours remaining: 71h

Current best validation score: 77.59% S1 Macro F0.5 (RECON-05 Baseline Scorer)
Current best leaderboard score: None

Best experiment: RECON-05
Best model: Standardized Logistic Regression Scorer (threshold=0.58)
Best feature set: RapidFuzz character/token similarities + address completeness + source indicators

Current hypothesis: Entity-level post-processing (duplicate-claim resolution, heuristic singleton gates) is an ineffective precision lever (0% duplicate collision in sample; singleton FPs are high-confidence ~0.90). Signal improvement must come directly from non-linear scoring (GBDT/LightGBM) to handle conditional logic (e.g. name matching when address is null) and fine-grained feature interactions.
Current next action: Proceed to GBDT / LightGBM baseline scorer (E001).

---

## PROBLEM

Objective: Business Entity Resolution across 3 sources (S1 reference, S2 & S3 noisy records).
Target: List of matching S2 and S3 entity IDs for each S1 entity.
Metric: Macro-averaged F_0.5 score across all S1 entities (precision weighted 2x over recall; singletons with 0 matches score 1.0 if empty, 0.0 otherwise).
Submission format: Tab-separated (.tsv) `matching_results.tsv` and `candidate_pairs.tsv`.
Key Constraints: No external API lookups / databases (strict disqualification); final model MIT/Apache 2.0 <= 8B params; test set includes France (unseen in train).

---

## DATASET

Train rows: S1 = 2,206,821 | S2 = 5,034,616 | S3 = 5,285,603 | Ground Truth = 2,206,821 rows
Test rows: S1 = 1,732,544 | S2 = 4,887,273 | S3 = 5,082,316
Features: `entity_id` (string), `business_name` (string), `business_address` (string), `country` (string)
Important categorical columns: `country` (Train: US 60.0%, India 40.0% | Test: India 46.8%, US 38.3%, France 15.0%)
Important numerical columns: None (raw data is purely tabular text)

Potential leakage: Zero ID overlap between train and test. Each S2/S3 entity links to at most 1 S1 entity.
Train/test distribution issues: France (15% of test S1) does not exist in train data (zero-shot language/region transfer). ~3% of S2 and S3 records have blank addresses.

---

## EXPERIMENT HISTORY

### E000 — Reconnaissance & Health Check
Model: None
Features: Raw tabular text analysis
Validation: Full corpus inspection
Score: N/A
Leaderboard: N/A
Result: Completed workspace inspection, missing value analysis, ground truth graph analysis, and submission rule verification.
Conclusion: Memory is constrained (16 GB total RAM, ~3.5 GB free), disk is ample (420 GB free). Country is an absolute hard partition (0 cross-country matches).

### RECON-03 — BM25 Retrieval Benchmark (Corrected)
Model: BM25 (char 4-gram + address, `n4addr`, max_df=5000)
Features: Normalized business name + address n-grams
Validation: Stratified holdout on 3,995 S1 entities (seed 42)
Score: Candidate Recall = 93.02% (50 S2 + 50 S3 = 100 cands) | 93.99% (100 S2 + 100 S3 = 200 cands)
Result: S2 recall = 93.83%, S3 recall = 92.48% (at 50+50). S3 is NOT a bottleneck. Query latency ~37 ms/query.

### RECON-04 — Targeted Retrieval Fallback Diagnostic
Model: `n4addr (50+50)` baseline + Char 4-gram name-only fallback (10 S2 + 10 S3)
Features: Name+addr 4-grams (primary) + Name-only 4-grams (fallback)
Validation: Identical 3,995 S1 holdout sample
Score: Candidate Recall = **94.55%** (Union Avg Candidates = 114.0)
Result: Char 4-gram fallback exclusively recovered 195 true matches missed by n4addr (+1.53% net additional recall). On missing-address records, recall surged from 49.26% to 66.78% (+17.52% net gain). India recall rose from 90.78% to 92.92% (+2.14%). Outperformed word-token fallback by 2x.

### RECON-05 — Baseline Candidate Scorer
Model: Standardized Logistic Regression Scorer (C=1.0)
Features: RapidFuzz name & address similarities (Levenshtein, Jaro-Winkler, token sort/set/jaccard), name length ratio, S1 name frequency, address missing indicators, retrieval ranks/sources (22 features total)
Validation: Strict S1 held-out split (50% Train S1 = 1,994, 50% Val S1 = 2,001, stratified). Threshold tuned on Train S1 only (th=0.58).
Score: **S1 Macro F0.5 = 77.59%** | Pair Precision = 86.56% | Candidate Pair Recall = 75.83% | End-to-End Recall = 71.37% | Pair F0.5 = 84.18%
Result: Demonstrates that the ~114-candidate retrieval set contains rich signal. Score distributions are cleanly separated (98.1% of false candidates score <0.05, median true match scores 0.922). However, a simple linear model collapses on missing-address candidates (7.5% recall, 15/200) and suffers FP leakage on singletons (F0.5 = 32.14%).

### RECON-06 — Validate Entity-Level Precision Lever (Red-Team Audit)
Model: RECON-05 Logistic Regression (th=0.58) + Post-Processing Grid (Duplicate Resolution + No-Match Gates)
Features: Same 22 features
Validation: Leak-Free S1 split (1,994 Train S1 tuned, 2,001 Val S1 evaluated)
Score: **S1 Macro F0.5 = 77.59%** (Delta = +0.00%) | Pair Precision = 86.56% | End-to-End Recall = 71.37%
Result: **FALSIFIED HYPOTHESIS.** Audited threshold 0.58: confirmed 100% leak-free (tuned only on Train S1). Duplicate claims at th=0.58 in Val S1: exactly 0 (0.00%). Affected S1s: 0. Singleton FPs have high confidence (median score 0.9009); heuristic suppression gates prune true matches faster than FPs, reducing macro F0.5. Best post-processing policy on Train S1 is baseline (no post-proc). Abandoning entity-level duplicate resolution.

---

## CURRENT BEST PIPELINE

Preprocessing: Unicode NFC, lowercase, punctuation strip, null removal
Features: 22 pair-level features (RapidFuzz char/token similarities for name & address, completeness flags, S1 log frequency, inverse retrieval ranks)
Candidate Generation: Primary char 4-gram BM25 on (name + address) [50 S2 + 50 S3] + fallback char 4-gram BM25 on (name only) [10 S2 + 10 S3] (~114 candidates / S1)
Model: Standardized Logistic Regression (C=1.0, threshold=0.58)
Validation strategy: Stratified holdout on S1 records (Val N=2,001). **S1 Macro F0.5 = 77.59%**. Candidate Recall = 94.55%.

---

## FAILED / ABANDONED IDEAS

- Loading full 12M train records into unified in-memory pandas DataFrame: Exceeds available RAM (16 GB total, ~3.5 GB free). Must use chunked streaming, disk-backed storage, or country-level partitioning.
- **RECON-06: Entity-level duplicate-claim resolution post-processing**: At threshold 0.58, exactly 0 duplicate claims exist in the held-out validation sample (sampling density ~0.08% makes collisions non-existent). Provides 0.00% precision gain.
- **RECON-06: Heuristic no-match / singleton suppression gates**: Singleton FPs are high-confidence (median score 0.9009) caused by real token overlaps (shared address or business tokens). Fixed threshold suppression gates prune true matches from difficult non-singletons faster than they prune singleton FPs, resulting in net negative macro F0.5.

---

## CURRENT HYPOTHESES

### H1: Country-Level Hard Partitioning
Matches never cross country lines (0 out of 7,638,365 pairs in train ground truth). Partitioning train and test by `country` provides a 100% loss-free space reduction.

Evidence:
Empirical scan of 100% of ground truth links showed exactly 0 cross-country links.

### H2: Precision-Driven Thresholding for F_0.5
Because F_0.5 weights precision 2x over recall and singletons penalized to 0 on false positives, high-confidence matching and aggressive pruning will outperform high-recall / low-precision approaches.

Evidence:
Singletons comprise 5.58% of the dataset; predicting a false match drops that entity's score from 1.0 to 0.0.

---

## NEXT EXPERIMENTS

### E001
Hypothesis: Establishing a leak-free local validation pipeline with stratified splitting and exact macro F_0.5 evaluation accurately mirrors leaderboard scoring.
Change: Build local validation script and baseline candidate blocking pipeline.
Why: Essential benchmark required before evaluating any candidate generation or matching models.
Expected value: Reliable local feedback loop under 5 minutes per run.

---

## SUBMISSION STATUS

Best submission: None
File: N/A
Leaderboard score: N/A

Backup submission: None
File: N/A
Score: N/A

---

## IMPORTANT DISCOVERIES

- Country is a 100% hard partition: 0 cross-country matches across 7.6M ground truth links.
- S2/S3 records link to at most ONE S1 entity (strictly 0 multi-parent S2/S3 records).
- France represents 15% of test S1 (259k entities) but 0% of train. Pipeline must be country-agnostic and handle French text/legal entities.
- Addresses in S2/S3 are missing in ~3% of records, requiring name-centric fallback matching.
- System RAM is ~16 GB with ~3.5 GB available; operations must be chunked or partitioned to avoid OOM.
- **RECON-02:** ~78% of true pairs have NO exact normalized name match. Exact-name blocking is severely insufficient.
- **RECON-02:** Country as an exact-match disambiguation key adds only +0.03% uniqueness. Its value is as a hard partition key ONLY.
- **RECON-02:** Exact (name+address) covers only 2.78% of S2 true pairs and 0.01% of S3 true pairs. S3 addresses are extremely noisy.
- **RECON-02:** India has lower exact-name recall (14-17%) vs US (25-26%) due to transliteration noise (Indic script variants).
- **RECON-02:** The top ambiguous names are all generic US healthcare terms ("primary care group" freq=253). These require address-level disambiguation.
- **RECON-02:** S1 name+address is 100% unique (perfect deduplicated reference). Use as ground-truth identity fingerprint.
- **RECON-03:** S3 is NOT a retrieval bottleneck when budgets are split per source. S2 recall = 93.8%, S3 recall = 92.5% at 50 candidates each (overall: 93.02%).
- **RECON-04:** Name-only fallback (Char 4-gram top-10 per source) adds **+1.53% net true matches** (+195 matches in sample), driving overall candidate recall to **94.55%** at an average candidate count of only 114.
- **RECON-04:** For missing-address target records, the char 4-gram fallback boosts recall from **49.26% to 66.78%** (+17.52% net gain), resolving the biggest blind spot of address-augmented BM25.
- **RECON-04:** Char 4-gram name-only fallback outperforms word-token fallback by **2x** in exclusive match recovery (+1.53% vs +0.79%) because word tokens fail on transliteration and spelling variants.
- **RECON-05:** Baseline Logistic Regression achieves **77.59% S1-level macro F0.5** and **86.56% pair precision** at decision threshold 0.58 on held-out validation S1 entities.
- **RECON-05:** Score separation is remarkably clean: 98.13% of false candidates score < 0.05 (median false score = 0.0003), while median true match scores 0.9221. Overlap occurs in only ~3-4% of edge cases.
- **RECON-05:** Address features dominate linear weights (`addr_token_set` coef = +2.74, `addr_jw_sim` = +1.01). Consequently, missing-address candidate matches collapse to **7.50% recall** (15/200 retrieved matches passed threshold), representing a critical blind spot of linear additive scoring.
- **RECON-05:** Singletons suffer heavy penalty: Singleton macro F0.5 is only **32.14%** (36/112 singletons remained empty; 76 received at least one false positive candidate scoring >= 0.58). Because each S1 has ~114 candidates, an FPR of 0.35% still yields ~0.4 expected false positives per S1.
- **RECON-05:** India macro F0.5 (**73.87%**) trails US (**80.08%**) due to transliteration discrepancies where character/token similarity is severely reduced.
- **RECON-05:** S2 and S3 perform identically under the scorer (S2 F0.5 = 82.90%, S3 F0.5 = 83.14%), confirming S3 is not inherently harder to score once retrieved.
- **RECON-06:** Red-team threshold audit proved RECON-05 decision threshold (0.58) is 100% leak-free (calibrated strictly on 1,994 Train S1 entities, exactly 0 validation data used).
- **RECON-06:** Entity-level duplicate claims are virtually non-existent: exactly 0 out of 5,706 predicted candidates in Val S1 are claimed by >1 S1 at threshold 0.58 (0 S1 entities affected, 0 FP pairs attributable). Sparse random sampling across 2.6M records makes candidate collisions negligibly rare.
- **RECON-06:** Singleton false positives exhibit high model confidence (median max score = 0.9009), driven by genuine token similarities (shared building addresses or business name tokens). Heuristic no-match gates (0.62–0.75) prune true matches from difficult non-singletons faster than they prune singleton FPs, resulting in net negative macro F0.5.
- **RECON-06:** Post-processing direction falsified: S1 macro F0.5 improvement is 0.00% (criteria FAIL). Effort must shift to non-linear model capacity (GBDT/LightGBM) to resolve missing-address and token-interaction errors directly.

---

## COMPETITION CLOCK

Last updated: 2026-09-25T17:15:00+05:30
Current phase: Entity Precision Lever Audited (RECON-06) -> Ready for GBDT Baseline Scorer (E001)