# RECON-05: Baseline Candidate Scorer Diagnostic Report

**Date**: 2026-09-25  
**Experiment ID**: RECON-05  
**Goal**: Given the ~114-candidate retrieval set (n4addr primary + n4name fallback), determine whether a simple, interpretable scorer can reliably distinguish true matches from false candidates.

---

## 1. Executive Summary

| Metric | Result (Held-Out Val S1) | Notes |
| :--- | :--- | :--- |
| **S1-Level Macro F0.5** | **77.59%** | Evaluated on 2,001 strictly held-out S1 entities (exact competition definition) |
| **Pair-Level Precision** | **86.56%** | 4,939 True Positives / 5,706 Predicted Pairs |
| **Candidate Pair Recall** | **75.83%** | 4,939 TP / 6,513 Retrieved Matches |
| **End-to-End Recall** | **71.37%** | 4,939 TP / 6,920 Ground Truth Matches (accounting for 5.45% retrieval loss) |
| **Pair-Level F0.5** | **84.18%** | Over all evaluated candidate pairs |
| **False Positive Rate** | **0.346%** | 767 FP / 221,621 False Candidates |
| **False Negative Rate** | **24.17%** | 1,574 retrieved true matches failed threshold |
| **Decision Threshold** | **0.58** | Calibrated exclusively on Train S1 split |

---

## 2. Experimental Setup & Leak-Free Validation

- **Candidate Pool**: Deterministic union of RECON-04 retrieval:
  - Primary `n4addr`: 4-gram (name + address), top-50 S2 + top-50 S3.
  - Fallback `n4name`: 4-gram (name only), top-10 S2 + top-10 S3.
  - Total candidate pool: 3,995 S1 entities, ~114 unique candidates / S1.
- **Held-Out S1 Split**:
  - Sample S1 entities were stratified by `(Country, Match_Count_Bucket)` and split 50/50:
    - **Train S1**: 1,994 entities (227,153 candidate pairs; 6,585 true matches = 2.899% prevalence).
    - **Validation S1**: 2,001 entities (228,134 candidate pairs; 6,513 true matches = 2.855% prevalence).
  - Validation S1 ground truth was strictly withheld during preprocessing, feature scaling, model fitting, and threshold selection.
- **Model**: Standardized `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`.
- **Threshold Calibration**: Scanned $\theta \in [0.10, 0.95]$ on Train S1 entities to maximize macro $F_{0.5}$. The optimal threshold was **0.58** (Train Macro $F_{0.5} = 0.7823$). This threshold was frozen and applied to Validation S1.

---

## 3. Feature Interpretability & Model Weights

Features were normalized using `StandardScaler` fit exclusively on Train S1 pairs.

| Rank | Feature | Description | Coefficient | Odds Ratio | Impact |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `addr_token_set` | RapidFuzz token set ratio on address | **+2.7429** | **15.53** | Strongest positive predictor |
| 2 | `addr_token_sort` | RapidFuzz token sort ratio on address | -1.3639 | 0.256 | Balances token set over-generosity |
| 3 | `both_have_addr` | Both S1 and candidate have address | -1.0823 | 0.339 | Address missing intercept offset |
| 4 | `cand_has_addr` | Candidate has non-empty address | -1.0823 | 0.339 | Address presence indicator |
| 5 | `name_len_diff` | Absolute difference in name lengths | **-1.0786** | **0.340** | Strong penalty for length mismatches |
| 6 | `addr_jw_sim` | Jaro-Winkler address similarity | **+1.0136** | **2.755** | Address string alignment |
| 7 | `addr_token_jaccard` | Token-level address Jaccard index | **+0.9912** | **2.694** | Word overlap in address |
| 8 | `name_jw_sim` | Jaro-Winkler name similarity | **+0.6335** | **1.884** | Character alignment in name |
| 9 | `name_token_jaccard` | Token-level name Jaccard index | **+0.5336** | **1.705** | Shared name tokens |
| 10 | `inv_rank_addr` | Inverse rank in primary retrieval ($1/\text{rank}$) | **+0.5111** | **1.667** | High retrieval rank boosts confidence |
| 11 | `is_s2` | Source 2 indicator (vs Source 3) | -0.4927 | 0.611 | Slight base penalty for S2 |
| 12 | `addr_lev_sim` | Normalized Levenshtein address similarity | -0.4429 | 0.642 | Redundant with JW / token metrics |
| 13 | `name_lev_sim` | Normalized Levenshtein name similarity | **+0.4016** | **1.494** | Character-level edit similarity |
| 14 | `name_len_ratio` | Min length / Max length ratio | -0.2589 | 0.772 | Secondary length metric |
| 15 | `is_india` | Country indicator (India vs US) | -0.2439 | 0.784 | Accounts for higher transliteration noise |
| 16 | `name_token_set` | RapidFuzz token set ratio on name | -0.2432 | 0.784 | Penalizes subset matches with extra tokens |
| 17 | `name_token_sort` | RapidFuzz token sort ratio on name | -0.1765 | 0.838 | Collinear with token Jaccard |
| 18 | `retrieved_by_addr` | Retrieved in top-50 `n4addr` | -0.1689 | 0.845 | Binary flag |
| 19 | `s1_name_log_freq` | $\log(1 + \text{S1 name frequency})$ | +0.1133 | 1.120 | Slight frequency adjustment |
| 20 | `retrieved_by_name` | Retrieved in top-10 `n4name` fallback | +0.1042 | 1.110 | Fallback origin indicator |
| 21 | `inv_rank_name` | Inverse rank in fallback retrieval | +0.0669 | 1.069 | Fallback rank score |
| 22 | `s1_has_addr` | S1 has address | +0.0000 | 1.000 | Invariant (all S1 have address) |
| — | **Intercept** | Base log-odds | **-7.7735** | — | Reflects extreme class imbalance (~2.85%) |

---

## 4. Score Separation Analysis

Comparing model predicted probabilities for true candidate pairs vs false candidate pairs on the held-out validation set:

| Percentile / Stat | True Candidate Pairs ($N = 6,513$) | False Candidate Pairs ($N = 221,621$) | Separation Gap |
| :--- | :---: | :---: | :---: |
| **Mean** | 0.7580 | 0.0072 | +0.7508 |
| **Median (P50)** | **0.9221** | **0.0003** | **+0.9218** |
| **Std Dev** | 0.3080 | 0.0545 | — |
| **P10** | 0.1898 | 0.0000 | — |
| **P25** | 0.6057 | 0.0001 | — |
| **P75** | 0.9861 | 0.0012 | — |
| **P90** | 0.9962 | 0.0052 | — |
| **P99** | 0.9995 | 0.1419 | — |
| **Max** | 0.9998 | 0.9963 | — |

### Score Distribution Histogram

| Score Bin | True Pairs Count | True % | False Pairs Count | False % |
| :--- | :---: | :---: | :---: | :---: |
| **0.00 – 0.05** | 239 | 3.67% | **217,471** | **98.13%** |
| **0.05 – 0.10** | 163 | 2.50% | 1,473 | 0.66% |
| **0.10 – 0.20** | 280 | 4.30% | 861 | 0.39% |
| **0.20 – 0.30** | 238 | 3.65% | 451 | 0.20% |
| **0.30 – 0.50** | 452 | 6.94% | 442 | 0.20% |
| **0.50 – 0.70** | 521 | 8.00% | 373 | 0.17% |
| **0.70 – 0.90** | 1,153 | 17.70% | 356 | 0.16% |
| **0.90 – 1.00** | **3,467** | **53.23%** | 194 | 0.09% |

**Key Finding**: 
- **98.13% of all false candidates are cleanly suppressed below 0.05**.
- **Over 70% of true matches score above 0.70**.
- The distribution is bimodal with a narrow overlap zone between 0.30 and 0.70 (representing ~1.5% of total pairs).

---

## 5. Granular Breakdowns (Held-Out Val S1)

### Category Breakdown

| Category / Slice | S1 Count | Ground Truth Matches | S1 Macro F0.5 | Avg Predicted Matches / S1 |
| :--- | :---: | :---: | :---: | :---: |
| **Overall** | **2,001** | **6,920** | **77.59%** | **2.85** |
| **Country: US** | 1,200 | 4,142 | **80.08%** | 2.89 |
| **Country: India** | 801 | 2,778 | **73.87%** | 2.79 |
| **S1: Singleton (0 matches)** | 112 | 0 | **32.14%** | 0.99 |
| **S1: Non-singleton** | 1,889 | 6,920 | **80.29%** | 2.96 |
| **Name Freq = 1 (unique)** | 1,253 | 4,274 | **78.84%** | 2.98 |
| **Name Freq: 2 – 5** | 334 | 1,175 | **76.90%** | 2.90 |
| **Name Freq: 6 – 20** | 128 | 464 | **74.69%** | 2.61 |
| **Name Freq: 21 – 100** | 273 | 955 | **73.73%** | 2.30 |
| **Name Freq: 101+ (generic)** | 13 | 52 | **84.99%** | 3.46 |

### Source-Level Breakdown (S2 vs S3)

| Target Source | Precision | Candidate Recall | F0.5 | False Positives | True Positives / GT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Source 2 (S2)** | 86.16% | 72.00% | 82.90% | 381 | 2,371 / 3,293 |
| **Source 3 (S3)** | 86.93% | 70.80% | 83.14% | 386 | 2,568 / 3,627 |

*Observation*: S2 and S3 behave almost identically in precision (86.2% vs 86.9%) and recall (72.0% vs 70.8%). S3 is neither harder to retrieve nor harder to score.

### Missing Candidate Address Slice

| Metric | Target Candidate Address Missing | Target Candidate Address Present |
| :--- | :---: | :---: |
| Candidate Pairs | 22,760 | 205,374 |
| True Matches in Slice | 200 | 6,313 |
| True Positives (Passed th=0.58) | **15** | 4,924 |
| False Positives | 10 | 757 |
| False Negatives | **185** | 1,389 |
| **Slice Precision** | 60.00% | 86.68% |
| **Slice Recall** | **7.50%** | **77.99%** |
| **Slice F0.5** | **25.00%** | **84.71%** |

*Critical Discovery*: The linear baseline collapses on missing-address candidates. Because address similarity features carry the heaviest weights (`addr_token_set` = +2.74, `addr_jw_sim` = +1.01), records without an address receive 0 on these features, pulling their probability down below 0.58 even when the business name is an exact match.

---

## 6. Failure Mode Analysis

### 1. High-Scoring False Positives
Occur when a candidate shares an almost identical business name and the same street address, but differs by a subtle suite/door number or minor legal extension:
- `S1`: 'modern ithax inc' | Addr: '482 madison avenue albany ny'
- `Cand`: 'modern ithax inc' | Addr: '489 madison avenue albany ny' (Score: 0.9963)
- `S1`: 'all producer private limited' | Addr: '13 1 1146 a 86 a kumar wadi asifnagar hyderabad'
- `Cand`: 'all producer private limited' | Addr: '13 1 1146 a 86 a kumar wadi asifnagar telangana' (Score: 0.9108) — multiple entities registered at identical shared corporate addresses.

### 2. Low-Scoring True Positives (Missed True Matches)
Occur when target records feature heavy spelling corruption or transliteration without clean address match:
- `S1`: 'first industrial partners' | Addr: '7612 hancock street fauquier county va'
- `Cand`: 'zephbrix' | Addr: '7612 hancock st bealeton virginia' (Score: 0.0001) — synthetic/noisy token corruption.
- `S1`: 'future management pvt ltd' | Addr: 's c o no 52 besment huda market sector 07 kurukshetra haryana'
- `Cand`: 'फ य चर म न जम ट प र ल' | Addr: 'हर य ण s c o no 52 kurukshetra' (Score: 0.0145) — Devanagari transliteration.

### 3. Singleton Fragility
For a singleton S1 (ground truth has 0 matches), the metric is binary: 1.0 if empty, 0.0 if even a single candidate is predicted.
- Across ~114 candidates per S1, an FPR of 0.35% produces an expected 0.40 false positives per entity.
- Consequently, 76 out of 112 singletons received at least one false positive, dropping singleton macro F0.5 to **32.14%**.

---

## 7. Conclusions & Strategic Implications

1. **Retrieval Signal is Rich**: The 114 candidates retrieved in RECON-04 contain ample signal: a basic linear model immediately achieves **86.56% pair precision** and pushes **98.13% of false candidates below 0.05**.
2. **Linear Additive Scoring Has Clear Architectural Bottlenecks**:
   - Missing addresses cause recall to plummet from 78.0% to 7.5%.
   - Singletons require a specialized entity-level non-match decision rather than naive independent pair thresholding.
   - Non-linear models (e.g., GBDTs / LightGBM) that learn split interactions (`if addr_missing then name_sim >= 0.95 else ...`) are mathematically necessary to bridge the gap between 71% end-to-end recall and the 94.55% candidate retrieval ceiling.
