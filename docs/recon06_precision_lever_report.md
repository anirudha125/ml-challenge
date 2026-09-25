# RECON-06: Entity-Level Precision Lever Validation & Threshold Audit

**Date**: 2026-09-25  
**Experiment ID**: RECON-06  
**Objective**: Audit the RECON-05 decision threshold selection and empirically measure whether entity-level structural constraints (duplicate-claim resolution, singleton/no-match gates) provide a meaningful precision lever before committing to complex global matching architectures.

---

## 1. Executive Summary

| Measurement / Question | Result | Status / Interpretation |
| :--- | :--- | :--- |
| **STEP 0: Threshold 0.58 Audit** | **100% Clean** | Frozen from Train S1 (1,994 entities); Val S1 labels **never** used |
| **Val S1 Candidates Claimed >1 S1** | **0 (0.00%)** | 0 of 5,706 predicted candidates claimed by multiple S1s |
| **Val S1 Entities Affected by Dups** | **0 (0.00%)** | No S1 collision in the held-out validation sample |
| **Val FPs from Duplicate Claims** | **0 / 767 (0.00%)** | Duplicate claims are **not** an empirical source of false positives |
| **Singleton FP Score Profile** | **High Confidence** | Median max score on singletons is **0.9009** (p25=0.831, max=0.992) |
| **No-Match / Singleton Gate Effect** | **Negative on Macro F0.5** | Cuts true matches on non-singletons faster than it eliminates singleton FPs |
| **Best Post-Processing Model** | **Baseline (th=0.58)** | Post-processing grid tuned on Train S1 yields $\Delta = +0.00\%$ |
| **Success Criteria** | **FAIL** | $< +0.30\%$ improvement threshold (actual: $+0.00\%$) |
| **Strategic Recommendation** | **Abandon global matching; proceed to non-linear GBDT / interaction features** |

---

## 2. Step 0 — Threshold Selection Audit

### Procedure and Code Verification
- **Audit Target File**: `scratch/recon05_baseline_scorer.py` (lines 553–580)
- **Train S1 Fold**: 1,994 entities (227,153 candidate pairs, 6,585 true matches)
- **Validation S1 Fold**: 2,001 entities (228,134 candidate pairs, 6,513 true matches)
- **Tuning Code**:
  ```python
  thresholds = np.arange(0.10, 0.96, 0.02)
  best_th = 0.50
  best_train_f05 = -1.0
  for th in thresholds:
      pred_dict = {}
      for s1_id in train_s1_ids:
          cands = train_preds_by_s1.get(s1_id, [])
          p_set = {cid for cid, p, _ in cands if p >= th}
          pred_dict[s1_id] = p_set
      f05 = evaluate_s1_macro_f05(train_s1_ids, s1_dict, pred_dict)
      if f05 > best_train_f05:
          best_train_f05 = f05
          best_th = th
  ```
- **Proof of Leak-Free Selection**:
  1. `train_s1_ids` alone was passed to `evaluate_s1_macro_f05`.
  2. The validation set `val_s1_ids` and its ground-truth annotations were strictly withheld until line 583, after `best_th = 0.58` was already determined and frozen.
  3. `StandardScaler` was fitted strictly on `X_train` and applied via `.transform(X_val)`.
  4. Conclusion: **Threshold 0.58 is 100% leak-free and mathematically verified.**

---

## 3. Step 1 — Structural Lever Diagnostics (Held-Out Val S1)

At the baseline decision threshold $\theta = 0.58$, we audited all 5,706 predicted candidate pairs across the 2,001 held-out S1 entities:

### 1. Duplicate Claim Counts
- Total unique S2/S3 records predicted: **5,706**
- Records predicted for $>1$ S1: **0 (0.00%)**
  - S2 records with $>1$ claim: **0**
  - S3 records with $>1$ claim: **0**
  - Claim multiplicity distribution: `{1: 5,706}`

*(Context: In the raw, un-thresholded retrieval pool of ~114 candidates/entity, 10,521 candidates appear in $>1$ S1 query result due to generic n-grams. However, 99.8% of these false candidates receive model scores $<0.05$. When the scorer acts at $\theta = 0.58$, exactly zero duplicate claims survive).*

### 2. S1 Entities Affected
- Affected S1 count: **0 / 2,001 (0.00%)**

### 3. True Parent of Duplicate Claims
- a) True parent IS one of claiming S1s: **0** (N/A)
- b) True parent is NEITHER claiming S1: **0** (N/A)

### 4. Validation FPs Attributable to Duplicate Claims
- Total validation false positive pairs: **767**
- FPs from multiply-claimed candidates: **0 (0.00%)**
- FPs from single-claimed candidates: **767 (100.00%)**

### 5. Singleton / No-Match FP Breakdown
The validation fold contains 112 true singletons ($|\text{gt}| = 0$). 76 received at least one false positive candidate scoring $\ge 0.58$ (total 111 FP predictions on singletons).
Categorization of these 111 singleton FPs:
- **a) Duplicate claims** (claimed by $>1$ S1): **0 (0.0%)**
- **b) Distractors** (high name match, completely disjoint address): **5 (4.5%)**
- **c) Shared-address conflicts** (same street/suite address, completely different business name): **19 (17.1%)**
- **d) Other / token-overlap combinations** (generic business name tokens + city/state matches): **87 (78.4%)**

### 6. Distribution of Score Margin
- Score margin $\Delta = s_{(1)} - s_{(2)}$ among multiply-claimed candidates: **N = 0**.
- Score profile of singleton false positives:
  - Minimum passing score: 0.5830
  - 25th percentile: 0.8314
  - **Median: 0.9009**
  - 75th percentile: 0.9424
  - 90th percentile: 0.9781
  - Maximum: 0.9919

*Discovery*: Singleton false positives are not marginal predictions hovering near the 0.58 boundary. They are high-confidence false alarms driven by genuine token similarities (e.g. shared building addresses or partial branch names).

---

## 4. Step 2 — Post-Processing Experiment

Parameters were evaluated and tuned **strictly on the 1,994 Train S1 fold**. The best-performing configuration on Train S1 was frozen and evaluated on the **untouched 2,001 Val S1 fold**.

### Train S1 Tuning Results (N=1,994)

| Configuration | Train Macro F0.5 | Delta vs Baseline | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline ($\theta=0.58$, No Post-Proc)** | **78.23%** | **+0.00%** | **87.00%** | **72.16%** |
| Dup: Top S1 only ($\Delta \ge 0.00$) | 78.23% | +0.00% | 87.00% | 72.16% |
| Dup: Top S1 only ($\Delta \ge 0.05$) | 78.23% | +0.00% | 87.00% | 72.16% |
| Dup: Top S1 only ($\Delta \ge 0.15$) | 78.23% | +0.00% | 87.00% | 72.16% |
| Dup: Drop ambiguous if $\Delta < 0.10$ | 78.23% | +0.00% | 87.00% | 72.16% |
| Gate: Max score $< 0.62 \to \emptyset$ | 78.02% | -0.22% | 87.07% | 72.09% |
| Gate: Max score $< 0.65 \to \emptyset$ | 77.98% | -0.25% | 87.18% | 72.02% |
| Gate: Max score $< 0.70 \to \emptyset$ | 77.99% | -0.24% | 87.26% | 71.87% |
| Gate: Max score $< 0.75 \to \emptyset$ | 77.85% | -0.39% | 87.37% | 71.74% |
| Gate: Solitary $< 0.65 \to \emptyset$ | 77.99% | -0.25% | 87.11% | 72.05% |
| Gate: Solitary $< 0.70 \to \emptyset$ | 78.10% | -0.14% | 87.16% | 72.02% |
| Gate: Solitary $< 0.75 \to \emptyset$ | 78.07% | -0.16% | 87.20% | 71.96% |
| Combo: Top S1 + MaxGate $< 0.62$ | 78.02% | -0.22% | 87.07% | 72.09% |
| Combo: Top S1 + Solitary $< 0.65$ | 77.99% | -0.25% | 87.11% | 72.05% |

*Train S1 Winner*: **Baseline (No Post-Processing)**. Every suppression gate degrades Train S1 macro F0.5.

---

### Untouched Validation S1 Evaluation (N=2,001)

| Configuration | Val Macro F0.5 | Delta | Precision | Recall | Sing F0.5 | US F0.5 | IN F0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($\theta=0.58$, No Post-Proc)** | **77.59%** | **+0.00%** | **86.56%** | **71.37%** | **32.14%** | **80.08%** | **73.87%** |
| Dup: Top S1 only (all variants) | 77.59% | +0.00% | 86.56% | 71.37% | 32.14% | 80.08% | 73.87% |
| Gate: Max score $< 0.62 \to \emptyset$ | 77.57% | -0.03% | 86.57% | 71.34% | 33.04% | 80.09% | 73.79% |
| Gate: Max score $< 0.65 \to \emptyset$ | 77.48% | -0.12% | 86.62% | 71.26% | 34.82% | 80.13% | 73.51% |
| Gate: Max score $< 0.70 \to \emptyset$ | 77.45% | -0.15% | 86.76% | 71.14% | 38.39% | 80.10% | 73.47% |
| Gate: Max score $< 0.75 \to \emptyset$ | 77.05% | -0.54% | 86.91% | 70.90% | 40.18% | 79.88% | 72.82% |
| Gate: Solitary $< 0.65 \to \emptyset$ | 77.52% | -0.07% | 86.62% | 71.29% | 34.82% | 80.13% | 73.62% |
| Gate: Solitary $< 0.70 \to \emptyset$ | 77.49% | -0.11% | 86.69% | 71.23% | 36.61% | 80.10% | 73.58% |
| Gate: Solitary $< 0.75 \to \emptyset$ | 77.26% | -0.33% | 86.74% | 71.11% | 38.39% | 80.05% | 73.09% |

---

## 5. Strategic Conclusions & Next Steps

1. **Why Duplicate Claims Are Non-Existent in the Validation Set**:
   In a randomly sampled validation set of 2,001 S1 entities drawn from a 2.6-million entity corpus (sampling fraction ~0.08%), the probability that two sample queries collide on the exact same true candidate is virtually zero. Duplicate claims cannot serve as a meaningful precision lever here.

2. **Why Heuristic No-Match Gates Fail**:
   Singletons that receive false positives are plagued by high-confidence errors (median score 0.9009). Raising an arbitrary cutoff gate (e.g. $\tau = 0.65$ or $0.70$) suppresses more true matches from difficult non-singletons than it removes false positives from singletons. The net effect is uniformly negative on macro $F_{0.5}$.

3. **Definitive Recommendation**:
   - **Abandon entity-level duplicate resolution and heuristic singleton gating.**
   - **Proceed directly to Non-Linear Gradient Boosted Decision Trees (LightGBM)** with rich feature interactions (e.g., conditional name weighting when address is null, address token set ratios, and character n-gram intersection ratios). This directly attacks the 767 actual false positives and the 1,574 missed true matches without destroying precision.
