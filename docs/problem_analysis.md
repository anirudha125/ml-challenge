# Problem Analysis: Business Entity Resolution Challenge (Amazon ML Challenge 2026)

## 1. Problem Overview and Objective

In enterprise commercial platforms, business identity records originate from disparate, independent data providers. These records have no shared primary keys or persistent identifiers. The challenge is **Entity Resolution (ER)**: resolving multi-source noisy, fragmented records to a canonical deduplicated reference entity.

Specifically:
- **Source 1 (`S1`)**: Deduplicated reference entity source.
- **Source 2 (`S2`)**: Independent noisy source with partial/corrupted records and distractors.
- **Source 3 (`S3`)**: Independent noisy source with partial/corrupted records and distractors.
- **Goal**: For every Source 1 entity, identify all corresponding matching entity IDs from Source 2 and Source 3.

---

## 2. Mathematical Formulation and Target Variable

### 2.1 Target Variable
For each $e \in S_1$, predict a set of matching entity IDs $M(e) \subseteq (S_2 \cup S_3)$.
- An entity $e$ can match zero records (singleton: $M(e) = \emptyset$).
- An entity $e$ can match one or multiple records across $S_2$ and $S_3$.
- In the training set:
  - Total $S_1$ entities: 2,206,821.
  - Singletons ($|M(e)| = 0$): 123,247 (5.58%).
  - Non-singletons ($|M(e)| \ge 1$): 2,083,574 (94.42%).
  - Matches across both $S_2$ and $S_3$: 80.48%.
  - Max matches for a single $S_1$: 11.
  - Unique $S_2/S_3$ entities matching multiple $S_1$ entities: **0** (strictly at most 1 $S_1$ entity per $S_2/S_3$ record).

### 2.2 Evaluation Metric: Macro $F_{0.5}$
The evaluation metric is the macro-averaged $F_{\beta}$ score with $\beta = 0.5$ computed across all $S_1$ records in the evaluation set:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

#### Boundary and Edge Case Rules:
1. **True Singletons ($|Y| = 0$):**
   - If $|\hat{Y}| = 0$: $F_{0.5} = 1.0$ (correctly predicting no match earns full credit).
   - If $|\hat{Y}| > 0$: $F_{0.5} = 0.0$ (false merge on a singleton is heavily penalized).
2. **True Non-Singletons ($|Y| > 0$):**
   - If $|\hat{Y}| = 0$: $F_{0.5} = 0.0$.
   - If $|Y \cap \hat{Y}| = 0$: $F_{0.5} = 0.0$.
3. **Macro Average:**
   - Evaluated individually per $S_1$ record, then arithmetic mean taken over all $N$ entities in the test set:
   
$$\text{Score} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5}(Y_i, \hat{Y}_i)$$

#### Metric Sensitivity:
Because $\beta = 0.5$, precision is weighted $2\times$ as heavily as recall:
- Precision = 1.0, Recall = 0.5 $\implies F_{0.5} = 0.8333$.
- Precision = 0.5, Recall = 1.0 $\implies F_{0.5} = 0.5556$.
This strongly dictates that false positives (incorrectly merging distinct businesses) severely harm the score compared to borderline omissions.

---

## 3. Train vs. Test Structure

| Split | File Name | Size (MB) | Row Count | Countries | Purpose |
|---|---|---|---|---|---|
| **Train** | `train_source1.tsv` | 200.3 MB | 2,206,821 | US (60.0%), India (40.0%) | Deduplicated reference |
| **Train** | `train_source2.tsv` | 466.6 MB | 5,034,616 | US (59.9%), India (40.1%) | Target records for resolution |
| **Train** | `train_source3.tsv` | 480.4 MB | 5,285,603 | US (60.0%), India (40.0%) | Target records for resolution |
| **Train** | `train_ground_truth.tsv` | 121.1 MB | 2,206,821 | — | Mapping $S_1 \to \{S_2, S_3\}$ |
| **Test** | `test_source1.tsv` | 166.9 MB | 1,732,544 | India (46.8%), US (38.3%), France (15.0%) | Query entities to resolve |
| **Test** | `test_source2.tsv` | 485.9 MB | 4,887,273 | India (47.3%), US (38.3%), France (14.4%) | Candidates for resolution |
| **Test** | `test_source3.tsv` | 482.6 MB | 5,082,316 | India (47.3%), US (38.3%), France (14.4%) | Candidates for resolution |

### Critical Out-of-Distribution (OOD) Observation:
- **France** is completely **absent from the training set** (0% in Train), but represents **15.0% of the test set** (259,452 Source 1 records; ~1.43M records in test S2/S3).
- Any hardcoded country logic or country-specific dictionary lookups will fail on France.
- Feature extractors and text normalization must generalize across language scripts and European address formats (e.g., French street terms: `Rue`, `Boulevard`, `Avenue`, `bis`, `CEDEX`, and legal forms: `SARL`, `SASU`, `SCI`).

---

## 4. Submission & Output Format Requirements

Two tab-separated files (`.tsv`) are required:

1. **`matching_results.tsv`** (Leaderboard scored):
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Must contain every $S_1$ entity from `test_source1.tsv` (1,732,544 rows).
   - Singletons have empty `matched_entity_ids`.
   - Multiple matched IDs are comma-separated without spaces: `S2-XXXXX,S3-YYYYY`.
   - Strict format: No self-matches (`S1-`), no invalid IDs, no intra-list duplicates.

2. **`candidate_pairs.tsv`** (Audited in final package):
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Exact candidate pool output from the blocking stage fed directly into the matching classifier.
   - All IDs in `matching_results.tsv` should be a subset of `candidate_pairs.tsv`.

### Final Submission Archive Structure:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 5. Strict Competition Rules & Prohibitions

1. **External Data Lookup Strictly Prohibited:**
   - No commercial ER APIs (e.g., Google Places, Dun & Bradstreet, OpenCorporates).
   - No government registry scraping (MCA, SEC, INSEE).
   - No geocoding APIs (e.g., Google Maps, Nominatim).
   - Violation causes **immediate disqualification**.
2. **Model Licensing & Size Constraints:**
   - Any pretrained model used must have an **MIT or Apache 2.0 license**.
   - Maximum model parameter count: **up to 8 Billion parameters**.
3. **Reproducibility:**
   - Runnable code package must reproduce both TSV files from scratch.
   - Pinned dependencies in `requirements.txt`.
