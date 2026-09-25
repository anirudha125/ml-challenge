# Exploratory Data Analysis (EDA) Report: Business Entity Resolution Challenge

## 1. Summary Statistics & Data Health

### 1.1 Volume Breakdown
- **Train Set:**
  - `train_source1.tsv`: 2,206,821 records (200.34 MB)
  - `train_source2.tsv`: 5,034,616 records (466.63 MB)
  - `train_source3.tsv`: 5,285,603 records (480.37 MB)
  - `train_ground_truth.tsv`: 2,206,821 rows (121.13 MB)
  - Total Train Records: 12,527,040 rows
- **Test Set:**
  - `test_source1.tsv`: 1,732,544 records (166.91 MB)
  - `test_source2.tsv`: 4,887,273 records (485.86 MB)
  - `test_source3.tsv`: 5,082,316 records (482.56 MB)
  - Total Test Records: 11,702,133 rows
- **Combined Data Footprint:** ~24.2 million records across 7 files (~2.4 GB uncompressed TSV).

### 1.2 Missingness Analysis
| File | Column | Missing / Empty Rows | Percentage | Notes |
|---|---|---|---|---|
| `train_source1` | `business_name` | 0 | 0.00% | Clean, 100% populated |
| `train_source1` | `business_address` | 0 | 0.00% | Clean, 100% populated |
| `train_source1` | `country` | 0 | 0.00% | Clean, 100% populated |
| `train_source2` | `business_address` | 168,967 | 3.36% | High missingness in S2 addresses |
| `train_source3` | `business_address` | 175,916 | 3.33% | High missingness in S3 addresses |
| `test_source1` | All columns | 0 | 0.00% | Clean, 100% populated |
| `test_source2` | `business_address` | 129,408 | 2.65% | High missingness in S2 test addresses |
| `test_source3` | `business_address` | 136,098 | 2.68% | High missingness in S3 test addresses |

**Key Takeaway:** While Source 1 records always have both name and address, ~2.7% - 3.4% of Source 2 and Source 3 records have empty address fields. A model cannot rely solely on address matching; name-only matching capability is necessary.

---

## 2. Ground Truth & Resolution Graph Properties

From analyzing all 2,206,821 ground truth links:
- **Total Matched Pairs $(S_1, S_{2/3})$:** 7,638,365 pairs.
- **Cross-Country Matches:** Exactly **0** (0.000%). Matches are strictly intra-country.
- **Multi-Parent S2/S3 Records:** Exactly **0** (0.000%). Every S2 and S3 entity matches at most 1 S1 entity.
- **Singletons ($S_1$ with 0 matches):** 123,247 entities (5.58%).
- **Non-Singletons ($S_1$ with $\ge 1$ matches):** 2,083,574 entities (94.42%).
- **Match Breakdown:**
  - Matches both $S_2$ and $S_3$: 1,776,047 (80.48%)
  - Matches only $S_2$: 143,029 (6.48%)
  - Matches only $S_3$: 164,498 (7.45%)

### Distribution of Matches per $S_1$ Record:
| Match Count | Number of $S_1$ Entities | Percentage | Cumulative % |
|---|---|---|---|
| 0 | 123,247 | 5.58% | 5.58% |
| 1 | 119,157 | 5.40% | 10.98% |
| 2 | 375,212 | 17.00% | 27.98% |
| 3 | 530,841 | 24.05% | 52.03% |
| 4 | 484,115 | 21.94% | 73.97% |
| 5 | 321,957 | 14.59% | 88.56% |
| 6 | 164,868 | 7.47% | 96.03% |
| 7 | 63,968 | 2.90% | 98.93% |
| 8 | 18,680 | 0.85% | 99.78% |
| 9 | 4,205 | 0.19% | 99.97% |
| 10 | 534 | 0.02% | 99.99% |
| 11 | 37 | 0.00% | 100.00% |

- Median matches per $S_1$: 3
- Mean matches per $S_1$: 3.46
- Maximum matches: 11 (max 5 from S2, max 6 from S3).

### Target Pool Coverage (Train):
- $S_2$ entities participating in matches: 3,693,619 / 5,034,616 (73.36%). Unmatched distractors: 26.64%.
- $S_3$ entities participating in matches: 3,944,746 / 5,285,603 (74.63%). Unmatched distractors: 25.37%.

---

## 3. Country Distributions & Train/Test Drift

| Country | Train $S_1$ | Train $S_2$ | Train $S_3$ | Test $S_1$ | Test $S_2$ | Test $S_3$ |
|---|---|---|---|---|---|---|
| **India** | 40.0% (883k) | 40.1% (2.02M) | 40.0% (2.12M) | 46.8% (810k) | 47.3% (2.31M) | 47.3% (2.41M) |
| **US** | 60.0% (1.32M) | 59.9% (3.02M) | 60.0% (3.17M) | 38.3% (663k) | 38.3% (1.87M) | 38.3% (1.95M) |
| **France** | **0.0% (0)** | **0.0% (0)** | **0.0% (0)** | **15.0% (259k)** | **14.4% (703k)** | **14.4% (732k)** |

**Key Takeaways:**
1. Hard country partitioning: Matches never cross country boundaries (0 observed in 7.6M pairs). Blocking by `country` is 100% safe and cuts the search space drastically.
2. France represents a pure zero-shot/domain-shift challenge. Normalization and tokenizers must not rely on English/Indic-only rules.

---

## 4. Qualitative Noise Patterns & Corruption Typology

Detailed qualitative inspection of real ground truth pairs revealed several distinct noise injection mechanisms:

### 4.1 Business Name Corruptions
1. **Transliteration & Multilingual Text:**
   - In India, business names are frequently transliterated into Indic scripts (Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`, Hindi/Devanagari: `एसएस फूड प्राइवेट लिमिटेड`, `रेड वेंचर्स प्राइवेट लिमिटेड`).
   - Mixed scripts: Some names combine English words with Indic words (e.g., `Raj Investments எல்எல்பி`).
2. **Accented Diacritics & Character Encodings:**
   - Accented characters appear in US and European names: `Payne Énterprises`, `Lumay Bóral`, `Dréxkor`, `Ptit Àmicale`.
3. **Severe Typos & Character Permutations:**
   - Inversions: `Enterpires` vs `Enterprises`, `ENRTPRMISES` vs `Enterprises`, `Etrepndiels` vs `Enterprises`.
   - Phonetic substitutions: `Wanye` vs `Wayne`, `Ponr` vs `Power`.
4. **Legal Entity Suffix Variations & Truncation:**
   - `LLC`, `LLP`, `Inc.`, `Inc`, `Corp`, `Corporation`, `Private Limited`, `Pvt Ltd`, `Center`, `Club`, `SARL`, `SASU`.
5. **Brand / DBA / Domain Name Aliases:**
   - In some records, the name is replaced by a domain name URL (`maurewilliamscolombier.com`) or a brand abbreviation (`Dréxkor`).

### 4.2 Business Address Corruptions
1. **Token Reordering / Inversion:**
   - Example: `Missouri, 630 45th Terrace, Kansas City` vs `630 45th Terrace, Kansas City, MO`.
   - City and state prepended or appended arbitrarily.
2. **Synthetic Garbage Tokens:**
   - The token `"null"` literally appears inserted into addresses (e.g. `KANSAS CITY, MO, 630 45ND TERRACE, null`).
3. **Missing Fields & Landmark Variations:**
   - PIN codes / zip codes omitted, or state abbreviations replaced by state names (`IL` vs `Illinois`, `MO` vs `Missouri`, `TN` vs `Tamil Nadu` vs `தமிழ்நாடு`).
   - Street abbreviations: `St`, `Street`, `Saint`, `Rd`, `Road`, `Ave`, `Avenue`, `Terrace`, `45ND TERRACE`.
4. **Completely Empty Addresses:**
   - ~3% of S2 and S3 records have blank addresses, meaning matching must rely entirely on business name similarity for these records.

---

## 5. Potential Leakage and Overlap Assessment
- **ID Overlap:** Zero overlap between Train and Test IDs ($S_1$, $S_2$, $S_3$). IDs are non-overlapping surrogate keys.
- **Leakage Risk:** If validation splitting is done purely by uniform random split across $S_1$, there is zero target leakage between folds because $S_1$ entities are completely independent deduplicated reference points, and each $S_2/S_3$ entity links to at most one $S_1$.
- **Validation Splitting:** Stratified K-fold by Country and Singleton status ensures that validation splits perfectly reflect the distribution of the challenge.
