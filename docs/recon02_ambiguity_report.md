# RECON-02: Name / Address Ambiguity Investigation Report

**Date:** 2026-09-25  
**Runtime:** 1,266.8s (~21 minutes) | Streaming over ~12.5M train records, no simultaneous full-dataset load  
**Normalization applied:** NFC + lowercase + punct→space + hyphen→space + `null` token removal + whitespace collapse

---

## 1. EXECUTIVE SUMMARY

The data is highly noisy. **~78% of all true matched pairs have NO exact normalized-name match** between the S1 record and its S2/S3 counterpart. Exact name+address covers only 2.78% of S2 true pairs and near-zero (0.01%) of S3 true pairs. Adding `country` to the exact name key adds negligible disambiguation (only ~0.03% improvement in S1 uniqueness). S1's name+address combination is 100% unique — it is the deduplicated reference — but S2 and S3 have only ~2% exact name+address duplicate rate, suggesting most S2/S3 records do represent distinct observations (with heavy noise, not outright copies).

**Core conclusion:** Any blocking strategy relying on exact string matching will miss ~78–99%+ of true pairs. Token-level soft matching (character n-grams, token Jaccard, TF-IDF cosine, etc.) is **mandatory** to achieve viable recall.

---

## 2. NAME AMBIGUITY (Normalized Name Only)

| Source | Total Rows | Unique Norm Names | % Rows w/ Unique Name | % Rows w/ Name Freq≥2 | % Rows w/ Name Freq≥10 | Max Freq |
|---|---|---|---|---|---|---|
| S1 | 2,206,821 | 1,520,684 | **60.72%** | 39.28% | 16.98% | 253 |
| S2 | 5,034,616 | 4,025,056 | **72.12%** | 27.88% | 7.43% | 526 |
| S3 | 5,285,603 | 4,280,399 | **73.41%** | 26.59% | 7.01% | 521 |

**Key findings:**
- **39% of S1 rows share their normalized name with at least one other S1 record.** Because S1 is the deduplicated reference, these are genuinely different businesses with the same (or same-normalizing) name.
- Max S1 name frequency = 253 (`"primary care group"`) — mostly medical specialty generics (see Part H).
- S2/S3 have lower duplication (27–28%) because they contain more unique noisy renderings of the same underlying set of businesses.
- The max 526/521 frequencies in S2/S3 result from many noisy variants of extremely common names all normalizing to the same string.

### Name Frequency Bucket Distribution (% of rows):

| Freq Bucket | S1 | S2 | S3 |
|---|---|---|---|
| 1 (unique) | 60.72% | 72.12% | 73.41% |
| 2 | 9.14% | 9.90% | 9.74% |
| 3–5 | 8.76% | 7.46% | 6.90% |
| 6–10 | 4.98% | 3.50% | 3.39% |
| 11–50 | 11.77% | 5.67% | 4.99% |
| 51–100 | 3.99% | 0.79% | 0.88% |
| 101–1000 | 0.64% | 0.57% | 0.69% |
| 1000+ | 0.00% | 0.00% | 0.00% |

---

## 3. NAME + COUNTRY AMBIGUITY

| Source | Total | Unique (name, country) | % Rows w/ Unique Key | Δ vs Name-only |
|---|---|---|---|---|
| S1 | 2,206,821 | 1,521,424 | **60.75%** | +0.03% |
| S2 | 5,034,616 | 4,030,166 | **72.20%** | +0.08% |
| S3 | 5,285,603 | 4,287,783 | **73.54%** | +0.13% |

**Key finding:** Country provides **near-zero disambiguation benefit** when used as part of an exact-match key. With only 2 training countries (US and India), almost all name duplicates already occur within a single country — so `(name, country)` ≈ `name` in terms of uniqueness.

However, `country` remains valuable as a **hard partition for blocking** (eliminating cross-country candidate pairs), not for row disambiguation.

---

## 4. NAME + ADDRESS AMBIGUITY

| Source | Rows w/ Address | % With Addr | Unique (name, addr) | % Rows Unique | Max Freq |
|---|---|---|---|---|---|
| S1 | 2,206,821 | **100.0%** | 2,206,821 | **100.00%** | 1 |
| S2 | 4,865,649 | 96.64% | 4,800,009 | **97.35%** | 5 |
| S3 | 5,109,687 | 96.67% | 5,061,073 | **98.13%** | 4 |

**Key findings:**
- **S1 name+address is 100% unique** — confirms S1 is a perfectly deduplicated reference source.
- S2 and S3 name+address is ~97–98% unique — the small ~2–3% of duplicated (name, addr) pairs are likely near-identical clones injected during noise simulation.
- Max frequency in S2/S3 is only 4–5 — there are no "name+address spam" clusters.
- Address missingness (3.36% S2, 3.33% S3) means ~168–176K records must rely on name-only matching.

---

## 5. EXACT MATCH COVERAGE OF TRUE PAIRS (Ground-Truth Analysis)

**This is the most critical section.** Using all 7,638,365 ground-truth matched pairs.

| Match Pairing | Total Pairs | Exact Norm Name | Name+Country | Name+Addr | No Exact Name |
|---|---|---|---|---|---|
| **All S1→S2** | 3,693,619 | **21.44%** | 21.44% | **2.78%** | **78.56%** |
| **All S1→S3** | 3,944,746 | **22.23%** | 22.23% | **0.01%** | **77.77%** |
| US: S1→S2 | 2,213,074 | 25.81% | 25.81% | 3.53% | 74.19% |
| US: S1→S3 | 2,365,448 | 25.85% | 25.85% | 0.00% | 74.15% |
| India: S1→S2 | 1,480,545 | **14.91%** | 14.91% | 1.66% | **85.09%** |
| India: S1→S3 | 1,579,298 | **16.81%** | 16.81% | 0.01% | **83.19%** |

### Critical Observations:

1. **~78% of true pairs have no exact normalized name match.** The entire 78% must be recalled through fuzzy/token-level blocking. This is the single most impactful finding.

2. **Name+Country = Name exactly** in exact-match recall (0% additional coverage). Since all matches are intra-country by design, exact-name filtering already implicit includes country identity.

3. **Exact name+address covers only 2.78% of S2 pairs and 0.01% of S3 pairs.** S3 addresses are dramatically noisier than S2 addresses in terms of post-normalization exact match. This means address exact-matching is far more useful for S2 than S3 as a high-precision seed.

4. **India harder than US:** India exact-name coverage is 14–17% vs US 25–26%. Indian business names have more transliteration variants (Indic scripts ↔ English) that survive normalization, resulting in lower exact-match rates.

---

## 6. S2 vs S3 COMPARISON

| Dimension | S2 | S3 | Verdict |
|---|---|---|---|
| Total rows | 5,034,616 | 5,285,603 | S3 larger |
| Missing addresses | 3.36% (168,967) | 3.33% (175,916) | Comparable |
| Unique name (% rows) | 72.12% | 73.41% | S3 slightly less duplicated |
| Unique name+country (% rows) | 72.20% | 73.54% | Same pattern |
| Unique name+addr (% rows with addr) | **97.35%** | **98.13%** | S3 slightly cleaner |
| Exact name coverage of true pairs | 21.44% | 22.23% | Nearly identical |
| Exact name+addr coverage of true pairs | **2.78%** | **0.01%** | S3 addresses far noisier |
| Max name frequency | 526 | 521 | Nearly identical |

**Conclusion:** S2 and S3 behave nearly identically for name-based blocking. The major difference is in address matching: S3 addresses are much noisier at the normalized exact-match level (nearly 0% of true S1→S3 pairs share an exact normalized address). A single universal blocking strategy is appropriate for both sources, but any address-based feature engineering must be fuzzy, not exact.

---

## 7. INDIA vs US ANALYSIS

| Metric | US S1 | India S1 | US S2 | India S2 | US S3 | India S3 |
|---|---|---|---|---|---|---|
| Rows | 1,323,633 | 883,188 | 3,016,817 | 2,017,799 | 3,170,056 | 2,115,547 |
| % Rows w/ Unique Name | 64.2% | **55.6%** | 73.4% | 70.4% | 73.8% | 73.2% |
| % Rows Dup by Name | 35.8% | **44.4%** | 26.6% | 29.6% | 26.2% | 26.8% |
| Max Name Freq | 253 | 99 | 526 | 128 | 521 | 183 |
| % Rows Unique Name+Addr | 100% | 100% | 97.6% | 97.0% | 97.8% | 98.6% |
| True pair exact-name recall | 25.8% | **14.9%** (S2) / **16.8%** (S3) | — | — | — | — |

**Key findings:**
- Indian S1 names are significantly more ambiguous (44.4% dup rows vs 35.8% for US), with lower max-frequency (max 99 vs 253 for US) — India has broadly higher background ambiguity but without extreme outliers.
- US high-frequency names are dominated by healthcare/medical generic terms (`"physical therapy"`, `"primary care group"`, `"orthopedic group"`) which likely represent hundreds of genuinely different practices with identical names.
- For India, the transliteration noise (English ↔ Indic scripts) is the dominant challenge — causing lower exact-name recall (14–17%) than US (25–26%).
- Name+address uniqueness is 100% for both countries in S1 (by definition). S2/S3 address duplicates are similarly low (~2–3%) for both countries.
- **Same blocking strategy applies to both countries**, but India will require more aggressive fuzzy tolerance to achieve comparable recall.

---

## 8. REPRESENTATIVE EXAMPLES

### Top 10 Most Ambiguous S1 Names (Highest Name Frequency)
All are generic US healthcare/medical terms:

| Freq | Normalized Name |
|---|---|
| 253 | `primary care group` |
| 251 | `ear nose throat group` |
| 222 | `pediatric group` |
| 220 | `womens health group` |
| 218 | `physical therapy group` |
| 216 | `pediatric dental group` |
| 215 | `behavioral health group` |
| 209 | `chiropractic group` |
| 208 | `orthopedic group` |
| 207 | `eye group` |

These are real distinct businesses that share identical normalized names — they can only be disambiguated by address.

### Names That Become Unique After Adding Country (Country Resolves Ambiguity)
Sample (freq=2 in name-only, freq=1 per country):
- `colline molitor sea llc` → unique per country
- `trident welfare society` → unique per country
- `pacific learning laboratories llc` → unique per country

### Names Still Ambiguous at Name+Country Level
| Freq | Country | Name |
|---|---|---|
| 15 | US | `primary care group` |
| 13 | US | `physical therapy group` |
| 12 | US | `internal medicine group` |
| 12 | US | `primary care specialists llc` |
| 11 | US | `cardiology center llc` |

### Sample Distinctive (Highly Unique) S1 Names
- `orelee s barbershop` | `1795 westchester drive high point nc`
- `prabhav business center` | `797 lake town block a kolkata howrah west bengal`
- `moore bitwise inc` | `337 oakland avenue michigan city in`

---

## 9. IMPLICATIONS FOR BLOCKING

| Observation | Implication |
|---|---|
| 78% of true pairs have no exact name match | Exact-name blocking alone achieves ≤22% recall ceiling — **unacceptable** |
| Country adds no additional exact-match recall | Use country only as a **hard partition key** (eliminates ~50–60% irrelevant cross-country candidates with 0% recall loss) |
| S1 name+addr is 100% unique | A (S1_name, S1_addr) key can serve as a high-confidence identity fingerprint |
| S2/S3 name+addr unique at 97–98% | Exact name+addr hits are near-certain true matches → use as a high-precision seed |
| S3 exact-name+addr coverage = 0.01% | S3 addresses are too noisy for exact-address matching — need token overlap |
| India true-pair exact-name recall = 14–17% | India needs more fuzzy blocking layers (especially for transliteration) |
| Generic names dominate ambiguity (medical specialty) | Address becomes the primary discriminator for top-ambiguous names |
| Max name freq = 253–526; max (name+addr) freq = 1–5 | No catastrophic blocking explosions from sparse keys |

### Blocking Architecture Requirements:
1. **Layer 0 (Hard Partition):** Partition by `country` — 0% recall loss, ~50–60% candidate space reduction.
2. **Layer 1 (Token-Level Name Blocking):** Character n-gram inverted index or token-set Jaccard on normalized name — required to capture the 78% of true pairs with no exact name match.
3. **Layer 2 (Address Token Blocking):** Normalized address token overlap — critical for high-frequency name disambiguation (the 253-frequency medical names).
4. **Layer 3 (High-Precision Seed):** Exact (norm_name, norm_addr) match → near-certain true positives.
5. **Layer 4 (Recall Floor):** For name-only records (missing address), rely entirely on fuzzy name matching.

---

## 10. DIRECT ANSWERS

**A. Is normalized-name blocking strong enough to form the backbone of candidate generation?**  
**No.** Exact normalized-name blocking achieves only ~22% recall on true pairs. It cannot be the backbone. **Token-level or fuzzy name blocking** (character n-grams, token overlap) is mandatory as the primary recall driver.

**B. Is name+country substantially safer than name alone?**  
**No** — for exact matching. Country adds only +0.03% uniqueness in S1 when used as part of an exact key (because all training matches are already intra-country). However, `country` is invaluable as a **hard partition key** to eliminate candidates — just not as a discriminative signal within a country.

**C. Is exact name+address strong enough to be a high-precision seed?**  
**Yes — with caveats.** S1 name+address is 100% unique. Exact matches between S1 and S2/S3 on (name, address) are extremely high-precision. However, this covers only **2.78% of S2 pairs and 0.01% of S3 pairs** — so it functions only as a high-confidence seed, not a recall-driving component.

**D. Roughly what percentage of true matches would be missed by exact normalized-name blocking?**  
**~78%** (78.56% for S2, 77.77% for S3). Of 7.6M true pairs, ~6M would be missed.

**E. Roughly what percentage would be missed by exact normalized-name+address blocking?**  
**~97–100%** (97.22% S2, 99.99% S3). Exact name+address blocking is nearly useless for recall and should only be used to identify certain true positives, not drive candidate generation.

**F. What is the single most important thing we learned?**  
The dominant challenge in this dataset is not entity disambiguation — it is **noise-robust blocking**. With 78%+ of true pairs having no exact name match, the entire problem shifts to designing a fuzzy candidate retrieval system that can surface the right S2/S3 record even when the business name has been transliterated, heavily misspelled, abbreviated, or replaced with a brand alias or domain. Soft token-level name matching is the core of the solution.
