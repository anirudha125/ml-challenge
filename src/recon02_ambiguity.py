"""
RECON-02: Name / Address Ambiguity Investigation
All parts: A (normalization), B (name freq), C (name+country), D (name+addr),
           E (GT coverage), F (S2 vs S3), G (India vs US), H (examples)

Design principles:
- Stream/chunk all large files; never load full dataset into memory at once.
- Use dict-of-Counter for frequency tracking, flushed per pass.
- Multiple passes over source files to avoid loading S1+S2+S3 simultaneously.
- Disk-backed temp structures only if needed.
- All normalization is transparent and documented.
"""

import os, sys, re, unicodedata, collections, time

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r"d:\amazon-ml-challenge-2026\student_resource\dataset"
TRAIN = os.path.join(DATA_DIR, "train")

# ============================================================
# PART A — NORMALIZATION (transparent, no external dicts)
# ============================================================

def normalize_name(text: str) -> str:
    """
    Normalization steps (in order):
    1. NFC Unicode normalization (compose canonical decompositions)
    2. Lowercase
    3. Remove all punctuation except alphanumeric and Unicode letters/digits
       (preserves Unicode scripts: Tamil, Hindi, Kannada, French accents etc.)
    4. Collapse multiple whitespace into single space
    5. Strip leading/trailing whitespace
    """
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    # Remove characters that are not alphanumeric (Unicode-aware), not space, not hyphen
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"[-]+", " ", t)        # replace hyphens with spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_address(text: str) -> str:
    """
    Additional address normalization:
    1. NFC, lowercase
    2. Remove punctuation (same as name) 
    3. Remove standalone "null" token (known synthetic noise from S2/S3)
    4. Collapse multiple whitespace
    5. Strip
    """
    if not text.strip():
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"[-]+", " ", t)
    # Remove standalone 'null' token (synthetic noise)
    t = re.sub(r"\bnull\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# HELPERS
# ============================================================

def stream_rows(path):
    """Yield (entity_id, norm_name, norm_addr, country) tuples."""
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            yield eid, normalize_name(name), normalize_address(addr), country


def count_frequencies(path, key_fn):
    """
    Stream path and count occurrences of key_fn(norm_name, norm_addr, country).
    Returns Counter.
    """
    ctr = collections.Counter()
    for eid, nn, na, country in stream_rows(path):
        ctr[key_fn(nn, na, country)] += 1
    return ctr


def freq_buckets(ctr):
    """Given a Counter, return bucket summary dict."""
    buckets = collections.OrderedDict([
        ("1", 0), ("2", 0), ("3-5", 0), ("6-10", 0),
        ("11-50", 0), ("51-100", 0), ("101-1000", 0), ("1000+", 0)
    ])
    total_rows = sum(ctr.values())
    for key, cnt in ctr.items():
        if cnt == 1:       buckets["1"] += cnt
        elif cnt == 2:     buckets["2"] += cnt
        elif cnt <= 5:     buckets["3-5"] += cnt
        elif cnt <= 10:    buckets["6-10"] += cnt
        elif cnt <= 50:    buckets["11-50"] += cnt
        elif cnt <= 100:   buckets["51-100"] += cnt
        elif cnt <= 1000:  buckets["101-1000"] += cnt
        else:              buckets["1000+"] += cnt
    return buckets, total_rows


def print_freq_table(ctr, label, total_rows=None):
    buckets, tr = freq_buckets(ctr)
    if total_rows is None:
        total_rows = tr
    unique_keys = len(ctr)
    print(f"  Unique {label}: {unique_keys:,}")
    print(f"  Rows analyzed: {total_rows:,}")
    print(f"  Uniqueness rate (key freq=1): {sum(1 for v in ctr.values() if v==1)/unique_keys*100:.2f}% of keys")
    print(f"  Max frequency: {ctr.most_common(1)[0][1]:,} (key: '{ctr.most_common(1)[0][0][:60]}')")
    pct_once = buckets['1'] / total_rows * 100
    pct_dup = (total_rows - buckets['1']) / total_rows * 100
    pct_10plus = sum(v for k, v in ctr.items() if v >= 10) / total_rows * 100
    pct_100plus = sum(v for k, v in ctr.items() if v >= 100) / total_rows * 100
    print(f"  Rows w/ freq=1 (unique key):   {buckets['1']:>10,}  ({pct_once:.2f}%)")
    print(f"  Rows w/ freq>=2 (ambiguous):   {total_rows - buckets['1']:>10,}  ({pct_dup:.2f}%)")
    print(f"  Rows w/ freq>=10:              {sum(v for k,v in ctr.items() if v>=10):>10,}  ({pct_10plus:.2f}%)")
    print(f"  Rows w/ freq>=100:             {sum(v for k,v in ctr.items() if v>=100):>10,}  ({pct_100plus:.2f}%)")
    print(f"  {'Bucket':>12} | {'Row count':>12} | {'% of rows':>10}")
    print(f"  {'-'*42}")
    for bname, bcount in buckets.items():
        print(f"  {bname:>12} | {bcount:>12,} | {bcount/total_rows*100:>9.2f}%")
    print()


# ============================================================
# PART B — NAME DUPLICATE / AMBIGUITY ANALYSIS
# ============================================================

def part_b():
    print("\n" + "="*60)
    print("PART B — NORMALIZED NAME FREQUENCY")
    print("="*60)
    files = {
        "S1 (train)": os.path.join(TRAIN, "train_source1.tsv"),
        "S2 (train)": os.path.join(TRAIN, "train_source2.tsv"),
        "S3 (train)": os.path.join(TRAIN, "train_source3.tsv"),
    }
    results = {}
    for label, path in files.items():
        t0 = time.time()
        print(f"\n--- {label} ---")
        ctr = count_frequencies(path, lambda nn, na, c: nn)
        print_freq_table(ctr, "norm_name", sum(ctr.values()))
        results[label] = ctr
        print(f"  (elapsed: {time.time()-t0:.1f}s)")
    return results


# ============================================================
# PART C — NAME + COUNTRY AMBIGUITY
# ============================================================

def part_c():
    print("\n" + "="*60)
    print("PART C — NORMALIZED NAME + COUNTRY FREQUENCY")
    print("="*60)
    files = {
        "S1 (train)": os.path.join(TRAIN, "train_source1.tsv"),
        "S2 (train)": os.path.join(TRAIN, "train_source2.tsv"),
        "S3 (train)": os.path.join(TRAIN, "train_source3.tsv"),
    }
    results = {}
    for label, path in files.items():
        t0 = time.time()
        print(f"\n--- {label} ---")
        ctr = count_frequencies(path, lambda nn, na, c: (nn, c))
        print_freq_table(ctr, "norm_name+country", sum(ctr.values()))
        results[label] = ctr
        print(f"  (elapsed: {time.time()-t0:.1f}s)")
    return results


# ============================================================
# PART D — NAME + ADDRESS AMBIGUITY
# ============================================================

def part_d():
    print("\n" + "="*60)
    print("PART D — NORMALIZED NAME + ADDRESS FREQUENCY")
    print("="*60)
    files = {
        "S1 (train)": os.path.join(TRAIN, "train_source1.tsv"),
        "S2 (train)": os.path.join(TRAIN, "train_source2.tsv"),
        "S3 (train)": os.path.join(TRAIN, "train_source3.tsv"),
    }
    results = {}
    for label, path in files.items():
        t0 = time.time()
        print(f"\n--- {label} ---")
        ctr_with_addr = collections.Counter()
        ctr_no_addr = collections.Counter()
        total_with = 0
        total_no = 0
        for eid, nn, na, country in stream_rows(path):
            if na:
                ctr_with_addr[(nn, na)] += 1
                total_with += 1
            else:
                ctr_no_addr[nn] += 1
                total_no += 1
        total = total_with + total_no
        print(f"  Total rows: {total:,} | With address: {total_with:,} ({total_with/total*100:.2f}%) | No address: {total_no:,} ({total_no/total*100:.2f}%)")
        print(f"\n  [Name+Address - for rows with non-empty address]")
        print_freq_table(ctr_with_addr, "norm_name+norm_addr (non-empty)", total_with)
        results[label] = ctr_with_addr
        print(f"  (elapsed: {time.time()-t0:.1f}s)")
    return results


# ============================================================
# PART E — GROUND-TRUTH-AWARE COVERAGE ANALYSIS
# ============================================================

def part_e():
    print("\n" + "="*60)
    print("PART E — EXACT MATCH COVERAGE OF TRUE PAIRS")
    print("="*60)
    
    # Load S1 data
    print("  Loading S1 norm data...")
    s1 = {}  # id -> (norm_name, norm_addr, country)
    for eid, nn, na, country in stream_rows(os.path.join(TRAIN, "train_source1.tsv")):
        s1[eid] = (nn, na, country)
    print(f"  Loaded {len(s1):,} S1 records.")

    # Load S2 norm data
    print("  Loading S2 norm data...")
    s2 = {}
    for eid, nn, na, country in stream_rows(os.path.join(TRAIN, "train_source2.tsv")):
        s2[eid] = (nn, na, country)
    print(f"  Loaded {len(s2):,} S2 records.")

    # Load S3 norm data
    print("  Loading S3 norm data...")
    s3 = {}
    for eid, nn, na, country in stream_rows(os.path.join(TRAIN, "train_source3.tsv")):
        s3[eid] = (nn, na, country)
    print(f"  Loaded {len(s3):,} S3 records.")

    # Analyze ground truth
    print("  Analyzing ground truth...")
    stats = {
        "S2": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
        "S3": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
        "US_S2": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
        "US_S3": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
        "India_S2": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
        "India_S3": {"name": 0, "name_country": 0, "name_addr": 0, "none": 0, "total": 0},
    }

    gt_path = os.path.join(TRAIN, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if not matched_str.strip():
                continue

            s1_nn, s1_na, s1_c = s1[s1_id]
            m_ids = matched_str.split(",")
            for mid in m_ids:
                if mid.startswith("S2-"):
                    src_dict = s2
                    src_key = "S2"
                elif mid.startswith("S3-"):
                    src_dict = s3
                    src_key = "S3"
                else:
                    continue

                country_key = s1_c  # India or US
                
                if mid not in src_dict:
                    continue
                m_nn, m_na, m_c = src_dict[mid]

                # Exact normalized name match
                exact_name = (s1_nn == m_nn)
                # Exact normalized name + country
                exact_name_c = (s1_nn == m_nn and s1_c == m_c)
                # Exact normalized name + address (only if both have address)
                exact_name_addr = (s1_nn == m_nn and s1_na != "" and m_na != "" and s1_na == m_na)
                # Neither
                no_exact = not exact_name

                stats[src_key]["total"] += 1
                if exact_name: stats[src_key]["name"] += 1
                if exact_name_c: stats[src_key]["name_country"] += 1
                if exact_name_addr: stats[src_key]["name_addr"] += 1
                if no_exact: stats[src_key]["none"] += 1

                ck = f"{country_key}_{src_key}"
                if ck in stats:
                    stats[ck]["total"] += 1
                    if exact_name: stats[ck]["name"] += 1
                    if exact_name_c: stats[ck]["name_country"] += 1
                    if exact_name_addr: stats[ck]["name_addr"] += 1
                    if no_exact: stats[ck]["none"] += 1

    print(f"\n  {'Key':20} | {'Total':>10} | {'Exact Name%':>12} | {'Name+Ctr%':>11} | {'Name+Addr%':>11} | {'No Exact%':>10}")
    print(f"  {'-'*85}")
    for k, v in stats.items():
        if v["total"] == 0:
            continue
        t = v["total"]
        print(f"  {k:20} | {t:>10,} | {v['name']/t*100:>11.2f}% | {v['name_country']/t*100:>10.2f}% | {v['name_addr']/t*100:>10.2f}% | {v['none']/t*100:>9.2f}%")
    
    # Free memory
    del s2, s3
    return stats, s1


# ============================================================
# PART G — INDIA vs US
# ============================================================

def part_g():
    print("\n" + "="*60)
    print("PART G — INDIA vs US NAME AMBIGUITY")
    print("="*60)

    for src_label, src_path in [
        ("S1", os.path.join(TRAIN, "train_source1.tsv")),
        ("S2", os.path.join(TRAIN, "train_source2.tsv")),
        ("S3", os.path.join(TRAIN, "train_source3.tsv")),
    ]:
        ctrs = {"US": {"name": collections.Counter(), "name_c": collections.Counter(), "name_a": collections.Counter()},
                "India": {"name": collections.Counter(), "name_c": collections.Counter(), "name_a": collections.Counter()}}
        for eid, nn, na, country in stream_rows(src_path):
            if country not in ("US", "India"):
                continue
            ctrs[country]["name"][nn] += 1
            ctrs[country]["name_c"][(nn, country)] += 1
            if na:
                ctrs[country]["name_a"][(nn, na)] += 1

        print(f"\n--- {src_label} ---")
        for cname in ("US", "India"):
            c = ctrs[cname]
            for key, label in [("name", "norm_name"), ("name_c", "norm_name+country"), ("name_a", "norm_name+addr")]:
                ctr = c[key]
                if not ctr:
                    continue
                total = sum(ctr.values())
                unique = len(ctr)
                uniq_rate = sum(1 for v in ctr.values() if v == 1) / unique * 100
                dup_rows = sum(v for v in ctr.values() if v >= 2)
                max_freq = ctr.most_common(1)[0][1]
                print(f"  [{cname}] {label}: total={total:,} | unique_keys={unique:,} | uniqueness%={uniq_rate:.1f}% | dup_rows={dup_rows:,} ({dup_rows/total*100:.1f}%) | max_freq={max_freq:,}")


# ============================================================
# PART H — REPRESENTATIVE EXAMPLES
# ============================================================

def part_h():
    print("\n" + "="*60)
    print("PART H — REPRESENTATIVE AMBIGUITY EXAMPLES")
    print("="*60)

    # Load S1 data to look for examples
    name_ctr = collections.Counter()
    rows_sample = []
    for eid, nn, na, country in stream_rows(os.path.join(TRAIN, "train_source1.tsv")):
        name_ctr[nn] += 1
        if len(rows_sample) < 100000:
            rows_sample.append((eid, nn, na, country))

    print("\n[1] Top 10 most common normalized S1 names (highest ambiguity):")
    for name, cnt in name_ctr.most_common(10):
        print(f"    freq={cnt:5,} | '{name[:60]}'")

    print("\n[2] Names that become unique after adding country (sample 5 from freq=2 in name, freq=1 in name+country):")
    name_country_ctr = collections.Counter()
    for eid, nn, na, country in rows_sample:
        name_country_ctr[(nn, country)] += 1
    shown = 0
    for nn, cnt in name_ctr.most_common():
        if cnt == 2 and shown < 5:
            # Check if both country entries are different
            found = False
            for (n2, c2), cnt2 in name_country_ctr.items():
                if n2 == nn and cnt2 == 1:
                    if not found:
                        print(f"    name='{nn[:50]}' | overall_freq=2 | (name,country) unique")
                        found = True
                        shown += 1
                        break

    print("\n[3] Top 5 names still ambiguous at name+country level (freq>=5 in any country):")
    country_name_ctr = collections.Counter()
    for (nn, c), cnt in name_country_ctr.items():
        country_name_ctr[(nn, c)] = cnt
    for (nn, c), cnt in country_name_ctr.most_common(10):
        if cnt >= 5:
            print(f"    freq={cnt:5,} | country={c} | '{nn[:60]}'")

    print("\n[4] Sample of distinctive (unique) S1 names:")
    shown2 = 0
    for eid, nn, na, country in rows_sample:
        if name_ctr[nn] == 1 and len(nn.split()) >= 3 and shown2 < 5:
            print(f"    '{nn[:70]}' | addr='{na[:50]}'")
            shown2 += 1

    print("\n[5] Names where exact name+address repeats (dup in S1):")
    name_addr_ctr = collections.Counter()
    for eid, nn, na, country in rows_sample:
        if na:
            name_addr_ctr[(nn, na)] += 1
    shown3 = 0
    for (nn, na), cnt in name_addr_ctr.most_common(5):
        if cnt >= 2:
            print(f"    freq={cnt} | name='{nn[:50]}' | addr='{na[:50]}'")
            shown3 += 1
    if shown3 == 0:
        print("    None found in sample (S1 is deduplicated reference).")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    print("RECON-02: Name / Address Ambiguity Investigation")
    print(f"  Normalization: NFC + lowercase + punct-strip + whitespace-collapse + null-token-removal")
    print()

    part_b()
    part_c()
    part_d()
    stats, s1 = part_e()
    del s1  # free memory after E
    part_g()
    part_h()

    print(f"\n{'='*60}")
    print(f"TOTAL ELAPSED: {time.time()-t_start:.1f}s")
    print("="*60)
