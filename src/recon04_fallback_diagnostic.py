"""
RECON-04: Targeted Retrieval Fallback Diagnostic

Objective:
Keep n4addr (50 S2 + 50 S3 = 100 total) as the primary baseline.
Test cheap name-only retrieval fallbacks:
  1. Word-token BM25 on business_name only (top-10 per source and top-20 per source)
  2. Character 4-gram BM25 on business_name only (top-10 per source)

Measure:
  - Baseline n4addr recall
  - Fallback standalone recall
  - Overlap (n4addr ∩ fallback)
  - True matches recovered EXCLUSIVELY by fallback (CRITICAL METRIC: additional_recall)
  - Union recall and candidate count increase
  - Breakdowns by US/India, S2/S3, missing-address targets, common-name frequencies
"""

import os, sys, time, math, collections, random, gc, unicodedata, re
import numpy as np
import psutil

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

DATA_DIR   = r"d:\amazon-ml-challenge-2026\student_resource\dataset"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
COUNTRIES  = ["US", "India"]
SAMPLE_N   = 4000
MAX_DF_ABS = 5000
BM25_K1    = 1.5
BM25_B     = 0.75

def ram_gb():
    return psutil.virtual_memory().available / (1024**3)

def used_ram_gb():
    return psutil.virtual_memory().used / (1024**3)

# ─────────────────────────────────────────────────────────────
# NORMALIZATION & TOKENIZATION
# ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\bnull\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_ngrams(text: str, n: int = 4) -> list:
    return [text[i:i+n] for i in range(max(0, len(text) - n + 1))]

def get_tokens(text: str) -> list:
    return text.split() if text else []

# ─────────────────────────────────────────────────────────────
# BM25 INDEX
# ─────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, n=4, use_addr=True, max_df=MAX_DF_ABS, k1=BM25_K1, b=BM25_B):
        self.n = n
        self.use_addr = use_addr
        self.max_df = max_df
        self.k1 = k1
        self.b = b
        self.index: dict = {}
        self.idf:   dict = {}
        self.tf_norm: dict = {}
        self.N = 0
        self.avgdl = 0.0
        self.entity_ids: list = []

    def _tokenize(self, name: str, addr: str) -> list:
        text = (name + " " + addr) if self.use_addr and addr else name
        if self.n is None:
            return list(set(get_tokens(text)))
        else:
            return list(set(get_ngrams(text, self.n)))

    def build(self, entity_ids: list, names: list, addrs: list) -> "BM25Index":
        t0 = time.time()
        self.entity_ids = entity_ids
        self.N = len(names)

        temp = collections.defaultdict(list)
        lengths = []
        for idx in range(self.N):
            toks = self._tokenize(names[idx], addrs[idx])
            for tok in toks:
                temp[tok].append(idx)
            lengths.append(len(toks))

        doc_lengths = np.array(lengths, dtype=np.int16)
        self.avgdl = float(doc_lengths.mean()) if lengths else 1.0

        skipped = kept = 0
        for tok, dl in temp.items():
            if len(dl) > self.max_df:
                skipped += 1
                continue
            pl = np.array(dl, dtype=np.int32)
            df = len(pl)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            dl_ratio = doc_lengths[pl].astype(np.float32) / self.avgdl
            tfn = (self.k1 + 1.0) / (1.0 + self.k1 * (1 - self.b + self.b * dl_ratio))
            self.index[tok]   = pl
            self.idf[tok]     = idf
            self.tf_norm[tok] = tfn
            kept += 1

        del temp, doc_lengths
        tok_lbl = f"{self.n}-gram" if self.n else "word-tok"
        addr_lbl = "+addr" if self.use_addr else " (name only)"
        print(f"      [{tok_lbl}{addr_lbl}] N={self.N:,} | kept={kept:,} | skipped={skipped:,} | {time.time()-t0:.1f}s", flush=True)
        return self

    def query(self, name: str, addr: str, top_k: int = 100) -> list:
        toks = self._tokenize(name, addr)
        if not toks:
            return []

        scores: dict = {}
        for tok in toks:
            pl = self.index.get(tok)
            if pl is None:
                continue
            idf = self.idf[tok]
            tfn = self.tf_norm[tok]
            for i, doc_idx in enumerate(pl):
                v = idf * tfn[i]
                if doc_idx in scores:
                    scores[doc_idx] += v
                else:
                    scores[doc_idx] = v

        if not scores:
            return []

        if len(scores) <= top_k:
            top = sorted(scores.keys(), key=lambda x: -scores[x])
        else:
            top = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]

        return [self.entity_ids[i] for i in top]

    def approx_mem_mb(self) -> float:
        total = sum(pl.nbytes for pl in self.index.values())
        total += sum(tf.nbytes for tf in self.tf_norm.values())
        total += len(self.entity_ids) * 14
        return total / (1024**2)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_source(fname: str, country: str):
    ids, names, addrs = [], [], []
    empty_addrs = set()
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4: p += [""] * (4 - len(p))
            if p[3] != country: continue
            eid = p[0]
            nm = normalize(p[1])
            ad = normalize(p[2])
            ids.append(eid)
            names.append(nm)
            addrs.append(ad)
            if not ad or ad == "null":
                empty_addrs.add(eid)
    return ids, names, addrs, empty_addrs

def load_gt() -> dict:
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1 = p[0]
            matched = p[1] if len(p) > 1 else ""
            gt[s1] = set(matched.split(",")) - {""} if matched.strip() else set()
    return gt

def load_s1_all():
    recs = []
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4: p += [""] * (4 - len(p))
            recs.append((p[0], normalize(p[1]), normalize(p[2]), p[3]))
    return recs

# ─────────────────────────────────────────────────────────────
# STRATIFIED SAMPLING (Identical to RECON-03)
# ─────────────────────────────────────────────────────────────
def sample_s1(s1_all: list, gt: dict, total: int = SAMPLE_N) -> list:
    def mc_bucket(n):
        if n == 0: return "0-sing"
        if n <= 2: return "1-2"
        if n <= 5: return "3-5"
        return "6+"

    buckets = collections.defaultdict(list)
    for rec in s1_all:
        eid = rec[0]; ctr = rec[3]
        nm = len(gt.get(eid, set()))
        buckets[(ctr, mc_bucket(nm))].append(rec)

    grand_total = sum(len(v) for v in buckets.values())
    sampled = []
    for key in sorted(buckets):
        pool = buckets[key]
        n = max(80, int(total * len(pool) / grand_total))
        n = min(n, len(pool))
        sampled.extend(random.sample(pool, n))
        print(f"    {str(key):35} pool={len(pool):,}  sample={n}")

    random.shuffle(sampled)
    return sampled[:total]

# ─────────────────────────────────────────────────────────────
# DIAGNOSTIC EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_fallback(sample: list, gt: dict, base_cands: dict, fb_cands: dict, empty_target_ids: set, name_freq: dict) -> dict:
    """
    Computes baseline recall, fallback standalone recall, overlap, exclusive recovery (additional recall),
    and union recall with fine-grained breakdowns.
    """
    tot_true = 0
    tot_base_hits = 0
    tot_fb_hits = 0
    tot_overlap_hits = 0
    tot_exclusive_hits = 0
    tot_union_hits = 0

    # Macro arrays
    base_rec_arr, fb_rec_arr, union_rec_arr, excl_rec_arr = [], [], [], []
    c_counts = []

    # Slices for breakdowns
    slice_data = collections.defaultdict(lambda: {"true": 0, "base": 0, "fb": 0, "overlap": 0, "excl": 0, "union": 0})

    # Empty address tracking
    empty_target_data = {"true": 0, "base": 0, "fb": 0, "excl": 0, "union": 0}

    for eid, nn, na, ctr in sample:
        true_m = gt.get(eid, set())
        cb = set(base_cands.get(eid, []))
        cfb = set(fb_cands.get(eid, []))
        c_union = cb | cfb
        c_counts.append(len(c_union))

        if not true_m:
            continue

        n_t = len(true_m)
        tot_true += n_t

        h_base = true_m & cb
        h_fb   = true_m & cfb
        h_over = h_base & h_fb
        h_excl = h_fb - h_base
        h_uni  = true_m & c_union

        tot_base_hits      += len(h_base)
        tot_fb_hits        += len(h_fb)
        tot_overlap_hits   += len(h_over)
        tot_exclusive_hits += len(h_excl)
        tot_union_hits     += len(h_uni)

        base_rec_arr.append(len(h_base) / n_t)
        fb_rec_arr.append(len(h_fb) / n_t)
        union_rec_arr.append(len(h_uni) / n_t)
        excl_rec_arr.append(len(h_excl) / n_t)

        # Slice: Country
        for sl_name in [ctr, "Overall"]:
            sd = slice_data[sl_name]
            sd["true"] += n_t; sd["base"] += len(h_base); sd["fb"] += len(h_fb)
            sd["overlap"] += len(h_over); sd["excl"] += len(h_excl); sd["union"] += len(h_uni)

        # Slice: S2 and S3 true matches
        s2t = {m for m in true_m if m.startswith("S2-")}
        s3t = {m for m in true_m if m.startswith("S3-")}
        if s2t:
            sd = slice_data["S2"]
            sd["true"] += len(s2t); sd["base"] += len(s2t & cb); sd["fb"] += len(s2t & cfb)
            sd["overlap"] += len(s2t & h_over); sd["excl"] += len(s2t & h_excl); sd["union"] += len(s2t & c_union)
        if s3t:
            sd = slice_data["S3"]
            sd["true"] += len(s3t); sd["base"] += len(s3t & cb); sd["fb"] += len(s3t & cfb)
            sd["overlap"] += len(s3t & h_over); sd["excl"] += len(s3t & h_excl); sd["union"] += len(s3t & c_union)

        # Slice: Missing-address targets
        em_t = true_m & empty_target_ids
        if em_t:
            empty_target_data["true"]  += len(em_t)
            empty_target_data["base"]  += len(em_t & cb)
            empty_target_data["fb"]    += len(em_t & cfb)
            empty_target_data["excl"]  += len(em_t & h_excl)
            empty_target_data["union"] += len(em_t & c_union)

        # Slice: Common-name frequency
        f = name_freq.get(nn, 1)
        freq_lbl = "freq=1 (unique)" if f == 1 else "freq=2-5" if f <= 5 else "freq=6-20" if f <= 20 else "freq=21-100" if f <= 100 else "freq=101+ (generic)"
        sd = slice_data[freq_lbl]
        sd["true"] += n_t; sd["base"] += len(h_base); sd["fb"] += len(h_fb)
        sd["overlap"] += len(h_over); sd["excl"] += len(h_excl); sd["union"] += len(h_uni)

    cc = np.array(c_counts)
    return {
        "tot_true": tot_true,
        "tot_base_hits": tot_base_hits,
        "tot_fb_hits": tot_fb_hits,
        "tot_overlap_hits": tot_overlap_hits,
        "tot_exclusive_hits": tot_exclusive_hits,
        "tot_union_hits": tot_union_hits,
        "micro_base_recall": (tot_base_hits / tot_true) * 100 if tot_true else 0,
        "micro_fb_recall": (tot_fb_hits / tot_true) * 100 if tot_true else 0,
        "micro_overlap_recall": (tot_overlap_hits / tot_true) * 100 if tot_true else 0,
        "micro_additional_recall": (tot_exclusive_hits / tot_true) * 100 if tot_true else 0,
        "micro_union_recall": (tot_union_hits / tot_true) * 100 if tot_true else 0,
        "macro_base_recall": float(np.mean(base_rec_arr) * 100),
        "macro_fb_recall": float(np.mean(fb_rec_arr) * 100),
        "macro_additional_recall": float(np.mean(excl_rec_arr) * 100),
        "macro_union_recall": float(np.mean(union_rec_arr) * 100),
        "cand_avg": float(cc.mean()),
        "cand_med": float(np.median(cc)),
        "cand_p95": float(np.percentile(cc, 95)),
        "cand_p99": float(np.percentile(cc, 99)),
        "cand_max": int(cc.max()),
        "slice_data": dict(slice_data),
        "empty_target_data": empty_target_data,
    }

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()
    initial_avail_ram = ram_gb()

    print("="*80)
    print("RECON-04: TARGETED RETRIEVAL FALLBACK DIAGNOSTIC")
    print(f"Initial Available RAM: {initial_avail_ram:.2f} GB")
    print("="*80 + "\n")

    gt = load_gt()
    s1_all = load_s1_all()
    name_freq = collections.Counter(nn for _, nn, _, _ in s1_all)

    print("Drawing stratified S1 validation sample (same seed 42, N=4000):")
    sample = sample_s1(s1_all, gt, SAMPLE_N)
    del s1_all; gc.collect()
    print(f"Sample drawn: {len(sample):,} S1 entities | Avail RAM={ram_gb():.2f}GB\n")

    # Candidate storage:
    # Baseline: n4addr (50 S2 + 50 S3 = 100)
    # Fallback 1: Word-token BM25 (top-10 per source -> 10 S2 + 10 S3 = 20)
    # Fallback 2: Word-token BM25 (top-20 per source -> 20 S2 + 20 S3 = 40)
    # Fallback 3: Char 4-gram name-only BM25 (top-10 per source -> 10 S2 + 10 S3 = 20)
    cands_n4addr_s2 = collections.defaultdict(list)
    cands_n4addr_s3 = collections.defaultdict(list)

    cands_wt_s2_top10 = collections.defaultdict(list)
    cands_wt_s3_top10 = collections.defaultdict(list)
    cands_wt_s2_top20 = collections.defaultdict(list)
    cands_wt_s3_top20 = collections.defaultdict(list)

    cands_n4name_s2_top10 = collections.defaultdict(list)
    cands_n4name_s3_top10 = collections.defaultdict(list)

    all_empty_target_ids = set()

    for ctr in COUNTRIES:
        s1_ctr = [(eid, nn, na, c) for eid, nn, na, c in sample if c == ctr]
        if not s1_ctr: continue
        print(f"── Processing Country={ctr} ({len(s1_ctr):,} sample S1) ──")

        # ── S2 ──
        print(f"  Loading S2({ctr})... (Avail RAM: {ram_gb():.1f} GB)")
        t0 = time.time()
        ids2, names2, addrs2, empty2 = load_source("train_source2.tsv", ctr)
        all_empty_target_ids.update(empty2)
        print(f"    Loaded {len(ids2):,} records in {time.time()-t0:.1f}s ({len(empty2):,} empty addrs)")

        # 1. n4addr (primary baseline)
        print(f"    Building [n4addr] S2({ctr})...")
        idx_n4addr = BM25Index(n=4, use_addr=True).build(ids2, names2, addrs2)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            cands_n4addr_s2[eid] = idx_n4addr.query(nn, na, top_k=50)
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s")
        del idx_n4addr; gc.collect()

        # 2. wordtok (name only fallback)
        print(f"    Building [wordtok (name only)] S2({ctr})...")
        idx_wt = BM25Index(n=None, use_addr=False).build(ids2, names2, addrs2)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            q_res = idx_wt.query(nn, "", top_k=20)
            cands_wt_s2_top10[eid] = q_res[:10]
            cands_wt_s2_top20[eid] = q_res[:20]
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s")
        del idx_wt; gc.collect()

        # 3. ngram4 name-only fallback
        print(f"    Building [n4name (name only)] S2({ctr})...")
        idx_n4name = BM25Index(n=4, use_addr=False).build(ids2, names2, addrs2)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            cands_n4name_s2_top10[eid] = idx_n4name.query(nn, "", top_k=10)
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s")
        del idx_n4name; gc.collect()

        del ids2, names2, addrs2, empty2; gc.collect()

        # ── S3 ──
        print(f"  Loading S3({ctr})... (Avail RAM: {ram_gb():.1f} GB)")
        t0 = time.time()
        ids3, names3, addrs3, empty3 = load_source("train_source3.tsv", ctr)
        all_empty_target_ids.update(empty3)
        print(f"    Loaded {len(ids3):,} records in {time.time()-t0:.1f}s ({len(empty3):,} empty addrs)")

        # 1. n4addr (primary baseline)
        print(f"    Building [n4addr] S3({ctr})...")
        idx_n4addr = BM25Index(n=4, use_addr=True).build(ids3, names3, addrs3)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            cands_n4addr_s3[eid] = idx_n4addr.query(nn, na, top_k=50)
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s")
        del idx_n4addr; gc.collect()

        # 2. wordtok (name only fallback)
        print(f"    Building [wordtok (name only)] S3({ctr})...")
        idx_wt = BM25Index(n=None, use_addr=False).build(ids3, names3, addrs3)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            q_res = idx_wt.query(nn, "", top_k=20)
            cands_wt_s3_top10[eid] = q_res[:10]
            cands_wt_s3_top20[eid] = q_res[:20]
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s")
        del idx_wt; gc.collect()

        # 3. ngram4 name-only fallback
        print(f"    Building [n4name (name only)] S3({ctr})...")
        idx_n4name = BM25Index(n=4, use_addr=False).build(ids3, names3, addrs3)
        t_q = time.time()
        for eid, nn, na, _ in s1_ctr:
            cands_n4name_s3_top10[eid] = idx_n4name.query(nn, "", top_k=10)
        print(f"      Queried {len(s1_ctr):,} in {time.time()-t_q:.1f}s\n")
        del idx_n4name; gc.collect()

        del ids3, names3, addrs3, empty3; gc.collect()

    print("All retrieval stages complete. Assembling candidate pools...")

    # Baseline pool (50 S2 + 50 S3 = 100)
    baseline_pool = {eid: cands_n4addr_s2[eid] + cands_n4addr_s3[eid] for eid, *_ in sample}

    # Fallback pools:
    # FB1: word-token top-10 per source (10 S2 + 10 S3 = 20)
    fb_wt_10_pool = {eid: cands_wt_s2_top10[eid] + cands_wt_s3_top10[eid] for eid, *_ in sample}
    # FB2: word-token top-20 per source (20 S2 + 20 S3 = 40)
    fb_wt_20_pool = {eid: cands_wt_s2_top20[eid] + cands_wt_s3_top20[eid] for eid, *_ in sample}
    # FB3: char 4-gram name-only top-10 per source (10 S2 + 10 S3 = 20)
    fb_n4name_10_pool = {eid: cands_n4name_s2_top10[eid] + cands_n4name_s3_top10[eid] for eid, *_ in sample}

    fallbacks_to_test = [
        ("Word-Token Fallback (10 S2 + 10 S3 = 20)", fb_wt_10_pool),
        ("Word-Token Fallback (20 S2 + 20 S3 = 40)", fb_wt_20_pool),
        ("Char 4-gram Name-Only (10 S2 + 10 S3 = 20)", fb_n4name_10_pool),
    ]

    print("\n" + "="*120)
    print("RECON-04 FALLBACK DIAGNOSTIC RESULTS")
    print("="*120)
    H = f"{'Fallback Configuration':40} | {'BaseRec':>7} | {'FB_Alone':>8} | {'Overlap':>7} | {'ADDITIONAL':>10} | {'UnionRec':>8} | {'AvgC':>5} | {'MedC':>5} | {'P95':>5}"
    print(H)
    print("-" * 120)

    fb_results = {}
    for fb_name, fb_pool in fallbacks_to_test:
        diag = evaluate_fallback(sample, gt, baseline_pool, fb_pool, all_empty_target_ids, name_freq)
        fb_results[fb_name] = diag
        print(f"{fb_name:40} | {diag['macro_base_recall']:>6.2f}% | {diag['macro_fb_recall']:>7.2f}% | "
              f"{diag['micro_overlap_recall']:>6.2f}% | {diag['macro_additional_recall']:>9.2f}% | "
              f"{diag['macro_union_recall']:>7.2f}% | {diag['cand_avg']:>5.1f} | {diag['cand_med']:>5.0f} | "
              f"{diag['cand_p95']:>5.0f}")
    print("="*120)

    # Detailed breakdown for the primary fallback (Word-Token 10S2+10S3)
    primary_name = "Word-Token Fallback (10 S2 + 10 S3 = 20)"
    p_diag = fb_results[primary_name]

    print(f"\n{'='*90}")
    print(f"DETAILED BREAKDOWN: {primary_name}")
    print(f"{'='*90}")
    print(f"{'Slice / Category':25} | {'TrueMatches':>11} | {'BaseRecall':>10} | {'ExclRecovered':>13} | {'ADDITIONAL':>10} | {'UnionRecall':>11}")
    print("-" * 90)

    for sl_name, sd in p_diag["slice_data"].items():
        if sd["true"] == 0: continue
        b_rec = (sd["base"] / sd["true"]) * 100
        excl_rec = (sd["excl"] / sd["true"]) * 100
        u_rec = (sd["union"] / sd["true"]) * 100
        print(f"{sl_name:25} | {sd['true']:>11,} | {b_rec:>9.2f}% | {sd['excl']:>13,} | {excl_rec:>9.2f}% | {u_rec:>10.2f}%")

    print("\nMissing-Address Target Records:")
    emd = p_diag["empty_target_data"]
    if emd["true"] > 0:
        b_rec = (emd["base"] / emd["true"]) * 100
        excl_rec = (emd["excl"] / emd["true"]) * 100
        u_rec = (emd["union"] / emd["true"]) * 100
        print(f"  Target matches with EMPTY address: N={emd['true']:,}")
        print(f"    Baseline n4addr Recall:              {b_rec:.2f}% ({emd['base']:,} / {emd['true']:,})")
        print(f"    Exclusively Recovered by Fallback:   {excl_rec:.2f}% ({emd['excl']:,} / {emd['true']:,})")
        print(f"    Union Recall on Empty Targets:       {u_rec:.2f}% ({emd['union']:,} / {emd['true']:,})")

    # Detailed breakdown for Char 4-gram Name-Only fallback
    n4_name = "Char 4-gram Name-Only (10 S2 + 10 S3 = 20)"
    n_diag = fb_results[n4_name]
    print(f"\n{'='*90}")
    print(f"DETAILED BREAKDOWN: {n4_name}")
    print(f"{'='*90}")
    print(f"{'Slice / Category':25} | {'TrueMatches':>11} | {'BaseRecall':>10} | {'ExclRecovered':>13} | {'ADDITIONAL':>10} | {'UnionRecall':>11}")
    print("-" * 90)
    for sl_name, sd in n_diag["slice_data"].items():
        if sd["true"] == 0: continue
        b_rec = (sd["base"] / sd["true"]) * 100
        excl_rec = (sd["excl"] / sd["true"]) * 100
        u_rec = (sd["union"] / sd["true"]) * 100
        print(f"{sl_name:25} | {sd['true']:>11,} | {b_rec:>9.2f}% | {sd['excl']:>13,} | {excl_rec:>9.2f}% | {u_rec:>10.2f}%")

    print("\nMissing-Address Target Records:")
    nemd = n_diag["empty_target_data"]
    if nemd["true"] > 0:
        b_rec = (nemd["base"] / nemd["true"]) * 100
        excl_rec = (nemd["excl"] / nemd["true"]) * 100
        u_rec = (nemd["union"] / nemd["true"]) * 100
        print(f"  Target matches with EMPTY address: N={nemd['true']:,}")
        print(f"    Baseline n4addr Recall:              {b_rec:.2f}% ({nemd['base']:,} / {nemd['true']:,})")
        print(f"    Exclusively Recovered by Fallback:   {excl_rec:.2f}% ({nemd['excl']:,} / {nemd['true']:,})")
        print(f"    Union Recall on Empty Targets:       {u_rec:.2f}% ({nemd['union']:,} / {nemd['true']:,})")

    # Candidate Count Distribution for Primary Fallback
    print(f"\n{'='*90}")
    print(f"CANDIDATE COUNT DISTRIBUTION ({primary_name})")
    print(f"{'='*90}")
    counts = np.array([len(set(baseline_pool[eid]) | set(fb_wt_10_pool[eid])) for eid, *_ in sample])
    bins = [0, 50, 100, 105, 110, 115, 120, np.inf]
    lbls = ["<50", "50-100", "101-105", "106-110", "111-115", "116-120", ">120"]
    for lo, hi, l in zip(bins, bins[1:], lbls):
        n = int(((counts >= lo) & (counts < hi)).sum())
        pct = (n / len(counts)) * 100
        print(f"  {l:>10}: {n:>5,} ({pct:>5.1f}%)")
    print(f"  Stats: avg={counts.mean():.1f} | med={np.median(counts):.0f} | p95={np.percentile(counts, 95):.0f} | p99={np.percentile(counts, 99):.0f} | max={counts.max()}")

    # Total runtime
    tot_time = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"RECON-04 COMPLETED in {tot_time/60:.2f} minutes ({tot_time:.1f} s)")
    print(f"Final Available RAM: {ram_gb():.2f} GB")
    print("="*90)
