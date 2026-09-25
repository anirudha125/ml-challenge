"""
RECON-03: Fuzzy Candidate Retrieval Benchmark
Experiments A (char n-gram BM25: 3/4/5-gram)
            B (word-token BM25)
            C (union of best n-gram + token)
            D (name + address n-gram union)
            + RapidFuzz diagnostic
            + Hard-case analysis

Design:
  - Each (source, country) pair is loaded ONCE; all index types built in sequence.
  - BM25 implemented from scratch over custom inverted index.
  - max_df_frac=0.10: n-grams appearing in >10% of docs are skipped (too generic).
  - Stratified S1 sample of ~18K for reliable statistics.
  - Ground truth used ONLY to compute recall AFTER retrieval; never to build index.
  - No external data, no leakage.
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
K_VALUES   = [10, 20, 50, 100]
SAMPLE_N   = 18000   # total S1 sample
MAX_DF_FRAC = 0.10   # skip terms appearing in > 10% of indexed docs
BM25_K1    = 1.5
BM25_B     = 0.75

# ─────────────────────────────────────────────────────────────
# 1. NORMALIZATION  (identical to RECON-02)
# ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\bnull\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def ngrams(text: str, n: int):
    return [text[i:i+n] for i in range(max(0, len(text) - n + 1))]

def word_tokens(text: str):
    return text.split() if text else []


# ─────────────────────────────────────────────────────────────
# 2. BM25 INVERTED INDEX
# ─────────────────────────────────────────────────────────────
class BM25Index:
    """
    Builds a BM25 inverted index over a list of text records.
    Tokenization: character n-grams (n>=2) OR word tokens (n=None).
    use_addr=True: concatenates name + addr before tokenizing.
    """
    def __init__(self, n=None, use_addr=False,
                 max_df_frac=MAX_DF_FRAC, k1=BM25_K1, b=BM25_B):
        self.n = n
        self.use_addr = use_addr
        self.max_df_frac = max_df_frac
        self.k1 = k1
        self.b = b
        # index: term -> np.int32 array of doc indices
        self.index: dict[str, np.ndarray] = {}
        self.doc_lengths: np.ndarray = None   # int16, unique tok count per doc
        self.N = 0
        self.avgdl = 0.0
        self.entity_ids: list[str] = []
        self._score: np.ndarray = None        # reused across queries

    def _tokenize(self, name: str, addr: str) -> list[str]:
        text = (name + " " + addr) if self.use_addr and addr else name
        if self.n is None:
            return list(set(word_tokens(text)))
        else:
            return list(set(ngrams(text, self.n)))

    def build(self, entity_ids: list, names: list, addrs: list) -> "BM25Index":
        t0 = time.time()
        self.entity_ids = entity_ids
        self.N = len(names)
        assert self.N == len(addrs)

        temp = collections.defaultdict(list)
        lengths = []
        for idx in range(self.N):
            toks = self._tokenize(names[idx], addrs[idx])
            for tok in toks:
                temp[tok].append(idx)
            lengths.append(len(toks))

        self.doc_lengths = np.array(lengths, dtype=np.int16)
        self.avgdl = float(self.doc_lengths.mean()) if lengths else 1.0

        max_df = max(1, int(self.max_df_frac * self.N))
        skipped = 0
        for tok, doc_list in temp.items():
            if len(doc_list) <= max_df:
                self.index[tok] = np.array(doc_list, dtype=np.int32)
            else:
                skipped += 1
        del temp
        gc.collect()

        self._score = np.zeros(self.N, dtype=np.float32)
        tok_type = f"{self.n}-gram" if self.n else "word-tok"
        addr_sfx = "+addr" if self.use_addr else ""
        print(f"      [{tok_type}{addr_sfx}] N={self.N:,} docs | "
              f"{len(self.index):,} terms | {skipped:,} high-df skipped | "
              f"{time.time()-t0:.1f}s", flush=True)
        return self

    def query(self, name: str, addr: str, top_k: int = 100) -> list[str]:
        toks = self._tokenize(name, addr)
        if not toks:
            return []

        sa = self._score
        accessed: list[np.ndarray] = []
        for tok in toks:
            pl = self.index.get(tok)
            if pl is None:
                continue
            df = len(pl)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            dl_ratio = self.doc_lengths[pl].astype(np.float32) / self.avgdl
            tf_n = (self.k1 + 1.0) / (1.0 + self.k1 * (1 - self.b + self.b * dl_ratio))
            np.add.at(sa, pl, idf * tf_n)
            accessed.append(pl)

        if not accessed:
            return []

        # top-K by score
        k = min(top_k, self.N)
        if k >= self.N:
            top_idx = np.argsort(-sa)
        else:
            top_idx = np.argpartition(sa, -k)[-k:]
            top_idx = top_idx[np.argsort(-sa[top_idx])]

        result = [self.entity_ids[i] for i in top_idx if sa[i] > 0.0]

        # reset touched entries only
        for pl in accessed:
            sa[pl] = 0.0

        return result[:top_k]

    def approx_mem_mb(self) -> float:
        total = sum(pl.nbytes for pl in self.index.values())
        total += self.doc_lengths.nbytes if self.doc_lengths is not None else 0
        total += self._score.nbytes if self._score is not None else 0
        # rough estimate for entity_ids strings
        total += len(self.entity_ids) * 14  # avg S2-/S3- id is ~12 chars
        return total / (1024 ** 2)


# ─────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_source(fname: str, country: str):
    """Stream source file, return (ids, names, addrs) lists filtered by country."""
    ids, names, addrs = [], [], []
    path = os.path.join(TRAIN_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                p += [""] * (4 - len(p))
            if p[3] != country:
                continue
            ids.append(p[0])
            names.append(normalize(p[1]))
            addrs.append(normalize(p[2]))
    return ids, names, addrs


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
            if len(p) < 4:
                p += [""] * (4 - len(p))
            recs.append((p[0], normalize(p[1]), normalize(p[2]), p[3]))
    return recs


# ─────────────────────────────────────────────────────────────
# 4. STRATIFIED SAMPLING
# ─────────────────────────────────────────────────────────────
def sample_s1(s1_all: list, gt: dict, total: int = SAMPLE_N) -> list:
    def mc_bucket(n):
        if n == 0: return "0-sing"
        if n <= 2: return "1-2"
        if n <= 5: return "3-5"
        return "6+"

    buckets = collections.defaultdict(list)
    for rec in s1_all:
        eid, nn, na, ctr = rec
        nm = len(gt.get(eid, set()))
        buckets[(ctr, mc_bucket(nm))].append(rec)

    grand_total = sum(len(v) for v in buckets.values())
    sampled = []
    for key in sorted(buckets):
        pool = buckets[key]
        n = max(300, int(total * len(pool) / grand_total))
        n = min(n, len(pool))
        chosen = random.sample(pool, n)
        sampled.extend(chosen)
        print(f"    {str(key):35} pool={len(pool):,}  sample={n}")

    random.shuffle(sampled)
    return sampled[:total]


# ─────────────────────────────────────────────────────────────
# 5. RECALL EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(sample: list, gt: dict,
             cands_by_s1: dict, K_values: list) -> dict:
    """Return nested dict K -> metric_name -> value."""
    results = {}
    for K in K_values:
        ns_recalls, us_r, in_r, s2_r, s3_r = [], [], [], [], []
        mc_recalls = collections.defaultdict(list)
        n_empty = 0
        cand_counts = []

        for eid, nn, na, ctr in sample:
            true_m = gt.get(eid, set())
            cands_k = cands_by_s1.get(eid, [])[:K]
            cand_counts.append(len(cands_k))
            if not cands_k:
                n_empty += 1
            if not true_m:
                continue   # singleton — excluded from recall

            cands_set = set(cands_k)
            tp = len(true_m & cands_set)
            rec = tp / len(true_m)
            ns_recalls.append(rec)

            s2t = {m for m in true_m if m.startswith("S2-")}
            s3t = {m for m in true_m if m.startswith("S3-")}
            if s2t: s2_r.append(len(s2t & cands_set) / len(s2t))
            if s3t: s3_r.append(len(s3t & cands_set) / len(s3t))
            if ctr == "US":   us_r.append(rec)
            elif ctr == "India": in_r.append(rec)

            nm = len(true_m)
            mc_recalls["1" if nm == 1 else "2-3" if nm <= 3 else "4-5" if nm <= 5 else "6+"].append(rec)

        A = lambda lst: np.mean(lst) * 100 if lst else float("nan")
        cc = np.array(cand_counts)
        results[K] = {
            "overall": A(ns_recalls), "S2": A(s2_r), "S3": A(s3_r),
            "US": A(us_r), "India": A(in_r),
            "by_mc": {k: A(v) for k, v in mc_recalls.items()},
            "avg_c": cc.mean(), "med_c": float(np.median(cc)),
            "p95_c": float(np.percentile(cc, 95)),
            "p99_c": float(np.percentile(cc, 99)),
            "max_c": int(cc.max()),
            "empty_pct": n_empty / len(sample) * 100,
            "n_ns": len(ns_recalls),
        }
    return results


def union_cands(cA: dict, cB: dict, max_k: int = 200) -> dict:
    """Merge two candidate dicts; A-ranked items first, new-from-B appended."""
    out = {}
    for eid in set(cA) | set(cB):
        a = cA.get(eid, [])
        b = cB.get(eid, [])
        seen = set(a)
        merged = list(a)
        for x in b:
            if x not in seen:
                seen.add(x)
                merged.append(x)
        out[eid] = merged[:max_k]
    return out


# ─────────────────────────────────────────────────────────────
# 6. MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────
def ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def run_all_experiments(sample: list, gt: dict):
    """
    For each (country, source):
      - Load data once
      - Build 5 indices (3-gram, 4-gram, 5-gram, word-tok, 4-gram+addr)
      - Query each S1 in sample, accumulate candidates

    Returns dict: experiment_name -> {s1_id: [cand_ids]}
    """
    EXPS = [
        ("A_3gram",     3,    False),
        ("A_4gram",     4,    False),
        ("A_5gram",     5,    False),
        ("B_wordtok",   None, False),
        ("D_4gram_addr",4,    True),
    ]

    # experiment -> s1_id -> list of candidate ids (from S2 + S3 combined)
    all_cands: dict[str, dict[str, list]] = {e[0]: {} for e in EXPS}

    for ctr in COUNTRIES:
        s1_ctr = [(eid, nn, na, c) for eid, nn, na, c in sample if c == ctr]
        if not s1_ctr:
            continue
        print(f"\n  ── Country={ctr}  S1-sample={len(s1_ctr):,} ──")

        for src_fname, src_label in [("train_source2.tsv","S2"),
                                     ("train_source3.tsv","S3")]:
            print(f"    Loading {src_label}({ctr})... RAM={ram_gb():.1f}GB", flush=True)
            t_load = time.time()
            ids, names, addrs = load_source(src_fname, ctr)
            print(f"      {len(ids):,} docs loaded in {time.time()-t_load:.1f}s")

            for exp_name, n_size, use_addr in EXPS:
                print(f"    Building index [{exp_name}] for {src_label}({ctr})...")
                idx = BM25Index(n=n_size, use_addr=use_addr).build(ids, names, addrs)
                mem_mb = idx.approx_mem_mb()
                print(f"      Index mem ≈ {mem_mb:.0f} MB | RAM avail = {ram_gb():.1f} GB")

                t_q = time.time()
                for eid, nn, na, _ in s1_ctr:
                    cands = idx.query(nn, na, top_k=max(K_VALUES))
                    if eid not in all_cands[exp_name]:
                        all_cands[exp_name][eid] = []
                    all_cands[exp_name][eid].extend(cands)
                elapsed = time.time() - t_q
                print(f"      Queried {len(s1_ctr):,} S1 in {elapsed:.1f}s "
                      f"({elapsed/len(s1_ctr)*1000:.2f}ms/query)", flush=True)

                del idx
                gc.collect()

            del ids, names, addrs
            gc.collect()

    # Deduplicate candidates per S1 (preserve order: S2-first, then new S3)
    for exp_name in all_cands:
        for eid in all_cands[exp_name]:
            seen = set()
            dedup = []
            for c in all_cands[exp_name][eid]:
                if c not in seen:
                    seen.add(c)
                    dedup.append(c)
            all_cands[exp_name][eid] = dedup

    return all_cands


# ─────────────────────────────────────────────────────────────
# 7. CANDIDATE VOLUME DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def cand_volume_report(cands_by_s1: dict, sample: list, K: int, label: str):
    counts = np.array([len(cands_by_s1.get(eid, [])[:K]) for eid, *_ in sample])
    bins = [0, 1, 11, 21, 51, 101, 501, np.inf]
    labels = ["0","1-10","11-20","21-50","51-100","101-500",">500"]
    print(f"\n  [{label} @ K={K}] Candidate count distribution ({len(counts):,} S1 records):")
    for lo, hi, lbl in zip(bins, bins[1:], labels):
        n = int(((counts >= lo) & (counts < hi)).sum())
        print(f"    {lbl:>8}: {n:>7,}  ({n/len(counts)*100:>5.1f}%)")
    print(f"    avg={counts.mean():.1f}  med={np.median(counts):.0f}  "
          f"p95={np.percentile(counts,95):.0f}  p99={np.percentile(counts,99):.0f}  "
          f"max={counts.max():.0f}  total_pairs={counts.sum():,}")


# ─────────────────────────────────────────────────────────────
# 8. HARD-CASE ANALYSIS
# ─────────────────────────────────────────────────────────────
def hard_case_analysis(sample: list, gt: dict,
                       best_cands: dict, K: int = 100):
    print(f"\n{'='*60}")
    print(f"HARD CASE ANALYSIS (best candidate set, K={K})")
    print(f"{'='*60}")

    cases = []
    for eid, nn, na, ctr in sample:
        true_m = gt.get(eid, set())
        if not true_m: continue
        cset = set(best_cands.get(eid, [])[:K])
        tp = len(true_m & cset)
        rec = tp / len(true_m)
        cases.append(dict(eid=eid, nn=nn, na=na, ctr=ctr,
                          n_true=len(true_m), rec=rec, true_m=true_m))

    cats = {
        "overall non-singleton": cases,
        "US": [c for c in cases if c["ctr"]=="US"],
        "India": [c for c in cases if c["ctr"]=="India"],
        "recall=0 (total miss)": [c for c in cases if c["rec"]==0],
        "recall=1 (perfect)": [c for c in cases if c["rec"]==1.0],
        "1 match": [c for c in cases if c["n_true"]==1],
        "2-3 matches": [c for c in cases if 2<=c["n_true"]<=3],
        "4-5 matches": [c for c in cases if 4<=c["n_true"]<=5],
        "6+ matches": [c for c in cases if c["n_true"]>=6],
    }
    print(f"  {'Category':35} | {'N':>6} | {'AvgRecall':>10} | {'Zero%':>8}")
    for cat, cc in cats.items():
        if not cc: continue
        ar = np.mean([c["rec"] for c in cc])*100
        zr = sum(1 for c in cc if c["rec"]==0) / len(cc)*100
        print(f"  {cat:35} | {len(cc):>6,} | {ar:>9.2f}% | {zr:>7.2f}%")

    # Sample failures
    misses = [c for c in cases if c["rec"]==0]
    print(f"\n  Sample total-miss cases (recall=0 at K={K}):")
    for c in misses[:8]:
        first_true = next(iter(c["true_m"]))
        print(f"    S1 [{c['ctr']}]: '{c['nn'][:55]}'")
        print(f"         addr: '{c['na'][:55]}'")
        print(f"    TrueID {first_true}  (n_true={c['n_true']})")


# ─────────────────────────────────────────────────────────────
# 9. COMMON-NAME ANALYSIS
# ─────────────────────────────────────────────────────────────
def common_name_analysis(sample: list, gt: dict,
                         best_cands: dict, name_freq: dict, K: int = 100):
    print(f"\n{'='*60}")
    print(f"COMMON-NAME RECALL ANALYSIS (K={K})")
    print(f"{'='*60}")
    bins = [
        ("freq=1 (unique)",    lambda f: f==1),
        ("freq=2-5",           lambda f: 2<=f<=5),
        ("freq=6-20",          lambda f: 6<=f<=20),
        ("freq=21-100",        lambda f: 21<=f<=100),
        ("freq=101+ (generic)",lambda f: f>100),
    ]
    bucket_recs = {b[0]: [] for b in bins}
    for eid, nn, na, ctr in sample:
        tm = gt.get(eid, set())
        if not tm: continue
        freq = name_freq.get(nn, 1)
        cset = set(best_cands.get(eid, [])[:K])
        rec = len(tm & cset) / len(tm)
        for bname, fn in bins:
            if fn(freq):
                bucket_recs[bname].append(rec)
                break
    print(f"  {'Bucket':30} | {'N':>6} | {'AvgRecall':>10}")
    for bname, recs in bucket_recs.items():
        print(f"  {bname:30} | {len(recs):>6,} | {np.mean(recs)*100:>9.2f}%" if recs else f"  {bname:30} | {'0':>6} |       N/A")


# ─────────────────────────────────────────────────────────────
# 10. RAPIDFUZZ DIAGNOSTIC
# ─────────────────────────────────────────────────────────────
def rapidfuzz_diagnostic(sample: list, gt: dict,
                         bm25_cands: dict, s23_dict: dict,
                         n_s1: int = 300, K_in: int = 20):
    try:
        from rapidfuzz import fuzz
    except ImportError:
        print("\n  rapidfuzz not available — skipping")
        return

    print(f"\n{'='*60}")
    print(f"RAPIDFUZZ RERANKING DIAGNOSTIC (K_in={K_in}, n_s1={n_s1})")
    print(f"{'='*60}")

    ns_sample = [(eid, nn, na, ctr) for eid, nn, na, ctr in sample
                 if gt.get(eid)][:n_s1]
    impr, same, degrad = 0, 0, 0

    t0 = time.time()
    for eid, nn, na, ctr in ns_sample:
        true_m = gt.get(eid, set())
        if not true_m: continue
        cands_in = bm25_cands.get(eid, [])[:K_in]
        if not cands_in: continue

        bm25_rec = len(true_m & set(cands_in))

        scored = []
        for cid in cands_in:
            if cid in s23_dict:
                s = fuzz.token_sort_ratio(nn, s23_dict[cid][0])
            else:
                s = 0
            scored.append((cid, s))
        scored.sort(key=lambda x: -x[1])
        reranked = [x[0] for x in scored]
        rf_rec = len(true_m & set(reranked[:K_in]))

        if rf_rec > bm25_rec: impr += 1
        elif rf_rec < bm25_rec: degrad += 1
        else: same += 1

    total = impr + same + degrad
    print(f"  Tested {total} S1 records")
    print(f"  Recall improved after RF reranking: {impr}  ({impr/total*100:.1f}%)")
    print(f"  Recall unchanged:                   {same}  ({same/total*100:.1f}%)")
    print(f"  Recall degraded:                    {degrad}  ({degrad/total*100:.1f}%)")
    print(f"  Time: {time.time()-t0:.1f}s")
    note = "helps" if impr > degrad else ("neutral" if impr == degrad else "hurts")
    print(f"  → RapidFuzz reranking at K={K_in}: **{note}**")


# ─────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────
def print_table(all_results: dict):
    hdr = f"{'Method':25}|{'K':>5}|{'Overall':>8}|{'S2':>7}|{'S3':>7}|{'US':>7}|{'India':>7}|{'AvgC':>6}|{'P95':>5}|{'P99':>5}|{'MaxC':>6}"
    print(f"\n{'='*len(hdr)}")
    print("CONSOLIDATED RECALL TABLE")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print("-"*len(hdr))
    for mname in sorted(all_results):
        for K in sorted(all_results[mname]):
            r = all_results[mname][K]
            print(f"{mname:25}|{K:>5}|{r['overall']:>7.2f}%|{r['S2']:>6.2f}%|{r['S3']:>6.2f}%|"
                  f"{r['US']:>6.2f}%|{r['India']:>6.2f}%|{r['avg_c']:>6.1f}|"
                  f"{r['p95_c']:>5.0f}|{r['p99_c']:>5.0f}|{r['max_c']:>6}")
            mc = r.get("by_mc", {})
            if mc:
                mc_str = "  by_mc:" + "  ".join(f"{k}:{v:.1f}%" for k,v in sorted(mc.items()))
                print(f"{'':25}|{'':5}|{mc_str}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T_TOTAL = time.time()
    print(f"RECON-03 Start | RAM avail: {ram_gb():.2f} GB\n")

    # ── Load GT and S1 ──
    print("Loading ground truth...")
    gt = load_gt()
    print(f"  {len(gt):,} S1 GT entries")

    print("Loading S1 records...")
    s1_all = load_s1_all()
    print(f"  {len(s1_all):,} S1 records")

    name_freq_s1 = collections.Counter(nn for _,nn,_,_ in s1_all)

    # ── Stratified sample ──
    print("\nStratified S1 sample:")
    sample = sample_s1(s1_all, gt, SAMPLE_N)
    del s1_all; gc.collect()
    print(f"  Final sample: {len(sample):,} | RAM: {ram_gb():.2f} GB")

    # ── Run all retrieval experiments ──
    print("\n" + "="*60)
    print("RUNNING RETRIEVAL EXPERIMENTS")
    print("="*60)
    all_cands = run_all_experiments(sample, gt)

    # ── Compute union experiments ──
    print("\nComputing union experiments...")
    # Best n-gram (will determine after seeing results — use 3-gram for union as it's broadest)
    # C: 3-gram union token
    all_cands["C_3g_tok"] = union_cands(all_cands["A_3gram"], all_cands["B_wordtok"])
    # Also 4-gram union token
    all_cands["C_4g_tok"] = union_cands(all_cands["A_4gram"], all_cands["B_wordtok"])
    # D: 4-gram name+addr (already computed as D_4gram_addr)
    # E: best_name_union + addr
    all_cands["E_C4g_addr"] = union_cands(all_cands["C_4g_tok"], all_cands["D_4gram_addr"])

    # ── Evaluate all experiments ──
    print("\nEvaluating recall for all experiments...")
    all_results = {}
    for exp_name, cands in all_cands.items():
        all_results[exp_name] = evaluate(sample, gt, cands, K_VALUES)

    print_table(all_results)

    # ── Candidate volume distribution for best experiment ──
    best_exp = "E_C4g_addr"
    cand_volume_report(all_cands[best_exp], sample, K=100, label=best_exp)
    cand_volume_report(all_cands["A_4gram"], sample, K=100, label="A_4gram")
    cand_volume_report(all_cands["C_4g_tok"], sample, K=100, label="C_4g_tok")

    # ── Hard case analysis ──
    hard_case_analysis(sample, gt, all_cands[best_exp], K=100)

    # ── Common name analysis ──
    common_name_analysis(sample, gt, all_cands[best_exp], name_freq_s1, K=100)

    # ── Load S2/S3 name data for RapidFuzz diagnostic ──
    print("\nLoading S2/S3 records needed for sample (rapidfuzz diagnostic)...")
    needed_ids = set()
    for eid, nn, na, ctr in sample:
        needed_ids.update(all_cands.get("A_4gram", {}).get(eid, [])[:20])
        needed_ids.update(gt.get(eid, set()))
    s23_dict = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4: p += [""]*(4-len(p))
                if p[0] in needed_ids:
                    s23_dict[p[0]] = (normalize(p[1]), normalize(p[2]), p[3])
    print(f"  Loaded {len(s23_dict):,} S2/S3 records")

    rapidfuzz_diagnostic(sample, gt, all_cands["A_4gram"], s23_dict, n_s1=300, K_in=20)

    # ── Index size measurement ──
    print(f"\n{'='*60}")
    print("INDEX SIZE MEASUREMENT (US-S2, standalone)")
    print("='*60")
    for n_size in [3, 4, 5, None]:
        ids_m, names_m, addrs_m = load_source("train_source2.tsv", "US")
        lbl = f"{n_size}-gram" if n_size else "word-tok"
        print(f"  Building {lbl} index for US-S2 ({len(ids_m):,} docs)...")
        idx_m = BM25Index(n=n_size).build(ids_m, names_m, addrs_m)
        sz = idx_m.approx_mem_mb()
        print(f"    In-memory size: {sz:.0f} MB | RAM avail: {ram_gb():.2f} GB")
        del idx_m, ids_m, names_m, addrs_m; gc.collect()

    # ── Extrapolation to test set ──
    print(f"\n{'='*60}")
    print("TEST-SET EXTRAPOLATION")
    print("="*60)
    for exp_name in ["A_4gram", "C_4g_tok", "E_C4g_addr"]:
        avg_c = np.mean([min(len(v), 100) for v in all_cands[exp_name].values()])
        est_pairs = 1_732_544 * avg_c
        print(f"  [{exp_name}] avg cand@K=100: {avg_c:.1f} → est test pairs: {est_pairs:,.0f}")

    elapsed = (time.time() - T_TOTAL) / 60
    print(f"\n{'='*60}")
    print(f"RECON-03 COMPLETE | Total runtime: {elapsed:.1f} minutes")
    print(f"  RAM at end: {ram_gb():.2f} GB")
    print("="*60)
