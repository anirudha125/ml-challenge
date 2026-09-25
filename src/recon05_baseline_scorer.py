"""
RECON-05: Baseline Candidate Scorer
Amazon ML Challenge 2026

Objective:
Given the ~114-candidate retrieval set (n4addr 50+50 + n4name 10+10),
can a simple, interpretable scorer reliably distinguish true matches from false candidates?

Methodology:
- Re-use exact candidate generation from RECON-04 (seed 42, 3,995 S1 entities).
- Feature extraction using rapidfuzz (character, token, address, rarity, source).
- Strict S1-level held-out split (50% Train S1, 50% Val S1, stratified).
- Train simple LogisticRegression on Train S1 candidate pairs.
- Tune decision threshold on Train S1 to maximize S1 macro F0.5.
- Freeze model and threshold, then evaluate on held-out Val S1.
- Report pair-level, S1-level macro F0.5, score separation, breakdowns, and error cases.
"""

import os, sys, time, math, collections, random, gc, unicodedata, re, pickle
import numpy as np
import psutil
import rapidfuzz
from rapidfuzz import fuzz, distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

DATA_DIR   = r"d:\amazon-ml-challenge-2026\student_resource\dataset"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
SCRATCH_DIR = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\b15afa6e-4b1b-492d-ba1c-a5c13926ce28\scratch"
CAND_CACHE_FILE = os.path.join(SCRATCH_DIR, "recon05_candidates.pkl")

COUNTRIES  = ["US", "India"]
SAMPLE_N   = 4000
MAX_DF_ABS = 5000
BM25_K1    = 1.5
BM25_B     = 0.75

def ram_gb():
    return psutil.virtual_memory().available / (1024**3)

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
# BM25 INDEX (from RECON-04)
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

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_source_records(fname: str, country: str):
    records = {}
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4: p += [""] * (4 - len(p))
            if p[3] != country: continue
            eid = p[0]
            nm = normalize(p[1])
            ad = normalize(p[2])
            records[eid] = (nm, ad)
    return records

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

    random.shuffle(sampled)
    return sampled[:total]

# ─────────────────────────────────────────────────────────────
# CANDIDATE RETRIEVAL PIPELINE (OR LOAD FROM CACHE)
# ─────────────────────────────────────────────────────────────
def get_candidate_data():
    if os.path.exists(CAND_CACHE_FILE):
        print(f"Loading cached candidates from: {CAND_CACHE_FILE}")
        t0 = time.time()
        with open(CAND_CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
        print(f"Loaded cached candidates in {time.time()-t0:.1f}s")
        return cached_data

    print("Candidate cache not found. Generating candidates via RECON-04 pipeline...")
    t_start = time.time()

    gt = load_gt()
    s1_all = load_s1_all()
    name_freq = collections.Counter(nn for _, nn, _, _ in s1_all)

    print("Drawing stratified S1 validation sample (seed 42, N=4000):")
    sample = sample_s1(s1_all, gt, SAMPLE_N)
    del s1_all; gc.collect()
    print(f"Sample drawn: {len(sample):,} S1 entities | Avail RAM={ram_gb():.2f}GB\n")

    # Map S1 data
    s1_dict = {eid: {"name": nn, "addr": na, "country": ctr, "gt": gt.get(eid, set())}
               for eid, nn, na, ctr in sample}

    # Store candidate info: s1_id -> dict of cand_id -> {cand_name, cand_addr, is_s2, rank_addr, rank_name}
    candidates_by_s1 = collections.defaultdict(dict)

    for ctr in COUNTRIES:
        s1_ctr = [(eid, s1_dict[eid]["name"], s1_dict[eid]["addr"]) 
                  for eid in s1_dict if s1_dict[eid]["country"] == ctr]
        if not s1_ctr: continue
        print(f"── Processing Country={ctr} ({len(s1_ctr):,} sample S1) ──")

        # S2
        print(f"  Loading S2({ctr})...")
        s2_records = load_source_records("train_source2.tsv", ctr)
        s2_ids = list(s2_records.keys())
        s2_names = [s2_records[k][0] for k in s2_ids]
        s2_addrs = [s2_records[k][1] for k in s2_ids]

        print(f"    Building [n4addr] S2({ctr})...")
        idx_n4addr = BM25Index(n=4, use_addr=True).build(s2_ids, s2_names, s2_addrs)
        for eid, nn, na in s1_ctr:
            q_res = idx_n4addr.query(nn, na, top_k=50)
            for rank, cid in enumerate(q_res):
                c_nm, c_ad = s2_records[cid]
                if cid not in candidates_by_s1[eid]:
                    candidates_by_s1[eid][cid] = {
                        "name": c_nm, "addr": c_ad, "is_s2": 1,
                        "rank_addr": rank + 1, "rank_name": 999
                    }
                else:
                    candidates_by_s1[eid][cid]["rank_addr"] = rank + 1
        del idx_n4addr; gc.collect()

        print(f"    Building [n4name] S2({ctr})...")
        idx_n4name = BM25Index(n=4, use_addr=False).build(s2_ids, s2_names, s2_addrs)
        for eid, nn, na in s1_ctr:
            q_res = idx_n4name.query(nn, "", top_k=10)
            for rank, cid in enumerate(q_res):
                c_nm, c_ad = s2_records[cid]
                if cid not in candidates_by_s1[eid]:
                    candidates_by_s1[eid][cid] = {
                        "name": c_nm, "addr": c_ad, "is_s2": 1,
                        "rank_addr": 999, "rank_name": rank + 1
                    }
                else:
                    candidates_by_s1[eid][cid]["rank_name"] = rank + 1
        del idx_n4name, s2_records, s2_ids, s2_names, s2_addrs; gc.collect()

        # S3
        print(f"  Loading S3({ctr})...")
        s3_records = load_source_records("train_source3.tsv", ctr)
        s3_ids = list(s3_records.keys())
        s3_names = [s3_records[k][0] for k in s3_ids]
        s3_addrs = [s3_records[k][1] for k in s3_ids]

        print(f"    Building [n4addr] S3({ctr})...")
        idx_n4addr = BM25Index(n=4, use_addr=True).build(s3_ids, s3_names, s3_addrs)
        for eid, nn, na in s1_ctr:
            q_res = idx_n4addr.query(nn, na, top_k=50)
            for rank, cid in enumerate(q_res):
                c_nm, c_ad = s3_records[cid]
                if cid not in candidates_by_s1[eid]:
                    candidates_by_s1[eid][cid] = {
                        "name": c_nm, "addr": c_ad, "is_s2": 0,
                        "rank_addr": rank + 1, "rank_name": 999
                    }
                else:
                    candidates_by_s1[eid][cid]["rank_addr"] = rank + 1
        del idx_n4addr; gc.collect()

        print(f"    Building [n4name] S3({ctr})...")
        idx_n4name = BM25Index(n=4, use_addr=False).build(s3_ids, s3_names, s3_addrs)
        for eid, nn, na in s1_ctr:
            q_res = idx_n4name.query(nn, "", top_k=10)
            for rank, cid in enumerate(q_res):
                c_nm, c_ad = s3_records[cid]
                if cid not in candidates_by_s1[eid]:
                    candidates_by_s1[eid][cid] = {
                        "name": c_nm, "addr": c_ad, "is_s2": 0,
                        "rank_addr": 999, "rank_name": rank + 1
                    }
                else:
                    candidates_by_s1[eid][cid]["rank_name"] = rank + 1
        del idx_n4name, s3_records, s3_ids, s3_names, s3_addrs; gc.collect()

    cached_data = {
        "sample": sample,
        "s1_dict": s1_dict,
        "name_freq": dict(name_freq),
        "candidates_by_s1": dict(candidates_by_s1)
    }

    print(f"Saving candidate data to: {CAND_CACHE_FILE}...")
    with open(CAND_CACHE_FILE, "wb") as f:
        pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Candidate generation complete and cached in {time.time()-t_start:.1f}s.")
    return cached_data

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # NAME features
    "name_lev_sim",         # normalized Levenshtein similarity [0, 1]
    "name_jw_sim",          # Jaro-Winkler similarity [0, 1]
    "name_token_sort",      # token sort ratio [0, 1]
    "name_token_set",       # token set ratio [0, 1]
    "name_token_jaccard",   # token jaccard [0, 1]
    "s1_name_log_freq",     # log(1 + frequency of S1 name)
    "name_len_diff",        # abs(len(s1) - len(c))
    "name_len_ratio",       # min_len / max_len [0, 1]
    # ADDRESS features
    "s1_has_addr",          # binary
    "cand_has_addr",        # binary
    "both_have_addr",       # binary
    "addr_lev_sim",         # normalized Levenshtein [0, 1] (0 if missing)
    "addr_jw_sim",          # Jaro-Winkler [0, 1] (0 if missing)
    "addr_token_sort",      # token sort ratio [0, 1] (0 if missing)
    "addr_token_set",       # token set ratio [0, 1] (0 if missing)
    "addr_token_jaccard",   # token jaccard [0, 1] (0 if missing)
    # OTHER features
    "is_s2",                # binary (1 if S2, 0 if S3)
    "is_india",             # binary (1 if India, 0 if US)
    "retrieved_by_addr",    # binary (in n4addr top-50)
    "retrieved_by_name",    # binary (in n4name top-10)
    "inv_rank_addr",        # 1.0 / rank_addr (0 if not in addr)
    "inv_rank_name",        # 1.0 / rank_name (0 if not in name)
]

def extract_features_for_pair(s1_name, s1_addr, cand_name, cand_addr, is_s2, is_india, rank_addr, rank_name, s1_freq):
    # Name features
    n_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name)
    n_jw  = distance.JaroWinkler.similarity(s1_name, cand_name)
    n_ts  = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    n_tset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0

    tok1 = set(s1_name.split())
    tok2 = set(cand_name.split())
    u = len(tok1 | tok2)
    n_jacc = (len(tok1 & tok2) / u) if u > 0 else 0.0

    log_freq = math.log1p(s1_freq)
    l1, l2 = len(s1_name), len(cand_name)
    l_diff = abs(l1 - l2)
    l_ratio = min(l1, l2) / max(l1, l2, 1)

    # Address features
    s1_has = 1.0 if s1_addr and s1_addr != "null" else 0.0
    c_has  = 1.0 if cand_addr and cand_addr != "null" else 0.0
    both   = 1.0 if (s1_has and c_has) else 0.0

    if both:
        a_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr)
        a_jw  = distance.JaroWinkler.similarity(s1_addr, cand_addr)
        a_ts  = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        a_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        atok1 = set(s1_addr.split())
        atok2 = set(cand_addr.split())
        au = len(atok1 | atok2)
        a_jacc = (len(atok1 & atok2) / au) if au > 0 else 0.0
    else:
        a_lev = a_jw = a_ts = a_tset = a_jacc = 0.0

    # Other features
    ret_addr = 1.0 if rank_addr <= 50 else 0.0
    ret_name = 1.0 if rank_name <= 10 else 0.0
    inv_r_addr = (1.0 / rank_addr) if rank_addr <= 50 else 0.0
    inv_r_name = (1.0 / rank_name) if rank_name <= 10 else 0.0

    return [
        n_lev, n_jw, n_ts, n_tset, n_jacc, log_freq, l_diff, l_ratio,
        s1_has, c_has, both, a_lev, a_jw, a_ts, a_tset, a_jacc,
        float(is_s2), float(is_india), ret_addr, ret_name, inv_r_addr, inv_r_name
    ]

# ─────────────────────────────────────────────────────────────
# COMPETITION METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_entity_f05(true_matches: set, pred_matches: set) -> float:
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0
    if len(pred_matches) == 0:
        return 0.0
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    p = tp / len(pred_matches)
    r = tp / len(true_matches)
    return (1.25 * p * r) / (0.25 * p + r)

def evaluate_s1_macro_f05(s1_list, s1_dict, pred_dict):
    scores = []
    for eid in s1_list:
        true_m = s1_dict[eid]["gt"]
        pred_m = pred_dict.get(eid, set())
        scores.append(compute_entity_f05(true_m, pred_m))
    return float(np.mean(scores))

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    t_global_start = time.time()
    print("="*80)
    print("RECON-05: BASELINE CANDIDATE SCORER")
    print(f"System Available RAM: {ram_gb():.2f} GB")
    print("="*80 + "\n")

    # 1. Load / Retrieve candidates
    cand_data = get_candidate_data()
    sample = cand_data["sample"]
    s1_dict = cand_data["s1_dict"]
    name_freq = cand_data["name_freq"]
    candidates_by_s1 = cand_data["candidates_by_s1"]

    # 2. Strict S1-level held-out split (50% Train, 50% Val)
    # Stratified by country and match count bucket
    def mc_bucket(n):
        if n == 0: return "0-sing"
        if n <= 2: return "1-2"
        if n <= 5: return "3-5"
        return "6+"

    buckets = collections.defaultdict(list)
    for eid, nn, na, ctr in sample:
        nm = len(s1_dict[eid]["gt"])
        buckets[(ctr, mc_bucket(nm))].append(eid)

    train_s1_ids = []
    val_s1_ids = []

    for key in sorted(buckets):
        pool = list(buckets[key])
        random.shuffle(pool)
        n_half = len(pool) // 2
        train_s1_ids.extend(pool[:n_half])
        val_s1_ids.extend(pool[n_half:])

    print(f"S1 Split: Train S1 = {len(train_s1_ids):,} | Val S1 = {len(val_s1_ids):,}")

    # 3. Build Pair Datasets and Features
    print("Extracting features for candidate pairs...")
    t_feat = time.time()

    def build_pair_dataset(s1_id_list):
        pairs_meta = []  # (s1_id, cand_id, is_match)
        X_rows = []
        y_rows = []
        for s1_id in s1_id_list:
            s1_info = s1_dict[s1_id]
            s1_nm = s1_info["name"]
            s1_ad = s1_info["addr"]
            ctr = s1_info["country"]
            gt_set = s1_info["gt"]
            is_india = 1 if ctr == "India" else 0
            freq = name_freq.get(s1_nm, 1)

            cands = candidates_by_s1.get(s1_id, {})
            for cand_id, c_info in cands.items():
                c_nm = c_info["name"]
                c_ad = c_info["addr"]
                is_s2 = c_info["is_s2"]
                rank_addr = c_info["rank_addr"]
                rank_name = c_info["rank_name"]

                feats = extract_features_for_pair(
                    s1_nm, s1_ad, c_nm, c_ad, is_s2, is_india, rank_addr, rank_name, freq
                )
                is_match = 1 if cand_id in gt_set else 0

                pairs_meta.append((s1_id, cand_id, is_match))
                X_rows.append(feats)
                y_rows.append(is_match)

        return pairs_meta, np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)

    train_meta, X_train, y_train = build_pair_dataset(train_s1_ids)
    val_meta, X_val, y_val = build_pair_dataset(val_s1_ids)

    print(f"Feature extraction done in {time.time()-t_feat:.1f}s.")
    print(f"  Train pairs: {len(X_train):,} (Matches: {y_train.sum():,}, Ratio: {y_train.mean():.4%})")
    print(f"  Val pairs:   {len(X_val):,} (Matches: {y_val.sum():,}, Ratio: {y_val.mean():.4%})")

    # 4. Standardize and Train Logistic Regression
    print("\nTraining Logistic Regression Baseline Scorer...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    print("\n--- MODEL COEFFICIENTS & INTERPRETABILITY ---")
    coefs = clf.coef_[0]
    sorted_idx = np.argsort(-np.abs(coefs))
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]:22} : Coef = {coefs[idx]:+7.4f} (Odds Ratio: {math.exp(coefs[idx]):6.3f})")
    print(f"  Intercept              : {clf.intercept_[0]:+7.4f}")

    # Predict probabilities
    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]

    # Organize predictions by S1
    train_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(train_meta, train_probs):
        train_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    val_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(val_meta, val_probs):
        val_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    # 5. Tune Threshold on TRAIN S1 Only
    print("\n--- TUNING DECISION THRESHOLD ON TRAIN S1 SET ---")
    thresholds = np.arange(0.10, 0.96, 0.02)
    best_th = 0.50
    best_train_f05 = -1.0

    print(f"{'Threshold':>10} | {'Train S1 Macro F0.5':>20} | {'Avg Pred/S1':>12} | {'Empty Pred %':>12}")
    print("-" * 62)
    for th in thresholds:
        pred_dict = {}
        n_preds = []
        n_empty = 0
        for s1_id in train_s1_ids:
            cands = train_preds_by_s1.get(s1_id, [])
            p_set = {cid for cid, p, _ in cands if p >= th}
            pred_dict[s1_id] = p_set
            n_preds.append(len(p_set))
            if not p_set: n_empty += 1

        f05 = evaluate_s1_macro_f05(train_s1_ids, s1_dict, pred_dict)
        if f05 > best_train_f05:
            best_train_f05 = f05
            best_th = th
        if round(th, 2) in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90] or th == best_th:
            print(f"{th:>10.2f} | {f05:>20.4f} | {np.mean(n_preds):>12.2f} | {n_empty/len(train_s1_ids)*100:>11.1f}%")

    print(f"\nOptimal Train Threshold: {best_th:.2f} (Train S1 Macro F0.5: {best_train_f05:.4f})")
    print("Freezing threshold and evaluating on strictly held-out Validation S1 entities...\n")

    # 6. Unbiased Held-out Evaluation on VAL S1 Set
    # Build validation predictions using frozen threshold
    val_pred_dict = {}
    val_pred_counts = []
    val_n_empty = 0
    for s1_id in val_s1_ids:
        cands = val_preds_by_s1.get(s1_id, [])
        p_set = {cid for cid, p, _ in cands if p >= best_th}
        val_pred_dict[s1_id] = p_set
        val_pred_counts.append(len(p_set))
        if not p_set: val_n_empty += 1

    # S1 Macro F0.5
    overall_val_f05 = evaluate_s1_macro_f05(val_s1_ids, s1_dict, val_pred_dict)

    # Pair-level evaluation on Val
    val_preds_binary = (val_probs >= best_th).astype(int)
    val_tp = int(np.sum((val_preds_binary == 1) & (y_val == 1)))
    val_fp = int(np.sum((val_preds_binary == 1) & (y_val == 0)))
    val_tn = int(np.sum((val_preds_binary == 0) & (y_val == 0)))
    val_fn_cand = int(np.sum((val_preds_binary == 0) & (y_val == 1)))

    pair_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
    pair_rec  = val_tp / (val_tp + val_fn_cand) if (val_tp + val_fn_cand) > 0 else 0.0
    pair_f05  = (1.25 * pair_prec * pair_rec) / (0.25 * pair_prec + pair_rec) if (0.25 * pair_prec + pair_rec) > 0 else 0.0

    fpr = val_fp / (val_fp + val_tn) if (val_fp + val_tn) > 0 else 0.0
    fnr = val_fn_cand / (val_fn_cand + val_tp) if (val_fn_cand + val_tp) > 0 else 0.0

    # Total ground truth matches for Val S1 entities (including unretrieved)
    val_total_gt = sum(len(s1_dict[eid]["gt"]) for eid in val_s1_ids)
    end_to_end_pair_rec = val_tp / val_total_gt if val_total_gt > 0 else 0.0

    print("="*80)
    print("RECON-05 EVALUATION RESULTS (HELD-OUT VAL S1)")
    print("="*80)
    print(f"PAIR-LEVEL (at threshold={best_th:.2f}):")
    print(f"  Precision:           {pair_prec*100:6.2f}% ({val_tp:,} TP / {val_tp+val_fp:,} Pred Matches)")
    print(f"  Recall (Candidates): {pair_rec*100:6.2f}% ({val_tp:,} TP / {val_tp+val_fn_cand:,} Retrieved Matches)")
    print(f"  Recall (End-to-End): {end_to_end_pair_rec*100:6.2f}% ({val_tp:,} TP / {val_total_gt:,} Total GT Matches)")
    print(f"  F0.5 (Candidate):    {pair_f05*100:6.2f}%")
    print(f"  False Positive Rate: {fpr*100:6.3f}% ({val_fp:,} FP / {val_fp+val_tn:,} False Cands)")
    print(f"  False Negative Rate: {fnr*100:6.2f}% ({val_fn_cand:,} Missed / {val_tp+val_fn_cand:,} Retrieved Matches)")

    print(f"\nS1-LEVEL COMPETITION METRIC:")
    print(f"  Macro F0.5 (Overall): {overall_val_f05*100:6.2f}% (over {len(val_s1_ids):,} held-out S1 entities)")
    print(f"  Predicted Matches/S1: mean={np.mean(val_pred_counts):.2f} | med={np.median(val_pred_counts):.0f} | min={np.min(val_pred_counts)} | max={np.max(val_pred_counts)}")
    print(f"  Empty Predictions:    {val_n_empty:,} / {len(val_s1_ids):,} ({val_n_empty/len(val_s1_ids)*100:.2f}%)")
    val_gt_singletons = sum(1 for eid in val_s1_ids if len(s1_dict[eid]["gt"]) == 0)
    print(f"  True Singletons:      {val_gt_singletons:,} / {len(val_s1_ids):,} ({val_gt_singletons/len(val_s1_ids)*100:.2f}%)")

    # 7. BREAKDOWNS ON HELD-OUT VAL S1
    print("\n" + "="*80)
    print("DETAILED BREAKDOWNS ON HELD-OUT VAL S1")
    print("="*80)

    # Slice evaluation helper
    def evaluate_slice(slice_name, s1_subset):
        if not s1_subset: return
        f05 = evaluate_s1_macro_f05(s1_subset, s1_dict, val_pred_dict)
        n_gt = sum(len(s1_dict[eid]["gt"]) for eid in s1_subset)
        preds_in_subset = [len(val_pred_dict.get(eid, set())) for eid in s1_subset]
        print(f"  {slice_name:25} | S1 N={len(s1_subset):>5,} | GT Matches={n_gt:>5,} | Macro F0.5={f05*100:>6.2f}% | AvgPred={np.mean(preds_in_subset):>4.2f}")

    print(f"{'Slice / Category':27} | {'S1 Count':>11} | {'GT Matches':>10} | {'Macro F0.5':>10} | {'AvgPred':>7}")
    print("-" * 80)

    # US vs India
    us_val_s1 = [eid for eid in val_s1_ids if s1_dict[eid]["country"] == "US"]
    in_val_s1 = [eid for eid in val_s1_ids if s1_dict[eid]["country"] == "India"]
    evaluate_slice("Country: US", us_val_s1)
    evaluate_slice("Country: India", in_val_s1)

    # Singleton vs Non-singleton
    sing_val_s1 = [eid for eid in val_s1_ids if len(s1_dict[eid]["gt"]) == 0]
    nonsing_val_s1 = [eid for eid in val_s1_ids if len(s1_dict[eid]["gt"]) > 0]
    evaluate_slice("S1: Singleton (0 matches)", sing_val_s1)
    evaluate_slice("S1: Non-singleton", nonsing_val_s1)

    # Missing address in S1
    empty_addr_val_s1 = [eid for eid in val_s1_ids if not s1_dict[eid]["addr"] or s1_dict[eid]["addr"] == "null"]
    has_addr_val_s1 = [eid for eid in val_s1_ids if s1_dict[eid]["addr"] and s1_dict[eid]["addr"] != "null"]
    evaluate_slice("S1: Missing Address", empty_addr_val_s1)
    evaluate_slice("S1: Has Address", has_addr_val_s1)

    # Name Frequency Buckets
    f1_val = [eid for eid in val_s1_ids if name_freq.get(s1_dict[eid]["name"], 1) == 1]
    f2_5_val = [eid for eid in val_s1_ids if 2 <= name_freq.get(s1_dict[eid]["name"], 1) <= 5]
    f6_20_val = [eid for eid in val_s1_ids if 6 <= name_freq.get(s1_dict[eid]["name"], 1) <= 20]
    f21_100_val = [eid for eid in val_s1_ids if 21 <= name_freq.get(s1_dict[eid]["name"], 1) <= 100]
    f101_val = [eid for eid in val_s1_ids if name_freq.get(s1_dict[eid]["name"], 1) > 100]
    evaluate_slice("Name Freq: 1 (unique)", f1_val)
    evaluate_slice("Name Freq: 2-5", f2_5_val)
    evaluate_slice("Name Freq: 6-20", f6_20_val)
    evaluate_slice("Name Freq: 21-100", f21_100_val)
    evaluate_slice("Name Freq: 101+ (generic)", f101_val)

    # S2 vs S3 Source Level Performance
    print("\n--- SOURCE-LEVEL MATCH PERFORMANCE (S2 vs S3) ---")
    s2_tp = sum(1 for (s1, cid, is_m), prob in zip(val_meta, val_probs) if cid.startswith("S2-") and prob >= best_th and is_m == 1)
    s2_fp = sum(1 for (s1, cid, is_m), prob in zip(val_meta, val_probs) if cid.startswith("S2-") and prob >= best_th and is_m == 0)
    s2_true_cands = sum(1 for (s1, cid, is_m) in val_meta if cid.startswith("S2-") and is_m == 1)
    s2_gt_tot = sum(len({m for m in s1_dict[eid]["gt"] if m.startswith("S2-")}) for eid in val_s1_ids)

    s3_tp = sum(1 for (s1, cid, is_m), prob in zip(val_meta, val_probs) if cid.startswith("S3-") and prob >= best_th and is_m == 1)
    s3_fp = sum(1 for (s1, cid, is_m), prob in zip(val_meta, val_probs) if cid.startswith("S3-") and prob >= best_th and is_m == 0)
    s3_true_cands = sum(1 for (s1, cid, is_m) in val_meta if cid.startswith("S3-") and is_m == 1)
    s3_gt_tot = sum(len({m for m in s1_dict[eid]["gt"] if m.startswith("S3-")}) for eid in val_s1_ids)

    s2_p = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0
    s2_r = s2_tp / s2_gt_tot if s2_gt_tot > 0 else 0
    s2_f05 = (1.25 * s2_p * s2_r) / (0.25 * s2_p + s2_r) if (0.25 * s2_p + s2_r) > 0 else 0

    s3_p = s3_tp / (s3_tp + s3_fp) if (s3_tp + s3_fp) > 0 else 0
    s3_r = s3_tp / s3_gt_tot if s3_gt_tot > 0 else 0
    s3_f05 = (1.25 * s3_p * s3_r) / (0.25 * s3_p + s3_r) if (0.25 * s3_p + s3_r) > 0 else 0

    print(f"  S2: Precision={s2_p*100:5.2f}% | Recall={s2_r*100:5.2f}% ({s2_tp}/{s2_gt_tot}) | F0.5={s2_f05*100:5.2f}% | FP={s2_fp}")
    print(f"  S3: Precision={s3_p*100:5.2f}% | Recall={s3_r*100:5.2f}% ({s3_tp}/{s3_gt_tot}) | F0.5={s3_f05*100:5.2f}% | FP={s3_fp}")

    # 8. SCORE SEPARATION ANALYSIS
    print("\n" + "="*80)
    print("SCORE SEPARATION ANALYSIS (TRUE vs FALSE CANDIDATES)")
    print("="*80)
    true_scores = val_probs[y_val == 1]
    false_scores = val_probs[y_val == 0]

    def stats_str(arr):
        pcts = np.percentile(arr, [0, 10, 25, 50, 75, 90, 99, 100])
        return (f"mean={arr.mean():.4f} | med={pcts[3]:.4f} | std={arr.std():.4f}\n"
                f"    min={pcts[0]:.4f} | p10={pcts[1]:.4f} | p25={pcts[2]:.4f} | p50={pcts[3]:.4f} | "
                f"p75={pcts[4]:.4f} | p90={pcts[5]:.4f} | p99={pcts[6]:.4f} | max={pcts[7]:.4f}")

    print("TRUE CANDIDATE PAIRS (N =", f"{len(true_scores):,}):")
    print("   ", stats_str(true_scores))
    print("\nFALSE CANDIDATE PAIRS (N =", f"{len(false_scores):,}):")
    print("   ", stats_str(false_scores))

    # Histogram comparison
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0001]
    labels = ["0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", "0.50-0.70", "0.70-0.90", "0.90-1.00"]
    print("\nScore Distribution Histogram:")
    print(f"  {'Score Bin':>12} | {'True Pairs Count':>16} | {'True %':>8} | {'False Pairs Count':>17} | {'False %':>8}")
    print("  " + "-" * 72)
    for lo, hi, lbl in zip(bins, bins[1:], labels):
        t_cnt = int(((true_scores >= lo) & (true_scores < hi)).sum())
        f_cnt = int(((false_scores >= lo) & (false_scores < hi)).sum())
        t_pct = (t_cnt / len(true_scores)) * 100
        f_pct = (f_cnt / len(false_scores)) * 100
        print(f"  {lbl:>12} | {t_cnt:>16,} | {t_pct:>7.2f}% | {f_cnt:>17,} | {f_pct:>7.2f}%")

    # 9. ERROR ANALYSIS
    print("\n" + "="*80)
    print("ERROR ANALYSIS: REPRESENTATIVE FAILURE MODES")
    print("="*80)

    # Collect pairs with features and scores
    detailed_val_pairs = []
    for (s1_id, cand_id, is_m), prob in zip(val_meta, val_probs):
        c_info = candidates_by_s1[s1_id][cand_id]
        detailed_val_pairs.append({
            "s1_id": s1_id,
            "cand_id": cand_id,
            "is_match": is_m,
            "prob": prob,
            "s1_name": s1_dict[s1_id]["name"],
            "s1_addr": s1_dict[s1_id]["addr"],
            "cand_name": c_info["name"],
            "cand_addr": c_info["addr"],
            "country": s1_dict[s1_id]["country"],
            "s1_freq": name_freq.get(s1_dict[s1_id]["name"], 1)
        })

    # Top False Positives
    fps = [p for p in detailed_val_pairs if p["is_match"] == 0 and p["prob"] >= best_th]
    fps.sort(key=lambda x: -x["prob"])
    print(f"\n1. TOP HIGH-SCORING FALSE POSITIVES (Total FPs: {len(fps):,}):")
    for i, p in enumerate(fps[:5]):
        print(f"  [{i+1}] Score: {p['prob']:.4f} | Country: {p['country']} | Cand: {p['cand_id']}")
        print(f"      S1:   '{p['s1_name']}' | Addr: '{p['s1_addr']}' (S1 Freq={p['s1_freq']})")
        print(f"      Cand: '{p['cand_name']}' | Addr: '{p['cand_addr']}'")

    # Top False Negatives (Retrieved True Matches that scored below threshold)
    fns = [p for p in detailed_val_pairs if p["is_match"] == 1 and p["prob"] < best_th]
    fns.sort(key=lambda x: x["prob"])
    print(f"\n2. TOP LOW-SCORING RETRIEVED TRUE POSITIVES (Missed True Matches, Total FNs: {len(fns):,}):")
    for i, p in enumerate(fns[:5]):
        print(f"  [{i+1}] Score: {p['prob']:.4f} | Country: {p['country']} | Cand: {p['cand_id']}")
        print(f"      S1:   '{p['s1_name']}' | Addr: '{p['s1_addr']}'")
        print(f"      Cand: '{p['cand_name']}' | Addr: '{p['cand_addr']}'")

    # Common-Name Failures
    cn_failures = [p for p in detailed_val_pairs if p["s1_freq"] >= 20 and ((p["is_match"] == 0 and p["prob"] >= best_th) or (p["is_match"] == 1 and p["prob"] < best_th))]
    print(f"\n3. COMMON-NAME FAILURES (Freq >= 20, Total: {len(cn_failures):,}):")
    for i, p in enumerate(cn_failures[:4]):
        err_type = "FP" if p["is_match"] == 0 else "FN"
        print(f"  [{err_type}] Score: {p['prob']:.4f} | Freq: {p['s1_freq']} | S1: '{p['s1_name']}'")
        print(f"      S1 Addr:   '{p['s1_addr']}'")
        print(f"      Cand Addr: '{p['cand_addr']}' ({p['cand_id']})")

    # Missing-Address Failures
    ma_failures = [p for p in detailed_val_pairs if (not p["s1_addr"] or not p["cand_addr"]) and ((p["is_match"] == 0 and p["prob"] >= best_th) or (p["is_match"] == 1 and p["prob"] < best_th))]
    print(f"\n4. MISSING-ADDRESS FAILURES (Total: {len(ma_failures):,}):")
    for i, p in enumerate(ma_failures[:4]):
        err_type = "FP" if p["is_match"] == 0 else "FN"
        print(f"  [{err_type}] Score: {p['prob']:.4f} | Cand: {p['cand_id']}")
        print(f"      S1:   '{p['s1_name']}' | Addr: '{p['s1_addr']}'")
        print(f"      Cand: '{p['cand_name']}' | Addr: '{p['cand_addr']}'")

    # India Transliteration Failures
    in_failures = [p for p in detailed_val_pairs if p["country"] == "India" and p["is_match"] == 1 and p["prob"] < best_th]
    print(f"\n5. INDIA TRANSLITERATION / NOISY TRUE MATCHES MISSED (Total: {len(in_failures):,}):")
    for i, p in enumerate(in_failures[:4]):
        print(f"  [FN] Score: {p['prob']:.4f} | Cand: {p['cand_id']}")
        print(f"      S1:   '{p['s1_name']}' | Addr: '{p['s1_addr']}'")
        print(f"      Cand: '{p['cand_name']}' | Addr: '{p['cand_addr']}'")

    # Total Runtime
    tot_time = time.time() - t_global_start
    print("\n" + "="*80)
    print(f"RECON-05 COMPLETED IN {tot_time/60:.2f} MINUTES ({tot_time:.1f}s)")
    print("="*80)

if __name__ == "__main__":
    main()
