"""
recon06_cache_scores.py
Uses the exact RECON-05 Train/Val S1 split from exact_splits.pkl,
extracts features, fits Logistic Regression, verifies exact metrics (th=0.58, 77.59% F0.5),
and caches the scored predictions.
"""
import os, sys, time, math, collections, pickle
import numpy as np
from rapidfuzz import fuzz, distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

SCRATCH_DIR_05 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\b15afa6e-4b1b-492d-ba1c-a5c13926ce28\scratch"
SCRATCH_DIR_06 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\a42fa9b5-3788-49d2-bd9b-801e45dfd4c2\scratch"
CAND_CACHE_FILE = os.path.join(SCRATCH_DIR_05, "recon05_candidates.pkl")
EXACT_SPLITS_FILE = os.path.join(SCRATCH_DIR_06, "exact_splits.pkl")
OUTPUT_SCORED_FILE = os.path.join(SCRATCH_DIR_06, "recon05_scored_pairs.pkl")

def extract_features_for_pair(s1_name, s1_addr, cand_name, cand_addr, is_s2, is_india, rank_addr, rank_name, s1_freq):
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

    ret_addr = 1.0 if rank_addr <= 50 else 0.0
    ret_name = 1.0 if rank_name <= 10 else 0.0
    inv_r_addr = (1.0 / rank_addr) if rank_addr <= 50 else 0.0
    inv_r_name = (1.0 / rank_name) if rank_name <= 10 else 0.0

    return [
        n_lev, n_jw, n_ts, n_tset, n_jacc, log_freq, l_diff, l_ratio,
        s1_has, c_has, both, a_lev, a_jw, a_ts, a_tset, a_jacc,
        float(is_s2), float(is_india), ret_addr, ret_name, inv_r_addr, inv_r_name
    ]

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
    scores = [compute_entity_f05(s1_dict[eid]["gt"], pred_dict.get(eid, set())) for eid in s1_list]
    return float(np.mean(scores))

def process_s1_batch(batch_s1_ids, s1_dict, candidates_by_s1, name_freq):
    pairs_meta = []
    X_rows = []
    y_rows = []
    for s1_id in batch_s1_ids:
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
    return pairs_meta, X_rows, y_rows

def main():
    print(f"Loading candidate cache from {CAND_CACHE_FILE}...")
    t0 = time.time()
    with open(CAND_CACHE_FILE, "rb") as f:
        cand_data = pickle.load(f)
    print(f"Loaded candidates in {time.time()-t0:.1f}s.")

    s1_dict = cand_data["s1_dict"]
    name_freq = cand_data["name_freq"]
    candidates_by_s1 = cand_data["candidates_by_s1"]

    with open(EXACT_SPLITS_FILE, "rb") as f:
        train_s1_ids, val_s1_ids = pickle.load(f)

    print(f"Loaded exact splits: Train S1 = {len(train_s1_ids):,} | Val S1 = {len(val_s1_ids):,}")

    print("Extracting features sequentially to preserve exact ordering...")
    t_feat = time.time()
    train_meta, X_train, y_train = process_s1_batch(train_s1_ids, s1_dict, candidates_by_s1, name_freq)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    val_meta, X_val, y_val = process_s1_batch(val_s1_ids, s1_dict, candidates_by_s1, name_freq)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    print(f"Feature extraction done in {time.time()-t_feat:.1f}s.")
    print(f"Train pairs: {len(train_meta):,} (Matches: {y_train.sum():,})")
    print(f"Val pairs:   {len(val_meta):,} (Matches: {y_val.sum():,})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]

    # Organize predictions by S1
    train_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(train_meta, train_probs):
        train_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    val_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(val_meta, val_probs):
        val_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    # Threshold search on Train S1
    thresholds = np.arange(0.10, 0.96, 0.02)
    best_th = 0.50
    best_train_f05 = -1.0
    for th in thresholds:
        pred_dict = {s1: {cid for cid, p, _ in cands if p >= th} for s1, cands in train_preds_by_s1.items()}
        f05 = evaluate_s1_macro_f05(train_s1_ids, s1_dict, pred_dict)
        if f05 > best_train_f05:
            best_train_f05 = f05
            best_th = th

    print(f"\nOptimal Train Threshold: {best_th:.2f} (Train S1 Macro F0.5: {best_train_f05:.4f})")

    # Evaluate on Val S1
    val_pred_dict = {s1: {cid for cid, p, _ in cands if p >= best_th} for s1, cands in val_preds_by_s1.items()}
    val_f05 = evaluate_s1_macro_f05(val_s1_ids, s1_dict, val_pred_dict)
    val_preds_binary = (val_probs >= best_th).astype(int)
    val_tp = int(np.sum((val_preds_binary == 1) & (y_val == 1)))
    val_fp = int(np.sum((val_preds_binary == 1) & (y_val == 0)))
    val_total_gt = sum(len(s1_dict[eid]["gt"]) for eid in val_s1_ids)
    print(f"Val S1 Macro F0.5: {val_f05*100:.2f}% | Val Precision: {val_tp/(val_tp+val_fp)*100:.2f}% | Val End-to-End Recall: {val_tp/val_total_gt*100:.2f}%")
    print(f"Val TP: {val_tp:,} | Val FP: {val_fp:,}")

    # Save scored pairs and metadata
    saved_data = {
        "train_s1_ids": train_s1_ids,
        "val_s1_ids": val_s1_ids,
        "s1_dict": s1_dict,
        "name_freq": name_freq,
        "candidates_by_s1": candidates_by_s1,
        "train_preds_by_s1": train_preds_by_s1,
        "val_preds_by_s1": val_preds_by_s1,
        "best_th": best_th
    }
    with open(OUTPUT_SCORED_FILE, "wb") as f:
        pickle.dump(saved_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Scored pairs cached to {OUTPUT_SCORED_FILE}")

if __name__ == "__main__":
    main()
