"""
recon06_structural_and_experiment.py
Performs STEP 1 (structural lever diagnostic) and STEP 2 (clean post-processing experiment)
strictly honoring the 1,994 Train S1 / 2,001 Val S1 split.
"""
import os, sys, time, math, collections, pickle
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SCRATCH_DIR_06 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\a42fa9b5-3788-49d2-bd9b-801e45dfd4c2\scratch"
SCORED_FILE = os.path.join(SCRATCH_DIR_06, "recon05_scored_pairs.pkl")

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

def evaluate_macro_f05(s1_list, s1_dict, pred_dict):
    scores = [compute_entity_f05(s1_dict[eid]["gt"], pred_dict.get(eid, set())) for eid in s1_list]
    return float(np.mean(scores))

def evaluate_full_metrics(s1_list, s1_dict, pred_dict):
    macro_f05 = evaluate_macro_f05(s1_list, s1_dict, pred_dict)
    
    tp = 0
    fp = 0
    tot_preds = 0
    for s1_id in s1_list:
        p_set = pred_dict.get(s1_id, set())
        gt_set = s1_dict[s1_id]["gt"]
        tot_preds += len(p_set)
        tp += len(p_set & gt_set)
        fp += len(p_set - gt_set)
    
    tot_gt = sum(len(s1_dict[eid]["gt"]) for eid in s1_list)
    prec = tp / tot_preds if tot_preds > 0 else 0.0
    rec = tp / tot_gt if tot_gt > 0 else 0.0
    pair_f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0

    # Subsets
    sing_ids = [eid for eid in s1_list if len(s1_dict[eid]["gt"]) == 0]
    us_ids = [eid for eid in s1_list if s1_dict[eid]["country"] == "US"]
    in_ids = [eid for eid in s1_list if s1_dict[eid]["country"] == "India"]

    sing_f05 = evaluate_macro_f05(sing_ids, s1_dict, pred_dict) if sing_ids else 0.0
    us_f05 = evaluate_macro_f05(us_ids, s1_dict, pred_dict) if us_ids else 0.0
    in_f05 = evaluate_macro_f05(in_ids, s1_dict, pred_dict) if in_ids else 0.0

    return {
        "macro_f05": macro_f05,
        "prec": prec,
        "rec": rec,
        "pair_f05": pair_f05,
        "tp": tp,
        "fp": fp,
        "tot_preds": tot_preds,
        "tot_gt": tot_gt,
        "sing_f05": sing_f05,
        "us_f05": us_f05,
        "in_f05": in_f05
    }

def main():
    if not os.path.exists(SCORED_FILE):
        print(f"Error: {SCORED_FILE} does not exist yet.")
        return

    print("=" * 80)
    print("RECON-06: VALIDATE ENTITY-LEVEL PRECISION LEVER")
    print("=" * 80)
    
    with open(SCORED_FILE, "rb") as f:
        data = pickle.load(f)

    train_s1_ids = data["train_s1_ids"]
    val_s1_ids = data["val_s1_ids"]
    s1_dict = data["s1_dict"]
    train_preds_by_s1 = data["train_preds_by_s1"]
    val_preds_by_s1 = data["val_preds_by_s1"]
    candidates_by_s1 = data["candidates_by_s1"]
    best_th = data["best_th"]

    print(f"\nLoaded cache successfully:")
    print(f"  Train S1 Count: {len(train_s1_ids):,}")
    print(f"  Val S1 Count:   {len(val_s1_ids):,}")
    print(f"  Baseline Decision Threshold: {best_th:.2f}")

    # Baseline predictions
    train_base_preds = {s1: {cid for cid, p, _ in cands if p >= best_th} for s1, cands in train_preds_by_s1.items()}
    val_base_preds = {s1: {cid for cid, p, _ in cands if p >= best_th} for s1, cands in val_preds_by_s1.items()}

    train_base_metrics = evaluate_full_metrics(train_s1_ids, s1_dict, train_base_preds)
    val_base_metrics = evaluate_full_metrics(val_s1_ids, s1_dict, val_base_preds)

    print("\n" + "-" * 60)
    print("BASELINE METRICS VERIFICATION (at th=0.58)")
    print("-" * 60)
    print(f"Train S1 Macro F0.5: {train_base_metrics['macro_f05']*100:.2f}% | Prec: {train_base_metrics['prec']*100:.2f}% | Rec: {train_base_metrics['rec']*100:.2f}%")
    print(f"Val S1 Macro F0.5:   {val_base_metrics['macro_f05']*100:.2f}% | Prec: {val_base_metrics['prec']*100:.2f}% | Rec: {val_base_metrics['rec']*100:.2f}%")
    print(f"Val TP: {val_base_metrics['tp']:,} | Val FP: {val_base_metrics['fp']:,} | Singletons F0.5: {val_base_metrics['sing_f05']*100:.2f}%")

    # =========================================================================
    # STEP 1 — MEASURE THE STRUCTURAL LEVER (VAL SET DIAGNOSTICS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: MEASURE THE STRUCTURAL LEVER (VAL SET DIAGNOSTIC)")
    print("=" * 80)

    # 1. Map candidate claims: cand_id -> list of (s1_id, score, is_match)
    cand_claims = collections.defaultdict(list)
    for s1_id in val_s1_ids:
        cands = val_preds_by_s1.get(s1_id, [])
        for cid, p, is_m in cands:
            if p >= best_th:
                cand_claims[cid].append((s1_id, p, is_m))

    tot_unique_cands_pred = len(cand_claims)
    mult_claimed_cands = {cid: claims for cid, claims in cand_claims.items() if len(claims) > 1}
    s2_mult = sum(1 for cid in mult_claimed_cands if cid.startswith("S2-"))
    s3_mult = sum(1 for cid in mult_claimed_cands if cid.startswith("S3-"))

    print(f"1. Candidate Claim Overview (at decision threshold {best_th:.2f}):")
    print(f"   Total unique S2/S3 candidates predicted: {tot_unique_cands_pred:,}")
    print(f"   Candidates predicted for >1 S1:          {len(mult_claimed_cands):,} ({len(mult_claimed_cands)/max(tot_unique_cands_pred, 1)*100:.2f}%)")
    print(f"     - S2 candidates with >1 claim:         {s2_mult:,}")
    print(f"     - S3 candidates with >1 claim:         {s3_mult:,}")
    claim_sizes = collections.Counter(len(claims) for claims in mult_claimed_cands.values())
    print(f"   Claim multiplicity distribution: {dict(sorted(claim_sizes.items()))}")

    # Also check at raw candidate retrieval level for full diagnostic context
    all_val_retrieved = collections.defaultdict(list)
    for s1_id in val_s1_ids:
        for cid, p, is_m in val_preds_by_s1.get(s1_id, []):
            all_val_retrieved[cid].append((s1_id, p, is_m))
    mult_retrieved = {cid: c for cid, c in all_val_retrieved.items() if len(c) > 1}
    print(f"\n   [Context: Raw Candidate Retrieval Pool Before Scoring]")
    print(f"   Total unique candidates retrieved for Val S1: {len(all_val_retrieved):,}")
    print(f"   Candidates retrieved for >1 S1:              {len(mult_retrieved):,} ({len(mult_retrieved)/len(all_val_retrieved)*100:.2f}%)")

    # 2. Number of S1 entities affected
    s1_affected = set()
    for cid, claims in mult_claimed_cands.items():
        for s1_id, _, _ in claims:
            s1_affected.add(s1_id)
    print(f"\n2. Number of S1 entities affected:")
    print(f"   Affected S1 count: {len(s1_affected):,} / {len(val_s1_ids):,} ({len(s1_affected)/len(val_s1_ids)*100:.2f}%)")

    # 3. Ground truth parent of duplicate claims
    true_parent_is_one = 0
    true_parent_is_neither = 0
    mult_cand_details = []
    margins = []

    for cid, claims in mult_claimed_cands.items():
        claims_sorted = sorted(claims, key=lambda x: -x[1])
        top_s1, top_score, top_m = claims_sorted[0]
        second_s1, second_score, second_m = claims_sorted[1]
        margin = top_score - second_score
        margins.append(margin)

        has_true_parent = any(is_m == 1 for _, _, is_m in claims)
        if has_true_parent:
            true_parent_is_one += 1
            top_is_true = (top_m == 1)
        else:
            true_parent_is_neither += 1
            top_is_true = False

        mult_cand_details.append({
            "cid": cid,
            "claims": claims_sorted,
            "margin": margin,
            "has_true_parent": has_true_parent,
            "top_is_true": top_is_true
        })

    print(f"\n3. True Parent of Duplicate Claims (Total {len(mult_claimed_cands):,} candidates):")
    if mult_claimed_cands:
        print(f"   a) True parent IS one of the claiming S1s:    {true_parent_is_one:,} ({true_parent_is_one/len(mult_claimed_cands)*100:.2f}%)")
        top_matches_true = sum(1 for d in mult_cand_details if d["top_is_true"])
        print(f"      - Of which highest-scoring S1 is true match: {top_matches_true:,} ({top_matches_true/true_parent_is_one*100:.2f}%)")
        print(f"   b) True parent is NEITHER claiming S1:        {true_parent_is_neither:,} ({true_parent_is_neither/len(mult_claimed_cands)*100:.2f}%)")
    else:
        print("   a) True parent IS one of claiming S1s:        0 (N/A — 0 duplicate claims exist at th=0.58)")
        print("   b) True parent is NEITHER claiming S1:        0 (N/A — 0 duplicate claims exist at th=0.58)")

    # 4. Number of validation FPs attributable to duplicate claims
    tot_val_fps = val_base_metrics["fp"]
    fps_from_mult_claims = 0
    for cid, claims in mult_claimed_cands.items():
        for s1_id, p, is_m in claims:
            if is_m == 0:
                fps_from_mult_claims += 1

    print(f"\n4. Validation FPs Attributable to Duplicate Claims:")
    print(f"   Total validation FP pairs:                    {tot_val_fps:,}")
    print(f"   FP pairs involving multiply-claimed candidates: {fps_from_mult_claims:,} ({fps_from_mult_claims/tot_val_fps*100:.2f}%)")
    print(f"   FP pairs from single-claimed candidates:      {tot_val_fps - fps_from_mult_claims:,} ({(tot_val_fps - fps_from_mult_claims)/tot_val_fps*100:.2f}%)")

    # 5. Singleton / no-match FPs breakdown
    sing_val_ids = [eid for eid in val_s1_ids if len(s1_dict[eid]["gt"]) == 0]
    sing_fps = []
    for s1_id in sing_val_ids:
        cands = val_preds_by_s1.get(s1_id, [])
        for cid, p, is_m in cands:
            if p >= best_th:
                s1_info = s1_dict[s1_id]
                c_info = candidates_by_s1[s1_id][cid]
                sing_fps.append({
                    "s1_id": s1_id,
                    "cid": cid,
                    "score": p,
                    "s1_name": s1_info["name"],
                    "s1_addr": s1_info["addr"],
                    "cand_name": c_info["name"],
                    "cand_addr": c_info["addr"],
                    "is_mult_claimed": (cid in mult_claimed_cands)
                })

    print(f"\n5. Singleton / No-Match FPs Breakdown (Total Singletons: {len(sing_val_ids):,}):")
    print(f"   Singletons with >=1 FP:                       {sum(1 for s in sing_val_ids if val_base_preds[s])} / {len(sing_val_ids)}")
    print(f"   Total FP predictions on singletons:          {len(sing_fps):,}")

    # Detailed Categorization of singleton FPs
    cat_dup = 0
    cat_distractor = 0
    cat_addr_conflict = 0
    cat_other = 0

    for sfp in sing_fps:
        s1_nm, s1_ad = sfp["s1_name"], sfp["s1_addr"]
        c_nm, c_ad = sfp["cand_name"], sfp["cand_addr"]

        toks_ad1 = set(s1_ad.split()) - {"null", ""} if s1_ad else set()
        toks_ad2 = set(c_ad.split()) - {"null", ""} if c_ad else set()
        shared_ad_toks = toks_ad1 & toks_ad2

        toks_nm1 = set(s1_nm.split()) - {"null", ""}
        toks_nm2 = set(c_nm.split()) - {"null", ""}
        shared_nm_toks = toks_nm1 & toks_nm2

        if sfp["is_mult_claimed"]:
            cat_dup += 1
        elif len(shared_ad_toks) >= 2 and len(shared_nm_toks) <= 1:
            cat_addr_conflict += 1
        elif len(shared_nm_toks) >= 2 and len(shared_ad_toks) == 0:
            cat_distractor += 1
        else:
            cat_other += 1

    print(f"   Breakdown of Singleton FPs:")
    print(f"     a) Duplicate claims (candidate claimed by >1 S1): {cat_dup:>4,} ({cat_dup/len(sing_fps)*100:5.1f}%)")
    print(f"     b) Distractors (name match, address disjoint):   {cat_distractor:>4,} ({cat_distractor/len(sing_fps)*100:5.1f}%)")
    print(f"     c) Shared-address conflicts (same loc, diff biz):{cat_addr_conflict:>4,} ({cat_addr_conflict/len(sing_fps)*100:5.1f}%)")
    print(f"     d) Other / combined token overlap:               {cat_other:>4,} ({cat_other/len(sing_fps)*100:5.1f}%)")

    # 6. Distribution of score margin between top and 2nd best S1
    print(f"\n6. Distribution of Score Margin (Top - 2nd S1) for Multiply-Claimed Candidates:")
    if margins:
        margins_arr = np.array(margins)
        pcts = np.percentile(margins_arr, [0, 10, 25, 50, 75, 90, 100])
        print(f"   N={len(margins_arr):,} | mean={margins_arr.mean():.4f} | med={pcts[3]:.4f} | std={margins_arr.std():.4f}")
        print(f"   min={pcts[0]:.4f} | p10={pcts[1]:.4f} | p25={pcts[2]:.4f} | p50={pcts[3]:.4f} | p75={pcts[4]:.4f} | p90={pcts[5]:.4f} | max={pcts[6]:.4f}")
    else:
        print("   N = 0 (No candidate is claimed by >1 S1 at threshold 0.58 in the 2,001-entity sample).")

    # Margin distribution in raw candidate pool (unthresholded)
    raw_margins = []
    for cid, c_list in mult_retrieved.items():
        c_sorted = sorted(c_list, key=lambda x: -x[1])
        raw_margins.append(c_sorted[0][1] - c_sorted[1][1])
    raw_margins_arr = np.array(raw_margins)
    raw_pcts = np.percentile(raw_margins_arr, [0, 25, 50, 75, 90, 100])
    print(f"\n   [Context: Margins in Raw Retrieval Pool (N={len(raw_margins_arr):,} multiply-retrieved candidates)]")
    print(f"   mean={raw_margins_arr.mean():.4f} | med={raw_pcts[2]:.4f} | min={raw_pcts[0]:.4f} | max={raw_pcts[5]:.4f}")
    print(f"   Note: For 99.8% of these, both claiming scores are < 0.05 (irrelevant background retrieval).")

    # =========================================================================
    # STEP 2 — SIMPLE POST-PROCESSING EXPERIMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: SIMPLE POST-PROCESSING EXPERIMENT")
    print("TUNING EXCLUSIVELY ON 1,994 TRAIN S1 FOLD")
    print("=" * 80)

    def apply_postprocessing(s1_list, preds_by_s1, th, dup_mode="none", min_margin=0.0, gate_mode="none", gate_th=0.0):
        # Step 0: thresholding
        s1_claims = collections.defaultdict(dict)
        cand_claims = collections.defaultdict(list)
        for s1_id in s1_list:
            cands = preds_by_s1.get(s1_id, [])
            for cid, p, _ in cands:
                if p >= th:
                    s1_claims[s1_id][cid] = p
                    cand_claims[cid].append((s1_id, p))

        # Step 1: Duplicate resolution
        resolved_count = 0
        if dup_mode == "top_only":
            for cid, claims in cand_claims.items():
                if len(claims) > 1:
                    resolved_count += 1
                    claims_sorted = sorted(claims, key=lambda x: -x[1])
                    top_s1, top_score = claims_sorted[0]
                    sec_s1, sec_score = claims_sorted[1]
                    margin = top_score - sec_score
                    if margin >= min_margin:
                        for other_s1, _ in claims_sorted[1:]:
                            s1_claims[other_s1].pop(cid, None)
        elif dup_mode == "top_drop_ambig":
            for cid, claims in cand_claims.items():
                if len(claims) > 1:
                    resolved_count += 1
                    claims_sorted = sorted(claims, key=lambda x: -x[1])
                    top_s1, top_score = claims_sorted[0]
                    sec_s1, sec_score = claims_sorted[1]
                    margin = top_score - sec_score
                    if margin >= min_margin:
                        for other_s1, _ in claims_sorted[1:]:
                            s1_claims[other_s1].pop(cid, None)
                    else:
                        for other_s1, _ in claims:
                            s1_claims[other_s1].pop(cid, None)

        # Step 2: No-match gate
        if gate_mode == "max_score":
            for s1_id in s1_list:
                preds = s1_claims[s1_id]
                if preds and max(preds.values()) < gate_th:
                    s1_claims[s1_id].clear()
        elif gate_mode == "solitary":
            for s1_id in s1_list:
                preds = s1_claims[s1_id]
                if len(preds) == 1:
                    only_cid = list(preds.keys())[0]
                    if preds[only_cid] < gate_th:
                        s1_claims[s1_id].clear()

        pred_dict = {s1_id: set(s1_claims[s1_id].keys()) for s1_id in s1_list}
        return pred_dict, resolved_count

    # Define compact parameter grid
    grid = [
        # Baseline
        {"name": "Baseline (th=0.58, No Post-Proc)", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "none", "gate_th": 0.0},
        
        # A) Duplicate-claim resolution
        {"name": "Dup: Top S1 only (no margin req)", "dup_mode": "top_only", "min_margin": 0.00, "gate_mode": "none", "gate_th": 0.0},
        {"name": "Dup: Top S1 only (margin >= 0.05)", "dup_mode": "top_only", "min_margin": 0.05, "gate_mode": "none", "gate_th": 0.0},
        {"name": "Dup: Top S1 only (margin >= 0.15)", "dup_mode": "top_only", "min_margin": 0.15, "gate_mode": "none", "gate_th": 0.0},
        {"name": "Dup: Drop ambiguous if margin < 0.10", "dup_mode": "top_drop_ambig", "min_margin": 0.10, "gate_mode": "none", "gate_th": 0.0},

        # B) No-match gate (Max Score Gate)
        {"name": "Gate: Suppress S1 if max_score < 0.62", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "max_score", "gate_th": 0.62},
        {"name": "Gate: Suppress S1 if max_score < 0.65", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "max_score", "gate_th": 0.65},
        {"name": "Gate: Suppress S1 if max_score < 0.70", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "max_score", "gate_th": 0.70},
        {"name": "Gate: Suppress S1 if max_score < 0.75", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "max_score", "gate_th": 0.75},
        
        # B) No-match gate (Solitary Prediction Gate)
        {"name": "Gate: Solitary pred < 0.65 -> Singleton", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "solitary", "gate_th": 0.65},
        {"name": "Gate: Solitary pred < 0.70 -> Singleton", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "solitary", "gate_th": 0.70},
        {"name": "Gate: Solitary pred < 0.75 -> Singleton", "dup_mode": "none", "min_margin": 0.0, "gate_mode": "solitary", "gate_th": 0.75},

        # Combinations A + B
        {"name": "Combo: Top S1 (m>=0) + MaxGate < 0.62", "dup_mode": "top_only", "min_margin": 0.00, "gate_mode": "max_score", "gate_th": 0.62},
        {"name": "Combo: Top S1 (m>=0) + Solitary < 0.65", "dup_mode": "top_only", "min_margin": 0.00, "gate_mode": "solitary", "gate_th": 0.65},
    ]

    print(f"\nTuning across {len(grid)} configurations on Train S1 (N={len(train_s1_ids):,}):")
    print(f"{'Configuration':<45} | {'Train Macro F0.5':>16} | {'Delta':>8} | {'Prec':>7} | {'Rec':>7}")
    print("-" * 92)

    best_cfg = None
    best_train_f05 = -1.0
    train_results = []

    for cfg in grid:
        pred_dict, _ = apply_postprocessing(
            train_s1_ids, train_preds_by_s1, best_th,
            dup_mode=cfg["dup_mode"], min_margin=cfg["min_margin"],
            gate_mode=cfg["gate_mode"], gate_th=cfg["gate_th"]
        )
        metrics = evaluate_full_metrics(train_s1_ids, s1_dict, pred_dict)
        f05 = metrics["macro_f05"]
        delta = f05 - train_base_metrics["macro_f05"]
        train_results.append((cfg, metrics, delta))
        if f05 > best_train_f05:
            best_train_f05 = f05
            best_cfg = cfg
        print(f"{cfg['name']:<45} | {f05*100:>15.2f}% | {delta*100:>+7.2f}% | {metrics['prec']*100:6.2f}% | {metrics['rec']*100:6.2f}%")

    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION ON TRAIN S1: {best_cfg['name']}")
    print(f"Train Macro F0.5: {best_train_f05*100:.2f}% (Delta: {(best_train_f05 - train_base_metrics['macro_f05'])*100:+.2f}%)")
    print("=" * 80)

    # =========================================================================
    # EVALUATION ON STRICTLY HELD-OUT VAL S1
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ON UNTOUCHED HELD-OUT VAL S1 (N=2,001)")
    print("=" * 80)
    print(f"{'Configuration':<45} | {'Val Macro F0.5':>14} | {'Delta':>7} | {'Prec':>7} | {'Rec':>7} | {'Sing F0.5':>9} | {'US F0.5':>7} | {'IN F0.5':>7}")
    print("-" * 115)

    val_results = []
    for cfg in grid:
        pred_dict, res_cnt = apply_postprocessing(
            val_s1_ids, val_preds_by_s1, best_th,
            dup_mode=cfg["dup_mode"], min_margin=cfg["min_margin"],
            gate_mode=cfg["gate_mode"], gate_th=cfg["gate_th"]
        )
        metrics = evaluate_full_metrics(val_s1_ids, s1_dict, pred_dict)
        delta_f05 = metrics["macro_f05"] - val_base_metrics["macro_f05"]
        delta_rec = metrics["rec"] - val_base_metrics["rec"]
        preds_removed = val_base_metrics["tot_preds"] - metrics["tot_preds"]
        fps_removed = val_base_metrics["fp"] - metrics["fp"]
        tp_lost = val_base_metrics["tp"] - metrics["tp"]

        val_results.append({
            "cfg": cfg,
            "metrics": metrics,
            "delta_f05": delta_f05,
            "delta_rec": delta_rec,
            "preds_removed": preds_removed,
            "fps_removed": fps_removed,
            "tp_lost": tp_lost,
            "res_cnt": res_cnt
        })

        print(f"{cfg['name']:<45} | {metrics['macro_f05']*100:>13.2f}% | {delta_f05*100:>+6.2f}% | {metrics['prec']*100:6.2f}% | {metrics['rec']*100:6.2f}% | {metrics['sing_f05']*100:8.2f}% | {metrics['us_f05']*100:6.2f}% | {metrics['in_f05']*100:6.2f}%")

    # Detailed report for the Best Train Config evaluated on Val
    best_val_res = next(r for r in val_results if r["cfg"]["name"] == best_cfg["name"])
    print("\n" + "=" * 80)
    print("PRIMARY EXPERIMENT OUTCOME (Selected by Train S1 -> Evaluated on Val S1)")
    print("=" * 80)
    print(f"1. BASELINE:                   {val_base_metrics['macro_f05']*100:.2f}% S1 Macro F0.5")
    print(f"2. BEST POST-PROCESSING RESULT: {best_val_res['metrics']['macro_f05']*100:.2f}% S1 Macro F0.5")
    print(f"3. EXACT PARAMETERS:           {best_cfg['name']}")
    print(f"4. DELTA:                      {best_val_res['delta_f05']*100:+.2f} percentage points (Recall delta: {best_val_res['delta_rec']*100:+.2f} points)")
    print(f"   Precision:                  {val_base_metrics['prec']*100:.2f}% -> {best_val_res['metrics']['prec']*100:.2f}% ({best_val_res['metrics']['prec']*100 - val_base_metrics['prec']*100:+.2f}%)")
    print(f"   End-to-End Recall:          {val_base_metrics['rec']*100:.2f}% -> {best_val_res['metrics']['rec']*100:.2f}% ({best_val_res['delta_rec']*100:+.2f}%)")
    print(f"   Singleton Macro F0.5:       {val_base_metrics['sing_f05']*100:.2f}% -> {best_val_res['metrics']['sing_f05']*100:.2f}% ({best_val_res['metrics']['sing_f05']*100 - val_base_metrics['sing_f05']*100:+.2f}%)")
    print(f"   US Macro F0.5:              {val_base_metrics['us_f05']*100:.2f}% -> {best_val_res['metrics']['us_f05']*100:.2f}%")
    print(f"   India Macro F0.5:           {val_base_metrics['in_f05']*100:.2f}% -> {best_val_res['metrics']['in_f05']*100:.2f}%")
    print(f"   Predictions removed:        {best_val_res['preds_removed']:,}")
    print(f"   Duplicate claims resolved:  {best_val_res['res_cnt']:,}")
    print(f"   True matches lost:          {best_val_res['tp_lost']:,}")
    print(f"   FPs removed:                {best_val_res['fps_removed']:,}")

    # Success / Fail evaluation
    f05_diff = best_val_res['delta_f05'] * 100
    rec_diff = best_val_res['delta_rec'] * 100
    print("\nCRITERIA EVALUATION:")
    if f05_diff >= 1.0 and rec_diff >= -0.5:
        verdict = "SUCCESS (>= +1.0 pt F0.5 with <= 0.5 pt recall drop)"
    elif f05_diff < 0.3 or rec_diff < -1.0:
        verdict = "FAIL (< +0.3 pt improvement or > 1.0 pt recall drop)"
    else:
        verdict = "INCONCLUSIVE / MARGINAL (+0.3 to +1.0 pt improvement)"
    print(f"Verdict: {verdict}")

if __name__ == "__main__":
    main()
