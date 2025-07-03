import pandas as pd


# Retrieval Metrics Computation Functions


def is_extended_match(match_type):
    return match_type in ["exact", "prefix"]
 

def compute_row_metrics_for_source(df, source="asr", k=3):
    assert source in ["asr", "aegon"]
    match_col = f"match_type_{source}"
    justify_col = f"justified_by_kb_{source}"
    gt_col = f"gt_article_ids_{source}"
    retrieved_col = f"top_{k}_{source}"
 
    total_queries = len(df)
 
    exact_count = (df[match_col] == "exact").sum()
    extended_count = df[match_col].apply(is_extended_match).sum()
    coverage_count = (df[match_col] == "no_match").sum()
 
    misleading_exact = df[(df[match_col] == "exact") & (~df[justify_col])].shape[0]
    misleading_extended = df[df[match_col].apply(is_extended_match) & (~df[justify_col])].shape[0]
    misleading_exact_rate = (misleading_exact / exact_count * 100) if exact_count > 0 else 0.0
    misleading_extended_rate = (misleading_extended / extended_count * 100) if extended_count > 0 else 0.0
    misleading_coverage_count = df[(df[match_col] == "no_match") & (~df[justify_col])].shape[0]
    misleading_coverage_rate = misleading_coverage_count / total_queries * 100
    justified_total = df[justify_col].sum()
    justification_rate = justified_total / total_queries * 100
 
    mrr_total, map_total = 0.0, 0.0
    hit_at_k = 0
    hit_at_1 = 0
 
    precision_at_k_numerator = 0
    recall_at_k_numerator = 0
    precision_at_1_numerator = 0
    recall_at_1_numerator = 0
 
    f1_at_k_sum = 0
 
    for _, row in df.iterrows():
        retrieved = row.get(retrieved_col, [])
        gt_ids = row.get(gt_col, [])
        if not isinstance(retrieved, list) or not isinstance(gt_ids, list) or len(gt_ids) == 0:
            continue
 
        gt_set = set(gt_ids)
        top_k = retrieved[:k]
        top_1 = retrieved[:1]
 
        relevant_in_top_k = set(top_k) & gt_set
        relevant_in_top_1 = set(top_1) & gt_set
 
        # Precision@k
        precision_at_k_numerator += len(relevant_in_top_k)
        # Recall@k
        recall_at_k_numerator += len(relevant_in_top_k)
 
        # Precision@1
        precision_at_1_numerator += len(relevant_in_top_1)
        # Recall@1
        recall_at_1_numerator += len(relevant_in_top_1)
 
        # F1@k per query
        prec = len(relevant_in_top_k) / k if k > 0 else 0
        rec = len(relevant_in_top_k) / len(gt_set) if len(gt_set) > 0 else 0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        f1_at_k_sum += f1
 
        # Accuracy@k
        if len(relevant_in_top_k) > 0:
            hit_at_k += 1
        if len(relevant_in_top_1) > 0:
            hit_at_1 += 1
 
        # MRR and MAP
        first_hit_found = False
        ap_sum = 0.0
        num_hits = 0
        for idx, item in enumerate(top_k):
            if item in gt_set:
                if not first_hit_found:
                    mrr_total += 1.0 / (idx + 1)
                    first_hit_found = True
                num_hits += 1
                ap_sum += num_hits / (idx + 1)
        if len(gt_set) > 0:
            map_total += ap_sum / len(gt_set)
 
    # Final metrics
    precision_at_k = (precision_at_k_numerator / (total_queries * k)) * 100 if total_queries > 0 else 0.0
    recall_at_k = (recall_at_k_numerator / (len(df) * len(gt_set))) * 100 if total_queries > 0 else 0.0  # careful
    recall_at_k = (recall_at_k_numerator / recall_at_k_numerator if recall_at_k_numerator > 0 else 1) * 100 if total_queries > 0 else 0.0
    recall_at_k = (recall_at_k_numerator / sum(len(set(row.get(gt_col, []))) for _, row in df.iterrows() if isinstance(row.get(gt_col, []), list) and len(row.get(gt_col, [])) > 0)) * 100
 
    recall_at_1 = (recall_at_1_numerator / sum(len(set(row.get(gt_col, []))) for _, row in df.iterrows() if isinstance(row.get(gt_col, []), list) and len(row.get(gt_col, [])) > 0)) * 100
 
    precision_at_1 = (precision_at_1_numerator / total_queries) * 100 if total_queries > 0 else 0.0
 
    accuracy_at_k = (hit_at_k / total_queries) * 100 if total_queries > 0 else 0.0
    accuracy_at_1 = (hit_at_1 / total_queries) * 100 if total_queries > 0 else 0.0
 
    f1_at_k = (f1_at_k_sum / total_queries) * 100 if total_queries > 0 else 0.0
    mrr_score = (mrr_total / total_queries) * 100 if total_queries > 0 else 0.0
    map_score = (map_total / total_queries) * 100 if total_queries > 0 else 0.0
 
    metrics = {
        "Retrieval Accuracy (Exact)": (exact_count, exact_count / total_queries * 100),
        "Extended Accuracy (Exact + Prefix)": (extended_count, extended_count / total_queries * 100),
        "Accuracy@1": (hit_at_1, accuracy_at_1),
        "Recall@1": (recall_at_1_numerator, recall_at_1),
        "Precision@1": (precision_at_1_numerator, precision_at_1),
        f"Accuracy@{k}": (hit_at_k, accuracy_at_k),
        f"Recall@{k}": (recall_at_k_numerator, recall_at_k),
        f"Precision@{k}": (precision_at_k_numerator, precision_at_k),
        f"F1@{k}": (f1_at_k_sum, f1_at_k),
        "MRR": (round(mrr_total, 2), mrr_score),
        "MAP": (round(map_total, 2), map_score),
        "Coverage (No Match)": (coverage_count, coverage_count / total_queries * 100),
        "Misleading Rate (Exact Not Justified)": (misleading_exact, misleading_exact / total_queries * 100),
        "Extended Misleading Rate": (misleading_extended, misleading_extended / total_queries * 100),
        "False Positive Rate (Exact)": (misleading_exact, misleading_exact_rate),
        "False Positive Rate (Extended)": (misleading_extended, misleading_extended_rate),
        "Justification Rate (All Matches)": (justified_total, justification_rate),
        "Misleading Coverage": (misleading_coverage_count, misleading_coverage_rate),
    }
    return metrics

def run_all_metrics_dual(df, filter_by_gt_quality, k=3):


    if filter_by_gt_quality:
        df = df[
            (df["gt_quality_issue_asr"] != True) &
            (df["gt_quality_issue_aegon"] != True)
        ].copy()

    metrics_asr = compute_row_metrics_for_source(df, source="asr", k=k)
    metrics_aegon = compute_row_metrics_for_source(df, source="aegon", k=k)
 
    rows = []
    for metric_name in metrics_asr:
        asr_count, asr_pct = metrics_asr[metric_name]
        aegon_count, aegon_pct = metrics_aegon[metric_name]
        rows.append({
            "Metric": metric_name,
            "ASR Count": asr_count,
            "ASR %": f"{asr_pct:.2f}%",
            "AEGON Count": aegon_count,
            "AEGON %": f"{aegon_pct:.2f}%"
        })
 
    return pd.DataFrame(rows)

print("running updated function")



