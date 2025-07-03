import pandas as pd

def compute_all_retrieval_metrics_dataframe(df, source="asr", k=3):
    assert source in ["asr", "aegon"]
    match_col = f"match_type_{source}"
    justify_col = f"justified_by_kb_{source}"
    gt_col = f"gt_article_ids_{source}"
    retrieved_col = f"top_{k}_{source}"

    rows = []
    for _, row in df.iterrows():
        out = {}
        retrieved = row.get(retrieved_col, [])
        gt_ids = row.get(gt_col, [])
        if not isinstance(retrieved, list) or not isinstance(gt_ids, list) or len(gt_ids) == 0:
            # Fill NA metrics
            out["precision_at_k"] = None
            out["recall_at_k"] = None
            out["f1_at_k"] = None
            out["accuracy_at_k"] = None
        else:
            gt_set = set(gt_ids)
            top_k = retrieved[:k]
            relevant = gt_set.intersection(top_k)

            prec = len(relevant) / k if k > 0 else 0
            rec = len(relevant) / len(gt_set) if len(gt_set) > 0 else 0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            acc = 1 if len(relevant) > 0 else 0

            out["precision_at_k"] = prec
            out["recall_at_k"] = rec
            out["f1_at_k"] = f1
            out["accuracy_at_k"] = acc

        # Add extra per-row info
        out["match_type"] = row.get(match_col, None)
        out["justified_by_kb"] = row.get(justify_col, None)
        rows.append(out)

    metrics_df = pd.DataFrame(rows)
    metrics_df.columns = [f"{source}_{col}" for col in metrics_df.columns]
    return metrics_df
