import pandas as pd
from tabulate import tabulate
from compute_retrieval_metrics import run_all_metrics_dual
from compute_retrieval_metrics_row import compute_all_retrieval_metrics_dataframe

# File paths
input_path = "../../data/processed_eval_data/queries_to_use_self_labeled.xlsx"
output_path = "../../data/outputs/retrieval_metric_test_results.xlsx"
output2_path = "../../data/outputs/retrieval_metric_test_results_by_row.xlsx"

df = pd.read_excel(input_path)

import ast

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

for col in ["top_3_asr", "top_3_aegon", "gt_article_ids_asr", "gt_article_ids_aegon"]:
    df[col] = df[col].apply(safe_eval)

for col in ["top_3_asr", "gt_article_ids_asr"]:
    print(f"{col} first row:", df[col].iloc[0], " type:", type(df[col].iloc[0]))

# Aggregate metrics
results_df = run_all_metrics_dual(df, filter_by_gt_quality=True)

# PostgreSQL-style print
print(tabulate(results_df, headers="keys", tablefmt="psql"))

# Save aggregate metrics
results_df.to_excel(output_path, index=False)
print(f"Retrieval metrics saved to: {output_path}")

# Per-row metrics DataFrame
row_metrics_df = compute_all_retrieval_metrics_dataframe(df, k=3)

# Merge per-row metrics into original DataFrame
df_combined = pd.concat([df, row_metrics_df], axis=1)

# Save per-row metrics
df_combined.to_excel(output2_path, index=False)
print(f"Retrieval metrics by row saved to: {output2_path}")

