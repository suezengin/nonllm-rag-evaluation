import pandas as pd
from tabulate import tabulate
from compute_retrieval_metrics import run_all_metrics_dual
from compute_retrieval_metrics_row import compute_all_retrieval_metrics_dataframe

# File paths
input_path = "../../data/processed_eval_data/retrieval_eval_results.pkl"
output_path = "../../data/outputs/retrieval_metric_results.xlsx"
output2_path = "../../data/outputs/retrieval_metric_results_by_row.xlsx"

# Read data
df = pd.read_pickle(input_path)

# Compute aggregate metrics
results_df = run_all_metrics_dual(df, filter_by_gt_quality=True)

# Show table
print(tabulate(results_df, headers="keys", tablefmt="psql"))

# Save aggregate metrics
results_df.to_excel(output_path, index=False)
print(f"Retrieval metrics saved to: {output_path}")

# Compute per-row metrics as DataFrames
asr_metrics_df = compute_all_retrieval_metrics_dataframe(df, source="asr", k=3)
asr_metrics_df.columns = [f"asr_{col}" for col in asr_metrics_df.columns]

aegon_metrics_df = compute_all_retrieval_metrics_dataframe(df, source="aegon", k=3)
aegon_metrics_df.columns = [f"aegon_{col}" for col in aegon_metrics_df.columns]

# Merge into original DataFrame
df_combined = pd.concat([df, asr_metrics_df, aegon_metrics_df], axis=1)

# Save row-level metrics
df_combined.to_excel(output2_path, index=False)
print(f"Retrieval metrics by row saved to: {output2_path}")

