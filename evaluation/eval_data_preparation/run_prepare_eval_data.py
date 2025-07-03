
import pandas as pd
from eval_data_utils import (
    preprocess_article_ids_fixed,
    classify_match,
    determine_kb_source_aegon,
    determine_kb_source_asr,
    is_justified_by_kb,
    has_gt_quality_issue
)
 
# Load input files

# File paths
input_path = "../../data/intermediate/merged_all_selected_articles.xlsx"
kb_path = "../../data/intermediate/all_kbs_combined.xlsx"

output_path = "../../data/processed_eval_data/retrieval_eval_results.xlsx"
output_path_pkl =  "../../data/processed_eval_data/retrieval_eval_results.pkl"


df = pd.read_excel(input_path)
kb_df = pd.read_excel(kb_path)
 
# Preprocess article IDs
df = preprocess_article_ids_fixed(df)

df["gt_quality_issue_aegon"] = df["gt_article_ids_aegon"].apply(has_gt_quality_issue)
df["gt_quality_issue_asr"] = df["gt_article_ids_asr"].apply(has_gt_quality_issue)
 
# Determine KB source
df["kb_source_aegon"] = df.apply(
    lambda row: determine_kb_source_aegon(row["polis_versie"], row["product"] ),
    axis=1
)

df["kb_source_asr"] = df.apply(
    lambda row: determine_kb_source_asr(row["source_gt_asr"]),
    axis=1
)

 
# Classification
 
df["match_type_aegon"] = df.apply(
    lambda row: classify_match(row["system_article_ids_aegon"], row["gt_article_ids_aegon"]), axis=1
)
df["match_type_asr"] = df.apply(
    lambda row: classify_match(row["system_article_ids_asr"], row["gt_article_ids_asr"]), axis=1
)
 
# KB Justification
 
df["justified_by_kb_aegon"] = df.apply(
    lambda row: is_justified_by_kb(row["system_article_ids_aegon"], row["gt_article_ids_aegon"], kb_df, row["kb_source_aegon"]),
    axis=1
)

df["justified_by_kb_asr"] = df.apply(
    lambda row: is_justified_by_kb(row["system_article_ids_asr"], row["gt_article_ids_asr"], kb_df, row["kb_source_asr"]),
    axis=1
)

 
# Save output

df.to_excel(output_path, index=False)
df.to_pickle(output_path_pkl)
print("Retrieval evaluation preparation complete.")