import pandas as pd
import os
 
 
# Load preprocessed parsed responses and ground truths
prefix = ("../../data/inputs/")
merged_pre = ("../../data/intermediate/")
merged_all = pd.read_excel(merged_pre + "merged_all.xlsx")
inboedel_topk = pd.read_excel(prefix + "inboedel_questions_selected_articles.xlsx")
reis_topk = pd.read_excel(prefix + "reis_questions_selected_articles.xlsx")
 

# Normalize 'product' column to ensure consistent merge keys
merged_all["product"] = merged_all["product"].str.strip().str.lower()
reis_topk["product"] = reis_topk["product"].str.strip().str.lower()
inboedel_topk["product"] = inboedel_topk["product"].str.strip().str.lower()

# Remove duplicates to avoid merge conflicts
inboedel_topk = inboedel_topk.drop_duplicates(subset=["question", "product", "dekking", "type_klant", "polis_versie"])
reis_topk = reis_topk.drop_duplicates(subset=["question", "product", "dekking", "type_klant"])
 
# Split merge all
inboedel_df = merged_all[merged_all["product"].str.contains("inboedel")].copy()
reis_df = merged_all[merged_all["product"].str.contains("reis")].copy()
 

inboedel_merged = inboedel_df.merge(
    inboedel_topk[["question", "product", "dekking", "polis_versie", "type_klant", "selected_articles_aegon", "selected_articles_asr", "top_3_aegon","top_3_asr"]],
    how="left",
    left_on=["question", "product", "dekking", "polis_versie", "type_klant"],
    right_on=["question", "product", "dekking", "polis_versie", "type_klant"]
)

reis_merged = reis_df.merge(
    reis_topk[["question", "product", "dekking", "type_klant", "selected_articles_aegon", "selected_articles_asr", "top_3_aegon","top_3_asr"]],
    how="left",
    left_on=["question", "product", "dekking", "type_klant"],
    right_on=["question", "product", "dekking", "type_klant"]
)


combined = pd.concat([inboedel_merged, reis_merged], ignore_index=True)


# Save
output_path = merged_pre + "merged_all_selected_articles.xlsx"
combined.to_excel(output_path, index=False)
output_path

print("Done, files saved")