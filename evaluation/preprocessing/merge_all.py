import pandas as pd
import os
 
#Normalize source names
def normalize_source_name(source: str) -> str:
    return (
        str(source)
        .lower()
        .replace("a.s.r.", "asr")
        .replace("ik_kies_zelf_(dr_2018)", "asr_ikz_2018")
        .replace("a.s.r", "asr")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .strip()
    )
 
# Load preprocessed parsed responses and ground truths
prefix = ("../../data/intermediate/")
inboedel_parsed = pd.read_excel(prefix + "parsed_inboedel_responses.xlsx")
reis_parsed = pd.read_excel(prefix + "parsed_reis_responses.xlsx")
inboedel_gt = pd.read_excel(prefix + "groundtruth_inboedel_enriched.xlsx")
reis_gt = pd.read_excel(prefix + "groundtruth_reis_enriched.xlsx")
 
# Normalize source in GT files
inboedel_gt["source"] = inboedel_gt["source"].apply(normalize_source_name)
reis_gt["source"] = reis_gt["source"].apply(normalize_source_name)
 
# Add empty polis_versie if not exists
if "polis_versie" not in reis_parsed.columns:
    reis_parsed["polis_versie"] = ""
 
# Merge all parsed and GT together
parsed_all = pd.concat([inboedel_parsed, reis_parsed], ignore_index=True)
gt_all = pd.concat([inboedel_gt, reis_gt], ignore_index=True)
 
# Normalize 'product' column to ensure consistent merge keys
parsed_all["product"] = parsed_all["product"].str.strip().str.lower()
gt_all["product"] = gt_all["product"].str.strip().str.lower()
 
# Separate AEGON and ASR GTs after normalization
aegon_gt = gt_all[gt_all["source"].str.startswith("aegon")].copy()
asr_gt = gt_all[gt_all["source"].str.startswith("asr_")].copy()
 
# Remove duplicates to avoid merge conflicts
aegon_gt = aegon_gt.drop_duplicates(subset=["vraag", "product", "dekking", "polis_versie"])
asr_gt = asr_gt.drop_duplicates(subset=["vraag", "product", "dekking", "type_klant"])
 
# Split parsed responses
inboedel_df = parsed_all[parsed_all["product"].str.contains("inboedel")].copy()
reis_df = parsed_all[parsed_all["product"].str.contains("reis")].copy()
 
# Merge AEGON

aegon_inboedel_merged = inboedel_df.merge(
    aegon_gt[["vraag", "product", "dekking", "polis_versie", "gold_article_id", "gold_answer", "source"]],
    how="left",
    left_on=["question", "product", "dekking", "polis_versie"],
    right_on=["vraag", "product", "dekking", "polis_versie"]
).rename(columns={
    "gold_article_id": "aegon_gold_article_id",
    "gold_answer": "aegon_gold_answer",
    "source": "source_gt_aegon"
})

aegon_reis_merged = reis_df.merge(
    aegon_gt[["vraag", "product", "dekking", "gold_article_id", "gold_answer", "source"]],
    how="left",
    left_on=["question", "product", "dekking"],
    right_on=["vraag", "product", "dekking"]
).rename(columns={
    "gold_article_id": "aegon_gold_article_id",
    "gold_answer": "aegon_gold_answer",
    "source": "source_gt_aegon"
})

aegon_combined = pd.concat([aegon_inboedel_merged, aegon_reis_merged], ignore_index=True)

# ASR Merge

asr_inboedel_merged = aegon_combined[aegon_combined["product"].str.contains("inboedel")].merge(
    asr_gt[["vraag", "product", "dekking", "type_klant", "gold_article_id", "gold_answer", "source"]],
    how="left",
    left_on=["question", "product", "dekking", "type_klant"],
    right_on=["vraag", "product", "dekking", "type_klant"]
).rename(columns={
    "gold_article_id": "asr_gold_article_id",
    "gold_answer": "asr_gold_answer",
    "source": "source_gt_asr"
})

asr_reis_merged = aegon_combined[aegon_combined["product"].str.contains("reis")].merge(
    asr_gt[["vraag", "product", "type_klant", "gold_article_id", "gold_answer", "source"]],
    how="left",
    left_on=["question", "product", "type_klant"],
    right_on=["vraag", "product", "type_klant"]
).rename(columns={
    "gold_article_id": "asr_gold_article_id",
    "gold_answer": "asr_gold_answer",
    "source": "source_gt_asr"
})

final_combined = pd.concat([asr_inboedel_merged, asr_reis_merged], ignore_index=True)

# Final cleanup
final_combined.drop(columns=[col for col in final_combined.columns if col.startswith("vraag")], inplace=True)

# Save
output_path = prefix + "merged_all.xlsx"
final_combined.to_excel(output_path, index=False)
output_path

print("Done, files saved")