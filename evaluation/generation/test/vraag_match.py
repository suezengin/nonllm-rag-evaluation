import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from tqdm import tqdm
from utils.embedding import embed_text
from utils.similarity import (
    compute_cosine_similarity,
    compute_jaccard_similarity,
    compute_edit_distance,
)
from utils.advanced_metrics import compute_rouge_all

load_dotenv(dotenv_path=".env")
# Azure client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("ENDPOINT_URL")
)
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
# Load data
prefix = ("../../data/intermediate/")
query_df = pd.read_excel(prefix + "query_data_ready.xlsx")
inboedel_gt = pd.read_excel(prefix + "groundtruth_inboedel_enriched.xlsx")
reis_gt = pd.read_excel(prefix + "groundtruth_reis_enriched.xlsx")
# Normalize source in GT files
inboedel_gt["source"] = inboedel_gt["source"].apply(normalize_source_name)
reis_gt["source"] = reis_gt["source"].apply(normalize_source_name)
# Embedding cache
embedding_cache = {}
 
def embed_with_cache(text):
    if text in embedding_cache:
        return embedding_cache[text]
    emb = embed_text(text, client)
    embedding_cache[text] = emb
    return emb
 
# Main loop
results = []
for idx, row in tqdm(query_df.iterrows(), total=len(query_df)):
    question = row["question"]
    product = str(row["product"]).lower().strip()
    polis_versie = str(row["polis_versie"]).lower().strip()
    dekking = str(row["dekking"]).lower().strip()
    type_klant = str(row["type_klant"]).lower().strip()
 
    if "inboedel" in product:
        gt_df = inboedel_gt.copy()
    elif "doorlopende reisverzekering" in product:
        gt_df = reis_gt.copy()
    else:
        results.append({})
        continue

    for col in ["polis_versie", "type_klant", "dekking"] : 
        gt_df[col] = gt_df[col].astype(str)
 
    # Filter GT by metadata
    filtered_gt = gt_df[
        (gt_df["polis_versie"].str.lower().str.strip() == polis_versie) &
        (gt_df["dekking"].str.lower().str.strip() == dekking) &
        (gt_df["type_klant"].str.lower().str.strip() == type_klant)
    ].copy()
 
    if filtered_gt.empty:
        results.append({})
        continue
 
    question_emb = embed_with_cache(question)
    filtered_gt["vraag_emb"] = filtered_gt["vraag"].apply(embed_with_cache)
    filtered_gt["cosine_sim"] = filtered_gt["vraag_emb"].apply(lambda emb: compute_cosine_similarity(question_emb, emb))
 
    best_row = filtered_gt.loc[filtered_gt["cosine_sim"].idxmax()]
    vraag_matched = best_row["vraag"]
    cos_sim = best_row["cosine_sim"]
    rouge_scores = compute_rouge_all(vraag_matched, question)
    jaccard = compute_jaccard_similarity(vraag_matched, question)
    edit_dist = compute_edit_distance(vraag_matched, question)
 
    aegon = filtered_gt[(filtered_gt["vraag"] == vraag_matched) & (filtered_gt["source"].str.startswith("aegon"))]
    asr = filtered_gt[(filtered_gt["vraag"] == vraag_matched) & (filtered_gt["source"].str.startswith("asr"))]
 
    result_row = {
        "vraag_matched": vraag_matched,
        "cosine_sim": cos_sim,
        "rouge_1": rouge_scores["rouge-1"],
        "rouge_2": rouge_scores["rouge-2"],
        "rouge_l": rouge_scores["rouge-l"],
        "jaccard_sim": jaccard,
        "edit_distance": edit_dist,
    }
  
    if not aegon.empty:
        aegon_row = aegon.iloc[0]

        result_row.update({
            "answer_aegon": aegon_row["antwoord"],
            "artikelnummer_aegon": aegon_row["artikelnummer"],
            "source_aegon": aegon_row["source"],
        })
    else:
        result_row.update({
            "answer_aegon": "", "artikelnummer_aegon": "", "source_aegon": ""
        })
    if not asr.empty:
        asr_row = asr.iloc[0]

        result_row.update({
            "answer_asr": asr_row["antwoord"],
            "artikelnummer_asr": asr_row["artikelnummer"],
            "source_asr": asr_row["source"],
        })
    else:

        result_row.update({
            "answer_asr": "", "artikelnummer_asr": "", "source_asr": ""
        })
     
 
    results.append(result_row)
 
# Merge with original query dataframe
results_df = pd.DataFrame(results)

print(results_df.head())


final_df = pd.concat([query_df, results_df], axis=1)

selected_columns = [
    'question', 'product', 'polis_versie', 'type_klant', 'dekking',
    'answer_aegon', 'answer_asr', 'comparison', 'citatie_aegon', 'citatie_asr',
    'artikelnummer_aegon', 'artikelnummer_asr',
    'vraag_matched', 'cosine_sim', 'source_aegon', 'source_asr'
]
 
final_df = final_df[selected_columns]

 
# Save and show
output_path = "../../data/intermediate/user_queries_matched_ground_truth_with_metrics.xlsx"
final_df.to_excel(output_path, index=False)


