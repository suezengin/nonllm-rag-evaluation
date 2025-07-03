import os
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from compute_generation_metrics import compute_generation_metrics
from pathlib import Path


load_dotenv(dotenv_path=".env")
 
# Azure client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("ENDPOINT_URL")
)
 

# File paths
input_path = "../../data/processed_eval_data/retrieval_eval_results.pkl"
output_path = "../../data/outputs/generation_metric_results.xlsx"
 
# --- Read Data ---
df = pd.read_pickle(input_path)

# Compute metrics for ASR
asr_metrics = compute_generation_metrics(df, source="asr", client=client)
asr_metrics.columns = [f"asr_{col}" for col in asr_metrics.columns]
asr_metrics.index = df.index  # align by row index
 
# Compute metrics for AEGON
aegon_metrics = compute_generation_metrics(df, source="aegon", client=client)
aegon_metrics.columns = [f"aegon_{col}" for col in aegon_metrics.columns]
aegon_metrics.index = df.index  # align by row index
 
# Merge into original DataFrame
df_combined = pd.concat([df, asr_metrics, aegon_metrics], axis=1)
 

# Save to Excel
df_combined.to_excel(output_path, index=False)
print(f"Generation metrics saved to: {output_path}")