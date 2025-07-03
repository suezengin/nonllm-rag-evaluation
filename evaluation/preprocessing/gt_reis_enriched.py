import pandas as pd

prefix = ("../../data/inputs/")
file_path = prefix + "Groundtruth_Reis.xlsx"
df_raw_reis= pd.read_excel(file_path, sheet_name="Blad1", header =None)

header_row_reis = df_raw_reis.iloc[0]
df_data_reis = df_raw_reis.iloc[2:].copy().reset_index(drop=True)
 
# Detect blocks
blocks_reis = []
col_count = df_data_reis.shape[1]
col_idx = 2
 
while col_idx + 1 < col_count:
    product_name = str(header_row_reis[col_idx]).strip().lower().replace(" ", "_")
    verzekerd_check = str(header_row_reis[col_idx + 1]).lower().replace("?", "").strip()
    if product_name == "" or "verzekerd" not in verzekerd_check:
        col_idx += 2
        continue
    blocks_reis.append({
        "vraag_col": 1,
        "artikel_col": col_idx,
        "verzekerd_col": col_idx + 1,
        "source": product_name
    })
    col_idx += 2
 
# Metadata enrichment function
def parse_reis_source(source: str):
    source = source.lower()
    if "aegon_basis" in source:
        return {
            "product": "Doorlopende Reisverzekering",
            "polis_versie": None,
            "type_klant": None,
            "dekking": "Basis",
        }
    elif "aegon_allrisk" in source:
        return {
            "product": "Doorlopende Reisverzekering",
            "polis_versie": None,
            "type_klant": None,
            "dekking": "Allrisk",
        }
    elif "a.s.r._vp_dr_2024" in source:
        return {
            "product": "Doorlopende Reisverzekering",
            "polis_versie": None,
            "type_klant": "Adviseur",
            "dekking": None,
        }
    elif "ik_kies_zelf_(dr_2018)" in source:
        return {
            "product": "Doorlopende Reisverzekering",
            "polis_versie": None,
            "type_klant": "Direct",
            "dekking": None,
        }
    return None
 
# Extract and clean
clean_blocks_reis = []
for block in blocks_reis:
    try:
        temp = df_data_reis.iloc[:, [block["vraag_col"], block["artikel_col"], block["verzekerd_col"]]].copy()
        if temp.shape[1] != 3 or temp.dropna(how="all").shape[0] == 0:
            continue
        temp.columns = ["vraag", "gold_article_id", "gold_answer"]
        temp["source"] = block["source"]
        parsed = parse_reis_source(block["source"])
        if parsed:
            for k, v in parsed.items():
                temp[k] = v
            # Remove "Hoofdstuk" and trim article id
            temp["gold_article_id"] = temp["gold_article_id"].astype(str).str.replace("Hoofdstuk", "", case=False).str.strip()
            clean_blocks_reis.append(temp)
    except Exception:
        continue
 
# Combine
df_cleaned_reis = pd.concat(clean_blocks_reis, ignore_index=True)
df_cleaned_reis = df_cleaned_reis[df_cleaned_reis["gold_article_id"].notna()]
print(df_cleaned_reis.head())


output_prefix = ("../../data/intermediate/")
output_path = output_prefix + "groundtruth_reis_enriched.xlsx"
df_cleaned_reis.to_excel(output_path, index=False, engine="openpyxl")

print("Done, flies saved")

