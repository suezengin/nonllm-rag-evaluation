import pandas as pd

prefix = ("../../data/inputs/")
file_path = prefix + "Groundtruth_inboedel.xlsx"
df_raw= pd.read_excel(file_path, sheet_name="Blad1", header =None)

df_raw = pd.read_excel(file_path, header=None)
 
# === Extract headers
header_product_row = df_raw.iloc[0].fillna(method="ffill")
header_version_row = df_raw.iloc[1].fillna("")
header_fields_row = df_raw.iloc[2]
df = df_raw.iloc[3:].copy().reset_index(drop=True)

# === Block detection helpers
def is_artikel(col_name):
    return "artikel" in str(col_name).lower().replace("arikel", "artikel")
 
def is_verzekerd(col_name):
    return "verzekerd" in str(col_name).lower().replace("?", "").strip()
 
# === Detect valid blocks
blocks = []
col_idx = 1
while col_idx < len(header_fields_row):
    if is_artikel(header_fields_row.iloc[col_idx]) and is_verzekerd(header_fields_row.iloc[col_idx + 1]):
        product_name = str(header_product_row[col_idx]).strip().lower().replace(" ", "_")
        version_name = str(header_version_row[col_idx]).strip().lower().replace(" ", "_")
        source = f"{product_name}_{version_name}"
        blocks.append({
            "vraag_col": 0,
            "artikel_col": col_idx,
            "verzekerd_col": col_idx + 1,
            "source": source
        })
    col_idx += 2
 
def parse_inboedel_source(source: str):

    source = source.lower()

    parts = source.split("_")
 
    # Aegon logic

    if "aegon" in parts:

        polis_versie = "Oud (3038)" if "oud" in parts else "Nieuw (3041)"

        dekking = "Allrisk" if "allrisk" in parts or "royaal" in source else "Basis"

        return {

            "product": "Inboedel",

            "polis_versie": polis_versie,

            "type_klant": None,

            "dekking": dekking

        }
 
    # ASR logic

    elif "a.s.r." in source:

        type_klant = "Adviseur" if "advies" in parts else "Direct"

        dekking = "Allrisk" if "topdekking" in source or "allrisk" in parts else "Basis"

        return {

            "product": "Inboedel",

            "polis_versie": None,

            "type_klant": type_klant,

            "dekking": dekking

        }
 
    return None

# === Extract and clean blocks
groundtruths = []
for block in blocks:
    try:
        temp = df.iloc[:, [block["vraag_col"], block["artikel_col"], block["verzekerd_col"]]].copy()
        if temp.shape[1] != 3 or temp.dropna(how="all").shape[0] == 0:
            continue
        temp.columns = ["vraag", "gold_article_id", "gold_answer"]
        # Clean 'Hoofdstuk' from gold_article_id
        temp["gold_article_id"] = (
            temp["gold_article_id"]
            .astype(str)
            .str.replace("Hoofdstuk", "", case=False)
            .str.strip()
        )
 
        temp["source"] = block["source"]
        metadata = parse_inboedel_source(block["source"])
        if metadata:
            for key, value in metadata.items():
                temp[key] = value
            groundtruths.append(temp)
    except Exception:
        continue


# === Combine and save
df_groundtruth = pd.concat(groundtruths, ignore_index=True)
df_groundtruth.columns = df_groundtruth.columns.str.strip()
df_groundtruth = df_groundtruth[df_groundtruth["gold_article_id"].notna()]
print(df_groundtruth.tail())


output_prefix = ("../../data/intermediate/")
output_path = output_prefix + "groundtruth_inboedel_enriched.xlsx"
df_groundtruth.to_excel(output_path, index=False, engine="openpyxl") 

print("Done, flies saved")