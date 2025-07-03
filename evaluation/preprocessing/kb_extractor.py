import re
import pandas as pd
from pathlib import Path

def clean_for_excel(text: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F]", "", text)

def extract_kb_structure_with_source(filepath: str, source_name: str) -> pd.DataFrame:
    with Path(filepath).open("r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    article_data = []
    for entry in entries:
        metadata = entry.get("metadata", {})
        article_number = metadata.get("article_number", "").strip()
        content = clean_for_excel(entry.get("content", "").strip())
        if article_number:
            source_group = "aegon" if "aegon" in source_name.lower() else "asr"
            article_data.append({
                "article_number": article_number,
                "content": content,
                "kb_source": source_name,
                "kb_source_group": source_group
            })

    df = pd.DataFrame(article_data)
    all_article_numbers = set(df["article_number"])

    # Regexler
    sub_article_pattern = re.compile(r"\b(\d+)\.")
    reference_pattern = re.compile(r"\b\d+(?:\.\d+){0,2}\b")

    output_rows = []

    for _, row in df.iterrows():
        main_article = row["article_number"]
        content = row["content"]

        # Sub-mentions
        sub_article_numbers = sorted(set([
            f"{main_article}.{m}"
            for m in sub_article_pattern.findall(content)
        ]))

        # Referenced articles
        referenced_articles = sorted(set([
            match for match in reference_pattern.findall(content)
            if match != main_article and (
                match in all_article_numbers or any(match.startswith(a + ".") for a in all_article_numbers)
            )
        ]))

        output_rows.append({
            "article_number": main_article,
            "content": content,
            "sub_mentions_in_content": ", ".join(sub_article_numbers),
            "mentions_other_articles": ", ".join(referenced_articles),
            "kb_source": source_name,
            "kb_source_group": row["kb_source_group"]
        })

        for sub_article in sub_article_numbers:
            output_rows.append({
                "article_number": sub_article,
                "content": content,
                "sub_mentions_in_content": "",
                "mentions_other_articles": "",
                "kb_source": source_name,
                "kb_source_group": row["kb_source_group"]
            })

    return pd.DataFrame(output_rows)

prefix = "../../data/inputs/knowledge_base/"

kb_paths = {
    "aegon_reis": prefix + "input_Polisvoorwaarden_Aegon_Doorlopende_Reisverzekering_3032_anl 2_new.jsonl",
    "aegon_inboedel_oud": prefix + "input_aegon_oud.jsonl",
    "aegon_inboedel_nieuw": prefix + "input_aegon_nieuw.jsonl",
    "asr_vp_dr_2024": prefix + "input_a.s.r. Reis Voordeelpakket Doorlopende reis VP DR 2024-01 - selling 2.jsonl",
    "asr_ikz_2018": prefix + "input_a.s.r. IKZ voorheen Ditzo Voorwaarden Doorlopende Reisverzekering D DR 2018-01 - selling 2.jsonl",
    "asr_advies_basis": prefix + "input_asr.jsonl",
    "asr_ikz_allrisk": prefix + "input_asr_ikz.jsonl"
}

combined_rows = []
for source_name, path in kb_paths.items():
    df = extract_kb_structure_with_source(path, source_name)
    combined_rows.append(df)

combined_df = pd.concat(combined_rows, ignore_index=True)

output_path = "../../data/intermediate/all_kbs_combined.xlsx"
combined_df.to_excel(output_path, index=False)

print("Excel file is ready")

