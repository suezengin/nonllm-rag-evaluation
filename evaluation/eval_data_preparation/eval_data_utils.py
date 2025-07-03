# retrieval_eval_utils.py
import re
import ast
import pandas as pd
 
def extract_article_numbers(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    if isinstance(text, list):
        return text
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except:
        pass
    article_pattern = re.compile(r"\b\d+(?:\.\d+){1,2}\b")
    return article_pattern.findall(str(text))


# Preprocess article IDs from input DataFrame
 
def preprocess_article_ids_fixed(df):
    df["system_article_ids_aegon"] = df["aegon_response_Artikel nummer"].apply(extract_article_numbers)
    df["system_article_ids_asr"] = df["asr_response_Artikel nummer"].apply(extract_article_numbers)
    df["gt_article_ids_aegon"] = df["aegon_gold_article_id"].apply(extract_article_numbers)
    df["gt_article_ids_asr"] = df["asr_gold_article_id"].apply(extract_article_numbers)
    
    df["selected_articles_aegon"] = df["selected_articles_aegon"].apply(extract_article_numbers)
    df["top_3_aegon"] = df["top_3_aegon"].apply(extract_article_numbers)
    df["selected_articles_asr"] = df["selected_articles_asr"].apply(extract_article_numbers)
    df["top_3_asr"] = df["top_3_asr"].apply(extract_article_numbers)
    return df

def has_gt_quality_issue(article_ids):
    if not isinstance(article_ids, list):
        return True
    if not article_ids:
        return True
    for aid in article_ids:
        if not re.match(r"^\d+(\.\d+)*$", str(aid)):
            return True
    return False
 
# Match classification logic
 
def classify_match(system_articles, gold_articles):
    if not system_articles or not gold_articles:
        return "no_match"
    if any(sa == ga for sa in system_articles for ga in gold_articles):
        return "exact"
    if any(sa.startswith(ga) or ga.startswith(sa) for sa in system_articles for ga in gold_articles):
        return "prefix"
    return "no_match"
 
# Determine KB source from merged_all
 
def determine_kb_source_aegon(polis_versie, product):
    polis_versie = str(polis_versie).lower().strip()
    product = str(product).lower().strip()
 
    if "inboedel" in product:
        if "nieuw" in polis_versie:
            return "aegon_inboedel_nieuw"
        elif "oud" in polis_versie:
            return "aegon_inboedel_oud"
    if "doorlopende reisverzekering" in product: 
            return "aegon_reis"
 

def determine_kb_source_asr(source_gt_asr):
    source_gt_asr = str(source_gt_asr).lower().strip()
 
    if "vp_dr_2024" in source_gt_asr:
        return "asr_vp_dr_2024"
 
    elif "ikz_allrisk" in source_gt_asr:
        return "asr_ikz_allrisk"
 
    elif "ikz_2018" in source_gt_asr:
        return "asr_ikz_2018"
    elif "advies_basis" in source_gt_asr:
        return "asr_advies_basis"
    
    return None
 
# Check justification using KB tables
 
def is_justified_by_kb(system_articles, gold_articles, kb_df, kb_source):
    if not kb_source:
        return False
 
    filtered_kb = kb_df[kb_df["kb_source"] == kb_source].copy()
    if filtered_kb.empty:
        return False
 
    # Normalize all fields to string list
    def parse_list_column(col):
        return col.apply(lambda x: extract_article_numbers(x)
                         )
 
    filtered_kb["article_number"] = filtered_kb["article_number"].astype(str)
    filtered_kb["sub_mentions_in_content"] = parse_list_column(filtered_kb["sub_mentions_in_content"])
    filtered_kb["mentions_other_articles"] = parse_list_column(filtered_kb["mentions_other_articles"])
 
    # Combine all possible article references in KB
    kb_all_refs = []
    for _, row in filtered_kb.iterrows():
        refs = set([row["article_number"]] + row["sub_mentions_in_content"] + row["mentions_other_articles"])
        kb_all_refs.append(refs)
 
    # Flatten for quick match
    flat_kb_articles = set().union(*kb_all_refs)
 
    match_type = classify_match(system_articles, gold_articles)
 
    if match_type == "exact":
        return any(x in flat_kb_articles for x in system_articles + gold_articles)
 
    if match_type == "prefix":
        return any(
            any(sa in refs for sa in system_articles) and
            any(ga in refs for ga in gold_articles)
            for refs in kb_all_refs
        )
 
    return any(
        any(sa in refs for sa in system_articles) and
        any(ga in refs for ga in gold_articles)
        for refs in kb_all_refs
    )