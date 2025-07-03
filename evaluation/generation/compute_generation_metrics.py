import pandas as pd
from utils.embedding import embed_text
from utils.similarity import (
    compute_cosine_similarity,
    compute_jaccard_similarity,
    compute_edit_distance,
)
from utils.advanced_metrics import (
    compute_bleu,
    compute_rouge_all,
    #compute_meteor,
)
 
 
def compute_generation_metrics(df, source="asr", client=None):
    """
    Compute generation metrics for a given source ("asr" or "aegon").
 
    Args:
        df (pd.DataFrame): DataFrame with system and gold answers.
        source (str): Either "asr" or "aegon".
        client: Azure OpenAI client for embedding.
 
    Returns:
        pd.DataFrame: Row-level metrics including similarity and lexical scores.
    """
    sys_col = f"{source}_response_Antwoord"
    gold_col = f"{source}_gold_answer"
    match_col = f"match_type_{source}"
 
    records = []
 
    for idx, row in df.iterrows():
        match_type = row.get(match_col)
        if match_type not in {"exact", "prefix", "no_match"}:
            continue
 
        sys_answer = row.get(sys_col, "")
        gold_answer = row.get(gold_col, "")
 
        # Embedding-based
        sys_emb = embed_text(sys_answer, client)
        gold_emb = embed_text(gold_answer, client)
        cosine_sim = compute_cosine_similarity(sys_emb, gold_emb)
 
        # String-based
        jaccard_sim = compute_jaccard_similarity(sys_answer, gold_answer)
        edit_dist = compute_edit_distance(sys_answer, gold_answer)
 
        # Lexical metrics
        bleu = compute_bleu(sys_answer, gold_answer)
        rouge = compute_rouge_all(sys_answer, gold_answer)  # returns dict
        #meteor = compute_meteor(sys_answer, gold_answer)
 
        record = {
            "index": idx,
            "match_type": match_type,
            "system_answer": sys_answer,
            "gold_answer": gold_answer,
            "cosine_similarity": cosine_sim,
            "jaccard_similarity": jaccard_sim,
            "edit_distance": edit_dist,
            "bleu": bleu,
            #"meteor": meteor,
            "rouge-1": rouge.get("rouge-1"),
            "rouge-2": rouge.get("rouge-2"),
            "rouge-l": rouge.get("rouge-l"),
        }
 
        records.append(record)
 
    return pd.DataFrame(records).set_index("index")