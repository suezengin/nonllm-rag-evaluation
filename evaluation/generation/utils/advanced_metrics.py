import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

 
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
 
 
def compute_bleu(reference, prediction):
    """
    Compute BLEU score between reference and prediction.
    """
    try:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            return 0.0
 
        ref_tokens = reference.lower().split()
        pred_tokens =prediction.lower().split()
        smoothie = SmoothingFunction().method4
 
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    except Exception:
        return 0.0
 
    

def compute_meteor(reference, prediction):
    reference_tokens = reference.lower().split()
    cand_tokens = prediction.lower().split()
 
    matches = sum(1 for w in cand_tokens if w in reference_tokens)
    if matches == 0:
        return 0.0
 
    precision = matches / len(cand_tokens)
    recall = matches / len(reference_tokens)
    fmean = (10 * precision * recall) / (9 * precision + recall)
 
    frag_penalty = 0.5  # constant
    score = fmean * (1 - frag_penalty)
    return score    


 
def compute_rouge_all(reference, prediction):
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    Returns a dictionary with each score.
    """
    try:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
 
        scorer = Rouge()
        scores = scorer.get_scores(prediction, reference)
        return {
            "rouge-1": scores[0]["rouge-1"]["f"],
            "rouge-2": scores[0]["rouge-2"]["f"],
            "rouge-l": scores[0]["rouge-l"]["f"],
        }
    except Exception:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    


#could not find sacrebleu package, which is better than NLTK for Dutch.
'''
import sacrebleu
from rouge import Rouge
 
def compute_bleu(reference, prediction):
    """
    Compute BLEU score using sacrebleu for better multilingual reliability.
    """
    try:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            return 0.0
        if not reference.strip() or not prediction.strip():
            return 0.0
 
        bleu = sacrebleu.sentence_bleu(prediction.strip(), [reference.strip()])
        return bleu.score / 100.0  # Normalize to [0,1]
    except Exception:
        return 0.0
 
def compute_meteor(reference, prediction):
    """
    Simulate METEOR score using chrF from sacrebleu as a proxy (no native Dutch METEOR).
    """
    try:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            return 0.0
        if not reference.strip() or not prediction.strip():
            return 0.0
 
        chrf = sacrebleu.sentence_chrf(prediction.strip(), [reference.strip()])
        return chrf.score / 100.0  # Normalize to [0,1]
    except Exception:
        return 0.0
    '''