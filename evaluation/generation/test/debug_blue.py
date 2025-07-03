from evaluation.generation.utils.advanced_metrics import compute_bleu
from evaluation.generation.utils.advanced_metrics import compute_meteor


# Örnek stringlerle test edelim
sys_answer = "Ja, schade is verzekerd bij diefstal van spullen"
gold_answer = "Bij diefstal is schade verzekerd"
 
bleu = compute_bleu(sys_answer, gold_answer)
print(f"BLEU DEBUG: {bleu:.4f} | SYS: {sys_answer} | GOLD: {gold_answer}")

score = compute_meteor(sys_answer, gold_answer)
print(f"Meteor DEBUG: {score:.4f} | SYS: {sys_answer} | GOLD: {gold_answer}")