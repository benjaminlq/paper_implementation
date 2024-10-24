import evaluate
from typing import List

BLEU_SCORER = evaluate.load("bleu")
ROUGE_SCORER = evaluate.load('rouge')

def evaluate_translation(
    references: List[str],
    predictions: List[str]
):
    bleu_score = BLEU_SCORER.compute(predictions=predictions, references=references)
    rouge_score = ROUGE_SCORER.compute(predictions=predictions, references=references)

    print("Bleu Score:", bleu_score["bleu"])
    print("Rouge Score:", rouge_score)

    return bleu_score, rouge_score