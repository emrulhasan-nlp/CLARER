import pandas as pd
import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def rouge_scores(references, generated):
    """Calculate ROUGE Precision, Recall, and F1-score for ROUGE-1, ROUGE-2, and ROUGE-L"""
    rouge_obj = Rouge()
    score = rouge_obj.get_scores([generated], [references])  # Ensure list format

    # Extracting precision, recall, and f-score
    rouge_s = {
        'rouge_1/f_score': score[0]['rouge-1']['f'] * 100,
        'rouge_1/precision': score[0]['rouge-1']['p'] * 100,
        'rouge_1/recall': score[0]['rouge-1']['r'] * 100,

        'rouge_2/f_score': score[0]['rouge-2']['f'] * 100,
        'rouge_2/precision': score[0]['rouge-2']['p'] * 100,
        'rouge_2/recall': score[0]['rouge-2']['r'] * 100,

        'rouge_l/f_score': score[0]['rouge-l']['f'] * 100,
        'rouge_l/precision': score[0]['rouge-l']['p'] * 100,
        'rouge_l/recall': score[0]['rouge-l']['r'] * 100
    }
    return rouge_s

def bleu_score(references, generated, n_gram=4, smooth=True):
    """Compute BLEU score for a given reference and generated text."""
    references_tokens = [ref.split() for ref in references]
    generated_tokens = generated.split()

    # Apply smoothing if smooth is True
    smoothing_function = SmoothingFunction().method4 if smooth else None

    bleu_s = sentence_bleu(references_tokens, generated_tokens,
                           weights=(1.0 / n_gram,) * n_gram,
                           smoothing_function=smoothing_function)
    return bleu_s * 100

def parse_ground_truth_generated(text):
    """Extract ground truth and generated text pairs from a given text input."""
    pattern = r"Ground Truth:\s*(.*?)\s*\nGenerated:\s*(.*?)\s*(?:\n-+|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(gt.strip(), gen.strip()) for gt, gen in matches]

# Specify dataset file path
#data = './yelp_1.0_criteriaFalse_aspectFalse_xpltest.txt'
#data='./yelp_1.0_criteriaFalse_aspectTrue_xpltest.txt'
#data='./trip_review_xplns_1.0_criteriaFalse_aspectTrue_xpltest.txt'
data='./trip_review_xplns_1.0_criteriaTrue_aspectFalse_xpltest.txt'

# Read file content
try:
    with open(data, "r", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    print(f"Error: File '{data}' not found.")
    exit()

# Extract ground truth and generated explanations
gt_gen_pairs = parse_ground_truth_generated(text)

# Filter out empty pairs
filtered_pairs = [(tgt, gen) for tgt, gen in gt_gen_pairs if tgt.strip() and gen.strip()]

if not filtered_pairs:
    print("No valid ground truth - generated pairs found. Exiting.")
    exit()

# Initialize metric storage
bleu_scores1, bleu_scores4 = [], []
rouge1_F1_scores, rouge2_F1_scores, rougeL_F1_scores = [], [], []
rouge1_R_scores, rouge2_R_scores, rougeL_R_scores = [], [], []
rouge1_P_scores, rouge2_P_scores, rougeL_P_scores = [], [], []

# Compute evaluation metrics
for explanation, predicted_explanation in filtered_pairs:
    bleu_scores1.append(bleu_score([explanation], predicted_explanation, n_gram=1))
    bleu_scores4.append(bleu_score([explanation], predicted_explanation, n_gram=4))
    
    try:
        result_scores = rouge_scores(explanation, predicted_explanation)
    except Exception as e:
        print(f"Skipping pair due to error: {e}")
        continue
    
    rouge1_F1_scores.append(result_scores.get('rouge_1/f_score', 0))
    rouge1_R_scores.append(result_scores.get('rouge_1/recall', 0))
    rouge1_P_scores.append(result_scores.get('rouge_1/precision', 0))

    rouge2_F1_scores.append(result_scores.get('rouge_2/f_score', 0))
    rouge2_R_scores.append(result_scores.get('rouge_2/recall', 0))
    rouge2_P_scores.append(result_scores.get('rouge_2/precision', 0))

    rougeL_F1_scores.append(result_scores.get('rouge_l/f_score', 0))
    rougeL_R_scores.append(result_scores.get('rouge_l/recall', 0))
    rougeL_P_scores.append(result_scores.get('rouge_l/precision', 0))

def safe_average(scores):
    """Compute the safe average of a list of scores, avoiding division by zero."""
    return sum(scores) / len(scores) if scores else 0

# Average scores
avg_bleu1 = safe_average(bleu_scores1)
avg_bleu4 = safe_average(bleu_scores4)

avg_rouge1_f = safe_average(rouge1_F1_scores)
avg_rouge2_f = safe_average(rouge2_F1_scores)
avg_rougeL_f = safe_average(rougeL_F1_scores)

avg_rouge1_R = safe_average(rouge1_R_scores)
avg_rouge2_R = safe_average(rouge2_R_scores)
avg_rougeL_R = safe_average(rougeL_R_scores)

avg_rouge1_P = safe_average(rouge1_P_scores)
avg_rouge2_P = safe_average(rouge2_P_scores)
avg_rougeL_P = safe_average(rougeL_P_scores)

# Print results
print(f"Average BLEU1: {avg_bleu1:.4f}")
print(f"Average BLEU4: {avg_bleu4:.4f}")

print(f"Average ROUGE-1 F1: {avg_rouge1_f:.4f}, R: {avg_rouge1_R:.4f}, P: {avg_rouge1_P:.4f}")
print(f"Average ROUGE-2 F1: {avg_rouge2_f:.4f}, R: {avg_rouge2_R:.4f}, P: {avg_rouge2_P:.4f}")
print(f"Average ROUGE-L F1: {avg_rougeL_f:.4f}, R: {avg_rougeL_R:.4f}, P: {avg_rougeL_P:.4f}")

