import pandas as pd
import re
from utils import bleu_score, rouge_scores

def parse_ground_truth_generated(text):

    pattern = r"Ground Truth:\s*(.*?)\nGenerated:\s*(.*?)\n-+"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(gt.strip(), gen.strip()) for gt, gen in matches]

#data='./1.5_criteriaFalse_aspectTrue_xpltest.txt'
#data='./criteriaFalse_aspectTrue_xpltest.txt'
#data='./yelp_1.0_criteriaFalse_aspectTrue_xpltest.txt'
#data='./amz_mt_1.0_criteriaFalse_aspectFalse_xpltest.txt'

#data='./trip_review_xplns_1.0_criteriaFalse_aspectFalse_xpltest.txt'

data='./yelp_1.0_criteriaFalse_aspectFalse_xpltest.txt'

#data='./trip_review_xplns_1.0_criteriaTrue_aspectFalse_xpltest.txt'

with open(data, "r", encoding="utf-8") as file:
    text = file.read()

gt_gen_pairs = parse_ground_truth_generated(text)

filtered_pairs = []
for tgt, gen in gt_gen_pairs:
    if len(tgt) != 0 or len(gen) != 0:
        filtered_pairs.append((tgt, gen))

bleu_scores1, bleu_scores4 = [],[]
rouge1_F1_scores, rouge2_F1_scores, rougeL_F1_scores = [], [], []
rouge1_R_scores, rouge2_R_scores, rougeL_R_scores = [], [], []
rouge1_P_scores, rouge2_P_scores, rougeL_P_scores = [], [], []

for explanation, predicted_explanation in filtered_pairs:
    
    bleu1 = bleu_score([explanation], [predicted_explanation], n_gram=1)
    bleu_scores1.append(bleu1)

    bleu4 = bleu_score([explanation], [predicted_explanation], n_gram=4)
    bleu_scores4.append(bleu4)
    try:
      result_scores = rouge_scores(explanation, predicted_explanation)
    except:
      #print("Error: One or both explanations are empty or contain only whitespace. Skipping this pair.")
      continue  # Skip this pair and move to the next
    rouge1_F1_scores.append(result_scores['rouge_1/f_score'])
    rouge1_R_scores.append(result_scores['rouge_1/recall'])
    rouge1_P_scores.append(result_scores['rouge_1/precision'])

    rouge2_F1_scores.append(result_scores['rouge_2/f_score'])
    rouge2_R_scores.append(result_scores['rouge_2/recall'])  
    rouge2_P_scores.append(result_scores['rouge_2/precision'])

    rougeL_F1_scores.append(result_scores['rouge_l/f_score'])
    rougeL_R_scores.append(result_scores['rouge_l/recall']) 
    rougeL_P_scores.append(result_scores['rouge_l/precision'])

# Compute averages safely to avoid ZeroDivisionError
def safe_average(scores):
    return sum(scores) / len(scores) if scores else 0  # Avoid division by zero

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

print(f"Average BLEU1: {avg_bleu1:.4f}")
print(f"Average BLEU4: {avg_bleu4:.4f}")

print(f"Average ROUGE-1 f1: {avg_rouge1_f:.4f}, R:{avg_rouge1_R:.4f}, P:{avg_rouge1_P:.4f}")
print(f"Average ROUGE-2 f1: {avg_rouge2_f:.4f},R: {avg_rouge2_R:.4f}, P:{avg_rouge2_P:.4f}")
print(f"Average ROUGE-L f1:: {avg_rougeL_f:.4f},R: {avg_rougeL_R:.4f},P: {avg_rougeL_P:.4f}")
