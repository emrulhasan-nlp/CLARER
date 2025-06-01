import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def parse_ground_truth_generated(text):
    """Extract (ground truth, generated) text pairs from a structured text block."""
    pattern = r"Ground Truth:\s*(.*?)\s*\nGenerated:\s*(.*?)\s*(?:\n-+|$)"
    return [(gt.strip(), gen.strip()) for gt, gen in re.findall(pattern, text, re.DOTALL)]

def rouge_scores(reference, generated):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, F1)."""
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    s = scores[0]
    return {
        'rouge_1': s['rouge-1'],
        'rouge_2': s['rouge-2'],
        'rouge_l': s['rouge-l'],
    }

def bleu_score(references, generated, n_gram=4, smooth=True):
    """Compute BLEU score for a single prediction with optional smoothing."""
    refs_tokens = [ref.split() for ref in references]
    gen_tokens = generated.split()
    weights = tuple((1.0 / n_gram,) * n_gram)
    smoother = SmoothingFunction().method4 if smooth else None
    score = sentence_bleu(refs_tokens, gen_tokens, weights=weights, smoothing_function=smoother)
    return score * 100

def safe_average(scores):
    """Return the average or zero if the list is empty."""
    return sum(scores) / len(scores) if scores else 0

# === CONFIGURATION ===
data_path = '../results/trip_review_xplns_1.0_criteriaTrue_aspectFalse_xpltest.txt'

# === READ FILE ===
try:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
except FileNotFoundError:
    print(f"Error: File '{data_path}' not found.")
    exit()

# === PARSE TEXT ===
pairs = parse_ground_truth_generated(raw_text)
pairs = [(gt, gen) for gt, gen in pairs if gt and gen]

if not pairs:
    print("No valid ground truth-generated pairs found. Exiting.")
    exit()

# === METRIC COLLECTION ===
metrics = {
    'bleu1': [],
    'bleu4': [],
    'rouge_1/f': [], 'rouge_1/p': [], 'rouge_1/r': [],
    'rouge_2/f': [], 'rouge_2/p': [], 'rouge_2/r': [],
    'rouge_l/f': [], 'rouge_l/p': [], 'rouge_l/r': [],
}

for reference, generated in pairs:
    metrics['bleu1'].append(bleu_score([reference], generated, n_gram=1))
    metrics['bleu4'].append(bleu_score([reference], generated, n_gram=4))

    try:
        r_scores = rouge_scores([reference], [generated])
    except Exception as e:
        print(f"Skipping pair due to error: {e}")
        continue

    for key in ['rouge_1', 'rouge_2', 'rouge_l']:
        metrics[f"{key}/f"].append(r_scores[key]['f'] * 100)
        metrics[f"{key}/p"].append(r_scores[key]['p'] * 100)
        metrics[f"{key}/r"].append(r_scores[key]['r'] * 100)

# === REPORT ===
print(f"Average BLEU-1: {safe_average(metrics['bleu1']):.4f}")
print(f"Average BLEU-4: {safe_average(metrics['bleu4']):.4f}")

for key in ['rouge_1', 'rouge_2', 'rouge_l']:
    f1 = safe_average(metrics[f"{key}/f"])
    p = safe_average(metrics[f"{key}/p"])
    r = safe_average(metrics[f"{key}/r"])
    print(f"Average {key.upper()} F1: {f1:.4f}, R: {r:.4f}, P: {p:.4f}")
