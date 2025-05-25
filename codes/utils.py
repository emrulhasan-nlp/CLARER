
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from transformers import AutoTokenizer
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.distributions import Categorical

#############Text encoder###########
def encode_texts(text_list, tokenizer, max_length,src_vocab_size):
    # Ensure all elements in the text_list are strings
    text_list = [str(text) if text is not None else '' for text in text_list]

    encoded = tokenizer(
        text_list, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    # Clamp token IDs to be within the vocabulary range
    encoded["input_ids"] = encoded["input_ids"].clamp(0, src_vocab_size - 1)
    return encoded["input_ids"]  # Get tokenized input_ids

def preparedata(filePath, max_seq_length, src_vocab_size, tgt_vocab_size, tripadvisor):
  
    df = pd.read_csv(filePath)#.sample(frac=0.1, random_state=42)
    
    # Get unique users and items
    user_to_index = {user: idx for idx, user in enumerate(df['user'].unique())}
    item_to_index = {item: idx for idx, item in enumerate(df['item'].unique())}

    # Convert users & items to indices efficiently
    df['user_idx'] = df['user'].map(user_to_index)
    df['item_idx'] = df['item'].map(item_to_index)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    src_text_to_id = encode_texts(df['text'].tolist(), tokenizer, max_seq_length, src_vocab_size)
    tgt_text_to_id = encode_texts(df['explanation'].tolist(), tokenizer, max_seq_length, tgt_vocab_size)

    ratings = df['rating'].tolist()
    num_users, num_items = len(user_to_index), len(item_to_index)

    # Handle TripAdvisor-specific criteria
    if tripadvisor:
        criteria = list(zip(df['value'], df['location'], df['sleep_quality'], df['rooms'], df['cleanliness'], ratings))
        data = list(zip(df['user_idx'], df['item_idx'], src_text_to_id, tgt_text_to_id, criteria))
    else:
        data = list(zip(df['user_idx'], df['item_idx'], src_text_to_id, tgt_text_to_id, ratings))

    return data, num_users, num_items

# split data into train-val-test
def datasplit(data, random_state=1702):

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=random_state)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=random_state)

    return train_data, val_data, test_data

def create_dataloader(data, batch_size, shuffle=True):
    user_ids, item_ids, src_texts, tgt_texts,ratings = zip(*data)

    # Convert src_texts and tgt_texts to PyTorch tensors with dtype=torch.long
    src_texts = torch.stack(src_texts)
    tgt_texts = torch.stack(tgt_texts)

    dataset = TensorDataset(torch.tensor(user_ids), torch.tensor(item_ids),
                           src_texts, tgt_texts, torch.tensor(ratings))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_and_test_explanations(test_data, model, device, criteria, calculate_aspRep, output_file):
    """Generates explanations for test data and saves them to a file."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    results = []
    test_xplanationPairs = []
    
    with open(output_file, "w") as f:
        for data_point in test_data:
            #print(data_point)
            #break
            user_id, item_id, review_tensor, gt_explanation_tensor, _ = data_point
            
            review_text = tokenizer.decode(review_tensor.tolist(), skip_special_tokens=True)
            ground_truth_explanation = tokenizer.decode(gt_explanation_tensor.tolist(), skip_special_tokens=True)
            target_length = max(2, len(ground_truth_explanation.split()))
            
            # Prepare input tensors
            user_id_tensor = torch.tensor([user_id], device=device)
            item_id_tensor = torch.tensor([item_id], device=device)
            decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
            text_input_ids = tokenizer(review_text, padding="max_length", truncation=True, max_length=target_length, return_tensors="pt")["input_ids"].to(device)
            
            # Generate explanation
            for _ in range(target_length):
                with torch.no_grad():
                    if criteria:
                        output, pred_rating, pred_criteria = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
                    elif calculate_aspRep:
                        output, pred_rating, aspRep = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
                    else:
                        output, pred_rating = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
                
                probs = torch.softmax(output[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            generated_explanation = tokenizer.decode(decoder_input_ids.squeeze().tolist(), skip_special_tokens=True)
            
            if generated_explanation:
                results.append(f"Ground Truth: {ground_truth_explanation}\nGenerated: {generated_explanation}\n" + "-" * 40 + "\n")
                test_xplanationPairs.append((ground_truth_explanation, generated_explanation))
        
        f.writelines(results)
    
    return test_xplanationPairs


############## Explanation Generation and Saving the File ################
def generate_explanation(model, user_id, item_id, review_text, tokenizer, device, max_length,criteria, calculate_aspRep):
    """Generates an explanation for a given review, without needing a ground truth sequence."""

    user_id_tensor = torch.tensor([user_id], device=device)
    item_id_tensor = torch.tensor([item_id], device=device)
    decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

    # Ensure text_input_ids is also on the same device
    text_input_ids = tokenizer(review_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_length):
        with torch.no_grad():  # No gradient calculation for inference

            if criteria:
                output, pred_rating, pred_criteria = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
            elif calculate_aspRep:
                output, pred_rating, aspRep = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
            else:
                 output, pred_rating = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)

        probs = torch.softmax(output[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        #next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)  # Get next word ID
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)  # Append token

        if next_token.item() == tokenizer.eos_token_id:
          break # Stop if
    return tokenizer.decode(decoder_input_ids.squeeze().tolist(), skip_special_tokens=True)

def test_explanation(test_data, model, device, criteria, calculate_aspRep,output_file):
    """Generates explanations for test data and saves them to a file."""
    with open(output_file, "w") as f:
        results = []
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        test_xplanationPairs=[]
        for data_point in test_data:
            user_id = data_point[0]
            item_id = data_point[1]

            review_tensor = data_point[2]
            gt_explanation_tensor = data_point[3]

            review_text = tokenizer.decode(review_tensor.tolist(), skip_special_tokens=True)
            ground_truth_explanation = tokenizer.decode(gt_explanation_tensor.tolist(), skip_special_tokens=True)

            target_length = max(2, len(ground_truth_explanation.split()))
            generated_explanation = generate_explanation(model, user_id, item_id, review_text, tokenizer, device, target_length, criteria, calculate_aspRep)

            if generated_explanation:  # Ensure it's valid
                results.append(f"Ground Truth: {ground_truth_explanation}\nGenerated: {generated_explanation}\n" + "-" * 40 + "\n")
                test_xplanationPairs.append((ground_truth_explanation, generated_explanation))

        f.writelines(results)

    return test_xplanationPairs

#####################################Evaluating the explanation##############
from rouge import Rouge

def rouge_scores(references, generated):
    """Calculate ROUGE Precision, Recall, and F1-score for ROUGE-1, ROUGE-2, and ROUGE-L"""
    rouge_obj = Rouge()
    score = rouge_obj.get_scores(generated, references)

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
    """a list of lists of tokens"""
    references_tokens = [ref.split() for ref in references]
    generated_tokens = generated[0].split()

    # Apply smoothing if smooth is True
    if smooth:
        smoothing_function = SmoothingFunction().method4  # Use method4 smoothing
    else:
        smoothing_function = None

    # Calculate BLEU score using sentence_bleu from nltk
    bleu_s = sentence_bleu(references_tokens, generated_tokens,
                           weights=(1.0 / n_gram,) * n_gram,
                           smoothing_function=smoothing_function)

    return bleu_s * 100


############################Evaluating############################
from rouge_score import rouge_scorer

def RougeScore(references, generated):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(explanation, predicted_explanation)

    rouge1_F1_scores.append(scores['rouge1'].fmeasure)
    rouge1_R_scores.append(scores['rouge1'].recall)
    rouge1_P_scores.append(scores['rouge1'].precision)

    rouge2_F1_scores.append(scores['rouge2'].fmeasure)
    rouge2_R_scores.append(scores['rouge2'].recall)
    rouge2_P_scores.append(scores['rouge2'].precision)

    rougeL_F1_scores.append(scores['rougeL'].fmeasure)
    rougeL_R_scores.append(scores['rougeL'].recall)
    rougeL_P_scores.append(scores['rougeL'].precision)


    rouge1_F1_scores, rouge1_R_scores, rouge_P_scores 







