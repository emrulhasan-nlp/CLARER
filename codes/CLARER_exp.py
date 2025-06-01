"""
Explainability
"""

import os
import time
import json
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import (
    preparedata,
    create_dataloader,
    datasplit,
    test_explanation,
    bleu_score,
    rouge_score
)
from models import NRTPlus

# ------------------------ Argument Parsing ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="trip_review_xplns", help="Dataset category (Default: TripAdvisor Review Data)")
parser.add_argument("-rs", "--random_seed", type=int, default=1702, help="Random seed (Default: 1702)")
parser.add_argument("-seq", "--seq_length", type=int, default=256, help="Maximum Sequence Length (Default: 256)")
parser.add_argument("-src_vocab", "--src_vocab_size", type=int, default=20000, help="Vocabulary size for source review (Default: 20000)")
parser.add_argument("-tgt_vocab", "--tgt_vocab_size", type=int, default=20000, help="Vocabulary size for target review (Default: 20000)")
parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size (Default: 64)")
args = parser.parse_args()

# ------------------------ Setup ------------------------
CATEGORY = args.dataset.strip().lower()
print(f"\nDataset: {CATEGORY}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
np.random.seed(args.random_seed)
random.seed(args.random_seed)

# ------------------------ Paths and Hyperparameters ------------------------
SOURCE_FOLDER = "../prepdata"
filePath = f"{SOURCE_FOLDER}/{CATEGORY}/{CATEGORY}.csv"
best_model_path = "best_model.pth"

max_seq_length = args.seq_length
src_vocab_size = args.src_vocab_size
tgt_vocab_size = args.tgt_vocab_size
batch_size = args.batch_size

# Model hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 1024
rating_dim = 1
criteria_dim = 5
dropout = 0.1
hidden_dim = 128
num_epochs = 20
patience = 3

# ------------------------ Load and Prepare Data ------------------------
data, num_users, num_items = preparedata(filePath, max_seq_length, src_vocab_size, tgt_vocab_size)
train_data, val_data, test_data = datasplit(data, random_state=args.random_seed)

train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)

# ------------------------ Model Initialization ------------------------
model = NRTPlus(num_users, num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                num_layers, d_ff, max_seq_length, dropout, hidden_dim, rating_dim, criteria_dim).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
rating_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# ------------------------ Training and Validation ------------------------
start_time = time.time()
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch = batch[2].to(device)
        tgt_batch = batch[3].to(device)
        criteria = batch[5][:, :5].float().to(device)
        ratings = batch[5][:, 5:6].float().to(device)

        optimizer.zero_grad()

        explanation, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])

        exp_loss = criterion(explanation.view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
        rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
        criteria_loss = rating_criterion(pred_criteria.view(-1), criteria.view(-1))
        loss = exp_loss + rating_loss + criteria_loss

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            userId = batch[0].to(device)
            itemId = batch[1].to(device)
            src_batch = batch[2].to(device)
            tgt_batch = batch[3].to(device)
            criteria = batch[5][:, :5].float().to(device)
            ratings = batch[5][:, 5:6].float().to(device)

            explanation, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])

            exp_loss = criterion(explanation.view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
            rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
            criteria_loss = rating_criterion(pred_criteria.view(-1), criteria.view(-1))
            loss = exp_loss + rating_loss + criteria_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved!")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

train_val_duration = time.time() - start_time
h, rem = divmod(train_val_duration, 3600)
m, s = divmod(rem, 60)
print(f"\nTraining/Validation Time: {int(h)}h {int(m)}m {int(s)}s")

# ------------------------ Testing ------------------------
start_time = time.time()

model.load_state_dict(torch.load(best_model_path))
model.eval()

y_trues, y_preds = [], []
with torch.no_grad():
    for batch in test_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch = batch[2].to(device)
        tgt_batch = batch[3].to(device)
        ratings = batch[5][:, 5:6].float().to(device)

        _, pred_rating, _ = model(userId, itemId, src_batch, tgt_batch[:, :-1])
        y_trues.extend(ratings.cpu().numpy().flatten())
        y_preds.extend(pred_rating.cpu().numpy().flatten())

mse = mean_squared_error(y_trues, y_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_trues, y_preds)

print(f"\nTest MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

test_duration = time.time() - start_time
h, rem = divmod(test_duration, 3600)
m, s = divmod(rem, 60)
print(f"Testing Time: {int(h)}h {int(m)}m {int(s)}s")

# ------------------------ Explanation Evaluation ------------------------
start_time = time.time()

save_explanation = f"../results/{CATEGORY}_test_exp.txt"
test_xplanationPairs = test_explanation(test_data, model, device, save_explanation)

bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []

for ref, pred in test_xplanationPairs:
    bleu_scores.append(bleu_score([ref], [pred]))
    rouge = rouge_score([ref], [pred])
    rouge1_scores.append(rouge['rouge_1/f_score'])
    rouge2_scores.append(rouge['rouge_2/f_score'])
    rougeL_scores.append(rouge['rouge_l/f_score'])

# Print average scores
print(f"\nAverage BLEU: {np.mean(bleu_scores):.4f}")
print(f"Average ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"Average ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"Average ROUGE-L: {np.mean(rougeL_scores):.4f}")

eval_duration = time.time() - start_time
h, rem = divmod(eval_duration, 3600)
m, s = divmod(rem, 60)
print(f"Explanation Evaluation Time: {int(h)}h {int(m)}m {int(s)}s")
