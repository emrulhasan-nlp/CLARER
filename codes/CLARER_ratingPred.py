import argparse
import json
import math
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer

from models import NRTPlus, ContrastiveLoss
from utils import (
    create_dataloader,
    datasplit,
    generate_and_test_explanations,
    preparedata,
)

warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description="Rating Prediction with Explanation")
parser.add_argument("-d", "--dataset", type=str, default="trip_review_xplns")
parser.add_argument("-rs", "--random_seed", type=int, default=1702)
parser.add_argument("-seq", "--seq_length", type=int, default=128)
parser.add_argument("-src_vocab", type=int, default=30522)
parser.add_argument("-tgt_vocab", type=int, default=30522)
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-ft", "--feat_type", type=bool, default=False)
parser.add_argument("-asRep", "--aspect_rep", type=bool, default=True)
args = parser.parse_args()

# Configuration
CATEGORY = args.dataset.strip().lower()
print(f"\nDataset: {CATEGORY}")

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.random_seed)
random.seed(args.random_seed)

# Load data
SOURCE_FOLDER = "../prepdata"
filePath = f"{SOURCE_FOLDER}/{CATEGORY}/{CATEGORY}.csv"
print(filePath)

data, num_users, num_items = preparedata(
    filePath,
    max_seq_length=args.seq_length,
    src_vocab_size=args.src_vocab,
    tgt_vocab_size=args.tgt_vocab,
    tripadvisor=False,
)
print(num_users, num_items)

train_data, val_data, test_data = datasplit(data, random_state=args.random_seed)
train_loader = create_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = create_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=args.batch_size, shuffle=False)

# Model setup
d_model, num_heads, num_layers = 256, 8, 6
d_ff, dropout, hidden_dim = 1024, 0.1, 64
criteria, calculate_aspRep = args.feat_type, args.aspect_rep
model = NRTPlus(num_users, num_items, args.src_vocab, args.tgt_vocab, d_model,
                 num_heads, num_layers, d_ff, args.seq_length, dropout,
                 hidden_dim, criteria, calculate_aspRep).to(device)

contrastive_loss = ContrastiveLoss()
criterion = nn.CrossEntropyLoss(ignore_index=0)
rating_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

print("Training started")

asp_emb = torch.from_numpy(np.load(f'../prepdata/{CATEGORY}/{CATEGORY}_emb.npy')).to(device)

# Training parameters
num_epochs, patience = 20, 3
best_val_loss = float('inf')
early_stop_counter = 0
best_model_path = f"{CATEGORY}_{criteria}_{calculate_aspRep}_best_model.pth"
print(best_model_path)

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        userId, itemId = batch[0].to(device), batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)
        ratings = batch[4].type(torch.float32).to(device)

        optimizer.zero_grad()

        if criteria:
            output, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])
            xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
            rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
            criteria_loss = rating_criterion(pred_criteria.reshape(-1), batch[4][:, 0:5].reshape(-1).to(device))
            loss = xpln_loss + 0.5 * rating_loss + 1.0 * criteria_loss

        elif calculate_aspRep:
            asp_embed = asp_emb.unsqueeze(0).expand(src_batch.size(0), -1, -1)
            output, pred_rating, aspRep = model(userId, itemId, src_batch, tgt_batch[:, :-1])
            xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
            rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
            contrast_loss = contrastive_loss(asp_embed, asp_embed)
            loss = xpln_loss + 0.5 * rating_loss + 1.0 * contrast_loss

        else:
            output, pred_rating = model(userId, itemId, src_batch, tgt_batch[:, :-1])
            xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
            rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
            loss = xpln_loss + 0.5 * rating_loss

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            userId, itemId = batch[0].to(device), batch[1].to(device)
            src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)
            ratings = batch[4].type(torch.float32).to(device)

            if criteria:
                output, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])
                xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
                rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
                criteria_loss = rating_criterion(pred_criteria.reshape(-1), batch[4][:, 0:5].reshape(-1).to(device))
                loss = xpln_loss + 0.5 * rating_loss + 1.0 * criteria_loss

            elif calculate_aspRep:
                asp_embed = asp_emb.unsqueeze(0).expand(src_batch.size(0), -1, -1)
                output, pred_rating, aspRep = model(userId, itemId, src_batch, tgt_batch[:, :-1])
                xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
                rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
                contrast_loss = contrastive_loss(asp_embed, asp_embed)
                loss = xpln_loss + 0.5 * rating_loss + 1.0 * contrast_loss

            else:
                output, pred_rating = model(userId, itemId, src_batch, tgt_batch[:, :-1])
                xpln_loss = criterion(output.view(-1, args.tgt_vocab), tgt_batch[:, 1:].reshape(-1))
                rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
                loss = xpln_loss + 0.5 * rating_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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

# Testing
model.load_state_dict(torch.load(best_model_path))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
y_trues, y_preds = [], []

with torch.no_grad():
    for batch in test_loader:
        userId, itemId = batch[0].to(device), batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)
        ratings = batch[4].type(torch.float32).to(device)

        if criteria:
            _, pred_rating, _ = model(userId, itemId, src_batch, tgt_batch[:, :-1])
        elif calculate_aspRep:
            _, pred_rating, _ = model(userId, itemId, src_batch, tgt_batch[:, :-1])
        else:
            _, pred_rating = model(userId, itemId, src_batch, tgt_batch[:, :-1])

        y_trues.extend(ratings.cpu().numpy().flatten())
        y_preds.extend(pred_rating.cpu().numpy().flatten())

mse = mean_squared_error(y_trues, y_preds)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_trues, y_preds)
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Explanation generation
output_file = f"{CATEGORY}_{1.0}_criteria{criteria}_aspect{calculate_aspRep}_xpltest.txt"
generate_and_test_explanations(test_data, model, device, criteria, calculate_aspRep, output_file)